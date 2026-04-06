#!/usr/bin/env python3
"""
evaluate_grasps.py — 评估生成抓握的物理质量 + 语义一致性

指标:
  - Penetration Depth (cm):       手穿入物体的深度, 越低越好
  - Intersection Volume (cm³):    手-物体交叉体积, 越低越好
  - Simulation Displacement (cm): 物理仿真中物体位移, 越低越好
  - Contact Ratio:                近表面接触样本比例, 越高越好
  - Contact Token F1:             最终 hand pose 反推接触矩阵，与 conditioning tokens 的 F1
  - Slot Hit Rate:                折叠 hand_part 后，GT slot 中被命中的比例

依赖:
  - lib.metrics.penetration      (穿透深度)
  - lib.metrics.intersection     (交叉体积)
  - lib.metrics.simulator        (物理仿真)
  - assets/closed_mano_faces.pkl (密封 MANO 面片)
  - proc_dir/ 下的 watertight/, voxel/, vhacd/ 预处理物体

用法:
  python evaluate_grasps.py --exp_path experiments/generated
"""

import os
import argparse
import pickle
import logging
import warnings
from collections import defaultdict

import numpy as np
import trimesh
import torch
from joblib import Parallel, delayed
from scipy.spatial import cKDTree

from data.token_encoder import TokenEncoder
from metrics.basic_metric import AverageMeter
from metrics.penetration import penetration
from metrics.intersection import solid_intersection_volume
from metrics.simulator import simulation_sample
from utils.mano_utils import MANOHelper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# 有效性检查
# ==============================================================================

# MANO 手部顶点数量
MANO_VERTS = 778

def check_hand_verts(hand_verts: np.ndarray, sample_id: str) -> tuple[bool, str]:
    """
    检查手部顶点有效性。

    Returns:
        (is_valid, reason)  reason 仅在 is_valid=False 时有意义
    """
    if hand_verts is None:
        return False, "hand_verts is None"
    if hand_verts.size == 0:
        return False, "hand_verts is empty"
    if hand_verts.ndim != 2 or hand_verts.shape[1] != 3:
        return False, f"wrong shape {hand_verts.shape}, expected ({MANO_VERTS}, 3)"
    if hand_verts.shape[0] != MANO_VERTS:
        return False, f"wrong vertex count {hand_verts.shape[0]}, expected {MANO_VERTS}"
    if np.any(np.isnan(hand_verts)):
        n_nan = np.isnan(hand_verts).sum()
        return False, f"contains {n_nan} NaN values"
    if np.any(np.isinf(hand_verts)):
        n_inf = np.isinf(hand_verts).sum()
        return False, f"contains {n_inf} Inf values"
    # 坐标范围检查：手部顶点不应超过 ±2m（单位为米时合理范围）
    coord_max = np.abs(hand_verts).max()
    if coord_max > 2.0:
        return False, f"coordinate magnitude too large: {coord_max:.2f}m (>2m)"
    return True, ""


def check_obj_mesh(obj_mesh: trimesh.Trimesh, sample_id: str, tag: str) -> tuple[bool, str]:
    """
    检查物体 mesh 有效性。

    Args:
        tag: 用于日志区分（如 "watertight"、"vhacd"）
    """
    if obj_mesh is None:
        return False, f"{tag} mesh is None"
    verts = np.asarray(obj_mesh.vertices, dtype=np.float32)
    faces = np.asarray(obj_mesh.faces, dtype=np.int32)
    if verts.size == 0:
        return False, f"{tag} mesh has no vertices"
    if faces.size == 0:
        return False, f"{tag} mesh has no faces"
    if np.any(np.isnan(verts)) or np.any(np.isinf(verts)):
        return False, f"{tag} mesh vertices contain NaN/Inf"
    # 面片索引越界检查
    if faces.max() >= len(verts):
        return False, f"{tag} mesh face index out of bounds"
    return True, ""


def check_obj_vox(obj_vox, sample_id: str) -> tuple[bool, str]:
    """检查体素化物体有效性"""
    if obj_vox is None:
        return False, "voxel is None"
    points = np.asarray(obj_vox.points, dtype=np.float32)
    if points.size == 0:
        return False, "voxel has no points"
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return False, "voxel points contain NaN/Inf"
    if obj_vox.element_volume <= 0:
        return False, f"invalid element_volume={obj_vox.element_volume}"
    return True, ""


def _log_skip(sample_id: str, reason: str):
    logger.warning(f"  [{sample_id}] Skipping: {reason}")


def min_surface_distance(obj_mesh: trimesh.Trimesh, query_points: np.ndarray) -> float:
    """
    计算点集到物体表面的最小距离（米）。

    优先使用 trimesh.proximity.closest_point 的精确表面距离；
    若 trimesh proximity 依赖缺失或失败，则退化为物体顶点 KDTree 最近邻距离。
    """
    query_points = np.asarray(query_points, dtype=np.float32)

    if len(obj_mesh.faces) > 0:
        try:
            # trimesh.closest_point 在退化三角形上可能触发 RuntimeWarning；
            # 这类样本直接退化到 KDTree 顶点最近邻，避免评估刷屏。
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=RuntimeWarning)
                _, dist, _ = trimesh.proximity.closest_point(obj_mesh, query_points)
            dist = np.asarray(dist, dtype=np.float32)
            if dist.size > 0 and np.all(np.isfinite(dist)):
                return float(dist.min())
        except Exception:
            pass

    verts = np.asarray(obj_mesh.vertices, dtype=np.float32)
    if verts.size == 0:
        return float("inf")
    tree = cKDTree(verts)
    dist, _ = tree.query(query_points)
    dist = np.asarray(dist, dtype=np.float32)
    return float(dist.min()) if dist.size > 0 else float("inf")


# ==============================================================================
# 语义指标
# ==============================================================================

def compute_token_recall_f1(pred_matrix: np.ndarray, gt_matrix: np.ndarray) -> tuple[float, float]:
    """Compute token recall/F1 directly from predicted vs GT contact matrices."""
    pred_hist = (np.asarray(pred_matrix) >= 0).reshape(-1).astype(np.float32)
    gt_hist = (np.asarray(gt_matrix) >= 0).reshape(-1).astype(np.float32)

    tp = float((pred_hist * gt_hist).sum())
    pred_pos = float(pred_hist.sum())
    gt_pos = float(gt_hist.sum())

    precision = tp / pred_pos if pred_pos > 0 else 0.0
    recall = tp / gt_pos if gt_pos > 0 else 0.0

    denom = precision + recall
    if denom <= 0:
        return float(recall), 0.0
    return float(recall), float(2.0 * precision * recall / denom)


class SemanticMetricHelper:
    """Recover contact semantics from the final hand pose and compare with tokens."""

    def __init__(
        self,
        config_path: str,
        mano_assets_root: str,
        bps_path: str,
        bps_slot_cache_dir: str,
        split: str,
        contact_distance: float,
    ):
        self.token_encoder = TokenEncoder(config_path)
        self.num_hand_parts = self.token_encoder.num_hand_parts
        self.num_slots = self.token_encoder.num_slots
        self.contact_distance = float(contact_distance)

        slot_dir = os.path.join(bps_slot_cache_dir, split)
        if not os.path.isdir(slot_dir):
            raise FileNotFoundError(f"BPS slot cache dir not found: {slot_dir}")
        self.slot_dir = slot_dir

        if not os.path.exists(bps_path):
            raise FileNotFoundError(f"BPS basis not found: {bps_path}")
        bps_data = np.load(bps_path)
        self.bps_basis = bps_data["basis"].astype(np.float32)
        if self.bps_basis.ndim == 3:
            self.bps_basis = self.bps_basis.squeeze(0)

        mano = MANOHelper(mano_assets_root, device="cpu")
        self.vert_to_part = self._build_vert_to_part(mano)
        self._slot_cache = {}

    def _build_vert_to_part(self, mano: MANOHelper) -> np.ndarray:
        weights = mano.mano.th_weights.detach().cpu().numpy()  # (778, 16)
        dominant_joint = weights.argmax(axis=1)

        hand_parts = self.token_encoder.mapper.hand_parts
        part_name_to_id = {name: i for i, name in enumerate(hand_parts)}
        joint_to_part_name = self.token_encoder.mapper.joint_to_part
        default_part = part_name_to_id.get("PALM", len(hand_parts) - 1)

        vert_to_part = np.empty(len(dominant_joint), dtype=np.int64)
        for i, joint_id in enumerate(dominant_joint):
            part_name = joint_to_part_name.get(int(joint_id), "PALM")
            vert_to_part[i] = part_name_to_id.get(part_name, default_part)
        return vert_to_part

    def _load_obj_slot_data(self, obj_id: str):
        if obj_id in self._slot_cache:
            return self._slot_cache[obj_id]

        npz_path = os.path.join(self.slot_dir, f"{obj_id}.npz")
        if not os.path.exists(npz_path):
            self._slot_cache[obj_id] = None
            return None

        with np.load(npz_path, allow_pickle=True) as data:
            if "bps_slot_labels" not in data.files or "obj_scale" not in data.files:
                self._slot_cache[obj_id] = None
                return None

            result = {
                "bps_slot_labels": data["bps_slot_labels"].astype(np.int64),
                "obj_scale": float(data["obj_scale"]),
            }
            self._slot_cache[obj_id] = result
            return result

    def propagate_slot_labels(self, obj_points: np.ndarray, obj_scale: float, bps_slot_labels: np.ndarray) -> np.ndarray:
        basis_tree = cKDTree(self.bps_basis)
        obj_norm = np.asarray(obj_points, dtype=np.float32) / max(float(obj_scale), 1e-6)
        _, nearest_basis = basis_tree.query(obj_norm, k=1)
        return np.asarray(bps_slot_labels, dtype=np.int64)[nearest_basis]

    def recover_contact_matrix(
        self,
        hand_verts: np.ndarray,
        obj_points: np.ndarray,
        obj_slot_labels: np.ndarray,
    ) -> np.ndarray:
        tree = cKDTree(np.asarray(obj_points, dtype=np.float32))
        dist, nearest_idx = tree.query(np.asarray(hand_verts, dtype=np.float32), k=1)
        nearest_slots = np.asarray(obj_slot_labels, dtype=np.int64)[nearest_idx]

        contact_matrix = np.full(
            (self.num_hand_parts, self.num_slots),
            -1,
            dtype=np.int32,
        )

        contact_mask = (dist < self.contact_distance) & (nearest_slots >= 0)
        for vid in np.flatnonzero(contact_mask):
            part_id = int(self.vert_to_part[vid])
            slot_id = int(nearest_slots[vid])
            if 0 <= part_id < self.num_hand_parts and 0 <= slot_id < self.num_slots:
                contact_matrix[part_id, slot_id] = 0

        return contact_matrix

    def compute_metrics(self, item: dict, hand_verts: np.ndarray, obj_points: np.ndarray):
        tokens = np.asarray(item.get("tokens", np.array([])), dtype=np.int64)
        if tokens.size == 0:
            return None

        slot_data = self._load_obj_slot_data(item["obj_id"])
        if slot_data is None:
            return None

        gt_matrix = self.token_encoder.decode(tokens)
        obj_slot_labels = self.propagate_slot_labels(
            obj_points=np.asarray(obj_points, dtype=np.float32),
            obj_scale=slot_data["obj_scale"],
            bps_slot_labels=slot_data["bps_slot_labels"],
        )
        pred_matrix = self.recover_contact_matrix(
            hand_verts=np.asarray(hand_verts, dtype=np.float32),
            obj_points=np.asarray(obj_points, dtype=np.float32),
            obj_slot_labels=obj_slot_labels,
        )

        token_recall, token_f1 = compute_token_recall_f1(pred_matrix, gt_matrix)
        return {
            "token_f1": token_f1,
            "token_recall": token_recall,
        }


# ==============================================================================
# 数据加载
# ==============================================================================

class GraspResultLoader:
    """加载 generate_grasps.py 的输出 + 预处理物体文件"""

    def __init__(self, results_dir, proc_dir):
        self.results_dir = results_dir
        self.proc_dir = proc_dir

        grasp_fnames = sorted(
            f for f in os.listdir(results_dir) if f.endswith(".pkl")
        )
        self.grasp_files = [
            os.path.join(results_dir, f) for f in grasp_fnames
        ]

        with open("assets/closed_mano_faces.pkl", "rb") as f:
            faces = pickle.load(f)
        self.hand_wt_faces = np.asarray(faces, dtype=np.int32)

        self._check_missing_objects()

    def _check_missing_objects(self):
        logger.info("Checking object preprocessing files...")
        missing_objs = {}

        for grasp_file in self.grasp_files:
            with open(grasp_file, "rb") as f:
                item = pickle.load(f)
            obj_id = item["obj_id"]
            if obj_id in missing_objs:
                continue

            missing_types = []
            for sub in ("watertight", "voxel", "vhacd"):
                ext = ".obj" if sub != "voxel" else ".binvox"
                path = os.path.join(self.proc_dir, sub, f"{obj_id}{ext}")
                if not os.path.exists(path):
                    missing_types.append(sub)

            if missing_types:
                missing_objs[obj_id] = missing_types

        if missing_objs:
            logger.warning(f"Found {len(missing_objs)} objects with missing files:")
            for obj_id, types in sorted(missing_objs.items())[:10]:
                logger.warning(f"  {obj_id}: missing {', '.join(types)}")
            if len(missing_objs) > 10:
                logger.warning(f"  ... and {len(missing_objs) - 10} more")
            logger.warning("These samples will be SKIPPED.\n")
        else:
            logger.info("All object files present.\n")

    def __len__(self):
        return len(self.grasp_files)

    def __getitem__(self, idx):
        grasp_file = self.grasp_files[idx]
        with open(grasp_file, "rb") as f:
            item = pickle.load(f)

        sample_id = os.path.splitext(os.path.basename(grasp_file))[0]
        item["sample_id"] = sample_id
        obj_id = item["obj_id"]

        obj_wt_path    = os.path.join(self.proc_dir, "watertight", f"{obj_id}.obj")
        obj_vox_path   = os.path.join(self.proc_dir, "voxel",      f"{obj_id}.binvox")
        obj_vhacd_path = os.path.join(self.proc_dir, "vhacd",      f"{obj_id}.obj")

        # 文件存在性检查
        for p in (obj_wt_path, obj_vox_path, obj_vhacd_path):
            if not os.path.exists(p):
                return None, f"missing file: {p}"

        # 加载物体资源
        try:
            obj_wt  = trimesh.load(obj_wt_path,  process=False)
            obj_vox = trimesh.load(obj_vox_path)
        except Exception as e:
            return None, f"trimesh load failed: {e}"

        # 物体 mesh 有效性
        ok, reason = check_obj_mesh(obj_wt, sample_id, "watertight")
        if not ok:
            return None, reason

        ok, reason = check_obj_vox(obj_vox, sample_id)
        if not ok:
            return None, reason

        item["obj_wt"]         = obj_wt
        item["obj_vox"]        = obj_vox
        item["obj_vhacd_path"] = obj_vhacd_path
        item["hand_faces"]     = self.hand_wt_faces
        return item, ""


# ==============================================================================
# 单样本评估
# ==============================================================================

def evaluate_single_grasp(idx, loader, sims_dir, contact_thresh, semantic_helper=None):
    """
    评估单个抓握: penetration + intersection + simulation

    Returns:
        dict  成功
        None  跳过，原因已记录
    """
    item, load_reason = loader[idx]

    # 加载失败
    if item is None:
        if load_reason:
            sample_id = os.path.splitext(
                os.path.basename(loader.grasp_files[idx])
            )[0]
            _log_skip(sample_id, load_reason)
        return None

    sample_id = item["sample_id"]

    # ---- 手部顶点有效性 ----
    hand_verts = np.asarray(item.get("hand_verts_r", np.array([])), dtype=np.float32)
    ok, reason = check_hand_verts(hand_verts, sample_id)
    if not ok:
        _log_skip(sample_id, f"hand_verts invalid: {reason}")
        return None

    # ---- 提取物体数据 ----
    obj_wt         = item["obj_wt"]
    obj_vox        = item["obj_vox"]
    obj_vhacd_path = item["obj_vhacd_path"]
    obj_rotmat     = item.get("obj_rotmat", np.eye(3))
    hand_faces     = item["hand_faces"]

    obj_wt_verts      = np.asarray(obj_wt.vertices,       dtype=np.float32)
    obj_wt_faces      = np.asarray(obj_wt.faces,          dtype=np.int32)
    obj_vox_points    = np.asarray(obj_vox.points,        dtype=np.float32)
    obj_element_volume = obj_vox.element_volume

    semantic_metrics = {
        "token_f1": None,
        "token_recall": None,
    }
    if semantic_helper is not None:
        try:
            semantic_result = semantic_helper.compute_metrics(
                item=item,
                hand_verts=hand_verts,
                obj_points=obj_wt_verts,
            )
            if semantic_result is not None:
                semantic_metrics.update(semantic_result)
        except Exception as e:
            logger.warning(f"  [{sample_id}] Semantic metrics unavailable: {e}")

    # ---- 穿透深度 ----
    try:
        pentr_dep = penetration(
            obj_verts=obj_wt_verts,
            obj_faces=obj_wt_faces,
            hand_verts=hand_verts,
        )
    except Exception as e:
        _log_skip(sample_id, f"penetration failed: {e}")
        return None

    if not np.isfinite(pentr_dep):
        _log_skip(sample_id, f"penetration returned non-finite: {pentr_dep}")
        return None

    # ---- 交叉体积 ----
    try:
        pentr_vol = solid_intersection_volume(
            hand_verts=hand_verts,
            hand_faces=hand_faces,
            obj_vox_points=obj_vox_points,
            obj_vox_el_vol=obj_element_volume,
            return_kin=False,
        )
    except Exception as e:
        _log_skip(sample_id, f"intersection_volume failed: {e}")
        return None

    if not np.isfinite(pentr_vol):
        _log_skip(sample_id, f"intersection_volume returned non-finite: {pentr_vol}")
        return None

    # ---- 物理仿真位移 ----
    try:
        sims_disp = simulation_sample(
            sample_idx=idx,
            sample_info={
                "sample_id":      sample_id,
                "hand_verts":     hand_verts,
                "hand_faces":     hand_faces,
                "obj_verts":      obj_wt_verts,
                "obj_faces":      obj_wt_faces,
                "obj_vhacd_fname": obj_vhacd_path,
                "obj_rotmat":     obj_rotmat,
            },
            save_gif_folder=os.path.join(sims_dir, "gif"),
            save_obj_folder=os.path.join(sims_dir, "vhacd"),
            tmp_folder=os.path.join(sims_dir, "tmp"),
            use_gui=False,
            sample_vis_freq=1,
        )
    except Exception as e:
        _log_skip(sample_id, f"simulation failed: {e}")
        return None

    if not np.isfinite(sims_disp):
        _log_skip(sample_id, f"simulation returned non-finite: {sims_disp}")
        return None

    try:
        min_contact_dist = min_surface_distance(obj_wt, hand_verts)
    except Exception as e:
        _log_skip(sample_id, f"surface contact distance failed: {e}")
        return None

    if not np.isfinite(min_contact_dist):
        _log_skip(sample_id, f"surface contact distance returned non-finite: {min_contact_dist}")
        return None

    return {
        "sample_id": sample_id,
        "obj_id":    item["obj_id"],
        "pentr_dep": float(pentr_dep),
        "pentr_vol": float(pentr_vol),
        "sims_disp": float(sims_disp),
        "min_contact_dist": float(min_contact_dist),
        "near_contact": bool(min_contact_dist <= contact_thresh),
        "token_f1": semantic_metrics["token_f1"],
        "token_recall": semantic_metrics["token_recall"],
    }


# ==============================================================================
# 汇总
# ==============================================================================

def evaluate(args):
    results_dir = os.path.join(args.exp_path, "results")
    sims_dir    = os.path.join(args.exp_path, "simulation")
    eval_dir    = os.path.join(args.exp_path, "evaluations")
    os.makedirs(sims_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    loader  = GraspResultLoader(results_dir, args.proc_dir)
    n_total = len(loader)
    logger.info(f"Evaluating {n_total} grasps (n_jobs={args.n_jobs})...")

    semantic_helper = None
    try:
        semantic_helper = SemanticMetricHelper(
            config_path=args.config_path,
            mano_assets_root=args.mano_assets_root,
            bps_path=args.bps_path,
            bps_slot_cache_dir=args.bps_slot_cache_dir,
            split=args.split,
            contact_distance=args.contact_thresh,
        )
        logger.info(
            f"Semantic metrics enabled: split={args.split}, "
            f"bps_slot_cache={os.path.join(args.bps_slot_cache_dir, args.split)}"
        )
    except Exception as e:
        logger.warning(f"Semantic metrics disabled: {e}")

    # 近表面接触阈值（米）：
    # 以 hand vertex 到 object watertight surface 的最小距离定义接触。
    tasks = [
        delayed(evaluate_single_grasp)(i, loader, sims_dir, args.contact_thresh, semantic_helper)
        for i in range(n_total)
    ]
    raw_results = Parallel(n_jobs=args.n_jobs, verbose=10)(tasks)

    valid     = [r for r in raw_results if r is not None]
    n_skipped = n_total - len(valid)

    # ---- skip 原因统计（按物体 ID 聚合）----
    skipped_by_obj = defaultdict(int)
    for r, raw in zip(raw_results, raw_results):
        pass  # 原因已在 worker 中打印，此处仅统计数量

    # ---- 聚合指标 ----
    pentr_dep_meter = AverageMeter("pentr_dep")
    pentr_vol_meter = AverageMeter("pentr_vol")
    sims_disp_meter = AverageMeter("sims_disp")
    sims_disp_list  = []
    min_contact_dist_meter = AverageMeter("min_contact_dist")
    contact_count   = 0
    token_f1_list = []
    token_recall_list = []

    for r in valid:
        pentr_dep_meter.update(r["pentr_dep"])
        pentr_vol_meter.update(r["pentr_vol"])
        sims_disp_meter.update(r["sims_disp"])
        sims_disp_list.append(r["sims_disp"])
        min_contact_dist_meter.update(r["min_contact_dist"])
        if r["near_contact"]:
            contact_count += 1
        if r.get("token_f1") is not None:
            token_f1_list.append(float(r["token_f1"]))
        if r.get("token_recall") is not None:
            token_recall_list.append(float(r["token_recall"]))

    n_valid        = len(valid)
    contact_ratio  = contact_count / n_valid if n_valid > 0 else 0.0
    skip_ratio     = n_skipped / n_total if n_total > 0 else 0.0

    # 单位换算 m → cm
    s, s3 = 100.0, 100.0 ** 3
    pentr_dep_cm  = pentr_dep_meter.avg * s
    pentr_vol_cm3 = pentr_vol_meter.avg * s3
    sims_disp_cm  = sims_disp_meter.avg * s
    sims_disp_std = float(np.std(sims_disp_list)) * s if sims_disp_list else 0.0
    min_contact_dist_mm = min_contact_dist_meter.avg * 1000.0
    token_f1 = (
        float(np.mean(token_f1_list)) if token_f1_list else None
    )
    token_recall = (
        float(np.mean(token_recall_list)) if token_recall_list else None
    )

    # ---- 保存详细结果 ----
    with open(os.path.join(eval_dir, "eval_results.pkl"), "wb") as f:
        pickle.dump(valid, f)

    # ---- 打印 ----
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  EVALUATION RESULTS")
    print(f"{sep}")
    print(f"  Total samples:           {n_total}")
    print(f"  Valid samples:           {n_valid}")
    print(f"  Skipped:                 {n_skipped}  ({skip_ratio*100:.1f}%)")
    print()
    print(f"  Penetration Depth (cm):  {pentr_dep_cm:.6f}")
    print(f"  Intersection Vol (cm³):  {pentr_vol_cm3:.6f}")
    print(f"  Sim Displacement (cm):   {sims_disp_cm:.6f} ± {sims_disp_std:.6f}")
    print(f"  Contact Ratio (<= {args.contact_thresh * 1000:.1f}mm): {contact_ratio:.4f}")
    print(f"  Min Surface Dist (mm):   {min_contact_dist_mm:.6f}")
    if token_f1 is not None:
        print(f"  Token F1:               {token_f1:.4f}")
    else:
        print(f"  Token F1:               N/A")
    if token_recall is not None:
        print(f"  Token Recall:           {token_recall:.4f}")
    else:
        print(f"  Token Recall:           N/A")
    print(f"{sep}")

    # ---- 保存文本报告 ----
    report_path = os.path.join(eval_dir, "Metric.txt")
    with open(report_path, "w") as f:
        f.write(f"Total samples:                   {n_total}\n")
        f.write(f"Valid samples:                   {n_valid}\n")
        f.write(f"Skipped samples:                 {n_skipped} ({skip_ratio*100:.1f}%)\n\n")
        f.write(f"Penetration Depth (mean, cm):    {pentr_dep_cm:.6f}\n")
        f.write(f"Intersection Volume (mean, cm³): {pentr_vol_cm3:.6f}\n")
        f.write(f"Sim Displacement (mean, cm):     {sims_disp_cm:.6f}\n")
        f.write(f"Sim Displacement (std, cm):      {sims_disp_std:.6f}\n")
        f.write(f"Contact Ratio (<= {args.contact_thresh * 1000:.1f}mm): {contact_ratio:.4f}\n")
        f.write(f"Min Surface Dist (mean, mm):     {min_contact_dist_mm:.6f}\n")
        if token_f1 is not None:
            f.write(f"Token F1 (mean):                 {token_f1:.4f}\n")
        else:
            f.write("Token F1 (mean):                 N/A\n")
        if token_recall is not None:
            f.write(f"Token Recall (mean):             {token_recall:.4f}\n")
        else:
            f.write("Token Recall (mean):             N/A\n")

    logger.info(f"Results saved to: {eval_dir}")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate generated grasps")
    parser.add_argument("--exp_path", type=str, required=True,
                        help="Experiment directory (contains 'results/')")
    parser.add_argument("--proc_dir", type=str,
                        default="cache/OakInkShape_object_process",
                        help="Preprocessed objects dir (watertight/ voxel/ vhacd/)")
    parser.add_argument("--n_jobs",   type=int, default=4)
    parser.add_argument("--contact_thresh", type=float, default=0.005,
                        help="Near-surface contact threshold in meters (default: 0.005 = 5mm)")
    
    parser.add_argument("--split", type=str, default="test",
                        help="Split used to find BPS slot cache (default: test)")
    parser.add_argument("--config_path", type=str, default="configs/token_config.yaml",
                        help="Path to token config for decoding tokens")
    parser.add_argument("--bps_path", type=str, default="configs/bps.npz",
                        help="Path to BPS basis (.npz)")
    parser.add_argument("--bps_slot_cache_dir", type=str, default="cache/bps_slot",
                        help="Directory containing per-object BPS slot cache")
    parser.add_argument("--mano_assets_root", type=str, default="assets/mano_v1_2",
                        help="MANO assets root for building vertex-to-hand-part mapping")
    parser.add_argument("-g", "--gpu_id", type=str, default="1")
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"]    = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    evaluate(args)
