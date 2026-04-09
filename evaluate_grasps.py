#!/usr/bin/env python3
"""
evaluate_grasps.py — 评估生成抓握的物理质量 + 语义一致性

指标:
  - Penetration Depth (cm):       手穿入物体的深度, 越低越好
  - Intersection Volume (cm³):    手-物体交叉体积, 越低越好
  - Simulation Displacement (cm): 物理仿真中物体位移, 越低越好
  - Contact Ratio:                近表面接触样本比例, 越高越好
  - Part Accuracy:                Text2Grasp-style dominant contacted part accuracy
  - Part Ratio:                   接触点中落在目标 part/slot 上的比例

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
import json
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
from data.text_generator import SLOT_NAME_BY_CATEGORY, DEFAULT_SLOT_NAME
from metrics.basic_metric import AverageMeter
from metrics.diversity import diversity_details, joints_to_diversity_feature
from metrics.penetration import penetration
from metrics.intersection import solid_intersection_volume
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


class SemanticMetricHelper:
    """Recover Text2Grasp-style target part accuracy from final hand pose."""

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

    def recover_contact_slot_counts(
        self,
        hand_verts: np.ndarray,
        obj_points: np.ndarray,
        obj_slot_labels: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Count object-side contacts per slot using nearest hand-vertex distance.

        This follows the spirit of Text2Grasp's part-accuracy metric:
        determine which object part/slot receives the most contact points.
        """
        hand_tree = cKDTree(np.asarray(hand_verts, dtype=np.float32))
        dist, _ = hand_tree.query(np.asarray(obj_points, dtype=np.float32), k=1)
        slots = np.asarray(obj_slot_labels, dtype=np.int64)
        valid_mask = slots >= 0
        contact_mask = (dist < self.contact_distance) & valid_mask

        if not np.any(contact_mask):
            return np.zeros(self.num_slots, dtype=np.int64), 0

        contact_slots = slots[contact_mask]
        counts = np.bincount(contact_slots, minlength=self.num_slots).astype(np.int64)
        return counts, int(contact_mask.sum())

    def _phrase_candidates_for_slot(self, cate_id: str, slot_name: str) -> list[str]:
        cate_key = str(cate_id or "").lower().strip()
        cate_map = SLOT_NAME_BY_CATEGORY.get(cate_key, {})
        base_phrase = str(cate_map.get(slot_name, DEFAULT_SLOT_NAME.get(slot_name, slot_name.lower()))).lower().strip()

        candidates = [base_phrase]
        if " or " in base_phrase:
            candidates.extend(part.strip() for part in base_phrase.split(" or "))
        if " and " in base_phrase:
            candidates.extend(part.strip() for part in base_phrase.split(" and "))

        extras = {
            "GRASP_SURFACE": ["body", "handle", "grip", "barrel"],
            "FUNCTIONAL_END": ["blade", "tip", "nozzle", "opening", "spout", "lens", "head"],
            "CONTROL": ["trigger", "button", "pump top", "control panel", "control area"],
            "CLOSURE": ["cap", "lid"],
        }
        candidates.extend(extras.get(slot_name, []))
        return sorted({c.strip() for c in candidates if c and len(c.strip()) >= 3}, key=len, reverse=True)

    def infer_target_from_text(self, item: dict) -> tuple[np.ndarray | None, int | None]:
        explicit_targets = item.get("text_target_slots", None)
        explicit_primary = item.get("text_primary_slot", None)

        if explicit_targets is not None:
            target_slots = np.asarray(explicit_targets, dtype=np.int64).reshape(-1)
            target_slots = target_slots[(target_slots >= 0) & (target_slots < self.num_slots)]
            if target_slots.size > 0:
                primary_slot = None
                if explicit_primary is not None:
                    primary_slot = int(explicit_primary)
                    if not (0 <= primary_slot < self.num_slots):
                        primary_slot = None
                if primary_slot is None:
                    primary_slot = int(target_slots[0])
                return target_slots, primary_slot

        text = str(item.get("text_prompt", item.get("text", "")) or "").lower().strip()
        if not text:
            return None, None

        cate_id = item.get("class_name", item.get("cate_id", ""))
        hits = []
        for slot_id, slot_name in enumerate(self.token_encoder.mapper.canonical_slots):
            for phrase in self._phrase_candidates_for_slot(cate_id, slot_name):
                pos = text.find(phrase)
                if pos >= 0:
                    hits.append((pos, slot_id))
                    break

        if not hits:
            return None, None

        hits.sort(key=lambda x: x[0])
        target_slots = np.asarray(sorted({slot_id for _, slot_id in hits}), dtype=np.int64)
        primary_slot = int(hits[0][1])
        return target_slots, primary_slot

    def compute_metrics(self, item: dict, hand_verts: np.ndarray, obj_points: np.ndarray):
        slot_data = self._load_obj_slot_data(item["obj_id"])
        if slot_data is None:
            return None

        obj_slot_labels = self.propagate_slot_labels(
            obj_points=np.asarray(obj_points, dtype=np.float32),
            obj_scale=slot_data["obj_scale"],
            bps_slot_labels=slot_data["bps_slot_labels"],
        )
        target_slots, primary_slot = self.infer_target_from_text(item)
        if target_slots is None or target_slots.size == 0 or primary_slot is None:
            return None

        slot_contact_counts, n_contact = self.recover_contact_slot_counts(
            hand_verts=np.asarray(hand_verts, dtype=np.float32),
            obj_points=np.asarray(obj_points, dtype=np.float32),
            obj_slot_labels=obj_slot_labels,
        )

        if n_contact <= 0:
            return {
                "part_acc": 0.0,
                "part_ratio": 0.0,
                "target_slots": target_slots.tolist(),
                "dominant_part": None,
            }

        dominant_slot = int(slot_contact_counts.argmax())
        part_acc = 1.0 if dominant_slot == int(primary_slot) else 0.0
        part_ratio = float(slot_contact_counts[target_slots].sum()) / float(n_contact)
        return {
            "part_acc": float(part_acc),
            "part_ratio": float(part_ratio),
            "target_slots": target_slots.tolist(),
            "primary_slot": int(primary_slot),
            "dominant_part": dominant_slot,
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

    from metrics.simulator import simulation_sample

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
        "part_acc": None,
        "part_ratio": None,
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
        "part_acc": semantic_metrics["part_acc"],
        "part_ratio": semantic_metrics["part_ratio"],
    }


# ==============================================================================
# 汇总
# ==============================================================================

def _diversity_group_key(item: dict, group_by: str) -> str:
    obj_id = str(item.get("obj_id", "unknown_obj"))

    if group_by == "group_id":
        return str(item.get("group_id", item.get("manifest_key", f"{obj_id}::{item.get('sample_idx', 'unknown')}")))
    if group_by == "obj_id":
        return obj_id
    if group_by == "obj_sample":
        sample_idx = item.get("sample_idx", "unknown")
        return f"{obj_id}::{sample_idx}"
    raise ValueError(f"Unsupported diversity_group_by: {group_by}")


def evaluate_diversity(args):
    results_dir = os.path.join(args.exp_path, "results")
    eval_dir = os.path.join(args.exp_path, "evaluations_diversity")
    os.makedirs(eval_dir, exist_ok=True)

    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    grasp_files = sorted(
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith(".pkl")
    )
    logger.info(f"Diversity mode: loading {len(grasp_files)} result files...")

    grouped_features = defaultdict(list)
    grouped_meta = {}
    all_features = []
    skipped_load = 0
    skipped_feature = 0

    for path in grasp_files:
        try:
            with open(path, "rb") as f:
                item = pickle.load(f)
        except Exception as e:
            logger.warning(f"Skipping unreadable result {path}: {e}")
            skipped_load += 1
            continue

        feat = joints_to_diversity_feature(
            item.get("hand_joints_r", None),
            unit=args.diversity_unit,
            canonical=args.diversity_canonical,
        )
        if feat is None:
            skipped_feature += 1
            continue

        group_key = _diversity_group_key(item, args.diversity_group_by)
        grouped_features[group_key].append(feat)
        all_features.append(feat)
        if group_key not in grouped_meta:
            grouped_meta[group_key] = {
                "group_id": group_key,
                "obj_id": item.get("obj_id"),
                "sample_idx": item.get("sample_idx"),
                "manifest_key": item.get("manifest_key"),
                "class_name": item.get("class_name"),
                "afford_name": item.get("afford_name"),
            }

    total_groups = len(grouped_features)
    valid_group_records = []
    singleton_groups = 0
    valid_group_sizes = []

    for group_key, feats in grouped_features.items():
        if len(feats) < 2:
            singleton_groups += 1
            continue

        details = diversity_details(np.asarray(feats, dtype=np.float32), cls_num=args.diversity_cls_num)
        details.update(grouped_meta.get(group_key, {}))
        details["num_samples"] = len(feats)
        valid_group_records.append(details)
        valid_group_sizes.append(len(feats))

    global_details = diversity_details(np.asarray(all_features, dtype=np.float32), cls_num=args.diversity_cls_num)

    mean_entropy = float(np.mean([r["entropy"] for r in valid_group_records])) if valid_group_records else 0.0
    mean_cluster_size = float(np.mean([r["cluster_size"] for r in valid_group_records])) if valid_group_records else 0.0
    std_entropy = float(np.std([r["entropy"] for r in valid_group_records])) if valid_group_records else 0.0
    std_cluster_size = float(np.std([r["cluster_size"] for r in valid_group_records])) if valid_group_records else 0.0
    mean_cluster_spread = float(np.mean([r.get("cluster_spread", 0.0) for r in valid_group_records])) if valid_group_records else 0.0
    std_cluster_spread = float(np.std([r.get("cluster_spread", 0.0) for r in valid_group_records])) if valid_group_records else 0.0
    avg_group_size = float(np.mean(valid_group_sizes)) if valid_group_sizes else 0.0

    summary = {
        "total_files": len(grasp_files),
        "valid_features": len(all_features),
        "skipped_load": skipped_load,
        "skipped_feature": skipped_feature,
        "group_by": args.diversity_group_by,
        "groups_total": total_groups,
        "groups_valid": len(valid_group_records),
        "groups_skipped_singleton": singleton_groups,
        "avg_group_size": avg_group_size,
        "cls_num": int(args.diversity_cls_num),
        "unit": args.diversity_unit,
        "canonical": bool(args.diversity_canonical),
        "cluster_size_definition": "mean number of samples per active cluster after k-means",
        "cluster_spread_definition": f"mean distance to assigned k-means center ({args.diversity_unit})",
        "grouped_mean_entropy": mean_entropy,
        "grouped_std_entropy": std_entropy,
        "grouped_mean_cluster_size": mean_cluster_size,
        "grouped_std_cluster_size": std_cluster_size,
        "grouped_mean_cluster_spread": mean_cluster_spread,
        "grouped_std_cluster_spread": std_cluster_spread,
        "global_entropy": float(global_details["entropy"]),
        "global_cluster_size": float(global_details["cluster_size"]),
        "global_cluster_spread": float(global_details.get("cluster_spread", 0.0)),
        "records": valid_group_records,
    }

    with open(os.path.join(eval_dir, "diversity_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    sep = "=" * 60
    print(f"\n{sep}")
    print("  DIVERSITY RESULTS")
    print(f"{sep}")
    print(f"  Total result files:       {summary['total_files']}")
    print(f"  Valid features:           {summary['valid_features']}")
    print(f"  Skipped load/feature:     {summary['skipped_load']}/{summary['skipped_feature']}")
    print(f"  Group by:                 {summary['group_by']}")
    print(f"  Valid groups (>=2):       {summary['groups_valid']} / {summary['groups_total']}")
    print(f"  Avg valid group size:     {summary['avg_group_size']:.4f}")
    print(f"  Entropy (grouped mean):   {summary['grouped_mean_entropy']:.4f} ± {summary['grouped_std_entropy']:.4f}")
    print(f"  Cluster Size (grouped):   {summary['grouped_mean_cluster_size']:.4f} ± {summary['grouped_std_cluster_size']:.4f}")
    print(f"  Cluster Spread (grouped): {summary['grouped_mean_cluster_spread']:.4f} ± {summary['grouped_std_cluster_spread']:.4f}")
    print(f"  Entropy (global):         {summary['global_entropy']:.4f}")
    print(f"  Cluster Size (global):    {summary['global_cluster_size']:.4f}")
    print(f"  Cluster Spread (global):  {summary['global_cluster_spread']:.4f}")
    print(f"{sep}")

    report_path = os.path.join(eval_dir, "Metric.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Total result files:              {summary['total_files']}\n")
        f.write(f"Valid features:                  {summary['valid_features']}\n")
        f.write(f"Skipped load:                    {summary['skipped_load']}\n")
        f.write(f"Skipped invalid feature:         {summary['skipped_feature']}\n")
        f.write(f"Group By:                        {summary['group_by']}\n")
        f.write(f"Groups total:                    {summary['groups_total']}\n")
        f.write(f"Groups valid (>=2):              {summary['groups_valid']}\n")
        f.write(f"Groups skipped singleton:        {summary['groups_skipped_singleton']}\n")
        f.write(f"Avg valid group size:            {summary['avg_group_size']:.4f}\n")
        f.write(f"KMeans cls_num (max):            {summary['cls_num']}\n")
        f.write(f"Unit:                            {summary['unit']}\n")
        f.write(f"Canonical:                       {int(summary['canonical'])}\n")
        f.write(f"Cluster Size Definition:         {summary['cluster_size_definition']}\n")
        f.write(f"Cluster Spread Definition:       {summary['cluster_spread_definition']}\n")
        f.write(f"Entropy (grouped mean, ln):      {summary['grouped_mean_entropy']:.4f}\n")
        f.write(f"Entropy (grouped std, ln):       {summary['grouped_std_entropy']:.4f}\n")
        f.write(f"Cluster Size (grouped mean):     {summary['grouped_mean_cluster_size']:.4f}\n")
        f.write(f"Cluster Size (grouped std):      {summary['grouped_std_cluster_size']:.4f}\n")
        f.write(f"Cluster Spread (grouped mean):   {summary['grouped_mean_cluster_spread']:.4f}\n")
        f.write(f"Cluster Spread (grouped std):    {summary['grouped_std_cluster_spread']:.4f}\n")
        f.write(f"Entropy (global, ln):            {summary['global_entropy']:.4f}\n")
        f.write(f"Cluster Size (global):           {summary['global_cluster_size']:.4f}\n")
        f.write(f"Cluster Spread (global):         {summary['global_cluster_spread']:.4f}\n")

    logger.info(f"Diversity results saved to: {eval_dir}")


def evaluate_quality(args):
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
    part_acc_list = []
    part_ratio_list = []

    for r in valid:
        pentr_dep_meter.update(r["pentr_dep"])
        pentr_vol_meter.update(r["pentr_vol"])
        sims_disp_meter.update(r["sims_disp"])
        sims_disp_list.append(r["sims_disp"])
        min_contact_dist_meter.update(r["min_contact_dist"])
        if r["near_contact"]:
            contact_count += 1
        if r.get("part_acc") is not None:
            part_acc_list.append(float(r["part_acc"]))
        if r.get("part_ratio") is not None:
            part_ratio_list.append(float(r["part_ratio"]))

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
    part_acc = (
        float(np.mean(part_acc_list)) if part_acc_list else None
    )
    part_ratio = (
        float(np.mean(part_ratio_list)) if part_ratio_list else None
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
    if part_acc is not None:
        print(f"  Part Accuracy:          {part_acc:.4f}")
    else:
        print(f"  Part Accuracy:          N/A")
    if part_ratio is not None:
        print(f"  Part Ratio:             {part_ratio:.4f}")
    else:
        print(f"  Part Ratio:             N/A")
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
        if part_acc is not None:
            f.write(f"Part Accuracy (mean):            {part_acc:.4f}\n")
        else:
            f.write("Part Accuracy (mean):            N/A\n")
        if part_ratio is not None:
            f.write(f"Part Ratio (mean):               {part_ratio:.4f}\n")
        else:
            f.write("Part Ratio (mean):               N/A\n")

    logger.info(f"Results saved to: {eval_dir}")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate generated grasps")
    parser.add_argument("--mode", type=str, default="quality",
                        choices=["quality", "diversity"],
                        help="Evaluation mode")
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
    parser.add_argument("--diversity_cls_num", type=int, default=20,
                        help="Upper bound of KMeans cluster count for diversity mode")
    parser.add_argument("--diversity_unit", type=str, default="cm",
                        choices=["m", "cm", "mm"],
                        help="Unit scaling applied before diversity clustering")
    parser.add_argument("--diversity_canonical", action=argparse.BooleanOptionalAction, default=True,
                        help="Canonicalize joints before diversity clustering")
    parser.add_argument("--diversity_group_by", type=str, default="group_id",
                        choices=["group_id", "obj_sample", "obj_id"],
                        help="Grouping key for diversity mode")
    parser.add_argument("-g", "--gpu_id", type=str, default="1")
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"]    = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    if args.mode == "diversity":
        evaluate_diversity(args)
    else:
        evaluate_quality(args)
