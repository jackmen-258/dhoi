#!/usr/bin/env python3
"""
evaluate_grasps.py — 评估生成抓握的物理质量

指标:
  - Penetration Depth (cm):       手穿入物体的深度, 越低越好
  - Intersection Volume (cm³):    手-物体交叉体积, 越低越好
  - Simulation Displacement (cm): 物理仿真中物体位移, 越低越好
  - Contact Ratio:                近表面接触样本比例, 越高越好

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
from joblib import Parallel, delayed
from scipy.spatial import cKDTree

from metrics.basic_metric import AverageMeter
from metrics.penetration import penetration
from metrics.intersection import solid_intersection_volume
from metrics.simulator import simulation_sample

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

def evaluate_single_grasp(idx, loader, sims_dir, contact_thresh):
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

    # 近表面接触阈值（米）：
    # 以 hand vertex 到 object watertight surface 的最小距离定义接触。
    tasks = [
        delayed(evaluate_single_grasp)(i, loader, sims_dir, args.contact_thresh)
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

    for r in valid:
        pentr_dep_meter.update(r["pentr_dep"])
        pentr_vol_meter.update(r["pentr_vol"])
        sims_disp_meter.update(r["sims_disp"])
        sims_disp_list.append(r["sims_disp"])
        min_contact_dist_meter.update(r["min_contact_dist"])
        if r["near_contact"]:
            contact_count += 1

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
    parser.add_argument("--n_jobs",   type=int, default=8)
    parser.add_argument("--contact_thresh", type=float, default=0.005,
                        help="Near-surface contact threshold in meters (default: 0.005 = 5mm)")
    parser.add_argument("-g", "--gpu_id", type=str, default="0")
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"]    = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    evaluate(args)
