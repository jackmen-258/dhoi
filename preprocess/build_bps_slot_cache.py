"""
build_bps_slot_cache.py
========================
预处理脚本：为每个物体计算 BPS 编码 + slot label 传播，保存到缓存。

缓存结构（按物体级别，同一 obj_id 只计算一次）:
    cache/bps_slot/{split}/{obj_id}.npz
        - bps_dists:       (4096,) float32  — basis→物体归一化距离
        - bps_slot_labels: (4096,) int8     — 每个 basis point 的 slot label
        - bps_nn_points:   (4096, 3) float32 — 每个 basis point 最近的归一化物体点
        - b2x_idxs:        (4096,) int32    — 最近物体顶点索引
        - obj_scale:        float32          — bbox 对角线 (归一化因子)
        - obj_center:       (3,) float32     — bbox 中心 (已在 OIShape 中减去)

用法:
    python preprocess/build_bps_slot_cache.py \\
        --split train --bps_path configs/bps.npz \\
        --save_dir cache/bps_slot

前置条件:
    1. 环境变量 OAKINK_DIR 指向 OakInk 数据根目录
    2. OakBase 下有物体 part 标注（part_XX.json + part_XX.ply）
    3. configs/bps.npz 存在（BPS basis 文件）
"""

import os
import sys
import json
import hashlib
import argparse
import logging
import time
import numpy as np
import trimesh
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional
from scipy.spatial import cKDTree

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.slot_mapping import SlotMapper
from data.bps_slot_encoder import normalize_object

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# 物体采样 (10000 点, 与 GrabNet OIShape 一致)
# ==============================================================================

def sample_object_points(
    obj_mesh: trimesh.Trimesh,
    n_samples: int = 10000,
    seed_str: str = "",
) -> np.ndarray:
    """确定性采样物体表面点。

    使用 subdivide + random sample 方式（与 GrabNet OIShape 一致）。

    Args:
        obj_mesh: bbox-centered 的物体网格
        n_samples: 采样点数
        seed_str: 确定性种子字符串

    Returns:
        (n_samples, 3) float32
    """
    seed = int(hashlib.md5(seed_str.encode("utf-8")).hexdigest(), 16) % (2**31)
    rng = np.random.RandomState(seed)

    mesh = obj_mesh.copy()
    while mesh.vertices.shape[0] < n_samples:
        mesh = mesh.subdivide()

    verts = np.array(mesh.vertices, dtype=np.float32)
    idx = rng.choice(verts.shape[0], n_samples, replace=False)
    return verts[idx]


# ==============================================================================
# Slot label 传播到采样点
# ==============================================================================

def compute_vertex_to_slot(
    sampled_points: np.ndarray,
    obj_mesh: trimesh.Trimesh,
    part_geometries: Dict[int, dict],
) -> np.ndarray:
    """为每个采样点计算 slot label。

    对每个 part 几何，计算采样点到 part 的最近距离，
    取最近的 part 的 slot_id。

    Args:
        sampled_points: (N, 3) 采样点
        obj_mesh: 完整物体网格（bbox-centered）
        part_geometries: {slot_id: {"surface": Optional[Trimesh], "points": (M, 3)}}

    Returns:
        (N,) int32, 每个点的 slot 标签 (-1 = 未分配)
    """
    n = len(sampled_points)
    slot_labels = np.full(n, -1, dtype=np.int32)
    best_dist = np.full(n, np.inf, dtype=np.float32)

    for slot_id, geom in part_geometries.items():
        part_points = np.asarray(geom.get("points", []), dtype=np.float32)
        if part_points.ndim != 2 or part_points.shape[0] == 0 or part_points.shape[1] != 3:
            continue

        dist = None
        part_surface = geom.get("surface")
        if part_surface is not None and hasattr(part_surface, "faces"):
            try:
                _, dist, _ = trimesh.proximity.closest_point(part_surface, sampled_points)
                dist = np.asarray(dist, dtype=np.float32)
            except Exception as e:
                logger.warning(
                    f"[compute_vertex_to_slot] closest_point failed for slot={slot_id}: {e}; "
                    "falling back to point-cloud nearest neighbors"
                )

        if dist is None:
            tree = cKDTree(part_points)
            dist, _ = tree.query(sampled_points)
            dist = dist.astype(np.float32)

        closer = dist < best_dist
        slot_labels[closer] = slot_id
        best_dist[closer] = dist[closer]

    slot_labels[best_dist == np.inf] = -1
    return slot_labels


def _geometry_vertices(geom) -> np.ndarray:
    """Extract vertices from a trimesh geometry or point cloud."""
    verts = getattr(geom, "vertices", None)
    if verts is None:
        return np.empty((0, 3), dtype=np.float32)
    verts = np.asarray(verts, dtype=np.float32)
    if verts.ndim != 2 or verts.shape[1] != 3:
        return np.empty((0, 3), dtype=np.float32)
    return verts


def _has_surface_faces(geom) -> bool:
    """Return True only for geometries with a usable face array."""
    faces = getattr(geom, "faces", None)
    return faces is not None and len(faces) > 0


def load_part_meshes(
    mapper: SlotMapper,
    obj_id: str,
    cate_id: str,
    oakbase_dir: str,
    obj_id_to_name: dict,
    obj_mesh: trimesh.Trimesh,
    raw_obj_id: str = None,
) -> Dict[int, dict]:
    """加载物体的 part 几何并按 slot 分组。

    OakBase 的 part_*.ply 在不同对象上可能是三角网格，也可能是 point cloud。
    这里统一保留:
      - surface: 可用的 Trimesh 表面（若存在）
      - points:  几何顶点/点云，始终可用于 cKDTree 最近邻回退
    """
    # 查找 part 目录
    part_dir = None
    candidate_ids = [obj_id]
    if raw_obj_id and raw_obj_id != obj_id:
        candidate_ids.append(raw_obj_id)

    for oid in candidate_ids:
        name = obj_id_to_name.get(oid)
        if name is None:
            continue
        d = os.path.join(oakbase_dir, cate_id, name)
        if os.path.isdir(d) and any(f.startswith("part_") and f.endswith(".json")
                                      for f in os.listdir(d)):
            part_dir = d
            break
        # 尝试其他类别目录
        if os.path.isdir(oakbase_dir):
            for subdir in os.listdir(oakbase_dir):
                alt = os.path.join(oakbase_dir, subdir, name)
                if os.path.isdir(alt) and any(f.startswith("part_") and f.endswith(".json")
                                               for f in os.listdir(alt)):
                    part_dir = alt
                    break
        if part_dir:
            break

    if part_dir is None:
        return {}

    parts = mapper.load_object_parts(part_dir, cate_id)
    bbox_center = (obj_mesh.vertices.min(0) + obj_mesh.vertices.max(0)) / 2

    slot_geometries = {}
    for _, info in parts.items():
        sid = info["slot_id"]
        ply = info["ply_path"]
        if sid is None or ply is None or not Path(ply).exists():
            continue
        # 不使用 force="mesh"：点云 PLY 会被错误地转为空 Trimesh
        geom = trimesh.load(ply, process=False)
        verts = np.asarray(geom.vertices, dtype=np.float32)
        if verts.ndim != 2 or verts.shape[0] == 0:
            continue

        verts = verts - bbox_center
        entry = slot_geometries.setdefault(sid, {"surface": None, "points": []})
        entry["points"].append(verts)

        # 仅当是真正的 Trimesh（有面片）时保留 surface
        if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0:
            geom.vertices = verts
            if entry["surface"] is None:
                entry["surface"] = geom
            else:
                try:
                    entry["surface"] = trimesh.util.concatenate([entry["surface"], geom])
                except Exception as e:
                    logger.warning(
                        f"[load_part_meshes] Failed to concatenate surfaces for obj_id={obj_id}, "
                        f"slot={sid}: {e}; using point cloud fallback only"
                    )
                    entry["surface"] = None

    merged = {}
    for sid, entry in slot_geometries.items():
        merged[sid] = {
            "surface": entry["surface"],
            "points": np.concatenate(entry["points"], axis=0) if entry["points"] else np.empty((0, 3), dtype=np.float32),
        }

    return merged


def encode_bps_slots(
    bps_basis: np.ndarray,
    obj_points_norm: np.ndarray,
    vertex_to_slot: np.ndarray,
    dist_threshold: float,
) -> dict:
    """Encode BPS distances, slot labels, and nearest normalized object points.

    Prefer the repository's ChamferDistance implementation when available so
    the behavior matches the rest of the project. If the runtime import is not
    ready yet, fall back to a cKDTree-based nearest-neighbor implementation.
    """
    basis = np.asarray(bps_basis, dtype=np.float32)
    if basis.ndim == 3:
        basis = basis.squeeze(0)

    points = np.asarray(obj_points_norm, dtype=np.float32)
    v2s = np.asarray(vertex_to_slot, dtype=np.int32)
    dists = None
    b2x_idx = None

    chamfer_ctor = None
    try:
        import chamfer_distance as chd
        chamfer_ctor = getattr(chd, "ChamferDistance", None)
    except Exception:
        chamfer_ctor = None

    if chamfer_ctor is None:
        try:
            from chamfer_distance.chamfer_distance import ChamferDistance as _ChamferDistance
            chamfer_ctor = _ChamferDistance
        except Exception:
            chamfer_ctor = None

    if chamfer_ctor is not None:
        try:
            import torch

            obj_t = torch.from_numpy(points).unsqueeze(0)
            basis_t = torch.from_numpy(basis).unsqueeze(0)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            obj_t = obj_t.to(device)
            basis_t = basis_t.to(device)

            ch = chamfer_ctor()
            _, _, b2x_idx_t, _ = ch(basis_t, obj_t)
            b2x_idx_t = b2x_idx_t.squeeze(0).long()
            nearest_pts = obj_t.squeeze(0)[b2x_idx_t]
            dists_t = (basis_t.squeeze(0) - nearest_pts).norm(dim=1)

            dists = dists_t.detach().cpu().numpy().astype(np.float32)
            b2x_idx = b2x_idx_t.detach().cpu().numpy().astype(np.int32)
        except Exception as e:
            logger.warning(
                f"[encode_bps_slots] ChamferDistance failed ({type(e).__name__}: {e}); "
                "falling back to cKDTree nearest neighbors"
            )

    if dists is None or b2x_idx is None:
        tree = cKDTree(points)
        dists, b2x_idx = tree.query(basis, k=1)
        dists = dists.astype(np.float32)
        b2x_idx = b2x_idx.astype(np.int32)

    bps_slot_labels = v2s[b2x_idx].astype(np.int32, copy=True)
    bps_nn_points = points[b2x_idx].astype(np.float32, copy=True)
    bps_slot_labels[dists > dist_threshold] = -1

    return {
        "bps_dists": dists,
        "bps_slot_labels": bps_slot_labels.astype(np.int8),
        "bps_nn_points": bps_nn_points,
        "b2x_idxs": b2x_idx,
    }


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build BPS slot cache for per-object BPS encoding + slot labels"
    )
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test", "all"])
    parser.add_argument("--config", type=str, default="configs/token_config.yaml")
    parser.add_argument("--bps_path", type=str, default="configs/bps.npz")
    parser.add_argument("--save_dir", type=str, default="cache/bps_slot")
    parser.add_argument("--n_samples", type=int, default=10000,
                        help="物体采样点数 (需 >= BPS 基点数 4096)")
    parser.add_argument("--dist_threshold", type=float, default=0.15,
                        help="BPS slot label 距离过滤阈值 (归一化空间)")
    parser.add_argument("--category", type=str, default="all")
    parser.add_argument("--intent", type=str, default="all")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    # 加载 BPS basis
    assert os.path.exists(args.bps_path), f"BPS basis 文件不存在: {args.bps_path}"
    bps_data = np.load(args.bps_path)
    bps_basis = bps_data['basis'].astype(np.float32)
    if bps_basis.ndim == 3:
        bps_basis = bps_basis.squeeze(0)
    logger.info(f"BPS basis: {bps_basis.shape} from {args.bps_path}")

    # 加载 SlotMapper
    mapper = SlotMapper(args.config)
    logger.info(f"SlotMapper: {len(mapper.hand_parts)} parts, "
                f"{len(mapper.canonical_slots)} slots")

    # 加载 obj_id → name 映射
    oakink_root = os.environ.get("OAKINK_DIR", "")
    assert oakink_root, "OAKINK_DIR 环境变量未设置"
    oakbase_dir = os.path.join(oakink_root, "OakBase")
    meta_dir = os.path.join(oakink_root, "shape", "metaV2")

    obj_id_to_name = {}
    for meta_file in ["object_id.json", "virtual_object_id.json"]:
        p = os.path.join(meta_dir, meta_file)
        if os.path.exists(p):
            with open(p, "r") as f:
                meta = json.load(f)
            for oid, info in meta.items():
                obj_id_to_name[oid] = info["name"]
    logger.info(f"Loaded {len(obj_id_to_name)} obj_id→name mappings")

    for split in splits:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing split: {split}")

        # 加载 OIShape 获取 obj_id 列表和网格
        from data.oishape_dataset import OIShape

        class _Cfg:
            DATA_SPLIT = split
            OBJ_CATES = args.category
            INTENT_MODE = args.intent

        oi = OIShape(_Cfg())
        logger.info(f"OIShape [{split}]: {len(oi)} grasps, "
                     f"{len(oi.obj_id_set)} unique objects")

        # 输出目录
        out_dir = os.path.join(args.save_dir, split)
        os.makedirs(out_dir, exist_ok=True)

        # 收集 obj_id → (idx, cate_id, raw_obj_id)
        obj_info = {}
        for idx, grasp in enumerate(oi.grasp_list):
            oid = grasp["obj_id"]
            if oid not in obj_info:
                obj_info[oid] = {
                    "idx": idx,
                    "cate_id": grasp["cate_id"],
                    "raw_obj_id": grasp.get("raw_obj_id", oid),
                }

        stats = {
            "total": len(obj_info),
            "success": 0,
            "no_parts": 0,
            "skipped": 0,
            "failed": 0,
            "all_unassigned_sampled": 0,
            "all_unassigned_bps": 0,
        }
        t0 = time.time()

        for obj_id, info in tqdm(obj_info.items(), desc=f"BPS slot [{split}]"):
            out_path = os.path.join(out_dir, f"{obj_id}.npz")
            if args.skip_existing and os.path.exists(out_path):
                stats["skipped"] += 1
                continue

            try:
                # 加载物体网格 (bbox-centered)
                obj_mesh = oi.get_obj_mesh(info["idx"])

                # 采样 10000 点 (确定性)
                sampled_pts = sample_object_points(
                    obj_mesh, n_samples=args.n_samples, seed_str=obj_id)

                # 加载 part meshes
                part_geometries = load_part_meshes(
                    mapper, obj_id, info["cate_id"], oakbase_dir,
                    obj_id_to_name, obj_mesh, info["raw_obj_id"])

                if not part_geometries:
                    stats["no_parts"] += 1
                    # 仍然保存 BPS 编码，slot labels 全为 -1
                    v2s = np.full(args.n_samples, -1, dtype=np.int32)
                else:
                    # 计算采样点的 slot 标签
                    v2s = compute_vertex_to_slot(sampled_pts, obj_mesh, part_geometries)
                    if np.all(v2s < 0):
                        stats["all_unassigned_sampled"] += 1
                        slot_sizes = {
                            int(sid): int(geom["points"].shape[0])
                            for sid, geom in part_geometries.items()
                        }
                        logger.warning(
                            f"[all -1 sampled] obj_id={obj_id} cate={info['cate_id']} "
                            f"slot_sizes={slot_sizes}"
                        )

                # 归一化
                pts_norm, center, scale = normalize_object(sampled_pts)

                # BPS 编码 + slot 传播
                result = encode_bps_slots(
                    bps_basis=bps_basis,
                    obj_points_norm=pts_norm,
                    vertex_to_slot=v2s,
                    dist_threshold=args.dist_threshold,
                )
                if np.all(result["bps_slot_labels"] < 0):
                    stats["all_unassigned_bps"] += 1
                    sampled_vals, sampled_cnt = np.unique(v2s, return_counts=True)
                    logger.warning(
                        f"[all -1 bps] obj_id={obj_id} cate={info['cate_id']} "
                        f"sampled_slots={dict(zip(sampled_vals.tolist(), sampled_cnt.tolist()))} "
                        f"dist_range=({result['bps_dists'].min():.4f}, {result['bps_dists'].max():.4f}) "
                        f"threshold={args.dist_threshold}"
                    )

                # 保存
                np.savez_compressed(
                    out_path,
                    bps_dists=result['bps_dists'],
                    bps_slot_labels=result['bps_slot_labels'],
                    bps_nn_points=result['bps_nn_points'],
                    b2x_idxs=result['b2x_idxs'],
                    obj_scale=scale,
                    obj_center=center,
                )
                stats["success"] += 1

            except Exception as e:
                logger.warning(f"Failed obj_id={obj_id}: {e}")
                stats["failed"] += 1
                continue

        elapsed = time.time() - t0
        logger.info(f"\nDone [{split}] in {elapsed:.0f}s")
        logger.info(f"  success={stats['success']}, skipped={stats['skipped']}, "
                     f"no_parts={stats['no_parts']}, failed={stats['failed']}")

        # 保存统计
        stats_path = os.path.join(out_dir, "build_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        # 保存索引
        index = {
            "split": split,
            "n_bps": bps_basis.shape[0],
            "n_samples": args.n_samples,
            "dist_threshold": args.dist_threshold,
            "has_bps_nn_points": True,
            "obj_ids": list(obj_info.keys()),
        }
        with open(os.path.join(out_dir, "index.json"), "w") as f:
            json.dump(index, f, indent=2)

    logger.info("\n✅ All done!")


if __name__ == "__main__":
    main()
