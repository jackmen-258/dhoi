"""
preprocess/build_cweights.py
=============================
从 OakInk 数据集统计真实接触频率，生成每顶点接触权重：

    grabnet/configs/c_weights.npy  — (778,) float32

统计方法：
  对每个 grasp 样本，计算 778 个 MANO 手部顶点到物体表面的最近距离；
  距离 < contact_distance（由 token_config.yaml 读取）的顶点视为"接触"；
  统计每个顶点在全体样本中的接触频率，归一化至均值 = 1.0。

使用已有的 npz 缓存（需先运行 build_contact_dataset.py）重建手部顶点，
无需再次解析 OakInk 原始数据，速度快且结果与训练时完全一致。

Usage:
    # 从 train split 的 npz 缓存统计（推荐）
    python preprocess/build_cweights.py

    # 指定 split 和缓存目录
    python preprocess/build_cweights.py \\
        --splits train val \\
        --cache_root cache/contact_tokens \\
        --config    configs/token_config.yaml \\
        --mano_root assets/mano_v1_2 \\
        --out_dir   grabnet/configs

前置条件：
    build_contact_dataset.py 已运行，cache/contact_tokens/{split}/*.npz 存在。
"""

import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ==============================================================================
# 工具函数
# ==============================================================================

def load_config(config_path: str) -> dict:
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_mano_layer(mano_root: str):
    from manotorch.manolayer import ManoLayer
    return ManoLayer(center_idx=0, mano_assets_root=mano_root)


def get_contact_distance(cfg: dict) -> float:
    """从 token_config.yaml 读取接触距离阈值（mm → m）"""
    ct = cfg["preprocessing"]["contact_thresholds"]
    return float(ct["contact_distance"]) * 1e-3


def load_split_index(cache_dir: str) -> list:
    """读取 dataset_index.json，返回 sample 列表。"""
    index_path = os.path.join(cache_dir, "dataset_index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"dataset_index.json 不存在: {index_path}\n"
            "请先运行 preprocess/build_contact_dataset.py。"
        )
    with open(index_path, "r", encoding="utf-8") as f:
        idx = json.load(f)
    return idx["samples"]


def load_sample_npz(npz_path: str) -> dict | None:
    """加载单个 npz，返回所需字段；失败返回 None。"""
    try:
        data = np.load(npz_path, allow_pickle=True)
        return {
            "hand_pose":  data["hand_pose"].astype(np.float32),   # (48,)
            "hand_shape": data["hand_shape"].astype(np.float32),  # (10,)
            "hand_tsl":   data["hand_tsl"].astype(np.float32),    # (3,)
        }
    except Exception as e:
        print(f"[WARN] 加载失败 {npz_path}: {e}")
        return None


# ==============================================================================
# MANO 批量前向（分批以节省显存）
# ==============================================================================

def compute_hand_verts_batched(
    mano_layer,
    poses:  np.ndarray,   # (N, 48)
    shapes: np.ndarray,   # (N, 10)
    tsls:   np.ndarray,   # (N, 3)
    batch_size: int = 512,
    device: str = "cpu",
) -> np.ndarray:
    """批量计算 MANO 手部顶点，返回 (N, 778, 3)。"""
    import torch

    mano_layer = mano_layer.to(device)
    N = poses.shape[0]
    all_verts = []

    for start in tqdm(range(0, N, batch_size), desc="  MANO forward", leave=False):
        end = min(start + batch_size, N)
        pose_t  = torch.from_numpy(poses[start:end]).to(device)
        shape_t = torch.from_numpy(shapes[start:end]).to(device)
        tsl_t   = torch.from_numpy(tsls[start:end]).to(device)

        with torch.no_grad():
            out = mano_layer(pose_t, shape_t)
        verts = out.verts + tsl_t.unsqueeze(1)  # (B, 778, 3)
        all_verts.append(verts.cpu().numpy())

    return np.concatenate(all_verts, axis=0)  # (N, 778, 3)


# ==============================================================================
# 物体点云加载（带缓存）
# ==============================================================================

def build_obj_warehouse(
    oakink_dir: str,
    sample_list: list,
    n_surf_pts: int = 4096,
) -> dict:
    """
    为所有涉及物体采样表面点，返回 {obj_id: (N, 3) surface points}。

    物体坐标系：bbox 居中（与 oishape_dataset.py 一致）。
    """
    import trimesh
    from data.oishape_dataset import OIShape
    from data.utils import get_obj_path

    # 收集全部 obj_id
    obj_ids = {}   # obj_id → cate_id（任意一个 sample 提供）
    for s in sample_list:
        obj_ids[s["obj_id"]] = s["cate_id"]

    data_dir  = os.path.join(oakink_dir, "shape")
    meta_dir  = os.path.join(data_dir, "metaV2")

    warehouse = {}
    for obj_id, cate_id in tqdm(obj_ids.items(), desc="  采样物体表面点"):
        try:
            obj_path = get_obj_path(obj_id, data_dir, meta_dir, use_downsample=True)
            mesh = trimesh.load(obj_path, process=False, force="mesh",
                                skip_materials=True)
            # bbox 居中
            bbox_center = (mesh.vertices.min(0) + mesh.vertices.max(0)) / 2
            mesh.vertices -= bbox_center
            # 均匀采样表面点
            pts, _ = trimesh.sample.sample_surface(mesh, n_surf_pts)
            warehouse[obj_id] = pts.astype(np.float32)
        except Exception as e:
            print(f"[WARN] 物体 {obj_id} 加载失败: {e}")

    print(f"  物体表面点库: {len(warehouse)} 个物体")
    return warehouse


def build_obj_proximity(warehouse: dict) -> dict:
    """为每个物体点云建 cKDTree，供距离查询。"""
    from scipy.spatial import cKDTree
    trees = {}
    for obj_id, pts in warehouse.items():
        trees[obj_id] = cKDTree(pts)
    return trees


# ==============================================================================
# 主统计逻辑
# ==============================================================================

def accumulate_contact_counts(
    sample_list:        list,          # dataset_index samples
    cache_dir:          str,           # npz 所在目录
    mano_layer,                        # ManoLayer
    obj_trees:          dict,          # {obj_id: cKDTree}
    contact_distance:   float,         # 接触阈值（米）
    mano_batch_size:    int = 512,
    device:             str = "cpu",
) -> np.ndarray:
    """
    遍历所有样本，统计每顶点接触次数。

    Returns:
        counts: (778,) float64，每顶点在全体样本中的接触次数
        n_valid: int，有效样本数
    """
    # 先收集所有可用样本的 pose/shape/tsl
    print("  读取 npz 缓存...")
    poses, shapes, tsls, meta = [], [], [], []
    missing_obj = 0

    for s in tqdm(sample_list, desc="  读取 npz", leave=False):
        obj_id = s["obj_id"]
        if obj_id not in obj_trees:
            missing_obj += 1
            continue

        npz_path = os.path.join(cache_dir, s["cache_file"])
        rec = load_sample_npz(npz_path)
        if rec is None:
            continue

        poses.append(rec["hand_pose"])
        shapes.append(rec["hand_shape"])
        tsls.append(rec["hand_tsl"])
        meta.append(obj_id)

    if missing_obj:
        print(f"  [INFO] {missing_obj} 个样本因物体不在仓库中被跳过")

    n_valid = len(poses)
    if n_valid == 0:
        raise RuntimeError("没有可用样本，请检查缓存路径和物体仓库。")
    print(f"  有效样本: {n_valid}")

    # 批量 MANO 前向
    print("  计算手部顶点...")
    poses_arr  = np.stack(poses,  axis=0)   # (N, 48)
    shapes_arr = np.stack(shapes, axis=0)   # (N, 10)
    tsls_arr   = np.stack(tsls,   axis=0)   # (N, 3)
    all_verts  = compute_hand_verts_batched(
        mano_layer, poses_arr, shapes_arr, tsls_arr,
        batch_size=mano_batch_size, device=device,
    )  # (N, 778, 3)

    # 统计接触频次
    print("  统计接触频次...")
    counts = np.zeros(778, dtype=np.float64)

    for i, obj_id in enumerate(tqdm(meta, desc="  接触统计", leave=False)):
        tree  = obj_trees[obj_id]
        verts = all_verts[i]           # (778, 3)
        dists, _ = tree.query(verts, k=1)   # (778,)
        counts += (dists < contact_distance).astype(np.float64)

    return counts, n_valid


# ==============================================================================
# 归一化与保存
# ==============================================================================

def normalize_weights(counts: np.ndarray, min_weight: float = 0.1) -> np.ndarray:
    """
    归一化至均值 = 1.0，并将零频顶点赋予最小权重。

    Args:
        counts:     (778,) 接触频次
        min_weight: 接触频次为零的顶点的最小权重（避免完全屏蔽）

    Returns:
        (778,) float32
    """
    w = counts.copy()
    mean = w.mean()
    if mean < 1e-9:
        raise RuntimeError("所有顶点接触频次为零，请检查 contact_distance 设置。")

    w = w / mean          # 均值 = 1.0
    w = np.maximum(w, min_weight)
    return w.astype(np.float32)


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="从 OakInk 统计接触频率，生成 c_weights.npy"
    )
    p.add_argument("--splits", nargs="+", default=["train"],
                   choices=["train", "val", "test"],
                   help="使用哪些 split 统计（默认仅 train）")
    p.add_argument("--cache_root", default="cache/contact_tokens",
                   help="contact token 缓存根目录")
    p.add_argument("--config", default="configs/token_config.yaml",
                   help="token_config.yaml 路径")
    p.add_argument("--mano_root", default="assets/mano_v1_2",
                   help="MANO 资产根目录")
    p.add_argument("--out_dir", default="configs",
                   help="输出目录（存放 c_weights.npy）")
    p.add_argument("--n_surf_pts", type=int, default=4096,
                   help="每个物体采样的表面点数（越多距离估计越精确）")
    p.add_argument("--mano_batch_size", type=int, default=512,
                   help="MANO 前向批量大小")
    p.add_argument("--min_weight", type=float, default=0.1,
                   help="零接触顶点的最小权重（避免完全屏蔽）")
    p.add_argument("--device", default="cpu",
                   help="MANO 前向设备（cpu / cuda）")
    return p.parse_args()


def main():
    args = parse_args()

    # 检查 OAKINK_DIR
    oakink_dir = os.environ.get("OAKINK_DIR", "")
    if not oakink_dir:
        print("[ERROR] 请设置环境变量 OAKINK_DIR")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("build_cweights.py — OakInk 接触频率统计")
    print("=" * 60)
    print(f"  Splits:         {args.splits}")
    print(f"  Cache root:     {args.cache_root}")
    print(f"  Config:         {args.config}")
    print(f"  OAKINK_DIR:     {oakink_dir}")
    print(f"  Contact dist:   ", end="")

    # 读配置
    cfg = load_config(args.config)
    contact_distance = get_contact_distance(cfg)
    print(f"{contact_distance * 1000:.1f} mm")
    print(f"  Output dir:     {args.out_dir}")
    print("=" * 60)

    # 收集所有 split 的样本列表
    print("\n[1/4] 收集样本索引...")
    all_samples = []
    for split in args.splits:
        cache_dir = os.path.join(args.cache_root, split)
        samples = load_split_index(cache_dir)
        # 为每个 sample 记录所在 cache_dir
        for s in samples:
            s["_cache_dir"] = cache_dir
        all_samples.extend(samples)
        print(f"  {split}: {len(samples)} 样本")
    print(f"  合计: {len(all_samples)} 样本")

    # 加载 MANO
    print("\n[2/4] 加载 MANO...")
    mano_layer = build_mano_layer(args.mano_root)
    print(f"  MANO 顶点数: 778, faces: {mano_layer.th_faces.shape[0]}")

    # 建立物体表面点库
    print("\n[3/4] 建立物体表面点库...")
    warehouse = build_obj_warehouse(oakink_dir, all_samples, args.n_surf_pts)
    obj_trees = build_obj_proximity(warehouse)

    # 按 cache_dir 分组统计
    print("\n[4/4] 统计接触频次...")

    # 按 split 分组（每个 split 有自己的 cache_dir）
    by_cache: dict = {}
    for s in all_samples:
        cd = s["_cache_dir"]
        by_cache.setdefault(cd, []).append(s)

    total_counts = np.zeros(778, dtype=np.float64)
    total_valid  = 0

    for cache_dir, samples in by_cache.items():
        print(f"\n  处理 {cache_dir} ({len(samples)} 样本)...")
        counts, n_valid = accumulate_contact_counts(
            sample_list=samples,
            cache_dir=cache_dir,
            mano_layer=mano_layer,
            obj_trees=obj_trees,
            contact_distance=contact_distance,
            mano_batch_size=args.mano_batch_size,
            device=args.device,
        )
        total_counts += counts
        total_valid  += n_valid

    print(f"\n  有效样本总数: {total_valid}")
    print(f"  接触频次范围: [{total_counts.min():.0f}, {total_counts.max():.0f}]")
    print(f"  接触频次均值: {total_counts.mean():.1f}")

    # 归一化
    c_weights = normalize_weights(total_counts, min_weight=args.min_weight)

    # 保存
    out_path = os.path.join(args.out_dir, "c_weights.npy")
    np.save(out_path, c_weights)

    print(f"\nc_weights.npy 已保存: {out_path}")
    print(f"  shape={c_weights.shape}")
    print(f"  min={c_weights.min():.3f}  max={c_weights.max():.3f}  mean={c_weights.mean():.3f}")

    # 打印各手指部位的权重统计（供参考）
    _print_part_stats(c_weights, mano_layer)

    print("\n完成。")


def _print_part_stats(c_weights: np.ndarray, mano_layer):
    """打印各手部区域的平均权重，供直觉检验。"""
    import torch

    JOINT_TO_PART = {
        0:  "PALM",
        1:  "INDEX", 2:  "INDEX", 3:  "INDEX",
        4:  "MIDDLE", 5:  "MIDDLE", 6:  "MIDDLE",
        7:  "LITTLE", 8:  "LITTLE", 9:  "LITTLE",
        10: "RING", 11: "RING", 12: "RING",
        13: "THUMB", 14: "THUMB", 15: "THUMB",
    }
    weights_np = mano_layer.th_weights.detach().cpu().numpy()  # (778, 16)
    dominant = weights_np.argmax(axis=1)                        # (778,)

    print("\n  各手部区域平均权重:")
    from collections import defaultdict
    part_w = defaultdict(list)
    for v, j in enumerate(dominant):
        part_w[JOINT_TO_PART[j]].append(c_weights[v])

    for part in ["THUMB", "INDEX", "MIDDLE", "RING", "LITTLE", "PALM"]:
        ws = part_w[part]
        print(f"    {part:<8}: n={len(ws):3d}  mean={np.mean(ws):.3f}  "
              f"min={np.min(ws):.3f}  max={np.max(ws):.3f}")


if __name__ == "__main__":
    main()
