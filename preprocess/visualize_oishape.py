#!/usr/bin/env python3
"""
visualize_oishape.py — 可视化 OIShape 样本，验证手-物坐标对齐
=============================================================

检查项:
  1. 手部顶点与物体点云是否在同一坐标系（手应包围物体）
  2. hand_tsl (wrist) 是否在合理位置
  3. 不同类别/抓握模式的样本是否都正确

用法:
  python visualize_oishape.py
  python visualize_oishape.py --n_samples 8 --save_dir ./vis_output
  python visualize_oishape.py --indices 0 100 500 1000
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def create_dataset(split="train"):
    from data.oishape_dataset import OIShape

    class Cfg:
        DATA_SPLIT = split
        OBJ_CATES = "all"
        INTENT_MODE = "all"

    return OIShape(Cfg())


def visualize_sample(sample, idx, save_path):
    """单个样本的 3D 可视化"""
    obj_verts  = sample["obj_verts"]     # (N, 3) 物体点云 (已居中)
    hand_verts = sample["hand_verts"]    # (V, 3) 手部顶点
    hand_tsl   = sample["hand_tsl"]      # (3,)   wrist 位置
    cate_id    = sample["cate_id"]
    obj_id     = sample["obj_id"]

    fig = plt.figure(figsize=(16, 5))

    # ---- 视角 1: 正面 ----
    ax1 = fig.add_subplot(131, projection="3d")
    _plot_3d(ax1, obj_verts, hand_verts, hand_tsl, elev=20, azim=45)
    ax1.set_title("View 1 (45°)")

    # ---- 视角 2: 侧面 ----
    ax2 = fig.add_subplot(132, projection="3d")
    _plot_3d(ax2, obj_verts, hand_verts, hand_tsl, elev=20, azim=135)
    ax2.set_title("View 2 (135°)")

    # ---- 视角 3: 俯视 ----
    ax3 = fig.add_subplot(133, projection="3d")
    _plot_3d(ax3, obj_verts, hand_verts, hand_tsl, elev=80, azim=45)
    ax3.set_title("View 3 (top)")

    # ---- 诊断信息 ----
    obj_center = obj_verts.mean(axis=0)
    hand_center = hand_verts.mean(axis=0)
    dist = np.linalg.norm(obj_center - hand_center)

    # 手-物最近距离
    from scipy.spatial import cKDTree
    tree = cKDTree(obj_verts)
    dists, _ = tree.query(hand_verts, k=1)
    min_dist = dists.min()
    mean_dist = dists.mean()
    pct_contact = (dists < 0.005).mean() * 100  # 5mm 内接触比例

    info = (f"[{idx}] {cate_id}/{obj_id}\n"
            f"obj center: ({obj_center[0]:.3f}, {obj_center[1]:.3f}, {obj_center[2]:.3f})\n"
            f"hand center: ({hand_center[0]:.3f}, {hand_center[1]:.3f}, {hand_center[2]:.3f})\n"
            f"wrist (tsl): ({hand_tsl[0]:.3f}, {hand_tsl[1]:.3f}, {hand_tsl[2]:.3f})\n"
            f"center dist: {dist:.4f}m  |  min hand-obj: {min_dist:.4f}m  |  "
            f"mean: {mean_dist:.4f}m  |  contact(<5mm): {pct_contact:.1f}%")

    fig.suptitle(info, fontsize=9, family="monospace", y=0.02, va="bottom")
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
    print(f"    {info}")


def _plot_3d(ax, obj_verts, hand_verts, hand_tsl, elev=20, azim=45):
    """绘制单个 3D 视角"""
    # 物体点云 (蓝)
    step_obj = max(1, len(obj_verts) // 500)
    ax.scatter(obj_verts[::step_obj, 0], obj_verts[::step_obj, 1],
               obj_verts[::step_obj, 2],
               c="royalblue", s=1, alpha=0.4, label="object")

    # 手部顶点 (红)
    step_hand = max(1, len(hand_verts) // 300)
    ax.scatter(hand_verts[::step_hand, 0], hand_verts[::step_hand, 1],
               hand_verts[::step_hand, 2],
               c="salmon", s=2, alpha=0.5, label="hand")

    # Wrist 位置 (绿星)
    ax.scatter([hand_tsl[0]], [hand_tsl[1]], [hand_tsl[2]],
               c="lime", s=80, marker="*", zorder=10, label="wrist")

    # 原点 (黑叉)
    ax.scatter([0], [0], [0], c="black", s=60, marker="x",
               zorder=10, label="origin")

    ax.view_init(elev=elev, azim=azim)
    _set_equal_axes(ax, obj_verts, hand_verts)
    ax.legend(fontsize=6, loc="upper right")


def _set_equal_axes(ax, pts1, pts2):
    """设置等比例坐标轴"""
    all_pts = np.concatenate([pts1, pts2], axis=0)
    center = all_pts.mean(axis=0)
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.1
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    ax.set_xlabel("X", fontsize=7)
    ax.set_ylabel("Y", fontsize=7)
    ax.set_zlabel("Z", fontsize=7)
    ax.tick_params(labelsize=6)


def pick_diverse_indices(dataset, n=6):
    """从不同类别中各选一个样本"""
    cate_to_idx = {}
    for i, g in enumerate(dataset.grasp_list):
        cate = g["cate_id"]
        if cate not in cate_to_idx:
            cate_to_idx[cate] = i

    indices = list(cate_to_idx.values())[:n]
    # 不够就随机补
    if len(indices) < n:
        rng = np.random.RandomState(42)
        extra = rng.choice(len(dataset), n - len(indices), replace=False)
        indices.extend(extra.tolist())

    return indices[:n]


def main():
    parser = argparse.ArgumentParser("Visualize OIShape samples")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--n_samples", type=int, default=6,
                        help="Number of samples to visualize (ignored if --indices)")
    parser.add_argument("--indices", type=int, nargs="+", default=None,
                        help="Specific sample indices to visualize")
    parser.add_argument("--save_dir", type=str, default="./vis_oishape")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Loading OIShape ({args.split})...")
    dataset = create_dataset(args.split)
    print(f"  {len(dataset)} samples loaded")

    if args.indices:
        indices = args.indices
    else:
        indices = pick_diverse_indices(dataset, args.n_samples)

    print(f"\nVisualizing {len(indices)} samples: {indices}\n")

    for i, idx in enumerate(indices):
        if idx >= len(dataset):
            print(f"  [SKIP] idx={idx} out of range")
            continue
        sample = dataset[idx]
        save_path = os.path.join(args.save_dir,
                                 f"sample_{idx:06d}_{sample['cate_id']}.png")
        visualize_sample(sample, idx, save_path)

    # ---- 汇总统计 ----
    print(f"\n{'='*60}")
    print("Quick sanity check on all samples (center distance):")
    rng = np.random.RandomState(42)
    check_indices = rng.choice(len(dataset), min(200, len(dataset)), replace=False)
    dists = []
    for idx in check_indices:
        s = dataset[idx]
        obj_c = s["obj_verts"].mean(axis=0)
        hand_c = s["hand_verts"].mean(axis=0)
        dists.append(np.linalg.norm(obj_c - hand_c))
    dists = np.array(dists)
    print(f"  Sampled {len(dists)} samples")
    print(f"  obj-hand center dist: "
          f"mean={dists.mean():.4f}m  std={dists.std():.4f}m  "
          f"max={dists.max():.4f}m")
    if dists.mean() > 0.15:
        print("  ⚠️  Mean distance > 15cm — possible coordinate misalignment!")
    else:
        print("  ✓  Looks reasonable (hand close to object)")
    print(f"{'='*60}")
    print(f"\nAll visualizations saved to: {args.save_dir}/")


if __name__ == "__main__":
    main()