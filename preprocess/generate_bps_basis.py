"""
generate_bps_basis.py
=====================
生成适配 dhoi 归一化空间的 BPS 基点。

dhoi 的 BPS 监督使用 bbox-center + bbox-diagonal normalization：
归一化后，物体点云整体落在半径不超过 0.5 的球内。

这里默认生成“球内均匀分布”的 BPS basis，而不是“球面壳层”上的点：
  1. 这和原始 GrabNet / bps_torch 的 basis 分布一致
  2. basis 覆盖整个体积，更适合细长/中空物体
  3. 旧的 configs/bps.npz 本身也是球内分布，而不是球面分布

用法:
    python preprocess/generate_bps_basis.py \
        --n_basis 4096 --radius 0.5 \
        --distribution ball \
        --save_path configs/bps.npz
"""

import os
import argparse
import numpy as np


def sample_ball_uniform(n_points: int, radius: float = 1.0, seed: int = 13) -> np.ndarray:
    """在三维球体内部均匀采样点。

    这与仓库内 bps_torch.tools.sample_sphere_uniform 的采样逻辑一致：
    先采样方向，再采样 r^(1/3) 的半径分布，得到体积均匀分布。

    Args:
        n_points: 采样点数
        radius: 球体半径
        seed: 随机种子

    Returns:
        (n_points, 3) float32
    """
    rng = np.random.RandomState(seed)

    x = rng.normal(size=(n_points, 3)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8

    r = rng.uniform(size=(n_points, 1)).astype(np.float32)
    u = np.power(r, 1.0 / 3.0)
    points = radius * x * u

    return points.astype(np.float32)


def sample_sphere_surface(n_points: int, radius: float = 1.0, seed: int = 42) -> np.ndarray:
    """在球面上均匀采样点（Fibonacci lattice）。

    比随机采样更均匀，避免聚集。

    Args:
        n_points: 采样点数
        radius: 球面半径
        seed: 随机种子（用于微小扰动）

    Returns:
        (n_points, 3) float32
    """
    rng = np.random.RandomState(seed)

    # Fibonacci sphere
    indices = np.arange(n_points, dtype=np.float64)
    golden_ratio = (1 + np.sqrt(5)) / 2

    # 极角 theta: arccos(1 - 2*(i+0.5)/N)
    theta = np.arccos(1 - 2 * (indices + 0.5) / n_points)
    # 方位角 phi: 2*pi*i/golden_ratio
    phi = 2 * np.pi * indices / golden_ratio

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    points = np.stack([x, y, z], axis=1) * radius

    # 加微小扰动避免完全确定性（可选）
    points += rng.randn(*points.shape) * (radius * 0.001)

    # 重新投影到精确半径
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms * radius

    return points.astype(np.float32)


def summarize_radius_stats(points: np.ndarray) -> str:
    norms = np.linalg.norm(points, axis=1)
    q = np.quantile(norms, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return (
        f"min={q[0]:.4f}, p10={q[1]:.4f}, p25={q[2]:.4f}, "
        f"p50={q[3]:.4f}, p75={q[4]:.4f}, p90={q[5]:.4f}, max={q[6]:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Generate BPS basis points")
    parser.add_argument("--n_basis", type=int, default=4096,
                        help="Number of basis points")
    parser.add_argument("--radius", type=float, default=0.5,
                        help="BPS support radius in normalized space")
    parser.add_argument("--distribution", type=str, default="ball",
                        choices=["ball", "sphere"],
                        help="`ball`: uniform inside the volume; `sphere`: only on the shell")
    parser.add_argument("--save_path", type=str, default="configs/bps.npz",
                        help="Output path")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    print(
        f"Generating {args.n_basis} BPS basis points "
        f"(distribution={args.distribution}, radius={args.radius})"
    )

    if args.distribution == "ball":
        basis = sample_ball_uniform(args.n_basis, args.radius, args.seed)
    else:
        basis = sample_sphere_surface(args.n_basis, args.radius, args.seed)

    # 验证
    print(f"  Shape: {basis.shape}")
    print(f"  Radius quantiles: {summarize_radius_stats(basis)}")
    print(f"  Coord range: [{basis.min():.4f}, {basis.max():.4f}]")

    # 保存（与 GrabNet bps.npz 格式一致: key='basis', shape=(1, N, 3)）
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    np.savez_compressed(args.save_path, basis=basis.reshape(1, -1, 3))
    print(f"  Saved to {args.save_path}")

    # 与旧 BPS 对比（如果存在）
    old_path = "grabnet/configs/bps.npz"
    if os.path.exists(old_path):
        old_basis = np.load(old_path)['basis']
        if old_basis.ndim == 3:
            old_basis = old_basis.squeeze(0)
        print(f"\n  Old BPS ({old_path}):")
        print(f"    Shape: {old_basis.shape}")
        print(f"    Radius quantiles: {summarize_radius_stats(old_basis)}")


if __name__ == "__main__":
    main()
