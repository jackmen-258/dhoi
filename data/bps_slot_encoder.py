"""
bps_slot_encoder.py
===================
将物体 part 的 slot 标注传播到 BPS basis points 上。

核心操作:
  1. BPS.enc_points(obj_verts) → b2x_idxs: 每个 basis point 最近的物体顶点
  2. vertex_to_slot[b2x_idxs] → 每个 basis point 的 slot label
  3. 距离过滤: 太远的 basis point 标为 NO_SLOT(-1)

输入:
  - obj_verts: (N_obj, 3) 物体顶点（归一化后）
  - vertex_to_slot: (N_obj,) int, 每个顶点的 slot_id (0~3 或 -1)
  - bps_basis: (K, 3) 固定 basis points

输出:
  - bps_dists: (K,) float, basis→object 距离
  - bps_slot_labels: (K,) int, ∈ {-1, 0, 1, 2, 3}
  - bps_nn_points: (K, 3) float, 每个 basis 对应的最近归一化物体点
  - b2x_idxs: (K,) int, basis→最近物体顶点索引
"""

import numpy as np
import torch
from typing import Optional

NO_SLOT = -1


class BPSSlotEncoder:
    """将 slot 语义从物体表面传播到 BPS basis points。

    用法:
        encoder = BPSSlotEncoder(bps_basis)
        result = encoder.encode(obj_verts_norm, vertex_to_slot)
        # result['bps_slot_labels'] -> (4096,) int8
    """

    def __init__(
        self,
        bps_basis: np.ndarray,         # (K, 3)
        num_slots: int = 4,
        dist_threshold: float = 0.15,  # 归一化空间中的距离阈值
    ):
        self.bps_basis = np.asarray(bps_basis, dtype=np.float32)
        if self.bps_basis.ndim == 3:
            self.bps_basis = self.bps_basis.squeeze(0)  # (1, K, 3) -> (K, 3)
        self.n_bps = self.bps_basis.shape[0]
        self.num_slots = num_slots
        self.dist_threshold = dist_threshold

    def encode(
        self,
        obj_verts: np.ndarray,           # (N, 3), 已归一化
        vertex_to_slot: np.ndarray,      # (N,) int, -1~3
    ) -> dict:
        """
        计算每个 BPS basis point 的 slot label。

        Returns:
            {
                'bps_dists':       (K,) float32  — basis→物体距离
                'bps_slot_labels': (K,) int8     — slot 归属 (-1 = 无效)
                'bps_nn_points':   (K, 3) float32 — 最近归一化物体点
                'b2x_idxs':        (K,) int32    — 最近物体顶点索引
            }
        """
        import chamfer_distance as chd

        obj_t = torch.from_numpy(np.asarray(obj_verts, dtype=np.float32)).unsqueeze(0)
        basis_t = torch.from_numpy(self.bps_basis).unsqueeze(0)

        # 如果 GPU 可用，在 GPU 上算 Chamfer 更快
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        obj_t = obj_t.to(device)
        basis_t = basis_t.to(device)

        ch = chd.ChamferDistance()
        # b2x: basis → object 最近邻
        b2x_dist, _, b2x_idx, _ = ch(basis_t, obj_t)
        b2x_idx = b2x_idx.squeeze(0).long()  # (K,)

        # 计算精确距离 (Chamfer 返回的是平方距离)
        nearest_pts = obj_t.squeeze(0)[b2x_idx]  # (K, 3)
        dists = (basis_t.squeeze(0) - nearest_pts).norm(dim=1)  # (K,)

        # 传播 slot label
        v2s = torch.from_numpy(np.asarray(vertex_to_slot, dtype=np.int64)).to(device)
        bps_slots = v2s[b2x_idx]  # (K,)

        # 距离过滤
        bps_slots[dists > self.dist_threshold] = NO_SLOT

        return {
            'bps_dists': dists.cpu().numpy().astype(np.float32),
            'bps_slot_labels': bps_slots.cpu().numpy().astype(np.int8),
            'bps_nn_points': nearest_pts.cpu().numpy().astype(np.float32),
            'b2x_idxs': b2x_idx.cpu().numpy().astype(np.int32),
        }


def normalize_object(obj_verts: np.ndarray, eps: float = 1e-6):
    """Per-object bbox 归一化: center + scale by diagonal.

    Args:
        obj_verts: (N, 3)

    Returns:
        obj_norm: (N, 3) 归一化后坐标
        center:   (3,)
        scale:    float, bbox 对角线
    """
    vmin = obj_verts.min(axis=0)
    vmax = obj_verts.max(axis=0)
    center = (vmin + vmax) / 2.0
    scale = float(np.linalg.norm(vmax - vmin))
    if scale < eps:
        scale = 1.0
    obj_norm = (obj_verts - center) / scale
    return obj_norm.astype(np.float32), center.astype(np.float32), np.float32(scale)


def compute_vertex_to_slot(
    sampled_points: np.ndarray,       # (N, 3) 采样点
    obj_mesh_vertices: np.ndarray,    # (M, 3) 原始网格顶点
    mesh_vertex_to_slot: np.ndarray,  # (M,) 原始顶点的 slot 标签
) -> np.ndarray:
    """将原始网格顶点的 slot 标签传播到采样点。

    通过最近邻查找: 每个采样点找最近的网格顶点，继承其 slot。

    Args:
        sampled_points: 采样的表面点
        obj_mesh_vertices: 原始网格的顶点
        mesh_vertex_to_slot: 原始顶点的 slot 标签

    Returns:
        (N,) int32, 每个采样点的 slot 标签
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(obj_mesh_vertices)
    _, idx = tree.query(sampled_points, k=1)
    return mesh_vertex_to_slot[idx].astype(np.int32)
