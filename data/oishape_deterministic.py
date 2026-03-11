"""
oishape_deterministic.py
========================
OIShape 确定性采样补丁 + 预处理 slot_labels 生成

问题：
  OIShape.__getitem__ 每次调用 trimesh.sample.sample_surface 产生不同的点，
  导致预处理阶段保存的 obj_slot_labels 与训练时加载的点云不匹配。

方案：
  用 obj_id 的 hash 作为随机种子，保证同一物体在任何时刻、任何进程中
  采样出完全相同的 2048 个点。

需要修改的文件：
  1. OIShape.__getitem__     — 采样前设种子
  2. build_contact_dataset.py — 用同样的确定性采样生成 slot_labels

本文件提供：
  - deterministic_sample_surface()  共用的确定性采样函数
  - sample_with_slot_labels()       预处理用，带 slot 标注的采样
  - OIShape.__getitem__ 的补丁说明
"""

import hashlib
import numpy as np
import trimesh
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# ==============================================================================
# 核心：确定性表面采样
# ==============================================================================

def deterministic_sample_surface(
    mesh:      trimesh.Trimesh,
    n_samples: int,
    seed_str:  str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    确定性表面采样：同一 mesh + 同一 seed_str → 完全相同的采样结果

    用 seed_str 的 MD5 hash 作为 numpy 随机种子，
    保证跨进程、跨时间的一致性。

    Args:
        mesh:      trimesh 网格
        n_samples: 采样点数
        seed_str:  种子字符串（通常用 obj_id）

    Returns:
        points:       (N, 3) float32  表面采样点
        normals:      (N, 3) float32  对应面法线
        face_indices: (N,)   int64    采样点所在面的索引
    """
    # obj_id → 确定性种子
    seed = int(hashlib.md5(seed_str.encode("utf-8")).hexdigest(), 16) % (2**31)
    rng = np.random.RandomState(seed)

    # trimesh.sample.sample_surface 内部使用 np.random
    # 为了完全确定性，我们手动实现按面积加权采样
    areas = mesh.area_faces
    probs = areas / areas.sum()

    # 按面积概率采样面
    face_indices = rng.choice(len(areas), size=n_samples, p=probs)

    # 在三角形内均匀采样（重心坐标）
    faces = mesh.faces[face_indices]                     # (N, 3)
    vertices = mesh.vertices                              # (V, 3)

    v0 = vertices[faces[:, 0]]                            # (N, 3)
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # 均匀重心坐标采样
    r1 = rng.random(n_samples).astype(np.float32)
    r2 = rng.random(n_samples).astype(np.float32)
    sqrt_r1 = np.sqrt(r1)

    # 重心坐标 (u, v, w)
    u = 1 - sqrt_r1
    v = sqrt_r1 * (1 - r2)
    w = sqrt_r1 * r2

    points = (u[:, None] * v0 + v[:, None] * v1 + w[:, None] * v2)
    normals = mesh.face_normals[face_indices]

    return (
        points.astype(np.float32),
        normals.astype(np.float32),
        face_indices.astype(np.int64),
    )


# ==============================================================================
# OIShape.__getitem__ 补丁
# ==============================================================================
# 只需把原来的随机采样替换为 deterministic_sample_surface：
#
#   BEFORE (oishape_dataset.py 第 125-127 行):
#       sample = trimesh.sample.sample_surface(obj_mesh, self.n_samples)
#       obj_verts = np.array(sample[0], dtype=np.float32)
#       obj_vn = np.array(obj_mesh.face_normals[sample[1]], dtype=np.float32)
#
#   AFTER:
#       from data.oishape_deterministic import deterministic_sample_surface
#       obj_verts, obj_vn, _ = deterministic_sample_surface(
#           obj_mesh, self.n_samples, seed_str=obj_id)
#
# 这保证：
#   - 同一 obj_id 在 diffusion 训练、decoder 训练、预处理中采样完全一致
#   - 不同 obj_id 的采样结果不同（种子不同）
#   - 同一 obj_id 的不同 grasp 共享相同的物体点云（正确行为：物体不变）


# ==============================================================================
# 带 slot 标注的确定性采样（预处理用）
# ==============================================================================

def sample_with_slot_labels(
    obj_mesh:        trimesh.Trimesh,
    obj_id:          str,
    part_meshes:     Dict[int, trimesh.Trimesh],   # {slot_id: part_mesh}
    n_samples:       int = 2048,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    确定性采样 + slot 标注

    策略：
      1. 用 deterministic_sample_surface 采样整个物体 mesh（保证与 OIShape 一致）
      2. 对每个采样点，找到它所属的 part mesh → slot_id

    分配方法：
      - 有面片的 part: trimesh.proximity.closest_point (精确表面距离)
      - 纯点云的 part: scipy.spatial.cKDTree (顶点最近邻)
      每个采样点取最近 part 的 slot_id。

    Args:
        obj_mesh:    完整物体网格（与 OIShape 使用的相同）
        obj_id:      物体 ID（确定性种子）
        part_meshes: {slot_id: 该 slot 对应的 part trimesh}
                     来自 SlotMapper.load_object_parts + 加载 ply
        n_samples:   采样数（与 OIShape.n_samples 一致）

    Returns:
        points:      (N, 3) float32  — 与 OIShape.__getitem__ 产生的完全一致
        normals:     (N, 3) float32
        slot_labels: (N,)   int32    — 每个点的 slot_id，-1=未分配
    """
    # Step 1: 确定性采样（与 OIShape 完全一致）
    points, normals, _ = deterministic_sample_surface(
        obj_mesh, n_samples, seed_str=obj_id)

    # Step 2: 为每个点分配 slot_id
    slot_labels = np.full(n_samples, -1, dtype=np.int32)

    if not part_meshes:
        return points, normals, slot_labels

    # 计算每个点到各 part mesh 的最近距离
    # part_meshes 通常只有 3-6 个，直接遍历
    best_dist = np.full(n_samples, np.inf, dtype=np.float32)

    for slot_id, part_mesh in part_meshes.items():
        # 有面片 → trimesh.proximity (精确表面距离)
        # 纯点云 → cKDTree (顶点最近邻)
        if len(part_mesh.faces) > 0:
            closest, dist, _ = trimesh.proximity.closest_point(
                part_mesh, points)
            dist = np.asarray(dist, dtype=np.float32)
        else:
            from scipy.spatial import cKDTree
            tree = cKDTree(np.array(part_mesh.vertices))
            dist, _ = tree.query(points)
            dist = dist.astype(np.float32)

        # 更新最近的 slot
        closer = dist < best_dist
        slot_labels[closer] = slot_id
        best_dist[closer] = dist[closer]

    # 所有有 part 的点都已被分配 (取最近的 slot)
    # 仅未被任何 part 覆盖的极端情况保留 -1 (best_dist == inf)
    slot_labels[best_dist == np.inf] = -1

    # 统计 (仅 debug, 不再 warn)
    n_assigned = (slot_labels >= 0).sum()

    return points, normals, slot_labels


# ==============================================================================
# 预处理集成示例
# ==============================================================================

def preprocess_slot_labels_for_object(
    obj_id:          str,
    obj_mesh:        trimesh.Trimesh,
    part_dir:        str,
    object_category: str,
    slot_mapper,     # SlotMapper instance
    n_samples:       int = 2048,
) -> np.ndarray:
    """
    为一个物体实例生成 slot_labels

    在 build_contact_dataset.py 的循环中调用：
        slot_labels = preprocess_slot_labels_for_object(
            obj_id, obj_mesh, part_dir, cate_id, mapper)
        np.savez(cache_path, ..., obj_slot_labels=slot_labels)

    Args:
        obj_id:          物体 ID
        obj_mesh:        完整物体网格
        part_dir:        包含 part_XX.json / part_XX.ply 的目录
        object_category: 物体类别名
        slot_mapper:     SlotMapper 实例
        n_samples:       采样点数（必须与 OIShape.n_samples 一致）

    Returns:
        slot_labels: (N,) int32
    """
    # 加载 part 信息
    parts = slot_mapper.load_object_parts(part_dir, object_category)

    # 加载 part mesh 并按 slot_id 分组
    part_meshes: Dict[int, trimesh.Trimesh] = {}
    for part_key, info in parts.items():
        sid = info["slot_id"]
        ply = info["ply_path"]
        if sid is not None and ply is not None and Path(ply).exists():
            mesh = trimesh.load(ply, process=False, force="mesh")
            # 与 OIShape.get_obj_mesh 的 bbox 中心化保持一致
            bbox_center = (obj_mesh.vertices.min(0) + obj_mesh.vertices.max(0)) / 2
            mesh.vertices = mesh.vertices - bbox_center
            if sid in part_meshes:
                # 同一 slot 多个 part：合并
                part_meshes[sid] = trimesh.util.concatenate(
                    [part_meshes[sid], mesh])
            else:
                part_meshes[sid] = mesh

    # 确定性采样 + slot 分配
    _, _, slot_labels = sample_with_slot_labels(
        obj_mesh, obj_id, part_meshes, n_samples)

    return slot_labels


# ==============================================================================
# 验证工具
# ==============================================================================

def verify_deterministic_consistency(
    obj_mesh:  trimesh.Trimesh,
    obj_id:    str,
    n_samples: int = 2048,
    n_trials:  int = 5,
) -> bool:
    """
    验证多次调用产生完全相同的结果

    Returns:
        True if all trials produce identical results
    """
    ref_pts, ref_nrm, ref_fi = deterministic_sample_surface(
        obj_mesh, n_samples, obj_id)

    for _ in range(n_trials):
        pts, nrm, fi = deterministic_sample_surface(
            obj_mesh, n_samples, obj_id)
        if not (np.array_equal(pts, ref_pts) and
                np.array_equal(nrm, ref_nrm) and
                np.array_equal(fi, ref_fi)):
            return False
    return True


# ==============================================================================
# 测试
# ==============================================================================

if __name__ == "__main__":
    # 创建一个简单的测试 mesh
    mesh = trimesh.creation.box(extents=[0.1, 0.05, 0.03])
    obj_id = "test_box_001"

    # 验证确定性
    assert verify_deterministic_consistency(mesh, obj_id), "不一致!"
    print("✅ 确定性验证通过")

    # 测试采样
    pts, nrm, fi = deterministic_sample_surface(mesh, 2048, obj_id)
    print(f"采样: {pts.shape}, 法线: {nrm.shape}")
    print(f"  点范围: [{pts.min():.4f}, {pts.max():.4f}]")
    print(f"  法线范数: {np.linalg.norm(nrm, axis=1).mean():.6f}")

    # 测试带 slot 的采样
    # 把 box 分成两个 "part"
    part_top = trimesh.creation.box(extents=[0.1, 0.05, 0.015])
    part_top.apply_translation([0, 0, 0.0075])
    part_bottom = trimesh.creation.box(extents=[0.1, 0.05, 0.015])
    part_bottom.apply_translation([0, 0, -0.0075])

    _, _, labels = sample_with_slot_labels(
        mesh, obj_id,
        part_meshes={0: part_top, 1: part_bottom},
        n_samples=2048,
    )

    n0 = (labels == 0).sum()
    n1 = (labels == 1).sum()
    n_unassigned = (labels == -1).sum()
    print(f"\nSlot 分配: slot_0={n0}, slot_1={n1}, unassigned={n_unassigned}")
    print("✅ slot_labels 测试通过")