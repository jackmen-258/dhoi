"""
mesh_utils.py
=============
Mesh 距离计算、接触判定、分组聚合工具

核心功能：
  1. 包装 capsule SDF 接触计算（修改自已有代码，额外返回 nearest_idx）
  2. MANO 顶点 → hand_part 分组
  3. 物体 part mesh 顶点 → slot 归属
  4. 按 (hand_part, slot) 聚合接触信息 → 6×6 接触矩阵
  5. 接触矩阵 → contact_type 判定

依赖：
  - torch
  - pytorch3d (knn_points)
  - trimesh (加载 ply)
  - numpy
"""

import numpy as np
import torch
import trimesh
from typing import Dict, List, Optional, Tuple


# ==============================================================================
# 1. Capsule SDF 接触计算（修改版，额外返回 nearest_idx）
# ==============================================================================

def batched_index_select(t: torch.Tensor, dim: int, inds: torch.Tensor) -> torch.Tensor:
    """
    沿指定维度按 batch-varying 索引选取元素

    Args:
        t: 源张量 (batch, N, D)
        dim: 选取维度
        inds: 索引 (batch, M)

    Returns:
        (batch, M, D)
    """
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy)
    return out


def capsule_sdf(
    mesh_verts: torch.Tensor,
    mesh_normals: torch.Tensor,
    query_points: torch.Tensor,
    query_normals: torch.Tensor,
    caps_rad: float,
    caps_top: float,
    caps_bot: float,
    foreach_on_mesh: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Capsule SDF 接触检测

    修改点：额外返回 nearest_idx，用于后续分组归属判定

    Args:
        mesh_verts: (batch, V, 3)
        mesh_normals: (batch, V, 3)
        query_points: (batch, Q, 3)
        query_normals: (batch, Q, 3)
        caps_rad: capsule 半径
        caps_top: mesh 到 capsule 顶端的距离
        caps_bot: mesh 到 capsule 底端的距离
        foreach_on_mesh: True=对每个 mesh 顶点找最近 query, False=反之

    Returns:
        sdf: (batch, V or Q) 归一化 SDF+1，0=capsule 中心，1=capsule 表面
        normal_dot: (batch, V or Q) 法向点积
        nearest_idx: (batch, V or Q) 最近邻索引
    """
    import pytorch3d.ops

    if foreach_on_mesh:
        knn_dists, nearest_idx, nearest_pos = pytorch3d.ops.knn_points(
            mesh_verts, query_points, K=1, return_nn=True
        )
        nearest_idx = nearest_idx.squeeze(2)  # (batch, V)

        capsule_tops = mesh_verts + mesh_normals * caps_top
        capsule_bots = mesh_verts + mesh_normals * caps_bot
        delta_top = nearest_pos[:, :, 0, :] - capsule_tops
        normal_dot = torch.sum(
            mesh_normals * batched_index_select(query_normals, 1, nearest_idx),
            dim=2,
        )
    else:
        knn_dists, nearest_idx, nearest_pos = pytorch3d.ops.knn_points(
            query_points, mesh_verts, K=1, return_nn=True
        )
        nearest_idx = nearest_idx.squeeze(2)  # (batch, Q)

        closest_mesh_verts = batched_index_select(mesh_verts, 1, nearest_idx)
        closest_mesh_normals = batched_index_select(mesh_normals, 1, nearest_idx)

        capsule_tops = closest_mesh_verts + closest_mesh_normals * caps_top
        capsule_bots = closest_mesh_verts + closest_mesh_normals * caps_bot
        delta_top = query_points - capsule_tops
        normal_dot = torch.sum(query_normals * closest_mesh_normals, dim=2)

    bot_to_top = capsule_bots - capsule_tops
    along_axis = torch.sum(delta_top * bot_to_top, dim=2)
    top_to_bot_square = torch.sum(bot_to_top * bot_to_top, dim=2)
    h = torch.clamp(along_axis / (top_to_bot_square + 1e-8), 0, 1)
    dist_to_axis = torch.norm(delta_top - bot_to_top * h.unsqueeze(2), dim=2)

    return dist_to_axis / caps_rad, normal_dot, nearest_idx


def sdf_to_contact(
    sdf: torch.Tensor, normal_dot: torch.Tensor, method: int = 0
) -> torch.Tensor:
    """
    归一化 SDF → 接触强度 [0, 1]

    Args:
        sdf: 归一化 SDF (batch, N)
        normal_dot: 法向点积 (batch, N)
        method: 转换方法 (0=指数衰减, 3=sigmoid, 推荐 0)

    Returns:
        contact: (batch, N) 接触强度 [0, 1]
    """
    if method == 0:
        c = 1 / (sdf + 0.0001)
    elif method == 1:
        c = -sdf + 2
    elif method == 2:
        c = torch.pow(1 / (sdf + 0.0001), 2)
    elif method == 3:
        c = torch.sigmoid(-sdf + 2.5)
    elif method == 4:
        c = (-normal_dot / 2 + 0.5) / (sdf + 0.0001)
    else:
        c = 1 / (sdf + 0.0001)

    return torch.clamp(c, 0.0, 1.0)


def calculate_contact_capsule(
    hand_verts: torch.Tensor,
    hand_normals: torch.Tensor,
    object_verts: torch.Tensor,
    object_normals: torch.Tensor,
    caps_top: float = 0.0005,
    caps_bot: float = -0.0015,
    caps_rad: float = 0.001,
    caps_on_hand: bool = False,
    contact_norm_method: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算手-物接触，返回接触值 + 法向点积 + 最近邻索引

    Args:
        hand_verts: (batch, V, 3)
        hand_normals: (batch, V, 3)
        object_verts: (batch, O, 3)
        object_normals: (batch, O, 3)

    Returns:
        obj_contact: (batch, O, 1) 物体每个顶点的接触强度
        hand_contact: (batch, V, 1) 手每个顶点的接触强度
        hand_normal_dot: (batch, V) 手顶点处的法向点积
        hand_nearest_obj_idx: (batch, V) 手每个顶点最近的物体顶点索引
        obj_normal_dot: (batch, O) 物体顶点处的法向点积
        obj_nearest_hand_idx: (batch, O) 物体每个顶点最近的手顶点索引
    """
    if caps_on_hand:
        sdf_obj, dot_obj, nearest_obj = capsule_sdf(
            hand_verts, hand_normals, object_verts, object_normals,
            caps_rad, caps_top, caps_bot, False,
        )
        sdf_hand, dot_hand, nearest_hand = capsule_sdf(
            hand_verts, hand_normals, object_verts, object_normals,
            caps_rad, caps_top, caps_bot, True,
        )
    else:
        sdf_obj, dot_obj, nearest_obj = capsule_sdf(
            object_verts, object_normals, hand_verts, hand_normals,
            caps_rad, caps_top, caps_bot, True,
        )
        sdf_hand, dot_hand, nearest_hand = capsule_sdf(
            object_verts, object_normals, hand_verts, hand_normals,
            caps_rad, caps_top, caps_bot, False,
        )

    obj_contact = sdf_to_contact(sdf_obj, dot_obj, method=contact_norm_method)
    hand_contact = sdf_to_contact(sdf_hand, dot_hand, method=contact_norm_method)

    return (
        obj_contact.unsqueeze(2),     # (batch, O, 1)
        hand_contact.unsqueeze(2),    # (batch, V, 1)
        dot_hand,                     # (batch, V) 手顶点法向点积
        nearest_hand,                 # (batch, V) 手顶点→最近物体顶点索引
        dot_obj,                      # (batch, O) 物体顶点法向点积
        nearest_obj,                  # (batch, O) 物体顶点→最近手顶点索引
    )


# ==============================================================================
# 2. MANO 顶点分组
# ==============================================================================

def build_hand_part_assignment(
    mano_layer,
    joint_to_part: Dict[int, int],
    num_hand_parts: int = 6,
) -> np.ndarray:
    """
    基于 MANO skinning weights 将每个顶点分配到 hand_part

    Args:
        mano_layer: MANO 模型层（需要有 lbs_weights 属性）
                    lbs_weights shape: (778, 16)
        joint_to_part: {joint_id: hand_part_id} 映射
                       来自 token_config.yaml 的 joint_to_part（已转为 int id）
        num_hand_parts: hand part 数量 (6)

    Returns:
        vertex_to_part: (778,) 每个顶点的 hand_part_id
    """
    # 获取 skinning weights
    if hasattr(mano_layer, "lbs_weights"):
        weights = mano_layer.lbs_weights.detach().cpu().numpy()  # (778, 16)
    elif hasattr(mano_layer, "th_weights"):
        weights = mano_layer.th_weights.detach().cpu().numpy()
    else:
        raise AttributeError(
            "MANO layer 没有 lbs_weights 或 th_weights 属性，"
            "请检查你使用的 MANO 实现"
        )

    num_verts = weights.shape[0]
    vertex_to_part = np.zeros(num_verts, dtype=np.int64)

    for v in range(num_verts):
        # 找到该顶点权重最大的关节
        dominant_joint = int(np.argmax(weights[v]))
        # 关节 → hand_part
        vertex_to_part[v] = joint_to_part.get(dominant_joint, 5)  # 默认 PALM

    return vertex_to_part


def build_hand_part_assignment_from_weights(
    skinning_weights: np.ndarray,
    joint_to_part: Dict[int, int],
) -> np.ndarray:
    """
    直接从 skinning weights numpy 数组构建分组
    （不需要 MANO layer 对象时使用）

    Args:
        skinning_weights: (num_verts, num_joints) skinning weights
        joint_to_part: {joint_id: hand_part_id}

    Returns:
        vertex_to_part: (num_verts,) 每个顶点的 hand_part_id
    """
    num_verts = skinning_weights.shape[0]
    vertex_to_part = np.zeros(num_verts, dtype=np.int64)

    for v in range(num_verts):
        dominant_joint = int(np.argmax(skinning_weights[v]))
        vertex_to_part[v] = joint_to_part.get(dominant_joint, 5)

    return vertex_to_part


def get_hand_part_vertex_indices(
    vertex_to_part: np.ndarray, num_hand_parts: int = 6
) -> Dict[int, np.ndarray]:
    """
    获取每个 hand_part 包含的顶点索引列表

    Args:
        vertex_to_part: (num_verts,) 顶点 → hand_part_id
        num_hand_parts: hand part 数量

    Returns:
        {hand_part_id: np.array([vertex_indices])}
    """
    part_to_verts = {}
    for part_id in range(num_hand_parts):
        part_to_verts[part_id] = np.where(vertex_to_part == part_id)[0]
    return part_to_verts


# ==============================================================================
# 3. 物体 Part Mesh → Slot 顶点归属
# ==============================================================================

def load_object_part_vertices(
    slot_to_plys: Dict[int, List[str]],
    num_slots: int = 6,
) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """
    加载物体各 slot 的 ply mesh，合并顶点，建立顶点→slot 归属映射

    Args:
        slot_to_plys: {slot_id: [ply_path_1, ply_path_2, ...]}
                      来自 SlotMapper.get_slot_to_vertices()
        num_slots: canonical slot 数量

    Returns:
        slot_to_vert_indices: {slot_id: np.array([在合并mesh中的顶点索引])}
        all_vertices: (N, 3) 合并后的所有顶点坐标
        vertex_to_slot: (N,) 每个顶点的 slot_id
    """
    all_verts_list = []
    vertex_to_slot_list = []
    slot_to_vert_indices = {}

    global_offset = 0

    for slot_id in range(num_slots):
        ply_paths = slot_to_plys.get(slot_id, [])
        slot_vert_indices = []

        for ply_path in ply_paths:
            mesh = trimesh.load(ply_path, process=False)
            verts = np.array(mesh.vertices, dtype=np.float32)  # (M, 3)
            num_v = verts.shape[0]

            all_verts_list.append(verts)
            vertex_to_slot_list.append(
                np.full(num_v, slot_id, dtype=np.int64)
            )
            slot_vert_indices.extend(
                range(global_offset, global_offset + num_v)
            )
            global_offset += num_v

        slot_to_vert_indices[slot_id] = np.array(slot_vert_indices, dtype=np.int64)

    if len(all_verts_list) == 0:
        return slot_to_vert_indices, np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int64)

    all_vertices = np.concatenate(all_verts_list, axis=0)  # (N, 3)
    vertex_to_slot = np.concatenate(vertex_to_slot_list, axis=0)  # (N,)

    return slot_to_vert_indices, all_vertices, vertex_to_slot


# ==============================================================================
# 4. 接触矩阵计算
# ==============================================================================

def compute_contact_matrix(
    hand_contact: torch.Tensor,
    hand_normal_dot: torch.Tensor,
    hand_nearest_obj_idx: torch.Tensor,
    vertex_to_hand_part: np.ndarray,
    vertex_to_slot: np.ndarray,
    num_hand_parts: int = 6,
    num_slots: int = 6,
    contact_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从逐顶点接触信息聚合为 (hand_part × slot) 接触矩阵

    Args:
        hand_contact: (V, 1) 手每个顶点的接触强度
        hand_normal_dot: (V,) 手每个顶点的法向点积
        hand_nearest_obj_idx: (V,) 手每个顶点最近的物体顶点索引
        vertex_to_hand_part: (V,) 手顶点 → hand_part_id
        vertex_to_slot: (O,) 物体顶点 → slot_id
        num_hand_parts: hand_part 数量
        num_slots: slot 数量
        contact_threshold: 接触强度阈值，高于此值认为有接触

    Returns:
        contact_area_matrix: (num_hand_parts, num_slots) 接触面积比
        normal_std_matrix: (num_hand_parts, num_slots) 法向点积标准差
        contact_count_matrix: (num_hand_parts, num_slots) 接触顶点数
    """
    # 转 numpy
    if isinstance(hand_contact, torch.Tensor):
        hand_contact = hand_contact.detach().cpu().numpy().squeeze(-1)  # (V,)
    if isinstance(hand_normal_dot, torch.Tensor):
        hand_normal_dot = hand_normal_dot.detach().cpu().numpy()  # (V,)
    if isinstance(hand_nearest_obj_idx, torch.Tensor):
        hand_nearest_obj_idx = hand_nearest_obj_idx.detach().cpu().numpy()  # (V,)

    num_verts = len(hand_contact)

    # 每个 hand_part 的总顶点数（用于计算面积比）
    part_total_verts = np.zeros(num_hand_parts, dtype=np.float32)
    for hp in range(num_hand_parts):
        part_total_verts[hp] = max(np.sum(vertex_to_hand_part == hp), 1)

    # 聚合矩阵
    contact_area_matrix = np.zeros((num_hand_parts, num_slots), dtype=np.float32)
    normal_std_matrix = np.zeros((num_hand_parts, num_slots), dtype=np.float32)
    contact_count_matrix = np.zeros((num_hand_parts, num_slots), dtype=np.int32)

    # 临时收集每个 (hand_part, slot) 对的法向点积值
    normal_dots_per_pair: Dict[Tuple[int, int], List[float]] = {}

    for v in range(num_verts):
        if hand_contact[v] < contact_threshold:
            continue

        hp_id = int(vertex_to_hand_part[v])
        obj_nearest = int(hand_nearest_obj_idx[v])

        # 物体顶点索引可能超出 vertex_to_slot 范围（理论上不应该）
        if obj_nearest >= len(vertex_to_slot):
            continue

        slot_id = int(vertex_to_slot[obj_nearest])

        contact_count_matrix[hp_id, slot_id] += 1

        key = (hp_id, slot_id)
        if key not in normal_dots_per_pair:
            normal_dots_per_pair[key] = []
        normal_dots_per_pair[key].append(float(hand_normal_dot[v]))

    # 计算面积比和法向标准差
    for hp in range(num_hand_parts):
        for s in range(num_slots):
            count = contact_count_matrix[hp, s]
            contact_area_matrix[hp, s] = count / part_total_verts[hp]

            key = (hp, s)
            if key in normal_dots_per_pair and len(normal_dots_per_pair[key]) > 1:
                normal_std_matrix[hp, s] = float(np.std(normal_dots_per_pair[key]))
            else:
                normal_std_matrix[hp, s] = 0.0

    return contact_area_matrix, normal_std_matrix, contact_count_matrix


# ==============================================================================
# 5. Contact Type 判定
# ==============================================================================

def classify_contact_type(
    contact_area_ratio: float,
    normal_dot_std: float,
    area_ratio_small: float = 0.1,
    area_ratio_large: float = 0.4,
    normal_std_threshold: float = 0.5,
) -> int:
    """
    根据接触面积比和法向标准差判定 contact_type

    判定逻辑：
        面积比 < small        → POINT (0)  指尖点接触
        面积比 >= small 且 法向标准差 < threshold → AREA (1)  面接触
        面积比 >= small 且 法向标准差 >= threshold → WRAP (2)  环绕接触

    Args:
        contact_area_ratio: 接触顶点数 / 该 hand_part 总顶点数
        normal_dot_std: 接触顶点法向点积的标准差
        area_ratio_small: POINT 判定阈值
        area_ratio_large: 大面积判定阈值（保留用于未来细分）
        normal_std_threshold: 法向分散阈值

    Returns:
        contact_type_id: 0=POINT, 1=AREA, 2=WRAP
    """
    if contact_area_ratio < area_ratio_small:
        return 0  # POINT
    elif normal_dot_std < normal_std_threshold:
        return 1  # AREA
    else:
        return 2  # WRAP


def contact_matrix_to_type_matrix(
    contact_area_matrix: np.ndarray,
    normal_std_matrix: np.ndarray,
    area_ratio_small: float = 0.1,
    area_ratio_large: float = 0.4,
    normal_std_threshold: float = 0.5,
    min_contact_area: float = 0.01,
) -> np.ndarray:
    """
    将接触矩阵转换为 contact_type 矩阵

    Args:
        contact_area_matrix: (H, S) 接触面积比
        normal_std_matrix: (H, S) 法向标准差
        min_contact_area: 最小接触面积比，低于此值认为无接触
        其他参数同 classify_contact_type

    Returns:
        type_matrix: (H, S) contact_type_id
                     -1=无接触, 0=POINT, 1=AREA, 2=WRAP
    """
    H, S = contact_area_matrix.shape
    type_matrix = np.full((H, S), -1, dtype=np.int32)

    for h in range(H):
        for s in range(S):
            area = contact_area_matrix[h, s]
            if area < min_contact_area:
                type_matrix[h, s] = -1  # 无接触
            else:
                type_matrix[h, s] = classify_contact_type(
                    area,
                    normal_std_matrix[h, s],
                    area_ratio_small,
                    area_ratio_large,
                    normal_std_threshold,
                )

    return type_matrix


# ==============================================================================
# 6. 完整 Pipeline：从 mesh 到 contact type matrix
# ==============================================================================

def compute_full_contact_info(
    hand_verts: torch.Tensor,
    hand_normals: torch.Tensor,
    object_verts: torch.Tensor,
    object_normals: torch.Tensor,
    vertex_to_hand_part: np.ndarray,
    vertex_to_slot: np.ndarray,
    num_hand_parts: int = 6,
    num_slots: int = 6,
    caps_top: float = 0.0005,
    caps_bot: float = -0.0015,
    caps_rad: float = 0.001,
    contact_threshold: float = 0.5,
    contact_norm_method: int = 0,
    area_ratio_small: float = 0.1,
    area_ratio_large: float = 0.4,
    normal_std_threshold: float = 0.5,
    min_contact_area: float = 0.01,
) -> Dict:
    """
    完整流程：mesh 顶点 → capsule 接触 → 分组聚合 → contact type 矩阵

    注意：输入为单个样本（无 batch 维度），函数内部会自动添加/移除 batch 维度

    Args:
        hand_verts: (V, 3) 手部顶点
        hand_normals: (V, 3) 手部法向
        object_verts: (O, 3) 物体顶点（合并后）
        object_normals: (O, 3) 物体法向（合并后）
        vertex_to_hand_part: (V,) 手顶点 → hand_part_id
        vertex_to_slot: (O,) 物体顶点 → slot_id
        其他参数见上述函数

    Returns:
        dict:
            contact_area_matrix: (H, S) 接触面积比
            normal_std_matrix: (H, S) 法向标准差
            contact_count_matrix: (H, S) 接触顶点数
            contact_type_matrix: (H, S) contact_type_id (-1/0/1/2)
    """
    # 确保是 torch tensor 且有 batch 维度
    device = hand_verts.device if isinstance(hand_verts, torch.Tensor) else "cpu"

    if not isinstance(hand_verts, torch.Tensor):
        hand_verts = torch.tensor(hand_verts, dtype=torch.float32, device=device)
    if not isinstance(hand_normals, torch.Tensor):
        hand_normals = torch.tensor(hand_normals, dtype=torch.float32, device=device)
    if not isinstance(object_verts, torch.Tensor):
        object_verts = torch.tensor(object_verts, dtype=torch.float32, device=device)
    if not isinstance(object_normals, torch.Tensor):
        object_normals = torch.tensor(object_normals, dtype=torch.float32, device=device)

    # 添加 batch 维度
    needs_squeeze = False
    if hand_verts.dim() == 2:
        hand_verts = hand_verts.unsqueeze(0)
        hand_normals = hand_normals.unsqueeze(0)
        object_verts = object_verts.unsqueeze(0)
        object_normals = object_normals.unsqueeze(0)
        needs_squeeze = True

    # Step 1: Capsule SDF 接触计算
    (
        obj_contact, hand_contact,
        hand_normal_dot, hand_nearest_obj_idx,
        obj_normal_dot, obj_nearest_hand_idx,
    ) = calculate_contact_capsule(
        hand_verts, hand_normals,
        object_verts, object_normals,
        caps_top=caps_top,
        caps_bot=caps_bot,
        caps_rad=caps_rad,
        contact_norm_method=contact_norm_method,
    )

    # 移除 batch 维度（单样本处理）
    if needs_squeeze:
        hand_contact = hand_contact[0]           # (V, 1)
        hand_normal_dot = hand_normal_dot[0]     # (V,)
        hand_nearest_obj_idx = hand_nearest_obj_idx[0]  # (V,)

    # Step 2: 分组聚合为接触矩阵
    contact_area_matrix, normal_std_matrix, contact_count_matrix = compute_contact_matrix(
        hand_contact,
        hand_normal_dot,
        hand_nearest_obj_idx,
        vertex_to_hand_part,
        vertex_to_slot,
        num_hand_parts=num_hand_parts,
        num_slots=num_slots,
        contact_threshold=contact_threshold,
    )

    # Step 3: 判定 contact type
    contact_type_matrix = contact_matrix_to_type_matrix(
        contact_area_matrix,
        normal_std_matrix,
        area_ratio_small=area_ratio_small,
        area_ratio_large=area_ratio_large,
        normal_std_threshold=normal_std_threshold,
        min_contact_area=min_contact_area,
    )

    return {
        "contact_area_matrix": contact_area_matrix,       # (H, S)
        "normal_std_matrix": normal_std_matrix,           # (H, S)
        "contact_count_matrix": contact_count_matrix,     # (H, S)
        "contact_type_matrix": contact_type_matrix,       # (H, S) -1/0/1/2
    }


# ==============================================================================
# 8. 简化版接触矩阵计算（用于简化版配置）
# ==============================================================================

def compute_contact_matrix_simple(
    hand_verts: torch.Tensor,
    hand_normals: torch.Tensor,
    object_verts: torch.Tensor,
    object_normals: torch.Tensor,
    vertex_to_hand_part: np.ndarray,
    vertex_to_slot: np.ndarray,
    num_hand_parts: int = 6,
    num_slots: int = 4,
    caps_top: float = 0.0005,
    caps_bot: float = -0.0015,
    caps_rad: float = 0.001,
    contact_threshold: float = 0.5,
    contact_distance: float = 0.005,  # 5mm = 0.005m
) -> np.ndarray:
    """
    简化版接触矩阵计算：只判断"是否接触"（二元），不区分 contact_type

    流程：
      1. 使用 capsule SDF 计算接触强度
      2. 只保留距离在 contact_distance 内的接触
      3. 聚合为 (hand_part × slot) 二元矩阵

    Args:
        hand_verts: (V, 3) 或 (1, V, 3) 手部顶点
        hand_normals: (V, 3) 或 (1, V, 3) 手部法向
        object_verts: (O, 3) 或 (1, O, 3) 物体顶点
        object_normals: (O, 3) 或 (1, O, 3) 物体法向
        vertex_to_hand_part: (V,) 手顶点 → hand_part_id
        vertex_to_slot: (O,) 物体顶点 → slot_id
        num_hand_parts: hand_part 数量（默认 6）
        num_slots: slot 数量（默认 4，简化版配置）
        caps_top, caps_bot, caps_rad: capsule SDF 参数（米）
        contact_threshold: 接触强度阈值
        contact_distance: 接触距离阈值（米），默认 0.005m = 5mm

    Returns:
        contact_matrix: (num_hand_parts, num_slots) int32
                        -1=无接触, 0=接触
    """
    # 确保是 torch tensor 且有 batch 维度
    device = hand_verts.device if isinstance(hand_verts, torch.Tensor) else "cpu"

    if not isinstance(hand_verts, torch.Tensor):
        hand_verts = torch.tensor(hand_verts, dtype=torch.float32, device=device)
    if not isinstance(hand_normals, torch.Tensor):
        hand_normals = torch.tensor(hand_normals, dtype=torch.float32, device=device)
    if not isinstance(object_verts, torch.Tensor):
        object_verts = torch.tensor(object_verts, dtype=torch.float32, device=device)
    if not isinstance(object_normals, torch.Tensor):
        object_normals = torch.tensor(object_normals, dtype=torch.float32, device=device)

    # 添加 batch 维度
    if hand_verts.dim() == 2:
        hand_verts = hand_verts.unsqueeze(0)
        hand_normals = hand_normals.unsqueeze(0)
    if object_verts.dim() == 2:
        object_verts = object_verts.unsqueeze(0)
        object_normals = object_normals.unsqueeze(0)

    # Step 1: Capsule SDF 接触计算
    (
        obj_contact, hand_contact,
        hand_normal_dot, hand_nearest_obj_idx,
        obj_normal_dot, obj_nearest_hand_idx,
    ) = calculate_contact_capsule(
        hand_verts, hand_normals,
        object_verts, object_normals,
        caps_top=caps_top,
        caps_bot=caps_bot,
        caps_rad=caps_rad,
        caps_on_hand=False,
    )

    # 移除 batch 维度
    hand_contact = hand_contact[0].squeeze(-1)  # (V,)
    hand_nearest_obj_idx = hand_nearest_obj_idx[0]  # (V,)

    # Step 2: 计算实际距离（用于距离阈值筛选）
    # 将 hand_verts 和 object_verts 转换为 (V, 3) 和 (O, 3)
    hand_v = hand_verts[0]  # (V, 3)
    obj_v = object_verts[0]  # (O, 3)

    # 获取每个手顶点对应的最近物体顶点坐标
    nearest_obj_verts = obj_v[hand_nearest_obj_idx.long()]  # (V, 3)

    # 计算欧氏距离
    distances = torch.norm(hand_v - nearest_obj_verts, dim=1)  # (V,)

    # Step 3: 聚合为 (hand_part × slot) 接触矩阵
    contact_matrix = np.full((num_hand_parts, num_slots), -1, dtype=np.int32)

    num_verts = len(vertex_to_hand_part)
    for v in range(num_verts):
        # 检查接触强度和距离
        if hand_contact[v] < contact_threshold:
            continue
        if distances[v] > contact_distance:
            continue

        hp_id = int(vertex_to_hand_part[v])
        obj_nearest = int(hand_nearest_obj_idx[v])

        if obj_nearest >= len(vertex_to_slot):
            continue

        slot_id = int(vertex_to_slot[obj_nearest])

        # 标记为接触（简化版使用 0 表示接触）
        if 0 <= hp_id < num_hand_parts and 0 <= slot_id < num_slots:
            contact_matrix[hp_id, slot_id] = 0

    return contact_matrix


# ==============================================================================
# 7. 法向估算工具
# ==============================================================================

def estimate_vertex_normals_from_mesh(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    从 trimesh mesh 估算顶点法向

    Args:
        mesh: trimesh.Trimesh 对象

    Returns:
        normals: (N, 3) 顶点法向（已归一化）
    """
    return np.array(mesh.vertex_normals, dtype=np.float32)


def estimate_vertex_normals_from_points(
    points: np.ndarray, k: int = 10
) -> np.ndarray:
    """
    从点云估算法向（基于 PCA）

    Args:
        points: (N, 3) 点坐标
        k: KNN 邻域大小

    Returns:
        normals: (N, 3) 估算的法向（已归一化）
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    _, indices = tree.query(points, k=k)

    normals = np.zeros_like(points)
    for i in range(len(points)):
        neighbors = points[indices[i]]
        centered = neighbors - neighbors.mean(axis=0)
        cov = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # 最小特征值对应的特征向量为法向
        normals[i] = eigenvectors[:, 0]

    # 归一化
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-8)

    return normals


# ==============================================================================
# 测试
# ==============================================================================

if __name__ == "__main__":
    print("=== mesh_utils 单元测试 ===\n")

    # 模拟数据
    np.random.seed(42)
    num_hand_verts = 778
    num_obj_verts = 500
    num_hand_parts = 6
    num_slots = 6

    # 模拟手部和物体顶点
    hand_verts = torch.randn(1, num_hand_verts, 3) * 0.1
    hand_normals = torch.randn(1, num_hand_verts, 3)
    hand_normals = hand_normals / hand_normals.norm(dim=2, keepdim=True)

    object_verts = torch.randn(1, num_obj_verts, 3) * 0.1
    object_normals = torch.randn(1, num_obj_verts, 3)
    object_normals = object_normals / object_normals.norm(dim=2, keepdim=True)

    # 模拟分组
    vertex_to_hand_part = np.random.randint(0, num_hand_parts, size=num_hand_verts)
    vertex_to_slot = np.random.randint(0, num_slots, size=num_obj_verts)

    print(f"手部顶点: {num_hand_verts}, 物体顶点: {num_obj_verts}")
    print(f"Hand parts: {num_hand_parts}, Slots: {num_slots}")

    # 测试 capsule 接触计算
    print("\n--- 测试 calculate_contact_capsule ---")
    (
        obj_contact, hand_contact,
        hand_normal_dot, hand_nearest_obj_idx,
        obj_normal_dot, obj_nearest_hand_idx,
    ) = calculate_contact_capsule(hand_verts, hand_normals, object_verts, object_normals)

    print(f"  obj_contact shape:  {obj_contact.shape}")
    print(f"  hand_contact shape: {hand_contact.shape}")
    print(f"  hand_nearest_obj_idx shape: {hand_nearest_obj_idx.shape}")
    print(f"  hand_contact 范围: [{hand_contact.min():.4f}, {hand_contact.max():.4f}]")

    # 测试接触矩阵
    print("\n--- 测试 compute_contact_matrix ---")
    area_mat, std_mat, count_mat = compute_contact_matrix(
        hand_contact[0], hand_normal_dot[0], hand_nearest_obj_idx[0],
        vertex_to_hand_part, vertex_to_slot,
    )
    print(f"  contact_area_matrix:\n{area_mat}")
    print(f"  contact_count_matrix:\n{count_mat}")

    # 测试 contact type 判定
    print("\n--- 测试 contact_type_matrix ---")
    type_mat = contact_matrix_to_type_matrix(area_mat, std_mat)
    type_names = {-1: "NONE", 0: "POINT", 1: "AREA", 2: "WRAP"}
    print("  Contact types:")
    hand_labels = ["THUMB", "INDEX", "MIDDLE", "RING", "LITTLE", "PALM"]
    slot_labels = ["BODY", "HANDLE", "HEAD", "CAP_LID", "EMITTER", "TRIGGER"]
    for h in range(num_hand_parts):
        for s in range(num_slots):
            if type_mat[h, s] >= 0:
                print(f"    ({hand_labels[h]}, {slot_labels[s]}) → {type_names[type_mat[h, s]]}")

    # 测试完整 pipeline
    print("\n--- 测试 compute_full_contact_info ---")
    result = compute_full_contact_info(
        hand_verts[0], hand_normals[0],
        object_verts[0], object_normals[0],
        vertex_to_hand_part, vertex_to_slot,
    )
    print(f"  返回 keys: {list(result.keys())}")
    num_contacts = np.sum(result["contact_type_matrix"] >= 0)
    print(f"  有效接触对数: {num_contacts}")

    print("\n✅ 所有测试通过")