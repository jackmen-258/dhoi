"""
contact_builder.py
==================
为 OakInk 数据集中的每个 grasp 样本构建 (hand_part × slot) 接触矩阵（简化版）

简化版变更：
  - 接触矩阵从 (6, 6) 简化为 (6, 4)
  - 移除 contact_type 维度（不再区分 POINT/AREA/WRAP）
  - 简化接触判定：只保留距离阈值，移除面积/法向标准差判断
  - 输出 contact_matrix: 0=接触, -1=无接触

路径解析逻辑（OakInk 数据结构）：
  obj_id (如 "C26001")
    → metaV2/object_id.json 查 name → "contactpose_fryingpan"
    → $OAKINK_DIR/OakBase/{cate_id}/{name}/part_01.json
       即 OakBase/frying_pan/contactpose_fryingpan/part_01.json

  对于虚拟物体 (is_virtual=True):
    obj_id 在 virtual_object_id.json 中
    part 标注复用 raw_obj_id 对应的真实物体

依赖：
  - data.slot_mapping.SlotMapper
  - utils.mesh_utils
"""

import os
import json
import numpy as np
import torch
import trimesh
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.slot_mapping import SlotMapper
from utils.mesh_utils import (
    compute_contact_matrix_simple,
    estimate_vertex_normals_from_mesh,
    estimate_vertex_normals_from_points,
    build_hand_part_assignment_from_weights,
    load_object_part_vertices,
)


class ContactBuilder:
    """
    为 OakInk 数据集中的每个 grasp 构建接触矩阵（简化版）

    用法：
        builder = ContactBuilder("configs/token_config.yaml")
        builder.init_hand_part_assignment(mano_layer=dataset.mano_layer)

        result = builder.build_contact_for_grasp(
            hand_verts=grasp["hand_verts"],
            obj_id=grasp["obj_id"],
            cate_id=grasp["cate_id"],
            raw_obj_id=grasp.get("raw_obj_id"),
        )
    """

    def __init__(self, config_path: str, oakink_root: str = None):
        """
        Args:
            config_path: token_config.yaml 路径
            oakink_root: OakInk 数据根目录（$OAKINK_DIR），不提供则从环境变量读取
        """
        self.mapper = SlotMapper(config_path)

        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # 简化版：只保留距离阈值
        ct = cfg["preprocessing"]["contact_thresholds"]
        # contact_distance 配置值单位为 mm（如 5.0），转换为米（0.005）
        self.contact_distance = ct["contact_distance"] * 0.001

        # 从配置读取 slot 数量
        self.num_slots = cfg["canonical_slots"]["num"]

        # capsule SDF 参数（与 OakInk 默认值一致）
        self.caps_top = 0.0005
        self.caps_bot = -0.0015
        self.caps_rad = 0.001
        self.contact_threshold = 0.5

        # MANO 顶点分组（延迟初始化）
        self._vertex_to_hand_part = None

        # 物体 part 缓存: {cache_key: (slot_to_vert_indices, verts, v2slot, normals)}
        self._obj_part_cache: Dict[str, Tuple] = {}

        # joint_to_part: {int joint_id: int part_id}
        self._joint_to_part_id = {}
        hand_part_to_id = {h: i for i, h in enumerate(cfg["hand_parts"]["labels"])}
        for joint_id_str, part_name in cfg["hand_parts"]["joint_to_part"].items():
            self._joint_to_part_id[int(joint_id_str)] = hand_part_to_id[part_name]

        # ====== OakInk 路径 ======
        if oakink_root is None:
            oakink_root = os.environ.get("OAKINK_DIR", "")
        self.oakink_root = oakink_root
        self.oakbase_dir = os.path.join(oakink_root, "OakBase")

        # ====== 加载 obj_id → name 映射 ======
        # object_id.json: 真实物体, virtual_object_id.json: 虚拟物体
        self._obj_id_to_name: Dict[str, str] = {}
        self._obj_id_to_info: Dict[str, dict] = {}
        meta_dir = os.path.join(oakink_root, "shape", "metaV2")
        for meta_file in ["object_id.json", "virtual_object_id.json"]:
            meta_path = os.path.join(meta_dir, meta_file)
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                for oid, info in meta.items():
                    self._obj_id_to_name[oid] = info["name"]
                    self._obj_id_to_info[oid] = info

        print(f"[ContactBuilder] 加载 {len(self._obj_id_to_name)} 个 obj_id→name 映射")
        print(f"[ContactBuilder] OakBase: {self.oakbase_dir}")

    # ==========================================================================
    # MANO 顶点分组
    # ==========================================================================

    def init_hand_part_assignment(
        self, mano_layer=None, skinning_weights: np.ndarray = None,
    ):
        """
        初始化 MANO 顶点 → hand_part 分组

        Args:
            mano_layer: ManoLayer 实例
            skinning_weights: (778, 16) numpy array
        """
        if skinning_weights is not None:
            weights = skinning_weights
        elif mano_layer is not None:
            if hasattr(mano_layer, "th_weights"):
                weights = mano_layer.th_weights.detach().cpu().numpy()
            elif hasattr(mano_layer, "lbs_weights"):
                weights = mano_layer.lbs_weights.detach().cpu().numpy()
            else:
                raise AttributeError("MANO layer 缺少 skinning weights 属性")
        else:
            raise ValueError("必须提供 mano_layer 或 skinning_weights")

        self._vertex_to_hand_part = build_hand_part_assignment_from_weights(
            weights, self._joint_to_part_id
        )

        counts = np.bincount(self._vertex_to_hand_part, minlength=6)
        labels = self.mapper.hand_parts
        parts_str = ", ".join(f"{labels[i]}={counts[i]}" for i in range(6))
        print(f"[ContactBuilder] 顶点分组: {parts_str}")

    @property
    def vertex_to_hand_part(self) -> np.ndarray:
        if self._vertex_to_hand_part is None:
            raise RuntimeError("请先调用 init_hand_part_assignment()")
        return self._vertex_to_hand_part

    # ==========================================================================
    # Part 目录查找（核心路径解析）
    # ==========================================================================

    def _find_part_dir(
        self,
        obj_id: str,
        cate_id: str,
        raw_obj_id: str = None,
    ) -> Optional[str]:
        """
        查找物体的 part 标注目录

        路径解析顺序：
          1. obj_id → metaV2 查 name → OakBase/{cate_id}/{name}/
          2. 若 obj_id 查不到，尝试 raw_obj_id（虚拟物体复用真实物体 part）
          3. 在 OakBase/{cate_id}/ 下遍历查找（兜底）

        Args:
            obj_id: 物体 ID（如 "C26001"）
            cate_id: 物体类别（如 "frying_pan"）
            raw_obj_id: 真实物体 ID（虚拟物体需要，可选）

        Returns:
            包含 part_*.json 的目录路径，找不到返回 None
        """

        def _has_part_files(d: str) -> bool:
            """检查目录是否包含 part_*.json"""
            if not os.path.isdir(d):
                return False
            return any(
                f.startswith("part_") and f.endswith(".json")
                for f in os.listdir(d)
            )

        # 按优先级尝试的 obj_id 列表
        candidate_ids = [obj_id]
        if raw_obj_id and raw_obj_id != obj_id:
            candidate_ids.append(raw_obj_id)

        for oid in candidate_ids:
            obj_name = self._obj_id_to_name.get(oid)
            if obj_name is None:
                continue

            # 主路径: OakBase/{cate_id}/{obj_name}/
            part_dir = os.path.join(self.oakbase_dir, cate_id, obj_name)
            if _has_part_files(part_dir):
                return part_dir

            # 备选：OakBase 下其他类别目录（名称可能跨类别）
            for subdir in os.listdir(self.oakbase_dir):
                alt_dir = os.path.join(self.oakbase_dir, subdir, obj_name)
                if _has_part_files(alt_dir):
                    return alt_dir

        # 兜底：遍历 OakBase/{cate_id}/ 下所有子目录
        cate_dir = os.path.join(self.oakbase_dir, cate_id)
        if os.path.isdir(cate_dir):
            for name in os.listdir(cate_dir):
                d = os.path.join(cate_dir, name)
                if _has_part_files(d):
                    # 检查 obj_name 是否和任何已知 ID 匹配
                    # 这里只要找到有 part 的就返回（同一 cate 的 part 结构相同）
                    pass  # 不做兜底匹配，避免误匹配

        return None

    # ==========================================================================
    # 物体 Part Mesh 加载
    # ==========================================================================

    def load_object_parts_for_contact(
        self,
        obj_id: str,
        cate_id: str,
        raw_obj_id: str = None,
    ) -> Optional[Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]]:
        """
        加载物体 part mesh，构建 slot 归属和法向

        Args:
            obj_id: 物体 ID
            cate_id: 物体类别
            raw_obj_id: 真实物体 ID（虚拟物体需要）

        Returns:
            (slot_to_vert_indices, all_vertices, vertex_to_slot, all_normals)
            加载失败返回 None
        """
        # 缓存 key：用实际解析到的 part 目录路径，
        # 这样同一物体（包括虚拟变体共享同一 part）只加载一次
        cache_key = f"{cate_id}_{obj_id}"
        if raw_obj_id:
            cache_key += f"_{raw_obj_id}"

        if cache_key in self._obj_part_cache:
            return self._obj_part_cache[cache_key]

        # 查找 part 目录
        part_dir = self._find_part_dir(obj_id, cate_id, raw_obj_id)
        if part_dir is None:
            self._obj_part_cache[cache_key] = None
            return None

        # 也用 part_dir 做缓存（不同 obj_id 可能指向同一 part_dir）
        if part_dir in self._obj_part_cache:
            result = self._obj_part_cache[part_dir]
            self._obj_part_cache[cache_key] = result
            return result

        # 用 SlotMapper 解析 part json → slot 映射
        parts = self.mapper.load_object_parts(part_dir, cate_id)
        if not parts:
            self._obj_part_cache[cache_key] = None
            self._obj_part_cache[part_dir] = None
            return None

        slot_to_plys = self.mapper.get_slot_to_vertices(parts)
        if not slot_to_plys:
            self._obj_part_cache[cache_key] = None
            self._obj_part_cache[part_dir] = None
            return None

        # 加载所有 part 顶点，构建 vertex_to_slot
        slot_to_vert_indices, all_vertices, vertex_to_slot = load_object_part_vertices(
            slot_to_plys, num_slots=self.num_slots
        )

        if len(all_vertices) == 0:
            self._obj_part_cache[cache_key] = None
            self._obj_part_cache[part_dir] = None
            return None

        # 估算法向：逐 part mesh 获取 vertex normals
        all_normals_list = []
        for slot_id in range(self.num_slots):
            for ply_path in slot_to_plys.get(slot_id, []):
                try:
                    mesh = trimesh.load(ply_path, process=False)
                    normals = estimate_vertex_normals_from_mesh(mesh)
                except Exception:
                    try:
                        n_v = len(trimesh.load(ply_path, process=False).vertices)
                    except Exception:
                        n_v = 0
                    normals = np.zeros((n_v, 3), dtype=np.float32)
                all_normals_list.append(normals)

        if all_normals_list:
            all_normals = np.concatenate(all_normals_list, axis=0)
        else:
            all_normals = estimate_vertex_normals_from_points(all_vertices)

        # 修复无效法向
        norms = np.linalg.norm(all_normals, axis=1, keepdims=True)
        invalid = norms.squeeze() < 1e-8
        if invalid.any():
            fallback = estimate_vertex_normals_from_points(all_vertices)
            all_normals[invalid] = fallback[invalid]
            norms = np.linalg.norm(all_normals, axis=1, keepdims=True)
        all_normals = all_normals / np.maximum(norms, 1e-8)

        result = (slot_to_vert_indices, all_vertices, vertex_to_slot, all_normals)
        self._obj_part_cache[cache_key] = result
        self._obj_part_cache[part_dir] = result
        return result

    # ==========================================================================
    # 手部法向
    # ==========================================================================

    @staticmethod
    def estimate_hand_normals(
        hand_verts: np.ndarray, mano_faces: np.ndarray
    ) -> np.ndarray:
        """从 MANO mesh 计算顶点法向"""
        mesh = trimesh.Trimesh(
            vertices=hand_verts, faces=mano_faces, process=False
        )
        return np.array(mesh.vertex_normals, dtype=np.float32)

    @staticmethod
    def estimate_hand_normals_no_faces(hand_verts: np.ndarray) -> np.ndarray:
        """无 faces 时从点云估算法向"""
        return estimate_vertex_normals_from_points(hand_verts, k=10)

    # ==========================================================================
    # 核心：构建单个 grasp 的接触矩阵（简化版）
    # ==========================================================================

    def build_contact_for_grasp(
        self,
        hand_verts: np.ndarray,
        obj_id: str,
        cate_id: str,
        raw_obj_id: str = None,
        hand_normals: Optional[np.ndarray] = None,
        mano_faces: Optional[np.ndarray] = None,
    ) -> Optional[Dict]:
        """
        为单个 grasp 构建接触矩阵（简化版）

        Args:
            hand_verts: (778, 3) 手部顶点
            obj_id: 物体 ID
            cate_id: 物体类别
            raw_obj_id: 真实物体 ID（虚拟物体需要）
            hand_normals: (778, 3) 可选
            mano_faces: (F, 3) MANO 面片

        Returns:
            {
                contact_matrix: (6, 4) int, 0=接触, -1=无接触
                obj_id, cate_id: str
            }
            失败返回 None
        """
        obj_data = self.load_object_parts_for_contact(
            obj_id, cate_id, raw_obj_id
        )
        if obj_data is None:
            return None

        slot_to_vert_indices, obj_verts, vertex_to_slot, obj_normals = obj_data

        # 手部法向
        if hand_normals is None:
            if mano_faces is not None:
                hand_normals = self.estimate_hand_normals(hand_verts, mano_faces)
            else:
                hand_normals = self.estimate_hand_normals_no_faces(hand_verts)

        # 简化版：只计算二元接触矩阵
        contact_matrix = compute_contact_matrix_simple(
            hand_verts=torch.tensor(hand_verts, dtype=torch.float32),
            hand_normals=torch.tensor(hand_normals, dtype=torch.float32),
            object_verts=torch.tensor(obj_verts, dtype=torch.float32),
            object_normals=torch.tensor(obj_normals, dtype=torch.float32),
            vertex_to_hand_part=self.vertex_to_hand_part,
            vertex_to_slot=vertex_to_slot,
            num_hand_parts=len(self.mapper.hand_parts),
            num_slots=len(self.mapper.canonical_slots),
            caps_top=self.caps_top,
            caps_bot=self.caps_bot,
            caps_rad=self.caps_rad,
            contact_threshold=self.contact_threshold,
            contact_distance=self.contact_distance,
        )

        return {
            "contact_matrix": contact_matrix,
            "obj_id": obj_id,
            "cate_id": cate_id,
        }

    # ==========================================================================
    # 批量处理整个 OIShape 数据集
    # ==========================================================================

    def build_contact_for_dataset(
        self,
        oi_shape_dataset,
        mano_faces: np.ndarray,
        save_dir: str,
        skip_existing: bool = True,
    ) -> Dict:
        """
        遍历 OIShape 数据集，为每个 grasp 计算接触矩阵并保存（简化版）

        Args:
            oi_shape_dataset: OIShape 实例
            mano_faces: (F, 3) MANO 面片索引
            save_dir: 缓存保存目录
            skip_existing: 跳过已有缓存
        """
        os.makedirs(save_dir, exist_ok=True)

        grasp_list = oi_shape_dataset.grasp_list

        stats = {"total": len(grasp_list), "success": 0,
                 "skipped": 0, "failed": 0, "no_parts": 0}

        for idx in tqdm(range(len(grasp_list)), desc="Building contacts"):
            grasp = grasp_list[idx]
            obj_id = grasp["obj_id"]
            cate_id = grasp["cate_id"]
            raw_obj_id = grasp.get("raw_obj_id", obj_id)

            cache_name = f"{cate_id}_{obj_id}_{idx:06d}.npz"
            cache_path = os.path.join(save_dir, cache_name)

            if skip_existing and os.path.exists(cache_path):
                stats["skipped"] += 1
                continue

            hand_verts = grasp["hand_verts"]
            if hand_verts is None:
                stats["failed"] += 1
                continue

            result = self.build_contact_for_grasp(
                hand_verts=hand_verts,
                obj_id=obj_id,
                cate_id=cate_id,
                raw_obj_id=raw_obj_id,
                mano_faces=mano_faces,
            )

            if result is None:
                stats["no_parts"] += 1
                continue

            # 简化版：只保存 contact_matrix
            np.savez_compressed(
                cache_path,
                contact_matrix=result["contact_matrix"],
                obj_id=obj_id,
                cate_id=cate_id,
                grasp_idx=idx,
            )
            stats["success"] += 1

        with open(os.path.join(save_dir, "build_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\n[ContactBuilder] 完成: "
              f"成功={stats['success']}, 跳过={stats['skipped']}, "
              f"失败={stats['failed']}, 无part={stats['no_parts']}")
        return stats

    # ==========================================================================
    # 缓存加载
    # ==========================================================================

    @staticmethod
    def load_cached_contact(cache_path: str) -> Optional[Dict]:
        if not os.path.exists(cache_path):
            return None
        data = np.load(cache_path, allow_pickle=True)
        return {
            "contact_matrix": data["contact_matrix"],
            "obj_id": str(data["obj_id"]),
            "cate_id": str(data["cate_id"]),
        }

    # ==========================================================================
    # 诊断工具
    # ==========================================================================

    def diagnose_object(self, obj_id: str, cate_id: str, raw_obj_id: str = None):
        """
        诊断单个物体的 part 查找结果，用于调试路径问题

        Args:
            obj_id: 物体 ID
            cate_id: 物体类别
            raw_obj_id: 真实物体 ID
        """
        print(f"\n--- 诊断 obj_id={obj_id}, cate_id={cate_id}, raw={raw_obj_id} ---")

        # 查 meta 映射
        name = self._obj_id_to_name.get(obj_id)
        print(f"  obj_id→name: {name}")
        if name is None and raw_obj_id:
            name = self._obj_id_to_name.get(raw_obj_id)
            print(f"  raw_obj_id→name: {name}")

        if name is None:
            print(f"  ❌ obj_id 不在 metaV2 映射中")
            return

        # 查 OakBase 路径
        part_dir = os.path.join(self.oakbase_dir, cate_id, name)
        print(f"  预期路径: {part_dir}")
        print(f"  目录存在: {os.path.isdir(part_dir)}")

        if os.path.isdir(part_dir):
            files = os.listdir(part_dir)
            print(f"  目录内容: {files}")
            part_jsons = [f for f in files if f.startswith("part_") and f.endswith(".json")]
            print(f"  Part JSON: {part_jsons}")
        else:
            # 在 OakBase/{cate_id}/ 下看看有什么
            cate_dir = os.path.join(self.oakbase_dir, cate_id)
            if os.path.isdir(cate_dir):
                entries = os.listdir(cate_dir)
                print(f"  OakBase/{cate_id}/ 下有: {entries[:10]}...")
            else:
                print(f"  ❌ OakBase/{cate_id}/ 不存在")
                print(f"  OakBase 下有: {os.listdir(self.oakbase_dir)[:15]}...")

        # 尝试完整查找
        result_dir = self._find_part_dir(obj_id, cate_id, raw_obj_id)
        print(f"  _find_part_dir 结果: {result_dir}")


# ==============================================================================
# 测试
# ==============================================================================

if __name__ == "__main__":
    print("=== ContactBuilder 初始化测试（简化版）===\n")

    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "token_config.yaml"
    )

    oakink_root = os.environ.get("OAKINK_DIR", "")
    if not oakink_root:
        print("OAKINK_DIR 未设置，跳过路径测试")
        print("仅测试分组逻辑...\n")

        builder = ContactBuilder(config_path, oakink_root="/tmp/fake_oakink")
        np.random.seed(42)
        fake_weights = np.random.rand(778, 16).astype(np.float32)
        for v in range(778):
            fake_weights[v] *= 0.1
            fake_weights[v, v % 16] = 1.0
        fake_weights /= fake_weights.sum(axis=1, keepdims=True)
        builder.init_hand_part_assignment(skinning_weights=fake_weights)
        print("✅ 分组测试通过")
    else:
        builder = ContactBuilder(config_path)
        print(f"\n已加载映射数，测试路径解析...")

        # 测试几个典型物体
        test_cases = [
            ("C26001", "frying_pan", None),     # contactpose_fryingpan
            ("C22001", "hammer", None),          # contactpose_hammer
            ("C10001", "mug", None),             # contactpose_mug
            ("Y27035", "power_drill", None),     # 035_power_drill
        ]

        for obj_id, cate_id, raw_id in test_cases:
            builder.diagnose_object(obj_id, cate_id, raw_id)
