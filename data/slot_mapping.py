"""
slot_mapping.py
===============
OakInk part name → canonical slot 映射模块（简化版）

功能：
  1. 加载 token_config.yaml 中的映射规则
  2. raw_name → stripped → normalized → canonical_slot 全流程转换
  3. 加载物体的 part json，返回 {part_id: canonical_slot_id} 映射
  4. 提供 token_id ↔ (hand_part, slot) 编解码（移除 contact_type）

简化版变更：
  - 移除 contact_type 维度
  - Token 空间 = hand_parts (6) × canonical_slots (4) = 24
  - token_id = hand_part_id × num_slots + slot_id

用法：
  mapper = SlotMapper("configs/token_config.yaml")
  slot = mapper.raw_name_to_slot("scissor_handle", object_category="scissor")
  # → "GRASP_SURFACE"

  token_id = mapper.encode_token("INDEX", "GRASP_SURFACE")
  # → 4
  h, s = mapper.decode_token(4)
  # → ("INDEX", "GRASP_SURFACE")
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


class SlotMapper:
    """OakInk 部件名称 → canonical slot 映射器 + token 编解码器（简化版）"""

    def __init__(self, config_path: str):
        """
        Args:
            config_path: token_config.yaml 的路径
        """
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # 加载各维度标签
        self.hand_parts = self.config["hand_parts"]["labels"]
        self.canonical_slots = self.config["canonical_slots"]["labels"]

        # 构建 label → id 映射
        self.hand_part_to_id = {h: i for i, h in enumerate(self.hand_parts)}
        self.slot_to_id = {s: i for i, s in enumerate(self.canonical_slots)}

        # 加载名称归一化规则
        norm_cfg = self.config["name_normalization"]
        self.raw_to_normalized = norm_cfg["raw_to_normalized"]
        self.normalized_to_slot = norm_cfg["normalized_to_slot"]

        # 加载 token 编码参数
        enc = self.config["encoding"]
        self.hand_part_stride = enc["hand_part_stride"]

        # 特殊 token
        self.special_tokens = self.config["token_space"]["special_tokens"]
        self.vocab_size = self.config["token_space"]["vocab_size"]
        self.max_token_length = self.config["token_space"]["max_token_length"]
        self.num_semantic_tokens = self.config["token_space"]["num_semantic_tokens"]

        # MANO 关节 → hand_part 映射
        self.joint_to_part = {
            int(k): v for k, v in self.config["hand_parts"]["joint_to_part"].items()
        }

    # ==========================================================================
    # 名称归一化流程
    # ==========================================================================

    def strip_object_prefix(self, raw_name: str, object_category: str) -> str:
        """
        去掉物体类别前缀
        例: scissor_handle + scissor → handle
            bottle_body + cylinder_bottle → body (处理类别名含下划线的情况)
            camera_grip + cameras → grip (处理单复数不一致)
        """
        name = raw_name.lower().strip()
        category = object_category.lower().strip()

        # 构建候选前缀列表（包含单复数变体）
        candidates = [category]
        # 去掉末尾 s/es（复数→单数）
        if category.endswith("ses"):
            candidates.append(category[:-2])   # glasses → glass
        if category.endswith("ies"):
            candidates.append(category[:-3] + "y")  # batteries → battery
        if category.endswith("s") and not category.endswith("ss"):
            candidates.append(category[:-1])   # cameras → camera
        # 加末尾 s（单数→复数）
        candidates.append(category + "s")

        # 下划线变体：fryingpan ↔ frying_pan
        for c in list(candidates):
            # 去掉下划线
            no_underscore = c.replace("_", "")
            if no_underscore != c:
                candidates.append(no_underscore)
            # 如果类别含下划线，把每段也加入候选
            if "_" in c:
                candidates.extend(c.split("_"))

        # 去重，按长度降序（优先匹配最长的前缀）
        candidates = sorted(set(candidates), key=len, reverse=True)

        # 尝试去掉每个候选前缀
        for prefix in candidates:
            if name.startswith(prefix + "_"):
                name = name[len(prefix) + 1:]
                break
            elif name.startswith(prefix) and len(name) > len(prefix):
                rest = name[len(prefix):]
                if rest.startswith("_"):
                    name = rest.lstrip("_")
                    break

        return name if name else raw_name.lower()

    def normalize_name(self, stripped_name: str) -> Optional[str]:
        """
        同义词归一化（简化版配置中直接使用 raw_to_normalized）
        例: handle → grasp_surface, blade → functional_end

        Args:
            stripped_name: 已去掉物体前缀的名称

        Returns:
            归一化名称，未找到返回 None
        """
        name = stripped_name.lower().strip()

        # 直接匹配
        if name in self.raw_to_normalized:
            return self.raw_to_normalized[name]

        # 下划线变体匹配 (处理可能的空格/下划线不一致)
        name_underscore = name.replace(" ", "_")
        if name_underscore in self.raw_to_normalized:
            return self.raw_to_normalized[name_underscore]

        name_space = name.replace("_", " ")
        if name_space in self.raw_to_normalized:
            return self.raw_to_normalized[name_space]

        return None

    def raw_name_to_slot(self, raw_name: str, object_category: str) -> Optional[str]:
        """
        完整流程: raw_name → stripped → normalized → canonical_slot

        Args:
            raw_name: OakInk part json 中的 name 字段
            object_category: 物体类别名

        Returns:
            canonical slot 名称 (如 "GRASP_SURFACE")，未匹配返回 None
        """
        # Step 1: 去物体前缀
        stripped = self.strip_object_prefix(raw_name, object_category)

        # Step 2: 同义词归一化（简化版无方向前缀步骤）
        normalized = self.normalize_name(stripped)

        # Step 3: 归一化名 → slot
        if normalized is not None:
            slot = self.normalized_to_slot.get(normalized, None)
            if slot is not None:
                return slot

        # Step 4 (fallback): 按 _ 拆分，尝试后缀子串
        parts = stripped.split("_")
        if len(parts) > 1:
            # 从短后缀到长后缀依次尝试
            for i in range(len(parts) - 1, 0, -1):
                suffix = "_".join(parts[i:])
                norm = self.normalize_name(suffix)
                if norm is not None:
                    slot = self.normalized_to_slot.get(norm, None)
                    if slot is not None:
                        return slot

        return None

    def raw_name_to_slot_id(self, raw_name: str, object_category: str) -> Optional[int]:
        """raw_name → slot_id (整数)"""
        slot = self.raw_name_to_slot(raw_name, object_category)
        if slot is None:
            return None
        return self.slot_to_id.get(slot, None)

    # ==========================================================================
    # 加载物体 part 文件
    # ==========================================================================

    def load_object_parts(
        self, part_dir: str, object_category: str
    ) -> Dict[str, dict]:
        """
        加载一个物体实例的所有 part json，返回每个 part 的信息

        Args:
            part_dir: 包含 part_01.json, part_02.json, ... 的目录
            object_category: 物体类别名

        Returns:
            {
                "part_01": {
                    "raw_name": "scissor_handle",
                    "attr": ["held_by_hand"],
                    "slot": "GRASP_SURFACE",
                    "slot_id": 0,
                    "ply_path": "/path/to/part_01.ply"
                },
                ...
            }
        """
        part_dir = Path(part_dir)
        parts = {}

        # 找到所有 part json 文件
        json_files = sorted(part_dir.glob("part_*.json"))

        for json_path in json_files:
            part_key = json_path.stem  # e.g. "part_01"

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            raw_name = data.get("name", "")
            attr = data.get("attr", [])
            slot = self.raw_name_to_slot(raw_name, object_category)
            slot_id = self.slot_to_id.get(slot) if slot else None

            # 对应的 ply 文件
            ply_path = json_path.with_suffix(".ply")

            parts[part_key] = {
                "raw_name": raw_name,
                "attr": attr,
                "slot": slot,
                "slot_id": slot_id,
                "ply_path": str(ply_path) if ply_path.exists() else None,
            }

            if slot is None:
                print(
                    f"[WARNING] 未能映射 part: {raw_name} "
                    f"(object={object_category}, file={json_path})"
                )

        return parts

    def get_slot_to_vertices(
        self, parts: Dict[str, dict]
    ) -> Dict[int, List[str]]:
        """
        将同一 slot 的 part 合并（处理一个物体有多个 part 映射到同一 slot 的情况）

        Args:
            parts: load_object_parts 的返回值

        Returns:
            {slot_id: [ply_path_1, ply_path_2, ...]}
        """
        slot_to_plys: Dict[int, List[str]] = {}

        for part_key, info in parts.items():
            sid = info["slot_id"]
            ply = info["ply_path"]
            if sid is not None and ply is not None:
                if sid not in slot_to_plys:
                    slot_to_plys[sid] = []
                slot_to_plys[sid].append(ply)

        return slot_to_plys

    # ==========================================================================
    # Token 编解码（简化版：移除 contact_type）
    # ==========================================================================

    def encode_token(
        self,
        hand_part: Union[str, int],
        slot: Union[str, int],
    ) -> int:
        """
        (hand_part, slot) → token_id

        Args:
            hand_part: 名称 (如 "INDEX") 或 id (如 1)
            slot: 名称 (如 "GRASP_SURFACE") 或 id (如 0)

        Returns:
            token_id (int)
        """
        h = hand_part if isinstance(hand_part, int) else self.hand_part_to_id[hand_part]
        s = slot if isinstance(slot, int) else self.slot_to_id[slot]

        token_id = h * self.hand_part_stride + s
        return token_id

    def decode_token(self, token_id: int) -> Tuple[str, str]:
        """
        token_id → (hand_part, slot) 名称二元组

        Args:
            token_id: 语义 token id (0 ~ num_semantic_tokens-1)

        Returns:
            (hand_part_name, slot_name)

        Raises:
            ValueError: token_id 超出语义 token 范围
        """
        if token_id < 0 or token_id >= self.num_semantic_tokens:
            # 检查是否为特殊 token
            for name, tid in self.special_tokens.items():
                if token_id == tid:
                    return (name, name)
            raise ValueError(
                f"token_id {token_id} 超出范围 [0, {self.num_semantic_tokens})"
            )

        h_id = token_id // self.hand_part_stride
        s_id = token_id % self.hand_part_stride

        return (
            self.hand_parts[h_id],
            self.canonical_slots[s_id],
        )

    def decode_token_ids(self, token_id: int) -> Tuple[int, int]:
        """
        token_id → (hand_part_id, slot_id) 整数二元组
        """
        if token_id < 0 or token_id >= self.num_semantic_tokens:
            return (-1, -1)

        h_id = token_id // self.hand_part_stride
        s_id = token_id % self.hand_part_stride
        return (h_id, s_id)

    def is_semantic_token(self, token_id: int) -> bool:
        """判断是否为语义 token（非特殊 token）"""
        return 0 <= token_id < self.num_semantic_tokens

    def is_special_token(self, token_id: int) -> bool:
        """判断是否为特殊 token (PAD/MASK)"""
        return token_id in self.special_tokens.values()

    # ==========================================================================
    # 文本约束相关
    # ==========================================================================

    def get_tokens_by_hand_part(self, hand_part: str) -> List[int]:
        """获取指定 hand_part 的所有 token_id"""
        h_id = self.hand_part_to_id[hand_part]
        start = h_id * self.hand_part_stride
        return list(range(start, start + self.hand_part_stride))

    def get_tokens_by_slot(self, slot: str) -> List[int]:
        """获取指定 slot 的所有 token_id"""
        s_id = self.slot_to_id[slot]
        tokens = []
        for h_id in range(len(self.hand_parts)):
            token_id = h_id * self.hand_part_stride + s_id
            tokens.append(token_id)
        return tokens

    def get_tokens_by_constraint(
        self,
        hand_parts: Optional[List[str]] = None,
        slots: Optional[List[str]] = None,
    ) -> List[int]:
        """
        根据约束条件筛选 token_id（用于文本引导）

        Args:
            hand_parts: 限定的手部部位列表，None 表示不限
            slots: 限定的物体 slot 列表，None 表示不限

        Returns:
            满足所有约束的 token_id 列表
        """
        h_ids = (
            [self.hand_part_to_id[h] for h in hand_parts]
            if hand_parts
            else list(range(len(self.hand_parts)))
        )
        s_ids = (
            [self.slot_to_id[s] for s in slots]
            if slots
            else list(range(len(self.canonical_slots)))
        )

        tokens = []
        for h in h_ids:
            for s in s_ids:
                tokens.append(h * self.hand_part_stride + s)
        return tokens

    # ==========================================================================
    # 工具方法
    # ==========================================================================

    def print_token_table(self, token_ids: Optional[List[int]] = None):
        """打印 token 语义表（调试用）"""
        if token_ids is None:
            token_ids = list(range(self.num_semantic_tokens))

        print(f"{'token_id':>8}  {'hand_part':<10}  {'slot':<18}")
        print("-" * 40)
        for tid in token_ids:
            if self.is_semantic_token(tid):
                h, s = self.decode_token(tid)
                print(f"{tid:>8}  {h:<10}  {s:<18}")
            else:
                for name, val in self.special_tokens.items():
                    if tid == val:
                        print(f"{tid:>8}  [{name}]")

    def summary(self):
        """打印配置摘要"""
        print("=" * 50)
        print("Token Space Summary (Simplified)")
        print("=" * 50)
        print(f"  Hand parts:      {len(self.hand_parts):>3}  {self.hand_parts}")
        print(f"  Canonical slots: {len(self.canonical_slots):>3}  {self.canonical_slots}")
        print(f"  Semantic tokens: {self.num_semantic_tokens:>3}")
        print(f"  Special tokens:  {self.special_tokens}")
        print(f"  Vocab size:      {self.vocab_size:>3}")
        print(f"  Max seq length:  {self.max_token_length:>3}")
        print("=" * 50)


# ==============================================================================
# 快捷函数
# ==============================================================================

def load_mapper(config_path: str = "configs/token_config.yaml") -> SlotMapper:
    """加载 SlotMapper 的快捷方式"""
    return SlotMapper(config_path)


# ==============================================================================
# 测试 / 示例
# ==============================================================================

if __name__ == "__main__":
    import sys

    config = sys.argv[1] if len(sys.argv) > 1 else "configs/token_config.yaml"
    mapper = SlotMapper(config)
    mapper.summary()

    print("\n--- 名称映射测试 ---")
    test_cases = [
        ("scissor_handle", "scissor"),
        ("scissor_blades", "scissor"),
        ("bottle_body", "cylinder_bottle"),
        ("drill_trigger", "power_drill"),
        ("wineglass_stem", "wineglass"),
        ("camera_grip_and_control_pannel", "camera"),
        ("pump_head", "lotion_bottle"),
        ("fryingpan_handle", "frying_pan"),
        ("hammer_head", "hammer"),
        ("mug_handle", "mug"),
    ]

    for raw_name, category in test_cases:
        slot = mapper.raw_name_to_slot(raw_name, category)
        print(f"  {raw_name:<40} ({category:<20}) → {slot}")

    print("\n--- Token 编解码测试 ---")
    examples = [
        ("THUMB", "GRASP_SURFACE"),
        ("INDEX", "GRASP_SURFACE"),
        ("PALM", "FUNCTIONAL_END"),
        ("INDEX", "CONTROL"),
        ("MIDDLE", "CLOSURE"),
    ]

    for h, s in examples:
        tid = mapper.encode_token(h, s)
        h2, s2 = mapper.decode_token(tid)
        assert (h, s) == (h2, s2), "编解码不一致!"
        print(f"  ({h}, {s}) → token_id={tid} → ({h2}, {s2}) ✓")

    print("\n--- 文本约束筛选测试 ---")
    # "用拇指和食指捏住 grasp_surface"
    constrained = mapper.get_tokens_by_constraint(
        hand_parts=["THUMB", "INDEX"],
        slots=["GRASP_SURFACE"],
    )
    print(f"  约束: hand=[THUMB,INDEX], slot=[GRASP_SURFACE]")
    print(f"  匹配 token 数: {len(constrained)}")
    mapper.print_token_table(constrained)
