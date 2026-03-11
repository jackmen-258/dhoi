"""
text_generator.py (v6 – 极简确定性映射)
========================================
生成稳定的自然语言描述（单个句子），用于 CLIP 编码。

设计原则：
  - 完全确定性：相同接触模式 → 相同文本 → 相同 CLIP embedding
  - 极简：仅编码 object + 哪些手指接触哪些部件
  - contact_type 不在文本中体现 (已由 token 序列携带)
  - intent 统一为 grasp (具体意图由 token 序列携带)

生成规则：
  - 格式: "To grasp the {obj}, {fingers} on the {slot1} and {fingers} on the {slot2}."
  - 无接触时: "No contact with the {obj}."
"""

import os
import sys
import numpy as np
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.slot_mapping import SlotMapper


# ---- 手部部位 → 自然语言 ----
HAND_NAME = {
    "THUMB":  "thumb",
    "INDEX":  "index",
    "MIDDLE": "middle",
    "RING":   "ring",
    "LITTLE": "little",
    "PALM":   "palm",
}

# ---- 物体部件 → 自然语言 ----
SLOT_NAME = {
    "GRASP_SURFACE":   "grasp surface",
    "FUNCTIONAL_END":  "functional end",
    "CONTROL":         "control",
    "CLOSURE":         "closure",
}

# ---- OakInk cate_id → 自然语言物体名 ----
CATE_TO_NAME = {
    "bottle":           "bottle",
    "cylinder_bottle":  "bottle",
    "lotion_bottle":    "bottle",
    "bowl":             "bowl",
    "camera":           "camera",
    "cameras":          "camera",
    "can":              "can",
    "cup":              "cup",
    "mug":              "mug",
    "eyeglasses":       "glasses",
    "faucet":           "faucet",
    "flashlight":       "flashlight",
    "flyswatter":       "fly swatter",
    "gamecontroller":   "game controller",
    "hammer":           "hammer",
    "headphones":       "headphones",
    "knife":            "knife",
    "light_bulb":       "light bulb",
    "lightbulb":        "light bulb",
    "mouse":            "mouse",
    "pen":              "pen",
    "phone":            "phone",
    "pincer":           "pliers",
    "power_drill":      "drill",
    "remote":           "remote",
    "scissor":          "scissors",
    "screwdriver":      "screwdriver",
    "spraybottle":      "spray bottle",
    "trigger_sprayer":  "spray bottle",
    "stapler":          "stapler",
    "teapot":           "teapot",
    "toothbrush":       "toothbrush",
    "trigger_spray":    "spray bottle",
    "wineglass":        "wine glass",
    "binoculars":       "binoculars",
}

# 手指生理顺序
HAND_ORDER = {h: i for i, h in enumerate(
    ["THUMB", "INDEX", "MIDDLE", "RING", "LITTLE", "PALM"])}


class ContactTextGenerator:
    """
    接触矩阵 → 自然语言描述 (完全确定性, 无随机)

    文本仅编码: object + 哪些手指接触哪些部件
    contact_type / intent 不在文本中体现 (已由 token 序列携带)
    """

    def __init__(self, mapper: SlotMapper, seed: Optional[int] = None):
        self.mapper = mapper

    @staticmethod
    def _obj_name(cate_id: str) -> str:
        """cate_id → 自然语言物体名"""
        key = cate_id.lower().strip()
        return CATE_TO_NAME.get(key, key.replace("_", " "))

    @staticmethod
    def _join_list(items: List[str]) -> str:
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"

    def _format_hand_parts(self, hps: List[str]) -> str:
        """智能化简手部组合"""
        has_palm = "PALM" in hps
        finger_keys = ["THUMB", "INDEX", "MIDDLE", "RING", "LITTLE"]
        fingers = [h for h in finger_keys if h in hps]
        non_thumb = [f for f in fingers if f != "THUMB"]
        has_thumb = "THUMB" in fingers

        # 全手
        if len(fingers) == 5 and has_palm:
            return "the whole hand"
        if len(fingers) == 5:
            return "all fingers"

        # 4 根非拇指手指
        if len(non_thumb) == 4 and not has_thumb:
            fingers_str = "four fingers"
        elif len(fingers) == 0:
            fingers_str = ""
        elif len(fingers) == 1:
            if fingers[0] == "THUMB":
                fingers_str = "the thumb"
            else:
                fingers_str = f"the {HAND_NAME[fingers[0]]} finger"
        else:
            if has_thumb:
                if len(non_thumb) == 1:
                    fingers_str = f"the thumb and {HAND_NAME[non_thumb[0]]} finger"
                else:
                    others = [HAND_NAME[f] for f in non_thumb]
                    others_str = ", ".join(others[:-1]) + f", and {others[-1]}" if len(others) > 2 else f"{others[0]} and {others[1]}"
                    fingers_str = f"the thumb, {others_str} fingers"
            else:
                names = [HAND_NAME[f] for f in fingers]
                if len(names) == 2:
                    fingers_str = f"the {names[0]} and {names[1]} fingers"
                else:
                    fingers_str = "the " + ", ".join(names[:-1]) + f", and {names[-1]} fingers"

        if has_palm:
            if not fingers_str:
                return "the palm"
            return f"{fingers_str} and palm"

        if not fingers_str:
            return "the hand"

        return fingers_str

    def _parse(self, mat: np.ndarray):
        """
        接触矩阵 → (hand, slot) 二元组列表
        按 (slot, hand) 排序保证确定性
        
        Args:
            mat: (H, S) 接触矩阵，-1=无接触, 0=接触
        """
        contacts = []
        H, S = mat.shape
        for h in range(H):
            for s in range(S):
                v = int(mat[h, s])
                if v >= 0:  # 接触
                    contacts.append((
                        self.mapper.hand_parts[h],
                        self.mapper.canonical_slots[s],
                    ))
        contacts.sort(key=lambda x: (x[1], HAND_ORDER.get(x[0], 99)))
        return contacts

    def _group_by_slot(self, contacts):
        """
        按 slot 分组 → {slot: [hand_parts]}
        只关心哪些手指接触了哪个部件
        """
        slot_data = {}
        for hp, sl in contacts:
            if sl not in slot_data:
                slot_data[sl] = []
            if hp not in slot_data[sl]:
                slot_data[sl].append(hp)
        return slot_data

    def _build_sentence(self, contacts, obj_name: str) -> str:
        """
        生成极简句子
        格式: "To grasp the {obj}, {fingers} on the {slot1} and {fingers} on the {slot2}."
        """
        slot_groups = self._group_by_slot(contacts)
        if not slot_groups:
            return f"No contact with the {obj_name}."

        clauses = []
        for sl, hands in slot_groups.items():
            hand_str = self._format_hand_parts(hands)
            slot_str = SLOT_NAME.get(sl, sl.lower())
            clauses.append(f"{hand_str} on the {slot_str}")

        if len(clauses) == 1:
            body = clauses[0]
        elif len(clauses) == 2:
            body = f"{clauses[0]} and {clauses[1]}"
        else:
            body = ", ".join(clauses[:-1]) + f", and {clauses[-1]}"

        return f"To grasp the {obj_name}, {body}."

    # ==================================================================
    # 公共接口
    # ==================================================================

    def generate_single(
        self,
        contact_matrix: np.ndarray,
        cate_id: str,
        action_id: str = "",
    ) -> str:
        """
        生成单个确定性描述

        Args:
            contact_matrix: (6, 4) 接触矩阵, -1=无接触, 0=接触
            cate_id: 物体类别 ID
            action_id: OakInk action ID (已忽略)

        Returns:
            自然语言描述字符串
        """
        contacts = self._parse(contact_matrix)
        obj_name = self._obj_name(cate_id)
        return self._build_sentence(contacts, obj_name)

    def generate(
        self,
        contact_matrix: np.ndarray,
        cate_id: str,
        action_id: str = "",
        num_texts: int = 0,
    ) -> List[str]:
        """返回包含单个句子的列表（保持接口兼容）"""
        return [self.generate_single(contact_matrix, cate_id, action_id)]

    def generate_for_dataset(
        self,
        matrices: List[np.ndarray],
        cate_ids: List[str],
        action_ids: Optional[List[str]] = None,
        texts_per_sample: int = 1,
    ) -> List[List[str]]:
        """批量生成"""
        if action_ids is None:
            action_ids = [""] * len(matrices)
        return [self.generate(m, c, a)
                for m, c, a in zip(matrices, cate_ids, action_ids)]