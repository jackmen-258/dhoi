"""
text_generator.py (v7 - deterministic natural prompts)
======================================================
生成稳定、自然、可用于 CLIP 编码的单句文本描述。

设计原则：
  - 完全确定性：相同输入始终生成相同文本
  - 更自然：尽量贴近图文预训练模型常见的 caption 风格
  - 轻量语义：编码 object + intent + hand-slot contact
  - schema-aware：slot 仍是 canonical slot，但会映射到更自然的物体部件词汇

生成规则：
  - 格式: "A hand {intent_phrase} the {obj}, with {fingers} on the {part_phrase}."
  - 无接触时: "A hand {intent_phrase} the {obj}, without visible contact."
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

# ---- 默认 slot → 自然语言 ----
DEFAULT_SLOT_NAME = {
    "GRASP_SURFACE": "body or handle",
    "FUNCTIONAL_END": "functional end",
    "CONTROL": "control area",
    "CLOSURE": "cap or lid",
}

# ---- OakInk cate_id → 自然语言物体名 ----
CATE_TO_NAME = {
    "bottle":           "bottle",
    "cylinder_bottle":  "bottle",
    "lotion_bottle":    "bottle",
    "lotion_pump":      "lotion pump bottle",
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

# ---- 动作意图 → 句式短语 ----
ACTION_TO_PHRASE = {
    "0001": "using",
    "0002": "holding",
    "0003": "lifting",
    "0004": "handing over",
}

# ---- 类别特定 slot 用词 ----
SLOT_NAME_BY_CATEGORY = {
    "bottle": {
        "GRASP_SURFACE": "bottle body",
        "FUNCTIONAL_END": "opening",
        "CLOSURE": "cap",
    },
    "cylinder_bottle": {
        "GRASP_SURFACE": "bottle body",
        "FUNCTIONAL_END": "opening",
        "CLOSURE": "cap",
    },
    "lotion_pump": {
        "GRASP_SURFACE": "bottle body",
        "FUNCTIONAL_END": "pump head",
        "CONTROL": "pump top",
    },
    "mug": {
        "GRASP_SURFACE": "mug body or handle",
    },
    "cup": {
        "GRASP_SURFACE": "cup body",
    },
    "bowl": {
        "GRASP_SURFACE": "rim or outer surface",
    },
    "wineglass": {
        "GRASP_SURFACE": "stem or bowl",
        "FUNCTIONAL_END": "rim",
    },
    "knife": {
        "GRASP_SURFACE": "handle",
        "FUNCTIONAL_END": "blade",
    },
    "hammer": {
        "GRASP_SURFACE": "handle",
        "FUNCTIONAL_END": "hammer head",
    },
    "pen": {
        "GRASP_SURFACE": "barrel",
        "FUNCTIONAL_END": "tip",
        "CLOSURE": "cap",
    },
    "flashlight": {
        "GRASP_SURFACE": "flashlight body",
        "FUNCTIONAL_END": "light end",
        "CONTROL": "button",
    },
    "cameras": {
        "GRASP_SURFACE": "camera body or grip",
        "FUNCTIONAL_END": "lens",
        "CONTROL": "button or control panel",
    },
    "eyeglasses": {
        "GRASP_SURFACE": "frame or temple",
        "FUNCTIONAL_END": "lens area",
    },
    "headphones": {
        "GRASP_SURFACE": "headband or ear cup",
    },
    "power_drill": {
        "GRASP_SURFACE": "handle",
        "FUNCTIONAL_END": "drill bit or chuck",
        "CONTROL": "trigger",
    },
    "trigger_sprayer": {
        "GRASP_SURFACE": "bottle body or handle",
        "FUNCTIONAL_END": "nozzle",
        "CONTROL": "trigger",
    },
    "teapot": {
        "GRASP_SURFACE": "handle or body",
        "FUNCTIONAL_END": "spout",
        "CLOSURE": "lid",
    },
    "toothbrush": {
        "GRASP_SURFACE": "handle",
        "FUNCTIONAL_END": "brush head",
    },
    "stapler": {
        "GRASP_SURFACE": "top surface",
        "FUNCTIONAL_END": "front end",
    },
    "mouse": {
        "GRASP_SURFACE": "mouse body",
        "CONTROL": "button area",
    },
    "binoculars": {
        "GRASP_SURFACE": "barrel",
        "FUNCTIONAL_END": "lens end",
    },
}

# 手指生理顺序
HAND_ORDER = {h: i for i, h in enumerate(
    ["THUMB", "INDEX", "MIDDLE", "RING", "LITTLE", "PALM"])}


class ContactTextGenerator:
    """
    接触矩阵 → 自然语言描述（完全确定性，无随机）

    文本编码：
      - object category
      - action intent
      - 哪些手部部位接触哪些 canonical slot
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

    @staticmethod
    def _intent_phrase(action_id: str) -> str:
        return ACTION_TO_PHRASE.get(str(action_id), "grasping")

    def _slot_phrase(self, cate_id: str, slot_name: str) -> str:
        cate_key = cate_id.lower().strip()
        cate_map = SLOT_NAME_BY_CATEGORY.get(cate_key, {})
        return cate_map.get(slot_name, DEFAULT_SLOT_NAME.get(slot_name, slot_name.lower()))

    def _format_hand_parts(self, hps: List[str]) -> str:
        """将 hand parts 转成较自然的英文短语"""
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
                    others_str = self._join_list(others)
                    fingers_str = f"the thumb, {others_str} fingers"
            else:
                names = [HAND_NAME[f] for f in fingers]
                if len(names) == 2:
                    fingers_str = f"the {names[0]} and {names[1]} fingers"
                else:
                    fingers_str = "the " + self._join_list(names) + " fingers"

        if has_palm:
            if not fingers_str:
                return "the palm"
            return f"{fingers_str} and the palm"

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
        slot_order = {s: i for i, s in enumerate(self.mapper.canonical_slots)}
        contacts.sort(key=lambda x: (slot_order.get(x[1], 99), HAND_ORDER.get(x[0], 99)))
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

    def _build_sentence(self, contacts, cate_id: str, obj_name: str, action_id: str) -> str:
        """
        生成自然、确定性的 caption 风格句子
        """
        intent_phrase = self._intent_phrase(action_id)
        slot_groups = self._group_by_slot(contacts)
        if not slot_groups:
            return f"A hand {intent_phrase} the {obj_name}, without visible contact."

        clauses = []
        for sl in self.mapper.canonical_slots:
            hands = slot_groups.get(sl)
            if not hands:
                continue
            hand_str = self._format_hand_parts(hands)
            slot_str = self._slot_phrase(cate_id, sl)
            clauses.append(f"{hand_str} on the {slot_str}")

        if len(clauses) == 1:
            body = clauses[0]
        elif len(clauses) == 2:
            body = f"{clauses[0]} and {clauses[1]}"
        else:
            body = self._join_list(clauses)

        return f"A hand {intent_phrase} the {obj_name}, with {body}."

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
            action_id: OakInk action ID

        Returns:
            自然语言描述字符串
        """
        contacts = self._parse(contact_matrix)
        obj_name = self._obj_name(cate_id)
        return self._build_sentence(contacts, cate_id, obj_name, action_id)

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
