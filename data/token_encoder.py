"""
token_encoder.py
================
接触矩阵 → 离散 token 序列的编码/解码模块（简化版）

功能：
  1. contact_matrix (6×4) → token 序列 [t1, t2, ..., PAD, PAD]
  2. token 序列 → contact_matrix（反向解码）
  3. 提供 OIShape 数据集集成接口
  4. token 频次统计（用于 loss 加权）

简化版变更：
  - 移除 contact_type，contact_matrix 为二元矩阵（0=接触, -1=无接触）
  - Token 空间 = 6 hand_parts × 4 slots = 24
  - 矩阵形状从 (6, 6) 变为 (6, 4)

数据流：
  OIShape.__getitem__
    → contact_builder.build_contact_for_grasp / load_cached_contact
    → token_encoder.encode(contact_matrix)
    → token_seq: (max_token_length,) int64
    → 送入离散扩散模型

依赖：
  - data.slot_mapping.SlotMapper
"""

import os
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from collections import Counter

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.slot_mapping import SlotMapper


class TokenEncoder:
    """
    接触矩阵 ↔ 离散 token 序列 编解码器（简化版）

    用法：
        encoder = TokenEncoder("configs/token_config.yaml")

        # 编码
        token_seq = encoder.encode(contact_matrix)
        # → np.array([4, 21, 91, 24, 24, ...])  (max_token_length,)

        # 解码
        matrix = encoder.decode(token_seq)
        # → (6, 4) int, 与原始 contact_matrix 一致
    """

    def __init__(self, config_path: str):
        self.mapper = SlotMapper(config_path)

        self.num_hand_parts = len(self.mapper.hand_parts)
        self.num_slots = len(self.mapper.canonical_slots)

        self.PAD = self.mapper.special_tokens["PAD"]
        self.MASK = self.mapper.special_tokens["MASK"]
        self.vocab_size = self.mapper.vocab_size
        self.max_token_length = self.mapper.max_token_length

    # ==========================================================================
    # 编码：contact_matrix → token 序列
    # ==========================================================================

    def encode(
        self,
        contact_matrix: np.ndarray,
        add_bos_eos: bool = False,
        sort_tokens: bool = True,
    ) -> np.ndarray:
        """
        将 (H, S) contact_matrix 编码为定长 token 序列

        规则：
          - contact_matrix[h, s] >= 0 的位置生成一个 token（接触）
          - contact_matrix[h, s] == -1 的位置跳过（无接触）
          - token_id = h * hand_part_stride + s
          - 序列补 PAD 到 max_token_length
          - 超出 max_token_length 的 token 按 token_id 排序后截断

        Args:
            contact_matrix: (num_hand_parts, num_slots) int
                            -1=无接触, >=0=接触（简化版不区分 contact_type）
            add_bos_eos: 是否在序列首尾添加 BOS/EOS（简化版通常不需要）
            sort_tokens: 是否对 token 排序（保证确定性）

        Returns:
            token_seq: (max_token_length,) int64, 填充 PAD
        """
        assert contact_matrix.shape == (self.num_hand_parts, self.num_slots), (
            f"期望 ({self.num_hand_parts}, {self.num_slots})，"
            f"得到 {contact_matrix.shape}"
        )

        tokens = []
        for h in range(self.num_hand_parts):
            for s in range(self.num_slots):
                val = int(contact_matrix[h, s])
                if val < 0:
                    continue  # 无接触，跳过
                token_id = self.mapper.encode_token(h, s)
                tokens.append(token_id)

        # 排序保证确定性（按 token_id 升序）
        if sort_tokens:
            tokens.sort()

        # 构建定长序列
        max_len = self.max_token_length
        if add_bos_eos:
            max_len -= 2  # 留位置给 BOS/EOS

        # 截断（超长时只保留前 max_len 个）
        if len(tokens) > max_len:
            tokens = tokens[:max_len]

        # 添加 BOS/EOS（如果启用）
        if add_bos_eos:
            # 简化版配置通常没有 BOS/EOS，但保留兼容性
            bos = getattr(self.mapper.special_tokens, "BOS", self.PAD)
            eos = getattr(self.mapper.special_tokens, "EOS", self.PAD)
            tokens = [bos] + tokens + [eos]

        # PAD 填充
        num_pad = self.max_token_length - len(tokens)
        token_seq = tokens + [self.PAD] * num_pad

        return np.array(token_seq, dtype=np.int64)

    # ==========================================================================
    # 解码：token 序列 → contact_matrix
    # ==========================================================================

    def decode(self, token_seq: np.ndarray) -> np.ndarray:
        """
        将 token 序列解码回 (H, S) contact_matrix

        Args:
            token_seq: (L,) int64, 可含 PAD/MASK

        Returns:
            contact_matrix: (num_hand_parts, num_slots) int
                            -1=无接触, 0=接触（简化版用 0 表示接触）
        """
        contact_matrix = np.full(
            (self.num_hand_parts, self.num_slots), -1, dtype=np.int32
        )

        for token_id in token_seq:
            token_id = int(token_id)

            # 跳过特殊 token
            if not self.mapper.is_semantic_token(token_id):
                continue

            h_id, s_id = self.mapper.decode_token_ids(token_id)
            if h_id < 0:
                continue

            # 如果同一 (h, s) 位置出现多次（理论上不应该），取后者
            contact_matrix[h_id, s_id] = 0  # 简化版用 0 表示接触

        return contact_matrix

    # ==========================================================================
    # Batch 编解码（方便训练使用）
    # ==========================================================================

    def encode_batch(
        self, matrices: List[np.ndarray], **kwargs
    ) -> np.ndarray:
        """
        批量编码

        Args:
            matrices: list of (H, S) int arrays

        Returns:
            (B, max_token_length) int64
        """
        seqs = [self.encode(m, **kwargs) for m in matrices]
        return np.stack(seqs, axis=0)

    def decode_batch(self, token_seqs: np.ndarray) -> List[np.ndarray]:
        """
        批量解码

        Args:
            token_seqs: (B, L) int64

        Returns:
            list of (H, S) int arrays
        """
        if isinstance(token_seqs, torch.Tensor):
            token_seqs = token_seqs.detach().cpu().numpy()
        return [self.decode(seq) for seq in token_seqs]

    # ==========================================================================
    # Token 序列属性计算
    # ==========================================================================

    def get_seq_length(self, token_seq: np.ndarray) -> int:
        """获取有效 token 数量（不含 PAD/MASK）"""
        count = 0
        for t in token_seq:
            t = int(t)
            if self.mapper.is_semantic_token(t):
                count += 1
        return count

    def get_active_hand_parts(self, token_seq: np.ndarray) -> List[str]:
        """获取序列中涉及的 hand_part 名称"""
        parts = set()
        for t in token_seq:
            t = int(t)
            if self.mapper.is_semantic_token(t):
                h, s = self.mapper.decode_token(t)
                parts.add(h)
        return sorted(parts)

    def get_active_slots(self, token_seq: np.ndarray) -> List[str]:
        """获取序列中涉及的 slot 名称"""
        slots = set()
        for t in token_seq:
            t = int(t)
            if self.mapper.is_semantic_token(t):
                h, s = self.mapper.decode_token(t)
                slots.add(s)
        return sorted(slots)

    # ==========================================================================
    # Token 频次统计（用于 loss 加权和类别均衡）
    # ==========================================================================

    def compute_token_frequencies(
        self, all_token_seqs: np.ndarray
    ) -> np.ndarray:
        """
        统计每个 token 在数据集中的出现频次

        Args:
            all_token_seqs: (N, max_token_length) int64

        Returns:
            freqs: (vocab_size,) float32, 每个 token 的出现频率
        """
        counts = np.zeros(self.vocab_size, dtype=np.float64)

        for seq in all_token_seqs:
            for t in seq:
                t = int(t)
                if self.mapper.is_semantic_token(t):
                    counts[t] += 1

        total = counts.sum()
        if total > 0:
            freqs = counts / total
        else:
            freqs = np.ones(self.vocab_size, dtype=np.float64) / self.vocab_size

        return freqs.astype(np.float32)

    def compute_loss_weights(
        self,
        all_token_seqs: np.ndarray,
        method: str = "inverse_freq",
        smoothing: float = 0.1,
    ) -> np.ndarray:
        """
        计算每个 token 的 loss 权重（用于类别均衡）

        Args:
            all_token_seqs: (N, max_token_length) int64
            method: "inverse_freq" 或 "sqrt_inverse_freq"
            smoothing: 频率平滑因子，防止除零

        Returns:
            weights: (vocab_size,) float32
        """
        freqs = self.compute_token_frequencies(all_token_seqs)
        freqs = freqs + smoothing  # 平滑

        if method == "inverse_freq":
            weights = 1.0 / freqs
        elif method == "sqrt_inverse_freq":
            weights = 1.0 / np.sqrt(freqs)
        else:
            raise ValueError(f"未知 method: {method}")

        # 归一化到均值 1
        # 只对语义 token 做归一化
        semantic_mask = np.zeros(self.vocab_size, dtype=bool)
        semantic_mask[:self.mapper.num_semantic_tokens] = True

        mean_weight = weights[semantic_mask].mean()
        if mean_weight > 0:
            weights = weights / mean_weight

        # 特殊 token 权重设为 0（不参与 loss）
        weights[~semantic_mask] = 0.0

        return weights.astype(np.float32)

    # ==========================================================================
    # 人类可读打印
    # ==========================================================================

    def describe_token_seq(self, token_seq: np.ndarray) -> str:
        """
        将 token 序列转为人类可读的描述

        Returns:
            如 "INDEX+GRASP_SURFACE, THUMB+GRASP_SURFACE, PALM+FUNCTIONAL_END"
        """
        parts = []
        for t in token_seq:
            t = int(t)
            if self.mapper.is_semantic_token(t):
                h, s = self.mapper.decode_token(t)
                parts.append(f"{h}+{s}")
        return ", ".join(parts) if parts else "(empty)"

    def describe_contact_matrix(self, contact_matrix: np.ndarray) -> str:
        """
        将 contact_matrix 转为人类可读的描述
        """
        type_names = {-1: "·", 0: "✓"}  # 简化版：✓=接触, ·=无接触
        lines = []

        # 表头
        header = f"{'':>8}" + "".join(
            f"{s[:6]:>8}" for s in self.mapper.canonical_slots
        )
        lines.append(header)

        # 行
        for h in range(self.num_hand_parts):
            row = f"{self.mapper.hand_parts[h]:>8}"
            for s in range(self.num_slots):
                val = int(contact_matrix[h, s])
                row += f"{type_names.get(val, '?'):>8}"
            lines.append(row)

        return "\n".join(lines)

    # ==========================================================================
    # 与 OIShape 集成的便捷方法
    # ==========================================================================

    def encode_from_cache(self, cache_path: str, **kwargs) -> Optional[np.ndarray]:
        """
        从缓存的 npz 文件直接编码 token 序列

        Args:
            cache_path: contact_builder 保存的 npz 路径

        Returns:
            token_seq: (max_token_length,) int64
        """
        if not os.path.exists(cache_path):
            return None

        data = np.load(cache_path, allow_pickle=True)
        # 简化版：使用 contact_matrix 而不是 contact_type_matrix
        contact_matrix = data["contact_matrix"]
        return self.encode(contact_matrix, **kwargs)

    def create_training_sample(
        self,
        contact_matrix: np.ndarray,
        hand_pose: np.ndarray,
        hand_shape: np.ndarray,
        hand_tsl: np.ndarray,
        cate_id: str,
        intent_id: int,
    ) -> Dict:
        """
        创建一个完整的训练样本（给 dataset.py 使用）

        Args:
            contact_matrix: (H, S) int, -1=无接触, 0=接触
            hand_pose: (48,) MANO pose
            hand_shape: (10,) MANO shape
            hand_tsl: (3,) 手部平移
            cate_id: 物体类别
            intent_id: 意图 ID

        Returns:
            {
                token_seq: (max_token_length,) int64
                token_length: int (有效 token 数)
                hand_pose: (48,) float32
                hand_shape: (10,) float32
                hand_tsl: (3,) float32
                contact_matrix: (H, S) int32
                cate_id: str
                intent_id: int
            }
        """
        token_seq = self.encode(contact_matrix)
        token_length = self.get_seq_length(token_seq)

        return {
            "token_seq": token_seq,
            "token_length": token_length,
            "hand_pose": np.array(hand_pose, dtype=np.float32),
            "hand_shape": np.array(hand_shape, dtype=np.float32),
            "hand_tsl": np.array(hand_tsl, dtype=np.float32),
            "contact_matrix": contact_matrix.astype(np.int32),
            "cate_id": cate_id,
            "intent_id": int(intent_id),
        }


# ==============================================================================
# 测试
# ==============================================================================

if __name__ == "__main__":
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "token_config.yaml"
    )
    encoder = TokenEncoder(config_path)

    print("=== TokenEncoder 测试（简化版）===\n")
    print(f"Vocab size: {encoder.vocab_size}")
    print(f"Max token length: {encoder.max_token_length}")
    print(f"PAD={encoder.PAD}, MASK={encoder.MASK}")
    print(f"Matrix shape: ({encoder.num_hand_parts}, {encoder.num_slots})")

    # 构造一个模拟 contact_matrix
    # 场景：用拇指和食指捏住 GRASP_SURFACE，掌心面接触 FUNCTIONAL_END
    contact_matrix = np.full((6, 4), -1, dtype=np.int32)
    contact_matrix[0, 0] = 0   # THUMB + GRASP_SURFACE = 接触
    contact_matrix[1, 0] = 0   # INDEX + GRASP_SURFACE = 接触
    contact_matrix[2, 0] = 0   # MIDDLE + GRASP_SURFACE = 接触
    contact_matrix[5, 1] = 0   # PALM + FUNCTIONAL_END = 接触

    print("\n--- 输入 contact_matrix ---")
    print(encoder.describe_contact_matrix(contact_matrix))

    # 编码
    token_seq = encoder.encode(contact_matrix)
    print(f"\n--- 编码结果 ---")
    print(f"  token_seq = {token_seq}")
    print(f"  有效 token 数 = {encoder.get_seq_length(token_seq)}")
    print(f"  语义描述: {encoder.describe_token_seq(token_seq)}")
    print(f"  涉及手指: {encoder.get_active_hand_parts(token_seq)}")
    print(f"  涉及部件: {encoder.get_active_slots(token_seq)}")

    # 解码
    decoded_matrix = encoder.decode(token_seq)
    print(f"\n--- 解码结果 ---")
    print(encoder.describe_contact_matrix(decoded_matrix))

    # 验证 round-trip
    assert np.array_equal(contact_matrix, decoded_matrix), "编解码不一致!"
    print("\n✅ Round-trip 验证通过")

    # 测试批量编码
    matrices = [contact_matrix, contact_matrix, np.full((6, 4), -1, dtype=np.int32)]
    batch = encoder.encode_batch(matrices)
    print(f"\n--- Batch 编码 ---")
    print(f"  shape: {batch.shape}")  # (3, 12)
    print(f"  空样本 token 数: {encoder.get_seq_length(batch[2])}")

    # 测试频次统计
    freqs = encoder.compute_token_frequencies(batch)
    active = np.sum(freqs > 0)
    print(f"\n--- 频次统计 ---")
    print(f"  活跃 token 数: {active}")

    # 测试 loss 权重
    weights = encoder.compute_loss_weights(batch)
    print(f"  权重范围: [{weights[weights > 0].min():.3f}, {weights[weights > 0].max():.3f}]")

    # 测试 create_training_sample
    sample = encoder.create_training_sample(
        contact_matrix=contact_matrix,
        hand_pose=np.zeros(48),
        hand_shape=np.zeros(10),
        hand_tsl=np.zeros(3),
        cate_id="scissor",
        intent_id=0,
    )
    print(f"\n--- Training sample ---")
    for k, v in sample.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {v}")

    print("\n✅ 所有测试通过")
