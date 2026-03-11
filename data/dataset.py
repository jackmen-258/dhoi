"""
dataset.py
==========
离散扩散模型训练用 PyTorch Dataset

功能：
  1. 从预处理缓存（npz）加载 token 序列和 MANO 参数
  2. 提供类别均衡采样权重
  3. 支持训练/验证/测试三种模式
  4. 提供条件信息（物体类别、意图）的编码

数据流：
  预处理阶段产出:
    cache/{split}/dataset_index.json     ← 样本索引
    cache/{split}/*.npz                  ← 每个 grasp 的数据
    cache/{split}/loss_weights.npy       ← token loss 权重
    cache/{split}/token_sequences.npz    ← 汇总 token（可选快速加载）

  训练时:
    ContactTokenDataset.__getitem__
      → 读 npz → token_seq, hand_pose, hand_shape, hand_tsl, cate_id, intent
      → 返回 dict of tensors → DataLoader → 离散扩散模型

依赖：
  - data.slot_mapping.SlotMapper
  - data.token_encoder.TokenEncoder
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Dict, List, Optional, Tuple
from collections import Counter

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.slot_mapping import SlotMapper
from data.token_encoder import TokenEncoder


class ContactTokenDataset(Dataset):
    """
    离散扩散模型训练用数据集

    每个样本返回：
        token_seq:            (max_token_length,) int64   — 离散 token 序列
        token_length:         int                         — 有效 token 数
        token_mask:           (max_token_length,) bool    — True=有效, False=PAD
        hand_pose:            (48,) float32               — MANO pose 参数
        hand_shape:           (10,) float32               — MANO shape 参数
        hand_tsl:             (3,) float32                — 手部平移
        cate_id:              int                         — 物体类别编号
        intent_id:            int                         — 意图编号
        contact_type_matrix:  (H, S) int32                — 接触类型矩阵（调试用）

    用法：
        dataset = ContactTokenDataset(
            cache_dir="cache/contact_tokens/train",
            config_path="configs/token_config.yaml",
        )
        loader = dataset.get_dataloader(batch_size=64, shuffle=True)

        for batch in loader:
            token_seq = batch["token_seq"]       # (B, max_token_length)
            token_mask = batch["token_mask"]      # (B, max_token_length)
            hand_pose = batch["hand_pose"]        # (B, 48)
            ...
    """

    def __init__(
        self,
        cache_dir: str,
        config_path: str,
        preload_all: bool = False,
        category_filter: Optional[List[str]] = None,
        intent_filter: Optional[List[int]] = None,
        min_token_length: int = 1,
    ):
        """
        Args:
            cache_dir: 预处理缓存目录（包含 dataset_index.json 和 npz 文件）
            config_path: token_config.yaml 路径
            preload_all: 是否预加载所有样本到内存（小数据集建议开启）
            category_filter: 只保留指定类别（None=全部）
            intent_filter: 只保留指定意图 ID（None=全部）
            min_token_length: 最小有效 token 数，低于此值的样本被过滤
        """
        super().__init__()

        self.cache_dir = cache_dir
        self.encoder = TokenEncoder(config_path)
        self.mapper = self.encoder.mapper

        # 加载索引
        index_path = os.path.join(cache_dir, "dataset_index.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"找不到 {index_path}，请先运行 preprocess/build_contact_dataset.py"
            )

        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        # 构建样本列表（应用过滤条件）
        self.samples = []
        for entry in index["samples"]:
            # 类别过滤
            if category_filter and entry["cate_id"] not in category_filter:
                continue
            # 意图过滤
            if intent_filter and int(entry.get("action_id", -1)) not in intent_filter:
                continue
            # 最小 token 长度过滤
            if entry["token_length"] < min_token_length:
                continue
            self.samples.append(entry)

        # 构建类别和意图的编码映射
        self._build_label_mappings()

        # 加载 loss 权重（如果存在）
        weights_path = os.path.join(cache_dir, "loss_weights.npy")
        if os.path.exists(weights_path):
            self.loss_weights = np.load(weights_path)
        else:
            self.loss_weights = None

        # 预加载
        self.preload_all = preload_all
        self._cache: Dict[int, Dict] = {}
        if preload_all:
            self._preload()

        print(f"[ContactTokenDataset] {cache_dir}")
        print(f"  样本数: {len(self.samples)}")
        print(f"  类别数: {len(self.cate_to_id)}")
        print(f"  意图数: {len(self.intent_to_id)}")
        print(f"  预加载: {preload_all}")

    def _build_label_mappings(self):
        """构建类别和意图的 str ↔ int 映射"""
        # 类别
        all_cates = sorted(set(s["cate_id"] for s in self.samples))
        self.cate_to_id = {c: i for i, c in enumerate(all_cates)}
        self.id_to_cate = {i: c for c, i in self.cate_to_id.items()}
        self.num_categories = len(all_cates)

        # 意图
        all_intents = sorted(set(
            str(s.get("action_id", "0")) for s in self.samples
        ))
        self.intent_to_id = {a: i for i, a in enumerate(all_intents)}
        self.id_to_intent = {i: a for a, i in self.intent_to_id.items()}
        self.num_intents = len(all_intents)

    def _preload(self):
        """预加载所有 npz 到内存"""
        print(f"  预加载 {len(self.samples)} 个样本...")
        for i, entry in enumerate(self.samples):
            self._cache[i] = self._load_npz(entry["cache_file"])

    def _load_npz(self, cache_file: str) -> Dict:
        """加载单个 npz 缓存文件"""
        path = os.path.join(self.cache_dir, cache_file)
        data = np.load(path, allow_pickle=True)
        
        # 兼容新旧配置：新配置使用 contact_matrix (6, 4)，旧配置使用 contact_type_matrix (6, 6)
        if "contact_matrix" in data:
            contact_matrix = data["contact_matrix"]
        else:
            # 旧缓存兼容
            contact_matrix = data["contact_type_matrix"]
        
        return {
            "token_seq": data["token_seq"],                       # (max_len,)
            "token_length": int(data["token_length"]),
            "hand_pose": data["hand_pose"],                       # (48,)
            "hand_shape": data["hand_shape"],                     # (10,)
            "hand_tsl": data["hand_tsl"],                         # (3,)
            "contact_matrix": contact_matrix,                     # (H, S) 新配置 (6, 4)
            "cate_id": str(data["cate_id"]),
            "action_id": str(data.get("action_id", "0")),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 加载数据
        if self.preload_all:
            raw = self._cache[idx]
        else:
            raw = self._load_npz(self.samples[idx]["cache_file"])

        token_seq = raw["token_seq"].astype(np.int64)
        token_length = raw["token_length"]
        max_len = self.encoder.max_token_length

        # token mask: True=有效语义 token, False=PAD/特殊 token
        token_mask = np.zeros(max_len, dtype=bool)
        for i, t in enumerate(token_seq):
            if self.mapper.is_semantic_token(int(t)):
                token_mask[i] = True

        # 类别和意图编码
        cate_id = self.cate_to_id.get(raw["cate_id"], 0)
        intent_id = self.intent_to_id.get(raw["action_id"], 0)

        return {
            "token_seq": torch.from_numpy(token_seq),                             # (max_len,) int64
            "token_length": torch.tensor(token_length, dtype=torch.long),         # scalar
            "token_mask": torch.from_numpy(token_mask),                           # (max_len,) bool
            "hand_pose": torch.from_numpy(raw["hand_pose"].astype(np.float32)),   # (48,)
            "hand_shape": torch.from_numpy(raw["hand_shape"].astype(np.float32)), # (10,)
            "hand_tsl": torch.from_numpy(raw["hand_tsl"].astype(np.float32)),     # (3,)
            "cate_id": torch.tensor(cate_id, dtype=torch.long),                   # scalar
            "intent_id": torch.tensor(intent_id, dtype=torch.long),               # scalar
            "contact_matrix": torch.from_numpy(
                raw["contact_matrix"].astype(np.int32)
            ),                                                                    # (H, S)
        }

    # ==========================================================================
    # 采样器（类别均衡）
    # ==========================================================================

    def get_category_balanced_sampler(self) -> WeightedRandomSampler:
        """
        返回类别均衡的 WeightedRandomSampler

        每个样本的权重 = 1 / 该类别的样本数
        使得每个 epoch 中各类别被采样的次数大致相等
        """
        cate_counts = Counter(s["cate_id"] for s in self.samples)

        weights = []
        for s in self.samples:
            w = 1.0 / cate_counts[s["cate_id"]]
            weights.append(w)

        weights = torch.DoubleTensor(weights)
        return WeightedRandomSampler(
            weights, num_samples=len(self.samples), replacement=True
        )

    # ==========================================================================
    # DataLoader 便捷方法
    # ==========================================================================

    def get_dataloader(
        self,
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 4,
        balanced_sampling: bool = False,
        drop_last: bool = True,
        pin_memory: bool = True,
    ) -> DataLoader:
        """
        创建 DataLoader

        Args:
            batch_size: 批大小
            shuffle: 是否打乱（balanced_sampling=True 时自动忽略）
            num_workers: 数据加载线程数
            balanced_sampling: 是否使用类别均衡采样
            drop_last: 是否丢弃最后不完整的 batch
            pin_memory: 是否 pin memory（GPU 训练建议开启）

        Returns:
            DataLoader 实例
        """
        sampler = None
        if balanced_sampling:
            sampler = self.get_category_balanced_sampler()
            shuffle = False  # sampler 和 shuffle 互斥

        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
            collate_fn=None,  # 默认 collate 即可（所有字段都是 tensor）
        )

    # ==========================================================================
    # 数据集信息
    # ==========================================================================

    def get_loss_weights_tensor(self, device: str = "cpu") -> Optional[torch.Tensor]:
        """获取 token loss 权重 tensor（用于 CrossEntropy 的 weight 参数）"""
        if self.loss_weights is None:
            return None
        return torch.from_numpy(self.loss_weights).float().to(device)

    def get_vocab_size(self) -> int:
        return self.encoder.vocab_size

    def get_max_token_length(self) -> int:
        return self.encoder.max_token_length

    def get_pad_token_id(self) -> int:
        return self.encoder.PAD

    def get_mask_token_id(self) -> int:
        return self.encoder.MASK

    def summary(self):
        """打印数据集摘要"""
        # token 长度分布
        lengths = [s["token_length"] for s in self.samples]
        lengths = np.array(lengths)

        # 类别分布
        cate_counts = Counter(s["cate_id"] for s in self.samples)

        print("=" * 60)
        print("ContactTokenDataset Summary")
        print("=" * 60)
        print(f"  样本数:        {len(self.samples)}")
        print(f"  Vocab size:    {self.get_vocab_size()}")
        print(f"  Max seq len:   {self.get_max_token_length()}")
        print(f"  PAD token:     {self.get_pad_token_id()}")
        print(f"  MASK token:    {self.get_mask_token_id()}")
        print(f"  类别数:        {self.num_categories}")
        print(f"  意图数:        {self.num_intents}")
        print(f"\n  Token 长度分布:")
        print(f"    min={lengths.min()}, max={lengths.max()}, "
              f"mean={lengths.mean():.1f}, median={np.median(lengths):.0f}")
        print(f"    p95={np.percentile(lengths, 95):.0f}, "
              f"p99={np.percentile(lengths, 99):.0f}")
        print(f"\n  类别分布 (top 10):")
        for cate, count in cate_counts.most_common(10):
            pct = count / len(self.samples) * 100
            print(f"    {cate:<20} {count:>5} ({pct:.1f}%)")
        if self.loss_weights is not None:
            active = np.sum(self.loss_weights > 0)
            print(f"\n  Loss weights: {active} active tokens")
        print("=" * 60)


# ==============================================================================
# 工具函数
# ==============================================================================

def build_dataloaders(
    config_path: str,
    cache_root: str,
    batch_size: int = 64,
    num_workers: int = 4,
    balanced_sampling: bool = True,
    preload_all: bool = False,
) -> Dict[str, DataLoader]:
    """
    一键构建 train/val/test 三个 DataLoader

    Args:
        config_path: token_config.yaml
        cache_root: 缓存根目录（下面应有 train/, val/, test/ 子目录）
        batch_size: 批大小
        num_workers: 加载线程数
        balanced_sampling: 训练集是否用类别均衡采样
        preload_all: 是否预加载全部数据

    Returns:
        {"train": DataLoader, "val": DataLoader, "test": DataLoader}
        缺失的 split 不会出现在返回值中
    """
    loaders = {}

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(cache_root, split)
        index_path = os.path.join(split_dir, "dataset_index.json")

        if not os.path.exists(index_path):
            print(f"[INFO] {split} 集缓存不存在，跳过: {split_dir}")
            continue

        dataset = ContactTokenDataset(
            cache_dir=split_dir,
            config_path=config_path,
            preload_all=preload_all,
        )

        is_train = (split == "train")
        loader = dataset.get_dataloader(
            batch_size=batch_size,
            shuffle=is_train and not balanced_sampling,
            num_workers=num_workers,
            balanced_sampling=is_train and balanced_sampling,
            drop_last=is_train,
            pin_memory=True,
        )

        loaders[split] = loader
        print(f"  {split}: {len(dataset)} samples, "
              f"{len(loader)} batches (bs={batch_size})")

    return loaders


# ==============================================================================
# 测试
# ==============================================================================

if __name__ == "__main__":
    import tempfile
    import shutil

    print("=== ContactTokenDataset 测试（模拟数据） ===\n")

    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "token_config.yaml"
    )
    encoder = TokenEncoder(config_path)

    # 创建临时缓存目录和模拟数据
    tmp_dir = tempfile.mkdtemp(prefix="contact_token_test_")
    print(f"临时目录: {tmp_dir}")

    try:
        num_samples = 50
        categories = ["scissor", "mug", "knife", "bottle", "hammer"]
        intents = ["0001", "0002", "0003"]
        index_samples = []

        np.random.seed(42)

        for i in range(num_samples):
            cate = categories[i % len(categories)]
            intent = intents[i % len(intents)]
            obj_id = f"{i:06d}"
            cache_name = f"{cate}_{obj_id}_{i:06d}.npz"

            # 随机生成 contact_matrix（新配置: 6 hand_parts × 4 slots）
            contact_matrix = np.full((6, 4), -1, dtype=np.int32)
            # 随机激活 2-6 个接触对
            num_contacts = np.random.randint(2, 7)
            for _ in range(num_contacts):
                h = np.random.randint(0, 6)
                s = np.random.randint(0, 4)
                contact_matrix[h, s] = 0  # 简化版: 0=接触

            token_seq = encoder.encode(contact_matrix)
            token_length = encoder.get_seq_length(token_seq)

            # 保存 npz
            np.savez_compressed(
                os.path.join(tmp_dir, cache_name),
                token_seq=token_seq,
                token_length=token_length,
                hand_pose=np.random.randn(48).astype(np.float32),
                hand_shape=np.random.randn(10).astype(np.float32),
                hand_tsl=np.random.randn(3).astype(np.float32),
                contact_matrix=contact_matrix,
                cate_id=cate,
                action_id=intent,
                grasp_idx=i,
            )

            index_samples.append({
                "idx": i,
                "cache_file": cache_name,
                "cate_id": cate,
                "obj_id": obj_id,
                "action_id": intent,
                "token_length": int(token_length),
            })

        # 保存 index
        with open(os.path.join(tmp_dir, "dataset_index.json"), "w") as f:
            json.dump({"total": num_samples, "num_valid": num_samples,
                       "samples": index_samples}, f)

        # 保存 loss weights
        all_seqs = np.stack([encoder.encode(np.full((6, 6), -1, dtype=np.int32))
                             for _ in range(num_samples)])
        # 用真实 token 重新算
        real_seqs = []
        for s in index_samples:
            data = np.load(os.path.join(tmp_dir, s["cache_file"]))
            real_seqs.append(data["token_seq"])
        real_seqs = np.stack(real_seqs)
        weights = encoder.compute_loss_weights(real_seqs)
        np.save(os.path.join(tmp_dir, "loss_weights.npy"), weights)

        # ==================== 测试 Dataset ====================
        print("\n--- 测试 Dataset 创建 ---")
        dataset = ContactTokenDataset(
            cache_dir=tmp_dir,
            config_path=config_path,
            preload_all=False,
        )
        dataset.summary()

        # 测试 __getitem__
        print("\n--- 测试 __getitem__ ---")
        sample = dataset[0]
        print("  返回字段:")
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
            else:
                print(f"    {k}: {v}")

        # 验证 token_mask
        seq = sample["token_seq"]
        mask = sample["token_mask"]
        length = sample["token_length"].item()
        num_true = mask.sum().item()
        print(f"\n  token_length={length}, mask 中 True 数={num_true}")
        assert num_true == length, f"mask 不一致: {num_true} != {length}"
        print("  ✅ token_mask 验证通过")

        # 测试 DataLoader
        print("\n--- 测试 DataLoader ---")
        loader = dataset.get_dataloader(
            batch_size=8, shuffle=True, num_workers=0, drop_last=False
        )

        for batch_idx, batch in enumerate(loader):
            if batch_idx == 0:
                print(f"  第一个 batch:")
                for k, v in batch.items():
                    print(f"    {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
            if batch_idx >= 2:
                break
        print(f"  共 {len(loader)} 个 batch ✅")

        # 测试均衡采样
        print("\n--- 测试类别均衡采样 ---")
        loader_balanced = dataset.get_dataloader(
            batch_size=16, balanced_sampling=True, num_workers=0, drop_last=False
        )
        cate_counter = Counter()
        for batch in loader_balanced:
            for c in batch["cate_id"].tolist():
                cate_counter[dataset.id_to_cate[c]] += 1
        print("  采样后类别分布:")
        for cate, count in cate_counter.most_common():
            print(f"    {cate}: {count}")
        # 均衡采样后各类别应大致相等
        counts = list(cate_counter.values())
        ratio = max(counts) / (min(counts) + 1)
        print(f"  最大/最小比: {ratio:.1f} (越接近 1 越均衡)")

        # 测试预加载模式
        print("\n--- 测试预加载模式 ---")
        dataset_preload = ContactTokenDataset(
            cache_dir=tmp_dir,
            config_path=config_path,
            preload_all=True,
        )
        sample_preload = dataset_preload[0]
        assert torch.equal(sample["token_seq"], sample_preload["token_seq"])
        print("  ✅ 预加载结果与按需加载一致")

        # 测试类别过滤
        print("\n--- 测试类别过滤 ---")
        dataset_filtered = ContactTokenDataset(
            cache_dir=tmp_dir,
            config_path=config_path,
            category_filter=["scissor", "mug"],
        )
        print(f"  过滤后样本数: {len(dataset_filtered)} (原 {len(dataset)})")
        filtered_cates = set(s["cate_id"] for s in dataset_filtered.samples)
        assert filtered_cates.issubset({"scissor", "mug"})
        print(f"  类别: {filtered_cates} ✅")

        # 测试 loss weights
        print("\n--- 测试 loss weights ---")
        lw = dataset.get_loss_weights_tensor()
        print(f"  shape: {lw.shape}, dtype: {lw.dtype}")
        print(f"  非零数: {(lw > 0).sum().item()}")
        print("  ✅ loss weights 加载成功")

        print("\n✅ 所有测试通过!")

    finally:
        shutil.rmtree(tmp_dir)
        print(f"\n已清理临时目录: {tmp_dir}")