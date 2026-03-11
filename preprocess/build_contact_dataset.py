"""
build_contact_dataset.py (v2 — 确定性采样 + slot labels)
=========================================================
一次性预处理脚本：遍历 OakInk 数据集，为每个 grasp 样本生成接触矩阵、token 序列
以及物体点云的逐点 slot 标注。

v2 变更（相对 v1）:
  - 新增 obj_slot_labels: (2048,) int32，标识每个采样点的 canonical slot 归属
  - 使用 deterministic_sample_surface 保证采样点与 OIShape.__getitem__ 完全一致
  - slot_labels 按 obj_id 缓存（同一物体只计算一次）
  - 新增 --no_slot_labels 标志跳过 slot 标注（兼容旧流程）
  - 新增 slot_label_stats.json 统计（覆盖率、各 slot 点数分布）

运行方式：
    # 基本用法（默认生成 slot labels）
    python preprocess/build_contact_dataset.py \\
        --config configs/token_config.yaml \\
        --split train --category all --intent all

    # 不生成 slot labels（兼容旧流程）
    python preprocess/build_contact_dataset.py \\
        --split train --no_slot_labels

    # 指定输出目录 + 跳过已有
    python preprocess/build_contact_dataset.py \\
        --split train --save_dir ./cache/contact_tokens/train --skip_existing

输出：
    {save_dir}/
        {cate_id}_{obj_id}_{idx:06d}.npz    # 每个 grasp 的接触数据 + slot_labels
        token_sequences.npz                  # 汇总的 token 序列（训练用）
        token_stats.json                     # token 频次统计
        build_stats.json                     # 构建统计
        dataset_index.json                   # 样本索引（dataset.py 使用）
        slot_label_stats.json                # [v2] slot 覆盖率统计

前置条件：
  1. 环境变量 OAKINK_DIR 指向 OakInk 数据根目录
  2. OakBase 下有物体 part 标注（part_XX.json + part_XX.ply）
  3. data/oishape_deterministic.py 已就位（确定性采样模块）
"""

import os
import sys
import json
import argparse
import numpy as np
import trimesh                                                      # [v2] 新增
from pathlib import Path
from tqdm import tqdm
from collections import Counter, defaultdict
from typing import Dict, Optional, Tuple                            # [v2] 新增

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.slot_mapping import SlotMapper
from data.contact_builder import ContactBuilder
from data.token_encoder import TokenEncoder
from data.oishape_deterministic import sample_with_slot_labels      # [v2] 新增


def parse_args():
    parser = argparse.ArgumentParser(
        description="预处理 OakInk 数据集：生成接触矩阵和 token 序列"
    )
    parser.add_argument(
        "--config", type=str, default="configs/token_config.yaml",
        help="token_config.yaml 路径",
    )
    parser.add_argument(
        "--split", type=str, default="train",
        choices=["train", "val", "test", "all"],
        help="数据集划分",
    )
    parser.add_argument(
        "--category", type=str, default="all",
        help="物体类别，逗号分隔或 'all'",
    )
    parser.add_argument(
        "--intent", type=str, default="all",
        help="意图模式，逗号分隔或 'all'",
    )
    parser.add_argument(
        "--save_dir", type=str, default=None,
        help="输出目录，默认 ./cache/contact_tokens/{split}",
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="跳过已有缓存文件",
    )
    # ---- [v2] 新增参数 ----
    parser.add_argument(
        "--no_slot_labels", action="store_true",
        help="不生成 obj_slot_labels（兼容旧流程）",
    )
    parser.add_argument(
        "--n_samples", type=int, default=2048,
        help="物体表面采样点数（必须与 OIShape.n_samples 一致）",
    )
    # ---- [v2] end ----
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="并行 worker 数（0=单进程）",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="只统计不实际计算（检查数据路径是否正确）",
    )
    return parser.parse_args()


class OIShapeConfig:
    """
    模拟 OIShape 所需的 cfg 对象

    OIShape.__init__ 需要 cfg 具有以下属性：
      - DATA_SPLIT
      - INTENT_MODE
      - OBJ_CATES
    """

    def __init__(self, split: str, category: str, intent: str):
        self.DATA_SPLIT = split
        self.OBJ_CATES = category
        self.INTENT_MODE = intent

        # 关闭数据增强（预处理阶段不需要）
        self.AUG_RIGID_P = 0.0


def create_oi_shape_dataset(split: str, category: str, intent: str):
    """
    创建 OIShape 数据集实例

    注意：需要设置环境变量 OAKINK_DIR 指向 OakInk 数据根目录
    """
    # 延迟导入（避免在没有 OakInk 数据时就报错）
    from data.oishape_dataset import OIShape

    cfg = OIShapeConfig(split, category, intent)
    dataset = OIShape(cfg)
    return dataset


def get_mano_faces(mano_layer) -> np.ndarray:
    """从 ManoLayer 获取面片索引"""
    if hasattr(mano_layer, "th_faces"):
        return mano_layer.th_faces.detach().cpu().numpy()
    elif hasattr(mano_layer, "faces"):
        if hasattr(mano_layer.faces, "numpy"):
            return mano_layer.faces.detach().cpu().numpy()
        return np.array(mano_layer.faces, dtype=np.int64)
    else:
        raise AttributeError("MANO layer 没有 faces 属性")


# ==============================================================================
# [v2] Slot Label 生成器（按 obj_id 缓存）
# ==============================================================================

class SlotLabelGenerator:
    """
    为物体点云生成逐点 slot 标注。

    工作流程：
      1. 用 deterministic_sample_surface 对整体 obj_mesh 采样
         （保证与 OIShape.__getitem__ 产生完全相同的 2048 个点）
      2. 加载各 part ply → {slot_id: part_trimesh}
      3. 对每个采样点，计算到各 part mesh 的最近距离，取最近者的 slot_id

    按 obj_id 缓存：同一物体的不同 grasp 共享完全相同的 slot_labels。
    """

    def __init__(
        self,
        mapper:         SlotMapper,
        oakbase_dir:    str,
        obj_id_to_name: Dict[str, str],
        n_samples:      int = 2048,
    ):
        """
        Args:
            mapper:         SlotMapper 实例
            oakbase_dir:    OakBase 根目录（包含各类别子目录）
            obj_id_to_name: obj_id → obj_name 映射（来自 ContactBuilder._obj_id_to_name）
            n_samples:      采样点数（必须与 OIShape.n_samples 一致）
        """
        self.mapper         = mapper
        self.oakbase_dir    = oakbase_dir
        self.obj_id_to_name = obj_id_to_name
        self.n_samples      = n_samples

        # 缓存: obj_id → slot_labels (N,) int32
        self._cache: Dict[str, np.ndarray] = {}
        # 统计
        self.stats = {"computed": 0, "cached": 0, "failed": 0}

    def get_slot_labels(
        self,
        obj_id:     str,
        cate_id:    str,
        obj_mesh:   trimesh.Trimesh,
        raw_obj_id: Optional[str] = None,
        bbox_center: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        获取物体点云的 slot 标注（带缓存）

        Args:
            obj_id:      物体 ID（也是确定性采样的种子）
            cate_id:     物体类别
            obj_mesh:    完整物体网格（已 bbox 中心化，与 OIShape.get_obj_mesh 一致）
            raw_obj_id:  真实物体 ID（虚拟物体需要，part 标注复用真实物体）
            bbox_center: 原始 bbox 中心 (从 OIShape.obj_bbox_centers 获取)，
                         用于正确平移 part PLY 到居中坐标系

        Returns:
            slot_labels: (n_samples,) int32, -1=未分配
        """
        # 缓存命中
        if obj_id in self._cache:
            self.stats["cached"] += 1
            return self._cache[obj_id]

        # 查找 part 目录（复用 ContactBuilder 的路径解析逻辑）
        part_dir = self._find_part_dir(obj_id, cate_id, raw_obj_id)
        if part_dir is None:
            # 无 part 标注 → 全 -1（ContactAwareLoss 自动退化为全局 pen）
            labels = np.full(self.n_samples, -1, dtype=np.int32)
            self._cache[obj_id] = labels
            self.stats["failed"] += 1
            return labels

        # 加载 part meshes → {slot_id: trimesh}
        part_meshes = self._load_part_meshes(part_dir, cate_id, obj_mesh,
                                             bbox_center=bbox_center)

        if not part_meshes:
            labels = np.full(self.n_samples, -1, dtype=np.int32)
            self._cache[obj_id] = labels
            self.stats["failed"] += 1
            return labels

        # 确定性采样 + slot 分配
        _, _, slot_labels = sample_with_slot_labels(
            obj_mesh, obj_id, part_meshes, self.n_samples,
        )

        self._cache[obj_id] = slot_labels
        self.stats["computed"] += 1
        return slot_labels

    def _find_part_dir(
        self, obj_id: str, cate_id: str, raw_obj_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        查找物体的 part 标注目录

        路径解析与 ContactBuilder._find_part_dir 一致：
          obj_id → metaV2 查 name → OakBase/{cate_id}/{name}/
          若 obj_id 查不到，尝试 raw_obj_id（虚拟物体复用真实物体 part）
        """
        def _has_parts(d: str) -> bool:
            return os.path.isdir(d) and any(
                f.startswith("part_") and f.endswith(".json")
                for f in os.listdir(d))

        candidate_ids = [obj_id]
        if raw_obj_id and raw_obj_id != obj_id:
            candidate_ids.append(raw_obj_id)

        for oid in candidate_ids:
            name = self.obj_id_to_name.get(oid)
            if name is None:
                continue
            # 主路径
            d = os.path.join(self.oakbase_dir, cate_id, name)
            if _has_parts(d):
                return d
            # 跨类别 fallback
            if os.path.isdir(self.oakbase_dir):
                for sub in os.listdir(self.oakbase_dir):
                    alt = os.path.join(self.oakbase_dir, sub, name)
                    if _has_parts(alt):
                        return alt
        return None

    def _load_part_meshes(
        self,
        part_dir:  str,
        cate_id:   str,
        obj_mesh:  trimesh.Trimesh,
        require_faces: bool = False,
        bbox_center: Optional[np.ndarray] = None,
    ) -> Dict[int, trimesh.Trimesh]:
        """
        加载 part meshes，应用 bbox 中心化

        Args:
            require_faces: True → 跳过无面片的 mesh
                           False (默认) → 保留纯点云 mesh
            bbox_center: 原始 bbox 中心偏移量 (从 OIShape.obj_bbox_centers 获取)
                         如果为 None, 则从 obj_mesh 重新计算
                         注意: obj_mesh 已居中时, 重新计算得到 ~0, part 不会被正确平移！

        Returns:
            {slot_id: part_trimesh（已中心化）}
        """
        parts = self.mapper.load_object_parts(part_dir, cate_id)

        # 使用提供的 bbox_center, 或从 obj_mesh 重新计算 (仅限未居中的 mesh)
        if bbox_center is None:
            bbox_center = (obj_mesh.vertices.min(0) + obj_mesh.vertices.max(0)) / 2

        part_meshes: Dict[int, trimesh.Trimesh] = {}
        for part_key, info in parts.items():
            sid = info["slot_id"]
            ply = info["ply_path"]
            if sid is None or ply is None or not Path(ply).exists():
                continue

            # 加载 PLY: 可能是 mesh, 点云, 或 Scene
            raw = trimesh.load(ply, process=False)
            if isinstance(raw, trimesh.Trimesh):
                verts = raw.vertices
                faces = raw.faces
            elif isinstance(raw, trimesh.PointCloud):
                verts = raw.vertices
                faces = np.zeros((0, 3), dtype=np.int64)
            elif isinstance(raw, trimesh.Scene):
                # Scene 可能包含多个 geometry
                all_v = [g.vertices for g in raw.geometry.values()
                         if hasattr(g, 'vertices') and len(g.vertices) > 0]
                if not all_v:
                    continue
                verts = np.vstack(all_v)
                faces = np.zeros((0, 3), dtype=np.int64)
            else:
                continue

            if len(verts) == 0:
                continue

            # 构建 Trimesh (纯点云时 faces 为空)
            mesh = trimesh.Trimesh(vertices=verts - bbox_center,
                                   faces=faces, process=False)

            # 跳过无面片的退化 mesh (slot_labels 的 closest_point 需要面片)
            if require_faces and len(mesh.faces) == 0:
                continue

            if sid in part_meshes:
                # 同一 slot 多个 part → 合并
                part_meshes[sid] = trimesh.util.concatenate(
                    [part_meshes[sid], mesh])
            else:
                part_meshes[sid] = mesh

        return part_meshes

    def print_stats(self):
        s = self.stats
        total = s["computed"] + s["cached"] + s["failed"]
        print(f"  [SlotLabels] 总请求: {total}, "
              f"新计算: {s['computed']}, 缓存命中: {s['cached']}, "
              f"无part标注: {s['failed']}")


# ==============================================================================
# [v2] Slot label 统计
# ==============================================================================

def compute_slot_label_stats(
    slot_label_gen: SlotLabelGenerator,
    save_dir: str,
    mapper: SlotMapper,
):
    """统计 slot label 的覆盖率和各 slot 点数分布"""
    cache = slot_label_gen._cache
    if not cache:
        return

    total_points = 0
    assigned     = 0
    slot_counts  = Counter()
    num_objects  = 0

    for key, labels in cache.items():
        num_objects += 1
        n = len(labels)
        total_points += n
        valid = labels[labels >= 0]
        assigned += len(valid)
        for sid in valid:
            slot_counts[int(sid)] += 1

    slot_names = mapper.canonical_slots
    stats = {
        "num_objects":      num_objects,
        "total_points":     total_points,
        "assigned_points":  assigned,
        "coverage":         assigned / max(total_points, 1),
        "slot_distribution": {
            slot_names[sid] if sid < len(slot_names) else f"slot_{sid}": count
            for sid, count in sorted(slot_counts.items())
        },
    }

    with open(os.path.join(save_dir, "slot_label_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n  [Slot Labels 统计]")
    print(f"    物体数: {stats['num_objects']}")
    print(f"    覆盖率: {stats['coverage']:.1%} "
          f"({stats['assigned_points']}/{stats['total_points']})")
    print(f"    各 slot 点数:")
    for name, count in stats["slot_distribution"].items():
        print(f"      {name}: {count}")


# ==============================================================================
# 单个 grasp 处理
# [v2] 新增 slot_labels 参数
# ==============================================================================

def process_single_grasp(
    builder: ContactBuilder,
    encoder: TokenEncoder,
    grasp: dict,
    mano_faces: np.ndarray,
    save_path: str,
    grasp_idx: int,
    slot_labels: Optional[np.ndarray] = None,       # [v2] 新增
) -> dict:
    """
    处理单个 grasp 样本

    Args:
        slot_labels: [v2] (N,) int32 物体点云的 slot 标注，None 则不保存

    Returns:
        {
            "status": "success" | "failed" | "no_parts",
            "token_seq": np.array or None,
            "token_length": int,
            "cate_id": str,
            "obj_id": str,
        }
    """
    hand_verts = grasp["hand_verts"]
    obj_id = grasp["obj_id"]
    cate_id = grasp["cate_id"]
    raw_obj_id = grasp.get("raw_obj_id", obj_id)

    if hand_verts is None:
        return {"status": "failed", "token_seq": None, "token_length": 0,
                "cate_id": cate_id, "obj_id": obj_id}

    # 构建接触矩阵
    contact_result = builder.build_contact_for_grasp(
        hand_verts=hand_verts,
        obj_id=obj_id,
        cate_id=cate_id,
        raw_obj_id=raw_obj_id,
        mano_faces=mano_faces,
    )

    if contact_result is None:
        return {"status": "no_parts", "token_seq": None, "token_length": 0,
                "cate_id": cate_id, "obj_id": obj_id}

    # 编码为 token 序列
    # 新配置: contact_matrix (6, 4) 二元接触矩阵
    contact_matrix = contact_result["contact_matrix"]
    token_seq = encoder.encode(contact_matrix)
    token_length = encoder.get_seq_length(token_seq)

    # 组装保存字典（简化版配置只保留 contact_matrix）
    save_dict = dict(
        # 接触矩阵（简化版：二元 0=接触, -1=无接触）
        contact_matrix=contact_matrix,
        # token 序列
        token_seq=token_seq,
        token_length=token_length,
        # MANO 参数（训练 pose decoder 时需要）
        hand_pose=np.array(grasp["hand_pose"], dtype=np.float32),
        hand_shape=np.array(grasp["hand_shape"], dtype=np.float32),
        hand_tsl=np.array(grasp["hand_tsl"], dtype=np.float32),
        # 元信息
        obj_id=obj_id,
        cate_id=cate_id,
        action_id=grasp.get("action_id", ""),
        grasp_idx=grasp_idx,
    )

    # [v2] 保存 slot labels
    if slot_labels is not None:
        save_dict["obj_slot_labels"] = slot_labels

    np.savez_compressed(save_path, **save_dict)

    return {
        "status": "success",
        "token_seq": token_seq,
        "token_length": token_length,
        "cate_id": cate_id,
        "obj_id": obj_id,
    }


# ==============================================================================
# 索引构建（与 v1 相同）
# ==============================================================================

def build_dataset_index(
    grasp_list: list,
    save_dir: str,
    results: list,
) -> dict:
    """
    构建数据集索引文件（供 dataset.py 快速加载）

    Returns:
        index dict
    """
    index = {
        "total": len(grasp_list),
        "samples": [],
    }

    for idx, (grasp, result) in enumerate(zip(grasp_list, results)):
        if result["status"] != "success":
            continue

        cate_id = grasp["cate_id"]
        obj_id = grasp["obj_id"]
        cache_name = f"{cate_id}_{obj_id}_{idx:06d}.npz"

        index["samples"].append({
            "idx": idx,
            "cache_file": cache_name,
            "cate_id": cate_id,
            "obj_id": obj_id,
            "action_id": grasp.get("action_id", ""),
            "token_length": result["token_length"],
        })

    index["num_valid"] = len(index["samples"])

    index_path = os.path.join(save_dir, "dataset_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    return index


# ==============================================================================
# Token 统计（与 v1 相同）
# ==============================================================================

def compute_and_save_stats(
    encoder: TokenEncoder,
    results: list,
    save_dir: str,
):
    """
    统计 token 分布并保存

    生成：
      - token_sequences.npz: 所有有效的 token 序列（训练用）
      - token_stats.json: 统计信息
    """
    # 收集所有有效 token 序列
    valid_seqs = []
    token_lengths = []
    cate_counter = Counter()
    # 简化版配置移除了 contact_type 维度

    mapper = encoder.mapper

    for r in results:
        if r["status"] != "success" or r["token_seq"] is None:
            continue
        valid_seqs.append(r["token_seq"])
        token_lengths.append(r["token_length"])
        cate_counter[r["cate_id"]] += 1

    if not valid_seqs:
        print("[WARNING] 没有有效样本!")
        return

    all_seqs = np.stack(valid_seqs, axis=0)  # (N, max_token_length)
    token_lengths = np.array(token_lengths)

    # 保存汇总 token 序列
    np.savez_compressed(
        os.path.join(save_dir, "token_sequences.npz"),
        token_seqs=all_seqs,
        token_lengths=token_lengths,
    )

    # 计算 token 频次和 loss 权重
    freqs = encoder.compute_token_frequencies(all_seqs)
    loss_weights = encoder.compute_loss_weights(all_seqs)

    np.save(os.path.join(save_dir, "token_frequencies.npy"), freqs)
    np.save(os.path.join(save_dir, "loss_weights.npy"), loss_weights)

    # 统计信息
    # token 级别统计（简化版：只有 hand_part + slot，无 contact_type）
    token_counts = Counter()
    for seq in all_seqs:
        for t in seq:
            t = int(t)
            if mapper.is_semantic_token(t):
                h, s = mapper.decode_token(t)
                token_counts[f"{h}+{s}"] += 1

    # 排序取 top-20
    top_tokens = token_counts.most_common(20)

    stats = {
        "num_valid_samples": len(valid_seqs),
        "token_length_stats": {
            "min": int(token_lengths.min()),
            "max": int(token_lengths.max()),
            "mean": float(token_lengths.mean()),
            "median": float(np.median(token_lengths)),
            "p95": float(np.percentile(token_lengths, 95)),
            "p99": float(np.percentile(token_lengths, 99)),
        },
        "category_distribution": dict(cate_counter.most_common()),
        "top_20_tokens": [
            {"token": name, "count": count} for name, count in top_tokens
        ],
        "num_unique_active_tokens": int(np.sum(freqs > 0)),
        "max_token_length_config": encoder.max_token_length,
        "samples_exceeding_max_length": int(
            np.sum(token_lengths > encoder.max_token_length)
        ),
    }

    stats_path = os.path.join(save_dir, "token_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # 打印摘要
    print("\n" + "=" * 60)
    print("Token 统计摘要")
    print("=" * 60)
    print(f"  有效样本数: {stats['num_valid_samples']}")
    print(f"  Token 长度: min={stats['token_length_stats']['min']}, "
          f"max={stats['token_length_stats']['max']}, "
          f"mean={stats['token_length_stats']['mean']:.1f}, "
          f"p95={stats['token_length_stats']['p95']:.0f}")
    print(f"  活跃 token 种类: {stats['num_unique_active_tokens']} / {encoder.mapper.num_semantic_tokens}")
    print(f"  超出 max_length 的样本: {stats['samples_exceeding_max_length']}")
    print(f"\n  类别分布 (top 10):")
    for cate, count in list(stats["category_distribution"].items())[:10]:
        print(f"    {cate}: {count}")
    print(f"\n  最常见 token (top 10):")
    for item in stats["top_20_tokens"][:10]:
        print(f"    {item['token']}: {item['count']}")
    print("=" * 60)


# ==============================================================================
# Dry run（与 v1 相同）
# ==============================================================================

def dry_run(dataset, builder: ContactBuilder, save_dir: str):
    """
    Dry run：不计算接触，只检查 part 目录可用性
    """
    grasp_list = dataset.grasp_list

    has_parts = 0
    no_parts = 0
    no_parts_objs = set()
    cate_stats = defaultdict(lambda: {"total": 0, "has_parts": 0})

    checked_objs = {}  # {(obj_id, raw_obj_id): bool}

    for idx in tqdm(range(len(grasp_list)), desc="Dry run: 检查 part 可用性"):
        grasp = grasp_list[idx]
        obj_id = grasp["obj_id"]
        cate_id = grasp["cate_id"]
        raw_obj_id = grasp.get("raw_obj_id", obj_id)

        cate_stats[cate_id]["total"] += 1

        check_key = (obj_id, raw_obj_id)
        if check_key in checked_objs:
            if checked_objs[check_key]:
                has_parts += 1
                cate_stats[cate_id]["has_parts"] += 1
            else:
                no_parts += 1
            continue

        part_dir = builder._find_part_dir(obj_id, cate_id, raw_obj_id)
        if part_dir is not None:
            checked_objs[check_key] = True
            has_parts += 1
            cate_stats[cate_id]["has_parts"] += 1
        else:
            checked_objs[check_key] = False
            no_parts += 1
            no_parts_objs.add((cate_id, obj_id))

    print("\n" + "=" * 60)
    print("Dry Run 结果")
    print("=" * 60)
    print(f"  总样本数: {len(grasp_list)}")
    print(f"  有 part 标注: {has_parts}")
    print(f"  无 part 标注: {no_parts}")
    print(f"  物体总数: {len(checked_objs)}")
    print(f"  无 part 的物体数: {len(no_parts_objs)}")

    print(f"\n  按类别统计:")
    for cate in sorted(cate_stats.keys()):
        s = cate_stats[cate]
        pct = s["has_parts"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"    {cate:<20} {s['has_parts']:>5}/{s['total']:<5} ({pct:.0f}%)")

    if no_parts_objs:
        print(f"\n  无 part 的物体 (前 20):")
        for cate, oid in sorted(no_parts_objs)[:20]:
            print(f"    {cate}/{oid}")

    # 保存 dry run 结果
    os.makedirs(save_dir, exist_ok=True)
    dry_run_path = os.path.join(save_dir, "dry_run_result.json")
    with open(dry_run_path, "w") as f:
        json.dump({
            "total": len(grasp_list),
            "has_parts": has_parts,
            "no_parts": no_parts,
            "no_parts_objects": [
                {"cate": c, "obj_id": o} for c, o in sorted(no_parts_objs)
            ],
            "category_stats": dict(cate_stats),
        }, f, indent=2)

    print(f"\n  结果已保存到: {dry_run_path}")
    print("=" * 60)


# ==============================================================================
# 主流程
# [v2] 集成 SlotLabelGenerator，步骤从 4 步增加到 5 步
# ==============================================================================

def main():
    args = parse_args()

    # 确定保存目录
    if args.save_dir is None:
        save_dir = os.path.join(PROJECT_ROOT, "cache", "contact_tokens", args.split)
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    generate_slot_labels = not args.no_slot_labels                  # [v2]

    print("=" * 60)
    print("OakInk 接触 Token 预处理" +
          (" (v2: +slot labels)" if generate_slot_labels else ""))   # [v2]
    print("=" * 60)
    print(f"  Config:       {args.config}")
    print(f"  Split:        {args.split}")
    print(f"  Category:     {args.category}")
    print(f"  Intent:       {args.intent}")
    print(f"  Save dir:     {save_dir}")
    print(f"  Slot labels:  {'ON' if generate_slot_labels else 'OFF'}")  # [v2]
    print(f"  n_samples:    {args.n_samples}")                      # [v2]
    print(f"  Dry run:      {args.dry_run}")
    print("=" * 60)

    # 检查环境变量
    if "OAKINK_DIR" not in os.environ:
        print("\n[ERROR] 请设置环境变量 OAKINK_DIR")
        print("  export OAKINK_DIR=/path/to/OakInk")
        sys.exit(1)

    # 初始化（ContactBuilder 自动从 OAKINK_DIR 加载 metaV2 映射）
    config_path = os.path.join(PROJECT_ROOT, args.config)
    builder = ContactBuilder(config_path)  # 自动读 OAKINK_DIR
    encoder = TokenEncoder(config_path)
    mapper  = SlotMapper(config_path)                               # [v2]

    # 创建 OIShape 数据集
    print("\n[1/5] 加载 OakInk 数据集...")
    dataset = create_oi_shape_dataset(args.split, args.category, args.intent)
    print(f"  加载完成: {len(dataset)} 个 grasp 样本")

    # 初始化 MANO 顶点分组
    print("\n[2/5] 初始化 MANO 分组...")
    builder.init_hand_part_assignment(mano_layer=dataset.mano_layer)

    # 获取 MANO faces
    mano_faces = get_mano_faces(dataset.mano_layer)
    print(f"  MANO faces: {mano_faces.shape}")

    # Dry run 模式
    if args.dry_run:
        dry_run(dataset, builder, save_dir)
        return

    # ---- [v2] 初始化 SlotLabelGenerator ----
    slot_label_gen = None
    if generate_slot_labels:
        print("\n[3/5] 初始化 SlotLabelGenerator...")
        oakink_root = os.environ["OAKINK_DIR"]
        oakbase_dir = os.path.join(oakink_root, "OakBase")
        if not os.path.isdir(oakbase_dir):
            print(f"  [WARNING] OakBase 目录不存在: {oakbase_dir}")
            print(f"  将回退到不生成 slot labels 的模式")
            generate_slot_labels = False
            slot_label_gen = None
        else:
            slot_label_gen = SlotLabelGenerator(
                mapper=mapper,
                oakbase_dir=oakbase_dir,
                obj_id_to_name=builder._obj_id_to_name,
                n_samples=args.n_samples,
            )
            print(f"  OakBase: {oakbase_dir}")
            print(f"  obj_id→name 映射数: {len(builder._obj_id_to_name)}")
            print(f"  采样点数: {args.n_samples}")
    else:
        print("\n[3/5] 跳过 SlotLabelGenerator（--no_slot_labels）")
    # ---- [v2] end ----

    # 正式处理
    print(f"\n[4/5] 计算接触矩阵、token 序列"
          + ("、slot labels..." if generate_slot_labels else "..."))
    grasp_list = dataset.grasp_list

    results = []
    stats = {"total": len(grasp_list), "success": 0,
             "skipped": 0, "failed": 0, "no_parts": 0}

    for idx in tqdm(range(len(grasp_list)), desc="Processing"):
        grasp = grasp_list[idx]
        obj_id = grasp["obj_id"]
        cate_id = grasp["cate_id"]

        cache_name = f"{cate_id}_{obj_id}_{idx:06d}.npz"
        save_path = os.path.join(save_dir, cache_name)

        # 跳过已有缓存
        if args.skip_existing and os.path.exists(save_path):
            # 尝试加载已有缓存来获取 token_seq
            try:
                cached = np.load(save_path, allow_pickle=True)
                result = {
                    "status": "success",
                    "token_seq": cached["token_seq"],
                    "token_length": int(cached["token_length"]),
                    "cate_id": cate_id,
                    "obj_id": obj_id,
                }
            except Exception:
                result = {
                    "status": "skipped",
                    "token_seq": None,
                    "token_length": 0,
                    "cate_id": cate_id,
                    "obj_id": obj_id,
                }
            results.append(result)
            stats["skipped"] += 1
            continue

        # ---- [v2] 获取 slot labels ----
        slot_labels = None
        if slot_label_gen is not None:
            # OIShape.get_obj_mesh：加载 + bbox 中心化 + 缓存
            # 保证 obj_mesh 与 OIShape.__getitem__ 中使用的完全一致
            obj_mesh = dataset.get_obj_mesh(idx)
            # 原始 bbox_center: part PLY 在原始坐标系, 需要用此偏移量对齐
            bbox_ctr = dataset.obj_bbox_centers.get(obj_id)
            slot_labels = slot_label_gen.get_slot_labels(
                obj_id=obj_id,
                cate_id=cate_id,
                obj_mesh=obj_mesh,
                raw_obj_id=grasp.get("raw_obj_id"),
                bbox_center=bbox_ctr,
            )
        # ---- [v2] end ----

        # 处理
        result = process_single_grasp(
            builder=builder,
            encoder=encoder,
            grasp=grasp,
            mano_faces=mano_faces,
            save_path=save_path,
            grasp_idx=idx,
            slot_labels=slot_labels,                                # [v2] 传入
        )
        results.append(result)
        stats[result["status"]] = stats.get(result["status"], 0) + 1

    # 保存构建统计
    with open(os.path.join(save_dir, "build_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n  构建统计: 成功={stats['success']}, "
          f"跳过={stats['skipped']}, 失败={stats['failed']}, "
          f"无part={stats['no_parts']}")

    # [v2] 打印 slot label 统计
    if slot_label_gen is not None:
        slot_label_gen.print_stats()

    # 构建索引和统计
    print(f"\n[5/5] 生成索引和统计...")
    build_dataset_index(grasp_list, save_dir, results)
    compute_and_save_stats(encoder, results, save_dir)

    # [v2] 保存 slot label 统计
    if slot_label_gen is not None:
        compute_slot_label_stats(slot_label_gen, save_dir, mapper)



    print(f"\n✅ 预处理完成！输出目录: {save_dir}")
    print(f"  缓存文件:     {save_dir}/*.npz")
    print(f"  数据集索引:   {save_dir}/dataset_index.json")
    print(f"  Token 序列:   {save_dir}/token_sequences.npz")
    print(f"  Token 统计:   {save_dir}/token_stats.json")
    print(f"  Loss 权重:    {save_dir}/loss_weights.npy")
    # [v2]
    if generate_slot_labels:
        print(f"  Slot 统计:    {save_dir}/slot_label_stats.json")
        print(f"  每个 npz 含:  obj_slot_labels (N,) int32")


if __name__ == "__main__":
    main()