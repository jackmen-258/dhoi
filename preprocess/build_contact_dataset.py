"""
build_contact_dataset.py
========================
一次性预处理脚本：遍历 OakInk 数据集，为每个 grasp 样本生成接触矩阵和 token 序列。

注意：per-object slot labels 已迁移到 build_bps_slot_cache.py（BPS 基点级别），
      本脚本不再生成 obj_slot_labels。

运行方式：
    python preprocess/build_contact_dataset.py \\
        --config configs/token_config.yaml \\
        --split train --category all --intent all

    # 指定输出目录 + 跳过已有
    python preprocess/build_contact_dataset.py \\
        --split train --save_dir ./cache/contact_tokens/train --skip_existing

输出：
    {save_dir}/
        {cate_id}_{obj_id}_{idx:06d}.npz    # 每个 grasp 的接触数据
        token_sequences.npz                  # 汇总的 token 序列（训练用）
        token_stats.json                     # token 频次统计
        build_stats.json                     # 构建统计
        dataset_index.json                   # 样本索引（dataset.py 使用）

前置条件：
  1. 环境变量 OAKINK_DIR 指向 OakInk 数据根目录
  2. OakBase 下有物体 part 标注（part_XX.json + part_XX.ply）
"""

import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.contact_builder import ContactBuilder
from data.token_encoder import TokenEncoder


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
# 单个 grasp 处理
# ==============================================================================

def process_single_grasp(
    builder: ContactBuilder,
    encoder: TokenEncoder,
    grasp: dict,
    mano_faces: np.ndarray,
    save_path: str,
    grasp_idx: int,
) -> dict:
    """
    处理单个 grasp 样本

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
# ==============================================================================

def main():
    args = parse_args()

    # 确定保存目录
    if args.save_dir is None:
        save_dir = os.path.join(PROJECT_ROOT, "cache", "contact_tokens", args.split)
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("OakInk 接触 Token 预处理")
    print("=" * 60)
    print(f"  Config:       {args.config}")
    print(f"  Split:        {args.split}")
    print(f"  Category:     {args.category}")
    print(f"  Intent:       {args.intent}")
    print(f"  Save dir:     {save_dir}")
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

    # 创建 OIShape 数据集
    print("\n[1/4] 加载 OakInk 数据集...")
    dataset = create_oi_shape_dataset(args.split, args.category, args.intent)
    print(f"  加载完成: {len(dataset)} 个 grasp 样本")

    # 初始化 MANO 顶点分组
    print("\n[2/4] 初始化 MANO 分组...")
    builder.init_hand_part_assignment(mano_layer=dataset.mano_layer)

    # 获取 MANO faces
    mano_faces = get_mano_faces(dataset.mano_layer)
    print(f"  MANO faces: {mano_faces.shape}")

    # Dry run 模式
    if args.dry_run:
        dry_run(dataset, builder, save_dir)
        return

    # 正式处理
    print(f"\n[3/4] 计算接触矩阵、token 序列...")
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

        result = process_single_grasp(
            builder=builder,
            encoder=encoder,
            grasp=grasp,
            mano_faces=mano_faces,
            save_path=save_path,
            grasp_idx=idx,
        )
        results.append(result)
        stats[result["status"]] = stats.get(result["status"], 0) + 1

    # 保存构建统计
    with open(os.path.join(save_dir, "build_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n  构建统计: 成功={stats['success']}, "
          f"跳过={stats['skipped']}, 失败={stats['failed']}, "
          f"无part={stats['no_parts']}")

    # 构建索引和统计
    print(f"\n[4/4] 生成索引和统计...")
    build_dataset_index(grasp_list, save_dir, results)
    compute_and_save_stats(encoder, results, save_dir)

    print(f"\n预处理完成！输出目录: {save_dir}")
    print(f"  缓存文件:     {save_dir}/*.npz")
    print(f"  数据集索引:   {save_dir}/dataset_index.json")
    print(f"  Token 序列:   {save_dir}/token_sequences.npz")
    print(f"  Token 统计:   {save_dir}/token_stats.json")
    print(f"  Loss 权重:    {save_dir}/loss_weights.npy")


if __name__ == "__main__":
    main()