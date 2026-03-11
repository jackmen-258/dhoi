#!/usr/bin/env python3
"""
build_text_cache.py — 预计算文本描述

从 token cache 中的 contact_type_matrix + dataset_index.json 生成确定性文本描述.

输出 (每个 split):
  {cache_dir}/text_cache.json — {cache_file: text_string}

用法:
  python preprocess/build_text_cache.py
  python preprocess/build_text_cache.py --splits train val test
"""

import os
import sys
import json
import argparse
import logging

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.slot_mapping import SlotMapper
from data.text_generator import ContactTextGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def build_cache_for_split(cache_dir, text_gen):
    """为单个 split 构建 text cache"""

    index_path = os.path.join(cache_dir, "dataset_index.json")
    if not os.path.isfile(index_path):
        logger.warning(f"  Skipped: {index_path} not found")
        return

    with open(index_path) as f:
        index = json.load(f)
    samples = index["samples"]
    logger.info(f"  {len(samples)} samples in index")

    text_map = {}
    n_empty = 0
    n_missing = 0

    for entry in tqdm(samples, desc="  Generating text"):
        cache_file = entry["cache_file"]
        npz_path = os.path.join(cache_dir, cache_file)
        if not os.path.isfile(npz_path):
            n_missing += 1
            continue

        data = np.load(npz_path, allow_pickle=True)
        # 新配置使用 contact_matrix (6, 4)，旧配置使用 contact_type_matrix (6, 6)
        if "contact_matrix" in data:
            contact_mat = data["contact_matrix"].astype(np.int32)
        else:
            # 兼容旧缓存
            contact_mat = data["contact_type_matrix"].astype(np.int32)

        # cate_id 和 action_id 从 index entry 获取 (比 npz 更可靠)
        cate_id   = entry.get("cate_id", "")
        action_id = entry.get("action_id", "")

        text = text_gen.generate_single(contact_mat, cate_id, action_id)
        text_map[cache_file] = text

        if "no contact" in text:
            n_empty += 1

    # 保存
    out_path = os.path.join(cache_dir, "text_cache.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(text_map, f, indent=2, ensure_ascii=False)

    logger.info(f"  Saved: {out_path}")
    logger.info(f"  Total: {len(text_map)}, empty contact: {n_empty}, "
                f"missing npz: {n_missing}")

    # 打印样例
    items = list(text_map.items())[:5]
    for k, v in items:
        logger.info(f"    {k}: \"{v}\"")


def main():
    parser = argparse.ArgumentParser("Build text cache for each split")
    parser.add_argument("--cache_root", type=str,
                        default="cache/contact_tokens")
    parser.add_argument("--splits", nargs="+",
                        default=["train", "val", "test"])
    parser.add_argument("--config_path", type=str,
                        default="configs/token_config.yaml")
    args = parser.parse_args()

    mapper = SlotMapper(args.config_path)
    text_gen = ContactTextGenerator(mapper)

    for split in args.splits:
        cache_dir = os.path.join(args.cache_root, split)
        logger.info(f"\n{'='*50}")
        logger.info(f"Split: {split} ({cache_dir})")
        logger.info(f"{'='*50}")

        if not os.path.isdir(cache_dir):
            logger.warning(f"  Directory not found, skipping")
            continue

        build_cache_for_split(cache_dir, text_gen)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()