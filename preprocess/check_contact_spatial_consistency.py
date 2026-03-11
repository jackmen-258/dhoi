#!/usr/bin/env python3
"""
粗略检查接触矩阵(contact_type_matrix)与手部姿态(hand pose)的一致性。

思路：
1) 将 contact_type_matrix 解码为 (hand_part, slot, contact_type) 三元组；
2) 用 MANO 参数重建 hand vertices；
3) 加载该物体的 part vertices (按 slot 归类)；
4) 对每个 active 三元组，计算 hand_part 顶点到该 slot 顶点的最近距离；
5) 若距离明显偏大，且明显劣于同一 hand_part 到其他 slot 的距离，则标为可疑。

这是“粗略几何 sanity check”，用于发现系统性偏差，不是严格物理判定。
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from manotorch.manolayer import ManoLayer

from data.contact_builder import ContactBuilder
from data.slot_mapping import SlotMapper
from utils.mesh_utils import build_hand_part_assignment_from_weights


@dataclass
class PairCheck:
    hand_part: str
    slot: str
    contact_type: str
    dist_to_expected: float
    best_slot: str
    best_dist: float
    suspicious: bool


def min_dist_points_to_points(src: np.ndarray, dst: np.ndarray, chunk: int = 256) -> float:
    """返回 src 到 dst 的最小欧氏距离（单位: 米，取决于数据坐标系）。"""
    if len(src) == 0 or len(dst) == 0:
        return float("inf")

    best = float("inf")
    for i in range(0, len(src), chunk):
        a = src[i : i + chunk]  # (Ca,3)
        # (Ca,1,3) - (1,Cb,3) -> (Ca,Cb,3)
        d2 = np.sum((a[:, None, :] - dst[None, :, :]) ** 2, axis=-1)
        cur = float(np.sqrt(np.min(d2)))
        if cur < best:
            best = cur
    return best


def decode_active_triples(type_matrix: np.ndarray, mapper: SlotMapper) -> List[Tuple[int, int, int, str, str, str]]:
    triples = []
    H, S = type_matrix.shape
    for h in range(H):
        for s in range(S):
            c = int(type_matrix[h, s])
            if c < 0:
                continue
            triples.append((h, s, c, mapper.hand_parts[h], mapper.canonical_slots[s], mapper.contact_types[c]))
    return triples


def rebuild_hand_verts(npz_data: np.lib.npyio.NpzFile, mano_layer: ManoLayer) -> np.ndarray:
    pose = torch.from_numpy(np.asarray(npz_data["hand_pose"], dtype=np.float32)).unsqueeze(0)
    shape = torch.from_numpy(np.asarray(npz_data["hand_shape"], dtype=np.float32)).unsqueeze(0)
    tsl = torch.from_numpy(np.asarray(npz_data["hand_tsl"], dtype=np.float32)).unsqueeze(0)

    out = mano_layer(pose, shape)
    verts = out.verts + tsl.unsqueeze(1)
    return verts[0].detach().cpu().numpy().astype(np.float32)


def check_one_sample(
    npz_path: str,
    mapper: SlotMapper,
    builder: ContactBuilder,
    mano_layer: ManoLayer,
    vertex_to_hand_part: np.ndarray,
    abs_dist_thr: float,
    rel_margin: float,
) -> Tuple[Optional[List[PairCheck]], Optional[str]]:
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        return None, f"load_failed: {e}"

    needed = ["contact_type_matrix", "obj_id", "cate_id", "hand_pose", "hand_shape", "hand_tsl"]
    for k in needed:
        if k not in data:
            return None, f"missing_key:{k}"

    obj_id = str(data["obj_id"])
    cate_id = str(data["cate_id"])
    raw_obj_id = str(data["raw_obj_id"]) if "raw_obj_id" in data else obj_id

    obj_parts = builder.load_object_parts_for_contact(obj_id=obj_id, cate_id=cate_id, raw_obj_id=raw_obj_id)
    if obj_parts is None:
        return None, "no_object_parts"

    _, obj_verts, vertex_to_slot, _ = obj_parts

    hand_verts = rebuild_hand_verts(data, mano_layer)
    type_matrix = np.asarray(data["contact_type_matrix"], dtype=np.int32)

    part_to_hand_verts: Dict[int, np.ndarray] = {}
    for h_id in range(len(mapper.hand_parts)):
        idx = np.where(vertex_to_hand_part == h_id)[0]
        part_to_hand_verts[h_id] = hand_verts[idx]

    slot_to_obj_verts: Dict[int, np.ndarray] = {}
    for s_id in range(len(mapper.canonical_slots)):
        idx = np.where(vertex_to_slot == s_id)[0]
        slot_to_obj_verts[s_id] = obj_verts[idx]

    checks: List[PairCheck] = []
    for h_id, s_id, c_id, h_name, s_name, c_name in decode_active_triples(type_matrix, mapper):
        src = part_to_hand_verts[h_id]
        dst = slot_to_obj_verts[s_id]
        d_expected = min_dist_points_to_points(src, dst)

        all_slot_dists = []
        for sid in range(len(mapper.canonical_slots)):
            d = min_dist_points_to_points(src, slot_to_obj_verts[sid])
            all_slot_dists.append((sid, d))
        best_sid, best_d = sorted(all_slot_dists, key=lambda x: x[1])[0]

        suspicious = (d_expected > abs_dist_thr) and (d_expected > best_d + rel_margin)
        checks.append(
            PairCheck(
                hand_part=h_name,
                slot=s_name,
                contact_type=c_name,
                dist_to_expected=d_expected,
                best_slot=mapper.canonical_slots[best_sid],
                best_dist=best_d,
                suspicious=suspicious,
            )
        )

    return checks, None


def main() -> None:
    ap = argparse.ArgumentParser(description="检查 contact_type_matrix 与手部空间对应的一致性")
    ap.add_argument("--cache_dir", type=str, required=True, help="如 cache/contact_tokens/train")
    ap.add_argument("--config", type=str, default="configs/token_config.yaml")
    ap.add_argument("--oakink_root", type=str, default=None, help="可选，不填则使用环境变量 OAKINK_DIR")
    ap.add_argument("--max_samples", type=int, default=300)
    ap.add_argument("--abs_dist_thr", type=float, default=0.02, help="绝对距离阈值(米)")
    ap.add_argument("--rel_margin", type=float, default=0.005, help="相对最优slot的距离劣势阈值(米)")
    ap.add_argument("--show_top", type=int, default=15, help="输出最可疑的样本数")
    args = ap.parse_args()

    npz_files = sorted(glob.glob(os.path.join(args.cache_dir, "*.npz")))
    npz_files = [p for p in npz_files if not os.path.basename(p).startswith("token_sequences")]
    if not npz_files:
        raise RuntimeError(f"未找到样本 npz: {args.cache_dir}")

    if args.max_samples > 0:
        npz_files = npz_files[: args.max_samples]

    mapper = SlotMapper(args.config)
    builder = ContactBuilder(config_path=args.config, oakink_root=args.oakink_root)

    mano_layer = ManoLayer(center_idx=0, mano_assets_root="assets/mano_v1_2")
    with torch.no_grad():
        weights = mano_layer.th_weights.detach().cpu().numpy()
    joint_to_part = {int(k): mapper.hand_part_to_id[v] for k, v in mapper.joint_to_part.items()}
    vertex_to_hand_part = build_hand_part_assignment_from_weights(weights, joint_to_part)

    total_pairs = 0
    suspicious_pairs = 0
    sample_scores = []
    skip_reasons: Dict[str, int] = {}

    for p in npz_files:
        checks, err = check_one_sample(
            npz_path=p,
            mapper=mapper,
            builder=builder,
            mano_layer=mano_layer,
            vertex_to_hand_part=vertex_to_hand_part,
            abs_dist_thr=args.abs_dist_thr,
            rel_margin=args.rel_margin,
        )
        if err is not None:
            skip_reasons[err] = skip_reasons.get(err, 0) + 1
            continue
        assert checks is not None

        n_bad = sum(1 for c in checks if c.suspicious)
        total_pairs += len(checks)
        suspicious_pairs += n_bad

        score = (n_bad / max(len(checks), 1), n_bad)
        sample_scores.append((score, p, checks))

    print("=" * 80)
    print("Contact/Hand Spatial Consistency Report")
    print("=" * 80)
    print(f"Checked samples: {len(sample_scores)} / {len(npz_files)}")
    print(f"Total active triples: {total_pairs}")
    ratio = suspicious_pairs / max(total_pairs, 1)
    print(f"Suspicious triples: {suspicious_pairs} ({ratio:.2%})")
    print(f"Thresholds: abs_dist_thr={args.abs_dist_thr:.4f}m, rel_margin={args.rel_margin:.4f}m")

    if skip_reasons:
        print("\nSkipped samples:")
        for k, v in sorted(skip_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {k}: {v}")

    sample_scores.sort(key=lambda x: x[0], reverse=True)
    topk = sample_scores[: args.show_top]

    print("\nTop suspicious samples:")
    if not topk:
        print("  (none)")
        return

    for rank, (score, path, checks) in enumerate(topk, start=1):
        ratio_bad, n_bad = score
        print(f"\n[{rank:02d}] {os.path.basename(path)}  suspicious={n_bad}/{len(checks)} ({ratio_bad:.1%})")
        bad_checks = [c for c in checks if c.suspicious]
        bad_checks.sort(key=lambda x: (x.dist_to_expected - x.best_dist), reverse=True)
        for c in bad_checks[:8]:
            print(
                "    - "
                f"({c.hand_part}, {c.slot}, {c.contact_type}) "
                f"dist={c.dist_to_expected:.4f}m, "
                f"best_slot={c.best_slot}({c.best_dist:.4f}m)"
            )


if __name__ == "__main__":
    main()
