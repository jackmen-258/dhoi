#!/usr/bin/env python3
"""
diagnose_affordpose_mapping.py
==============================

Use AffordPose GT hand meshes to diagnose raw-part -> dhoi-slot mappings.

Outputs:
  - per-category dominant-slot accuracy / target-contact ratio
  - worst objects ranked by target-contact ratio
  - per-label GT contact statistics

Example:
  python preprocess/diagnose_affordpose_mapping.py \
    --manifest configs/affordpose_test_manifest.json \
    --mapping configs/affordpose_part_to_slot.json \
    --output configs/affordpose_mapping_diagnostics.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


TARGET_SLOT_BY_AFFORDANCE = {
    ("bottle", "Twist"): 3,
    ("bottle", "Wrap-grasp"): 0,
    ("dispenser", "Press"): 2,
    ("dispenser", "Twist"): 3,
    ("dispenser", "Wrap-grasp"): 0,
    ("earphone", "Lift"): 0,
    ("knife", "Handle-grasp"): 0,
    ("mug", "Handle-grasp"): 0,
    ("mug", "Support"): 0,
    ("mug", "Wrap-grasp"): 0,
    ("scissors", "Handle-grasp"): 0,
}


def parse_obj_with_labels(lines):
    verts = []
    labels = []
    for line in lines:
        if not line.startswith("v "):
            continue
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if len(parts) >= 5:
            labels.append(int(float(parts[4])))
        else:
            labels.append(-1)
    return np.asarray(verts, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def parse_obj(lines):
    verts = []
    for line in lines:
        if not line.startswith("v "):
            continue
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(verts, dtype=np.float32)


def normalize_pair(obj_verts, hand_verts):
    bbox_center = (obj_verts.min(0) + obj_verts.max(0)) / 2.0
    return obj_verts - bbox_center, hand_verts - bbox_center


def compute_contact_mask(obj_verts, hand_verts, primary_thresh=0.005, fallback_thresh=0.008):
    hand_tree = cKDTree(hand_verts)
    dist, _ = hand_tree.query(obj_verts, k=1)
    contact = dist <= primary_thresh
    thresh = primary_thresh
    if contact.sum() == 0 and fallback_thresh > primary_thresh:
        contact = dist <= fallback_thresh
        thresh = fallback_thresh
    return contact, dist, thresh


def build_mapping_lookup(mapping_json):
    lookup = {}
    for cls, cls_info in mapping_json["categories"].items():
        raw_map = {}
        for raw_label, info in cls_info.get("raw_part_labels", {}).items():
            raw_map[int(raw_label)] = int(info["slot_id"])
        lookup[cls] = raw_map
    return lookup


def summarize_counter(counter_obj):
    return {str(k): int(v) for k, v in sorted(counter_obj.items(), key=lambda kv: kv[0])}


def main():
    parser = argparse.ArgumentParser("Diagnose AffordPose slot mapping with GT hand contacts")
    parser.add_argument("--manifest", type=str, default="configs/affordpose_test_manifest.json")
    parser.add_argument("--mapping", type=str, default="configs/affordpose_part_to_slot.json")
    parser.add_argument("--output", type=str, default="configs/affordpose_mapping_diagnostics.json")
    parser.add_argument("--contact_thresh", type=float, default=0.005)
    parser.add_argument("--fallback_thresh", type=float, default=0.008)
    parser.add_argument("--top_k", type=int, default=30)
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)["entries"]
    with open(args.mapping, "r", encoding="utf-8") as f:
        mapping_json = json.load(f)

    mapping_lookup = build_mapping_lookup(mapping_json)

    category_stats = defaultdict(lambda: {
        "n_samples": 0,
        "dominant_ok": 0,
        "target_ratio": [],
        "unassigned_ratio": [],
        "dominant_slots": Counter(),
        "target_slots": Counter(),
    })
    object_stats = defaultdict(lambda: {
        "n_samples": 0,
        "class_name": None,
        "dominant_ok": 0,
        "target_ratio": [],
        "unassigned_ratio": [],
        "dominant_slots": Counter(),
        "affordances": Counter(),
    })
    label_stats = defaultdict(Counter)

    for entry in manifest:
        class_name = entry["class_name"]
        afford_name = entry["afford_name"]
        target_slot = TARGET_SLOT_BY_AFFORDANCE.get((class_name, afford_name), None)
        if target_slot is None:
            continue

        with open(entry["sample_json"], "r", encoding="utf-8") as f:
            sample = json.load(f)

        obj_verts, raw_labels = parse_obj_with_labels(sample["object_mesh"])
        hand_verts = parse_obj(sample["rhand_mesh"])
        if len(obj_verts) == 0 or len(hand_verts) == 0:
            continue

        obj_verts, hand_verts = normalize_pair(obj_verts, hand_verts)
        contact_mask, _, _ = compute_contact_mask(
            obj_verts, hand_verts,
            primary_thresh=args.contact_thresh,
            fallback_thresh=args.fallback_thresh,
        )
        if contact_mask.sum() == 0:
            continue

        raw_contact_counts = Counter(raw_labels[contact_mask].tolist())
        slot_counts = Counter()
        unassigned = 0

        for raw_label, count in raw_contact_counts.items():
            slot_id = mapping_lookup.get(class_name, {}).get(int(raw_label), None)
            label_stats[(class_name, int(raw_label))]["contacts"] += int(count)
            label_stats[(class_name, int(raw_label))][afford_name] += int(count)
            if slot_id is None or slot_id < 0:
                unassigned += int(count)
            else:
                slot_counts[int(slot_id)] += int(count)

        n_contact = int(contact_mask.sum())
        dominant_slot = max(slot_counts.items(), key=lambda kv: kv[1])[0] if slot_counts else -1
        target_ratio = float(slot_counts.get(target_slot, 0)) / float(n_contact)
        unassigned_ratio = float(unassigned) / float(n_contact)
        dominant_ok = int(dominant_slot == target_slot)

        cat = category_stats[class_name]
        cat["n_samples"] += 1
        cat["dominant_ok"] += dominant_ok
        cat["target_ratio"].append(target_ratio)
        cat["unassigned_ratio"].append(unassigned_ratio)
        cat["dominant_slots"][dominant_slot] += 1
        cat["target_slots"][target_slot] += 1

        obj_key = f"{class_name}_{entry['obj_id']}"
        obj = object_stats[obj_key]
        obj["n_samples"] += 1
        obj["class_name"] = class_name
        obj["dominant_ok"] += dominant_ok
        obj["target_ratio"].append(target_ratio)
        obj["unassigned_ratio"].append(unassigned_ratio)
        obj["dominant_slots"][dominant_slot] += 1
        obj["affordances"][afford_name] += 1

    category_summary = {}
    for class_name, stats in sorted(category_stats.items()):
        n_samples = max(int(stats["n_samples"]), 1)
        category_summary[class_name] = {
            "n_samples": int(stats["n_samples"]),
            "dominant_part_acc": float(stats["dominant_ok"]) / float(n_samples),
            "mean_target_ratio": float(np.mean(stats["target_ratio"])) if stats["target_ratio"] else 0.0,
            "mean_unassigned_ratio": float(np.mean(stats["unassigned_ratio"])) if stats["unassigned_ratio"] else 0.0,
            "dominant_slots": summarize_counter(stats["dominant_slots"]),
            "target_slots": summarize_counter(stats["target_slots"]),
        }

    worst_objects = []
    for obj_key, stats in object_stats.items():
        n_samples = max(int(stats["n_samples"]), 1)
        worst_objects.append({
            "object_id": obj_key,
            "class_name": stats["class_name"],
            "n_samples": int(stats["n_samples"]),
            "dominant_part_acc": float(stats["dominant_ok"]) / float(n_samples),
            "mean_target_ratio": float(np.mean(stats["target_ratio"])) if stats["target_ratio"] else 0.0,
            "mean_unassigned_ratio": float(np.mean(stats["unassigned_ratio"])) if stats["unassigned_ratio"] else 0.0,
            "affordances": summarize_counter(stats["affordances"]),
            "dominant_slots": summarize_counter(stats["dominant_slots"]),
        })
    worst_objects.sort(key=lambda row: (row["mean_target_ratio"], row["dominant_part_acc"], row["mean_unassigned_ratio"]))

    label_summary = {}
    for (class_name, raw_label), counts in sorted(label_stats.items()):
        label_summary.setdefault(class_name, {})[str(raw_label)] = {
            key: int(val) for key, val in sorted(counts.items(), key=lambda kv: kv[0])
        }

    report = {
        "manifest": str(Path(args.manifest).resolve()),
        "mapping": str(Path(args.mapping).resolve()),
        "contact_thresh": float(args.contact_thresh),
        "fallback_thresh": float(args.fallback_thresh),
        "n_entries": len(manifest),
        "category_summary": category_summary,
        "worst_objects": worst_objects[: int(args.top_k)],
        "label_contact_summary": label_summary,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Saved diagnostic report to: {output_path}")
    print("Category summary:")
    for class_name, stats in category_summary.items():
        print(
            f"  {class_name:10s} "
            f"acc={stats['dominant_part_acc']:.3f} "
            f"target_ratio={stats['mean_target_ratio']:.3f} "
            f"unassigned={stats['mean_unassigned_ratio']:.3f}"
        )
    print("\nWorst objects:")
    for row in report["worst_objects"][:10]:
        print(
            f"  {row['object_id']}: "
            f"target_ratio={row['mean_target_ratio']:.3f}, "
            f"acc={row['dominant_part_acc']:.3f}, "
            f"unassigned={row['mean_unassigned_ratio']:.3f}"
        )


if __name__ == "__main__":
    main()
