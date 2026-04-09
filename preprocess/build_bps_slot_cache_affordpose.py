#!/usr/bin/env python3
"""
build_bps_slot_cache_affordpose.py
==================================

Build AffordPose BPS slot caches from the object meshes embedded in manifest
JSON files and the manually prepared raw-part-label -> dhoi-slot mapping.

Outputs one cache file per unique object (or per manifest key, depending on
cache_id mode), in the same format used by the OakInk BPS slot cache:

    {save_dir}/{split}/{cache_id}.npz
        - bps_dists:       (4096,) float32
        - bps_slot_labels: (4096,) int8
        - bps_nn_points:   (4096, 3) float32
        - b2x_idxs:        (4096,) int32
        - obj_scale:       float32
        - obj_center:      (3,) float32

Additional metadata files:
    {save_dir}/{split}/index.json
    {save_dir}/{split}/build_stats.json

Example:
    python preprocess/build_bps_slot_cache_affordpose.py \
        --manifest configs/affordpose_test_manifest.json \
        --mapping configs/affordpose_part_to_slot.json \
        --bps_path configs/bps.npz \
        --save_dir cache/bps_slot_affordpose \
        --split test
"""

import os
import sys
import json
import argparse
import logging
import time
from collections import defaultdict

import numpy as np
import trimesh
from tqdm import tqdm


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.bps_slot_encoder import normalize_object, compute_vertex_to_slot
from preprocess.build_bps_slot_cache import sample_object_points, encode_bps_slots


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_obj_lines(lines, expect_label=False):
    verts = []
    faces = []
    labels = []

    for line in lines:
        if not line:
            continue
        if line.startswith("v "):
            parts = line.strip().split()
            if expect_label:
                if len(parts) < 5:
                    raise ValueError(f"Expected labeled vertex line, got: {line!r}")
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                labels.append(int(parts[4]))
            else:
                if len(parts) < 4:
                    raise ValueError(f"Expected vertex line, got: {line!r}")
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("f "):
            items = line.strip().split()[1:]
            idxs = []
            for item in items:
                raw = item.split("/")[0]
                if not raw:
                    continue
                idxs.append(int(raw) - 1)
            if len(idxs) < 3:
                continue
            for i in range(1, len(idxs) - 1):
                faces.append([idxs[0], idxs[i], idxs[i + 1]])

    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if expect_label:
        labels = np.asarray(labels, dtype=np.int32)
        if len(labels) != len(verts):
            raise ValueError("Number of raw labels does not match vertex count")
        return verts, faces, labels
    return verts, faces


def reverse_slot_names(canonical_slots):
    out = {}
    for key, value in canonical_slots.items():
        try:
            out[value] = int(key)
        except ValueError:
            continue
    return out


def build_slot_mapping(mapping_cfg, class_name, use_proposed=False):
    canonical_slots = mapping_cfg["canonical_slots"]
    slot_name_to_id = reverse_slot_names(canonical_slots)
    class_cfg = mapping_cfg["categories"].get(class_name, {})
    raw_cfg = class_cfg.get("raw_part_labels", {})

    slot_by_raw = {}
    status_by_raw = {}
    for raw_label, info in raw_cfg.items():
        slot_name = info.get("slot_name")
        if slot_name is None and use_proposed:
            slot_name = info.get("proposed_slot_name")

        slot_id = info.get("slot_id")
        if slot_id is None and slot_name is not None:
            slot_id = slot_name_to_id.get(slot_name)
        if slot_id is None:
            slot_id = -1

        raw_int = int(raw_label)
        slot_by_raw[raw_int] = int(slot_id)
        status_by_raw[raw_int] = info.get("status", "todo")
    return slot_by_raw, status_by_raw


def apply_raw_to_slot_mapping(raw_labels, slot_by_raw):
    raw_labels = np.asarray(raw_labels, dtype=np.int32)
    slot_labels = np.full(raw_labels.shape, -1, dtype=np.int32)
    missing_raw = sorted(set(int(x) for x in raw_labels.tolist()) - set(slot_by_raw.keys()))
    for raw_label, slot_id in slot_by_raw.items():
        slot_labels[raw_labels == int(raw_label)] = int(slot_id)
    return slot_labels, missing_raw


def load_affordpose_object(sample_json):
    data = load_json(sample_json)
    verts, faces, raw_labels = parse_obj_lines(data["object_mesh"], expect_label=True)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return mesh, raw_labels, data


def normalize_filter_values(values):
    if not values:
        return None
    out = []
    for value in values:
        if value is None:
            continue
        for token in str(value).split(","):
            token = token.strip()
            if token:
                out.append(token)
    if not out:
        return None
    return list(dict.fromkeys(out))


def select_manifest_entries(manifest, keys=None, classes=None, obj_ids=None, limit=None):
    entries = manifest["entries"]

    keys = normalize_filter_values(keys)
    classes = normalize_filter_values(classes)
    obj_ids = normalize_filter_values(obj_ids)

    if keys:
        key_set = set(keys)
        entries = [e for e in entries if e["key"] in key_set]

    if classes:
        class_set = set(classes)
        entries = [e for e in entries if e["class_name"] in class_set]

    if obj_ids:
        obj_set = set(obj_ids)
        entries = [e for e in entries if str(e["obj_id"]) in obj_set]

    if limit is not None:
        entries = entries[: int(limit)]

    return entries


def build_cache_id(entry, mode):
    if mode == "obj_id":
        return str(entry["obj_id"])
    if mode == "class_obj_id":
        return f"{entry['class_name']}_{entry['obj_id']}"
    if mode == "manifest_key":
        return str(entry["key"])
    raise ValueError(f"Unsupported cache_id mode: {mode}")


def group_entries(entries, cache_id_mode):
    grouped = defaultdict(list)
    for entry in entries:
        grouped[build_cache_id(entry, cache_id_mode)].append(entry)
    return grouped


def unique_sorted_int(values):
    return sorted(set(int(v) for v in values))


def count_by_value(arr):
    vals, cnt = np.unique(np.asarray(arr), return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(vals.tolist(), cnt.tolist())}


def main():
    parser = argparse.ArgumentParser(
        description="Build AffordPose BPS slot cache from embedded labeled object meshes"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="configs/affordpose_test_manifest.json",
        help="AffordPose manifest JSON generated earlier",
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default="configs/affordpose_part_to_slot.json",
        help="Raw-part-label -> dhoi slot mapping JSON",
    )
    parser.add_argument("--bps_path", type=str, default="configs/bps.npz")
    parser.add_argument("--save_dir", type=str, default="cache/bps_slot_affordpose")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--cache_id_mode",
        type=str,
        default="class_obj_id",
        choices=["class_obj_id", "obj_id", "manifest_key"],
        help="How output cache files are named",
    )
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--dist_threshold", type=float, default=0.15)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--use_proposed", action="store_true",
                        help="Fallback to proposed_slot_name when slot_name is missing")
    parser.add_argument("--keys", nargs="+", default=None)
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--obj_ids", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    assert os.path.exists(args.manifest), f"Manifest not found: {args.manifest}"
    assert os.path.exists(args.mapping), f"Mapping not found: {args.mapping}"
    assert os.path.exists(args.bps_path), f"BPS basis not found: {args.bps_path}"

    manifest = load_json(args.manifest)
    mapping_cfg = load_json(args.mapping)

    selected_entries = select_manifest_entries(
        manifest,
        keys=args.keys,
        classes=args.classes,
        obj_ids=args.obj_ids,
        limit=args.limit,
    )
    if not selected_entries:
        raise ValueError("No AffordPose manifest entries matched the requested filters")

    grouped = group_entries(selected_entries, args.cache_id_mode)
    if args.cache_id_mode == "obj_id":
        cache_to_classes = defaultdict(set)
        for cache_id, entries in grouped.items():
            for entry in entries:
                cache_to_classes[cache_id].add(entry["class_name"])
        collisions = {k: v for k, v in cache_to_classes.items() if len(v) > 1}
        if collisions:
            preview = {k: sorted(v) for k, v in list(collisions.items())[:5]}
            raise ValueError(
                "cache_id_mode=obj_id causes cross-category collisions. "
                f"Examples: {preview}. Use --cache_id_mode class_obj_id instead."
            )

    bps_data = np.load(args.bps_path)
    bps_basis = bps_data["basis"].astype(np.float32)
    if bps_basis.ndim == 3:
        bps_basis = bps_basis.squeeze(0)

    out_dir = os.path.join(args.save_dir, args.split)
    os.makedirs(out_dir, exist_ok=True)

    stats = {
        "split": args.split,
        "cache_id_mode": args.cache_id_mode,
        "selected_manifest_entries": len(selected_entries),
        "selected_cache_objects": len(grouped),
        "success": 0,
        "skipped": 0,
        "failed": 0,
        "objects_with_missing_raw_labels": 0,
        "objects_all_unassigned_mesh": 0,
        "objects_all_unassigned_sampled": 0,
        "objects_all_unassigned_bps": 0,
    }

    manifest_to_cache_id = {}
    index_objects = []

    logger.info("=" * 60)
    logger.info(f"AffordPose manifest entries selected: {len(selected_entries)}")
    logger.info(f"Unique cache objects:             {len(grouped)}")
    logger.info(f"Cache id mode:                   {args.cache_id_mode}")
    logger.info(f"Output dir:                      {out_dir}")
    logger.info("=" * 60)

    t0 = time.time()

    for cache_id, entries in tqdm(grouped.items(), desc=f"BPS slot [{args.split}]"):
        out_path = os.path.join(out_dir, f"{cache_id}.npz")
        for entry in entries:
            manifest_to_cache_id[entry["key"]] = cache_id

        if args.skip_existing and os.path.exists(out_path):
            stats["skipped"] += 1
            index_objects.append({
                "cache_id": cache_id,
                "status": "skipped_existing",
                "class_name": entries[0]["class_name"],
                "obj_id": str(entries[0]["obj_id"]),
                "manifest_keys": [e["key"] for e in entries],
                "sample_json": entries[0]["sample_json"],
            })
            continue

        entry0 = entries[0]
        class_name = entry0["class_name"]
        obj_id = str(entry0["obj_id"])
        sample_json = entry0["sample_json"]

        try:
            obj_mesh, raw_labels, _ = load_affordpose_object(sample_json)
            slot_by_raw, status_by_raw = build_slot_mapping(
                mapping_cfg, class_name, use_proposed=args.use_proposed
            )
            mesh_slot_labels, missing_raw = apply_raw_to_slot_mapping(raw_labels, slot_by_raw)

            if missing_raw:
                stats["objects_with_missing_raw_labels"] += 1
                logger.warning(
                    f"[missing mapping] cache_id={cache_id} class={class_name} "
                    f"raw_labels={missing_raw}"
                )

            if np.all(mesh_slot_labels < 0):
                stats["objects_all_unassigned_mesh"] += 1
                logger.warning(
                    f"[all -1 mesh] cache_id={cache_id} class={class_name} "
                    f"raw_counts={count_by_value(raw_labels)}"
                )

            sampled_pts = sample_object_points(
                obj_mesh, n_samples=args.n_samples, seed_str=cache_id
            )
            sampled_slot_labels = compute_vertex_to_slot(
                sampled_pts,
                np.asarray(obj_mesh.vertices, dtype=np.float32),
                mesh_slot_labels,
            )
            if np.all(sampled_slot_labels < 0):
                stats["objects_all_unassigned_sampled"] += 1
                logger.warning(
                    f"[all -1 sampled] cache_id={cache_id} class={class_name} "
                    f"mesh_slot_counts={count_by_value(mesh_slot_labels)}"
                )

            pts_norm, center, scale = normalize_object(sampled_pts)
            result = encode_bps_slots(
                bps_basis=bps_basis,
                obj_points_norm=pts_norm,
                vertex_to_slot=sampled_slot_labels,
                dist_threshold=args.dist_threshold,
            )
            if np.all(result["bps_slot_labels"] < 0):
                stats["objects_all_unassigned_bps"] += 1
                logger.warning(
                    f"[all -1 bps] cache_id={cache_id} class={class_name} "
                    f"sampled_slot_counts={count_by_value(sampled_slot_labels)} "
                    f"dist_range=({result['bps_dists'].min():.4f}, {result['bps_dists'].max():.4f}) "
                    f"threshold={args.dist_threshold}"
                )

            np.savez_compressed(
                out_path,
                bps_dists=result["bps_dists"],
                bps_slot_labels=result["bps_slot_labels"],
                bps_nn_points=result["bps_nn_points"],
                b2x_idxs=result["b2x_idxs"],
                obj_scale=scale,
                obj_center=center,
                cache_id=np.asarray(cache_id),
                class_name=np.asarray(class_name),
                obj_id=np.asarray(obj_id),
            )

            index_objects.append({
                "cache_id": cache_id,
                "status": "success",
                "class_name": class_name,
                "obj_id": obj_id,
                "manifest_keys": [e["key"] for e in entries],
                "sample_json": sample_json,
                "num_manifest_entries": len(entries),
                "raw_part_labels": unique_sorted_int(raw_labels.tolist()),
                "missing_raw_labels": missing_raw,
                "raw_label_counts": count_by_value(raw_labels),
                "mesh_slot_counts": count_by_value(mesh_slot_labels),
                "sampled_slot_counts": count_by_value(sampled_slot_labels),
                "bps_slot_counts": count_by_value(result["bps_slot_labels"]),
                "mapping_status_by_raw": {
                    str(raw): status_by_raw.get(int(raw), "missing")
                    for raw in unique_sorted_int(raw_labels.tolist())
                },
            })
            stats["success"] += 1

        except Exception as e:
            logger.warning(f"Failed cache_id={cache_id}: {type(e).__name__}: {e}")
            index_objects.append({
                "cache_id": cache_id,
                "status": "failed",
                "class_name": class_name,
                "obj_id": obj_id,
                "manifest_keys": [e["key"] for e in entries],
                "sample_json": sample_json,
                "error": f"{type(e).__name__}: {e}",
            })
            stats["failed"] += 1

    elapsed = time.time() - t0
    stats["elapsed_seconds"] = elapsed
    stats["n_bps"] = int(bps_basis.shape[0])
    stats["n_samples"] = int(args.n_samples)
    stats["dist_threshold"] = float(args.dist_threshold)

    index = {
        "schema_version": 1,
        "dataset_name": "AffordPose",
        "split": args.split,
        "manifest_path": os.path.abspath(args.manifest),
        "mapping_path": os.path.abspath(args.mapping),
        "bps_path": os.path.abspath(args.bps_path),
        "cache_id_mode": args.cache_id_mode,
        "n_bps": int(bps_basis.shape[0]),
        "n_samples": int(args.n_samples),
        "dist_threshold": float(args.dist_threshold),
        "selected_manifest_entries": len(selected_entries),
        "selected_cache_objects": len(grouped),
        "manifest_key_to_cache_id": manifest_to_cache_id,
        "objects": index_objects,
    }

    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    with open(os.path.join(out_dir, "build_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info("\nDone!")
    logger.info(f"  success={stats['success']}  skipped={stats['skipped']}  failed={stats['failed']}")
    logger.info(f"  elapsed={elapsed:.1f}s")
    logger.info(f"  index={os.path.join(out_dir, 'index.json')}")


if __name__ == "__main__":
    main()
