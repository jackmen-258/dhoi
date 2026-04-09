#!/usr/bin/env python3
"""
visualize_affordpose_labels.py
==============================

Visualize AffordPose object raw part labels and their tentative mapping to
dhoi canonical slots.

Outputs per selected entry:
  - object_raw_labels.ply   : object mesh colored by AffordPose raw part label
  - object_slot_labels.ply  : object mesh colored by mapped dhoi slot
  - hand_gt.ply             : GT hand mesh from the same json
  - preview.png             : quick multi-view scatter preview
  - summary.json            : raw-label counts and current mapping status

Examples:
  python preprocess/visualize_affordpose_labels.py
  python preprocess/visualize_affordpose_labels.py --classes bottle mug --samples_per_class 3
  python preprocess/visualize_affordpose_labels.py --keys bottle_3418_Twist mug_8554_Handle-grasp
  python preprocess/visualize_affordpose_labels.py --use_proposed --limit 8
"""

import os
import sys
import json
import argparse
from collections import Counter

import numpy as np
import trimesh

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


RAW_PALETTE = np.array([
    [31, 119, 180, 255],
    [255, 127, 14, 255],
    [44, 160, 44, 255],
    [214, 39, 40, 255],
    [148, 103, 189, 255],
    [140, 86, 75, 255],
    [227, 119, 194, 255],
    [127, 127, 127, 255],
    [188, 189, 34, 255],
    [23, 190, 207, 255],
], dtype=np.uint8)

SLOT_COLORS = {
    -1: np.array([180, 180, 180, 255], dtype=np.uint8),  # unresolved / unassigned
    0: np.array([52, 152, 219, 255], dtype=np.uint8),    # GRASP_SURFACE
    1: np.array([230, 126, 34, 255], dtype=np.uint8),    # FUNCTIONAL_END
    2: np.array([46, 204, 113, 255], dtype=np.uint8),    # CONTROL
    3: np.array([231, 76, 60, 255], dtype=np.uint8),     # CLOSURE
}


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
            parts = line.strip().split()[1:]
            face = []
            for item in parts[:3]:
                face.append(int(item.split("/")[0]) - 1)
            if len(face) == 3:
                faces.append(face)

    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if expect_label:
        labels = np.asarray(labels, dtype=np.int32)
        return verts, faces, labels
    return verts, faces


def build_raw_color_map(raw_labels):
    unique = sorted(set(int(x) for x in raw_labels))
    return {
        lab: RAW_PALETTE[i % len(RAW_PALETTE)]
        for i, lab in enumerate(unique)
    }


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
    name_by_raw = {}
    note_by_raw = {}

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
        name_by_raw[raw_int] = slot_name
        note_by_raw[raw_int] = info.get("notes", "")

    return slot_by_raw, status_by_raw, name_by_raw, note_by_raw


def make_colored_mesh(vertices, faces, colors):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual.vertex_colors = np.asarray(colors, dtype=np.uint8)
    return mesh


def colors_from_raw_labels(raw_labels):
    raw_color_map = build_raw_color_map(raw_labels.tolist())
    colors = np.stack([raw_color_map[int(lab)] for lab in raw_labels], axis=0)
    return colors, raw_color_map


def colors_from_slot_labels(slot_labels):
    colors = np.stack([SLOT_COLORS.get(int(lab), SLOT_COLORS[-1]) for lab in slot_labels], axis=0)
    return colors


def set_equal_axes(ax, points):
    center = points.mean(axis=0)
    extent = points.max(axis=0) - points.min(axis=0)
    radius = max(float(extent.max()) / 2.0, 1e-3) * 1.1
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("X", fontsize=7)
    ax.set_ylabel("Y", fontsize=7)
    ax.set_zlabel("Z", fontsize=7)
    ax.tick_params(labelsize=6)


def preview_scatter(ax, obj_verts, obj_colors, hand_verts, elev, azim, title):
    obj_step = max(1, len(obj_verts) // 1800)
    hand_step = max(1, len(hand_verts) // 500)
    ax.scatter(
        obj_verts[::obj_step, 0],
        obj_verts[::obj_step, 1],
        obj_verts[::obj_step, 2],
        c=obj_colors[::obj_step] / 255.0,
        s=2,
        alpha=0.9,
    )
    ax.scatter(
        hand_verts[::hand_step, 0],
        hand_verts[::hand_step, 1],
        hand_verts[::hand_step, 2],
        c=np.array([[0.1, 0.1, 0.1]]),
        s=1.5,
        alpha=0.25,
    )
    ax.view_init(elev=elev, azim=azim)
    set_equal_axes(ax, np.concatenate([obj_verts, hand_verts], axis=0))
    ax.set_title(title, fontsize=8)


def save_preview(
    save_path,
    key,
    afford_name,
    obj_verts,
    hand_verts,
    raw_colors,
    slot_colors,
    raw_counts,
    slot_summary_lines,
):
    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(221, projection="3d")
    ax2 = fig.add_subplot(222, projection="3d")
    ax3 = fig.add_subplot(223, projection="3d")
    ax4 = fig.add_subplot(224, projection="3d")

    preview_scatter(ax1, obj_verts, raw_colors, hand_verts, elev=25, azim=45, title="Raw Labels (45°)")
    preview_scatter(ax2, obj_verts, raw_colors, hand_verts, elev=25, azim=135, title="Raw Labels (135°)")
    preview_scatter(ax3, obj_verts, slot_colors, hand_verts, elev=25, azim=45, title="Mapped Slots (45°)")
    preview_scatter(ax4, obj_verts, slot_colors, hand_verts, elev=80, azim=45, title="Mapped Slots (Top)")

    raw_desc = ", ".join([f"{lab}:{cnt}" for lab, cnt in sorted(raw_counts.items())])
    mapping_desc = " | ".join(slot_summary_lines) if slot_summary_lines else "No slot mapping found"
    fig.suptitle(
        f"{key}  |  affordance={afford_name}\n"
        f"raw label counts: {raw_desc}\n"
        f"slot mapping: {mapping_desc}",
        fontsize=9,
        family="monospace",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def select_entries(manifest, keys=None, classes=None, limit=None, samples_per_class=2):
    entries = manifest["entries"]
    if keys:
        key_set = set(keys)
        return [entry for entry in entries if entry["key"] in key_set]

    if classes:
        cls_set = set(classes)
        selected = []
        per_class = Counter()
        for entry in entries:
            cls = entry["class_name"]
            if cls not in cls_set:
                continue
            if per_class[cls] >= samples_per_class:
                continue
            selected.append(entry)
            per_class[cls] += 1
        return selected[:limit] if limit else selected

    selected = []
    per_class = Counter()
    for entry in entries:
        cls = entry["class_name"]
        if per_class[cls] >= samples_per_class:
            continue
        selected.append(entry)
        per_class[cls] += 1

    if limit is not None:
        return selected[:limit]
    return selected


def summarize_mapping(raw_counts, slot_by_raw, slot_name_by_raw):
    lines = []
    for raw_label, count in sorted(raw_counts.items()):
        slot_id = slot_by_raw.get(raw_label, -1)
        slot_name = slot_name_by_raw.get(raw_label)
        if slot_name is None:
            slot_name = "UNRESOLVED"
        lines.append(f"{raw_label}->{slot_name}[{slot_id}] x{count}")
    return lines


def export_entry(entry, mapping_cfg, save_root, use_proposed=False, save_preview_flag=True, save_meshes=True):
    data = load_json(entry["sample_json"])
    obj_verts, obj_faces, raw_labels = parse_obj_lines(data["object_mesh"], expect_label=True)
    hand_verts, hand_faces = parse_obj_lines(data["rhand_mesh"], expect_label=False)

    raw_counts = Counter(int(x) for x in raw_labels.tolist())
    slot_by_raw, status_by_raw, slot_name_by_raw, note_by_raw = build_slot_mapping(
        mapping_cfg, entry["class_name"], use_proposed=use_proposed
    )
    slot_labels = np.asarray([slot_by_raw.get(int(lab), -1) for lab in raw_labels], dtype=np.int32)

    raw_colors, raw_color_map = colors_from_raw_labels(raw_labels)
    slot_colors = colors_from_slot_labels(slot_labels)

    entry_dir = os.path.join(save_root, entry["key"])
    os.makedirs(entry_dir, exist_ok=True)

    if save_meshes:
        raw_mesh = make_colored_mesh(obj_verts, obj_faces, raw_colors)
        raw_mesh.export(os.path.join(entry_dir, "object_raw_labels.ply"))

        slot_mesh = make_colored_mesh(obj_verts, obj_faces, slot_colors)
        slot_mesh.export(os.path.join(entry_dir, "object_slot_labels.ply"))

        hand_mesh = trimesh.Trimesh(vertices=hand_verts, faces=hand_faces, process=False)
        hand_mesh.export(os.path.join(entry_dir, "hand_gt.ply"))

    slot_summary_lines = summarize_mapping(raw_counts, slot_by_raw, slot_name_by_raw)
    if save_preview_flag:
        save_preview(
            os.path.join(entry_dir, "preview.png"),
            key=entry["key"],
            afford_name=entry["afford_name"],
            obj_verts=obj_verts,
            hand_verts=hand_verts,
            raw_colors=raw_colors,
            slot_colors=slot_colors,
            raw_counts=raw_counts,
            slot_summary_lines=slot_summary_lines,
        )

    summary = {
        "key": entry["key"],
        "class_name": entry["class_name"],
        "obj_id": entry["obj_id"],
        "afford_name": entry["afford_name"],
        "sample_json": entry["sample_json"],
        "use_proposed": bool(use_proposed),
        "raw_label_counts": {str(k): int(v) for k, v in sorted(raw_counts.items())},
        "raw_label_color_rgba": {str(k): raw_color_map[int(k)].tolist() for k in raw_counts},
        "raw_to_slot": {
            str(raw_label): {
                "slot_id": int(slot_by_raw.get(raw_label, -1)),
                "slot_name": slot_name_by_raw.get(raw_label),
                "status": status_by_raw.get(raw_label),
                "notes": note_by_raw.get(raw_label, ""),
            }
            for raw_label in sorted(raw_counts)
        },
    }
    with open(os.path.join(entry_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return entry_dir


def parse_args():
    parser = argparse.ArgumentParser("Visualize AffordPose raw labels and slot mapping")
    parser.add_argument(
        "--manifest",
        type=str,
        default=os.path.join(PROJECT_ROOT, "configs", "affordpose_test_manifest.json"),
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default=os.path.join(PROJECT_ROOT, "configs", "affordpose_part_to_slot.json"),
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "vis_affordpose_labels"),
    )
    parser.add_argument("--keys", nargs="+", default=None, help="Specific manifest keys to visualize")
    parser.add_argument("--classes", nargs="+", default=None, help="Filter by class_name")
    parser.add_argument("--limit", type=int, default=None, help="Max number of entries to export")
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=2,
        help="Used when --keys is not set; picks the first N entries per class",
    )
    parser.add_argument(
        "--use_proposed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use proposed_slot_name from affordpose_part_to_slot.json when slot_name is still empty",
    )
    parser.add_argument(
        "--save_preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save preview PNGs",
    )
    parser.add_argument(
        "--save_meshes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save colored PLY meshes",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    manifest = load_json(args.manifest)
    mapping_cfg = load_json(args.mapping)
    entries = select_entries(
        manifest,
        keys=args.keys,
        classes=args.classes,
        limit=args.limit,
        samples_per_class=args.samples_per_class,
    )

    print(f"Loaded manifest: {args.manifest}")
    print(f"Loaded mapping:  {args.mapping}")
    print(f"Selected {len(entries)} entries")
    print(f"Output dir:      {args.save_dir}")

    if not entries:
        print("No entries selected. Nothing to do.")
        return

    exported = []
    for entry in entries:
        out_dir = export_entry(
            entry,
            mapping_cfg,
            save_root=args.save_dir,
            use_proposed=args.use_proposed,
            save_preview_flag=args.save_preview,
            save_meshes=args.save_meshes,
        )
        exported.append({
            "key": entry["key"],
            "output_dir": out_dir,
        })
        print(f"  Saved: {out_dir}")

    index = {
        "manifest": args.manifest,
        "mapping": args.mapping,
        "use_proposed": bool(args.use_proposed),
        "num_entries": len(exported),
        "entries": exported,
    }
    index_path = os.path.join(args.save_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"Index written to: {index_path}")


if __name__ == "__main__":
    main()
