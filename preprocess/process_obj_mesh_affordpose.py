#!/usr/bin/env python3
"""
process_obj_mesh_affordpose.py
==============================

Pre-process AffordPose object meshes for physical evaluation.

Compared with OakInk's original `scripts/process_obj_mesh.py`, the main
difference is the data source:
  - OakInk: object meshes come from OakInkShape
  - AffordPose: object meshes are embedded as OBJ text inside each sample JSON

This script groups AffordPose samples by unique object (`class_name + obj_id`),
exports a bbox-centered raw OBJ for each unique object, then runs the same
three-stage pipeline used by OakInk evaluation:
  1. watertight  (ManifoldPlus)
  2. voxel       (binvox)
  3. vhacd       (pybullet.vhacd)

Output layout:
    {proc_dir}/
      raw/{cache_obj_id}.obj
      watertight/{cache_obj_id}.obj
      voxel/{cache_obj_id}.binvox
      vhacd/{cache_obj_id}.obj
      object_index.json

Example:
    python preprocess/process_obj_mesh_affordpose.py --stage watertight
    python preprocess/process_obj_mesh_affordpose.py --stage voxel
    python preprocess/process_obj_mesh_affordpose.py --stage vhacd
"""

import argparse
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict

import numpy as np
import trimesh
from joblib import Parallel, delayed
from tqdm import tqdm


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_tool_candidates():
    repo_parent = os.path.dirname(PROJECT_ROOT)
    return {
        "manifold": [
            os.path.join(PROJECT_ROOT, "thirdparty", "ManifoldPlus", "build", "manifold"),
            os.path.join(repo_parent, "OakInk-Grasp-Generation", "thirdparty", "ManifoldPlus", "build", "manifold"),
        ],
        "binvox": [
            os.path.join(PROJECT_ROOT, "thirdparty", "binvox"),
            os.path.join(repo_parent, "OakInk-Grasp-Generation", "thirdparty", "binvox"),
        ],
    }


def resolve_tool_path(explicit_path, tool_name):
    if explicit_path:
        return explicit_path

    for candidate in _default_tool_candidates()[tool_name]:
        if os.path.isfile(candidate):
            return candidate
    return None


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_filter_values(values):
    if not values:
        return None

    normalized = []
    for value in values:
        if value is None:
            continue
        for token in str(value).split(","):
            token = token.strip()
            if token:
                normalized.append(token)

    if not normalized:
        return None
    return list(dict.fromkeys(normalized))


def parse_obj_lines(lines):
    verts = []
    faces = []

    for line in lines:
        if not line:
            continue
        if line.startswith("v "):
            parts = line.strip().split()
            if len(parts) < 4:
                raise ValueError(f"Invalid vertex line: {line!r}")
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("f "):
            items = line.strip().split()[1:]
            idxs = []
            for item in items:
                raw = item.split("/")[0]
                if raw:
                    idxs.append(int(raw) - 1)
            if len(idxs) < 3:
                continue
            for i in range(1, len(idxs) - 1):
                faces.append([idxs[0], idxs[i], idxs[i + 1]])

    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    return verts, faces


def select_manifest_entries(manifest, keys=None, classes=None, obj_ids=None, affordances=None, limit=None):
    entries = list(manifest["entries"])

    keys = normalize_filter_values(keys)
    classes = normalize_filter_values(classes)
    obj_ids = normalize_filter_values(obj_ids)
    affordances = normalize_filter_values(affordances)

    if keys:
        key_set = set(keys)
        entries = [e for e in entries if e["key"] in key_set]
    if classes:
        class_set = set(classes)
        entries = [e for e in entries if e["class_name"] in class_set]
    if obj_ids:
        obj_set = set(obj_ids)
        entries = [
            e for e in entries
            if str(e["obj_id"]) in obj_set or f"{e['class_name']}_{e['obj_id']}" in obj_set
        ]
    if affordances:
        afford_set = set(affordances)
        entries = [e for e in entries if e["afford_name"] in afford_set]
    if limit is not None:
        entries = entries[: int(limit)]
    return entries


def build_object_records(entries):
    grouped = defaultdict(list)
    for entry in entries:
        cache_obj_id = f"{entry['class_name']}_{entry['obj_id']}"
        grouped[cache_obj_id].append(entry)

    objects = []
    for cache_obj_id, items in grouped.items():
        entry0 = items[0]
        objects.append({
            "cache_obj_id": cache_obj_id,
            "class_name": entry0["class_name"],
            "obj_id": str(entry0["obj_id"]),
            "sample_json": entry0["sample_json"],
            "manifest_keys": [e["key"] for e in items],
            "affordances": sorted({e["afford_name"] for e in items}),
            "num_entries": len(items),
        })
    objects.sort(key=lambda x: x["cache_obj_id"])
    return objects


def load_affordpose_mesh(sample_json):
    data = load_json(sample_json)
    verts, faces = parse_obj_lines(data["object_mesh"])
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    bbox_center = (mesh.vertices.min(0) + mesh.vertices.max(0)) / 2.0
    mesh.vertices = mesh.vertices - bbox_center
    return mesh


def export_raw_mesh(record, raw_dir, overwrite=False):
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = os.path.join(raw_dir, f"{record['cache_obj_id']}.obj")
    if os.path.exists(raw_path) and not overwrite:
        return raw_path

    mesh = load_affordpose_mesh(record["sample_json"])
    mesh.visual = trimesh.visual.ColorVisuals()
    trimesh.exchange.export.export_mesh(mesh, raw_path, file_type="obj")
    return raw_path


def watertight_one(record, raw_dir, export_dir, manifold_path, overwrite=False, depth_list=None):
    os.makedirs(export_dir, exist_ok=True)
    oid = record["cache_obj_id"]
    out_path = os.path.join(export_dir, f"{oid}.obj")
    err_path = os.path.join(export_dir, f"{oid}.err")

    if os.path.exists(out_path) and not overwrite:
        return {"status": "skipped", "cache_obj_id": oid}

    raw_path = export_raw_mesh(record, raw_dir, overwrite=overwrite)
    depth_list = depth_list or [8, 7, 6]

    try:
        if os.path.exists(err_path):
            os.remove(err_path)

        for depth in depth_list:
            command = [
                manifold_path,
                "--input", raw_path,
                "--output", out_path,
                "--depth", str(depth),
            ]
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            obj_reload = trimesh.load(out_path, process=False, force="mesh", skip_materials=True)
            assert obj_reload.is_watertight, f"{out_path} is not watertight"
            if obj_reload.faces.shape[0] < 30000:
                break
        return {"status": "success", "cache_obj_id": oid}
    except Exception as e:
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(str(e))
        return {"status": "failed", "cache_obj_id": oid, "error": str(e)}


def voxel_one(oid, input_mesh_path, export_dir, binvox_path, overwrite=False):
    os.makedirs(export_dir, exist_ok=True)
    out_mesh_dest = os.path.join(export_dir, f"{oid}.binvox")
    err_path = os.path.join(export_dir, f"{oid}.err")
    if os.path.exists(out_mesh_dest) and not overwrite:
        return {"status": "skipped", "cache_obj_id": oid}

    command = [binvox_path, "-d", "128", input_mesh_path]
    res = subprocess.run(command, capture_output=True, check=False)

    res_binvox_path = input_mesh_path.replace(".obj", ".binvox")
    if not os.path.exists(res_binvox_path):
        stdout_text = res.stdout.decode("latin1", "replace")
        stderr_text = res.stderr.decode("latin1", "replace")
        hint = ""
        if "failed to open display" in stderr_text.lower():
            hint = (
                "binvox requires an OpenGL display. On a remote server, start Xvfb "
                "and export DISPLAY before running the voxel stage."
            )
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(f"returncode: {res.returncode}\n")
            if hint:
                f.write(f"hint: {hint}\n")
            f.write("\n[stdout]\n")
            f.write(stdout_text)
            f.write("\n[stderr]\n")
            f.write(stderr_text)
        return {
            "status": "failed",
            "cache_obj_id": oid,
            "error": (
                f"binvox output not found: {res_binvox_path} "
                f"(returncode={res.returncode})"
                + (f"; {hint}" if hint else "")
            ),
        }

    shutil.move(res_binvox_path, out_mesh_dest)
    if os.path.exists(err_path):
        os.remove(err_path)
    return {"status": "success", "cache_obj_id": oid}


def vhacd_one(oid, input_mesh_path, export_dir, overwrite=False):
    os.makedirs(export_dir, exist_ok=True)
    res_vhacd_path = os.path.join(export_dir, f"{oid}.obj")
    if os.path.exists(res_vhacd_path) and not overwrite:
        return {"status": "skipped", "cache_obj_id": oid}

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import pybullet as pbl

            pbl.vhacd(
                fileNameIn=input_mesh_path,
                fileNameOut=res_vhacd_path,
                fileNameLogging="/dev/null",
                resolution=100000,
                concavity=0.0025,
                planeDownsampling=4,
                convexhullDownsampling=4,
                alpha=0.05,
                beta=0.0,
                pca=0,
                mode=0,
                maxNumVerticesPerCH=64,
                minVolumePerCH=0.0001,
            )
        if not os.path.exists(res_vhacd_path):
            raise FileNotFoundError(f"VHACD output missing: {res_vhacd_path}")
        return {"status": "success", "cache_obj_id": oid}
    except Exception as e:
        err_path = os.path.join(export_dir, f"{oid}.err")
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(str(e))
        return {"status": "failed", "cache_obj_id": oid, "error": str(e)}


def write_object_index(proc_dir, manifest_path, objects, filters):
    index = {
        "dataset_name": "AffordPose",
        "manifest_path": os.path.abspath(manifest_path),
        "num_unique_objects": len(objects),
        "filters": filters,
        "objects": objects,
    }
    with open(os.path.join(proc_dir, "object_index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def summarize_results(stage, results, out_dir):
    status_count = defaultdict(int)
    failures = []
    for item in results:
        status_count[item["status"]] += 1
        if item["status"] == "failed":
            failures.append(item)

    summary = {
        "stage": stage,
        "success": status_count["success"],
        "skipped": status_count["skipped"],
        "failed": status_count["failed"],
        "results": results,
    }
    out_path = os.path.join(out_dir, f"{stage}_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary, out_path, failures


def missing_input_result(oid, input_mesh_path):
    return {
        "status": "failed",
        "cache_obj_id": oid,
        "error": f"watertight mesh missing: {input_mesh_path}",
    }


def watertight(args, objects):
    manifold_path = resolve_tool_path(args.manifold_path, "manifold")
    if not manifold_path:
        raise FileNotFoundError(
            "ManifoldPlus executable not found. "
            "Pass --manifold_path or install it under dhoi/thirdparty or OakInk-Grasp-Generation/thirdparty."
        )

    raw_dir = os.path.join(args.proc_dir, "raw")
    export_dir = os.path.join(args.proc_dir, "watertight")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(export_dir, exist_ok=True)

    tasks = []
    for record in tqdm(objects, desc="AffordPose raw export"):
        export_raw_mesh(record, raw_dir, overwrite=args.overwrite)
        tasks.append(delayed(watertight_one)(
            record, raw_dir, export_dir, manifold_path,
            overwrite=args.overwrite,
        ))

    return Parallel(n_jobs=args.n_jobs, verbose=10, timeout=args.timeout)(tasks)


def voxel(args, objects):
    binvox_path = resolve_tool_path(args.binvox_path, "binvox")
    if not binvox_path:
        raise FileNotFoundError(
            "binvox executable not found. "
            "Pass --binvox_path or install it under dhoi/thirdparty or OakInk-Grasp-Generation/thirdparty."
        )

    wt_dir = os.path.join(args.proc_dir, "watertight")
    export_dir = os.path.join(args.proc_dir, "voxel")
    if not os.path.isdir(wt_dir):
        raise FileNotFoundError(f"Watertight dir not found: {wt_dir}")
    if not os.environ.get("DISPLAY"):
        print(
            "[warn] DISPLAY is not set. binvox may fail on headless servers. "
            "If voxelization fails with 'failed to open display', run Xvfb first."
        )

    tasks = []
    for record in objects:
        input_mesh_path = os.path.join(wt_dir, f"{record['cache_obj_id']}.obj")
        if not os.path.exists(input_mesh_path):
            tasks.append(delayed(missing_input_result)(record["cache_obj_id"], input_mesh_path))
            continue
        tasks.append(delayed(voxel_one)(
            record["cache_obj_id"], input_mesh_path, export_dir,
            binvox_path, overwrite=args.overwrite,
        ))
    return Parallel(n_jobs=args.n_jobs, verbose=10, timeout=args.timeout)(tasks)


def vhacd(args, objects):
    wt_dir = os.path.join(args.proc_dir, "watertight")
    export_dir = os.path.join(args.proc_dir, "vhacd")
    if not os.path.isdir(wt_dir):
        raise FileNotFoundError(f"Watertight dir not found: {wt_dir}")

    tasks = []
    for record in objects:
        input_mesh_path = os.path.join(wt_dir, f"{record['cache_obj_id']}.obj")
        if not os.path.exists(input_mesh_path):
            tasks.append(delayed(missing_input_result)(record["cache_obj_id"], input_mesh_path))
            continue
        tasks.append(delayed(vhacd_one)(
            record["cache_obj_id"], input_mesh_path, export_dir,
            overwrite=args.overwrite,
        ))
    return Parallel(n_jobs=args.n_jobs, verbose=10, timeout=args.timeout)(tasks)


def main():
    parser = argparse.ArgumentParser(description="Process AffordPose object meshes for evaluation")
    parser.add_argument("--manifest", type=str, default="configs/affordpose_test_manifest.json")
    parser.add_argument("--proc_dir", type=str, default="data/AffordPose_object_process")
    parser.add_argument("--stage", type=str, default="watertight",
                        choices=["watertight", "voxel", "vhacd"])
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--manifold_path", type=str, default=None)
    parser.add_argument("--binvox_path", type=str, default=None)
    parser.add_argument("--keys", nargs="+", default=None)
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--obj_ids", nargs="+", default=None)
    parser.add_argument("--affordances", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    os.makedirs(args.proc_dir, exist_ok=True)

    manifest = load_json(args.manifest)
    entries = select_manifest_entries(
        manifest,
        keys=args.keys,
        classes=args.classes,
        obj_ids=args.obj_ids,
        affordances=args.affordances,
        limit=args.limit,
    )
    if not entries:
        raise ValueError("No AffordPose entries matched the requested filters")

    objects = build_object_records(entries)
    write_object_index(
        args.proc_dir,
        args.manifest,
        objects,
        filters={
            "keys": normalize_filter_values(args.keys),
            "classes": normalize_filter_values(args.classes),
            "obj_ids": normalize_filter_values(args.obj_ids),
            "affordances": normalize_filter_values(args.affordances),
            "limit": args.limit,
        },
    )

    print("=" * 60)
    print(f"Stage:          {args.stage}")
    print(f"Manifest:       {args.manifest}")
    print(f"Proc dir:       {args.proc_dir}")
    print(f"Unique objects: {len(objects)}")
    print("=" * 60)

    if args.stage == "watertight":
        results = watertight(args, objects)
    elif args.stage == "voxel":
        results = voxel(args, objects)
    elif args.stage == "vhacd":
        results = vhacd(args, objects)
    else:
        raise ValueError(f"Unsupported stage: {args.stage}")

    summary, summary_path, failures = summarize_results(args.stage, results, args.proc_dir)
    print(f"\nDone: success={summary['success']} skipped={summary['skipped']} failed={summary['failed']}")
    print(f"Summary: {summary_path}")
    if failures:
        print("First failures:")
        for item in failures[:5]:
            print(f"  {item['cache_obj_id']}: {item.get('error', 'unknown error')}")


if __name__ == "__main__":
    main()
