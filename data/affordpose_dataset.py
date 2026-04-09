import hashlib
import json
from typing import Iterable, Optional

import numpy as np
import trimesh

SLOT_NAME_TO_ID = {
    "GRASP_SURFACE": 0,
    "FUNCTIONAL_END": 1,
    "CONTROL": 2,
    "CLOSURE": 3,
}


def _normalize_filter_values(values: Optional[Iterable[str]]):
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


def _parse_obj_lines(lines):
    verts = []
    faces = []

    for line in lines:
        if not line:
            continue
        if line.startswith("v "):
            parts = line.strip().split()
            if len(parts) < 4:
                continue
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


def _join_list(items):
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _action_prompt(entry):
    class_name = entry["class_name"]
    afford_name = entry["afford_name"]

    prompt_map = {
        ("bottle", "Twist"): "A hand twisting the bottle cap.",
        ("bottle", "Wrap-grasp"): "A hand wrapping around the bottle body.",
        ("dispenser", "Press"): "A hand pressing the spray bottle trigger.",
        ("dispenser", "Twist"): "A hand twisting the dispenser cap.",
        ("dispenser", "Wrap-grasp"): "A hand wrapping around the spray bottle body.",
        ("earphone", "Lift"): "A hand lifting the earphone.",
        ("knife", "Handle-grasp"): "A hand grasping the knife handle.",
        ("mug", "Handle-grasp"): "A hand grasping the mug handle.",
        ("mug", "Support"): "A hand supporting the mug from below.",
        ("mug", "Wrap-grasp"): "A hand wrapping around the mug body.",
        ("scissors", "Handle-grasp"): "A hand grasping the scissors handle.",
    }
    if (class_name, afford_name) in prompt_map:
        return prompt_map[(class_name, afford_name)]

    obj_name_map = {
        "bottle": "bottle",
        "dispenser": "spray bottle",
        "earphone": "earphone",
        "knife": "knife",
        "mug": "mug",
        "scissors": "scissors",
    }
    afford_verb_map = {
        "Twist": "twisting the",
        "Wrap-grasp": "wrapping around the",
        "Press": "pressing the",
        "Lift": "lifting the",
        "Handle-grasp": "grasping the",
        "Support": "supporting the",
    }
    obj_name = obj_name_map.get(class_name, class_name.replace("_", " "))
    verb = afford_verb_map.get(afford_name, f"performing {afford_name.lower()} on the")
    return f"A hand {verb} {obj_name}."


def _dhoi_style_prompt(entry):
    class_name = entry["class_name"]
    afford_name = entry["afford_name"]

    prompt_map = {
        ("bottle", "Twist"): (
            "grasping",
            [("the thumb and index finger", "cap", "CLOSURE")],
        ),
        ("bottle", "Wrap-grasp"): (
            "grasping",
            [("all fingers", "bottle body", "GRASP_SURFACE")],
        ),
        ("dispenser", "Press"): (
            "using",
            [
                ("the index finger", "trigger", "CONTROL"),
                ("the thumb, middle, ring, and little fingers", "bottle body or handle", "GRASP_SURFACE"),
            ],
        ),
        ("dispenser", "Twist"): (
            "grasping",
            [("the thumb and index finger", "cap", "CLOSURE")],
        ),
        ("dispenser", "Wrap-grasp"): (
            "grasping",
            [("all fingers", "bottle body or handle", "GRASP_SURFACE")],
        ),
        ("earphone", "Lift"): (
            "lifting",
            [("the thumb and index finger", "earphone body", "GRASP_SURFACE")],
        ),
        ("knife", "Handle-grasp"): (
            "grasping",
            [("the thumb and fingers", "handle", "GRASP_SURFACE")],
        ),
        ("mug", "Handle-grasp"): (
            "grasping",
            [("the thumb and fingers", "mug body or handle", "GRASP_SURFACE")],
        ),
        ("mug", "Support"): (
            "holding",
            [("the palm and fingers", "mug body", "GRASP_SURFACE")],
        ),
        ("mug", "Wrap-grasp"): (
            "grasping",
            [("all fingers", "mug body or handle", "GRASP_SURFACE")],
        ),
        ("scissors", "Handle-grasp"): (
            "grasping",
            [("the thumb and fingers", "handle", "GRASP_SURFACE")],
        ),
    }
    obj_name_map = {
        "bottle": "bottle",
        "dispenser": "spray bottle",
        "earphone": "earphone",
        "knife": "knife",
        "mug": "mug",
        "scissors": "scissors",
    }

    obj_name = obj_name_map.get(class_name, class_name.replace("_", " "))
    intent_phrase, clauses = prompt_map.get(
        (class_name, afford_name),
        ("grasping", [("the hand", obj_name, "GRASP_SURFACE")]),
    )
    clause_text = _join_list([f"{hand_part} on the {slot_phrase}" for hand_part, slot_phrase, _ in clauses])
    target_slot_names = [slot_name for _, _, slot_name in clauses]
    primary_slot_name = target_slot_names[0]
    return {
        "text_prompt": f"A hand {intent_phrase} the {obj_name}, with {clause_text}.",
        "text_prompt_style": "dhoi",
        "text_target_slot_names": target_slot_names,
        "text_target_slots": [SLOT_NAME_TO_ID[name] for name in target_slot_names],
        "text_primary_slot_name": primary_slot_name,
        "text_primary_slot": SLOT_NAME_TO_ID[primary_slot_name],
    }


def _build_prompt(entry, prompt_style):
    if prompt_style == "dhoi":
        return _dhoi_style_prompt(entry)
    if prompt_style == "action":
        prompt = _action_prompt(entry)
        class_name = entry["class_name"]
        afford_name = entry["afford_name"]
        slot_map = {
            ("bottle", "Twist"): ["CLOSURE"],
            ("bottle", "Wrap-grasp"): ["GRASP_SURFACE"],
            ("dispenser", "Press"): ["CONTROL"],
            ("dispenser", "Twist"): ["CLOSURE"],
            ("dispenser", "Wrap-grasp"): ["GRASP_SURFACE"],
            ("earphone", "Lift"): ["GRASP_SURFACE"],
            ("knife", "Handle-grasp"): ["GRASP_SURFACE"],
            ("mug", "Handle-grasp"): ["GRASP_SURFACE"],
            ("mug", "Support"): ["GRASP_SURFACE"],
            ("mug", "Wrap-grasp"): ["GRASP_SURFACE"],
            ("scissors", "Handle-grasp"): ["GRASP_SURFACE"],
        }
        target_slot_names = slot_map.get((class_name, afford_name), ["GRASP_SURFACE"])
        primary_slot_name = target_slot_names[0]
        return {
            "text_prompt": prompt,
            "text_prompt_style": "action",
            "text_target_slot_names": target_slot_names,
            "text_target_slots": [SLOT_NAME_TO_ID[name] for name in target_slot_names],
            "text_primary_slot_name": primary_slot_name,
            "text_primary_slot": SLOT_NAME_TO_ID[primary_slot_name],
        }
    raise ValueError(f"Unsupported AffordPose prompt_style: {prompt_style}")


class AffordPoseDataset:
    def __init__(
        self,
        manifest_path,
        n_samples=10000,
        keys=None,
        classes=None,
        affordances=None,
        limit=None,
        prompt_style="dhoi",
    ):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        entries = list(manifest["entries"])
        keys = _normalize_filter_values(keys)
        classes = _normalize_filter_values(classes)
        affordances = _normalize_filter_values(affordances)

        if keys:
            key_set = set(keys)
            entries = [e for e in entries if e["key"] in key_set]
        if classes:
            class_set = set(classes)
            entries = [e for e in entries if e["class_name"] in class_set]
        if affordances:
            afford_set = set(affordances)
            entries = [e for e in entries if e["afford_name"] in afford_set]
        if limit is not None:
            entries = entries[: int(limit)]

        self.manifest_path = manifest_path
        self.n_samples = int(n_samples)
        self.entries = entries
        self.prompt_style = str(prompt_style).strip().lower()
        self.obj_warehouse = {}
        self.obj_bbox_centers = {}

        if not self.entries:
            raise ValueError("AffordPoseDataset has no entries after filtering")

    def __len__(self):
        return len(self.entries)

    def _cache_obj_id(self, idx):
        entry = self.entries[idx]
        return f"{entry['class_name']}_{entry['obj_id']}"

    def _load_mesh(self, idx):
        cache_obj_id = self._cache_obj_id(idx)
        if cache_obj_id not in self.obj_warehouse:
            entry = self.entries[idx]
            with open(entry["sample_json"], "r", encoding="utf-8") as f:
                sample = json.load(f)
            verts, faces = _parse_obj_lines(sample["object_mesh"])
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            bbox_center = (mesh.vertices.min(0) + mesh.vertices.max(0)) / 2.0
            mesh.vertices = mesh.vertices - bbox_center
            self.obj_warehouse[cache_obj_id] = mesh
            self.obj_bbox_centers[cache_obj_id] = bbox_center.astype(np.float32)
        return self.obj_warehouse[cache_obj_id]

    def get_obj_mesh(self, idx):
        return self._load_mesh(idx)

    def get_obj_id(self, idx):
        return self._cache_obj_id(idx)

    def get_match_obj_ids(self, idx):
        entry = self.entries[idx]
        return [
            self._cache_obj_id(idx),
            str(entry["obj_id"]),
            entry["key"],
        ]

    def __getitem__(self, idx):
        entry = self.entries[idx]
        obj_mesh = self.get_obj_mesh(idx)
        obj_id = self.get_obj_id(idx)

        seed = int(hashlib.md5(entry["key"].encode("utf-8")).hexdigest(), 16) % (2**31)
        np.random.seed(seed)
        sample_pts, face_idx = trimesh.sample.sample_surface(obj_mesh, self.n_samples)
        obj_verts = np.asarray(sample_pts, dtype=np.float32)
        obj_vn = np.asarray(obj_mesh.face_normals[face_idx], dtype=np.float32)
        prompt_spec = _build_prompt(entry, self.prompt_style)

        return {
            "obj_verts": obj_verts,
            "obj_vn": obj_vn,
            "obj_id": obj_id,
            "cache_obj_id": obj_id,
            "raw_obj_id": str(entry["obj_id"]),
            "cate_id": entry["class_name"],
            "class_name": entry["class_name"],
            "afford_name": entry["afford_name"],
            "manifest_key": entry["key"],
            "afford_dir": entry["afford_dir"],
            "sample_json": entry["sample_json"],
            "obj_rotmat": np.eye(3, dtype=np.float32),
            "sample_idx": idx,
            "text_prompt": prompt_spec["text_prompt"],
            "text_prompt_style": prompt_spec["text_prompt_style"],
            "text_target_slot_names": prompt_spec["text_target_slot_names"],
            "text_target_slots": prompt_spec["text_target_slots"],
            "text_primary_slot_name": prompt_spec["text_primary_slot_name"],
            "text_primary_slot": prompt_spec["text_primary_slot"],
        }
