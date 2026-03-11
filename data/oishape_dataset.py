import hashlib
import os
import re
import numpy as np
import torch
import trimesh
from manotorch.manolayer import ManoLayer, MANOOutput
from .utils import (
    ALL_CAT,
    ALL_INTENT,
    ALL_SPLIT,
    CENTER_IDX,
    check_valid,
    get_hand_parameter,
    get_obj_path,
    to_list,
    suppress_trimesh_logging
)
from tqdm import tqdm


class OIShape:
    def __init__(self, cfg):
        self.cfg = cfg

        self.data_split = cfg.DATA_SPLIT
        self.is_train = ("train" in self.data_split)

        self.intent_mode = cfg.INTENT_MODE
        self.category = cfg.OBJ_CATES

        self.use_downsample_mesh = True
        self.n_samples = 2048
        self.mano_assets_root = "assets/mano_v1_2"

        assert 'OAKINK_DIR' in os.environ, "environment variable 'OAKINK_DIR' is not set"
        data_dir = os.path.join(os.environ['OAKINK_DIR'], "shape")
        oi_shape_dir = os.path.join(data_dir, "oakink_shape_v2")
        meta_dir = os.path.join(data_dir, "metaV2")

        self.data_dir = data_dir
        self.meta_dir = meta_dir
        self.oi_shape_dir = oi_shape_dir

        if self.data_split == 'all':
            self.data_split = ALL_SPLIT
        if self.category == 'all':
            self.category = ALL_CAT
        if self.intent_mode == 'all':
            self.intent_mode = list(ALL_INTENT)

        self.data_split = to_list(self.data_split)
        self.categories = to_list(self.category)
        self.intent_mode = to_list(self.intent_mode)
        assert (check_valid(self.data_split, ALL_SPLIT) and check_valid(self.categories, ALL_CAT) and
                check_valid(self.intent_mode, list(ALL_INTENT))), "invalid data split, category, or intent!"

        self.intent_idx = [ALL_INTENT[i] for i in self.intent_mode]
        self.action_id_to_intent = {v: k for k, v in ALL_INTENT.items()}

        self.mano_layer = ManoLayer(center_idx=CENTER_IDX, mano_assets_root=self.mano_assets_root)

        self.obj_warehouse = {}
        self.obj_bbox_centers = {}

        self.grasp_list = self._prepare_data()
        self.obj_id_set = {g["obj_id"] for g in self.grasp_list}

    def _prepare_data(self):
        # region ===== filter with regex >>>>>
        grasp_list = []
        category_begin_idx = []
        seq_cat_matcher = re.compile(r"(.+)/(.{6})_(.{4})_([_0-9]+)/([\-0-9]+)")
        for cat in tqdm(self.categories, desc="Process categories"):
            real_matcher = re.compile(rf"({cat}/(.{{6}})/.{{10}})/hand_param\.pkl$")
            virtual_matcher = re.compile(rf"({cat}/(.{{6}})/.{{10}})/(.{{6}})/hand_param\.pkl$")
            path = os.path.join(self.oi_shape_dir, cat)
            category_begin_idx.append(len(grasp_list))
            for cur, dirs, files in os.walk(path, followlinks=False):
                dirs.sort()
                for f in files:
                    re_match = virtual_matcher.findall(os.path.join(cur, f))
                    is_virtual = len(re_match) > 0
                    re_match = re_match + real_matcher.findall(os.path.join(cur, f))
                    if len(re_match) > 0:
                        assert len(re_match) == 1, "regex should return only one match"
                        source = open(os.path.join(self.oi_shape_dir, re_match[0][0], "source.txt")).read()
                        grasp_cat_match = seq_cat_matcher.findall(source)[0]
                        pass_stage, raw_obj_id, action_id, subject_id = (grasp_cat_match[0], grasp_cat_match[1],
                                                                         grasp_cat_match[2], grasp_cat_match[3])
                        obj_id = re_match[0][2] if is_virtual else re_match[0][1]
                        assert (is_virtual and raw_obj_id == re_match[0][1]) or obj_id == raw_obj_id
                        if action_id not in self.intent_idx:
                            continue
                        obj_id_hash = int(hashlib.md5(obj_id.encode("utf-8")).hexdigest(), 16)
                        if obj_id_hash % 10 < 8 and "train" not in self.data_split:
                            continue
                        elif obj_id_hash % 10 == 8 and "val" not in self.data_split:
                            continue
                        elif obj_id_hash % 10 == 9 and "test" not in self.data_split:
                            continue

                        hand_pose, hand_shape, hand_tsl = get_hand_parameter(os.path.join(cur, f))
                        grasp_item = {
                            "cate_id": cat,
                            "obj_id": obj_id,
                            "hand_joints": None,
                            "hand_verts": None,
                            "hand_pose": hand_pose,
                            "hand_shape": hand_shape,
                            "hand_tsl": hand_tsl,
                            "is_virtual": is_virtual,
                            "raw_obj_id": raw_obj_id,
                            "action_id": action_id,
                            "subject_id": subject_id,
                            "file_path": os.path.join(cur, f),
                        }
                        grasp_list.append(grasp_item)
        # endregion <<<<

        # region ===== cal hand joints >>>>>
        batch_hand_pose = []
        batch_hand_shape = []
        batch_hand_tsl = []
        for _, g in enumerate(grasp_list):
            batch_hand_pose.append(g["hand_pose"])
            batch_hand_shape.append(g["hand_shape"])
            batch_hand_tsl.append(g["hand_tsl"])
        batch_hand_shape = torch.from_numpy(np.stack(batch_hand_shape))
        batch_hand_pose = torch.from_numpy(np.stack(batch_hand_pose))
        batch_hand_tsl = np.stack(batch_hand_tsl)
        mano_output: MANOOutput = self.mano_layer(batch_hand_pose, batch_hand_shape)
        batch_hand_joints = mano_output.joints.numpy() + batch_hand_tsl[:, None, :]
        batch_hand_verts = mano_output.verts.numpy() + batch_hand_tsl[:, None, :]
        batch_hand_tsl = batch_hand_joints[:, CENTER_IDX]  # equal to original hand_tsl
        for i in range(len(grasp_list)):
            grasp_list[i]["hand_joints"] = batch_hand_joints[i]
            grasp_list[i]["hand_verts"] = batch_hand_verts[i]
            grasp_list[i]["hand_tsl"] = batch_hand_tsl[i]
        # endregion <<<<<

        return grasp_list

    def __len__(self):
        return len(self.grasp_list)

    def get_obj_mesh(self, idx):
        obj_id = self.grasp_list[idx]["obj_id"]
        if obj_id not in self.obj_warehouse:
            obj_path = get_obj_path(obj_id, self.data_dir, self.meta_dir, use_downsample=self.use_downsample_mesh)
            obj_trimesh = trimesh.load(obj_path, process=False, force="mesh", skip_materials=True)
            bbox_center = (obj_trimesh.vertices.min(0) + obj_trimesh.vertices.max(0)) / 2
            obj_trimesh.vertices = obj_trimesh.vertices - bbox_center
            self.obj_warehouse[obj_id] = obj_trimesh
            self.obj_bbox_centers[obj_id] = bbox_center.astype(np.float32)
        return self.obj_warehouse[obj_id]

    def get_obj_id(self, idx):
        return self.grasp_list[idx]["obj_id"]

    def get_hand_joints(self, idx):
        return self.grasp_list[idx]["hand_joints"]

    def get_hand_shape(self, idx):
        return self.grasp_list[idx]["hand_shape"]

    def get_hand_pose(self, idx):
        return self.grasp_list[idx]["hand_pose"]

    def get_obj_rotmat(self, idx):
        return np.eye(3, dtype=np.float32)

    def get_intent(self, idx):
        act_id = self.grasp_list[idx]["action_id"]
        intent_name = self.action_id_to_intent[act_id]
        return int(act_id), intent_name

    def __getitem__(self, idx):
        grasp = self.grasp_list[idx]
        obj_mesh = self.get_obj_mesh(idx)
        obj_id = self.get_obj_id(idx)

        # 确定性采样，保证同一物体每次采样的点云一致
        seed = int(hashlib.md5(obj_id.encode()).hexdigest(), 16) % (2**31)
        np.random.seed(seed)

        sample = trimesh.sample.sample_surface(obj_mesh, self.n_samples)
        obj_verts = np.array(sample[0], dtype=np.float32)
        obj_vn = np.array(obj_mesh.face_normals[sample[1]], dtype=np.float32)

        hand_pose  = np.array(grasp["hand_pose"],  dtype=np.float32)  # (48,)
        hand_shape = np.array(grasp["hand_shape"], dtype=np.float32)  # (10,)

        # 坐标系对齐: 物体已 bbox 居中, 手部坐标也需减去同一偏移
        bbox_center = self.obj_bbox_centers[obj_id]                   # (3,)
        hand_tsl   = np.array(grasp["hand_tsl"],   dtype=np.float32) - bbox_center
        hand_verts = np.array(grasp["hand_verts"], dtype=np.float32) - bbox_center[None, :]

        intent_id, intent_name = self.get_intent(idx)
        intent_id = intent_id - 1  # [1, 2, 3, 4] -> [0, 1, 2, 3]

        return {
            "obj_verts":   obj_verts,
            "obj_vn":      obj_vn,
            "hand_pose":   hand_pose,
            "hand_tsl":    hand_tsl,
            "hand_shape":  hand_shape,
            "hand_verts":  hand_verts,
            "intent_id":   intent_id,
            "intent_name": intent_name,
            "obj_id":      obj_id,
            "cate_id":     grasp["cate_id"],
            "obj_rotmat":  self.get_obj_rotmat(idx),
            "sample_idx":  idx,
        }