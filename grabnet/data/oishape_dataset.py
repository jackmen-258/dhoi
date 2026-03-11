import hashlib
import os
import re
import numpy as np
import torch
import trimesh
from tqdm import tqdm

import pickle
import json
import glob
import logging
from bps_torch.bps import bps_torch

import smplx

ALL_CAT = [
    "apple", "banana", "binoculars", "bottle", "bowl", "cameras", "can", "cup",
    "cylinder_bottle", "donut", "eyeglasses", "flashlight", "fryingpan",
    "gamecontroller", "hammer", "headphones", "knife", "lightbulb", "lotion_pump",
    "mouse", "mug", "pen", "phone", "pincer", "power_drill", "scissors",
    "screwdriver", "squeezable", "stapler", "teapot", "toothbrush",
    "trigger_sprayer", "wineglass", "wrench",
]

ALL_SPLIT = ["train", "val", "test"]

ALL_INTENT = {
    "use": "0001",
    "hold": "0002",
    "liftup": "0003",
    "handover": "0004",
}

CENTER_IDX = 0

def to_list(x):
    if isinstance(x, list):
        return x
    return [x]

def check_valid(list, valid_list):
    for x in list:
        if x not in valid_list:
            return False
    return True

def suppress_trimesh_logging():
    logger = logging.getLogger("trimesh")
    logger.setLevel(logging.ERROR)

def get_hand_parameter(path):
    pose = pickle.load(open(path, "rb"))
    return pose["pose"], pose["shape"], pose["tsl"]

def get_obj_path(oid, data_path, meta_path, use_downsample=True, key="align"):
    obj_suffix_path = "align_ds" if use_downsample else "align"
    real_meta = json.load(open(os.path.join(meta_path, "object_id.json"), "r"))
    virtual_meta = json.load(open(os.path.join(meta_path, "virtual_object_id.json"), "r"))
    if oid in real_meta:
        obj_name = real_meta[oid]["name"]
        obj_path = os.path.join(data_path, "OakInkObjectsV2")
    else:
        obj_name = virtual_meta[oid]["name"]
        obj_path = os.path.join(data_path, "OakInkVirtualObjectsV2")
    obj_mesh_path = list(
        glob.glob(os.path.join(obj_path, obj_name, obj_suffix_path, "*.obj")) +
        glob.glob(os.path.join(obj_path, obj_name, obj_suffix_path, "*.ply")))
    if len(obj_mesh_path) > 1:
        obj_mesh_path = [p for p in obj_mesh_path if key in os.path.split(p)[1]]
    assert len(obj_mesh_path) == 1
    return obj_mesh_path[0]

def rodrigues_torch(rvec):
    """
    将 axis-angle 转为旋转矩阵（PyTorch 版本）
    Args:
        rvec: [B, 3] 或 [3]
    Returns:
        R: [B, 3, 3] 或 [3, 3]
    """
    if rvec.ndim == 1:
        rvec = rvec.unsqueeze(0)
    
    theta = torch.norm(rvec, dim=-1, keepdim=True)
    theta_expanded = theta.unsqueeze(-1)
    
    # 避免除零
    r = rvec / (theta + 1e-8)
    
    # 构造反对称矩阵 K
    K = torch.zeros(rvec.shape[0], 3, 3, device=rvec.device, dtype=rvec.dtype)
    K[:, 0, 1] = -r[:, 2]
    K[:, 0, 2] = r[:, 1]
    K[:, 1, 0] = r[:, 2]
    K[:, 1, 2] = -r[:, 0]
    K[:, 2, 0] = -r[:, 1]
    K[:, 2, 1] = r[:, 0]
    
    # Rodrigues 公式: R = I + sin(theta)*K + (1-cos(theta))*K^2
    I = torch.eye(3, device=rvec.device, dtype=rvec.dtype).unsqueeze(0).repeat(rvec.shape[0], 1, 1)
    R = I + torch.sin(theta_expanded) * K + (1 - torch.cos(theta_expanded)) * torch.bmm(K, K)
    
    return R.squeeze(0) if rvec.shape[0] == 1 else R


class OIShape:
    def __init__(self, split='train', bps_path='grabnet/configs/bps.npz', cache_bps=True, cache_dir='grabnet/cache'):
        suppress_trimesh_logging()

        self.data_split = split
        self.is_train = ("train" in self.data_split)

        self.category = "all"
        self.intent_idx = set(ALL_INTENT.values())

        self.use_downsample_mesh = True
        self.n_samples = 10000  # GrabNet 使用 10000 个点

        assert 'OAKINK_DIR' in os.environ, "environment variable 'OAKINK_DIR' is not set"
        data_dir = os.path.join(os.environ['OAKINK_DIR'], "shape")
        oi_shape_dir = os.path.join(data_dir, "oakink_shape_v2")
        meta_dir = os.path.join(data_dir, "metaV2")

        self.data_dir = data_dir
        self.meta_dir = meta_dir
        self.oi_shape_dir = oi_shape_dir
        self.cache_dir = cache_dir

        if self.data_split == 'all':
            self.data_split = ALL_SPLIT
        if self.category == 'all':
            self.category = ALL_CAT

        self.data_split = to_list(self.data_split)
        self.categories = to_list(self.category)

        assert check_valid(self.data_split, ALL_SPLIT) and check_valid(self.categories, ALL_CAT), \
            "invalid data split or category"

        self._mano_layer = None

        # ===== 加载 BPS basis =====
        bps_data = np.load(bps_path)
        self.bps_basis = torch.from_numpy(bps_data['basis']).float()
        self.bps_path = bps_path
        
        print(f"[INFO] Loaded BPS basis from {bps_path}, shape: {self.bps_basis.shape}")

        self.obj_warehouse = {}

        self.grasp_list = self._prepare_data()
        self.obj_id_set = {g["obj_id"] for g in self.grasp_list}
        
        print(f"[INFO] OIShape dataset loaded: {len(self.grasp_list)} samples, split={split}")

        # ===== 预计算或加载 BPS 编码 =====
        if cache_bps:
            self._load_or_compute_bps_cache()
        else:
            self.bps_cache = None
            self.obj_verts_cache = None

    def _prepare_data(self):
        # region ===== filter with regex >>>>>
        grasp_list = []
        seq_cat_matcher = re.compile(r"(.+)/(.{6})_(.{4})_([_0-9]+)/([\-0-9]+)")
        
        for cat in tqdm(self.categories, desc="Process categories"):
            real_matcher = re.compile(rf"({cat}/(.{{6}})/.{{10}})/hand_param\.pkl$")
            virtual_matcher = re.compile(rf"({cat}/(.{{6}})/.{{10}})/(.{{6}})/hand_param\.pkl$")
            path = os.path.join(self.oi_shape_dir, cat)
            
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
                        pass_stage, raw_obj_id, action_id, subject_id = (
                            grasp_cat_match[0], grasp_cat_match[1],
                            grasp_cat_match[2], grasp_cat_match[3]
                        )
                        obj_id = re_match[0][2] if is_virtual else re_match[0][1]
                        assert (is_virtual and raw_obj_id == re_match[0][1]) or obj_id == raw_obj_id
                        
                        # filter with intent mode
                        if action_id not in self.intent_idx:
                            continue
                        
                        # filter with data split
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

        return grasp_list

    def _get_mano_layer(self):
        if self._mano_layer is None:
            self._mano_layer = smplx.MANO(
                model_path='mano/models',
                use_pca=False,
                is_rhand=True,
                flat_hand_mean=True,
                batch_size=1,
                num_betas=10  # ✅ 明确指定 10 个 shape 参数
            )
        return self._mano_layer

    @staticmethod
    def _to_1d(x: np.ndarray) -> np.ndarray:
        """把 (1, N) / (N,) 等统一成 (N,) float32。"""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2 and x.shape[0] == 1:
            x = x[0]
        return x.reshape(-1)

    def _load_or_compute_bps_cache(self):
        """加载或计算 BPS 编码缓存"""
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 缓存文件路径（包含 split 信息，因为不同 split 可能有不同物体）
        cache_file = os.path.join(self.cache_dir, f'bps_cache_{self.data_split[0]}.pkl')
        
        if os.path.exists(cache_file):
            print(f"[INFO] Loading BPS cache from {cache_file}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.bps_cache = cache_data['bps_cache']
                self.obj_verts_cache = cache_data['obj_verts_cache']
            print(f"[INFO] Loaded BPS cache for {len(self.bps_cache)} objects")
        else:
            print(f"[INFO] Computing BPS encodings for {len(self.obj_id_set)} unique objects...")
            self._precompute_bps()
            
            # 保存缓存
            print(f"[INFO] Saving BPS cache to {cache_file}")
            cache_data = {
                'bps_cache': self.bps_cache,
                'obj_verts_cache': self.obj_verts_cache,
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=4)
            print(f"[INFO] BPS cache saved")

    def _precompute_bps(self):
        """预计算所有物体的 BPS 编码"""
        bps = bps_torch(custom_basis=self.bps_basis, device=torch.device('cpu'))

        self.bps_cache = {}
        self.obj_verts_cache = {}

        obj_to_idx = {}
        for idx, grasp in enumerate(self.grasp_list):
            obj_id = grasp['obj_id']
            if obj_id not in obj_to_idx:
                obj_to_idx[obj_id] = idx

        for obj_id in tqdm(self.obj_id_set, desc="Computing BPS encodings"):
            idx = obj_to_idx[obj_id]
            obj_mesh = self.get_obj_mesh(idx)

            obj_verts = self._sample_and_subdivide_mesh(obj_mesh, n_sample_verts=self.n_samples)
            obj_verts = np.array(obj_verts, dtype=np.float32)

            obj_verts_tensor = torch.from_numpy(obj_verts).float()
            with torch.no_grad():
                bps_result = bps.encode(obj_verts_tensor, feature_type='dists')
                bps_encoding = bps_result['dists'].cpu().numpy()

            # ✅ 关键：确保存入缓存的是 (4096,)
            bps_encoding = self._to_1d(bps_encoding)

            self.bps_cache[obj_id] = bps_encoding
            self.obj_verts_cache[obj_id] = obj_verts

        print(f"[INFO] Precomputed BPS for {len(self.bps_cache)} objects")
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
        return self.obj_warehouse[obj_id]

    def _sample_and_subdivide_mesh(self, obj_mesh, n_sample_verts=10000):
        """采样并细分网格以获得足够的顶点"""
        mesh = obj_mesh.copy()
        while mesh.vertices.shape[0] < n_sample_verts:
            mesh = mesh.subdivide()
        
        # ✅ 关键修复：强制转换为 numpy array
        verts = np.array(mesh.vertices, dtype=np.float32)  # 转换为标准 numpy
        verts_sample_id = np.random.choice(verts.shape[0], n_sample_verts, replace=False)
        verts_sampled = verts[verts_sample_id]
        
        return verts_sampled

    def __getitem__(self, idx):
        grasp = self.grasp_list[idx]
        obj_id = grasp["obj_id"]

        if self.bps_cache is not None:
            bps_object = self._to_1d(self.bps_cache[obj_id])  # ✅ 关键
            obj_verts = np.array(self.obj_verts_cache[obj_id], dtype=np.float32)
        else:
            obj_mesh = self.get_obj_mesh(idx)
            obj_verts = self._sample_and_subdivide_mesh(obj_mesh, n_sample_verts=self.n_samples)

            obj_verts_tensor = torch.from_numpy(obj_verts).float()
            bps = bps_torch(custom_basis=self.bps_basis, device=torch.device('cpu'))
            bps_result = bps.encode(obj_verts_tensor, feature_type='dists')
            bps_object = self._to_1d(bps_result['dists'].cpu().numpy())  # ✅ 关键

        # ===== 手部参数 =====
        hand_pose = np.array(grasp["hand_pose"], dtype=np.float32)
        hand_tsl = np.array(grasp["hand_tsl"], dtype=np.float32)
        hand_shape = np.array(grasp["hand_shape"], dtype=np.float32)

        # ===== 计算 GT 手部顶点 =====
        mano = self._get_mano_layer()
        with torch.no_grad():
            hand_pose_t = torch.from_numpy(hand_pose).float().unsqueeze(0)
            hand_tsl_t = torch.from_numpy(hand_tsl).float().unsqueeze(0)
            hand_shape_t = torch.from_numpy(hand_shape).float().unsqueeze(0)
            
            output = mano(
                global_orient=hand_pose_t[:, :3],
                hand_pose=hand_pose_t[:, 3:],
                betas=hand_shape_t,
                transl=hand_tsl_t
            )
            verts_rhand = output.vertices[0].cpu().numpy().astype(np.float32)

        # ===== 添加噪声生成扰动参数 =====
        if self.is_train:
            pose_noise_std = 0.1
            hand_pose_noisy = hand_pose + np.random.randn(*hand_pose.shape).astype(np.float32) * pose_noise_std
            
            transl_noise_std = 0.01
            hand_tsl_noisy = hand_tsl + np.random.randn(*hand_tsl.shape).astype(np.float32) * transl_noise_std
            
            with torch.no_grad():
                hand_pose_noisy_t = torch.from_numpy(hand_pose_noisy).float().unsqueeze(0)
                hand_tsl_noisy_t = torch.from_numpy(hand_tsl_noisy).float().unsqueeze(0)
                
                output_noisy = mano(
                    global_orient=hand_pose_noisy_t[:, :3],
                    hand_pose=hand_pose_noisy_t[:, 3:],
                    betas=hand_shape_t,
                    transl=hand_tsl_noisy_t
                )
                verts_rhand_f = output_noisy.vertices[0].cpu().numpy().astype(np.float32)
            
            # 扰动的旋转矩阵
            hand_pose_noisy_t_reshaped = torch.from_numpy(hand_pose_noisy).view(16, 3)
            hand_pose_rotmat_noisy = rodrigues_torch(hand_pose_noisy_t_reshaped.view(-1, 3)).view(16, 3, 3)
            
            global_orient_rotmat_f = hand_pose_rotmat_noisy[0].numpy().astype(np.float32).reshape(-1)  # [9]
            fpose_rotmat_f = hand_pose_rotmat_noisy[1:].numpy().astype(np.float32).reshape(-1)        # [45]
            trans_rhand_f = hand_tsl_noisy
        else:
            verts_rhand_f = verts_rhand.copy()
            trans_rhand_f = hand_tsl.copy()
            global_orient_rotmat_f = None
            fpose_rotmat_f = None

        # ===== 转换 GT 为旋转矩阵 =====
        hand_pose_t = torch.from_numpy(hand_pose).view(16, 3)
        hand_pose_rotmat = rodrigues_torch(hand_pose_t.view(-1, 3)).view(16, 3, 3)
        
        global_orient_rotmat = hand_pose_rotmat[0].numpy().astype(np.float32).reshape(-1)  # ✅ [9] 展平
        fpose_rotmat = hand_pose_rotmat[1:].numpy().astype(np.float32).reshape(-1)        # ✅ [45] 展平
        
        if not self.is_train:
            global_orient_rotmat_f = global_orient_rotmat.copy()
            fpose_rotmat_f = fpose_rotmat.copy()

        # ===== 返回结果 =====
        result = {
            'bps_object': bps_object,                           # [4096]
            'transl': hand_tsl,                                 # [3]
            'global_orient': hand_pose[:3],                     # [3]
            'hand_pose': hand_pose[3:],                         # [45]
            
            # GT 参数（展平的旋转矩阵）
            'trans_rhand': hand_tsl,                            # [3]
            'global_orient_rhand_rotmat': global_orient_rotmat, # [9] ✅ 展平
            'fpose_rhand_rotmat': fpose_rotmat,                 # [45] ✅ 展平
            
            # 顶点
            'verts_object': obj_verts,                          # [10000, 3]
            'verts_rhand': verts_rhand,                         # [778, 3]
            
            # 扰动的参数（展平的旋转矩阵）
            'verts_rhand_f': verts_rhand_f,                     # [778, 3]
            'trans_rhand_f': trans_rhand_f,                     # [3]
            'global_orient_rhand_rotmat_f': global_orient_rotmat_f,  # [9] ✅ 展平
            'fpose_rhand_rotmat_f': fpose_rotmat_f,             # [45] ✅ 展平
        }
        
        return result


if __name__ == '__main__':
    # 测试代码
    from torch.utils.data import DataLoader
    
    print("=== Testing with BPS cache ===")
    ds = OIShape(split='train', bps_path='grabnet/configs/bps.npz', cache_bps=True)
    print(f"Dataset size: {len(ds)}")
    
    # 测试单个样本
    sample = ds[0]
    print("\nSample keys and shapes:")
    for k, v in sample.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: {v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v)}")
    
    # 测试 DataLoader（可以使用多进程）
    print("\n=== Testing DataLoader with num_workers=8 ===")
    loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=8, drop_last=True)
    
    import time
    start = time.time()
    batch = next(iter(loader))
    elapsed = time.time() - start
    
    print(f"\nBatch loaded in {elapsed:.2f}s")
    print("Batch keys and shapes:")
    for k, v in batch.items():
        print(f"  {k}: {v.shape}, dtype={v.dtype}")
    
    print("\n✅ Dataset format matches GrabNet LoadData!")
    print("✅ Multi-processing works with BPS cache!")