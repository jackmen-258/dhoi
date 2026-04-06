"""
train_decoder.py (GrabNet-style CVAE + RefineNet Joint Training)
================================================================
PoseGrabModel (CoarseNet) + RefineNet joint training script.

Following GrabNet's training approach, CoarseNet and RefineNet are trained
jointly in the same epoch loop, each with their own optimizer.

Architecture:
  - CoarseNet: CVAE with raw BPS(4096) + scheme-B token conditioning
  - RefineNet: iterative h2o-distance refinement (sub-module of PoseGrabModel)
  - Part-Contact Consistency Loss: encourages hand-part → object-slot contact
  - No shape/betas prediction (mean shape)

Usage:
  python train_decoder.py --epochs 200 --batch_size 64
"""

import os
import json
import math
import time
import random
import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import chamfer_distance as chd
from torch.utils.data import Dataset, DataLoader

from data.token_encoder import TokenEncoder
from models.condition_encoder import BPSSlotPredictor, build_token_histogram
from models.pose_decoder import (
    build_grab_model, ROT6D_POSE_DIM, TRANS_DIM, NUM_HAND_PARTS, NUM_SLOTS,
    point2point_signed, compute_geo_loss,
    build_vert_to_part, build_part_contact_masks,
    propagate_slot_labels, compute_part_contact_loss,
    RefineNet, compute_refine_loss, sanitize_posegrab_state_dict_for_load,
)
from utils.mano_utils import MANOHelper
from utils.pose_utils import aa_to_rot6d


COARSE_CKPT_PREFIXES = (
    "token_encoder",
    "coarse_net",
)
REFINE_CKPT_PREFIXES = ("refine_net",)
PREDICTOR_PREFIX = "bps_slot_predictor."


# ==============================================================================
# Contact Token Dropout
# ==============================================================================

@torch.no_grad()
def apply_token_dropout(token_mask, drop_prob, adaptive_pivot: int = 6):
    """
    k-adaptive token dropout.

    Sparse sequences (k < adaptive_pivot) have reduced drop_prob to avoid
    losing all semantic information in short sequences.
    """
    if drop_prob <= 0:
        return token_mask

    B, L = token_mask.shape
    mask = token_mask.clone()

    k_per_sample = token_mask.sum(dim=1).float()
    scale = (k_per_sample / adaptive_pivot).clamp(max=1.0)
    effective_prob = drop_prob * scale

    rand = torch.rand(B, L, device=mask.device)
    drop = rand < effective_prob.unsqueeze(1)
    mask = mask & ~drop

    empty = ~mask.any(dim=1)
    if empty.any():
        mask[empty] = token_mask[empty]

    return mask


@torch.no_grad()
def apply_token_substitution(tokens, token_mask, sub_prob, n_semantic: int = 24):
    """Randomly substitute some valid tokens with other semantic tokens."""
    if sub_prob <= 0 or n_semantic <= 1:
        return tokens

    B, L = tokens.shape
    tokens = tokens.clone()

    sub_mask = token_mask & (torch.rand(B, L, device=tokens.device) < sub_prob)
    if not sub_mask.any():
        return tokens

    random_ids = torch.randint(0, n_semantic, (B, L), device=tokens.device)
    same = random_ids == tokens
    if same.any():
        random_ids = torch.where(same, (random_ids + 1) % n_semantic, random_ids)

    tokens = torch.where(sub_mask, random_ids, tokens)
    return tokens

@torch.no_grad()
def recover_contact_matrix_from_obj_slots(
    hand_verts: torch.Tensor,      # (B, 778, 3)
    obj_pc: torch.Tensor,          # (B, N, 3)
    obj_slot_labels: torch.Tensor, # (B, N)
    vert_to_part: torch.Tensor,    # (778,)
    contact_distance: float,
    num_slots: int = NUM_SLOTS,
) -> torch.Tensor:
    """Recover a binary (hand_part x slot) contact matrix from a predicted grasp."""
    device = hand_verts.device
    B, V, _ = hand_verts.shape

    ch = chd.ChamferDistance()
    _, _, xidx_near, _ = ch(hand_verts, obj_pc)  # (B, V)
    xidx_near = xidx_near.long()

    nearest_obj = torch.gather(
        obj_pc, 1, xidx_near.unsqueeze(-1).expand(-1, -1, 3)
    )
    nearest_slots = torch.gather(obj_slot_labels.long(), 1, xidx_near)
    distances = torch.norm(hand_verts - nearest_obj, dim=-1)

    contact_mask = (distances < contact_distance) & (nearest_slots >= 0)
    part_ids = vert_to_part.to(device).view(1, V).expand(B, -1)

    contact_matrix = torch.full(
        (B, NUM_HAND_PARTS, num_slots),
        -1,
        dtype=torch.long,
        device=device,
    )

    for part_id in range(NUM_HAND_PARTS):
        part_mask = part_ids == part_id
        part_contact = contact_mask & part_mask
        for slot_id in range(num_slots):
            present = (part_contact & (nearest_slots == slot_id)).any(dim=1)
            contact_matrix[present, part_id, slot_id] = 0

    return contact_matrix


@torch.no_grad()
def compute_token_recall_f1(
    pred_contact_matrix: torch.Tensor,  # (B, 6, 4), -1/0
    gt_tokens: torch.Tensor,            # (B, L)
    gt_token_mask: torch.Tensor,        # (B, L)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compare predicted contact matrix with conditioning tokens."""
    gt_hist = build_token_histogram(
        gt_tokens,
        gt_token_mask,
        n_semantic=NUM_HAND_PARTS * NUM_SLOTS,
        binary=True,
    )
    pred_hist = (pred_contact_matrix.view(pred_contact_matrix.shape[0], -1) >= 0).float()

    tp = (pred_hist * gt_hist).sum(dim=1)
    pred_pos = pred_hist.sum(dim=1)
    gt_pos = gt_hist.sum(dim=1)

    recall = torch.where(
        gt_pos > 0,
        tp / gt_pos.clamp(min=1.0),
        torch.zeros_like(tp),
    )
    precision = torch.where(
        pred_pos > 0,
        tp / pred_pos.clamp(min=1.0),
        torch.zeros_like(tp),
    )
    denom = precision + recall
    f1 = torch.where(
        denom > 0,
        2.0 * precision * recall / denom,
        torch.zeros_like(tp),
    )
    return recall, f1


def _matches_prefix(key: str, prefixes) -> bool:
    return any(key == prefix or key.startswith(f"{prefix}.") for prefix in prefixes)


def _filter_state_dict_by_prefix(state_dict: Dict[str, torch.Tensor], prefixes):
    return {k: v for k, v in state_dict.items() if _matches_prefix(k, prefixes)}


def _strip_predictor_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(key.startswith(PREDICTOR_PREFIX) for key in state_dict):
        return {
            key[len(PREDICTOR_PREFIX):]: value
            for key, value in state_dict.items()
            if key.startswith(PREDICTOR_PREFIX)
        }
    return state_dict


# ==============================================================================
# Config
# ==============================================================================

@dataclass
class TrainConfig:
    # ---- data ----
    cache_dir:        str = "cache/contact_tokens/train"
    val_cache_dir:    str = "cache/contact_tokens/val"
    config_path:      str = "configs/token_config.yaml"
    mano_assets_root: str = "assets/mano_v1_2"
    n_pc_points:      int = 10000
    oi_category:      str = "all"
    oi_intent:        str = "all"
    bps_path:         str = "configs/bps.npz"

    # ---- model ----
    model_config: str = "base"
    n_betas:      int = 10
    vocab_size:   int = 26
    kl_coef:      float = 0.005

    # ---- BPS basis / slot supervision ----
    bps_slot_cache_dir:   str   = "cache/bps_slot"
    predictor_ckpt:       str   = "checkpoints/predictor/best_predictor.pt"

    # ---- RefineNet (joint training) ----
    refine_n_iters: int   = 3
    refine_h_size:  int   = 512
    refine_n_neurons: int = 512
    refine_lr:      float = 5e-4    # same as CoarseNet (GrabNet uses same lr for both)
    refine_pose_noise_std: float = 0.1
    refine_trans_noise_std: float = 0.01

    # ---- Part-Contact Consistency Loss ----
    w_part_contact: float = 10.0

    # ---- training ----
    epochs:            int   = 200
    batch_size:        int   = 128
    lr:                float = 5e-4    # GrabNet default: 5e-4
    weight_decay:      float = 5e-4   # GrabNet: reg_coef = 0.0005
    max_grad_norm:     float = 0.0  # 0 = disabled (GrabNet does not clip gradients)

    # ---- ablation ----
    no_token_cond:      bool  = False   # zero out tok_summary for ablation
    no_part_target_pos: bool  = False   # zero slot-position term, keep token hist + valid mask
    no_refine:          bool  = False   # disable RefineNet training/validation

    # ---- augmentation ----
    token_drop_prob:    float = 0.0
    token_sub_prob:     float = 0.005
    token_dropout_pivot: int = 6
    contact_thresh:     float = 0.005
    token_eval_seed_base: int = 12345

    # ---- logging / save ----
    log_dir:    str = "logs/decoder"
    save_dir:   str = "checkpoints/decoder"
    log_every:  int = 50
    val_every:  int = 1000
    save_every: int = 5000
    save_top_k: int = 5
    early_stop_patience: int = 30

    # ---- resume ----
    resume: str = ""

    # ---- hardware ----
    num_workers: int = 4
    seed:        int = 42
    device:      str = "cuda"


# ==============================================================================
# OIShape helper
# ==============================================================================

class OIShapeConfig:
    def __init__(self, split, category="all", intent="all"):
        self.DATA_SPLIT  = split
        self.OBJ_CATES   = category
        self.INTENT_MODE = intent


def _load_oishape(split, category="all", intent="all", n_samples=2048):
    from data.oishape_dataset import OIShape
    cfg = OIShapeConfig(split, category, intent)
    oi = OIShape(cfg)
    oi.n_samples = n_samples
    return oi


# ==============================================================================
# Dataset
# ==============================================================================

class DecoderDataset(Dataset):
    def __init__(
        self,
        cache_dir:   str,
        config_path: str,
        split:       str = "train",
        n_pc_points: int = 10000,
        category:    str = "all",
        intent:      str = "all",
        bps_path:    str = "configs/bps.npz",
        bps_slot_cache_dir: str = "",
        require_bps_nn_points: bool = False,
    ):
        self.cache_dir = cache_dir
        self.encoder   = TokenEncoder(config_path)
        self.mapper    = self.encoder.mapper
        self.n_pc_points = n_pc_points
        self.require_bps_nn_points = require_bps_nn_points

        with open(os.path.join(cache_dir, "dataset_index.json")) as f:
            index = json.load(f)
        self.samples = [s for s in index["samples"] if s["token_length"] >= 1]

        logging.info(f"[DecoderDataset] Loading OIShape split={split}, "
                     f"n_samples={n_pc_points} ...")
        self.oishape = _load_oishape(
            split, category, intent,
            n_samples=n_pc_points,
        )
        logging.info(f"[DecoderDataset] OIShape: {len(self.oishape)} grasps")

        max_idx = len(self.oishape) - 1
        before = len(self.samples)
        self.samples = [s for s in self.samples if s["idx"] <= max_idx]
        if len(self.samples) < before:
            logging.warning(f"  Filtered {before - len(self.samples)} out-of-range samples")

        self._npz_cache = {}

        # ---- BPS slot cache: per-object npz files ----
        self._bps_slot_cache = {}
        self._bps_slot_dir = ""
        self._bps_has_nn_points = False
        if bps_slot_cache_dir:
            slot_dir = os.path.join(bps_slot_cache_dir, split)
            if os.path.isdir(slot_dir):
                self._bps_slot_dir = slot_dir
                self._bps_has_nn_points = self._probe_bps_slot_cache_field("bps_nn_points")
                logging.info(f"[DecoderDataset] BPS slot cache: {slot_dir}")
                if self.require_bps_nn_points and not self._bps_has_nn_points:
                    raise RuntimeError(
                        "[DecoderDataset] BPS slot cache does not contain bps_nn_points. "
                        "Run `python preprocess/build_bps_slot_cache.py` to regenerate the cache."
                    )
            else:
                raise FileNotFoundError(
                    f"BPS slot cache directory not found: {slot_dir}\n"
                    f"Run `python preprocess/build_bps_slot_cache.py` first."
                )

        # Build obj_id index from OIShape grasp_list
        self._idx_to_obj_id = {}
        if hasattr(self.oishape, 'grasp_list'):
            for gi, g in enumerate(self.oishape.grasp_list):
                self._idx_to_obj_id[gi] = g.get("obj_id", "")

        logging.info(f"[DecoderDataset] {len(self.samples)} samples")

    def _load_npz(self, cache_file):
        if cache_file not in self._npz_cache:
            data = np.load(os.path.join(self.cache_dir, cache_file),
                           allow_pickle=True)
            self._npz_cache[cache_file] = {
                "token_seq": data["token_seq"].astype(np.int64),
            }
        return self._npz_cache[cache_file]

    def _probe_bps_slot_cache_field(self, field_name: str) -> bool:
        if not self._bps_slot_dir:
            return False
        for fname in sorted(os.listdir(self._bps_slot_dir)):
            if not fname.endswith(".npz"):
                continue
            npz_path = os.path.join(self._bps_slot_dir, fname)
            with np.load(npz_path, allow_pickle=True) as data:
                return field_name in data.files
        return False

    def _load_bps_slot(self, obj_id: str):
        """Load per-object BPS slot cache.

        Returns:
            dict with 'bps_dists', 'bps_slot_labels', optional 'bps_nn_points',
            'obj_scale', or None.
        """
        if not self._bps_slot_dir or not obj_id:
            return None
        if obj_id in self._bps_slot_cache:
            return self._bps_slot_cache[obj_id]

        npz_path = os.path.join(self._bps_slot_dir, f"{obj_id}.npz")
        if not os.path.exists(npz_path):
            self._bps_slot_cache[obj_id] = None
            return None

        data = np.load(npz_path, allow_pickle=True)
        result = {
            "bps_dists":       data["bps_dists"].astype(np.float32),      # (4096,)
            "bps_slot_labels": data["bps_slot_labels"].astype(np.int64),  # (4096,)
            "bps_nn_points": (
                data["bps_nn_points"].astype(np.float32)                  # (4096, 3)
                if "bps_nn_points" in data.files else None
            ),
            "obj_scale":       float(data["obj_scale"]),
        }
        self._bps_slot_cache[obj_id] = result
        return result

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        entry = self.samples[i]
        oi    = self.oishape[entry["idx"]]
        tok   = self._load_npz(entry["cache_file"])

        token_seq  = tok["token_seq"]
        max_len    = self.encoder.max_token_length
        token_mask = np.zeros(max_len, dtype=bool)
        for j, t in enumerate(token_seq):
            if j < max_len and self.mapper.is_semantic_token(int(t)):
                token_mask[j] = True

        result = {
            "token_seq":   torch.from_numpy(token_seq),
            "token_mask":  torch.from_numpy(token_mask),
            "obj_pc":      torch.from_numpy(oi["obj_verts"]),
            "obj_vn":      torch.from_numpy(oi["obj_vn"]),
            "hand_pose":   torch.from_numpy(oi["hand_pose"]),
            "hand_shape":  torch.from_numpy(oi["hand_shape"]),
            "hand_tsl":    torch.from_numpy(oi["hand_tsl"]),
            "hand_verts":  torch.from_numpy(oi["hand_verts"]),
        }

        # BPS encoding from slot cache (required)
        obj_id = self._idx_to_obj_id.get(entry["idx"], "")
        bps_slot_data = self._load_bps_slot(obj_id)

        if bps_slot_data is not None:
            result["bps_object"] = torch.from_numpy(bps_slot_data["bps_dists"])
            result["bps_slot_labels"] = torch.from_numpy(bps_slot_data["bps_slot_labels"])
            if self._bps_has_nn_points:
                bps_nn_points = bps_slot_data["bps_nn_points"]
                if self.require_bps_nn_points and bps_nn_points is None:
                    raise RuntimeError(
                        f"Mixed BPS slot cache formats detected for obj_id='{obj_id}'. "
                        "Please regenerate the full cache with "
                        "`python preprocess/build_bps_slot_cache.py`."
                    )
                if bps_nn_points is None:
                    bps_nn_points = np.zeros(
                        (bps_slot_data["bps_dists"].shape[0], 3),
                        dtype=np.float32,
                    )
                result["bps_nn_points"] = torch.from_numpy(bps_nn_points)
            result["obj_scale"] = torch.tensor(bps_slot_data["obj_scale"], dtype=torch.float32)
        else:
            raise RuntimeError(
                f"BPS slot cache not found for obj_id='{obj_id}' "
                f"(sample idx={entry['idx']}, cache_file={entry['cache_file']}). "
                f"Run `python preprocess/build_bps_slot_cache.py` first."
            )

        # GT rotmat for CoarseNet encoder
        hand_pose = oi["hand_pose"]
        hand_pose_t = torch.from_numpy(hand_pose).float()
        from utils.pose_utils import _aa_to_rotmat_stable
        global_orient_aa = hand_pose_t[:3].unsqueeze(0)
        global_orient_rotmat = _aa_to_rotmat_stable(global_orient_aa).squeeze(0)
        result["global_orient_rhand_rotmat"] = global_orient_rotmat.reshape(-1)  # (9,)

        return result


def _collate(batch):
    keys = list(batch[0].keys())
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


# ==============================================================================
# LR Schedule
# ==============================================================================

def cosine_warmup_schedule(optimizer, warmup, total):
    total = max(int(total), 1)
    warmup = max(0, min(int(warmup), total - 1))

    def fn(step):
        step = min(int(step) + 1, total)
        if warmup > 0 and step <= warmup:
            return step / warmup
        p = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


# ==============================================================================
# Trainer
# ==============================================================================

class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg         = cfg
        self.device      = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.global_step = 0
        self.epoch       = 0
        self.best_loss_cnet = float("inf")
        self.best_loss_rnet = float("inf")
        self.no_improve_cnet = 0
        self.no_improve_rnet = 0
        self.fit_cnet = True
        self.fit_rnet = not cfg.no_refine
        self.current_kl_coef = 0.0

        self._set_seed(cfg.seed)
        os.makedirs(cfg.save_dir, exist_ok=True)
        os.makedirs(cfg.log_dir,  exist_ok=True)
        self._setup_logging()
        self.mano = MANOHelper(
            mano_assets_root=cfg.mano_assets_root,
            device=str(self.device))

        self._build_data()
        self._build_model()
        self._build_optimizer()

        if cfg.resume and os.path.isfile(cfg.resume):
            self._load_checkpoint(cfg.resume)

        self._apply_current_kl_coef()

        # Load vertex weights and edge pairs for GrabNet-style loss
        self._load_loss_assets()

        # Part-Contact Consistency Loss setup
        self._setup_part_contact()

        self._log_info()

    def _set_seed(self, s):
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(self.cfg.log_dir, "train.log")),
                logging.StreamHandler()])
        self.logger = logging.getLogger(__name__)
        self.tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(self.cfg.log_dir)
        except ImportError:
            pass

    def _build_data(self):
        cfg = self.cfg
        require_bps_nn_points = True
        self.train_ds = DecoderDataset(
            cfg.cache_dir, cfg.config_path, "train",
            n_pc_points=cfg.n_pc_points,
            category=cfg.oi_category, intent=cfg.oi_intent,
            bps_path=cfg.bps_path,
            bps_slot_cache_dir=cfg.bps_slot_cache_dir,
            require_bps_nn_points=require_bps_nn_points)

        self.train_dl = DataLoader(
            self.train_ds, batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True,
            drop_last=True, collate_fn=_collate)

        self.val_dl = None
        if cfg.val_cache_dir and os.path.isdir(cfg.val_cache_dir):
            val_ds = DecoderDataset(
                cfg.val_cache_dir, cfg.config_path, "val",
                n_pc_points=cfg.n_pc_points,
                category=cfg.oi_category, intent=cfg.oi_intent,
                bps_path=cfg.bps_path,
                bps_slot_cache_dir=cfg.bps_slot_cache_dir,
                require_bps_nn_points=require_bps_nn_points)
            self.val_dl = DataLoader(
                val_ds, batch_size=cfg.batch_size, shuffle=False,
                num_workers=cfg.num_workers, pin_memory=True,
                drop_last=True, collate_fn=_collate)

    def _build_model(self):
        cfg = self.cfg
        self.model = build_grab_model(
            cfg.model_config,
            vocab_size=cfg.vocab_size,
            n_betas=cfg.n_betas,
            kl_coef=cfg.kl_coef,
            refine_n_iters=cfg.refine_n_iters,
            refine_n_neurons=cfg.refine_n_neurons,
        ).to(self.device)
        if cfg.no_token_cond:
            self.model.no_token_cond = True
        if cfg.no_part_target_pos:
            self.model.no_part_target_pos = True
        self.slot_predictor = None
        self.predictor_ckpt_path = None
        self.slot_label_source = "gt"
        if cfg.predictor_ckpt:
            self._build_slot_predictor(cfg.predictor_ckpt)
            self.slot_label_source = "predictor"

    def _build_slot_predictor(self, ckpt_path: str):
        cfg = self.cfg
        resolved_ckpt = ckpt_path
        if not os.path.isfile(resolved_ckpt):
            fallback_candidates = [
                ckpt_path.replace(
                    "checkpoints/predictor/",
                    "checkpoints/full/predictor/",
                ),
                ckpt_path.replace(
                    "checkpoints/full/predictor/",
                    "checkpoints/predictor/",
                ),
            ]
            for candidate in fallback_candidates:
                if candidate != resolved_ckpt and os.path.isfile(candidate):
                    resolved_ckpt = candidate
                    break
        if not os.path.isfile(resolved_ckpt):
            raise FileNotFoundError(
                f"Predictor checkpoint not found: {ckpt_path}\n"
                "Train it first with `python dhoi/train_predictor.py`."
            )
        if resolved_ckpt != ckpt_path:
            self.logger.warning(
                f"Predictor checkpoint not found at {ckpt_path}; using {resolved_ckpt} instead."
            )
        self.predictor_ckpt_path = resolved_ckpt

        ckpt = torch.load(resolved_ckpt, map_location=self.device)
        raw_state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        predictor_state = _strip_predictor_prefix(raw_state)
        predictor_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

        bps_data = np.load(cfg.bps_path)
        basis = bps_data["basis"].astype(np.float32)
        if basis.ndim == 3:
            basis = basis.squeeze(0)

        self.slot_predictor = BPSSlotPredictor(
            hidden_dim=int(predictor_cfg.get("hidden_dim", 256)),
            basis_points=basis,
            num_neighbors=int(predictor_cfg.get("num_neighbors", 16)),
            num_layers=int(predictor_cfg.get("num_layers", 2)),
            dropout=float(predictor_cfg.get("dropout", 0.0)),
            chunk_size=int(predictor_cfg.get("chunk_size", 256)),
        ).to(self.device)
        missing, unexpected = self.slot_predictor.load_state_dict(predictor_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Predictor checkpoint is incompatible with current BPSSlotPredictor "
                f"(missing={len(missing)}, unexpected={len(unexpected)})."
            )
        self.slot_predictor.eval()
        for param in self.slot_predictor.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def _predict_bps_slot_labels(
        self,
        bps_object: torch.Tensor,
        bps_nn_points: torch.Tensor,
    ) -> torch.Tensor:
        if self.slot_predictor is None:
            raise RuntimeError("slot predictor is not initialized")
        return self.slot_predictor.predict(
            bps_object,
            bps_nn_points=bps_nn_points,
        )

    @torch.no_grad()
    def _build_obj_slot_labels(
        self,
        obj_pc: torch.Tensor,
        obj_scale: torch.Tensor,
        bps_object: torch.Tensor,
        bps_nn_points: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.bps_basis is None:
            return None
        bps_slot_labels = self._predict_bps_slot_labels(bps_object, bps_nn_points)
        return propagate_slot_labels(
            obj_pc,
            obj_scale,
            self.bps_basis,
            bps_slot_labels,
        )

    def _build_optimizer(self):
        cfg = self.cfg

        # CoarseNet params only; RefineNet keeps its separate optimizer.
        coarse_params = [p for n, p in self.model.named_parameters()
                         if p.requires_grad and not n.startswith("refine_net.")]
        self.optimizer = torch.optim.Adam(
            coarse_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min"
        )

        # RefineNet optimizer (separate, following GrabNet)
        if cfg.no_refine:
            self.optimizer_refine = None
            self.scheduler_refine = None
        else:
            self.optimizer_refine = torch.optim.Adam(
                self.model.refine_net.parameters(), lr=cfg.refine_lr,
                weight_decay=cfg.weight_decay)
            self.scheduler_refine = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_refine, mode="min"
            )

    def _compute_current_kl_coef(self, epoch: float | None = None) -> float:
        del epoch
        return float(self.cfg.kl_coef)

    def _apply_current_kl_coef(self, epoch: float | None = None) -> float:
        current_kl = self._compute_current_kl_coef(epoch)
        self.model.kl_coef = current_kl
        self.current_kl_coef = current_kl
        return current_kl

    def _coarse_patience_start_epoch(self) -> float:
        return 0.0

    def _is_coarse_patience_active(self, epoch: float | None = None) -> bool:
        if epoch is None:
            epoch = float(self.epoch)
        return float(epoch) >= self._coarse_patience_start_epoch()

    def _load_loss_assets(self):
        """Load vertex weights and edge pairs for GrabNet-style loss."""
        cfg = self.cfg

        # Reuse the already-initialized MANOHelper instead of depending on a
        # separate `mano.load(...)` API, which varies across environments.
        mano_faces = getattr(self.mano, "faces", None)
        if mano_faces is None:
            raise AttributeError(
                "MANOHelper does not expose `faces`; cannot build GrabNet losses."
            )
        rh_f = torch.from_numpy(np.asarray(mano_faces, dtype=np.int32)).view(1, -1, 3)
        self.rh_faces = rh_f.repeat(cfg.batch_size, 1, 1).to(self.device).long()

        # Vertex weights (try to load, fallback to ones)
        c_weights_path = os.path.join("configs", "c_weights.npy")
        if os.path.exists(c_weights_path):
            v_weights = torch.from_numpy(np.load(c_weights_path)).float().to(self.device)
            self.v_weights = v_weights
            self.v_weights2 = torch.pow(v_weights, 1.0 / 2.5)
        else:
            self.v_weights = torch.ones(778, device=self.device)
            self.v_weights2 = torch.ones(778, device=self.device)
            self.logger.warning("c_weights.npy not found, using uniform vertex weights")

        # Edge pairs
        vpe_path = os.path.join("configs", "vpe.npy")
        if os.path.exists(vpe_path):
            self.vpe = torch.from_numpy(np.load(vpe_path)).to(self.device).long()
        else:
            self.vpe = None
            self.logger.warning("vpe.npy not found, edge loss disabled")

        # BPS basis for slot label propagation
        if os.path.exists(cfg.bps_path):
            bps_basis = np.load(cfg.bps_path)['basis']
            if bps_basis.ndim == 3:
                bps_basis = bps_basis.squeeze(0)
            self.bps_basis = torch.from_numpy(bps_basis).float().to(self.device)
        else:
            self.bps_basis = None
            self.logger.warning(f"BPS basis not found at {cfg.bps_path}")

    def _setup_part_contact(self):
        """Setup Part-Contact Consistency Loss."""
        self.vert_to_part = build_vert_to_part(self.mano).to(self.device)
        self.pc_part_masks = build_part_contact_masks(self.mano).to(self.device)
        self.logger.info(f"[Part-Contact] Built vertex→part mapping (778 → {NUM_HAND_PARTS} parts)")

        # Log full-part counts and the compact distal/palm supervision subsets.
        part_names = ["THUMB", "INDEX", "MIDDLE", "RING", "LITTLE", "PALM"]
        for p in range(NUM_HAND_PARTS):
            cnt = (self.vert_to_part == p).sum().item()
            self.logger.info(f"  {part_names[p]}: {cnt} vertices")
        self.logger.info("[Part-Contact] Supervision subset (distal fingers + palm)")
        for p in range(NUM_HAND_PARTS):
            cnt = self.pc_part_masks[p].sum().item()
            self.logger.info(f"  {part_names[p]}: {cnt} vertices")

    def _log_info(self):
        n_model = self.model.count_parameters()
        cfg = self.cfg
        self.logger.info("=" * 60)
        self.logger.info("PoseGrabModel — GrabNet-style CVAE + RefineNet (Joint Training)")
        self.logger.info(f"  CoarseNet:     {n_model:,} params ({n_model/1e6:.2f}M)")

        n_refine = sum(p.numel() for p in self.model.refine_net.parameters() if p.requires_grad)
        self.logger.info(f"  RefineNet:     {n_refine:,} params ({n_refine/1e6:.2f}M)")
        self.logger.info(
            f"  Refine init:   GT + noise (pose_std={cfg.refine_pose_noise_std:.4f}, "
            f"trans_std={cfg.refine_trans_noise_std:.4f})"
        )

        self.logger.info(f"  Output:        rot6d={ROT6D_POSE_DIM} + trans={TRANS_DIM} (no shape)")
        self.logger.info(f"  Latent z:      {self.model.latentD}")
        self.logger.info("  Posterior GT:  global orient rot6d + translation")
        self.logger.info(f"  Vocab size:    {cfg.vocab_size}")
        self.logger.info(f"  KL coef:       fixed {cfg.kl_coef}")
        self.logger.info(f"  Part-Contact w: {cfg.w_part_contact}")
        self.logger.info(f"  Slot labels:   {self.slot_label_source}")
        if self.slot_predictor is not None:
            self.logger.info(f"  Predictor ckpt:{getattr(self, 'predictor_ckpt_path', cfg.predictor_ckpt)}")
        self.logger.info(
            f"  Token eval:    single fixed prior seed ({cfg.token_eval_seed_base})"
        )
        if cfg.no_token_cond:
            self.logger.info("  *** ABLATION: token conditioning DISABLED ***")
        if cfg.no_part_target_pos:
            self.logger.info("  *** ABLATION: part_target_pos ZEROED (valid_mask kept) ***")
        if cfg.no_refine:
            self.logger.info("  *** ABLATION: RefineNet DISABLED ***")
        self.logger.info("  LR schedule:   ReduceLROnPlateau per stage (GrabNet-style)")
        self.logger.info(
            f"  Token aug:     drop_p={cfg.token_drop_prob}  "
            f"sub_p={cfg.token_sub_prob}  pivot={cfg.token_dropout_pivot}"
        )
        if cfg.no_refine:
            self.logger.info("  Best ckpt:     save best_cnet.pt by val coarse loss")
        else:
            self.logger.info(
                "  Best ckpt:     save best_cnet.pt by val coarse loss, "
                "and best_rnet.pt by val refine loss"
            )
        self.logger.info("  Coarse ES:     track val coarse total_loss, active immediately (fixed KL)")
        self.logger.info(f"  Train:         {len(self.train_ds)} samples, "
                         f"{len(self.train_dl)} steps/epoch")
        self.logger.info("=" * 60)

    # ================================================================
    # Train step — CoarseNet CVAE
    # ================================================================

    @torch.no_grad()
    def _build_refine_noisy_init(
        self,
        gt_pose: torch.Tensor,
        gt_trans: torch.Tensor,
        obj_pc: torch.Tensor,
        obj_vn: torch.Tensor,
        shape: torch.Tensor,
    ):
        """Build RefineNet inputs from GT pose/trans with small Gaussian noise."""
        cfg = self.cfg

        noisy_pose = gt_pose
        noisy_trans = gt_trans
        if cfg.refine_pose_noise_std > 0:
            noisy_pose = noisy_pose + torch.randn_like(noisy_pose) * cfg.refine_pose_noise_std
        if cfg.refine_trans_noise_std > 0:
            noisy_trans = noisy_trans + torch.randn_like(noisy_trans) * cfg.refine_trans_noise_std

        noisy_verts, _ = self.mano(noisy_pose, noisy_trans, shape)
        noisy_rot6d = aa_to_rot6d(noisy_pose)
        h2o_dist = RefineNet._compute_h2o_dist(noisy_verts, obj_pc, obj_vn)
        return noisy_pose, noisy_trans, noisy_rot6d, noisy_verts, h2o_dist

    def _train_step_coarse(self, batch) -> Dict[str, float]:
        cfg = self.cfg
        current_kl = self.current_kl_coef

        bps_object = batch["bps_object"].to(self.device)
        bps_slot_labels = batch["bps_slot_labels"].to(self.device)
        bps_nn_points = batch["bps_nn_points"].to(self.device)
        tokens     = batch["token_seq"].to(self.device)
        t_mask     = batch["token_mask"].to(self.device)
        gt_trans   = batch["hand_tsl"].to(self.device)
        gt_pose    = batch["hand_pose"].to(self.device)
        gt_pose_rot6d = aa_to_rot6d(gt_pose)
        obj_pc     = batch["obj_pc"].to(self.device)
        obj_vn     = batch["obj_vn"].to(self.device)
        obj_scale  = batch["obj_scale"].to(self.device)

        # ---- Augmentation ----
        if cfg.token_drop_prob > 0:
            t_mask = apply_token_dropout(
                t_mask,
                cfg.token_drop_prob,
                adaptive_pivot=cfg.token_dropout_pivot,
            )
        if cfg.token_sub_prob > 0:
            tokens = apply_token_substitution(
                tokens, t_mask, cfg.token_sub_prob,
                n_semantic=cfg.vocab_size - 2,
            )

        self.optimizer.zero_grad()

        # ---- Token summary (semantic conditioning) ----
        tok_summary = self.model.encode_token_condition(
            tokens,
            t_mask,
            bps_slot_labels=bps_slot_labels,
            bps_nn_points=bps_nn_points,
        )

        # ---- CoarseNet forward ----
        drec = self.model.forward_coarse(
            bps_object,
            tok_summary,
            gt_pose_rot6d,
            gt_trans,
        )

        # ---- MANO forward: both pred and GT use mean shape (zeros) ----
        B = gt_trans.shape[0]
        zeros_shape = torch.zeros(B, cfg.n_betas, device=self.device)

        with torch.cuda.amp.autocast(enabled=False):
            drec_float32 = {k: v.float() if isinstance(v, torch.Tensor) else v
                           for k, v in drec.items()}
            verts_pred, _ = self.mano(
                drec_float32["fullpose_aa"],
                drec_float32["transl"],
                zeros_shape,
            )

        # Recompute GT verts with mean shape to match inference
        with torch.no_grad():
            gt_verts, _ = self.mano(gt_pose, gt_trans, zeros_shape)

        # ---- CoarseNet Loss ----
        dorig = {"vpe": self.vpe} if self.vpe is not None else {}
        rh_faces = self.rh_faces[:B]

        loss_total, loss_dict = self.model.compute_coarse_loss(
            drec, dorig, verts_pred, gt_verts, obj_pc,
            rh_faces, self.v_weights, self.v_weights2, B,
            obj_normals=obj_vn,
        )

        # ---- Part-Contact Consistency Loss ----
        if cfg.w_part_contact > 0:
            obj_slot_labels = self._build_obj_slot_labels(
                obj_pc,
                obj_scale,
                bps_object,
                bps_nn_points,
            )

            loss_pc = compute_part_contact_loss(
                hand_verts=verts_pred,
                obj_pc=obj_pc,
                tokens=tokens,
                token_mask=t_mask,
                part_vert_masks=self.pc_part_masks,
                obj_slot_labels=obj_slot_labels,
            )
            loss_total = loss_total + cfg.w_part_contact * loss_pc
            loss_dict["loss_part_contact"] = loss_pc

        loss_dict["loss_total"] = loss_total
        loss_dict["kl_coef_cur"] = float(current_kl)

        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v
                   for k, v in loss_dict.items()}

        loss_total.backward()
        if cfg.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
        self.optimizer.step()

        return metrics

    # ================================================================
    # Train step — RefineNet (GT + noise initialization)
    # ================================================================

    def _train_step_refine(self, batch) -> Dict[str, float]:
        """RefineNet training step from GT pose/trans plus small Gaussian noise."""
        cfg = self.cfg

        obj_pc     = batch["obj_pc"].to(self.device)
        obj_vn     = batch["obj_vn"].to(self.device)
        gt_pose    = batch["hand_pose"].to(self.device)
        gt_trans   = batch["hand_tsl"].to(self.device)

        B = gt_pose.shape[0]
        zeros_shape = torch.zeros(B, cfg.n_betas, device=self.device)

        # ---- Compute GT verts & h2o distances with mean shape ----
        with torch.no_grad():
            gt_verts, _ = self.mano(gt_pose, gt_trans, zeros_shape)
            h2o_gt = RefineNet._compute_h2o_dist(gt_verts, obj_pc, obj_vn)
            _, coarse_trans, coarse_rot6d, _, h2o_dist = self._build_refine_noisy_init(
                gt_pose, gt_trans, obj_pc, obj_vn, zeros_shape,
            )

        # ---- RefineNet forward ----
        self.optimizer_refine.zero_grad()
        result = self.model.refine_net(
            h2o_dist, coarse_rot6d, coarse_trans,
            obj_pc, self.mano, shape=zeros_shape, obj_normals=obj_vn,
        )

        # Compute refined vertices
        refined_verts, _ = self.mano(result["pose"], result["trans"], zeros_shape)

        # ---- GrabNet-style RefineNet loss ----
        loss, metrics = compute_refine_loss(
            pred_verts=refined_verts,
            gt_verts=gt_verts,
            obj_pc=obj_pc,
            v_weights=self.v_weights,
            v_weights2=self.v_weights2,
            vpe=self.vpe,
            kl_coef=self.current_kl_coef,
            h2o_gt=h2o_gt,
            obj_normals=obj_vn,
        )

        loss.backward()
        if cfg.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model.refine_net.parameters(), cfg.max_grad_norm)
        self.optimizer_refine.step()

        return {f"rnet_{k}": v for k, v in metrics.items()}

    # ================================================================
    # Combined train step
    # ================================================================

    def _train_step(self, batch) -> Dict[str, float]:
        """Joint training step: CoarseNet + RefineNet."""
        self._apply_current_kl_coef()
        coarse_metrics = {}
        if self.fit_cnet:
            coarse_metrics = self._train_step_coarse(batch)

        refine_metrics = {}
        if self.fit_rnet:
            refine_metrics = self._train_step_refine(batch)

        coarse_metrics.update(refine_metrics)
        return coarse_metrics

    # ================================================================
    # Validation
    # ================================================================

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        if self.val_dl is None:
            return {}
        cfg = self.cfg
        current_kl = self._apply_current_kl_coef()
        self.model.eval()
        if self.fit_rnet:
            self.model.refine_net.eval()
        totals = defaultdict(float)
        n = 0
        token_metric_samples = 0
        token_eval_seed = int(cfg.token_eval_seed_base)

        for batch in self.val_dl:
            bps_object = batch["bps_object"].to(self.device)
            bps_slot_labels = batch["bps_slot_labels"].to(self.device)
            bps_nn_points = batch["bps_nn_points"].to(self.device)
            tokens = batch["token_seq"].to(self.device)
            t_mask = batch["token_mask"].to(self.device)
            gt_trans = batch["hand_tsl"].to(self.device)
            gt_pose = batch["hand_pose"].to(self.device)
            gt_pose_rot6d = aa_to_rot6d(gt_pose)
            obj_pc = batch["obj_pc"].to(self.device)
            obj_vn = batch["obj_vn"].to(self.device)
            obj_scale = batch["obj_scale"].to(self.device)
            B = gt_trans.shape[0]
            zeros_shape = torch.zeros(B, cfg.n_betas, device=self.device)

            # Recompute GT verts with mean shape
            gt_verts, _ = self.mano(gt_pose, gt_trans, zeros_shape)

            obj_slot_labels = self._build_obj_slot_labels(
                obj_pc,
                obj_scale,
                bps_object,
                bps_nn_points,
            )
            tok_summary = self.model.encode_token_condition(
                tokens,
                t_mask,
                bps_slot_labels=bps_slot_labels,
                bps_nn_points=bps_nn_points,
            )
            drec = self.model.forward_coarse(
                bps_object,
                tok_summary,
                gt_pose_rot6d,
                gt_trans,
            )

            verts_pred, _ = self.mano(
                drec["fullpose_aa"].float(),
                drec["transl"].float(),
                zeros_shape,
            )

            dorig = {"vpe": self.vpe} if self.vpe is not None else {}
            rh_faces = self.rh_faces[:B]
            loss_total, loss_dict = self.model.compute_coarse_loss(
                drec, dorig, verts_pred, gt_verts, obj_pc,
                rh_faces, self.v_weights, self.v_weights2, B,
                obj_normals=obj_vn,
            )

            # Sample-based metrics use a single fixed prior seed for determinism.
            pred = self.model.sample(
                bps_object, tokens, t_mask,
                bps_slot_labels=bps_slot_labels,
                bps_nn_points=bps_nn_points,
                seed=token_eval_seed,
            )

            pred_verts, _ = self.mano(
                pred.pose, pred.trans,
                torch.zeros(B, cfg.n_betas, device=self.device),
            )

            _, h2o_signed, _ = point2point_signed(
                pred_verts, obj_pc, y_normals=obj_vn,
            )
            pen_depth = torch.relu(-h2o_signed)
            contact_mask = h2o_signed.abs() < cfg.contact_thresh
            sample_has_contact = contact_mask.any(dim=1).float()

            # RefineNet validation
            if self.fit_rnet:
                _, coarse_trans, coarse_rot6d, _, h2o_dist = self._build_refine_noisy_init(
                    gt_pose, gt_trans, obj_pc, obj_vn, zeros_shape,
                )
                rnet_result = self.model.refine_net(
                    h2o_dist, coarse_rot6d, coarse_trans,
                    obj_pc, self.mano, shape=zeros_shape, obj_normals=obj_vn)
                refined_verts, _ = self.mano(rnet_result["pose"], rnet_result["trans"], zeros_shape)

                # RefineNet loss / geometry (GrabNet-style validation)
                rnet_loss, rnet_metrics = compute_refine_loss(
                    pred_verts=refined_verts,
                    gt_verts=gt_verts,
                    obj_pc=obj_pc,
                    v_weights=self.v_weights,
                    v_weights2=self.v_weights2,
                    vpe=self.vpe,
                    kl_coef=current_kl,
                    obj_normals=obj_vn,
                )
                totals["rnet_loss_total"] += rnet_loss.item() * B
                for rk, rv in rnet_metrics.items():
                    totals[f"rnet_{rk}"] += rv * B

                # Use signed distance for penetration monitoring
                _, refined_h2o_signed, _ = point2point_signed(
                    refined_verts, obj_pc, y_normals=obj_vn)
                pen_refined = torch.relu(-refined_h2o_signed).mean().item() * 1000.0
                contact_refined = (refined_h2o_signed.abs() < cfg.contact_thresh).any(dim=1).float().mean().item()
                totals["pen_refined_mm"] += pen_refined * B
                totals["contact_refined"] += contact_refined * B

            if obj_slot_labels is not None:
                token_eval_mask = (obj_slot_labels >= 0).any(dim=1)
            else:
                token_eval_mask = None

            if token_eval_mask is not None and token_eval_mask.any():
                eval_B = int(token_eval_mask.sum().item())

                def _tok_metrics(verts_eval):
                    cm = recover_contact_matrix_from_obj_slots(
                        verts_eval[token_eval_mask],
                        obj_pc[token_eval_mask],
                        obj_slot_labels[token_eval_mask],
                        self.vert_to_part,
                        contact_distance=cfg.contact_thresh,
                        num_slots=NUM_SLOTS,
                    )
                    return compute_token_recall_f1(
                        cm, tokens[token_eval_mask], t_mask[token_eval_mask]
                    )

                recall, f1 = _tok_metrics(pred_verts)
                totals["token_recall"] += recall.mean().item() * eval_B
                totals["token_f1"] += f1.mean().item() * eval_B

                token_metric_samples += eval_B
            else:
                totals["token_recall"] += 0.0
                totals["token_f1"] += 0.0

            for k, v in loss_dict.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                totals[k] += val * B
            totals["pen_depth_mm"] += pen_depth.mean().item() * 1000.0 * B
            totals["contact_rate"] += sample_has_contact.mean().item() * B
            n += B

        self.model.train()
        if self.fit_rnet:
            self.model.refine_net.train()
        _token_keys = {
            "token_recall",
            "token_f1"
        }
        metrics = {}
        for k, v in totals.items():
            if k in _token_keys:
                metrics[f"val/{k}"] = v / max(token_metric_samples, 1)
            else:
                metrics[f"val/{k}"] = v / max(n, 1)
        metrics["val/token_eval_samples"] = float(token_metric_samples)
        metrics["val/kl_coef"] = float(current_kl)
        return metrics

    # ================================================================
    # Save / Load
    # ================================================================

    def _save_checkpoint(self, tag="latest", stage=None):
        path = os.path.join(self.cfg.save_dir, f"{tag}.pt")
        stage_name = stage or "full"
        model_state = self.model.state_dict()
        if stage_name == "cnet":
            model_state = _filter_state_dict_by_prefix(model_state, COARSE_CKPT_PREFIXES)
        elif stage_name == "rnet":
            model_state = _filter_state_dict_by_prefix(model_state, REFINE_CKPT_PREFIXES)
        elif stage_name != "full":
            raise ValueError(f"Unknown checkpoint stage: {stage_name}")

        save_dict = {
            "stage": stage_name,
            "model": model_state,
            "epoch":       self.epoch,
            "global_step": self.global_step,
            "best_loss_cnet": self.best_loss_cnet,
            "best_loss_rnet": self.best_loss_rnet,
            "no_improve_cnet": self.no_improve_cnet,
            "no_improve_rnet": self.no_improve_rnet,
            "fit_cnet": self.fit_cnet,
            "fit_rnet": self.fit_rnet,
            "config":      self.cfg.__dict__,
        }
        if stage_name in ("full", "cnet"):
            save_dict["optimizer"] = self.optimizer.state_dict()
            save_dict["scheduler"] = self.scheduler.state_dict() if self.scheduler is not None else None
        if stage_name == "full":
            save_dict["optimizer_refine"] = (
                self.optimizer_refine.state_dict() if self.optimizer_refine is not None else None
            )
            save_dict["scheduler_refine"] = (
                self.scheduler_refine.state_dict() if self.scheduler_refine is not None else None
            )
        elif stage_name == "rnet":
            save_dict["optimizer_refine"] = (
                self.optimizer_refine.state_dict() if self.optimizer_refine is not None else None
            )
            save_dict["scheduler_refine"] = (
                self.scheduler_refine.state_dict() if self.scheduler_refine is not None else None
            )

        torch.save(save_dict, path)
        self.logger.info(f"  Saved [{stage_name}]: {path}")

    def _load_checkpoint(self, path):
        self.logger.info(f"Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device)
        ckpt_stage = ckpt.get("stage", "full")
        raw_model_state = ckpt["model"]
        model_state, dropped = sanitize_posegrab_state_dict_for_load(
            raw_model_state, self.model.state_dict()
        )
        if dropped:
            self.logger.warning(
                f"  Dropped {len(dropped)} incompatible checkpoint tensors during load"
            )
            for key, src_shape, dst_shape in dropped[:5]:
                self.logger.warning(
                    f"    shape mismatch: {key} ckpt={src_shape} model={dst_shape}"
                )
        missing, unexpected = self.model.load_state_dict(model_state, strict=False)
        relevant_prefixes = None
        if ckpt_stage == "cnet":
            relevant_prefixes = COARSE_CKPT_PREFIXES
        elif ckpt_stage == "rnet":
            relevant_prefixes = REFINE_CKPT_PREFIXES
        if relevant_prefixes is not None:
            missing = [k for k in missing if _matches_prefix(k, relevant_prefixes)]
            unexpected = [k for k in unexpected if _matches_prefix(k, relevant_prefixes)]
        if missing or unexpected:
            self.logger.warning(f"  Checkpoint compat: missing={len(missing)}, unexpected={len(unexpected)}")
            for k in missing[:5]:
                self.logger.warning(f"    missing: {k}")
            for k in unexpected[:5]:
                self.logger.warning(f"    unexpected: {k}")
        if ckpt_stage in ("full", "cnet"):
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except (KeyError, ValueError) as e:
                self.logger.warning(f"  Optimizer state_dict mismatch (type changed?), skipping: {e}")
            if self.scheduler and "scheduler" in ckpt and ckpt["scheduler"]:
                try:
                    self.scheduler.load_state_dict(ckpt["scheduler"])
                except (KeyError, ValueError) as e:
                    self.logger.warning(f"  Scheduler state_dict mismatch, skipping: {e}")

        if ckpt_stage in ("full", "rnet") and self.optimizer_refine is not None:
            try:
                self.optimizer_refine.load_state_dict(ckpt["optimizer_refine"])
            except (KeyError, ValueError) as e:
                self.logger.warning(f"  Refine optimizer state_dict mismatch, skipping: {e}")
            if self.scheduler_refine and "scheduler_refine" in ckpt and ckpt["scheduler_refine"]:
                try:
                    self.scheduler_refine.load_state_dict(ckpt["scheduler_refine"])
                except (KeyError, ValueError) as e:
                    self.logger.warning(f"  Refine scheduler state_dict mismatch, skipping: {e}")

        self.epoch       = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)
        self.best_loss_cnet = ckpt.get("best_loss_cnet", ckpt.get("best_val", float("inf")))
        self.best_loss_rnet = ckpt.get("best_loss_rnet", float("inf"))
        self.no_improve_cnet = ckpt.get("no_improve_cnet", 0)
        self.no_improve_rnet = ckpt.get("no_improve_rnet", 0)
        self.fit_cnet = ckpt.get("fit_cnet", True)
        self.fit_rnet = ckpt.get("fit_rnet", True)
        if self.cfg.no_refine:
            self.fit_rnet = False
        self.logger.info(
            f"  Resumed [{ckpt_stage}]: epoch={self.epoch}, step={self.global_step}"
        )

    def _cleanup_checkpoints(self):
        files = sorted([
            f for f in os.listdir(self.cfg.save_dir)
            if f.startswith("step_") and f.endswith(".pt")
        ])
        while len(files) > self.cfg.save_top_k:
            os.remove(os.path.join(self.cfg.save_dir, files.pop(0)))

    # ================================================================
    # Main loop
    # ================================================================

    def train(self):
        cfg = self.cfg
        self.model.train()
        if self.fit_rnet:
            self.model.refine_net.train()
        self.logger.info(f"\nStart training: {cfg.epochs} epochs")

        for ep in range(self.epoch, cfg.epochs):
            self.epoch = ep
            self.logger.info(f"Epoch {ep}/{cfg.epochs}")

            epoch_loss = 0.0
            n_steps = 0
            t0 = time.time()

            for batch in self.train_dl:
                log = self._train_step(batch)
                self.global_step += 1
                n_steps += 1
                epoch_loss += log.get("loss_total", 0.0) + log.get("rnet_refine_loss", 0.0)

                if self.global_step % cfg.log_every == 0:
                    parts = [f"  S{self.global_step:06d}"]
                    if self.fit_cnet and "loss_total" in log:
                        pc_loss = f"  pc={log.get('loss_part_contact', 0):.4f}" if cfg.w_part_contact > 0 else ""
                        parts.append(
                            f"coarse: total={log.get('loss_total', 0):.4f}  "
                            f"kl={log.get('loss_kl', 0):.4f}  "
                            f"mesh={log.get('loss_mesh_rec', 0):.4f}  "
                            f"dist_h={log.get('loss_dist_h', 0):.4f}  "
                            f"dist_o={log.get('loss_dist_o', 0):.4f}  "
                            f"edge={log.get('loss_edge', 0):.4f}"
                            f"{pc_loss}"
                        )
                    if self.fit_rnet and "rnet_refine_loss" in log:
                        parts.append(
                            f"refine: total={log.get('rnet_refine_loss', 0):.4f}  "
                            f"dist_h={log.get('rnet_refine_dist_h', 0):.4f}  "
                            f"mesh={log.get('rnet_refine_mesh_rec', 0):.4f}  "
                            f"edge={log.get('rnet_refine_edge', 0):.4f}"
                        )
                    self.logger.info("  | ".join(parts))

                    if self.tb:
                        for k, v in log.items():
                            if isinstance(v, (int, float)):
                                self.tb.add_scalar(f"train/{k}", v, self.global_step)

                if self.global_step % cfg.save_every == 0:
                    self._save_checkpoint("latest")
                    self._save_checkpoint(f"step_{self.global_step:07d}")
                    self._cleanup_checkpoints()

            # --- End of epoch: validate + scheduler step (following GrabNet) ---
            dt = time.time() - t0
            self.logger.info(
                f"  Epoch {ep} done  {dt:.0f}s  "
                f"avg_loss={epoch_loss / max(n_steps, 1):.4f}")

            val = self._validate()
            if val:
                val_token_f1 = val.get("val/token_f1", 0.0)
                val_token_recall = val.get("val/token_recall", 0.0)
                vt_cnet = val.get("val/loss_total", float("inf"))
                pen_mm = val.get("val/pen_depth_mm", 0.0)
                coarse_patience_active = self._is_coarse_patience_active(ep)
                coarse_patience_start = self._coarse_patience_start_epoch()
                self.logger.info(
                    f"  [VAL][Coarse] total={vt_cnet:.4f}  "
                    f"contact={val.get('val/contact_rate', 0):.3f}  "
                    f"pen={pen_mm:.2f}mm  "
                    f"token_f1={val_token_f1:.3f}  "
                    f"token_recall={val_token_recall:.3f}"
                )
                vt_rnet = float("inf")
                if self.fit_rnet:
                    vt_rnet = val.get("val/rnet_loss_total", val.get("val/rnet_refine_loss", float("inf")))
                    self.logger.info(
                        f"  [VAL][Refine] total={vt_rnet:.4f}  "
                        f"contact={val.get('val/contact_refined', 0):.3f}  "
                        f"pen={val.get('val/pen_refined_mm', 0):.2f}mm"
                    )

                if self.tb:
                    for k, v in val.items():
                        self.tb.add_scalar(k, v, self.global_step)

                if self.fit_cnet and self.scheduler is not None:
                    prev_lr = self.optimizer.param_groups[0]['lr']
                    self.scheduler.step(vt_cnet)
                    cur_lr = self.optimizer.param_groups[0]['lr']
                    if cur_lr != prev_lr:
                        self.logger.info(f"  [LR][Coarse] {prev_lr:.2e} -> {cur_lr:.2e}")

                if self.fit_rnet and self.scheduler_refine is not None:
                    prev_lr_r = self.optimizer_refine.param_groups[0]['lr']
                    self.scheduler_refine.step(vt_rnet)
                    cur_lr_r = self.optimizer_refine.param_groups[0]['lr']
                    if cur_lr_r != prev_lr_r:
                        self.logger.info(f"  [LR][Refine] {prev_lr_r:.2e} -> {cur_lr_r:.2e}")

                if self.fit_cnet:
                    if vt_cnet < self.best_loss_cnet:
                        prev_best = self.best_loss_cnet
                        self.best_loss_cnet = vt_cnet
                        self.no_improve_cnet = 0
                        self._save_checkpoint("best_cnet", stage="cnet")
                        if prev_best == float("inf"):
                            self.logger.info(f"  * New best_cnet: val_total={vt_cnet:.4f}")
                        else:
                            self.logger.info(
                                f"  * New best_cnet: val_total={vt_cnet:.4f} "
                                f"(prev={prev_best:.4f})"
                            )
                    else:
                        if coarse_patience_active:
                            self.no_improve_cnet += 1
                            remaining = max(cfg.early_stop_patience - self.no_improve_cnet, 0)
                            self.logger.info(
                                f"  [EARLY STOP][Coarse] no total loss improvement for {self.no_improve_cnet}/"
                                f"{cfg.early_stop_patience} (remaining={remaining})"
                            )
                            if self.no_improve_cnet >= cfg.early_stop_patience:
                                self.fit_cnet = False
                                self.logger.info("  [EARLY STOP][Coarse] disabled further CoarseNet updates.")
                        else:
                            self.no_improve_cnet = 0
                            self.logger.info(
                                f"  [EARLY STOP][Coarse] patience inactive during KL warmup "
                                f"(starts at epoch {coarse_patience_start:.0f})"
                            )

                if self.fit_rnet:
                    if vt_rnet < self.best_loss_rnet:
                        prev_best_r = self.best_loss_rnet
                        self.best_loss_rnet = vt_rnet
                        self.no_improve_rnet = 0
                        self._save_checkpoint("best_rnet", stage="rnet")
                        if prev_best_r == float("inf"):
                            self.logger.info(f"  * New best_rnet: total={vt_rnet:.4f}")
                        else:
                            self.logger.info(
                                f"  * New best_rnet: total={vt_rnet:.4f} (prev={prev_best_r:.4f})"
                            )
                    else:
                        self.no_improve_rnet += 1
                        remaining = max(cfg.early_stop_patience - self.no_improve_rnet, 0)
                        self.logger.info(
                            f"  [EARLY STOP][Refine] no improvement for {self.no_improve_rnet}/"
                            f"{cfg.early_stop_patience} (remaining={remaining})"
                        )
                        if self.no_improve_rnet >= cfg.early_stop_patience:
                            self.fit_rnet = False
                            self.logger.info("  [EARLY STOP][Refine] disabled further RefineNet updates.")

            if not self.fit_cnet and not self.fit_rnet:
                self.logger.info("Early stopping triggered: both CoarseNet and RefineNet are inactive.")
                break

        self._save_checkpoint("final")
        self.logger.info("Training complete.")


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train PoseGrabModel (CVAE + RefineNet joint)")

    p.add_argument("--cache_dir",     type=str, default="cache/contact_tokens/train")
    p.add_argument("--val_cache_dir", type=str, default="cache/contact_tokens/val")
    p.add_argument("--n_pc_points",   type=int, default=10000)
    p.add_argument("--bps_path",      type=str, default="configs/bps.npz")

    p.add_argument("--model_config",  type=str, default="base")
    p.add_argument("--kl_coef",       type=float, default=0.005)

    # Slot supervision / part-contact auxiliaries
    p.add_argument("--bps_slot_cache_dir", type=str, default="cache/bps_slot",
                   help="Directory for per-object BPS slot cache")
    p.add_argument("--predictor_ckpt", type=str, default="checkpoints/full/predictor/best_predictor.pt",
                   help="Frozen BPSSlotPredictor checkpoint used to predict slot labels")

    # RefineNet
    p.add_argument("--refine_n_iters", type=int, default=3)
    p.add_argument("--refine_lr", type=float, default=5e-4)
    p.add_argument("--refine_pose_noise_std", type=float, default=0.1,
                   help="Std of Gaussian noise added to GT MANO pose for RefineNet training")
    p.add_argument("--refine_trans_noise_std", type=float, default=0.01,
                   help="Std of Gaussian noise added to GT translation for RefineNet training")

    # Part-Contact Loss
    p.add_argument("--w_part_contact", type=float, default=10.0,
                   help="Weight for Part-Contact Consistency Loss")

    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=5e-4)
    p.add_argument("--resume",       type=str,   default="")
    p.add_argument("--save_dir",     type=str,   default="checkpoints/full/decoder")
    p.add_argument("--log_dir",      type=str,   default="logs/full/decoder")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--token_drop_prob", type=float, default=0.0)
    p.add_argument("--token_sub_prob",  type=float, default=0.005)
    p.add_argument("--token_dropout_pivot", type=int, default=6,
                   help="Adaptive pivot used by token dropout")
    p.add_argument("--contact_thresh",  type=float, default=0.005)
    p.add_argument("--token_eval_seed_base", type=int, default=12345,
                   help="Fixed random seed for deterministic validation prior sampling")

    # Ablation
    p.add_argument("--no_token_cond", action="store_true",
                   help="Zero out token conditioning (ablation test)")
    p.add_argument("--no_part_target_pos", action="store_true",
                   help="Zero out slot-derived part_target_pos while keeping token histogram and valid_mask")
    p.add_argument("--no_refine", action="store_true",
                   help="Disable RefineNet training/validation (coarse-only ablation)")

    return p.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(
        cache_dir=args.cache_dir,
        val_cache_dir=args.val_cache_dir,
        n_pc_points=args.n_pc_points,
        bps_path=args.bps_path,
        model_config=args.model_config,
        kl_coef=args.kl_coef,
        bps_slot_cache_dir=args.bps_slot_cache_dir,
        predictor_ckpt=args.predictor_ckpt,
        refine_n_iters=args.refine_n_iters,
        refine_lr=args.refine_lr,
        refine_pose_noise_std=args.refine_pose_noise_std,
        refine_trans_noise_std=args.refine_trans_noise_std,
        w_part_contact=args.w_part_contact,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume=args.resume,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        seed=args.seed,
        token_drop_prob=args.token_drop_prob,
        token_sub_prob=args.token_sub_prob,
        token_dropout_pivot=args.token_dropout_pivot,
        contact_thresh=args.contact_thresh,
        token_eval_seed_base=args.token_eval_seed_base,
        no_token_cond=args.no_token_cond,
        no_part_target_pos=args.no_part_target_pos,
        no_refine=args.no_refine,
    )
    Trainer(cfg).train()


if __name__ == "__main__":
    main()
