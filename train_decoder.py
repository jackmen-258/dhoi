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
import torch.nn.functional as F
import chamfer_distance as chd
from torch.utils.data import Dataset, DataLoader

from data.token_encoder import TokenEncoder
from models.condition_encoder import build_token_histogram
from models.pose_decoder import (
    build_grab_model, ROT6D_POSE_DIM, TRANS_DIM, NUM_HAND_PARTS, NUM_SLOTS,
    point2point_signed, compute_geo_loss,
    build_vert_to_part, propagate_slot_labels, compute_part_contact_loss,
    RefineNet, compute_refine_loss,
)
from utils.mano_utils import MANOHelper
from utils.pose_utils import aa_to_rot6d, rot6d_to_aa


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

    # ---- BPS basis / auxiliary slot supervision ----
    bps_slot_cache_dir:   str   = "cache/bps_slot"
    slot_grounder_weight: float = 1.0    # CE loss weight for SlotGrounder

    # ---- RefineNet (joint training) ----
    fit_refine:     bool  = True    # Joint training (GrabNet trains both together)
    refine_n_iters: int   = 3
    refine_h_size:  int   = 512
    refine_n_neurons: int = 512
    refine_lr:      float = 5e-4    # same as CoarseNet (GrabNet uses same lr for both)

    # ---- Part-Contact Consistency Loss ----
    w_part_contact: float = 10.0

    # ---- training ----
    epochs:            int   = 100
    batch_size:        int   = 128
    lr:                float = 5e-4    # GrabNet default: 5e-4
    weight_decay:      float = 5e-4   # GrabNet: reg_coef = 0.0005
    warmup_steps:      int   = 1000
    max_grad_norm:     float = 0.0  # 0 = disabled (GrabNet does not clip gradients)

    # ---- ablation ----
    no_token_cond:      bool  = False   # zero out tok_summary for ablation

    # ---- augmentation ----
    token_drop_prob:    float = 0.0
    token_sub_prob:     float = 0.0
    contact_thresh:     float = 0.005

    # ---- logging / save ----
    log_dir:    str = "logs/decoder"
    save_dir:   str = "checkpoints/decoder"
    log_every:  int = 50
    val_every:  int = 1000
    save_every: int = 5000
    save_top_k: int = 5
    early_stop_patience: int = 10

    # ---- best checkpoint selection ----
    best_geom_contact_min: float = 0.90
    best_geom_pen_mm_max:  float = 12.0

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
    ):
        self.cache_dir = cache_dir
        self.encoder   = TokenEncoder(config_path)
        self.mapper    = self.encoder.mapper
        self.n_pc_points = n_pc_points

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
        if bps_slot_cache_dir:
            slot_dir = os.path.join(bps_slot_cache_dir, split)
            if os.path.isdir(slot_dir):
                self._bps_slot_dir = slot_dir
                logging.info(f"[DecoderDataset] BPS slot cache: {slot_dir}")
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

    def _load_bps_slot(self, obj_id: str):
        """Load per-object BPS slot cache: bps_dists(4096) + bps_slot_labels(4096).

        Returns:
            dict with 'bps_dists', 'bps_slot_labels', 'obj_scale', or None.
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
        self.best_val    = float("inf")
        self.best_token_f1 = float("-inf")
        self.no_improve_evals = 0

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
        self.train_ds = DecoderDataset(
            cfg.cache_dir, cfg.config_path, "train",
            n_pc_points=cfg.n_pc_points,
            category=cfg.oi_category, intent=cfg.oi_intent,
            bps_path=cfg.bps_path,
            bps_slot_cache_dir=cfg.bps_slot_cache_dir)

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
                bps_slot_cache_dir=cfg.bps_slot_cache_dir)
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

    def _build_optimizer(self):
        cfg = self.cfg
        total_steps = max(cfg.epochs * len(self.train_dl), 1)

        # CoarseNet + TokenEncoder + SlotGrounder params (exclude refine_net)
        coarse_params = [p for n, p in self.model.named_parameters()
                         if not n.startswith("refine_net.")]
        self.optimizer = torch.optim.Adam(
            coarse_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = cosine_warmup_schedule(
            self.optimizer, cfg.warmup_steps, total_steps,
        )
        for group, base_lr in zip(self.optimizer.param_groups, self.scheduler.base_lrs):
            group["lr"] = base_lr * self.scheduler.lr_lambdas[0](0)

        # RefineNet optimizer (separate, following GrabNet)
        if cfg.fit_refine:
            self.optimizer_refine = torch.optim.Adam(
                self.model.refine_net.parameters(), lr=cfg.refine_lr,
                weight_decay=cfg.weight_decay)
            self.scheduler_refine = cosine_warmup_schedule(
                self.optimizer_refine, cfg.warmup_steps, total_steps,
            )
            for group, base_lr in zip(self.optimizer_refine.param_groups, self.scheduler_refine.base_lrs):
                group["lr"] = base_lr * self.scheduler_refine.lr_lambdas[0](0)
        else:
            self.optimizer_refine = None
            self.scheduler_refine = None

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
        self.logger.info(f"[Part-Contact] Built vertex→part mapping (778 → {NUM_HAND_PARTS} parts)")

        # Log per-part vertex counts
        for p in range(NUM_HAND_PARTS):
            cnt = (self.vert_to_part == p).sum().item()
            part_names = ["THUMB", "INDEX", "MIDDLE", "RING", "LITTLE", "PALM"]
            self.logger.info(f"  {part_names[p]}: {cnt} vertices")

    def _log_info(self):
        n_model = self.model.count_parameters()
        cfg = self.cfg
        self.logger.info("=" * 60)
        self.logger.info("PoseGrabModel — GrabNet-style CVAE + RefineNet (Joint Training)")
        self.logger.info(f"  CoarseNet:     {n_model:,} params ({n_model/1e6:.2f}M)")
        if cfg.fit_refine:
            n_refine = sum(p.numel() for p in self.model.refine_net.parameters() if p.requires_grad)
            self.logger.info(f"  RefineNet:     {n_refine:,} params ({n_refine/1e6:.2f}M)")
        self.logger.info(f"  Output:        rot6d={ROT6D_POSE_DIM} + trans={TRANS_DIM} (no shape)")
        self.logger.info(f"  Vocab size:    {cfg.vocab_size}")
        self.logger.info(f"  KL coef:       {cfg.kl_coef}")
        self.logger.info(f"  Grounder w:    {cfg.slot_grounder_weight}")
        self.logger.info(f"  Part-Contact w: {cfg.w_part_contact}")
        if cfg.no_token_cond:
            self.logger.info("  *** ABLATION: token conditioning DISABLED ***")
        self.logger.info(f"  LR schedule:   warmup_steps={cfg.warmup_steps} + cosine decay")
        self.logger.info(f"  Token aug:     drop_p={cfg.token_drop_prob}  sub_p={cfg.token_sub_prob}")
        self.logger.info(
            f"  Best ckpt:     geom(contact>={cfg.best_geom_contact_min:.2f}, "
            f"pen<={cfg.best_geom_pen_mm_max:.2f}mm) then max token_f1"
        )
        self.logger.info(f"  Train:         {len(self.train_ds)} samples, "
                         f"{len(self.train_dl)} steps/epoch")
        self.logger.info("=" * 60)

    # ================================================================
    # Train step — CoarseNet CVAE
    # ================================================================

    def _train_step_coarse(self, batch) -> Dict[str, float]:
        cfg = self.cfg

        bps_object = batch["bps_object"].to(self.device)
        tokens     = batch["token_seq"].to(self.device)
        t_mask     = batch["token_mask"].to(self.device)
        gt_trans   = batch["hand_tsl"].to(self.device)
        gt_orient_rotmat = batch["global_orient_rhand_rotmat"].to(self.device)
        gt_pose    = batch["hand_pose"].to(self.device)
        obj_pc     = batch["obj_pc"].to(self.device)
        obj_vn     = batch["obj_vn"].to(self.device)
        bps_slot_labels = batch["bps_slot_labels"].to(self.device)
        obj_scale  = batch["obj_scale"].to(self.device)

        # ---- Augmentation ----
        if cfg.token_drop_prob > 0:
            t_mask = apply_token_dropout(t_mask, cfg.token_drop_prob)
        if cfg.token_sub_prob > 0:
            tokens = apply_token_substitution(
                tokens, t_mask, cfg.token_sub_prob,
                n_semantic=cfg.vocab_size - 2,
            )

        self.optimizer.zero_grad()

        # ---- Token summary (semantic conditioning) ----
        tok_summary = self.model.token_encoder(tokens, t_mask)   # (B, token_embed_dim)
        if cfg.no_token_cond:
            tok_summary = torch.zeros_like(tok_summary)

        # ---- SlotGrounder auxiliary loss ----
        grounder_logits = self.model.slot_grounder(bps_object, tok_summary)
        loss_grounder = self.model.slot_grounder.compute_loss(
            grounder_logits, bps_slot_labels)

        # ---- CoarseNet forward ----
        drec = self.model.forward_coarse(
            bps_object, tok_summary, gt_trans, gt_orient_rotmat,
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

        # ---- SlotGrounder loss ----
        if cfg.slot_grounder_weight > 0:
            loss_total = loss_total + cfg.slot_grounder_weight * loss_grounder
            loss_dict["loss_grounder"] = loss_grounder

        # ---- Part-Contact Consistency Loss ----
        if cfg.w_part_contact > 0:
            # Propagate slot labels to object surface points
            obj_slot_labels = None
            if self.bps_basis is not None:
                obj_slot_labels = propagate_slot_labels(
                    obj_pc, obj_scale, self.bps_basis, bps_slot_labels)

            loss_pc = compute_part_contact_loss(
                hand_verts=verts_pred,
                obj_pc=obj_pc,
                tokens=tokens,
                token_mask=t_mask,
                vert_to_part=self.vert_to_part,
                obj_slot_labels=obj_slot_labels,
            )
            loss_total = loss_total + cfg.w_part_contact * loss_pc
            loss_dict["loss_part_contact"] = loss_pc

        loss_dict["loss_total"] = loss_total

        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v
                   for k, v in loss_dict.items()}

        loss_total.backward()
        if cfg.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return metrics

    # ================================================================
    # Train step — RefineNet (GrabNet joint training)
    # ================================================================

    def _train_step_refine(self, batch) -> Dict[str, float]:
        """RefineNet training step — GrabNet-style.

        GrabNet feeds CoarseNet's actual reconstruction output to RefineNet.
        CoarseNet encode→decode acts as a natural perturbation of the GT,
        so RefineNet learns to correct realistic CoarseNet errors.

        The CoarseNet forward is run with torch.no_grad() — RefineNet
        gradients do not flow back into CoarseNet (separate optimizer).
        """
        cfg = self.cfg
        if not cfg.fit_refine:
            return {}

        obj_pc     = batch["obj_pc"].to(self.device)
        obj_vn     = batch["obj_vn"].to(self.device)
        gt_pose    = batch["hand_pose"].to(self.device)
        gt_trans   = batch["hand_tsl"].to(self.device)
        bps_object = batch["bps_object"].to(self.device)
        tokens     = batch["token_seq"].to(self.device)
        t_mask     = batch["token_mask"].to(self.device)

        B = gt_pose.shape[0]
        zeros_shape = torch.zeros(B, cfg.n_betas, device=self.device)

        # ---- Compute GT verts & h2o distances with mean shape ----
        with torch.no_grad():
            gt_verts, _ = self.mano(gt_pose, gt_trans, zeros_shape)
            h2o_gt = RefineNet._compute_h2o_dist(gt_verts, obj_pc, obj_vn)

        # ---- Get CoarseNet reconstruction output (detached) ----
        with torch.no_grad():
            gt_orient_rotmat = batch["global_orient_rhand_rotmat"].to(self.device)
            tok_summary = self.model.token_encoder(tokens, t_mask)
            if self.model.no_token_cond:
                tok_summary = torch.zeros_like(tok_summary)

            drec = self.model.coarse_net(
                bps_object, gt_trans, gt_orient_rotmat, tok_summary,
            )
            coarse_rot6d = drec["pose_rot6d"]      # (B, 96)
            coarse_trans = drec["transl"]            # (B, 3)
            coarse_verts, _ = self.mano(drec["fullpose_aa"], coarse_trans, zeros_shape)
            h2o_dist = RefineNet._compute_h2o_dist(coarse_verts, obj_pc, obj_vn)

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
            kl_coef=cfg.kl_coef,
            h2o_gt=h2o_gt,
            obj_normals=obj_vn,
        )

        loss.backward()
        if cfg.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model.refine_net.parameters(), cfg.max_grad_norm)
        self.optimizer_refine.step()
        if self.scheduler_refine is not None:
            self.scheduler_refine.step()

        return {f"rnet_{k}": v for k, v in metrics.items()}

    # ================================================================
    # Combined train step
    # ================================================================

    def _train_step(self, batch) -> Dict[str, float]:
        """Joint training step: CoarseNet + RefineNet."""
        # CoarseNet training
        coarse_metrics = self._train_step_coarse(batch)

        # RefineNet training (following GrabNet's approach)
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
        self.model.eval()
        if cfg.fit_refine:
            self.model.refine_net.eval()
        totals = defaultdict(float)
        n = 0
        token_metric_samples = 0

        for batch in self.val_dl:
            bps_object = batch["bps_object"].to(self.device)
            tokens = batch["token_seq"].to(self.device)
            t_mask = batch["token_mask"].to(self.device)
            gt_trans = batch["hand_tsl"].to(self.device)
            gt_pose = batch["hand_pose"].to(self.device)
            gt_orient_rotmat = batch["global_orient_rhand_rotmat"].to(self.device)
            obj_pc = batch["obj_pc"].to(self.device)
            obj_vn = batch["obj_vn"].to(self.device)
            obj_scale = batch["obj_scale"].to(self.device)
            B = gt_trans.shape[0]
            zeros_shape = torch.zeros(B, cfg.n_betas, device=self.device)

            # Recompute GT verts with mean shape
            gt_verts, _ = self.mano(gt_pose, gt_trans, zeros_shape)

            bps_slot_labels = batch["bps_slot_labels"].to(self.device)
            obj_slot_labels = None
            if self.bps_basis is not None:
                obj_slot_labels = propagate_slot_labels(
                    obj_pc, obj_scale, self.bps_basis, bps_slot_labels,
                )
            tok_summary = self.model.token_encoder(tokens, t_mask)
            if cfg.no_token_cond:
                tok_summary = torch.zeros_like(tok_summary)
            drec = self.model.forward_coarse(
                bps_object, tok_summary, gt_trans, gt_orient_rotmat,
            )

            # SlotGrounder accuracy
            pred_labels = self.model.slot_grounder.predict(bps_object, tok_summary)
            valid_mask = bps_slot_labels >= 0
            if valid_mask.any():
                correct = (pred_labels[valid_mask] == bps_slot_labels[valid_mask]).float()
                totals["grounder_acc"] += correct.mean().item() * B

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

            # Sample-based metrics
            pred = self.model.sample(bps_object, tokens, t_mask)

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

            final_verts = pred_verts
            final_pen_mm = pen_depth.mean().item() * 1000.0
            final_contact = sample_has_contact.mean().item()

            # RefineNet validation
            if cfg.fit_refine:
                coarse_rot6d = aa_to_rot6d(pred.pose)
                coarse_verts = pred_verts
                h2o_dist = RefineNet._compute_h2o_dist(coarse_verts, obj_pc, obj_vn)
                rnet_result = self.model.refine_net(
                    h2o_dist, coarse_rot6d, pred.trans,
                    obj_pc, self.mano, shape=pred.shape, obj_normals=obj_vn)
                refined_verts, _ = self.mano(rnet_result["pose"], rnet_result["trans"], pred.shape)

                # RefineNet loss (for scheduler and monitoring)
                rnet_loss, rnet_metrics = compute_refine_loss(
                    pred_verts=refined_verts,
                    gt_verts=gt_verts,
                    obj_pc=obj_pc,
                    v_weights=self.v_weights,
                    v_weights2=self.v_weights2,
                    vpe=self.vpe,
                    kl_coef=cfg.kl_coef,
                    obj_normals=obj_vn,
                )
                for rk, rv in rnet_metrics.items():
                    totals[f"rnet_{rk}"] += rv * B

                # Use signed distance for penetration monitoring
                _, refined_h2o_signed, _ = point2point_signed(
                    refined_verts, obj_pc, y_normals=obj_vn)
                pen_refined = torch.relu(-refined_h2o_signed).mean().item() * 1000.0
                contact_refined = (refined_h2o_signed.abs() < cfg.contact_thresh).any(dim=1).float().mean().item()
                totals["pen_refined_mm"] += pen_refined * B
                totals["contact_refined"] += contact_refined * B
                final_verts = refined_verts
                final_pen_mm = pen_refined
                final_contact = contact_refined

            if obj_slot_labels is not None:
                token_eval_mask = (obj_slot_labels >= 0).any(dim=1)
            else:
                token_eval_mask = None

            if token_eval_mask is not None and token_eval_mask.any():
                eval_B = int(token_eval_mask.sum().item())

                def _tok_f1(verts_eval):
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

                # Coarse token_f1 (before RefineNet) — used for best.pt selection
                coarse_recall, coarse_f1 = _tok_f1(pred_verts)
                totals["token_recall"] += coarse_recall.mean().item() * eval_B
                totals["token_f1"] += coarse_f1.mean().item() * eval_B

                # Refined token_f1 (after RefineNet) — diagnostic only
                if cfg.fit_refine:
                    refined_recall, refined_f1 = _tok_f1(final_verts)
                    totals["token_f1_refined"] += refined_f1.mean().item() * eval_B
                    totals["token_recall_refined"] += refined_recall.mean().item() * eval_B

                token_metric_samples += eval_B
            else:
                totals["token_recall"] += 0.0
                totals["token_f1"] += 0.0

            for k, v in loss_dict.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                totals[k] += val * B
            totals["pen_depth_mm"] += pen_depth.mean().item() * 1000.0 * B
            totals["contact_rate"] += sample_has_contact.mean().item() * B
            totals["final_pen_mm"] += final_pen_mm * B
            totals["final_contact_rate"] += final_contact * B
            n += B

        self.model.train()
        if cfg.fit_refine:
            self.model.refine_net.train()
        _token_keys = {"token_recall", "token_f1", "token_f1_refined", "token_recall_refined"}
        metrics = {}
        for k, v in totals.items():
            if k in _token_keys:
                metrics[f"val/{k}"] = v / max(token_metric_samples, 1)
            else:
                metrics[f"val/{k}"] = v / max(n, 1)
        metrics["val/token_eval_samples"] = float(token_metric_samples)
        return metrics

    # ================================================================
    # Save / Load
    # ================================================================

    def _save_checkpoint(self, tag="latest"):
        path = os.path.join(self.cfg.save_dir, f"{tag}.pt")
        save_dict = {
            "model":       self.model.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "scheduler":   self.scheduler.state_dict() if self.scheduler else {},
            "epoch":       self.epoch,
            "global_step": self.global_step,
            "best_val":    self.best_val,
            "best_token_f1": self.best_token_f1,
            "no_improve_evals": self.no_improve_evals,
            "config":      self.cfg.__dict__,
        }
        if self.optimizer_refine is not None:
            save_dict["optimizer_refine"] = self.optimizer_refine.state_dict()
            if self.scheduler_refine:
                save_dict["scheduler_refine"] = self.scheduler_refine.state_dict()
        torch.save(save_dict, path)
        self.logger.info(f"  Saved: {path}")

    def _load_checkpoint(self, path):
        self.logger.info(f"Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device)
        missing, unexpected = self.model.load_state_dict(ckpt["model"], strict=False)
        if missing or unexpected:
            self.logger.warning(f"  Checkpoint compat: missing={len(missing)}, unexpected={len(unexpected)}")
            for k in missing[:5]:
                self.logger.warning(f"    missing: {k}")
            for k in unexpected[:5]:
                self.logger.warning(f"    unexpected: {k}")
        try:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        except (KeyError, ValueError) as e:
            self.logger.warning(f"  Optimizer state_dict mismatch (type changed?), skipping: {e}")
        if self.scheduler and "scheduler" in ckpt and ckpt["scheduler"]:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler"])
            except (KeyError, ValueError) as e:
                self.logger.warning(f"  Scheduler state_dict mismatch, skipping: {e}")

        # RefineNet optimizer/scheduler (weights already loaded with model)
        if self.optimizer_refine is not None:
            if "optimizer_refine" in ckpt:
                self.optimizer_refine.load_state_dict(ckpt["optimizer_refine"])

        self.epoch       = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)
        self.best_val    = ckpt.get("best_val", float("inf"))
        self.best_token_f1 = ckpt.get("best_token_f1", float("-inf"))
        self.no_improve_evals = ckpt.get("no_improve_evals", 0)
        self.logger.info(f"  Resumed: epoch={self.epoch}, step={self.global_step}")

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
        if cfg.fit_refine:
            self.model.refine_net.train()
        self.logger.info(f"\nStart training: {cfg.epochs} epochs")
        should_stop = False

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
                epoch_loss += log.get("loss_total", 0.0)

                if self.global_step % cfg.log_every == 0:
                    grnd = f"  grnd={log.get('loss_grounder', 0):.4f}"
                    pc_loss = f"  pc={log.get('loss_part_contact', 0):.4f}" if cfg.w_part_contact > 0 else ""
                    rnet = ""
                    if cfg.fit_refine:
                        rnet = (f"  | rnet_loss={log.get('rnet_refine_loss', 0):.4f}"
                                f"  rnet_dist={log.get('rnet_refine_dist_h', 0):.4f}"
                                f"  rnet_mesh={log.get('rnet_refine_mesh_rec', 0):.4f}")
                    self.logger.info(
                        f"  S{self.global_step:06d}  "
                        f"total={log.get('loss_total', 0):.4f}  "
                        f"kl={log.get('loss_kl', 0):.4f}  "
                        f"mesh={log.get('loss_mesh_rec', 0):.4f}  "
                        f"dist_h={log.get('loss_dist_h', 0):.4f}  "
                        f"dist_o={log.get('loss_dist_o', 0):.4f}  "
                        f"edge={log.get('loss_edge', 0):.4f}"
                        f"{grnd}{pc_loss}{rnet}")

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
                vt = val.get("val/loss_total", float("inf"))
                val_token_f1 = val.get("val/token_f1", 0.0)
                val_token_recall = val.get("val/token_recall", 0.0)
                final_contact = val.get("val/final_contact_rate", val.get("val/contact_rate", 0.0))
                final_pen_mm = val.get("val/final_pen_mm", val.get("val/pen_depth_mm", float("inf")))
                val_msg = (
                    f"  [VAL]    "
                    f"total={vt:.4f}  "
                    f"contact={val.get('val/contact_rate', 0):.3f}  "
                    f"pen={val.get('val/pen_depth_mm', 0):.2f}mm  "
                    f"coarse_f1={val_token_f1:.3f}  "
                    f"coarse_recall={val_token_recall:.3f}")
                if cfg.fit_refine:
                    val_msg += (
                        f"  | refined: loss={val.get('val/rnet_refine_loss', 0):.4f}"
                        f"  pen={val.get('val/pen_refined_mm', 0):.2f}mm"
                        f"  contact={val.get('val/contact_refined', 0):.3f}"
                        f"  f1={val.get('val/token_f1_refined', 0):.3f}")
                val_msg += (
                    f"  | best-gate: final_contact={final_contact:.3f}"
                    f"  final_pen={final_pen_mm:.2f}mm"
                )
                self.logger.info(val_msg)

                if self.tb:
                    for k, v in val.items():
                        self.tb.add_scalar(k, v, self.global_step)

                cur_lr = self.optimizer.param_groups[0]['lr']
                refine_lr_msg = ""
                if self.optimizer_refine is not None:
                    refine_lr = self.optimizer_refine.param_groups[0]['lr']
                    refine_lr_msg = f"  refine_lr={refine_lr:.2e}"
                self.logger.info(f"  lr={cur_lr:.2e}{refine_lr_msg}")

                geom_ok = (
                    final_contact >= cfg.best_geom_contact_min
                    and final_pen_mm <= cfg.best_geom_pen_mm_max
                )
                if geom_ok:
                    if val_token_f1 > self.best_token_f1:
                        prev_best = self.best_token_f1
                        self.best_token_f1 = val_token_f1
                        self._save_checkpoint("best")
                        if prev_best == float("-inf"):
                            self.logger.info(
                                f"  * New best: token_f1={val_token_f1:.4f} "
                                f"(geometry gate passed)"
                            )
                        else:
                            self.logger.info(
                                f"  * New best: token_f1={val_token_f1:.4f} "
                                f"(prev={prev_best:.4f}, geometry gate passed)"
                            )
                    else:
                        self.logger.info(
                            f"  Best gate passed, but token_f1={val_token_f1:.4f} "
                            f"did not beat best={self.best_token_f1:.4f}"
                        )
                else:
                    self.logger.info(
                        f"  Best gate failed: need contact>={cfg.best_geom_contact_min:.2f} "
                        f"and pen<={cfg.best_geom_pen_mm_max:.2f}mm"
                    )

                # Early stopping: track val loss improvement, but only trigger
                # if best.pt has been saved at least once (geometry gate passed).
                if vt < self.best_val:
                    self.best_val = vt
                    self.no_improve_evals = 0
                    self.logger.info(f"  * New loss best: total={vt:.4f}")
                else:
                    self.no_improve_evals += 1
                    remaining = max(cfg.early_stop_patience - self.no_improve_evals, 0)
                    self.logger.info(
                        f"  [EARLY STOP] no improvement for {self.no_improve_evals}/"
                        f"{cfg.early_stop_patience} (remaining={remaining})")
                    if self.no_improve_evals >= cfg.early_stop_patience:
                        if self.best_token_f1 > float("-inf"):
                            self.logger.info("Early stopping triggered (best.pt exists).")
                            should_stop = True
                        else:
                            self.logger.info(
                                "Early stop patience exhausted, but geometry gate "
                                "never passed — continuing training."
                            )

            if should_stop:
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
    p.add_argument("--kl_coef",       type=float, default=0.01)

    # Slot supervision / part-contact auxiliaries
    p.add_argument("--bps_slot_cache_dir", type=str, default="cache/bps_slot",
                   help="Directory for per-object BPS slot cache")
    p.add_argument("--slot_grounder_weight", type=float, default=1.0,
                   help="Weight for SlotGrounder CE loss")

    # RefineNet
    p.add_argument("--fit_refine", action=argparse.BooleanOptionalAction, default=True,
                   help="Joint training of RefineNet (default: True)")
    p.add_argument("--refine_n_iters", type=int, default=3)
    p.add_argument("--refine_lr", type=float, default=5e-4)

    # Part-Contact Loss
    p.add_argument("--w_part_contact", type=float, default=10.0,
                   help="Weight for Part-Contact Consistency Loss")

    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=5e-4)
    p.add_argument("--warmup_steps", type=int,   default=1000)
    p.add_argument("--resume",       type=str,   default="")
    p.add_argument("--save_dir",     type=str,   default="checkpoints/decoder")
    p.add_argument("--log_dir",      type=str,   default="logs/decoder")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--token_drop_prob", type=float, default=0.0)
    p.add_argument("--token_sub_prob",  type=float, default=0.0)
    p.add_argument("--contact_thresh",  type=float, default=0.005)
    p.add_argument("--best_geom_contact_min", type=float, default=0.90)
    p.add_argument("--best_geom_pen_mm_max",  type=float, default=12.0)

    # Ablation
    p.add_argument("--no_token_cond", action="store_true",
                   help="Zero out token conditioning (ablation test)")

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
        slot_grounder_weight=args.slot_grounder_weight,
        fit_refine=args.fit_refine,
        refine_n_iters=args.refine_n_iters,
        refine_lr=args.refine_lr,
        w_part_contact=args.w_part_contact,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        resume=args.resume,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        seed=args.seed,
        token_drop_prob=args.token_drop_prob,
        token_sub_prob=args.token_sub_prob,
        contact_thresh=args.contact_thresh,
        best_geom_contact_min=args.best_geom_contact_min,
        best_geom_pen_mm_max=args.best_geom_pen_mm_max,
        no_token_cond=args.no_token_cond,
    )
    Trainer(cfg).train()


if __name__ == "__main__":
    main()
