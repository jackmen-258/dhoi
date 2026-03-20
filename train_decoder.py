"""
train_decoder.py (GrabNet-style CVAE + RefineNet Joint Training)
================================================================
PoseGrabModel (CoarseNet) + RefineNet joint training script.

Following GrabNet's training approach, CoarseNet and RefineNet are trained
jointly in the same epoch loop, each with their own optimizer.

Architecture:
  - CoarseNet: CVAE with slot-grounded BPS(4096) + contact token conditioning
  - RefineNet: iterative h2o-distance refinement (from models/refine.py)
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
from torch.utils.data import Dataset, DataLoader

from data.token_encoder import TokenEncoder
from models.pose_decoder import (
    build_grab_model, ROT6D_POSE_DIM, TRANS_DIM, NUM_HAND_PARTS, NUM_SLOTS,
    point2point_signed, compute_geo_loss,
    build_vert_to_part, propagate_slot_labels, compute_part_contact_loss,
)
from models.refine import RefineNet, compute_refine_loss, _compute_vertex_normals
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
    n_pc_points:      int = 2048
    oi_category:      str = "all"
    oi_intent:        str = "all"
    bps_path:         str = "grabnet/configs/bps.npz"

    # ---- model ----
    model_config: str = "base"
    n_betas:      int = 10
    vocab_size:   int = 26
    kl_coef:      float = 0.005

    # ---- slot-grounded BPS ----
    bps_slot_cache_dir:   str   = "cache/bps_slot"
    slot_grounder_weight: float = 1.0    # CE loss weight for SlotGrounder

    # ---- RefineNet (joint training) ----
    fit_refine:     bool  = True
    refine_n_iters: int   = 3
    refine_h_size:  int   = 512
    refine_n_neurons: int = 512
    refine_lr:      float = 5e-5
    # RefineNet loss weights
    w_refine_vert:    float = 5.0
    w_refine_dist:    float = 5.0
    w_refine_pen:     float = 100.0
    w_refine_contact: float = 2.0
    w_refine_reg:     float = 1.0
    # Noise augmentation for RefineNet
    pose_noise_std:   float = 0.02
    trans_noise_std:  float = 0.005

    # ---- Part-Contact Consistency Loss ----
    w_part_contact: float = 1.0

    # ---- training ----
    epochs:            int   = 200
    batch_size:        int   = 64
    lr:                float = 1e-4
    weight_decay:      float = 0.01
    warmup_steps:      int   = 1000
    max_grad_norm:     float = 1.0

    # ---- augmentation ----
    token_drop_prob:    float = 0.2
    token_sub_prob:     float = 0.05
    contact_thresh:     float = 0.005

    # ---- logging / save ----
    log_dir:    str = "logs/decoder"
    save_dir:   str = "checkpoints/decoder"
    log_every:  int = 50
    val_every:  int = 1000
    save_every: int = 5000
    save_top_k: int = 5
    early_stop_patience: int = 10

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
        n_pc_points: int = 2048,
        category:    str = "all",
        intent:      str = "all",
        bps_path:    str = "grabnet/configs/bps.npz",
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
    def fn(step):
        if step < warmup:
            return step / max(1, warmup)
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
                collate_fn=_collate)

    def _build_model(self):
        cfg = self.cfg
        self.model = build_grab_model(
            cfg.model_config,
            vocab_size=cfg.vocab_size,
            n_betas=cfg.n_betas,
            kl_coef=cfg.kl_coef,
        ).to(self.device)

        # RefineNet — joint training following GrabNet approach
        if cfg.fit_refine:
            self.refine_net = RefineNet(
                n_iters=cfg.refine_n_iters,
                h_size=cfg.refine_h_size,
                n_neurons=cfg.refine_n_neurons,
            ).to(self.device)
        else:
            self.refine_net = None

    def _build_optimizer(self):
        cfg = self.cfg
        # CoarseNet optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr,
            weight_decay=cfg.weight_decay, betas=(0.9, 0.99))

        total = cfg.epochs * len(self.train_dl)
        self.scheduler = cosine_warmup_schedule(
            self.optimizer, cfg.warmup_steps, total)

        # RefineNet optimizer (separate, following GrabNet)
        if self.refine_net is not None:
            self.optimizer_refine = torch.optim.AdamW(
                self.refine_net.parameters(), lr=cfg.refine_lr,
                weight_decay=cfg.weight_decay, betas=(0.9, 0.99))
            self.scheduler_refine = cosine_warmup_schedule(
                self.optimizer_refine, cfg.warmup_steps, total)
        else:
            self.optimizer_refine = None
            self.scheduler_refine = None

    def _load_loss_assets(self):
        """Load vertex weights and edge pairs for GrabNet-style loss."""
        import mano as mano_pkg
        cfg = self.cfg

        # MANO faces for Meshes
        with torch.no_grad():
            rhm = mano_pkg.load(
                model_path=os.path.join(cfg.mano_assets_root, "models"),
                model_type='mano', num_pca_comps=45,
                batch_size=1, flat_hand_mean=True,
            )
            rh_f = torch.from_numpy(rhm.faces.astype(np.int32)).view(1, -1, 3)
        self.rh_faces = rh_f.repeat(cfg.batch_size, 1, 1).to(self.device).long()

        # Vertex weights (try to load, fallback to ones)
        c_weights_path = os.path.join("grabnet", "configs", "c_weights.npy")
        if os.path.exists(c_weights_path):
            v_weights = torch.from_numpy(np.load(c_weights_path)).float().to(self.device)
            self.v_weights = v_weights
            self.v_weights2 = torch.pow(v_weights, 1.0 / 2.5)
        else:
            self.v_weights = torch.ones(778, device=self.device)
            self.v_weights2 = torch.ones(778, device=self.device)
            self.logger.warning("c_weights.npy not found, using uniform vertex weights")

        # Edge pairs
        vpe_path = os.path.join("grabnet", "configs", "vpe.npy")
        if os.path.exists(vpe_path):
            self.vpe = torch.from_numpy(np.load(vpe_path)).to(self.device).long()
        else:
            self.vpe = None
            self.logger.warning("vpe.npy not found, edge loss disabled")

        # BPS basis for slot label propagation
        if os.path.exists(cfg.bps_path):
            self.bps_basis = torch.from_numpy(
                np.load(cfg.bps_path)['basis']
            ).float().to(self.device)
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
        if self.refine_net is not None:
            n_refine = sum(p.numel() for p in self.refine_net.parameters() if p.requires_grad)
            self.logger.info(f"  RefineNet:     {n_refine:,} params ({n_refine/1e6:.2f}M)")
        self.logger.info(f"  Output:        rot6d={ROT6D_POSE_DIM} + trans={TRANS_DIM} (no shape)")
        self.logger.info(f"  Vocab size:    {cfg.vocab_size}")
        self.logger.info(f"  KL coef:       {cfg.kl_coef}")
        self.logger.info(f"  Grounder w:    {cfg.slot_grounder_weight}")
        self.logger.info(f"  Part-Contact w: {cfg.w_part_contact}")
        self.logger.info(f"  Token aug:     drop_p={cfg.token_drop_prob}  sub_p={cfg.token_sub_prob}")
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
        gt_verts   = batch["hand_verts"].to(self.device)
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

        # ---- Slot-modulated BPS feature ----
        slot_bps_feat = self.model.encode_slot_bps(
            bps_object, bps_slot_labels, tokens, t_mask)

        # ---- SlotGrounder auxiliary loss ----
        tok_summary = self.model.slot_bps._encode_token_summary(tokens, t_mask)
        grounder_logits = self.model.slot_grounder(bps_object, tok_summary)
        loss_grounder = self.model.slot_grounder.compute_loss(
            grounder_logits, bps_slot_labels)

        # ---- CoarseNet forward ----
        drec = self.model.forward_coarse(
            bps_object, slot_bps_feat, gt_trans, gt_orient_rotmat,
        )

        # ---- MANO forward for predicted verts ----
        with torch.cuda.amp.autocast(enabled=False):
            drec_float32 = {k: v.float() if isinstance(v, torch.Tensor) else v
                           for k, v in drec.items()}
            verts_pred, _ = self.mano(
                drec_float32["fullpose_aa"],
                drec_float32["transl"],
                torch.zeros(gt_trans.shape[0], cfg.n_betas, device=self.device),
            )

        # ---- CoarseNet Loss ----
        B = gt_trans.shape[0]
        dorig = {"vpe": self.vpe} if self.vpe is not None else {}
        rh_faces = self.rh_faces[:B]

        loss_total, loss_dict = self.model.compute_coarse_loss(
            drec, dorig, verts_pred, gt_verts, obj_pc,
            rh_faces, self.v_weights, self.v_weights2, B,
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
        self.scheduler.step()

        return metrics

    # ================================================================
    # Train step — RefineNet (GrabNet joint training)
    # ================================================================

    def _train_step_refine(self, batch) -> Dict[str, float]:
        """RefineNet training step following GrabNet's joint training approach.

        1. Generate coarse prediction from frozen/detached CoarseNet output
        2. Compute h2o distances
        3. RefineNet forward (iterative refinement)
        4. Compute refine loss and backprop (only RefineNet params)
        """
        cfg = self.cfg
        if self.refine_net is None:
            return {}

        obj_pc   = batch["obj_pc"].to(self.device)
        obj_vn   = batch["obj_vn"].to(self.device)
        gt_verts = batch["hand_verts"].to(self.device)
        gt_shape = batch["hand_shape"].to(self.device)
        tokens   = batch["token_seq"].to(self.device)
        t_mask   = batch["token_mask"].to(self.device)
        bps_object = batch["bps_object"].to(self.device)
        bps_slot_labels = batch["bps_slot_labels"].to(self.device)

        # Generate coarse prediction (detached from CoarseNet graph)
        with torch.no_grad():
            pred = self.model.sample(bps_object, tokens, t_mask,
                                     bps_slot_labels=bps_slot_labels)
            coarse_rot6d = aa_to_rot6d(pred.pose)
            coarse_trans = pred.trans
            coarse_shape = pred.shape

            # Noise augmentation
            if cfg.pose_noise_std > 0:
                coarse_rot6d = coarse_rot6d + torch.randn_like(coarse_rot6d) * cfg.pose_noise_std
            if cfg.trans_noise_std > 0:
                coarse_trans = coarse_trans + torch.randn_like(coarse_trans) * cfg.trans_noise_std

            # Compute coarse vertices and h2o distance
            coarse_pose_aa = rot6d_to_aa(coarse_rot6d)
            coarse_verts, _ = self.mano(coarse_pose_aa, coarse_trans, coarse_shape)
            coarse_hand_normals = _compute_vertex_normals(
                coarse_verts, self.mano.faces_tensor)

        h2o_dist = RefineNet._compute_h2o_dist(
            coarse_verts, obj_pc, obj_vn, hand_normals=coarse_hand_normals)

        # RefineNet forward
        self.optimizer_refine.zero_grad()
        result = self.refine_net(
            h2o_dist, coarse_rot6d, coarse_trans,
            obj_pc, self.mano, shape=coarse_shape, obj_normals=obj_vn,
        )

        # Compute refined vertices
        refined_verts, _ = self.mano(result["pose"], result["trans"], coarse_shape)

        # Loss
        loss, metrics = compute_refine_loss(
            pred_verts=refined_verts,
            gt_verts=gt_verts,
            obj_pc=obj_pc,
            mano_fn=self.mano,
            obj_normals=obj_vn,
            coarse_verts=coarse_verts,
            w_vert=cfg.w_refine_vert,
            w_dist=cfg.w_refine_dist,
            w_pen=cfg.w_refine_pen,
            w_contact=cfg.w_refine_contact,
            w_reg=cfg.w_refine_reg,
            contact_thresh=cfg.contact_thresh,
        )

        loss.backward()
        if cfg.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.refine_net.parameters(), cfg.max_grad_norm)
        self.optimizer_refine.step()
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
        self.model.eval()
        if self.refine_net is not None:
            self.refine_net.eval()
        totals = defaultdict(float)
        n = 0
        cfg = self.cfg

        for batch in self.val_dl:
            bps_object = batch["bps_object"].to(self.device)
            tokens = batch["token_seq"].to(self.device)
            t_mask = batch["token_mask"].to(self.device)
            gt_trans = batch["hand_tsl"].to(self.device)
            gt_orient_rotmat = batch["global_orient_rhand_rotmat"].to(self.device)
            gt_verts = batch["hand_verts"].to(self.device)
            obj_pc = batch["obj_pc"].to(self.device)
            obj_vn = batch["obj_vn"].to(self.device)
            B = gt_trans.shape[0]

            bps_slot_labels = batch["bps_slot_labels"].to(self.device)
            slot_bps_feat = self.model.encode_slot_bps(
                bps_object, bps_slot_labels, tokens, t_mask)
            drec = self.model.forward_coarse(
                bps_object, slot_bps_feat, gt_trans, gt_orient_rotmat,
            )

            # SlotGrounder accuracy
            pred_labels = self.model.predict_slot_labels(bps_object, tokens, t_mask)
            valid_mask = bps_slot_labels >= 0
            if valid_mask.any():
                correct = (pred_labels[valid_mask] == bps_slot_labels[valid_mask]).float()
                totals["grounder_acc"] += correct.mean().item() * B

            verts_pred, _ = self.mano(
                drec["fullpose_aa"].float(),
                drec["transl"].float(),
                torch.zeros(B, cfg.n_betas, device=self.device),
            )

            dorig = {"vpe": self.vpe} if self.vpe is not None else {}
            rh_faces = self.rh_faces[:B]
            loss_total, loss_dict = self.model.compute_coarse_loss(
                drec, dorig, verts_pred, gt_verts, obj_pc,
                rh_faces, self.v_weights, self.v_weights2, B,
            )

            # Sample-based metrics
            pred = self.model.sample(bps_object, tokens, t_mask,
                                     bps_slot_labels=bps_slot_labels)

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
            if self.refine_net is not None:
                coarse_rot6d = aa_to_rot6d(pred.pose)
                coarse_verts = pred_verts
                coarse_hand_normals = _compute_vertex_normals(
                    coarse_verts, self.mano.faces_tensor)
                h2o_dist = RefineNet._compute_h2o_dist(
                    coarse_verts, obj_pc, obj_vn, hand_normals=coarse_hand_normals)
                rnet_result = self.refine_net(
                    h2o_dist, coarse_rot6d, pred.trans,
                    obj_pc, self.mano, shape=pred.shape, obj_normals=obj_vn)
                refined_verts, _ = self.mano(rnet_result["pose"], rnet_result["trans"], pred.shape)

                refined_h2o = RefineNet._compute_h2o_dist(
                    refined_verts, obj_pc, obj_vn,
                    hand_normals=_compute_vertex_normals(refined_verts, self.mano.faces_tensor))
                pen_refined = torch.relu(-refined_h2o).mean().item() * 1000.0
                contact_refined = (refined_h2o.abs() < cfg.contact_thresh).any(dim=1).float().mean().item()
                totals["pen_refined_mm"] += pen_refined * B
                totals["contact_refined"] += contact_refined * B

            for k, v in loss_dict.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                totals[k] += val * B
            totals["pen_depth_mm"] += pen_depth.mean().item() * 1000.0 * B
            totals["contact_rate"] += sample_has_contact.mean().item() * B
            n += B

        self.model.train()
        if self.refine_net is not None:
            self.refine_net.train()
        return {f"val/{k}": v / max(n, 1) for k, v in totals.items()}

    # ================================================================
    # Save / Load
    # ================================================================

    def _save_checkpoint(self, tag="latest"):
        path = os.path.join(self.cfg.save_dir, f"{tag}.pt")
        save_dict = {
            "model":       self.model.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "scheduler":   self.scheduler.state_dict(),
            "epoch":       self.epoch,
            "global_step": self.global_step,
            "best_val":    self.best_val,
            "no_improve_evals": self.no_improve_evals,
            "config":      self.cfg.__dict__,
        }
        if self.refine_net is not None:
            save_dict["refine_net"] = self.refine_net.state_dict()
            save_dict["optimizer_refine"] = self.optimizer_refine.state_dict()
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
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])

        # RefineNet
        if self.refine_net is not None and "refine_net" in ckpt:
            self.refine_net.load_state_dict(ckpt["refine_net"])
            if "optimizer_refine" in ckpt:
                self.optimizer_refine.load_state_dict(ckpt["optimizer_refine"])
            if "scheduler_refine" in ckpt:
                self.scheduler_refine.load_state_dict(ckpt["scheduler_refine"])

        self.epoch       = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)
        self.best_val    = ckpt.get("best_val", float("inf"))
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
        if self.refine_net is not None:
            self.refine_net.train()
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
                    if self.refine_net is not None:
                        rnet = (f"  | rnet_loss={log.get('rnet_refine_loss', 0):.4f}"
                                f"  rnet_pen={log.get('rnet_refine_pen', 0):.4f}")
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

                if self.global_step % cfg.val_every == 0:
                    val = self._validate()
                    if val:
                        vt = val.get("val/loss_total", float("inf"))
                        val_msg = (
                            f"  [VAL]    "
                            f"total={vt:.4f}  "
                            f"contact={val.get('val/contact_rate', 0):.3f}  "
                            f"pen={val.get('val/pen_depth_mm', 0):.2f}mm")
                        if self.refine_net is not None:
                            val_msg += (
                                f"  | refined: pen={val.get('val/pen_refined_mm', 0):.2f}mm"
                                f"  contact={val.get('val/contact_refined', 0):.3f}")
                        self.logger.info(val_msg)

                        if self.tb:
                            for k, v in val.items():
                                self.tb.add_scalar(k, v, self.global_step)

                        if vt < self.best_val:
                            self.best_val = vt
                            self.no_improve_evals = 0
                            self._save_checkpoint("best")
                            self.logger.info(f"  * New best: total={vt:.4f}")
                        else:
                            self.no_improve_evals += 1
                            remaining = max(cfg.early_stop_patience - self.no_improve_evals, 0)
                            self.logger.info(
                                f"  [EARLY STOP] no improvement for {self.no_improve_evals}/"
                                f"{cfg.early_stop_patience} (remaining={remaining})")
                            if self.no_improve_evals >= cfg.early_stop_patience:
                                self.logger.info("Early stopping triggered.")
                                should_stop = True

                if self.global_step % cfg.save_every == 0:
                    self._save_checkpoint("latest")
                    self._save_checkpoint(f"step_{self.global_step:07d}")
                    self._cleanup_checkpoints()

                if should_stop:
                    break

            dt = time.time() - t0
            self.logger.info(
                f"  Epoch {ep} done  {dt:.0f}s  "
                f"avg_loss={epoch_loss / max(n_steps, 1):.4f}")
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
    p.add_argument("--n_pc_points",   type=int, default=2048)
    p.add_argument("--bps_path",      type=str, default="grabnet/configs/bps.npz")

    p.add_argument("--model_config",  type=str, default="base")
    p.add_argument("--kl_coef",       type=float, default=0.005)

    # Slot-grounded BPS
    p.add_argument("--bps_slot_cache_dir", type=str, default="cache/bps_slot",
                   help="Directory for per-object BPS slot cache")
    p.add_argument("--slot_grounder_weight", type=float, default=1.0,
                   help="Weight for SlotGrounder CE loss")

    # RefineNet
    p.add_argument("--fit_refine", action=argparse.BooleanOptionalAction, default=True,
                   help="Joint training of RefineNet (default: True)")
    p.add_argument("--refine_n_iters", type=int, default=3)
    p.add_argument("--refine_lr", type=float, default=5e-5)

    # Part-Contact Loss
    p.add_argument("--w_part_contact", type=float, default=1.0,
                   help="Weight for Part-Contact Consistency Loss")

    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--resume",       type=str,   default="")
    p.add_argument("--save_dir",     type=str,   default="checkpoints/decoder")
    p.add_argument("--log_dir",      type=str,   default="logs/decoder")
    p.add_argument("--seed",         type=int,   default=7725)
    p.add_argument("--token_drop_prob", type=float, default=0.2)
    p.add_argument("--token_sub_prob",  type=float, default=0.05)
    p.add_argument("--contact_thresh",  type=float, default=0.005)

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
        resume=args.resume,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        seed=args.seed,
        token_drop_prob=args.token_drop_prob,
        token_sub_prob=args.token_sub_prob,
        contact_thresh=args.contact_thresh,
    )
    Trainer(cfg).train()


if __name__ == "__main__":
    main()
