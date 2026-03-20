"""
train_refine.py — RefineNet Training (Stage 3)
===============================================
Trains a RefineNet to iteratively refine coarse hand poses from the
Flow Decoder (Stage 2), reducing penetration and improving contact.

Training strategy:
  1. Load frozen Stage 1 (diffusion) + Stage 2 (decoder) + PointNet2 encoder
  2. For each GT sample, produce a coarse prediction via the decoder
  3. Train RefineNet to map coarse → GT, with losses emphasizing
     penetration removal, contact preservation, and vertex accuracy

Usage:
  python train_refine.py --decoder_ckpt checkpoints/decoder/best.pt
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
from models.object_normalization import normalize_object_points
from models.pose_decoder import build_grab_model, PoseDecoderOutput
from models.refine import RefineNet, compute_refine_loss
from utils.mano_utils import MANOHelper
from utils.pose_utils import aa_to_rot6d, rot6d_to_aa


# ==============================================================================
# Config
# ==============================================================================

@dataclass
class RefineConfig:
    # ---- data ----
    cache_dir:        str = "cache/contact_tokens/train"
    val_cache_dir:    str = "cache/contact_tokens/val"
    config_path:      str = "configs/token_config.yaml"
    mano_assets_root: str = "assets/mano_v1_2"
    n_pc_points:      int = 2048
    oi_category:      str = "all"
    oi_intent:        str = "all"

    # ---- frozen models ----
    decoder_ckpt:  str = "checkpoints/decoder/best.pt"

    # ---- decoder sampling ----
    decoder_sample_steps: int = 20

    # ---- model ----
    n_iters:    int = 3
    h_size:     int = 512
    n_neurons:  int = 512

    # ---- training ----
    epochs:         int   = 100
    batch_size:     int   = 64
    lr:             float = 5e-5
    weight_decay:   float = 0.01
    warmup_steps:   int   = 500
    max_grad_norm:  float = 1.0

    # ---- loss weights ----
    w_vert:     float = 5.0     # vertex L1 reconstruction
    w_dist:     float = 5.0     # h2o distance field matching (GrabNet-style)
    w_pen:      float = 100.0   # penetration (quadratic, per-penetrating-vertex mean)
    w_contact:  float = 2.0     # outside-only contact attraction
    w_reg:      float = 1.0     # stay close to coarse
    contact_thresh: float = 0.005  # 5mm

    # ---- noise augmentation ----
    # Add noise to coarse pose during training so RefineNet learns to
    # handle varied imperfections, not just the decoder's specific errors
    pose_noise_std:  float = 0.02
    trans_noise_std: float = 0.005

    # ---- logging / save ----
    log_dir:    str = "logs/refine"
    save_dir:   str = "checkpoints/refine"
    log_every:  int = 50
    val_every:  int = 500
    save_every: int = 2000
    save_top_k: int = 5
    early_stop_patience: int = 20

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
# Dataset (reuses DecoderDataset structure)
# ==============================================================================

class RefineDataset(Dataset):
    """Loads token sequences + OIShape GT data for refinement training."""

    def __init__(
        self,
        cache_dir:   str,
        config_path: str,
        split:       str = "train",
        n_pc_points: int = 2048,
        category:    str = "all",
        intent:      str = "all",
    ):
        self.cache_dir = cache_dir
        self.encoder   = TokenEncoder(config_path)
        self.mapper    = self.encoder.mapper
        self.n_pc_points = n_pc_points

        with open(os.path.join(cache_dir, "dataset_index.json")) as f:
            index = json.load(f)
        self.samples = [s for s in index["samples"] if s["token_length"] >= 1]

        logging.info(f"[RefineDataset] Loading OIShape split={split} ...")
        self.oishape = _load_oishape(split, category, intent, n_samples=n_pc_points)
        logging.info(f"[RefineDataset] OIShape: {len(self.oishape)} grasps")

        max_idx = len(self.oishape) - 1
        before = len(self.samples)
        self.samples = [s for s in self.samples if s["idx"] <= max_idx]
        if len(self.samples) < before:
            logging.warning(f"  Filtered {before - len(self.samples)} out-of-range samples")

        self._npz_cache = {}
        logging.info(f"[RefineDataset] {len(self.samples)} samples")

    def _load_npz(self, cache_file):
        if cache_file not in self._npz_cache:
            data = np.load(os.path.join(self.cache_dir, cache_file),
                           allow_pickle=True)
            slot_labels = data.get("obj_slot_labels", None)
            if slot_labels is None:
                slot_labels = np.full(self.n_pc_points, -1, dtype=np.int32)
            elif slot_labels.ndim != 1 or len(slot_labels) != self.n_pc_points:
                raise ValueError(
                    f"{cache_file}: obj_slot_labels shape mismatch, "
                    f"expected ({self.n_pc_points},), got {slot_labels.shape}"
                )
            self._npz_cache[cache_file] = {
                "token_seq":   data["token_seq"].astype(np.int64),
                "slot_labels": slot_labels.astype(np.int32),
            }
        return self._npz_cache[cache_file]

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

        return {
            "token_seq":   torch.from_numpy(token_seq),
            "token_mask":  torch.from_numpy(token_mask),
            "slot_labels": torch.from_numpy(tok["slot_labels"]),
            "obj_pc":      torch.from_numpy(oi["obj_verts"]),
            "obj_vn":      torch.from_numpy(oi["obj_vn"]),
            "hand_pose":   torch.from_numpy(oi["hand_pose"]),
            "hand_shape":  torch.from_numpy(oi["hand_shape"]),
            "hand_tsl":    torch.from_numpy(oi["hand_tsl"]),
            "hand_verts":  torch.from_numpy(oi["hand_verts"]),
        }


def _collate(batch):
    keys = ["token_seq", "token_mask", "slot_labels",
            "obj_pc", "obj_vn", "hand_pose", "hand_shape", "hand_tsl", "hand_verts"]
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

class RefineTrainer:
    def __init__(self, cfg: RefineConfig):
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
        self._load_frozen_models()
        self._build_refine_model()
        self._build_optimizer()

        if cfg.resume and os.path.isfile(cfg.resume):
            self._load_checkpoint(cfg.resume)

        self._log_info()

    # ================================================================
    # Init helpers
    # ================================================================

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
        self.train_ds = RefineDataset(
            cfg.cache_dir, cfg.config_path, "train",
            n_pc_points=cfg.n_pc_points,
            category=cfg.oi_category, intent=cfg.oi_intent)

        self.train_dl = DataLoader(
            self.train_ds, batch_size=cfg.batch_size,
            shuffle=True, num_workers=cfg.num_workers,
            pin_memory=True, drop_last=True, collate_fn=_collate)

        self.val_dl = None
        if cfg.val_cache_dir and os.path.isdir(cfg.val_cache_dir):
            val_ds = RefineDataset(
                cfg.val_cache_dir, cfg.config_path, "val",
                n_pc_points=cfg.n_pc_points,
                category=cfg.oi_category, intent=cfg.oi_intent)
            self.val_dl = DataLoader(
                val_ds, batch_size=cfg.batch_size, shuffle=False,
                num_workers=cfg.num_workers, pin_memory=True,
                collate_fn=_collate)

    def _load_frozen_models(self):
        """Load frozen CVAE decoder from Stage 2 checkpoint."""
        cfg = self.cfg
        self.logger.info(f"Loading frozen decoder: {cfg.decoder_ckpt}")
        dec_state = torch.load(cfg.decoder_ckpt, map_location=self.device)
        dec_cfg = dec_state.get("config", {})

        use_slot_bps = dec_state.get("use_slot_bps", dec_cfg.get("use_slot_bps", True))
        self.use_slot_bps = use_slot_bps
        self.logger.info(f"  use_slot_bps={use_slot_bps}")

        self.decoder = build_grab_model(
            dec_cfg.get("model_config", "base"),
            vocab_size=dec_cfg.get("vocab_size", 26),
            n_betas=dec_cfg.get("n_betas", 10),
            use_slot_bps=use_slot_bps,
        ).to(self.device)
        self.decoder.load_state_dict(dec_state["model"])
        self.decoder.eval()
        for p in self.decoder.parameters():
            p.requires_grad = False

        # BPS basis for object encoding
        bps_path = dec_cfg.get("bps_path", "grabnet/configs/bps.npz")
        if os.path.exists(bps_path):
            self.bps_basis = torch.from_numpy(
                np.load(bps_path)['basis']
            ).float().to(self.device)
            self.logger.info(f"  BPS basis loaded from {bps_path}")
        else:
            self.bps_basis = None
            self.logger.warning(f"  BPS basis not found at {bps_path}")

        self.logger.info("  Decoder frozen (CVAE, no obj encoder needed)")

    def _build_refine_model(self):
        cfg = self.cfg
        self.refine = RefineNet(
            n_iters=cfg.n_iters,
            h_size=cfg.h_size,
            n_neurons=cfg.n_neurons,
        ).to(self.device)

    def _build_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.refine.parameters(), lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay, betas=(0.9, 0.99))

        total = self.cfg.epochs * len(self.train_dl)
        self.scheduler = cosine_warmup_schedule(
            self.optimizer, self.cfg.warmup_steps, total)

    def _log_info(self):
        n_params = sum(p.numel() for p in self.refine.parameters() if p.requires_grad)
        cfg = self.cfg
        self.logger.info("=" * 60)
        self.logger.info("RefineNet — Iterative Grasp Pose Refinement (Stage 3)")
        self.logger.info(f"  RefineNet:       {n_params:,} params ({n_params/1e6:.2f}M)")
        self.logger.info(f"  Iterations:      {cfg.n_iters}")
        self.logger.info(f"  Hidden size:     {cfg.h_size}")
        self.logger.info(f"  Decoder ckpt:    {cfg.decoder_ckpt}")
        self.logger.info(f"  Decoder steps:   {cfg.decoder_sample_steps}")
        self.logger.info(f"  Loss weights:    vert={cfg.w_vert} dist={cfg.w_dist} "
                         f"pen={cfg.w_pen} contact={cfg.w_contact} reg={cfg.w_reg}")
        self.logger.info(f"  Noise aug:       pose_std={cfg.pose_noise_std} "
                         f"trans_std={cfg.trans_noise_std}")
        self.logger.info(f"  Contact thresh:  {cfg.contact_thresh * 1000:.1f}mm")
        self.logger.info(f"  Train:           {len(self.train_ds)} samples, "
                         f"{len(self.train_dl)} steps/epoch")
        self.logger.info("=" * 60)

    # ================================================================
    # Encode & generate coarse
    # ================================================================

    def _compute_bps(self, obj_pc):
        """Compute BPS encoding for a batch of object point clouds."""
        from bps_torch.bps import bps_torch
        bps = bps_torch(custom_basis=self.bps_basis.cpu(), device=obj_pc.device)
        B = obj_pc.shape[0]
        bps_list = []
        for i in range(B):
            result = bps.encode(obj_pc[i], feature_type='dists')
            bps_list.append(result['dists'].reshape(-1))
        return torch.stack(bps_list)  # (B, 4096)

    @torch.no_grad()
    def _generate_coarse(self, batch):
        """
        Generate coarse pose predictions from the frozen CVAE decoder.

        Uses GT tokens to produce coarse hand pose via BPS + token conditioning,
        then optionally adds pose/translation noise for augmentation.
        """
        cfg = self.cfg
        obj_pc   = batch["obj_pc"].to(self.device)
        tokens   = batch["token_seq"].to(self.device)
        t_mask   = batch["token_mask"].to(self.device)

        bps_object = self._compute_bps(obj_pc)
        pred = self.decoder.sample(bps_object, tokens, t_mask)

        # Convert to rot6d for RefineNet
        coarse_rot6d = aa_to_rot6d(pred.pose)  # (B, 96)
        coarse_trans = pred.trans               # (B, 3)
        coarse_shape = pred.shape               # (B, 10)

        # Noise augmentation to increase variety
        if self.refine.training and cfg.pose_noise_std > 0:
            coarse_rot6d = coarse_rot6d + torch.randn_like(coarse_rot6d) * cfg.pose_noise_std
        if self.refine.training and cfg.trans_noise_std > 0:
            coarse_trans = coarse_trans + torch.randn_like(coarse_trans) * cfg.trans_noise_std

        return coarse_rot6d, coarse_trans, coarse_shape

    # ================================================================
    # Train step
    # ================================================================

    def _train_step(self, batch) -> Dict[str, float]:
        cfg = self.cfg
        obj_pc   = batch["obj_pc"].to(self.device)
        obj_vn   = batch["obj_vn"].to(self.device)
        gt_pose  = batch["hand_pose"].to(self.device)
        gt_trans = batch["hand_tsl"].to(self.device)
        gt_shape = batch["hand_shape"].to(self.device)
        gt_verts = batch["hand_verts"].to(self.device)

        # Generate coarse prediction
        coarse_rot6d, coarse_trans, coarse_shape = self._generate_coarse(batch)

        # Compute coarse hand vertices and normals for h2o distance
        with torch.no_grad():
            coarse_pose_aa = rot6d_to_aa(coarse_rot6d)
            coarse_verts, _ = self.mano(coarse_pose_aa, coarse_trans, coarse_shape)
            from models.refine import _compute_vertex_normals
            coarse_hand_normals = _compute_vertex_normals(
                coarse_verts, self.mano.faces_tensor
            )

        # Compute initial h2o distance (with hand normals for reliable sign)
        h2o_dist = RefineNet._compute_h2o_dist(
            coarse_verts, obj_pc, obj_vn, hand_normals=coarse_hand_normals
        )

        # RefineNet forward
        self.optimizer.zero_grad()
        result = self.refine(
            h2o_dist, coarse_rot6d, coarse_trans,
            obj_pc, self.mano, shape=coarse_shape, obj_normals=obj_vn,
        )

        # Compute refined vertices via MANO
        refined_verts, _ = self.mano(result["pose"], result["trans"], coarse_shape)

        # Loss
        loss, metrics = compute_refine_loss(
            pred_verts=refined_verts,
            gt_verts=gt_verts,
            obj_pc=obj_pc,
            mano_fn=self.mano,
            obj_normals=obj_vn,
            coarse_verts=coarse_verts,
            w_vert=cfg.w_vert,
            w_dist=cfg.w_dist,
            w_pen=cfg.w_pen,
            w_contact=cfg.w_contact,
            w_reg=cfg.w_reg,
            contact_thresh=cfg.contact_thresh,
        )

        loss.backward()
        if cfg.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.refine.parameters(), cfg.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        return metrics

    # ================================================================
    # Validation
    # ================================================================

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        if self.val_dl is None:
            return {}
        self.refine.eval()
        totals = defaultdict(float)
        n = 0

        for batch in self.val_dl:
            obj_pc   = batch["obj_pc"].to(self.device)
            obj_vn   = batch["obj_vn"].to(self.device)
            gt_verts = batch["hand_verts"].to(self.device)
            gt_shape = batch["hand_shape"].to(self.device)

            coarse_rot6d, coarse_trans, coarse_shape = self._generate_coarse(batch)

            with torch.no_grad():
                coarse_pose_aa = rot6d_to_aa(coarse_rot6d)
                coarse_verts, _ = self.mano(coarse_pose_aa, coarse_trans, coarse_shape)
                from models.refine import _compute_vertex_normals
                coarse_hand_normals = _compute_vertex_normals(
                    coarse_verts, self.mano.faces_tensor
                )

            h2o_dist = RefineNet._compute_h2o_dist(
                coarse_verts, obj_pc, obj_vn, hand_normals=coarse_hand_normals
            )

            result = self.refine(
                h2o_dist, coarse_rot6d, coarse_trans,
                obj_pc, self.mano, shape=coarse_shape, obj_normals=obj_vn,
            )

            refined_verts, _ = self.mano(result["pose"], result["trans"], coarse_shape)
            B = refined_verts.shape[0]

            loss, metrics = compute_refine_loss(
                pred_verts=refined_verts,
                gt_verts=gt_verts,
                obj_pc=obj_pc,
                mano_fn=self.mano,
                obj_normals=obj_vn,
                coarse_verts=coarse_verts,
                w_vert=self.cfg.w_vert,
                w_dist=self.cfg.w_dist,
                w_pen=self.cfg.w_pen,
                w_contact=self.cfg.w_contact,
                w_reg=self.cfg.w_reg,
                contact_thresh=self.cfg.contact_thresh,
            )

            # Compute evaluation metrics (with hand normals)
            with torch.no_grad():
                refined_hand_normals = _compute_vertex_normals(
                    refined_verts, self.mano.faces_tensor
                )
            h2o_refined = RefineNet._compute_h2o_dist(
                refined_verts, obj_pc, obj_vn, hand_normals=refined_hand_normals
            )
            h2o_coarse = RefineNet._compute_h2o_dist(
                coarse_verts, obj_pc, obj_vn, hand_normals=coarse_hand_normals
            )

            pen_refined = torch.relu(-h2o_refined).mean().item() * 1000.0
            pen_coarse = torch.relu(-h2o_coarse).mean().item() * 1000.0

            contact_refined = (h2o_refined.abs() < self.cfg.contact_thresh).any(dim=1).float().mean().item()
            contact_coarse = (h2o_coarse.abs() < self.cfg.contact_thresh).any(dim=1).float().mean().item()

            vert_err = F.l1_loss(refined_verts, gt_verts).item() * 1000.0  # mm

            for k, v in metrics.items():
                totals[k] += float(v) * B
            totals["pen_refined_mm"] += pen_refined * B
            totals["pen_coarse_mm"] += pen_coarse * B
            totals["contact_refined"] += contact_refined * B
            totals["contact_coarse"] += contact_coarse * B
            totals["vert_err_mm"] += vert_err * B
            n += B

        self.refine.train()
        return {f"val/{k}": v / max(n, 1) for k, v in totals.items()}

    # ================================================================
    # Save / Load
    # ================================================================

    def _save_checkpoint(self, tag="latest"):
        path = os.path.join(self.cfg.save_dir, f"{tag}.pt")
        torch.save({
            "refine":      self.refine.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "scheduler":   self.scheduler.state_dict(),
            "epoch":       self.epoch,
            "global_step": self.global_step,
            "best_val":    self.best_val,
            "no_improve_evals": self.no_improve_evals,
            "config":      self.cfg.__dict__,
        }, path)
        self.logger.info(f"  Saved: {path}")

    def _load_checkpoint(self, path):
        self.logger.info(f"Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.refine.load_state_dict(ckpt["refine"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
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
        self.refine.train()
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
                epoch_loss += log["refine_loss"]

                if self.global_step % cfg.log_every == 0:
                    self.logger.info(
                        f"  S{self.global_step:06d}  "
                        f"loss={log['refine_loss']:.4f}  "
                        f"vert={log['refine_vert']:.4f}  "
                        f"dist={log['refine_dist']:.4f}  "
                        f"pen={log['refine_pen']:.4f}  "
                        f"contact={log['refine_contact']:.4f}  "
                        f"pen_ratio={log['pen_ratio']:.3f}")

                    if self.tb:
                        for k, v in log.items():
                            if isinstance(v, (int, float)):
                                self.tb.add_scalar(f"train/{k}", v, self.global_step)

                if self.global_step % cfg.val_every == 0:
                    val = self._validate()
                    if val:
                        vl = val.get("val/refine_loss", float("inf"))
                        self.logger.info(
                            f"  [VAL]    "
                            f"loss={vl:.4f}  "
                            f"vert={val.get('val/vert_err_mm', 0.0):.2f}mm  "
                            f"pen_coarse={val.get('val/pen_coarse_mm', 0.0):.2f}mm  "
                            f"pen_refined={val.get('val/pen_refined_mm', 0.0):.2f}mm  "
                            f"contact_c={val.get('val/contact_coarse', 0.0):.3f}  "
                            f"contact_r={val.get('val/contact_refined', 0.0):.3f}")

                        if self.tb:
                            for k, v in val.items():
                                self.tb.add_scalar(k, v, self.global_step)

                        if vl < self.best_val:
                            self.best_val = vl
                            self.no_improve_evals = 0
                            self._save_checkpoint("best")
                            self.logger.info(f"  * New best: loss={vl:.4f}")
                        else:
                            self.no_improve_evals += 1
                            remaining = max(cfg.early_stop_patience - self.no_improve_evals, 0)
                            self.logger.info(
                                f"  [EARLY STOP] no improvement for {self.no_improve_evals}/"
                                f"{cfg.early_stop_patience} (best={self.best_val:.4f}, "
                                f"cur={vl:.4f}, remaining={remaining})")
                            if self.no_improve_evals >= cfg.early_stop_patience:
                                self.logger.info(
                                    f"Early stopping at epoch={self.epoch}, step={self.global_step}.")
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
    p = argparse.ArgumentParser(
        description="Train RefineNet (Stage 3 — iterative pose refinement)")

    p.add_argument("--cache_dir",     type=str, default="cache/contact_tokens/train")
    p.add_argument("--val_cache_dir", type=str, default="cache/contact_tokens/val")
    p.add_argument("--n_pc_points",   type=int, default=2048)

    p.add_argument("--decoder_ckpt",  type=str, default="checkpoints/decoder/best.pt")
    p.add_argument("--decoder_sample_steps", type=int, default=20)

    p.add_argument("--n_iters",   type=int, default=3)
    p.add_argument("--h_size",    type=int, default=512)

    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=5e-5)
    p.add_argument("--resume",       type=str,   default="")
    p.add_argument("--save_dir",     type=str,   default="checkpoints/refine")
    p.add_argument("--log_dir",      type=str,   default="logs/refine")
    p.add_argument("--seed",         type=int,   default=42)

    p.add_argument("--w_vert",    type=float, default=5.0)
    p.add_argument("--w_dist",    type=float, default=5.0)
    p.add_argument("--w_pen",     type=float, default=100.0)
    p.add_argument("--w_contact", type=float, default=2.0)
    p.add_argument("--w_reg",     type=float, default=1.0)
    p.add_argument("--contact_thresh", type=float, default=0.005)

    p.add_argument("--pose_noise_std",  type=float, default=0.02)
    p.add_argument("--trans_noise_std", type=float, default=0.005)

    return p.parse_args()


def main():
    args = parse_args()
    cfg = RefineConfig(
        cache_dir=args.cache_dir,
        val_cache_dir=args.val_cache_dir,
        n_pc_points=args.n_pc_points,
        decoder_ckpt=args.decoder_ckpt,
        decoder_sample_steps=args.decoder_sample_steps,
        n_iters=args.n_iters,
        h_size=args.h_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume=args.resume,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        seed=args.seed,
        w_vert=args.w_vert,
        w_dist=args.w_dist,
        w_pen=args.w_pen,
        w_contact=args.w_contact,
        w_reg=args.w_reg,
        contact_thresh=args.contact_thresh,
        pose_noise_std=args.pose_noise_std,
        trans_noise_std=args.trans_noise_std,
    )
    RefineTrainer(cfg).train()


if __name__ == "__main__":
    main()
