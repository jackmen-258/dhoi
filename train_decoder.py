"""
train_decoder.py (GrabNet-style CVAE)
======================================
PoseGrabModel (CoarseNet + RefineNet) training script.

Architecture:
  - CoarseNet: CVAE with BPS(4096) + contact token conditioning
  - RefineNet: iterative h2o-distance refinement
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
from torch.utils.data import Dataset, DataLoader

from data.token_encoder import TokenEncoder
from models.pose_decoder import (
    build_grab_model, ROT6D_POSE_DIM, TRANS_DIM,
    point2point_signed, compute_geo_loss,
)
from utils.mano_utils import MANOHelper
from utils.pose_utils import aa_to_rot6d


# ==============================================================================
# Contact Token Dropout
# ==============================================================================

@torch.no_grad()
def apply_token_dropout(token_mask, drop_prob, adaptive_pivot: int = 6):
    """
    k 自适应 token dropout.

    稀疏序列（k < adaptive_pivot）按比例降低 drop_prob，避免精捏等
    短序列因过度 dropout 丢失全部语义信息。具体地：
        effective_prob = drop_prob * min(1, k / adaptive_pivot)

    Args:
        token_mask:     (B, L) bool — True = 有效 token
        drop_prob:      float — 基准 drop 概率（k >= pivot 时使用）
        adaptive_pivot: int   — k 达到此值时使用完整 drop_prob（默认 6）

    Returns:
        new token_mask, 保证每个样本至少保留 1 个 valid token
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
    """
    随机将部分 valid token 替换为其他语义 token（模拟扩散模型的 token 预测错误）.
    """
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
    use_slot_bps:         bool  = True
    bps_slot_cache_dir:   str   = "cache/bps_slot"
    slot_grounder_weight: float = 1.0    # CE loss weight for SlotGrounder

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

    # ---- training stages ----
    fit_coarse: bool = True
    fit_refine: bool = True

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


def _load_oishape(split, category="all", intent="all", n_samples=2048, bps_path=None):
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
        use_slot_bps: bool = True,
    ):
        self.cache_dir = cache_dir
        self.encoder   = TokenEncoder(config_path)
        self.mapper    = self.encoder.mapper
        self.n_pc_points = n_pc_points
        self.use_slot_bps = use_slot_bps

        with open(os.path.join(cache_dir, "dataset_index.json")) as f:
            index = json.load(f)
        self.samples = [s for s in index["samples"] if s["token_length"] >= 1]

        logging.info(f"[DecoderDataset] Loading OIShape split={split}, "
                     f"n_samples={n_pc_points} ...")
        self.oishape = _load_oishape(
            split, category, intent,
            n_samples=n_pc_points, bps_path=bps_path,
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
        if use_slot_bps and bps_slot_cache_dir:
            slot_dir = os.path.join(bps_slot_cache_dir, split)
            if os.path.isdir(slot_dir):
                self._bps_slot_dir = slot_dir
                logging.info(f"[DecoderDataset] BPS slot cache: {slot_dir}")
            else:
                logging.warning(f"[DecoderDataset] BPS slot cache dir not found: {slot_dir}")

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
            slot_labels = data.get("obj_slot_labels", None)
            if slot_labels is None:
                slot_labels = np.full(self.n_pc_points, -1, dtype=np.int32)
            elif slot_labels.ndim != 1 or len(slot_labels) != self.n_pc_points:
                raise ValueError(
                    f"{cache_file}: obj_slot_labels shape mismatch, "
                    f"expected ({self.n_pc_points},), got {slot_labels.shape}"
                )
            self._npz_cache[cache_file] = {
                "token_seq":      data["token_seq"].astype(np.int64),
                "slot_labels":    slot_labels.astype(np.int32),
            }
        return self._npz_cache[cache_file]

    def _load_bps_slot(self, obj_id: str):
        """Load per-object BPS slot cache: bps_dists(4096) + bps_slot_labels(4096).

        Returns:
            dict with 'bps_dists' and 'bps_slot_labels', or None if not available.
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

        # BPS encoding — prefer slot cache (normalized), fallback to OIShape BPS
        obj_id = self._idx_to_obj_id.get(entry["idx"], "")
        bps_slot_data = self._load_bps_slot(obj_id) if self.use_slot_bps else None

        if bps_slot_data is not None:
            # Use BPS from slot cache (per-object normalized BPS distances)
            result["bps_object"] = torch.from_numpy(bps_slot_data["bps_dists"])
            result["bps_slot_labels"] = torch.from_numpy(bps_slot_data["bps_slot_labels"])
        else:
            # Fallback to OIShape BPS (no slot labels)
            if "bps_object" in oi:
                result["bps_object"] = torch.from_numpy(
                    np.asarray(oi["bps_object"], dtype=np.float32)
                )
            else:
                result["bps_object"] = torch.zeros(4096)
            result["bps_slot_labels"] = torch.full((4096,), -1, dtype=torch.long)

        # GT rotmat for CoarseNet encoder
        hand_pose = oi["hand_pose"]
        hand_pose_t = torch.from_numpy(hand_pose).float()
        # Convert axis-angle to rotation matrix for global orient
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
            bps_slot_cache_dir=cfg.bps_slot_cache_dir,
            use_slot_bps=cfg.use_slot_bps)

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
                use_slot_bps=cfg.use_slot_bps)
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
            use_slot_bps=cfg.use_slot_bps,
        ).to(self.device)

    def _build_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay, betas=(0.9, 0.99))

        total = self.cfg.epochs * len(self.train_dl)
        self.scheduler = cosine_warmup_schedule(
            self.optimizer, self.cfg.warmup_steps, total)

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

    def _log_info(self):
        n_model = self.model.count_parameters()
        cfg = self.cfg
        self.logger.info("=" * 60)
        self.logger.info("PoseGrabModel — GrabNet-style CVAE + RefineNet")
        self.logger.info(f"  Model:         {n_model:,} params ({n_model/1e6:.2f}M)")
        self.logger.info(f"  Output:        rot6d={ROT6D_POSE_DIM} + trans={TRANS_DIM} (no shape)")
        self.logger.info(f"  Vocab size:    {cfg.vocab_size}")
        self.logger.info(f"  KL coef:       {cfg.kl_coef}")
        self.logger.info(f"  Slot BPS:      {cfg.use_slot_bps} "
                         f"(grounder_w={cfg.slot_grounder_weight})")
        self.logger.info(f"  Token aug:     drop_p={cfg.token_drop_prob}  sub_p={cfg.token_sub_prob}")
        self.logger.info(f"  Train:         {len(self.train_ds)} samples, "
                         f"{len(self.train_dl)} steps/epoch")
        self.logger.info("=" * 60)

    # ================================================================
    # Train step — CVAE
    # ================================================================

    def _train_step(self, batch) -> Dict[str, float]:
        cfg = self.cfg

        bps_object = batch["bps_object"].to(self.device)
        tokens     = batch["token_seq"].to(self.device)
        t_mask     = batch["token_mask"].to(self.device)
        gt_trans   = batch["hand_tsl"].to(self.device)
        gt_orient_rotmat = batch["global_orient_rhand_rotmat"].to(self.device)
        gt_verts   = batch["hand_verts"].to(self.device)
        obj_pc     = batch["obj_pc"].to(self.device)
        obj_vn     = batch["obj_vn"].to(self.device)

        # ---- Augmentation ----
        if cfg.token_drop_prob > 0:
            t_mask = apply_token_dropout(t_mask, cfg.token_drop_prob)
        if cfg.token_sub_prob > 0:
            tokens = apply_token_substitution(
                tokens, t_mask, cfg.token_sub_prob,
                n_semantic=cfg.vocab_size - 2,
            )

        self.optimizer.zero_grad()

        # ---- Conditioning ----
        if cfg.use_slot_bps:
            bps_slot_labels = batch["bps_slot_labels"].to(self.device)

            # Slot-modulated BPS feature
            slot_bps_feat = self.model.encode_slot_bps(
                bps_object, bps_slot_labels, tokens, t_mask)

            # SlotGrounder auxiliary loss (co-train to predict slot labels)
            tok_summary = self.model.slot_bps._encode_token_summary(tokens, t_mask)
            grounder_logits = self.model.slot_grounder(bps_object, tok_summary)
            loss_grounder = self.model.slot_grounder.compute_loss(
                grounder_logits, bps_slot_labels)

            # CoarseNet forward: encoder gets raw BPS, decoder gets slot BPS
            drec = self.model.forward_coarse(
                bps_object, slot_bps_feat, gt_trans, gt_orient_rotmat,
            )
        else:
            token_cond = self.model.encode_tokens(tokens, t_mask)
            drec = self.model.forward_coarse(
                bps_object, bps_object, gt_trans, gt_orient_rotmat,
                token_cond=token_cond,
            )
            loss_grounder = torch.tensor(0.0, device=self.device)

        # MANO forward for predicted verts
        with torch.cuda.amp.autocast(enabled=False):
            drec_float32 = {k: v.float() if isinstance(v, torch.Tensor) else v
                           for k, v in drec.items()}
            verts_pred = self.mano.forward_verts(
                drec_float32["fullpose_aa"],
                drec_float32["transl"],
                torch.zeros(gt_trans.shape[0], cfg.n_betas, device=self.device),
            )

        # ---- Loss ----
        B = gt_trans.shape[0]
        dorig = {"vpe": self.vpe} if self.vpe is not None else {}
        rh_faces = self.rh_faces[:B]

        loss_total, loss_dict = self.model.compute_coarse_loss(
            drec, dorig, verts_pred, gt_verts, obj_pc,
            rh_faces, self.v_weights, self.v_weights2, B,
        )

        # Add SlotGrounder loss
        if cfg.use_slot_bps and cfg.slot_grounder_weight > 0:
            loss_total = loss_total + cfg.slot_grounder_weight * loss_grounder
            loss_dict["loss_grounder"] = loss_grounder
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
    # Validation
    # ================================================================

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        if self.val_dl is None:
            return {}
        self.model.eval()
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

            if cfg.use_slot_bps:
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
            else:
                token_cond = self.model.encode_tokens(tokens, t_mask)
                drec = self.model.forward_coarse(
                    bps_object, bps_object, gt_trans, gt_orient_rotmat,
                    token_cond=token_cond,
                )

            verts_pred = self.mano.forward_verts(
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

            # Sample-based metrics (use SlotGrounder at inference)
            if cfg.use_slot_bps:
                pred = self.model.sample(bps_object, tokens, t_mask,
                                         bps_slot_labels=bps_slot_labels)
            else:
                pred = self.model.sample(bps_object, tokens, t_mask)

            pred_verts = self.mano.forward_verts(
                pred.pose, pred.trans,
                torch.zeros(B, cfg.n_betas, device=self.device),
            )

            _, h2o_signed, _ = point2point_signed(
                pred_verts, obj_pc, y_normals=obj_vn,
            )
            pen_depth = torch.relu(-h2o_signed)
            contact_mask = h2o_signed.abs() < cfg.contact_thresh
            sample_has_contact = contact_mask.any(dim=1).float()

            for k, v in loss_dict.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                totals[k] += val * B
            totals["pen_depth_mm"] += pen_depth.mean().item() * 1000.0 * B
            totals["contact_rate"] += sample_has_contact.mean().item() * B
            n += B

        self.model.train()
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
        # Store use_slot_bps flag at top level for easy access by generate_grasps
        save_dict["use_slot_bps"] = self.cfg.use_slot_bps
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
                    grnd = f"  grnd={log.get('loss_grounder', 0):.4f}" if cfg.use_slot_bps else ""
                    self.logger.info(
                        f"  S{self.global_step:06d}  "
                        f"total={log.get('loss_total', 0):.4f}  "
                        f"kl={log.get('loss_kl', 0):.4f}  "
                        f"mesh={log.get('loss_mesh_rec', 0):.4f}  "
                        f"dist_h={log.get('loss_dist_h', 0):.4f}  "
                        f"dist_o={log.get('loss_dist_o', 0):.4f}  "
                        f"edge={log.get('loss_edge', 0):.4f}"
                        f"{grnd}")

                    if self.tb:
                        for k, v in log.items():
                            if isinstance(v, (int, float)):
                                self.tb.add_scalar(f"train/{k}", v, self.global_step)

                if self.global_step % cfg.val_every == 0:
                    val = self._validate()
                    if val:
                        vt = val.get("val/loss_total", float("inf"))
                        self.logger.info(
                            f"  [VAL]    "
                            f"total={vt:.4f}  "
                            f"contact={val.get('val/contact_rate', 0):.3f}  "
                            f"pen={val.get('val/pen_depth_mm', 0):.2f}mm")

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
    p = argparse.ArgumentParser(description="Train PoseGrabModel (CVAE + RefineNet)")

    p.add_argument("--cache_dir",     type=str, default="cache/contact_tokens/train")
    p.add_argument("--val_cache_dir", type=str, default="cache/contact_tokens/val")
    p.add_argument("--n_pc_points",   type=int, default=2048)
    p.add_argument("--bps_path",      type=str, default="grabnet/configs/bps.npz")

    p.add_argument("--model_config",  type=str, default="base")
    p.add_argument("--kl_coef",       type=float, default=0.005)

    # Slot-grounded BPS
    p.add_argument("--use_slot_bps", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable slot-grounded BPS conditioning (default: True)")
    p.add_argument("--bps_slot_cache_dir", type=str, default="cache/bps_slot",
                   help="Directory for per-object BPS slot cache")
    p.add_argument("--slot_grounder_weight", type=float, default=1.0,
                   help="Weight for SlotGrounder CE loss")

    p.add_argument("--epochs",       type=int,   default=100)
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
        use_slot_bps=args.use_slot_bps,
        bps_slot_cache_dir=args.bps_slot_cache_dir,
        slot_grounder_weight=args.slot_grounder_weight,
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
