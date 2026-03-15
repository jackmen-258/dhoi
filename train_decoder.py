"""
train_decoder.py (Flow Matching — OT-CFM)
==========================================
PoseFlowModel training script.

Changes from regression version:
  - Training loss: OT-CFM velocity matching (single MSE, no lambda weighting)
  - Validation: Euler ODE sampling → geo/trs/vert metrics
  - lambda_rot/trans/shape/vert retained ONLY for val-score checkpoint selection

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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from data.token_encoder import TokenEncoder
from models.object_normalization import normalize_object_points
from models.point_encoder import PointNet2Encoder
from models.pose_decoder import build_flow_model, ROT6D_POSE_DIM, TRANS_DIM
from utils.pose_utils import aa_to_rot6d
from utils.pose_utils import _aa_to_rotmat_stable as _aa_to_rotmat
from utils.pose_utils import _rotmat_to_aa_stable as _rotmat_to_aa
from utils.mano_utils import MANOHelper


# ==============================================================================
# Unified metric helpers
# ==============================================================================

def vert_l2_mm(pred_verts, gt_verts):
    """Per-vertex L2 norm, mean over vertices, in mm."""
    return (pred_verts - gt_verts).norm(dim=-1).mean() * 1000.0


def trans_rmse_mm(pred_trans, gt_trans):
    """Translation RMSE in mm."""
    return torch.sqrt(((pred_trans - gt_trans) ** 2).mean()) * 1000.0


# ==============================================================================
# Contact Token Dropout
# ==============================================================================

@torch.no_grad()
def apply_token_dropout(token_mask, drop_prob):
    """
    随机 drop 部分 valid contact tokens.

    Args:
        token_mask: (B, L) bool — True = 有效 token
        drop_prob:  float — 每个 valid token 被 drop 的概率

    Returns:
        new token_mask, 保证每个样本至少保留 1 个 valid token
    """
    if drop_prob <= 0:
        return token_mask

    B, L = token_mask.shape
    mask = token_mask.clone()

    # 只对 valid token 做 dropout
    drop = torch.rand(B, L, device=mask.device) < drop_prob
    mask = mask & ~drop

    # 兜底: 如果某个样本的 valid token 全被 drop 了, 恢复原始 mask
    empty = ~mask.any(dim=1)
    if empty.any():
        mask[empty] = token_mask[empty]

    return mask


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

    # ---- model ----
    model_config: str = "base"
    n_betas:      int = 10
    obj_global_dim: int = 256
    obj_point_dim:  int = 256
    obj_xyz_dim:    int = 3
    vocab_size:   int = 26

    # ---- training ----
    epochs:            int   = 200
    batch_size:        int   = 64
    lr:                float = 1e-4
    weight_decay:      float = 0.01
    warmup_steps:      int   = 1000
    max_grad_norm:     float = 1.0
    balanced_sampling: bool  = False

    # ---- augmentation ----
    token_drop_prob: float = 0.2

    # ---- flow matching ----
    n_sample_steps: int = 20     # Euler ODE steps for validation sampling

    # ---- val-score weights (NOT used for training loss) ----
    lambda_rot:   float = 1.0
    lambda_trans: float = 0.01
    lambda_shape: float = 1.0
    lambda_vert:  float = 0.02

    # ---- logging / save ----
    log_dir:    str = "logs/decoder"
    save_dir:   str = "checkpoints/decoder"
    log_every:  int = 50
    val_every:  int = 1000
    save_every: int = 5000
    save_top_k: int = 5

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
        self.oishape = _load_oishape(split, category, intent, n_samples=n_pc_points)
        logging.info(f"[DecoderDataset] OIShape: {len(self.oishape)} grasps")

        max_idx = len(self.oishape) - 1
        before = len(self.samples)
        self.samples = [s for s in self.samples if s["idx"] <= max_idx]
        if len(self.samples) < before:
            logging.warning(f"  Filtered {before - len(self.samples)} out-of-range samples")

        self._npz_cache = {}
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

class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg         = cfg
        self.device      = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.global_step = 0
        self.epoch       = 0
        self.best_val    = float("inf")

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
        self.train_ds = DecoderDataset(
            cfg.cache_dir, cfg.config_path, "train",
            n_pc_points=cfg.n_pc_points,
            category=cfg.oi_category, intent=cfg.oi_intent)

        sampler = None
        shuffle = True
        if cfg.balanced_sampling:
            cate_counts = defaultdict(int)
            for s in self.train_ds.samples:
                cate_counts[s["cate_id"]] += 1
            weights = [
                1.0 / math.sqrt(cate_counts[s["cate_id"]])
                for s in self.train_ds.samples
            ]
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            shuffle = False
            self.logger.info(
                f"Balanced sampling: sqrt-inverse over {len(cate_counts)} categories"
            )

        self.train_dl = DataLoader(
            self.train_ds, batch_size=cfg.batch_size,
            shuffle=shuffle, sampler=sampler,
            num_workers=cfg.num_workers, pin_memory=True,
            drop_last=True, collate_fn=_collate)

        self.val_dl = None
        if cfg.val_cache_dir and os.path.isdir(cfg.val_cache_dir):
            val_ds = DecoderDataset(
                cfg.val_cache_dir, cfg.config_path, "val",
                n_pc_points=cfg.n_pc_points,
                category=cfg.oi_category, intent=cfg.oi_intent)
            self.val_dl = DataLoader(
                val_ds, batch_size=cfg.batch_size, shuffle=False,
                num_workers=cfg.num_workers, pin_memory=True,
                collate_fn=_collate)

    def _build_model(self):
        cfg = self.cfg
        self.obj_enc = PointNet2Encoder(
            in_dim=6,
            global_dim=cfg.obj_global_dim,
            point_dim=cfg.obj_point_dim,
        ).to(self.device)

        self.model = build_flow_model(
            cfg.model_config,
            vocab_size=cfg.vocab_size,
            obj_global_dim=cfg.obj_global_dim,
            obj_point_dim=cfg.obj_point_dim,
            obj_xyz_dim=cfg.obj_xyz_dim,
            n_betas=cfg.n_betas,
        ).to(self.device)

    def _build_optimizer(self):
        trainable_params = list(self.model.parameters()) + list(self.obj_enc.parameters())
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay, betas=(0.9, 0.99))

        total = self.cfg.epochs * len(self.train_dl)
        self.scheduler = cosine_warmup_schedule(
            self.optimizer, self.cfg.warmup_steps, total)

    def _log_info(self):
        n_model = self.model.count_parameters()
        cfg = self.cfg
        self.logger.info("=" * 60)
        self.logger.info("PoseFlowModel — OT Conditional Flow Matching")
        self.logger.info(f"  Model:         {n_model:,} params ({n_model/1e6:.2f}M)")
        self.logger.info(f"  State dim:     {self.model.x_dim} "
                         f"(rot6d={ROT6D_POSE_DIM} + trans={TRANS_DIM} + shape={cfg.n_betas})")
        self.logger.info(f"  Vocab size:    {cfg.vocab_size}")
        self.logger.info(f"  Token drop:    p={cfg.token_drop_prob}")
        self.logger.info(f"  ODE steps:     {cfg.n_sample_steps} (val sampling)")
        self.logger.info(f"  Val-score λ:   rot={cfg.lambda_rot}  trans={cfg.lambda_trans}  "
                         f"shape={cfg.lambda_shape}  vert={cfg.lambda_vert}")
        self.logger.info(f"  Train:         {len(self.train_ds)} samples, "
                         f"{len(self.train_dl)} steps/epoch")
        self.logger.info("=" * 60)

    def _compute_val_score(self, val: Dict[str, float]) -> float:
        cfg = self.cfg
        return (
            cfg.lambda_rot * val.get("val/geo_rad", 0.0)
            + cfg.lambda_trans * val.get("val/trs_mm", 0.0)
            + cfg.lambda_shape * val.get("val/shape", 0.0)
            + cfg.lambda_vert * val.get("val/vert_mm", 0.0)
        )

    # ================================================================
    # Encode
    # ================================================================

    def _encode_obj(self, obj_pc, obj_vn, return_fps_indices: bool = False):
        """PointNet2 encode."""
        obj_pc_norm, _ = normalize_object_points(obj_pc)
        obj_input = torch.cat([obj_pc_norm, obj_vn], dim=-1)
        result = self.obj_enc(obj_input, return_fps_indices=return_fps_indices)
        if return_fps_indices:
            global_feat, point_feat, point_xyz, fps_indices = result
            return global_feat, point_feat, point_xyz, fps_indices
        global_feat, point_feat, point_xyz = result
        return global_feat, point_feat, point_xyz

    def _downsample_slot_labels(self, slot_labels: torch.Tensor, fps_indices: Dict[str, torch.Tensor]) -> torch.Tensor:
        fps2_idx = fps_indices["fps2"].to(device=slot_labels.device, dtype=torch.long)
        if slot_labels.dim() != 2:
            raise ValueError(f"slot_labels must be (B, N), got {slot_labels.shape}")
        if fps2_idx.dim() != 2:
            raise ValueError(f"fps2 must be (B, 128), got {fps2_idx.shape}")
        return torch.gather(slot_labels, 1, fps2_idx)

    # ================================================================
    # Train step — Flow Matching
    # ================================================================

    def _train_step(self, batch) -> Dict[str, float]:
        cfg = self.cfg

        obj_pc   = batch["obj_pc"].to(self.device)
        obj_vn   = batch["obj_vn"].to(self.device)
        tokens   = batch["token_seq"].to(self.device)
        t_mask   = batch["token_mask"].to(self.device)
        slot_labels = batch["slot_labels"].to(self.device)
        gt_pose  = batch["hand_pose"].to(self.device)
        gt_trans = batch["hand_tsl"].to(self.device)
        gt_shape = batch["hand_shape"].to(self.device)

        # ---- Augmentation ----
        if cfg.token_drop_prob > 0:
            t_mask = apply_token_dropout(t_mask, cfg.token_drop_prob)

        # ---- Object point encode ----
        obj_global_feat, obj_point_feat, obj_point_xyz, fps_indices = self._encode_obj(
            obj_pc, obj_vn, return_fps_indices=True
        )
        slot_labels_128 = self._downsample_slot_labels(slot_labels, fps_indices)

        # ---- Forward: flow matching ----
        self.optimizer.zero_grad()

        cond = self.model.encode_condition(
            tokens, t_mask, obj_global_feat,
            obj_point_feat, obj_point_xyz,
            slot_labels=slot_labels_128,
            obj_pc=obj_pc,
        )

        x_data = self.model.pack_data(gt_pose, gt_trans, gt_shape, cond)
        loss, metrics = self.model.compute_loss(x_data, cond)

        loss.backward()
        if cfg.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.obj_enc.parameters()),
                cfg.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        return metrics

    # ================================================================
    # Validation — ODE sampling + full metric computation
    # ================================================================

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        if self.val_dl is None:
            return {}
        self.model.eval()
        self.obj_enc.eval()
        totals = defaultdict(float)
        n = 0

        for i, batch in enumerate(self.val_dl):
            obj_pc = batch["obj_pc"].to(self.device)
            obj_vn = batch["obj_vn"].to(self.device)
            obj_global_feat, obj_point_feat, obj_point_xyz, fps_indices = self._encode_obj(
                obj_pc, obj_vn, return_fps_indices=True
            )
            tokens   = batch["token_seq"].to(self.device)
            t_mask   = batch["token_mask"].to(self.device)
            slot_labels = batch["slot_labels"].to(self.device)
            gt_pose  = batch["hand_pose"].to(self.device)
            gt_trans = batch["hand_tsl"].to(self.device)
            gt_shape = batch["hand_shape"].to(self.device)
            slot_labels_128 = self._downsample_slot_labels(slot_labels, fps_indices)

            cond = self.model.encode_condition(
                tokens, t_mask, obj_global_feat,
                obj_point_feat, obj_point_xyz,
                slot_labels=slot_labels_128,
                obj_pc=obj_pc,
            )
            out = self.model.sample(cond, n_steps=self.cfg.n_sample_steps)

            B = gt_pose.shape[0]

            from models.loss import geodesic_loss
            geo_rad = geodesic_loss(
                out.pose.view(B, -1, 3), gt_pose.view(B, -1, 3))
            trs_mm = trans_rmse_mm(out.trans, gt_trans)

            pred_verts, _ = self.mano(out.pose, out.trans, out.shape)
            gt_verts, _   = self.mano(gt_pose, gt_trans, gt_shape)
            v_mm = vert_l2_mm(pred_verts, gt_verts)

            l_shape = F.mse_loss(out.shape, gt_shape)

            totals["geo_rad"]  += geo_rad.item() * B
            totals["trs_mm"]   += trs_mm.item() * B
            totals["shape"]    += l_shape.item() * B
            totals["vert_mm"]  += v_mm.item() * B
            n += B

        self.model.train()
        self.obj_enc.train()

        return {f"val/{k}": v / max(n, 1) for k, v in totals.items()}

    # ================================================================
    # Save / Load
    # ================================================================

    def _save_checkpoint(self, tag="latest"):
        path = os.path.join(self.cfg.save_dir, f"{tag}.pt")
        torch.save({
            "model":       self.model.state_dict(),
            "obj_enc":     self.obj_enc.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "scheduler":   self.scheduler.state_dict(),
            "epoch":       self.epoch,
            "global_step": self.global_step,
            "best_val":    self.best_val,
            "best_score":  self.best_val,
            "config":      self.cfg.__dict__,
        }, path)
        self.logger.info(f"  Saved: {path}")

    def _load_checkpoint(self, path):
        self.logger.info(f"Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.obj_enc.load_state_dict(ckpt["obj_enc"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.epoch       = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)
        self.best_val    = ckpt.get("best_score", ckpt.get("best_val", float("inf")))
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
        self.obj_enc.train()
        self.logger.info(f"\nStart training: {cfg.epochs} epochs")

        for ep in range(self.epoch, cfg.epochs):
            self.epoch = ep
            self.logger.info(f"Epoch {ep}/{cfg.epochs}")

            epoch_loss = 0.0
            n_steps    = 0
            t0         = time.time()

            for batch in self.train_dl:
                log = self._train_step(batch)
                self.global_step += 1
                n_steps += 1
                epoch_loss += log["flow_loss"]

                if self.global_step % cfg.log_every == 0:
                    self.logger.info(
                        f"  S{self.global_step:06d}  "
                        f"flow={log['flow_loss']:.4f}  "
                        f"v_rot={log['v_rot']:.4f}  "
                        f"v_trans={log['v_trans']:.4f}  "
                        f"v_shape={log['v_shape']:.4f}")

                    if self.tb:
                        for k, v in log.items():
                            if isinstance(v, (int, float)):
                                self.tb.add_scalar(f"train/{k}", v, self.global_step)

                if self.global_step % cfg.val_every == 0:
                    val = self._validate()
                    if val:
                        vg = val.get("val/geo_rad", float("inf"))
                        vt = val.get("val/trs_mm", 0)
                        vv = val.get("val/vert_mm", float("inf"))
                        vs = self._compute_val_score(val)

                        self.logger.info(
                            f"  [VAL]    "
                            f"geo={vg:.3f}rad  "
                            f"trs={vt:.1f}mm  "
                            f"vert={vv:.1f}mm  "
                            f"score={vs:.4f}")

                        if self.tb:
                            for k, v in val.items():
                                self.tb.add_scalar(k, v, self.global_step)
                            self.tb.add_scalar("val/score", vs, self.global_step)

                        if vs < self.best_val:
                            self.best_val = vs
                            self._save_checkpoint("best")
                            self.logger.info(
                                f"  * New best: score={vs:.4f}  "
                                f"geo={vg:.3f}rad  trs={vt:.1f}mm  vert={vv:.1f}mm")

                if self.global_step % cfg.save_every == 0:
                    self._save_checkpoint("latest")
                    self._save_checkpoint(f"step_{self.global_step:07d}")
                    self._cleanup_checkpoints()

            dt = time.time() - t0
            self.logger.info(
                f"  Epoch {ep} done  {dt:.0f}s  "
                f"avg_flow_loss={epoch_loss / max(n_steps, 1):.4f}")

        self._save_checkpoint("final")
        self.logger.info("Training complete.")


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Train PoseFlowModel (OT-CFM)")

    p.add_argument("--cache_dir",     type=str, default="cache/contact_tokens/train")
    p.add_argument("--val_cache_dir", type=str, default="cache/contact_tokens/val")
    p.add_argument("--n_pc_points",   type=int, default=2048)

    p.add_argument("--model_config",  type=str, default="base")
    p.add_argument("--obj_global_dim", type=int, default=256)
    p.add_argument("--obj_point_dim",  type=int, default=256)
    p.add_argument("--obj_xyz_dim",    type=int, default=3)

    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--n_sample_steps", type=int, default=20)

    # Val-score weights (not used for training loss)
    p.add_argument("--lambda_rot",   type=float, default=1.0)
    p.add_argument("--lambda_trans", type=float, default=0.01)
    p.add_argument("--lambda_shape", type=float, default=1.0)
    p.add_argument("--lambda_vert",  type=float, default=0.02)

    p.add_argument("--resume",       type=str, default="")
    p.add_argument("--save_dir",     type=str, default="checkpoints/decoder")
    p.add_argument("--log_dir",      type=str, default="logs/decoder")
    p.add_argument("--seed",         type=int, default=7725)
    p.add_argument(
        "--balanced_sampling",
        action="store_true",
        help="使用 1/sqrt(category_count) 的弱类别均衡采样",
    )

    return p.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(
        cache_dir=args.cache_dir,
        val_cache_dir=args.val_cache_dir,
        n_pc_points=args.n_pc_points,
        model_config=args.model_config,
        obj_global_dim=args.obj_global_dim,
        obj_point_dim=args.obj_point_dim,
        obj_xyz_dim=args.obj_xyz_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_sample_steps=args.n_sample_steps,
        lambda_rot=args.lambda_rot,
        lambda_trans=args.lambda_trans,
        lambda_shape=args.lambda_shape,
        lambda_vert=args.lambda_vert,
        resume=args.resume,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        seed=args.seed,
        balanced_sampling=args.balanced_sampling,
    )
    Trainer(cfg).train()


if __name__ == "__main__":
    main()
