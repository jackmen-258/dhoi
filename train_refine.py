"""
train_refine.py — RefineNet Training with PoseDecoderModel v23
===============================================================
Pipeline: frozen PoseDecoderModel → coarse pose → RefineNet → refined pose

v2 适配 PoseDecoderModel v23 (纯 Transformer 回归):
  - PoseDecoderModel 替代 DiffusionPoseModel (无 DDIM, 无 norm stats)
  - coarse 生成: 单次 forward (确定性回归), 不再需要 num_steps
  - 移除 diff_steps, ddim_coarse_steps 等扩散相关配置
  - PointNet2 编码传入 obj_global/obj_point/obj_xyz

训练策略:
  1. 冻结 PointNet2 + PoseDecoderModel, 单次 forward 生成 coarse pose
  2. RefineNet 迭代修正 (MANO forward 有梯度, h2o_dist 作为物体信息)
  3. 三项损失:
     - vertex L1:    ‖refined_verts − gt_verts‖₁
     - penetration:  惩罚手穿入物体的顶点 (h2o_signed < 0)
     - contact:      鼓励 GT 接触区域在 refined 结果中保持接触

用法:
  python train_refine.py --decoder_ckpt checkpoints/decoder/best.pt
"""

import os
import sys
import json
import math
import time
import random
import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict

# Ensure project root is importable even when launching via absolute script path
# from outside repository root.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast

from data.token_encoder import TokenEncoder
from models.object_normalization import normalize_object_points
from models.point_encoder import PointNet2Encoder
from models.pose_decoder import build_pose_model
from models.refine import RefineNet, build_refine_net, point2point_signed
from utils.mano_utils import MANOHelper


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

    # ---- pretrained pose decoder (v23, Transformer regression) ----
    decoder_ckpt:     str = "checkpoints/decoder/best.pt"
    decoder_config:   str = "base"
    vocab_size:       int = 112
    obj_global_dim:   int = 256
    obj_point_dim:    int = 256
    obj_xyz_dim:      int = 3

    # ---- refine model ----
    refine_config: str = "base"

    # ---- training ----
    epochs:            int   = 50
    batch_size:        int   = 128
    lr:                float = 3e-4
    weight_decay:      float = 0.01
    warmup_steps:      int   = 500
    max_grad_norm:     float = 1.0
    balanced_sampling: bool  = False

    # ---- AMP ----
    use_amp: bool = True

    # ---- loss weights ----
    lambda_vertex:  float = 10.0       # vertex L1 loss
    lambda_pen:     float = 5.0        # penetration penalty
    lambda_contact: float = 2.0        # contact attraction

    # ---- physics thresholds (meters) ----
    contact_thresh:    float = 0.005   # 5mm

    # ---- logging / save ----
    log_dir:    str = "logs/refine"
    save_dir:   str = "checkpoints/refine"
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


def _load_oishape(split, category="all", intent="all", n_samples=10000):
    from data.oishape_dataset import OIShape
    cfg = OIShapeConfig(split, category, intent)
    oi = OIShape(cfg)
    oi.n_samples = n_samples
    return oi


# ==============================================================================
# Dataset
# ==============================================================================

class RefineDataset(Dataset):
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

        with open(os.path.join(cache_dir, "dataset_index.json")) as f:
            index = json.load(f)
        self.samples = [s for s in index["samples"] if s["token_length"] >= 1]

        logging.info(f"[RefineDataset] Loading OIShape split={split}, "
                     f"n_samples={n_pc_points} ...")
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
            self._npz_cache[cache_file] = {
                "token_seq":           data["token_seq"].astype(np.int64),
                "contact_type_matrix": data["contact_type_matrix"].astype(np.int32),
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
            "obj_pc":      torch.from_numpy(oi["obj_verts"]),
            "obj_vn":      torch.from_numpy(oi["obj_vn"]),
            "hand_pose":   torch.from_numpy(oi["hand_pose"]),
            "hand_shape":  torch.from_numpy(oi["hand_shape"]),
            "hand_tsl":    torch.from_numpy(oi["hand_tsl"]),
        }


def _collate(batch):
    keys = ["token_seq", "token_mask",
            "obj_pc", "obj_vn", "hand_pose", "hand_shape", "hand_tsl"]
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
        self._build_frozen_decoder()
        self._build_model()
        self._build_optimizer()

        self.scaler = GradScaler(enabled=cfg.use_amp)

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

        sampler = None
        shuffle = True
        if cfg.balanced_sampling:
            cate_counts = defaultdict(int)
            for s in self.train_ds.samples:
                cate_counts[s["cate_id"]] += 1
            weights = [1.0 / cate_counts[s["cate_id"]]
                       for s in self.train_ds.samples]
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            shuffle = False
            self.logger.info(f"Balanced sampling: {len(cate_counts)} categories")

        self.train_dl = DataLoader(
            self.train_ds, batch_size=cfg.batch_size,
            shuffle=shuffle, sampler=sampler,
            num_workers=cfg.num_workers, pin_memory=True,
            drop_last=True, collate_fn=_collate)

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

    def _build_frozen_decoder(self):
        """加载预训练的 PointNet2 + PoseDecoderModel (v23), 冻结所有参数."""
        cfg = self.cfg

        self.obj_enc = PointNet2Encoder(
            in_dim=6,
            global_dim=cfg.obj_global_dim,
            point_dim=cfg.obj_point_dim,
        ).to(self.device)

        # v23: PoseDecoderModel (纯 Transformer 回归, 无扩散)
        self.pose_decoder = build_pose_model(
            cfg.decoder_config,
            vocab_size=cfg.vocab_size,
            obj_global_dim=cfg.obj_global_dim,
            obj_point_dim=cfg.obj_point_dim,
            obj_xyz_dim=cfg.obj_xyz_dim,
        ).to(self.device)

        # 加载 checkpoint
        if os.path.isfile(cfg.decoder_ckpt):
            ckpt = torch.load(cfg.decoder_ckpt, map_location=self.device)
            self.pose_decoder.load_state_dict(ckpt["model"])
            if "obj_enc" in ckpt:
                self.obj_enc.load_state_dict(ckpt["obj_enc"])
                self.logger.info(f"Loaded object encoder from: {cfg.decoder_ckpt}")
            else:
                self.logger.warning("Object encoder weights not found in decoder checkpoint; using random frozen obj_enc")
            self.logger.info(f"Loaded pose decoder from: {cfg.decoder_ckpt}")
        else:
            self.logger.warning(f"Decoder checkpoint not found: {cfg.decoder_ckpt}")
            self.logger.warning("  RefineNet will train with random coarse poses!")

        # 冻结
        self.pose_decoder.eval()
        self.obj_enc.eval()
        for p in self.pose_decoder.parameters():
            p.requires_grad_(False)
        for p in self.obj_enc.parameters():
            p.requires_grad_(False)
        self.logger.info(f"Pose decoder frozen: "
                         f"{sum(p.numel() for p in self.pose_decoder.parameters()):,} params")
        self.logger.info(f"Object encoder frozen: "
                         f"{sum(p.numel() for p in self.obj_enc.parameters()):,} params")

    def _build_model(self):
        cfg = self.cfg
        self.model = build_refine_net(cfg.refine_config).to(self.device)
        self.model.mano_fn = self.mano

    def _build_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay, betas=(0.9, 0.99))

        total = self.cfg.epochs * len(self.train_dl)
        self.scheduler = cosine_warmup_schedule(
            self.optimizer, self.cfg.warmup_steps, total)

    def _log_info(self):
        n   = self.model.count_parameters()
        cfg = self.cfg
        self.logger.info("=" * 60)
        self.logger.info("RefineNet — GrabNet-Style Iterative Pose Refinement")
        self.logger.info(f"  RefineNet:     {n:,} params ({n/1e6:.2f}M)")
        self.logger.info(f"  Iterations:    {self.model.n_iters}")
        self.logger.info(f"  Object info:   h2o_dist only (coarse from PointNet2+Decoder)")
        self.logger.info(f"  MANO gradient: YES (inside iterations)")
        self.logger.info(f"  Coarse source: PoseDecoderModel v23 (deterministic)")
        self.logger.info(f"  Loss λ:        vertex={cfg.lambda_vertex}  "
                         f"pen={cfg.lambda_pen}  contact={cfg.lambda_contact}")
        self.logger.info(f"  Contact thresh: {cfg.contact_thresh * 1000:.1f}mm")
        self.logger.info(f"  AMP:           {cfg.use_amp}")
        self.logger.info(f"  Train:         {len(self.train_ds)} samples, "
                         f"{len(self.train_dl)} steps/epoch")
        self.logger.info("=" * 60)

    # ================================================================
    # Object encoding
    # ================================================================

    def _encode_obj(self, batch):
        obj_pc = batch["obj_pc"].to(self.device)
        obj_vn = batch["obj_vn"].to(self.device)
        obj_pc_norm, _ = normalize_object_points(obj_pc)
        obj_in = torch.cat([obj_pc_norm, obj_vn], dim=-1)
        with torch.no_grad():
            obj_global_feat, obj_point_feat, obj_point_xyz = self.obj_enc(obj_in)
        return obj_global_feat, obj_point_feat, obj_point_xyz

    # ================================================================
    # Generate coarse pose from frozen PoseDecoderModel
    # ================================================================

    @torch.no_grad()
    def _generate_coarse(self, batch, obj_global_feat, obj_point_feat, obj_point_xyz):
        """用冻结的 PoseDecoderModel 生成 coarse pose (单次 forward, 无 DDIM)."""
        tokens = batch["token_seq"].to(self.device)
        t_mask = batch["token_mask"].to(self.device)

        cond = self.pose_decoder.encode_condition(
            tokens, t_mask, obj_global_feat, obj_point_feat, obj_point_xyz, obj_pc=batch["obj_pc"].to(self.device))
        out = self.pose_decoder.sample(cond)  # 确定性回归, 无 num_steps

        return out.pose, out.trans, out.shape

    # ================================================================
    # Loss computation
    # ================================================================

    def _compute_losses(self, refined_verts, gt_verts, obj_pc, obj_vn):
        """三项损失: vertex L1 + penetration + contact."""
        # 1. Vertex L1
        loss_vertex = F.l1_loss(refined_verts, gt_verts)

        # 2. Penetration (h2o_signed < 0 → hand inside object)
        _, h2o_signed, _ = point2point_signed(
            refined_verts, obj_pc, y_normals=obj_vn)
        penetration = F.relu(-h2o_signed)
        loss_pen = penetration.mean()

        # 3. Contact (GT contact regions should stay close)
        with torch.no_grad():
            _, gt_h2o, _ = point2point_signed(gt_verts, obj_pc)
            contact_mask = (gt_h2o.abs() < self.cfg.contact_thresh)
            n_contact = contact_mask.float().sum().clamp(min=1)

        loss_contact = (h2o_signed.abs() * contact_mask.float()).sum() / n_contact

        with torch.no_grad():
            pen_ratio = (h2o_signed < 0).float().mean()

        return loss_vertex, loss_pen, loss_contact, {"pen_ratio": pen_ratio.item()}

    # ================================================================
    # Train step
    # ================================================================

    def _train_step(self, batch) -> Dict[str, float]:
        cfg    = self.cfg
        device = self.device

        obj_pc   = batch["obj_pc"].to(device)
        obj_vn   = batch["obj_vn"].to(device)
        gt_pose  = batch["hand_pose"].to(device)
        gt_trans = batch["hand_tsl"].to(device)
        gt_shape = batch["hand_shape"].to(device)

        # 1. PointNet2 编码 (frozen)
        obj_global_feat, obj_point_feat, obj_point_xyz = self._encode_obj(batch)

        # 2. 生成 coarse pose (frozen decoder, 确定性)
        coarse_pose, coarse_trans, _ = self._generate_coarse(
            batch, obj_global_feat, obj_point_feat, obj_point_xyz)

        # 3. RefineNet 迭代修正 (有梯度)
        self.optimizer.zero_grad()

        with autocast(enabled=cfg.use_amp):
            refined_parms = self.model.forward_simple(
                coarse_pose, coarse_trans, gt_shape, obj_pc,
                obj_vn=obj_vn)

            refined_pose_aa = torch.cat([
                refined_parms['global_orient'],
                refined_parms['hand_pose'],
            ], dim=1)
            refined_verts, _ = self.mano(
                refined_pose_aa, refined_parms['transl'], gt_shape)

            with torch.no_grad():
                gt_verts, _ = self.mano(gt_pose, gt_trans, gt_shape)

            # 4. 损失计算
            loss_vertex, loss_pen, loss_contact, extra = self._compute_losses(
                refined_verts, gt_verts, obj_pc, obj_vn)

        loss = (cfg.lambda_vertex  * loss_vertex +
                cfg.lambda_pen     * loss_pen +
                cfg.lambda_contact * loss_contact)

        self.scaler.scale(loss).backward()
        if cfg.max_grad_norm > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return {
            "total":    loss.item(),
            "vertex":   loss_vertex.item(),
            "pen":      loss_pen.item(),
            "contact":  loss_contact.item(),
            **extra,
        }

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

        for batch in self.val_dl:
            device = self.device
            obj_pc   = batch["obj_pc"].to(device)
            obj_vn   = batch["obj_vn"].to(device)
            gt_pose  = batch["hand_pose"].to(device)
            gt_trans = batch["hand_tsl"].to(device)
            gt_shape = batch["hand_shape"].to(device)

            obj_global_feat, obj_point_feat, obj_point_xyz = self._encode_obj(batch)
            coarse_pose, coarse_trans, _ = self._generate_coarse(
                batch, obj_global_feat, obj_point_feat, obj_point_xyz)

            refined_parms = self.model.forward_simple(
                coarse_pose, coarse_trans, gt_shape, obj_pc, obj_vn=obj_vn)

            refined_pose_aa = torch.cat([
                refined_parms['global_orient'],
                refined_parms['hand_pose'],
            ], dim=1)
            refined_verts, _ = self.mano(
                refined_pose_aa, refined_parms['transl'], gt_shape)
            gt_verts, _ = self.mano(gt_pose, gt_trans, gt_shape)

            B = gt_pose.shape[0]

            # Vertex error (mm)
            vert_err = (refined_verts - gt_verts).norm(dim=-1).mean()
            totals["vert_err_mm"] += vert_err.item() * 1000 * B

            # Coarse baseline
            coarse_verts, _ = self.mano(coarse_pose, coarse_trans, gt_shape)
            coarse_vert_err = (coarse_verts - gt_verts).norm(dim=-1).mean()
            totals["coarse_vert_err_mm"] += coarse_vert_err.item() * 1000 * B

            # Improvement
            improvement = (coarse_vert_err - vert_err) / coarse_vert_err.clamp(min=1e-6)
            totals["improvement_pct"] += improvement.item() * 100 * B

            # Penetration metrics
            _, h2o_signed, _ = point2point_signed(
                refined_verts, obj_pc, y_normals=obj_vn)
            pen_depth = F.relu(-h2o_signed)
            totals["pen_depth_mm"] += pen_depth.mean().item() * 1000 * B
            totals["pen_ratio"] += (h2o_signed < 0).float().mean().item() * B

            # Contact metrics
            _, gt_h2o, _ = point2point_signed(gt_verts, obj_pc)
            contact_mask = (gt_h2o.abs() < self.cfg.contact_thresh)
            if contact_mask.any():
                contact_dist = h2o_signed.abs()[contact_mask].mean()
                totals["contact_dist_mm"] += contact_dist.item() * 1000 * B

            n += B

        self.model.train()
        return {f"val/{k}": v / max(n, 1) for k, v in totals.items()}

    # ================================================================
    # Save / Load
    # ================================================================

    def _save_checkpoint(self, tag="latest"):
        path = os.path.join(self.cfg.save_dir, f"{tag}.pt")
        torch.save({
            "model":       self.model.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "scheduler":   self.scheduler.state_dict(),
            "scaler":      self.scaler.state_dict(),
            "epoch":       self.epoch,
            "global_step": self.global_step,
            "best_val":    self.best_val,
            "config":      self.cfg.__dict__,
        }, path)
        self.logger.info(f"  Saved: {path}")

    def _load_checkpoint(self, path):
        self.logger.info(f"Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.epoch       = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)
        self.best_val    = ckpt.get("best_val", float("inf"))
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

        for ep in range(self.epoch, cfg.epochs):
            self.epoch = ep
            self.logger.info(f"Epoch {ep}/{cfg.epochs}")

            epoch_losses = defaultdict(float)
            n_steps = 0
            t0 = time.time()

            for batch in self.train_dl:
                log = self._train_step(batch)
                self.global_step += 1
                n_steps += 1
                for k, v in log.items():
                    epoch_losses[k] += v

                if self.global_step % cfg.log_every == 0:
                    parts = [
                        f"S{self.global_step:06d}",
                        f"loss={log['total']:.4f}",
                        f"vert={log['vertex']:.5f}",
                        f"pen={log['pen']:.5f}",
                        f"cntct={log['contact']:.5f}",
                        f"pen%={log['pen_ratio']:.3f}",
                    ]
                    self.logger.info("  " + "  ".join(parts))

                    if self.tb:
                        for k, v in log.items():
                            self.tb.add_scalar(f"train/{k}", v, self.global_step)

                if self.global_step % cfg.val_every == 0:
                    val = self._validate()
                    if val:
                        refined_err = val.get("val/vert_err_mm", 999)
                        coarse_err  = val.get("val/coarse_vert_err_mm", 999)
                        improv      = val.get("val/improvement_pct", 0)
                        pen_mm      = val.get("val/pen_depth_mm", 0)
                        contact_mm  = val.get("val/contact_dist_mm", 0)

                        self.logger.info(
                            f"  [VAL] refined={refined_err:.2f}mm "
                            f"(coarse={coarse_err:.2f}mm, improv={improv:.1f}%) "
                            f"pen={pen_mm:.3f}mm  contact={contact_mm:.3f}mm")

                        if self.tb:
                            for k, v in val.items():
                                self.tb.add_scalar(k, v, self.global_step)

                        if refined_err < self.best_val:
                            self.best_val = refined_err
                            self._save_checkpoint("best")
                            self.logger.info(f"  ★ New best: {refined_err:.2f}mm")

                if self.global_step % cfg.save_every == 0:
                    self._save_checkpoint("latest")
                    self._save_checkpoint(f"step_{self.global_step:07d}")
                    self._cleanup_checkpoints()

            dt = time.time() - t0
            avg = {k: v / max(n_steps, 1) for k, v in epoch_losses.items()}
            self.logger.info(
                f"  Epoch {ep} done  {dt:.0f}s  "
                f"avg_loss={avg['total']:.4f}  "
                f"avg_pen%={avg.get('pen_ratio', 0):.3f}")

        self._save_checkpoint("final")
        self.logger.info("Training complete.")


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train RefineNet (GrabNet-style)")

    p.add_argument("--cache_dir",     type=str, default="cache/contact_tokens/train")
    p.add_argument("--val_cache_dir", type=str, default="cache/contact_tokens/val")
    p.add_argument("--n_pc_points",   type=int, default=2048)

    p.add_argument("--decoder_ckpt",    type=str, default="checkpoints/decoder/best.pt")
    p.add_argument("--decoder_config",  type=str, default="base")
    p.add_argument("--obj_global_dim",  type=int, default=256)
    p.add_argument("--obj_point_dim",   type=int, default=256)
    p.add_argument("--obj_xyz_dim",     type=int, default=3)

    p.add_argument("--refine_config", type=str, default="base")

    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch_size", type=int,   default=128)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--no_amp",     action="store_true")

    p.add_argument("--lambda_vertex",  type=float, default=10.0)
    p.add_argument("--lambda_pen",     type=float, default=5.0)
    p.add_argument("--lambda_contact", type=float, default=2.0)
    p.add_argument("--contact_thresh", type=float, default=0.005)

    p.add_argument("--resume",   type=str, default="")
    p.add_argument("--save_dir", type=str, default="checkpoints/refine")
    p.add_argument("--log_dir",  type=str, default="logs/refine")
    p.add_argument("--device",   type=str, default="cuda")
    p.add_argument("--seed",     type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(
        cache_dir=args.cache_dir,
        val_cache_dir=args.val_cache_dir,
        n_pc_points=args.n_pc_points,
        decoder_ckpt=args.decoder_ckpt,
        decoder_config=args.decoder_config,
        obj_global_dim=args.obj_global_dim,
        obj_point_dim=args.obj_point_dim,
        obj_xyz_dim=args.obj_xyz_dim,
        refine_config=args.refine_config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_vertex=args.lambda_vertex,
        lambda_pen=args.lambda_pen,
        lambda_contact=args.lambda_contact,
        contact_thresh=args.contact_thresh,
        use_amp=not args.no_amp,
        resume=args.resume,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        device=args.device,
        seed=args.seed,
    )
    Trainer(cfg).train()


if __name__ == "__main__":
    main()
