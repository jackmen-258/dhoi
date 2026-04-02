"""
train_diffusion.py (v16 — simplified-token diffusion trainer)
=============================================================
Contact token denoiser 训练脚本。

核心改动：
  - 默认词表切到简化版 token schema（vocab=26, PAD=24, MASK=25）
  - 去掉训练脚本外层 text-only CFG dropout，统一交给 denoiser.cond_dropout
  - training_loss 显式传入 token_mask，使 PAD 不参与扩散与监督
"""

import functools
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
from torch.cuda.amp import GradScaler, autocast

from data.token_encoder import TokenEncoder
from models.object_normalization import normalize_object_points
from models.point_encoder import PointNet2Encoder
from models.token_denoiser import build_denoiser
from models.discrete_diffusion import AbsorbingDiffusion
from models.clip_encoder import CLIPTextEncoder


@dataclass
class TrainConfig:
    # ---- data ----
    cache_dir:     str = "cache/contact_tokens/train"
    val_cache_dir: str = "cache/contact_tokens/val"
    config_path:   str = "configs/token_config.yaml"
    n_pc_points:   int = 2048
    oi_category:   str = "all"
    oi_intent:     str = "all"

    # ---- model ----
    denoiser_config: str = "base"
    obj_global_dim:  int = 256
    obj_point_dim:   int = 256
    clip_model:      str = "ViT-B/32"
    text_feat_dim:   int = 512
    use_text:        bool = True
    cond_dropout:    float = 0.2

    # ---- diffusion ----
    num_timesteps: int = 100
    schedule:      str = "cosine"
    vocab_size:    int = 26
    pad_token_id:  int = 24
    mask_token_id: int = 25
    min_tokens:    int = 1
    pad_logit_bias: float = 0.0

    # ---- training ----
    epochs:            int   = 100
    batch_size:        int   = 128
    lr:                float = 1e-4
    weight_decay:      float = 0.01
    warmup_steps:      int   = 2000
    max_grad_norm:     float = 1.0

    # ---- AMP ----
    use_amp: bool = True

    # ---- logging / save ----
    log_dir:         str = "logs/diffusion"
    save_dir:        str = "checkpoints/diffusion"
    log_every:       int = 50
    val_every:       int = 2000
    save_every:      int = 5000
    save_top_k:      int = 5
    num_val_samples: int = 16
    resume:          str = ""

    # ---- hardware ----
    num_workers: int = 4
    seed:        int = 42
    device:      str = "cuda"


class OIShapeConfig:
    def __init__(self, split, category="all", intent="all"):
        self.DATA_SPLIT = split
        self.OBJ_CATES = category
        self.INTENT_MODE = intent


def _load_oishape(split, category="all", intent="all", n_samples=2048):
    from data.oishape_dataset import OIShape
    cfg = OIShapeConfig(split, category, intent)
    oi = OIShape(cfg)
    oi.n_samples = n_samples
    return oi


class DiffusionTrainDataset(Dataset):
    """只加载 token denoiser 训练所需字段。"""

    def __init__(
        self,
        cache_dir: str,
        config_path: str,
        split: str = "train",
        n_pc_points: int = 2048,
        category: str = "all",
        intent: str = "all",
    ):
        self.cache_dir = cache_dir
        self.encoder = TokenEncoder(config_path)
        self.mapper = self.encoder.mapper

        index_path = os.path.join(cache_dir, "dataset_index.json")
        with open(index_path) as f:
            index = json.load(f)
        self.samples = [s for s in index["samples"] if s["token_length"] >= 1]

        logging.info(f"[Dataset] Loading OIShape split={split}, n_samples={n_pc_points} ...")
        self.oishape = _load_oishape(split, category, intent, n_samples=n_pc_points)
        logging.info(f"[Dataset] OIShape: {len(self.oishape)} grasps")

        max_oi_idx = len(self.oishape) - 1
        before = len(self.samples)
        self.samples = [s for s in self.samples if s["idx"] <= max_oi_idx]
        if len(self.samples) < before:
            logging.warning(f"  Filtered {before - len(self.samples)} out-of-range samples")

        self._text_map = {}
        tc_path = os.path.join(cache_dir, "text_cache.json")
        if os.path.isfile(tc_path):
            with open(tc_path, encoding="utf-8") as f:
                self._text_map = json.load(f)
            logging.info(f"  Text cache: {len(self._text_map)} entries")
        else:
            logging.warning(f"  text_cache.json not found at {cache_dir}")

        logging.info(f"[DiffusionTrainDataset] {len(self.samples)} samples")

    @functools.lru_cache(maxsize=1024)
    def _load_npz(self, cache_file: str) -> dict:
        path = os.path.join(self.cache_dir, cache_file)
        data = np.load(path, allow_pickle=True)
        return {"token_seq": data["token_seq"].astype(np.int64)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        entry = self.samples[i]
        oi = self.oishape[entry["idx"]]
        tok = self._load_npz(entry["cache_file"])

        token_seq = tok["token_seq"]
        max_len = self.encoder.max_token_length
        # 生成 token_mask：标记有效的 semantic token 位置
        valid_tokens = [self.mapper.is_semantic_token(int(t)) for t in token_seq]
        token_mask = np.zeros(max_len, dtype=bool)
        token_mask[:len(valid_tokens)] = valid_tokens

        obj_pc = oi["obj_verts"]
        obj_vn = oi["obj_vn"]

        return {
            "token_seq": torch.from_numpy(token_seq),
            "token_mask": torch.from_numpy(token_mask),
            "obj_pc": torch.from_numpy(obj_pc),
            "obj_vn": torch.from_numpy(obj_vn),
            "text": self._text_map.get(entry["cache_file"], ""),
            "sample_id": os.path.splitext(entry["cache_file"])[0],
            "cache_file": entry["cache_file"],
        }


def _collate_fn(batch):
    return {
        "token_seq": torch.stack([b["token_seq"] for b in batch]),
        "token_mask": torch.stack([b["token_mask"] for b in batch]),
        "obj_pc": torch.stack([b["obj_pc"] for b in batch]),
        "obj_vn": torch.stack([b["obj_vn"] for b in batch]),
        "text": [b["text"] for b in batch],
        "sample_id": [b["sample_id"] for b in batch],
        "cache_file": [b["cache_file"] for b in batch],
    }


def _semantic_token_hist(
    tokens: torch.Tensor,
    semantic_mask: torch.Tensor,
    n_semantic: int,
) -> torch.Tensor:
    """Build per-sample semantic token histograms while ignoring PAD/MASK/order."""
    tok_ids = tokens.clamp(min=0, max=n_semantic - 1)
    one_hot = F.one_hot(tok_ids, n_semantic).float()
    return (one_hot * semantic_mask.unsqueeze(-1).float()).sum(dim=1)


@torch.no_grad()
def compute_stage1_token_noise_stats(
    pred_tokens: torch.Tensor,
    gt_tokens: torch.Tensor,
    gt_token_mask: torch.Tensor,
    n_semantic: int,
) -> dict:
    """Decompose Stage-1 token errors into drop / substitution / insertion counts.

    Since the semantic token set is unordered, we compare multisets rather than
    sequence positions:
      - missing = GT semantic tokens not recovered by Stage 1
      - extra   = predicted semantic tokens absent from GT
      - sub     = min(missing, extra)
      - drop    = missing - sub
      - insert  = extra - sub

    This decomposition lets Stage 2 map observed Stage-1 errors to the two
    augmentation knobs it actually has: dropout and substitution.
    """
    pred_sem_mask = (pred_tokens >= 0) & (pred_tokens < n_semantic)
    gt_sem_mask = gt_token_mask.bool() & (gt_tokens >= 0) & (gt_tokens < n_semantic)

    pred_hist = _semantic_token_hist(pred_tokens, pred_sem_mask, n_semantic)
    gt_hist = _semantic_token_hist(gt_tokens, gt_sem_mask, n_semantic)

    matched = torch.minimum(pred_hist, gt_hist)
    gt_count = gt_hist.sum(dim=1)
    pred_count = pred_hist.sum(dim=1)
    missing = (gt_hist - matched).clamp(min=0).sum(dim=1)
    extra = (pred_hist - matched).clamp(min=0).sum(dim=1)
    sub = torch.minimum(missing, extra)
    drop = missing - sub
    insert = extra - sub

    return {
        "gt_count": gt_count,
        "pred_count": pred_count,
        "missing_count": missing,
        "extra_count": extra,
        "sub_count": sub,
        "drop_count": drop,
        "insert_count": insert,
    }


def cosine_warmup_schedule(optimizer, warmup, total):
    def fn(step):
        if step < warmup:
            return step / max(1, warmup)
        p = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


class DiffusionTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

        self._set_seed(cfg.seed)
        os.makedirs(cfg.save_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)
        self._setup_logging()

        self._build_models()
        self._build_dataloaders()
        self._build_optimizer()

        self.scaler = GradScaler(enabled=cfg.use_amp)

        if cfg.resume and os.path.isfile(cfg.resume):
            self._load_checkpoint(cfg.resume)

        self._log_model_info()

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(self.cfg.log_dir, "train.log")),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)
        self.tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(self.cfg.log_dir)
        except ImportError:
            pass

    def _build_models(self):
        cfg = self.cfg

        self.obj_enc = PointNet2Encoder(
            in_dim=6,
            global_dim=cfg.obj_global_dim,
            point_dim=cfg.obj_point_dim,
        ).to(self.device)

        self.denoiser = build_denoiser(
            cfg.denoiser_config,
            vocab_size=cfg.vocab_size,
            obj_feat_dim=cfg.obj_global_dim,
            obj_point_feat_dim=cfg.obj_point_dim,
            text_feat_dim=cfg.text_feat_dim if cfg.use_text else 0,
            cond_dropout=cfg.cond_dropout,
            pad_token_id=cfg.pad_token_id,
            mask_token_id=cfg.mask_token_id,
        ).to(self.device)

        self.diffusion = AbsorbingDiffusion(
            vocab_size=cfg.vocab_size,
            mask_token_id=cfg.mask_token_id,
            pad_token_id=cfg.pad_token_id,
            num_timesteps=cfg.num_timesteps,
            schedule=cfg.schedule,
            min_tokens=cfg.min_tokens,
            pad_logit_bias=cfg.pad_logit_bias,
        ).to(self.device)

        self.clip_enc = None
        if cfg.use_text:
            self.clip_enc = CLIPTextEncoder(cfg.clip_model, str(self.device))
            self.logger.info(f"CLIP encoder: {cfg.clip_model}")

    def _build_dataloaders(self):
        cfg = self.cfg

        self.train_ds = DiffusionTrainDataset(
            cache_dir=cfg.cache_dir,
            config_path=cfg.config_path,
            split="train",
            n_pc_points=cfg.n_pc_points,
            category=cfg.oi_category,
            intent=cfg.oi_intent,
        )

        sampler = None
        shuffle = True

        self.train_dl = DataLoader(
            self.train_ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=_collate_fn,
        )

        self.val_dl = None
        if cfg.val_cache_dir and os.path.isdir(cfg.val_cache_dir):
            val_ds = DiffusionTrainDataset(
                cache_dir=cfg.val_cache_dir,
                config_path=cfg.config_path,
                split="val",
                n_pc_points=cfg.n_pc_points,
                category=cfg.oi_category,
                intent=cfg.oi_intent,
            )
            self.val_dl = DataLoader(
                val_ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=True,
                collate_fn=_collate_fn,
            )

    def _build_split_dataloader(
        self,
        cache_dir: str,
        split: str,
        batch_size: int | None = None,
    ):
        ds = DiffusionTrainDataset(
            cache_dir=cache_dir,
            config_path=self.cfg.config_path,
            split=split,
            n_pc_points=self.cfg.n_pc_points,
            category=self.cfg.oi_category,
            intent=self.cfg.oi_intent,
        )
        dl = DataLoader(
            ds,
            batch_size=batch_size or self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            collate_fn=_collate_fn,
        )
        return ds, dl

    def _build_optimizer(self):
        cfg = self.cfg
        self.optimizer = torch.optim.AdamW(
            list(self.denoiser.parameters()) + list(self.obj_enc.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.99),
        )
        total = cfg.epochs * len(self.train_dl)
        self.scheduler = cosine_warmup_schedule(self.optimizer, cfg.warmup_steps, total)

    def _log_model_info(self):
        n = sum(p.numel() for p in self.denoiser.parameters() if p.requires_grad)
        n_obj = self.obj_enc.count_parameters()
        cfg = self.cfg
        self.logger.info("=" * 60)
        self.logger.info("Token Denoiser v16 — PointNet + PAD-aware Absorbing Diffusion")
        self.logger.info(f"  Denoiser:     {n:,} params ({n/1e6:.2f}M)")
        self.logger.info(f"  ObjEnc:       {n_obj:,} params ({n_obj/1e6:.2f}M)")
        self.logger.info(f"  PointEnc:     global={cfg.obj_global_dim}  point={cfg.obj_point_dim}")
        self.logger.info(f"  PC points:    {cfg.n_pc_points} (via OIShape.n_samples)")
        self.logger.info(f"  Timesteps:    {cfg.num_timesteps}")
        self.logger.info(f"  Vocab:        {cfg.vocab_size}  PAD={cfg.pad_token_id}  MASK={cfg.mask_token_id}")
        self.logger.info(f"  Text:         {'CLIP ' + cfg.clip_model if cfg.use_text else 'off'}  cond_drop={cfg.cond_dropout}")
        self.logger.info(
            f"  Sample rule:  min_tokens={cfg.min_tokens}  pad_logit_bias={cfg.pad_logit_bias}"
        )
        self.logger.info(f"  Train:        {len(self.train_ds)} samples, {len(self.train_dl)} steps/epoch")
        self.logger.info(f"  AMP:          {cfg.use_amp}")
        self.logger.info("=" * 60)

    def _build_conditions(self, batch, training: bool = True) -> Dict[str, torch.Tensor]:
        del training  # 条件 dropout 统一在 denoiser 内部完成
        cfg = self.cfg

        obj_pc = batch["obj_pc"].to(self.device)
        obj_vn = batch["obj_vn"].to(self.device)
        obj_pc_norm, _ = normalize_object_points(obj_pc)
        obj_input = torch.cat([obj_pc_norm, obj_vn], dim=-1)
        obj_global_feat, obj_point_feat, _ = self.obj_enc(obj_input)

        cond = {
            "obj_feat": obj_global_feat,
            "obj_point_feat": obj_point_feat,
        }

        if cfg.use_text and self.clip_enc is not None:
            texts = batch["text"]
            with torch.no_grad():
                cond["text_feat"] = self.clip_enc.encode(texts)

        return cond

    def train(self):
        cfg = self.cfg
        self.logger.info(f"\nStart training: {cfg.epochs} epochs")

        for epoch in range(self.epoch, cfg.epochs):
            self.epoch = epoch
            self._train_one_epoch()

            if self.val_dl is not None:
                val_loss = self._validate()
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint("best.pt")

        self._save_checkpoint("final.pt")
        self.logger.info("Training complete.")

    def _train_one_epoch(self):
        cfg = self.cfg
        self.denoiser.train()
        self.obj_enc.train()

        epoch_loss = 0.0
        n_steps = 0
        t0 = time.time()

        for batch in self.train_dl:
            log = self._train_step(batch)
            n_steps += 1
            epoch_loss += log.get("loss", 0.0)

            if self.global_step % cfg.log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                parts = [
                    f"E{self.epoch:03d} S{self.global_step:06d}",
                    f"loss={log.get('loss', 0):.4f}",
                    f"mask={log.get('loss_mask_only', 0):.4f}",
                    f"ratio={log.get('mask_ratio', 0):.3f}",
                    f"acc={log.get('masked_acc', 0):.3f}",
                    f"sem_acc={log.get('semantic_acc_masked', 0):.3f}",
                    f"pad_acc={log.get('pad_acc_masked', 0):.3f}",
                    f"lr={lr:.2e}",
                ]
                self.logger.info("  ".join(parts))

                if self.tb:
                    for k, v in log.items():
                        if isinstance(v, (int, float)):
                            self.tb.add_scalar(f"train/{k}", v, self.global_step)
                    self.tb.add_scalar("train/lr", lr, self.global_step)

            if self.val_dl is not None and self.global_step > 0 and self.global_step % cfg.val_every == 0:
                val_loss = self._validate()
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint("best.pt")

            if self.global_step > 0 and self.global_step % cfg.save_every == 0:
                self._save_checkpoint(f"step_{self.global_step:07d}.pt")
                self._save_checkpoint("latest.pt")
                self._cleanup_checkpoints()

        dt = time.time() - t0
        avg = epoch_loss / max(n_steps, 1)
        self.logger.info(f"[Epoch {self.epoch:03d}] avg_loss={avg:.4f}  time={dt:.0f}s ({dt/max(n_steps,1):.3f}s/step)")

    def _train_step(self, batch) -> Dict[str, float]:
        x_0 = batch["token_seq"].to(self.device)
        token_mask = batch["token_mask"].to(self.device).bool()
        cond = self._build_conditions(batch, training=True)

        self.optimizer.zero_grad()
        with autocast(enabled=self.cfg.use_amp):
            loss_dict = self.diffusion.training_loss(
                self.denoiser,
                x_0,
                token_mask=token_mask,
                cond=cond,
            )
            loss = loss_dict["loss"]

        self.scaler.scale(loss).backward()
        if self.cfg.max_grad_norm > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                list(self.denoiser.parameters()) + list(self.obj_enc.parameters()),
                self.cfg.max_grad_norm,
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.global_step += 1

        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

    @torch.no_grad()
    def _validate(self) -> float:
        if self.val_dl is None:
            return float("inf")
        self.denoiser.eval()
        self.obj_enc.eval()

        total_loss = 0.0
        n = 0

        for batch in self.val_dl:
            x_0 = batch["token_seq"].to(self.device)
            token_mask = batch["token_mask"].to(self.device).bool()
            cond = self._build_conditions(batch, training=False)

            with autocast(enabled=self.cfg.use_amp):
                loss_dict = self.diffusion.training_loss(
                    self.denoiser,
                    x_0,
                    token_mask=token_mask,
                    cond=cond,
                )

            B = x_0.size(0)
            total_loss += loss_dict["loss"].item() * B
            n += B

        avg_loss = total_loss / max(n, 1)
        self.logger.info(f"[Val S{self.global_step:06d}] loss={avg_loss:.4f}")

        if self.tb:
            self.tb.add_scalar("val/loss", avg_loss, self.global_step)

        self._sample_and_log()

        self.denoiser.train()
        self.obj_enc.train()
        return avg_loss

    @torch.no_grad()
    def _sample_and_log(self):
        cfg = self.cfg
        if self.val_dl is None:
            return
        n = min(cfg.num_val_samples, cfg.batch_size)

        batch = next(iter(self.val_dl))
        small = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                small[k] = v[:n]
            elif isinstance(v, list):
                small[k] = v[:n]
            else:
                small[k] = v
        cond = self._build_conditions(small, training=False)

        seq_len = self.train_ds.encoder.max_token_length
        result = self.diffusion.sample(
            self.denoiser,
            cond=cond,
            batch_size=n,
            seq_length=seq_len,
            temperature=0.9,
            guidance_scale=2.0,
            min_tokens=cfg.min_tokens,
        )

        samples = result["samples"].cpu()
        gt = small["token_seq"][:n]
        mapper = self.train_ds.mapper

        self.logger.info(f"--- Sample comparison (first {min(4, n)}) ---")
        for i in range(min(4, n)):
            gt_toks = [int(t) for t in gt[i]]
            pred_toks = [int(t) for t in samples[i]]
            gt_sem = sum(1 for t in gt_toks if mapper.is_semantic_token(t))
            gt_pad = sum(1 for t in gt_toks if t == cfg.pad_token_id)
            pred_sem = sum(1 for t in pred_toks if mapper.is_semantic_token(t))
            pred_pad = sum(1 for t in pred_toks if t == cfg.pad_token_id)
            self.logger.info(f"  [{i}] GT:   {gt_toks}  (sem={gt_sem}, pad={gt_pad})")
            self.logger.info(
                f"  [{i}] Pred: {pred_toks}  (sem={pred_sem}, pad={pred_pad})"
            )

    @staticmethod
    def _slice_batch(batch, n_keep: int):
        sliced = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                sliced[k] = v[:n_keep]
            elif isinstance(v, list):
                sliced[k] = v[:n_keep]
            else:
                sliced[k] = v
        return sliced

    @torch.no_grad()
    def analyze_stage1_token_noise(
        self,
        split: str = "test",
        cache_dir: str = "cache/contact_tokens/test",
        out_path: str | None = None,
        batch_size: int | None = None,
        temperature: float = 0.9,
        guidance_scale: float = 2.0,
        max_samples: int = 0,
        adaptive_pivot: int = 6,
    ) -> dict:
        """Estimate Stage-1 semantic token noise for Stage-2 augmentation calibration."""
        ds, dl = self._build_split_dataloader(cache_dir, split, batch_size=batch_size)
        n_semantic = ds.mapper.num_semantic_tokens
        seq_len = ds.encoder.max_token_length

        was_training_denoiser = self.denoiser.training
        was_training_obj = self.obj_enc.training
        self.denoiser.eval()
        self.obj_enc.eval()

        totals = defaultdict(float)
        per_sample = []
        seen = 0

        for batch in dl:
            if max_samples > 0 and seen >= max_samples:
                break
            if max_samples > 0:
                remaining = max_samples - seen
                if remaining < len(batch["sample_id"]):
                    batch = self._slice_batch(batch, remaining)

            gt_tokens = batch["token_seq"].to(self.device)
            gt_mask = batch["token_mask"].to(self.device).bool()
            cond = self._build_conditions(batch, training=False)

            result = self.diffusion.sample(
                self.denoiser,
                cond=cond,
                batch_size=gt_tokens.shape[0],
                seq_length=seq_len,
                temperature=temperature,
                guidance_scale=guidance_scale,
                min_tokens=self.cfg.min_tokens,
            )
            pred_tokens = result["samples"].to(self.device)

            batch_stats = compute_stage1_token_noise_stats(
                pred_tokens=pred_tokens,
                gt_tokens=gt_tokens,
                gt_token_mask=gt_mask,
                n_semantic=n_semantic,
            )

            gt_count = batch_stats["gt_count"].float()
            adaptive_scale = (gt_count / float(adaptive_pivot)).clamp(max=1.0)
            totals["num_samples"] += gt_tokens.shape[0]
            totals["total_gt_tokens"] += gt_count.sum().item()
            totals["total_pred_tokens"] += batch_stats["pred_count"].sum().item()
            totals["total_missing_tokens"] += batch_stats["missing_count"].sum().item()
            totals["total_extra_tokens"] += batch_stats["extra_count"].sum().item()
            totals["total_drop_tokens"] += batch_stats["drop_count"].sum().item()
            totals["total_sub_tokens"] += batch_stats["sub_count"].sum().item()
            totals["total_insert_tokens"] += batch_stats["insert_count"].sum().item()
            totals["adaptive_drop_denom"] += (gt_count * adaptive_scale).sum().item()

            for i, sample_id in enumerate(batch["sample_id"]):
                per_sample.append({
                    "sample_id": sample_id,
                    "cache_file": batch["cache_file"][i],
                    "gt_tokens": int(batch_stats["gt_count"][i].item()),
                    "pred_tokens": int(batch_stats["pred_count"][i].item()),
                    "drop_tokens": int(batch_stats["drop_count"][i].item()),
                    "sub_tokens": int(batch_stats["sub_count"][i].item()),
                    "insert_tokens": int(batch_stats["insert_count"][i].item()),
                })
            seen += gt_tokens.shape[0]

        total_gt = max(totals["total_gt_tokens"], 1.0)
        total_remaining_after_drop = max(
            totals["total_gt_tokens"] - totals["total_drop_tokens"], 1.0
        )
        adaptive_drop_denom = max(totals["adaptive_drop_denom"], 1.0)

        calibrated_drop_prob = min(totals["total_drop_tokens"] / adaptive_drop_denom, 1.0)
        calibrated_sub_prob = min(totals["total_sub_tokens"] / total_remaining_after_drop, 1.0)

        summary = {
            "split": split,
            "cache_dir": cache_dir,
            "checkpoint": self.cfg.resume,
            "num_samples": int(totals["num_samples"]),
            "temperature": float(temperature),
            "guidance_scale": float(guidance_scale),
            "adaptive_pivot": int(adaptive_pivot),
            "mean_gt_tokens_per_sample": float(totals["total_gt_tokens"] / max(totals["num_samples"], 1.0)),
            "mean_pred_tokens_per_sample": float(totals["total_pred_tokens"] / max(totals["num_samples"], 1.0)),
            "mean_missing_tokens_per_sample": float(totals["total_missing_tokens"] / max(totals["num_samples"], 1.0)),
            "mean_extra_tokens_per_sample": float(totals["total_extra_tokens"] / max(totals["num_samples"], 1.0)),
            "mean_drop_tokens_per_sample": float(totals["total_drop_tokens"] / max(totals["num_samples"], 1.0)),
            "mean_sub_tokens_per_sample": float(totals["total_sub_tokens"] / max(totals["num_samples"], 1.0)),
            "mean_insert_tokens_per_sample": float(totals["total_insert_tokens"] / max(totals["num_samples"], 1.0)),
            "drop_rate_per_gt_token": float(totals["total_drop_tokens"] / total_gt),
            "sub_rate_per_surviving_gt_token": float(totals["total_sub_tokens"] / total_remaining_after_drop),
            "calibrated_token_drop_prob": float(calibrated_drop_prob),
            "calibrated_token_sub_prob": float(calibrated_sub_prob),
            "notes": (
                "missing = drop + sub, extra = insert + sub, sub = min(missing, extra). "
                "Drop probability is calibrated against the adaptive dropout denominator "
                "sum(k * min(k/pivot, 1)); substitution probability is calibrated on "
                "the surviving GT token mass after drops."
            ),
            "per_sample": per_sample,
        }

        if out_path:
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            self.logger.info(f"[Stage1 Noise] Saved stats to {out_path}")

        self.logger.info(
            "[Stage1 Noise] split=%s  samples=%d  drop/sample=%.3f  sub/sample=%.3f  "
            "drop_prob=%.4f  sub_prob=%.4f",
            split,
            summary["num_samples"],
            summary["mean_drop_tokens_per_sample"],
            summary["mean_sub_tokens_per_sample"],
            summary["calibrated_token_drop_prob"],
            summary["calibrated_token_sub_prob"],
        )

        if was_training_denoiser:
            self.denoiser.train()
        if was_training_obj:
            self.obj_enc.train()
        return summary

    def _save_checkpoint(self, filename):
        path = os.path.join(self.cfg.save_dir, filename)
        torch.save({
            "obj_enc": self.obj_enc.state_dict(),
            "denoiser": self.denoiser.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "config": vars(self.cfg),
        }, path)
        self.logger.info(f"  Saved: {path}")

    def _load_checkpoint(self, path):
        self.logger.info(f"Resuming from {path}")
        state = torch.load(path, map_location=self.device)
        if "obj_enc" in state:
            self.obj_enc.load_state_dict(state["obj_enc"])
        else:
            self.logger.warning("  obj_enc not found in checkpoint, using current initialization")
        self.denoiser.load_state_dict(state["denoiser"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        if "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])
        self.global_step = state["global_step"]
        self.epoch = state["epoch"]
        self.best_val_loss = state.get("best_val_loss", float("inf"))
        self.logger.info(f"  Resumed: epoch={self.epoch}, step={self.global_step}")

    def _cleanup_checkpoints(self):
        files = sorted([
            f for f in os.listdir(self.cfg.save_dir)
            if f.startswith("step_") and f.endswith(".pt")
        ])
        while len(files) > self.cfg.save_top_k:
            os.remove(os.path.join(self.cfg.save_dir, files.pop(0)))


def parse_args():
    p = argparse.ArgumentParser(description="Train Token Denoiser (v16 simplified tokens)")

    p.add_argument("--cache_dir", type=str, default="cache/contact_tokens/train")
    p.add_argument("--val_cache_dir", type=str, default="cache/contact_tokens/val")
    p.add_argument("--config_path", type=str, default="configs/token_config.yaml")
    p.add_argument("--n_pc_points", type=int, default=2048, help="物体点云采样点数")

    p.add_argument("--denoiser_config", type=str, default="base", choices=["tiny", "small", "base"])
    p.add_argument("--obj_global_dim", type=int, default=256)
    p.add_argument("--obj_point_dim", type=int, default=256)
    p.add_argument("--clip_model", type=str, default="ViT-B/32")
    p.add_argument("--text_feat_dim", type=int, default=512)
    p.add_argument("--use_text", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cond_dropout", type=float, default=0.2)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--log_dir", type=str, default="logs/diffusion")
    p.add_argument("--save_dir", type=str, default="checkpoints/diffusion")
    p.add_argument("--seed", type=int, default=7725)

    p.add_argument("--vocab_size", type=int, default=26)
    p.add_argument("--pad_token_id", type=int, default=24)
    p.add_argument("--mask_token_id", type=int, default=25)
    p.add_argument("--min_tokens", type=int, default=1)
    p.add_argument("--pad_logit_bias", type=float, default=0.0)

    # Stage-1 token noise analysis for Stage-2 augmentation calibration
    p.add_argument("--analyze_token_noise", action="store_true",
                   help="Run Stage-1 token noise analysis instead of training")
    p.add_argument("--analysis_ckpt", type=str, default="",
                   help="Checkpoint to analyze; if set, overrides --resume for analysis mode")
    p.add_argument("--analysis_split", type=str, default="test",
                   choices=["train", "val", "test"],
                   help="Dataset split used for Stage-1 token noise analysis")
    p.add_argument("--analysis_cache_dir", type=str, default="cache/contact_tokens/test",
                   help="Token cache dir used for Stage-1 token noise analysis")
    p.add_argument("--analysis_out", type=str, default="",
                   help="Optional JSON path to save Stage-1 token noise statistics")
    p.add_argument("--analysis_batch_size", type=int, default=0,
                   help="Batch size for analysis; 0 means reuse --batch_size")
    p.add_argument("--analysis_temperature", type=float, default=0.9)
    p.add_argument("--analysis_guidance_scale", type=float, default=2.0)
    p.add_argument("--analysis_max_samples", type=int, default=0,
                   help="Limit analysis to the first N samples; 0 means all")
    p.add_argument("--analysis_adaptive_pivot", type=int, default=6,
                   help="Adaptive dropout pivot used to calibrate token_drop_prob")

    return p.parse_args()


def _override_args_from_checkpoint(args, ckpt_path: str):
    if not ckpt_path or not os.path.isfile(ckpt_path):
        return

    state = torch.load(ckpt_path, map_location="cpu")
    saved_cfg = state.get("config", {})
    if not isinstance(saved_cfg, dict):
        return

    keys = [
        "config_path",
        "n_pc_points",
        "denoiser_config",
        "obj_global_dim",
        "obj_point_dim",
        "clip_model",
        "text_feat_dim",
        "use_text",
        "cond_dropout",
        "vocab_size",
        "pad_token_id",
        "mask_token_id",
        "min_tokens",
        "pad_logit_bias",
    ]
    for key in keys:
        if key in saved_cfg:
            setattr(args, key, saved_cfg[key])


def main():
    args = parse_args()
    analysis_ckpt = args.analysis_ckpt or args.resume
    if args.analyze_token_noise and not analysis_ckpt:
        raise ValueError("analysis mode requires --analysis_ckpt (or --resume) to load a trained Stage-1 model")
    if args.analyze_token_noise and analysis_ckpt:
        _override_args_from_checkpoint(args, analysis_ckpt)
        args.resume = analysis_ckpt

    cfg = TrainConfig(
        cache_dir=args.cache_dir,
        val_cache_dir=args.val_cache_dir,
        config_path=args.config_path,
        n_pc_points=args.n_pc_points,
        denoiser_config=args.denoiser_config,
        obj_global_dim=args.obj_global_dim,
        obj_point_dim=args.obj_point_dim,
        clip_model=args.clip_model,
        text_feat_dim=args.text_feat_dim,
        use_text=args.use_text,
        cond_dropout=args.cond_dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume=args.resume,
        log_dir=args.log_dir,
        save_dir=args.save_dir,
        seed=args.seed,
        vocab_size=args.vocab_size,
        pad_token_id=args.pad_token_id,
        mask_token_id=args.mask_token_id,
        min_tokens=args.min_tokens,
        pad_logit_bias=args.pad_logit_bias,
    )

    trainer = DiffusionTrainer(cfg)
    if args.analyze_token_noise:
        out_path = args.analysis_out
        if not out_path:
            out_path = os.path.join(
                cfg.save_dir,
                f"stage1_token_noise_{args.analysis_split}.json",
            )
        trainer.analyze_stage1_token_noise(
            split=args.analysis_split,
            cache_dir=args.analysis_cache_dir,
            out_path=out_path,
            batch_size=args.analysis_batch_size or cfg.batch_size,
            temperature=args.analysis_temperature,
            guidance_scale=args.analysis_guidance_scale,
            max_samples=args.analysis_max_samples,
            adaptive_pivot=args.analysis_adaptive_pivot,
        )
        return

    trainer.train()


if __name__ == "__main__":
    main()
