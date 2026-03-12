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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast

from data.token_encoder import TokenEncoder
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
    balanced_sampling: bool  = False

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
        }


def _collate_fn(batch):
    return {
        "token_seq": torch.stack([b["token_seq"] for b in batch]),
        "token_mask": torch.stack([b["token_mask"] for b in batch]),
        "obj_pc": torch.stack([b["obj_pc"] for b in batch]),
        "obj_vn": torch.stack([b["obj_vn"] for b in batch]),
        "text": [b["text"] for b in batch],
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
        obj_input = torch.cat([obj_pc, obj_vn], dim=-1)
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
    p.add_argument("--cond_dropout", type=float, default=0.2)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--seed", type=int, default=7725)
    p.add_argument(
        "--balanced_sampling",
        action="store_true",
        help="使用 1/sqrt(category_count) 的弱类别均衡采样",
    )

    p.add_argument("--vocab_size", type=int, default=26)
    p.add_argument("--pad_token_id", type=int, default=24)
    p.add_argument("--mask_token_id", type=int, default=25)
    p.add_argument("--min_tokens", type=int, default=1)
    p.add_argument("--pad_logit_bias", type=float, default=0.0)

    return p.parse_args()


def main():
    args = parse_args()

    cfg = TrainConfig(
        cache_dir=args.cache_dir,
        val_cache_dir=args.val_cache_dir,
        config_path=args.config_path,
        n_pc_points=args.n_pc_points,
        denoiser_config=args.denoiser_config,
        obj_global_dim=args.obj_global_dim,
        obj_point_dim=args.obj_point_dim,
        cond_dropout=args.cond_dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume=args.resume,
        seed=args.seed,
        balanced_sampling=args.balanced_sampling,
        vocab_size=args.vocab_size,
        pad_token_id=args.pad_token_id,
        mask_token_id=args.mask_token_id,
        min_tokens=args.min_tokens,
        pad_logit_bias=args.pad_logit_bias,
    )

    DiffusionTrainer(cfg).train()


if __name__ == "__main__":
    main()
