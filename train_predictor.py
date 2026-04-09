#!/usr/bin/env python3
"""
train_predictor.py
==================
Standalone trainer for BPSSlotPredictor.

This script trains the geometry-only BPS slot classifier using the same cached
decoder dataset, but keeps the predictor lifecycle fully separate from decoder
training. Checkpoints are still saved with the `bps_slot_predictor.` prefix for
backward compatibility with earlier integrated experiments.
"""

import os
import time
import random
import argparse
import logging
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.condition_encoder import BPSSlotPredictor
from train_decoder import DecoderDataset, _collate


PREDICTOR_PREFIX = "bps_slot_predictor."


@dataclass
class PredictorTrainConfig:
    # ---- data ----
    cache_dir: str = "cache/contact_tokens/train"
    val_cache_dir: str = "cache/contact_tokens/val"
    config_path: str = "configs/token_config.yaml"
    n_pc_points: int = 10000
    oi_category: str = "all"
    oi_intent: str = "all"
    bps_path: str = "configs/bps.npz"
    bps_slot_cache_dir: str = "cache/bps_slot"

    # ---- model ----
    hidden_dim: int = 256
    num_neighbors: int = 16
    num_layers: int = 2
    dropout: float = 0.0
    chunk_size: int = 256
    class_weight_power: float = 0.5

    # ---- training ----
    epochs: int = 100
    batch_size: int = 128
    lr: float = 5e-4
    weight_decay: float = 5e-4
    max_grad_norm: float = 0.0
    early_stop_patience: int = 10

    # ---- logging / save ----
    log_dir: str = "logs/predictor"
    save_dir: str = "checkpoints/predictor"
    log_every: int = 50

    # ---- resume ----
    resume: str = ""

    # ---- hardware ----
    num_workers: int = 4
    seed: int = 42
    device: str = "cuda"


def _prefix_state_dict(state_dict: dict) -> dict:
    return {f"{PREDICTOR_PREFIX}{key}": value for key, value in state_dict.items()}


def _strip_predictor_prefix(state_dict: dict) -> dict:
    if any(key.startswith(PREDICTOR_PREFIX) for key in state_dict):
        return {
            key[len(PREDICTOR_PREFIX):]: value
            for key, value in state_dict.items()
            if key.startswith(PREDICTOR_PREFIX)
        }
    return state_dict


class PredictorTrainer:
    def __init__(self, cfg: PredictorTrainConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")
        self.no_improve = 0

        self._set_seed(cfg.seed)
        os.makedirs(cfg.save_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)

        self._setup_logging()
        self._build_data()
        self._build_model()
        self._build_class_weights()
        self._build_optimizer()

        if cfg.resume:
            self._load_checkpoint(cfg.resume)

        self._log_info()

    def _set_seed(self, seed: int):
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

    def _build_data(self):
        cfg = self.cfg
        self.train_ds = DecoderDataset(
            cfg.cache_dir,
            cfg.config_path,
            "train",
            n_pc_points=cfg.n_pc_points,
            category=cfg.oi_category,
            intent=cfg.oi_intent,
            bps_path=cfg.bps_path,
            bps_slot_cache_dir=cfg.bps_slot_cache_dir,
            require_bps_nn_points=True,
        )
        self.train_dl = DataLoader(
            self.train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=_collate,
        )

        self.val_dl = None
        if cfg.val_cache_dir and os.path.isdir(cfg.val_cache_dir):
            self.val_ds = DecoderDataset(
                cfg.val_cache_dir,
                cfg.config_path,
                "val",
                n_pc_points=cfg.n_pc_points,
                category=cfg.oi_category,
                intent=cfg.oi_intent,
                bps_path=cfg.bps_path,
                bps_slot_cache_dir=cfg.bps_slot_cache_dir,
                require_bps_nn_points=True,
            )
            self.val_dl = DataLoader(
                self.val_ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=_collate,
            )

    def _build_model(self):
        cfg = self.cfg
        bps_data = np.load(cfg.bps_path)
        basis = bps_data["basis"].astype(np.float32)
        if basis.ndim == 3:
            basis = basis.squeeze(0)
        self.model = BPSSlotPredictor(
            hidden_dim=cfg.hidden_dim,
            basis_points=basis,
            num_neighbors=cfg.num_neighbors,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            chunk_size=cfg.chunk_size,
        ).to(self.device)
        self.class_names = [f"slot{i}" for i in range(self.model.num_slots)] + ["unassigned"]

    def _build_class_weights(self):
        cfg = self.cfg
        self.class_counts = torch.zeros(self.model.n_classes, dtype=torch.float64)
        self.class_weights = None

        if cfg.class_weight_power <= 0:
            return

        obj_freq = defaultdict(int)
        for entry in self.train_ds.samples:
            obj_id = self.train_ds._idx_to_obj_id.get(entry["idx"], "")
            if obj_id:
                obj_freq[obj_id] += 1

        for obj_id, repeat in obj_freq.items():
            bps_slot_data = self.train_ds._load_bps_slot(obj_id)
            if bps_slot_data is None:
                continue

            labels = torch.from_numpy(bps_slot_data["bps_slot_labels"]).long()
            labels[labels < 0] = self.model.num_slots
            bincount = torch.bincount(labels, minlength=self.model.n_classes).to(torch.float64)
            self.class_counts += bincount * repeat

        present = self.class_counts > 0
        if not present.any():
            return

        weights = torch.ones(self.model.n_classes, dtype=torch.float32)
        weights[present] = self.class_counts[present].to(torch.float32).pow(-cfg.class_weight_power)
        weights[present] = weights[present] / weights[present].mean().clamp(min=1e-12)
        self.class_weights = weights.to(self.device)

    def _build_optimizer(self):
        cfg = self.cfg
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
        )

    def _log_info(self):
        n_model = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info("=" * 60)
        self.logger.info("BPSSlotPredictor — Standalone Training")
        self.logger.info(f"  Params:        {n_model:,} ({n_model/1e6:.2f}M)")
        self.logger.info(
            f"  Backbone:      light point transformer  "
            f"layers={self.cfg.num_layers}  knn={self.cfg.num_neighbors}"
        )
        self.logger.info(
            f"  Hidden dim:    {self.cfg.hidden_dim}  "
            f"dropout={self.cfg.dropout:.2f}  chunk={self.cfg.chunk_size}"
        )
        if self.class_weights is None:
            self.logger.info("  Class weight:  disabled")
        else:
            total = self.class_counts.sum().item()
            parts = []
            weights = self.class_weights.detach().cpu()
            for idx, name in enumerate(self.class_names):
                freq = self.class_counts[idx].item() / max(total, 1.0)
                parts.append(f"{name}={freq:.3f}/w={weights[idx].item():.2f}")
            self.logger.info(
                f"  Class weight:  inv_freq^{self.cfg.class_weight_power:.2f}"
            )
            self.logger.info("  Train labels:  " + "  ".join(parts))
        self.logger.info(f"  Train:         {len(self.train_ds)} samples, {len(self.train_dl)} steps/epoch")
        if self.val_dl is not None:
            self.logger.info(f"  Val:           {len(self.val_ds)} samples, {len(self.val_dl)} steps")
        else:
            self.logger.info("  Val:           disabled")
        self.logger.info("=" * 60)

    def _compute_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float | np.ndarray]]:
        loss = self.model.compute_loss(
            logits,
            labels,
            class_weights=self.class_weights,
        )
        pred_cls = logits.argmax(dim=-1)

        target = labels.clone().long()
        target[target < 0] = self.model.num_slots

        class_correct = np.zeros(self.model.n_classes, dtype=np.float64)
        class_total = np.zeros(self.model.n_classes, dtype=np.float64)
        semantic_accs = []

        for cls_idx in range(self.model.n_classes):
            cls_mask = target == cls_idx
            cls_total = int(cls_mask.sum().item())
            class_total[cls_idx] = cls_total
            if cls_total == 0:
                continue

            cls_correct = int((pred_cls[cls_mask] == cls_idx).sum().item())
            class_correct[cls_idx] = cls_correct
            if cls_idx < self.model.num_slots:
                semantic_accs.append(cls_correct / cls_total)

        semantic_correct = float(class_correct[:self.model.num_slots].sum())
        semantic_total = float(class_total[:self.model.num_slots].sum())
        acc = semantic_correct / max(semantic_total, 1.0)
        macro_acc = float(np.mean(semantic_accs)) if semantic_accs else 0.0
        return loss, {
            "bps_slot_loss": loss.item(),
            "bps_slot_acc": acc,
            "bps_slot_macro_acc": macro_acc,
            "valid_correct": semantic_correct,
            "valid_total": semantic_total,
            "class_correct": class_correct,
            "class_total": class_total,
        }

    def _init_metric_state(self) -> dict[str, float | np.ndarray]:
        return {
            "loss_sum": 0.0,
            "n_samples": 0.0,
            "valid_correct": 0.0,
            "valid_total": 0.0,
            "class_correct": np.zeros(self.model.n_classes, dtype=np.float64),
            "class_total": np.zeros(self.model.n_classes, dtype=np.float64),
        }

    def _update_metric_state(
        self,
        state: dict[str, float | np.ndarray],
        stats: dict[str, float | np.ndarray],
        batch_size: int,
    ) -> None:
        state["loss_sum"] += float(stats["bps_slot_loss"]) * batch_size
        state["n_samples"] += batch_size
        state["valid_correct"] += float(stats["valid_correct"])
        state["valid_total"] += float(stats["valid_total"])
        state["class_correct"] += stats["class_correct"]
        state["class_total"] += stats["class_total"]

    def _summarize_metric_state(
        self,
        state: dict[str, float | np.ndarray],
        prefix: str = "",
    ) -> dict[str, float]:
        loss = float(state["loss_sum"]) / max(float(state["n_samples"]), 1.0)
        acc = float(state["valid_correct"]) / max(float(state["valid_total"]), 1.0)

        class_correct = np.asarray(state["class_correct"], dtype=np.float64)
        class_total = np.asarray(state["class_total"], dtype=np.float64)
        semantic_mask = class_total[:self.model.num_slots] > 0
        if semantic_mask.any():
            macro_acc = float(
                np.mean(
                    class_correct[:self.model.num_slots][semantic_mask]
                    / class_total[:self.model.num_slots][semantic_mask]
                )
            )
        else:
            macro_acc = 0.0

        metrics = {
            f"{prefix}bps_slot_loss": loss,
            f"{prefix}bps_slot_acc": acc,
            f"{prefix}bps_slot_macro_acc": macro_acc,
        }

        for idx, name in enumerate(self.class_names):
            if class_total[idx] > 0:
                metrics[f"{prefix}bps_slot_acc_{name}"] = float(class_correct[idx] / class_total[idx])

        return metrics

    def _format_per_class(self, metrics: dict[str, float], prefix: str = "") -> str:
        parts = []
        for name in self.class_names:
            key = f"{prefix}bps_slot_acc_{name}"
            if key in metrics:
                parts.append(f"{name}={metrics[key]:.3f}")
            else:
                parts.append(f"{name}=-")
        return "  ".join(parts)

    def _train_step(self, batch) -> dict[str, float | np.ndarray]:
        cfg = self.cfg
        bps_object = batch["bps_object"].to(self.device)
        bps_nn_points = batch["bps_nn_points"].to(self.device)
        bps_slot_labels = batch["bps_slot_labels"].to(self.device)

        self.optimizer.zero_grad()
        logits = self.model(bps_object, bps_nn_points=bps_nn_points)
        loss, stats = self._compute_metrics(logits, bps_slot_labels)
        loss.backward()
        if cfg.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
        self.optimizer.step()

        return stats

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        if self.val_dl is None:
            return {}

        self.model.eval()
        totals = self._init_metric_state()

        for batch in self.val_dl:
            bps_object = batch["bps_object"].to(self.device)
            bps_nn_points = batch["bps_nn_points"].to(self.device)
            bps_slot_labels = batch["bps_slot_labels"].to(self.device)
            batch_size = bps_object.shape[0]

            logits = self.model(bps_object, bps_nn_points=bps_nn_points)
            _, stats = self._compute_metrics(logits, bps_slot_labels)
            self._update_metric_state(totals, stats, batch_size)

        self.model.train()
        return self._summarize_metric_state(totals, prefix="val/")

    def _save_checkpoint(self, tag: str):
        path = os.path.join(self.cfg.save_dir, f"{tag}.pt")
        torch.save(
            {
                "stage": "predictor",
                "model": _prefix_state_dict(self.model.state_dict()),
                "epoch": self.epoch,
                "global_step": self.global_step,
                "best_loss": self.best_loss,
                "no_improve": self.no_improve,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "config": self.cfg.__dict__,
            },
            path,
        )
        self.logger.info(f"  Saved [predictor]: {path}")

    def _load_checkpoint(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        self.logger.info(f"Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device)
        raw_state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        predictor_state = _strip_predictor_prefix(raw_state)
        missing, unexpected = self.model.load_state_dict(predictor_state, strict=False)
        if missing or unexpected:
            self.logger.warning(
                f"  Predictor checkpoint compat: missing={len(missing)}, unexpected={len(unexpected)}"
            )
            for key in missing[:5]:
                self.logger.warning(f"    missing: {key}")
            for key in unexpected[:5]:
                self.logger.warning(f"    unexpected: {key}")

        if "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except (KeyError, ValueError) as exc:
                self.logger.warning(f"  Optimizer state mismatch, skipping: {exc}")
        if self.scheduler and "scheduler" in ckpt and ckpt["scheduler"]:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler"])
            except (KeyError, ValueError) as exc:
                self.logger.warning(f"  Scheduler state mismatch, skipping: {exc}")

        self.epoch = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)
        self.best_loss = ckpt.get("best_loss", float("inf"))
        self.no_improve = ckpt.get("no_improve", 0)
        self.logger.info(
            f"  Resumed [predictor]: epoch={self.epoch}, step={self.global_step}"
        )

    def train(self):
        cfg = self.cfg
        self.model.train()
        self.logger.info(f"\nStart training: {cfg.epochs} epochs")

        for ep in range(self.epoch, cfg.epochs):
            self.epoch = ep
            self.logger.info(f"Epoch {ep}/{cfg.epochs}")

            epoch_totals = self._init_metric_state()
            t0 = time.time()

            for batch in self.train_dl:
                stats = self._train_step(batch)
                self.global_step += 1
                self._update_metric_state(epoch_totals, stats, batch["bps_object"].shape[0])

                if self.global_step % cfg.log_every == 0:
                    step_log = {
                        "bps_slot_loss": float(stats["bps_slot_loss"]),
                        "bps_slot_acc": float(stats["bps_slot_acc"]),
                        "bps_slot_macro_acc": float(stats["bps_slot_macro_acc"]),
                    }
                    self.logger.info(
                        f"  S{self.global_step:06d}  "
                        f"bps_slot_loss={step_log['bps_slot_loss']:.4f}  "
                        f"bps_slot_acc={step_log['bps_slot_acc']:.4f}  "
                        f"bps_slot_macro_acc={step_log['bps_slot_macro_acc']:.4f}"
                    )
                    if self.tb:
                        for key, value in step_log.items():
                            self.tb.add_scalar(f"train/{key}", value, self.global_step)

            dt = time.time() - t0
            train = self._summarize_metric_state(epoch_totals)
            train_loss = train["bps_slot_loss"]
            self.logger.info(
                f"  Epoch {ep} done  {dt:.0f}s  "
                f"bps_slot_loss={train['bps_slot_loss']:.4f}  "
                f"bps_slot_acc={train['bps_slot_acc']:.4f}  "
                f"bps_slot_macro_acc={train['bps_slot_macro_acc']:.4f}"
            )
            self.logger.info("    [TRAIN] " + self._format_per_class(train))
            if self.tb:
                for key, value in train.items():
                    self.tb.add_scalar(f"train_epoch/{key}", value, self.global_step)

            val = self._validate()
            monitor_loss = train_loss
            if val:
                monitor_loss = val["val/bps_slot_loss"]
                self.logger.info(
                    f"  [VAL] bps_slot_loss={val['val/bps_slot_loss']:.4f}  "
                    f"bps_slot_acc={val['val/bps_slot_acc']:.4f}  "
                    f"bps_slot_macro_acc={val['val/bps_slot_macro_acc']:.4f}"
                )
                self.logger.info("        " + self._format_per_class(val, prefix="val/"))
                if self.tb:
                    for key, value in val.items():
                        self.tb.add_scalar(key, value, self.global_step)

            if self.scheduler is not None:
                prev_lr = self.optimizer.param_groups[0]["lr"]
                self.scheduler.step(monitor_loss)
                cur_lr = self.optimizer.param_groups[0]["lr"]
                if cur_lr != prev_lr:
                    self.logger.info(f"  [LR] {prev_lr:.2e} -> {cur_lr:.2e}")

            self._save_checkpoint("latest")
            if monitor_loss < self.best_loss:
                prev_best = self.best_loss
                self.best_loss = monitor_loss
                self.no_improve = 0
                self._save_checkpoint("best_predictor")
                if prev_best == float("inf"):
                    self.logger.info(f"  * New best_predictor: loss={monitor_loss:.4f}")
                else:
                    self.logger.info(
                        f"  * New best_predictor: loss={monitor_loss:.4f} (prev={prev_best:.4f})"
                    )
            else:
                self.no_improve += 1
                remaining = max(cfg.early_stop_patience - self.no_improve, 0)
                self.logger.info(
                    f"  [EARLY STOP] no improvement for {self.no_improve}/"
                    f"{cfg.early_stop_patience} (remaining={remaining})"
                )
                if self.no_improve >= cfg.early_stop_patience:
                    self.logger.info("Early stopping triggered for BPSSlotPredictor.")
                    break

        self._save_checkpoint("final")
        self.logger.info("Training complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train BPSSlotPredictor standalone")
    parser.add_argument("--cache_dir", type=str, default="cache/contact_tokens/train")
    parser.add_argument("--val_cache_dir", type=str, default="cache/contact_tokens/val")
    parser.add_argument("--config_path", type=str, default="configs/token_config.yaml")
    parser.add_argument("--n_pc_points", type=int, default=10000)
    parser.add_argument("--oi_category", type=str, default="all")
    parser.add_argument("--oi_intent", type=str, default="all")
    parser.add_argument("--bps_path", type=str, default="configs/bps.npz")
    parser.add_argument("--bps_slot_cache_dir", type=str, default="cache/bps_slot")

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_neighbors", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument(
        "--class_weight_power",
        type=float,
        default=0.5,
        help="Use inverse-frequency^p class weights for CE; set <=0 to disable",
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--max_grad_norm", type=float, default=0.0)
    parser.add_argument("--early_stop_patience", type=int, default=10)

    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="checkpoints/predictor")
    parser.add_argument("--log_dir", type=str, default="logs/predictor")
    parser.add_argument("--log_every", type=int, default=50)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = PredictorTrainConfig(
        cache_dir=args.cache_dir,
        val_cache_dir=args.val_cache_dir,
        config_path=args.config_path,
        n_pc_points=args.n_pc_points,
        oi_category=args.oi_category,
        oi_intent=args.oi_intent,
        bps_path=args.bps_path,
        bps_slot_cache_dir=args.bps_slot_cache_dir,
        hidden_dim=args.hidden_dim,
        num_neighbors=args.num_neighbors,
        num_layers=args.num_layers,
        dropout=args.dropout,
        chunk_size=args.chunk_size,
        class_weight_power=args.class_weight_power,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        early_stop_patience=args.early_stop_patience,
        resume=args.resume,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        log_every=args.log_every,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
    )
    PredictorTrainer(cfg).train()


if __name__ == "__main__":
    main()
