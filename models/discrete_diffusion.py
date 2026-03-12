"""
discrete_diffusion.py (v16 — PAD-aware Absorbing Diffusion)
===========================================================
Absorbing-State Discrete Diffusion (D3PM) for contact tokens.

相对旧版的核心改动：
  - 适配简化后的 token schema（默认 vocab=26, PAD=24, MASK=25）
  - PAD 不参与前向扩散；只对有效 semantic token 位加噪
  - 训练 loss 只在“有效位且被 mask 的位置”上计算
  - 采样时按位置约束 PAD：前 min_tokens 位禁止 PAD
  - 每一步采样后都强制 PAD 连续，避免非法中间态反复滚动
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class AbsorbingDiffusion(nn.Module):
    """
    Absorbing-state 离散扩散

    q(x_t|x_0): 有效 semantic token 以 1-ā_t 概率替换为 MASK；PAD 位保持 PAD
    p(x_0|x_t): 由 TokenDenoiser 参数化
    """

    def __init__(
        self,
        vocab_size: int = 26,
        mask_token_id: int = 25,
        pad_token_id: int = 24,
        num_timesteps: int = 100,
        schedule: str = "cosine",
        loss_type: str = "x0_ce",
        schedule_s: float = 0.008,
        min_tokens: int = 1,
        pad_logit_bias: float = -4.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.num_timesteps = num_timesteps
        self.loss_type = loss_type
        self.min_tokens = int(min_tokens)
        self.pad_logit_bias = float(pad_logit_bias)

        alpha_bar = self._build_schedule(schedule, num_timesteps, schedule_s)
        self.register_buffer("alpha_bar", alpha_bar)  # (T+1,)

        alpha = torch.cat([torch.tensor([1.0]), alpha_bar[1:] / alpha_bar[:-1]])
        self.register_buffer("alpha", alpha.clamp(min=1e-5, max=1.0))

    @staticmethod
    def _build_schedule(schedule, T, s=0.008):
        if schedule == "cosine":
            steps = torch.arange(T + 1, dtype=torch.float64)
            f = torch.cos(((steps / T) + s) / (1 + s) * (math.pi / 2)) ** 2
            return (f / f[0]).clamp(min=1e-5, max=1.0).float()
        if schedule == "linear":
            return (1.0 - torch.linspace(0, 1, T + 1)).clamp(min=1e-5).float()
        raise ValueError(f"Unknown schedule: {schedule}")

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, token_mask: Optional[torch.Tensor] = None):
        """
        对 x_0 中的所有位置做 absorbing corruption，包括尾部 PAD 位。

        这样模型在训练时也会看到 “MASK -> PAD” 的恢复任务，
        否则采样阶段从全 MASK 开始时，模型学不会何时终止序列。

        Args:
            x_0: (B, L)
            t:   (B,)
            token_mask: 保留兼容接口，当前不参与 corruption 决策
        """
        B, L = x_0.shape
        ab = self.alpha_bar[t].unsqueeze(1).expand(B, L)
        keep = torch.rand(B, L, device=x_0.device) < ab
        x_t = torch.where(keep, x_0, self.mask_token_id)
        return x_t

    def q_posterior(self, x_0, x_t, t):
        """
        q(x_{t-1}|x_t, x_0) 的 unmask 概率。
        对 MASK 位置： (ā_{t-1} - ā_t) / (1 - ā_t)
        对非 MASK：保持不变。
        """
        B, L = x_0.shape
        ab_t = self.alpha_bar[t].unsqueeze(1).expand(B, L)
        ab_s = self.alpha_bar[(t - 1).clamp(min=0)].unsqueeze(1).expand(B, L)
        prob = ((ab_s - ab_t) / (1 - ab_t).clamp(min=1e-8)).clamp(0.0, 1.0)
        return torch.where(x_t == self.mask_token_id, prob, torch.ones_like(prob))

    def training_loss(self, model, x_0, token_mask=None, cond=None):
        B, L = x_0.shape
        device = x_0.device

        if token_mask is None:
            token_mask = (x_0 != self.pad_token_id)
        token_mask = token_mask.bool()
        pad_mask = ~token_mask

        t = torch.randint(1, self.num_timesteps + 1, (B,), device=device)
        x_t = self.q_sample(x_0, t, token_mask=token_mask)
        logits = model(x_t, t, cond)  # (B, L, V)

        is_masked = (x_t == self.mask_token_id)
        supervise_mask = is_masked

        flat_logits = logits.reshape(-1, self.vocab_size)
        flat_targets = x_0.reshape(-1)

        ce = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        ce = ce.view(B, L)

        n_supervised = supervise_mask.float().sum().clamp(min=1.0)
        loss = (ce * supervise_mask.float()).sum() / n_supervised

        with torch.no_grad():
            masked_acc = torch.tensor(0.0, device=device)
            semantic_acc = torch.tensor(0.0, device=device)
            pad_acc = torch.tensor(0.0, device=device)
            semantic_supervise_mask = supervise_mask & token_mask
            pad_supervise_mask = supervise_mask & pad_mask

            if supervise_mask.any():
                pred = logits.argmax(dim=-1)
                masked_acc = (pred[supervise_mask] == x_0[supervise_mask]).float().mean()
            if semantic_supervise_mask.any():
                pred = logits.argmax(dim=-1)
                semantic_acc = (
                    pred[semantic_supervise_mask] == x_0[semantic_supervise_mask]
                ).float().mean()
            if pad_supervise_mask.any():
                pred = logits.argmax(dim=-1)
                pad_acc = (
                    pred[pad_supervise_mask] == x_0[pad_supervise_mask]
                ).float().mean()

        return {
            "loss": loss,
            "loss_mask_only": loss,
            "mask_ratio": supervise_mask.float().mean(),
            "semantic_mask_ratio": semantic_supervise_mask.float().mean(),
            "pad_mask_ratio": pad_supervise_mask.float().mean(),
            "masked_acc": masked_acc,
            "semantic_acc_masked": semantic_acc,
            "pad_acc_masked": pad_acc,
        }

    @torch.no_grad()
    def sample(
        self,
        model,
        cond=None,
        batch_size: int = 1,
        seq_length: int = 12,
        temperature: float = 0.9,
        top_k: int = 0,
        guidance_scale: float = 2.0,
        min_tokens: Optional[int] = None,
    ):
        device = next(model.parameters()).device
        use_cfg = guidance_scale > 0
        min_tokens = self.min_tokens if min_tokens is None else int(min_tokens)
        min_tokens = max(0, min(min_tokens, seq_length))

        x_t = torch.full((batch_size, seq_length), self.mask_token_id,
                         dtype=torch.long, device=device)
        trajectories = [x_t.clone()]

        valid_tokens = torch.zeros(self.vocab_size, dtype=torch.bool, device=device)
        valid_tokens[:self.pad_token_id] = True   # semantic tokens
        valid_tokens[self.pad_token_id] = True    # PAD
        invalid_tokens = ~valid_tokens

        pos = torch.arange(seq_length, device=device)

        for t_val in range(self.num_timesteps, 0, -1):
            t = torch.full((batch_size,), t_val, dtype=torch.long, device=device)
            is_masked = (x_t == self.mask_token_id)

            if not is_masked.any():
                trajectories.append(x_t.clone())
                continue

            if use_cfg:
                logits = model.forward_cfg(x_t, t, cond, guidance_scale)
            else:
                logits = model(x_t, t, cond)

            if temperature != 1.0:
                logits = logits / temperature

            logits[:, :, invalid_tokens] = float("-inf")

            # 前 min_tokens 位禁止生成 PAD
            if min_tokens > 0:
                early = pos < min_tokens
                logits[:, early, self.pad_token_id] = float("-inf")

            # 对靠前位置整体压低 PAD 倾向
            if self.pad_logit_bias != 0.0 and seq_length > 1:
                ramp = pos.float() / float(seq_length - 1)
                pad_bias = self.pad_logit_bias * (1.0 - ramp)
                if min_tokens > 0:
                    pad_bias = torch.where(early, torch.full_like(pad_bias, -1e9), pad_bias)
                logits[:, :, self.pad_token_id] += pad_bias.view(1, seq_length)

            if top_k > 0:
                kth, _ = logits.topk(top_k, dim=-1)
                logits = torch.where(
                    logits >= kth[..., -1:],
                    logits,
                    torch.full_like(logits, float("-inf"))
                )

            probs = F.softmax(logits, dim=-1)  # (B, L, V)

            x_0_pred = x_t.clone()                  # 非 MASK 位直接保持原值
            masked_probs = probs[is_masked]         # (N_masked, V)

            masked_probs = torch.nan_to_num(
                masked_probs, nan=0.0, posinf=0.0, neginf=0.0
            )
            row_sum = masked_probs.sum(dim=-1, keepdim=True)

            # fallback：只在合法 token 集上均匀
            fallback = torch.zeros_like(masked_probs)
            fallback[:, :self.pad_token_id] = 1.0
            fallback[:, self.pad_token_id] = 1.0
            fallback = fallback / fallback.sum(dim=-1, keepdim=True)

            masked_probs = torch.where(
                row_sum > 0,
                masked_probs / row_sum.clamp(min=1e-8),
                fallback,
            )

            sampled_tokens = torch.multinomial(masked_probs, 1).squeeze(-1)  # (N_masked,)
            x_0_pred[is_masked] = sampled_tokens

            prob_unmask = self.q_posterior(x_0_pred, x_t, t)
            should_unmask = torch.rand_like(prob_unmask) < prob_unmask

            x_t = torch.where(is_masked & should_unmask, x_0_pred, x_t)
            x_t = self._enforce_contiguous_pad(x_t)
            trajectories.append(x_t.clone())

        x_t = self._enforce_contiguous_pad(x_t)
        return {"samples": x_t, "trajectories": trajectories}

    def _enforce_contiguous_pad(self, x: torch.Tensor) -> torch.Tensor:
        """第一个 PAD 出现后，后续全部强制 PAD。"""
        is_pad = (x == self.pad_token_id)
        pad_cumsum = is_pad.long().cumsum(dim=1)
        force_pad = (pad_cumsum > 0)
        return torch.where(force_pad, torch.full_like(x, self.pad_token_id), x)

    def get_schedule_info(self):
        return {
            "alpha_bar": self.alpha_bar,
            "alpha": self.alpha,
            "mask_rate": 1.0 - self.alpha_bar,
        }

    def estimate_snr(self, t):
        ab = self.alpha_bar[t]
        return ab / (1 - ab).clamp(min=1e-8)
