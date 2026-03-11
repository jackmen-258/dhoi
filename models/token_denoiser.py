"""
token_denoiser.py (v16 — fixed CFG dropout + stronger classifier head)
=====================================================================
DiT-style Transformer Denoiser for Contact-Token Discrete Diffusion.

相对旧版的核心改动：
  - 适配简化后的 token schema（默认 vocab=26, PAD=24, MASK=25）
  - 修复 cross-attention 条件 dropout 逻辑（True=保留，不再取反）
  - 移除与训练脚本外部 CFG dropout 的重叠假设，统一由本模块负责 cond_dropout
  - 分类 head 改为小随机初始化，改善冷启动
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class AdaLayerNorm(nn.Module):
    """c → (scale, shift) → LayerNorm with affine modulation"""

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, hidden_dim * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(c).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    AdaLN → SelfAttn → [AdaLN → CrossAttn] → AdaLN → FFN
    """

    def __init__(self, hidden_dim, cond_dim, num_heads=8,
                 mlp_ratio=4.0, dropout=0.1, cross_attn=False):
        super().__init__()
        self.norm1 = AdaLayerNorm(hidden_dim, cond_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)

        self.has_cross_attn = cross_attn
        if cross_attn:
            self.norm_cross = AdaLayerNorm(hidden_dim, cond_dim)
            self.cross_attn = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm2 = AdaLayerNorm(hidden_dim, cond_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, c, pad_mask=None, cross_ctx=None):
        h = self.norm1(x, c)
        h, _ = self.self_attn(h, h, h, key_padding_mask=pad_mask)
        x = x + h

        if self.has_cross_attn and cross_ctx is not None:
            h = self.norm_cross(x, c)
            h, _ = self.cross_attn(query=h, key=cross_ctx, value=cross_ctx)
            x = x + h

        x = x + self.ffn(self.norm2(x, c))
        return x


class TokenDenoiser(nn.Module):
    """
    cond dict 键值：
        obj_feat:       (B, D_obj)       → AdaLN
        obj_point_feat: (B, N, D_pt)     → Cross-Attn
        text_feat:      (B, S, D_clip) / (B, D_clip)
    """

    def __init__(
        self,
        vocab_size: int = 26,
        max_seq_len: int = 12,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        pad_token_id: int = 24,
        mask_token_id: int = 25,
        cond_dropout: float = 0.1,
        obj_feat_dim: int = 256,
        obj_point_feat_dim: int = 256,
        text_feat_dim: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.cond_dropout = cond_dropout

        D = hidden_dim

        self.tok_emb = nn.Embedding(vocab_size, D)
        self.pos_emb = nn.Embedding(max_seq_len, D)

        self.time_mlp = nn.Sequential(
            SinusoidalTimestepEmbedding(D),
            nn.Linear(D, D), nn.SiLU(), nn.Linear(D, D),
        )

        self.use_obj_feat = obj_feat_dim > 0
        if self.use_obj_feat:
            self.obj_feat_proj = nn.Sequential(
                nn.Linear(obj_feat_dim, D), nn.SiLU(), nn.Linear(D, D),
            )

        self.use_txt = text_feat_dim > 0
        if self.use_txt:
            self.txt_global_proj = nn.Sequential(
                nn.Linear(text_feat_dim, D), nn.SiLU(), nn.Linear(D, D),
            )
            self.txt_proj = nn.Sequential(
                nn.Linear(text_feat_dim, D), nn.SiLU(), nn.Linear(D, D),
            )

        cond_in_dim = D
        if self.use_obj_feat:
            cond_in_dim += D
        if self.use_txt:
            cond_in_dim += D
        self.cond_fuse_mlp = nn.Sequential(
            nn.Linear(cond_in_dim, D), nn.SiLU(), nn.Linear(D, D),
        )

        self.use_pt = obj_point_feat_dim > 0
        if self.use_pt:
            self.pt_proj = nn.Sequential(
                nn.Linear(obj_point_feat_dim, D), nn.SiLU(), nn.Linear(D, D),
            )

        need_cross = self.use_pt or self.use_txt
        self.blocks = nn.ModuleList([
            DiTBlock(D, D, num_heads, mlp_ratio, dropout, cross_attn=need_cross)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.head:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _drop_mask(self, B: int, device) -> torch.Tensor:
        """(B,) bool；True = keep，False = drop"""
        return torch.rand(B, device=device) > self.cond_dropout

    def _build_adaln(self, t, cond):
        device = t.device
        B = t.shape[0]
        components = [self.time_mlp(t)]

        if self.use_obj_feat:
            if "obj_feat" in cond:
                g = self.obj_feat_proj(cond["obj_feat"])
                if self.training and self.cond_dropout > 0:
                    keep = self._drop_mask(B, device)
                    g = g * keep.float().unsqueeze(1)
                components.append(g)
            else:
                components.append(torch.zeros(B, self.hidden_dim, device=device))

        if self.use_txt:
            if "text_feat" in cond:
                txt = cond["text_feat"]
                txt_global = txt.mean(dim=1) if txt.dim() == 3 else txt
                tg = self.txt_global_proj(txt_global)
                if self.training and self.cond_dropout > 0:
                    keep = self._drop_mask(B, device)
                    tg = tg * keep.float().unsqueeze(1)
                components.append(tg)
            else:
                components.append(torch.zeros(B, self.hidden_dim, device=device))

        c_cat = torch.cat(components, dim=1)
        return self.cond_fuse_mlp(c_cat)

    def _build_cross_ctx(self, cond, B, device):
        parts = []

        if self.use_pt and "obj_point_feat" in cond:
            p = self.pt_proj(cond["obj_point_feat"])
            if self.training and self.cond_dropout > 0:
                keep = self._drop_mask(B, device)
                p = p * keep.float().view(B, 1, 1)
            parts.append(p)

        if self.use_txt and "text_feat" in cond:
            txt = cond["text_feat"]
            if txt.dim() == 2:
                txt = txt.unsqueeze(1)
            txt = self.txt_proj(txt)
            if self.training and self.cond_dropout > 0:
                keep = self._drop_mask(B, device)
                txt = txt * keep.float().view(B, 1, 1)
            parts.append(txt)

        return torch.cat(parts, dim=1) if parts else None

    def forward(self, x_t, t, cond=None):
        B, L = x_t.shape
        cond = cond or {}

        c = self._build_adaln(t, cond)
        cross = self._build_cross_ctx(cond, B, x_t.device)

        pos = torch.arange(L, device=x_t.device).unsqueeze(0)
        h = self.tok_emb(x_t) + self.pos_emb(pos)
        pad = (x_t == self.pad_token_id)

        for blk in self.blocks:
            h = blk(h, c, pad_mask=pad, cross_ctx=cross)

        return self.head(self.final_norm(h))

    def forward_cfg(self, x_t, t, cond, guidance_scale=2.0):
        logits_c = self.forward(x_t, t, cond)
        logits_u = self.forward(x_t, t, {})
        return logits_u + guidance_scale * (logits_c - logits_u)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        n = self.count_parameters()
        print(
            f"TokenDenoiser | L={len(self.blocks)} D={self.hidden_dim} "
            f"| cross={self.use_pt or self.use_txt} | {n:,} ({n/1e6:.2f}M)"
        )


_PRESETS = {
    "tiny":  dict(hidden_dim=128, num_layers=4, num_heads=4),
    "small": dict(hidden_dim=256, num_layers=6, num_heads=8),
    "base":  dict(hidden_dim=384, num_layers=8, num_heads=8),
}


def build_denoiser(config="small", **kw):
    p = _PRESETS.get(config, {}).copy() if isinstance(config, str) else dict(config)
    p.update(kw)
    return TokenDenoiser(**p)
