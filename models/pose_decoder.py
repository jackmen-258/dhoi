"""
pose_decoder.py — Flow Matching Pose Decoder
=============================================
Conditional Flow Matching (OT-CFM) model for generating hand pose (rot6d),
translation, and shape parameters from contact tokens + object geometry.

State vector x ∈ R^{X_DIM}:
    x = [rot6d (96), trans_norm (3), shape (10)]
    where trans_norm = trans / obj_scale

Training:  Learn velocity field v_θ(x_t, t, cond) via OT-CFM loss.
Sampling:  Euler ODE integration from t=0 (noise) to t=1 (data).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from models.object_normalization import compute_object_scale
from utils.pose_utils import rot6d_to_aa, aa_to_rot6d


NUM_HAND_PARTS = 6
NUM_JOINTS = 16
AA_POSE_DIM = 48
ROT6D_POSE_DIM = 96
TRANS_DIM = 3
NUM_SLOTS = 4
DEFAULT_N_BETAS = 10

# Token hand-part ids are assumed to follow:
#   0=THUMB, 1=INDEX, 2=MIDDLE, 3=RING, 4=LITTLE, 5=PALM
# MANO joints:
#   0=wrist, 1-3=index, 4-6=middle, 7-9=little, 10-12=ring, 13-15=thumb
JOINT_TO_PART = [
    5,
    1, 1, 1,
    2, 2, 2,
    4, 4, 4,
    3, 3, 3,
    0, 0, 0,
]


def ensure_nonempty_token_mask(token_mask: torch.Tensor) -> torch.Tensor:
    """Ensure each sample has at least one valid token."""
    any_valid = token_mask.any(dim=1)
    if any_valid.all():
        return token_mask
    safe_mask = token_mask.clone()
    safe_mask[~any_valid, 0] = True
    return safe_mask


@dataclass
class PoseDecoderOutput:
    pose: torch.Tensor    # (B, 48) axis-angle
    trans: torch.Tensor   # (B, 3) metric space
    shape: torch.Tensor   # (B, 10) betas


# ==============================================================================
# Conditioning (unchanged from regression version)
# ==============================================================================

class ConditionBlock(nn.Module):
    """Transformer block with optional spatial cross-attention."""

    def __init__(self, D: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.1, use_spatial: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(D)
        self.self_attn = nn.MultiheadAttention(
            D, num_heads, dropout=dropout, batch_first=True
        )

        self.use_spatial = use_spatial
        if use_spatial:
            self.norm_ca = nn.LayerNorm(D)
            self.norm_kv = nn.LayerNorm(D)
            self.cross_attn = nn.MultiheadAttention(
                D, num_heads, dropout=dropout, batch_first=True
            )

        self.norm2 = nn.LayerNorm(D)
        h = int(D * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(D, h), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h, D), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None,
                spatial_kv: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h, key_padding_mask=pad_mask)
        if pad_mask is not None:
            h = h.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        x = x + h

        if self.use_spatial and spatial_kv is not None:
            h2 = self.norm_ca(x)
            kv = self.norm_kv(spatial_kv)
            h2, _ = self.cross_attn(h2, kv, kv)
            if pad_mask is not None:
                h2 = h2.masked_fill(pad_mask.unsqueeze(-1), 0.0)
            x = x + h2

        x = x + self.ffn(self.norm2(x))
        return x


class ConditionEncoder(nn.Module):
    """
    Condition encoder for simplified (hand_part, slot) tokens.

    Conditioning layout:
      - 128 SA2 points provide geometry tokens
      - slot_labels are aligned to the same 128 SA2 points
    """

    def __init__(
        self,
        vocab_size: int = 26,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        obj_global_dim: int = 256,
        obj_point_dim: int = 256,
        obj_xyz_dim: int = 3,
        num_hand_parts: int = NUM_HAND_PARTS,
        num_slots: int = NUM_SLOTS,
        slot_point_emb_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        D = hidden_dim
        self.hidden_dim = D
        self.num_slots = num_slots
        self.unknown_slot_id = num_slots

        # token decomposition embeddings
        self.hand_part_emb = nn.Embedding(num_hand_parts + 1, D)
        self.slot_emb = nn.Embedding(num_slots + 1, D)

        # global object conditioning
        self.global_proj = nn.Sequential(
            nn.Linear(obj_global_dim, D), nn.SiLU(), nn.Linear(D, D)
        )
        self.obj_to_scale_shift = nn.Sequential(
            nn.Linear(obj_global_dim, D), nn.SiLU(), nn.Linear(D, D * 2)
        )
        nn.init.zeros_(self.obj_to_scale_shift[-1].weight)
        nn.init.zeros_(self.obj_to_scale_shift[-1].bias)

        self.slot_point_emb_dim = int(slot_point_emb_dim)
        self.slot_point_emb = nn.Embedding(num_slots + 1, self.slot_point_emb_dim)

        spatial_in_dim = obj_point_dim + obj_xyz_dim + self.slot_point_emb_dim
        self.use_spatial = spatial_in_dim > 0
        if self.use_spatial:
            self.spatial_proj = nn.Sequential(
                nn.Linear(spatial_in_dim, D), nn.SiLU(), nn.Linear(D, D)
            )

        self.blocks = nn.ModuleList([
            ConditionBlock(D, num_heads, dropout=dropout, use_spatial=self.use_spatial)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(D)

        self._build_token_lookup(vocab_size, num_hand_parts, num_slots)
        self._init_weights()

    def _build_token_lookup(self, V: int, H: int, S: int):
        hp = torch.full((V,), H, dtype=torch.long)
        sl = torch.full((V,), S, dtype=torch.long)
        n_semantic = min(V, H * S)
        for tok in range(n_semantic):
            hp[tok] = tok // S
            sl[tok] = tok % S
        self.register_buffer("_hp_ids", hp)
        self.register_buffer("_sl_ids", sl)

    def _init_weights(self):
        nn.init.normal_(self.hand_part_emb.weight, std=0.02)
        nn.init.normal_(self.slot_emb.weight, std=0.02)
        nn.init.normal_(self.slot_point_emb.weight, std=0.02)

    def _embed_slot_labels(
        self,
        slot_labels: torch.Tensor | None,
        batch_size: int,
        num_points: int,
        device: torch.device,
    ) -> torch.Tensor:
        if slot_labels is None:
            ids = torch.full(
                (batch_size, num_points),
                self.unknown_slot_id,
                dtype=torch.long,
                device=device,
            )
        else:
            ids = slot_labels.to(device=device, dtype=torch.long)
            ids = torch.where(
                (ids >= 0) & (ids < self.num_slots),
                ids,
                torch.full_like(ids, self.unknown_slot_id),
            )
        return self.slot_point_emb(ids)

    def forward(self, contact_tokens: torch.Tensor, token_mask: torch.Tensor,
                obj_global_feat: torch.Tensor,
                obj_point_feat: torch.Tensor | None = None,
                obj_point_xyz: torch.Tensor | None = None,
                slot_labels: torch.Tensor | None = None,
                slot_geo_feat: torch.Tensor | None = None,
                obj_pc: torch.Tensor | None = None):
        del slot_geo_feat  # kept only for backward-compatible call signatures

        B, L = contact_tokens.shape
        token_mask = ensure_nonempty_token_mask(token_mask)

        hp = self._hp_ids[contact_tokens]
        sl = self._sl_ids[contact_tokens]

        h = self.hand_part_emb(hp) + self.slot_emb(sl)

        c_global = self.global_proj(obj_global_feat)
        scale, shift = self.obj_to_scale_shift(obj_global_feat).chunk(2, dim=-1)
        h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        spatial_summary = c_global
        if self.use_spatial and (obj_point_feat is not None or obj_point_xyz is not None):
            if obj_point_feat is None:
                raise ValueError("obj_point_feat must be provided when using spatial conditioning.")
            if obj_point_xyz is None:
                obj_point_xyz = obj_point_feat.new_zeros(
                    obj_point_feat.shape[0], obj_point_feat.shape[1], 3
                )

            num_points = obj_point_feat.shape[1]
            slot_feat = self._embed_slot_labels(slot_labels, B, num_points, obj_point_feat.device)
            spatial_in = torch.cat([obj_point_feat, obj_point_xyz, slot_feat], dim=-1)
            spatial_kv = self.spatial_proj(spatial_in)
            spatial_summary = spatial_kv.mean(dim=1)
        else:
            spatial_kv = None

        if obj_pc is not None:
            obj_scale = compute_object_scale(obj_pc.to(device=obj_global_feat.device, dtype=obj_global_feat.dtype))
        else:
            obj_scale = obj_global_feat.new_ones(B, 1)

        pad_mask = ~token_mask
        for blk in self.blocks:
            h = blk(h, pad_mask=pad_mask, spatial_kv=spatial_kv)
        h = self.final_norm(h)

        return {
            "c_global": c_global,
            "token_feats": h,
            "token_mask": token_mask,
            "token_parts": hp,
            "token_slots": sl,
            "spatial_kv": spatial_kv,
            "spatial_summary": spatial_summary,
            "obj_scale": obj_scale,
            "obj_point_xyz": obj_point_xyz,
        }


# ==============================================================================
# Transformer blocks (reused from regression version)
# ==============================================================================

class PoseHeadBlock(nn.Module):
    """Pose head block with weak part / slot priors over token attention."""

    def __init__(self, D: int, cond_dim: int, num_heads: int = 8,
                 num_slots: int = NUM_SLOTS, dropout: float = 0.1):
        super().__init__()
        assert D % num_heads == 0
        self.num_heads = num_heads
        self.num_slots = num_slots
        self.D = D
        self.head_dim = D // num_heads

        self.film = nn.Linear(cond_dim, D * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

        self.norm_sa = nn.LayerNorm(D)
        self.self_attn = nn.MultiheadAttention(D, num_heads, dropout=dropout, batch_first=True)

        self.norm_ca = nn.LayerNorm(D)
        self.norm_kv = nn.LayerNorm(D)
        self.cross_q = nn.Linear(D, D)
        self.cross_k = nn.Linear(D, D)
        self.cross_v = nn.Linear(D, D)
        self.cross_out = nn.Linear(D, D)
        self.cross_drop = nn.Dropout(dropout)
        self.norm_ca_spatial = nn.LayerNorm(D)
        self.norm_spatial_kv = nn.LayerNorm(D)
        self.spatial_attn = nn.MultiheadAttention(
            D, num_heads, dropout=dropout, batch_first=True
        )

        self.part_attn_bias = nn.Parameter(torch.zeros(num_heads))
        self.slot_attn_bias = nn.Parameter(torch.zeros(num_heads, num_slots))

        self.norm_ff = nn.LayerNorm(D)
        self.ffn = nn.Sequential(
            nn.Linear(D, D * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(D * 4, D), nn.Dropout(dropout),
        )

    def forward(self, h: torch.Tensor, cond_vec: torch.Tensor,
                token_kv: torch.Tensor, token_mask: torch.Tensor,
                part_align_mask: torch.Tensor, token_slots: torch.Tensor,
                spatial_kv: torch.Tensor | None = None) -> torch.Tensor:
        B, L = token_kv.shape[:2]

        s, sh = self.film(cond_vec).chunk(2, dim=-1)
        h = h * (1 + s.unsqueeze(1)) + sh.unsqueeze(1)

        h2 = self.norm_sa(h)
        h2, _ = self.self_attn(h2, h2, h2)
        h = h + h2

        h3 = self.norm_ca(h)
        kv = self.norm_kv(token_kv)

        part_bias = (
            part_align_mask.float().unsqueeze(1)
            * self.part_attn_bias.view(1, self.num_heads, 1, 1)
        )

        clamped_slots = token_slots.clamp(min=0, max=self.num_slots - 1)
        slot_bias_per_token = self.slot_attn_bias[:, clamped_slots]   # (H, B, L)
        slot_bias = slot_bias_per_token.permute(1, 0, 2).unsqueeze(2)  # (B, H, 1, L)

        q = self.cross_q(h3).view(B, NUM_JOINTS, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.cross_k(kv).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.cross_v(kv).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores + part_bias + slot_bias
        scores = scores.masked_fill((~token_mask).view(B, 1, 1, L), torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1)
        attn = self.cross_drop(attn)
        h3 = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, NUM_JOINTS, self.D)
        h3 = self.cross_out(h3)
        h = h + h3

        if spatial_kv is not None:
            h4 = self.norm_ca_spatial(h)
            spatial_kv = self.norm_spatial_kv(spatial_kv)
            h4, _ = self.spatial_attn(h4, spatial_kv, spatial_kv)
            h = h + h4

        h = h + self.ffn(self.norm_ff(h))
        return h


# ==============================================================================
# Flow Matching components
# ==============================================================================

class TimestepEmbedding(nn.Module):
    """Sinusoidal → MLP timestep embedding."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) in [0, 1] → (B, dim)."""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class FlowTrunk(nn.Module):
    """
    Velocity field network for flow matching.

    Takes noisy state x_t, timestep t, and conditioning cond to predict velocity.
    Architecture mirrors SharedTaskTrunk but injects x_t and t:
      - Per-joint noisy rot6d → projected and added to joint queries
      - Noisy trans_norm + shape → projected into cond_vec (FiLM)
      - Timestep → sinusoidal embedding added to cond_vec
    """

    def __init__(self, hidden_dim: int = 512, num_blocks: int = 8,
                 num_heads: int = 8, num_slots: int = NUM_SLOTS,
                 n_betas: int = DEFAULT_N_BETAS, dropout: float = 0.1):
        super().__init__()
        D = hidden_dim
        self.n_betas = n_betas
        self.x_dim = ROT6D_POSE_DIM + TRANS_DIM + n_betas

        # Joint queries + positional embedding
        self.joint_queries = nn.Parameter(torch.randn(NUM_JOINTS, D) * 0.02)
        self.joint_pos_emb = nn.Embedding(NUM_JOINTS, D)
        nn.init.normal_(self.joint_pos_emb.weight, std=0.02)
        self.register_buffer("joint_parts", torch.tensor(JOINT_TO_PART, dtype=torch.long))

        # Timestep embedding
        self.time_emb = TimestepEmbedding(D)

        # Noisy state injection — trans and shape are separate
        self.x_rot_proj = nn.Sequential(
            nn.Linear(6, D), nn.SiLU(), nn.Linear(D, D)
        )
        self.x_trans_proj = nn.Sequential(
            nn.Linear(TRANS_DIM, D), nn.SiLU(), nn.Linear(D, D)
        )
        self.x_shape_proj = nn.Sequential(
            nn.Linear(n_betas, D), nn.SiLU(), nn.Linear(D, D)
        )

        # Condition projection
        self.cond_proj = nn.Sequential(nn.Linear(D, D), nn.SiLU(), nn.Linear(D, D))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            PoseHeadBlock(D, D, num_heads=num_heads, num_slots=num_slots, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.joint_norm = nn.LayerNorm(D)

        # Global latent
        self.global_proj = nn.Sequential(
            nn.Linear(D * 3, D), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(D, D),
        )

        # Per-block trans spatial cross-attention (deep spatial reasoning for translation)
        self.trans_spatial_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.trans_spatial_blocks.append(nn.ModuleDict({
                "norm_q": nn.LayerNorm(D),
                "norm_kv": nn.LayerNorm(D),
                "attn": nn.MultiheadAttention(D, num_heads, dropout=dropout, batch_first=True),
                "film": nn.Linear(D, D * 2),
            }))
            nn.init.zeros_(self.trans_spatial_blocks[-1]["film"].weight)
            nn.init.zeros_(self.trans_spatial_blocks[-1]["film"].bias)
        self.trans_final_norm = nn.LayerNorm(D)

        # Translation fuse: merge deep trans query with global state
        self.trans_fuse = nn.Sequential(
            nn.Linear(D * 2, D), nn.SiLU(), nn.Linear(D, D)
        )

        # Velocity output heads (zero-init for stable start)
        self.vel_rot_head = nn.Linear(D, 6)
        nn.init.zeros_(self.vel_rot_head.weight)
        nn.init.zeros_(self.vel_rot_head.bias)

        self.vel_trans_head = nn.Sequential(
            nn.Linear(D, D // 2), nn.SiLU(),
            nn.Linear(D // 2, TRANS_DIM),
        )
        nn.init.zeros_(self.vel_trans_head[-1].weight)
        nn.init.zeros_(self.vel_trans_head[-1].bias)

        self.vel_shape_head = nn.Sequential(
            nn.Linear(D, D // 2), nn.SiLU(),
            nn.Linear(D // 2, n_betas),
        )
        nn.init.zeros_(self.vel_shape_head[-1].weight)
        nn.init.zeros_(self.vel_shape_head[-1].bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: dict) -> torch.Tensor:
        """
        Predict velocity field v_θ(x_t, t, cond).

        Args:
            x_t: (B, X_DIM) noisy state [rot6d, trans_norm, shape]
            t:   (B,) timestep in [0, 1]
            cond: dict from ConditionEncoder

        Returns:
            v: (B, X_DIM) predicted velocity
        """
        B = x_t.shape[0]
        device = x_t.device

        # Decompose noisy state
        rot6d_t = x_t[:, :ROT6D_POSE_DIM].reshape(B, NUM_JOINTS, 6)
        trans_t = x_t[:, ROT6D_POSE_DIM:ROT6D_POSE_DIM + TRANS_DIM]  # (B, 3)
        shape_t = x_t[:, ROT6D_POSE_DIM + TRANS_DIM:]                # (B, n_betas)

        # Embeddings — trans and shape encoded separately
        t_emb = self.time_emb(t)                        # (B, D)
        rot_emb = self.x_rot_proj(rot6d_t)              # (B, 16, D)
        trans_emb = self.x_trans_proj(trans_t)           # (B, D)
        shape_emb = self.x_shape_proj(shape_t)           # (B, D)

        # Build joint queries with noisy rot6d injection
        h = self.joint_queries.unsqueeze(0).expand(B, -1, -1)
        h = h + self.joint_pos_emb(torch.arange(NUM_JOINTS, device=device)).unsqueeze(0)
        h = h + rot_emb

        # Conditioning from tokens + object
        token_kv = cond["token_feats"]
        token_mask = cond["token_mask"]
        token_parts = cond["token_parts"]
        token_slots = cond["token_slots"]
        c_global = cond["c_global"]
        spatial_kv = cond.get("spatial_kv", None)

        part_align_mask = (
            (self.joint_parts.view(1, NUM_JOINTS, 1) == token_parts.unsqueeze(1))
            & token_mask.unsqueeze(1)
        )

        # cond_vec = object context + timestep + noisy trans + noisy shape
        cond_vec = self.cond_proj(c_global) + t_emb + trans_emb + shape_emb

        # Parallel trans query — deep spatial reasoning for translation
        trans_h = trans_emb.unsqueeze(1)  # (B, 1, D) — single query token

        # Transformer blocks (joint queries + parallel trans query)
        for blk, trans_blk in zip(self.blocks, self.trans_spatial_blocks):
            h = blk(
                h,
                cond_vec,
                token_kv,
                token_mask,
                part_align_mask,
                token_slots,
                spatial_kv=spatial_kv,
            )
            # Trans spatial cross-attention at each layer
            if spatial_kv is not None:
                s, sh = trans_blk["film"](cond_vec).chunk(2, dim=-1)
                trans_h = trans_h * (1 + s.unsqueeze(1)) + sh.unsqueeze(1)
                tq = trans_blk["norm_q"](trans_h)
                tkv = trans_blk["norm_kv"](spatial_kv)
                ta, _ = trans_blk["attn"](tq, tkv, tkv)
                trans_h = trans_h + ta

        joint_latent = self.joint_norm(h)
        trans_h = self.trans_final_norm(trans_h).squeeze(1)  # (B, D)

        # Global latent
        mask_f = token_mask.float().unsqueeze(-1)
        token_denom = mask_f.sum(1).clamp(min=1.0)
        token_summary = (token_kv * mask_f).sum(1) / token_denom
        joint_summary = joint_latent.mean(dim=1)
        spatial_summary = cond.get("spatial_summary", c_global)
        global_latent = self.global_proj(
            torch.cat([joint_summary, token_summary, spatial_summary], dim=-1)
        )
        global_state = global_latent + t_emb + trans_emb + shape_emb

        if spatial_kv is not None:
            trans_latent = self.trans_fuse(
                torch.cat([global_state, trans_h], dim=-1)
            )
        else:
            trans_latent = global_state

        # Predict velocity
        v_rot = self.vel_rot_head(joint_latent).reshape(B, ROT6D_POSE_DIM)
        v_trans = self.vel_trans_head(trans_latent)
        v_shape = self.vel_shape_head(global_state)

        return torch.cat([v_rot, v_trans, v_shape], dim=-1)


# ==============================================================================
# Top-level Flow Matching Model
# ==============================================================================

class PoseFlowModel(nn.Module):
    """
    Flow Matching pose decoder.

    Learns a conditional velocity field to generate hand pose, translation,
    and shape from contact tokens and object geometry.
    """

    def __init__(
        self,
        vocab_size: int = 26,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        obj_global_dim: int = 256,
        obj_point_dim: int = 256,
        obj_xyz_dim: int = 3,
        num_hand_parts: int = NUM_HAND_PARTS,
        num_slots: int = NUM_SLOTS,
        n_betas: int = DEFAULT_N_BETAS,
        dropout: float = 0.1,
        num_flow_blocks: int = 8,
        slot_point_emb_dim: int = 64,
        loss_weight_rot: float = 1.0,
        loss_weight_trans: float = 1.0,
        loss_weight_shape: float = 1.0,
        num_contact_types: int | None = None,   # backward compat, ignored
    ):
        super().__init__()
        del num_contact_types
        self.n_betas = n_betas
        self.x_dim = ROT6D_POSE_DIM + TRANS_DIM + n_betas
        self.loss_weight_rot = float(loss_weight_rot)
        self.loss_weight_trans = float(loss_weight_trans)
        self.loss_weight_shape = float(loss_weight_shape)

        total_loss_weight = (
            self.loss_weight_rot
            + self.loss_weight_trans
            + self.loss_weight_shape
        )
        if total_loss_weight <= 0:
            raise ValueError("loss weights must sum to a positive value.")
        self.total_loss_weight = total_loss_weight

        self.cond_encoder = ConditionEncoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            obj_global_dim=obj_global_dim,
            obj_point_dim=obj_point_dim,
            obj_xyz_dim=obj_xyz_dim,
            num_hand_parts=num_hand_parts,
            num_slots=num_slots,
            slot_point_emb_dim=slot_point_emb_dim,
            dropout=dropout,
        )

        self.flow_trunk = FlowTrunk(
            hidden_dim=hidden_dim,
            num_blocks=num_flow_blocks,
            num_heads=num_heads,
            num_slots=num_slots,
            n_betas=n_betas,
            dropout=dropout,
        )

    # ----------------------------------------------------------------
    # Condition encoding (same API as regression version)
    # ----------------------------------------------------------------

    def encode_condition(self, tokens: torch.Tensor, mask: torch.Tensor,
                         obj_global_feat: torch.Tensor,
                         obj_point_feat: torch.Tensor | None = None,
                         obj_point_xyz: torch.Tensor | None = None,
                         slot_labels: torch.Tensor | None = None,
                         slot_geo_feat: torch.Tensor | None = None,
                         obj_pc: torch.Tensor | None = None) -> dict:
        cond = self.cond_encoder(
            tokens,
            mask,
            obj_global_feat,
            obj_point_feat=obj_point_feat,
            obj_point_xyz=obj_point_xyz,
            slot_labels=slot_labels,
            slot_geo_feat=slot_geo_feat,
            obj_pc=obj_pc,
        )
        if obj_pc is not None:
            cond["obj_pc"] = obj_pc
        return cond

    # ----------------------------------------------------------------
    # State packing / unpacking
    # ----------------------------------------------------------------

    def pack_data(self, gt_pose_aa: torch.Tensor, gt_trans: torch.Tensor,
                  gt_shape: torch.Tensor, cond: dict) -> torch.Tensor:
        """Pack ground truth into normalized flow state vector."""
        rot6d = aa_to_rot6d(gt_pose_aa)                 # (B, 96)
        scale = cond.get("obj_scale",
                         gt_trans.new_ones(gt_trans.shape[0], 1))
        trans_norm = gt_trans / scale                   # (B, 3)
        return torch.cat([rot6d, trans_norm, gt_shape], dim=-1)

    def unpack_data(self, x: torch.Tensor, cond: dict) -> PoseDecoderOutput:
        """Unpack flow state vector to pose (aa), trans (metric), shape."""
        rot6d = x[:, :ROT6D_POSE_DIM]
        trans_norm = x[:, ROT6D_POSE_DIM:ROT6D_POSE_DIM + TRANS_DIM]
        shape = x[:, ROT6D_POSE_DIM + TRANS_DIM:]

        pose_aa = rot6d_to_aa(rot6d)
        scale = cond.get("obj_scale",
                         trans_norm.new_ones(trans_norm.shape[0], 1))
        trans = trans_norm * scale

        return PoseDecoderOutput(pose=pose_aa, trans=trans, shape=shape)

    # ----------------------------------------------------------------
    # Velocity prediction
    # ----------------------------------------------------------------

    def forward_velocity(self, x_t: torch.Tensor, t: torch.Tensor,
                         cond: dict) -> torch.Tensor:
        """Predict velocity v_θ(x_t, t, cond)."""
        return self.flow_trunk(x_t, t, cond)

    # ----------------------------------------------------------------
    # Training: OT-CFM loss
    # ----------------------------------------------------------------

    def compute_loss(self, x_data: torch.Tensor, cond: dict) -> tuple:
        """
        OT Conditional Flow Matching loss.

        Path:  x_t = (1-t) * noise + t * x_data
        Target velocity:  v = x_data - noise

        Returns:
            loss: scalar
            metrics: dict with per-component MSE
        """
        B = x_data.shape[0]
        device = x_data.device

        t = torch.rand(B, device=device)
        noise = torch.randn_like(x_data)

        t_expand = t.unsqueeze(-1)
        x_t = (1.0 - t_expand) * noise + t_expand * x_data
        v_target = x_data - noise

        v_pred = self.forward_velocity(x_t, t, cond)

        rot_end = ROT6D_POSE_DIM
        trans_end = rot_end + TRANS_DIM
        v_rot_mse = F.mse_loss(v_pred[:, :rot_end], v_target[:, :rot_end])
        v_trans_mse = F.mse_loss(v_pred[:, rot_end:trans_end], v_target[:, rot_end:trans_end])
        v_shape_mse = F.mse_loss(v_pred[:, trans_end:], v_target[:, trans_end:])

        loss = (
            self.loss_weight_rot * v_rot_mse
            + self.loss_weight_trans * v_trans_mse
            + self.loss_weight_shape * v_shape_mse
        ) / self.total_loss_weight

        return loss, {
            "flow_loss": loss.item(),
            "v_rot": v_rot_mse.item(),
            "v_trans": v_trans_mse.item(),
            "v_shape": v_shape_mse.item(),
        }

    # ----------------------------------------------------------------
    # Sampling: Euler ODE integration
    # ----------------------------------------------------------------

    @torch.no_grad()
    def sample(self, cond: dict, n_steps: int = 20) -> PoseDecoderOutput:
        """
        Generate samples via Euler ODE integration.

        Integrates from t=0 (noise) to t=1 (data).
        """
        B = cond["c_global"].shape[0]
        device = cond["c_global"].device

        x = torch.randn(B, self.x_dim, device=device)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=device)
            v = self.forward_velocity(x, t, cond)
            x = x + v * dt

        return self.unpack_data(x, cond)

    # ----------------------------------------------------------------
    # Convenience forward (for inference / train_refine.py compat)
    # ----------------------------------------------------------------

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor,
                obj_global_feat: torch.Tensor,
                obj_point_feat: torch.Tensor | None = None,
                obj_point_xyz: torch.Tensor | None = None,
                slot_labels: torch.Tensor | None = None,
                slot_geo_feat: torch.Tensor | None = None,
                obj_pc: torch.Tensor | None = None,
                n_steps: int = 20):
        cond = self.encode_condition(
            tokens, mask, obj_global_feat,
            obj_point_feat=obj_point_feat,
            obj_point_xyz=obj_point_xyz,
            slot_labels=slot_labels,
            slot_geo_feat=slot_geo_feat,
            obj_pc=obj_pc,
        )
        return self.sample(cond, n_steps=n_steps)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==============================================================================
# GrabNet-style geometric auxiliary loss for flow matching training
# ==============================================================================

def compute_geo_loss(
    pred: PoseDecoderOutput,
    gt_verts: torch.Tensor,
    obj_pc: torch.Tensor,
    mano_fn,
    obj_normals: torch.Tensor | None = None,
    w_vert: float = 10.0,
    w_dist_h: float = 5.0,
    w_dist_o: float = 5.0,
    w_pen: float = 10.0,
):
    """
    GrabNet-inspired geometric supervision on denoised samples.

    Call this on one-step denoised predictions at high t (e.g. t > 0.7)
    to provide geometric grounding without slowing down every step.

    Args:
        pred: PoseDecoderOutput from model.unpack_data()
        gt_verts: (B, 778, 3) ground-truth hand vertices
        obj_pc: (B, N, 3) object point cloud
        mano_fn: callable(pose, trans, shape) -> (verts, joints)
        obj_normals: (B, N, 3) optional, for signed distance
        w_vert, w_dist_h, w_dist_o, w_pen: loss weights

    Returns:
        loss: scalar
        metrics: dict
    """
    from models.refine import point2point_signed

    def compute_vertex_normals(verts: torch.Tensor, faces) -> torch.Tensor | None:
        if faces is None:
            return None
        if not torch.is_tensor(faces):
            faces_t = torch.as_tensor(faces, device=verts.device, dtype=torch.long)
        else:
            faces_t = faces.to(device=verts.device, dtype=torch.long)
        if faces_t.numel() == 0:
            return None

        v0 = verts[:, faces_t[:, 0], :]
        v1 = verts[:, faces_t[:, 1], :]
        v2 = verts[:, faces_t[:, 2], :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)

        normals = torch.zeros_like(verts)
        for corner in range(3):
            idx = faces_t[:, corner].view(1, -1, 1).expand(verts.shape[0], -1, 3)
            normals.scatter_add_(1, idx, face_normals)
        return F.normalize(normals, dim=-1, eps=1e-6)

    pred_verts, _ = mano_fn(pred.pose, pred.trans, pred.shape)
    mano_faces = getattr(mano_fn, "faces", None)
    pred_hand_normals = compute_vertex_normals(pred_verts, mano_faces)
    gt_hand_normals = compute_vertex_normals(gt_verts, mano_faces)

    # Vertex reconstruction loss (weighted L1, like GrabNet loss_mesh_rec_w)
    loss_vert = F.l1_loss(pred_verts, gt_verts)

    # Signed hand/object distances. With hand and object normals we recover the
    # GrabNet-style signed object-to-hand term instead of an unsigned proxy.
    if obj_normals is not None:
        o2h_signed, h2o_signed, _ = point2point_signed(
            pred_verts, obj_pc,
            x_normals=pred_hand_normals,
            y_normals=obj_normals,
        )
        o2h_gt, h2o_gt, _ = point2point_signed(
            gt_verts, obj_pc,
            x_normals=gt_hand_normals,
            y_normals=obj_normals,
        )
    else:
        o2h_signed, h2o_signed, _ = point2point_signed(pred_verts, obj_pc)
        o2h_gt, h2o_gt, _ = point2point_signed(gt_verts, obj_pc)

    # h2o distance matching (like GrabNet loss_dist_h)
    loss_dist_h = F.l1_loss(h2o_signed.abs(), h2o_gt.abs())

    # o2h distance matching (like GrabNet loss_dist_o)
    loss_dist_o = F.l1_loss(o2h_signed, o2h_gt)

    # Penetration penalty: penalize negative signed distance (hand inside object)
    if obj_normals is not None:
        pen_depth = torch.relu(-h2o_signed)
        loss_pen = pen_depth.mean()
    else:
        loss_pen = pred_verts.new_zeros(1).squeeze()

    loss = (
        w_vert * loss_vert
        + w_dist_h * loss_dist_h
        + w_dist_o * loss_dist_o
        + w_pen * loss_pen
    )

    return loss, {
        "geo_loss": loss.item(),
        "geo_vert": loss_vert.item(),
        "geo_dist_h": loss_dist_h.item(),
        "geo_dist_o": loss_dist_o.item(),
        "geo_pen": loss_pen.item(),
    }


# ==============================================================================
# Presets & builders
# ==============================================================================

_PRESETS = {
    "small": dict(hidden_dim=256, num_layers=2, num_heads=8, num_flow_blocks=6),
    "base": dict(hidden_dim=512, num_layers=3, num_heads=8, num_flow_blocks=8),
    "large": dict(hidden_dim=768, num_layers=4, num_heads=12, num_flow_blocks=10),
}


def build_flow_model(config: str = "base", **kw):
    params = _PRESETS[config].copy()
    params.update(kw)
    return PoseFlowModel(**params)


# Backward-compatible alias for train_refine.py
build_pose_model = build_flow_model
