import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from models.object_normalization import normalize_object_points
from utils.pose_utils import rot6d_to_aa


NUM_HAND_PARTS = 6
NUM_JOINTS = 16
AA_POSE_DIM = 48
ROT6D_POSE_DIM = 96
TRANS_DIM = 3
NUM_SLOTS = 4

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
    pose: torch.Tensor
    trans: torch.Tensor
    shape: torch.Tensor


@dataclass
class PoseDecoderTrainOutput:
    pred_pose: torch.Tensor
    pred_trans: torch.Tensor
    pred_shape: torch.Tensor
    loss_rot: torch.Tensor
    loss_trans: torch.Tensor
    loss_shape: torch.Tensor


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
      - original dense slot labels provide 4 slot summary tokens
      - slot labels are no longer forced to align with SA2 points one-by-one
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

        # SA2 geometry tokens stay point-wise.
        spatial_in_dim = obj_point_dim + obj_xyz_dim
        self.use_spatial = spatial_in_dim > 0
        if self.use_spatial:
            self.spatial_proj = nn.Sequential(
                nn.Linear(spatial_in_dim, D), nn.SiLU(), nn.Linear(D, D)
            )

        # Dense slot labels are summarized into one token per slot.
        self.slot_token_emb_dim = int(slot_point_emb_dim)
        self.slot_token_emb = nn.Embedding(num_slots, D)
        self.slot_stat_dim = obj_xyz_dim * 2 + 2  # centroid, spread, ratio, present
        self.slot_stat_proj = nn.Sequential(
            nn.Linear(self.slot_stat_dim, D),
            nn.SiLU(),
            nn.Linear(D, D),
        )

        self.blocks = nn.ModuleList([
            ConditionBlock(D, num_heads, dropout=dropout, use_spatial=self.use_spatial)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(D)

        self._build_token_lookup(vocab_size, num_hand_parts, num_slots)
        self._init_weights()

    def _build_token_lookup(self, V: int, H: int, S: int):
        # semantic tokens are assumed to be [0, H*S)
        # special / invalid ids map to padding buckets
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
        nn.init.normal_(self.slot_token_emb.weight, std=0.02)

    def _build_slot_tokens(
        self,
        slot_labels: torch.Tensor | None,
        obj_pc: torch.Tensor | None,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if obj_pc is None:
            batch_size = slot_labels.shape[0] if slot_labels is not None else 1
            xyz = torch.zeros(
                batch_size, 1, 3, device=device, dtype=self.slot_token_emb.weight.dtype
            )
            labels = None
            obj_scale = torch.ones(batch_size, 1, device=device, dtype=xyz.dtype)
        else:
            xyz = obj_pc.to(device=device, dtype=self.slot_token_emb.weight.dtype)
            xyz, obj_scale = normalize_object_points(xyz)
            batch_size = xyz.shape[0]
            labels = slot_labels.to(device=device, dtype=torch.long) if slot_labels is not None else None

        slot_ids = torch.arange(self.num_slots, device=device, dtype=torch.long)
        slot_base = self.slot_token_emb(slot_ids).unsqueeze(0).expand(batch_size, -1, -1)

        stats = torch.zeros(
            batch_size, self.num_slots, self.slot_stat_dim,
            device=device, dtype=slot_base.dtype,
        )
        slot_centroids = torch.zeros(
            batch_size, self.num_slots, 3,
            device=device, dtype=slot_base.dtype,
        )
        slot_present = torch.zeros(
            batch_size, self.num_slots,
            device=device, dtype=slot_base.dtype,
        )
        if labels is None or obj_pc is None:
            return slot_base + self.slot_stat_proj(stats), slot_centroids, slot_present, obj_scale

        valid = (labels >= 0) & (labels < self.num_slots)
        slot_masks = torch.stack([(labels == s) & valid for s in range(self.num_slots)], dim=-1)
        slot_masks = slot_masks.to(dtype=xyz.dtype)

        counts = slot_masks.sum(dim=1)  # (B, S)
        denom = counts.clamp(min=1.0).unsqueeze(-1)

        centroid = torch.einsum("bns,bnd->bsd", slot_masks, xyz) / denom
        diff = (xyz.unsqueeze(2) - centroid.unsqueeze(1)).abs()
        spread = (diff * slot_masks.unsqueeze(-1)).sum(dim=1) / denom

        num_points = max(xyz.shape[1], 1)
        ratio = counts / float(num_points)
        present = (counts > 0).to(dtype=xyz.dtype)
        stats = torch.cat(
            [centroid, spread, ratio.unsqueeze(-1), present.unsqueeze(-1)],
            dim=-1,
        )
        return slot_base + self.slot_stat_proj(stats), centroid, present, obj_scale

    def _build_trans_anchor(
        self,
        token_slots: torch.Tensor,
        token_mask: torch.Tensor,
        slot_centroids: torch.Tensor,
        slot_present: torch.Tensor,
    ) -> torch.Tensor:
        valid_slots = (
            token_mask
            & (token_slots >= 0)
            & (token_slots < self.num_slots)
        )
        slot_ids = token_slots.clamp(min=0, max=self.num_slots - 1)
        slot_weights = F.one_hot(slot_ids, num_classes=self.num_slots).to(slot_centroids.dtype)
        slot_weights = slot_weights * valid_slots.unsqueeze(-1)
        slot_weights = slot_weights.sum(dim=1)
        slot_weights = slot_weights * slot_present

        fallback = slot_present
        has_weight = slot_weights.sum(dim=-1, keepdim=True) > 0
        slot_weights = torch.where(has_weight, slot_weights, fallback)
        denom = slot_weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return torch.einsum("bs,bsd->bd", slot_weights, slot_centroids) / denom

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

        geometry_tokens = None
        spatial_summary = c_global
        if self.use_spatial and (obj_point_feat is not None or obj_point_xyz is not None):
            if obj_point_feat is None:
                raise ValueError("obj_point_feat must be provided when using spatial conditioning.")
            if obj_point_xyz is None:
                obj_point_xyz = obj_point_feat.new_zeros(
                    obj_point_feat.shape[0], obj_point_feat.shape[1], 3
                )

            spatial_in = torch.cat([obj_point_feat, obj_point_xyz], dim=-1)
            geometry_tokens = self.spatial_proj(spatial_in)

        slot_tokens, slot_centroids, slot_present, obj_scale = self._build_slot_tokens(
            slot_labels, obj_pc, obj_global_feat.device
        )
        trans_anchor = self._build_trans_anchor(sl, token_mask, slot_centroids, slot_present)

        cond_kv_parts = []
        summary_parts = []
        if geometry_tokens is not None:
            cond_kv_parts.append(geometry_tokens)
            summary_parts.append(geometry_tokens.mean(dim=1))
        if slot_tokens is not None:
            cond_kv_parts.append(slot_tokens)
            summary_parts.append(slot_tokens.mean(dim=1))
        spatial_kv = torch.cat(cond_kv_parts, dim=1) if cond_kv_parts else None
        if summary_parts:
            spatial_summary = torch.stack(summary_parts, dim=1).mean(dim=1)

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
            "spatial_summary": spatial_summary,
            "slot_tokens": slot_tokens,
            "slot_centroids": slot_centroids,
            "slot_present": slot_present,
            "trans_anchor": trans_anchor,
            "obj_scale": obj_scale,
            "obj_point_xyz": obj_point_xyz,
        }


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

        self.part_attn_bias = nn.Parameter(torch.zeros(num_heads))
        self.slot_attn_bias = nn.Parameter(torch.zeros(num_heads, num_slots))

        self.norm_ff = nn.LayerNorm(D)
        self.ffn = nn.Sequential(
            nn.Linear(D, D * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(D * 4, D), nn.Dropout(dropout),
        )

    def forward(self, h: torch.Tensor, cond_vec: torch.Tensor,
                token_kv: torch.Tensor, token_mask: torch.Tensor,
                part_align_mask: torch.Tensor, token_slots: torch.Tensor) -> torch.Tensor:
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

        h = h + self.ffn(self.norm_ff(h))
        return h


class PoseHead(nn.Module):
    """Predict 6D rotation for 16 joints."""

    def __init__(self, hidden_dim: int = 512, num_blocks: int = 8,
                 num_heads: int = 8, num_slots: int = NUM_SLOTS,
                 dropout: float = 0.1):
        super().__init__()
        D = hidden_dim

        self.joint_queries = nn.Parameter(torch.randn(NUM_JOINTS, D) * 0.02)
        self.joint_pos_emb = nn.Embedding(NUM_JOINTS, D)
        nn.init.normal_(self.joint_pos_emb.weight, std=0.02)

        self.register_buffer("joint_parts", torch.tensor(JOINT_TO_PART, dtype=torch.long))

        self.cond_proj = nn.Sequential(
            nn.Linear(D, D), nn.SiLU(), nn.Linear(D, D)
        )

        self.blocks = nn.ModuleList([
            PoseHeadBlock(D, D, num_heads=num_heads, num_slots=num_slots, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.out_norm = nn.LayerNorm(D)

        self.wrist_head = nn.Linear(D, 6)
        self.finger_head = nn.Linear(D, 6)
        nn.init.normal_(self.wrist_head.weight, std=0.01)
        nn.init.zeros_(self.wrist_head.bias)
        nn.init.normal_(self.finger_head.weight, std=0.01)
        nn.init.zeros_(self.finger_head.bias)

    def build_input(self, cond: dict) -> dict:
        B = cond["c_global"].shape[0]
        device = cond["c_global"].device
        return {
            "token_kv": cond["token_feats"],
            "token_mask": cond["token_mask"],
            "token_parts": cond["token_parts"],
            "token_slots": cond["token_slots"],
            "cond_vec": self.cond_proj(cond["c_global"]),
            "device": device,
            "B": B,
        }

    def forward(self, token_kv: torch.Tensor, token_mask: torch.Tensor,
                token_parts: torch.Tensor, token_slots: torch.Tensor,
                cond_vec: torch.Tensor, device: torch.device, B: int) -> torch.Tensor:
        h = self.joint_queries.unsqueeze(0).expand(B, -1, -1)
        h = h + self.joint_pos_emb(torch.arange(NUM_JOINTS, device=device)).unsqueeze(0)

        part_align_mask = (
            (self.joint_parts.view(1, NUM_JOINTS, 1) == token_parts.unsqueeze(1))
            & token_mask.unsqueeze(1)
        )

        for blk in self.blocks:
            h = blk(h, cond_vec, token_kv, token_mask, part_align_mask, token_slots)
        h = self.out_norm(h)

        wrist_rot = self.wrist_head(h[:, 0:1, :])
        finger_rot = self.finger_head(h[:, 1:, :])
        rot_6d = torch.cat([wrist_rot, finger_rot], dim=1)
        return rot_6d.reshape(B, ROT6D_POSE_DIM)


class TransHead(nn.Module):
    """Predict normalized residual translation on top of an object anchor."""

    def __init__(self, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        D = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(D * 3 + TRANS_DIM, D), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(D, D // 2), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(D // 2, TRANS_DIM),
        )
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def build_input(self, cond: dict) -> torch.Tensor:
        token_feats = cond["token_feats"]
        token_mask = cond["token_mask"]
        mask_f = token_mask.float().unsqueeze(-1)
        denom = mask_f.sum(1).clamp(min=1.0)
        token_summary = (token_feats * mask_f).sum(1) / denom
        spatial_summary = cond.get("spatial_summary", cond["c_global"])
        trans_anchor = cond.get("trans_anchor")
        if trans_anchor is None:
            trans_anchor = cond["c_global"].new_zeros(cond["c_global"].shape[0], TRANS_DIM)
        return torch.cat([cond["c_global"], token_summary, spatial_summary, trans_anchor], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ShapeHead(nn.Module):
    """Predict MANO shape from global and contact-conditioned summaries."""

    def __init__(self, hidden_dim: int = 512, n_betas: int = 10, dropout: float = 0.1):
        super().__init__()
        D = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(D * 2, D), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(D, D // 2), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(D // 2, n_betas),
        )

    def build_input(self, cond: dict) -> torch.Tensor:
        token_feats = cond["token_feats"]
        token_mask = cond["token_mask"]
        mask_f = token_mask.float().unsqueeze(-1)
        denom = mask_f.sum(1).clamp(min=1.0)
        token_summary = (token_feats * mask_f).sum(1) / denom
        return torch.cat([cond["c_global"], token_summary], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PoseDecoderModel(nn.Module):
    """
    Pose decoder aligned with simplified (hand_part, slot) tokens.

    Key updates:
      - 128 SA2 points stay as geometry conditioning
      - dense slot labels are summarized into slot tokens from the original object points
      - slot labels no longer need point-wise alignment with SA2 samples
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
        n_betas: int = 10,
        dropout: float = 0.1,
        num_pose_blocks: int = 8,
        slot_point_emb_dim: int = 64,
        num_contact_types: int | None = None,
    ):
        super().__init__()
        del num_contact_types  # backward compatibility only
        self.num_slots = num_slots

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

        self.pose_head = PoseHead(
            hidden_dim=hidden_dim,
            num_blocks=num_pose_blocks,
            num_heads=num_heads,
            num_slots=num_slots,
            dropout=dropout,
        )
        self.trans_head = TransHead(hidden_dim=hidden_dim, dropout=dropout)
        self.shape_head = ShapeHead(hidden_dim=hidden_dim, n_betas=n_betas, dropout=dropout)

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

    def predict_pose(self, cond: dict) -> torch.Tensor:
        inputs = self.pose_head.build_input(cond)
        rot_6d = self.pose_head(**inputs)
        return rot6d_to_aa(rot_6d)

    def predict_trans(self, cond: dict) -> torch.Tensor:
        x = self.trans_head.build_input(cond)
        delta = self.trans_head(x)
        anchor = cond.get("trans_anchor")
        if anchor is None:
            anchor = delta.new_zeros(delta.shape)
        pred_trans_norm = anchor + delta
        obj_scale = cond.get("obj_scale")
        if obj_scale is None:
            return pred_trans_norm
        return pred_trans_norm * obj_scale

    def predict_shape(self, cond: dict) -> torch.Tensor:
        x = self.shape_head.build_input(cond)
        return self.shape_head(x)

    def _predict_all(self, cond: dict):
        pose_inputs = self.pose_head.build_input(cond)
        rot_6d = self.pose_head(**pose_inputs)

        trans_x = self.trans_head.build_input(cond)
        trans = self.trans_head(trans_x)

        shape_x = self.shape_head.build_input(cond)
        shape = self.shape_head(shape_x)
        return rot_6d, trans, shape

    def forward_train(self, gt_pose_aa: torch.Tensor, gt_trans: torch.Tensor, cond: dict):
        rot_6d, pred_trans, _ = self._predict_all(cond)
        pred_pose_aa = rot6d_to_aa(rot_6d)

        from models.loss import geodesic_loss
        loss_rot = geodesic_loss(
            pred_pose_aa.view(-1, NUM_JOINTS, 3),
            gt_pose_aa.view(-1, NUM_JOINTS, 3),
        )
        loss_trans = torch.sqrt(((pred_trans - gt_trans) ** 2).mean().clamp_min(1e-12))

        return loss_rot, loss_trans, {
            "loss_rot": loss_rot.item(),
            "loss_trans": loss_trans.item(),
            "pred_pose": pred_pose_aa,
            "pred_trans": pred_trans,
        }

    def forward_train_all(self, gt_pose_aa: torch.Tensor, gt_trans: torch.Tensor,
                          gt_shape: torch.Tensor, cond: dict) -> PoseDecoderTrainOutput:
        rot_6d, pred_trans, pred_shape = self._predict_all(cond)
        pred_pose_aa = rot6d_to_aa(rot_6d)

        from models.loss import geodesic_loss
        loss_rot = geodesic_loss(
            pred_pose_aa.view(-1, NUM_JOINTS, 3),
            gt_pose_aa.view(-1, NUM_JOINTS, 3),
        )
        loss_trans = torch.sqrt(((pred_trans - gt_trans) ** 2).mean().clamp_min(1e-12))
        loss_shape = F.mse_loss(pred_shape, gt_shape)

        return PoseDecoderTrainOutput(
            pred_pose=pred_pose_aa,
            pred_trans=pred_trans,
            pred_shape=pred_shape,
            loss_rot=loss_rot,
            loss_trans=loss_trans,
            loss_shape=loss_shape,
        )

    @torch.no_grad()
    def sample(self, cond: dict) -> PoseDecoderOutput:
        rot_6d, trans, shape = self._predict_all(cond)
        pose_aa = rot6d_to_aa(rot_6d)
        return PoseDecoderOutput(pose=pose_aa, trans=trans, shape=shape)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor,
                obj_global_feat: torch.Tensor,
                obj_point_feat: torch.Tensor | None = None,
                obj_point_xyz: torch.Tensor | None = None,
                slot_labels: torch.Tensor | None = None,
                slot_geo_feat: torch.Tensor | None = None,
                obj_pc: torch.Tensor | None = None):
        cond = self.encode_condition(
            tokens,
            mask,
            obj_global_feat,
            obj_point_feat=obj_point_feat,
            obj_point_xyz=obj_point_xyz,
            slot_labels=slot_labels,
            slot_geo_feat=slot_geo_feat,
            obj_pc=obj_pc,
        )
        return self.sample(cond)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


_PRESETS = {
    "small": dict(hidden_dim=256, num_layers=2, num_heads=8, num_pose_blocks=6),
    "base": dict(hidden_dim=512, num_layers=3, num_heads=8, num_pose_blocks=8),
    "large": dict(hidden_dim=768, num_layers=4, num_heads=12, num_pose_blocks=10),
}


def build_pose_model(config: str = "base", **kw):
    params = _PRESETS[config].copy()
    params.update(kw)
    return PoseDecoderModel(**params)
