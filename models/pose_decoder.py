"""
pose_decoder.py — Slot-Grounded BPS Pose Decoder
==================================================
GrabNet-style CVAE (CoarseNet) with slot-grounded BPS conditioning.

Architecture:
  SlotModulatedBPS: tokens + bps_dists + bps_slot_labels → slot-aware BPS feature
  CoarseNet:  CVAE, slot_bps_feat → rot6d(96) + trans(3)
  SlotGrounder: (inference) predict bps_slot_labels from bps_dists + tokens

RefineNet lives in models/refine.py (joint-trained, see train_decoder.py).

No shape prediction — uses mean hand shape (following GrabNet).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import chamfer_distance as chd
from utils.pose_utils import rot6d_to_aa, aa_to_rot6d


NUM_HAND_PARTS = 6
NUM_JOINTS = 16
AA_POSE_DIM = 48
ROT6D_POSE_DIM = 96
TRANS_DIM = 3
NUM_SLOTS = 4
DEFAULT_N_BETAS = 10

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
    trans: torch.Tensor   # (B, 3)
    shape: torch.Tensor   # (B, 10) betas — zeros (mean shape)


# ==============================================================================
# Signed Distance (from GrabNet)
# ==============================================================================

def point2point_signed(x, y, x_normals=None, y_normals=None):
    """
    Signed distance between two point clouds.

    Args:
        x: (N, P1, D) — e.g. hand vertices
        y: (N, P2, D) — e.g. object vertices
        x_normals: Optional (N, P1, D)
        y_normals: Optional (N, P2, D)

    Returns:
        y2x_signed: (N, P2)
        x2y_signed: (N, P1) — h2o_dist
        yidx_near:  (N, P2)
    """
    N, P1, D = x.shape
    P2 = y.shape[1]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")

    ch_dist = chd.ChamferDistance()
    x_near, y_near, xidx_near, yidx_near = ch_dist(x, y)

    xidx_near_expanded = xidx_near.view(N, P1, 1).expand(N, P1, D).to(torch.long)
    x_near = y.gather(1, xidx_near_expanded)

    yidx_near_expanded = yidx_near.view(N, P2, 1).expand(N, P2, D).to(torch.long)
    y_near = x.gather(1, yidx_near_expanded)

    x2y = x - x_near
    y2x = y - y_near

    if x_normals is not None:
        y_nn = x_normals.gather(1, yidx_near_expanded)
        in_out = torch.bmm(y_nn.reshape(-1, 1, 3), y2x.reshape(-1, 3, 1)).reshape(N, -1).sign()
        y2x_signed = y2x.norm(dim=2) * in_out
    else:
        y2x_signed = y2x.norm(dim=2)

    if y_normals is not None:
        x_nn = y_normals.gather(1, xidx_near_expanded)
        in_out_x = torch.bmm(x_nn.reshape(-1, 1, 3), x2y.reshape(-1, 3, 1)).reshape(N, -1).sign()
        x2y_signed = x2y.norm(dim=2) * in_out_x
    else:
        x2y_signed = x2y.norm(dim=2)

    return y2x_signed, x2y_signed, yidx_near


# ==============================================================================
# SlotModulatedBPS — Slot-aware BPS feature encoding
# ==============================================================================

class SlotModulatedBPS(nn.Module):
    """Slot-aware modulation of BPS distances.

    Each BPS basis point's distance is modulated by learnable per-slot
    affine parameters, weighted by token-derived slot activation.

    Output: (B, out_dim) feature vector that replaces bps + token_cond
    in CoarseNet.

    Design:
      1. Per-slot affine: dist * scale[s] + bias[s]  per basis point
      2. Token → per-slot activation: part-slot attention weights
      3. Concatenate modulated BPS(4096) + token summary → project to out_dim
    """

    def __init__(
        self,
        n_bps: int = 4096,
        num_slots: int = NUM_SLOTS,
        num_hand_parts: int = NUM_HAND_PARTS,
        vocab_size: int = 26,
        token_embed_dim: int = 64,
        out_dim: int = 4096,
    ):
        super().__init__()
        self.n_bps = n_bps
        self.num_slots = num_slots
        self.num_hand_parts = num_hand_parts
        self.out_dim = out_dim
        n_channels = num_slots + 1  # +1 for unassigned (-1)

        # Per-slot affine modulation
        self.slot_scale = nn.Parameter(torch.ones(n_channels))
        self.slot_bias = nn.Parameter(torch.zeros(n_channels))

        # Part-slot interaction: modulates how much each (part, slot) pair
        # contributes to activation.  Shape (num_hand_parts, num_slots).
        self.part_slot_attn = nn.Parameter(torch.ones(num_hand_parts, num_slots))
        nn.init.xavier_uniform_(self.part_slot_attn)

        # Token summary encoder: small token embedding → pooled → projection
        self.hp_emb = nn.Embedding(num_hand_parts + 1, token_embed_dim)
        self.sl_emb = nn.Embedding(num_slots + 1, token_embed_dim)
        nn.init.normal_(self.hp_emb.weight, std=0.02)
        nn.init.normal_(self.sl_emb.weight, std=0.02)

        # Token → (hand_part, slot) lookup
        hp_ids = torch.full((vocab_size,), num_hand_parts, dtype=torch.long)
        sl_ids = torch.full((vocab_size,), num_slots, dtype=torch.long)
        n_semantic = min(vocab_size, num_hand_parts * num_slots)
        for tok in range(n_semantic):
            hp_ids[tok] = tok // num_slots
            sl_ids[tok] = tok % num_slots
        self.register_buffer("_hp_ids", hp_ids)
        self.register_buffer("_sl_ids", sl_ids)

        # Output projection: modulated_bps(4096) + token_summary(token_embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(n_bps + token_embed_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def _compute_slot_activation(
        self, tokens: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-slot activation from token sequence.

        Returns:
            (B, num_slots) activation weights per slot.
        """
        B, L = tokens.shape
        device = tokens.device
        mask = ensure_nonempty_token_mask(mask)

        # Accumulate part_slot_attn for active tokens
        activation = torch.zeros(B, self.num_slots, device=device)
        hp = self._hp_ids[tokens]  # (B, L)
        sl = self._sl_ids[tokens]  # (B, L)

        for s in range(self.num_slots):
            # For each slot s, sum part_slot_attn[hp, s] for valid tokens with slot==s
            slot_match = (sl == s) & mask  # (B, L) bool
            if slot_match.any():
                hp_for_slot = hp.clamp(max=self.num_hand_parts - 1)
                weights = self.part_slot_attn[:, s][hp_for_slot]  # (B, L)
                activation[:, s] = (weights * slot_match.float()).sum(dim=1)

        return activation

    def _encode_token_summary(
        self, tokens: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode tokens to a summary vector via masked mean pooling.

        Returns:
            (B, token_embed_dim)
        """
        mask = ensure_nonempty_token_mask(mask)
        hp = self._hp_ids[tokens]
        sl = self._sl_ids[tokens]
        h = self.hp_emb(hp) + self.sl_emb(sl)  # (B, L, embed_dim)

        mask_f = mask.float().unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        return (h * mask_f).sum(dim=1) / denom  # (B, embed_dim)

    def forward(
        self,
        bps_dists: torch.Tensor,        # (B, K) raw BPS distances
        bps_slot_labels: torch.Tensor,   # (B, K) int, -1~3
        tokens: torch.Tensor,           # (B, L) token ids
        token_mask: torch.Tensor,       # (B, L) bool
    ) -> torch.Tensor:
        """Compute slot-modulated BPS feature.

        Returns:
            (B, out_dim) slot-aware BPS feature vector.
        """
        B, K = bps_dists.shape
        device = bps_dists.device

        # 1. Per-slot activation from tokens
        slot_act = self._compute_slot_activation(tokens, token_mask)  # (B, num_slots)

        # 2. Per-slot affine modulation of BPS distances
        # Map -1 → index num_slots (the "unassigned" channel)
        label_idx = bps_slot_labels.clone()
        label_idx[label_idx < 0] = self.num_slots  # (B, K) in [0, num_slots]

        scales = self.slot_scale[label_idx]  # (B, K)
        biases = self.slot_bias[label_idx]   # (B, K)

        modulated = bps_dists * scales + biases  # (B, K)

        # 3. Apply slot activation: amplify active slots, suppress inactive
        # For each basis point, if its slot is active, scale up; else scale down.
        # Unassigned points (-1) are not modulated by activation.
        for s in range(self.num_slots):
            s_mask = (bps_slot_labels == s).float()  # (B, K)
            act = slot_act[:, s:s+1]                  # (B, 1)
            # Soft modulation: act clamped to [0.1, 2.0] to avoid complete zeroing
            act_clamped = act.clamp(min=0.1, max=2.0)
            modulated = modulated + s_mask * bps_dists * (act_clamped - 1.0) * 0.5

        # 4. Token summary
        tok_summary = self._encode_token_summary(tokens, token_mask)  # (B, embed_dim)

        # 5. Concatenate and project
        feat = torch.cat([modulated, tok_summary], dim=1)  # (B, K + embed_dim)
        return self.proj(feat)  # (B, out_dim)


# ==============================================================================
# SlotGrounder — Predict BPS slot labels at inference time
# ==============================================================================

class SlotGrounder(nn.Module):
    """Predict bps_slot_labels from raw bps_dists + token embedding.

    Trained with cross-entropy on GT bps_slot_labels during decoder training.
    Used at inference to replace GT slot labels.

    Architecture: lightweight MLP operating on per-point features.
    """

    def __init__(
        self,
        n_bps: int = 4096,
        num_slots: int = NUM_SLOTS,
        token_cond_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.n_bps = n_bps
        self.num_slots = num_slots
        self.n_classes = num_slots + 1  # +1 for "unassigned"

        # Global context from BPS distances + token
        self.context_net = nn.Sequential(
            nn.Linear(n_bps + token_cond_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
        )

        # Per-point classification head
        # Input: per-point dist(1) + global context(hidden_dim)
        self.point_head = nn.Sequential(
            nn.Linear(1 + hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, self.n_classes),
        )

    def forward(
        self,
        bps_dists: torch.Tensor,      # (B, K)
        token_summary: torch.Tensor,   # (B, token_cond_dim)
    ) -> torch.Tensor:
        """Predict per-basis-point slot logits.

        Returns:
            logits: (B, K, n_classes)
        """
        B, K = bps_dists.shape

        # Global context
        ctx_in = torch.cat([bps_dists, token_summary], dim=1)  # (B, K + D)
        ctx = self.context_net(ctx_in)  # (B, hidden)

        # Broadcast to each point
        ctx_expanded = ctx.unsqueeze(1).expand(B, K, -1)       # (B, K, hidden)
        per_point = bps_dists.unsqueeze(-1)                     # (B, K, 1)
        point_feat = torch.cat([per_point, ctx_expanded], dim=2)  # (B, K, 1+hidden)

        logits = self.point_head(point_feat)  # (B, K, n_classes)
        return logits

    def predict(
        self,
        bps_dists: torch.Tensor,
        token_summary: torch.Tensor,
    ) -> torch.Tensor:
        """Predict slot labels (argmax).

        Returns:
            (B, K) long — predicted slot labels (0~num_slots-1 or num_slots=unassigned)
        """
        logits = self.forward(bps_dists, token_summary)
        pred = logits.argmax(dim=-1)  # (B, K)
        # Map "unassigned" class back to -1
        pred[pred == self.num_slots] = -1
        return pred

    def compute_loss(
        self,
        logits: torch.Tensor,    # (B, K, n_classes)
        gt_labels: torch.Tensor,  # (B, K) int, -1~num_slots-1
    ) -> torch.Tensor:
        """Cross-entropy loss. Maps -1 labels to the "unassigned" class."""
        target = gt_labels.clone().long()
        target[target < 0] = self.num_slots
        B, K, C = logits.shape
        return F.cross_entropy(
            logits.reshape(B * K, C),
            target.reshape(B * K),
        )


# ==============================================================================
# ResBlock (from GrabNet)
# ==============================================================================

class ResBlock(nn.Module):
    def __init__(self, Fin: int, Fout: int, n_neurons: int = 256):
        super().__init__()
        self.Fin = Fin
        self.Fout = Fout

        self.fc1 = nn.Linear(Fin, n_neurons)
        self.bn1 = nn.BatchNorm1d(n_neurons)
        self.fc2 = nn.Linear(n_neurons, Fout)
        self.bn2 = nn.BatchNorm1d(Fout)

        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor, final_nl: bool = True) -> torch.Tensor:
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))
        Xout = self.ll(self.bn1(self.fc1(x)))
        Xout = self.bn2(self.fc2(Xout))
        Xout = Xin + Xout
        return self.ll(Xout) if final_nl else Xout


# ==============================================================================
# CoarseNet — CVAE (slot-grounded BPS)
# ==============================================================================

class CoarseNet(nn.Module):
    """CVAE for coarse pose generation.

    Encoder sees raw BPS + GT pose (for VAE posterior).
    Decoder sees z + slot_bps_feat (slot-grounded BPS feature from SlotModulatedBPS).
    """

    def __init__(
        self,
        n_neurons: int = 512,
        latentD: int = 16,
        in_bps: int = 4096,
        in_pose: int = 12,
    ):
        super().__init__()
        self.latentD = latentD

        # ---- Encoder: raw bps + GT pose → z ----
        enc_in = in_bps + in_pose
        self.enc_bn1 = nn.BatchNorm1d(enc_in)
        self.enc_rb1 = ResBlock(enc_in, n_neurons)
        self.enc_rb2 = ResBlock(n_neurons + enc_in, n_neurons)
        self.enc_mu = nn.Linear(n_neurons, latentD)
        self.enc_var = nn.Linear(n_neurons, latentD)
        self.do = nn.Dropout(p=0.1)

        # ---- Decoder: z + slot_bps_feat → rot6d + trans ----
        dec_in = latentD + in_bps  # slot_bps_feat has dim in_bps
        self.dec_bn1 = nn.BatchNorm1d(in_bps)
        self.dec_rb1 = ResBlock(dec_in, n_neurons)
        self.dec_rb2 = ResBlock(n_neurons + dec_in, n_neurons)
        self.dec_pose = nn.Linear(n_neurons, NUM_JOINTS * 6)
        self.dec_trans = nn.Linear(n_neurons, 3)

    def encode(
        self,
        bps_raw: torch.Tensor,
        trans_rhand: torch.Tensor,
        global_orient_rhand_rotmat: torch.Tensor,
    ) -> torch.distributions.Normal:
        """Encode GT into posterior q(z|x, o). Uses raw BPS (not slot-modulated)."""
        bs = bps_raw.shape[0]
        X = torch.cat([
            bps_raw,
            global_orient_rhand_rotmat.reshape(bs, -1),
            trans_rhand,
        ], dim=1)
        X0 = self.enc_bn1(X)
        X = self.enc_rb1(X0, True)
        X = self.enc_rb2(torch.cat([X0, X], dim=1), True)
        return torch.distributions.Normal(
            self.enc_mu(X), F.softplus(self.enc_var(X))
        )

    def decode(self, z: torch.Tensor, bps_feat: torch.Tensor) -> dict:
        """Decode z + bps_feat → pose + trans.

        Args:
            z: (B, latentD)
            bps_feat: (B, in_bps) — slot_bps_feat from SlotModulatedBPS
        """
        o_bps = self.dec_bn1(bps_feat)
        X0 = torch.cat([z, o_bps], dim=1)
        X = self.dec_rb1(X0, True)
        X = self.dec_rb2(torch.cat([X0, X], dim=1), True)

        pose = self.dec_pose(X)
        trans = self.dec_trans(X)

        results = _parms_decode(pose, trans)
        results["z"] = z
        results["pose_rot6d"] = pose
        return results

    def forward(
        self,
        bps_raw: torch.Tensor,
        trans_rhand: torch.Tensor,
        global_orient_rhand_rotmat: torch.Tensor,
        bps_feat: torch.Tensor,
        **kwargs,
    ) -> dict:
        """Training forward: encode GT → sample z → decode.

        Args:
            bps_raw: (B, 4096) raw BPS distances (for encoder)
            trans_rhand: (B, 3) GT translation
            global_orient_rhand_rotmat: (B, 9) GT global orient
            bps_feat: (B, out_dim) slot-modulated BPS (for decoder)
        """
        z_dist = self.encode(bps_raw, trans_rhand, global_orient_rhand_rotmat)
        z_s = z_dist.rsample()
        results = self.decode(z_s, bps_feat)
        results["mean"] = z_dist.mean
        results["std"] = z_dist.scale
        return results

    @torch.no_grad()
    def sample_poses(
        self,
        bps_feat: torch.Tensor,
        seed: int | None = None,
    ) -> dict:
        """Sample from prior: z ~ N(0, I) → decode."""
        bs = bps_feat.shape[0]
        device = bps_feat.device
        dtype = bps_feat.dtype
        if seed is not None:
            np.random.seed(seed)
        z = torch.randn(bs, self.latentD, device=device, dtype=dtype)
        return self.decode(z, bps_feat)


# ==============================================================================
# Parameter decode: rot6d → axis-angle MANO dict
# ==============================================================================

def _parms_decode(pose_rot6d: torch.Tensor, trans: torch.Tensor) -> dict:
    """Convert rot6d(96) + trans(3) to MANO-compatible parameter dict."""
    bs = trans.shape[0]
    pose_aa = rot6d_to_aa(pose_rot6d.reshape(bs, -1))

    global_orient = pose_aa[:, :3]
    hand_pose = pose_aa[:, 3:]

    return {
        "global_orient": global_orient,
        "hand_pose": hand_pose,
        "transl": trans,
        "fullpose_aa": pose_aa,
    }


# ==============================================================================
# Top-level wrapper
# ==============================================================================

class PoseGrabModel(nn.Module):
    """GrabNet-style coarse pose decoder with slot-grounded BPS.

    Contains:
      - SlotModulatedBPS: tokens + bps_dists + bps_slot_labels → slot-aware BPS
      - SlotGrounder: predict bps_slot_labels at inference (no GT)
      - CoarseNet: CVAE, slot_bps → rot6d + trans
    """

    def __init__(
        self,
        vocab_size: int = 26,
        num_hand_parts: int = NUM_HAND_PARTS,
        num_slots: int = NUM_SLOTS,
        cond_dim: int = 512,
        n_neurons: int = 512,
        latentD: int = 16,
        in_bps: int = 4096,
        n_betas: int = DEFAULT_N_BETAS,
        kl_coef: float = 0.005,
        slot_grounder_hidden: int = 256,
        **kwargs,
    ):
        super().__init__()
        self.n_betas = n_betas
        self.kl_coef = kl_coef
        self.latentD = latentD

        self.slot_bps = SlotModulatedBPS(
            n_bps=in_bps,
            num_slots=num_slots,
            num_hand_parts=num_hand_parts,
            vocab_size=vocab_size,
            token_embed_dim=64,
            out_dim=in_bps,
        )
        self.slot_grounder = SlotGrounder(
            n_bps=in_bps,
            num_slots=num_slots,
            token_cond_dim=64,
            hidden_dim=slot_grounder_hidden,
        )

        self.coarse_net = CoarseNet(
            n_neurons=n_neurons,
            latentD=latentD,
            in_bps=in_bps,
        )

    # ----------------------------------------------------------------
    # Slot-grounded BPS encoding
    # ----------------------------------------------------------------

    def encode_slot_bps(
        self,
        bps_dists: torch.Tensor,
        bps_slot_labels: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute slot-modulated BPS feature. (B, in_bps)"""
        return self.slot_bps(bps_dists, bps_slot_labels, tokens, token_mask)

    def predict_slot_labels(
        self,
        bps_dists: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Inference: predict bps_slot_labels from BPS + tokens.

        Returns:
            (B, K) long, -1~num_slots-1
        """
        tok_summary = self.slot_bps._encode_token_summary(tokens, token_mask)
        return self.slot_grounder.predict(bps_dists, tok_summary)

    # ----------------------------------------------------------------
    # Coarse stage (CVAE)
    # ----------------------------------------------------------------

    def forward_coarse(
        self,
        bps_raw: torch.Tensor,
        slot_bps_feat: torch.Tensor,
        trans_rhand: torch.Tensor,
        global_orient_rhand_rotmat: torch.Tensor,
    ) -> dict:
        """Training forward through CoarseNet.

        Args:
            bps_raw: (B, 4096) raw BPS dists (for encoder)
            slot_bps_feat: (B, 4096) slot-modulated BPS (for decoder)
            trans_rhand: GT translation
            global_orient_rhand_rotmat: GT orient
        """
        return self.coarse_net(
            bps_raw, trans_rhand, global_orient_rhand_rotmat,
            slot_bps_feat,
        )

    def compute_coarse_loss(
        self,
        drec: dict,
        dorig: dict,
        verts_pred: torch.Tensor,
        verts_gt: torch.Tensor,
        verts_object: torch.Tensor,
        rh_faces: torch.Tensor,
        v_weights: torch.Tensor,
        v_weights2: torch.Tensor,
        batch_size: int,
    ) -> tuple:
        """GrabNet-style CoarseNet loss: KL + vertex + edge + dist."""
        device = verts_gt.device
        dtype = verts_gt.dtype

        from pytorch3d.structures import Meshes
        rh_mesh = Meshes(verts=verts_pred, faces=rh_faces).verts_normals_packed().view(-1, 778, 3)
        rh_mesh_gt = Meshes(verts=verts_gt, faces=rh_faces).verts_normals_packed().view(-1, 778, 3)

        o2h_signed, h2o, _ = point2point_signed(verts_pred, verts_object, rh_mesh)
        o2h_signed_gt, h2o_gt, _ = point2point_signed(verts_gt, verts_object, rh_mesh_gt)

        w_dist = (o2h_signed_gt < 0.01) & (o2h_signed_gt > -0.005)
        w_dist_neg = o2h_signed < 0.0
        w = torch.ones(verts_object.shape[0], verts_object.shape[1], device=device)
        w[~w_dist] = 0.1
        w[w_dist_neg] = 1.5

        loss_dist_h = 35 * (1.0 - self.kl_coef) * torch.mean(
            torch.einsum("ij,j->ij", torch.abs(h2o.abs() - h2o_gt.abs()), v_weights2)
        )
        loss_dist_o = 30 * (1.0 - self.kl_coef) * torch.mean(
            torch.einsum("ij,ij->ij", torch.abs(o2h_signed - o2h_signed_gt), w)
        )
        loss_mesh_rec = 35 * (1.0 - self.kl_coef) * torch.mean(
            torch.einsum("ijk,j->ijk", torch.abs(verts_gt - verts_pred), v_weights)
        )

        vpe = dorig.get("vpe", None)
        if vpe is not None:
            edges_pred = verts_pred[:, vpe[:, 0]] - verts_pred[:, vpe[:, 1]]
            edges_gt = verts_gt[:, vpe[:, 0]] - verts_gt[:, vpe[:, 1]]
            loss_edge = 30 * (1.0 - self.kl_coef) * F.l1_loss(edges_pred, edges_gt)
        else:
            loss_edge = torch.tensor(0.0, device=device)

        q_z = torch.distributions.Normal(drec["mean"], drec["std"])
        p_z = torch.distributions.Normal(
            torch.zeros(batch_size, self.latentD, device=device, dtype=dtype),
            torch.ones(batch_size, self.latentD, device=device, dtype=dtype),
        )
        loss_kl = self.kl_coef * torch.mean(
            torch.sum(torch.distributions.kl_divergence(q_z, p_z), dim=1)
        )

        loss_dict = {
            "loss_kl": loss_kl,
            "loss_edge": loss_edge,
            "loss_mesh_rec": loss_mesh_rec,
            "loss_dist_h": loss_dist_h,
            "loss_dist_o": loss_dist_o,
        }
        loss_total = torch.stack(list(loss_dict.values())).sum()
        loss_dict["loss_total"] = loss_total

        return loss_total, loss_dict

    # ----------------------------------------------------------------
    # Sampling (inference)
    # ----------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        bps_dists: torch.Tensor,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        bps_slot_labels: torch.Tensor | None = None,
        n_betas: int | None = None,
        seed: int | None = None,
    ) -> PoseDecoderOutput:
        """Generate coarse pose from BPS + contact tokens.

        Args:
            bps_dists: (B, 4096) raw BPS distances
            tokens: (B, L) contact token ids
            mask: (B, L) token mask
            bps_slot_labels: (B, 4096) slot labels. If None, predicted via SlotGrounder.
            n_betas: number of shape parameters
            seed: random seed for CVAE sampling
        """
        B = bps_dists.shape[0]
        nb = n_betas if n_betas is not None else self.n_betas

        # Get or predict slot labels
        if bps_slot_labels is None:
            bps_slot_labels = self.predict_slot_labels(bps_dists, tokens, mask)

        # Compute slot-modulated BPS
        slot_bps_feat = self.encode_slot_bps(
            bps_dists, bps_slot_labels, tokens, mask)
        results = self.coarse_net.sample_poses(slot_bps_feat)

        pose_aa = results["fullpose_aa"]
        trans = results["transl"]
        shape = torch.zeros(B, nb, device=bps_dists.device)

        return PoseDecoderOutput(pose=pose_aa, trans=trans, shape=shape)

    # ----------------------------------------------------------------
    # Convenience forward
    # ----------------------------------------------------------------

    def forward(
        self,
        bps_dists: torch.Tensor,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        trans_rhand: torch.Tensor | None = None,
        global_orient_rhand_rotmat: torch.Tensor | None = None,
        bps_slot_labels: torch.Tensor | None = None,
        **kwargs,
    ):
        """Training forward: if GT is provided, run CVAE. Otherwise sample."""
        if bps_slot_labels is None:
            bps_slot_labels = self.predict_slot_labels(bps_dists, tokens, mask)
        slot_bps_feat = self.encode_slot_bps(
            bps_dists, bps_slot_labels, tokens, mask)

        if trans_rhand is not None and global_orient_rhand_rotmat is not None:
            return self.forward_coarse(
                bps_dists, slot_bps_feat, trans_rhand,
                global_orient_rhand_rotmat)
        else:
            return self.coarse_net.sample_poses(slot_bps_feat)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==============================================================================
# GrabNet-style geometric auxiliary loss
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
    reduction: str = "mean",
):
    """GrabNet-inspired geometric supervision."""
    if reduction not in {"mean", "none"}:
        raise ValueError(f"Unsupported reduction: {reduction}")

    def compute_vertex_normals(verts, faces):
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
    B = pred_verts.shape[0]

    def mean_per_sample(x):
        return x.reshape(B, -1).mean(dim=1)

    loss_vert = mean_per_sample(torch.abs(pred_verts - gt_verts))

    if obj_normals is not None:
        o2h_signed, h2o_signed, _ = point2point_signed(
            pred_verts, obj_pc, x_normals=pred_hand_normals, y_normals=obj_normals)
        o2h_gt, h2o_gt, _ = point2point_signed(
            gt_verts, obj_pc, x_normals=gt_hand_normals, y_normals=obj_normals)
    else:
        o2h_signed, h2o_signed, _ = point2point_signed(pred_verts, obj_pc)
        o2h_gt, h2o_gt, _ = point2point_signed(gt_verts, obj_pc)

    loss_dist_h = mean_per_sample(torch.abs(h2o_signed.abs() - h2o_gt.abs()))
    loss_dist_o = mean_per_sample(torch.abs(o2h_signed - o2h_gt))

    if obj_normals is not None:
        pen_depth = torch.relu(-h2o_signed)
        is_pen = pen_depth > 0
        n_pen = is_pen.float().sum(dim=1).clamp(min=1.0)
        loss_pen = pen_depth.sum(dim=1) / n_pen
        pen_mean_all = pen_depth.mean(dim=1)
    else:
        loss_pen = pred_verts.new_zeros(B)
        pen_mean_all = pred_verts.new_zeros(B)

    loss_per_sample = (
        w_vert * loss_vert
        + w_dist_h * loss_dist_h
        + w_dist_o * loss_dist_o
        + w_pen * loss_pen
    )

    loss = loss_per_sample.mean() if reduction == "mean" else loss_per_sample

    return loss, {
        "geo_loss": loss_per_sample.mean().item(),
        "geo_vert": loss_vert.mean().item(),
        "geo_dist_h": loss_dist_h.mean().item(),
        "geo_dist_o": loss_dist_o.mean().item(),
        "geo_pen": loss_pen.mean().item(),
        "geo_pen_all": pen_mean_all.mean().item(),
    }


# ==============================================================================
# Part-Contact Consistency Loss
# ==============================================================================

def build_vert_to_part(mano_fn) -> torch.Tensor:
    """Build vertex → hand-part mapping from MANO skinning weights.

    Uses MANO's linear blend skinning weights to determine the dominant
    joint for each vertex, then maps joints to hand parts via JOINT_TO_PART.

    Returns:
        (778,) long tensor, values in [0, NUM_HAND_PARTS-1]
    """
    weights = mano_fn.mano.th_weights.detach().cpu()  # (778, 16)
    dominant_joint = weights.argmax(dim=1)  # (778,)
    jtp = torch.tensor(JOINT_TO_PART, dtype=torch.long)
    return jtp[dominant_joint]  # (778,)


def propagate_slot_labels(
    obj_pc: torch.Tensor,           # (B, N, 3)
    obj_scale: torch.Tensor,        # (B,) normalization scale
    bps_basis: torch.Tensor,        # (4096, 3) BPS basis coordinates
    bps_slot_labels: torch.Tensor,  # (B, 4096) slot label per basis point
) -> torch.Tensor:
    """Propagate BPS slot labels to object surface points via nearest BPS basis point.

    Object points are normalized to BPS space using obj_scale, then for each
    normalized point, the nearest BPS basis point is found and its slot label
    is assigned to the object point.

    Returns:
        (B, N) long — slot labels for each surface point
    """
    B, N, _ = obj_pc.shape
    device = obj_pc.device

    # Normalize obj_pc to BPS space (already bbox-centered, just scale)
    obj_norm = obj_pc / obj_scale.view(B, 1, 1).clamp(min=1e-6)

    # Find nearest BPS basis point for each surface point (chunked to save memory)
    basis = bps_basis.to(device)  # (4096, 3)
    labels = torch.empty(B, N, dtype=torch.long, device=device)

    chunk_size = 8
    for i in range(0, B, chunk_size):
        j = min(i + chunk_size, B)
        dists = torch.cdist(obj_norm[i:j], basis.unsqueeze(0).expand(j - i, -1, -1))  # (chunk, N, 4096)
        nearest_basis = dists.argmin(dim=2)  # (chunk, N)
        labels[i:j] = torch.gather(bps_slot_labels[i:j], 1, nearest_basis)

    return labels


def compute_part_contact_loss(
    hand_verts: torch.Tensor,       # (B, 778, 3)
    obj_pc: torch.Tensor,           # (B, N, 3)
    tokens: torch.Tensor,           # (B, L)
    token_mask: torch.Tensor,       # (B, L) bool
    vert_to_part: torch.Tensor,     # (778,) long — vertex → part_id
    obj_slot_labels: torch.Tensor | None = None,  # (B, N) long — per-point slot labels
    num_slots: int = NUM_SLOTS,
) -> torch.Tensor:
    """Part-Contact Consistency Loss.

    For each contact token (part_id, slot_id):
      - part_id's hand vertices should be close to the object
      - If obj_slot_labels is provided: specifically close to slot_id region

    This loss is differentiable through hand_verts and encourages the predicted
    hand pose to match the contact specification from the token sequence.

    Returns:
        scalar loss (0 if no valid tokens)
    """
    B, L = tokens.shape
    device = hand_verts.device
    n_semantic = NUM_HAND_PARTS * num_slots

    total_loss = hand_verts.new_zeros(())
    n_terms = 0

    # Pre-compute vert_to_part on device
    vtp = vert_to_part.to(device)

    for b in range(B):
        for j in range(L):
            if not token_mask[b, j]:
                continue
            tok = tokens[b, j].item()
            if tok >= n_semantic:  # skip PAD/MASK
                continue

            part_id = tok // num_slots
            slot_id = tok % num_slots

            # Get hand vertices for this part
            part_mask = (vtp == part_id)
            part_verts = hand_verts[b, part_mask]  # (Vp, 3)

            if obj_slot_labels is not None:
                # Full consistency: part vertices → slot region on object
                slot_mask = (obj_slot_labels[b] == slot_id)
                target_points = obj_pc[b, slot_mask]  # (Ns, 3)
                if target_points.shape[0] == 0:
                    # Fallback to full surface if slot has no points
                    target_points = obj_pc[b]
            else:
                target_points = obj_pc[b]  # (N, 3)

            # Min distance from each part vertex to target region
            dists = torch.cdist(part_verts, target_points)  # (Vp, Ns)
            min_dists = dists.min(dim=1).values  # (Vp,)
            total_loss = total_loss + min_dists.mean()
            n_terms += 1

    if n_terms > 0:
        total_loss = total_loss / n_terms

    return total_loss


# ==============================================================================
# Presets & builders
# ==============================================================================

_PRESETS = {
    "small": dict(
        n_neurons=256, latentD=16, cond_dim=256,
    ),
    "base": dict(
        n_neurons=512, latentD=16, cond_dim=512,
    ),
    "large": dict(
        n_neurons=768, latentD=32, cond_dim=512,
    ),
}


def build_grab_model(config: str = "base", **kw) -> PoseGrabModel:
    """Build a GrabNet-style pose decoder with slot-grounded BPS conditioning.

    Args:
        config: preset name ("small", "base", "large")
    """
    params = _PRESETS[config].copy()
    params.update(kw)
    return PoseGrabModel(**params)
