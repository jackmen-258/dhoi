"""
pose_decoder.py — BPS + Token Pose Decoder
===========================================
GrabNet-style CVAE (CoarseNet) with BPS + contact token conditioning.

Architecture (following GrabNet Figure 7(2) intent conditioning):
  TokenHistogramEncoder : tokens -> raw 24D histogram
  CoarseNet CVAE        : [BPS(4096) | tok_hist(24)] -> rot6d(96) + trans(3)

RefineNet is integrated as a sub-module of PoseGrabModel (joint-trained, see train_decoder.py).
No shape prediction — uses mean hand shape (following GrabNet).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import chamfer_distance as chd
from utils.pose_utils import rot6d_to_aa, aa_to_rot6d
from models.condition_encoder import (
    TokenHistogramEncoder, point2point_signed,
    NUM_HAND_PARTS, NUM_JOINTS, AA_POSE_DIM, ROT6D_POSE_DIM,
    TRANS_DIM, NUM_SLOTS, DEFAULT_N_BETAS, JOINT_TO_PART,
)


@dataclass
class PoseDecoderOutput:
    pose: torch.Tensor    # (B, 48) axis-angle
    trans: torch.Tensor   # (B, 3)
    shape: torch.Tensor   # (B, 10) betas — zeros (mean shape)


def remap_legacy_posegrab_state_dict(state_dict: dict) -> dict:
    """Return checkpoint keys unchanged.

    Legacy slot-grounder / per-slot-feature modules are no longer part of the
    decoder architecture, so their weights are simply ignored during load.
    """
    return dict(state_dict)


def sanitize_posegrab_state_dict_for_load(
    state_dict: dict,
    model_state_dict: dict,
) -> tuple[dict, list[tuple[str, tuple[int, ...], tuple[int, ...]]]]:
    """Remap legacy keys and drop tensors with incompatible shapes."""
    remapped = remap_legacy_posegrab_state_dict(state_dict)
    sanitized = {}
    dropped = []

    for key, value in remapped.items():
        target_value = model_state_dict.get(key)
        if (
            target_value is not None
            and hasattr(value, "shape")
            and hasattr(target_value, "shape")
            and tuple(value.shape) != tuple(target_value.shape)
        ):
            dropped.append((key, tuple(value.shape), tuple(target_value.shape)))
            continue
        sanitized[key] = value

    return sanitized, dropped


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
# CoarseNet — CVAE (BPS + token conditioning)
# ==============================================================================

class CoarseNet(nn.Module):
    """CVAE for coarse pose generation.

    Follows GrabNet Figure 7(2) intent-conditioning pattern:
      Encoder: [bps_raw(4096) | orient(9) | trans(3) | tok_hist(24)] → z(16)
      Decoder: [z(16) | BN(bps_raw)(4096) | tok_hist(24)] → rot6d(96) + trans(3)
    """

    def __init__(
        self,
        n_neurons: int = 512,
        latentD: int = 16,
        in_bps: int = 4096,
        in_pose: int = 12,
        token_cond_dim: int = NUM_HAND_PARTS * NUM_SLOTS,
    ):
        super().__init__()
        self.latentD = latentD
        self.in_bps = in_bps
        self.in_pose = in_pose
        self.token_cond_dim = token_cond_dim

        # ---- Encoder: bps_raw + GT pose + tok_hist → z ----
        # Independent BN per signal prevents 4096D BPS from dominating smaller conditions.
        enc_in = in_bps + in_pose + token_cond_dim
        self.enc_bn_bps = nn.BatchNorm1d(in_bps)
        self.enc_bn_pose = nn.BatchNorm1d(in_pose)
        self.enc_bn_tok = nn.BatchNorm1d(token_cond_dim)
        self.enc_rb1 = ResBlock(enc_in, n_neurons)
        self.enc_rb2 = ResBlock(n_neurons + enc_in, n_neurons)
        self.enc_mu = nn.Linear(n_neurons, latentD)
        self.enc_var = nn.Linear(n_neurons, latentD)

        # ---- Decoder: z + bps_raw + tok_hist → rot6d + trans ----
        dec_in = latentD + in_bps + token_cond_dim
        self.dec_bn_bps = nn.BatchNorm1d(in_bps)
        self.dec_bn_tok = nn.BatchNorm1d(token_cond_dim)
        self.dec_rb1 = ResBlock(dec_in, n_neurons)
        self.dec_rb2 = ResBlock(n_neurons + dec_in, n_neurons)
        self.dec_pose = nn.Linear(n_neurons, NUM_JOINTS * 6)
        self.dec_trans = nn.Linear(n_neurons, 3)

        # Initialize dec_pose bias to identity rotation rot6d: [1,0,0,0,1,0]
        # This prevents Gram-Schmidt instability from near-zero outputs at init.
        with torch.no_grad():
            self.dec_pose.weight.zero_()
            identity_r6d = torch.tensor([1., 0., 0., 0., 1., 0.]).repeat(NUM_JOINTS)
            self.dec_pose.bias.copy_(identity_r6d)

    def encode(
        self,
        bps_raw: torch.Tensor,
        trans_rhand: torch.Tensor,
        global_orient_rhand_rotmat: torch.Tensor,
        tok_summary: torch.Tensor,
    ) -> torch.distributions.Normal:
        """Encode GT into posterior q(z | x, o, tok)."""
        bs = bps_raw.shape[0]
        pose_in = torch.cat([
            global_orient_rhand_rotmat.reshape(bs, -1),
            trans_rhand,
        ], dim=1)
        X0 = torch.cat([
            self.enc_bn_bps(bps_raw),
            self.enc_bn_pose(pose_in),
            self.enc_bn_tok(tok_summary),
        ], dim=1)
        X = self.enc_rb1(X0, True)
        X = self.enc_rb2(torch.cat([X0, X], dim=1), True)
        return torch.distributions.Normal(
            self.enc_mu(X), F.softplus(self.enc_var(X))
        )

    def decode(
        self,
        z: torch.Tensor,
        bps_raw: torch.Tensor,
        tok_summary: torch.Tensor,
    ) -> dict:
        """Decode z + bps_raw + token histogram → pose + trans.

        Args:
            z:           (B, latentD)
            bps_raw:     (B, 4096) raw BPS distances
            tok_summary: (B, token_cond_dim) raw token histogram
        """
        o_bps = self.dec_bn_bps(bps_raw)
        o_tok = self.dec_bn_tok(tok_summary)
        X0 = torch.cat([z, o_bps, o_tok], dim=1)
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
        tok_summary: torch.Tensor,
        **kwargs,
    ) -> dict:
        """Training forward: encode GT → sample z → decode.

        Args:
            bps_raw:                    (B, 4096) raw BPS distances
            trans_rhand:                (B, 3) GT translation
            global_orient_rhand_rotmat: (B, 9) GT global orient
            tok_summary:                (B, token_cond_dim) raw token histogram
        """
        z_dist = self.encode(
            bps_raw, trans_rhand, global_orient_rhand_rotmat, tok_summary
        )
        z_s = z_dist.rsample()
        results = self.decode(z_s, bps_raw, tok_summary)
        results["mean"] = z_dist.mean
        results["std"] = z_dist.scale
        return results

    @torch.no_grad()
    def sample_poses(
        self,
        bps_raw: torch.Tensor,
        tok_summary: torch.Tensor,
        seed: int | None = None,
    ) -> dict:
        """Sample from prior: z ~ N(0, I) → decode."""
        bs = bps_raw.shape[0]
        device = bps_raw.device
        dtype = bps_raw.dtype
        if seed is not None:
            np.random.seed(seed)
        z = torch.randn(bs, self.latentD, device=device, dtype=dtype)
        return self.decode(z, bps_raw, tok_summary)


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
# RefineNet — Iterative pose refinement (geometry-only, no token conditioning)
# ==============================================================================

class RefineNet(nn.Module):
    """
    Iterative hand-pose refinement network — faithful GrabNet replica.

    Architecture matches GrabNet's RefineNet exactly:
      - BN on h2o_dist input (not hidden layers)
      - 3 ResBlocks with skip connections to X0
      - Dropout(0.3) after each ResBlock
      - Delta accumulation over n_iters iterations

    At each iteration:
      1. BN-normalize h2o_dist (778,)
      2. Concatenate [h2o_bn(778) | rot6d(96) | trans(3)] → X0
      3. ResBlock chain with X0 skip → delta
      4. Accumulate: rot6d += delta_rot6d, trans += delta_trans
    """

    def __init__(self, n_iters: int = 3, n_neurons: int = 512):
        super().__init__()
        self.n_iters = n_iters
        in_dim = 778 + 96 + 3  # h2o_dist + rot6d + trans

        # BN on h2o_dist input (GrabNet: self.bn1 = nn.BatchNorm1d(778))
        self.bn1 = nn.BatchNorm1d(778)

        # ResBlock chain with X0 skip connections (GrabNet structure)
        self.rb1 = ResBlock(in_dim, n_neurons, n_neurons)
        self.rb2 = ResBlock(in_dim + n_neurons, n_neurons, n_neurons)
        self.rb3 = ResBlock(in_dim + n_neurons, n_neurons, n_neurons)

        self.out_pose = nn.Linear(n_neurons, 96)   # delta_rot6d
        self.out_trans = nn.Linear(n_neurons, 3)    # delta_trans
        self.dout = nn.Dropout(0.3)

    @staticmethod
    def _compute_h2o_dist(
        hand_verts: torch.Tensor,     # (B, 778, 3)
        obj_pc: torch.Tensor,         # (B, N, 3)
        obj_normals: torch.Tensor = None,
    ) -> torch.Tensor:
        """Unsigned per-vertex hand-to-object distance (B, 778).

        GrabNet uses unsigned h2o distances (object normals are noisy).
        The obj_normals parameter is kept for API compatibility but ignored.
        """
        _, h2o, _ = point2point_signed(hand_verts, obj_pc)
        return h2o.abs()

    def forward(
        self,
        h2o_dist: torch.Tensor,       # (B, 778)
        coarse_rot6d: torch.Tensor,    # (B, 96)
        coarse_trans: torch.Tensor,    # (B, 3)
        obj_pc: torch.Tensor,          # (B, N, 3)
        mano_fn,
        shape: torch.Tensor = None,
        obj_normals: torch.Tensor = None,
    ) -> dict:
        """GrabNet-faithful forward: BN(h2o) → ResBlock chain with X0 skip."""
        cur_rot6d = coarse_rot6d
        cur_trans = coarse_trans
        cur_h2o = h2o_dist

        for i in range(self.n_iters):
            if i > 0:
                cur_pose_aa = rot6d_to_aa(cur_rot6d)
                cur_verts, _ = mano_fn(cur_pose_aa, cur_trans, shape)
                cur_h2o = self._compute_h2o_dist(cur_verts, obj_pc, obj_normals)

            # GrabNet structure: BN on h2o_dist, then concat, then ResBlocks with X0 skip
            h2o_bn = self.bn1(cur_h2o)
            X0 = torch.cat([h2o_bn, cur_rot6d, cur_trans], dim=-1)
            X = self.rb1(X0)
            X = self.dout(X)
            X = self.rb2(torch.cat([X, X0], dim=1))
            X = self.dout(X)
            X = self.rb3(torch.cat([X, X0], dim=1))
            X = self.dout(X)

            cur_rot6d = cur_rot6d + self.out_pose(X)
            cur_trans = cur_trans + self.out_trans(X)

        final_pose_aa = rot6d_to_aa(cur_rot6d)
        return {"pose": final_pose_aa, "trans": cur_trans, "rot6d": cur_rot6d}


def compute_refine_loss(
    pred_verts: torch.Tensor,
    gt_verts: torch.Tensor,
    obj_pc: torch.Tensor,
    v_weights: torch.Tensor = None,
    v_weights2: torch.Tensor = None,
    vpe: torch.Tensor = None,
    kl_coef: float = 0.005,
    h2o_gt: torch.Tensor = None,
    obj_normals: torch.Tensor = None,
    **kwargs,
) -> tuple:
    """GrabNet-style RefineNet loss: dist_h + mesh_rec + edge.

    Uses signed h2o distances for dist_h loss, so the network learns to
    match not just distance magnitude but also inside/outside status.

    Returns:
        loss    (scalar Tensor)
        metrics (dict of float)
    """
    B = pred_verts.shape[0]
    device = pred_verts.device

    if v_weights is None:
        v_weights = torch.ones(778, device=device)
    if v_weights2 is None:
        v_weights2 = torch.ones(778, device=device)

    # Signed h2o distances (positive=outside, negative=penetrating)
    h2o = RefineNet._compute_h2o_dist(pred_verts, obj_pc, obj_normals)

    if h2o_gt is None:
        h2o_gt = RefineNet._compute_h2o_dist(gt_verts, obj_pc, obj_normals)

    # ---- dist loss (GrabNet RefineNet: weight 35) ----
    loss_dist_h = 35 * (1.0 - kl_coef) * torch.mean(
        torch.einsum('ij,j->ij', torch.abs(h2o - h2o_gt), v_weights2))

    # ---- vertex reconstruction loss (GrabNet RefineNet: weight 20) ----
    loss_mesh_rec = 20 * (1.0 - kl_coef) * torch.mean(
        torch.einsum('ijk,j->ijk', torch.abs(gt_verts - pred_verts), v_weights2))

    # ---- edge loss (GrabNet RefineNet: weight 10) ----
    if vpe is not None:
        edges_pred = pred_verts[:, vpe[:, 0]] - pred_verts[:, vpe[:, 1]]
        edges_gt = gt_verts[:, vpe[:, 0]] - gt_verts[:, vpe[:, 1]]
        loss_edge = 10 * (1.0 - kl_coef) * F.l1_loss(edges_pred, edges_gt)
    else:
        loss_edge = torch.tensor(0.0, device=device)

    loss = loss_dist_h + loss_mesh_rec + loss_edge

    metrics = {
        "refine_loss": loss.item(),
        "refine_dist_h": loss_dist_h.item(),
        "refine_mesh_rec": loss_mesh_rec.item(),
        "refine_edge": loss_edge.item(),
    }
    return loss, metrics


def _compute_vertex_normals(verts, faces_t):
    """Compute per-vertex normals from mesh faces."""
    v0 = verts[:, faces_t[:, 0], :]
    v1 = verts[:, faces_t[:, 1], :]
    v2 = verts[:, faces_t[:, 2], :]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    normals = torch.zeros_like(verts)
    for c in range(3):
        idx = faces_t[:, c].view(1, -1, 1).expand(verts.shape[0], -1, 3)
        normals.scatter_add_(1, idx, face_normals)
    return F.normalize(normals, dim=-1, eps=1e-6)


# ==============================================================================
# Top-level wrapper
# ==============================================================================

class PoseGrabModel(nn.Module):
    """GrabNet-style pose decoder: CoarseNet (CVAE) + RefineNet (iterative).

    Contains:
      - TokenHistogramEncoder : tokens → raw 24D histogram
      - CoarseNet             : CVAE, [BPS ‖ tok_hist] → rot6d + trans
      - RefineNet             : iterative geometry-based refinement (h2o_dist → delta)
    """

    def __init__(
        self,
        vocab_size: int = 26,
        num_hand_parts: int = NUM_HAND_PARTS,
        num_slots: int = NUM_SLOTS,
        token_embed_dim: int = NUM_HAND_PARTS * NUM_SLOTS,
        token_combo_dim: int | None = None,  # legacy config, ignored
        n_neurons: int = 512,
        latentD: int = 16,
        in_bps: int = 4096,
        n_betas: int = DEFAULT_N_BETAS,
        kl_coef: float = 0.005,
        refine_n_iters: int = 3,
        refine_n_neurons: int = 512,
        **kwargs,
    ):
        super().__init__()
        del num_slots, token_combo_dim
        kwargs.pop("slot_feat_dim", None)
        kwargs.pop("bps_set_hidden_dim", None)
        kwargs.pop("bps_slot_predictor_hidden", None)
        kwargs.pop("slot_grounder_hidden", None)

        self.n_betas = n_betas
        self.kl_coef = kl_coef
        self.latentD = latentD
        self.no_token_cond = False   # ablation flag

        # Token → raw 24D histogram
        self.token_encoder = TokenHistogramEncoder(
            vocab_size=vocab_size,
            num_hand_parts=num_hand_parts,
            num_slots=NUM_SLOTS,
        )

        self.coarse_net = CoarseNet(
            n_neurons=n_neurons,
            latentD=latentD,
            in_bps=in_bps,
            token_cond_dim=token_embed_dim,
        )

        # RefineNet — iterative geometry-based refinement
        self.refine_net = RefineNet(
            n_iters=refine_n_iters,
            n_neurons=refine_n_neurons,
        )

    # ----------------------------------------------------------------
    # Coarse stage (CVAE)
    # ----------------------------------------------------------------

    def encode_token_condition(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build the interpretable token histogram condition."""
        tok_summary = self.token_encoder(tokens, token_mask)

        if self.no_token_cond:
            tok_summary = torch.zeros_like(tok_summary)

        return tok_summary

    def forward_coarse(
        self,
        bps_raw: torch.Tensor,
        tok_summary: torch.Tensor,
        trans_rhand: torch.Tensor,
        global_orient_rhand_rotmat: torch.Tensor,
    ) -> dict:
        """Training forward through CoarseNet.

        Args:
            bps_raw:                    (B, 4096) raw BPS distances
            tok_summary:                (B, token_embed_dim) raw token histogram
            trans_rhand:                (B, 3) GT translation
            global_orient_rhand_rotmat: (B, 9) GT global orient
        """
        return self.coarse_net(
            bps_raw,
            trans_rhand,
            global_orient_rhand_rotmat,
            tok_summary,
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
        obj_normals: torch.Tensor = None,
    ) -> tuple:
        """CoarseNet loss with signed h2o distances for penetration awareness."""
        device = verts_gt.device
        dtype = verts_gt.dtype

        faces_t = rh_faces[0].long()  # (F, 3) — same topology for all samples
        rh_mesh = _compute_vertex_normals(verts_pred, faces_t)
        rh_mesh_gt = _compute_vertex_normals(verts_gt, faces_t)

        # GrabNet-faithful: hand normals for o2h sign, NO object normals for h2o
        # h2o uses unsigned distance (object normals are noisy on sampled point clouds)
        o2h_signed, h2o, _ = point2point_signed(
            verts_pred, verts_object, rh_mesh)       # h2o is unsigned
        o2h_signed_gt, h2o_gt, _ = point2point_signed(
            verts_gt, verts_object, rh_mesh_gt)

        # Adaptive weights for o2h loss (penetration-aware)
        w_dist = (o2h_signed_gt < 0.01) & (o2h_signed_gt > -0.005)
        w_dist_neg = o2h_signed < 0.0
        w = torch.ones(verts_object.shape[0], verts_object.shape[1], device=device)
        w[~w_dist] = 0.1
        w[w_dist_neg] = 1.5

        # h2o: unsigned distance matching (GrabNet style)
        loss_dist_h = 35 * (1.0 - self.kl_coef) * torch.mean(
            torch.einsum("ij,j->ij", torch.abs(h2o.abs() - h2o_gt.abs()), v_weights2)
        )
        # o2h: signed distance matching (hand normals are reliable)
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
        n_betas: int | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> PoseDecoderOutput:
        """Generate coarse pose from BPS + contact tokens (inference).

        Args:
            bps_dists: (B, 4096) raw BPS distances
            tokens:    (B, L) contact token ids
            mask:      (B, L) token mask
            n_betas:   number of shape parameters (default: self.n_betas)
            seed:      random seed for CVAE prior sampling
        """
        del kwargs
        B = bps_dists.shape[0]
        nb = n_betas if n_betas is not None else self.n_betas

        tok_summary = self.encode_token_condition(tokens, mask)
        results = self.coarse_net.sample_poses(
            bps_dists, tok_summary, seed=seed
        )

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
        **kwargs,
    ):
        """Training forward: if GT is provided, run CVAE. Otherwise sample."""
        del kwargs
        tok_summary = self.encode_token_condition(tokens, mask)

        if trans_rhand is not None and global_orient_rhand_rotmat is not None:
            return self.forward_coarse(
                bps_dists,
                tok_summary,
                trans_rhand,
                global_orient_rhand_rotmat,
            )
        else:
            return self.coarse_net.sample_poses(
                bps_dists,
                tok_summary,
            )

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

    def _safe_vertex_normals(verts, faces):
        if faces is None:
            return None
        if not torch.is_tensor(faces):
            faces_t = torch.as_tensor(faces, device=verts.device, dtype=torch.long)
        else:
            faces_t = faces.to(device=verts.device, dtype=torch.long)
        if faces_t.numel() == 0:
            return None
        return _compute_vertex_normals(verts, faces_t)

    pred_verts, _ = mano_fn(pred.pose, pred.trans, pred.shape)
    mano_faces = getattr(mano_fn, "faces", None)
    pred_hand_normals = _safe_vertex_normals(pred_verts, mano_faces)
    gt_hand_normals = _safe_vertex_normals(gt_verts, mano_faces)
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
    """Part-Contact Consistency Loss (ChamferDistance-based).

    For each contact token (part_id, slot_id):
      - a contact patch within part_id should be close to the object
      - If obj_slot_labels is provided: specifically close to slot_id region

    Uses ChamferDistance CUDA kernel for nearest-neighbor lookup instead of
    torch.cdist full pairwise matrix, reducing memory from O(n_sel*Vp*N) to
    O(n_sel*Vp). Slot masking is achieved by replacing non-target points
    with a far sentinel so ChamferDistance naturally ignores them. If a
    slot has no propagated object points, that term is skipped instead of
    falling back to the full object.

    Returns:
        scalar loss (0 if no valid tokens)
    """
    B, L = tokens.shape
    V = hand_verts.shape[1]  # 778
    N = obj_pc.shape[1]
    device = hand_verts.device
    n_semantic = NUM_HAND_PARTS * num_slots

    vtp = vert_to_part.to(device)  # (V,)

    # Filter valid semantic tokens
    valid = token_mask & (tokens < n_semantic)  # (B, L)
    if not valid.any():
        return hand_verts.new_zeros(())

    # Decode token → part_id, slot_id
    tok_clamped = tokens.clamp(min=0, max=n_semantic - 1)
    part_ids = torch.div(tok_clamped, num_slots, rounding_mode="floor")  # (B, L)
    slot_ids = tok_clamped % num_slots   # (B, L)

    # Pre-compute per-part vertex masks: (NUM_HAND_PARTS, V)
    part_masks = torch.stack([vtp == p for p in range(NUM_HAND_PARTS)])  # (P, V)

    # Collect all valid (b, part, slot) triples and deduplicate
    b_idx, l_idx = valid.nonzero(as_tuple=True)
    parts = part_ids[b_idx, l_idx]  # (K,)
    slots = slot_ids[b_idx, l_idx]  # (K,)

    keys = b_idx * (NUM_HAND_PARTS * num_slots) + parts * num_slots + slots
    unique_keys, inv = keys.unique(return_inverse=True)
    u_b = torch.div(
        unique_keys, NUM_HAND_PARTS * num_slots, rounding_mode="floor"
    )
    remainder = unique_keys % (NUM_HAND_PARTS * num_slots)
    u_part = torch.div(remainder, num_slots, rounding_mode="floor")
    u_slot = remainder % num_slots

    if unique_keys.shape[0] == 0:
        return hand_verts.new_zeros(())

    ch = chd.ChamferDistance()
    FAR = 1e6  # sentinel for masked-out object points

    total_loss = hand_verts.new_zeros(())
    valid_terms = 0

    for p in range(NUM_HAND_PARTS):
        p_sel = (u_part == p)
        if not p_sel.any():
            continue

        pmask = part_masks[p]  # (V,) bool
        if not pmask.any():
            continue

        sel_b = u_b[p_sel]       # batch indices for this part
        sel_slot = u_slot[p_sel]  # slot ids for this part

        # Gather part vertices: (n_sel, Vp, 3)
        pv = hand_verts[sel_b][:, pmask]

        if obj_slot_labels is not None:
            sel_obj = obj_pc[sel_b]               # (n_sel, N, 3)
            sel_labels = obj_slot_labels[sel_b]   # (n_sel, N)

            # Mask non-target slot points with far sentinel
            slot_match = sel_labels == sel_slot.unsqueeze(1)  # (n_sel, N)
            has_slot_pts = slot_match.any(dim=1)               # (n_sel,)
            if not has_slot_pts.any():
                continue

            pv = pv[has_slot_pts]
            sel_obj = sel_obj[has_slot_pts].clone()
            slot_match = slot_match[has_slot_pts]
            sel_obj = sel_obj.masked_fill(~slot_match.unsqueeze(-1), FAR)
        else:
            sel_obj = obj_pc[sel_b]

        # ChamferDistance: x_near = squared dist from each pv to nearest sel_obj
        # Memory: O(n_sel * Vp) instead of O(n_sel * Vp * N)
        x_near, _, _, _ = ch(pv, sel_obj)  # x_near: (n_sel, Vp)
        min_dists = x_near.clamp_min(1e-12).sqrt()  # ChamferDistance returns squared distances

        # Encourage a local contact patch instead of pulling the entire part.
        contact_k = min(
            min_dists.shape[1],
            max(4, min(16, (min_dists.shape[1] + 3) // 4)),
        )
        patch_dists = torch.topk(
            min_dists,
            k=contact_k,
            dim=1,
            largest=False,
        ).values

        total_loss = total_loss + patch_dists.mean(dim=1).sum()
        valid_terms += patch_dists.shape[0]

    if valid_terms == 0:
        return hand_verts.new_zeros(())

    return total_loss / valid_terms


# ==============================================================================
# Presets & builders
# ==============================================================================

_PRESETS = {
    "small": dict(n_neurons=256, latentD=16, token_embed_dim=NUM_HAND_PARTS * NUM_SLOTS),
    "base":  dict(n_neurons=512, latentD=16, token_embed_dim=NUM_HAND_PARTS * NUM_SLOTS),
    "large": dict(n_neurons=768, latentD=32, token_embed_dim=NUM_HAND_PARTS * NUM_SLOTS),
}


def build_grab_model(config: str = "base", **kw) -> PoseGrabModel:
    """Build a GrabNet-style pose decoder with BPS + token conditioning.

    Args:
        config: preset name ("small", "base", "large")
    """
    params = _PRESETS[config].copy()
    params.update(kw)
    return PoseGrabModel(**params)
