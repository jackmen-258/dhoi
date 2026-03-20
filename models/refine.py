"""
refine.py — Test-Time Pose Optimization
========================================
Post-processor for generated hand poses:
  1. Anatomical constraint enforcement (joint angle clamping)
  2. Penetration removal (gradient-based optimization)

No neural network, no training, no checkpoint needed.

Usage:
    optimizer = GraspOptimizer(mano_fn, device="cuda")
    refined_pose, refined_trans = optimizer.optimize(
        pose_aa, trans, shape, obj_pc, obj_normals
    )
"""

import logging
import torch
import torch.nn.functional as F
import chamfer_distance as chd

from utils.pose_utils import aa_to_rot6d, rot6d_to_aa

logger = logging.getLogger(__name__)


# ==============================================================================
# Anatomical Joint Angle Limits for MANO
# ==============================================================================
#
# MANO joint layout (right hand, 16 joints × 3 axis-angle):
#   Joint 0:     Wrist (global orientation)
#   Joints 1-3:  INDEX  (MCP, PIP, DIP)
#   Joints 4-6:  MIDDLE (MCP, PIP, DIP)
#   Joints 7-9:  LITTLE (MCP, PIP, DIP)
#   Joints 10-12: RING  (MCP, PIP, DIP)
#   Joints 13-15: THUMB (CMC, MCP, IP)
#
# Max angle magnitude per joint (radians).
# These are conservative upper bounds from hand biomechanics:
#   - Wrist: ~80° flexion/extension, ~30° abduction → magnitude ≤ ~1.6 rad
#   - MCP: ~90° flexion + ~30° abduction → magnitude ≤ ~2.0 rad
#   - PIP: ~110° flexion, minimal other → magnitude ≤ ~2.0 rad
#   - DIP: ~80° flexion, minimal other → magnitude ≤ ~1.5 rad
#   - Thumb: wider range but still bounded

# Per-joint maximum rotation angle (radians)
JOINT_ANGLE_LIMITS = torch.tensor([
    3.14,   # 0:  wrist (global) — wide range
    2.0,    # 1:  index MCP
    2.0,    # 2:  index PIP
    1.5,    # 3:  index DIP
    2.0,    # 4:  middle MCP
    2.0,    # 5:  middle PIP
    1.5,    # 6:  middle DIP
    2.0,    # 7:  little MCP
    2.0,    # 8:  little PIP
    1.5,    # 9:  little DIP
    2.0,    # 10: ring MCP
    2.0,    # 11: ring PIP
    1.5,    # 12: ring DIP
    2.5,    # 13: thumb CMC (most mobile)
    1.8,    # 14: thumb MCP
    1.5,    # 15: thumb IP
], dtype=torch.float32)


def clamp_joint_angles(pose_aa, limits=None):
    """
    Clamp per-joint rotation angles to anatomical limits.

    Scales down the axis-angle vector for joints that exceed their
    maximum rotation angle, preserving the rotation axis direction.

    Args:
        pose_aa: (B, 48) axis-angle pose (16 joints × 3)
        limits:  (16,) optional per-joint max angle in radians

    Returns:
        clamped_pose: (B, 48) clamped axis-angle pose
        n_clamped:    int, number of joints that were clamped
    """
    if limits is None:
        limits = JOINT_ANGLE_LIMITS.to(pose_aa.device)
    else:
        limits = limits.to(pose_aa.device)

    B = pose_aa.shape[0]
    pose = pose_aa.reshape(B, 16, 3)

    # Compute per-joint rotation angle (magnitude of axis-angle)
    angles = pose.norm(dim=-1, keepdim=True)  # (B, 16, 1)

    # Scale factor: min(1, limit / angle)
    max_angles = limits.view(1, 16, 1)
    scale = torch.where(
        angles > max_angles,
        max_angles / angles.clamp(min=1e-8),
        torch.ones_like(angles),
    )

    n_clamped = (angles.squeeze(-1) > max_angles.squeeze(-1)).sum().item()
    clamped = pose * scale
    return clamped.reshape(B, 48), int(n_clamped)


def compute_anatomical_loss(pose_aa, limits=None, reduction: str = "mean"):
    """
    Differentiable soft penalty for joint angles exceeding anatomical limits.

    Returns smooth quadratic penalty that increases as joints exceed their limits.

    Args:
        pose_aa: (B, 48) axis-angle pose
        limits:  (16,) optional per-joint max angle

    Returns:
        loss:
            scalar if reduction="mean"
            (B,) if reduction="none"
    """

    if reduction not in {"mean", "none"}:
        raise ValueError(f"Unsupported reduction: {reduction}")

    if limits is None:
        limits = JOINT_ANGLE_LIMITS.to(pose_aa.device)
    else:
        limits = limits.to(pose_aa.device)

    B = pose_aa.shape[0]
    pose = pose_aa.reshape(B, 16, 3)
    angles = pose.norm(dim=-1)  # (B, 16)
    max_angles = limits.view(1, 16)

    excess = torch.relu(angles - max_angles)  # (B, 16)
    
    loss_per_sample = excess.pow(2).mean(dim=1)
    return loss_per_sample.mean() if reduction == "mean" else loss_per_sample


# ==============================================================================
# Signed Distance & Penetration
# ==============================================================================

SIGN_RELIABLE_RADIUS = 0.02  # 2cm


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


def compute_penetration(hand_verts, obj_pc, obj_normals, hand_normals=None):
    """
    Compute per-vertex penetration depth (differentiable).

    Returns:
        pen_depth:    (B, 778) penetration depth (0 for non-penetrating)
        h2o_unsigned: (B, 778) unsigned distance to nearest object point
        is_inside:    (B, 778) bool mask of penetrating vertices
    """
    ch = chd.ChamferDistance()
    B, P1, D = hand_verts.shape

    _, _, xidx_near, _ = ch(hand_verts, obj_pc)
    xidx_expanded = xidx_near.view(B, P1, 1).expand(B, P1, D).long()
    nearest_obj = obj_pc.gather(1, xidx_expanded)
    h2o_vec = hand_verts - nearest_obj
    h2o_unsigned = h2o_vec.norm(dim=-1)

    # Object normal test
    nn_normals = obj_normals.gather(1, xidx_expanded)
    dot_obj = (h2o_vec * nn_normals).sum(dim=-1)
    is_inside = dot_obj < 0

    # Hand normal cross-validation (if available)
    if hand_normals is not None:
        dot_hand = (h2o_vec * hand_normals).sum(dim=-1)
        inside_by_hand = dot_hand < 0
        is_inside = is_inside & inside_by_hand

    # Proximity filter
    is_inside = is_inside & (h2o_unsigned < SIGN_RELIABLE_RADIUS)

    pen_depth = torch.where(is_inside, h2o_unsigned, torch.zeros_like(h2o_unsigned))
    return pen_depth, h2o_unsigned, is_inside


# ==============================================================================
# GraspOptimizer — combines anatomical + penetration optimization
# ==============================================================================

class GraspOptimizer:
    """
    Test-time optimization for generated hand grasps.

    Two-phase process:
      Phase 1: Hard clamp joint angles to anatomical limits (instant, no grad)
      Phase 2: Gradient-based optimization to reduce penetration while
               maintaining anatomical validity and pose similarity.

    Loss = w_pen * penetration²
         + w_anat * anatomical_excess²
         + w_reg_pose * ||delta_rot6d||²
         + w_reg_trans * ||delta_trans||²
    """

    def __init__(
        self,
        mano_fn,
        device="cuda",
        # Phase 1
        clamp_joints=True,
        # Phase 2
        n_iters=80,
        lr=0.003,
        w_pen=1.0,
        w_anat=5.0,
        w_reg_pose=0.1,
        w_reg_trans=0.5,
        verbose=False,
    ):
        self.mano_fn = mano_fn
        self.device = device
        self.clamp_joints = clamp_joints
        self.n_iters = n_iters
        self.lr = lr
        self.w_pen = w_pen
        self.w_anat = w_anat
        self.w_reg_pose = w_reg_pose
        self.w_reg_trans = w_reg_trans
        self.verbose = verbose

        self.faces_t = getattr(mano_fn, "faces_tensor", None)

    def optimize(self, pose_aa, trans, shape, obj_pc, obj_normals):
        """
        Optimize hand pose: first clamp, then gradient optimization.

        Args:
            pose_aa:     (B, 48) axis-angle hand pose
            trans:       (B, 3)  hand translation
            shape:       (B, 10) MANO shape parameters (frozen)
            obj_pc:      (B, N, 3) object point cloud
            obj_normals: (B, N, 3) object surface normals

        Returns:
            refined_pose: (B, 48)
            refined_trans: (B, 3)
            info: dict
        """
        B = pose_aa.shape[0]

        # ============ Phase 1: Anatomical clamp ============
        if self.clamp_joints:
            pose_aa, n_clamped = clamp_joint_angles(pose_aa.detach())
            if self.verbose and n_clamped > 0:
                logger.info(f"  Phase 1: clamped {n_clamped} joints")
        else:
            pose_aa = pose_aa.detach()
            n_clamped = 0

        # ============ Phase 2: Gradient optimization ============
        init_rot6d = aa_to_rot6d(pose_aa)
        init_trans = trans.detach().clone()

        # Check initial penetration
        with torch.no_grad():
            init_verts, _ = self.mano_fn(pose_aa, init_trans, shape)
            init_hand_normals = None
            if self.faces_t is not None:
                init_hand_normals = _compute_vertex_normals(init_verts, self.faces_t)
            init_pen, _, _ = compute_penetration(
                init_verts, obj_pc, obj_normals, init_hand_normals
            )
            init_pen_mean = init_pen.mean().item()

        # Skip optimization if no penetration
        if init_pen_mean < 1e-5 and self.n_iters > 0:
            return pose_aa, init_trans, {
                "skipped": True,
                "n_clamped": n_clamped,
                "init_pen_mm": init_pen_mean * 1000,
                "final_pen_mm": init_pen_mean * 1000,
                "n_iters": 0,
            }

        if self.n_iters <= 0:
            # Only clamp, no optimization
            return pose_aa, init_trans, {
                "skipped": True,
                "n_clamped": n_clamped,
                "init_pen_mm": init_pen_mean * 1000,
                "final_pen_mm": init_pen_mean * 1000,
                "n_iters": 0,
            }

        # Optimizable deltas
        delta_rot6d = torch.zeros_like(init_rot6d, requires_grad=True)
        delta_trans = torch.zeros_like(init_trans, requires_grad=True)
        optimizer = torch.optim.Adam([delta_rot6d, delta_trans], lr=self.lr)

        best_pen = init_pen_mean
        best_rot6d = init_rot6d.clone()
        best_trans = init_trans.clone()

        for step in range(self.n_iters):
            optimizer.zero_grad()

            cur_rot6d = init_rot6d + delta_rot6d
            cur_trans = init_trans + delta_trans
            cur_pose_aa = rot6d_to_aa(cur_rot6d)
            cur_verts, _ = self.mano_fn(cur_pose_aa, cur_trans, shape)

            # Penetration
            cur_hand_normals = None
            if self.faces_t is not None:
                cur_hand_normals = _compute_vertex_normals(cur_verts, self.faces_t)
            pen_depth, _, is_inside = compute_penetration(
                cur_verts, obj_pc, obj_normals, cur_hand_normals
            )
            n_pen = is_inside.float().sum().clamp(min=1.0)
            loss_pen = pen_depth.pow(2).sum() / n_pen

            # Anatomical constraint (soft, differentiable)
            loss_anat = compute_anatomical_loss(cur_pose_aa)

            # Regularization
            loss_reg_pose = delta_rot6d.pow(2).mean()
            loss_reg_trans = delta_trans.pow(2).mean()

            loss = (
                self.w_pen * loss_pen
                + self.w_anat * loss_anat
                + self.w_reg_pose * loss_reg_pose
                + self.w_reg_trans * loss_reg_trans
            )

            loss.backward()
            optimizer.step()

            cur_pen = pen_depth.mean().item()
            if cur_pen < best_pen:
                best_pen = cur_pen
                best_rot6d = cur_rot6d.detach().clone()
                best_trans = cur_trans.detach().clone()

            if self.verbose and (step % 20 == 0 or step == self.n_iters - 1):
                logger.info(
                    f"    Step {step:3d}  "
                    f"pen={cur_pen*1000:.3f}mm  "
                    f"anat={loss_anat.item():.4f}  "
                    f"n_pen={is_inside.float().sum().item():.0f}"
                )

        # Final hard clamp to guarantee anatomical validity
        refined_pose = rot6d_to_aa(best_rot6d)
        if self.clamp_joints:
            refined_pose, _ = clamp_joint_angles(refined_pose)

        return refined_pose, best_trans, {
            "skipped": False,
            "n_clamped": n_clamped,
            "init_pen_mm": init_pen_mean * 1000,
            "final_pen_mm": best_pen * 1000,
            "n_iters": self.n_iters,
        }


# ==============================================================================
# Backward compat alias
# ==============================================================================
PenetrationOptimizer = GraspOptimizer


# ==============================================================================
# RefineNet — Trainable Iterative Refinement (Stage 3)
# ==============================================================================

class _ResBlock(torch.nn.Module):
    """Residual block: Linear → BN → LeakyReLU → Linear → BN → skip → LeakyReLU."""

    def __init__(self, n):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n, n),
            torch.nn.BatchNorm1d(n),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Linear(n, n),
            torch.nn.BatchNorm1d(n),
        )
        self.act = torch.nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class RefineNet(torch.nn.Module):
    """
    Iterative hand-pose refinement network (GrabNet-style).

    At each iteration:
      1. Compute signed h2o distances (778,) from current hand vertices.
      2. Concatenate [h2o_dist | rot6d | trans] → MLP → delta_rot6d, delta_trans.
      3. Accumulate: rot6d += delta_rot6d, trans += delta_trans.

    Constructor args:
        n_iters   (int): number of refinement iterations
        h_size    (int): hidden/output width of MLP
        n_neurons (int): width for ResBlocks (may equal h_size)
    """

    def __init__(self, n_iters: int = 3, h_size: int = 512, n_neurons: int = 512):
        super().__init__()
        from utils.pose_utils import rot6d_to_aa  # local import to avoid circular
        self._rot6d_to_aa = rot6d_to_aa

        self.n_iters = n_iters
        in_dim = 778 + 96 + 3  # h2o_dist + rot6d + trans

        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, n_neurons),
            torch.nn.BatchNorm1d(n_neurons),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Dropout(0.3),
            _ResBlock(n_neurons),
            _ResBlock(n_neurons),
            _ResBlock(n_neurons),
        )
        self.out_pose  = torch.nn.Linear(n_neurons, 96)  # delta_rot6d
        self.out_trans = torch.nn.Linear(n_neurons, 3)   # delta_trans

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_h2o_dist(
        hand_verts:   torch.Tensor,   # (B, 778, 3)
        obj_pc:       torch.Tensor,   # (B, N, 3)
        obj_normals:  torch.Tensor,   # (B, N, 3)
        hand_normals: torch.Tensor = None,  # (B, 778, 3) optional
    ) -> torch.Tensor:
        """Signed per-vertex hand-to-object distance (B, 778).

        Positive  → vertex outside the object.
        Negative  → vertex inside (penetrating).
        """
        import chamfer_distance as chd

        B, P, D = hand_verts.shape
        ch = chd.ChamferDistance()

        # Nearest object point for each hand vertex
        _, _, xidx, _ = ch(hand_verts, obj_pc)
        xidx_exp = xidx.view(B, P, 1).expand(B, P, D).long()

        nearest_obj = obj_pc.gather(1, xidx_exp)          # (B, 778, 3)
        h2o_vec     = hand_verts - nearest_obj             # (B, 778, 3)
        unsigned    = h2o_vec.norm(dim=-1)                 # (B, 778)

        # Sign by object normal: dot < 0 → inside → negative distance
        nn_normals = obj_normals.gather(1, xidx_exp)
        dot_obj    = (h2o_vec * nn_normals).sum(dim=-1)

        if hand_normals is not None:
            dot_hand = (h2o_vec * hand_normals).sum(dim=-1)
            inside   = (dot_obj < 0) & (dot_hand < 0)
        else:
            inside = dot_obj < 0

        signed = torch.where(inside, -unsigned, unsigned)
        return signed  # (B, 778)

    # ------------------------------------------------------------------
    def forward(
        self,
        h2o_dist:    torch.Tensor,    # (B, 778)  initial signed distances
        coarse_rot6d: torch.Tensor,   # (B, 96)
        coarse_trans: torch.Tensor,   # (B, 3)
        obj_pc:       torch.Tensor,   # (B, N, 3)
        mano_fn,                      # MANOHelper with __call__(pose_aa, trans, shape)
        shape:        torch.Tensor = None,  # (B, 10) MANO betas
        obj_normals:  torch.Tensor = None,  # (B, N, 3)
    ) -> dict:
        """
        Returns dict with keys:
            "pose"  : (B, 48)  final axis-angle
            "trans" : (B, 3)   final translation
            "rot6d" : (B, 96)  final rot6d
        """
        cur_rot6d  = coarse_rot6d
        cur_trans  = coarse_trans
        cur_h2o    = h2o_dist

        for i in range(self.n_iters):
            if i > 0 and obj_normals is not None:
                # Recompute h2o_dist from current pose
                with torch.no_grad():
                    cur_pose_aa = self._rot6d_to_aa(cur_rot6d)
                    cur_verts, _ = mano_fn(cur_pose_aa, cur_trans, shape)
                cur_h2o = self._compute_h2o_dist(cur_verts, obj_pc, obj_normals)

            x    = torch.cat([cur_h2o, cur_rot6d, cur_trans], dim=-1)
            feat = self.net(x)
            cur_rot6d  = cur_rot6d  + self.out_pose(feat)
            cur_trans  = cur_trans  + self.out_trans(feat)

        final_pose_aa = self._rot6d_to_aa(cur_rot6d)
        return {"pose": final_pose_aa, "trans": cur_trans, "rot6d": cur_rot6d}


# ==============================================================================
# compute_refine_loss
# ==============================================================================

def compute_refine_loss(
    pred_verts:    torch.Tensor,   # (B, 778, 3)  refined hand vertices
    gt_verts:      torch.Tensor,   # (B, 778, 3)  ground-truth hand vertices
    obj_pc:        torch.Tensor,   # (B, N, 3)
    mano_fn,
    obj_normals:   torch.Tensor,   # (B, N, 3)
    coarse_verts:  torch.Tensor,   # (B, 778, 3)  coarse hand vertices (for reg)
    w_vert:        float = 5.0,
    w_dist:        float = 5.0,
    w_pen:         float = 100.0,
    w_contact:     float = 2.0,
    w_reg:         float = 1.0,
    contact_thresh: float = 0.005,
) -> tuple:
    """
    Compute refinement training loss.

    Returns:
        loss    (scalar Tensor)
        metrics (dict of float)
    """
    # ---- vertex reconstruction (L1) ----
    loss_vert = F.l1_loss(pred_verts, gt_verts)

    # ---- h2o distance matching ----
    pred_h2o = RefineNet._compute_h2o_dist(pred_verts, obj_pc, obj_normals)
    gt_h2o   = RefineNet._compute_h2o_dist(gt_verts,   obj_pc, obj_normals)
    loss_dist = F.l1_loss(pred_h2o, gt_h2o)

    # ---- penetration (quadratic) ----
    pen_depth, h2o_unsigned, is_inside = compute_penetration(
        pred_verts, obj_pc, obj_normals)
    n_pen  = is_inside.float().sum().clamp(min=1.0)
    loss_pen = pen_depth.pow(2).sum() / n_pen
    pen_ratio = is_inside.float().mean().item()

    # ---- contact attraction ----
    # For GT-contact vertices that are currently outside, pull them toward surface
    with torch.no_grad():
        _, gt_h2o_unsigned, _ = compute_penetration(gt_verts, obj_pc, obj_normals)
        gt_contact = gt_h2o_unsigned < contact_thresh   # (B, 778) bool
    is_outside = ~is_inside
    loss_contact = (h2o_unsigned * gt_contact.float() * is_outside.float()).mean()

    # ---- regularisation: stay close to coarse prediction ----
    loss_reg = F.l1_loss(pred_verts, coarse_verts)

    loss = (w_vert    * loss_vert
            + w_dist    * loss_dist
            + w_pen     * loss_pen
            + w_contact * loss_contact
            + w_reg     * loss_reg)

    metrics = {
        "refine_loss":    loss.item(),
        "refine_vert":    loss_vert.item(),
        "refine_dist":    loss_dist.item(),
        "refine_pen":     loss_pen.item(),
        "refine_contact": loss_contact.item(),
        "pen_ratio":      pen_ratio,
    }
    return loss, metrics
