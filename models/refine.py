"""
refine.py — GrabNet-Style Iterative Pose Refinement Network
=============================================================
Coarse pose (from PoseDecoderModel) → RefineNet → physically plausible pose

v2: 使用 pose_utils 中的旋转转换函数, 删除所有内联重复实现.

GrabNet 设计原则:
  - h2o_dist (778 维) 作为唯一的物体信息输入, 不接收显式 BPS/PointNet 特征
  - 6D continuous rotation 表示 (Zhou et al., 2019)
  - 姿态分为 global_orient(6D) + hand_pose(15×6D) + trans(3)
  - BatchNorm + LeakyReLU + ResBlock (GrabNet 风格)
  - 迭代修正: MANO forward 有梯度, 每步重算 h2o_dist
  - 零初始化输出层: 第一次迭代预测零残差, 训练稳定

信息流 (每次迭代 i > 0):
  1. rot6d_to_aa(pose_6d) → pose_aa
  2. MANO(pose_aa, trans, shape) → hand_verts     [有梯度]
  3. point2point_signed(hand_verts, obj_verts) → h2o_dist
  4. net([h2o_dist, pose_6d, trans]) → Δpose_6d, Δtrans
  5. pose_6d += Δpose_6d, trans += Δtrans
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import chamfer_distance as chd

from utils.pose_utils import aa_to_rot6d, rot6d_to_aa


# ==============================================================================
# Constants
# ==============================================================================

N_HAND_VERTS = 778
N_JOINTS     = 16       # 1 (global orient) + 15 (finger joints)
ROT6D_DIM    = N_JOINTS * 6   # 96
TRANS_DIM    = 3


# ==============================================================================
# Signed Distance — 直接照搬 GrabNet train_tools.py
# ==============================================================================

def point2point_signed(x, y, x_normals=None, y_normals=None):
    """
    Signed distance between two point clouds (GrabNet implementation).

    Args:
        x: (N, P1, D) — e.g. hand vertices
        y: (N, P2, D) — e.g. object vertices
        x_normals: Optional (N, P1, D)
        y_normals: Optional (N, P2, D)

    Returns:
        y2x_signed: (N, P2) — signed dist from object to hand
        x2y_signed: (N, P1) — signed dist from hand to object (h2o_dist)
        yidx_near:  (N, P2) — indices of nearest hand vertex for each obj vertex
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
# Pose Encode / Decode (AA ↔ 6D) — 薄封装 pose_utils
# ==============================================================================

def parms_decode(pose_6d, trans):
    """
    6D pose + trans → MANO-compatible axis-angle parameters.

    Args:
        pose_6d: (B, 96) — 16 joints × 6D continuous rotation
        trans:   (B, 3)

    Returns:
        dict with:
            'global_orient': (B, 3)   axis-angle
            'hand_pose':     (B, 45)  axis-angle
            'transl':        (B, 3)
    """
    pose_aa = rot6d_to_aa(pose_6d)                             # (B, 48)

    return {
        'global_orient': pose_aa[:, :3],
        'hand_pose':     pose_aa[:, 3:],
        'transl':        trans,
    }


# ==============================================================================
# GrabNet-Style ResBlock
# ==============================================================================

class ResBlock(nn.Module):
    """
    Residual block with BatchNorm + LeakyReLU (GrabNet design).

    Fin → [FC → BN → LeakyReLU → FC → BN] + skip → LeakyReLU
    """

    def __init__(self, Fin, Fout, n_neurons=256):
        super(ResBlock, self).__init__()
        self.Fin  = Fin
        self.Fout = Fout

        self.fc1 = nn.Linear(Fin, n_neurons)
        self.bn1 = nn.BatchNorm1d(n_neurons)

        self.fc2 = nn.Linear(n_neurons, Fout)
        self.bn2 = nn.BatchNorm1d(Fout)

        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))

        Xout = self.fc1(x)
        Xout = self.bn1(Xout)
        Xout = self.ll(Xout)

        Xout = self.fc2(Xout)
        Xout = self.bn2(Xout)
        Xout = Xin + Xout

        if final_nl:
            return self.ll(Xout)
        return Xout


# ==============================================================================
# RefineNet
# ==============================================================================

class RefineNet(nn.Module):
    """
    GrabNet-style iterative pose refinement.

    输入: h2o_dist (778) + pose_6d (96) + trans (3) = 877 维
    输出: Δpose_6d (96) + Δtrans (3)

    不接收显式 BPS/PointNet 特征 — 物体信息完全通过 h2o_dist 隐式传递.
    迭代时 MANO forward 有梯度, 让梯度通过 h2o_dist → MANO → pose 回传.
    """

    def __init__(
        self,
        in_size: int = N_HAND_VERTS + ROT6D_DIM + TRANS_DIM,  # 877
        h_size:  int = 512,
        n_iters: int = 3,
    ):
        super(RefineNet, self).__init__()

        self.n_iters = n_iters

        self.bn1 = nn.BatchNorm1d(N_HAND_VERTS)

        self.rb1 = ResBlock(in_size, h_size)
        self.rb2 = ResBlock(in_size + h_size, h_size)
        self.rb3 = ResBlock(in_size + h_size, h_size)

        self.out_p = nn.Linear(h_size, ROT6D_DIM)
        self.out_t = nn.Linear(h_size, TRANS_DIM)

        self.dout = nn.Dropout(0.3)

        nn.init.zeros_(self.out_p.weight)
        nn.init.zeros_(self.out_p.bias)
        nn.init.zeros_(self.out_t.weight)
        nn.init.zeros_(self.out_t.bias)

        # MANO 引用, 由 Trainer 外部设置
        # 接口: mano_fn(pose_aa_48, trans_3, shape_10) → (verts, joints)
        self.mano_fn = None

    def _normalize_h2o(self, h2o_dist):
        """
        BatchNorm1d 在训练态下要求 batch size > 1。
        小 batch 调试时退化为使用 running stats，避免无意义崩溃。
        """
        if self.training and h2o_dist.shape[0] == 1:
            return F.batch_norm(
                h2o_dist,
                self.bn1.running_mean,
                self.bn1.running_var,
                self.bn1.weight,
                self.bn1.bias,
                training=False,
                momentum=0.0,
                eps=self.bn1.eps,
            )
        return self.bn1(h2o_dist)

    def _check_inputs(self, coarse_pose_aa, coarse_trans, shape, verts_object):
        if self.mano_fn is None:
            raise RuntimeError(
                "RefineNet.mano_fn is not set. Inject a MANO callable before forward, "
                "e.g. model.mano_fn = mano_helper."
            )

        if coarse_pose_aa.ndim != 2 or coarse_pose_aa.shape[1] != 48:
            raise ValueError(
                f"coarse_pose_aa must have shape (B, 48), got {tuple(coarse_pose_aa.shape)}"
            )
        if coarse_trans.ndim != 2 or coarse_trans.shape[1] != TRANS_DIM:
            raise ValueError(
                f"coarse_trans must have shape (B, 3), got {tuple(coarse_trans.shape)}"
            )
        if shape.ndim != 2:
            raise ValueError(f"shape must have shape (B, 10), got {tuple(shape.shape)}")
        if verts_object.ndim != 3 or verts_object.shape[-1] != 3:
            raise ValueError(
                f"verts_object must have shape (B, N, 3), got {tuple(verts_object.shape)}"
            )

        batch_size = coarse_pose_aa.shape[0]
        if coarse_trans.shape[0] != batch_size or shape.shape[0] != batch_size or verts_object.shape[0] != batch_size:
            raise ValueError("Batch size mismatch among refine inputs.")

    def forward(self, *args, **kwargs):
        return self.forward_simple(*args, **kwargs)

    def forward_simple(
        self,
        coarse_pose_aa,
        coarse_trans,
        shape,
        verts_object,
        obj_vn=None,
        obj_global_feat=None,
        obj_point_feat=None,
        obj_point_xyz=None,
    ):
        """
        简化接口: 接收 axis-angle, 内部处理所有转换.

        Args:
            coarse_pose_aa: (B, 48) axis-angle (from PoseDecoderModel)
            coarse_trans:   (B, 3)
            shape:          (B, 10) MANO shape parameters
            verts_object:   (B, N, 3) object vertices
            obj_vn:         (B, N, 3) object vertex normals
            obj_global_feat / obj_point_feat / obj_point_xyz:
                             为兼容 PointNet 接口预留, 当前 refine 网络不直接使用

        Returns:
            dict: {'global_orient': (B,3), 'hand_pose': (B,45), 'transl': (B,3)}
        """
        self._check_inputs(coarse_pose_aa, coarse_trans, shape, verts_object)

        # ---- 初始 h2o_dist (无梯度, 来自 coarse) ----
        with torch.no_grad():
            coarse_verts, _ = self.mano_fn(coarse_pose_aa, coarse_trans, shape)
        _, h2o_dist, _ = point2point_signed(
            coarse_verts, verts_object, y_normals=obj_vn)

        # ---- AA → 6D (迭代在 6D 空间中进行) ----
        init_pose  = aa_to_rot6d(coarse_pose_aa)                # (B, 96)
        init_trans = coarse_trans                                 # (B, 3)

        # ---- 迭代修正 ----
        for i in range(self.n_iters):
            if i != 0:
                # 有梯度的 MANO forward: 重算 h2o_dist
                hand_parms = parms_decode(init_pose, init_trans)
                pose_aa = torch.cat([
                    hand_parms['global_orient'],
                    hand_parms['hand_pose'],
                ], dim=1)                                         # (B, 48)
                verts_rhand, _ = self.mano_fn(pose_aa, init_trans, shape)
                _, h2o_dist, _ = point2point_signed(
                    verts_rhand, verts_object, y_normals=obj_vn)

            # 网络前向
            h2o_normed = self._normalize_h2o(h2o_dist)
            X0 = torch.cat([h2o_normed, init_pose, init_trans], dim=1)

            X = self.rb1(X0)
            X = self.dout(X)
            X = self.rb2(torch.cat([X, X0], dim=1))
            X = self.dout(X)
            X = self.rb3(torch.cat([X, X0], dim=1))
            X = self.dout(X)

            delta_pose  = self.out_p(X)
            delta_trans = self.out_t(X)

            init_pose  = init_pose  + delta_pose
            init_trans = init_trans + delta_trans

        # ---- 最终解码 ----
        return parms_decode(init_pose, init_trans)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==============================================================================
# Builder
# ==============================================================================

_PRESETS = {
    "small": dict(h_size=512, n_iters=5),
    "base":  dict(h_size=768, n_iters=5),
}


def build_refine_net(config="small", **kw):
    params = _PRESETS[config].copy()
    params.update(kw)
    return RefineNet(**params)
