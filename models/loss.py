import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# ==============================================================================
# Geodesic Loss
# ==============================================================================

def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """axis-angle (B, K, 3) → rotation matrix (B, K, 3, 3)"""
    B, K, _ = aa.shape
    angle = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis  = aa / angle

    cos = angle.cos().unsqueeze(-1)
    sin = angle.sin().unsqueeze(-1)

    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    O = torch.zeros_like(x)
    K_mat = torch.stack([
        torch.stack([O, -z,  y], dim=-1),
        torch.stack([z,  O, -x], dim=-1),
        torch.stack([-y, x,  O], dim=-1),
    ], dim=-2)

    eye = torch.eye(3, device=aa.device).expand(B, K, 3, 3)
    R = cos * eye + sin * K_mat + (1 - cos) * torch.einsum('...i,...j->...ij', axis, axis)
    return R


def geodesic_loss(pred_aa: torch.Tensor, gt_aa: torch.Tensor) -> torch.Tensor:
    """
    SO(3) 测地距离, 返回标量 mean (radians).
    """
    B = pred_aa.shape[0]
    K = pred_aa.numel() // (B * 3)
    pred_R = axis_angle_to_matrix(pred_aa.reshape(B, K, 3))
    gt_R   = axis_angle_to_matrix(gt_aa.reshape(B, K, 3))

    R_diff = pred_R.transpose(-2, -1) @ gt_R
    trace  = R_diff.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos    = ((trace - 1.0) / 2.0).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(cos).mean()