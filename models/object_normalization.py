import torch
from typing import Tuple


def compute_object_scale(point_xyz: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute a per-object scale from the bounding-box diagonal.

    Args:
        point_xyz: (B, N, 3)

    Returns:
        scale: (B, 1)
    """
    xyz_min = point_xyz.amin(dim=1)
    xyz_max = point_xyz.amax(dim=1)
    scale = (xyz_max - xyz_min).norm(dim=-1, keepdim=True)
    return scale.clamp_min(eps)


def normalize_object_points(point_xyz: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize object coordinates by per-object scale.

    Args:
        point_xyz: (B, N, 3)

    Returns:
        normalized_xyz: (B, N, 3)
        scale:          (B, 1)
    """
    scale = compute_object_scale(point_xyz, eps=eps)
    return point_xyz / scale.unsqueeze(1), scale
