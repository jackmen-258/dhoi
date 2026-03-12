import torch
import torch.nn as nn
from typing import Tuple

from models.object_normalization import compute_object_scale, normalize_object_points
from models.pointnet2 import PointNetSetAbstraction

class PointNet2Encoder(nn.Module):
    """
    PointNet++ SA-only encoder

    输出：
      global_feat:  (B, global_dim)
      point_feat:   (B, 128, point_dim)  - SA2 geometry tokens
      point_xyz:    (B, 128, 3)          - SA2 token coordinates
    """

    def __init__(
        self,
        in_dim: int = 3,
        hidden_dim: int = 128,
        global_dim: int = 256,
        point_dim: int = 256,
    ):
        """
        Args:
            in_dim: 输入点维度（3=xyz, 6=xyz+normal）
            hidden_dim: 基础隐藏维度
            global_dim: 全局特征输出维度
            point_dim: 逐点特征输出维度
        """
        super().__init__()
        self.in_dim = in_dim
        self.global_dim = global_dim
        self.point_dim = point_dim

        # SA1 in_channel 计算:
        #   in_dim=3 → l0_points=None → in_channel=3 (只有 grouped_xyz_norm)
        #   in_dim=6 → l0_points=3dim → in_channel=6 (grouped_xyz_norm + feat)
        sa1_in = in_dim if in_dim > 3 else 3

        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.1, nsample=32,
            in_channel=sa1_in,
            mlp=[hidden_dim, hidden_dim * 2],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.2, nsample=64,
            in_channel=hidden_dim * 2 + 3,
            mlp=[hidden_dim * 2, hidden_dim * 4],
            group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=hidden_dim * 4 + 3,
            mlp=[hidden_dim * 4, hidden_dim * 8],
            group_all=True,
        )

        # SA3 → global_feat
        self.global_head = nn.Sequential(
            nn.Linear(hidden_dim * 8, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 4, global_dim),
        )

        # SA2 → point_feat
        self.point_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, point_dim),
        )

    def forward(
        self, point_cloud: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            point_cloud: (B, N, in_dim)

        Returns:
            global_feat: (B, global_dim)
            point_feat:  (B, 128, point_dim)
            point_xyz:   (B, 128, 3)
        """
        x = point_cloud.permute(0, 2, 1)       # (B, D, N)

        l0_xyz = x[:, :3, :]                    # (B, 3, N)
        l0_points = x[:, 3:, :] if x.shape[1] > 3 else None

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # global: (B, h*8, 1) → (B, h*8) → (B, global_dim)
        global_feat = self.global_head(l3_points.squeeze(-1))

        # point: (B, h*4, 128) → (B, 128, h*4) → (B, 128, point_dim)
        point_feat = self.point_head(l2_points.permute(0, 2, 1))

        # point_xyz: (B, 3, 128) → (B, 128, 3)
        point_xyz = l2_xyz.permute(0, 2, 1).contiguous() 

        return global_feat, point_feat, point_xyz

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
