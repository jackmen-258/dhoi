"""
mano_helper.py
==============
MANO 前向封装（参数冻结，手指顶点分组）

用法:
    from models.mano_helper import MANOHelper

    mano = MANOHelper(mano_assets_root="assets/mano_v1_2", device="cuda")
    verts, joints = mano(pose, trans, shape)
"""

import logging
from typing import Dict

import torch
import torch.nn as nn

# MANO joint → hand_part 映射
# joint 0=wrist → PALM, 1-3=INDEX, 4-6=MIDDLE, 7-9=LITTLE, 10-12=RING, 13-15=THUMB
HAND_PART_JOINT_MAPPING = {
    "THUMB":  [13, 14, 15],
    "INDEX":  [1, 2, 3],
    "MIDDLE": [4, 5, 6],
    "RING":   [10, 11, 12],
    "LITTLE": [7, 8, 9],
    "PALM":   [0],
}

logger = logging.getLogger(__name__)


class MANOHelper(nn.Module):
    """
    MANO 前向封装

    - 参数全部冻结
    - 提供 finger_vertex_ids 分组（用于 contact loss）
    - 提供 faces（用于 mesh 导出）
    """

    def __init__(self, mano_assets_root: str, device: str = "cuda"):
        super().__init__()
        from manotorch.manolayer import ManoLayer

        self._device = device

        self.mano = ManoLayer(
            center_idx=0,
            mano_assets_root=mano_assets_root
        ).to(device)

        for p in self.mano.parameters():
            p.requires_grad = False

        # 面信息（用于 mesh 导出）
        self.faces = self.mano.th_faces.detach().cpu().numpy()

        # 手指顶点分组（用于 contact loss）
        self.finger_vertex_ids = self._build_finger_groups()

    def _build_finger_groups(self) -> Dict[str, torch.Tensor]:
        weights = self.mano.th_weights.detach().cpu().numpy()
        dominant_joint = weights.argmax(axis=1)

        finger_ids = {}
        for part, joints in HAND_PART_JOINT_MAPPING.items():
            jset = set(joints)
            vids = [i for i, j in enumerate(dominant_joint) if j in jset]
            finger_ids[part] = torch.tensor(
                vids, dtype=torch.long, device=self._device
            )

        counts = {k: len(v) for k, v in finger_ids.items()}
        logger.info(f"[MANOHelper] 手指顶点: {counts}")
        return finger_ids

    def forward(self, pose, trans, shape):
        """
        Args:
            pose:  (B, 48) axis-angle
            trans: (B, 3)
            shape: (B, 10)

        Returns:
            verts:  (B, 778, 3)
            joints: (B, 21, 3)
        """
        mano_out = self.mano(pose, shape)
        verts = mano_out.verts + trans.unsqueeze(1)
        joints = mano_out.joints + trans.unsqueeze(1)
        return verts, joints