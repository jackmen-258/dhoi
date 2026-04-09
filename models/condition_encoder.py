import torch
import torch.nn as nn
import torch.nn.functional as F
import chamfer_distance as chd
import math


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


# ==============================================================================
# TokenHistogramEncoder — raw 24D token histogram conditioning
# ==============================================================================

def build_token_histogram(
    tokens: torch.Tensor,
    token_mask: torch.Tensor,
    n_semantic: int = NUM_HAND_PARTS * NUM_SLOTS,
    binary: bool = False,
) -> torch.Tensor:
    """Build a semantic token histogram from padded token sequences.

    Args:
        tokens:      (B, L) token ids
        token_mask:  (B, L) bool mask for valid tokens
        n_semantic:  number of semantic tokens (default: 24)
        binary:      if True, return multi-hot presence instead of raw counts

    Returns:
        (B, n_semantic) float tensor
    """
    mask = ensure_nonempty_token_mask(token_mask)
    semantic_mask = mask & (tokens < n_semantic)
    tok_ids = tokens.clamp(min=0, max=n_semantic - 1)

    one_hot = F.one_hot(tok_ids, n_semantic).float()
    one_hot = one_hot * semantic_mask.float().unsqueeze(-1)
    hist = one_hot.sum(dim=1)
    if binary:
        hist = (hist > 0).float()
    return hist


class TokenHistogramEncoder(nn.Module):
    """Encode tokens as a raw 24D semantic histogram.

    This keeps the conditioning signal in an interpretable, nearly lossless form:
    each dimension corresponds directly to one (hand_part, slot) token.
    """

    def __init__(
        self,
        vocab_size: int = 26,
        num_hand_parts: int = NUM_HAND_PARTS,
        num_slots: int = NUM_SLOTS,
        binary: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_parts = num_hand_parts
        self.n_slots = num_slots
        self.n_semantic = num_hand_parts * num_slots
        self.token_embed_dim = self.n_semantic
        self.binary = binary

    def forward(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        return build_token_histogram(
            tokens,
            token_mask,
            n_semantic=self.n_semantic,
            binary=self.binary,
        )

# ==============================================================================
# BPSSlotPredictor — Predict BPS slot labels (auxiliary, for Part-Contact Loss)
# ==============================================================================


def _build_fixed_knn_graph(
    points: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a fixed kNN graph for BPS basis points."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (K, 3), got {tuple(points.shape)}")

    n_points = points.shape[0]
    if n_points == 0:
        raise ValueError("points must not be empty")

    k = max(1, min(int(k), n_points))
    with torch.no_grad():
        dist = torch.cdist(points.unsqueeze(0), points.unsqueeze(0)).squeeze(0)
        topk = min(k + 1, n_points)
        idx = dist.topk(k=topk, largest=False).indices
        if topk > k:
            idx = idx[:, 1:]
        if idx.shape[1] < k:
            self_idx = torch.arange(n_points, device=points.device).unsqueeze(1)
            idx = torch.cat([idx, self_idx.expand(-1, k - idx.shape[1])], dim=1)
        rel_pos = points[idx] - points.unsqueeze(1)

    return idx.long(), rel_pos.float()


def _gather_knn_features(
    feat: torch.Tensor,
    neighbor_idx: torch.Tensor,
) -> torch.Tensor:
    """Gather neighbor features for a fixed kNN graph."""
    batch_size = feat.shape[0]
    idx = neighbor_idx.unsqueeze(0).expand(batch_size, -1, -1)
    batch = torch.arange(batch_size, device=feat.device).view(batch_size, 1, 1)
    batch = batch.expand_as(idx)
    return feat[batch, idx]


class _LightPointTransformerBlock(nn.Module):
    """Lightweight local attention block over a fixed BPS graph."""

    def __init__(
        self,
        dim: int,
        attn_dim: int,
        value_dim: int,
        dropout: float = 0.0,
        chunk_size: int = 256,
    ):
        super().__init__()
        self.chunk_size = max(1, int(chunk_size))
        self.scale = math.sqrt(max(attn_dim, 1))

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.q_proj = nn.Linear(dim, attn_dim)
        self.k_proj = nn.Linear(dim, attn_dim)
        self.v_proj = nn.Linear(dim, value_dim)
        self.out_proj = nn.Linear(value_dim, dim)

        self.pos_attn = nn.Sequential(
            nn.Linear(3, attn_dim),
            nn.ReLU(inplace=True),
            nn.Linear(attn_dim, attn_dim),
        )
        self.pos_value = nn.Sequential(
            nn.Linear(3, value_dim),
            nn.ReLU(inplace=True),
            nn.Linear(value_dim, value_dim),
        )
        self.score_mlp = nn.Sequential(
            nn.Linear(attn_dim, attn_dim),
            nn.ReLU(inplace=True),
            nn.Linear(attn_dim, 1),
        )

        self.drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        knn_idx: torch.Tensor,
        knn_rel_pos: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm1(x)
        q_all = self.q_proj(h)
        k_all = self.k_proj(h)
        v_all = self.v_proj(h)

        chunks = []
        n_points = x.shape[1]

        for start in range(0, n_points, self.chunk_size):
            end = min(start + self.chunk_size, n_points)
            idx = knn_idx[start:end]
            rel = knn_rel_pos[start:end]

            q = q_all[:, start:end]
            k = _gather_knn_features(k_all, idx)
            v = _gather_knn_features(v_all, idx)

            rel_attn = self.pos_attn(rel).unsqueeze(0)
            rel_value = self.pos_value(rel).unsqueeze(0)

            score = self.score_mlp(torch.tanh(q.unsqueeze(2) - k + rel_attn)).squeeze(-1)
            attn = F.softmax(score / self.scale, dim=2)
            ctx = torch.sum(attn.unsqueeze(-1) * (v + rel_value), dim=2)
            chunks.append(self.out_proj(ctx))

        x = x + self.drop(torch.cat(chunks, dim=1))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x

class BPSSlotPredictor(nn.Module):
    """Predict bps_slot_labels with a lightweight point transformer.

    Trained with cross-entropy on GT bps_slot_labels in train_predictor.py.

    Slot labels are defined per object and do not depend on contact tokens, so
    this branch is geometry-only. The legacy token_cond_dim argument is kept
    for config compatibility but ignored.
    """

    def __init__(
        self,
        n_bps: int = 4096,
        num_slots: int = NUM_SLOTS,
        token_cond_dim: int = NUM_HAND_PARTS * NUM_SLOTS,
        hidden_dim: int = 256,
        basis_points: torch.Tensor | None = None,
        num_neighbors: int = 16,
        num_layers: int = 2,
        attn_dim: int | None = None,
        value_dim: int | None = None,
        dropout: float = 0.0,
        chunk_size: int = 256,
    ):
        super().__init__()
        del token_cond_dim
        self.n_bps = n_bps
        self.num_slots = num_slots
        self.n_classes = num_slots + 1  # +1 for "unassigned"
        self.hidden_dim = hidden_dim
        self.num_neighbors = num_neighbors
        self.num_layers = num_layers

        if basis_points is None:
            basis_points = torch.zeros(n_bps, 3, dtype=torch.float32)
        basis_points = torch.as_tensor(basis_points, dtype=torch.float32)
        if basis_points.ndim == 3:
            basis_points = basis_points.squeeze(0)
        if basis_points.shape != (n_bps, 3):
            raise ValueError(
                f"basis_points must have shape {(n_bps, 3)}, got {tuple(basis_points.shape)}"
            )

        knn_idx, knn_rel_pos = _build_fixed_knn_graph(basis_points, num_neighbors)
        self.register_buffer("basis_points", basis_points, persistent=False)
        self.register_buffer("knn_idx", knn_idx, persistent=False)
        self.register_buffer("knn_rel_pos", knn_rel_pos, persistent=False)

        attn_dim = attn_dim or max(16, hidden_dim // 4)
        value_dim = value_dim or max(32, hidden_dim // 2)

        # Input per basis point: basis xyz + nearest xyz + offset + dist.
        self.input_proj = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            _LightPointTransformerBlock(
                dim=hidden_dim,
                attn_dim=attn_dim,
                value_dim=value_dim,
                dropout=dropout,
                chunk_size=chunk_size,
            )
            for _ in range(max(1, num_layers))
        )
        self.global_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.n_classes),
        )

    def forward(
        self,
        bps_dists: torch.Tensor,      # (B, K)
        bps_nn_points: torch.Tensor | None = None,  # (B, K, 3)
        token_summary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict per-basis-point slot logits.

        Returns:
            logits: (B, K, n_classes)
        """
        del token_summary
        B, K = bps_dists.shape
        if K != self.n_bps:
            raise ValueError(f"Expected {self.n_bps} BPS points, got {K}")

        basis = self.basis_points.unsqueeze(0).expand(B, -1, -1)

        if bps_nn_points is not None:
            if bps_nn_points.ndim != 3 or bps_nn_points.shape != (B, K, 3):
                raise ValueError(
                    f"bps_nn_points must have shape {(B, K, 3)}, got {tuple(bps_nn_points.shape)}"
                )
        else:
            # Fallback for legacy callers/caches that do not provide nearest points.
            bps_nn_points = basis

        offset = basis - bps_nn_points
        point_input = torch.cat(
            [basis, bps_nn_points, offset, bps_dists.unsqueeze(-1)],
            dim=-1,
        )
        point_feat = self.input_proj(point_input)  # (B, K, hidden)

        for block in self.blocks:
            point_feat = block(point_feat, self.knn_idx, self.knn_rel_pos)

        global_feat = self.global_proj(point_feat.mean(dim=1, keepdim=True))
        fused = torch.cat([point_feat, global_feat.expand(-1, K, -1)], dim=-1)
        logits = self.head(fused)  # (B, K, n_classes)
        return logits

    def predict(
        self,
        bps_dists: torch.Tensor,
        bps_nn_points: torch.Tensor | None = None,
        token_summary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict slot labels (argmax).

        Returns:
            (B, K) long — predicted slot labels (0~num_slots-1 or num_slots=unassigned)
        """
        logits = self.forward(
            bps_dists,
            bps_nn_points=bps_nn_points,
            token_summary=token_summary,
        )
        pred = logits.argmax(dim=-1)  # (B, K)
        # Map "unassigned" class back to -1
        pred[pred == self.num_slots] = -1
        return pred

    def compute_loss(
        self,
        logits: torch.Tensor,    # (B, K, n_classes)
        gt_labels: torch.Tensor,  # (B, K) int, -1~num_slots-1
        class_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cross-entropy loss. Maps -1 labels to the "unassigned" class."""
        target = gt_labels.clone().long()
        target[target < 0] = self.num_slots
        B, K, C = logits.shape
        return F.cross_entropy(
            logits.reshape(B * K, C),
            target.reshape(B * K),
            weight=class_weights,
        )


# ==============================================================================
# Signed Distance
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
# Penetration
# ==============================================================================
SIGN_RELIABLE_RADIUS = 0.02  # 2cm

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
