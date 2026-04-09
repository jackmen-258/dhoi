import numpy as np
import torch
from scipy.cluster.vq import kmeans, vq
from scipy.stats import entropy as scipy_entropy


def unit_scale(unit: str) -> float:
    unit = (unit or "cm").lower()
    if unit == "m":
        return 1.0
    if unit == "cm":
        return 100.0
    if unit == "mm":
        return 1000.0
    raise ValueError(f"Unsupported diversity unit: {unit}")


def diversity_details(cluster_array, cls_num=20):
    """
    Reference-aligned diversity metric:
      - KMeans: scipy.cluster.vq.kmeans
      - Entropy: scipy.stats.entropy(counts)
      - cluster_size: mean number of samples per active cluster
    """
    if cluster_array is None:
        return {
            "entropy": 0.0,
            "cluster_size": 0.0,
            "cluster_spread": 0.0,
            "counts": [],
            "num_clusters": 0,
            "num_active_clusters": 0,
            "mean_active_cluster_count": 0.0,
        }

    cluster_array = np.asarray(cluster_array, dtype=np.float32)
    if cluster_array.ndim != 2 or cluster_array.shape[0] < 2:
        return {
            "entropy": 0.0,
            "cluster_size": 0.0,
            "cluster_spread": 0.0,
            "counts": [],
            "num_clusters": 0,
            "num_active_clusters": 0,
            "mean_active_cluster_count": 0.0,
        }

    cls_num = int(cls_num)
    cls_num = max(1, min(cls_num, cluster_array.shape[0]))

    codes, _ = kmeans(cluster_array, cls_num)
    vecs, dist = vq(cluster_array, codes)
    counts = np.bincount(vecs, minlength=len(codes)).astype(np.int64)
    active = counts[counts > 0]
    cluster_spread = float(np.mean(dist)) if dist is not None and len(dist) > 0 else 0.0
    mean_active_cluster_count = float(np.mean(active)) if len(active) > 0 else 0.0

    return {
        "entropy": float(scipy_entropy(counts)),
        "cluster_size": mean_active_cluster_count,
        "cluster_spread": cluster_spread,
        "counts": counts.tolist(),
        "num_clusters": int(len(codes)),
        "num_active_clusters": int(len(active)),
        "mean_active_cluster_count": mean_active_cluster_count,
    }


def diversity(cluster_array, cls_num=20):
    details = diversity_details(cluster_array, cls_num=cls_num)
    return details["entropy"], details["cluster_size"]


def xyz_to_xyz1(xyz):
    ones = torch.ones([*xyz.shape[:-1], 1], device=xyz.device)
    return torch.cat([xyz, ones], dim=-1)


def pad34_to_44(mat):
    last_row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=mat.device).reshape(1, 4)
    last_row = last_row.repeat(*mat.shape[:-2], 1, 1)
    return torch.cat([mat, last_row], dim=-2)


def convert_joints(joints, source, target):
    halo_joint_to_mano = np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20])
    mano_joint_to_halo = np.array([0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20])
    mano_joint_to_biomech = np.array([0, 1, 5, 9, 13, 17, 2, 6, 10, 14, 18, 3, 7, 11, 15, 19, 4, 8, 12, 16, 20])
    biomech_joint_to_mano = np.array([0, 1, 6, 11, 16, 2, 7, 12, 17, 3, 8, 13, 18, 4, 9, 14, 19, 5, 10, 15, 20])
    halo_joint_to_biomech = np.array([0, 13, 1, 4, 10, 7, 14, 2, 5, 11, 8, 15, 3, 6, 12, 9, 16, 17, 18, 19, 20])
    biomech_joint_to_halo = np.array([0, 2, 7, 12, 3, 8, 13, 5, 10, 15, 4, 9, 14, 1, 6, 11, 16, 17, 18, 19, 20])

    if source == "halo" and target == "biomech":
        return joints[:, halo_joint_to_biomech]
    if source == "biomech" and target == "halo":
        return joints[:, biomech_joint_to_halo]
    if source == "mano" and target == "biomech":
        return joints[:, mano_joint_to_biomech]
    if source == "biomech" and target == "mano":
        return joints[:, biomech_joint_to_mano]
    if source == "halo" and target == "mano":
        return joints[:, halo_joint_to_mano]
    if source == "mano" and target == "halo":
        return joints[:, mano_joint_to_halo]
    return joints


def normalize(bv, eps=1e-8):
    eps_mat = torch.tensor(eps, device=bv.device)
    norm = torch.max(torch.norm(bv, dim=-1, keepdim=True), eps_mat)
    return bv / norm


def angle2(v1, v2):
    eps = 1e-10
    eps_mat = torch.tensor([eps], device=v1.device)
    n_v1 = v1 / torch.max(torch.norm(v1, dim=-1, keepdim=True), eps_mat)
    n_v2 = v2 / torch.max(torch.norm(v2, dim=-1, keepdim=True), eps_mat)
    return 2 * torch.atan2(
        torch.norm(n_v1 - n_v2, dim=-1),
        torch.norm(n_v1 + n_v2, dim=-1),
    )


def rotation_matrix(angles, axis):
    batch_size = angles.shape[0]
    sina = torch.sin(angles).view(batch_size, 1, 1)
    cosa_1_minus = (1 - torch.cos(angles)).view(batch_size, 1, 1)
    a_batch = axis.view(batch_size, 3)
    o = torch.zeros((batch_size, 1), device=angles.device)
    a0 = a_batch[:, 0:1]
    a1 = a_batch[:, 1:2]
    a2 = a_batch[:, 2:3]
    cprod = torch.cat((o, -a2, a1, a2, o, -a0, -a1, a0, o), 1).view(batch_size, 3, 3)
    eye = torch.eye(3, device=angles.device).view(1, 3, 3)
    return eye + cprod * sina + cprod.bmm(cprod) * cosa_1_minus


def cross(bv_1, bv_2, do_normalize=False):
    cross_prod = torch.cross(bv_1, bv_2, dim=-1)
    if do_normalize:
        cross_prod = normalize(cross_prod)
    return cross_prod


def get_alignment_mat(v1, v2):
    axis = cross(v1, v2, do_normalize=True)
    ang = angle2(v1, v2)
    return rotation_matrix(ang, axis)


def compute_canonical_transform(kp3d, skeleton="bmc"):
    del skeleton
    assert len(kp3d.shape) == 3, "kp3d need to be BS x 21 x 3"

    dev = kp3d.device
    bs = kp3d.shape[0]
    kp3d = kp3d.clone().detach()
    kp3d[0, :, 1] *= -1

    tx = kp3d[:, 0, 0]
    ty = kp3d[:, 0, 1]
    tz = kp3d[:, 0, 2]

    t_t = torch.zeros((bs, 3, 4), device=dev)
    t_t[:, 0, 3] = -tx
    t_t[:, 1, 3] = -ty
    t_t[:, 2, 3] = -tz
    t_t[:, 0, 0] = 1
    t_t[:, 1, 1] = 1
    t_t[:, 2, 2] = 1

    y_axis = torch.tensor([[0.0, -1.0, 0.0]], device=dev).expand(bs, 3)
    v_mrb = normalize(kp3d[:, 3] - kp3d[:, 0])
    r_1 = get_alignment_mat(v_mrb, y_axis)

    v_irb = normalize(kp3d[:, 2] - kp3d[:, 0])
    normal = cross(v_mrb, v_irb).view(-1, 1, 3)
    normal_rot = torch.matmul(normal, r_1.transpose(1, 2)).view(-1, 3)
    z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=dev).expand(bs, 3)
    r_2 = get_alignment_mat(normal_rot, z_axis)

    t_t[0, 1, 1] = -1
    return torch.bmm(r_2, torch.bmm(r_1, t_t))


def transform_to_canonical(kp3d, skeleton="bmc"):
    normalization_mat = compute_canonical_transform(kp3d, skeleton=skeleton)
    kp3d = xyz_to_xyz1(kp3d)
    kp3d_canonical = torch.matmul(normalization_mat.unsqueeze(1), kp3d.unsqueeze(-1))
    kp3d_canonical = kp3d_canonical.squeeze(-1)
    normalization_mat = pad34_to_44(normalization_mat)
    return kp3d_canonical, normalization_mat


def joints_to_diversity_feature(joints, unit="cm", canonical=True):
    joints = np.asarray(joints, dtype=np.float32)
    if joints.ndim != 2 or joints.shape[1] != 3:
        return None

    joints = joints * unit_scale(unit)
    if canonical:
        with torch.no_grad():
            jt = torch.from_numpy(joints).float().unsqueeze(0)
            jt = convert_joints(jt, source="mano", target="biomech")
            jt_after, _ = transform_to_canonical(jt)
            jt_after = convert_joints(jt_after, source="biomech", target="mano")
            joints = jt_after[0].cpu().numpy().astype(np.float32)

    return joints.reshape(-1)
