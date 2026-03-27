import torch
import torch.nn.functional as F

AA_POSE_DIM     = 48      # 16 × 3 axis-angle
ROT6D_POSE_DIM  = 96      # 16 × 6 continuous rotation

# ==============================================================================
# 6D Rotation Utilities
# ==============================================================================

def _aa_to_rotmat_stable(axis_angle: torch.Tensor) -> torch.Tensor:
    """axis-angle (..., 3) → rotation matrix (..., 3, 3)."""
    batch_shape = axis_angle.shape[:-1]
    aa_flat = axis_angle.reshape(-1, 3)
    _angle_axis = aa_flat.unsqueeze(1)
    theta2 = torch.bmm(_angle_axis, _angle_axis.transpose(1, 2)).squeeze(1)
    eps = 1e-6
    theta = torch.sqrt(theta2 + eps)
    wxyz = aa_flat / (theta + eps)
    wx, wy, wz = wxyz[:, 0:1], wxyz[:, 1:2], wxyz[:, 2:3]
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    r00 = cos_t + wx*wx*(1-cos_t); r01 = wx*wy*(1-cos_t)-wz*sin_t; r02 = wy*sin_t+wx*wz*(1-cos_t)
    r10 = wz*sin_t+wx*wy*(1-cos_t); r11 = cos_t+wy*wy*(1-cos_t); r12 = -wx*sin_t+wy*wz*(1-cos_t)
    r20 = -wy*sin_t+wx*wz*(1-cos_t); r21 = wx*sin_t+wy*wz*(1-cos_t); r22 = cos_t+wz*wz*(1-cos_t)
    rot_normal = torch.cat([r00,r01,r02,r10,r11,r12,r20,r21,r22], 1).view(-1,3,3)
    rx, ry, rz = aa_flat[:,0:1], aa_flat[:,1:2], aa_flat[:,2:3]
    ones = torch.ones_like(rx)
    rot_taylor = torch.cat([ones,-rz,ry,rz,ones,-rx,-ry,rx,ones], 1).view(-1,3,3)
    mask = (theta2.squeeze(-1)>eps).unsqueeze(-1).unsqueeze(-1).float()
    return (mask*rot_normal + (1-mask)*rot_taylor).reshape(*batch_shape, 3, 3)


def _rotmat_to_aa_stable(R: torch.Tensor) -> torch.Tensor:
    """rotation matrix (..., 3, 3) → axis-angle (..., 3). Via quaternion."""
    batch_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)
    R_34 = F.pad(R_flat, [0, 1])
    rmat_t = R_34.transpose(1, 2)
    eps = 1e-6
    mask_d2 = rmat_t[:,2,2] < eps
    mask_d0_d1 = rmat_t[:,0,0] > rmat_t[:,1,1]
    mask_d0_nd1 = rmat_t[:,0,0] < -rmat_t[:,1,1]
    t0 = 1+rmat_t[:,0,0]-rmat_t[:,1,1]-rmat_t[:,2,2]
    q0 = torch.stack([rmat_t[:,1,2]-rmat_t[:,2,1],t0,rmat_t[:,0,1]+rmat_t[:,1,0],rmat_t[:,2,0]+rmat_t[:,0,2]],-1)
    t1 = 1-rmat_t[:,0,0]+rmat_t[:,1,1]-rmat_t[:,2,2]
    q1 = torch.stack([rmat_t[:,2,0]-rmat_t[:,0,2],rmat_t[:,0,1]+rmat_t[:,1,0],t1,rmat_t[:,1,2]+rmat_t[:,2,1]],-1)
    t2 = 1-rmat_t[:,0,0]-rmat_t[:,1,1]+rmat_t[:,2,2]
    q2 = torch.stack([rmat_t[:,0,1]-rmat_t[:,1,0],rmat_t[:,2,0]+rmat_t[:,0,2],rmat_t[:,1,2]+rmat_t[:,2,1],t2],-1)
    t3 = 1+rmat_t[:,0,0]+rmat_t[:,1,1]+rmat_t[:,2,2]
    q3 = torch.stack([t3,rmat_t[:,1,2]-rmat_t[:,2,1],rmat_t[:,2,0]-rmat_t[:,0,2],rmat_t[:,0,1]-rmat_t[:,1,0]],-1)
    mc0 = (mask_d2 & mask_d0_d1).unsqueeze(-1).float()
    mc1 = (mask_d2 & ~mask_d0_d1).unsqueeze(-1).float()
    mc2 = (~mask_d2 & mask_d0_nd1).unsqueeze(-1).float()
    mc3 = (~mask_d2 & ~mask_d0_nd1).unsqueeze(-1).float()
    q = q0*mc0 + q1*mc1 + q2*mc2 + q3*mc3
    tr = t0.unsqueeze(-1)*mc0 + t1.unsqueeze(-1)*mc1 + t2.unsqueeze(-1)*mc2 + t3.unsqueeze(-1)*mc3
    q = q / (torch.sqrt(tr.clamp(min=1e-8))) * 0.5
    q1v,q2v,q3v = q[...,1],q[...,2],q[...,3]
    sin_sq = q1v*q1v + q2v*q2v + q3v*q3v
    sin_t = torch.sqrt(sin_sq.clamp(min=1e-10))
    cos_t = q[...,0]
    two_t = 2.0*torch.where(cos_t<0, torch.atan2(-sin_t,-cos_t), torch.atan2(sin_t,cos_t))
    k = torch.where(sin_sq>1e-10, two_t/sin_t.clamp(min=1e-10), 2.0*torch.ones_like(sin_t))
    return torch.stack([q1v*k, q2v*k, q3v*k], -1).reshape(*batch_shape, 3)


def _rotmat_to_rot6d(R):
    """Extract first two columns of rotation matrix as 6D representation.

    R[..., :2] gives (..., 3, 2), but .reshape(6) would interleave rows.
    We need column-major order: [col0(3), col1(3)] to match _rot6d_to_rotmat
    which splits as a1=[:3], a2=[3:6].
    """
    # R[..., 0] = first column (3,), R[..., 1] = second column (3,)
    return torch.cat([R[..., 0], R[..., 1]], dim=-1)

def _rot6d_to_rotmat(r6d):
    a1, a2 = r6d[..., :3], r6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1*a2).sum(-1,keepdim=True)*b1, dim=-1)
    return torch.stack([b1, b2, torch.cross(b1,b2,dim=-1)], dim=-1)

def aa_to_rot6d(aa):
    B = aa.shape[0]; K = aa.shape[-1]//3
    return _rotmat_to_rot6d(_aa_to_rotmat_stable(aa.reshape(B,K,3))).reshape(B, K*6)

def rot6d_to_aa(r6d):
    B = r6d.shape[0]; K = r6d.shape[-1]//6
    return _rotmat_to_aa_stable(_rot6d_to_rotmat(r6d.reshape(B,K,6))).reshape(B, K*3)

def aa_x0_to_rot6d_x0(x0_aa):
    return torch.cat([aa_to_rot6d(x0_aa[:,:AA_POSE_DIM]), x0_aa[:,AA_POSE_DIM:]], -1)

def rot6d_x0_to_aa_x0(x0_6d):
    return torch.cat([rot6d_to_aa(x0_6d[:,:ROT6D_POSE_DIM]), x0_6d[:,ROT6D_POSE_DIM:]], -1)