#!/usr/bin/env python3
"""
generate_grasps.py (v18 — GrabNet-style CVAE + RefineNet)
从物体点云生成抓握姿态

架构:
  Stage 1: diffusion_obj_enc + token denoiser → contact tokens
  Stage 2: BPS + token_cond → CoarseNet CVAE → coarse pose + trans
           PoseGrabModel.refine_net → geometry-only iterative refinement (no shape)
"""

import os
import sys
import json
import argparse
import pickle
import logging
import time

import numpy as np
import torch
from tqdm import tqdm

from models.object_normalization import normalize_object_points
from models.point_encoder import PointNet2Encoder
from models.token_denoiser import build_denoiser
from models.pose_decoder import build_grab_model, RefineNet
from models.discrete_diffusion import AbsorbingDiffusion
from models.clip_encoder import CLIPTextEncoder
from utils.mano_utils import MANOHelper
from utils.pose_utils import aa_to_rot6d

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class OIShapeConfig:
    def __init__(self, split, category="all", intent="all"):
        self.DATA_SPLIT = split
        self.OBJ_CATES = category
        self.INTENT_MODE = intent
        self.AUG_RIGID_P = 0.0
        self.AUG_RIGID_YAW_RANGE_DEG = 0.0
        self.AUG_RIGID_TILT_STD_DEG = 0.0
        self.AUG_RIGID_TRANS_STD_M = 0.0


def load_test_data(args):
    from data.oishape_dataset import OIShape

    oi = OIShape(OIShapeConfig(args.split, args.oi_category, args.oi_intent))
    oi.n_samples = args.n_pc_points
    logger.info(f"OIShape [{args.split}]: {len(oi)} samples")
    return oi


def build_generation_indices(oi):
    n_items = len(oi)

    return list(range(n_items))



def load_models(args, device):
    logger.info(f"Loading diffusion: {args.diffusion_ckpt}")
    diff_state = torch.load(args.diffusion_ckpt, map_location=device)
    diff_cfg = diff_state.get("config", {})

    obj_global_dim = diff_cfg.get("obj_global_dim", args.obj_global_dim)
    obj_point_dim = diff_cfg.get("obj_point_dim", args.obj_point_dim)
    denoiser_config = diff_cfg.get("denoiser_config", args.denoiser_config)

    vocab_size = diff_cfg.get("vocab_size", args.vocab_size)
    pad_token_id = diff_cfg.get("pad_token_id", args.pad_token_id)
    mask_token_id = diff_cfg.get("mask_token_id", args.mask_token_id)
    num_timesteps = diff_cfg.get("num_timesteps", args.num_timesteps)
    text_feat_dim = diff_cfg.get("text_feat_dim", args.text_feat_dim)
    use_text = diff_cfg.get("use_text", args.use_text)

    diffusion_obj_enc = PointNet2Encoder(
        in_dim=6,
        global_dim=obj_global_dim,
        point_dim=obj_point_dim,
    ).to(device)

    if "obj_enc" in diff_state:
        diffusion_obj_enc.load_state_dict(diff_state["obj_enc"])
        logger.info("  Diffusion obj encoder loaded from diffusion checkpoint ✓")
    else:
        logger.warning("  obj_enc not found in diffusion checkpoint, using initialized weights")

    denoiser = build_denoiser(
        denoiser_config,
        vocab_size=vocab_size,
        obj_feat_dim=obj_global_dim,
        obj_point_feat_dim=obj_point_dim,
        text_feat_dim=text_feat_dim if use_text else 0,
        pad_token_id=pad_token_id,
        mask_token_id=mask_token_id,
    ).to(device)
    denoiser.load_state_dict(diff_state["denoiser"])

    diffusion = AbsorbingDiffusion(
        vocab_size=vocab_size,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
        num_timesteps=num_timesteps,
    ).to(device)

    logger.info(f"Loading decoder: {args.decoder_ckpt}")
    dec_state = torch.load(args.decoder_ckpt, map_location=device)
    dec_cfg = dec_state.get("config", {})

    decoder = build_grab_model(
        dec_cfg.get("model_config", args.decoder_config),
        vocab_size=dec_cfg.get("vocab_size", vocab_size),
        n_betas=dec_cfg.get("n_betas", args.n_betas),
    ).to(device)
    missing, unexpected = decoder.load_state_dict(dec_state["model"], strict=False)
    if missing:
        logger.warning(f"  Decoder load: {len(missing)} missing keys (first 5: {missing[:5]})")
    if unexpected:
        logger.warning(f"  Decoder load: {len(unexpected)} unexpected keys (first 5: {unexpected[:5]})")
    has_refine = any(k.startswith("refine_net.") for k in dec_state["model"].keys())
    logger.info(f"  Decoder loaded (BPS + scheme-B token conditioning, RefineNet={'✓' if has_refine else '✗'})")

    # BPS basis for object encoding
    bps_path = dec_cfg.get("bps_path", args.bps_path)
    bps_basis = None
    if os.path.exists(bps_path):
        bps_basis = torch.from_numpy(np.load(bps_path)['basis']).float().to(device)
        logger.info(f"  BPS basis loaded from {bps_path} ✓")
    else:
        logger.warning(f"  BPS basis not found at {bps_path}")

    diffusion_obj_enc.eval()
    denoiser.eval()
    decoder.eval()
    for model in [diffusion_obj_enc, denoiser, decoder]:
        for p in model.parameters():
            p.requires_grad = False

    clip_model = diff_cfg.get("clip_model", args.clip_model)
    clip_encoder = CLIPTextEncoder(model_name=clip_model, device=str(device))

    mano = MANOHelper(args.mano_assets_root, device=device)
    mano.eval()

    return {
        "diffusion_obj_enc": diffusion_obj_enc,
        "denoiser": denoiser,
        "diffusion": diffusion,
        "decoder": decoder,
        "has_refine": has_refine,
        "bps_basis": bps_basis,
        "mano": mano,
        "clip_encoder": clip_encoder,
        "pad_token_id": pad_token_id,
    }


def _compute_bps(obj_pc, bps_basis):
    """Compute BPS encoding for a single object point cloud.

    Args:
        obj_pc: (1, N, 3) object point cloud
        bps_basis: (K, 3) BPS basis points (K=4096 typically)
    Returns:
        bps_object: (1, K) BPS distance encoding
    """
    from bps_torch.bps import bps_torch

    # Match training/cache preprocessing:
    # bbox center + bbox diagonal normalization in BPS space.
    xyz_min = obj_pc.amin(dim=1, keepdim=True)
    xyz_max = obj_pc.amax(dim=1, keepdim=True)
    center = (xyz_min + xyz_max) / 2.0
    scale = (xyz_max - xyz_min).norm(dim=-1, keepdim=True).clamp_min(1e-6)
    obj_pc_norm = (obj_pc - center) / scale

    basis = bps_basis
    if basis.ndim == 3:
        basis = basis.squeeze(0)

    bps = bps_torch(custom_basis=basis.cpu(), device=obj_pc.device)
    result = bps.encode(obj_pc_norm.squeeze(0), feature_type='dists')
    return result['dists'].reshape(1, -1)


def load_bps_slot_cache(args, split):
    """Load per-object BPS slot cache for inference.

    Returns a dict for looking up bps_dists and bps_slot_labels by obj_id,
    or None if cache not found.
    """
    slot_dir = os.path.join(args.bps_slot_cache_dir, split)
    if not os.path.isdir(slot_dir):
        logger.info(f"BPS slot cache not found at {slot_dir}; "
                     f"will compute BPS online for inference")
        return None

    # Just store the directory path; load individual files on demand
    logger.info(f"BPS slot cache: {slot_dir}")
    return {"dir": slot_dir, "cache": {}}


def get_bps_slot_data(bps_slot_cache, obj_id, device):
    """Get per-object BPS slot data from cache.

    Returns:
        (bps_dists, bps_slot_labels) tensors, or (None, None) if not found.
    """
    if bps_slot_cache is None or not obj_id:
        return None, None

    cache = bps_slot_cache["cache"]
    if obj_id not in cache:
        npz_path = os.path.join(bps_slot_cache["dir"], f"{obj_id}.npz")
        if os.path.exists(npz_path):
            data = np.load(npz_path, allow_pickle=True)
            cache[obj_id] = {
                "bps_dists": data["bps_dists"].astype(np.float32),
                "bps_slot_labels": data["bps_slot_labels"].astype(np.int64),
            }
        else:
            cache[obj_id] = None

    entry = cache.get(obj_id)
    if entry is None:
        return None, None

    bps_dists = torch.from_numpy(entry["bps_dists"]).unsqueeze(0).to(device)
    bps_slot_labels = torch.from_numpy(entry["bps_slot_labels"]).unsqueeze(0).to(device)
    return bps_dists, bps_slot_labels


@torch.no_grad()
def generate_single(models, obj_pc, obj_vn, args, text_feat=None,
                    bps_dists_cached=None):
    diffusion_obj_enc = models["diffusion_obj_enc"]
    denoiser = models["denoiser"]
    diffusion = models["diffusion"]
    decoder = models["decoder"]
    has_refine = models["has_refine"]
    bps_basis = models["bps_basis"]
    mano = models["mano"]

    diff_obj_pc_norm, _ = normalize_object_points(obj_pc)
    diff_obj_input = torch.cat([diff_obj_pc_norm, obj_vn], dim=-1)
    diff_obj_global_feat, diff_obj_point_feat, _ = diffusion_obj_enc(diff_obj_input)

    cond = {
        "obj_feat": diff_obj_global_feat,
        "obj_point_feat": diff_obj_point_feat,
    }
    if text_feat is not None:
        cond["text_feat"] = text_feat

    # ---- Stage 1: Contact token generation ----
    result = diffusion.sample(
        denoiser,
        cond=cond,
        batch_size=1,
        seq_length=args.seq_length,
        temperature=args.temperature,
        guidance_scale=args.guidance_scale,
    )
    tokens = result["samples"]
    inferred_mask = tokens != models["pad_token_id"]

    # ---- Stage 2: CVAE decoder → coarse pose → RefineNet ----
    if bps_dists_cached is not None:
        # Use cached BPS dists from slot cache (normalized per-object)
        bps_object = bps_dists_cached
    else:
        bps_object = _compute_bps(obj_pc, bps_basis)

    pred = decoder.sample(
        bps_object, tokens, inferred_mask,
    )

    final_pose = pred.pose
    final_trans = pred.trans
    final_shape = pred.shape

    if has_refine:
        coarse_rot6d = aa_to_rot6d(pred.pose)
        coarse_verts, _ = mano(pred.pose, pred.trans, pred.shape)
        h2o_dist = RefineNet._compute_h2o_dist(coarse_verts, obj_pc, obj_vn)
        refined = decoder.refine_net(
            h2o_dist, coarse_rot6d, pred.trans,
            obj_pc, mano, shape=pred.shape, obj_normals=obj_vn,
        )
        final_pose = refined["pose"]
        final_trans = refined["trans"]

    verts, joints = mano(final_pose, final_trans, final_shape)

    return (
        verts[0].cpu().numpy(),
        joints[0].cpu().numpy(),
        {
            "pose": final_pose[0].cpu().numpy(),
            "trans": final_trans[0].cpu().numpy(),
            "shape": final_shape[0].cpu().numpy(),
        },
        tokens[0].cpu().numpy(),
    )


def main():
    parser = argparse.ArgumentParser("Generate grasps (v18, diffusion + CVAE decoder + RefineNet)")

    parser.add_argument("--diffusion_ckpt", type=str, default="checkpoints/diffusion/best.pt")
    parser.add_argument("--decoder_ckpt", type=str, default="checkpoints/decoder/best.pt")

    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--oi_category", type=str, default="all")
    parser.add_argument("--oi_intent", type=str, default="all")
    parser.add_argument("--category", dest="oi_category", type=str)
    parser.add_argument("--intent", dest="oi_intent", type=str)
    parser.add_argument("--n_pc_points", type=int, default=10000)

    parser.add_argument("--decoder_config", type=str, default="base")
    parser.add_argument("--denoiser_config", type=str, default="base")
    parser.add_argument("--obj_global_dim", type=int, default=256)
    parser.add_argument("--obj_point_dim", type=int, default=256)
    parser.add_argument("--n_betas", type=int, default=10)
    parser.add_argument("--bps_path", type=str, default="configs/bps.npz")
    parser.add_argument("--text_feat_dim", type=int, default=512)
    parser.add_argument("--use_text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vocab_size", type=int, default=26)
    parser.add_argument("--pad_token_id", type=int, default=24)
    parser.add_argument("--mask_token_id", type=int, default=25)
    parser.add_argument("--seq_length", type=int, default=12)
    parser.add_argument("--num_timesteps", type=int, default=100)
    parser.add_argument("--mano_assets_root", type=str, default="assets/mano_v1_2")

    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--guidance_scale", type=float, default=2.0)

    parser.add_argument("--text_cache_dir", type=str, default="cache/contact_tokens/test")
    parser.add_argument("--slot_cache_dir", type=str, default="cache/contact_tokens/test")
    parser.add_argument("--no_text", action="store_true")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")

    # Slot-grounded BPS
    parser.add_argument("--bps_slot_cache_dir", type=str, default="cache/bps_slot",
                        help="Per-object BPS slot cache directory")

    parser.add_argument("--save_dir", type=str, default="experiments/generated")
    parser.add_argument("--save_mesh", action="store_true")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    results_dir = os.path.join(args.save_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    models = load_models(args, device)

    clip_enc = models["clip_encoder"]
    text_mode = None
    idx_to_text = {}

    if args.no_text:
        text_mode = "none"
        logger.info("Text condition: disabled")
    else:
        tc_path = os.path.join(args.text_cache_dir, "text_cache.json")
        idx_path = os.path.join(args.text_cache_dir, "dataset_index.json")
        if not (os.path.isfile(tc_path) and os.path.isfile(idx_path)):
            logger.error(f"text_cache.json or dataset_index.json not found in {args.text_cache_dir}")
            sys.exit(1)
        with open(tc_path, encoding="utf-8") as f:
            text_map = json.load(f)
        with open(idx_path) as f:
            index = json.load(f)
        for entry in index["samples"]:
            cf = entry["cache_file"]
            if cf in text_map:
                idx_to_text[entry["idx"]] = text_map[cf]
        text_mode = "cache"
        logger.info(f"Text cache: {len(idx_to_text)} entries")

    if text_mode is None:
        logger.error("Text condition required. Use --no_text or --text_cache_dir")
        sys.exit(1)

    oi = load_test_data(args)
    n_items = len(oi)
    generation_indices = build_generation_indices(oi)
    n_generate = len(generation_indices)

    # Load BPS slot cache for inference (per-object BPS dists + slot labels)
    bps_slot_cache = load_bps_slot_cache(args, args.split)

    logger.info("=" * 60)
    logger.info(f"  Split:           {args.split}")
    logger.info(f"  Source samples:  {n_items}")
    logger.info(f"  Generate count:  {n_generate}")
    decoder_desc = "GrabNet-style CVAE + RefineNet" if models["has_refine"] else "GrabNet-style CVAE (coarse only)"
    logger.info(f"  Decoder:         {decoder_desc}")
    logger.info(f"  Temperature:     {args.temperature}")
    logger.info(f"  Guidance scale:  {args.guidance_scale}")
    logger.info(f"  Text condition:  {text_mode}")
    logger.info(f"  BPS slot cache:  {'loaded (cached BPS dists available)' if bps_slot_cache else 'none (compute BPS online)'}")
    logger.info(f"  Output:          {results_dir}")
    logger.info("=" * 60)

    t0 = time.time()
    total_generated = 0

    try:
        import trimesh
        has_trimesh = True
    except ImportError:
        has_trimesh = False
        if args.save_mesh:
            logger.warning("trimesh not available, --save_mesh disabled")
            args.save_mesh = False

    for gen_idx, si in enumerate(tqdm(generation_indices, desc="Generating grasps")):
        oi_sample = oi[si]

        obj_pc = torch.from_numpy(oi_sample["obj_verts"].astype(np.float32)).unsqueeze(0).to(device)
        obj_vn = torch.from_numpy(oi_sample["obj_vn"].astype(np.float32)).unsqueeze(0).to(device)
        obj_id = oi_sample.get("obj_id", f"obj_{si:05d}")

        # BPS slot cache lookup by obj_id
        bps_dists_cached, _ = get_bps_slot_data(
            bps_slot_cache, obj_id, device)

        if text_mode == "cache":
            text_str = idx_to_text.get(si, "no contact")
            text_feat = clip_enc.encode([text_str])
        else:
            text_feat = None
            text_str = ""

        hand_verts, hand_joints, mano_params, tokens = generate_single(
            models, obj_pc, obj_vn, args, text_feat=text_feat,
            bps_dists_cached=bps_dists_cached,
        )

        grasp_result = {
            "obj_id": obj_id,
            "sample_idx": si,
            "generation_idx": gen_idx,
            "hand_verts_r": hand_verts,
            "hand_joints_r": hand_joints,
            "hand_pose": mano_params["pose"],
            "hand_trans": mano_params["trans"],
            "hand_shape": mano_params["shape"],
            "tokens": tokens,
            "obj_rotmat": np.eye(3, dtype=np.float32),
            "text": text_str,
        }

        fname = f"{obj_id}_{si:05d}"
        pkl_path = os.path.join(results_dir, f"{fname}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(grasp_result, f)

        if args.save_mesh and has_trimesh:
            mesh_dir = os.path.join(results_dir, f"{fname}_vis")
            os.makedirs(mesh_dir, exist_ok=True)

            faces = models["mano"].faces
            hand_mesh = trimesh.Trimesh(vertices=hand_verts, faces=faces, process=False)
            hand_mesh.export(os.path.join(mesh_dir, "hand_pred.ply"))

            gt_verts = oi_sample.get("hand_verts", None)
            if gt_verts is not None:
                gt_mesh = trimesh.Trimesh(vertices=gt_verts, faces=faces, process=False)
                gt_mesh.export(os.path.join(mesh_dir, "hand_gt.ply"))

            obj_mesh = oi.get_obj_mesh(si)
            obj_mesh.export(os.path.join(mesh_dir, "object.ply"))

        total_generated += 1

    elapsed = time.time() - t0
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Done! Generated {total_generated} grasps in {elapsed:.1f}s")
    logger.info(f"  ({elapsed / max(total_generated, 1):.2f}s per grasp)")
    logger.info(f"Results saved to: {results_dir}")
    logger.info("\nEvaluate with:")
    logger.info(f"  python evaluate_grasps.py --exp_path {args.save_dir}")

    meta = {
        "args": vars(args),
        "decoder_type": "cvae+refine" if models["has_refine"] else "cvae",
        "total_generated": total_generated,
        "elapsed_seconds": elapsed,
        "num_source_samples": n_items,
        "num_generation_indices": n_generate,
        "num_unique_source_samples": len(set(generation_indices)),
    }
    with open(os.path.join(args.save_dir, "generate_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)


if __name__ == "__main__":
    main()
