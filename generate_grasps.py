#!/usr/bin/env python3
"""
generate_grasps.py (v14 — Point Encoder Inference)
从物体点云生成抓握姿态

架构:
  单一 PointNet2Encoder (与 train_diffusion.py / train_decoder.py 对齐)
  Stage 1: Point features → token denoiser → contact tokens
  Stage 2: Point features → pose decoder → pose + trans + shape
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

from models.point_encoder import PointNet2Encoder
from models.token_denoiser import build_denoiser
from models.pose_decoder import build_pose_model
from models.discrete_diffusion import AbsorbingDiffusion
from models.clip_encoder import CLIPTextEncoder
from utils.mano_utils import MANOHelper

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

    obj_enc = PointNet2Encoder(
        in_dim=6,
        global_dim=obj_global_dim,
        point_dim=obj_point_dim,
    ).to(device)

    # 优先使用 diffusion checkpoint 的 obj_enc（与训练一致）
    if "obj_enc" in diff_state:
        obj_enc.load_state_dict(diff_state["obj_enc"])
        logger.info("  Obj encoder loaded from diffusion checkpoint ✓")
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

    decoder = build_pose_model(
        dec_cfg.get("model_config", args.decoder_config),
        vocab_size=dec_cfg.get("vocab_size", vocab_size),
        obj_global_dim=dec_cfg.get("obj_global_dim", obj_global_dim),
        obj_point_dim=dec_cfg.get("obj_point_dim", obj_point_dim),
        obj_xyz_dim=dec_cfg.get("obj_xyz_dim", args.obj_xyz_dim),
        n_betas=dec_cfg.get("n_betas", args.n_betas),
    ).to(device)
    decoder.load_state_dict(dec_state["model"])

    # 如果 decoder checkpoint 里也有 obj_enc，则覆盖为 decoder 对应版本
    if "obj_enc" in dec_state:
        obj_enc.load_state_dict(dec_state["obj_enc"])
        logger.info("  Obj encoder overridden from decoder checkpoint ✓")

    obj_enc.eval()
    denoiser.eval()
    decoder.eval()
    for model in (obj_enc, denoiser, decoder):
        for p in model.parameters():
            p.requires_grad = False

    norm_stats_path = os.path.join(os.path.dirname(args.decoder_ckpt), "norm_stats.npz")
    if os.path.isfile(norm_stats_path):
        stats = np.load(norm_stats_path)
        decoder.set_normalization(
            torch.from_numpy(stats["mean"]).to(device),
            torch.from_numpy(stats["std"]).to(device),
        )
        logger.info(f"  Norm stats loaded from {norm_stats_path}")
    else:
        logger.warning(f"  norm_stats.npz not found at {norm_stats_path}")

    clip_model = diff_cfg.get("clip_model", args.clip_model)
    clip_encoder = CLIPTextEncoder(model_name=clip_model, device=str(device))

    mano = MANOHelper(args.mano_assets_root, device=device)
    mano.eval()

    return {
        "obj_enc": obj_enc,
        "denoiser": denoiser,
        "diffusion": diffusion,
        "decoder": decoder,
        "mano": mano,
        "clip_encoder": clip_encoder,
        "pad_token_id": pad_token_id,
    }


@torch.no_grad()
def generate_single(models, obj_pc, obj_vn, args, text_feat=None):
    obj_enc = models["obj_enc"]
    denoiser = models["denoiser"]
    diffusion = models["diffusion"]
    decoder = models["decoder"]
    mano = models["mano"]

    obj_input = torch.cat([obj_pc, obj_vn], dim=-1)
    obj_global_feat, obj_point_feat, obj_point_xyz = obj_enc(obj_input)

    cond = {
        "obj_feat": obj_global_feat,
        "obj_point_feat": obj_point_feat,
    }
    if text_feat is not None:
        cond["text_feat"] = text_feat

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

    pred = decoder(tokens, inferred_mask, obj_global_feat, obj_point_feat, obj_point_xyz)
    verts, joints = mano(pred.pose, pred.trans, pred.shape)

    return (
        verts[0].cpu().numpy(),
        joints[0].cpu().numpy(),
        {
            "pose": pred.pose[0].cpu().numpy(),
            "trans": pred.trans[0].cpu().numpy(),
            "shape": pred.shape[0].cpu().numpy(),
        },
        tokens[0].cpu().numpy(),
    )


def main():
    parser = argparse.ArgumentParser("Generate grasps (v14, point encoder)")

    parser.add_argument("--diffusion_ckpt", type=str, required=True)
    parser.add_argument("--decoder_ckpt", type=str, required=True)

    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--oi_category", type=str, default="all")
    parser.add_argument("--oi_intent", type=str, default="all")
    parser.add_argument("--category", dest="oi_category", type=str)
    parser.add_argument("--intent", dest="oi_intent", type=str)
    parser.add_argument("--n_pc_points", type=int, default=2048)

    parser.add_argument("--decoder_config", type=str, default="base")
    parser.add_argument("--denoiser_config", type=str, default="base")
    parser.add_argument("--obj_global_dim", type=int, default=256)
    parser.add_argument("--obj_point_dim", type=int, default=256)
    parser.add_argument("--obj_xyz_dim", type=int, default=3)
    parser.add_argument("--n_betas", type=int, default=10)
    parser.add_argument("--text_feat_dim", type=int, default=512)
    parser.add_argument("--use_text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vocab_size", type=int, default=112)
    parser.add_argument("--pad_token_id", type=int, default=108)
    parser.add_argument("--mask_token_id", type=int, default=109)
    parser.add_argument("--seq_length", type=int, default=12)
    parser.add_argument("--num_timesteps", type=int, default=100)
    parser.add_argument("--mano_assets_root", type=str, default="assets/mano_v1_2")

    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--guidance_scale", type=float, default=2.0)

    parser.add_argument("--text_cache_dir", type=str, default=None)
    parser.add_argument("--no_text", action="store_true")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")

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

    logger.info("=" * 60)
    logger.info(f"  Split:           {args.split}")
    logger.info(f"  Samples:         {n_items}")
    logger.info(f"  PointEnc:        global={args.obj_global_dim} point={args.obj_point_dim}")
    logger.info(f"  Temperature:     {args.temperature}")
    logger.info(f"  Guidance scale:  {args.guidance_scale}")
    logger.info(f"  Text condition:  {text_mode}")
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

    for si in tqdm(range(n_items), desc="Generating grasps"):
        oi_sample = oi[si]

        obj_pc = torch.from_numpy(oi_sample["obj_verts"].astype(np.float32)).unsqueeze(0).to(device)
        obj_vn = torch.from_numpy(oi_sample["obj_vn"].astype(np.float32)).unsqueeze(0).to(device)
        obj_id = oi_sample.get("obj_id", f"obj_{si:05d}")

        if text_mode == "cache":
            text_str = idx_to_text.get(si, "no contact")
            text_feat = clip_enc.encode([text_str])
        else:
            text_feat = None
            text_str = ""

        hand_verts, hand_joints, mano_params, tokens = generate_single(
            models, obj_pc, obj_vn, args, text_feat=text_feat
        )

        grasp_result = {
            "obj_id": obj_id,
            "sample_idx": si,
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
        "total_generated": total_generated,
        "elapsed_seconds": elapsed,
        "num_source_samples": n_items,
    }
    with open(os.path.join(args.save_dir, "generate_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)


if __name__ == "__main__":
    main()
