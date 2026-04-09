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
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial import cKDTree

from metrics.diversity import diversity_details, joints_to_diversity_feature
from models.object_normalization import normalize_object_points
from models.point_encoder import PointNet2Encoder
from models.token_denoiser import build_denoiser
from models.pose_decoder import (
    build_grab_model, RefineNet, sanitize_posegrab_state_dict_for_load,
)
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
    if args.dataset == "oishape":
        from data.oishape_dataset import OIShape

        oi = OIShape(OIShapeConfig(args.split, args.oi_category, args.oi_intent))
        oi.n_samples = args.n_pc_points
        logger.info(f"OIShape [{args.split}]: {len(oi)} samples")
        return oi

    if args.dataset == "affordpose":
        from data.affordpose_dataset import AffordPoseDataset

        ds = AffordPoseDataset(
            manifest_path=args.affordpose_manifest,
            n_samples=args.n_pc_points,
            keys=args.affordpose_keys,
            classes=args.affordpose_classes,
            affordances=args.affordpose_affordances,
            limit=args.affordpose_limit,
        )
        logger.info(f"AffordPose manifest [{args.split}]: {len(ds)} samples")
        return ds

    raise ValueError(f"Unsupported dataset: {args.dataset}")


def _normalize_obj_id_filter(obj_ids):
    if not obj_ids:
        return None

    normalized = []
    for value in obj_ids:
        if value is None:
            continue
        for token in str(value).split(","):
            token = token.strip()
            if token:
                normalized.append(token)

    if not normalized:
        return None
    return list(dict.fromkeys(normalized))


def build_generation_indices(oi, obj_ids=None):
    n_items = len(oi)
    obj_ids = _normalize_obj_id_filter(obj_ids)

    if not obj_ids:
        return list(range(n_items))

    selected = []
    target_ids = set(obj_ids)
    missing_ids = set(target_ids)

    grasp_list = getattr(oi, "grasp_list", None)
    if grasp_list is not None:
        for idx, grasp in enumerate(grasp_list):
            obj_id = grasp.get("obj_id")
            if obj_id in target_ids:
                selected.append(idx)
                missing_ids.discard(obj_id)
    else:
        for idx in range(n_items):
            candidates = None
            if hasattr(oi, "get_match_obj_ids"):
                candidates = set(str(x) for x in oi.get_match_obj_ids(idx))
            else:
                candidates = {oi.get_obj_id(idx)}
            if candidates & target_ids:
                selected.append(idx)
                missing_ids -= candidates

    if missing_ids:
        logger.warning(
            "Requested obj_id(s) not found under current split/category/intent: "
            + ", ".join(sorted(missing_ids))
        )

    if not selected:
        raise ValueError(
            "No samples matched the requested obj_id filter: "
            + ", ".join(obj_ids)
        )

    logger.info(
        f"obj_id filter: {len(obj_ids)} requested, "
        f"{len(selected)} matching samples, "
        f"{len({oi.get_obj_id(i) for i in selected})} unique objects"
    )

    return selected


def maybe_subsample_generation_indices(indices, max_samples=None, sample_seed=42):
    if max_samples is None:
        return indices

    max_samples = int(max_samples)
    if max_samples <= 0:
        raise ValueError("--max_samples must be >= 1")
    if len(indices) <= max_samples:
        return indices

    rng = np.random.default_rng(int(sample_seed))
    picked = rng.choice(len(indices), size=max_samples, replace=False)
    picked = np.sort(picked)
    return [indices[i] for i in picked.tolist()]


def maybe_subsample_affordpose_per_category(
    dataset,
    indices,
    instances_per_category=None,
    sample_seed=42,
):
    """Randomly keep at most N AffordPose entries per class_name."""
    if instances_per_category is None:
        return indices, None

    instances_per_category = int(instances_per_category)
    if instances_per_category <= 0:
        raise ValueError("--affordpose_instances_per_category must be >= 1")

    category_to_indices = defaultdict(list)
    entries = getattr(dataset, "entries", None)
    for idx in indices:
        if entries is not None:
            class_name = str(entries[idx]["class_name"])
        else:
            class_name = str(dataset[idx].get("class_name", dataset[idx].get("cate_id", "unknown")))
        category_to_indices[class_name].append(int(idx))

    rng = np.random.default_rng(int(sample_seed))
    selected = []
    stats = {}
    total_available = 0

    for class_name in sorted(category_to_indices):
        class_indices = category_to_indices[class_name]
        total_available += len(class_indices)
        if len(class_indices) > instances_per_category:
            picked = rng.choice(class_indices, size=instances_per_category, replace=False)
            picked = sorted(int(x) for x in picked.tolist())
        else:
            picked = sorted(class_indices)
        selected.extend(picked)
        stats[class_name] = {
            "available": int(len(class_indices)),
            "selected": int(len(picked)),
        }

    selected.sort()
    summary = {
        "instances_per_category": int(instances_per_category),
        "num_categories": int(len(category_to_indices)),
        "total_available": int(total_available),
        "total_selected": int(len(selected)),
        "categories": stats,
    }
    return selected, summary


def _matches_prefix(key, prefixes):
    return any(key == prefix or key.startswith(f"{prefix}.") for prefix in prefixes)


def _load_decoder_stage(
    decoder,
    ckpt_path,
    device,
    only_prefixes,
    stage_name,
):
    logger.info(f"Loading decoder {stage_name}: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    raw_model_state = state["model"]
    filtered_state = {
        k: v for k, v in raw_model_state.items()
        if _matches_prefix(k, only_prefixes)
    }
    model_state, dropped = sanitize_posegrab_state_dict_for_load(
        filtered_state, decoder.state_dict()
    )
    if dropped:
        logger.warning(f"  Dropped {len(dropped)} incompatible decoder tensors from {stage_name}")
        for key, src_shape, dst_shape in dropped[:5]:
            logger.warning(f"    shape mismatch: {key} ckpt={src_shape} model={dst_shape}")
    missing, unexpected = decoder.load_state_dict(model_state, strict=False)
    relevant_missing = [k for k in missing if _matches_prefix(k, only_prefixes)]
    relevant_unexpected = [k for k in unexpected if _matches_prefix(k, only_prefixes)]
    if relevant_missing:
        logger.warning(
            f"  Decoder {stage_name} load: {len(relevant_missing)} missing keys "
            f"(first 5: {relevant_missing[:5]})"
        )
    if relevant_unexpected:
        logger.warning(
            f"  Decoder {stage_name} load: {len(relevant_unexpected)} unexpected keys "
            f"(first 5: {relevant_unexpected[:5]})"
        )
    return state



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

    cnet_state = torch.load(args.decoder_cnet_ckpt, map_location=device)
    dec_cfg = cnet_state.get("config", {})

    decoder = build_grab_model(
        dec_cfg.get("model_config", args.decoder_config),
        vocab_size=dec_cfg.get("vocab_size", vocab_size),
        n_betas=dec_cfg.get("n_betas", args.n_betas),
    ).to(device)
    if dec_cfg.get("no_token_cond", False):
        decoder.no_token_cond = True
        logger.info("  Decoder ablation from checkpoint: token conditioning DISABLED")
    if dec_cfg.get("no_part_target_pos", False) or getattr(args, "no_part_target_pos", False):
        decoder.no_part_target_pos = True
        logger.info("  Decoder ablation: part_target_pos ZEROED")
    if getattr(args, "posterior_refine_steps", 0) > 0:
        logger.info(f"  Posterior refinement: {args.posterior_refine_steps} steps")

    _load_decoder_stage(
        decoder, args.decoder_cnet_ckpt, device,
        only_prefixes=("token_encoder", "coarse_net"),
        stage_name="CoarseNet"
    )
    disable_refine = bool(dec_cfg.get("no_refine", False) or getattr(args, "no_refine", False))
    has_refine = False
    if disable_refine:
        logger.info("  Decoder ablation: RefineNet DISABLED (coarse-only inference)")
    elif args.decoder_rnet_ckpt and os.path.isfile(args.decoder_rnet_ckpt):
        rnet_state = torch.load(args.decoder_rnet_ckpt, map_location=device)
        if cnet_state.get("config", {}) and rnet_state.get("config", {}):
            cnet_model_cfg = cnet_state["config"].get("model_config")
            rnet_model_cfg = rnet_state["config"].get("model_config")
            if cnet_model_cfg != rnet_model_cfg:
                logger.warning(
                    f"  Decoder config mismatch: best_cnet={cnet_model_cfg} "
                    f"best_rnet={rnet_model_cfg}"
                )
        _load_decoder_stage(
            decoder, args.decoder_rnet_ckpt, device,
            only_prefixes=("refine_net",),
            stage_name="RefineNet"
        )
        has_refine = True
        logger.info("  Decoder loaded from separate best_cnet.pt + best_rnet.pt ✓")
    else:
        logger.warning("  decoder_rnet_ckpt not found; running coarse-only inference")

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
    """Compute BPS distances and nearest normalized object points.

    Args:
        obj_pc: (1, N, 3) object point cloud
        bps_basis: (K, 3) BPS basis points (K=4096 typically)
    Returns:
        bps_object: (1, K) BPS distance encoding
        bps_nn_points: (1, K, 3) nearest normalized object points
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
    result = bps.encode(obj_pc_norm.squeeze(0), feature_type=['dists', 'closest'])
    return (
        result['dists'].reshape(1, -1),
        result['closest'].reshape(1, -1, 3),
    )


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

    has_nn_points = False
    for fname in sorted(os.listdir(slot_dir)):
        if not fname.endswith(".npz"):
            continue
        npz_path = os.path.join(slot_dir, fname)
        with np.load(npz_path, allow_pickle=True) as data:
            has_nn_points = "bps_nn_points" in data.files
        break

    # Just store the directory path; load individual files on demand
    logger.info(f"BPS slot cache: {slot_dir}")
    if not has_nn_points:
        logger.info(
            "  Cache format is legacy: bps_nn_points not found. "
            "Re-run preprocess/build_bps_slot_cache.py if you need nearest normalized object points."
        )
    return {"dir": slot_dir, "cache": {}, "has_nn_points": has_nn_points}


def get_bps_slot_data(bps_slot_cache, obj_id, device):
    """Get per-object BPS slot data from cache.

    Returns:
        (bps_dists, bps_slot_labels, bps_nn_points) tensors,
        where bps_nn_points may be None for legacy caches.
    """
    if bps_slot_cache is None or not obj_id:
        return None, None, None

    cache = bps_slot_cache["cache"]
    if obj_id not in cache:
        npz_path = os.path.join(bps_slot_cache["dir"], f"{obj_id}.npz")
        if os.path.exists(npz_path):
            data = np.load(npz_path, allow_pickle=True)
            cache[obj_id] = {
                "bps_dists": data["bps_dists"].astype(np.float32),
                "bps_slot_labels": data["bps_slot_labels"].astype(np.int64),
                "bps_nn_points": (
                    data["bps_nn_points"].astype(np.float32)
                    if "bps_nn_points" in data.files else None
                ),
            }
        else:
            cache[obj_id] = None

    entry = cache.get(obj_id)
    if entry is None:
        return None, None, None

    bps_dists = torch.from_numpy(entry["bps_dists"]).unsqueeze(0).to(device)
    bps_slot_labels = torch.from_numpy(entry["bps_slot_labels"]).unsqueeze(0).to(device)
    bps_nn_points = None
    if entry["bps_nn_points"] is not None:
        bps_nn_points = torch.from_numpy(entry["bps_nn_points"]).unsqueeze(0).to(device)
    return bps_dists, bps_slot_labels, bps_nn_points


def _build_group_id(oi_sample, sample_idx: int, obj_id: str) -> str:
    if "manifest_key" in oi_sample:
        return str(oi_sample["manifest_key"])
    return f"{obj_id}_{sample_idx:05d}"


def _get_text_string_for_index(oi, si, text_mode, idx_to_text):
    if text_mode == "cache":
        return idx_to_text.get(si, "no contact")
    if text_mode == "sample":
        oi_sample = oi[si]
        return oi_sample.get("text_prompt", "")
    return ""


def build_text2grasp_prompt_plan(oi, generation_indices, args, text_mode, idx_to_text):
    if text_mode == "none":
        raise ValueError("Text2Grasp-style diversity requires text prompts; do not use --no_text")

    obj_to_rep_idx = {}
    obj_to_prompt_pool = defaultdict(list)
    obj_to_prompt_set = defaultdict(set)

    for si in generation_indices:
        if hasattr(oi, "get_obj_id"):
            obj_id = str(oi.get_obj_id(si))
        else:
            obj_id = str(oi[si].get("obj_id", f"obj_{si:05d}"))

        if obj_id not in obj_to_rep_idx:
            obj_to_rep_idx[obj_id] = si

        prompt = str(_get_text_string_for_index(oi, si, text_mode, idx_to_text) or "").strip()
        if not prompt or prompt.lower() == "no contact":
            continue
        if prompt not in obj_to_prompt_set[obj_id]:
            obj_to_prompt_set[obj_id].add(prompt)
            obj_to_prompt_pool[obj_id].append(prompt)

    object_ids = sorted(obj_to_rep_idx.keys())
    object_ids = maybe_subsample_generation_indices(
        object_ids,
        max_samples=args.max_samples,
        sample_seed=args.sample_seed,
    )

    rng = np.random.default_rng(int(args.sample_seed))
    plans = []
    prompt_pool_sizes = []
    replacement_count = 0

    for obj_id in object_ids:
        prompt_pool = obj_to_prompt_pool.get(obj_id, [])
        if not prompt_pool:
            prompt_pool = [str(_get_text_string_for_index(oi, obj_to_rep_idx[obj_id], text_mode, idx_to_text) or "").strip()]
            prompt_pool = [p for p in prompt_pool if p]
        if not prompt_pool:
            logger.warning(f"Text2Grasp diversity: no prompt available for obj_id={obj_id}; skipping")
            continue

        replace = len(prompt_pool) < args.num_grasps_per_sample
        if replace:
            replacement_count += 1
        prompt_indices = rng.choice(len(prompt_pool), size=args.num_grasps_per_sample, replace=replace)
        selected_prompts = [prompt_pool[i] for i in prompt_indices.tolist()]

        prompt_pool_sizes.append(len(prompt_pool))
        plans.append({
            "si": obj_to_rep_idx[obj_id],
            "obj_id": obj_id,
            "group_id": obj_id,
            "prompt_pool_size": len(prompt_pool),
            "uses_replacement": bool(replace),
            "prompts": selected_prompts,
        })

    stats = {
        "num_objects": len(plans),
        "prompt_pool_min": int(min(prompt_pool_sizes)) if prompt_pool_sizes else 0,
        "prompt_pool_mean": float(np.mean(prompt_pool_sizes)) if prompt_pool_sizes else 0.0,
        "prompt_pool_max": int(max(prompt_pool_sizes)) if prompt_pool_sizes else 0,
        "objects_with_prompt_replacement": int(replacement_count),
    }
    return plans, stats


def _compute_group_diversity(hand_joints_list, args):
    features = []
    for hand_joints in hand_joints_list:
        feat = joints_to_diversity_feature(
            hand_joints,
            unit=args.diversity_unit,
            canonical=args.diversity_canonical,
        )
        if feat is not None:
            features.append(feat)

    details = diversity_details(features, cls_num=args.diversity_cls_num)
    details["num_samples"] = len(features)
    details["unit"] = args.diversity_unit
    details["canonical"] = bool(args.diversity_canonical)
    return details


def _write_diversity_summary(save_dir, records, args):
    if not records:
        return None

    entropy_vals = [r["entropy"] for r in records]
    cluster_vals = [r["cluster_size"] for r in records]
    active_counts = [r["num_active_clusters"] for r in records]
    cluster_spread_vals = [r.get("cluster_spread", 0.0) for r in records]

    summary = {
        "num_groups": len(records),
        "num_grasps_per_sample": args.num_grasps_per_sample,
        "cls_num": args.diversity_cls_num,
        "unit": args.diversity_unit,
        "canonical": bool(args.diversity_canonical),
        "cluster_size_definition": "mean number of samples per active cluster after k-means",
        "cluster_spread_definition": f"mean distance to assigned k-means center ({args.diversity_unit})",
        "mean_entropy": float(np.mean(entropy_vals)),
        "std_entropy": float(np.std(entropy_vals)),
        "mean_cluster_size": float(np.mean(cluster_vals)),
        "std_cluster_size": float(np.std(cluster_vals)),
        "mean_cluster_spread": float(np.mean(cluster_spread_vals)),
        "std_cluster_spread": float(np.std(cluster_spread_vals)),
        "mean_active_clusters": float(np.mean(active_counts)),
        "records": records,
    }

    json_path = os.path.join(save_dir, "diversity_metrics.json")
    txt_path = os.path.join(save_dir, "diversity_metrics.txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("DIVERSITY SUMMARY\n")
        f.write(f"Groups: {summary['num_groups']}\n")
        f.write(f"Samples per group: {summary['num_grasps_per_sample']}\n")
        f.write(f"KMeans clusters (max): {summary['cls_num']}\n")
        f.write(f"Unit: {summary['unit']}\n")
        f.write(f"Canonical: {int(summary['canonical'])}\n")
        f.write(f"Cluster Size Definition: {summary['cluster_size_definition']}\n")
        f.write(f"Cluster Spread Definition: {summary['cluster_spread_definition']}\n")
        f.write(f"Mean Entropy (ln): {summary['mean_entropy']:.4f}\n")
        f.write(f"Std Entropy (ln): {summary['std_entropy']:.4f}\n")
        f.write(f"Mean Cluster Size: {summary['mean_cluster_size']:.4f}\n")
        f.write(f"Std Cluster Size: {summary['std_cluster_size']:.4f}\n")
        f.write(f"Mean Cluster Spread: {summary['mean_cluster_spread']:.4f}\n")
        f.write(f"Std Cluster Spread: {summary['std_cluster_spread']:.4f}\n")
        f.write(f"Mean Active Clusters: {summary['mean_active_clusters']:.4f}\n")

    return summary


def _score_grasp(hand_verts, hand_joints, obj_pc_np, obj_vn_np):
    """Score a grasp candidate. Lower is better.

    The score is intentionally biased toward three properties:
      1. low penetration,
      2. moderate surface contact rather than "touch as many vertices as possible",
      3. better quasi-stability via fingertip support and spatially spread contacts.

    This is still only a lightweight proxy for simulation stability, but it
    behaves better than the previous "more contact is always better" score.

    Args:
        hand_verts: (778, 3) hand vertices.
        hand_joints: (21, 3) MANO joints.
        obj_pc_np: (N, 3) object point cloud.
        obj_vn_np: (N, 3) object point normals.
    """
    from scipy.spatial import cKDTree

    def _signed_distance(query_points, obj_tree):
        dists, idx = obj_tree.query(query_points, k=1)
        nearest_pts = obj_pc_np[idx]
        nearest_normals = obj_vn_np[idx]
        diff = query_points - nearest_pts
        sign = np.sum(diff * nearest_normals, axis=1)
        signed = np.where(sign >= 0, dists, -dists)
        return signed, dists, nearest_pts

    obj_tree = cKDTree(obj_pc_np)
    signed_dist, dists, nearest_pts = _signed_distance(hand_verts, obj_tree)

    # Penalize both the total penetration depth and how many vertices go inside.
    penetrating = signed_dist < -0.001
    pen_depth = float(np.sum(np.abs(signed_dist[penetrating]))) if penetrating.any() else 0.0
    pen_ratio = float(np.mean(penetrating))
    max_pen = float(np.max(np.abs(signed_dist[penetrating]))) if penetrating.any() else 0.0

    # Encourage moderate contact: enough support to hold the object, but not an
    # over-compressed "all available vertices glued to the surface" solution.
    contact_mask = (signed_dist >= 0.0) & (signed_dist < 0.005)
    close_mask = (signed_dist >= 0.0) & (signed_dist < 0.012)
    contact_ratio = float(np.mean(contact_mask))
    close_ratio = float(np.mean(close_mask))

    target_contact = 0.08
    low_contact_pen = max(0.0, target_contact - contact_ratio)
    high_contact_pen = max(0.0, contact_ratio - 0.18)

    if close_mask.any():
        avg_close_dist = float(np.mean(dists[close_mask]))
    else:
        avg_close_dist = 0.012

    # Stability proxies:
    # - fingertip support: count fingertips that are just outside the surface
    # - contact spread: reward contacts distributed over a wider region
    fingertip_ids = np.asarray([4, 8, 12, 16, 20], dtype=np.int64)
    tip_signed, _, _ = _signed_distance(hand_joints[fingertip_ids], obj_tree)
    tip_contact_count = int(np.sum((tip_signed >= 0.0) & (tip_signed < 0.010)))

    contact_points = nearest_pts[contact_mask]
    if contact_points.shape[0] >= 3:
        contact_centroid = contact_points.mean(axis=0, keepdims=True)
        contact_spread = float(
            np.mean(np.linalg.norm(contact_points - contact_centroid, axis=1))
        )
    else:
        contact_spread = 0.0
    contact_spread = min(contact_spread, 0.03)

    score = 0.0
    score += 240.0 * pen_depth
    score += 18.0 * pen_ratio
    score += 12.0 * max_pen
    score += 10.0 * low_contact_pen
    score += 4.0 * high_contact_pen
    score += 10.0 * max(0.0, 0.06 - close_ratio)
    score += 20.0 * avg_close_dist
    score -= 0.45 * tip_contact_count
    score -= 18.0 * contact_spread
    score -= 1.5 * min(close_ratio, 0.18)
    return float(score)



@torch.no_grad()
def generate_single(models, obj_pc, obj_vn, args, text_feat=None,
                    bps_dists_cached=None,
                    bps_slot_labels_cached=None,
                    bps_nn_points_cached=None):
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

    bps_object = bps_dists_cached
    bps_slot_labels = bps_slot_labels_cached
    bps_nn_points = bps_nn_points_cached

    if bps_object is None or bps_nn_points is None:
        bps_object_online, bps_nn_points_online = _compute_bps(obj_pc, bps_basis)
        if bps_object is None:
            bps_object = bps_object_online
        if bps_nn_points is None:
            bps_nn_points = bps_nn_points_online

    best_of_n = int(getattr(args, "best_of_n", 1))
    best_of_n = max(best_of_n, 1)
    if best_of_n > 1:
        obj_pc_np = obj_pc[0].cpu().numpy()
        obj_vn_np = obj_vn[0].cpu().numpy()
    else:
        obj_pc_np = obj_vn_np = None

    best_candidate = None
    best_score = float("inf")
    best_trial = 0

    for trial_idx in range(best_of_n):
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

        pred = decoder.sample(
            bps_object, tokens, inferred_mask,
            bps_slot_labels=bps_slot_labels,
            bps_nn_points=bps_nn_points,
            posterior_refine_steps=getattr(args, "posterior_refine_steps", 0),
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
        hand_verts = verts[0].detach().cpu().numpy()
        hand_joints = joints[0].detach().cpu().numpy()

        candidate = (
            hand_verts,
            hand_joints,
            {
                "pose": final_pose[0].detach().cpu().numpy(),
                "trans": final_trans[0].detach().cpu().numpy(),
                "shape": final_shape[0].detach().cpu().numpy(),
                "best_of_n_score": None,
                "best_of_n_trial": trial_idx,
            },
            tokens[0].detach().cpu().numpy(),
        )

        if best_of_n == 1:
            return candidate

        score = _score_grasp(hand_verts, hand_joints, obj_pc_np, obj_vn_np)
        if score < best_score:
            best_score = score
            best_trial = trial_idx
            best_candidate = candidate

    best_hand_verts, best_hand_joints, best_params, best_tokens = best_candidate
    best_params["best_of_n_score"] = float(best_score)
    best_params["best_of_n_trial"] = int(best_trial)
    return best_hand_verts, best_hand_joints, best_params, best_tokens


def main():
    parser = argparse.ArgumentParser("Generate grasps (v18, diffusion + CVAE decoder + RefineNet)")

    parser.add_argument("--diffusion_ckpt", type=str, default="checkpoints/full/diffusion/best.pt")
    parser.add_argument("--decoder_cnet_ckpt", type=str, default="checkpoints/w_pen/decoder/best_cnet.pt")
    parser.add_argument("--decoder_rnet_ckpt", type=str, default="checkpoints/w_pen/decoder/best_rnet.pt")

    parser.add_argument("--dataset", type=str, default="oishape",
                        choices=["oishape", "affordpose"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--oi_category", type=str, default="all")
    parser.add_argument("--oi_intent", type=str, default="all")
    parser.add_argument("--category", dest="oi_category", type=str)
    parser.add_argument("--intent", dest="oi_intent", type=str)
    parser.add_argument(
        "--obj_id",
        type=str,
        nargs="+",
        default=None,
        help="Only generate samples whose obj_id matches one of these ids. "
             "Supports multiple values and comma-separated ids.",
    )
    parser.add_argument("--n_pc_points", type=int, default=10000)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Randomly subsample at most this many source samples before generation")
    parser.add_argument("--sample_seed", type=int, default=42,
                        help="Random seed used by --max_samples subsampling")
    parser.add_argument("--affordpose_manifest", type=str,
                        default="configs/affordpose_test_manifest.json")
    parser.add_argument("--affordpose_keys", nargs="+", default=None,
                        help="Only use specific AffordPose manifest keys")
    parser.add_argument("--affordpose_classes", nargs="+", default=None,
                        help="Filter AffordPose entries by class_name")
    parser.add_argument("--affordpose_affordances", nargs="+", default=None,
                        help="Filter AffordPose entries by afford_name")
    parser.add_argument("--affordpose_limit", type=int, default=None,
                        help="Limit the number of AffordPose manifest entries")
    parser.add_argument("--affordpose_instances_per_category", type=int, default=None,
                        help="For AffordPose only, randomly keep at most this many entries per class_name after filtering; uses --sample_seed")

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

    # Slot-grounded BPS cache
    parser.add_argument("--bps_slot_cache_dir", type=str, default="cache/bps_slot",
                        help="Per-object BPS slot cache directory")

    parser.add_argument("--save_dir", type=str, default="experiments/ablation_full")
    parser.add_argument("--save_mesh", action="store_true")
    parser.add_argument("--diversity_mode", action="store_true",
                        help="Generate multiple grasps per input and summarize diversity")
    parser.add_argument("--text2grasp_diversity", action="store_true",
                        help="Text2Grasp-style diversity: for each object, sample multiple prompts from that object's prompt pool")
    parser.add_argument("--num_grasps_per_sample", type=int, default=1,
                        help="Number of grasps to generate for each source sample")
    parser.add_argument("--repeat_per_sample", type=int, default=None,
                        help="Alias for --num_grasps_per_sample: repeat sampling each source sample N times")
    parser.add_argument("--diversity_cls_num", type=int, default=20,
                        help="Upper bound of KMeans cluster count for diversity mode")
    parser.add_argument("--diversity_unit", type=str, default="cm",
                        choices=["m", "cm", "mm"],
                        help="Unit scaling applied before diversity clustering")
    parser.add_argument("--diversity_canonical", action=argparse.BooleanOptionalAction, default=True,
                        help="Canonicalize joints before diversity clustering")

    parser.add_argument("--posterior_refine_steps", type=int, default=0,
                        help="Iterative posterior refinement steps at inference (0=disabled)")
    parser.add_argument("--best_of_n", type=int, default=1,
                        help="Generate N candidates per saved grasp and keep the best (1=disabled)")
    parser.add_argument("--no_refine", action="store_true",
                        help="Disable RefineNet even if a refine checkpoint is provided")
    parser.add_argument("--no_part_target_pos", action="store_true",
                        help="Zero out slot-derived part_target_pos at inference")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.repeat_per_sample is not None:
        if args.repeat_per_sample < 1:
            raise ValueError("--repeat_per_sample must be >= 1")
        if args.num_grasps_per_sample != 1 and args.num_grasps_per_sample != args.repeat_per_sample:
            raise ValueError(
                "--repeat_per_sample conflicts with --num_grasps_per_sample; "
                "please specify only one of them or give them the same value"
            )
        args.num_grasps_per_sample = int(args.repeat_per_sample)

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    results_dir = os.path.join(args.save_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    if args.diversity_mode and args.num_grasps_per_sample < 2:
        raise ValueError("--diversity_mode requires --num_grasps_per_sample >= 2")
    if args.num_grasps_per_sample < 1:
        raise ValueError("--num_grasps_per_sample must be >= 1")
    if args.best_of_n < 1:
        raise ValueError("--best_of_n must be >= 1")
    if args.text2grasp_diversity and args.num_grasps_per_sample < 2:
        raise ValueError("--text2grasp_diversity requires --num_grasps_per_sample >= 2")
    if args.affordpose_instances_per_category is not None and args.dataset != "affordpose":
        logger.warning("--affordpose_instances_per_category is only used when --dataset affordpose; ignoring it")

    models = load_models(args, device)

    clip_enc = models["clip_encoder"]
    text_mode = None
    idx_to_text = {}

    if args.no_text:
        text_mode = "none"
        logger.info("Text condition: disabled")
    elif args.dataset == "oishape":
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
    elif args.dataset == "affordpose":
        text_mode = "sample"
        logger.info("Text condition: AffordPose auto prompt")

    if text_mode is None:
        logger.error("Text condition required. Use --no_text or --text_cache_dir")
        sys.exit(1)

    oi = load_test_data(args)
    n_items = len(oi)
    generation_indices = build_generation_indices(oi, args.obj_id)
    n_candidates_before_category_subsample = len(generation_indices)
    affordpose_category_sample_stats = None
    if args.dataset == "affordpose":
        generation_indices, affordpose_category_sample_stats = maybe_subsample_affordpose_per_category(
            oi,
            generation_indices,
            instances_per_category=args.affordpose_instances_per_category,
            sample_seed=args.sample_seed,
        )
    n_candidates = len(generation_indices)

    text2grasp_plan = None
    text2grasp_stats = None
    if args.text2grasp_diversity:
        text2grasp_plan, text2grasp_stats = build_text2grasp_prompt_plan(
            oi, generation_indices, args, text_mode, idx_to_text
        )
        n_generate = len(text2grasp_plan)
    else:
        generation_indices = maybe_subsample_generation_indices(
            generation_indices,
            max_samples=args.max_samples,
            sample_seed=args.sample_seed,
        )
        n_generate = len(generation_indices)

    # Load BPS slot cache for inference (per-object BPS distances when available)
    bps_slot_cache = load_bps_slot_cache(args, args.split)

    logger.info("=" * 60)
    logger.info(f"  Dataset:         {args.dataset}")
    logger.info(f"  Split:           {args.split}")
    logger.info(f"  Source samples:  {n_items}")
    logger.info(f"  Candidates:      {n_candidates_before_category_subsample}")
    if affordpose_category_sample_stats is not None:
        logger.info(
            f"  AffordPose cat:  {affordpose_category_sample_stats['instances_per_category']}/class "
            f"(seed={args.sample_seed}) -> {affordpose_category_sample_stats['total_selected']} samples "
            f"across {affordpose_category_sample_stats['num_categories']} classes"
        )
        logger.info(
            "  AffordPose sel:  "
            + ", ".join(
                f"{class_name}={stats['selected']}/{stats['available']}"
                for class_name, stats in affordpose_category_sample_stats["categories"].items()
            )
        )
    logger.info(f"  Generate count:  {n_generate}")
    if args.max_samples is not None:
        logger.info(f"  Max samples:     {args.max_samples} (seed={args.sample_seed})")
    if args.repeat_per_sample is not None:
        logger.info(f"  Repeat/sample:   {args.num_grasps_per_sample}")
    logger.info(f"  Grasp/sample:    {args.num_grasps_per_sample}")
    if args.text2grasp_diversity:
        logger.info("  Prompt mode:     Text2Grasp-style object prompt pool")
        logger.info(
            f"  Prompt pool:     min/mean/max="
            f"{text2grasp_stats['prompt_pool_min']}/"
            f"{text2grasp_stats['prompt_pool_mean']:.2f}/"
            f"{text2grasp_stats['prompt_pool_max']}"
        )
        logger.info(
            f"  Prompt replace:  {text2grasp_stats['objects_with_prompt_replacement']}"
            f" / {text2grasp_stats['num_objects']} objects"
        )
    if args.best_of_n > 1:
        logger.info(f"  Best-of-N:       {args.best_of_n}")
    if args.obj_id:
        logger.info(
            "  obj_id filter:   " + ", ".join(_normalize_obj_id_filter(args.obj_id))
        )
    decoder_desc = "GrabNet-style CVAE + RefineNet" if models["has_refine"] else "GrabNet-style CVAE (coarse only)"
    logger.info(f"  Decoder:         {decoder_desc}")
    logger.info(f"  Temperature:     {args.temperature}")
    logger.info(f"  Guidance scale:  {args.guidance_scale}")
    logger.info(f"  Text condition:  {text_mode}")
    if args.diversity_mode:
        logger.info(
            f"  Diversity:       enabled "
            f"(k={args.diversity_cls_num}, unit={args.diversity_unit}, "
            f"canonical={int(args.diversity_canonical)})"
        )
    logger.info(
        f"  BPS slot cache:  "
        f"{'loaded (cached BPS dists when available)' if bps_slot_cache else 'none (compute BPS online)'}"
    )
    logger.info(f"  Output:          {results_dir}")
    logger.info("=" * 60)

    t0 = time.time()
    total_generated = 0
    diversity_records = []

    try:
        import trimesh
        has_trimesh = True
    except ImportError:
        has_trimesh = False
        if args.save_mesh:
            logger.warning("trimesh not available, --save_mesh disabled")
            args.save_mesh = False

    if args.text2grasp_diversity:
        iterator = enumerate(tqdm(text2grasp_plan, desc="Generating grasps"))
    else:
        iterator = enumerate(tqdm(generation_indices, desc="Generating grasps"))

    for gen_idx, item_ref in iterator:
        if args.text2grasp_diversity:
            plan_item = item_ref
            si = plan_item["si"]
            oi_sample = oi[si]
        else:
            si = item_ref
            plan_item = None
            oi_sample = oi[si]

        obj_pc = torch.from_numpy(oi_sample["obj_verts"].astype(np.float32)).unsqueeze(0).to(device)
        obj_vn = torch.from_numpy(oi_sample["obj_vn"].astype(np.float32)).unsqueeze(0).to(device)
        obj_id = oi_sample.get("obj_id", f"obj_{si:05d}")
        cache_obj_id = oi_sample.get("cache_obj_id", obj_id)

        # BPS slot cache lookup by obj_id
        bps_dists_cached, bps_slot_labels_cached, bps_nn_points_cached = get_bps_slot_data(
            bps_slot_cache, cache_obj_id, device)

        if text_mode == "cache":
            text_str = idx_to_text.get(si, "no contact")
            text_feat = clip_enc.encode([text_str]) if text_str else None
        elif text_mode == "sample":
            text_str = oi_sample.get("text_prompt", "")
            text_feat = clip_enc.encode([text_str]) if text_str else None
        else:
            text_feat = None
            text_str = ""

        group_hand_joints = []
        if args.text2grasp_diversity:
            group_id = plan_item["group_id"]
            prompt_list = plan_item["prompts"]
        else:
            group_id = _build_group_id(oi_sample, si, obj_id)
            prompt_list = None

        for rep_idx in range(args.num_grasps_per_sample):
            if prompt_list is not None:
                text_str = prompt_list[rep_idx]
                text_feat = clip_enc.encode([text_str]) if text_str else None
            hand_verts, hand_joints, mano_params, tokens = generate_single(
                models, obj_pc, obj_vn, args, text_feat=text_feat,
                bps_dists_cached=bps_dists_cached,
                bps_slot_labels_cached=bps_slot_labels_cached,
                bps_nn_points_cached=bps_nn_points_cached,
            )

            grasp_result = {
                "dataset": args.dataset,
                "obj_id": obj_id,
                "cache_obj_id": cache_obj_id,
                "raw_obj_id": oi_sample.get("raw_obj_id", obj_id),
                "sample_idx": si,
                "generation_idx": rep_idx,
                "group_id": group_id,
                "hand_verts_r": hand_verts,
                "hand_joints_r": hand_joints,
                "hand_pose": mano_params["pose"],
                "hand_trans": mano_params["trans"],
                "hand_shape": mano_params["shape"],
                "best_of_n_score": mano_params.get("best_of_n_score"),
                "best_of_n_trial": mano_params.get("best_of_n_trial"),
                "tokens": tokens,
                "obj_rotmat": np.eye(3, dtype=np.float32),
                "text": text_str,
                "text_prompt": text_str,
            }
            if args.text2grasp_diversity:
                grasp_result["text2grasp_diversity"] = True
                grasp_result["prompt_pool_size"] = int(plan_item["prompt_pool_size"])
                grasp_result["prompt_uses_replacement"] = bool(plan_item["uses_replacement"])
            for key in [
                "class_name",
                "afford_name",
                "manifest_key",
                "sample_json",
                "text_prompt_style",
                "text_target_slot_names",
                "text_target_slots",
                "text_primary_slot_name",
                "text_primary_slot",
            ]:
                if key in oi_sample:
                    grasp_result[key] = oi_sample[key]

            if args.num_grasps_per_sample == 1:
                fname = f"{obj_id}_{si:05d}"
            else:
                fname = f"{obj_id}_{si:05d}_g{rep_idx:03d}"
            pkl_path = os.path.join(results_dir, f"{fname}.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(grasp_result, f)

            if text_str:
                prompt_path = os.path.join(results_dir, f"{fname}_text_prompt.txt")
                with open(prompt_path, "w", encoding="utf-8") as f:
                    f.write(text_str)

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

            group_hand_joints.append(hand_joints)
            total_generated += 1

        if args.diversity_mode:
            details = _compute_group_diversity(group_hand_joints, args)
            details.update({
                "group_id": group_id,
                "obj_id": obj_id,
                "sample_idx": si,
            })
            if args.text2grasp_diversity:
                details["prompt_pool_size"] = int(plan_item["prompt_pool_size"])
                details["prompt_uses_replacement"] = bool(plan_item["uses_replacement"])
            for key in ["class_name", "afford_name", "manifest_key"]:
                if key in oi_sample:
                    details[key] = oi_sample[key]
            diversity_records.append(details)

    elapsed = time.time() - t0
    diversity_summary = None
    if args.diversity_mode:
        diversity_summary = _write_diversity_summary(args.save_dir, diversity_records, args)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Done! Generated {total_generated} grasps in {elapsed:.1f}s")
    logger.info(f"  ({elapsed / max(total_generated, 1):.2f}s per grasp)")
    logger.info(f"Results saved to: {results_dir}")
    if diversity_summary is not None:
        logger.info(
            "Diversity summary: "
            f"entropy={diversity_summary['mean_entropy']:.4f}, "
            f"cluster_size={diversity_summary['mean_cluster_size']:.4f}, "
            f"cluster_spread={diversity_summary['mean_cluster_spread']:.4f}, "
            f"active_clusters={diversity_summary['mean_active_clusters']:.2f}"
        )
        logger.info(f"  Saved to: {os.path.join(args.save_dir, 'diversity_metrics.json')}")
    logger.info("\nEvaluate with:")
    logger.info(f"  python evaluate_grasps.py --exp_path {args.save_dir}")

    meta = {
        "args": vars(args),
        "dataset": args.dataset,
        "decoder_type": "cvae+refine" if models["has_refine"] else "cvae",
        "total_generated": total_generated,
        "elapsed_seconds": elapsed,
        "num_source_samples": n_items,
        "num_candidate_indices_before_category_subsample": n_candidates_before_category_subsample,
        "num_candidate_indices": n_candidates,
        "num_generation_indices": n_generate,
        "num_unique_source_samples": n_generate if args.text2grasp_diversity else len(set(generation_indices)),
        "num_generation_groups": n_generate,
        "obj_id_filter": _normalize_obj_id_filter(args.obj_id),
        "max_samples": args.max_samples,
        "sample_seed": args.sample_seed,
        "repeat_per_sample": args.repeat_per_sample,
        "affordpose_instances_per_category": args.affordpose_instances_per_category,
        "affordpose_category_sample_stats": affordpose_category_sample_stats,
        "best_of_n": args.best_of_n,
        "diversity_mode": bool(args.diversity_mode),
        "text2grasp_diversity": bool(args.text2grasp_diversity),
        "text2grasp_stats": text2grasp_stats,
        "num_grasps_per_sample": args.num_grasps_per_sample,
    }
    with open(os.path.join(args.save_dir, "generate_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)


if __name__ == "__main__":
    main()
