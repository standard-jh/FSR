#!/usr/bin/env python3
"""Paper-style x2 generated-image metrics for RF and local LUA baselines.

This script follows the local LUA evaluation protocol more closely than the
5-image visual diagnostic:

- generated set: saved FLUX 1024 latent/prompt records
- target setting: x2, 1024 -> 2048
- real reference: cached OpenImages HR Inception full-image/patch features
- metrics: FID, pFID, KID, pKID, CLIP, and per-method runtime

It is still not the exact LUA paper table, because this repo evaluates FLUX
1024 saved latents and our RF x2 method. LSRNA-DemoFusion can be appended only
when a full matched output directory exists.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
LUA_ROOT = WORKSPACE / "LUA"
LUA_SCRIPT_DIR = LUA_ROOT / "test_scripts"
TRAIN_SCRIPT = ROOT / "scripts" / "train_feature_rectified_flow_sr.py"

for path in [str(LUA_ROOT), str(LUA_SCRIPT_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from eval_flux_latent_4096_pfid_pkid import (  # noqa: E402
    extract_tensor_patch_features,
    load_records,
    patch_boxes,
)
from eval_flux_latent_4096_real_openimages_metrics import fid_low_rank, kid_unbiased  # noqa: E402
from eval_lua_swinir_x4_pfid_pkid import InceptionPool  # noqa: E402
from eval_lua_swinir_x4_psnr import resolve_dtype, upscale_lua_safe  # noqa: E402
from lua import load_model as load_lua_model  # noqa: E402


DEFAULT_METHODS = ["bicubic_x2", "RF_f3_one_step_x2", "LUA_x2_to_2048"]


def load_train_module():
    spec = importlib.util.spec_from_file_location("feature_rf_train", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-style OpenImages metrics for x2 RF/LUA generated outputs.")
    parser.add_argument(
        "--latent_root",
        type=Path,
        default=WORKSPACE / "FALSR/output/flux_random_1024_merged_179_20260501",
    )
    parser.add_argument(
        "--real_feature_dir",
        type=Path,
        default=WORKSPACE / "LUA/experiments/eval_flux_latent_4096_real_openimages_metrics_179v150/features",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=WORKSPACE
        / "runs/feature_rectified_flow_x2_f3_resume_bench_wandb/train_main/checkpoints/last.pt",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "results/paper_style_openimages_x2",
    )
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_images", type=int, default=0)
    parser.add_argument("--vae_name_or_path", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--vae_subfolder", default="vae")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--lua_weights", type=str, default=None)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cut_name", default="f3")
    parser.add_argument("--feature_channels", type=int, default=512)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--enable_gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gate_channels", default="one")
    parser.add_argument("--gate_init_bias", type=float, default=-1.0)
    parser.add_argument("--vector_scale", type=float, default=1.0)
    parser.add_argument("--target_size", type=int, default=2048)
    parser.add_argument("--inception_size", type=int, default=299)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--patches_per_image", type=int, default=16)
    parser.add_argument("--feature_batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--compute_clip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    parser.add_argument(
        "--lsrna_output_root",
        type=Path,
        default=Path(""),
        help="Optional root with raw_outputs/generated/<sample>/lsrna_sdxl_trg2048.png.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def clear_cuda() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


@torch.inference_mode()
def decode_raw_vae(vae, latent: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    x_m11 = vae.decode(latent.to(device=device, dtype=dtype), return_dict=False)[0]
    return ((x_m11.float() + 1.0) / 2.0).clamp(0.0, 1.0)


def tensor_to_png(x_01: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (x_01[0].detach().float().permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def image_path_to_tensor(path: Path, device: torch.device, target_size: int) -> torch.Tensor:
    with Image.open(path) as raw:
        img = raw.convert("RGB")
    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), Image.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device)


def normalize_inception(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.inference_mode()
def inception_feature(model, image_01: torch.Tensor, device: torch.device, size: int) -> np.ndarray:
    x = image_01.detach().float().clamp(0.0, 1.0).to(device=device)
    if x.shape[-2:] != (size, size):
        x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False, antialias=True)
    x = normalize_inception(x.clamp(0.0, 1.0))
    return model(x).detach().float().cpu().numpy().astype(np.float64)


def build_clip(args: argparse.Namespace, device: torch.device):
    if not bool(args.compute_clip):
        return None
    try:
        from transformers import CLIPModel, CLIPTokenizer
    except Exception as exc:
        print(f"[WARN] CLIP unavailable: {exc}")
        return None
    try:
        model = CLIPModel.from_pretrained(args.clip_model, use_safetensors=False).to(device)
        tokenizer = CLIPTokenizer.from_pretrained(args.clip_model)
    except Exception as exc:
        print(f"[WARN] Failed to load CLIP model {args.clip_model}: {exc}")
        return None
    model.eval()
    model.requires_grad_(False)
    return {"model": model, "tokenizer": tokenizer}


@torch.inference_mode()
def clip_text_features(clip_bundle, prompts: list[str], device: torch.device) -> torch.Tensor | None:
    if clip_bundle is None:
        return None
    tokenizer = clip_bundle["tokenizer"]
    model = clip_bundle["model"]
    inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
    feats = model.get_text_features(**inputs)
    if not torch.is_tensor(feats):
        feats = feats.pooler_output
    return F.normalize(feats.float(), dim=-1).detach().cpu()


@torch.inference_mode()
def clip_score(clip_bundle, image_01: torch.Tensor, text_feat_cpu: torch.Tensor | None, device: torch.device) -> float:
    if clip_bundle is None or text_feat_cpu is None:
        return float("nan")
    model = clip_bundle["model"]
    x = image_01.detach().float().clamp(0.0, 1.0).to(device=device)
    x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=x.dtype, device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=x.dtype, device=device).view(1, 3, 1, 1)
    image_feat = model.get_image_features(pixel_values=(x - mean) / std)
    if not torch.is_tensor(image_feat):
        image_feat = image_feat.pooler_output
    image_feat = F.normalize(image_feat.float(), dim=-1)
    return float((image_feat * text_feat_cpu.to(device=device, dtype=torch.float32)).sum(dim=-1).mean().item())


def load_rf_model(args: argparse.Namespace, train, vae, device: torch.device):
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {})
    feature_channels = int(ckpt_args.get("feature_channels", args.feature_channels))
    hidden_channels = int(ckpt_args.get("hidden_channels", args.hidden_channels))
    num_blocks = int(ckpt_args.get("num_blocks", args.num_blocks))
    enable_gate = bool(ckpt_args.get("enable_gate", args.enable_gate))
    gate_channels = str(ckpt_args.get("gate_channels", args.gate_channels))
    gate_init_bias = float(ckpt_args.get("gate_init_bias", args.gate_init_bias))
    cut_name = str((ckpt.get("cut") or {}).get("name", args.cut_name))

    cut = train.resolve_cut(vae.decoder, cut_name, fallback_name=args.cut_name)
    model = train.FeatureRectifiedFlowNet(
        feature_channels=feature_channels,
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
        enable_gate=enable_gate,
        gate_channels=gate_channels,
        gate_init_bias=gate_init_bias,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    stats = train.FeatureStats(feature_channels, device)
    stats.load_state_dict(ckpt["feature_stats"], device)
    return cut, model, stats


@torch.inference_mode()
def run_rf_x2(train, vae, cut, model, stats, direct_1024_01: torch.Tensor, dtype: torch.dtype, amp_dtype: torch.dtype, vector_scale: float) -> torch.Tensor:
    lr_m11 = direct_1024_01.float() * 2.0 - 1.0
    with train.autocast_context(direct_1024_01.device, amp_dtype):
        z_l = train.encode_flux_scaled(vae, lr_m11, dtype)
        f_l = train.decoder_front_to_cut(vae, z_l, dtype, cut)
        f_b = train.interpolate_feature(f_l, (f_l.shape[-2] * 2, f_l.shape[-1] * 2))
        f_hat, _ = train.sample_feature_ode(model, stats, f_b, amp_dtype, 1, vector_scale, "euler")
        x_rf_m11 = train.decoder_tail_from_cut(vae, cut, f_hat.to(dtype=dtype)).float()
    return ((x_rf_m11 + 1.0) / 2.0).clamp(0.0, 1.0)


def maybe_append_lsrna_method(methods: list[str], root: Path) -> list[str]:
    if str(root) not in {"", "."} and (root / "raw_outputs/generated").exists() and "LSRNA_SDXL_x2_to_2048" not in methods:
        return [*methods, "LSRNA_SDXL_x2_to_2048"]
    return methods


def summarize(
    rows: list[dict[str, Any]],
    real_full: np.ndarray,
    real_patch: np.ndarray,
    full_features: dict[str, list[np.ndarray]],
    patch_features: dict[str, list[np.ndarray]],
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for method, parts in full_features.items():
        if not parts:
            continue
        full = np.concatenate(parts, axis=0).astype(np.float64)
        patch = np.concatenate(patch_features[method], axis=0).astype(np.float64)
        subset = [r for r in rows if r["method"] == method]
        kid = kid_unbiased(real_full, full)
        pkid = kid_unbiased(real_patch, patch)
        summary.append(
            {
                "resolution": "2048x2048",
                "method": method,
                "n_real_images": int(real_full.shape[0]),
                "n_generated_images": int(full.shape[0]),
                "n_real_patches": int(real_patch.shape[0]),
                "n_generated_patches": int(patch.shape[0]),
                "FID": fid_low_rank(real_full, full),
                "pFID": fid_low_rank(real_patch, patch),
                "KID": kid,
                "pKID": pkid,
                "CLIP": float(np.nanmean([float(r["clip"]) for r in subset])) if subset else float("nan"),
                "time_sec_mean": float(np.nanmean([float(r["time_sec"]) for r in subset])) if subset else float("nan"),
                "time_sec_median": float(np.nanmedian([float(r["time_sec"]) for r in subset])) if subset else float("nan"),
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.precision)
    amp_dtype = dtype
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and (args.output_dir / "summary.json").exists():
        print((args.output_dir / "summary.json").read_text(encoding="utf-8"))
        return
    records = load_records(args.latent_root.resolve(), int(args.max_images))
    methods = maybe_append_lsrna_method(list(args.methods), args.lsrna_output_root)
    real_full = np.load(args.real_feature_dir / "real_openimages_full.npy")
    real_patch = np.load(args.real_feature_dir / "real_openimages_patch.npy")

    print("[INFO] Loading feature extractors")
    inception = InceptionPool().to(device)
    clip_bundle = build_clip(args, device)
    text_feats = clip_text_features(clip_bundle, [str(r.get("prompt", "")) for r in records], device)

    print("[INFO] Loading FLUX VAE")
    train = load_train_module()
    vae = train.load_vae(
        SimpleNamespace(
            vae_name_or_path=args.vae_name_or_path,
            vae_subfolder=args.vae_subfolder,
            local_files_only=args.local_files_only,
        ),
        dtype,
        device,
    )
    vae.requires_grad_(False)
    vae.eval()

    lua_model = None
    if "LUA_x2_to_2048" in methods:
        print("[INFO] Loading LUA")
        lua_model = load_lua_model(weights_path=args.lua_weights, device=device, dtype=torch.float32).eval()

    rf_bundle = None
    if "RF_f3_one_step_x2" in methods:
        print("[INFO] Loading RF checkpoint")
        rf_bundle = load_rf_model(args, train, vae, device)

    rows: list[dict[str, Any]] = []
    full_features: dict[str, list[np.ndarray]] = {method: [] for method in methods}
    patch_features: dict[str, list[np.ndarray]] = {method: [] for method in methods}
    image_dir = args.output_dir / "images"

    for ordinal, record in enumerate(tqdm(records, desc="paper-style x2 eval", dynamic_ncols=True)):
        name = Path(str(record["image_filename"])).stem
        latent_path = args.latent_root / "latents" / str(record["latent_filename"])
        latent = torch.load(latent_path, map_location="cpu").to(device=device, dtype=torch.float32)

        t0 = time.perf_counter()
        direct_1024 = decode_raw_vae(vae, latent, device, dtype)
        direct_time = time.perf_counter() - t0

        outputs: dict[str, tuple[torch.Tensor, float]] = {}
        if "FLUX_direct_1024" in methods:
            outputs["FLUX_direct_1024"] = (direct_1024, direct_time)
        if "bicubic_x2" in methods:
            t0 = time.perf_counter()
            outputs["bicubic_x2"] = (
                F.interpolate(direct_1024, size=(args.target_size, args.target_size), mode="bicubic", align_corners=False, antialias=True).clamp(0.0, 1.0),
                time.perf_counter() - t0,
            )
        if "LUA_x2_to_2048" in methods:
            assert lua_model is not None
            t0 = time.perf_counter()
            z_lua2 = upscale_lua_safe(lua_model, latent, head="x2")
            outputs["LUA_x2_to_2048"] = (decode_raw_vae(vae, z_lua2, device, dtype), time.perf_counter() - t0)
        if "RF_f3_one_step_x2" in methods:
            assert rf_bundle is not None
            cut, rf_model, rf_stats = rf_bundle
            t0 = time.perf_counter()
            outputs["RF_f3_one_step_x2"] = (
                run_rf_x2(train, vae, cut, rf_model, rf_stats, direct_1024, dtype, amp_dtype, args.vector_scale),
                time.perf_counter() - t0,
            )
        if "LSRNA_SDXL_x2_to_2048" in methods:
            lsrna_path = args.lsrna_output_root / "raw_outputs/generated" / name / "lsrna_sdxl_trg2048.png"
            time_path = args.lsrna_output_root / "raw_outputs/generated" / name / "timing.json"
            if lsrna_path.exists():
                elapsed = float("nan")
                if time_path.exists():
                    elapsed = float(json.loads(time_path.read_text(encoding="utf-8")).get("elapsed_sec", float("nan")))
                outputs["LSRNA_SDXL_x2_to_2048"] = (image_path_to_tensor(lsrna_path, device, args.target_size), elapsed)

        for method, (image_01, elapsed) in outputs.items():
            boxes = patch_boxes(
                image_01.shape[-1],
                image_01.shape[-2],
                int(args.patch_size),
                int(args.patches_per_image),
                dataset=method,
                name=name,
                seed=int(args.seed),
            )
            full_feat = inception_feature(inception, image_01, device, int(args.inception_size))
            patch_feat = extract_tensor_patch_features(
                image_01,
                boxes,
                model=inception,
                device=device,
                inception_size=int(args.inception_size),
                batch_size=int(args.feature_batch_size),
            )
            full_features[method].append(full_feat)
            patch_features[method].append(patch_feat)
            text_feat = text_feats[ordinal : ordinal + 1] if text_feats is not None else None
            cscore = clip_score(clip_bundle, image_01, text_feat, device)
            rows.append(
                {
                    "ordinal": ordinal,
                    "name": name,
                    "method": method,
                    "height": int(image_01.shape[-2]),
                    "width": int(image_01.shape[-1]),
                    "time_sec": float(elapsed),
                    "clip": float(cscore),
                    "prompt": str(record.get("prompt", "")),
                }
            )
            if ordinal < int(args.save_images):
                tensor_to_png(image_01, image_dir / method / f"{name}.png")

        del latent, direct_1024, outputs
        clear_cuda()
        if (ordinal + 1) % 10 == 0:
            write_csv(args.output_dir / "per_image_metrics.csv", rows)

    summary = summarize(rows, real_full, real_patch, full_features, patch_features)
    write_csv(args.output_dir / "per_image_metrics.csv", rows)
    write_csv(args.output_dir / "paper_style_summary.csv", summary)
    for method in methods:
        if full_features[method]:
            np.save(args.output_dir / f"{method}_full_features.npy", np.concatenate(full_features[method], axis=0))
            np.save(args.output_dir / f"{method}_patch_features.npy", np.concatenate(patch_features[method], axis=0))
    metadata = {
        "experiment": "evaluate_x2_paper_style_openimages",
        "note": "Local paper-style protocol for FLUX 1024 saved latent x2 outputs; not the exact LUA paper table.",
        "latent_root": str(args.latent_root.resolve()),
        "real_feature_dir": str(args.real_feature_dir.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "methods": methods,
        "num_records": len(records),
        "patch_size": int(args.patch_size),
        "patches_per_image": int(args.patches_per_image),
        "summary": summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
