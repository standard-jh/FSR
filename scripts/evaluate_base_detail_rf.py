#!/usr/bin/env python3
"""Evaluate base preservation and detail gain for the f3 RF checkpoint.

This is a lightweight post-hoc evaluator. It does not train the model. It loads
the saved feature rectified-flow checkpoint, evaluates one-step RF on paired SR
benchmark sets, and optionally runs LR-only x2 inference on generated 1024px
images to compare base preservation against existing LUA/LSRNA visual outputs.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_feature_rectified_flow_sr.py"


def load_train_module():
    spec = importlib.util.spec_from_file_location("feature_rf_train", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_name_path(entry: str) -> tuple[str, Path]:
    if "=" in entry:
        name, raw = entry.split("=", 1)
        return name.strip(), Path(raw).expanduser()
    path = Path(entry).expanduser()
    return path.name, path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def mean_finite(values: list[float]) -> float:
    xs = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(xs)) if xs else float("nan")


def gradient_l1_01(a: torch.Tensor, b: torch.Tensor, train) -> float:
    a01 = train.m11_to_01(a).float()
    b01 = train.m11_to_01(b).float()
    ax = a01[..., :, 1:] - a01[..., :, :-1]
    bx = b01[..., :, 1:] - b01[..., :, :-1]
    ay = a01[..., 1:, :] - a01[..., :-1, :]
    by = b01[..., 1:, :] - b01[..., :-1, :]
    return float((F.l1_loss(ax, bx) + F.l1_loss(ay, by)).item())


def hf_energy(x: torch.Tensor, train) -> float:
    return float(train.highpass(x.float(), 1.0).abs().mean().item())


def metric_row(
    dataset: str,
    sample_id: str,
    mode: str,
    output_m11: torch.Tensor,
    target_m11: torch.Tensor | None,
    raw_m11: torch.Tensor | None,
    lr_m11: torch.Tensor,
    base_up_m11: torch.Tensor,
    train,
) -> dict[str, Any]:
    down = F.interpolate(output_m11.float(), size=lr_m11.shape[-2:], mode="bicubic", align_corners=False)
    out_hf = hf_energy(output_m11, train)
    base_hf = hf_energy(base_up_m11, train)
    return {
        "dataset": dataset,
        "sample_id": sample_id,
        "mode": mode,
        "height": int(output_m11.shape[-2]),
        "width": int(output_m11.shape[-1]),
        "psnr_vs_vae": train.psnr_01_from_m11(output_m11, target_m11) if target_m11 is not None else float("nan"),
        "ssim_vs_vae": train.ssim_01_from_m11(output_m11, target_m11) if target_m11 is not None else float("nan"),
        "psnr_vs_raw": train.psnr_01_from_m11(output_m11, raw_m11) if raw_m11 is not None else float("nan"),
        "ssim_vs_raw": train.ssim_01_from_m11(output_m11, raw_m11) if raw_m11 is not None else float("nan"),
        "base_psnr_rgb": train.psnr_01_from_m11(down, lr_m11),
        "base_ssim_rgb": train.ssim_01_from_m11(down, lr_m11),
        "base_l1_rgb": float(F.l1_loss(train.m11_to_01(down), train.m11_to_01(lr_m11)).item()),
        "base_grad_l1": gradient_l1_01(down, lr_m11, train),
        "hf_energy": out_hf,
        "hf_gain_vs_base_up": out_hf / max(base_hf, 1e-12),
        "hf_error_vs_vae": float(F.l1_loss(train.highpass(output_m11, 1.0), train.highpass(target_m11, 1.0)).item())
        if target_m11 is not None
        else float("nan"),
        "lf_drift_vs_base": float(F.l1_loss(train.lowpass(output_m11, 2.0), train.lowpass(base_up_m11, 2.0)).item()),
    }


def labeled_grid(items: list[tuple[str, Image.Image]], cell_w: int = 256) -> Image.Image:
    label_h = 26
    resized: list[tuple[str, Image.Image]] = []
    for label, im in items:
        im = ImageOps.exif_transpose(im).convert("RGB")
        scale = cell_w / max(1, im.width)
        cell_h = max(1, int(round(im.height * scale)))
        resized.append((label, im.resize((cell_w, cell_h), Image.Resampling.BICUBIC)))
    h = max(im.height for _, im in resized) + label_h
    canvas = Image.new("RGB", (cell_w * len(resized), h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, im) in enumerate(resized):
        x = i * cell_w
        draw.text((x + 4, 4), label, fill=(0, 0, 0))
        canvas.paste(im, (x, label_h))
    return canvas


def load_checkpoint_model(args, train, device: torch.device, dtype: torch.dtype):
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vae = train.load_vae(
        SimpleNamespace(
            vae_name_or_path=args.vae_name_or_path,
            vae_subfolder=args.vae_subfolder,
            local_files_only=args.local_files_only,
        ),
        dtype,
        device,
    )
    cut = train.resolve_cut(vae.decoder, args.cut_name, fallback_name="f3")
    model = train.FeatureRectifiedFlowNet(
        feature_channels=args.feature_channels,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        enable_gate=args.enable_gate,
        gate_channels=args.gate_channels,
        gate_init_bias=args.gate_init_bias,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    stats = train.FeatureStats(args.feature_channels, device)
    stats.load_state_dict(ckpt["feature_stats"], device)
    return vae, cut, model, stats


@torch.no_grad()
def evaluate_paired(args, train, vae, cut, model, stats, device, dtype, amp_dtype) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    eval_args = SimpleNamespace(scale=args.scale, vae_multiple=args.vae_multiple, vector_scale=args.vector_scale)
    for entry in args.paired_roots:
        name, root = parse_name_path(entry)
        resolved_name, resolved_root = train.resolve_benchmark_root(f"{name}={root}")
        paths = sorted(p for p in resolved_root.iterdir() if p.is_file() and p.suffix.lower() in train.IMAGE_EXTS)
        loader = DataLoader(train.BenchmarkImageDataset(paths, args.scale, args.vae_multiple), batch_size=1, shuffle=False, num_workers=0)
        print(f"[paired] {resolved_name}: {len(paths)} images")
        for batch in loader:
            pair = train.build_feature_pair(eval_args, vae, cut, batch, device, dtype, include_feat_bic=True, include_rgb_targets=True)
            with train.autocast_context(device, amp_dtype):
                f_hat, _ = train.sample_feature_ode(model, stats, pair.f_b, amp_dtype, 1, args.vector_scale, "euler")
                x_rf = train.decoder_tail_from_cut(vae, cut, f_hat.to(dtype=pair.f_h.dtype)).float()
            lr_m11 = train.tensor01_to_m11(pair.lr_01).to(device)
            raw_m11 = train.tensor01_to_m11(pair.hr_01).to(device)
            sample_id = pair.sample_ids[0]
            rows.append(metric_row(resolved_name, sample_id, "feature_bicubic", pair.x_feat_bic.float(), pair.x_h_rec.float(), raw_m11, lr_m11, pair.x_base.float(), train))
            rows.append(metric_row(resolved_name, sample_id, "rf_1step", x_rf.float(), pair.x_h_rec.float(), raw_m11, lr_m11, pair.x_base.float(), train))
            del pair, f_hat, x_rf
            train.clear_cuda()
    for dataset in sorted({r["dataset"] for r in rows}):
        for mode in ["feature_bicubic", "rf_1step"]:
            subset = [r for r in rows if r["dataset"] == dataset and r["mode"] == mode]
            out = {"dataset": dataset, "mode": mode, "num_images": len(subset)}
            for key in [
                "psnr_vs_vae",
                "ssim_vs_vae",
                "psnr_vs_raw",
                "ssim_vs_raw",
                "base_psnr_rgb",
                "base_ssim_rgb",
                "base_l1_rgb",
                "base_grad_l1",
                "hf_energy",
                "hf_gain_vs_base_up",
                "hf_error_vs_vae",
                "lf_drift_vs_base",
            ]:
                out[key] = mean_finite([float(r[key]) for r in subset])
            summary.append(out)
    return rows, summary


@torch.no_grad()
def infer_generated_x2(args, train, vae, cut, model, stats, device, dtype, amp_dtype):
    rows: list[dict[str, Any]] = []
    image_paths = sorted(p for p in args.generated_root.glob("*") if p.is_file() and p.suffix.lower() in train.IMAGE_EXTS)
    if args.generated_limit > 0:
        image_paths = image_paths[: args.generated_limit]
    print(f"[generated] RF x2 LR-only: {len(image_paths)} images")
    out_images = args.output_dir / "generated_rf_x2_outputs"
    out_panels = args.output_dir / "generated_rf_lua_lsrna_panels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_panels.mkdir(parents=True, exist_ok=True)
    for idx, path in enumerate(image_paths):
        sample_id = path.stem
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        image = train.center_crop_to_multiple(image, args.scale * args.vae_multiple)
        lr_01 = train.pil_to_tensor01(image).unsqueeze(0).to(device)
        lr_m11 = train.tensor01_to_m11(lr_01)
        with train.autocast_context(device, amp_dtype):
            z_l = train.encode_flux_scaled(vae, lr_m11, dtype)
            f_l = train.decoder_front_to_cut(vae, z_l, dtype, cut)
            f_b = train.interpolate_feature(f_l, (f_l.shape[-2] * args.scale, f_l.shape[-1] * args.scale))
            f_hat, _ = train.sample_feature_ode(model, stats, f_b, amp_dtype, 1, args.vector_scale, "euler")
            x_feat = train.decoder_tail_from_cut(vae, cut, f_b.to(dtype=dtype)).float()
            x_rf = train.decoder_tail_from_cut(vae, cut, f_hat.to(dtype=dtype)).float()
        base_up = F.interpolate(lr_m11.float(), size=x_rf.shape[-2:], mode="bicubic", align_corners=False)
        rows.append(metric_row("FLUX179_x2_unpaired", sample_id, "feature_bicubic", x_feat, None, None, lr_m11, base_up, train))
        rows.append(metric_row("FLUX179_x2_unpaired", sample_id, "rf_1step", x_rf, None, None, lr_m11, base_up, train))
        if idx < args.save_generated_limit:
            rf_path = out_images / f"{sample_id}_rf_1step_x2.png"
            train.image_from_m11(x_rf).save(rf_path)
            train.image_from_m11(x_feat).save(out_images / f"{sample_id}_feature_bicubic_x2.png")
            direct_dir = args.generated_compare_root / sample_id
            lsrna_dir = args.generated_lsrna_root / sample_id
            panel_items = [
                ("base 1024", image),
                ("bicubic x2", train.image_from_m11(base_up)),
                ("RF f3 1-step", train.image_from_m11(x_rf)),
            ]
            lua_path = direct_dir / "lua2.png"
            lsrna_path = lsrna_dir / "lsrna_sdxl_trg2048.png"
            if lua_path.exists():
                panel_items.append(("LUA x2", Image.open(lua_path)))
            if lsrna_path.exists():
                panel_items.append(("LSRNA x2", Image.open(lsrna_path)))
            labeled_grid(panel_items, cell_w=256).save(out_panels / f"{sample_id}_rf_lua_lsrna_panel.png")
        del lr_01, lr_m11, z_l, f_l, f_b, f_hat, x_feat, x_rf, base_up
        train.clear_cuda()
    summary: list[dict[str, Any]] = []
    for mode in ["feature_bicubic", "rf_1step"]:
        subset = [r for r in rows if r["mode"] == mode]
        out = {"dataset": "FLUX179_x2_unpaired", "mode": mode, "num_images": len(subset)}
        for key in ["base_psnr_rgb", "base_ssim_rgb", "base_l1_rgb", "base_grad_l1", "hf_energy", "hf_gain_vs_base_up", "lf_drift_vs_base"]:
            out[key] = mean_finite([float(r[key]) for r in subset])
        summary.append(out)
    return rows, summary


def copy_assets(args) -> None:
    asset_dir = ROOT / "assets" / "generated_flux179_rf_lua_lsrna"
    asset_dir.mkdir(parents=True, exist_ok=True)
    for panel in sorted((args.output_dir / "generated_rf_lua_lsrna_panels").glob("*.png"))[: args.save_generated_limit]:
        shutil.copy2(panel, asset_dir / panel.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--vae_name_or_path", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--vae_subfolder", default="vae")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--vae_multiple", type=int, default=8)
    parser.add_argument("--cut_name", default="f3")
    parser.add_argument("--feature_channels", type=int, default=512)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--enable_gate", action="store_true")
    parser.add_argument("--gate_channels", default="one")
    parser.add_argument("--gate_init_bias", type=float, default=-1.0)
    parser.add_argument("--vector_scale", type=float, default=1.0)
    parser.add_argument("--paired_roots", nargs="*", default=[])
    parser.add_argument("--generated_root", type=Path, default=Path(""))
    parser.add_argument("--generated_compare_root", type=Path, default=Path(""))
    parser.add_argument("--generated_lsrna_root", type=Path, default=Path(""))
    parser.add_argument("--generated_limit", type=int, default=0)
    parser.add_argument("--save_generated_limit", type=int, default=5)
    args = parser.parse_args()

    train = load_train_module()
    device = torch.device(args.device)
    dtype = train.dtype_from_name(args.precision)
    amp_dtype = dtype
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vae, cut, model, stats = load_checkpoint_model(args, train, device, dtype)

    paired_rows: list[dict[str, Any]] = []
    paired_summary: list[dict[str, Any]] = []
    if args.paired_roots:
        paired_rows, paired_summary = evaluate_paired(args, train, vae, cut, model, stats, device, dtype, amp_dtype)
        write_csv(args.output_dir / "rf_paired_base_detail_metrics.csv", paired_rows)
        write_csv(args.output_dir / "rf_paired_base_detail_summary.csv", paired_summary)

    if args.generated_root.exists():
        gen_rows, gen_summary = infer_generated_x2(args, train, vae, cut, model, stats, device, dtype, amp_dtype)
        write_csv(args.output_dir / "rf_generated_x2_base_detail_metrics.csv", gen_rows)
        write_csv(args.output_dir / "rf_generated_x2_base_detail_summary.csv", gen_summary)
        copy_assets(args)

    print(f"[done] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
