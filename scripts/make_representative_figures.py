#!/usr/bin/env python3
"""Build publication-style summary figures for the decoder-feature-flow SR repo."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent

SAMPLE_ID = "img_0000003"
CROP_BOX = (120, 1020, 680, 1580)  # x0, y0, x1, y1 in 2048-space.

PATHS = {
    "Base x2": WORKSPACE
    / f"FALSR/output/flux_random_1024_merged_179_20260501/images/{SAMPLE_ID}.png",
    "RF one-step (ours)": WORKSPACE
    / f"runs/feature_rectified_flow_x2_f3_base_detail_eval/generated_rf_x2_outputs/{SAMPLE_ID}_rf_1step_x2.png",
    "LUA x2": WORKSPACE
    / f"LUA/experiments/x2_visual_panels_featuresr_lua_swinir/raw_outputs/generated/{SAMPLE_ID}/lua2.png",
    "LSRNA x2": WORKSPACE
    / f"LUA/experiments/x2_visual_panels_featuresr_lua_swinir_lsrna_prompt/raw_outputs/generated/{SAMPLE_ID}/lsrna_sdxl_trg2048.png",
}

METRIC_CSV = ROOT / "results/tables/generated_flux179_visual5_x2_base_detail.csv"
OUT_REP = ROOT / "assets/representative_base_detail_crop.png"
OUT_COST = ROOT / "assets/training_cost_comparison.png"
OUT_COST_CSV = ROOT / "results/tables/training_cost_comparison.csv"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def load_metrics() -> Dict[str, Dict[str, float]]:
    alias = {
        "bicubic_x2": "Base x2",
        "DecoderFeatureFlowSR RF one-step": "RF one-step (ours)",
        "LUA x2": "LUA x2",
        "LSRNA x2": "LSRNA x2",
    }
    metrics: Dict[str, Dict[str, float]] = {}
    with METRIC_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["sample_id"] != SAMPLE_ID:
                continue
            name = alias.get(row["method"])
            if not name:
                continue
            metrics[name] = {
                "base_l1": float(row["base_l1_rgb"]),
                "base_ssim": float(row["base_ssim_rgb"]),
                "hf_gain": float(row["hf_gain_vs_bicubic"]),
            }
    return metrics


def open_2048(path: Path, label: str) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    im = Image.open(path).convert("RGB")
    if im.size != (2048, 2048):
        im = im.resize((2048, 2048), Image.Resampling.BICUBIC)
    return im


def draw_label_box(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, fill: Tuple[int, int, int]) -> None:
    x, y = xy
    f = font(38, bold=True)
    pad_x, pad_y = 16, 10
    box = draw.textbbox((x, y), text, font=f)
    w, h = box[2] - box[0], box[3] - box[1]
    draw.rounded_rectangle((x, y, x + w + pad_x * 2, y + h + pad_y * 2), radius=10, fill=fill)
    draw.text((x + pad_x, y + pad_y - 2), text, fill=(255, 255, 255), font=f)


def make_representative() -> None:
    metrics = load_metrics()
    images = {name: open_2048(path, name) for name, path in PATHS.items()}

    bg = (248, 248, 245)
    ink = (25, 28, 33)
    muted = (84, 91, 101)
    accent = (230, 88, 38)
    green = (22, 126, 92)
    blue = (48, 96, 190)
    red = (173, 58, 58)

    col_w = 590
    gap = 34
    margin = 64
    full_h = 400
    crop_h = 590
    width = margin * 2 + col_w * 4 + gap * 3
    height = 1640
    canvas = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(canvas)

    draw.text((margin, 38), "Base-preserving detail synthesis in decoder feature space", fill=ink, font=font(54, True))
    draw.text(
        (margin, 106),
        f"FLUX179 generated sample {SAMPLE_ID}, x2 upscaling. Crop highlights a sharp, high-frequency region.",
        fill=muted,
        font=font(30),
    )

    colors = {
        "Base x2": (96, 105, 120),
        "RF one-step (ours)": green,
        "LUA x2": blue,
        "LSRNA x2": red,
    }

    y_full = 178
    y_crop = 760
    for idx, (name, im) in enumerate(images.items()):
        x = margin + idx * (col_w + gap)
        thumb = im.resize((col_w, full_h), Image.Resampling.LANCZOS)
        canvas.paste(thumb, (x, y_full))

        sx = col_w / 2048
        sy = full_h / 2048
        x0, y0, x1, y1 = CROP_BOX
        rect = (x + int(x0 * sx), y_full + int(y0 * sy), x + int(x1 * sx), y_full + int(y1 * sy))
        draw.rectangle(rect, outline=accent, width=6)
        draw_label_box(draw, (x + 16, y_full + 16), name, colors[name])

        crop = im.crop(CROP_BOX).resize((col_w, crop_h), Image.Resampling.LANCZOS)
        canvas.paste(crop, (x, y_crop))
        draw.rectangle((x, y_crop, x + col_w, y_crop + crop_h), outline=(225, 225, 218), width=2)

        m = metrics.get(name)
        if m:
            line1 = f"base L1 {m['base_l1']:.4f} | base SSIM {m['base_ssim']:.3f}"
            line2 = f"HF gain vs bicubic {m['hf_gain']:.2f}x"
        else:
            line1 = "base L1 - | base SSIM -"
            line2 = "HF gain vs bicubic -"
        draw.text((x, y_crop + crop_h + 20), line1, fill=ink, font=font(25, False))
        draw.text((x, y_crop + crop_h + 55), line2, fill=colors[name], font=font(29, True))

    draw.rounded_rectangle((margin, 1500, width - margin, 1590), radius=14, fill=(235, 239, 235), outline=(215, 221, 214))
    draw.text(
        (margin + 28, 1520),
        "Reading: RF keeps the generated base close to LUA, but adds more local HF detail; LSRNA adds texture while drifting far from the base.",
        fill=ink,
        font=font(29, True),
    )
    OUT_REP.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT_REP)


def write_training_cost_csv() -> None:
    rows = [
        {
            "method": "DecoderFeatureFlowSR RF one-step",
            "scope": "this repo, x2 f3 prototype",
            "hardware": "1x RTX 3090 24GB",
            "wall_hours": 9.98,
            "gpu_count": 1,
            "gpu_hours": 9.98,
            "optimizer_steps": 6063,
            "reported_train_pairs": "DIV2K crops, about 48.5K crop presentations (6063 steps x grad_accum 8)",
            "source": "local run summary",
            "notes": "Single selected f3 cut, x2 only, overnight hypothesis test.",
        },
        {
            "method": "LSRNA LSR module",
            "scope": "paper v1 latent SR module, arbitrary scale",
            "hardware": "1x NVIDIA Tesla V100-SXM2",
            "wall_hours": 26.0,
            "gpu_count": 1,
            "gpu_hours": 26.0,
            "optimizer_steps": 200000,
            "reported_train_pairs": "4.7M LR-HR latent pairs",
            "source": "arXiv:2503.18446 / CVPR 2025 supplement",
            "notes": "LSR module only; LSRNA inference also uses RNA plus a guided denoising stage.",
        },
        {
            "method": "LUA latent upscaler",
            "scope": "paper multi-scale x2/x4 adapter",
            "hardware": "8x NVIDIA H100 80GB",
            "wall_hours": 34.1,
            "gpu_count": 8,
            "gpu_hours": 272.8,
            "optimizer_steps": 375000,
            "reported_train_pairs": "3.8M OpenImages latent pairs",
            "source": "arXiv:2511.10629",
            "notes": "Three 125K-step stages; single checkpoint supports x2 and x4.",
        },
    ]
    OUT_COST_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_COST_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_training_cost_plot() -> None:
    write_training_cost_csv()
    labels = ["RF f3\nours", "LSRNA\npaper", "LUA\npaper"]
    gpu_hours = [9.98, 26.0, 272.8]
    steps = [6063, 200000, 375000]
    colors = ["#167e5c", "#ad3a3a", "#3060be"]

    plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=180)
    fig.patch.set_facecolor("#f8f8f5")

    axes[0].bar(labels, gpu_hours, color=colors)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("GPU-hours (log scale)")
    axes[0].set_title("Reported training compute")
    for i, v in enumerate(gpu_hours):
        axes[0].text(i, v * 1.12, f"{v:g}", ha="center", va="bottom", fontweight="bold")

    axes[1].bar(labels, steps, color=colors)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Optimizer steps / iterations (log scale)")
    axes[1].set_title("Training length")
    for i, v in enumerate(steps):
        axes[1].text(i, v * 1.12, f"{v/1000:.1f}K" if v >= 10000 else f"{v}", ha="center", va="bottom", fontweight="bold")

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("A small overnight f3 RF run vs. published latent-SR training budgets", fontsize=16, fontweight="bold")
    fig.text(
        0.5,
        0.02,
        "Caveat: LUA is a multi-scale x2/x4 adapter; LSRNA trains an arbitrary-scale LSR module. RF here is an x2 f3 prototype.",
        ha="center",
        color="#555b65",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.02, 0.07, 0.98, 0.92))
    OUT_COST.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_COST, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    make_representative()
    make_training_cost_plot()
    print(OUT_REP)
    print(OUT_COST)
    print(OUT_COST_CSV)


if __name__ == "__main__":
    main()
