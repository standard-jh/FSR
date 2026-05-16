#!/usr/bin/env python3
"""OpenImages-reference FID/KID metrics for the shared generated visual subset.

This evaluates only the five generated images for which RF, LUA, and LSRNA x2
outputs all exist in the workspace. The sample count is intentionally small, so
the resulting FID/KID table should be read as a diagnostic companion to the
base/detail metrics, not as a leaderboard.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.models import Inception_V3_Weights, inception_v3
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent


METHOD_TEMPLATES = {
    "bicubic_x2": WORKSPACE / "FALSR/output/flux_random_1024_merged_179_20260501/images/{sample}.png",
    "RF one-step": WORKSPACE
    / "runs/feature_rectified_flow_x2_f3_base_detail_eval/generated_rf_x2_outputs/{sample}_rf_1step_x2.png",
    "LUA x2": WORKSPACE / "LUA/experiments/x2_visual_panels_featuresr_lua_swinir/raw_outputs/generated/{sample}/lua2.png",
    "LSRNA x2": WORKSPACE
    / "LUA/experiments/x2_visual_panels_featuresr_lua_swinir_lsrna_prompt/raw_outputs/generated/{sample}/lsrna_sdxl_trg2048.png",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenImages FID/KID for shared RF/LUA/LSRNA visual subset.")
    parser.add_argument("--samples", nargs="+", default=[f"img_{i:07d}" for i in range(5)])
    parser.add_argument(
        "--real_feature_dir",
        type=Path,
        default=WORKSPACE / "LUA/experiments/eval_flux_latent_4096_real_openimages_metrics_179v150/features",
    )
    parser.add_argument("--output_csv", type=Path, default=ROOT / "results/tables/openimages_visual5_fid_kid.csv")
    parser.add_argument("--output_json", type=Path, default=ROOT / "results/tables/openimages_visual5_fid_kid_summary.json")
    parser.add_argument("--plot_path", type=Path, default=ROOT / "assets/openimages_visual5_fid_kid.png")
    parser.add_argument("--inception_size", type=int, default=299)
    parser.add_argument("--target_size", type=int, default=2048)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--patches_per_image", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


class InceptionPool(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weights = Inception_V3_Weights.IMAGENET1K_V1
        self.model = inception_v3(weights=weights, aux_logits=True, transform_input=False)
        self.model.fc = nn.Identity()
        self.model.eval()
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def normalize_inception(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def stable_seed(*parts: str, seed: int) -> int:
    text = "::".join([str(seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") & 0x7FFFFFFF


def read_image(path: Path, target_size: int) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    with Image.open(path) as raw:
        img = raw.convert("RGB")
    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), Image.Resampling.BICUBIC)
    return img


def image_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def patch_boxes(name: str, width: int, height: int, patch_size: int, count: int, seed: int) -> list[tuple[int, int, int, int]]:
    if width < patch_size or height < patch_size:
        return [(0, 0, width, height)] * count
    max_x = width - patch_size
    max_y = height - patch_size
    anchors = [
        (0, 0),
        (max_x, 0),
        (0, max_y),
        (max_x, max_y),
        (max_x // 2, max_y // 2),
    ]
    boxes = [(x, y, x + patch_size, y + patch_size) for x, y in anchors[:count]]
    rng = np.random.default_rng(stable_seed(name, seed=seed))
    while len(boxes) < count:
        x = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
        y = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
        boxes.append((x, y, x + patch_size, y + patch_size))
    return boxes


@torch.inference_mode()
def flush_features(batch: list[torch.Tensor], model: nn.Module, device: torch.device) -> np.ndarray | None:
    if not batch:
        return None
    x = torch.stack(batch, dim=0).to(device=device, non_blocking=True)
    x = normalize_inception(x.clamp(0.0, 1.0))
    y = model(x).detach().float().cpu().numpy().astype(np.float64)
    batch.clear()
    return y


def extract_features(
    paths: Iterable[tuple[str, Path]],
    *,
    model: nn.Module,
    device: torch.device,
    target_size: int,
    inception_size: int,
    patch_size: int,
    patches_per_image: int,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    full_parts: list[np.ndarray] = []
    patch_parts: list[np.ndarray] = []
    full_batch: list[torch.Tensor] = []
    patch_batch: list[torch.Tensor] = []

    for name, path in tqdm(list(paths), desc="generated features", dynamic_ncols=True):
        img = read_image(path, target_size)
        full = img.resize((inception_size, inception_size), Image.Resampling.BICUBIC)
        full_batch.append(image_to_tensor(full))
        if len(full_batch) >= batch_size:
            part = flush_features(full_batch, model, device)
            if part is not None:
                full_parts.append(part)

        for box in patch_boxes(name, img.width, img.height, patch_size, patches_per_image, seed):
            patch = img.crop(box).resize((inception_size, inception_size), Image.Resampling.BICUBIC)
            patch_batch.append(image_to_tensor(patch))
            if len(patch_batch) >= batch_size:
                part = flush_features(patch_batch, model, device)
                if part is not None:
                    patch_parts.append(part)

    part = flush_features(full_batch, model, device)
    if part is not None:
        full_parts.append(part)
    part = flush_features(patch_batch, model, device)
    if part is not None:
        patch_parts.append(part)

    return np.concatenate(full_parts, axis=0), np.concatenate(patch_parts, axis=0)


def fid_low_rank(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mx = x.mean(axis=0)
    my = y.mean(axis=0)
    xc = x - mx
    yc = y - my
    nx = max(x.shape[0] - 1, 1)
    ny = max(y.shape[0] - 1, 1)
    trace_x = float(np.sum(xc * xc) / nx)
    trace_y = float(np.sum(yc * yc) / ny)
    cross = (xc @ yc.T) / math.sqrt(nx * ny)
    nuclear = float(np.linalg.svd(cross, compute_uv=False).sum())
    mean_term = float(np.sum((mx - my) ** 2))
    return float(max(mean_term + trace_x + trace_y - 2.0 * nuclear, 0.0))


def kid_unbiased(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    d = float(x.shape[1])
    k_xx = (x @ x.T / d + 1.0) ** 3
    k_yy = (y @ y.T / d + 1.0) ** 3
    k_xy = (x @ y.T / d + 1.0) ** 3
    n = x.shape[0]
    m = y.shape[0]
    if n < 2 or m < 2:
        return float("nan")
    xx = (float(k_xx.sum()) - float(np.trace(k_xx))) / (n * (n - 1))
    yy = (float(k_yy.sum()) - float(np.trace(k_yy))) / (m * (m - 1))
    return float(xx + yy - 2.0 * float(k_xy.mean()))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, rows: list[dict[str, object]]) -> None:
    labels = [str(row["method"]) for row in rows]
    colors = {
        "bicubic_x2": "#697078",
        "RF one-step": "#167e5c",
        "LUA x2": "#3060be",
        "LSRNA x2": "#ad3a3a",
    }
    metrics = [
        ("FID", "FID"),
        ("KID_x1000", "KID x1000"),
        ("pFID", "pFID"),
        ("pKID_x1000", "pKID x1000"),
    ]
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), dpi=180)
    fig.patch.set_facecolor("#f8f8f5")
    for ax, (key, title) in zip(axes.flatten(), metrics):
        values = [float(row[key]) for row in rows]
        ax.bar(labels, values, color=[colors.get(label, "#777777") for label in labels])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", rotation=20)
        for i, value in enumerate(values):
            ax.text(i, value * 1.01, f"{value:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    fig.suptitle("OpenImages-reference distribution metrics on shared visual5 subset", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.02,
        "Lower is better. Diagnostic only: RF/LUA/LSRNA overlap has 5 images; pFID/pKID use 80 generated patches.",
        ha="center",
        color="#555b65",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    real_full = np.load(args.real_feature_dir / "real_openimages_full.npy")
    real_patch = np.load(args.real_feature_dir / "real_openimages_patch.npy")

    model = InceptionPool().to(device)
    rows: list[dict[str, object]] = []
    for method, template in METHOD_TEMPLATES.items():
        paths = [(sample, Path(str(template).format(sample=sample))) for sample in args.samples]
        full, patch = extract_features(
            paths,
            model=model,
            device=device,
            target_size=int(args.target_size),
            inception_size=int(args.inception_size),
            patch_size=int(args.patch_size),
            patches_per_image=int(args.patches_per_image),
            batch_size=int(args.batch_size),
            seed=int(args.seed),
        )
        kid = kid_unbiased(real_full, full)
        pkid = kid_unbiased(real_patch, patch)
        rows.append(
            {
                "reference": "OpenImages HR real features",
                "method": method,
                "n_real_images": int(real_full.shape[0]),
                "n_generated_images": int(full.shape[0]),
                "n_real_patches": int(real_patch.shape[0]),
                "n_generated_patches": int(patch.shape[0]),
                "feature_dim": int(real_full.shape[1]),
                "FID": fid_low_rank(real_full, full),
                "KID_mmd2": kid,
                "KID_x1000": kid * 1000.0,
                "pFID": fid_low_rank(real_patch, patch),
                "pKID_mmd2": pkid,
                "pKID_x1000": pkid * 1000.0,
            }
        )

    write_csv(args.output_csv, rows)
    write_plot(args.plot_path, rows)
    summary = {
        "samples": args.samples,
        "real_feature_dir": str(args.real_feature_dir),
        "plot_path": str(args.plot_path),
        "note": "Diagnostic only: RF/LUA/LSRNA shared subset has five generated images.",
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
