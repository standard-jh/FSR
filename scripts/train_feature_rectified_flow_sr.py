#!/usr/bin/env python3
"""Feature-space rectified-flow SR inside the frozen FLUX VAE decoder.

This experiment learns a vector field in an intermediate VAE decoder feature
space:

    f0 = bicubic(D_<=k(E(x_LR))) -> shape of D_<=k(E(x_HR))
    f1 = D_<=k(E(x_HR))
    ft = (1 - t) f0 + t f1 + sigma(t) eps
    v_theta(ft, t, cond=f0) ~= f1 - f0

At inference it performs one-step or few-step Euler/Heun integration in feature
space and renders through the frozen decoder tail D_>k. No diffusion UNet or
scheduler noise path is used.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from PIL import Image, ImageDraw, ImageOps
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from torch.utils.data import DataLoader, Dataset


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

LUA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/home/juhwan/Documents/sr/BasicSR/datasets/DIV2K/DIV2K_train_HR")
DEFAULT_VAL_ROOT = Path("/home/juhwan/Documents/sr/BasicSR/datasets/DIV2K/DIV2K_valid_HR")
DEFAULT_OUT_DIR = LUA_ROOT / "runs/feature_rectified_flow_x2_autocut_512"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

LEVEL_DEFINITIONS = {
    "f1": "decoder.conv_in output",
    "f2": "decoder.up_blocks.0 output",
    "f3": "decoder.up_blocks.1 output",
    "f4": "decoder.up_blocks.2 output",
    "f5": "decoder.conv_act output / decoder.conv_out input",
}


@dataclass(frozen=True)
class CutSpec:
    name: str
    stage: str
    level_def: str
    fallback_from: str | None = None


@dataclass
class FeaturePair:
    sample_ids: list[str]
    hr_01: torch.Tensor
    lr_01: torch.Tensor
    f_h: torch.Tensor
    f_l: torch.Tensor
    f_b: torch.Tensor
    x_h_rec: torch.Tensor
    x_base: torch.Tensor
    x_feat_bic: torch.Tensor | None = None
    time_front_ms: float = float("nan")
    time_tail_ms: float = float("nan")


@dataclass
class TrainResult:
    cut: CutSpec
    final_step: int
    best_metric: float
    last_checkpoint: Path
    best_checkpoint: Path | None
    summary: dict[str, Any]


class HRImageDataset(Dataset):
    def __init__(self, paths: list[Path], hr_size: int, scale: int, random_crop: bool, seed: int):
        super().__init__()
        self.paths = paths
        self.hr_size = hr_size
        self.scale = scale
        self.lr_size = hr_size // scale
        self.random_crop = random_crop
        self.seed = seed

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.paths[idx]
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        image = resize_min_side(image, self.hr_size)
        if self.random_crop:
            rng = random.Random(self.seed + idx * 1000003 + int(time.time_ns() % 1000003))
        else:
            rng = random.Random(self.seed + idx * 1000003)
        image = crop_square(image, self.hr_size, self.random_crop, rng)
        hr_01 = pil_to_tensor01(image)
        lr_01 = F.interpolate(
            hr_01.unsqueeze(0),
            size=(self.lr_size, self.lr_size),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).clamp(0.0, 1.0)
        return {
            "sample_id": path.stem,
            "path": str(path),
            "hr_01": hr_01,
            "lr_01": lr_01,
        }


class BenchmarkImageDataset(Dataset):
    def __init__(self, paths: list[Path], scale: int, vae_multiple: int):
        super().__init__()
        self.paths = paths
        self.scale = scale
        self.align = max(scale * vae_multiple, scale)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.paths[idx]
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        image = center_crop_to_multiple(image, self.align)
        hr_01 = pil_to_tensor01(image)
        lr_size = (hr_01.shape[-2] // self.scale, hr_01.shape[-1] // self.scale)
        lr_01 = F.interpolate(
            hr_01.unsqueeze(0),
            size=lr_size,
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).clamp(0.0, 1.0)
        return {
            "sample_id": path.stem,
            "path": str(path),
            "hr_01": hr_01,
            "lr_01": lr_01,
        }


class DepthwiseTimeBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        groups = choose_group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pw1 = nn.Conv2d(channels, channels * 2, 1)
        self.pw2 = nn.Conv2d(channels, channels, 1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.time = nn.Linear(time_dim, channels * 2)
        self.act = nn.SiLU()
        nn.init.zeros_(self.pw2.weight)
        nn.init.zeros_(self.pw2.bias)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        residual = x
        scale_shift = self.time(temb).to(dtype=x.dtype).view(x.shape[0], -1, 1, 1)
        scale, shift = scale_shift.chunk(2, dim=1)
        y = self.norm1(x)
        y = y * (1.0 + 0.1 * scale) + 0.1 * shift
        y = self.dw(self.act(y))
        a, b = self.pw1(y).chunk(2, dim=1)
        y = a * torch.sigmoid(b)
        y = self.pw2(self.act(self.norm2(y)))
        return residual + y


class FeatureRectifiedFlowNet(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        hidden_channels: int = 128,
        num_blocks: int = 8,
        enable_gate: bool = False,
        gate_channels: str = "one",
        gate_init_bias: float = -1.0,
    ):
        super().__init__()
        self.feature_channels = feature_channels
        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks
        self.enable_gate = enable_gate
        self.gate_channels = gate_channels
        time_dim = hidden_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.in_proj = nn.Conv2d(feature_channels * 2, hidden_channels, 3, padding=1)
        self.blocks = nn.ModuleList([DepthwiseTimeBlock(hidden_channels, time_dim) for _ in range(num_blocks)])
        self.out_norm = nn.GroupNorm(choose_group_count(hidden_channels), hidden_channels)
        self.out_act = nn.SiLU()
        self.out_velocity = nn.Conv2d(hidden_channels, feature_channels, 3, padding=1)
        nn.init.zeros_(self.out_velocity.weight)
        nn.init.zeros_(self.out_velocity.bias)
        if enable_gate:
            gate_out_channels = feature_channels if gate_channels == "per_channel" else 1
            self.out_gate = nn.Conv2d(hidden_channels, gate_out_channels, 3, padding=1)
            nn.init.zeros_(self.out_gate.weight)
            nn.init.constant_(self.out_gate.bias, gate_init_bias)
        else:
            self.out_gate = None

    def forward(
        self,
        f_t: torch.Tensor,
        f0_cond: torch.Tensor,
        t: torch.Tensor,
        vector_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if t.ndim == 0:
            t = t.view(1).repeat(f_t.shape[0])
        if t.ndim == 4:
            t = t.flatten()
        if t.ndim == 2:
            t = t[:, 0]
        temb = sinusoidal_embedding(t.float(), self.hidden_channels)
        temb = self.time_mlp(temb)
        x = self.in_proj(torch.cat([f_t, f0_cond], dim=1))
        for block in self.blocks:
            x = block(x, temb)
        h = self.out_act(self.out_norm(x))
        v_pred = self.out_velocity(h) * float(vector_scale)
        gate = torch.sigmoid(self.out_gate(h)) if self.out_gate is not None else None
        v_eff = v_pred if gate is None else gate * v_pred
        return v_eff, gate, v_pred


class FeatureStats:
    def __init__(self, channels: int, device: torch.device, eps: float = 1e-6, momentum: float = 0.01):
        self.channels = channels
        self.eps = eps
        self.momentum = momentum
        self.mean = torch.zeros(1, channels, 1, 1, device=device, dtype=torch.float32)
        self.std = torch.ones(1, channels, 1, 1, device=device, dtype=torch.float32)
        self.count = 0

    def update(self, *features: torch.Tensor) -> None:
        with torch.no_grad():
            items = [x.detach().float() for x in features if x is not None]
            if not items:
                return
            x = torch.cat(items, dim=0)
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            std = x.std(dim=(0, 2, 3), unbiased=False, keepdim=True).clamp_min(self.eps)
            if self.count <= 0:
                self.mean.copy_(mean)
                self.std.copy_(std)
            else:
                m = float(self.momentum)
                self.mean.mul_(1.0 - m).add_(mean, alpha=m)
                self.std.mul_(1.0 - m).add_(std, alpha=m)
            self.count += int(x.shape[0])

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self.mean) / self.std.clamp_min(self.eps)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() * self.std.clamp_min(self.eps) + self.mean

    def state_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.detach().cpu(), "std": self.std.detach().cpu(), "count": self.count}

    def load_state_dict(self, state: dict[str, Any], device: torch.device) -> None:
        self.mean = state["mean"].to(device=device, dtype=torch.float32)
        self.std = state["std"].to(device=device, dtype=torch.float32)
        self.count = int(state.get("count", 0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train feature-space rectified-flow vector field SR.")
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--val_root", type=Path, default=DEFAULT_VAL_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--vae_name_or_path", "--vae_path", dest="vae_name_or_path", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--vae_subfolder", "--subfolder", dest="vae_subfolder", type=str, default="vae")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--hr_size", type=int, default=512)
    parser.add_argument("--vae_multiple", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_train_images", type=int, default=0)
    parser.add_argument("--num_val_images", type=int, default=0)
    parser.add_argument("--train_ratio_if_no_val", type=float, default=0.95)

    parser.add_argument("--auto_probe_cuts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--probe_cuts", nargs="+", default=["f1", "f2", "f3", "f4", "f5"])
    parser.add_argument("--probe_num_images", type=int, default=32)
    parser.add_argument("--probe_then_train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--topk_train", type=int, default=1)
    parser.add_argument("--cut_name", type=str, default="f3")
    parser.add_argument("--force_cut", type=str, default="")
    parser.add_argument("--ambiguous_score_delta", type=float, default=0.05)
    parser.add_argument("--ambiguous_warmup_steps", type=int, default=2000)
    parser.add_argument("--probe_broken_absmax", type=float, default=8.0)
    parser.add_argument("--probe_broken_rgb_l1", type=float, default=2.0)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", "--grad_accum_steps", dest="grad_accum", type=int, default=8)
    parser.add_argument("--num_steps", "--max_steps", dest="num_steps", type=int, default=30000)
    parser.add_argument(
        "--max_hours",
        type=float,
        default=10.0,
        help="Global wall-clock budget for probe + warm-up + main training. <=0 disables the time limit.",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--sigma_max", type=float, default=0.03)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--enable_gate", action="store_true")
    parser.add_argument("--gate_channels", choices=["one", "per_channel"], default="one")
    parser.add_argument("--gate_init_bias", type=float, default=-1.0)
    parser.add_argument("--vector_scale", type=float, default=1.0)
    parser.add_argument("--feature_stats_momentum", type=float, default=0.01)
    parser.add_argument("--feature_stats_eps", type=float, default=1e-6)
    parser.add_argument("--pixel_loss_every", type=int, default=1)
    parser.add_argument("--checkpoint_tail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fft_loss_max_size", type=int, default=256)

    parser.add_argument("--lambda_flow", type=float, default=1.0)
    parser.add_argument("--lambda_endpoint", type=float, default=0.5)
    parser.add_argument("--lambda_fft", type=float, default=0.1)
    parser.add_argument("--lambda_1step_feat", type=float, default=1.0)
    parser.add_argument("--lambda_rgb", type=float, default=0.5)
    parser.add_argument("--lambda_lf_anchor", type=float, default=0.2)
    parser.add_argument("--lambda_hf", type=float, default=0.2)
    parser.add_argument("--lambda_drift", type=float, default=0.005)
    parser.add_argument("--lambda_gate_sparse", type=float, default=0.001)
    parser.add_argument("--lambda_gate_tv", type=float, default=0.001)
    parser.add_argument("--lambda_feature_stats", type=float, default=0.0)
    parser.add_argument("--feature_range_z", type=float, default=0.0)

    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--val_every", type=int, default=1000)
    parser.add_argument("--vis_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--num_val", type=int, default=8)
    parser.add_argument(
        "--benchmark_roots",
        nargs="+",
        default=[
            "/home/juhwan/Documents/sr/BasicSR/datasets/Set5",
            "/home/juhwan/Documents/sr/BasicSR/datasets/Set14",
            "/home/juhwan/Documents/sr/BasicSR/datasets/B100",
        ],
        help="Benchmark HR dataset roots or name=path entries. Set5/Set14 use GTmod12; B100 uses HR.",
    )
    parser.add_argument("--benchmark_every", type=int, default=1000)
    parser.add_argument("--benchmark_max_images_per_set", type=int, default=0)
    parser.add_argument("--benchmark_during_warmup", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save_benchmark_images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--butterfly_name", type=str, default="butterfly")
    parser.add_argument("--sample_etas", nargs="+", type=float, default=[0.0, 0.03, 0.05, 0.10])
    parser.add_argument("--eta_vis", type=float, default=0.05)
    parser.add_argument("--heun4", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--zoom_size", type=int, default=128)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--keep_last_checkpoints", type=int, default=5)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb_project", type=str, default="feature-rectified-flow-sr")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_name", type=str, default="")
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb_api_key", type=str, default="")
    return parser.parse_args()


def choose_group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    if half <= 0:
        return t[:, None]
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def list_images(root: Path, limit: int = 0) -> list[Path]:
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if limit > 0:
        paths = paths[:limit]
    if not paths:
        raise RuntimeError(f"No images found under {root}")
    return paths


def resolve_benchmark_root(entry: str) -> tuple[str, Path]:
    if "=" in entry:
        name, raw = entry.split("=", 1)
        root = Path(raw).expanduser()
        name = name.strip() or root.name
    else:
        root = Path(entry).expanduser()
        name = root.name
    candidates = [
        root,
        root / "GTmod12",
        root / "HR",
        root / "original",
    ]
    if name.lower() in {"set5", "set14"}:
        candidates = [root / "GTmod12", root / "HR", root / "original", root]
    if name.lower() in {"b100", "bsd100"}:
        candidates = [root / "HR", root / "GTmod12", root / "original", root]
    for candidate in candidates:
        if candidate.exists() and any(p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in candidate.iterdir()):
            return name, candidate
    raise RuntimeError(f"No benchmark images found for {entry}")


def build_benchmark_loaders(args: argparse.Namespace, device: torch.device) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for entry in args.benchmark_roots:
        name, root = resolve_benchmark_root(str(entry))
        paths = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if args.benchmark_max_images_per_set > 0:
            paths = paths[: args.benchmark_max_images_per_set]
        if not paths:
            continue
        ds = BenchmarkImageDataset(paths, args.scale, args.vae_multiple)
        loaders[name] = DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        print(f"[BENCH] {name}: {len(paths)} images from {root}")
    return loaders


def split_train_val(paths: list[Path], train_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    paths = list(paths)
    random.Random(seed).shuffle(paths)
    if len(paths) < 2:
        return paths, paths
    n_train = min(max(1, int(round(len(paths) * train_ratio))), len(paths) - 1)
    return paths[:n_train], paths[n_train:]


def resize_min_side(image: Image.Image, min_side: int) -> Image.Image:
    w, h = image.size
    if min(w, h) >= min_side:
        return image
    scale = min_side / float(min(w, h))
    return image.resize((max(min_side, round(w * scale)), max(min_side, round(h * scale))), Image.Resampling.BICUBIC)


def crop_square(image: Image.Image, size: int, random_crop: bool, rng: random.Random) -> Image.Image:
    w, h = image.size
    if w < size or h < size:
        image = resize_min_side(image, size)
        w, h = image.size
    if random_crop:
        left = rng.randint(0, max(0, w - size))
        top = rng.randint(0, max(0, h - size))
    else:
        left = max(0, (w - size) // 2)
        top = max(0, (h - size) // 2)
    return image.crop((left, top, left + size, top + size))


def center_crop_to_multiple(image: Image.Image, multiple: int) -> Image.Image:
    w, h = image.size
    new_w = max(multiple, (w // multiple) * multiple)
    new_h = max(multiple, (h // multiple) * multiple)
    new_w = min(new_w, w)
    new_h = min(new_h, h)
    left = max(0, (w - new_w) // 2)
    top = max(0, (h - new_h) // 2)
    return image.crop((left, top, left + new_w, top + new_h))


def pil_to_tensor01(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def tensor01_to_m11(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


def m11_to_01(x: torch.Tensor) -> torch.Tensor:
    return ((x.float() + 1.0) * 0.5).clamp(0.0, 1.0)


def image_from_01(x: torch.Tensor) -> Image.Image:
    arr = x.detach().float().clamp(0.0, 1.0)[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8))


def image_from_m11(x: torch.Tensor) -> Image.Image:
    return image_from_01(m11_to_01(x))


def map_to_image(x: torch.Tensor, size: tuple[int, int] | None = None) -> Image.Image:
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3:
        if x.shape[0] != 1:
            x = x.float().abs().mean(dim=0, keepdim=True)
    else:
        x = x.view(1, *x.shape[-2:])
    x = x.float()
    x = (x - x.amin()) / (x.amax() - x.amin()).clamp_min(1e-6)
    if size is not None and x.shape[-2:] != size:
        x = F.interpolate(x.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)
    rgb = x.repeat(3, 1, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8))


def abs_error_image(pred: torch.Tensor, target: torch.Tensor) -> Image.Image:
    err = (pred.detach().float() - target.detach().float()).abs().mean(dim=1, keepdim=True)
    return map_to_image(err, size=pred.shape[-2:])


def make_labeled_grid(items: list[tuple[str, Image.Image]], cols: int, cell_w: int = 220) -> Image.Image:
    if not items:
        return Image.new("RGB", (1, 1), "white")
    label_h = 24
    resized: list[tuple[str, Image.Image]] = []
    for label, image in items:
        image = image.convert("RGB")
        scale = cell_w / image.width
        cell_h = max(1, int(round(image.height * scale)))
        resized.append((label, image.resize((cell_w, cell_h), Image.Resampling.LANCZOS)))
    cell_h = max(img.height for _, img in resized)
    rows = int(math.ceil(len(resized) / cols))
    canvas = Image.new("RGB", (cols * cell_w, rows * (cell_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, image) in enumerate(resized):
        r, c = divmod(idx, cols)
        x = c * cell_w
        y = r * (cell_h + label_h)
        draw.text((x + 4, y + 4), label[:36], fill=(0, 0, 0))
        canvas.paste(image, (x, y + label_h))
    return canvas


def crop_pil(image: Image.Image, center: tuple[int, int], size: int) -> Image.Image:
    cx, cy = center
    half = size // 2
    left = max(0, min(image.width - size, cx - half))
    top = max(0, min(image.height - size, cy - half))
    return image.crop((left, top, left + size, top + size))


def max_center_from_map(x: torch.Tensor, out_size: tuple[int, int], crop_size: int) -> tuple[int, int]:
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3:
        x = x.float().abs().mean(dim=0)
    yx = int(torch.argmax(x.flatten()).item())
    h, w = x.shape[-2:]
    y = yx // w
    x0 = yx % w
    out_h, out_w = out_size
    cx = int(round((x0 + 0.5) * out_w / max(w, 1)))
    cy = int(round((y + 0.5) * out_h / max(h, 1)))
    half = crop_size // 2
    cx = max(half, min(out_w - half, cx))
    cy = max(half, min(out_h - half, cy))
    return cx, cy


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(name)


def normalize_subfolder(vae_path: str, subfolder: str | None) -> str | None:
    if subfolder is None or subfolder.lower() in {"", "none", "null", "."}:
        return None
    local_path = Path(vae_path).expanduser()
    if local_path.exists() and not (local_path / subfolder).exists() and (local_path / "config.json").exists():
        return None
    return subfolder


def expanded_model_path(path: str) -> str:
    local_path = Path(path).expanduser()
    return str(local_path) if local_path.exists() else path


def load_vae(args: argparse.Namespace, dtype: torch.dtype, device: torch.device) -> AutoencoderKL:
    subfolder = normalize_subfolder(args.vae_name_or_path, args.vae_subfolder)
    kwargs: dict[str, Any] = {"torch_dtype": dtype, "local_files_only": args.local_files_only}
    if subfolder is not None:
        kwargs["subfolder"] = subfolder
    model_path = expanded_model_path(args.vae_name_or_path)
    print(f"[LOAD] FLUX VAE: {model_path}, subfolder={subfolder}, dtype={dtype}")
    vae = AutoencoderKL.from_pretrained(model_path, **kwargs)
    vae.eval()
    vae.requires_grad_(False)
    vae.to(device=device, dtype=dtype)
    if hasattr(vae.config, "force_upcast") and dtype != torch.float32:
        vae.config.force_upcast = False
    return vae


def get_scaling_and_shift(vae: AutoencoderKL) -> tuple[float, float]:
    scaling = float(getattr(vae.config, "scaling_factor", 1.0) or 1.0)
    shift = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
    return scaling, shift


def scaled_to_decode_latent(z: torch.Tensor, vae: AutoencoderKL) -> torch.Tensor:
    scaling, shift = get_scaling_and_shift(vae)
    return z / scaling + shift


@torch.no_grad()
def encode_flux_scaled(vae: AutoencoderKL, image_m11: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    z_raw = vae.encode(image_m11.to(dtype=dtype), return_dict=True).latent_dist.mode()
    scaling, shift = get_scaling_and_shift(vae)
    return scaling * (z_raw - shift)


def prepare_decoder_input(vae: AutoencoderKL, z_decode: torch.Tensor) -> torch.Tensor:
    if getattr(vae, "post_quant_conv", None) is not None:
        return vae.post_quant_conv(z_decode)
    return z_decode


def call_mid_or_up(module: nn.Module, sample: torch.Tensor) -> torch.Tensor:
    try:
        return module(sample, None)
    except TypeError:
        return module(sample)


def decoder_has_stage(decoder: nn.Module, stage: str) -> bool:
    if stage in {"conv_in", "mid_block", "conv_norm_out", "conv_act"}:
        return hasattr(decoder, stage)
    if stage.startswith("up_blocks."):
        try:
            idx = int(stage.split(".")[-1])
        except ValueError:
            return False
        return hasattr(decoder, "up_blocks") and idx < len(decoder.up_blocks)
    return False


def resolve_cut(decoder: nn.Module, name: str, fallback_name: str | None = None) -> CutSpec:
    raw = name.strip()
    key = raw.lower()
    aliases = {
        "f1": ("conv_in", LEVEL_DEFINITIONS["f1"]),
        "f2": ("up_blocks.0", LEVEL_DEFINITIONS["f2"]),
        "f3": ("up_blocks.1", LEVEL_DEFINITIONS["f3"]),
        "f4": ("up_blocks.2", LEVEL_DEFINITIONS["f4"]),
        "f5": ("conv_act", LEVEL_DEFINITIONS["f5"]),
        "conv_in": ("conv_in", "decoder.conv_in output"),
        "mid_block": ("mid_block", "decoder.mid_block output"),
        "up_blocks.0": ("up_blocks.0", "decoder.up_blocks.0 output"),
        "up_blocks.1": ("up_blocks.1", "decoder.up_blocks.1 output"),
        "up_blocks.2": ("up_blocks.2", "decoder.up_blocks.2 output"),
        "up_blocks.3": ("up_blocks.3", "decoder.up_blocks.3 output"),
        "conv_norm_out": ("conv_norm_out", "decoder.conv_norm_out output"),
        "conv_act": ("conv_act", "decoder.conv_act output / conv_out input"),
    }
    candidates: list[tuple[str, str, str | None]] = []
    if key in aliases:
        stage, definition = aliases[key]
        candidates.append((stage, definition, None))

    if key == "f3":
        candidates.extend([
            ("up_blocks.0", "fallback for f3: decoder.up_blocks.0 output", "f3"),
        ])
    if key == "f1":
        candidates.extend([
            ("mid_block", "fallback for f1: decoder.mid_block output", "f1"),
            ("up_blocks.0", "fallback for f1: decoder.up_blocks.0 output", "f1"),
        ])
    if fallback_name:
        fb = fallback_name.lower()
        if fb in aliases:
            stage, definition = aliases[fb]
            candidates.append((stage, f"fallback --cut_name {fallback_name}: {definition}", raw))

    for stage, definition, fallback_from in candidates:
        if decoder_has_stage(decoder, stage):
            canonical = key if key in LEVEL_DEFINITIONS else stage
            if fallback_from is not None:
                canonical = stage
            return CutSpec(name=canonical, stage=stage, level_def=definition, fallback_from=fallback_from)
    raise RuntimeError(f"Could not resolve decoder cut {name!r}.")


def decoder_front_to_cut(vae: AutoencoderKL, z_scaled: torch.Tensor, dtype: torch.dtype, cut: CutSpec) -> torch.Tensor:
    decoder = vae.decoder
    x = prepare_decoder_input(vae, scaled_to_decode_latent(z_scaled.to(dtype=dtype), vae))
    x = decoder.conv_in(x)
    if cut.stage == "conv_in":
        return x
    x = call_mid_or_up(decoder.mid_block, x)
    if cut.stage == "mid_block":
        return x
    for i, block in enumerate(decoder.up_blocks):
        x = call_mid_or_up(block, x)
        if cut.stage == f"up_blocks.{i}":
            return x
    x = decoder.conv_norm_out(x)
    if cut.stage == "conv_norm_out":
        return x
    x = decoder.conv_act(x)
    if cut.stage == "conv_act":
        return x
    raise ValueError(f"Unsupported cut stage: {cut.stage}")


def decoder_tail_from_cut(vae: AutoencoderKL, cut: CutSpec, feature: torch.Tensor) -> torch.Tensor:
    decoder = vae.decoder
    x = feature
    if cut.stage == "conv_in":
        x = call_mid_or_up(decoder.mid_block, x)
        start_up = 0
    elif cut.stage == "mid_block":
        start_up = 0
    elif cut.stage.startswith("up_blocks."):
        start_up = int(cut.stage.split(".")[-1]) + 1
    elif cut.stage == "conv_norm_out":
        x = decoder.conv_act(x)
        return decoder.conv_out(x)
    elif cut.stage == "conv_act":
        return decoder.conv_out(x)
    else:
        raise ValueError(f"Unsupported cut stage: {cut.stage}")

    for block in decoder.up_blocks[start_up:]:
        x = call_mid_or_up(block, x)
    x = decoder.conv_norm_out(x)
    x = decoder.conv_act(x)
    return decoder.conv_out(x)


def decoder_tail_train(
    vae: AutoencoderKL,
    cut: CutSpec,
    feature: torch.Tensor,
    use_checkpoint: bool,
) -> torch.Tensor:
    if use_checkpoint and feature.requires_grad:
        return activation_checkpoint(lambda y: decoder_tail_from_cut(vae, cut, y), feature, use_reentrant=False)
    return decoder_tail_from_cut(vae, cut, feature)


@torch.no_grad()
def decode_flux_scaled(vae: AutoencoderKL, z_scaled: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return vae.decode(scaled_to_decode_latent(z_scaled.to(dtype=dtype), vae), return_dict=False)[0]


def interpolate_feature(x: torch.Tensor, size: tuple[int, int], mode: str = "bicubic") -> torch.Tensor:
    if x.shape[-2:] == size:
        return x
    if mode == "nearest":
        return F.interpolate(x, size=size, mode=mode)
    return F.interpolate(x, size=size, mode=mode, align_corners=False)


def autocast_context(device: torch.device, dtype: torch.dtype):
    enabled = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clear_cuda(reset_peak: bool = False, device: torch.device | None = None) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if reset_peak and (device is None or device.type == "cuda"):
            torch.cuda.reset_peak_memory_stats(device)


def cuda_peak_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024**3))


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    radius = max(1, int(math.ceil(sigma * 3.0)))
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-(coords**2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    c = x.shape[1]
    kernel_x = kernel.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    kernel_y = kernel.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    y = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    y = F.conv2d(y, kernel_x, groups=c)
    y = F.pad(y, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(y, kernel_y, groups=c)


def lowpass(x: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    return gaussian_blur(x.float(), sigma)


def highpass(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    xf = x.float()
    return xf - gaussian_blur(xf, sigma)


def charbonnier(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((pred.float() - target.float()).square() + eps * eps).mean()


def fft_abs_l1(pred: torch.Tensor, target: torch.Tensor, max_size: int = 0) -> torch.Tensor:
    a = pred.float()
    b = target.float()
    if max_size > 0 and max(a.shape[-2:]) > max_size:
        scale = max_size / float(max(a.shape[-2:]))
        size = (max(8, int(round(a.shape[-2] * scale))), max(8, int(round(a.shape[-1] * scale))))
        a = F.interpolate(a, size=size, mode="bilinear", align_corners=False)
        b = F.interpolate(b, size=size, mode="bilinear", align_corners=False)
    fa = torch.fft.rfft2(a, norm="ortho").abs()
    fb = torch.fft.rfft2(b, norm="ortho").abs()
    return F.l1_loss(fa, fb)


def total_variation(x: torch.Tensor | None) -> torch.Tensor:
    if x is None:
        return torch.zeros((), device="cpu")
    return (x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (x[..., :, 1:] - x[..., :, :-1]).abs().mean()


def psnr_01_from_m11(pred: torch.Tensor, target: torch.Tensor) -> float:
    p = m11_to_01(pred)
    t = m11_to_01(target)
    mse = F.mse_loss(p, t).item()
    return 99.0 if mse <= 1e-12 else float(-10.0 * math.log10(mse))


def ssim_01_from_m11(pred: torch.Tensor, target: torch.Tensor) -> float:
    x = m11_to_01(pred).float()
    y = m11_to_01(target).float()
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = gaussian_blur(x, 1.5)
    mu_y = gaussian_blur(y, 1.5)
    sigma_x = gaussian_blur(x * x, 1.5) - mu_x * mu_x
    sigma_y = gaussian_blur(y * y, 1.5) - mu_y * mu_y
    sigma_xy = gaussian_blur(x * y, 1.5) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    ).clamp_min(1e-12)
    return float(ssim.mean().clamp(-1, 1).item())


def build_lpips(device: torch.device):
    try:
        import lpips
    except Exception as exc:
        print(f"[WARN] LPIPS unavailable: {exc}")
        return None
    try:
        net = lpips.LPIPS(net="alex").to(device)
        net.eval()
        net.requires_grad_(False)
        return net
    except Exception as exc:
        print(f"[WARN] LPIPS init failed; continuing without LPIPS: {exc}")
        return None


def lpips_metric(net, pred_m11: torch.Tensor, target_m11: torch.Tensor, size: int = 256) -> float:
    if net is None:
        return float("nan")
    with torch.no_grad():
        pred = pred_m11.float().clamp(-1, 1)
        target = target_m11.float().clamp(-1, 1)
        if size > 0 and max(pred.shape[-2:]) > size:
            pred = F.interpolate(pred, size=(size, size), mode="bicubic", align_corners=False)
            target = F.interpolate(target, size=(size, size), mode="bicubic", align_corners=False)
        return float(net(pred, target).mean().item())


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, CutSpec):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), indent=2, ensure_ascii=False), encoding="utf-8")


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def init_wandb(args: argparse.Namespace, config: dict[str, Any]):
    if not args.wandb or args.wandb_mode == "disabled":
        return None
    try:
        import wandb

        if args.wandb_api_key:
            os.environ["WANDB_API_KEY"] = args.wandb_api_key
            wandb.login(key=args.wandb_api_key, relogin=True)
        name = args.wandb_name or f"feature_rf_{time.strftime('%Y%m%d_%H%M%S')}"
        return wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=name,
            mode=args.wandb_mode,
            config=to_jsonable(config),
        )
    except Exception as exc:
        print(f"[WARN] wandb init failed; continuing with local logs only: {exc}")
        return None


def wandb_image(path: Path, caption: str):
    try:
        import wandb

        return wandb.Image(str(path), caption=caption)
    except Exception:
        return None


def mean_finite(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def global_elapsed_hours(args: argparse.Namespace) -> float:
    started_at = float(getattr(args, "run_started_at", time.time()))
    return (time.time() - started_at) / 3600.0


def remaining_hours(args: argparse.Namespace) -> float:
    max_hours = float(getattr(args, "max_hours", 0.0) or 0.0)
    if max_hours <= 0:
        return float("inf")
    return max_hours - global_elapsed_hours(args)


def time_budget_exhausted(args: argparse.Namespace) -> bool:
    max_hours = float(getattr(args, "max_hours", 0.0) or 0.0)
    return max_hours > 0 and global_elapsed_hours(args) >= max_hours


@torch.no_grad()
def build_feature_pair(
    args: argparse.Namespace,
    vae: AutoencoderKL,
    cut: CutSpec,
    batch: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    include_feat_bic: bool = False,
    include_rgb_targets: bool = True,
    collect_timing: bool = False,
) -> FeaturePair:
    sample_ids = [str(x) for x in batch["sample_id"]]
    hr_01 = batch["hr_01"].to(device=device, dtype=torch.float32, non_blocking=True)
    lr_01 = batch["lr_01"].to(device=device, dtype=torch.float32, non_blocking=True)
    hr_m11 = tensor01_to_m11(hr_01)
    lr_m11 = tensor01_to_m11(lr_01)
    synchronize(device)
    t_front0 = time.perf_counter()
    z_h = encode_flux_scaled(vae, hr_m11, dtype)
    z_l = encode_flux_scaled(vae, lr_m11, dtype)
    f_h = decoder_front_to_cut(vae, z_h, dtype, cut).detach()
    f_l = decoder_front_to_cut(vae, z_l, dtype, cut).detach()
    f_b = interpolate_feature(f_l.float(), size=f_h.shape[-2:], mode="bicubic").to(dtype=f_h.dtype).detach()
    synchronize(device)
    t_front1 = time.perf_counter()

    x_h_rec = torch.empty(0, device=device)
    x_base = torch.empty(0, device=device)
    x_feat_bic = None
    if include_rgb_targets or include_feat_bic:
        x_h_rec = decoder_tail_from_cut(vae, cut, f_h).float().detach()
        x_l_rec = decoder_tail_from_cut(vae, cut, f_l).float().detach()
        x_base = F.interpolate(x_l_rec, size=x_h_rec.shape[-2:], mode="bicubic", align_corners=False).detach()
        if include_feat_bic:
            x_feat_bic = decoder_tail_from_cut(vae, cut, f_b).float().detach()
    synchronize(device)
    t_tail1 = time.perf_counter()

    return FeaturePair(
        sample_ids=sample_ids,
        hr_01=hr_01.detach(),
        lr_01=lr_01.detach(),
        f_h=f_h,
        f_l=f_l,
        f_b=f_b,
        x_h_rec=x_h_rec,
        x_base=x_base,
        x_feat_bic=x_feat_bic,
        time_front_ms=(t_front1 - t_front0) * 1000.0 if collect_timing else float("nan"),
        time_tail_ms=(t_tail1 - t_front1) * 1000.0 if collect_timing else float("nan"),
    )


def cut_probe(
    args: argparse.Namespace,
    vae: AutoencoderKL,
    probe_loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    lpips_net,
) -> tuple[CutSpec | None, list[dict[str, Any]], dict[str, Any]]:
    probe_dir = args.output_dir / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    visual_items: list[tuple[str, Image.Image]] = []

    print("[PROBE] decoder cut probing 시작")
    for cut_name in args.probe_cuts:
        try:
            cut = resolve_cut(vae.decoder, cut_name, fallback_name=args.cut_name)
        except Exception as exc:
            rows.append({"cut_name": cut_name, "status": "resolve_failed", "error": f"{type(exc).__name__}: {exc}"})
            continue
        print(f"[PROBE] {cut_name} -> {cut.stage} ({cut.level_def})")
        for idx, batch in enumerate(probe_loader):
            if idx >= args.probe_num_images:
                break
            clear_cuda(reset_peak=True, device=device)
            t0 = time.perf_counter()
            try:
                pair = build_feature_pair(args, vae, cut, batch, device, dtype, include_feat_bic=True, collect_timing=True)
                assert pair.x_feat_bic is not None
                eps_scale = 0.01 * pair.f_b.float().std(unbiased=False).clamp_min(1e-6)
                eps = eps_scale * torch.randn_like(pair.f_b.float())
                x_perturb = decoder_tail_from_cut(vae, cut, (pair.f_b.float() + eps).to(dtype=pair.f_b.dtype)).float()
                denom = eps.abs().mean().clamp_min(1e-12)
                sensitivity = float((x_perturb - pair.x_feat_bic).abs().mean().div(denom).item())
                x_absmax = float(pair.x_feat_bic.abs().amax().item())
                rgb_l1 = float(F.l1_loss(pair.x_feat_bic, pair.x_h_rec).item())
                broken = (
                    (not torch.isfinite(pair.x_feat_bic).all().item())
                    or x_absmax > args.probe_broken_absmax
                    or rgb_l1 > args.probe_broken_rgb_l1
                )
                row = {
                    "cut_name": cut_name,
                    "resolved_name": cut.name,
                    "stage": cut.stage,
                    "level_def": cut.level_def,
                    "sample_id": pair.sample_ids[0],
                    "status": "ok",
                    "broken": bool(broken),
                    "feature_shape_b": "x".join(str(v) for v in pair.f_b.shape),
                    "feature_shape_h": "x".join(str(v) for v in pair.f_h.shape),
                    "feature_l1": float(F.l1_loss(pair.f_b.float(), pair.f_h.float()).item()),
                    "feature_fft": float(fft_abs_l1(pair.f_b, pair.f_h, args.fft_loss_max_size).item()),
                    "rgb_l1": rgb_l1,
                    "rgb_lpips": lpips_metric(lpips_net, pair.x_feat_bic, pair.x_h_rec),
                    "low_freq_error": float(F.l1_loss(lowpass(pair.x_feat_bic, 2.0), lowpass(pair.x_base, 2.0)).item()),
                    "high_freq_error": float(F.l1_loss(highpass(pair.x_feat_bic, 1.0), highpass(pair.x_h_rec, 1.0)).item()),
                    "feature_mean_std_gap": float(
                        (pair.f_b.float().mean() - pair.f_h.float().mean()).abs().item()
                        + (pair.f_b.float().std(unbiased=False) - pair.f_h.float().std(unbiased=False)).abs().item()
                    ),
                    "decoder_sensitivity": sensitivity,
                    "decoded_absmax": x_absmax,
                    "runtime_sec": float(time.perf_counter() - t0),
                    "front_ms": pair.time_front_ms,
                    "tail_ms": pair.time_tail_ms,
                    "vram_gb": cuda_peak_gb(),
                }
                if idx == 0:
                    visual_items.extend(
                        [
                            (f"{cut.name} HR target", image_from_m11(pair.x_h_rec)),
                            (f"{cut.name} feat bic", image_from_m11(pair.x_feat_bic)),
                            (f"{cut.name} abs err", abs_error_image(pair.x_feat_bic, pair.x_h_rec)),
                        ]
                    )
                rows.append(row)
                del pair, x_perturb, eps
            except torch.cuda.OutOfMemoryError as exc:
                rows.append(
                    {
                        "cut_name": cut_name,
                        "resolved_name": cut.name,
                        "stage": cut.stage,
                        "status": "oom",
                        "error": str(exc).splitlines()[0],
                        "runtime_sec": float(time.perf_counter() - t0),
                        "vram_gb": cuda_peak_gb(),
                    }
                )
                clear_cuda(reset_peak=True, device=device)
                break
            except Exception as exc:
                rows.append(
                    {
                        "cut_name": cut_name,
                        "resolved_name": cut.name,
                        "stage": cut.stage,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "runtime_sec": float(time.perf_counter() - t0),
                        "vram_gb": cuda_peak_gb(),
                    }
                )
                clear_cuda(reset_peak=True, device=device)
                break

    write_csv(probe_dir / "probe_metrics.csv", rows)
    if visual_items:
        make_labeled_grid(visual_items, cols=3, cell_w=220).save(probe_dir / "probe_visual_grid.png")

    summary_rows = summarize_probe_rows(rows)
    selected, selection = select_probe_cut(args, vae.decoder, summary_rows)
    write_json(
        probe_dir / "probe_summary.json",
        {
            "selected_cut": selected,
            "selection": selection,
            "summary": summary_rows,
            "probe_metric_csv": probe_dir / "probe_metrics.csv",
            "probe_visual_grid": probe_dir / "probe_visual_grid.png",
        },
    )
    write_csv(probe_dir / "probe_summary.csv", summary_rows)
    if selected is not None:
        print(f"[PROBE] selected_cut_name={selected.name}, stage={selected.stage}, reason={selection.get('reason')}")
    else:
        print(f"[PROBE] 자동 선택 실패: {selection.get('reason')}")
    return selected, summary_rows, selection


def summarize_probe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get("resolved_name", row.get("cut_name", "unknown")))
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    metrics = [
        "feature_l1",
        "feature_fft",
        "rgb_l1",
        "rgb_lpips",
        "low_freq_error",
        "high_freq_error",
        "feature_mean_std_gap",
        "decoder_sensitivity",
        "decoded_absmax",
        "runtime_sec",
        "front_ms",
        "tail_ms",
        "vram_gb",
    ]
    for name, group in groups.items():
        oks = [r for r in group if r.get("status") == "ok"]
        row: dict[str, Any] = {
            "cut_name": name,
            "stage": next((r.get("stage") for r in group if r.get("stage")), ""),
            "level_def": next((r.get("level_def") for r in group if r.get("level_def")), ""),
            "n": len(group),
            "ok_n": len(oks),
            "broken_n": sum(1 for r in oks if bool(r.get("broken", False))),
            "oom_n": sum(1 for r in group if r.get("status") == "oom"),
            "failed_n": sum(1 for r in group if r.get("status") not in {"ok", "oom"}),
        }
        for metric in metrics:
            vals = [float(r[metric]) for r in oks if metric in r and math.isfinite(float(r[metric]))]
            row[f"{metric}_mean"] = float(np.mean(vals)) if vals else float("nan")
            row[f"{metric}_median"] = float(np.median(vals)) if vals else float("nan")
            row[f"{metric}_max"] = float(np.max(vals)) if vals else float("nan")
        row["broken_fraction"] = row["broken_n"] / max(1, row["ok_n"])
        out.append(row)
    return out


def normalize_scores(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    vals = [(str(r["cut_name"]), float(r.get(key, float("nan")))) for r in rows]
    finite = [v for _, v in vals if math.isfinite(v)]
    if not finite:
        return {name: 0.0 for name, _ in vals}
    lo, hi = min(finite), max(finite)
    if abs(hi - lo) < 1e-12:
        return {name: 0.5 for name, _ in vals}
    return {name: (v - lo) / (hi - lo) if math.isfinite(v) else 1.0 for name, v in vals}


def select_probe_cut(
    args: argparse.Namespace,
    decoder: nn.Module,
    summary_rows: list[dict[str, Any]],
) -> tuple[CutSpec | None, dict[str, Any]]:
    stable: list[dict[str, Any]] = []
    reasons: dict[str, str] = {}
    for row in summary_rows:
        name = str(row["cut_name"])
        reason_parts = []
        ok = int(row.get("ok_n", 0)) > 0
        if not ok:
            reason_parts.append("no successful probe samples")
        if float(row.get("broken_fraction", 1.0)) > 0.25:
            reason_parts.append("feature-bicubic decode looked unstable")
        if int(row.get("oom_n", 0)) > 0 and int(row.get("ok_n", 0)) == 0:
            reason_parts.append("OOM")
        if not math.isfinite(float(row.get("feature_l1_mean", float("nan")))):
            reason_parts.append("feature gap is not finite")
        if float(row.get("feature_l1_mean", 0.0)) < 1e-6:
            reason_parts.append("feature gap is too small")
        if float(row.get("high_freq_error_mean", 0.0)) < 1e-6:
            reason_parts.append("high-frequency error is too small")
        if reason_parts:
            reasons[name] = "; ".join(reason_parts)
        else:
            stable.append(row)

    if not stable:
        return None, {"reason": "no stable probe cut", "rejections": reasons}

    hf = normalize_scores(stable, "high_freq_error_mean")
    fft = normalize_scores(stable, "feature_fft_mean")
    lf = normalize_scores(stable, "low_freq_error_mean")
    sens = normalize_scores(stable, "decoder_sensitivity_mean")
    runtime = normalize_scores(stable, "runtime_sec_mean")
    scored: list[dict[str, Any]] = []
    for row in stable:
        name = str(row["cut_name"])
        score = 1.0 * hf[name] + 0.5 * fft[name] - 0.7 * lf[name] - 0.7 * sens[name] - 0.3 * runtime[name]
        item = dict(row)
        item["auto_score"] = float(score)
        item["score_terms"] = {
            "normalized_hf_error": hf[name],
            "normalized_feature_fft_gap": fft[name],
            "normalized_low_freq_drift": lf[name],
            "normalized_decoder_sensitivity": sens[name],
            "normalized_runtime": runtime[name],
        }
        scored.append(item)
    scored.sort(key=lambda r: float(r["auto_score"]), reverse=True)
    winner_name = str(scored[0]["cut_name"])
    try:
        selected = resolve_cut(decoder, winner_name, fallback_name=args.cut_name)
    except Exception:
        selected = resolve_cut(decoder, args.cut_name, fallback_name="f3")
        return selected, {
            "reason": f"winner {winner_name} could not be resolved at train time; fallback to {selected.name}",
            "scored": scored,
            "rejections": reasons,
        }
    reason = (
        f"score={scored[0]['auto_score']:.4f}; stable cut with high HF/FFT gap, limited LF drift/sensitivity/runtime"
    )
    return selected, {"reason": reason, "scored": scored, "rejections": reasons}


def infer_feature_shape(
    args: argparse.Namespace,
    vae: AutoencoderKL,
    cut: CutSpec,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[int, tuple[int, int], FeaturePair]:
    batch = next(iter(loader))
    pair = build_feature_pair(args, vae, cut, batch, device, dtype, include_feat_bic=False, include_rgb_targets=False)
    channels = int(pair.f_b.shape[1])
    spatial = (int(pair.f_b.shape[-2]), int(pair.f_b.shape[-1]))
    return channels, spatial, pair


def model_v_eff(
    model: FeatureRectifiedFlowNet,
    stats: FeatureStats,
    f_raw: torch.Tensor,
    f0_raw: torch.Tensor,
    t_value: float | torch.Tensor,
    amp_dtype: torch.dtype,
    vector_scale: float,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    f_n = stats.normalize(f_raw).to(device=f_raw.device)
    f0_n = stats.normalize(f0_raw).to(device=f0_raw.device)
    if isinstance(t_value, torch.Tensor):
        t = t_value.to(device=f_raw.device, dtype=torch.float32)
    else:
        t = torch.full((f_raw.shape[0],), float(t_value), device=f_raw.device, dtype=torch.float32)
    v_eff_n, gate, v_pred_n = model(f_n.to(dtype=amp_dtype), f0_n.to(dtype=amp_dtype), t, vector_scale=vector_scale)
    f_next_raw = stats.denormalize(f_n + v_eff_n.float())
    return v_eff_n, gate, v_pred_n, f_next_raw


def sample_feature_ode(
    model: FeatureRectifiedFlowNet,
    stats: FeatureStats,
    f_b: torch.Tensor,
    amp_dtype: torch.dtype,
    steps: int,
    vector_scale: float,
    method: str = "euler",
    eta: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    cond_n = stats.normalize(f_b).to(device=f_b.device)
    f_n = cond_n.clone()
    gate0: torch.Tensor | None = None
    if eta > 0:
        t0 = torch.zeros((f_b.shape[0],), device=f_b.device)
        with torch.no_grad():
            _, gate_probe, _ = model(f_n.to(dtype=amp_dtype), cond_n.to(dtype=amp_dtype), t0, vector_scale=vector_scale)
        if gate_probe is None:
            mask = highpass(f_b.float(), 1.0).abs().mean(dim=1, keepdim=True)
            mask = mask / mask.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        else:
            mask = gate_probe.float().detach()
        f_n = f_n + float(eta) * mask * torch.randn_like(f_n)
    dt = 1.0 / float(steps)
    for i in range(steps):
        t = torch.full((f_b.shape[0],), i * dt, device=f_b.device, dtype=torch.float32)
        v1, gate, _ = model(f_n.to(dtype=amp_dtype), cond_n.to(dtype=amp_dtype), t, vector_scale=vector_scale)
        if i == 0:
            gate0 = gate
        if method == "heun":
            f_pred = f_n + dt * v1.float()
            t2 = torch.full((f_b.shape[0],), min(1.0, (i + 1) * dt), device=f_b.device, dtype=torch.float32)
            v2, _, _ = model(f_pred.to(dtype=amp_dtype), cond_n.to(dtype=amp_dtype), t2, vector_scale=vector_scale)
            f_n = f_n + 0.5 * dt * (v1.float() + v2.float())
        else:
            f_n = f_n + dt * v1.float()
    return stats.denormalize(f_n), gate0


def compute_train_loss(
    args: argparse.Namespace,
    vae: AutoencoderKL,
    cut: CutSpec,
    model: FeatureRectifiedFlowNet,
    stats: FeatureStats,
    pair: FeaturePair,
    step: int,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    f_b = pair.f_b.detach()
    f_h = pair.f_h.detach()
    stats.update(f_b, f_h)
    f_b_n = stats.normalize(f_b)
    f_h_n = stats.normalize(f_h)
    b = f_b.shape[0]
    t = torch.rand((b, 1, 1, 1), device=device, dtype=torch.float32)
    sigma_t = float(args.sigma_max) * t * (1.0 - t)
    f_t_n = (1.0 - t) * f_b_n + t * f_h_n + sigma_t * torch.randn_like(f_b_n)
    v_target_n = f_h_n - f_b_n

    with autocast_context(device, amp_dtype):
        v_eff_n, gate, v_pred_n = model(
            f_t_n.to(dtype=amp_dtype),
            f_b_n.to(dtype=amp_dtype),
            t.flatten(),
            vector_scale=args.vector_scale,
        )
        f1_from_t_n = f_t_n + (1.0 - t) * v_eff_n.float()
        zero_t = torch.zeros((b,), device=device, dtype=torch.float32)
        v_eff_0_n, gate0, _ = model(
            f_b_n.to(dtype=amp_dtype),
            f_b_n.to(dtype=amp_dtype),
            zero_t,
            vector_scale=args.vector_scale,
        )
        f1_onestep_n = f_b_n + v_eff_0_n.float()

    f1_from_t_raw = stats.denormalize(f1_from_t_n)
    f1_onestep_raw = stats.denormalize(f1_onestep_n)

    loss_flow = charbonnier(v_eff_n, v_target_n)
    loss_endpoint = charbonnier(f1_from_t_n, f_h_n)
    loss_fft = fft_abs_l1(f1_from_t_raw, f_h, args.fft_loss_max_size)
    loss_1step_feat = charbonnier(f1_onestep_n, f_h_n)

    do_pixel = args.pixel_loss_every <= 1 or (step % max(1, args.pixel_loss_every) == 0)
    if do_pixel:
        with autocast_context(device, amp_dtype):
            x_1step = decoder_tail_train(
                vae,
                cut,
                f1_onestep_raw.to(dtype=pair.f_h.dtype),
                use_checkpoint=bool(args.checkpoint_tail),
            ).float()
        loss_rgb = charbonnier(x_1step, pair.x_h_rec)
        loss_lf_anchor = F.l1_loss(lowpass(x_1step, 2.0), lowpass(pair.x_base, 2.0))
        loss_hf = F.l1_loss(highpass(x_1step, 1.0), highpass(pair.x_h_rec, 1.0))
    else:
        x_1step = pair.x_base.detach()
        loss_rgb = f_b_n.new_zeros(())
        loss_lf_anchor = f_b_n.new_zeros(())
        loss_hf = f_b_n.new_zeros(())

    loss_drift = (f1_onestep_raw - f_b.float()).abs().mean()
    if gate is not None:
        loss_gate_sparse = gate.float().mean()
        loss_gate_tv = total_variation(gate.float()).to(device=device)
    else:
        loss_gate_sparse = f_b_n.new_zeros(())
        loss_gate_tv = f_b_n.new_zeros(())

    if args.lambda_feature_stats > 0:
        mean_gap = (f1_onestep_raw.mean(dim=(0, 2, 3)) - f_h.float().mean(dim=(0, 2, 3))).abs().mean()
        std_gap = (
            f1_onestep_raw.std(dim=(0, 2, 3), unbiased=False)
            - f_h.float().std(dim=(0, 2, 3), unbiased=False)
        ).abs().mean()
        loss_feature_stats = mean_gap + std_gap
    else:
        loss_feature_stats = f_b_n.new_zeros(())

    if args.feature_range_z > 0:
        excess = F.relu(f1_onestep_n.abs() - float(args.feature_range_z))
        loss_feature_range = excess.mean()
    else:
        loss_feature_range = f_b_n.new_zeros(())

    loss = (
        args.lambda_flow * loss_flow
        + args.lambda_endpoint * loss_endpoint
        + args.lambda_fft * loss_fft
        + args.lambda_1step_feat * loss_1step_feat
        + args.lambda_rgb * loss_rgb
        + args.lambda_lf_anchor * loss_lf_anchor
        + args.lambda_hf * loss_hf
        + args.lambda_drift * loss_drift
        + args.lambda_gate_sparse * loss_gate_sparse
        + args.lambda_gate_tv * loss_gate_tv
        + args.lambda_feature_stats * loss_feature_stats
        + loss_feature_range
    )
    metrics = {
        "loss": float(loss.detach().item()),
        "loss_flow": float(loss_flow.detach().item()),
        "loss_endpoint": float(loss_endpoint.detach().item()),
        "loss_fft": float(loss_fft.detach().item()),
        "loss_1step_feat": float(loss_1step_feat.detach().item()),
        "loss_rgb": float(loss_rgb.detach().item()),
        "loss_lf_anchor": float(loss_lf_anchor.detach().item()),
        "loss_hf": float(loss_hf.detach().item()),
        "loss_drift": float(loss_drift.detach().item()),
        "loss_gate_sparse": float(loss_gate_sparse.detach().item()),
        "loss_gate_tv": float(loss_gate_tv.detach().item()),
        "loss_feature_stats": float(loss_feature_stats.detach().item()),
        "loss_feature_range": float(loss_feature_range.detach().item()),
        "pixel_loss_active": float(do_pixel),
        "f_B_mean": float(f_b.float().mean().item()),
        "f_B_std": float(f_b.float().std(unbiased=False).item()),
        "f_H_mean": float(f_h.float().mean().item()),
        "f_H_std": float(f_h.float().std(unbiased=False).item()),
        "f_hat_mean": float(f1_onestep_raw.float().mean().item()),
        "f_hat_std": float(f1_onestep_raw.float().std(unbiased=False).item()),
        "f_hat_minus_f_B_abs": float((f1_onestep_raw - f_b.float()).abs().mean().item()),
        "gate_mean": float(gate0.float().mean().item()) if gate0 is not None else float("nan"),
        "v_pred_abs": float(v_pred_n.float().abs().mean().item()),
        "v_eff_abs": float(v_eff_n.float().abs().mean().item()),
    }
    tensors = {
        "x_1step": x_1step.detach(),
        "f1_onestep": f1_onestep_raw.detach(),
        "gate0": gate0.detach() if gate0 is not None else torch.empty(0, device=device),
    }
    return loss, metrics, tensors


@torch.no_grad()
def evaluate_model(
    args: argparse.Namespace,
    vae: AutoencoderKL,
    cut: CutSpec,
    model: FeatureRectifiedFlowNet,
    stats: FeatureStats,
    val_loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    amp_dtype: torch.dtype,
    step: int,
    out_dir: Path,
    lpips_net,
    save_visuals: bool,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, Any]] = []
    sample_count = 0
    vis_dir = out_dir / "validation" / f"step_{step:07d}"
    if save_visuals:
        vis_dir.mkdir(parents=True, exist_ok=True)
    for batch in val_loader:
        if sample_count >= args.num_val:
            break
        clear_cuda(reset_peak=True, device=device)
        pair = build_feature_pair(args, vae, cut, batch, device, dtype, include_feat_bic=True, collect_timing=True)
        assert pair.x_feat_bic is not None
        synchronize(device)
        t_vec0 = time.perf_counter()
        with autocast_context(device, amp_dtype):
            f_hat_1, gate0 = sample_feature_ode(model, stats, pair.f_b, amp_dtype, 1, args.vector_scale, "euler")
            f_hat_2, _ = sample_feature_ode(model, stats, pair.f_b, amp_dtype, 2, args.vector_scale, "euler")
            f_hat_4, _ = sample_feature_ode(model, stats, pair.f_b, amp_dtype, 4, args.vector_scale, "euler")
            f_hat_heun4, _ = sample_feature_ode(model, stats, pair.f_b, amp_dtype, 4, args.vector_scale, "heun") if args.heun4 else (None, None)
            eta_outputs: dict[float, torch.Tensor] = {}
            for eta in args.sample_etas:
                if eta > 0:
                    eta_outputs[float(eta)], _ = sample_feature_ode(
                        model, stats, pair.f_b, amp_dtype, 1, args.vector_scale, "euler", eta=float(eta)
                    )
        synchronize(device)
        vector_ms = (time.perf_counter() - t_vec0) * 1000.0

        t_tail0 = time.perf_counter()
        x_hat_1 = decoder_tail_from_cut(vae, cut, f_hat_1.to(dtype=pair.f_h.dtype)).float()
        x_hat_2 = decoder_tail_from_cut(vae, cut, f_hat_2.to(dtype=pair.f_h.dtype)).float()
        x_hat_4 = decoder_tail_from_cut(vae, cut, f_hat_4.to(dtype=pair.f_h.dtype)).float()
        x_eta_vis = None
        if eta_outputs:
            eta_key = min(eta_outputs.keys(), key=lambda v: abs(v - args.eta_vis))
            x_eta_vis = decoder_tail_from_cut(vae, cut, eta_outputs[eta_key].to(dtype=pair.f_h.dtype)).float()
        if x_eta_vis is None:
            x_eta_vis = x_hat_1
        synchronize(device)
        tail_ms = (time.perf_counter() - t_tail0) * 1000.0

        base_row = {
            "step": step,
            "sample_id": pair.sample_ids[0],
            "cut_name": cut.name,
            "stage": cut.stage,
            "D_leq_k_ms": pair.time_front_ms,
            "vector_field_ms": vector_ms,
            "D_gt_k_ms": tail_ms,
            "total_one_step_ms": pair.time_front_ms + vector_ms + tail_ms,
            "vram_gb": cuda_peak_gb(),
            "gate_mean": float(gate0.float().mean().item()) if gate0 is not None else float("nan"),
            "feature_l1_base": float(F.l1_loss(pair.f_b.float(), pair.f_h.float()).item()),
            "feature_l1_1step": float(F.l1_loss(f_hat_1.float(), pair.f_h.float()).item()),
            "feature_l1_2step": float(F.l1_loss(f_hat_2.float(), pair.f_h.float()).item()),
            "feature_l1_4step": float(F.l1_loss(f_hat_4.float(), pair.f_h.float()).item()),
            "feature_fft_1step": float(fft_abs_l1(f_hat_1, pair.f_h, args.fft_loss_max_size).item()),
            "feature_delta_abs_1step": float((f_hat_1.float() - pair.f_b.float()).abs().mean().item()),
            "feature_mean_std_mismatch_1step": float(
                (f_hat_1.float().mean() - pair.f_h.float().mean()).abs().item()
                + (f_hat_1.float().std(unbiased=False) - pair.f_h.float().std(unbiased=False)).abs().item()
            ),
        }

        mode_tensors = {
            "feature_bicubic": pair.x_feat_bic,
            "rf_1step": x_hat_1,
            "rf_2step": x_hat_2,
            "rf_4step": x_hat_4,
        }
        if f_hat_heun4 is not None:
            x_heun4 = decoder_tail_from_cut(vae, cut, f_hat_heun4.to(dtype=pair.f_h.dtype)).float()
            mode_tensors["rf_heun4"] = x_heun4
        for mode, rgb in mode_tensors.items():
            rows.append(
                base_row
                | {
                    "mode": mode,
                    "rgb_l1": float(F.l1_loss(rgb, pair.x_h_rec).item()),
                    "psnr": psnr_01_from_m11(rgb, pair.x_h_rec),
                    "ssim": ssim_01_from_m11(rgb, pair.x_h_rec),
                    "lpips": lpips_metric(lpips_net, rgb, pair.x_h_rec),
                    "lf_error_vs_base": float(F.l1_loss(lowpass(rgb, 2.0), lowpass(pair.x_base, 2.0)).item()),
                    "lf_error_vs_hr": float(F.l1_loss(lowpass(rgb, 2.0), lowpass(pair.x_h_rec, 2.0)).item()),
                    "hf_error_vs_hr": float(F.l1_loss(highpass(rgb, 1.0), highpass(pair.x_h_rec, 1.0)).item()),
                    "low_frequency_drift_from_base": float(F.l1_loss(lowpass(rgb, 2.0), lowpass(pair.x_base, 2.0)).item()),
                }
            )
        for eta, f_eta in eta_outputs.items():
            x_eta = decoder_tail_from_cut(vae, cut, f_eta.to(dtype=pair.f_h.dtype)).float()
            rows.append(
                base_row
                | {
                    "mode": f"rf_1step_eta_{eta:.3f}",
                    "rgb_l1": float(F.l1_loss(x_eta, pair.x_h_rec).item()),
                    "psnr": psnr_01_from_m11(x_eta, pair.x_h_rec),
                    "ssim": ssim_01_from_m11(x_eta, pair.x_h_rec),
                    "lpips": lpips_metric(lpips_net, x_eta, pair.x_h_rec),
                    "lf_error_vs_base": float(F.l1_loss(lowpass(x_eta, 2.0), lowpass(pair.x_base, 2.0)).item()),
                    "lf_error_vs_hr": float(F.l1_loss(lowpass(x_eta, 2.0), lowpass(pair.x_h_rec, 2.0)).item()),
                    "hf_error_vs_hr": float(F.l1_loss(highpass(x_eta, 1.0), highpass(pair.x_h_rec, 1.0)).item()),
                    "low_frequency_drift_from_base": float(F.l1_loss(lowpass(x_eta, 2.0), lowpass(pair.x_base, 2.0)).item()),
                }
            )

        if save_visuals:
            save_validation_visuals(
                vis_dir,
                pair.sample_ids[0],
                pair,
                x_hat_1,
                x_hat_2,
                x_hat_4,
                x_eta_vis,
                f_hat_1,
                gate0,
                args.zoom_size,
            )
        sample_count += 1
        del pair, f_hat_1, f_hat_2, f_hat_4, x_hat_1, x_hat_2, x_hat_4
        clear_cuda()

    if rows:
        write_csv(vis_dir / "validation_metrics_per_sample.csv", rows)
    summary: dict[str, float] = {"val/samples": float(sample_count)}
    for mode in sorted({str(r["mode"]) for r in rows}):
        mode_rows = [r for r in rows if r["mode"] == mode]
        for key in [
            "rgb_l1",
            "psnr",
            "ssim",
            "lpips",
            "lf_error_vs_base",
            "lf_error_vs_hr",
            "hf_error_vs_hr",
            "feature_l1_base",
            "feature_l1_1step",
            "feature_l1_2step",
            "feature_l1_4step",
            "feature_fft_1step",
            "feature_delta_abs_1step",
            "gate_mean",
            "D_leq_k_ms",
            "vector_field_ms",
            "D_gt_k_ms",
            "total_one_step_ms",
            "vram_gb",
        ]:
            summary[f"val/{mode}/{key}"] = mean_finite(r.get(key, float("nan")) for r in mode_rows)
    append_csv(out_dir / "val_log.csv", {"step": step, **summary})
    return summary


def save_validation_visuals(
    vis_dir: Path,
    sample_id: str,
    pair: FeaturePair,
    x_hat_1: torch.Tensor,
    x_hat_2: torch.Tensor,
    x_hat_4: torch.Tensor,
    x_eta: torch.Tensor,
    f_hat_1: torch.Tensor,
    gate: torch.Tensor | None,
    zoom_size: int,
) -> None:
    sample_dir = vis_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    target_size = pair.x_h_rec.shape[-2:]
    delta_map = (f_hat_1.float() - pair.f_b.float()).abs().mean(dim=1, keepdim=True)
    gate_map = gate.float().mean(dim=1, keepdim=True) if gate is not None else torch.zeros_like(delta_map[:, :1])
    grid_items = [
        ("HR raw", image_from_01(pair.hr_01)),
        ("HR VAE target", image_from_m11(pair.x_h_rec)),
        ("LR base up", image_from_m11(pair.x_base)),
        ("feature bicubic", image_from_m11(pair.x_feat_bic if pair.x_feat_bic is not None else pair.x_base)),
        ("RF one-step", image_from_m11(x_hat_1)),
        ("RF two-step", image_from_m11(x_hat_2)),
        ("RF four-step", image_from_m11(x_hat_4)),
        ("RF eta", image_from_m11(x_eta)),
        ("abs err 1-step", abs_error_image(x_hat_1, pair.x_h_rec)),
        ("feature delta", map_to_image(delta_map, size=target_size)),
        ("gate map", map_to_image(gate_map, size=target_size)),
    ]
    make_labeled_grid(grid_items, cols=len(grid_items), cell_w=160).save(sample_dir / "comparison_grid.png")

    hf_err = highpass(x_hat_1, 1.0).sub(highpass(pair.x_h_rec, 1.0)).abs().mean(dim=1, keepdim=True)
    centers = {
        "zoom_hf_error": max_center_from_map(hf_err, target_size, zoom_size),
        "zoom_feature_delta": max_center_from_map(delta_map, target_size, zoom_size),
        "zoom_center": (target_size[1] // 2, target_size[0] // 2),
    }
    crop_sources = [
        ("HR target", image_from_m11(pair.x_h_rec)),
        ("base", image_from_m11(pair.x_base)),
        ("feat bic", image_from_m11(pair.x_feat_bic if pair.x_feat_bic is not None else pair.x_base)),
        ("RF 1", image_from_m11(x_hat_1)),
        ("RF 4", image_from_m11(x_hat_4)),
        ("err", abs_error_image(x_hat_1, pair.x_h_rec)),
    ]
    for name, center in centers.items():
        crops = [(label, crop_pil(img, center, zoom_size)) for label, img in crop_sources]
        make_labeled_grid(crops, cols=len(crops), cell_w=160).save(sample_dir / f"{name}.png")


@torch.no_grad()
def evaluate_benchmarks(
    args: argparse.Namespace,
    vae: AutoencoderKL,
    cut: CutSpec,
    model: FeatureRectifiedFlowNet,
    stats: FeatureStats,
    benchmark_loaders: dict[str, DataLoader],
    device: torch.device,
    dtype: torch.dtype,
    amp_dtype: torch.dtype,
    step: int,
    out_dir: Path,
    lpips_net,
    wandb_run=None,
) -> dict[str, float]:
    if not benchmark_loaders:
        return {}
    model.eval()
    bench_dir = out_dir / "benchmarks" / f"step_{step:07d}"
    bench_dir.mkdir(parents=True, exist_ok=True)
    all_summary: dict[str, float] = {"benchmark/step": float(step)}
    wandb_payload: dict[str, Any] = {}

    for dataset_name, loader in benchmark_loaders.items():
        rows: list[dict[str, Any]] = []
        for batch in loader:
            pair = build_feature_pair(args, vae, cut, batch, device, dtype, include_feat_bic=True, collect_timing=True)
            assert pair.x_feat_bic is not None
            with autocast_context(device, amp_dtype):
                f_hat_1, gate0 = sample_feature_ode(model, stats, pair.f_b, amp_dtype, 1, args.vector_scale, "euler")
                f_hat_4, _ = sample_feature_ode(model, stats, pair.f_b, amp_dtype, 4, args.vector_scale, "euler")
            x_hat_1 = decoder_tail_from_cut(vae, cut, f_hat_1.to(dtype=pair.f_h.dtype)).float()
            x_hat_4 = decoder_tail_from_cut(vae, cut, f_hat_4.to(dtype=pair.f_h.dtype)).float()
            hr_raw_m11 = tensor01_to_m11(pair.hr_01)

            mode_rgbs = {
                "feature_bicubic": pair.x_feat_bic,
                "rf_1step": x_hat_1,
                "rf_4step": x_hat_4,
            }
            for mode, rgb in mode_rgbs.items():
                rows.append(
                    {
                        "step": step,
                        "dataset": dataset_name,
                        "sample_id": pair.sample_ids[0],
                        "mode": mode,
                        "height": int(rgb.shape[-2]),
                        "width": int(rgb.shape[-1]),
                        "psnr_vs_vae": psnr_01_from_m11(rgb, pair.x_h_rec),
                        "ssim_vs_vae": ssim_01_from_m11(rgb, pair.x_h_rec),
                        "psnr_vs_raw": psnr_01_from_m11(rgb, hr_raw_m11),
                        "ssim_vs_raw": ssim_01_from_m11(rgb, hr_raw_m11),
                        "rgb_l1_vs_vae": float(F.l1_loss(rgb, pair.x_h_rec).item()),
                        "hf_error_vs_vae": float(F.l1_loss(highpass(rgb, 1.0), highpass(pair.x_h_rec, 1.0)).item()),
                        "lf_drift_vs_base": float(F.l1_loss(lowpass(rgb, 2.0), lowpass(pair.x_base, 2.0)).item()),
                        "lpips_vs_vae": lpips_metric(lpips_net, rgb, pair.x_h_rec),
                        "feature_l1_base": float(F.l1_loss(pair.f_b.float(), pair.f_h.float()).item()),
                        "feature_l1_1step": float(F.l1_loss(f_hat_1.float(), pair.f_h.float()).item()),
                        "feature_l1_4step": float(F.l1_loss(f_hat_4.float(), pair.f_h.float()).item()),
                        "gate_mean": float(gate0.float().mean().item()) if gate0 is not None else float("nan"),
                    }
                )

            if (
                args.save_benchmark_images
                and dataset_name.lower() == "set5"
                and pair.sample_ids[0].lower() == args.butterfly_name.lower()
            ):
                butterfly_dir = out_dir / "benchmarks" / "Set5_butterfly_trend"
                butterfly_dir.mkdir(parents=True, exist_ok=True)
                grid_path = butterfly_dir / f"step_{step:07d}_butterfly.png"
                items = [
                    ("HR raw", image_from_01(pair.hr_01)),
                    ("HR VAE target", image_from_m11(pair.x_h_rec)),
                    ("base", image_from_m11(pair.x_base)),
                    ("feature bicubic", image_from_m11(pair.x_feat_bic)),
                    ("RF one-step", image_from_m11(x_hat_1)),
                    ("RF four-step", image_from_m11(x_hat_4)),
                    ("abs err 1-step", abs_error_image(x_hat_1, pair.x_h_rec)),
                    ("feature delta", map_to_image((f_hat_1.float() - pair.f_b.float()).abs().mean(dim=1, keepdim=True), size=pair.x_h_rec.shape[-2:])),
                ]
                make_labeled_grid(items, cols=len(items), cell_w=170).save(grid_path)
                make_labeled_grid(items, cols=len(items), cell_w=170).save(butterfly_dir / "latest_butterfly.png")
                wb_img = wandb_image(grid_path, f"Set5 butterfly step {step}")
                if wb_img is not None:
                    wandb_payload["benchmark/Set5_butterfly"] = wb_img

            del pair, f_hat_1, f_hat_4, x_hat_1, x_hat_4
            clear_cuda()

        write_csv(bench_dir / f"{dataset_name}_metrics.csv", rows)
        for mode in sorted({str(r["mode"]) for r in rows}):
            mode_rows = [r for r in rows if r["mode"] == mode]
            for key in [
                "psnr_vs_vae",
                "ssim_vs_vae",
                "psnr_vs_raw",
                "ssim_vs_raw",
                "rgb_l1_vs_vae",
                "hf_error_vs_vae",
                "lf_drift_vs_base",
                "lpips_vs_vae",
                "feature_l1_base",
                "feature_l1_1step",
                "feature_l1_4step",
                "gate_mean",
            ]:
                all_summary[f"benchmark/{dataset_name}/{mode}/{key}"] = mean_finite(r.get(key, float("nan")) for r in mode_rows)
    append_csv(out_dir / "benchmark_log.csv", {"step": step, **all_summary})
    if wandb_run is not None:
        wandb_run.log({"step": step, **all_summary, **wandb_payload}, step=step)
    return all_summary


def save_checkpoint(
    path: Path,
    model: FeatureRectifiedFlowNet,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    stats: FeatureStats,
    step: int,
    best_metric: float,
    cut: CutSpec,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "best_metric": best_metric,
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "feature_stats": stats.state_dict(),
            "cut": asdict(cut),
            "args": vars(args),
        },
        path,
    )


def load_training_checkpoint(
    ckpt_path: Path,
    model: FeatureRectifiedFlowNet,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    stats: FeatureStats,
    device: torch.device,
    load_optimizer: bool = True,
) -> tuple[int, float]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if load_optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if load_optimizer and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    if "feature_stats" in ckpt:
        stats.load_state_dict(ckpt["feature_stats"], device)
    return int(ckpt.get("step", 0)), float(ckpt.get("best_metric", float("inf")))


def prune_checkpoints(ckpt_dir: Path, keep: int) -> None:
    if keep <= 0:
        return
    ckpts = sorted(ckpt_dir.glob("iter_*.pt"), key=lambda p: p.stat().st_mtime)
    for path in ckpts[:-keep]:
        path.unlink(missing_ok=True)


def train_for_cut(
    args: argparse.Namespace,
    vae: AutoencoderKL,
    cut: CutSpec,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    amp_dtype: torch.dtype,
    out_dir: Path,
    lpips_net,
    num_steps: int,
    benchmark_loaders: dict[str, DataLoader] | None = None,
    wandb_run=None,
    initial_checkpoint: Path | None = None,
    warmup: bool = False,
) -> TrainResult:
    if benchmark_loaders is None:
        benchmark_loaders = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    channels, spatial, init_pair = infer_feature_shape(args, vae, cut, train_loader, device, dtype)
    stats = FeatureStats(channels, device, eps=args.feature_stats_eps, momentum=args.feature_stats_momentum)
    stats.update(init_pair.f_b, init_pair.f_h)
    model = FeatureRectifiedFlowNet(
        feature_channels=channels,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        enable_gate=args.enable_gate,
        gate_channels=args.gate_channels,
        gate_init_bias=args.gate_init_bias,
    ).to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and dtype == torch.float16))

    start_step = 0
    best_metric = float("inf")
    if initial_checkpoint is not None and initial_checkpoint.exists():
        start_step, best_metric = load_training_checkpoint(
            initial_checkpoint, model, optimizer, scaler, stats, device, load_optimizer=True
        )
        print(f"[RESUME] {initial_checkpoint} step={start_step} best_metric={best_metric:.6f}")
    elif args.resume is not None and args.resume.exists() and not warmup:
        start_step, best_metric = load_training_checkpoint(args.resume, model, optimizer, scaler, stats, device)
        print(f"[RESUME] {args.resume} step={start_step} best_metric={best_metric:.6f}")

    config = {
        "experiment": "feature_rectified_flow_sr",
        "warmup": warmup,
        "cut": cut,
        "feature_channels": channels,
        "feature_spatial": list(spatial),
        "vae_name_or_path": args.vae_name_or_path,
        "vae_subfolder": normalize_subfolder(args.vae_name_or_path, args.vae_subfolder),
        "scale": args.scale,
        "hr_size": args.hr_size,
        "num_steps": num_steps,
        "max_hours": args.max_hours,
        "loss": {
            "flow": args.lambda_flow,
            "endpoint": args.lambda_endpoint,
            "fft": args.lambda_fft,
            "one_step_feature": args.lambda_1step_feat,
            "rgb": args.lambda_rgb,
            "lf_anchor": args.lambda_lf_anchor,
            "hf": args.lambda_hf,
            "drift": args.lambda_drift,
            "gate_sparse": args.lambda_gate_sparse,
            "gate_tv": args.lambda_gate_tv,
        },
        "args": vars(args),
    }
    write_json(out_dir / "train_config.json", config)
    write_korean_run_note(out_dir, cut, config)

    print("=" * 120)
    print("Feature-space Rectified Flow SR")
    print(f"Output dir       : {out_dir}")
    print(f"Selected cut     : {cut.name} / {cut.stage} / {cut.level_def}")
    print(f"Feature shape    : C={channels}, HxW={spatial[0]}x{spatial[1]}")
    print(f"Model            : hidden={args.hidden_channels}, blocks={args.num_blocks}, gate={args.enable_gate}")
    print(f"Training         : steps={num_steps}, batch={args.batch_size}, grad_accum={args.grad_accum}, lr={args.lr}")
    print(f"Time budget      : max_hours={args.max_hours}, remaining_hours={remaining_hours(args):.3f}")
    print(f"RF               : sigma_max={args.sigma_max}, vector_scale={args.vector_scale}, pixel_loss_every={args.pixel_loss_every}")
    print("=" * 120)

    train_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    step = start_step
    micro_step = 0
    start_time = time.time()
    last_log_time = start_time
    rolling: dict[str, list[float]] = {}

    stopped_by_time = False
    while step < num_steps:
        if time_budget_exhausted(args):
            stopped_by_time = True
            print(
                f"[TIME] max_hours={args.max_hours} reached at step={step}; "
                "saving checkpoint and running final validation."
            )
            break
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        model.train()
        need_rgb_targets = args.pixel_loss_every <= 1 or ((step + 1) % max(1, args.pixel_loss_every) == 0)
        pair = build_feature_pair(
            args,
            vae,
            cut,
            batch,
            device,
            dtype,
            include_feat_bic=False,
            include_rgb_targets=need_rgb_targets,
        )
        loss, metrics, tensors = compute_train_loss(args, vae, cut, model, stats, pair, step + 1, device, amp_dtype)
        scaler.scale(loss / max(1, args.grad_accum)).backward()
        micro_step += 1

        if micro_step % max(1, args.grad_accum) == 0:
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item())
            else:
                grad_norm = 0.0
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            metrics["grad_norm"] = grad_norm
            metrics["lr"] = float(optimizer.param_groups[0]["lr"])
            metrics["elapsed_hours"] = (time.time() - start_time) / 3600.0
            metrics["global_elapsed_hours"] = global_elapsed_hours(args)
            metrics["remaining_hours"] = remaining_hours(args)
            metrics["sec_per_step"] = (time.time() - last_log_time) / max(1, args.log_every)
            for key, value in metrics.items():
                if math.isfinite(float(value)):
                    rolling.setdefault(key, []).append(float(value))

            if step % args.log_every == 0 or step == 1:
                log_row = {
                    "step": step,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    **{f"train/{k}": float(np.mean(v[-args.log_every :])) for k, v in rolling.items()},
                }
                append_csv(out_dir / "train_log.csv", log_row)
                if wandb_run is not None:
                    wandb_run.log(log_row | {"phase": "warmup" if warmup else "main"}, step=step)
                print(
                    f"[train {step:07d}] "
                    f"loss={log_row.get('train/loss', float('nan')):.5f} "
                    f"flow={log_row.get('train/loss_flow', float('nan')):.5f} "
                    f"1step={log_row.get('train/loss_1step_feat', float('nan')):.5f} "
                    f"rgb={log_row.get('train/loss_rgb', float('nan')):.5f} "
                    f"gate={log_row.get('train/gate_mean', float('nan')):.4f} "
                    f"|df|={log_row.get('train/f_hat_minus_f_B_abs', float('nan')):.5f}"
                )
                last_log_time = time.time()

            if args.val_every > 0 and (step % args.val_every == 0 or step == 1 or (warmup and step == num_steps)):
                save_visuals = (args.vis_every > 0 and step % args.vis_every == 0) or step == 1 or warmup
                val_metrics = evaluate_model(
                    args,
                    vae,
                    cut,
                    model,
                    stats,
                    val_loader,
                    device,
                    dtype,
                    amp_dtype,
                    step,
                    out_dir,
                    lpips_net,
                    save_visuals=save_visuals,
                )
                metric = float(val_metrics.get("val/rf_1step/rgb_l1", float("inf")))
                base_l1 = float(val_metrics.get("val/feature_bicubic/rgb_l1", float("nan")))
                four_l1 = float(val_metrics.get("val/rf_4step/rgb_l1", float("nan")))
                print(f"[val {step:07d}] bic={base_l1:.5f} one={metric:.5f} four={four_l1:.5f}")
                if wandb_run is not None:
                    wandb_run.log({"step": step, **val_metrics}, step=step)
                if metric < best_metric:
                    best_metric = metric
                    save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scaler, stats, step, best_metric, cut, args)

            run_benchmark = (
                args.benchmark_every > 0
                and step % args.benchmark_every == 0
                and benchmark_loaders
                and (not warmup or args.benchmark_during_warmup)
            )
            if run_benchmark:
                bench = evaluate_benchmarks(
                    args,
                    vae,
                    cut,
                    model,
                    stats,
                    benchmark_loaders,
                    device,
                    dtype,
                    amp_dtype,
                    step,
                    out_dir,
                    lpips_net,
                    wandb_run=wandb_run,
                )
                if bench:
                    set5_one = bench.get("benchmark/Set5/rf_1step/psnr_vs_vae", float("nan"))
                    set5_four = bench.get("benchmark/Set5/rf_4step/psnr_vs_vae", float("nan"))
                    print(f"[bench {step:07d}] Set5 PSNR(VAE): one={set5_one:.3f} four={set5_four:.3f}")

            if args.save_every > 0 and (step % args.save_every == 0 or step == num_steps):
                save_checkpoint(ckpt_dir / f"iter_{step:07d}.pt", model, optimizer, scaler, stats, step, best_metric, cut, args)
                save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scaler, stats, step, best_metric, cut, args)
                prune_checkpoints(ckpt_dir, args.keep_last_checkpoints)

        del loss, metrics, tensors, pair
        clear_cuda()

    save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scaler, stats, step, best_metric, cut, args)
    final_val = evaluate_model(
        args,
        vae,
        cut,
        model,
        stats,
        val_loader,
        device,
        dtype,
        amp_dtype,
        step,
        out_dir,
        lpips_net,
        save_visuals=True,
    )
    final_benchmark = {}
    if benchmark_loaders and (not warmup or args.benchmark_during_warmup):
        final_benchmark = evaluate_benchmarks(
            args,
            vae,
            cut,
            model,
            stats,
            benchmark_loaders,
            device,
            dtype,
            amp_dtype,
            step,
            out_dir,
            lpips_net,
            wandb_run=wandb_run,
        )
    summary = {
        "final_step": step,
        "stopped_by_time": stopped_by_time,
        "max_hours": args.max_hours,
        "global_elapsed_hours": global_elapsed_hours(args),
        "best_metric_rf_1step_rgb_l1": best_metric,
        "final_val": final_val,
        "final_benchmark": final_benchmark,
        "cut": cut,
        "last_checkpoint": ckpt_dir / "last.pt",
        "best_checkpoint": ckpt_dir / "best.pt" if (ckpt_dir / "best.pt").exists() else None,
        "elapsed_hours": (time.time() - start_time) / 3600.0,
    }
    write_json(out_dir / "summary.json", summary)
    write_korean_summary(out_dir, summary)
    return TrainResult(
        cut=cut,
        final_step=step,
        best_metric=best_metric,
        last_checkpoint=ckpt_dir / "last.pt",
        best_checkpoint=ckpt_dir / "best.pt" if (ckpt_dir / "best.pt").exists() else None,
        summary=summary,
    )


def write_korean_run_note(out_dir: Path, cut: CutSpec, config: dict[str, Any]) -> None:
    lines = [
        "# Feature-space Rectified Flow SR 실행 기록",
        "",
        f"- 선택 cut: `{cut.name}` (`{cut.stage}`)",
        f"- cut 설명: {cut.level_def}",
        "- 핵심 질문: bicubic decoder feature `f_B`에서 HR decoder feature `f_H`로 가는 feature-space vector field를 학습할 수 있는가?",
        "- 학습 신호: random `t`에서 flow matching을 주 신호로 쓰고, one-step endpoint와 decoder RGB 손실을 함께 둔다.",
        "- 금지 경로: pretrained diffusion UNet, scheduler.add_noise, SD/FLUX denoising UNet 입력은 사용하지 않는다.",
        "",
        "## 아침에 먼저 볼 항목",
        "",
        "1. `validation/*/comparison_grid.png`에서 feature bicubic / RF one-step / RF four-step 비교",
        "2. `val_log.csv`의 `val/feature_bicubic/rgb_l1`, `val/rf_1step/rgb_l1`, `val/rf_4step/rgb_l1`",
        "3. `benchmark_log.csv`의 Set5/Set14/B100 PSNR/SSIM 추세",
        "4. `benchmarks/Set5_butterfly_trend/latest_butterfly.png`에서 나비 이미지 변화",
        "5. LF drift와 HF error가 같이 움직이는지 확인",
        "",
        "## 설정 요약",
        "",
        f"- feature channels: {config.get('feature_channels')}",
        f"- feature spatial: {config.get('feature_spatial')}",
        f"- sigma_max: {config.get('args', {}).get('sigma_max')}",
        f"- gate 사용: {config.get('args', {}).get('enable_gate')}",
        f"- 최대 실행 시간: {config.get('max_hours')} hours",
        f"- benchmark sets: {config.get('args', {}).get('benchmark_roots')}",
    ]
    (out_dir / "run_notes_ko.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_korean_summary(out_dir: Path, summary: dict[str, Any]) -> None:
    final_val = summary.get("final_val", {})
    final_benchmark = summary.get("final_benchmark", {})
    bic = final_val.get("val/feature_bicubic/rgb_l1", float("nan"))
    one = final_val.get("val/rf_1step/rgb_l1", float("nan"))
    four = final_val.get("val/rf_4step/rgb_l1", float("nan"))
    lines = [
        "# Feature-space Rectified Flow SR 결과 요약",
        "",
        f"- 최종 step: {summary.get('final_step')}",
        f"- 선택 cut: `{summary.get('cut', {}).get('name') if isinstance(summary.get('cut'), dict) else getattr(summary.get('cut'), 'name', '')}`",
        f"- best RF one-step RGB L1: {summary.get('best_metric_rf_1step_rgb_l1')}",
        f"- 최종 feature bicubic RGB L1: {bic}",
        f"- 최종 RF one-step RGB L1: {one}",
        f"- 최종 RF four-step RGB L1: {four}",
        f"- Set5 RF one-step PSNR(VAE): {final_benchmark.get('benchmark/Set5/rf_1step/psnr_vs_vae', float('nan'))}",
        f"- Set14 RF one-step PSNR(VAE): {final_benchmark.get('benchmark/Set14/rf_1step/psnr_vs_vae', float('nan'))}",
        f"- B100 RF one-step PSNR(VAE): {final_benchmark.get('benchmark/B100/rf_1step/psnr_vs_vae', float('nan'))}",
        "",
        "## 해석 가이드",
        "",
        "- RF four-step이 feature bicubic보다 좋아지면 feature-space vector field가 의미 있는 방향을 배웠다는 신호다.",
        "- RF one-step이 four-step에 가까우면 one-step 압축 가능성이 있다.",
        "- HF error/LPIPS는 좋아지는데 LF drift가 커지면 anchor를 강화해야 한다.",
        "- 깨진 픽셀이나 색 폭주가 보이면 sigma/vector scale을 낮추거나 cut을 바꿔야 한다.",
    ]
    (out_dir / "summary_ko.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_ambiguous_warmup(
    args: argparse.Namespace,
    vae: AutoencoderKL,
    probe_selection: dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    amp_dtype: torch.dtype,
    lpips_net,
    benchmark_loaders: dict[str, DataLoader],
    wandb_run=None,
) -> tuple[CutSpec | None, Path | None]:
    scored = probe_selection.get("scored", [])
    if len(scored) < 2 or args.ambiguous_warmup_steps <= 0:
        return None, None
    top = scored[0]
    second = scored[1]
    if float(top["auto_score"]) - float(second["auto_score"]) > args.ambiguous_score_delta:
        return None, None
    print("[PROBE] auto selection ambiguous; top-2 warm-up training 시작")
    warm_results: list[TrainResult] = []
    warm_failures: list[dict[str, Any]] = []
    for item in scored[:2]:
        cut = resolve_cut(vae.decoder, str(item["cut_name"]), fallback_name=args.cut_name)
        cut_out = args.output_dir / "warmup_top2" / cut.name.replace(".", "_")
        try:
            result = train_for_cut(
                args,
                vae,
                cut,
                train_loader,
                val_loader,
                device,
                dtype,
                amp_dtype,
                cut_out,
                lpips_net,
                num_steps=args.ambiguous_warmup_steps,
                benchmark_loaders=benchmark_loaders,
                wandb_run=wandb_run,
                initial_checkpoint=None,
                warmup=True,
            )
            warm_results.append(result)
        except torch.cuda.OutOfMemoryError as exc:
            msg = str(exc).splitlines()[0]
            print(f"[WARN] warm-up OOM for {cut.name}/{cut.stage}; skipping this candidate: {msg}")
            warm_failures.append({"cut": asdict(cut), "status": "oom", "error": msg})
            write_json(cut_out / "warmup_failure.json", warm_failures[-1])
            clear_cuda(reset_peak=True, device=device)
            continue
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"[WARN] warm-up failed for {cut.name}/{cut.stage}; skipping this candidate: {msg}")
            warm_failures.append({"cut": asdict(cut), "status": "failed", "error": msg})
            write_json(cut_out / "warmup_failure.json", warm_failures[-1])
            clear_cuda(reset_peak=True, device=device)
            continue
    write_json(args.output_dir / "warmup_top2" / "warmup_selection.json", {
        "completed": [
            {
                "cut": asdict(r.cut),
                "best_metric": r.best_metric,
                "last_checkpoint": r.last_checkpoint,
                "best_checkpoint": r.best_checkpoint,
            }
            for r in warm_results
        ],
        "failures": warm_failures,
    })
    if not warm_results:
        print("[WARN] all ambiguous warm-up candidates failed; falling back to probe top cut.")
        return resolve_cut(vae.decoder, str(top["cut_name"]), fallback_name=args.cut_name), None
    warm_results.sort(key=lambda r: r.best_metric)
    best = warm_results[0]
    print(f"[PROBE] warm-up winner={best.cut.name}, best one-step rgb_l1={best.best_metric:.6f}")
    return best.cut, best.best_checkpoint or best.last_checkpoint


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    args.run_started_at = time.time()
    set_seed(args.seed)
    if args.hr_size % args.scale != 0:
        raise ValueError("--hr_size must be divisible by --scale")
    if args.hr_size % args.vae_multiple != 0 or (args.hr_size // args.scale) % args.vae_multiple != 0:
        raise ValueError("--hr_size and LR size must be divisible by --vae_multiple")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false.")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32
    dtype = dtype_from_name(args.precision)
    amp_dtype = dtype

    train_paths = list_images(args.data_root, args.num_train_images)
    if args.val_root.exists():
        val_paths = list_images(args.val_root, args.num_val_images)
    else:
        train_paths, val_paths = split_train_val(train_paths, args.train_ratio_if_no_val, args.seed)

    train_loader = DataLoader(
        HRImageDataset(train_paths, args.hr_size, args.scale, random_crop=True, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        HRImageDataset(val_paths, args.hr_size, args.scale, random_crop=False, seed=args.seed + 17),
        batch_size=1,
        shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 1,
    )
    probe_loader = DataLoader(
        HRImageDataset(train_paths[: max(args.probe_num_images, 1)], args.hr_size, args.scale, random_crop=False, seed=args.seed + 31),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    vae = load_vae(args, dtype, device)
    lpips_net = build_lpips(device)
    benchmark_loaders = build_benchmark_loaders(args, device)
    run_config = {
        "script": Path(__file__).name,
        "data_root": args.data_root,
        "val_root": args.val_root,
        "num_train": len(train_paths),
        "num_val": len(val_paths),
        "benchmark_sets": {name: len(loader.dataset) for name, loader in benchmark_loaders.items()},
        "args": vars(args),
    }
    write_json(args.output_dir / "run_config.json", run_config)
    wandb_run = init_wandb(args, run_config)

    selected_cut: CutSpec | None = None
    probe_summary: list[dict[str, Any]] = []
    probe_selection: dict[str, Any] = {}
    initial_checkpoint: Path | None = None

    if args.force_cut:
        selected_cut = resolve_cut(vae.decoder, args.force_cut, fallback_name=args.cut_name)
        probe_selection = {"reason": f"--force_cut used: {args.force_cut}"}
        print(f"[CUT] force_cut={selected_cut.name} stage={selected_cut.stage}")
    elif args.auto_probe_cuts:
        selected_cut, probe_summary, probe_selection = cut_probe(args, vae, probe_loader, device, dtype, lpips_net)
        warm_cut, warm_ckpt = maybe_ambiguous_warmup(
            args,
            vae,
            probe_selection,
            train_loader,
            val_loader,
            device,
            dtype,
            amp_dtype,
            lpips_net,
            benchmark_loaders,
            wandb_run=wandb_run,
        )
        if warm_cut is not None:
            selected_cut = warm_cut
            initial_checkpoint = warm_ckpt
    else:
        selected_cut = resolve_cut(vae.decoder, args.cut_name, fallback_name="f3")
        probe_selection = {"reason": f"auto probing disabled; using --cut_name {args.cut_name}"}

    if selected_cut is None:
        fallback = args.cut_name if args.cut_name else "f3"
        try:
            selected_cut = resolve_cut(vae.decoder, fallback, fallback_name="f3")
            probe_selection["fallback"] = f"auto probing failed; fallback to {selected_cut.name}/{selected_cut.stage}"
        except Exception:
            selected_cut = resolve_cut(vae.decoder, "f3", fallback_name="up_blocks.0")
            probe_selection["fallback"] = f"hard fallback to {selected_cut.name}/{selected_cut.stage}"

    write_json(
        args.output_dir / "selected_cut.json",
        {
            "selected_cut_name": selected_cut.name,
            "selected_cut_stage": selected_cut.stage,
            "selected_cut_reason": probe_selection.get("reason", ""),
            "selected_cut": selected_cut,
            "probe_metrics_for_all_cuts": probe_summary,
            "probe_selection": probe_selection,
            "initial_checkpoint": initial_checkpoint,
        },
    )
    print(
        f"[CUT] selected_cut_name={selected_cut.name}, stage={selected_cut.stage}, "
        f"reason={probe_selection.get('reason', '')}"
    )

    if not args.probe_then_train:
        print("[DONE] --no-probe_then_train specified; stopping after probe/selection.")
        return

    result = train_for_cut(
        args,
        vae,
        selected_cut,
        train_loader,
        val_loader,
        device,
        dtype,
        amp_dtype,
        args.output_dir / "train_main",
        lpips_net,
        num_steps=args.num_steps,
        benchmark_loaders=benchmark_loaders,
        wandb_run=wandb_run,
        initial_checkpoint=initial_checkpoint,
        warmup=False,
    )
    write_json(
        args.output_dir / "summary.json",
        {
            "selected_cut": selected_cut,
            "selected_cut_reason": probe_selection.get("reason", ""),
            "train_result": result.summary,
            "last_checkpoint": result.last_checkpoint,
            "best_checkpoint": result.best_checkpoint,
            "max_hours": args.max_hours,
            "global_elapsed_hours": global_elapsed_hours(args),
            "morning_priority": "Compare feature bicubic vs RF one-step vs RF four-step in train_main/validation.",
        },
    )
    print("\n[SUMMARY]")
    print(f"- selected_cut_name: {selected_cut.name}")
    print(f"- selected_cut_stage: {selected_cut.stage}")
    print(f"- best one-step RGB L1: {result.best_metric:.6f}")
    print(f"- last checkpoint: {result.last_checkpoint}")
    print(f"- output_dir: {args.output_dir}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
