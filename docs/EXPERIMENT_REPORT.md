# Experiment Report

## Goal

Test whether an overnight feature-space rectified-flow model can transport a
bicubic-upsampled decoder feature `f_B` toward the HR decoder feature `f_H`, then
produce a useful one-step super-resolution output through the frozen VAE decoder
tail.

## Setup

```text
VAE: black-forest-labs/FLUX.1-dev, vae subfolder
Scale: x2
HR crop: 512
Precision: bf16
GPU: NVIDIA RTX 3090 24 GB
Batch size: 1
Gradient accumulation: 8
Model width: 128
Blocks: 8
Gate: enabled
sigma_max: 0.03
```

Training data:

```text
DIV2K_train_HR
```

Evaluation data:

```text
Set5, Set14, B100, Urban100, Manga109, FLUX179 generated images
```

## Cut Probe Outcome

The f1-f5 cut probe selected `f3 = decoder.up_blocks.1`.

Top two cuts:

```text
f3 score = 0.4707
f4 score = 0.4315
```

`f4` was promising by probe metrics, but the default warm-up attempted at 512
OOMed because the f4 tensor is spatially very large:

```text
f4 feature shape: 1 x 256 x 512 x 512
```

The f3 tensor is earlier/lower resolution:

```text
f3 feature shape: 1 x 512 x 256 x 256
```

This made f3 the practical choice for the overnight run.

## Training Timeline

| Phase | End Step | Time |
|---|---:|---:|
| f3 warm-up | 2000 | 3.26 h |
| f3 main resume | 6063 | 6.72 h |
| Total f3 updates | 6063 | ~9.98 h |

The main run stopped by time budget at step 6063 and saved final checkpoints.

## Internal Validation

Final internal validation against `x_H_rec`:

| Method | RGB L1 | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| feature bicubic | 0.12526 | 20.62 | 0.5166 | 0.2319 |
| RF one-step | 0.09245 | 23.49 | 0.6565 | 0.0776 |

## Benchmark Summary

Primary VAE-target metrics:

| Dataset | Feature Bicubic PSNR | RF 1-step PSNR | RF 1-step SSIM |
|---|---:|---:|---:|
| Set5 | 24.606 | 28.478 | 0.8301 |
| Set14 | 22.713 | 26.161 | 0.7321 |
| B100 | 22.465 | 25.516 | 0.6834 |
| Urban100 | 20.022 | 24.114 | 0.7488 |
| Manga109 | 21.679 | 26.942 | 0.8508 |
| FLUX179 | 27.209 | 30.964 | 0.8853 |

All values are measured against the VAE reconstruction target.

Contextual raw-HR RGB comparison:

| Dataset | RF 1-step | LUA x2 | LSRNA x2 |
|---|---:|---:|---:|
| Set5 | 28.026 / 0.8138 | 27.988 / 0.8297 | 15.772 / 0.3903 |
| Set14 | 25.566 / 0.7058 | 26.085 / 0.7406 | 15.116 / 0.3744 |
| B100 | 25.284 / 0.6742 | 25.850 / 0.7142 | 15.325 / 0.3709 |
| Urban100 | 23.764 / 0.7381 | 24.985 / 0.7861 | 14.253 / 0.3965 |
| Manga109 | 26.549 / 0.8382 | 27.468 / 0.8647 | 15.385 / 0.5344 |

The raw-HR table is contextual, not a strict leaderboard: RF uses this run's
raw RGB logs, LUA uses the local FLUX VAE x2 benchmark, and LSRNA uses the local
SDXL VAE x2 benchmark.

## Base and Detail Check

A second post-hoc evaluator measured whether outputs preserve the LR/base image
after downsampling while changing high-frequency content.

Macro average over Set5/Set14/B100/Urban100/Manga109:

| Method | Raw RGB PSNR | Raw RGB SSIM | Base L1 RGB | Base Grad L1 |
|---|---:|---:|---:|---:|
| RF one-step | 25.837 | 0.7540 | 0.0300 | 0.0767 |
| LUA x2 | 26.475 | 0.7871 | 0.0237 | 0.0155 |
| LSRNA x2 | 15.170 | 0.4133 | 0.1123 | 0.0671 |

For FLUX179 generated images, RF was also run as LR-only 1024 -> 2048
inference. This matches the existing generated x2 visual setup.

| Method | Base PSNR RGB | Base SSIM RGB | Base L1 RGB | HF Gain vs Base |
|---|---:|---:|---:|---:|
| feature bicubic | 31.380 | 0.9048 | 0.01533 | 1.758 |
| RF one-step | 34.245 | 0.9460 | 0.01360 | 1.156 |

On the shared generated 5-image visual subset, RF preserves base nearly as well
as LUA while increasing high-frequency energy more than LUA. LSRNA changes
global content in this subset, which shows up as very poor base PSNR/SSIM.

## Interpretation

The result is positive for the experiment's core question:

1. RF one-step improves over feature bicubic on all reported VAE-target sets,
   including Manga109 and the generated FLUX179 set.
   This is the main evidence that the f3 vector field is useful.

2. Against LUA, RF one-step is roughly competitive on Set5 and behind on
   Set14/B100/Urban100. Against the current LSRNA snapshot it is much stronger.

3. Multi-step Euler remains a diagnostic only.
   It was useful for checking whether the vector field behaves like a flow, but
   the method claim should be the one-step result.

4. One-step improves high-frequency perceptual appearance without large
   low-frequency drift.
   The LF anchor appears strong enough for this run.

5. The vector field is not the largest runtime component.
   The frozen decoder tail dominates one-step inference cost at large outputs.

## Non-SOTA Disclaimer

This is not a claim of state-of-the-art SR. The target is VAE reconstruction,
not raw HR. The value of this repo is the controlled feature-space transport
experiment and the finding that a random-time rectified-flow objective at f3 can
be useful after only about ten hours on a single 3090.

## Raw Files

Raw summaries copied from the run are in:

```text
results/raw/
```

Compact tables are in:

```text
results/tables/
```
