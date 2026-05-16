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
Set5, Set14, B100, Urban100
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
| RF four-step | 0.09435 | 23.31 | 0.6552 | 0.0654 |

## Benchmark Summary

| Dataset | Feature Bicubic PSNR | RF 1-step PSNR | RF 4-step PSNR |
|---|---:|---:|---:|
| Set5 | 24.606 | 28.478 | 28.406 |
| Set14 | 22.713 | 26.161 | 26.014 |
| B100 | 22.465 | 25.516 | 25.347 |
| Urban100 | 20.022 | 24.114 | 23.874 |

All values are measured against the VAE reconstruction target.

## Interpretation

The result is positive for the experiment's core question:

1. Four-step RF improves over feature bicubic.
   This suggests the learned feature-space vector field is meaningful.

2. One-step RF is close to four-step RF.
   This suggests that at f3 the learned vector field can be used in a compressed
   one-step form.

3. One-step improves high-frequency perceptual appearance without large
   low-frequency drift.
   The LF anchor appears strong enough for this run.

4. The vector field is not the largest runtime component.
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
