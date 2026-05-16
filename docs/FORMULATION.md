# Formulation

This document spells out the experiment as a small bridge/rectified-flow problem
in a frozen VAE decoder feature space.

## Frozen Decoder Split

Let `E` be the frozen FLUX VAE encoder and `D` be the frozen decoder. For an
internal decoder feature cut `k`, split:

```text
D = D_>k o D_<=k
```

`D_<=k` maps a latent tensor into an intermediate decoder feature tensor.
`D_>k` maps that feature tensor to RGB.

In this run the selected cut was:

```text
k = f3 = decoder.up_blocks.1
```

The feature shape for 512 HR crops at f3 was:

```text
f3: 1 x 512 x 256 x 256
```

## Pair Construction

For each HR crop:

```text
x_HR: HR crop, 512 x 512 by default
x_LR = BicubicDownsample(x_HR, scale=2)
z_H = E(x_HR)
z_L = E(x_LR)
f_H = D_<=k(z_H)
f_L = D_<=k(z_L)
f_B = Bicubic(f_L, spatial_size=f_H)
x_H_rec = D_>k(f_H)
x_base = Bicubic(D(z_L), size=x_H_rec)
```

The transport task is:

```text
f_B -> f_H
```

The pixel-space target is `x_H_rec`, not raw `x_HR`, because the method is
designed to learn a path on the frozen VAE decoder's own feature manifold.

## Rectified-Flow Bridge

Define:

```text
f_0 = f_B
f_1 = f_H
t ~ Uniform(0, 1)
eps ~ Normal(0, I)
sigma(t) = sigma_max * t * (1 - t)
f_t = (1 - t) * f_0 + t * f_1 + sigma(t) * eps
v_target = f_1 - f_0
```

Train a vector field:

```text
v_theta(f_t, t, cond=f_0) ~= v_target
```

At inference:

```text
f_hat_1step = f_B + v_theta(f_B, t=0, cond=f_B)
x_hat_1step = D_>k(f_hat_1step)
```

Few-step Euler diagnostics:

```text
f <- f_B
for t in {0, 1/N, ..., (N-1)/N}:
    f <- f + (1/N) * v_theta(f, t, cond=f_B)
x_hat_N = D_>k(f)
```

## Feature Normalization

The script maintains running per-channel statistics for decoder features:

```text
mean_c, std_c over f_B and f_H
```

The vector field is trained in normalized feature coordinates. Predicted
velocity is converted back to raw feature coordinates before passing through
`D_>k`.

This helped avoid feature-scale instability and color explosions.

## Losses

The total loss used in the overnight run was:

```text
L = 1.0   * L_flow
  + 0.5   * L_endpoint
  + 0.1   * L_fft
  + 1.0   * L_1step_feat
  + 0.5   * L_rgb
  + 0.2   * L_lf_anchor
  + 0.2   * L_hf
  + 0.005 * L_drift
  + 0.001 * L_gate_sparse
  + 0.001 * L_gate_tv
```

Where:

```text
L_flow        = Charbonnier(v_eff, f_H - f_B)
L_endpoint    = Charbonnier(f_t + (1 - t) * v_eff, f_H)
L_fft         = L1(abs(FFT2(f_t + (1 - t) * v_eff)), abs(FFT2(f_H)))
L_1step_feat  = Charbonnier(f_B + v_theta(f_B,0), f_H)
L_rgb         = Charbonnier(D_>k(f_B + v_theta(f_B,0)), x_H_rec)
L_lf_anchor   = L1(LP(x_1step), LP(x_base))
L_hf          = L1(HP(x_1step), HP(x_H_rec))
L_drift       = mean(abs(f_1step - f_B))
```

`v_eff = gate * v_pred` when the gate is enabled.

## What This Is Not

This experiment deliberately avoids:

- pretrained diffusion UNets
- scheduler noise paths
- `scheduler.add_noise()`
- SDXL/SD2.1/FLUX denoising transformers
- a plain endpoint-only deterministic feature projector as the main method

The random-time flow-matching term is the central training signal.
