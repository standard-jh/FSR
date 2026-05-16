# Datasets

## Training

The overnight run used DIV2K HR training images:

```text
/home/juhwan/Documents/sr/BasicSR/datasets/DIV2K/DIV2K_train_HR
```

For each HR image:

1. A 512 x 512 crop was sampled.
2. The LR input was produced by bicubic downsampling with scale 2.
3. HR and LR were encoded by the frozen FLUX VAE.
4. Decoder features were extracted at the selected cut.

The validation root passed to the command was:

```text
/home/juhwan/Documents/sr/BasicSR/datasets/DIV2K/DIV2K_valid_HR
```

In the local environment that path was missing, so the script fell back to a
train split for internal validation.

## Benchmarks

The benchmark paths used locally were:

```text
/home/juhwan/Documents/sr/BasicSR/datasets/Set5
/home/juhwan/Documents/sr/BasicSR/datasets/Set14
/home/juhwan/Documents/sr/BasicSR/datasets/B100
/home/juhwan/Documents/sr/BasicSR/datasets/Urban100
```

Resolution handling:

- Set5/Set14 use `GTmod12` when available.
- B100 and Urban100 use `HR` when available.
- Images are center-cropped to a multiple compatible with the scale and VAE.

## Metric Target

Main reported metrics are against:

```text
x_H_rec = D_>k(D_<=k(E(x_HR)))
```

This differs from conventional SR papers that report against raw HR. The reason
is that this experiment tests transport on the frozen VAE decoder feature
manifold. Raw-HR metrics are stored in the raw CSVs for reference.
