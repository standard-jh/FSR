# Runtime Notes

## RF x2 Inference

Measured on RTX 3090 with bf16.

| Output | Front | Vector Field | Decoder Tail | Total | Peak |
|---:|---:|---:|---:|---:|---:|
| 512 | 15.7 ms | 22.9 ms | 39.6 ms | 78.2 ms | 0.98 GiB |
| 1024 | 56.3 ms | 87.3 ms | 157.4 ms | 300.8 ms | 3.37 GiB |
| 2048 | 240.5 ms | 347.4 ms | 634.8 ms | 1.22 s | 12.96 GiB |

The vector field is not the largest component. The decoder tail becomes the
main bottleneck as output resolution grows.

## LUA x2 Comparison

Measured on the same GPU.

| Output | Encode | LUA Model | Decode | Total | Peak |
|---:|---:|---:|---:|---:|---:|
| 1024 | 31.3 ms | 140.6 ms | 249.3 ms | 421.1 ms | 2.79 GiB |
| 2048 | 132.5 ms | 605.6 ms | 1192.2 ms | 1.93 s | 9.94 GiB |

The f3 RF path is faster in this x2 setup because it decodes only from f3
through the decoder tail, while LUA produces an HR latent and then pays the full
VAE decode cost.

## x4 Caveat

Existing SwinIR fp16 RGB benchmark:

```text
1024 -> 4096 x4: 3484.3 ms, 7.41 GiB
```

This is not a direct comparison to the current f3 RF run because the RF model
here is x2. Fair x4 comparison requires either training an x4 feature-flow model
or evaluating a two-stage/tiled x2 pipeline.
