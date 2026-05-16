# 한국어 요약

이 repository는 FLUX VAE decoder의 중간 feature space에서 rectified-flow
vector field를 학습한 x2 super-resolution 실험을 정리한 것입니다.

## 핵심 질문

기존 latent SR이나 단순 feature regression이 아니라, frozen VAE decoder의
중간 feature `f3`에서 다음 transport를 학습할 수 있는지 확인했습니다.

```text
f_B = Bicubic(D_<=f3(E(x_LR)))
f_H = D_<=f3(E(x_HR))
```

학습 목표는 다음 vector field입니다.

```text
v_theta(f_t, t, cond=f_B) ~= f_H - f_B
```

추론에서는 한 번만 적용합니다.

```text
f_hat = f_B + v_theta(f_B, t=0, cond=f_B)
x_hat = D_>f3(f_hat)
```

즉, diffusion UNet을 한 step 돌린 것이 아니라, decoder feature space에서
직접 학습한 rectified-flow field를 one-step Euler로 적용한 실험입니다.

## 왜 f3인가?

f1-f5 cut probe를 먼저 수행했습니다.

| Cut | Stage | Score | 해석 |
|---|---|---:|---|
| f3 | up_blocks.1 | 0.4707 | 선택된 cut. 안정성과 개선 여지가 가장 좋았음 |
| f4 | up_blocks.2 | 0.4315 | probe 2위. 기본 512 학습에서는 OOM |
| f2 | up_blocks.0 | 0.0014 | decoder sensitivity와 drift가 큼 |
| f1 | conv_in | -0.5271 | 너무 early feature라 불안정 |
| f5 | conv_act | -0.7000 | feature gap이 거의 작고 sensitivity가 큼 |

f4도 가능성은 있었지만 feature shape가 `1 x 256 x 512 x 512`라
hidden 128 / blocks 8 / 512 crop 학습에서 OOM이 났습니다. f4는 추후
`hr_size=384`, `hidden=64/96`, `num_blocks=4`, `pixel_loss_every=4`,
또는 feature tiling이 필요합니다.

## 학습 시간과 step

RTX 3090 24GB 한 장에서 진행했습니다.

| Phase | Step | 시간 |
|---|---:|---:|
| f3 warm-up | 2000 | 3.26시간 |
| f3 main resume | 6063까지 | 6.72시간 |
| 전체 f3 update | 6063 steps | 약 9.98시간 |

즉, 약 10시간 / 6063 optimizer steps만으로 feature bicubic 대비 명확한
개선이 나왔습니다.

## 결과

VAE reconstruction target 기준입니다.

| Dataset | feature bicubic PSNR | RF 1-step PSNR | RF 4-step PSNR |
|---|---:|---:|---:|
| Set5 | 24.606 | 28.478 | 28.406 |
| Set14 | 22.713 | 26.161 | 26.014 |
| B100 | 22.465 | 25.516 | 25.347 |
| Urban100 | 20.022 | 24.114 | 23.874 |

가장 중요한 관찰은 다음입니다.

1. RF 4-step이 feature bicubic보다 확실히 좋습니다.
   따라서 feature-space vector field 자체가 의미 있습니다.

2. RF 1-step이 RF 4-step과 거의 비슷하고, PSNR/L1에서는 오히려 조금 더
   좋았습니다.
   따라서 f3에서는 one-step 압축이 잘 되는 편입니다.

3. Set5/Set14/B100/Urban100 모두에서 같은 방향의 개선이 나왔습니다.

4. SOTA 주장은 아니지만, 실험 가설에는 꽤 긍정적인 결과입니다.

## 속도

x2 기준으로는 기존 LUA full pipeline보다 빠르게 측정됐습니다.

| Method | Input -> Output | Total |
|---|---:|---:|
| DecoderFeatureFlowSR | 512 -> 1024 | 300.8 ms |
| LUA x2 | 512 -> 1024 | 421.1 ms |
| DecoderFeatureFlowSR | 1024 -> 2048 | 1.22 s |
| LUA x2 | 1024 -> 2048 | 1.93 s |

다만 병목은 vector field 자체가 아니라 FLUX VAE decoder tail입니다.
1024 output 기준으로 RF vector field는 약 87 ms이고 decoder tail은 약
157 ms였습니다.

## 결론

이 실험은 "decoder feature space에서 rectified flow를 학습하면, 단순
feature bicubic보다 나은 one-step SR이 가능한가?"라는 질문에 대해
긍정적인 신호를 보여줍니다.

논문 SOTA 실험이라기보다는, 다음 연구 방향을 열어주는 compact prototype에
가깝습니다.

다음 단계는 f4 축소 학습, tiled 4096 inference, raw-HR benchmark protocol,
x4 모델 학습입니다.
