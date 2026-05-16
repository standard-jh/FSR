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

| Dataset | feature bicubic PSNR | RF 1-step PSNR | RF 1-step SSIM |
|---|---:|---:|---:|
| Set5 | 24.606 | 28.478 | 0.8301 |
| Set14 | 22.713 | 26.161 | 0.7321 |
| B100 | 22.465 | 25.516 | 0.6834 |
| Urban100 | 20.022 | 24.114 | 0.7488 |
| Manga109 | 21.679 | 26.942 | 0.8508 |
| FLUX179 | 27.209 | 30.964 | 0.8853 |

가장 중요한 관찰은 다음입니다.

1. RF 1-step이 feature bicubic보다 모든 benchmark에서 확실히 좋습니다.
   따라서 f3 feature space에서 학습한 vector field가 실제로 유효합니다.

2. 이 repo의 main comparison은 RF 4-step이 아니라 RF 1-step vs LUA/LSRNA입니다.
   RF 4-step은 vector field 진단용 artifact로만 남겼습니다.

3. Set5/Set14/B100/Urban100/Manga109 및 생성형 FLUX179 모두에서 같은
   방향의 개선이 나왔습니다.

4. SOTA 주장은 아니지만, 실험 가설에는 꽤 긍정적인 결과입니다.

Raw-HR RGB 기준으로 LUA/LSRNA와 비교하면 다음과 같습니다.

| Dataset | RF 1-step | LUA x2 | LSRNA x2 |
|---|---:|---:|---:|
| Set5 | 28.026 / 0.8138 | 27.988 / 0.8297 | 15.772 / 0.3903 |
| Set14 | 25.566 / 0.7058 | 26.085 / 0.7406 | 15.116 / 0.3744 |
| B100 | 25.284 / 0.6742 | 25.850 / 0.7142 | 15.325 / 0.3709 |
| Urban100 | 23.764 / 0.7381 | 24.985 / 0.7861 | 14.253 / 0.3965 |
| Manga109 | 26.549 / 0.8382 | 27.468 / 0.8647 | 15.385 / 0.5344 |

단, 이 표는 엄밀한 leaderboard가 아니라 contextual comparison입니다.
RF는 이번 f3 실험의 raw RGB log이고, LUA는 FLUX VAE x2 benchmark,
LSRNA는 SDXL VAE x2 benchmark에서 온 값입니다. 논문화하려면 세 방법을
같은 evaluator, 같은 crop border, 같은 color space에서 다시 돌리는 편이
맞습니다.

## base 유지와 detail 생성

우리가 실제로 보고 싶었던 것은 "RGB/base 구조는 유지하면서 latent/decoder
feature 쪽에서 디테일이 생기는가?"였습니다. 그래서 별도 post-hoc evaluator를
추가해 SR 결과를 다시 LR/base 해상도로 내렸을 때 얼마나 원래 base와 맞는지
계산했습니다.

Set5/Set14/B100/Urban100/Manga109 macro 평균:

| Method | Raw RGB PSNR | Raw RGB SSIM | Base L1 RGB | Base Grad L1 |
|---|---:|---:|---:|---:|
| RF 1-step | 25.837 | 0.7540 | 0.0300 | 0.0767 |
| LUA x2 | 26.475 | 0.7871 | 0.0237 | 0.0155 |
| LSRNA x2 | 15.170 | 0.4133 | 0.1123 | 0.0671 |

생성형 FLUX179에서는 1024 이미지를 base로 두고 RF를 1024 -> 2048 LR-only
inference로도 돌렸습니다. 전체 179장 기준:

| Method | Base PSNR RGB | Base SSIM RGB | Base L1 RGB | HF Gain |
|---|---:|---:|---:|---:|
| feature bicubic | 31.380 | 0.9048 | 0.01533 | 1.758 |
| RF 1-step | 34.245 | 0.9460 | 0.01360 | 1.156 |

기존 LUA/LSRNA 생성형 x2 visual subset 5장과 같은 축에서 보면:

| Method | Base PSNR RGB | Base SSIM RGB | Base L1 RGB | HF Gain vs Bicubic |
|---|---:|---:|---:|---:|
| bicubic x2 | 41.937 | 0.9875 | 0.00426 | 1.000 |
| RF 1-step | 33.784 | 0.9153 | 0.01378 | 1.285 |
| LUA x2 | 34.279 | 0.9185 | 0.01307 | 0.992 |
| LSRNA x2 | 9.673 | 0.3717 | 0.25925 | 1.609 |

해석하면 RF는 생성형 x2에서 LUA와 비슷한 수준으로 base를 유지하면서, LUA보다
high-frequency 변화량이 더 큽니다. 반면 이 LSRNA snapshot은 high-frequency
변화는 크지만 global/base가 많이 바뀌는 쪽입니다.

생성형 비교 이미지는 다음 위치에 저장했습니다.

```text
assets/generated_flux179_rf_lua_lsrna/
```

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
