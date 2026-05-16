#!/usr/bin/env bash
set -euo pipefail

python scripts/train_feature_rectified_flow_sr.py \
  --data_root /home/juhwan/Documents/sr/BasicSR/datasets/DIV2K/DIV2K_train_HR \
  --val_root /home/juhwan/Documents/sr/BasicSR/datasets/DIV2K/DIV2K_valid_HR \
  --vae_name_or_path black-forest-labs/FLUX.1-dev \
  --vae_subfolder vae \
  --scale 2 \
  --hr_size 512 \
  --auto_probe_cuts \
  --probe_cuts f1 f2 f3 f4 f5 \
  --probe_num_images 32 \
  --probe_then_train \
  --topk_train 1 \
  --precision bf16 \
  --batch_size 1 \
  --grad_accum 8 \
  --num_steps 30000 \
  --max_hours 10 \
  --lr 2e-4 \
  --sigma_max 0.03 \
  --hidden_channels 128 \
  --num_blocks 8 \
  --enable_gate \
  --val_every 1000 \
  --vis_every 1000 \
  --save_every 1000 \
  --num_val 8 \
  --benchmark_roots \
    /home/juhwan/Documents/sr/BasicSR/datasets/Set5 \
    /home/juhwan/Documents/sr/BasicSR/datasets/Set14 \
    /home/juhwan/Documents/sr/BasicSR/datasets/B100 \
  --benchmark_every 1000 \
  --save_benchmark_images \
  --sample_etas 0.0 0.03 0.05 0.10 \
  --allow_tf32 \
  --wandb \
  --wandb_project feature-rectified-flow-sr \
  --wandb_name decoder_feature_flow_sr_f3_x2 \
  --output_dir runs/decoder_feature_flow_sr_f3_x2
