#!/usr/bin/env bash
set -euo pipefail

# SFT 训练（DeepSpeed 由 configs/accelerate.yaml 引入，真正参数见 configs/ds_zero3.json）。
# 单卡：accelerate launch --config_file configs/accelerate.yaml --num_processes 1
accelerate launch \
  --config_file configs/accelerate.yaml \
  src/train/sft.py \
  --config configs/sft.yaml