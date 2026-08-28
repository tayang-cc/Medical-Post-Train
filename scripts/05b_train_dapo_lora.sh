#!/usr/bin/env bash
set -euo pipefail

# LoRA 版 DAPO（单卡低显存，RTX 6000 Ada 48G / 5090 均可用）。
# 同 GRPO LoRA：直接 python 单进程运行，勿走 accelerate/deepspeed。
#
# 前置：
#   1. 先起 Judge vLLM（端口 8001，可 AWQ 量化）：
#        bash scripts/serve_vllm.sh Qwen/Qwen2.5-32B-Instruct-AWQ 8001 1
#   2. 单卡时 per_device_train_batch_size 需 >= num_generations（本脚本默认 --per-device-train-batch-size 8）

python src/train/dapo.py \
  --config configs/dapo.yaml \
  --lora \
  --per-device-train-batch-size 8