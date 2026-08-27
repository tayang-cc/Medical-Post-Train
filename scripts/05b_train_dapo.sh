#!/usr/bin/env bash
set -euo pipefail

# DAPO 训练（token-level loss + clip-higher + dynamic sampling + overlong shaping）。
# 先启动 Judge 模型的 vLLM 服务（连续打分用）：
#   bash scripts/serve_vllm.sh Qwen/Qwen2.5-72B-Instruct 8001 1
# 单卡：accelerate launch --config_file configs/accelerate.yaml --num_processes 1
#   （并保证 num_generations 能整除 per_device_train_batch_size x num_processes）

accelerate launch \
  --config_file configs/accelerate.yaml \
  src/train/dapo.py \
  --config configs/dapo.yaml