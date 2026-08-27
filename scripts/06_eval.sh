#!/usr/bin/env bash
set -euo pipefail

# 评测。先用 vLLM 启动待评测模型：
# vllm serve checkpoints/grpo --port 8000

python src/eval/medqa.py \
  --data data/raw/medqa_test.jsonl \
  --base-url http://localhost:8000/v1 \
  --model checkpoints/grpo \
  --out logs/eval_medqa.json
