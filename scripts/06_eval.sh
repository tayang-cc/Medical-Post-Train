#!/usr/bin/env bash
set -euo pipefail

# 评测（双轨：抽取准确率 + LLM-as-Judge 质量分）。
# 注意：CMB-test 官方文件不含答案，评测集用带答案的 CMB-val（240 题）；
# 需要更大评测集可另从 train 切 hold-out（train 带答案）。
#
# 1. 先用 vLLM 启动待评测模型（端口 8000）：
#      bash scripts/serve_vllm.sh checkpoints/grpo 8000 1
# 2. 第二轨 LLM-as-Judge 再起一个模型（端口 8001，可用 AWQ 量化降显存）：
#      bash scripts/serve_vllm.sh Qwen/Qwen2.5-32B-Instruct-AWQ 8001 1
#    若不起 judge 服务，去掉 --judge-model 参数即可跑单轨。

python src/eval/medqa.py \
  --data data/raw/cmb_val.jsonl \
  --base-url http://localhost:8000/v1 \
  --model checkpoints/grpo \
  --judge-base-url http://localhost:8001/v1 \
  --judge-model Qwen/Qwen2.5-32B-Instruct-AWQ \
  --out logs/eval_medqa.json