#!/usr/bin/env bash
set -euo pipefail

# 通用 vLLM 服务启动脚本（教师模型 / Judge 模型均可）。
#
# 用法:
#   bash scripts/serve_vllm.sh <模型> <端口> <张量并行度>
#
# 示例:
#   # 教师模型：SFT 阶段 CoT 构造（completions API, 端口 8000）
#   bash scripts/serve_vllm.sh Qwen/Qwen2.5-72B-Instruct 8000 1
#
#   # Judge 模型：过程验证 + 强化学习连续打分（chat API, 端口 8001）
#   bash scripts/serve_vllm.sh Qwen/Qwen2.5-72B-Instruct 8001 1

MODEL="${1:-Qwen/Qwen2.5-72B-Instruct}"
PORT="${2:-8000}"
TP="${3:-1}"

echo "启动 vLLM: model=${MODEL} port=${PORT} tensor_parallel=${TP}"

vllm serve "${MODEL}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9
