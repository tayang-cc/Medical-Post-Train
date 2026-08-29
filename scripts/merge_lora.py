"""把 LoRA adapter merge 进基座，产出完整权重（SFT→GRPO 标准链路的前置步骤）。

用法:
  python scripts/merge_lora.py \
    --base /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
    --adapter checkpoints/sft \
    --out checkpoints/sft_merged
"""
from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="基座模型路径")
    parser.add_argument("--adapter", required=True, help="LoRA adapter 路径")
    parser.add_argument("--out", required=True, help="输出完整模型路径")
    args = parser.parse_args()

    print(f"[merge] 加载基座 {args.base} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    print(f"[merge] 加载 adapter {args.adapter} ...", flush=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    print("[merge] merge_and_unload ...", flush=True)
    model = model.merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.out)
    print(f"[merge] 完成 -> {args.out}", flush=True)


if __name__ == "__main__":
    main()