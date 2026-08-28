"""LoRA 配置构建（GRPO / DAPO 共用）。

注意：trl 0.15 在 deepspeed zero3 + peft 下仍会加载独立的 ref model（抵消省显存效果），
因此 LoRA 方案应直接 `python src/train/grpo.py --lora` 单进程运行，不要走 accelerate/deepspeed。
"""
from __future__ import annotations

import argparse


def add_lora_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lora", action="store_true", help="启用 LoRA（单卡低显存）")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="all-linear",
        help="逗号分隔的目标模块名，默认 all-linear",
    )


def build_lora_config(args: argparse.Namespace):
    if not getattr(args, "lora", False):
        return None
    from peft import LoraConfig

    raw = args.lora_target_modules.strip()
    targets = (
        "all-linear"  # peft 魔法字符串，展开为全部线性层；列表形式会被当字面模块名
        if raw.lower() == "all-linear"
        else [m.strip() for m in raw.split(",") if m.strip()]
    )
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=targets,
        task_type="CAUSAL_LM",
    )