"""SFT 训练：Qwen2.5-7B-Instruct 在高质量（含过程验证过滤）CoT 数据上微调。

输入来自 scripts/03_process_verification.py 输出的 train_cot_verified.jsonl，
按 process_score / num_fallacies 过滤掉过程不严谨的样本。
"""
from __future__ import annotations

import argparse
import json
import re

import yaml
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from src.train.lora import add_lora_args, build_lora_config


def build_prompt(question: str, options: dict) -> str:
    opt_text = "\n".join(f"{k}. {v}" for k, v in options.items())
    return (
        f"问题：{question}\n选项：\n{opt_text}\n"
        "请逐步推理，最后一行以 `最终答案: <选项字母>` 结束。"
    )


def normalize_completion(cot: str, answer: str) -> str:
    """去掉 CoT 末尾已有的答案行，再补统一格式，避免答案重复出现。"""
    lines = [ln.strip() for ln in (cot or "").splitlines() if ln.strip()]
    if lines and re.search(r"(?:最终答案|答案)\s*[:：]?\s*[A-Ea-e]", lines[-1]):
        lines = lines[:-1]
    body = "\n".join(lines)
    return f"{body}\n最终答案: {answer}" if body else f"最终答案: {answer}"


def load_cot_data(
    path: str,
    min_process_score: float | None = None,
    max_fallacies: int | None = None,
) -> Dataset:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if min_process_score is not None and r.get("process_score") is not None:
                if r["process_score"] < min_process_score:
                    continue
            if max_fallacies is not None and r.get("num_fallacies") is not None:
                if r["num_fallacies"] > max_fallacies:
                    continue
            # 用消息列表：trl 0.15 会把 prompt/completion 拼成 messages 并套 chat template
            rows.append(
                {
                    "prompt": [
                        {"role": "user", "content": build_prompt(r["question"], r["options"])}
                    ],
                    "completion": [
                        {"role": "assistant",
                         "content": normalize_completion(r["cot"], r["answer"])}
                    ],
                }
            )
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft.yaml")
    add_lora_args(parser)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    lora_config = build_lora_config(args)
    if lora_config is not None:
        print(
            "[LoRA] 已启用 LoRA；单卡请直接 python 运行本脚本"
            "（全参 7B + fp32 AdamW 优化器在 48G 单卡会 OOM）"
        )

    dataset = load_cot_data(
        cfg["dataset_path"],
        min_process_score=cfg.get("min_process_score"),
        max_fallacies=cfg.get("max_fallacies"),
    )

    sft_config = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        max_seq_length=cfg["max_seq_length"],
        bf16=cfg["bf16"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        logging_steps=cfg["logging_steps"],
        save_strategy=cfg["save_strategy"],
        save_steps=cfg["save_steps"],
    )

    trainer = SFTTrainer(
        model=cfg["model_name_or_path"],
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])


if __name__ == "__main__":
    main()