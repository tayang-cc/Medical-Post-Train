"""CoT 构造入口：调用教师模型对原始多选题生成推理链。"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.data.cot_construction import (
    CoTConstructor,
    load_jsonl,
    save_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--input", default="data/raw/medqa_train.jsonl")
    parser.add_argument("--output", default="data/cot/train_cot.jsonl")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ctor = CoTConstructor(
        teacher_base_url=cfg["vllm_base_url"],
        teacher_model=cfg["teacher_model"],
        num_samples=cfg["cot"]["num_samples"],
        temperature=cfg["cot"]["temperature"],
        top_p=cfg["cot"]["top_p"],
        max_new_tokens=cfg["cot"]["max_new_tokens"],
        keep_only_correct=cfg["cot"]["keep_only_correct"],
    )

    rows = load_jsonl(args.input)
    out = ctor.construct(rows)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(out, args.output)
    print(f"生成 {len(out)} 条 CoT -> {args.output}")


if __name__ == "__main__":
    main()
