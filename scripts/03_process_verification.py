"""过程验证入口：对 CoT 中间步骤做细粒度打分，标记谬误。"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.data.cot_construction import load_jsonl, save_jsonl
from src.data.process_verification import ProcessVerifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--input", default="data/cot/train_cot.jsonl")
    parser.add_argument("--output", default="data/cot/train_cot_verified.jsonl")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    verifier = ProcessVerifier(
        judge_base_url=cfg["vllm_base_url"],
        judge_model=cfg["judge_model"],
        granularity=cfg["process_verification"]["step_granularity"],
        score_threshold=cfg["process_verification"]["score_threshold"],
    )

    rows = load_jsonl(args.input)
    out = [verifier.verify(r) for r in rows]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(out, args.output)
    print(f"过程验证完成 {len(out)} 条 -> {args.output}")


if __name__ == "__main__":
    main()
