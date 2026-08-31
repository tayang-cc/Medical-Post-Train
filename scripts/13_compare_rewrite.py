"""对比改写前后：PRM min-score + 答案一致性。

输入含 cot_original(原始) 与 cot(改写) 的 jsonl，分别打分对比。
用法:
  python scripts/13_compare_rewrite.py --input data/cot/cmb_cot_rewritten_100.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics

from src.answer_extraction import extract_final_answer
from src.data.step_annotation import split_steps
from src.process.prm_scorer import PRMScorer


def score_cot(scorer: PRMScorer, records, field="cot", batch_size=64) -> list[float]:
    items: list[dict] = []
    ranges: list[tuple[int, int]] = []
    for r in records:
        steps = split_steps(r[field])
        start = len(items)
        context = ""
        for s in steps:
            items.append({"question": r["question"], "options": r["options"],
                          "context": context, "step": s})
            context = context + s + "\n"
        ranges.append((start, len(items)))
    scores: list[float] = []
    for i in range(0, len(items), batch_size):
        scores.extend(scorer.score_batch(items[i:i + batch_size]))
    mins = []
    for s, e in ranges:
        seg = scores[s:e]
        mins.append(min(seg) if seg else 0.0)
    return mins


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/cot/cmb_cot_rewritten_100.jsonl")
    parser.add_argument("--prm-base", default="/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--prm-adapter", default="checkpoints/prm")
    args = parser.parse_args()

    scorer = PRMScorer(args.prm_base, args.prm_adapter)
    records = [json.loads(l) for l in open(args.input, encoding="utf-8")]

    orig = score_cot(scorer, records, "cot_original")
    rew = score_cot(scorer, records, "cot")

    print(f"样本 {len(records)} 条")
    print(f"原始 min-score: 均值 {statistics.mean(orig):.4f} / 中位 {statistics.median(orig):.4f}")
    print(f"改写 min-score: 均值 {statistics.mean(rew):.4f} / 中位 {statistics.median(rew):.4f}")
    up = sum(1 for a, b in zip(orig, rew) if b - a > 0.02)
    down = sum(1 for a, b in zip(orig, rew) if a - b > 0.02)
    print(f"逐条: 升 {up} / 降 {down} / 平 {len(records) - up - down}")
    print(">=0.5 占比: 原始", f"{sum(1 for m in orig if m >= 0.5) / len(orig):.1%}",
          "/ 改写", f"{sum(1 for m in rew if m >= 0.5) / len(rew):.1%}")

    # 答案一致性：改写后抽取答案是否仍等于标准答案
    changed = 0
    for r in records:
        pred = extract_final_answer(r["cot"])
        if pred is not None and pred != r["answer"].upper():
            changed += 1
    print(f"改写后结论与标准答案不一致: {changed}/{len(records)}")


if __name__ == "__main__":
    main()
