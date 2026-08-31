"""格式脱敏验证：去 markdown 后 PRM 分数是否回升（判断域偏移是否源于格式）。

对样本 CoT 做机械格式脱敏（去 **bold**、编号、标题、bullet、行内代码），
分别用 PRM 打分，对比 min-score 分布与逐条变化。
用法:
  python scripts/11_validate_style.py --sample 200
"""
from __future__ import annotations

import argparse
import json
import re
import statistics

from src.data.step_annotation import split_steps
from src.process.prm_scorer import PRMScorer


def strip_markdown(cot: str) -> str:
    lines = []
    for ln in (cot or "").splitlines():
        s = ln.strip()
        s = re.sub(r"^#{1,6}\s*", "", s)                 # 标题
        s = re.sub(r"^(\d+[.、)]|[-*•])\s*", "", s)       # 编号/bullet
        s = s.replace("**", "")                            # 加粗
        s = re.sub(r"`([^`]*)`", r"\1", s)                 # 行内代码
        if s:
            lines.append(s)
    return "\n".join(lines)


def score_cot(scorer: PRMScorer, records, batch_size=64) -> list[float]:
    items: list[dict] = []
    ranges: list[tuple[int, int]] = []
    for r in records:
        steps = split_steps(r["cot"])
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
    parser.add_argument("--input", default="data/cot/cmb_2000_cot_verified.jsonl")
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--prm-base", default="/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--prm-adapter", default="checkpoints/prm")
    args = parser.parse_args()

    scorer = PRMScorer(args.prm_base, args.prm_adapter)
    records = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    records = records[: args.sample]

    orig = score_cot(scorer, records)
    norm = score_cot(scorer, [{**r, "cot": strip_markdown(r["cot"])} for r in records])

    print(f"样本 {len(records)} 条")
    print(f"原始  min-score: 均值 {statistics.mean(orig):.4f} / 中位 "
          f"{statistics.median(orig):.4f}")
    print(f"脱敏  min-score: 均值 {statistics.mean(norm):.4f} / 中位 "
          f"{statistics.median(norm):.4f}")
    up = sum(1 for a, b in zip(orig, norm) if b - a > 0.02)
    down = sum(1 for a, b in zip(orig, norm) if a - b > 0.02)
    flat = len(records) - up - down
    print(f"逐条变化: 升 {up} / 降 {down} / 平 {flat}")
    print(">=0.5 占比: 原始", f"{sum(1 for m in orig if m >= 0.5) / len(orig):.1%}",
          "/ 脱敏", f"{sum(1 for m in norm if m >= 0.5) / len(norm):.1%}")


if __name__ == "__main__":
    main()
