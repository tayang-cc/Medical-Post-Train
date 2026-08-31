"""PRM 软过滤 SFT 数据（方向2折中）：只删极端低分样本，不做二分类硬截断。

域偏移使 PRM 对 32B 老师 CoT 整体低分，但极端低分样本大概率真存在逻辑断裂/过程错误。
策略：给所有样本打分并记录 prm_min_score，仅丢弃 min-score < 极端阈值 的样本。
用法:
  python scripts/10_clean_sft_data.py --extreme-threshold 0.02
"""
from __future__ import annotations

import argparse
import json
import statistics

from src.data.cot_construction import save_jsonl
from src.data.step_annotation import split_steps
from src.process.prm_scorer import PRMScorer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/cot/cmb_2000_cot_verified.jsonl")
    parser.add_argument("--output", default="data/cot/cmb_cot_prm_clean.jsonl")
    parser.add_argument("--scored", default="data/cot/cmb_cot_prm_scored.jsonl",
                        help="全量打分结果（含 prm_min_score，供分析）")
    parser.add_argument("--prm-base", default="/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--prm-adapter", default="checkpoints/prm")
    parser.add_argument("--extreme-threshold", type=float, default=0.02,
                        help="min-score 低于该值的样本视为极端低分，丢弃")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    scorer = PRMScorer(args.prm_base, args.prm_adapter)
    records = [json.loads(l) for l in open(args.input, encoding="utf-8")]

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
    for i in range(0, len(items), args.batch_size):
        scores.extend(scorer.score_batch(items[i:i + args.batch_size]))

    kept = []
    mins = []
    for r, (s, e) in zip(records, ranges):
        seg = scores[s:e]
        mn = min(seg) if seg else 0.0
        mins.append(mn)
        r["prm_min_score"] = round(mn, 4)
        if mn >= args.extreme_threshold:
            kept.append(r)

    save_jsonl(records, args.scored)          # 全量打分存档
    save_jsonl(kept, args.output)             # 软过滤结果
    print(f"软过滤: {len(records)} 条 → 丢弃 min<{args.extreme_threshold} 的 "
          f"{len(records) - len(kept)} 条，保留 {len(kept)} 条")
    print(f"min-score: 均值 {statistics.mean(mins):.4f} / 中位 {statistics.median(mins):.4f} "
          f"/ min {min(mins):.4f} / max {max(mins):.4f}")
    print("低分端直方图（0.00-0.20，步长 0.01）:")
    for lo in [round(x * 0.01, 2) for x in range(20)]:
        hi = round(lo + 0.01, 2)
        n = sum(1 for m in mins if lo <= m < hi)
        if n:
            bar = "#" * (n // 5 + 1)
            print(f"  [{lo:.2f},{hi:.2f}) {n:4d} {bar}")


if __name__ == "__main__":
    main()
