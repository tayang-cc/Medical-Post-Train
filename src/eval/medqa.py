"""MedQA / MedMCQA 评测：多选准确率。"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from openai import OpenAI

from src.answer_extraction import extract_final_answer


def build_prompt(question: str, options: dict) -> str:
    opt_text = "\n".join(f"{k}. {v}" for k, v in options.items())
    return (
        f"问题：{question}\n选项：\n{opt_text}\n"
        "请逐步推理，最后一行以 `最终答案: <选项字母>` 结束。"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="测试集 jsonl")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="logs/eval_results.json")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")

    rows = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    per_dataset = defaultdict(lambda: {"correct": 0, "total": 0})
    results = []

    for r in rows:
        ds = r.get("dataset", "unknown")
        prompt = build_prompt(r["question"], r["options"])
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.0,
        )
        completion = resp.choices[0].message.content or ""
        pred = extract_final_answer(completion)
        correct = pred == r["answer"].upper()

        per_dataset[ds]["total"] += 1
        per_dataset[ds]["correct"] += int(correct)
        results.append({"id": r.get("id"), "dataset": ds, "pred": pred,
                        "answer": r["answer"], "correct": correct})

    report = {}
    for ds, stat in per_dataset.items():
        report[ds] = {
            "accuracy": stat["correct"] / stat["total"],
            "correct": stat["correct"],
            "total": stat["total"],
        }
        print(f"{ds}: {stat['correct']}/{stat['total']} "
              f"= {stat['correct'] / stat['total']:.4f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"per_dataset": report, "details": results}, f,
                  ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
