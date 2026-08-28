"""MedQA / MedMCQA 评测：抽取准确率 + LLM-as-Judge 质量分 双轨。

第一轨（规则/抽取）：从模型回答抽取最终选项字母，与标准答案比对得到准确率。
第二轨（LLM-as-Judge）：对回答的正确性与推理质量打 0-1 连续分（--judge-model 开启）。

另输出：
- miss_rate    未命中率（抽不到答案，幻觉代理指标）
- avg_length   平均回答长度
- judge_mean   Judge 均分；judge_pass_rate 达标率（score >= --judge-threshold）
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from src.answer_extraction import extract_final_answer
from src.reward.composite import JUDGE_SYSTEM, _parse_score


def build_prompt(question: str, options: dict) -> str:
    opt_text = "\n".join(f"{k}. {v}" for k, v in options.items())
    return (
        f"问题：{question}\n选项：\n{opt_text}\n"
        "请逐步推理，最后一行以 `最终答案: <选项字母>` 结束。"
    )


def judge_score(client, model, question, options, answer, completion,
                threshold=0.5) -> tuple[float, bool]:
    opt_text = "\n".join(f"{k}. {v}" for k, v in options.items())
    prompt = (
        f"问题：{question}\n选项：\n{opt_text}\n标准答案：{answer}\n模型回答：{completion}\n"
        "请给出正确性与推理质量的综合评分。"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=16,
        )
        score = _parse_score(resp.choices[0].message.content)
    except Exception:
        score = None
    score = score if score is not None else 0.0
    return score, score >= threshold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="测试集 jsonl")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="logs/eval_results.json")
    parser.add_argument("--judge-base-url", default="http://localhost:8001/v1")
    parser.add_argument("--judge-model", default=None,
                        help="Judge 模型名，不填则只跑抽取准确率单轨")
    parser.add_argument("--judge-threshold", type=float, default=0.5)
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    judge_client = (
        OpenAI(base_url=args.judge_base_url, api_key="EMPTY")
        if args.judge_model else None
    )

    rows = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    per_dataset = defaultdict(lambda: {
        "correct": 0, "total": 0, "miss": 0,
        "jsum": 0.0, "jpass": 0, "jfail": 0, "len_sum": 0,
    })
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

        stat = per_dataset[ds]
        stat["total"] += 1
        stat["correct"] += int(correct)
        stat["miss"] += int(pred is None)
        stat["len_sum"] += len(completion)

        results.append({
            "id": r.get("id"), "dataset": ds, "pred": pred,
            "answer": r["answer"], "correct": correct,
            "question": r["question"], "options": r["options"],
            "completion": completion,
        })

    if judge_client is not None:
        def _judge(item):
            return item, judge_score(
                judge_client, args.judge_model,
                item["question"], item["options"], item["answer"],
                item["completion"], args.judge_threshold,
            )

        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(_judge, r) for r in results]
            for fut in futures:
                item, (score, passed) = fut.result()
                stat = per_dataset[item["dataset"]]
                item["judge_score"] = score
                stat["jsum"] += score
                stat["jpass"] += int(passed)
                stat["jfail"] += int(score == 0.0 and item["completion"])

    report = {}
    for ds, stat in per_dataset.items():
        total = max(stat["total"], 1)
        entry = {
            "accuracy": stat["correct"] / total,
            "miss_rate": stat["miss"] / total,
            "avg_length": stat["len_sum"] / total,
        }
        if judge_client is not None:
            entry["judge_mean"] = stat["jsum"] / total
            entry["judge_pass_rate"] = stat["jpass"] / total
            if stat["jfail"]:
                print(f"[warn] {ds}: {stat['jfail']} 条 Judge 打分失败(记 0 分)")
        report[ds] = {"count": stat["total"], **entry}

        line = (f"{ds}: acc={entry['accuracy']:.4f} miss={entry['miss_rate']:.4f} "
                f"len={entry['avg_length']:.1f}")
        if judge_client is not None:
            line += (f" judge={entry['judge_mean']:.4f} "
                     f"pass={entry['judge_pass_rate']:.4f}")
        print(line)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "judge_enabled": judge_client is not None,
            "per_dataset": report,
            "details": results,
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()