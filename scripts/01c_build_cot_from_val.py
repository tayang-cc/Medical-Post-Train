"""从 CMB-val 的官方解析(explanation)构造 SFT CoT 数据（无需 teacher 模型）。

在 vllm/flashinfer 环境问题未解决时，用官方解析作为现成推理过程，
先把 SFT 阶段跑通。注意：val 同时用于评测会产生数据泄漏，
正式实验请改用 teacher 生成的 CoT。
"""
from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val", required=True, help="CMB-val-merge.json 路径")
    parser.add_argument("--out", default="data/cot/smoke_cot.jsonl")
    args = parser.parse_args()

    with open(args.val, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for it in data:
        q = it.get("question")
        opt = it.get("option")
        ans = it.get("answer")
        exp = it.get("explanation")
        if not q or not isinstance(opt, dict) or not opt or not ans or not exp:
            continue
        if it.get("question_type", "单项选择题") != "单项选择题":
            continue
        opts = {k.upper(): v for k, v in opt.items() if k.upper() in "ABCDE"}
        if ans.upper() not in opts:
            continue
        rows.append({
            "id": it.get("id", len(rows)),
            "dataset": "cmb",
            "question": q,
            "options": opts,
            "answer": ans.upper(),
            "cot": exp,
        })

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"SFT CoT {len(rows)} 条 -> {args.out}")


if __name__ == "__main__":
    main()