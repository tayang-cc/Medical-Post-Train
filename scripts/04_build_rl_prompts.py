"""构建 Step4 RL 训练数据：从 pool_rl 取题，排除 PRM 源题（严格隔离），含 options。

输出 data/processed/rl_prompts.jsonl，每行:
  {"prompt": [{"role":"user","content":...}], "question":..., "answer":..., "options":...}

用法:
  python scripts/04_build_rl_prompts.py --limit 500
"""
from __future__ import annotations

import argparse
import json

from src.data.cot_construction import save_jsonl


def build_prompt_messages(question: str, options: dict) -> list[dict]:
    opt_text = "\n".join(f"{k}. {v}" for k, v in options.items())
    content = (
        f"问题：{question}\n选项：\n{opt_text}\n"
        "请逐步推理，最后一行以 `最终答案: <选项字母>` 结束。"
    )
    return [{"role": "user", "content": content}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/splits/pool_rl.jsonl")
    parser.add_argument("--prm-src", nargs="+",
                        default=["data/prm/prm_train.jsonl",
                                 "data/prm/prm_val.jsonl"],
                        help="PRM 源标注文件（用于排除，保证 RL 与 PRM 源隔离）")
    parser.add_argument("--output", default="data/processed/rl_prompts.jsonl")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    prm_src: set[str] = set()
    for p in args.prm_src:
        for line in open(p, encoding="utf-8"):
            prm_src.add(json.loads(line)["question"])

    seen: set[str] = set()
    out = []
    for line in open(args.input, encoding="utf-8"):
        r = json.loads(line)
        q = r["question"]
        if q in prm_src or q in seen:   # 排除 PRM 源 + 去重（pool_rl 有重复题）
            continue
        seen.add(q)
        out.append({
            "prompt": build_prompt_messages(q, r["options"]),
            "question": q,
            "answer": r["answer"].upper(),
            "options": r["options"],
        })
        if len(out) >= args.limit:
            break

    save_jsonl(out, args.output)
    print(f"RL 训练数据 {len(out)} 题 -> {args.output}"
          f"（排除 PRM 源 {len(prm_src)} 题，已去重）")


if __name__ == "__main__":
    main()
