"""用 32B 把 SFT CoT 语义改写为简洁逐步风格（对齐 PRM 训练分布，缓解域偏移）。

32B 只改表达、不改推理内容与结论（不引入 7B 自蒸馏）。
用法:
  python scripts/12_rewrite_cot.py --limit 100
"""
from __future__ import annotations

import argparse
import json

from openai import OpenAI

from src.data.cot_construction import save_jsonl

REWRITE_SYSTEM = (
    "你是医学推理改写助手。把给出的医学推理过程改写成简洁的逐步推理。"
    "要求：1) 保留所有推理要点、医学事实和最终结论；2) 去掉加粗、编号、标题、"
    "列表符号等格式标记；3) 用简短直白的句子，每步单独一行；4) 不补充新信息、不改变结论。"
    "只输出改写后的推理正文，不要任何解释或前缀。"
)


def build_user(question: str, options: dict, cot: str) -> str:
    opt = "\n".join(f"{k}. {v}" for k, v in options.items())
    return f"问题：{question}\n选项：\n{opt}\n\n原始推理：\n{cot}\n\n请改写。"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/cot/cmb_2000_cot_verified.jsonl")
    parser.add_argument("--output", default="data/cot/cmb_cot_rewritten.jsonl")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen2.5-32B-Instruct-AWQ")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    records = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    if args.limit:
        records = records[: args.limit]

    out = []
    for i, r in enumerate(records):
        resp = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": build_user(r["question"], r["options"], r["cot"])},
            ],
            temperature=0.0,
            max_tokens=args.max_new_tokens,
        )
        rewritten = resp.choices[0].message.content or ""
        r["cot_original"] = r["cot"]
        r["cot"] = rewritten.strip()
        out.append(r)
        if (i + 1) % 50 == 0:
            print(f"[{i + 1}/{len(records)}] 完成", flush=True)

    save_jsonl(out, args.output)
    print(f"改写完成 {len(out)} 条 -> {args.output}")


if __name__ == "__main__":
    main()
