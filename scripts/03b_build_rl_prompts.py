"""构造 GRPO 训练数据：把可验证的多选题打包为 messages 形式的 prompt。

输出 data/processed/rl_prompts.jsonl，每行:
  {"prompt": [{"role": "user", "content": "..."}], "question": "...", "answer": "A"}

prompt 用 messages 列表，trl 会据此自动套 Qwen 的 chat template（否则少了 instruct 格式）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.cot_construction import load_jsonl, save_jsonl


def build_prompt_messages(question: str, options: dict) -> list[dict]:
    opt_text = "\n".join(f"{k}. {v}" for k, v in options.items())
    content = (
        f"问题：{question}\n选项：\n{opt_text}\n"
        "请逐步推理，最后一行以 `最终答案: <选项字母>` 结束。"
    )
    return [{"role": "user", "content": content}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/raw/medqa_train.jsonl",
            "data/raw/medmcqa_train.jsonl",
        ],
        help="可验证多选题文件（01_prepare_data.py 的输出）",
    )
    parser.add_argument("--output", default="data/processed/rl_prompts.jsonl")
    args = parser.parse_args()

    out = []
    for path in args.inputs:
        if not Path(path).exists():
            print(f"跳过不存在的文件: {path}")
            continue
        for r in load_jsonl(path):
            out.append(
                {
                    "prompt": build_prompt_messages(r["question"], r["options"]),
                    "question": r["question"],
                    "answer": r["answer"],
                }
            )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(out, args.output)
    print(f"GRPO 提示词 {len(out)} 条 -> {args.output}")


if __name__ == "__main__":
    main()