"""链式评测：base + 第一个 adapter(merge) + 第二个 adapter 叠加后评测。

用于评测 GRPO（在 SFT_clean merge 基座上再训的 LoRA）。
用法:
  python scripts/eval_stacked.py --model <base> --merge-adapter checkpoints/sft_clean \
      --adapter checkpoints/grpo --data data/eval/cmb_eval_200.jsonl --batch-size 8
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.answer_extraction import extract_final_answer


def build_prompt(question: str, options: dict) -> str:
    opt_text = "\n".join(f"{k}. {v}" for k, v in options.items())
    return (
        f"问题：{question}\n选项：\n{opt_text}\n"
        "请逐步推理，最后一行以 `最终答案: <选项字母>` 结束。"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--merge-adapter", default=None, help="先 merge 进基座的 adapter")
    parser.add_argument("--adapter", default=None, help="叠加在 merge 结果上的 adapter")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="logs/eval_stacked.json")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    if args.merge_adapter:
        model = PeftModel.from_pretrained(model, args.merge_adapter)
        model = model.merge_and_unload()
        print(f"[merge] {args.merge_adapter}", flush=True)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"[adapter] {args.adapter}", flush=True)
    model.eval()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    per = defaultdict(lambda: {"correct": 0, "total": 0, "miss": 0, "len": 0})
    bs = max(1, args.batch_size)
    for i in range(0, len(rows), bs):
        batch = rows[i:i + bs]
        texts = [build_prompt(r["question"], r["options"]) for r in batch]
        msgs = [[{"role": "user", "content": t}] for t in texts]
        prompts = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                          tokenize=False)
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False)
        decoded = tok.batch_decode(out[:, enc.input_ids.shape[1]:],
                                   skip_special_tokens=True)
        for r, d in zip(batch, decoded):
            key = r.get("dataset", "cmb")
            pred = extract_final_answer(d)
            per[key]["total"] += 1
            per[key]["len"] += len(d)
            if pred is None:
                per[key]["miss"] += 1
            elif pred == r["answer"].upper():
                per[key]["correct"] += 1

    for k, v in per.items():
        acc = v["correct"] / v["total"]
        print(f"{k}: acc={acc:.4f} miss={v['miss'] / v['total']:.4f} "
              f"len={v['len'] / v['total']:.1f}", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({k: dict(v) for k, v in per.items()}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
