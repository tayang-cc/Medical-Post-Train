"""transformers 直接评测（不依赖 vllm），支持批量生成提速。

加载模型（可选 LoRA adapter）后用 chat template 生成，抽取答案算准确率。
批量生成（batch 内 left-padding），比逐条快数倍。

用法:
  python src/eval/eval_transformers.py --model <基座> [--adapter <LoRA>] \
      --data data/eval/cmb_eval_200.jsonl --batch-size 8
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
    parser.add_argument("--model", required=True, help="基座模型路径")
    parser.add_argument("--adapter", default=None, help="LoRA adapter 路径（可选）")
    parser.add_argument("--data", required=True, help="评测集 jsonl")
    parser.add_argument("--out", default="logs/eval_transformers.json")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"[adapter] 已加载 {args.adapter}", flush=True)
    model.eval()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    per = defaultdict(lambda: {"correct": 0, "total": 0, "miss": 0, "len": 0})
    results = []
    bs = max(1, args.batch_size)

    for i in range(0, len(rows), bs):
        batch = rows[i:i + bs]
        texts = [
            tok.apply_chat_template(
                [{"role": "user", "content": build_prompt(r["question"], r["options"])}],
                tokenize=False, add_generation_prompt=True,
            )
            for r in batch
        ]
        enc = tok(texts, return_tensors="pt", padding=True,
                  truncation=True, max_length=2048)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        plen = enc["input_ids"].shape[1]  # 左填充后各样本 prompt 长度一致
        for j, r in enumerate(batch):
            completion = tok.decode(out[j][plen:], skip_special_tokens=True)
            pred = extract_final_answer(completion)
            correct = pred == r["answer"].upper()
            ds = r.get("dataset", "unknown")
            s = per[ds]
            s["total"] += 1
            s["correct"] += int(correct)
            s["miss"] += int(pred is None)
            s["len"] += len(completion)
            results.append({
                "id": r.get("id"), "dataset": ds, "pred": pred,
                "answer": r["answer"], "correct": correct,
            })
            print(f"[{s['total']}/{len(rows)}] pred={pred} ans={r['answer']} "
                  f"{'OK' if correct else 'x'}", flush=True)

    report = {}
    for ds, s in per.items():
        report[ds] = {
            "accuracy": s["correct"] / max(s["total"], 1),
            "miss_rate": s["miss"] / max(s["total"], 1),
            "avg_length": s["len"] / max(s["total"], 1),
            "count": s["total"],
        }
        print(f"{ds}: acc={report[ds]['accuracy']:.4f} "
              f"miss={report[ds]['miss_rate']:.4f} "
              f"len={report[ds]['avg_length']:.1f}", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"per_dataset": report, "details": results},
                  f, ensure_ascii=False, indent=2)
    print(f"结果已写入 {args.out}", flush=True)


if __name__ == "__main__":
    main()