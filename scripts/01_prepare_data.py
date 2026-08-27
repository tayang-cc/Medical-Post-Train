"""数据准备：下载 MedQA / MedMCQA 并统一为标准多选题 JSONL 格式。

标准格式:
  {"id", "dataset", "question", "options": {"A":..,"B":..,"C":..,"D":..}, "answer": "A"}

也可选用 HuatuoGPT 现成 SFT 数据 (FreedomIntelligence/medical-o1-reasoning-SFT) 加速复现。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

LETTERS = list("ABCDE")


def normalize_medqa(row: dict, split: str) -> dict | None:
    q = row.get("question")
    options = row.get("options")
    if not q or not options:
        return None
    # options 可能是 {'A':...,'B':...} 或 {'opa':...}
    keys = list(options.keys())
    if "A" in keys or "a" in keys:
        opts = {k.upper(): v for k, v in options.items() if k.upper() in LETTERS}
        answer = str(row.get("answer_idx") or row.get("answer") or "").upper()
    else:
        opts = {}
        for i, k in enumerate(sorted(keys)):
            if i < 5:
                opts[LETTERS[i]] = options[k]
        idx = row.get("answer_idx")
        if isinstance(idx, (int, float)):
            answer = LETTERS[int(idx)]
        else:
            answer = str(row.get("answer") or "").upper()
    if answer not in opts:
        return None
    return {
        "id": row.get("id", ""),
        "dataset": "medqa",
        "question": q,
        "options": opts,
        "answer": answer,
    }


def normalize_medmcqa(row: dict, split: str) -> dict | None:
    q = row.get("question")
    if not q:
        return None
    opts = {
        "A": row.get("opa"),
        "B": row.get("opb"),
        "C": row.get("opc"),
        "D": row.get("opd"),
    }
    if any(v is None for v in opts.values()):
        return None
    cop = row.get("cop")  # 1-4
    answer = LETTERS[int(cop) - 1]
    return {
        "id": row.get("id", ""),
        "dataset": "medmcqa",
        "question": q,
        "options": opts,
        "answer": answer,
    }


def load_and_write(dataset_name: str, split: str, out_path: Path,
                   normalizer) -> None:
    ds = load_dataset(dataset_name, split=split)
    rows = []
    for r in ds:
        norm = normalizer(r, split)
        if norm:
            rows.append(norm)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[{dataset_name} / {split}] {len(rows)} 条 -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # MedQA (USMLE, 4 options)
    try:
        load_and_write("openlifescienceai/medqa", "train",
                       out / "medqa_train.jsonl", normalize_medqa)
        load_and_write("openlifescienceai/medqa", "test",
                       out / "medqa_test.jsonl", normalize_medqa)
    except Exception as e:  # noqa: BLE001
        print(f"MedQA 下载失败（可能需授权）：{e}")

    # MedMCQA
    try:
        load_and_write("openlifescienceai/medmcqa", "train",
                       out / "medmcqa_train.jsonl", normalize_medmcqa)
        load_and_write("openlifescienceai/medmcqa", "validation",
                       out / "medmcqa_val.jsonl", normalize_medmcqa)
    except Exception as e:  # noqa: BLE001
        print(f"MedMCQA 下载失败：{e}")


if __name__ == "__main__":
    main()
