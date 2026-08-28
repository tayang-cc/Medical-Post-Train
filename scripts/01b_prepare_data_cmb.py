"""从 CMB（Chinese Medical Benchmark，中文医学基准）构造标准多选题 JSONL。

不依赖 HF gated 数据集（无需 token）。CMB 是 HuatuoGPT 同实验室（FreedomIntelligence）
开源的中文医学基准（Apache-2.0），CMB-Exam 为单选/多选客观题，本文只保留「单项选择题」。

用 ijson 流式解析大 JSON（如 CMB-train-merge.json 148MB），避免低内存容器 OOM。

数据获取：
  modelscope download --dataset FreedomIntelligence/CMB --local_dir <dir>
  或 git clone --depth 1 https://github.com/FreedomIntelligence/CMB.git

用法：
  python scripts/01b_prepare_data_cmb.py --source /path/to/CMB/CMB-Exam --out data/raw
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import ijson

SPLIT_DIRS = {"train": "CMB-train", "val": "CMB-val", "test": "CMB-test"}
LETTERS = "ABCDE"


def find_json_files(cmb_dir: Path, split: str) -> list[Path]:
    """优先按官方 split 目录，否则全盘扫描并按文件名推断。"""
    subdir = cmb_dir / SPLIT_DIRS[split]
    if subdir.exists():
        files = sorted(subdir.glob("*.json"))
        if files:
            return files
    files = []
    for fp in sorted(cmb_dir.rglob("*.json")):
        name = fp.name.lower()
        if split == "train" and ("train" in name or ("test" not in name and "val" not in name)):
            files.append(fp)
        elif split == "val" and "val" in name:
            files.append(fp)
        elif split == "test" and "test" in name:
            files.append(fp)
    return files


def to_record(it, split: str, idx: int):
    """单条 CMB 记录 -> 标准格式 dict；无效返回 None。"""
    if not isinstance(it, dict):
        return None
    q = it.get("question")
    opt = it.get("option") or it.get("options")
    ans = it.get("answer")
    if not q or not isinstance(opt, dict) or not opt or not ans:
        return None
    if it.get("question_type", "单项选择题") != "单项选择题":
        return None
    opts = {k.upper(): v for k, v in opt.items() if k.upper() in LETTERS}
    if ans.upper() not in opts:
        return None
    return {
        "id": it.get("id", f"{split}-{idx}"),
        "dataset": "cmb",
        "question": q,
        "options": opts,
        "answer": ans.upper(),
    }


def write_split(cmb_dir: Path, split: str, out: Path) -> int:
    count = 0
    with open(out, "w", encoding="utf-8") as f:
        for fp in find_json_files(cmb_dir, split):
            with open(fp, "rb") as raw:
                first = raw.read(1)
                raw.seek(0)
                if first == b"[":  # 顶层数组 -> ijson 流式
                    for it in ijson.items(raw, "item"):
                        row = to_record(it, split, count)
                        if row is None:
                            continue
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        count += 1
                else:  # 顶层 dict（小文件，如 hierarchy 元数据）
                    data = json.load(raw)
                    items = data if isinstance(data, list) else (
                        data.get("data", [data]) if isinstance(data, dict) else []
                    )
                    for it in items:
                        row = to_record(it, split, count)
                        if row is None:
                            continue
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        count += 1
    print(f"[cmb/{split}] {count} 条 -> {out}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="CMB-Exam 所在目录")
    parser.add_argument("--out", default="data/raw", help="输出目录")
    args = parser.parse_args()

    cmb_dir = Path(args.source)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLIT_DIRS:
        write_split(cmb_dir, split, out_dir / f"cmb_{split}.jsonl")


if __name__ == "__main__":
    main()