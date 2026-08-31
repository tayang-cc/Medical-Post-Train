"""构建可验证问题库：确定性过滤 + SFT/RL/VAL/TEST 严格互斥切分。

用法:
  python scripts/00_build_problem_library.py --pool data/raw/cmb_train.jsonl --out data/splits

产出:
  data/splits/pool_sft.jsonl / pool_rl.jsonl / pool_val.jsonl / pool_test.jsonl
  各集合互斥（脚本末尾校验），后续 SFT/RL/评测都从对应文件取数。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.problem_library import assign_splits, check_disjoint, load_pool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="data/raw/cmb_train.jsonl")
    parser.add_argument("--out", default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sft", type=float, default=0.70)
    parser.add_argument("--rl", type=float, default=0.15)
    parser.add_argument("--val", type=float, default=0.05)
    args = parser.parse_args()

    rows = load_pool(args.pool)
    print(f"确定性问题池: {len(rows)} 条")

    ratios = {"sft": args.sft, "rl": args.rl, "val": args.val}
    splits = assign_splits(rows, ratios, seed=args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        path = out / f"pool_{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(items)} 条 -> {path}")

    if check_disjoint(splits):
        print("[OK] 集合互斥校验通过")
    else:
        print("[FAIL] 存在集合重叠，请检查")


if __name__ == "__main__":
    main()