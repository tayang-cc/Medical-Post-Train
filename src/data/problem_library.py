"""可验证问题库：确定性过滤 + 严格集合隔离（SFT/RL/VAL/TEST 互斥）+ 难度分层。

所有下游（CoT 构造、SFT、RL、评测）都从本库的互斥切分取数，防止集合泄露。
难度分层默认均匀；若提供 difficulty_fn（如 teacher 是否正确、题目难度代理），
会在每层内按比例切分，保证各集合难度分布一致。
"""
from __future__ import annotations

import json
import random


def load_pool(path: str) -> list[dict]:
    """加载原始多选题池，只保留确定性答案（单选、答案在选项内、无歧义）。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            q = r.get("question")
            opts = r.get("options")
            ans = (r.get("answer") or "").upper()
            if not q or not isinstance(opts, dict) or not opts:
                continue
            if ans not in opts:  # 答案必须确定落在选项内
                continue
            rows.append(r)
    return rows


def _bucket_by_difficulty(rows: list[dict], difficulty_fn) -> dict:
    buckets = {}
    for r in rows:
        buckets.setdefault(difficulty_fn(r), []).append(r)
    return buckets


def assign_splits(
    rows: list[dict],
    ratios: dict,
    seed: int = 42,
    difficulty_fn=None,
) -> dict:
    """先按难度分桶，再在每个桶内按比例切分，保证集合互斥且难度分布一致。

    ratios: {"sft": 0.70, "rl": 0.15, "val": 0.05}，剩余归 test。
    """
    assert sum(ratios.values()) < 1.0, "ratios 之和需 < 1（剩余归 test）"
    buckets = _bucket_by_difficulty(rows, difficulty_fn or (lambda r: 0))
    rng = random.Random(seed)
    result = {"sft": [], "rl": [], "val": [], "test": []}
    for _level, bucket in buckets.items():
        rng.shuffle(bucket)
        n = len(bucket)
        n_sft = int(n * ratios["sft"])
        n_rl = int(n * ratios["rl"])
        n_val = int(n * ratios["val"])
        result["sft"] += bucket[:n_sft]
        result["rl"] += bucket[n_sft:n_sft + n_rl]
        result["val"] += bucket[n_sft + n_rl:n_sft + n_rl + n_val]
        result["test"] += bucket[n_sft + n_rl + n_val:]
    return result


def check_disjoint(splits: dict) -> bool:
    """校验各集合严格互斥（按 id 判重）。"""
    ids = {name: {r.get("id", r.get("question")) for r in items}
           for name, items in splits.items()}
    names = list(ids)
    ok = True
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = ids[names[i]] & ids[names[j]]
            if overlap:
                ok = False
                print(f"[warn] {names[i]} ∩ {names[j]} 重叠 {len(overlap)} 条")
    return ok