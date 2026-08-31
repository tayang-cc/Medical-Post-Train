"""步级标注入口 v2：PRM 源题目（仅取自 RL 集合）内部切分 PRM-train/PRM-val。

- 源题目取 pool_rl 前 `--limit` 题，与 SFT/TEST 集合严格隔离；
- 内部切分 PRM-train / PRM-val（默认 1600/400），PRM-val 用于训练早停；
- 每题 7B 策略模型生成 4 条轨迹（含对/错），32B Judge 每步投票 K=3 标注；
- 增量保存，中断用 --start 续跑。

用法:
  python scripts/07_step_annotation.py \
    --limit 2000 --train-ratio 0.8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.data.cot_construction import load_jsonl
from src.data.step_annotation import StepAnnotator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--input", default="data/splits/pool_rl.jsonl")
    parser.add_argument("--out", default="data/prm")
    parser.add_argument("--limit", type=int, default=2000, help="PRM 源题目数（取自 RL 集合）")
    parser.add_argument("--num-trajectories", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.8, help="源题目内 train/val 切分比")
    parser.add_argument("--start", type=int, default=0, help="续跑起点（源题目序号）")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ann_cfg = cfg.get("step_annotation", {})

    annotator = StepAnnotator(
        generator_base_url=ann_cfg.get("generator_base_url", cfg["vllm_base_url"]),
        generator_model=ann_cfg.get("generator_model", cfg["teacher_model"]),
        judge_base_url=ann_cfg.get("judge_base_url", cfg["vllm_base_url"]),
        judge_model=ann_cfg.get("judge_model", cfg["judge_model"]),
        k=ann_cfg.get("vote_k", 3),
        min_steps=ann_cfg.get("min_steps", 3),
    )

    questions = load_jsonl(args.input)
    # 按题干去重（CMB 存在勘误重复题），再切分，保证题目级隔离
    uniq: dict = {}
    for q in questions:
        uniq.setdefault(q["question"], q)
    questions = list(uniq.values())
    total = min(args.limit, len(questions))
    n_train = int(total * args.train_ratio)
    print(f"去重后源题目 {len(questions)} 题，取前 {total} 题（train {n_train} / val {total - n_train}）")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_f = open(out_dir / "prm_train.jsonl", "a", encoding="utf-8")
    val_f = open(out_dir / "prm_val.jsonl", "a", encoding="utf-8")

    try:
        kept = train_kept = val_kept = 0
        for i in range(args.start, total):
            q = questions[i]
            cots = annotator.generate_trajectories(q["question"], q["options"],
                                                   n=args.num_trajectories)
            f_out = train_f if i < n_train else val_f
            for cot in cots:
                rec = annotator.annotate(q["question"], q["options"], cot,
                                         q["answer"].upper())
                if rec is None:
                    continue  # 无效轨迹（步数过少/全步丢弃）
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
                if i < n_train:
                    train_kept += 1
                else:
                    val_kept += 1
            if (i - args.start + 1) % 10 == 0 or i == total - 1:
                train_f.flush()
                val_f.flush()
                print(f"[{i + 1}/{total}] 题，累计有效轨迹 {kept}"
                      f"（train {train_kept} / val {val_kept}）", flush=True)
    finally:
        train_f.close()
        val_f.close()

    print(f"完成：源题目 {total} 题 → PRM-train {n_train} 题 / PRM-val {total - n_train} 题")
    print(f"有效轨迹: {kept} 条（train {train_kept} / val {val_kept}）-> {out_dir}/prm_{{train,val}}.jsonl")


if __name__ == "__main__":
    main()