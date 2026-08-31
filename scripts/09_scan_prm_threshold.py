"""PRM 阈值扫描：AUC 选出排序最优模型后，为下游（SFT 过滤 / RL 奖励）确定实际分数阈值。

两个层面：
1) step 级：threshold 0.05~0.95 → accuracy / 错误步 precision-recall-F1（医疗核心=错误检出）
2) trajectory 级：min(step score) 作为轨迹分数 → 预测 trajectory_correct
   （这是 RL min-聚合奖励的真实信号，直接决定 reward 门控阈值）

输出 data/prm/threshold_report.json（Step 4 RL 接入直接读取）。
用法:
  python scripts/09_scan_prm_threshold.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.data.step_annotation import build_text
from src.process.prm_scorer import PRMScorer


def load_records(path: str, max_context_len: int = 1500):
    """每条轨迹 -> {trajectory_correct, items:[{context,step,label}]}。"""
    recs = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        context = ""
        items = []
        for s in r["labels"]:
            items.append({"context": context, "step": s["step"],
                          "label": float(s["label"])})
            context = (context + s["step"] + "\n")[-max_context_len:]
        recs.append({"question": r["question"], "options": r["options"],
                     "trajectory_correct": int(r["trajectory_correct"]),
                     "items": items})
    return recs


def f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def step_metrics(scores, labels, th):
    pred_err = [s < th for s in scores]  # 预测为错误步
    tp = sum(p and l == 0 for p, l in zip(pred_err, labels))  # 错误步正确检出
    fp = sum(p and l == 1 for p, l in zip(pred_err, labels))
    fn = sum((not p) and l == 0 for p, l in zip(pred_err, labels))
    acc = sum((not p) == (l == 1) for p, l in zip(pred_err, labels)) / len(labels)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return acc, prec, rec, f1(prec, rec)


def auc_fast(y_true, y_score):
    """ROC-AUC（Mann-Whitney U）。"""
    import numpy as np
    y = np.asarray(y_true)
    s = np.asarray(y_score)
    m = int((y == 1).sum())
    n = int((y == 0).sum())
    if m == 0 or n == 0:
        return 0.5
    neg = np.sort(s[y == 0])
    pos = s[y == 1]
    lt = np.searchsorted(neg, pos, side="left")
    le = np.searchsorted(neg, pos, side="right")
    return float((lt.sum() + 0.5 * (le - lt).sum()) / (m * n))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/prm.yaml")
    parser.add_argument("--base", default="/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", default="checkpoints/prm")
    parser.add_argument("--val", default="data/prm/prm_val.jsonl")
    parser.add_argument("--out", default="data/prm/threshold_report.json")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    scorer = PRMScorer(args.base, args.adapter)
    records = load_records(args.val)
    print(f"val 轨迹: {len(records)}")

    step_scores, step_labels = [], []
    traj_min, traj_all_correct, traj_correct = [], [], []
    for rec in records:
        items = rec["items"]
        q, opt = rec["question"], rec["options"]
        scores = scorer.score_batch(
            [{"question": q, "options": opt, "context": i["context"],
              "step": i["step"]} for i in items])
        step_scores.extend(scores)
        step_labels.extend(i["label"] for i in items)
        traj_min.append(min(scores))
        # 与 min 聚合对齐的 GT：所有步骤都正确（min(label)==1）
        traj_all_correct.append(int(min(i["label"] for i in items) == 1))
        # 次参考：最终答案正确（可能存在「步骤错但答案蒙对」的噪声）
        traj_correct.append(rec["trajectory_correct"])

    n = len(step_scores)
    print(f"step 样本: {n}，错误步占比 {sum(1 for l in step_labels if l == 0) / n:.1%}")
    print(f"PRM 越界占比: {scorer.oob_ratio:.3%}")

    report = {"step_samples": n, "n_trajectories": len(records)}
    print("\n=== step 级阈值扫描 ===")
    best_f1 = (0.0, None)
    for th in [round(t, 2) for t in [x / 20 for x in range(1, 20)]]:
        acc, prec, rec, f = step_metrics(step_scores, step_labels, th)
        if f > best_f1[0]:
            best_f1 = (f, th)
        if abs(th - 0.5) < 1e-9 or th in (0.4, 0.6) or th == best_f1[1]:
            print(f"  th={th:.2f}  acc={acc:.3f}  err_prec={prec:.3f}  "
                  f"err_rec={rec:.3f}  err_f1={f:.3f}")
    print(f"  * 错误步 F1 最优: th={best_f1[1]}  f1={best_f1[0]:.3f}")
    report["step_best_f1_threshold"] = best_f1[1]
    report["step_best_f1"] = best_f1[0]

    print("\n=== trajectory 级（min 聚合）===")
    print("  主标签 = 所有步正确(min(label)==1)，对齐 min 聚合；次参考 = 最终答案正确")
    print(f"  min-score vs 所有步正确 AUC: {auc_fast(traj_all_correct, traj_min):.3f}")
    print(f"  min-score vs 最终答案正确 AUC: {auc_fast(traj_correct, traj_min):.3f}  (含步骤错答案对噪声)")
    bt = None
    for th in [round(t, 2) for t in [x / 20 for x in range(1, 20)]]:
        pred = [s >= th for s in traj_min]
        acc = sum(p == c for p, c in zip(pred, traj_all_correct)) / len(traj_all_correct)
        tp = sum(p and c for p, c in zip(pred, traj_all_correct))
        fp = sum(p and not c for p, c in zip(pred, traj_all_correct))
        fn = sum(not p and c for p, c in zip(pred, traj_all_correct))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f = f1(prec, rec)
        if bt is None or f > bt[2]:
            bt = (th, acc, f, rec)
        if th in (0.3, 0.5, 0.7):
            print(f"  th={th:.2f}  acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  f1={f:.3f}")
    print(f"  * 轨迹 F1 最优: th={bt[0]}  f1={bt[2]:.3f}  acc={bt[1]:.3f}  rec={bt[3]:.3f}")
    report["traj_best_f1_threshold"] = bt[0]
    report["traj_best_f1"] = bt[2]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已存: {args.out}")


if __name__ == "__main__":
    main()