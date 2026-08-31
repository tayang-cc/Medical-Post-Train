"""训练 7B-PRM：回归头（num_labels=1，MSE）在步级标注数据上，输出 step 正确性分数。

数据: data/prm/prm_{train,val}.jsonl（question + 前置上下文 + 当前步骤 -> label 1.0/0.0）
输入不含标准答案（避免答案泄露进 step 打分）。

规范要点：
- 划分必须按题目（07 已按源题目序号切分），本脚本断言 train/val 题目零重叠，禁止按 step 随机切；
- 评估加二分类判别指标（accuracy + 错误步骤检出 precision/recall/F1），不只看 eval_loss；
- 回归头输出无值域约束，推理侧由 src/process/prm_scorer.py 强制 clip 到 [0,1]；
- LoRA 收敛超参：attention-only 目标、低 r、高 dropout + weight_decay + 早停，防过拟合标注噪声。

用法:
  python scripts/08_train_prm.py --config configs/prm.yaml
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.data.step_annotation import build_text


def flatten(path: str, max_context_len: int = 1500):
    """把轨迹标注展开成 (question, options, context, step, label) 样本。"""
    samples = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        context = ""
        for s in r["labels"]:  # 有序步骤，重建前置上下文
            samples.append({
                "question": r["question"],
                "options": r["options"],
                "context": context,
                "step": s["step"],
                "label": float(s["label"]),
            })
            context = (context + s["step"] + "\n")[-max_context_len:]
    return samples


def question_set(path: str) -> set:
    return {json.loads(l)["question"] for l in open(path, encoding="utf-8")}


def roc_auc(y_true, y_score):
    """ROC-AUC（Mann-Whitney U，免阈值判别指标）。"""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    m = int((y_true == 1).sum())
    n = int((y_true == 0).sum())
    if m == 0 or n == 0:
        return 0.5
    neg = np.sort(y_score[y_true == 0])
    pos = y_score[y_true == 1]
    lt = np.searchsorted(neg, pos, side="left")     # neg < pos
    le = np.searchsorted(neg, pos, side="right")    # neg <= pos
    u = lt.sum() + 0.5 * (le - lt).sum()
    return float(u / (m * n))


def compute_metrics(eval_pred):
    """二分类判别指标：accuracy + AUC + 错误步(0 类)检出 + 正确步(1 类)召回（监控过纠偏）。"""
    logits = np.asarray(eval_pred.predictions).reshape(-1)
    labels = np.asarray(eval_pred.label_ids).reshape(-1)
    probs = 1.0 / (1.0 + np.exp(-logits))            # sigmoid，天然落 [0,1]
    probs = np.clip(probs, 0.0, 1.0)                 # 双保险
    preds = (probs >= 0.5).astype(int)
    tp = int(np.sum((preds == 0) & (labels == 0)))  # 错误步正确检出
    fp = int(np.sum((preds == 0) & (labels == 1)))
    fn = int(np.sum((preds == 1) & (labels == 0)))
    tn = int(np.sum((preds == 1) & (labels == 1)))  # 正确步正确识别
    acc = float(np.mean(preds == labels))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    pos_total = int((labels == 1).sum())
    pos_rec = tn / pos_total if pos_total else 0.0
    return {
        "eval_accuracy": acc,
        "eval_auc": roc_auc(labels, probs),
        "eval_err_precision": prec,
        "eval_err_recall": rec,
        "eval_err_f1": f1,
        "eval_pos_recall": pos_rec,   # 监控 pos_weight+label_smoothing 是否过度纠偏
        "eval_mean_score": float(probs.mean()),
    }


class PRMTrainer(Trainer):
    """BCE 损失定制：label smoothing（抗硬标签饱和）+ pos_weight（加权错误步）。

    pos_weight < 1 时负类（错误步）相对权重更高，契合医疗「漏检代价 > 误判」。
    """

    def __init__(self, pos_weight: float = 0.55, label_smoothing: float = 0.05,
                 **kwargs):
        super().__init__(**kwargs)
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1).float()
        eps = self.label_smoothing
        smooth_labels = labels * (1.0 - 2.0 * eps) + eps  # 1->1-eps, 0->eps
        crit = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.pos_weight], device=logits.device))
        loss = crit(logits, smooth_labels)
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/prm.yaml")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_q = question_set(cfg["train_data"])
    val_q = question_set(cfg["val_data"])
    overlap = train_q & val_q
    assert not overlap, f"题目级隔离失败：train/val 重叠 {len(overlap)} 题！"
    print(f"题目级隔离校验通过：train {len(train_q)} 题 / val {len(val_q)} 题，零重叠")

    tok = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    def encode(samples):
        n = len(samples["question"])
        texts = [build_text(samples["question"][j], samples["options"][j],
                            samples["context"][j], samples["step"][j])
                 for j in range(n)]
        enc = tok(texts, truncation=True, max_length=cfg["max_length"],
                  padding=True)
        enc["labels"] = [samples["label"][j] for j in range(n)]
        return enc

    cols = ["question", "options", "context", "step"]
    train_ds = Dataset.from_list(flatten(cfg["train_data"])).map(
        encode, batched=True, remove_columns=cols)
    val_ds = Dataset.from_list(flatten(cfg["val_data"])).map(
        encode, batched=True, remove_columns=cols)
    print(f"步样本: train {len(train_ds)} / val {len(val_ds)}")

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name_or_path"],
        num_labels=1,
        problem_type="single_label_classification",  # sigmoid 头 + BCE，输出落 [0,1]
        torch_dtype=torch.bfloat16,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id  # 分类头池化需要
    lora = LoraConfig(
        r=cfg.get("lora_r", 8), lora_alpha=cfg.get("lora_alpha", 16),
        lora_dropout=cfg.get("lora_dropout", 0.1),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # attention-only，收敛
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora)
    if cfg.get("gradient_checkpointing", False):
        model.enable_input_require_grads()  # 仅开启检查点时需要（否则冻结层梯度不流经 LoRA）
    model.print_trainable_parameters()

    train_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 0.05),
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        bf16=cfg["bf16"],
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        logging_steps=cfg.get("logging_steps", 50),
        save_strategy="steps", save_steps=cfg.get("save_steps", 500),
        save_total_limit=cfg.get("save_total_limit", 2),
        eval_strategy="steps", eval_steps=cfg.get("eval_steps", 500),
        load_best_model_at_end=True, metric_for_best_model="eval_auc",
        greater_is_better=True,
        remove_unused_columns=False,
    )

    trainer = PRMTrainer(
        model=model, args=train_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tok,
        compute_metrics=compute_metrics,
        pos_weight=cfg.get("pos_weight", 0.55),          # <1 → 加权错误步
        label_smoothing=cfg.get("label_smoothing", 0.05),
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=cfg.get("early_stopping_patience", 3))],
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    print(f"best eval: {trainer.state.best_metric}")


if __name__ == "__main__":
    main()