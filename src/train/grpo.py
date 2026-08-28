"""GRPO 强化学习：复合 Reward（格式 + Judge）+ KL 退火 + Entropy 监控。

动态 KL 调节：随训练步数对 beta 做 cosine/linear 退火，训练早期约束强、后期放松，
抑制 Reward Hacking，并同步监控 policy 熵（-mean log p 代理）与 KL 散度。

注意：GRPO 接口按 trl==0.15.0 适配（requirements.txt 已固定），升级 trl 需自行适配。
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch
import yaml
from datasets import Dataset
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from src.reward.composite import CompositeReward
from src.train.lora import add_lora_args, build_lora_config


def build_prompt(question: str, options: dict) -> list[dict]:
    opt_text = "\n".join(f"{k}. {v}" for k, v in options.items())
    content = (
        f"问题：{question}\n选项：\n{opt_text}\n"
        "请逐步推理，最后一行以 `最终答案: <选项字母>` 结束。"
    )
    # 用 messages 列表，trl 会据此自动套 chat template（Qwen2.5-Instruct 需要 instruct 格式）
    return [{"role": "user", "content": content}]


def load_prompts(path: str) -> Dataset:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # 优先用 03b 已构造好的 messages prompt，兼容 {question, options} 原始格式
            if isinstance(r.get("prompt"), list):
                prompt = r["prompt"]
            else:
                prompt = build_prompt(r["question"], r["options"])
            rows.append(
                {
                    "prompt": prompt,
                    "question": r["question"],
                    "answer": r["answer"],
                }
            )
    return Dataset.from_list(rows)


def check_generations(cfg: dict) -> None:
    """trl 要求全局 batch（per_device x num_processes）能被 num_generations 整除。"""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    g = int(cfg["num_generations"])
    per_device = int(cfg["per_device_train_batch_size"])
    global_batch = per_device * world
    if g < 2 or global_batch < g or global_batch % g != 0:
        raise SystemExit(
            f"num_generations={g} 不合法：需要 2 <= G <= per_device_batch({per_device}) "
            f"x num_processes({world}) 且整除。\n"
            f"单卡请把 configs/grpo.yaml 的 per_device_train_batch_size 设为 {g}，"
            f"或把 num_generations 改为 {global_batch}。"
        )


class KLAnnealingCallback(TrainerCallback):
    """KL 退火：cosine / linear 调度更新 GRPO 的 beta 系数。"""

    def __init__(self, trainer, beta_start, beta_end, total_steps=None,
                 schedule="cosine"):
        self.trainer = trainer
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.total_steps = total_steps
        self.schedule = schedule

    def on_step_begin(self, args, state, control, **kwargs):
        total = self.total_steps or state.max_steps or 1
        frac = min(state.global_step / max(total, 1), 1.0)
        if self.schedule == "linear":
            beta = self.beta_start + (self.beta_end - self.beta_start) * frac
        else:  # cosine
            beta = self.beta_end + (self.beta_start - self.beta_end) * 0.5 * (
                1 + math.cos(math.pi * frac)
            )
        # GRPO loss 读取 self.beta，需同时更新实例属性与 args
        self.trainer.beta = beta
        self.trainer.args.beta = beta


class MedicalGRPOTrainer(GRPOTrainer):
    """带 policy 熵监控的 GRPOTrainer。

    熵代理 = -mean(log p) 按 completion mask 平均，复用 compute_loss 里已算好的
    per-token logps（无额外前向），因此必须让 compute_loss 先于熵计算执行。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._policy_entropy = None
        self._last_policy_logps = None

    def _prepare_inputs(self, inputs):
        # trl 0.15 在 ref logps 计算时用 disable_adapter()，某些 peft 版本退出后
        # adapter 未恢复启用，导致 compute_loss 前向无梯度。这里显式恢复。
        result = super()._prepare_inputs(inputs)
        model = getattr(self, "model", None)
        if model is not None:
            try:
                unwrapped = self.accelerator.unwrap_model(model)
                if hasattr(unwrapped, "enable_adapter"):
                    unwrapped.enable_adapter()
            except Exception:
                pass
        return result

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        logps = super()._get_per_token_logps(
            model, input_ids, attention_mask, logits_to_keep
        )
        # compute_loss 里唯一一次调用即 policy 前向，保存后即为当步策略 logps
        self._last_policy_logps = logps.detach()
        return logps

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        self._policy_entropy = None
        loss = super().compute_loss(
            model, inputs, return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
        self._compute_entropy_proxy(inputs)
        return loss

    def train(self, *args, **kwargs):
        # trl 0.15 + 部分 transformers 版本在训练收尾 _maybe_log_save_evaluate
        # 会因 self.control 类型问题崩溃（训练本身已完成）。捕获后手动保存。
        try:
            return super().train(*args, **kwargs)
        except AttributeError as e:
            if "should_evaluate" in str(e):
                print("[GRPO] 训练已完成，收尾保存崩溃被捕获，手动保存模型")
                self.save_model(self.args.output_dir)
                return
            raise

    @torch.no_grad()
    def _compute_entropy_proxy(self, inputs):
        logps = self._last_policy_logps
        mask = inputs.get("completion_mask") if isinstance(inputs, dict) else None
        if logps is None or mask is None:
            return
        denom = mask.sum().clamp(min=1.0)
        self._policy_entropy = float(-(logps * mask).sum() / denom)


class MonitoringCallback(TrainerCallback):
    """每个 log 步记录 policy 熵与当前 beta。"""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if getattr(self.trainer, "_policy_entropy", None) is not None:
            logs["policy_entropy"] = self.trainer._policy_entropy
        logs["beta"] = getattr(self.trainer, "beta", None)
        return logs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo.yaml")
    parser.add_argument("--per-device-train-batch-size", type=int, default=None,
                        help="覆盖配置里的 batch size（单卡 LoRA 建议 >= num_generations）")
    parser.add_argument("--num-generations", type=int, default=None)
    add_lora_args(parser)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.per_device_train_batch_size is not None:
        cfg["per_device_train_batch_size"] = args.per_device_train_batch_size
    if args.num_generations is not None:
        cfg["num_generations"] = args.num_generations

    check_generations(cfg)

    dataset = load_prompts(cfg["dataset_path"])

    reward = CompositeReward(
        judge_base_url=cfg["reward"]["judge_base_url"],
        judge_model=cfg["reward"]["judge_model"],
        format_weight=cfg["reward"]["format_weight"],
        judge_weight=cfg["reward"]["judge_weight"],
        enable_judge=cfg["reward"].get("judge_enabled", True),
    )

    def reward_func(prompts, completions, question, answer, **kwargs):
        return reward(
            completions=completions,
            questions=question,
            answers=answer,
        )

    grpo_config = GRPOConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        num_generations=cfg["num_generations"],
        max_completion_length=cfg["max_completion_length"],
        temperature=cfg["temperature"],
        beta=cfg["beta"],
        bf16=cfg["bf16"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        logging_steps=cfg["logging_steps"],
    )

    lora_config = build_lora_config(args)
    if lora_config is not None:
        print(
            "[LoRA] 已启用 LoRA；单卡请直接 python 运行本脚本（勿走 accelerate/deepspeed，"
            "trl 0.15 在 zero3+peft 下仍会加载独立 ref model，反而吃显存）。"
        )

    trainer = MedicalGRPOTrainer(
        model=cfg["model_name_or_path"],
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_func,
        peft_config=lora_config,
    )

    anneal = cfg.get("kl_annealing", {})
    if anneal.get("enabled", False):
        trainer.add_callback(KLAnnealingCallback(
            trainer=trainer,
            beta_start=anneal.get("beta_start", cfg["beta"]),
            beta_end=anneal.get("beta_end", cfg["beta"]),
            total_steps=anneal.get("total_steps"),
            schedule=anneal.get("schedule", "cosine"),
        ))
    trainer.add_callback(MonitoringCallback(trainer))

    trainer.train()
    trainer.save_model(cfg["output_dir"])


if __name__ == "__main__":
    main()