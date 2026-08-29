"""DAPO：Decoupled Clip and Dynamic sAmpling Policy Optimization（基于 trl 0.15 GRPOTrainer）。

参考论文：DAPO: An Open-Source LLM Reinforcement Learning System at Scale
https://arxiv.org/abs/2503.14476 （ByteDance Seed）

相对 GRPO 的四个改动（对应论文 §3）：
1. Token-Level Loss（§3.3）：loss 按全 batch 的 token 总数归一（1/Σ|o_i|），
   长样本对梯度贡献更大，避免短样本主导 / 长样本的乱码重复被稀释。
2. Dynamic Sampling（§3.2）：丢弃组内 reward 全相同（全对或全错）的 prompt。
   这类组 advantage≈0、梯度≈0，只会放大 batch 梯度噪声。用 |adv| 上界阈值判定。
3. Clip-Higher（§3.1）：非对称 clip，下界 1-eps_low、上界 1+eps_high，
   给低概率「探索」token 留更多上升空间，缓解熵坍缩。
4. Overlong Reward Shaping（§3.4，Eq.13）：对超长/截断样本加软惩罚，
   避免「推理对了只因太长被截断」引入奖励噪声。
5. 去掉 KL 惩罚（§2.3）：beta 默认 0；需要时也可在配置里开启。

实现说明（诚实标注）：
trl 0.15 每步「采样一次 + 更新一次」，没有 old-policy 缓冲，严格 PPO 式
old-ratio 拿不到；故 Clip-Higher 的 ratio 用「当前策略 vs 冻结参考模型」
r=exp(logp_θ - logp_ref)。训练中策略相对参考模型漂移，clip 约束的正是
每次更新的偏离量，语义等价、且随训练自动生效。
"""
from __future__ import annotations

import argparse

import torch
import yaml

from src.train.grpo import (
    KLAnnealingCallback,
    MedicalGRPOTrainer,
    MonitoringCallback,
    check_generations,
    load_prompts,
)
from src.reward.composite import CompositeReward
from src.train.lora import add_lora_args, build_lora_config


class DAPOTrainer(MedicalGRPOTrainer):
    """在 MedicalGRPOTrainer（GRPO + 熵监控）基础上实现 DAPO 的 loss。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # DAPO 超参默认值；main() 里会从配置覆盖
        self.dapo_eps_low = 0.2
        self.dapo_eps_high = 0.28
        self.dapo_dynamic_sampling = True
        self.dapo_dynamic_sampling_eps = 1e-3
        self.dapo_overlong_shaping = True
        self.dapo_max_length = self.max_completion_length
        self.dapo_punish_cache = 256
        self.entropy_coef = 0.0  # 熵正则系数；>0 时在 loss 中加熵奖励，抑制策略坍缩

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        if return_outputs:
            raise ValueError("DAPOTrainer 不支持 return_outputs=True")
        self._policy_entropy = None

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps(
            model, input_ids, attention_mask, logits_to_keep
        )
        ref_per_token_logps = inputs["ref_per_token_logps"]
        advantages = inputs["advantages"]

        # ---------- Clip-Higher：当前策略 vs 参考模型的非对称 clip ----------
        # r = π_θ / π_ref，随训练漂移；上界 1+eps_high 放开探索 token 的上升空间
        ratio = torch.exp(per_token_logps - ref_per_token_logps)
        ratio_clipped = ratio.clamp(
            1.0 - self.dapo_eps_low, 1.0 + self.dapo_eps_high
        )

        def surrogate(adv: torch.Tensor) -> torch.Tensor:
            a = adv.unsqueeze(1)
            return torch.min(ratio * a, ratio_clipped * a)

        # ---------- Overlong Reward Shaping：超长/截断样本软惩罚 ----------
        adv_eff = advantages
        if self.dapo_overlong_shaping:
            lengths = completion_mask.sum(dim=1).float()
            lmax = float(self.dapo_max_length)
            lcache = float(self.dapo_punish_cache)
            penalty = torch.zeros_like(lengths)
            soft = (lengths > lmax - lcache) & (lengths <= lmax)
            hard = lengths > lmax
            penalty[soft] = (lmax - lcache - lengths[soft]) / lcache
            penalty[hard] = -1.0
            adv_eff = advantages + penalty
            self._metrics["dapo/overlong_frac"].append(
                (lengths > lmax - lcache).float().mean().item()
            )

        pg = surrogate(adv_eff)

        # ---------- 可选 KL（DAPO 默认移除）----------
        if self.beta > 0:
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )
            per_token_loss = -(pg - self.beta * per_token_kl)
        else:
            per_token_loss = -pg

        # ---------- Dynamic Sampling：过滤组内 reward 全相同的 prompt ----------
        sample_keep = torch.ones(
            advantages.size(0), dtype=torch.bool, device=advantages.device
        )
        if self.dapo_dynamic_sampling:
            g = self.num_generations
            group_keep = (
                advantages.view(-1, g).abs().amax(dim=1)
                >= self.dapo_dynamic_sampling_eps
            )  # (num_groups,)
            sample_keep = group_keep.repeat_interleave(g)
            self._metrics["dapo/dynamic_filter_frac"].append(
                (1.0 - sample_keep.float().mean()).item()
            )
        keep = sample_keep.unsqueeze(1).float()  # (B,1)

        # ---------- Token-Level Loss：按全 batch token 总数归一 ----------
        masked = per_token_loss * completion_mask * keep
        denom = (completion_mask * keep).sum().clamp(min=1.0)
        loss = masked.sum() / denom

        # ---------- 熵正则：loss += entropy_coef * mean(log p)（采样代理 = -H 奖励）----------
        if self.entropy_coef > 0:
            e_denom = completion_mask.sum().clamp(min=1.0)
            loss = loss + self.entropy_coef * (
                per_token_logps * completion_mask
            ).sum() / e_denom

        # 与 GRPO 同口径的监控指标
        completion_length = completion_mask.sum(dim=1).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        mean_kl = (
            ((per_token_kl if self.beta > 0 else torch.zeros_like(per_token_logps))
             * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)
        ).mean().item()
        self._metrics["kl"].append(mean_kl)

        self._compute_entropy_proxy(inputs)
        return loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dapo.yaml")
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

    from trl import GRPOConfig

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
        beta=cfg.get("beta", 0.0),
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

    trainer = DAPOTrainer(
        model=cfg["model_name_or_path"],
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_func,
        peft_config=lora_config,
    )

    d = cfg.get("dapo", {})
    trainer.dapo_eps_low = float(d.get("eps_low", 0.2))
    trainer.dapo_eps_high = float(d.get("eps_high", 0.28))
    trainer.dapo_dynamic_sampling = bool(d.get("dynamic_sampling", True))
    trainer.dapo_dynamic_sampling_eps = float(d.get("dynamic_sampling_eps", 1e-3))
    trainer.dapo_overlong_shaping = bool(d.get("overlong_shaping", True))
    trainer.dapo_max_length = float(d.get("max_length", cfg["max_completion_length"]))
    trainer.dapo_punish_cache = float(d.get("punish_cache", 256))
    trainer.entropy_coef = float(cfg.get("entropy_coef", 0.0))

    anneal = cfg.get("kl_annealing", {})
    if anneal.get("enabled", False):
        trainer.add_callback(KLAnnealingCallback(
            trainer=trainer,
            beta_start=anneal.get("beta_start", cfg.get("beta", 0.0)),
            beta_end=anneal.get("beta_end", cfg.get("beta", 0.0)),
            total_steps=anneal.get("total_steps"),
            schedule=anneal.get("schedule", "cosine"),
        ))
    trainer.add_callback(MonitoringCallback(trainer))

    trainer.train()
    trainer.save_model(cfg["output_dir"])


if __name__ == "__main__":
    main()