"""步级标注流水线 v2：7B 策略模型生成多样本轨迹（含对/错）→ 分步过滤 → 32B Judge 投票(K=3, ≥2 一致保留) → 步级 PRM 数据集。

关键规范（对应评审结论）：
- 轨迹生成用 **7B 策略模型**（非 32B teacher），保留答案错误轨迹以提供负 step 样本；
- 过滤步数过少的无效轨迹、过滤过短步骤；
- 每 step 独立 K=3 次 32B Judge 调用（temp>0），≥2 票一致保留标签，无法形成多数则丢弃该 step；
- prompt 强制结构化输出（整串仅 0/1），解析失败重试一次；
- 标签用 1.0/0.0 回归；
- **32B Judge 只在离线构建 PRM 数据集时调用；RL 循环内部不调用**。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from src.answer_extraction import extract_final_answer
from src.data.cot_construction import build_prompt


STEP_JUDGE_SYSTEM = (
    "你是一名医学推理步骤评估专家。给定医学问题、推理到当前步骤为止的前置上下文、"
    "以及当前这一步，请判断该步骤在医学上是否正确、严谨、无谬误。"
    "只输出一个整数：1 或 0（1=正确，0=错误），不要输出任何其他文字。"
)

GENERATOR_SYSTEM = (
    "你是一名医学解答者。请对下面的医学问题逐步推理，"
    "最后一行以 `最终答案: <选项字母>` 给出结论。"
)

MIN_STEP_LEN = 4   # 过短步骤视为切分噪声，过滤


def build_text(q: str, options: dict, context: str, step: str) -> str:
    """PRM 输入格式：与训练/打分共用，不含答案。"""
    opt = "\n".join(f"{k}. {v}" for k, v in options.items())
    return (
        f"问题：{q}\n选项：\n{opt}\n前置上下文：{context}\n当前步骤：{step}"
    )


def split_steps(cot: str, min_len: int = MIN_STEP_LEN) -> list[str]:
    """切分步骤并做长度过滤；分布校验由调用方统计。"""
    parts = re.split(r"(?<=[。.!?])\s*|\n+", (cot or "").strip())
    steps = [s.strip() for s in parts if s.strip()]
    return [s for s in steps if len(s) >= min_len]


def _parse_label(text: str | None) -> int | None:
    """强制结构化解析：整串仅为一个 0 或 1。"""
    if not text:
        return None
    m = re.match(r"^\s*([01])\s*$", text)
    return int(m.group(1)) if m else None


@dataclass
class StepAnnotator:
    generator_base_url: str   # 7B 策略模型 vllm 服务
    generator_model: str
    judge_base_url: str       # 32B Judge vllm 服务
    judge_model: str
    k: int = 3                # 每步投票次数
    generator_temp: float = 1.0
    max_new_tokens: int = 2048
    min_steps: int = 3        # 步数少于该值视为无效轨迹

    _gen: Any = field(init=False, repr=False)
    _judge: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._gen = OpenAI(base_url=self.generator_base_url, api_key="EMPTY")
        self._judge = OpenAI(base_url=self.judge_base_url, api_key="EMPTY")

    def generate_trajectories(self, question: str, options: dict, n: int = 4) -> list[str]:
        resp = self._gen.chat.completions.create(
            model=self.generator_model,
            messages=[
                {"role": "system", "content": GENERATOR_SYSTEM},
                {"role": "user", "content": build_prompt(question, options)},
            ],
            n=n,
            temperature=self.generator_temp,
            max_tokens=self.max_new_tokens,
        )
        return [c.message.content or "" for c in resp.choices]

    def judge_step(self, question: str, options: dict, context: str,
                   step: str, k: int | None = None) -> tuple[float | None, int]:
        """返回 (多数标签 or None, 有效票数)。

        None 表示无法形成多数（有效票 <2），该 step 样本丢弃。
        """
        k = k or self.k
        opt_text = "\n".join(f"{a}. {b}" for a, b in options.items())
        votes: list[int] = []
        for _ in range(k):
            label = None
            for _retry in range(2):  # 解析容错：失败重试一次
                try:
                    resp = self._judge.chat.completions.create(
                        model=self.judge_model,
                        messages=[
                            {"role": "system", "content": STEP_JUDGE_SYSTEM},
                            {"role": "user", "content":
                                f"问题：{question}\n选项：\n{opt_text}\n"
                                f"前置上下文：{context}\n当前步骤：{step}"},
                        ],
                        temperature=0.7,
                        max_tokens=2,
                    )
                    label = _parse_label(resp.choices[0].message.content)
                except Exception:
                    label = None
                if label is not None:
                    break
            if label is not None:
                votes.append(label)
        if len(votes) < 2:  # 无法形成多数，丢弃该 step
            return None, len(votes)
        ones = sum(votes)
        return (1.0 if ones >= (len(votes) + 1) // 2 else 0.0), len(votes)

    def annotate(self, question: str, options: dict, cot: str,
                 answer: str) -> dict | None:
        """单条轨迹 -> 步级标注记录；无效轨迹（步数过少/全部步丢弃）返回 None。"""
        steps = split_steps(cot)
        if len(steps) < self.min_steps:
            return None
        context = ""
        labels = []
        for step in steps:
            label, n_votes = self.judge_step(question, options, context, step)
            if label is None:
                continue  # 该 step 样本丢弃
            labels.append({"step": step, "label": label, "n_votes": n_votes, "k": self.k})
            context = context + step + "\n"
        if not labels:
            return None
        pred = extract_final_answer(cot)
        return {
            "question": question,
            "options": options,
            "answer": answer,
            "trajectory_correct": int(pred == answer),
            "n_steps": len(steps),
            "n_kept_steps": len(labels),
            "labels": labels,
        }