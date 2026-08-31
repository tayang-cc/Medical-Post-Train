"""复合 Reward：格式检查（硬约束）+ LLM-as-Judge（连续打分）。

针对 GRPO 组内奖励同质化 / 优势函数失效问题：
- 格式分：对最终答案做结构化抽取，与标准答案硬匹配，给出稀疏但可靠的 0/1 信号；
- Judge 分：LLM-as-Judge 对答案与推理过程打连续分，拉开组内相对优势差距。

Judge 用线程池并发调用，避免逐条串行成为 RL 训练瓶颈。
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from src.answer_extraction import extract_final_answer
from src.data.step_annotation import split_steps
from src.process.prm_scorer import PRMScorer


def to_text(completion: Any) -> str:
    """兼容 trl 对话模式下把 completion 包装成 messages 列表的情况。"""
    if isinstance(completion, list):
        parts = []
        for m in completion:
            if isinstance(m, dict):
                parts.append(m.get("content", "") or "")
            else:
                parts.append(str(m))
        return "\n".join(parts)
    return completion or ""


def format_reward(completion: Any, answer: str) -> float:
    """硬约束：最终答案可解析且与标准答案一致得 1，否则 0。"""
    pred = extract_final_answer(to_text(completion))
    return 1.0 if pred == answer else 0.0


JUDGE_SYSTEM = (
    "你是一名医学评分专家。给定问题、标准答案和模型的回答，"
    "请评估模型答案的正确性与推理质量。只输出 0 到 1 之间的分数"
    "（可保留两位小数），不要输出任何其他文字。"
)


def _parse_score(text: str | None) -> float | None:
    if not text:
        return None
    m = re.search(r"(0(?:\.\d+)?|1(?:\.0+)?)", text)
    if not m:
        return None
    return max(0.0, min(1.0, float(m.group(1))))


@dataclass
class CompositeReward:
    judge_base_url: str
    judge_model: str
    format_weight: float = 1.0
    judge_weight: float = 1.0
    judge_max_workers: int = 8
    enable_judge: bool = True  # False = 纯规则 reward（跳过 LLM-as-Judge，无需 judge 模型）

    _client: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = OpenAI(base_url=self.judge_base_url, api_key="EMPTY")

    def judge_score(self, question: str, answer: str, completion: Any) -> float:
        prompt = (
            f"问题：{question}\n标准答案：{answer}\n模型回答：{to_text(completion)}\n"
            "请给出正确性与推理质量的综合评分。"
        )
        try:
            resp = self._client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=16,
            )
            score = _parse_score(resp.choices[0].message.content)
        except Exception:
            score = None
        return score if score is not None else 0.0

    def __call__(
        self, completions: list[Any], questions: list[str], answers: list[str]
    ) -> list[float]:
        fmt = [format_reward(c, a) for c, a in zip(completions, answers)]
        if self.enable_judge:
            with ThreadPoolExecutor(max_workers=self.judge_max_workers) as ex:
                futures = [
                    ex.submit(self.judge_score, q, a, c)
                    for c, q, a in zip(completions, questions, answers)
                ]
                jdg = [f.result() for f in futures]
        else:
            jdg = [0.0] * len(completions)
        return [
            self.format_weight * f + self.judge_weight * j
            for f, j in zip(fmt, jdg)
        ]


@dataclass
class PRMCompositeReward:
    """进程内 7B-PRM 复合奖励：格式门控 + 二进制结果 + min 聚合过程分。

    RL 循环内部不调用 32B Judge（32B 仅离线构建 PRM 数据集）。
    公式：reward = format_gate * (result_weight*result + process_weight*min(step_score))
    - format_gate：最终答案可解析才通过（否则 0，防格式劫持）；
    - result：抽取答案 == 标准答案（二进制）；
    - process：整条轨迹分步打分后取 min（最薄弱步骤决定过程质量）。
    """

    scorer: PRMScorer
    result_weight: float = 1.0
    process_weight: float = 1.0
    batch_size: int = 64

    def __call__(self, completions: list[Any], questions: list[str],
                 answers: list[str], options_list: list[dict]) -> list[float]:
        texts = [to_text(c) for c in completions]
        preds = [extract_final_answer(t) for t in texts]

        # 1) 所有 completion 分步，展开成 (question, options, context, step) 打分项
        items: list[dict] = []
        ranges: list[tuple[int, int]] = []
        for t, q, opt in zip(texts, questions, options_list):
            steps = split_steps(t)
            start = len(items)
            context = ""
            for s in steps:
                items.append({"question": q, "options": opt,
                              "context": context, "step": s})
                context = context + s + "\n"
            ranges.append((start, len(items)))

        # 2) 批量打分（进程内 7B-PRM）
        scores: list[float] = []
        for i in range(0, len(items), self.batch_size):
            scores.extend(self.scorer.score_batch(items[i:i + self.batch_size]))

        # 3) 逐条合成
        out: list[float] = []
        for i, (pred, (s, e)) in enumerate(zip(preds, ranges)):
            if pred is None:            # 格式门控失败
                out.append(0.0)
                continue
            result = 1.0 if pred == answers[i] else 0.0
            seg = scores[s:e]
            process = min(seg) if seg else 0.0
            out.append(self.result_weight * result
                       + self.process_weight * process)
        return out