"""过程监督 (PRM / LLM-as-Judge)：对 CoT 中间步骤做细粒度过程验证。"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI


JUDGE_SYSTEM = (
    "你是一名医学推理过程评估专家。给定一个医学问题和一段推理步骤，"
    "请判断该推理步骤在医学上是否严谨、正确、无谬误。"
    "只输出一个 0 到 1 之间的分数（可保留两位小数），不要输出任何其他文字。"
)


def split_steps(cot: str, granularity: str = "sentence") -> list[str]:
    cot = cot.strip()
    if granularity == "paragraph":
        return [s for s in re.split(r"\n\s*\n", cot) if s.strip()]
    # sentence: 按句号/换行切分
    parts = re.split(r"(?<=[。.!?])\s*|\n+", cot)
    return [s.strip() for s in parts if s.strip()]


@dataclass
class ProcessVerifier:
    judge_base_url: str
    judge_model: str
    granularity: str = "sentence"
    score_threshold: float = 0.5

    _client: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = OpenAI(base_url=self.judge_base_url, api_key="EMPTY")

    def score_step(self, question: str, step: str) -> float:
        prompt = f"问题：{question}\n推理步骤：{step}"
        for _ in range(2):  # 重试一次，失败默认 0.5（中性），避免误杀
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
                text = resp.choices[0].message.content
                if text:
                    m = re.search(r"(0(?:\.\d+)?|1(?:\.0+)?)", text)
                    if m:
                        return max(0.0, min(1.0, float(m.group(1))))
            except Exception:
                continue
        return 0.5

    def verify(self, row: dict) -> dict:
        """对单条 CoT 记录做过程验证，标记低分步骤。"""
        cot = row.get("cot", "")
        question = row["question"]
        steps = split_steps(cot, self.granularity)
        scored = [
            {"step": s, "score": self.score_step(question, s)} for s in steps
        ]
        fallacies = [s for s in scored if s["score"] < self.score_threshold]
        rec = dict(row)
        rec["reasoning_steps"] = scored
        rec["num_steps"] = len(steps)
        rec["num_fallacies"] = len(fallacies)
        rec["process_score"] = (
            sum(s["score"] for s in scored) / len(scored) if scored else 0.0
        )
        return rec
