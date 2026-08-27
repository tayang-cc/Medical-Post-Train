"""SFT 阶段 CoT 构造：用教师模型对医学多选题生成思维链推理轨迹。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from src.answer_extraction import extract_final_answer


SYSTEM_PROMPT = (
    "你是一名资深医学专家。请对下面的医学问题逐步推理，"
    "先给出严谨的推理过程，最后一行以 `最终答案: <选项字母>` 的形式给出结论。"
    "推理过程要包含：鉴别诊断要点、排除其他选项的理由、最终选择的依据。"
)


def build_prompt(question: str, options: dict[str, str]) -> str:
    opt_text = "\n".join(f"{k}. {v}" for k, v in options.items())
    return f"问题：{question}\n选项：\n{opt_text}\n请逐步推理并给出最终答案。"


@dataclass
class CoTConstructor:
    teacher_base_url: str
    teacher_model: str
    num_samples: int = 1
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 2048
    keep_only_correct: bool = True

    _client: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = OpenAI(base_url=self.teacher_base_url, api_key="EMPTY")

    def generate(self, question: str, options: dict[str, str]) -> list[str]:
        prompt = build_prompt(question, options)
        resp = self._client.chat.completions.create(
            model=self.teacher_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            n=self.num_samples,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
        )
        return [
            c.message.content or ""
            for c in resp.choices
        ]

    def construct(self, raw_rows: list[dict]) -> list[dict]:
        out: list[dict] = []
        for row in raw_rows:
            question = row["question"]
            options = row["options"]
            answer = row["answer"].upper()
            for cot in self.generate(question, options):
                pred = extract_final_answer(cot)
                if self.keep_only_correct and pred != answer:
                    continue
                rec = dict(row)
                rec["cot"] = cot
                rec["pred"] = pred
                rec["correct"] = (pred == answer)
                out.append(rec)
        return out


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
