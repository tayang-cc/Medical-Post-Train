"""7B-PRM 推理打分器（供 RL 奖励 / 数据清洗用）。

- 头为 sigmoid+BCE（problem_type=single_label_classification），输出天然落 [0,1]；
- 推理侧仍强制 clip 双保险；越界次数累积告警；
- 输入 question + 前置上下文 + 当前步骤，返回该步正确性分数。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.step_annotation import build_text  # 与训练一致的输入格式

_CLAMP_MIN, _CLAMP_MAX = 0.0, 1.0


class PRMScorer:
    def __init__(self, base_path: str, adapter_path: str,
                 device: str = "cuda") -> None:
        self.tok = AutoTokenizer.from_pretrained(base_path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        base = AutoModelForSequenceClassification.from_pretrained(
            base_path, num_labels=1, problem_type="single_label_classification",
            torch_dtype=torch.bfloat16)
        if base.config.pad_token_id is None:
            base.config.pad_token_id = self.tok.pad_token_id  # 分类头池化需要
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.to(device).eval()
        self.device = device
        self._out_of_range = 0
        self._calls = 0

    @torch.no_grad()
    def score(self, question: str, options: dict, context: str,
              step: str) -> float:
        text = build_text(question, options, context, step)
        enc = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=2048).to(self.device)
        logit = self.model(**enc).logits.squeeze().float().item()
        score = float(F.sigmoid(torch.tensor(logit)))
        self._calls += 1
        if not _CLAMP_MIN <= score <= _CLAMP_MAX:
            self._out_of_range += 1
        return min(_CLAMP_MAX, max(_CLAMP_MIN, score))

    @torch.no_grad()
    def score_batch(self, items: list[dict]) -> list[float]:
        """items: [{question, options, context, step}, ...]，批量打分。"""
        texts = [build_text(i["question"], i["options"], i["context"], i["step"])
                 for i in items]
        enc = self.tok(texts, return_tensors="pt", truncation=True,
                       padding=True, max_length=2048).to(self.device)
        logits = self.model(**enc).logits.reshape(-1).float()
        scores = F.sigmoid(logits).cpu().tolist()
        self._calls += len(scores)
        oob = sum(1 for v in scores if not _CLAMP_MIN <= v <= _CLAMP_MAX)
        self._out_of_range += oob
        return [min(_CLAMP_MAX, max(_CLAMP_MIN, v)) for v in scores]

    @property
    def oob_ratio(self) -> float:
        return self._out_of_range / max(1, self._calls)