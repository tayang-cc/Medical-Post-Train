"""统一的最终答案抽取：reward / CoT 构造 / 评测共用一份实现，避免行为不一致。

显式答案格式优先；兜底只匹配「最后一行」的独立选项字母，避免把推理过程中的
「维生素A」「选项C」等误判成答案。
"""
from __future__ import annotations

import re

_PATTERNS = [
    r"最终答案\s*[:：]?\s*[（(]?\s*([A-Ea-e])\s*[)）]?",
    r"正确答案\s*[:：]?\s*[（(]?\s*([A-Ea-e])\s*[)）]?",
    r"答案\s*[:：]?\s*[（(]?\s*([A-Ea-e])\s*[)）]?",
    r"选择\s*[（(]?\s*([A-Ea-e])\s*[)）]",
    r"\\boxed\{([A-Ea-e])\}",
    r"<answer>\s*([A-Ea-e])\s*</answer>",
]


def extract_final_answer(text: str | None) -> str | None:
    if not text:
        return None
    for p in _PATTERNS:
        m = re.findall(p, text, flags=re.IGNORECASE)
        if m:
            return m[-1].upper()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    m = re.findall(r"\b([A-E])\b", lines[-1])
    return m[-1].upper() if m else None