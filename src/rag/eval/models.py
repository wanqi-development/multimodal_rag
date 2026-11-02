"""与数据来源无关的评测结果行。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag.eval.judge import JudgeVerdict


@dataclass
class EvaluationRow:
    """一次检索 / 生成观测，可由任意数据集或线上快照构造。"""

    id: str
    question: str
    tags: tuple[str, ...] = ()
    retrieval_case: bool = False
    must_refuse: bool = False
    rank: int = 0
    top: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    context_size: int = 0
    verdict: JudgeVerdict | None = None
    error: str = ""
    faithfulness: float | None = None
