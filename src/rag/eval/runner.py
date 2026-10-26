"""把黄金集跑一遍：检索（免费）→ 可选的生成 + 判定（花钱）。

两个依赖都是**注入**的（`retrieve` / `generate` / `ask`），所以这一层能用替身单测，
而 CLI 那边注入的是生产链路本身（`build_retriever` / `assemble` + `generator`）——
评测跑的必须是线上那条路，另写一条「简化版检索」等于给自己发一张假成绩单。

三条容易写错、这里刻意钉住的规矩：

1. **语料外的问题不参与检索指标**（它本来就不该命中），只参与「该不该拒答」那一项；
2. **判不了 ≠ 判通过**：模型输出解析不出来时记 `verdict.error`，报告里单列（见 report.py）；
3. 每条问题的**排名记 0 表示没命中**，`metrics.py` 的所有指标据此退化成 0。
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from rag.contracts import ScoredChunk
from rag.eval.dataset import GoldenCase, is_relevant
from rag.eval.judge import JudgeVerdict, judge_case

# 检索：问题 → 命中。与 retrieval 层的签名一致，只是不带 filters。
RetrieveFn = Callable[[str, int], Awaitable[list[ScoredChunk]]]
# 生成：(问题, 命中) → (答案, 引用清单)。引用清单里的 text 要**给全片段**，判定才有依据。
GenerateFn = Callable[[str, list[ScoredChunk]], Awaitable[tuple[str, list[dict[str, Any]]]]]
# 判定用的一次问答：prompt → 模型原文
AskFn = Callable[[str], Awaitable[str]]


@dataclass
class CaseResult:
    id: str
    question: str
    tags: tuple[str, ...] = ()
    # 这条是不是「该命中语料」的条目：**没命中的也要进分母**，否则指标会虚高
    retrieval_case: bool = False
    # 语料外的问题（该拒答）：报告用它算「漏拒 / 误拒」
    must_refuse: bool = False
    # 第一条相关结果的排名（1 起）；0 = 没命中
    rank: int = 0
    # 每条命中的概要，报告里打出来 —— 「期望落空」时要能一眼看出它召回了什么
    top: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    # 答案里**真的用到**的引用（判定只看这些）；`context_size` 是装配进上下文的块数
    citations: list[dict[str, Any]] = field(default_factory=list)
    context_size: int = 0
    verdict: JudgeVerdict | None = None
    error: str = ""
    # ragas 的 faithfulness（可选：只有 --ragas 时才有值）
    faithfulness: float | None = None

    @property
    def judged(self) -> bool:
        return self.verdict is not None


def _outline(hits: Sequence[ScoredChunk], case: GoldenCase, *, keep: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, hit in enumerate(hits[:keep], 1):
        chunk = hit.chunk
        rows.append(
            {
                "rank": position,
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "source_uri": str(chunk.metadata.get("source_uri", "")),
                "heading_path": chunk.heading_path,
                "modality": chunk.modality,
                "score": round(float(hit.score), 4),
                "relevant": is_relevant(
                    case,
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    heading_path=chunk.heading_path,
                ),
            }
        )
    return rows


def first_relevant_rank(hits: Sequence[ScoredChunk], case: GoldenCase) -> int:
    """第一条相关材料排第几；没命中记 0。"""
    for position, hit in enumerate(hits, 1):
        chunk = hit.chunk
        if is_relevant(
            case,
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            heading_path=chunk.heading_path,
        ):
            return position
    return 0


def used_citations(answer: str, citations: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """答案里**真的引用到**的条目（按编号）。

    **必须过滤，否则指标问错了问题**：`assemble` 会给装进上下文的**每一块**都发编号，而答案
    常常只用了其中几条。把没用到的也算进分母，等于在问「你有没有把材料引用完」——
    实测这一条把引用准确率从 ~95% 压到 76%（第一次 --judge 跑出来的就是那个数）。
    引用准确率要回答的是：**你做出的每一次引用，站得住吗**。
    """
    used = {int(match) for match in re.findall(r"\[(\d+)\]", answer)}
    return [item for item in citations if item.get("index") in used]


async def run_eval(
    cases: Sequence[GoldenCase],
    *,
    retrieve: RetrieveFn,
    generate: GenerateFn | None = None,
    ask: AskFn | None = None,
    faithfulness: Callable[[str, str, list[str]], Awaitable[float]] | None = None,
    top_k: int = 8,
    outline_size: int = 5,
) -> list[CaseResult]:
    """跑一遍。`generate` / `ask` 不给就只算检索指标（免费的那一半）。

    `generate` 与 `ask` 要**一起**给：没有判定的生成结果只能看，不能算分。
    """
    if (generate is None) != (ask is None):
        raise ValueError("generate 与 ask 要么都给（跑生成侧判定），要么都不给（只跑检索）")

    results: list[CaseResult] = []
    for case in cases:
        hits = await retrieve(case.question, top_k)
        result = CaseResult(
            id=case.id,
            question=case.question,
            tags=case.tags,
            retrieval_case=case.is_retrieval_case,
            must_refuse=case.must_refuse,
            rank=first_relevant_rank(hits, case) if case.is_retrieval_case else 0,
            top=_outline(hits, case, keep=outline_size),
        )
        if generate is not None and ask is not None and case.is_judged_case:
            try:
                answer, citations = await generate(case.question, hits)
                result.answer = answer
                result.context_size = len(citations)
                result.citations = used_citations(answer, citations)
                result.verdict = await judge_case(
                    ask, case, answer=answer, citations=result.citations
                )
                if faithfulness is not None:
                    contexts = [
                        str(item.get("text") or "") for item in citations if item.get("text")
                    ]
                    with contextlib.suppress(Exception):
                        # 判不了就不记（拒答/没上下文那类本来就不适用），别让整轮挂掉
                        result.faithfulness = await faithfulness(case.question, answer, contexts)
            except Exception as exc:  # noqa: BLE001 —— 单条失败不该带走整轮评测
                result.error = f"{type(exc).__name__}: {exc}"
        results.append(result)
    return results
