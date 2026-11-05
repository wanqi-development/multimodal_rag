"""不依赖固定数据集的通用评测原语。"""

from __future__ import annotations

import asyncio
import json

import pytest

from rag.eval import (
    EvaluationRow,
    JudgeVerdict,
    build_report,
    judge_case,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    used_citations,
)
from rag.eval.judge import parse_verdict


def test_eval_retrieve_preserves_shared_service_result(monkeypatch):
    from types import SimpleNamespace

    from rag.config.schema import Settings
    from rag.contracts import RetrievalResult
    from rag.eval.wiring import build_retrieve
    from rag.service.retrieval import RetrievalService

    settings = Settings()
    seen = []

    async def retrieve(self, question, top_k):
        seen.append((self._settings, question, top_k))
        return RetrievalResult(query=question, rerank_status="failed")

    monkeypatch.setattr(RetrievalService, "retrieve", retrieve)
    heavy = SimpleNamespace(index=object(), embedder=object(), reranker=object(), chat=object())
    result = asyncio.run(build_retrieve(settings, heavy)("q", 3))
    assert result.chunks == []
    assert result.rerank_status == "failed"
    assert seen == [(settings, "q", 3)]


@pytest.mark.parametrize("has_hits", [False, True])
def test_eval_generation_preserves_degradations_and_full_citation_text(monkeypatch, has_hits):
    from types import SimpleNamespace

    from rag.config.schema import Settings
    from rag.contracts import Chunk, RetrievalDegradation, RetrievalResult, ScoredChunk
    from rag.eval.wiring import build_generate, build_retrieve
    from rag.service.retrieval import RetrievalService

    result = RetrievalResult(
        query="q",
        chunks=[
            ScoredChunk(
                chunk=Chunk(chunk_id="c", doc_id="d", text="全文证据"),
                score=0.9,
                channel="a",
                rank=0,
            )
        ]
        if has_hits
        else [],
        score_kind="rerank",
        rerank_status="succeeded",
        degradations=[RetrievalDegradation(stage="b", code="channel_failed", message="故障")],
    )

    async def retrieve(self, question, top_k):
        return result

    async def generate(question, context):
        assert context.score_kind == "rerank"
        assert context.rerank_status == "succeeded"
        assert context.degradations == result.degradations
        assert "部分检索通道失败" in context.retrieval_notice()
        assert "不能断言语料中没有相关内容" in context.retrieval_notice()
        return "答案 [1]" if has_hits else "检索不完整"

    monkeypatch.setattr(RetrievalService, "retrieve", retrieve)
    heavy = SimpleNamespace(
        index=None,
        embedder=None,
        reranker=None,
        chat=None,
        generator=SimpleNamespace(generate=generate),
    )
    settings = Settings()

    async def run():
        retrieval = await build_retrieve(settings, heavy)("q", 3)
        return await build_generate(settings, heavy)("q", retrieval)

    answer, citations = asyncio.run(run())
    assert answer == ("答案 [1]" if has_hits else "检索不完整")
    assert [item["text"] for item in citations] == (["全文证据"] if has_hits else [])


def test_recall_counts_only_within_k() -> None:
    ranks = [1, 3, 0, 8]

    assert recall_at_k(ranks, 3) == 0.5
    assert recall_at_k(ranks, 8) == 0.75
    assert recall_at_k([], 5) == 0.0


def test_missing_hits_pull_the_average_down() -> None:
    assert reciprocal_rank([1, 0]) == 0.5
    assert reciprocal_rank([0, 0]) == 0.0


def test_ndcg_prefers_relevant_material_ranked_higher() -> None:
    assert ndcg_at_k([1], 8) == pytest.approx(1.0)
    assert ndcg_at_k([2], 8) < ndcg_at_k([1], 8)
    assert ndcg_at_k([0], 8) == 0.0


def test_verdict_survives_code_fences_and_string_booleans() -> None:
    verdict = parse_verdict(
        "结果如下：\n```json\n"
        '{"citations": [{"index": "[1]", "supported": "true"}], '
        '"points": [{"point": "甲", "covered": "false"}], "refused": false}\n```'
    )

    assert verdict.error == ""
    assert verdict.citations[0]["index"] == 1
    assert verdict.citations[0]["supported"] is True
    assert verdict.points[0]["covered"] is False
    assert verdict.citation_accuracy == 1.0
    assert verdict.point_coverage == 0.0


def test_unparseable_judge_output_is_an_error_not_a_pass() -> None:
    verdict = parse_verdict("我觉得还不错")

    assert verdict.error
    assert verdict.citations == []
    assert verdict.citation_accuracy is None


def test_only_citations_used_by_the_answer_are_selected() -> None:
    citations = [{"index": index, "text": f"片段{index}"} for index in (1, 2, 3)]

    assert [item["index"] for item in used_citations("结论 [2]", citations)] == [2]


def test_judge_case_accepts_arbitrary_question_and_answer_points() -> None:
    prompts: list[str] = []

    async def ask(prompt: str) -> str:
        prompts.append(prompt)
        return json.dumps(
            {
                "citations": [{"index": 1, "supported": True}],
                "points": [{"point": "要点", "covered": True}],
                "refused": False,
            }
        )

    verdict = asyncio.run(
        judge_case(
            ask,
            question="任意问题",
            answer="答案 [1]",
            citations=[{"index": 1, "text": "证据"}],
            answer_points=("要点",),
        )
    )

    assert verdict.citation_accuracy == 1.0
    assert verdict.point_coverage == 1.0
    assert "任意问题" in prompts[0] and "要点" in prompts[0]


def test_report_keeps_macro_and_micro_citation_accuracy_separate() -> None:
    rows = [
        EvaluationRow(
            id="one-good",
            question="甲",
            retrieval_case=True,
            rank=1,
            verdict=JudgeVerdict(citations=[{"index": 1, "supported": True}]),
        ),
        EvaluationRow(
            id="eight-bad",
            question="乙",
            retrieval_case=True,
            rank=0,
            verdict=JudgeVerdict(
                citations=[{"index": index, "supported": False} for index in range(1, 9)]
            ),
        ),
    ]

    summary = build_report(rows, top_k=8)["summary"]

    assert summary["retrieval_cases"] == 2
    assert summary["retrieval"]["recall@8"] == 0.5
    assert summary["citation_accuracy"] == 0.5
    assert summary["citation_accuracy_micro"] == pytest.approx(1 / 9)
