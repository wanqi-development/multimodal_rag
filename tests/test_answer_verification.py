"""生成后的答案忠实度核验。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from rag.graphs.verification import AnswerVerifier, build_answer_verifier


def _verifier(score: float = 0.9) -> AnswerVerifier:
    async def faithfulness(question: str, answer: str, contexts: list[str]) -> float:
        assert question == "问题"
        assert answer == "答案"
        assert contexts == ["证据"]
        return score

    return AnswerVerifier(threshold=0.5, timeout_s=1, faithfulness=faithfulness)


def test_supported_answer_has_no_web_action() -> None:
    result = asyncio.run(_verifier(0.8).verify("问题", "答案", ["证据"]))

    assert result["status"] == "supported"
    assert result["score"] == 0.8
    assert result["can_search_web"] is False


def test_low_faithfulness_requests_web_permission() -> None:
    result = asyncio.run(_verifier(0.2).verify("问题", "答案", ["证据"]))

    assert result["status"] == "low"
    assert result["score"] == 0.2
    assert result["can_search_web"] is True
    assert "是否允许联网" in result["message"]


@pytest.mark.parametrize("retrieval_incomplete", [False, True])
@pytest.mark.parametrize("score", [0.2, 0.8])
def test_evidence_is_verified_even_when_retrieval_is_incomplete(
    retrieval_incomplete: bool, score: float
) -> None:
    result = asyncio.run(
        _verifier(score).verify("问题", "答案", ["证据"], retrieval_incomplete=retrieval_incomplete)
    )

    assert result["status"] == ("supported" if score >= 0.5 else "low")
    assert result["score"] == score
    assert result["can_search_web"] is (score < 0.5)


@pytest.mark.parametrize("retrieval_incomplete", [False, True])
@pytest.mark.parametrize("answer", ["答案", ""])
def test_missing_contexts_do_not_call_ragas(retrieval_incomplete: bool, answer: str) -> None:
    called = False

    async def faithfulness(question: str, answer: str, contexts: list[str]) -> float:
        nonlocal called
        called = True
        return 1.0

    verifier = AnswerVerifier(threshold=0.5, timeout_s=1, faithfulness=faithfulness)
    result = asyncio.run(
        verifier.verify("问题", answer, [], retrieval_incomplete=retrieval_incomplete)
    )

    assert not called
    assert result["status"] == "low"
    assert result["score"] is None
    assert result["can_search_web"] is True
    assert "是否允许联网搜索" in result["message"]
    if retrieval_incomplete:
        assert "检索不完整" in result["message"]
        assert "未能取得可核验资料" in result["message"]
        assert "没有检索到能够回答该问题的资料" not in result["message"]
    elif answer:
        assert "没有检索到" in result["message"]
    else:
        assert "没有生成可核验的答案" in result["message"]


def test_verification_failure_does_not_claim_support_or_low_quality() -> None:
    async def faithfulness(question: str, answer: str, contexts: list[str]) -> float:
        raise RuntimeError("judge down")

    verifier = AnswerVerifier(threshold=0.5, timeout_s=1, faithfulness=faithfulness)
    result = asyncio.run(verifier.verify("问题", "答案", ["证据"]))

    assert result["status"] == "unavailable"
    assert result["can_search_web"] is False


def test_disabled_verification_is_not_built() -> None:
    settings = SimpleNamespace(
        answer_verification=SimpleNamespace(enabled=False, threshold=0.5, timeout_s=1)
    )

    assert build_answer_verifier(settings, SimpleNamespace()) is None
