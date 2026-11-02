"""生成后的答案忠实度核验。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

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


def test_missing_contexts_do_not_call_ragas() -> None:
    called = False

    async def faithfulness(question: str, answer: str, contexts: list[str]) -> float:
        nonlocal called
        called = True
        return 1.0

    verifier = AnswerVerifier(threshold=0.5, timeout_s=1, faithfulness=faithfulness)
    result = asyncio.run(verifier.verify("问题", "答案", []))

    assert not called
    assert result["status"] == "low"
    assert result["can_search_web"] is True
    assert "没有检索到" in result["message"]


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
