"""统一答案引擎：直线、Graph 预检和非流式 / 流式生成共用一套上下文。"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from rag.config.schema import Settings
from rag.contracts import (
    AssembledContext,
    Chunk,
    RetrievalDegradation,
    RetrievalResult,
    ScoredChunk,
)
from rag.graphs.answer import AnswerEngine
from rag.graphs.verification import AnswerVerifier


def _hit(chunk_id: str = "d1:0001") -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id="d1",
            text="Savepoint 是作业状态的一致快照。",
            metadata={"source_uri": "flink.md"},
        ),
        score=0.9,
        channel="rrf",
        rank=0,
    )


class _Retriever:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @property
    def last_stats(self) -> list:
        return []

    async def retrieve(self, query, top_k, filters=None, image=None):
        self.calls.append({"query": query, "top_k": top_k, "filters": filters})
        return [_hit()]


class _Chat:
    def __init__(self, *responses: dict[str, Any]) -> None:
        self.responses = list(responses)
        self.calls = 0

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.calls += 1
        payload = self.responses.pop(0) if self.responses else {"score": 1.0}
        import json

        return AIMessage(
            content=json.dumps(payload),
            usage_metadata={"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
        )


class _Generator:
    def __init__(self) -> None:
        self.contexts: list[AssembledContext] = []

    async def generate(
        self,
        query: str,
        context: AssembledContext,
        *,
        usage: dict[str, Any] | None = None,
    ) -> str:
        self.contexts.append(context)
        if usage is not None:
            usage["calls"] = int(usage.get("calls") or 0) + 1
            usage["input_tokens"] = int(usage.get("input_tokens") or 0) + 4
            usage["output_tokens"] = int(usage.get("output_tokens") or 0) + 3
            usage["total_tokens"] = int(usage.get("total_tokens") or 0) + 7
        return "答案 [1]"

    async def astream(
        self,
        query: str,
        context: AssembledContext,
        *,
        usage: dict[str, Any] | None = None,
    ):
        self.contexts.append(context)
        for piece in ("答", "案 [1]"):
            yield piece
        if usage is not None:
            usage["calls"] = int(usage.get("calls") or 0) + 1
            usage["input_tokens"] = int(usage.get("input_tokens") or 0) + 4
            usage["output_tokens"] = int(usage.get("output_tokens") or 0) + 3
            usage["total_tokens"] = int(usage.get("total_tokens") or 0) + 7


class _Heavy:
    def __init__(self, chat: _Chat) -> None:
        self.index = object()
        self.embedder = object()
        self.reranker = None
        self.chat = chat
        self.generator = _Generator()


def _settings(*, graph: bool) -> Settings:
    settings = Settings()
    settings.graph.enabled = graph
    settings.graph.max_corrective_rounds = 0
    return settings


def _patch_retriever(monkeypatch: pytest.MonkeyPatch, retriever: _Retriever) -> None:
    import rag.service.retrieval

    monkeypatch.setattr(rag.service.retrieval, "build_retriever", lambda *a, **k: retriever)


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("has_hits", [False, True])
@pytest.mark.parametrize("code", [None, "channel_failed", "rerank_failed"])
@pytest.mark.parametrize("judge_score", [0.2, 0.8])
def test_retrieval_status_reaches_generation_and_verification(
    monkeypatch, graph, has_hits, code, judge_score
):
    from rag.service.retrieval import RetrievalService

    retrieval = RetrievalResult(
        query="q",
        chunks=[_hit()] if has_hits else [],
        score_kind="rrf" if code == "rerank_failed" else "rerank",
        rerank_status="failed" if code == "rerank_failed" else "succeeded",
        degradations=(
            [RetrievalDegradation(stage="b", code=code, message="内部异常不应进提示词")]
            if code
            else []
        ),
    )

    async def retrieve(self, query, top_k, **kwargs):
        return retrieval

    monkeypatch.setattr(RetrievalService, "retrieve", retrieve)
    checked = []

    async def faithfulness(question, answer, contexts):
        checked.append(contexts)
        # Ragas 只能看到引用的语料，不能把运行提示或异常当作证据。
        assert contexts == [_hit().chunk.text]
        return judge_score

    heavy = _Heavy(_Chat())
    engine = AnswerEngine(
        _settings(graph=graph),
        object(),
        heavy,
        verifier=AnswerVerifier(threshold=0.5, timeout_s=1, faithfulness=faithfulness),
    )
    answer = asyncio.run(engine.answer("q", 5))

    async def collect():
        return [event async for event in engine.stream("q", 5)]

    events = asyncio.run(collect())
    assert len(checked) == (2 if has_hits else 0)
    assert answer.verification is not None
    supported = has_hits and judge_score >= 0.5
    assert answer.verification["status"] == ("supported" if supported else "low")
    assert answer.verification["score"] == (judge_score if has_hits else None)
    assert answer.verification["can_search_web"] is not supported
    assert set(answer.verification) == {"status", "score", "threshold", "message", "can_search_web"}
    assert "内部异常" not in answer.verification["message"]
    if not has_hits:
        if code == "channel_failed":
            assert "检索不完整" in answer.verification["message"]
            assert "未能取得可核验资料" in answer.verification["message"]
        else:
            assert "没有检索到" in answer.verification["message"]
        assert "是否允许联网搜索" in answer.verification["message"]
    assert events[-2] == {"type": "verification", "verification": answer.verification}
    assert events[-1]["verification"] == answer.verification
    assert answer.score_kind == retrieval.score_kind
    assert answer.rerank_status == retrieval.rerank_status
    assert answer.degradations == retrieval.degradations
    for context in heavy.generator.contexts:
        assert context.score_kind == retrieval.score_kind
        assert context.rerank_status == retrieval.rerank_status
        assert context.degradations == retrieval.degradations
        notice = context.retrieval_notice()
        assert "内部异常" not in notice
        if code:
            assert "非语料证据，不可引用" in notice
            assert f"score_kind={retrieval.score_kind}" in notice
            if code == "channel_failed":
                assert "部分检索通道失败" in notice
                assert "不能断言语料中没有相关内容" in notice
            else:
                assert "重排失败" in notice
        else:
            assert not notice
    # SSE 保持原字段集，不把内部状态加入 retrieval/done 事件。
    assert set(events[0]) == {"type", "hits", "stats", "rounds", "grade", "mode"}
    assert set(events[-1]) == {
        "type",
        "answer",
        "citations",
        "rounds",
        "grade",
        "mode",
        "usage",
        "verification",
    }


def test_plain_answer_and_stream_use_the_same_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = _Retriever()
    _patch_retriever(monkeypatch, retriever)
    engine = AnswerEngine(_settings(graph=False), object(), _Heavy(_Chat()))

    answer = asyncio.run(engine.answer("Savepoint 是什么", 5))

    async def collect() -> list[dict[str, Any]]:
        return [event async for event in engine.stream("Savepoint 是什么", 5)]

    events = asyncio.run(collect())

    assert answer.mode == "plain"
    assert answer.answer == "答案 [1]"
    assert events[0]["type"] == "retrieval" and events[0]["mode"] == "plain"
    assert [event["text"] for event in events if event["type"] == "token"] == ["答", "案 [1]"]
    assert events[-1]["type"] == "done"
    assert events[-1]["citations"][0].chunk_id == "d1:0001"
    assert retriever.calls == [
        {"query": "Savepoint 是什么", "top_k": 5, "filters": None},
        {"query": "Savepoint 是什么", "top_k": 5, "filters": None},
    ]


def test_stream_gets_graph_preflight_before_its_first_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = _Retriever()
    _patch_retriever(monkeypatch, retriever)
    chat = _Chat(
        {"sub_queries": ["Flink Savepoint"], "filters": {}, "channels": []},
        {"score": 0.9, "missing": ""},
    )
    engine = AnswerEngine(_settings(graph=True), object(), _Heavy(chat))

    async def collect() -> list[dict[str, Any]]:
        return [event async for event in engine.stream("Savepoint 是什么", 5)]

    events = asyncio.run(collect())

    assert chat.calls == 2, "流式之前先完成 plan + grade"
    assert retriever.calls[0]["query"] == "Flink Savepoint"
    assert events[0]["type"] == "retrieval"
    assert events[0]["stats"] == []
    assert events[0]["rounds"] == 1
    assert events[0]["grade"] == {"score": 0.9, "missing": ""}
    assert events[0]["mode"] == "graph"
    assert events[-1]["type"] == "done"
    assert events[-1]["mode"] == "graph"
    assert events[-1]["usage"] == {
        "calls": 3,
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
    }


def test_answer_and_stream_emit_low_faithfulness_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = _Retriever()
    _patch_retriever(monkeypatch, retriever)

    async def faithfulness(question: str, answer: str, contexts: list[str]) -> float:
        assert question == "Savepoint 是什么"
        assert answer == "答案 [1]"
        assert contexts == ["Savepoint 是作业状态的一致快照。"]
        return 0.2

    verifier = AnswerVerifier(threshold=0.5, timeout_s=1, faithfulness=faithfulness)
    engine = AnswerEngine(
        _settings(graph=False),
        object(),
        _Heavy(_Chat()),
        verifier=verifier,
    )

    answer = asyncio.run(engine.answer("Savepoint 是什么", 5))

    async def collect() -> list[dict[str, Any]]:
        return [event async for event in engine.stream("Savepoint 是什么", 5)]

    events = asyncio.run(collect())
    verification = [event for event in events if event["type"] == "verification"]

    assert answer.verification is not None
    assert answer.verification["status"] == "low"
    assert answer.verification["can_search_web"] is True
    assert len(verification) == 1
    assert verification[0]["verification"]["score"] == 0.2
    assert events[-1]["type"] == "done"
    assert events[-1]["verification"]["status"] == "low"
