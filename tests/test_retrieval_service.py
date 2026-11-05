"""公共接线：参数保真、旧插件兼容及每次调用的状态隔离。"""

import asyncio
from pathlib import Path

import pytest

from rag.config.schema import ChannelConfig, Settings
from rag.contracts import Chunk, RetrievalDegradation, RetrievalResult, ScoredChunk
from rag.service.retrieval import (
    RetrievalService,
    UnknownChannelError,
    assess_relevance,
    resolve_channels,
)


def _hit(kind="unknown", score=0.01):
    return ScoredChunk(
        chunk=Chunk(chunk_id="a", doc_id="d", text="正文"),
        channel="legacy",
        rank=0,
        score=score,
        score_kind=kind,
    )


@pytest.mark.parametrize(
    ("channels", "expected"),
    [(None, {"a"}), ([], {"a"}), (["b"], {"b"}), (["b", "a", "b"], {"a", "b"})],
)
def test_resolve_channels_preserves_defaults_and_explicit_selection(channels, expected):
    settings = Settings()
    settings.retrieval.channels = [
        ChannelConfig(name="a", type="milvus"),
        ChannelConfig(name="b", type="milvus", enabled=False),
    ]
    assert resolve_channels(settings, channels) == expected
    assert [item.enabled for item in settings.retrieval.channels] == [True, False]


@pytest.mark.parametrize("channels", [["missing"], ["a", "missing"]])
def test_service_rejects_any_unknown_channel_before_building(monkeypatch, channels):
    import rag.service.retrieval as module

    def unexpected_build(*args, **kwargs):
        raise AssertionError("未知通道必须在装配前拒绝")

    monkeypatch.setattr(module, "build_retriever", unexpected_build)
    settings = Settings()
    settings.retrieval.channels = [ChannelConfig(name="a", type="milvus")]
    before = settings.model_dump()
    with pytest.raises(UnknownChannelError, match="未知的通道.*missing"):
        asyncio.run(
            RetrievalService(settings, None, None, None, None).retrieve("q", 2, channels=channels)
        )
    assert settings.model_dump() == before


@pytest.mark.parametrize("kind", ["unknown", "rrf", "weighted", "dense", "rerank"])
@pytest.mark.parametrize("status", ["unknown", "disabled", "skipped", "failed", "succeeded"])
@pytest.mark.parametrize("score", [0.1, 0.2, 0.8])
def test_relevance_truth_table(kind, status, score):
    result = RetrievalResult(
        query="q", chunks=[_hit(kind, score)], score_kind=kind, rerank_status=status
    )
    expected = "unknown"
    if kind == "rerank" and status == "succeeded":
        expected = "relevant" if score >= 0.2 else "irrelevant"
    assert assess_relevance(result, 0.2) == expected
    assert assess_relevance(result.model_copy(update={"chunks": []}), 0.2) == "irrelevant"


@pytest.mark.parametrize(
    ("code", "expected"),
    [("channel_failed", "unknown"), ("rerank_failed", "irrelevant")],
)
def test_empty_relevance_uses_channel_failure_code(code, expected):
    result = RetrievalResult(
        query="q",
        degradations=[RetrievalDegradation(stage="any", code=code, message="任意描述")],
    )
    assert assess_relevance(result, 0.2) == expected


@pytest.mark.parametrize("score", [0.1, 0.8])
@pytest.mark.parametrize("failed", [False, True])
def test_partial_channel_failure_makes_hit_relevance_unknown(score, failed):
    result = RetrievalResult(
        query="q",
        chunks=[_hit("rerank", score)],
        score_kind="rerank",
        rerank_status="succeeded",
        degradations=(
            [RetrievalDegradation(stage="b", code="channel_failed", message="故障")]
            if failed
            else []
        ),
    )
    expected = "unknown" if failed else ("relevant" if score >= 0.2 else "irrelevant")
    assert assess_relevance(result, 0.2) == expected


def test_relevance_uses_highest_score_not_first():
    result = RetrievalResult(
        query="q",
        chunks=[_hit("rerank", 0.1), _hit("rerank", 0.8)],
        score_kind="rerank",
        rerank_status="succeeded",
    )
    assert assess_relevance(result, 0.2) == "relevant"


def test_service_isolates_config_and_prefers_atomic_result(monkeypatch):
    import rag.service.retrieval as module

    settings = Settings()
    settings.retrieval.channels = [
        ChannelConfig(name="a", type="milvus"),
        ChannelConfig(name="b", type="milvus"),
    ]
    components = [object() for _ in range(4)]
    seen = []
    summaries = []

    class Atomic:
        async def retrieve_result(self, query, top_k, **kwargs):
            await asyncio.sleep(0)
            return RetrievalResult(
                query=query,
                chunks=[_hit()],
                rerank_status="failed" if query == "a" else "succeeded",
            )

        async def retrieve(self, *args, **kwargs):
            raise AssertionError("不应使用旧投影")

    def build(local, index, embedder, reranker, *, hyde):
        seen.append((local, index, embedder, reranker, hyde))
        return Atomic()

    monkeypatch.setattr(module, "build_retriever", build)
    monkeypatch.setattr(module, "configured_hyde", lambda local, chat: chat)
    monkeypatch.setattr(module, "record_retrieval", lambda result, **kw: summaries.append(result))
    service = RetrievalService(settings, *components)

    async def run():
        return await asyncio.gather(
            service.retrieve("a", 3, channels=["a"], rerank=False),
            service.retrieve("b", 4, channels=["b"]),
        )

    a, b = asyncio.run(run())
    assert (a.query, a.rerank_status, b.query, b.rerank_status) == ("a", "failed", "b", "succeeded")
    assert len(summaries) == 2
    assert [c.enabled for c in seen[0][0].retrieval.channels] == [True, False]
    assert [c.enabled for c in seen[1][0].retrieval.channels] == [False, True]
    assert not seen[0][0].retrieval.rerank.enabled
    assert settings.retrieval.rerank.enabled
    assert all(c.enabled for c in settings.retrieval.channels)
    assert list(seen[0][1:]) == components


@pytest.mark.parametrize("kinds", [("dense", "rrf"), ("rerank", "unknown")])
def test_mixed_legacy_scores_are_unknown(monkeypatch, kinds):
    import rag.service.retrieval as module

    class Legacy:
        async def retrieve(self, *args, **kwargs):
            return [_hit(kind) for kind in kinds]

    monkeypatch.setattr(module, "build_retriever", lambda *a, **kw: Legacy())
    result = asyncio.run(RetrievalService(Settings(), None, None, None, None).retrieve("q", 2))
    assert result.score_kind == "unknown" and result.rerank_status == "unknown"
    assert assess_relevance(result, 0.2) == "unknown"


def test_legacy_plugin_parameters_and_errors(monkeypatch):
    import rag.service.retrieval as module

    seen = []

    class Legacy:
        @property
        def last_stats(self):
            raise AssertionError("不可读取共享状态")

        async def retrieve(self, query, top_k, **kwargs):
            seen.append((query, top_k, kwargs))
            if query == "error":
                raise RuntimeError("原异常")
            return [_hit()]

    monkeypatch.setattr(module, "build_retriever", lambda *a, **kw: Legacy())
    settings = Settings()
    settings.retrieval.rerank.enabled = False
    service = RetrievalService(settings, object(), object(), None, None)
    image = Path("q.png")
    result = asyncio.run(service.retrieve("q", 3, filters={"x": 1}, image=image))
    assert result.score_kind == "unknown"
    assert result.rerank_status == "disabled"
    assert seen == [("q", 3, {"filters": {"x": 1}, "image": image})]
    with pytest.raises(RuntimeError, match="原异常"):
        asyncio.run(service.retrieve("error", 3))


@pytest.mark.parametrize("channel_count", [1, 2])
def test_explicit_rerank_false_ignores_injected_reranker(monkeypatch, channel_count):
    import rag.retrieval as wiring

    class Channel:
        name = "legacy"

        async def retrieve(self, query, top_k, **kwargs):
            return [_hit()]

    class Reranker:
        name = "must-not-run"

        async def rerank(self, *args, **kwargs):
            raise AssertionError("显式关闭时不可运行重排")

    settings = Settings()
    settings.retrieval.channels = [
        ChannelConfig(name=f"c{i}", type="milvus") for i in range(channel_count)
    ]
    monkeypatch.setattr(wiring, "build_channel", lambda *a: Channel())
    result = asyncio.run(
        RetrievalService(settings, None, None, Reranker(), None).retrieve("q", 3, rerank=False)
    )
    assert result.chunks
    assert result.rerank_status == "disabled"
    assert result.score_kind != "rerank"
    assert settings.retrieval.rerank.enabled


@pytest.mark.parametrize("mode", ["ok", "error", "empty", "disabled", "channel_error"])
def test_service_real_wiring_preserves_failure_semantics_and_one_summary(monkeypatch, mode):
    import rag.retrieval as wiring
    import rag.service.retrieval as module
    from rag.retrieval.multichannel import RetrievalError

    calls = []
    summaries = []

    class Channel:
        name = "legacy"

        async def retrieve(self, query, top_k, **kwargs):
            if mode == "channel_error":
                raise TimeoutError("注入通道故障")
            return [_hit()]

    class Reranker:
        name = "reranker"

        async def rerank(self, query, hits, top_n):
            calls.append(query)
            if mode == "error":
                raise RuntimeError("注入重排故障")
            return (
                [] if mode == "empty" else [hit.model_copy(update={"score": 0.8}) for hit in hits]
            )

    settings = Settings()
    settings.retrieval.channels = [
        ChannelConfig(name="a", type="milvus"),
        ChannelConfig(name="b", type="milvus"),
    ]
    monkeypatch.setattr(wiring, "build_channel", lambda *a: Channel())
    monkeypatch.setattr(
        module, "record_retrieval", lambda result, **kw: summaries.append((result, kw))
    )
    service = RetrievalService(settings, None, None, Reranker(), None)
    if mode == "channel_error":
        with pytest.raises(RetrievalError, match="全部失败"):
            asyncio.run(service.retrieve("q", 3))
        assert len(summaries) == 1
        assert summaries[0][1] == {"error": "retrieval_failed"}
        assert not calls
        return

    result = asyncio.run(service.retrieve("q", 3, rerank=mode != "disabled"))
    assert len(summaries) == 1
    assert summaries[0][0] is result
    assert len(calls) == (0 if mode == "disabled" else 1)
    assert settings.retrieval.rerank.enabled
    assert result.score_kind == ("rerank" if mode == "ok" else "rrf")
    assert result.rerank_status == (
        "succeeded" if mode == "ok" else "disabled" if mode == "disabled" else "failed"
    )
    assert assess_relevance(result, 0.2) == ("relevant" if mode == "ok" else "unknown")
