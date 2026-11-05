"""QueryGraph：plan → retrieve → grade →（不达标改写重查）→ assemble → generate。

全程用替身：检索、两个 LLM 调用、生成都是注入进来的 —— 图本身不该知道 Milvus 或哪家 chat。
"""

from __future__ import annotations

import asyncio
import json

import pytest

from rag.config.schema import Settings
from rag.contracts import (
    AssembledContext,
    ChannelStats,
    Chunk,
    RetrievalDegradation,
    RetrievalResult,
    ScoredChunk,
)
from rag.graphs.query import (
    MAX_SUB_QUERIES,
    build_query_graph,
    build_query_preflight,
    normalise_plan,
    parse_json_object,
    run_query_graph,
    run_query_preflight,
)


def _hit(chunk_id: str) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(chunk_id=chunk_id, doc_id="d1", text=f"正文 {chunk_id}"),
        score=0.5,
        channel="hybrid:milvus:c",
        rank=0,
    )


class _Retriever:
    """替身检索：按调用顺序吐预先排好的结果，并记下每次的参数。"""

    def __init__(self, rounds: list[list[str]]) -> None:
        self._rounds = rounds
        self.calls: list[dict] = []

    async def __call__(self, query, top_k, *, filters=None, channels=None):
        index = min(len(self.calls), len(self._rounds) - 1)
        self.calls.append(
            {"query": query, "top_k": top_k, "filters": filters, "channels": channels}
        )
        ids = self._rounds[index]
        stats = [ChannelStats(channel="hybrid:milvus:c", candidates=len(ids), kept=len(ids))]
        return RetrievalResult(
            query=query, chunks=[_hit(chunk_id) for chunk_id in ids], stats=stats
        )


class _Json:
    """替身 LLM：按次序返回预设的 JSON（可以是字符串也可以直接给 dict）。"""

    def __init__(self, *answers: object) -> None:
        self._answers = list(answers)
        self.prompts: list[str] = []

    async def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        answer = self._answers[min(len(self.prompts) - 1, len(self._answers) - 1)]
        return answer if isinstance(answer, str) else json.dumps(answer)


def _settings(**over) -> Settings:
    settings = Settings()
    for key, value in over.items():
        setattr(settings.graph, key, value)
    return settings


def _build(settings, retriever, planner, grader, answers: list[str]):
    async def generate(question: str, context: AssembledContext) -> str:
        answers.append(context.text)
        return "答案"

    return build_query_graph(
        settings,
        retrieve=retriever,
        plan_llm=planner,
        grade_llm=grader,
        generate=generate,
        channels=["milvus", "kg"],
    )


# ---- JSON 解析与收敛 ----（模型输出不可信，这两步是防线）


def test_json_is_taken_out_of_fences_and_surrounding_prose() -> None:
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('好的，这是计划：{"a": 1} 以上。') == {"a": 1}


def test_output_without_json_is_an_error() -> None:
    """解析不出来要报错：给个默认值就等于静默用错计划去检索。"""
    with pytest.raises(ValueError, match="JSON"):
        parse_json_object("我觉得应该先看看 Flink 那一章")


def test_plan_is_capped_and_unknown_channels_are_dropped() -> None:
    plan = normalise_plan(
        {
            "sub_queries": ["a", "b", "c", "d", "e"],
            "channels": ["milvus", "kg", "es"],
            "filters": {"modality": "image"},
        },
        available=["milvus", "kg"],
    )

    assert len(plan["sub_queries"]) == MAX_SUB_QUERIES
    assert plan["channels"] == ["milvus", "kg"], "编出来的通道名要滤掉，不然整次查询会挂"
    assert plan["filters"] == {"modality": "image"}


def test_a_plan_with_no_usable_channel_means_all_configured_ones() -> None:
    """写了个不认识的名字 → 收敛成**空**，语义是「走配置里启用的那几路」。

    以前这里是回退到写死的 `["hybrid"]` —— 通道进了配置之后就没有那个固定名字了。
    """
    plan = normalise_plan({"sub_queries": [], "channels": ["nope"]}, available=["milvus"])

    assert plan["channels"] == []
    assert plan["sub_queries"] == [""], "一条都没有时留给上层用原问题"
    assert plan["filters"] == {}


# ---- 整图 ----（不连库、不调线上）


def test_a_good_grade_goes_straight_to_generate() -> None:
    settings = _settings()
    retriever = _Retriever([["a", "b"]])
    planner = _Json({"sub_queries": ["问题改写"], "channels": ["hybrid"], "filters": {}})
    grader = _Json({"score": 0.9, "missing": ""})
    contexts: list[str] = []
    graph = _build(settings, retriever, planner, grader, contexts)

    result = asyncio.run(run_query_graph(graph, "原问题", top_k=5))

    assert result["answer"] == "答案"
    assert result["rounds"] == 1
    assert result["grade"]["score"] == 0.9
    assert [hit.chunk.chunk_id for hit in result["hits"]] == ["a", "b"]
    assert retriever.calls[0]["query"] == "问题改写", "检索用的是计划改写后的查询"
    assert len(planner.prompts) == 1 and len(grader.prompts) == 1
    assert "资料" in grader.prompts[0], "grade 拿到的 prompt 里要有候选内容"


def test_preflight_stops_after_assemble_without_generating() -> None:
    """流式生成前只要检索上下文；generate 不能在图里被偷偷调一次。"""
    settings = _settings()
    retriever = _Retriever([["a", "b"]])
    planner = _Json({"sub_queries": ["问题改写"]})
    grader = _Json({"score": 0.9, "missing": ""})
    graph = build_query_preflight(
        settings,
        retrieve=retriever,
        plan_llm=planner,
        grade_llm=grader,
        channels=["milvus"],
    )

    result = asyncio.run(run_query_preflight(graph, "原问题", top_k=5))

    assert result["rounds"] == 1
    assert result["grade"]["score"] == 0.9
    assert [hit.chunk.chunk_id for hit in result["hits"]] == ["a", "b"]
    assert "[1]" in result["context"].text
    assert result["context"].citations[0].chunk_id == "a"


def test_a_low_grade_rewrites_and_retries_with_the_missing_note() -> None:
    settings = _settings(max_corrective_rounds=1)
    retriever = _Retriever([["a"], ["b"]])
    planner = _Json({"sub_queries": ["第一轮"]}, {"sub_queries": ["第二轮"]})
    grader = _Json({"score": 0.1, "missing": "缺配置示例"}, {"score": 0.9, "missing": ""})
    contexts: list[str] = []
    graph = _build(settings, retriever, planner, grader, contexts)

    result = asyncio.run(run_query_graph(graph, "怎么配", top_k=5))

    assert result["rounds"] == 2
    assert len(planner.prompts) == 2
    assert "缺配置示例" in planner.prompts[1], "上一轮「缺什么」要喂给下一轮"
    assert retriever.calls[1]["query"] == "第二轮"
    assert contexts and "b" in contexts[0], "最终用的是第二轮召回到的资料"


def test_the_loop_stops_at_max_rounds_and_still_answers() -> None:
    """轮数用尽仍不达标：**照常生成**，把「可能不足」交给生成提示词去说。

    硬拒绝（直接回「不知道」）风险更大 —— 评分模型自己也可能判错，一次误判不该让
    本来能答的问题变成拒答。
    """
    settings = _settings(max_corrective_rounds=1)
    retriever = _Retriever([["a"], ["b"]])
    planner = _Json({"sub_queries": ["q"]})
    grader = _Json({"score": 0.1, "missing": "还是不够"})
    contexts: list[str] = []
    graph = _build(settings, retriever, planner, grader, contexts)

    result = asyncio.run(run_query_graph(graph, "问题", top_k=5))

    assert result["rounds"] == 2, "最多 max_corrective_rounds 次改写"
    assert result["answer"] == "答案"
    assert contexts


def test_zero_hits_skips_the_grade_call() -> None:
    """一条都没召回到就别问 grade 模型了 —— 它只会把「没有资料」再判一遍，白花一次调用。

    **但改写重查要照做**：零召回最常见的原因就是问法与语料用词对不上，换一种说法重来
    正是这条循环最值钱的地方。轮数用尽仍零召回时照样进生成 —— 那时上下文是空的，
    `CitationGenerator` 会回「没有检索到相关资料」，即老实拒答。
    """
    settings = _settings()
    retriever = _Retriever([[]])
    planner = _Json({"sub_queries": ["q"]})
    grader = _Json({"score": 1.0})
    contexts: list[str] = []
    graph = _build(settings, retriever, planner, grader, contexts)

    result = asyncio.run(run_query_graph(graph, "问题", top_k=5))

    assert grader.prompts == [], "没调 grade"
    assert result["grade"]["score"] == 0.0
    assert result["rounds"] == settings.graph.max_corrective_rounds + 1, "改写重查照做"
    assert len(planner.prompts) == result["rounds"]
    assert contexts == [""], "最后交给生成的是空上下文（真实实现会返回「没有检索到相关资料」）"


def test_multiple_sub_queries_are_fused_into_one_board() -> None:
    settings = _settings()
    retriever = _Retriever([["a", "b"], ["c"]])
    planner = _Json({"sub_queries": ["甲", "乙"]})
    grader = _Json({"score": 0.9})
    contexts: list[str] = []
    graph = _build(settings, retriever, planner, grader, contexts)

    result = asyncio.run(run_query_graph(graph, "宽问题", top_k=5))

    assert [call["query"] for call in retriever.calls] == ["甲", "乙"]
    assert {hit.chunk.chunk_id for hit in result["hits"]} == {"a", "b", "c"}
    assert all(hit.score_kind == "rrf" for hit in result["hits"])


@pytest.mark.parametrize("sub_queries", [["甲"], ["甲", "乙"]])
def test_query_graph_shared_wiring_preserves_result_and_refusion_semantics(
    monkeypatch, sub_queries
):
    from types import SimpleNamespace

    from rag.contracts import RetrievalResult
    from rag.graphs.wiring import build_retrieve
    from rag.service.retrieval import RetrievalService

    settings = _settings()
    heavy = SimpleNamespace(
        settings=settings, index=object(), embedder=object(), reranker=object(), chat=object()
    )
    seen = []

    async def retrieve(self, query, top_k, **kwargs):
        assert self._settings is settings
        seen.append((query, top_k, kwargs))
        hit = _hit(query).model_copy(update={"score_kind": "rerank"})
        return RetrievalResult(
            query=query,
            chunks=[hit],
            stats=[ChannelStats(channel="reranker", candidates=1, kept=1)],
            score_kind="rerank",
            rerank_status="succeeded",
        )

    monkeypatch.setattr(RetrievalService, "retrieve", retrieve)
    grader = _Json({"score": 0.9})
    graph = _build(
        settings,
        build_retrieve(heavy),
        _Json({"sub_queries": sub_queries, "channels": ["milvus"]}),
        grader,
        [],
    )
    result = asyncio.run(run_query_graph(graph, "问题", top_k=5))
    assert seen == [(query, 5, {"filters": None, "channels": ["milvus"]}) for query in sub_queries]
    expected_kind = "rerank" if len(sub_queries) == 1 else "rrf"
    assert all(hit.score_kind == expected_kind for hit in result["hits"])
    assert len(result["stats"]) == len(sub_queries)
    assert len(grader.prompts) == 1, "检索相关性不能替代 QueryGraph 的资料充分性评估"


@pytest.mark.parametrize("sub_queries", [["甲"], ["甲", "乙"]])
@pytest.mark.parametrize("failed_rerank", [False, True])
def test_query_graph_keeps_degradations_from_every_subquery(
    monkeypatch, sub_queries, failed_rerank
):
    from types import SimpleNamespace

    from rag.graphs.wiring import build_retrieve
    from rag.service.retrieval import RetrievalService

    settings = _settings(max_corrective_rounds=0)
    heavy = SimpleNamespace(settings=settings, index=None, embedder=None, reranker=None, chat=None)
    expected_degradations = []

    async def retrieve(self, query, top_k, **kwargs):
        degradation = RetrievalDegradation(stage=query, code="channel_failed", message="内部故障")
        expected_degradations.append(degradation)
        return RetrievalResult(
            query=query,
            chunks=[_hit(query)],
            score_kind="rerank",
            rerank_status="failed" if failed_rerank and query == "甲" else "succeeded",
            degradations=[degradation],
        )

    monkeypatch.setattr(RetrievalService, "retrieve", retrieve)
    grader = _Json({"score": 0.9})
    graph = build_query_preflight(
        settings,
        retrieve=build_retrieve(heavy),
        plan_llm=_Json({"sub_queries": sub_queries}),
        grade_llm=grader,
    )
    result = asyncio.run(run_query_preflight(graph, "问题"))
    retrieval = result["retrieval"]
    context = result["context"]
    assert retrieval.score_kind == ("rrf" if len(sub_queries) > 1 else "rerank")
    assert retrieval.rerank_status == ("failed" if failed_rerank else "succeeded")
    assert retrieval.degradations == expected_degradations
    assert context.degradations == expected_degradations
    assert context.score_kind == retrieval.score_kind
    assert context.rerank_status == retrieval.rerank_status
    assert "部分检索通道失败" in grader.prompts[0]
    assert "不能断言语料中没有相关内容" in grader.prompts[0]


def test_the_plan_filters_and_channels_reach_the_retriever() -> None:
    settings = _settings()
    retriever = _Retriever([["a"]])
    planner = _Json(
        {"sub_queries": ["图里的数据源"], "channels": ["milvus"], "filters": {"modality": "image"}}
    )
    grader = _Json({"score": 0.9})
    graph = _build(settings, retriever, planner, grader, [])

    asyncio.run(run_query_graph(graph, "图里画了啥", top_k=5))

    assert retriever.calls[0]["filters"] == {"modality": "image"}
    assert retriever.calls[0]["channels"] == ["milvus"]
