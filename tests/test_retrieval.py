"""检索层：上下文装配、过滤表达式、融合策略选择。

装配那几件事（去重、预算、编号）都在 assembler 里一次算完，
所以这里重点测它们两两之间会不会互相破坏。
"""

from __future__ import annotations

import pytest
from pymilvus import RRFRanker, WeightedRanker

from rag.config.schema import (
    ChannelConfig,
    ChannelLegConfig,
    FusionConfig,
    HybridChannelConfig,
    Settings,
)
from rag.contracts import EMBEDDERS, Chunk, Embeddings, ScoredChunk
from rag.indexing import build_index
from rag.retrieval import (
    MilvusHybridRetriever,
    MultiChannelRetriever,
    assemble,
    build_retriever,
    estimate_tokens,
)
from rag.retrieval.channels import build_ranker, filter_expr


def _hit(
    chunk_id: str,
    text: str,
    *,
    uri: str = "a.md",
    score: float = 1.0,
    channel: str = "hybrid",
    rank: int = 0,
) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id="d1",
            text=text,
            metadata={"source_uri": uri},
        ),
        score=score,
        channel=channel,
        rank=rank,
    )


# ---- token 估算 ----


def test_estimate_tokens_counts_cjk_by_char_latin_by_four() -> None:
    """中文 1 字算 1 token，拉丁 4 字符算 1 token。"""
    assert estimate_tokens("检索增强生成") == 7  # 6 字 + max(1, 0)
    assert estimate_tokens("abcdefgh") == 2  # max(1, 8 // 4)


def test_estimate_tokens_never_underestimates() -> None:
    """宁可估高：估低了会超模型窗口被静默截断，估高只是少放几条资料。"""
    for text in ["检索增强生成", "abcdefgh", "混合 检索 hybrid", ""]:
        assert estimate_tokens(text) >= len(text) // 4


def test_estimate_tokens_treats_fullwidth_punctuation_as_cjk() -> None:
    """全角标点是中文语境的一部分，算成拉丁会低估。"""
    assert estimate_tokens("，。") == 3


# ---- 装配 ----


def test_assemble_dedupes_and_keeps_the_best_ranked_occurrence() -> None:
    """同一块被多路召回是 hybrid 的常态，只保留排名最靠前的那次。"""
    hits = [
        _hit("c1", "先到", score=0.9, channel="dense", rank=0),
        _hit("c2", "中间", score=0.5, channel="sparse", rank=1),
        _hit("c1", "后到", score=0.1, channel="hybrid", rank=2),
    ]
    ctx = assemble(hits, max_tokens=10_000)

    assert [c.chunk_id for c in ctx.citations] == ["c1", "c2"]
    assert ctx.dropped == 1
    assert "先到" in ctx.text
    assert "后到" not in ctx.text


def test_assemble_numbers_citations_from_one_and_renders_source() -> None:
    """编号和正文出自同一次装配，模型引用的 [n] 才对得上引用表。"""
    ctx = assemble([_hit("c1", "甲", uri="docs/a.md"), _hit("c2", "乙", uri="docs/b.md")], 10_000)

    assert [c.index for c in ctx.citations] == [1, 2]
    assert ctx.citations[0].source_uri == "docs/a.md"
    assert ctx.text.startswith("[1] 来源：docs/a.md\n甲")
    assert "[2] 来源：docs/b.md" in ctx.text


def test_assemble_skips_over_budget_item_and_keeps_going() -> None:
    """一条长块不该让后面放得下的块一起被浪费。"""
    hits = [_hit("long", "长" * 400), _hit("short", "短")]
    ctx = assemble(hits, max_tokens=20)

    assert [c.chunk_id for c in ctx.citations] == ["short"]
    assert ctx.dropped == 1


def test_assemble_reports_budget_it_actually_used() -> None:
    ctx = assemble([_hit("c1", "甲"), _hit("c2", "乙")], max_tokens=10_000)

    assert 0 < ctx.tokens_estimate < 10_000
    assert ctx.tokens_estimate <= 10_000


def test_assemble_of_nothing_is_empty_not_an_error() -> None:
    ctx = assemble([], 1000)

    assert ctx.text == ""
    assert ctx.citations == []
    assert ctx.dropped == 0


def test_assemble_snippet_is_single_line_and_bounded() -> None:
    """摘要是给引用表展示的，不能带换行把表格撑破。"""
    ctx = assemble([_hit("c1", "第一行\n第二行\n" + "填" * 300)], 10_000)

    snippet = ctx.citations[0].snippet
    assert "\n" not in snippet
    assert len(snippet) <= 80


# ---- 过滤表达式 ----


def test_filter_expr_of_nothing_is_none() -> None:
    """空过滤要返回 None 而不是空串：调用方靠它决定要不要带 filter 参数。"""
    assert filter_expr(None) is None
    assert filter_expr({}) is None


def test_filter_expr_joins_multiple_fields_with_and() -> None:
    expr = filter_expr({"h1": "检索", "page_count": 3})

    assert expr == 'h1 == "检索" and page_count == 3'


def test_filter_expr_escapes_quotes_and_backslashes() -> None:
    """过滤值来自调用方，不转义就等于把注入面交给上层。"""
    assert filter_expr({"h1": 'a"b'}) == 'h1 == "a\\"b"'
    assert filter_expr({"h1": "a\\b"}) == 'h1 == "a\\\\b"'


def test_filter_expr_renders_bool_before_int() -> None:
    """bool 是 int 的子类，判序反了会写成 1 / 0，Milvus 不认。"""
    assert filter_expr({"flag": True}) == "flag == true"
    assert filter_expr({"flag": False}) == "flag == false"


def test_filter_expr_supports_string_lists_as_in() -> None:
    assert filter_expr({"tag": ["a", "b"]}) == 'tag in ["a", "b"]'


def test_filter_expr_supports_like_for_heading_paths() -> None:
    """层级检索：`heading_path` 是一整条路径，等值匹配要求一字不差，而问法千变万化 ——
    所以给一个单键字典走模糊匹配（通配符由调用方给）。"""
    assert filter_expr({"heading_path": {"like": "%2.1%"}}) == 'heading_path like "%2.1%"'
    assert filter_expr({"heading_path": {"like": '%a"b%'}}) == 'heading_path like "%a\\"b%"'


@pytest.mark.parametrize("value", [None, {"k": 1}, {"like": "x", "gt": 1}, [1, 2], 3.5j])
def test_filter_expr_rejects_types_it_cannot_safely_render(value: object) -> None:
    """多键字典也拒：那说明调用方在表达更复杂的东西，猜它的意思不如报错。"""
    with pytest.raises(TypeError, match="不支持的过滤值"):
        filter_expr({"field": value})


# ---- 融合策略 ----


def _hybrid_cfg(
    strategy: str = "rrf",
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    k: int = 60,
) -> HybridChannelConfig:
    return HybridChannelConfig(
        dense=ChannelLegConfig(field="dense", weight=dense_weight),
        sparse=ChannelLegConfig(field="sparse", weight=sparse_weight),
        fusion=FusionConfig(strategy=strategy, k=k),
    )


def test_build_ranker_defaults_to_rrf() -> None:
    ranker = build_ranker(_hybrid_cfg())

    assert isinstance(ranker, RRFRanker)


def test_build_ranker_refuses_rrf_with_weights() -> None:
    """RRFRanker 只有 k、不支持按腿加权，静默忽略权重比报错更坏。"""
    with pytest.raises(ValueError, match="weighted"):
        build_ranker(_hybrid_cfg(dense_weight=2.0))


def test_build_ranker_weighted_passes_weights_through() -> None:
    ranker = build_ranker(_hybrid_cfg(strategy="weighted", dense_weight=2.0, sparse_weight=1.0))

    assert isinstance(ranker, WeightedRanker)


def test_build_ranker_rejects_unknown_strategy() -> None:
    with pytest.raises(KeyError, match="未知的融合策略"):
        build_ranker(_hybrid_cfg(strategy="rrf-plus"))


# ---- 装配 ----


class _StubEmbedder:
    """只为走通装配路径，向量本身在检索层里是常量。"""

    name = "stub"
    dim = 512

    async def embed_documents(self, texts: list[str]) -> Embeddings:
        return Embeddings(dense=[[0.0] * self.dim for _ in texts])

    async def embed_query(self, text: str) -> Embeddings:
        return Embeddings(dense=[[0.0] * self.dim])


def test_the_default_config_is_one_milvus_channel() -> None:
    """默认配置只有一路：Milvus（融合向量 + 内建 BM25）。单路时直接返回它，不包一层。"""
    settings = Settings()
    # 重排默认开着就得给 reranker，跟本用例无关，先关掉
    settings.retrieval.rerank.enabled = False
    retriever = build_retriever(settings, build_index(settings), _StubEmbedder())

    assert isinstance(retriever, MilvusHybridRetriever)
    assert retriever.name == "milvus", "通道名 = 配置里的 name（日志/账本/--channel 都用它）"


def test_an_unknown_channel_type_is_rejected() -> None:
    """`type` 找不到适配器就当场报错（插件没装 / 名字写错）—— 静默少一路最难查。"""
    settings = Settings()
    settings.retrieval.channels = [ChannelConfig(name="es", type="elasticsearch")]
    settings.retrieval.rerank.enabled = False

    with pytest.raises(KeyError, match="未知的通道类型"):
        build_retriever(settings, build_index(settings), _StubEmbedder())


def test_two_configured_channels_are_fused() -> None:
    """配置里写两路就 L2 融合成一张榜（名字就是配置里的 name）。

    这里**用一个测试自己的通道类型**，不依赖任何插件包 —— 图谱那一路已经搬成插件了
    （`plugins/rag-channel-neo4j`），核心测试不该因为它没装而红。
    """
    from rag.contracts import CHANNEL_TYPES
    from rag.retrieval.plugins import resolve_channel_type

    class _StubChannel:
        complement_only = True

        def __init__(self) -> None:
            self.name = "stub"

        async def retrieve(self, query, top_k, filters=None, image=None):  # noqa: ANN001, ANN202
            return []

    CHANNEL_TYPES.register("stub-in-tests")(lambda **_: _StubChannel())
    resolve_channel_type.cache_clear()

    settings = Settings()
    settings.retrieval.channels = [
        ChannelConfig(name="milvus", type="milvus"),
        ChannelConfig(name="kg", type="stub-in-tests"),
    ]
    settings.retrieval.rerank.enabled = False

    try:
        retriever = build_retriever(settings, build_index(settings), _StubEmbedder())
    finally:
        resolve_channel_type.cache_clear()

    assert isinstance(retriever, MultiChannelRetriever)
    assert retriever.name == "multi[milvus,kg]"


def test_keyword_is_not_a_channel_any_more() -> None:
    """**关键词不再是一个通道**（2026-10-08 定）：它就是 Milvus 那一路的 sparse 腿
    （dense + 内建 BM25 在服务端融合）。

    判据是注册表：`bm25` 必须不在里面 —— 留着它，配置里就能再写出一路"同一个库的
    另一种搜法"，那正是这条要禁掉的形状（实测关掉它 recall@8 一样、MRR 反而略好）。
    """
    import rag.retrieval  # noqa: F401  触发通道注册
    from rag.contracts import CHANNEL_TYPES

    assert "bm25" not in CHANNEL_TYPES.names()
    # 内核只剩 `milvus` 一路：图谱（neo4j）2026-10-08 搬成了插件包，
    # 想用它得装 `plugins/rag-channel-neo4j`（配置里写 type: neo4j）。
    assert "milvus" in CHANNEL_TYPES.names()
    assert "neo4j" not in CHANNEL_TYPES.names()


def test_an_unknown_optional_channel_is_rejected() -> None:
    """配了却不生效是最难查的 —— 所以 `type` 写错要当场报（插件化解析会列出有什么）。"""
    settings = Settings()
    # web 是「文档里写了、还没落地」的那一路：它必须被拒，而不是静默当作没有
    settings.retrieval.channels = [ChannelConfig(name="web", type="web")]
    settings.retrieval.rerank.enabled = False

    with pytest.raises(KeyError, match="未知的通道类型"):
        build_retriever(settings, build_index(settings), _StubEmbedder())


class _StubReranker:
    name = "rerank:stub"

    async def rerank(self, query: str, chunks: list, top_n: int):
        return chunks[:top_n]


def test_rerank_enabled_without_a_reranker_is_rejected() -> None:
    """开了重排却没给重排器 = 配置与效果不符 —— 当场报，不静默跳过。"""
    settings = Settings()
    settings.retrieval.rerank.enabled = True

    with pytest.raises(ValueError, match="没有传 reranker"):
        build_retriever(settings, build_index(settings), _StubEmbedder())


def test_a_reranker_alone_also_wraps_into_one_retriever() -> None:
    """只有一路、但要重排时也要包一层 —— 重排得有地方挂（名字用配置里的通道名）。"""
    settings = Settings()
    settings.retrieval.rerank.enabled = True

    retriever = build_retriever(settings, build_index(settings), _StubEmbedder(), _StubReranker())

    assert isinstance(retriever, MultiChannelRetriever)
    assert retriever.name == "multi[milvus,rerank:stub]"


def test_embedder_registry_is_populated_by_providers_import() -> None:
    """providers/__init__ 里的 import 是注册的触发点，删了就没有 embedder 可用。"""
    import rag.providers  # noqa: F401

    assert "fastembed" in EMBEDDERS
