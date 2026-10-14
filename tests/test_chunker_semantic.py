"""markdown-semantic 切分器：四级流水线的每一级都要能单独验。

用替身 embedder 把「语义断点」变成可控的东西 —— 真模型跑出来的断点位置会随模型变，
不适合当断言；但「给定距离，断点该不该落在这里」是可以钉死的。
"""

from __future__ import annotations

import pytest

from rag.config.schema import ChunkingConfig, SemanticChunkConfig
from rag.contracts import Document, Embeddings
from rag.ingest.chunkers import build_chunker


class _StubEmbedder:
    """按句首字给向量：甲→[1,0]，乙→[0,1]。断点就出现在甲/乙交界处。"""

    def __init__(self) -> None:
        self.calls = 0
        self.batches: list[list[str]] = []

    @property
    def dim(self) -> int:
        return 2

    def _vector(self, text: str) -> list[float]:
        return [1.0, 0.0] if text.startswith("甲") else [0.0, 1.0]

    async def embed_documents(self, texts: list[str]) -> Embeddings:
        self.calls += 1
        self.batches.append(list(texts))
        return Embeddings(dense=[self._vector(t) for t in texts])

    async def embed_query(self, text: str) -> Embeddings:
        return Embeddings(dense=[self._vector(text)])


def _cfg(*, size: int = 20, min_size: int = 4, percentile: float = 90.0) -> ChunkingConfig:
    return ChunkingConfig(
        strategy="markdown-semantic",
        size=size,
        overlap=0,
        semantic=SemanticChunkConfig(min_size=min_size, percentile=percentile),
    )


def _split(text: str, cfg: ChunkingConfig, embedder: _StubEmbedder) -> list[str]:
    doc = Document(doc_id="d", source_uri="x.md", text=text)
    return [c.text for c in build_chunker(cfg, embedder).split([doc])]


# ---- 缺 embedder 不能静默退化 ----­


def test_semantic_strategy_without_an_embedder_fails_loudly() -> None:
    """缺 embedder 时偷偷只做硬切，会让人以为语义切分生效了。"""
    with pytest.raises(ValueError, match="embedder"):
        build_chunker(_cfg())


# ---- 第 1 级：header ----­


def test_heading_path_and_numbers_reach_every_chunk() -> None:
    text = "# 年报\n\n## 财务\n\n营收增长。\n\n## 风险\n\n汇率波动。"

    doc = Document(doc_id="d", source_uri="x.md", text=text)
    chunks = build_chunker(_cfg(size=1000), _StubEmbedder()).split([doc])

    # 路径是累积的；编号是**位置派生**的：第 1 个 h1 下的第 1、2 个 h2 → 1-1、1-2
    assert chunks[0].heading_path == "年报 -> 财务"
    assert chunks[0].heading_numbers == "1-1"
    assert chunks[1].heading_path == "年报 -> 风险"
    assert chunks[1].heading_numbers == "1-2"


def test_heading_is_attached_in_front_not_embedded() -> None:
    """标题**不在正文里**：切分时剥掉，切完贴到**开头**。

    贴开头是拿向量区分度换阅读顺序的取舍：实测末尾的对比度更好（0.68 vs 0.82，
    见 docs/milvus-schema-design.md §2.4），但读这一块时先看到所属章节更自然。
    """
    text = "# 年报\n\n## 财务\n\n营收增长。"

    doc = Document(doc_id="d", source_uri="x.md", text=text)
    chunk = build_chunker(_cfg(size=1000), _StubEmbedder()).split([doc])[0]

    assert chunk.text == "年报 -> 财务\n\n营收增长。"


# ---- 第 2 级：只用三个分隔符 ----­


def test_comma_only_text_is_not_split_by_the_recursive_stage() -> None:
    """逗号不在分隔符里 —— 这正是「只有逗号的长串」会漏到语义那一级的原因。"""
    text = "甲甲甲，甲甲甲，甲甲甲，甲甲甲，甲甲甲，甲甲甲，"
    embedder = _StubEmbedder()

    chunks = _split(text, _cfg(size=20), embedder)

    # 递归那级切不动，只能由语义那级接手 —— 它一定被调用过
    assert embedder.calls >= 1
    assert all(len(c) <= 20 for c in chunks)


def test_nothing_exceeds_size() -> None:
    text = "甲" * 95  # 没有任何分隔符

    chunks = _split(text, _cfg(size=30), _StubEmbedder())

    assert [len(c) for c in chunks] == [30, 30, 30, 5]


# ---- 第 3 级：语义断点 ----­


def test_breakpoint_lands_where_similarity_drops() -> None:
    """甲→乙 的交界处相似度骤降，应当正好断在那里。"""
    text = "甲甲甲，甲甲甲，甲甲甲，乙乙乙，乙乙乙，乙乙乙，"

    # size 必须小于原文长度，否则递归那级就把它整个留下了、根本到不了语义这一步
    chunks = _split(text, _cfg(size=20), _StubEmbedder())

    assert chunks == ["甲甲甲，甲甲甲，甲甲甲，", "乙乙乙，乙乙乙，乙乙乙，"]


def test_embedder_is_not_called_when_everything_already_fits() -> None:
    """语义那一步是唯一贵的，能不走就不走。"""
    embedder = _StubEmbedder()

    _split("短句。", _cfg(size=1000), embedder)

    assert embedder.calls == 0


def test_min_size_prevents_micro_chunks() -> None:
    """裸 percentile 会切出 3 个字的块；min_size 把它拦住。"""
    text = "甲甲甲，乙乙乙，甲甲甲，乙乙乙，甲甲甲，乙乙乙，"

    chunks = _split(text, _cfg(size=1000, min_size=8), _StubEmbedder())

    assert all(len(c) >= 8 for c in chunks[:-1]), chunks
    assert len(chunks) < 6, f"min_size 没拦住，切碎了：{chunks}"


# ---- 第 4 级：尺寸护栏 ----­


def test_size_cap_forces_a_cut_even_when_semantics_say_nothing() -> None:
    """语义断点不是尺寸控制：阈值调到 100（等于没有断点）时，仍然不许超 size。"""
    text = "甲甲甲，" * 10  # 向量全一样，没有任何语义断点

    chunks = _split(text, _cfg(size=20, percentile=100.0), _StubEmbedder())

    assert len(chunks) > 1
    assert all(len(c) <= 20 for c in chunks)


def test_a_single_oversized_sentence_is_hard_cut() -> None:
    """连句子都分不出来时（表格 HTML 那种），只能按字符切 —— 但必须切到 size 以内。"""
    text = "<table>" + "<tr><td>值</td></tr>" * 20

    chunks = _split(text, _cfg(size=50), _StubEmbedder())

    assert all(len(c) <= 50 for c in chunks)
    assert "".join(chunks) == text


def test_no_chunk_is_empty() -> None:
    text = "甲甲甲。\n\n乙乙乙。\n\n# 标题\n\n丙丙丙。"

    chunks = _split(text, _cfg(size=1000), _StubEmbedder())

    assert all(c.strip() for c in chunks)
