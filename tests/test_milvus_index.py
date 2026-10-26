"""索引层集成测试：要一台真 Milvus，默认不跑。

    RAG_TEST_MILVUS=1 python -m pytest tests/test_milvus_index.py

地址跟 CLI 走同一份配置（config/base.yaml 的 ${MILVUS_URI} + .env），
不另开一个环境变量，免得「CLI 连的是 A、测试连的是 B」。

为什么必须用真的：索引层依赖的全是 pymilvus 的运行时行为 —— 写后要 flush 才查得到、
主键按 schema 字段名取、sparse 由服务端 BM25 生成因而既不能写也不能读、
VARCHAR 的 max_length 按 UTF-8 字节算。把客户端 mock 掉等于把这些行为一起 mock 掉，
测试就只剩「调用发生过了」。

集合名带 uuid 前缀 rag_test_，用完必删：这些用例可能打到共用的开发实例上。
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Iterator

import pytest
from pymilvus import MilvusClient

from rag.config import load_settings
from rag.config.schema import FieldIndexConfig, VectorIndexConfig
from rag.contracts import Chunk, Embeddings
from rag.indexing.milvus import MilvusIndex

pytestmark = pytest.mark.skipif(
    os.environ.get("RAG_TEST_MILVUS") != "1",
    reason="集成测试：需要真 Milvus，设 RAG_TEST_MILVUS=1 才跑",
)

DIM = 8
URI = load_settings().indexing.vector.uri
PREFIX = "rag_test_"


def _server_up() -> bool:
    try:
        MilvusClient(uri=URI, timeout=3).list_collections()
    except Exception:
        return False
    return True


@pytest.fixture(scope="module", autouse=True)
def _require_server() -> None:
    if not _server_up():
        pytest.skip(f"连不上 Milvus：{URI}")


@pytest.fixture
def index() -> Iterator[MilvusIndex]:
    """每例一个一次性集合。fixture 是同步的：建表/删表要在用例的事件循环之外做。"""
    cfg = VectorIndexConfig(
        uri=URI,
        collection=f"{PREFIX}{uuid.uuid4().hex[:12]}",
        # FLAT 而不是配置默认的 IVF_FLAT：这里只有 3 行，聚类索引在小数据上没有意义
        indexes={
            "dense": FieldIndexConfig(type="FLAT", metric="COSINE"),
            "sparse": FieldIndexConfig(type="SPARSE_INVERTED_INDEX", metric="BM25"),
        },
    )
    idx = MilvusIndex(cfg, dim=DIM)
    asyncio.run(idx.ensure_ready())
    try:
        yield idx
    finally:
        asyncio.run(idx.drop())


def _vec(axis: int) -> list[float]:
    """单位向量：每条占一个正交轴，COSINE 下最近的必然是同一个轴。"""
    vector = [0.0] * DIM
    vector[axis] = 1.0
    return vector


def _three_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="t1:0000",
            doc_id="t1",
            index=0,
            text="Milvus 用内置 BM25 函数从原文生成稀疏向量，不需要额外的编码模型",
            metadata={"source_uri": "a.md", "h1": "稀疏向量"},
        ),
        Chunk(
            chunk_id="t2:0000",
            doc_id="t2",
            index=0,
            text="检索不到刚写入的数据时，先检查有没有 flush，以及查询是否用了强一致性",
            metadata={"source_uri": "b.txt", "h1": "一致性"},
        ),
        Chunk(
            chunk_id="t3:0000",
            doc_id="t3",
            index=0,
            text="VARCHAR 字段的 max_length 按 UTF-8 字节计算，不是字符数",
            metadata={"source_uri": "c.pdf", "h1": "字段长度"},
        ),
    ]


async def _seed(idx: MilvusIndex) -> None:
    await idx.upsert(_three_chunks(), Embeddings(dense=[_vec(0), _vec(1), _vec(2)]))


# ---- 生命周期 ----


async def test_ensure_ready_is_idempotent_and_loads(index: MilvusIndex) -> None:
    """同一个集合连着调两次不能报错：ingest 每次都会调它。"""
    await index.ensure_ready()
    await index.ensure_ready()

    assert await index.count() == 0


async def test_new_collection_starts_empty(index: MilvusIndex) -> None:
    assert await index.count() == 0


# ---- 写 ----


async def test_upsert_then_count(index: MilvusIndex) -> None:
    await _seed(index)

    assert await index.count() == 3


async def test_upsert_same_chunk_id_overwrites_in_place(index: MilvusIndex) -> None:
    """幂等灌库的前提：重复 ingest 同一份语料不该翻倍，且正文要是新的。"""
    await _seed(index)
    updated = _three_chunks()[0].model_copy(update={"text": "改过的正文：稀疏向量由内置 BM25 生成"})
    await index.upsert([updated], Embeddings(dense=[_vec(0)]))

    assert await index.count() == 3
    hits = await index.search_dense(_vec(0), top_k=1)
    assert hits[0].chunk.text == "改过的正文：稀疏向量由内置 BM25 生成"


async def test_oversized_text_is_rejected_before_it_reaches_the_server(index: MilvusIndex) -> None:
    """max_length 按字节算，中文只有约 1/3 的余量：本地就该拦住，报错要说清怎么办。"""
    huge = Chunk(chunk_id="big:0000", doc_id="big", text="填" * 3000, index=0)

    with pytest.raises(ValueError, match="chunk.size"):
        await index.upsert([huge], Embeddings(dense=[_vec(0)]))


async def test_mismatched_vector_count_is_rejected(index: MilvusIndex) -> None:
    with pytest.raises(ValueError, match="不一致"):
        await index.upsert(_three_chunks(), Embeddings(dense=[_vec(0)]))


async def test_wrong_dimension_is_rejected(index: MilvusIndex) -> None:
    with pytest.raises(ValueError, match="维度"):
        await index.upsert(_three_chunks()[:1], Embeddings(dense=[[0.0] * (DIM + 1)]))


# ---- 读 ----


async def test_dense_search_finds_the_nearest_and_carries_metadata(index: MilvusIndex) -> None:
    await _seed(index)
    hits = await index.search_dense(_vec(1), top_k=3)

    assert hits[0].chunk.chunk_id == "t2:0000"
    assert hits[0].channel == "dense"
    assert hits[0].rank == 0
    assert hits[0].chunk.metadata["h1"] == "一致性"
    assert hits[0].chunk.metadata["source_uri"] == "b.txt"


async def test_sparse_search_matches_on_terms_not_vectors(index: MilvusIndex) -> None:
    """sparse 路直接传查询原文：分词与词权重都由服务端 BM25 算。"""
    await _seed(index)
    hits = await index.search_sparse("字节", top_k=3)

    assert [h.chunk.chunk_id for h in hits][0] == "t3:0000"
    assert hits[0].channel == "sparse"


async def test_hybrid_returns_server_side_fused_scores(index: MilvusIndex) -> None:
    await _seed(index)
    hits = await index.search_hybrid(_vec(0), "稀疏向量", top_k=3, dense_limit=3, sparse_limit=3)

    assert hits[0].chunk.chunk_id == "t1:0000"
    assert all(h.channel == "hybrid" for h in hits)
    assert all(h.score > 0 for h in hits)


async def test_filter_on_plain_field_narrows_results(index: MilvusIndex) -> None:
    await _seed(index)
    hits = await index.search_dense(_vec(0), top_k=3, filter_expr='doc_id == "t3"')

    assert [h.chunk.chunk_id for h in hits] == ["t3:0000"]


async def test_filter_on_json_metadata_narrows_results(index: MilvusIndex) -> None:
    """块级元数据在 JSON 字段里，过滤要写成 metadata["键"] 这种下标形式。"""
    await _seed(index)
    hits = await index.search_dense(_vec(0), top_k=3, filter_expr='metadata["h1"] == "字段长度"')

    assert [h.chunk.chunk_id for h in hits] == ["t3:0000"]


async def test_empty_filter_expression_is_not_a_filter(index: MilvusIndex) -> None:
    await _seed(index)
    hits = await index.search_dense(_vec(0), top_k=3, filter_expr=None)

    assert len(hits) == 3


# ---- 删 ----


async def test_delete_by_chunk_id_removes_exactly_those(index: MilvusIndex) -> None:
    await _seed(index)
    await index.delete(["t1:0000", "t2:0000"])

    assert await index.count() == 1


async def test_delete_docs_clears_a_whole_document(index: MilvusIndex) -> None:
    await index.upsert(
        [
            Chunk(chunk_id="t1:0000", doc_id="t1", index=0, text="第一段"),
            Chunk(chunk_id="t1:0001", doc_id="t1", index=1, text="第二段"),
            Chunk(chunk_id="t1:0002", doc_id="t1", index=2, text="第三段"),
            Chunk(chunk_id="t9:0000", doc_id="t9", index=0, text="别人家的"),
        ],
        Embeddings(dense=[_vec(0), _vec(1), _vec(2), _vec(3)]),
    )
    await index.delete_docs(["t1"])

    remaining = await index.search_dense(_vec(0), top_k=10)
    assert [h.chunk.doc_id for h in remaining] == ["t9"]


async def test_reingesting_a_shortened_document_leaves_no_orphan_chunks(index: MilvusIndex) -> None:
    """chunk_id 是 doc_id + 序号，upsert 只覆盖同序号的块。

    文档改短以后多出来的旧序号块不会自己消失，会变成「检索得到、正文却已不存在」
    的孤儿。CLI 的 ingest 先 delete_docs 再 upsert 就是为了这个。
    """
    long_version = [
        Chunk(chunk_id=f"t1:{i:04d}", doc_id="t1", index=i, text=f"第 {i} 段") for i in range(3)
    ]
    await index.upsert(long_version, Embeddings(dense=[_vec(0), _vec(1), _vec(2)]))
    assert await index.count() == 3

    await index.delete_docs(["t1"])
    await index.upsert(long_version[:1], Embeddings(dense=[_vec(0)]))

    assert await index.count() == 1


# ---- 删表 ----


async def test_drop_removes_the_collection(index: MilvusIndex) -> None:
    await _seed(index)
    await index.drop()

    assert not MilvusClient(uri=URI, timeout=3).has_collection(index.collection)


async def test_drop_is_safe_on_a_collection_that_is_gone(index: MilvusIndex) -> None:
    """测试的 teardown 会无条件调它，重复调用不能报错。"""
    await index.drop()
    await index.drop()
