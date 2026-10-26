"""图谱存储与**图谱通道**（M6）：归一/过滤的判据、写库幂等、邻居扩展、映射回块。

全部离线：Neo4j 驱动是替身（记下每条 Cypher 与参数），Milvus 也是替身。
这一层要钉住的是**判据**（哪些写法算同一个节点、哪些名字不像概念）与**打分规矩**
（跳数越近越值钱、同一块取最大值而不是求和）—— 真连库的部分在真机验证里做。

**图谱通道的实现已经搬成插件包**（`plugins/rag-channel-neo4j`，2026-10-08）：核心只留
「按 type 找工厂」这套机制。所以这一层里跟通道有关的用例会 `importorskip` 那个包 ——
没装插件时跳过（核心的图谱存储、判据这些照常测）。
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from rag.config.schema import GraphChannelConfig
from rag.contracts import Chunk

# **图谱相关的一切都在插件包里**（2026-10-08 搬出去的）：没装就跳过这一层，
# 核心的其余测试照常跑。
_plugin = pytest.importorskip(
    "rag_channel_neo4j", reason="图谱是插件包（pip install -e plugins/rag-channel-neo4j）"
)
GraphRetriever = _plugin.GraphRetriever
_store = pytest.importorskip("rag_channel_neo4j.store")
GraphStore, is_usable_entity, norm_key = _store.GraphStore, _store.is_usable_entity, _store.norm_key


class _Record:
    def __init__(self, row: dict[str, Any]) -> None:
        self._row = row

    def data(self) -> dict[str, Any]:
        return self._row


class _Session:
    def __init__(self, driver: _FakeDriver) -> None:
        self._driver = driver

    def __enter__(self) -> _Session:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def run(self, cypher: str, **params: Any) -> list[_Record]:
        self._driver.calls.append((cypher, params))
        return [_Record(row) for row in self._driver.respond(cypher, params)]


class _FakeDriver:
    """替身驱动：按 Cypher 决定回什么，并记下所有语句。"""

    def __init__(self, respond=None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._respond = respond or (lambda cypher, params: [])
        self.verified = False

    def session(self) -> _Session:
        return _Session(self)

    def verify_connectivity(self) -> None:
        self.verified = True

    def respond(self, cypher: str, params: dict) -> list[dict]:
        return self._respond(cypher, params)

    def queries(self) -> list[str]:
        return [cypher for cypher, _ in self.calls]


def _store(driver: _FakeDriver) -> GraphStore:
    return GraphStore(GraphChannelConfig(), "pw", driver=driver)


# ---- 归一与过滤：这两条判据决定图里有多少节点 ----


def test_aliases_land_on_one_key() -> None:
    """实测：同一张图里 `Flink`×24 与 `Apache Flink`×4 是两个节点 —— 不合并就漏边。"""
    assert norm_key("Flink") == norm_key("Apache Flink") == "flink"
    assert norm_key("  Apache  Flink ") == "flink"
    assert norm_key("StreamExecutionEnvironment") == "streamexecutionenvironment"


def test_namespaced_class_names_fold_to_the_class() -> None:
    assert norm_key("org.apache.flink.api.java.ExecutionEnvironment") == "executionenvironment"
    assert norm_key("java.util.List") == "list"
    # 带 `-`/`_` 的坐标：去掉 ${...} 之后收尾
    assert norm_key("flink-scala_${scala.binary.version}") == "flink-scala"


def test_unlike_concepts_stay_apart() -> None:
    assert norm_key("Spark") != norm_key("SparkStreaming") != norm_key("Flink")


def test_junk_names_are_filtered_by_shape_not_length() -> None:
    """判据是**字符构成**：实测「超长」里大多数是对的 Java 类全名，按长度砍会砍掉好的。"""
    assert is_usable_entity("org.apache.flink.api.java.ExecutionEnvironment")
    assert is_usable_entity("StreamExecutionEnvironment")

    assert not is_usable_entity("flink-scala_${scala.binary.version}")
    assert not is_usable_entity("```\nval env = ...")
    assert not is_usable_entity("org.apache.flink api.java ExecutionEnvironment")
    assert not is_usable_entity("https://kafka.apache.org/documentation")
    assert not is_usable_entity("   ")
    assert not is_usable_entity("x" * 61)


# ---- 写库：幂等 + 归一 + 记账 ----


def _entries() -> dict[str, dict]:
    return {
        "d1:0001": {
            "heading_path": "第一章 -> 1.1",
            "page": 3,
            "entities": [
                {"name": "Flink", "type": "技术"},
                {"name": "Apache Flink", "type": "技术"},
                {"name": "窗口", "type": "概念"},
                {"name": "flink-scala_${scala.binary.version}", "type": "组件"},
            ],
            "relations": [
                {"source": "Flink", "relation": "支持", "target": "窗口", "evidence": "支持窗口"},
                {
                    "source": "Apache Flink",
                    "relation": "支持",
                    "target": "窗口",
                    "evidence": "重复",
                },
                {"source": "Flink", "relation": "是", "target": "Flink", "evidence": "自环"},
            ],
        }
    }


def test_load_deletes_this_document_first_then_writes() -> None:
    """按文档幂等：先删这篇的边，再写新的（与 Milvus 那边同一个规矩）。"""
    driver = _FakeDriver()

    stats = asyncio.run(_store(driver).load_document("d1", _entries()))

    assert "MATCH ()-[r:REL {doc_id: $doc_id}]->() DELETE r" in driver.queries()[0]
    assert driver.calls[0][1] == {"doc_id": "d1"}
    assert stats.entities == 2, "Flink / Apache Flink 归一成一个，加上「窗口」"
    assert stats.relations == 1, "重复的关系与自环都该丢掉"
    assert stats.skipped >= 2, "Maven 坐标与自环要记进 skipped"


def test_nodes_carry_every_original_name_and_the_doc() -> None:
    driver = _FakeDriver()

    asyncio.run(_store(driver).load_document("d1", _entries()))

    nodes = next(params["nodes"] for cypher, params in driver.calls if "UNWIND $nodes" in cypher)
    flink = next(node for node in nodes if node["key"] == "flink")
    assert sorted(flink["names"]) == ["Apache Flink", "Flink"], "原文写法一个都不能丢"
    assert flink["docs"] == ["d1"]

    edges = next(params["edges"] for cypher, params in driver.calls if "UNWIND $edges" in cypher)
    assert edges[0]["chunk_id"] == "d1:0001" and edges[0]["page"] == 3
    assert edges[0]["heading_path"] == "第一章 -> 1.1", "证据要能定位回原文"
    assert edges[0]["key"], "关系也要有稳定 key（MERGE 靠它去重）"


def test_edge_keys_are_stable_across_loads() -> None:
    first, second = _FakeDriver(), _FakeDriver()

    asyncio.run(_store(first).load_document("d1", _entries()))
    asyncio.run(_store(second).load_document("d1", _entries()))

    key_of = lambda driver: next(  # noqa: E731
        params["edges"][0]["key"] for cypher, params in driver.calls if "UNWIND $edges" in cypher
    )
    assert key_of(first) == key_of(second)


# ---- 查询：实体链接与多跳扩展 ----


def test_find_entities_matches_names_and_normalized_keys() -> None:
    seen: list[dict] = []

    def respond(cypher: str, params: dict) -> list[dict]:
        seen.append(params)
        return [{"key": "flink"}]

    keys = asyncio.run(_store(_FakeDriver(respond)).find_entities("Flink 的架构图", limit=5))

    assert keys == ["flink"] and seen[0]["limit"] == 5


def _neighbor_rows(params: dict) -> list[dict]:
    """一跳：从 flink 出发；二跳：从第一跳新发现的实体出发。"""
    if params["keys"] == ["flink"]:
        return [
            {
                "source": "flink",
                "neighbor": "窗口",
                "relation": "支持",
                "evidence": "支持窗口",
                "chunk_id": "d1:0001",
                "doc_id": "d1",
                "heading_path": "1.1",
                "page": 3,
            }
        ]
    return [
        {
            "source": "窗口",
            "neighbor": "时间语义",
            "relation": "包含",
            "evidence": "窗口包含时间语义",
            "chunk_id": "d1:0002",
            "doc_id": "d1",
            "heading_path": "1.2",
            "page": 4,
        }
    ]


def test_neighbors_expand_hop_by_hop_and_dedupe() -> None:
    driver = _FakeDriver(lambda cypher, params: _neighbor_rows(params))

    edges = asyncio.run(_store(driver).neighbors(["flink"], hops=2, limit=10))

    assert [(edge["hop"], edge["chunk_id"]) for edge in edges] == [(1, "d1:0001"), (2, "d1:0002")]


def test_neighbors_stop_at_the_requested_hops() -> None:
    driver = _FakeDriver(lambda cypher, params: _neighbor_rows(params))

    edges = asyncio.run(_store(driver).neighbors(["flink"], hops=1, limit=10))

    assert [edge["hop"] for edge in edges] == [1]


# ---- 通道：映射回块 + 打分 + 过滤 ----


class _FakeStore:
    def __init__(self, keys: list[str], edges: list[dict]) -> None:
        self._keys, self._edges = keys, edges
        self.name = "neo4j:fake"

    async def find_entities(self, question: str, *, limit: int) -> list[str]:
        return self._keys

    async def neighbors(self, keys: list[str], *, hops: int, limit: int) -> list[dict]:
        return self._edges


class _FakeIndex:
    name = "milvus:fake"

    def __init__(self) -> None:
        self.asked: list[str] = []

    async def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        self.asked.append(list(chunk_ids))
        return [
            Chunk(chunk_id=cid, doc_id="d1", text=f"{cid} 的正文", heading_path="1.1")
            for cid in chunk_ids
        ]


def _edge(chunk_id: str, hop: int) -> dict:
    return {
        "source": "flink",
        "neighbor": "窗口",
        "relation": "支持",
        "evidence": "支持窗口",
        "chunk_id": chunk_id,
        "doc_id": "d1",
        "heading_path": "1.1",
        "page": 3,
        "hop": hop,
    }


def test_the_channel_scores_by_hop_and_keeps_the_max_per_chunk() -> None:
    """同一块被多条关系命中取**最大值**：求和会让热门实体的边把分数堆到虚高。"""
    store = _FakeStore(["flink"], [_edge("d1:0001", 1), _edge("d1:0001", 2), _edge("d1:0002", 2)])
    index = _FakeIndex()
    channel = GraphRetriever(index, store, GraphChannelConfig(hops=2))

    hits = asyncio.run(channel.retrieve("Flink 支持什么", top_k=5))

    assert [hit.chunk.chunk_id for hit in hits] == ["d1:0001", "d1:0002"]
    assert hits[0].score == 1.0 and hits[1].score == 0.5
    assert hits[0].channel.startswith("graph:")


def test_the_channel_returns_nothing_without_entities() -> None:
    channel = GraphRetriever(
        _FakeIndex(), _FakeStore([], [_edge("d1:0001", 1)]), GraphChannelConfig()
    )

    assert asyncio.run(channel.retrieve("完全无关的问题", top_k=5)) == []
    assert asyncio.run(channel.retrieve("   ", top_k=5)) == [], "空白查询不该去查图"


def test_the_channel_applies_filters_after_fetching() -> None:
    """别的路把过滤下推给 Milvus；这条路的数据在 Neo4j，所以取回块之后再判。"""
    store = _FakeStore(["flink"], [_edge("d1:0001", 1), _edge("d1:0002", 2)])
    channel = GraphRetriever(_FakeIndex(), store, GraphChannelConfig())

    hits = asyncio.run(channel.retrieve("Flink", top_k=5, filters={"doc_id": "d9"}))
    kept = asyncio.run(channel.retrieve("Flink", top_k=5, filters={"doc_id": "d1"}))

    assert hits == [] and len(kept) == 2


def test_the_neo4j_type_comes_from_the_plugin_not_the_core() -> None:
    """**图谱通道是插件**（2026-10-08 搬出去的）：核心的内置类型里没有 `neo4j`，
    它是按 entry point 找到 `rag_channel_neo4j` 才装得起来的。

    这条同时钉住「通道名 = 配置里的 name」（不再让适配器自己取名），以及
    「核心不认识 Neo4j 本身」这条边界 —— 内核里只剩 `milvus` 一路。
    """
    from rag.config.schema import ChannelConfig, Settings
    from rag.contracts import CHANNEL_TYPES
    from rag.retrieval import build_channel
    from rag.retrieval.plugins import resolve_channel_type

    assert "neo4j" not in CHANNEL_TYPES.names(), "核心不该再内置图谱通道"
    assert callable(resolve_channel_type("neo4j")), "但插件装了就该找得到"

    channel = build_channel(
        ChannelConfig(name="kg", type="neo4j"),
        Settings(),
        None,  # type: ignore[arg-type]
    )

    assert channel.name == "kg"
