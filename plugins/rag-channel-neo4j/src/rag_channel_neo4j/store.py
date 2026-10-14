"""知识图谱的存储层（Neo4j，M6）：写入、统计、以及通道要的两条查询。

**为什么归一与过滤放在入库时**（2026-10-05 定）：

* **归一**：同一概念在原文里有多种写法 —— 实测 Flink 那张图里同时出现
  `Flink`（24 次）与 `Apache Flink`（4 次），第二章还有
  `StreamExecutionEnvironment` 与 `org.apache.flink.api.java.ExecutionEnvironment`。
  不合并就等于把同一个概念的边拆到几个节点上，**一跳查询必然漏**。所以节点身份用
  `key`（归一后的键），原文写法全部留在 `names` 里（不丢信息，人也能看）。
* **过滤**：代码块会抽出 `flink-scala_${scala.binary.version}` 这类 Maven 坐标。判据是
  **字符构成**而不是长度 —— 实测「超长实体名」里大多数是对的 Java 类全名
  （`StreamExecutionEnvironment`、`org.apache.flink.api.java.ExecutionEnvironment`），
  按长度砍会把好的砍掉。

写库**按文档幂等**：先删这篇文档的边、再写新的（与 Milvus 那边「先删后写」同一个规矩），
所以重灌一份文档不会留下上一次的边。
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from rag.config.schema import GraphChannelConfig

# 名字里出现这些就是代码/坐标而不是概念：Maven 的 ${...}、花括号、换行、制表
_JUNK = re.compile(r"[$}{]|\n|\r|\t")
# 「两个点路径被空白隔开」= OCR 把点读成了空格（实测：
# org.apache.flink api.java ExecutionEnvironment）。这种名字对应的正确写法本来就有，
# 留着它只会多一个查不到的节点。
_SPLIT_PATH = re.compile(r"\w+\.\w+\s+\S*\w+\.\w+")
# 顶级别名：取最后一段就是类名
_NAMESPACE = re.compile(r"^(?:org|com|io|net|java|javax|scala|akka)\.")
# 组织前缀：`Apache Flink` 与 `Flink` 该是同一个节点
_ORG = re.compile(r"^(?:apache|the)\s+")

MAX_NAME_CHARS = 60


def is_usable_entity(name: str) -> bool:
    """这个名字像不像「概念」而不是「代码片段」。判据全在字符构成上（见模块说明）。"""
    text = (name or "").strip()
    if not text or len(text) > MAX_NAME_CHARS:
        return False
    if _JUNK.search(text) or "http://" in text or "https://" in text:
        return False
    return not _SPLIT_PATH.search(text)


def norm_key(name: str) -> str:
    """归一键：它决定「哪些写法是同一个节点」。

    规则（都在实测的样本上验过）：

    1. NFKC + 小写 + 去首尾空白（全角/半角、大小写先统一）；
    2. 去掉 Maven 坐标里的 `${...}`（`flink-scala_${scala.binary.version}` → `flink-scala_`）；
    3. 带命名空间的全名取最后一段（`org.apache.flink...ExecutionEnvironment` →
       `executionenvironment`）；**只在没有空格时做**，带空格的路径是 OCR 拼坏的，已被过滤掉；
    4. 去掉组织前缀 `apache ` / `the `（`Apache Flink` → `flink`）；
    5. 去掉首尾的标点与分隔符、折叠空白。
    """
    text = unicodedata.normalize("NFKC", name or "").strip().lower()
    text = re.sub(r"\$\{[^}]*\}", "", text)
    if "." in text and " " not in text:
        text = re.sub(_NAMESPACE, "", text).rsplit(".", 1)[-1]
    text = _ORG.sub("", text)
    text = text.strip(" -_.")
    return re.sub(r"\s+", " ", text)


@dataclass
class LoadStats:
    """一次写库的账：写进去多少、因为不像概念丢了多少。"""

    doc_id: str
    entities: int = 0
    relations: int = 0
    skipped: int = 0


@dataclass
class GraphStats:
    entities: int = 0
    relations: int = 0
    by_doc: dict[str, int] = field(default_factory=dict)


_SCHEMA = "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE"
_DELETE_DOC = "MATCH ()-[r:REL {doc_id: $doc_id}]->() DELETE r"
_UPSERT_NODES = """
UNWIND $nodes AS n
MERGE (e:Entity {key: n.key})
SET e.names = coalesce(e.names, []) + [x IN n.names WHERE NOT x IN coalesce(e.names, [])],
    e.types = coalesce(e.types, []) + [x IN n.types WHERE NOT x IN coalesce(e.types, [])],
    e.docs  = coalesce(e.docs,  []) + [x IN n.docs  WHERE NOT x IN coalesce(e.docs,  [])]
"""
_UPSERT_EDGES = """
UNWIND $edges AS e
MERGE (a:Entity {key: e.source})
MERGE (b:Entity {key: e.target})
MERGE (a)-[r:REL {key: e.key}]->(b)
SET r.relation = e.relation, r.evidence = e.evidence, r.chunk_id = e.chunk_id,
    r.doc_id = e.doc_id, r.heading_path = e.heading_path, r.page = e.page
"""
_COUNT_ENTITIES = "MATCH (e:Entity) RETURN count(e) AS n"
_COUNT_RELATIONS = "MATCH ()-[r:REL]->() RETURN count(r) AS n"
_COUNT_BY_DOC = "MATCH ()-[r:REL]->() RETURN r.doc_id AS doc, count(r) AS n ORDER BY doc"
_FIND = """
MATCH (e:Entity)
WHERE any(n IN e.names WHERE size(n) >= 2 AND toLower($question) CONTAINS toLower(n))
   OR toLower($question) CONTAINS e.key
RETURN e.key AS key
ORDER BY size(e.names) DESC
LIMIT $limit
"""
_NEIGHBORS = """
MATCH (a:Entity)-[r:REL]-(b:Entity)
WHERE a.key IN $keys
RETURN a.key AS source, b.key AS neighbor, r.relation AS relation, r.evidence AS evidence,
       r.chunk_id AS chunk_id, r.doc_id AS doc_id, r.heading_path AS heading_path,
       r.page AS page
ORDER BY r.doc_id, r.chunk_id
LIMIT $limit
"""


class GraphStore:
    """Neo4j 客户端。同步驱动 + ``to_thread``（与 Milvus 那边同一个路子，别堵事件循环）。"""

    def __init__(
        self, cfg: GraphChannelConfig, password: str | None, *, driver: Any = None
    ) -> None:
        self._cfg = cfg
        self._password = password or ""
        self._driver = driver
        self.name = f"neo4j:{cfg.uri}"

    @property
    def driver(self) -> Any:
        """懒建驱动，但**注入的替身优先** —— 否则测试会真的去连库（实测踩过：
        `cached_property` 会绕开注入的 driver，用例于是打到了真 Neo4j 上）。"""
        if self._driver is None:
            self._driver = self._build_driver()
        return self._driver

    def _build_driver(self) -> Any:
        from neo4j import GraphDatabase

        return GraphDatabase.driver(self._cfg.uri, auth=(self._cfg.user, self._password))

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()

    # ---- 同步内核（都在线程里跑）----

    def _apply(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        with self.driver.session() as session:
            result = session.run(cypher, **params)
            return [record.data() for record in result]

    def _verify(self) -> None:
        self.driver.verify_connectivity()

    async def _run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        return await asyncio.wait_for(
            asyncio.to_thread(self._apply, cypher, **params), self._cfg.timeout_s
        )

    # ---- 对外 ----

    async def ping(self, timeout_s: float | None = None) -> None:
        """探活：连不上就在超时内抛出来（通道会把它记成「这一路失败」，其余照常）。"""
        await asyncio.wait_for(asyncio.to_thread(self._verify), timeout_s or self._cfg.timeout_s)

    async def ensure_schema(self) -> None:
        await self._run(_SCHEMA)

    async def load_document(
        self, doc_id: str, entries: dict[str, dict[str, Any]], *, source_uri: str = ""
    ) -> LoadStats:
        """把一篇文档的抽取产物写进图。`entries` = `{chunk_id: {entities, relations, ...}}`。

        与 Milvus 同一个规矩：**先删这篇文档的边，再写新的** —— 重灌不会留旧边。
        节点不删（它可能被别的文档共用），归一之后本来就该共用。
        """
        nodes: dict[str, dict[str, Any]] = {}
        # 按 key 去重：同一块里 `Flink --支持--> 窗口` 与 `Apache Flink --支持--> 窗口`
        # 归一到同一个 key，写进图里本来就是一条边 —— **账也要按这个口径记**，
        # 否则「关系数」会比图里真实的多（实测踩过）
        edges: dict[str, dict[str, Any]] = {}
        skipped = 0
        for chunk_id, entry in entries.items():
            heading = str(entry.get("heading_path") or "")
            page = int(entry.get("page") or 0)
            key_of: dict[str, str] = {}
            for entity in entry.get("entities") or []:
                name = str(entity.get("name") or "").strip()
                if not is_usable_entity(name):
                    skipped += 1
                    continue
                key = norm_key(name)
                if not key:
                    skipped += 1
                    continue
                key_of[name] = key
                bucket = nodes.setdefault(key, {"key": key, "names": [], "types": [], "docs": []})
                if name not in bucket["names"]:
                    bucket["names"].append(name)
                kind = str(entity.get("type") or "").strip()
                if kind and kind not in bucket["types"]:
                    bucket["types"].append(kind)
                if doc_id not in bucket["docs"]:
                    bucket["docs"].append(doc_id)
            for relation in entry.get("relations") or []:
                source, target = (
                    str(relation.get("source") or ""),
                    str(relation.get("target") or ""),
                )
                if not (is_usable_entity(source) and is_usable_entity(target)):
                    skipped += 1
                    continue
                source_key = key_of.get(source) or norm_key(source)
                target_key = key_of.get(target) or norm_key(target)
                if not source_key or not target_key or source_key == target_key:
                    skipped += 1
                    continue
                raw = f"{doc_id}|{chunk_id}|{source_key}|{relation.get('relation')}|{target_key}"
                key = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
                edges[key] = {
                    "key": key,
                    "source": source_key,
                    "target": target_key,
                    "relation": str(relation.get("relation") or ""),
                    "evidence": str(relation.get("evidence") or ""),
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "heading_path": heading,
                    "page": page,
                }

        await self._run(_DELETE_DOC, doc_id=doc_id)
        if nodes:
            await self._run(_UPSERT_NODES, nodes=list(nodes.values()))
        if edges:
            await self._run(_UPSERT_EDGES, edges=list(edges.values()))
        return LoadStats(doc_id=doc_id, entities=len(nodes), relations=len(edges), skipped=skipped)

    async def stats(self) -> GraphStats:
        entities = await self._run(_COUNT_ENTITIES)
        relations = await self._run(_COUNT_RELATIONS)
        by_doc = await self._run(_COUNT_BY_DOC)
        return GraphStats(
            entities=int(entities[0]["n"]) if entities else 0,
            relations=int(relations[0]["n"]) if relations else 0,
            by_doc={str(row["doc"]): int(row["n"]) for row in by_doc},
        )

    async def find_entities(self, question: str, *, limit: int) -> list[str]:
        """问题里出现了哪些图里的实体（按归一键返回）。**零额外模型调用**：纯子串匹配。"""
        rows = await self._run(_FIND, question=question, limit=limit)
        return [str(row["key"]) for row in rows]

    async def neighbors(self, keys: list[str], *, hops: int, limit: int) -> list[dict[str, Any]]:
        """从这些实体出发走 `hops` 跳，返回边的列表（每条带 `hop`）。

        一跳一跳地扩（而不是一句 `*1..2`）：每一跳的 frontier 是**上一跳新发现的实体**，
        这样每一跳都能单独限流，也好把 `hop` 记在边上当打分依据。
        """
        seen: set[str] = set()
        edges: list[dict[str, Any]] = []
        frontier = list(dict.fromkeys(keys))
        for hop in range(1, max(1, hops) + 1):
            if not frontier:
                break
            rows = await self._run(_NEIGHBORS, keys=frontier, limit=limit)
            next_frontier: list[str] = []
            for row in rows:
                edge_key = f"{row['source']}|{row['relation']}|{row['neighbor']}|{row['chunk_id']}"
                if edge_key in seen:
                    continue
                seen.add(edge_key)
                edges.append({**row, "hop": hop})
                neighbor = str(row["neighbor"])
                if neighbor not in frontier:
                    next_frontier.append(neighbor)
            frontier = next_frontier
        return edges


def build_graph_store(
    settings: Any, secrets: Any = None, *, driver: Any = None, cfg: Any = None
) -> GraphStore:
    """按配置建图库客户端（密码从 Secrets 来 —— 不进 yaml）。

    `cfg` 给了就用它：**插件要开第二路同类源**时，参数写在那一路的 `options` 里
    （见 `plugins/rag-channel-neo4j`），而不是去改核心的 `retrieval.graph`。
    """
    if secrets is None:
        from rag.config import get_secrets

        secrets = get_secrets()
    return GraphStore(
        cfg or settings.retrieval.graph,
        getattr(secrets, "neo4j_password", None),
        driver=driver,
    )
