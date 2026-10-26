"""文档注册表：**每篇一行**（走到哪一步 + 入库信息）。2026-10-09 从 Milvus 搬到数据库。

为什么要搬（使用者 2026-10-09 定）：「解析的 document 文档列表其实可以走 PG 的，
走 milvus 意义不大」。实测也印证了：Milvus 一抖（两张表卡在 `Loading 50%`，query
等 **9013ms** 才回 503 `channel not available`），文档列表与详情就白等 9 秒 ——
而它们要的不过是「这份到哪一步了」。这是 O(文档数) 的关系型数据，本来就该在关系库里；
Milvus 只留向量本体与检索。

三条口径：

1. **表里带状态**：`stage` 记这份文档走到哪一步
   （`uploaded → parsed → reviewed → chunked → indexed`，见 `contracts.DOC_STAGES`）。
   列表页**只读这一列**，不去翻产物、不算哈希 —— 那是详情页的活。
2. **PG + SQLite 双实现**：单机 SQLite（`data/documents.sqlite3`，WAL），多 worker PG；
   `documents.backend: auto|sqlite|postgres`，`auto` = 配了 PG 就用 PG。
3. **对外形状不变**：`ensure_ready / all / get / upsert / delete / drop` 与原来那个
   Milvus 版同名同义，所以 `commit_chunks`、CLI、agent 工具、文档接口都不用改调用方式。

两处刻意的选择：

* **同步实现 + `asyncio.to_thread` 门面**：查询是毫秒级的小活，为它引一套 async 驱动
  不划算；SQLite 的驱动本来就是同步的，两条路各写一套反而是两个坑。
* **连接用完就关**（`_session`）：SQLite 的 `with conn` 只管事务、不关连接（实测），
  照抄会在列表页上每请求漏一个句柄。psycopg3 的 `with conn` 会关，但两条路统一显式关，
  行为才不会分家。
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag.contracts import DocRecord

DEFAULT_SQLITE_PATH = "./data/documents.sqlite3"

# 两张库共用同一段 DDL：全是文本/整数列，`doc_id` 直接当主键（不需要自增，两个库写法一致）
_SCHEMA = (
    """
    create table if not exists document_registry (
        doc_id           text primary key,
        source_uri       text not null default '',
        content_hash     text not null default '',
        chunk_count      integer not null default 0,
        ingested_at      integer not null default 0,
        pipeline_version text not null default '',
        stage            text not null default 'uploaded',
        updated_at       text not null default ''
    )
    """,
)

_COLUMNS = (
    "doc_id",
    "source_uri",
    "content_hash",
    "chunk_count",
    "ingested_at",
    "pipeline_version",
    "stage",
    "updated_at",
)

# **每进程只建一次表**：`setup()` 是一句 `create table if not exists`，但读/写都是
# 每请求新建一个 store 实例，逐次都打卡就变成每请求一次 DDL。用进程级的名字集合兜住。
_READY: set[str] = set()
_READY_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _to_record(row: dict[str, Any]) -> DocRecord:
    return DocRecord(
        doc_id=str(row.get("doc_id") or ""),
        source_uri=str(row.get("source_uri") or ""),
        content_hash=str(row.get("content_hash") or ""),
        chunk_count=int(row.get("chunk_count") or 0),
        ingested_at=int(row.get("ingested_at") or 0),
        pipeline_version=str(row.get("pipeline_version") or ""),
        stage=str(row.get("stage") or ""),
    )


class SqliteDocumentStore:
    """单机版：一个文件、WAL、短事务。**多进程读没问题**（CLI 与 serve 可以同时开着）。"""

    backend = "sqlite"
    # 占位符：SQLite 用 `?`、PG 用 `%s` —— 下面的 SQL 两种库共用一份
    placeholder = "?"

    def __init__(self, path: str | Path = DEFAULT_SQLITE_PATH) -> None:
        self.path = Path(path)
        self.name = f"sqlite:{self.path}"
        # 建表打卡的键（PG 那份换成连接串，见 `PgDocumentStore`）
        self._ready_key = self.name

    def _connect(self) -> Any:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.execute("pragma journal_mode=wal")
        return conn

    @contextmanager
    def _session(self) -> Iterator[Any]:
        """开一次连接、做完就关。**两条路都显式关**（见模块说明）。"""
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:  # noqa: BLE001 —— 回滚后原样抛，别吞
            conn.rollback()
            raise
        finally:
            conn.close()

    def setup(self) -> None:
        """建表，幂等。"""
        with self._session() as conn:
            for statement in _SCHEMA:
                conn.execute(statement)
        _READY.add(self._ready_key)

    def _ready(self) -> None:
        """读/写前的自愈：这个进程还没建过表就先建一次（建过就直接返回）。"""
        if self._ready_key in _READY:
            return
        with _READY_LOCK:
            if self._ready_key not in _READY:
                self.setup()

    # ---- 同步实现（异步门面在下面）----

    def sync_all(self) -> dict[str, DocRecord]:
        self._ready()
        sql = f"select {', '.join(_COLUMNS)} from document_registry"
        with self._session() as conn:
            rows = conn.execute(sql).fetchall()
        return {row[0]: _to_record(dict(zip(_COLUMNS, row, strict=False))) for row in rows}

    def sync_stages(self) -> dict[str, str]:
        """只取 `doc_id → stage`：列表页要的就是这个，别的列一个都不读。"""
        self._ready()
        with self._session() as conn:
            rows = conn.execute("select doc_id, stage from document_registry").fetchall()
        return {str(row[0]): str(row[1] or "") for row in rows}

    def sync_get(self, doc_id: str) -> DocRecord | None:
        self._ready()
        sql = (
            f"select {', '.join(_COLUMNS)} from document_registry where doc_id = {self.placeholder}"
        )
        with self._session() as conn:
            row = conn.execute(sql, (doc_id,)).fetchone()
        return _to_record(dict(zip(_COLUMNS, row, strict=False))) if row else None

    def sync_upsert(self, records: Iterable[DocRecord]) -> int:
        """写注册表行（入库那一步）。`stage` 为空 = 不动原来那一列。"""
        items = list(records)
        if not items:
            return 0
        self._ready()
        plain = [column for column in _COLUMNS[1:] if column != "stage"]
        updates = ", ".join(f"{column}=excluded.{column}" for column in plain)
        updates += (
            ", stage=case when excluded.stage <> '' then excluded.stage"
            " else document_registry.stage end"
        )
        sql = (
            f"insert into document_registry ({', '.join(_COLUMNS)})"
            f" values ({', '.join([self.placeholder] * len(_COLUMNS))})"
            f" on conflict(doc_id) do update set {updates}"
        )
        at = _now()
        with self._session() as conn:
            for record in items:
                conn.execute(
                    sql,
                    (
                        record.doc_id,
                        record.source_uri,
                        record.content_hash,
                        int(record.chunk_count),
                        int(record.ingested_at),
                        record.pipeline_version,
                        record.stage,
                        at,
                    ),
                )
        return len(items)

    def sync_set_stage(
        self, doc_id: str, stage: str, source_uri: str = "", content_hash: str = ""
    ) -> None:
        """推进/回退一份文档的状态。**没有这一行就先建一行**（上传就该有行）。

        `source_uri` / `content_hash` 只在这一行还不存在（或原来是空）时写进去 ——
        推进状态时不该把已有的值覆盖成空。

        **content_hash 从上传那一步就该有值**（2026-10-10 补）：以前只有入库（`upsert`）
        会写它，于是「解析完但没入库」的文档这一列是空串 —— 而 `plan_incremental()`
        判「这份变了没有」用的正是它（`incremental.py:139`），空值会被判成"变了"，
        批量 ingest 时白白重跑一遍 OCR。
        """
        self._ready()
        sql = (
            "insert into document_registry (doc_id, source_uri, content_hash, stage, updated_at)"
            f" values ({self.placeholder}, {self.placeholder}, {self.placeholder},"
            f" {self.placeholder}, {self.placeholder})"
            " on conflict(doc_id) do update set"
            " stage=excluded.stage, updated_at=excluded.updated_at,"
            " source_uri=case when excluded.source_uri <> '' then excluded.source_uri"
            " else document_registry.source_uri end,"
            " content_hash=case when excluded.content_hash <> '' then excluded.content_hash"
            " else document_registry.content_hash end"
        )
        with self._session() as conn:
            conn.execute(sql, (doc_id, source_uri, content_hash, stage, _now()))

    def sync_delete(self, doc_ids: Iterable[str]) -> int:
        items = [doc_id for doc_id in doc_ids if doc_id]
        if not items:
            return 0
        self._ready()
        sql = f"delete from document_registry where doc_id = {self.placeholder}"
        with self._session() as conn:
            for doc_id in items:
                conn.execute(sql, (doc_id,))
        return len(items)

    def sync_drop(self) -> None:
        """删表。只给测试与重建用 —— 生产环境没有「顺手删库」这种操作。"""
        with self._session() as conn:
            conn.execute("drop table if exists document_registry")
        _READY.discard(self._ready_key)

    # ---- 异步门面：与原来那个 Milvus 版同名同义，调用点不用改 ----

    async def ensure_ready(self) -> None:
        await asyncio.to_thread(self.setup)

    async def all(self) -> dict[str, DocRecord]:
        return await asyncio.to_thread(self.sync_all)

    async def stages(self) -> dict[str, str]:
        return await asyncio.to_thread(self.sync_stages)

    async def get(self, doc_id: str) -> DocRecord | None:
        return await asyncio.to_thread(self.sync_get, doc_id)

    async def upsert(self, records: Iterable[DocRecord]) -> int:
        return await asyncio.to_thread(self.sync_upsert, records)

    async def set_stage(
        self, doc_id: str, stage: str, *, source_uri: str = "", content_hash: str = ""
    ) -> None:
        await asyncio.to_thread(self.sync_set_stage, doc_id, stage, source_uri, content_hash)

    async def delete(self, doc_ids: Iterable[str]) -> int:
        return await asyncio.to_thread(self.sync_delete, doc_ids)

    async def drop(self) -> None:
        await asyncio.to_thread(self.sync_drop)


class PgDocumentStore(SqliteDocumentStore):
    """多 worker 版：与 chat / 日志 / 模型选型同库。SQL 与 SQLite 那份**一模一样**，
    只有连接方式与占位符不同（`%s`）。"""

    backend = "postgres"
    placeholder = "%s"

    def __init__(self, conninfo: str) -> None:
        self.conninfo = conninfo
        self.name = "postgres:document_registry"
        # 打卡键用连接串（含库名 / schema）：一个进程连两个库时才不会互相以为建过表
        self._ready_key = conninfo

    def _connect(self) -> Any:  # pragma: no cover —— 需要真库才跑到，用例在 SQLite 那侧
        import psycopg

        # 短连接，用完就关（`_session` 负责）。**试过进程级连接池，撤了**：在
        # `getconn()`/`close()` 这套用法下会 PoolTimeout 卡住（实测），而省下的那点
        # 连接开销（本机 ~0.2s）不值得换一个会挂的池 —— 要池化就整条链改用
        # `Persistence.pool` 那套 async 连接，不是在这里塞一个同步池。
        # autocommit：`_session` 里的 commit/rollback 在它下面变成空操作，两条路行为一致。
        return psycopg.connect(self.conninfo, autocommit=True)


def build_document_store(settings: Any, *, backend: str | None = None) -> Any:
    """按**选库开关**建（见 `rag/persistence/backend.py`）。**不主动探活。**

    优先级：`documents.backend` 显式写了 sqlite/postgres 就听它；留 `auto`（默认）就往上问
    统一开关 `persistence.backend`（.env 里的 `RAG_DB_BACKEND`）。
    """
    from rag.persistence.backend import resolve_backend

    cfg = getattr(settings, "documents", None)
    if resolve_backend(settings, backend or getattr(cfg, "backend", "auto")) == "postgres":
        from rag.persistence.pg import build_persistence

        return PgDocumentStore(build_persistence(settings).conninfo())
    sqlite_path = getattr(cfg, "sqlite_path", "") or DEFAULT_SQLITE_PATH
    return SqliteDocumentStore(sqlite_path)
