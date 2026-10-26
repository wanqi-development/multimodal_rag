"""上传件台账：`upload_meta` 一行 = 一份上传件的**附属信息**（2026-10-09）。

它原先是 `data/uploads/_index.json`（一个 JSON 文件整体读-改-写）。多 worker 下那套不行：
两个人同时上传就会**互相覆盖**，丢的是真名与来源，表现出来却是"页面显示的文件名不对"，
最难查。挪进数据库后每份上传各写自己那一行（`on conflict(doc_id) do update`），
不同文件互不干扰，同一文件并发上传天然幂等（主键就是 doc_id）。

**单机也走这张表**（使用者 2026-10-09 定「PG + 单机 sqllite 一起做」）：DB 的选择沿用
`documents.backend` 那套 —— `auto` = 配了 PG 用 PG，否则单机 SQLite
（`data/documents.sqlite3`，与文档注册表同一个文件、零配置）。这样 local 与 minio
两个后端只有 blob 那一层不同，台账只有一份实现。

字段的取舍（对着 `data/uploads/_index.json` 的真实结构定的）：

* `name`（落盘名 = **对象键**）与 `size` 也进表：搬到对象存储后，列表页不可能靠列桶去
  反推文件名，`stat` 也变成一次 HEAD 往返；
* `attachments` 存 **JSON 文本**而不是子表：它只用在"删文档时把抓来的图一起清掉"，
  从来不需要"按附件反查文档"，拆子表只会多一次 join；
* `replaced` 不进表：那是 `save()` 当次的返回值，不是持久状态。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag.persistence.documents import DEFAULT_SQLITE_PATH

# 与 documents.py 同一个做法：**每进程只建一次表**（读/写都按请求新建 store 实例，
# 逐次都打卡就变成每请求一次 DDL）。
_READY: set[str] = set()
_READY_LOCK = threading.Lock()

# 两张库共用同一段 DDL（与 documents.py 同一个理由：全是文本/整数列）
_SCHEMA = (
    """
    create table if not exists upload_meta (
        doc_id       text primary key,
        name         text not null default '',
        display_name text not null default '',
        sha1         text not null default '',
        size         integer not null default 0,
        source_url   text not null default '',
        final_url    text not null default '',
        content_type text not null default '',
        created_at   integer not null default 0,
        attachments  text not null default '[]',
        updated_at   text not null default ''
    )
    """,
    # `by_display()` 要按真名查（页面问「同名的是不是已经有一份」），给它一条索引
    "create index if not exists upload_meta_display on upload_meta (display_name)",
)

_COLUMNS = (
    "doc_id",
    "name",
    "display_name",
    "sha1",
    "size",
    "source_url",
    "final_url",
    "content_type",
    "created_at",
    "attachments",
    "updated_at",
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _row(row: Any) -> dict[str, Any]:
    item = dict(zip(_COLUMNS, row, strict=False))
    try:
        item["attachments"] = list(json.loads(item.get("attachments") or "[]"))
    except ValueError:
        item["attachments"] = []
    item["size"] = int(item.get("size") or 0)
    item["created_at"] = int(item.get("created_at") or 0)
    return item


class SqliteUploadMetaStore:
    """单机版（WAL）。表与 `document_registry` 同库同文件。"""

    backend = "sqlite"
    placeholder = "?"
    _key_prefix = "sqlite-uploads"

    def __init__(self, path: str | Path = DEFAULT_SQLITE_PATH) -> None:
        self.path = Path(path)
        self.name = f"sqlite:{self.path}#upload_meta"

    def _connect(self) -> Any:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.execute("pragma journal_mode=wal")
        return conn

    @contextmanager
    def _session(self) -> Iterator[Any]:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:  # noqa: BLE001 —— 回滚后原样抛
            conn.rollback()
            raise
        finally:
            conn.close()

    @property
    def _ready_key(self) -> str:
        return f"{self._key_prefix}:{self.name}"

    def setup(self) -> None:
        with self._session() as conn:
            for statement in _SCHEMA:
                conn.execute(statement)
        _READY.add(self._ready_key)

    def _ready(self) -> None:
        if self._ready_key in _READY:
            return
        with _READY_LOCK:
            if self._ready_key not in _READY:
                self.setup()

    # ---- 同步实现 ----

    def all(self) -> dict[str, dict[str, Any]]:
        """全部台账行，键是 doc_id。"""
        self._ready()
        with self._session() as conn:
            rows = conn.execute(f"select {', '.join(_COLUMNS)} from upload_meta").fetchall()
        return {str(item["doc_id"]): item for item in (_row(row) for row in rows)}

    def get(self, doc_id: str) -> dict[str, Any] | None:
        self._ready()
        sql = f"select {', '.join(_COLUMNS)} from upload_meta where doc_id = {self.placeholder}"
        with self._session() as conn:
            found = conn.execute(sql, (doc_id,)).fetchone()
        return _row(found) if found else None

    def by_display(self, display_name: str) -> list[dict[str, Any]]:
        """同一个**真名**已经落过的版本（页面拿它问「覆盖还是保留两份」）。"""
        self._ready()
        sql = (
            f"select {', '.join(_COLUMNS)} from upload_meta"
            f" where display_name = {self.placeholder} order by created_at"
        )
        with self._session() as conn:
            rows = conn.execute(sql, (display_name,)).fetchall()
        return [_row(row) for row in rows]

    def upsert(self, items: Iterable[dict[str, Any]], *, only_if_absent: bool = False) -> int:
        """写台账。`only_if_absent=True` 时**只补空位**（导入老 `_index.json` 用这个，
        绝不覆盖已经写好的行）。"""
        rows = list(items)
        if not rows:
            return 0
        self._ready()
        conflict = (
            "do nothing"
            if only_if_absent
            else "do update set "
            + ", ".join(f"{column}=excluded.{column}" for column in _COLUMNS[1:])
        )
        sql = (
            f"insert into upload_meta ({', '.join(_COLUMNS)})"
            f" values ({', '.join([self.placeholder] * len(_COLUMNS))})"
            f" on conflict(doc_id) {conflict}"
        )
        at = _now()
        with self._session() as conn:
            for item in rows:
                conn.execute(
                    sql,
                    (
                        str(item.get("doc_id") or ""),
                        str(item.get("name") or ""),
                        str(item.get("display_name") or ""),
                        str(item.get("sha1") or ""),
                        int(item.get("size") or 0),
                        str(item.get("source_url") or ""),
                        str(item.get("final_url") or ""),
                        str(item.get("content_type") or ""),
                        int(item.get("created_at") or 0),
                        json.dumps(list(item.get("attachments") or []), ensure_ascii=False),
                        at,
                    ),
                )
        return len(rows)

    def delete(self, doc_ids: Iterable[str]) -> int:
        items = [doc_id for doc_id in doc_ids if doc_id]
        if not items:
            return 0
        self._ready()
        sql = f"delete from upload_meta where doc_id = {self.placeholder}"
        with self._session() as conn:
            for doc_id in items:
                conn.execute(sql, (doc_id,))
        return len(items)

    def drop(self) -> None:
        """删表。只给测试与重建用。"""
        with self._session() as conn:
            conn.execute("drop table if exists upload_meta")
        _READY.discard(self._ready_key)


class PgUploadMetaStore(SqliteUploadMetaStore):
    """多 worker 版：与 chat / 日志 / 模型选型 / 文档注册表同库。SQL 与 SQLite 那份一样，
    只有连接方式与占位符不同（`%s`）。"""

    backend = "postgres"
    placeholder = "%s"
    _key_prefix = "pg-uploads"

    def __init__(self, conninfo: str) -> None:
        self.conninfo = conninfo
        self.name = "postgres:upload_meta"

    def _connect(self) -> Any:  # pragma: no cover —— 要真库才跑到，用例在 SQLite 那侧
        import psycopg

        return psycopg.connect(self.conninfo, autocommit=True)


def build_upload_meta_store(settings: Any, *, backend: str | None = None) -> Any:
    """按 `documents.backend` 建（**与文档注册表同一个库**，不另开一套配置）。

    留 `auto`（默认）就往上问统一开关 `persistence.backend`（.env 的 `RAG_DB_BACKEND`），
    见 `rag/persistence/backend.py`。
    """
    from rag.persistence.backend import resolve_backend

    cfg = getattr(settings, "documents", None)
    if resolve_backend(settings, backend or getattr(cfg, "backend", "auto")) == "postgres":
        from rag.persistence.pg import build_persistence

        return PgUploadMetaStore(build_persistence(settings).conninfo())
    sqlite_path = getattr(cfg, "sqlite_path", "") or DEFAULT_SQLITE_PATH
    return SqliteUploadMetaStore(sqlite_path)
