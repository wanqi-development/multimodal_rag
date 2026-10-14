"""会话账本：`list` / `prune` / `--keep` 的选择。

全部用替身连接（不连 PG）：这一层要验的是**SQL 问对了没、删的时候顺序对不对、
「在等人批准」有没有被标出来** —— 真连库的部分在真机验证里做。
"""

from __future__ import annotations

from typing import Any

from rag.persistence.sessions import SessionStore, SessionSummary, select_prune_targets


class _Cursor:
    def __init__(self, rows: list[tuple], rowcount: int = 0) -> None:
        self._rows = rows
        self.rowcount = rowcount

    def fetchall(self) -> list[tuple]:
        return self._rows


class _Conn:
    """替身连接：按 SQL 文本分派，记下每条语句与参数。"""

    def __init__(self, summary=(), writes=(), deleted=None) -> None:
        self._summary = list(summary)
        self._writes = list(writes)
        self._deleted = deleted or {}
        self.statements: list[tuple[str, Any]] = []

    def __enter__(self) -> _Conn:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def execute(self, query: Any, params: Any = None) -> _Cursor:
        # `sql.Composed`（带 Identifier 的那条）要用 as_string 才拿得到 SQL 文本，
        # 直接 str() 出来的是 repr —— 替身得两种都认
        text = (query.as_string(None) if hasattr(query, "as_string") else str(query)).strip()
        self.statements.append((text, params))
        if text.startswith("select") and "from checkpoint_writes" in text:
            return _Cursor(self._writes)
        if text.startswith("select") and "from checkpoints" in text:
            return _Cursor(self._summary)
        for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints", "store"):
            if f"delete from {table}" in text:
                return _Cursor([], rowcount=self._deleted.get(table, 0))
        return _Cursor([])


def _store(conn: _Conn) -> SessionStore:
    return SessionStore("postgresql://x", connect=lambda _conninfo: conn)


def test_list_marks_who_is_waiting_for_approval() -> None:
    """停在等人批准的那次会话**必须看得见** —— 删了它就等于把待批准的事一起扔掉。"""
    conn = _Conn(
        summary=[("s-busy", 12, "0002"), ("s-done", 30, "0001")],
        writes=[("s-busy", 9, 1), ("s-done", 23, 0)],
    )

    sessions = _store(conn).list()

    assert [item.thread_id for item in sessions] == ["s-busy", "s-done"], "最新一版在前"
    assert sessions[0].waiting_for_approval is True
    assert sessions[0].interrupts == 1 and sessions[0].writes == 9
    assert sessions[1].waiting_for_approval is False


def test_a_session_without_writes_still_shows_up() -> None:
    """只有版本、没有待写行的会话（跑完就没人动过）也得列出来，别让它成为隐形的。"""
    conn = _Conn(summary=[("s-only", 3, "0003")], writes=[])

    sessions = _store(conn).list()

    assert [(s.thread_id, s.writes, s.interrupts) for s in sessions] == [("s-only", 0, 0)]


def test_prune_deletes_children_first_and_reports_the_counts() -> None:
    """三张表 + 引用表都要清；顺序是先叶子后主表（免得留下孤儿行）。"""
    conn = _Conn(
        deleted={
            "checkpoint_writes": 230,
            "checkpoint_blobs": 70,
            "checkpoints": 99,
            "store": 1,
        }
    )

    counts = _store(conn).prune(["s-1", "s-2", "s-1"])

    assert counts == {
        "checkpoint_writes": 230,
        "checkpoint_blobs": 70,
        "checkpoints": 99,
        "store": 1,
    }
    order = [
        text.split("where")[0].split()[-1]
        for text, _ in conn.statements
        if text.startswith("delete")
    ]
    assert order == ["checkpoint_writes", "checkpoint_blobs", "checkpoints", "store"]
    assert conn.statements[0][1] == (["s-1", "s-2"],), "重复的 id 只该传一次"


def test_prune_without_ids_does_nothing() -> None:
    conn = _Conn()

    assert _store(conn).prune([]) == {}
    assert conn.statements == [], "没给 id 就不该发任何语句"


def test_ensure_schema_quotes_the_name() -> None:
    """schema 名进的是 DDL，**必须用 Identifier 引起来**（拼字符串就是注入面）。"""
    conn = _Conn()

    _store(conn).ensure_schema("rag")

    assert conn.statements[0][0] == 'create schema if not exists "rag"'


def test_keep_only_the_newest_n() -> None:
    sessions = [
        SessionSummary("old", 1, 0, 0, "0001"),
        SessionSummary("new", 1, 0, 0, "0003"),
        SessionSummary("mid", 1, 0, 0, "0002"),
    ]

    assert [item.thread_id for item in select_prune_targets(sessions, keep=1)] == ["mid", "old"]
    assert select_prune_targets(sessions, keep=3) == []
    assert len(select_prune_targets(sessions, keep=0)) == 3, "keep=0 = 全删"
