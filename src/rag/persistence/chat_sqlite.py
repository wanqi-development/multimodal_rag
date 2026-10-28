"""UI 会话的 SQLite 存储：本地 profile 的三张会话表。"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, TypeVar

from rag.persistence.chat import (
    ROLE_AI,
    ROLE_HUMAN,
    STATUS_ANSWERED,
)

T = TypeVar("T")

_NOW = "strftime('%Y-%m-%dT%H:%M:%fZ','now')"

SCHEMA = (
    f"""
    create table if not exists chat_session (
        session_id text primary key,
        owner text not null default 'local',
        title text not null default '',
        mode text not null default 'chat',
        thread_id text,
        created_at text not null default ({_NOW}),
        updated_at text not null default ({_NOW})
    )
    """,
    "create index if not exists chat_session_updated_idx on chat_session (updated_at desc)",
    """
    create table if not exists chat_message (
        message_id integer primary key autoincrement,
        session_id text not null references chat_session(session_id) on delete cascade,
        seq integer not null,
        role text not null,
        content text not null default '',
        status text not null default 'answered',
        created_at text not null default (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
        unique (session_id, seq)
    )
    """,
    """
    create table if not exists token_usage (
        usage_id integer primary key autoincrement,
        session_id text not null references chat_session(session_id) on delete cascade,
        message_id integer references chat_message(message_id) on delete cascade,
        model text not null default '',
        calls integer not null default 1,
        input_tokens integer,
        output_tokens integer,
        total_tokens integer,
        created_at text not null default (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
    )
    """,
    "create index if not exists token_usage_session_idx on token_usage (session_id)",
    "create index if not exists token_usage_created_idx on token_usage (created_at desc)",
)


class SqliteChatStore:
    """与 PostgreSQL `ChatStore` 同一套异步接口，连接按操作短开短关。"""

    backend = "sqlite"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @property
    def name(self) -> str:
        return f"sqlite:{self.path}"

    @contextlib.contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma foreign_keys = on")
        conn.execute("pragma busy_timeout = 5000")
        try:
            yield conn
        finally:
            conn.close()

    def _read(self, work: Callable[[sqlite3.Connection], T]) -> T:
        with self._connection() as conn:
            return work(conn)

    def _write(self, work: Callable[[sqlite3.Connection], T]) -> T:
        with self._connection() as conn:
            conn.execute("begin immediate")
            try:
                result = work(conn)
                conn.execute("commit")
                return result
            except Exception:
                conn.execute("rollback")
                raise

    async def setup(self) -> None:
        await asyncio.to_thread(self._setup_sync)

    def _setup_sync(self) -> None:
        with self._connection() as conn:
            conn.execute("pragma journal_mode = wal")
            conn.execute("pragma synchronous = normal")
            for statement in SCHEMA:
                conn.execute(statement)

    async def create(
        self,
        *,
        session_id: str,
        owner: str = "local",
        title: str = "",
        mode: str = "chat",
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        def run(conn: sqlite3.Connection) -> dict[str, Any]:
            conn.execute(
                "insert into chat_session "
                "(session_id, owner, title, mode, thread_id, created_at, updated_at) "
                f"values (?, ?, ?, ?, ?, {_NOW}, {_NOW}) "
                f"on conflict(session_id) do update set updated_at = {_NOW}",
                (session_id, owner, title, mode, thread_id),
            )
            return {
                "session_id": session_id,
                "owner": owner,
                "title": title,
                "mode": mode,
                "thread_id": thread_id,
            }

        return await asyncio.to_thread(self._write, run)

    async def touch(
        self,
        session_id: str,
        *,
        title: str | None = None,
        mode: str | None = None,
        thread_id: str | None = None,
    ) -> None:
        def run(conn: sqlite3.Connection) -> None:
            conn.execute(
                f"update chat_session set updated_at = {_NOW},"
                " title = coalesce(nullif(title, ''), ?),"
                " mode = coalesce(?, mode),"
                " thread_id = coalesce(?, thread_id)"
                " where session_id = ?",
                (title, mode, thread_id, session_id),
            )

        await asyncio.to_thread(self._write, run)

    async def list(self, *, limit: int = 20) -> list[dict[str, Any]]:
        def run(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            rows = conn.execute(_LIST_SQL, (limit,)).fetchall()
            return [dict(row) for row in rows]

        return await asyncio.to_thread(self._read, run)

    async def get(self, session_id: str) -> dict[str, Any] | None:
        def run(conn: sqlite3.Connection) -> dict[str, Any] | None:
            row = conn.execute(
                "select session_id, title, mode, thread_id, created_at, updated_at"
                " from chat_session where session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            session = dict(row)
            session["messages"] = [
                dict(item)
                for item in conn.execute(
                    "select message_id, seq, role, content, status, created_at"
                    " from chat_message where session_id = ? order by seq",
                    (session_id,),
                ).fetchall()
            ]
            per_message = {
                int(item["message_id"]): dict(item)
                for item in conn.execute(
                    "select message_id, calls, input_tokens, output_tokens, total_tokens"
                    " from token_usage where session_id = ?",
                    (session_id,),
                ).fetchall()
                if item["message_id"] is not None
            }
            for item in session["messages"]:
                item["usage"] = per_message.get(int(item["message_id"]))
            session["usage"] = _usage_total(conn, session_id)
            return session

        return await asyncio.to_thread(self._read, run)

    async def delete(self, session_id: str) -> bool:
        def run(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute("delete from chat_session where session_id = ?", (session_id,))
            return (cursor.rowcount or 0) > 0

        return await asyncio.to_thread(self._write, run)

    async def summary(self) -> dict[str, Any]:
        def run(conn: sqlite3.Connection) -> dict[str, Any]:
            row = conn.execute(_SUMMARY_SQL).fetchone()
            return dict(row) if row else {"sessions": 0, "turns": 0}

        return await asyncio.to_thread(self._read, run)

    async def append_turn(
        self,
        session_id: str,
        *,
        question: str,
        answer: str = "",
        mode: str = "chat",
        status: str = STATUS_ANSWERED,
        thread_id: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> int:
        def run(conn: sqlite3.Connection) -> int:
            row = conn.execute(
                "select coalesce(max(seq), 0) as last_seq from chat_message where session_id = ?",
                (session_id,),
            ).fetchone()
            base = int((row["last_seq"] if row else 0) or 0)
            conn.execute(
                "insert into chat_message (session_id, seq, role, content, status)"
                " values (?, ?, ?, ?, ?), (?, ?, ?, ?, ?)",
                (
                    session_id,
                    base + 1,
                    ROLE_HUMAN,
                    question,
                    STATUS_ANSWERED,
                    session_id,
                    base + 2,
                    ROLE_AI,
                    answer,
                    status,
                ),
            )
            conn.execute(
                f"update chat_session set updated_at = {_NOW}, mode = ?,"
                " thread_id = coalesce(?, thread_id) where session_id = ?",
                (mode, thread_id, session_id),
            )
            if usage is not None:
                _record_usage(conn, session_id, base + 2, usage)
            return base + 2

        return await asyncio.to_thread(self._write, run)

    async def update_pending(
        self,
        session_id: str,
        *,
        answer: str,
        status: str = STATUS_ANSWERED,
        usage: dict[str, Any] | None = None,
    ) -> bool:
        def run(conn: sqlite3.Connection) -> bool:
            row = conn.execute(
                "select seq from chat_message where session_id = ? and role = ?"
                " and status <> ? order by seq desc limit 1",
                (session_id, ROLE_AI, STATUS_ANSWERED),
            ).fetchone()
            if row is None:
                return False
            seq = int(row["seq"] or 0)
            conn.execute(
                "update chat_message set content = ?, status = ? where session_id = ? and seq = ?",
                (answer, status, session_id, seq),
            )
            conn.execute(
                f"update chat_session set updated_at = {_NOW} where session_id = ?",
                (session_id,),
            )
            if usage is not None:
                _record_usage(conn, session_id, seq, usage)
            return True

        return await asyncio.to_thread(self._write, run)


_LIST_SQL = """
select s.session_id, s.title, s.mode, s.thread_id, s.created_at, s.updated_at,
       coalesce(m.turns, 0)      as turns,
       coalesce(u.tokens_in, 0)  as input_tokens,
       coalesce(u.tokens_out, 0) as output_tokens,
       coalesce(u.tokens_all, 0) as total_tokens,
       coalesce(u.calls, 0)      as calls
  from chat_session s
  left join (
       select session_id,
              sum(case when role = 'ai' then 1 else 0 end) as turns
         from chat_message group by session_id
  ) m on m.session_id = s.session_id
  left join (
       select session_id, sum(input_tokens) as tokens_in, sum(output_tokens) as tokens_out,
              sum(total_tokens) as tokens_all, sum(calls) as calls
         from token_usage group by session_id
  ) u on u.session_id = s.session_id
 order by s.updated_at desc
 limit ?
"""

_SUMMARY_SQL = """
select (select count(*) from chat_session) as sessions,
       (select count(*) from chat_message where role = 'ai') as turns,
       (select coalesce(sum(input_tokens), 0) from token_usage) as input_tokens,
       (select coalesce(sum(output_tokens), 0) from token_usage) as output_tokens,
       (select coalesce(sum(total_tokens), 0) from token_usage) as total_tokens
"""


def _usage_total(conn: sqlite3.Connection, session_id: str) -> dict[str, Any]:
    row = conn.execute(
        "select coalesce(sum(input_tokens), 0) as tokens_in,"
        " coalesce(sum(output_tokens), 0) as tokens_out,"
        " coalesce(sum(total_tokens), 0) as tokens_all,"
        " coalesce(sum(calls), 0) as calls"
        " from token_usage where session_id = ?",
        (session_id,),
    ).fetchone()
    return {
        "input_tokens": int(row["tokens_in"] or 0) if row else 0,
        "output_tokens": int(row["tokens_out"] or 0) if row else 0,
        "total_tokens": int(row["tokens_all"] or 0) if row else 0,
        "calls": int(row["calls"] or 0) if row else 0,
    }


def _record_usage(
    conn: sqlite3.Connection,
    session_id: str,
    seq: int,
    usage: dict[str, Any],
) -> None:
    row = conn.execute(
        "select message_id from chat_message where session_id = ? and seq = ?",
        (session_id, seq),
    ).fetchone()
    conn.execute(
        "insert into token_usage (session_id, message_id, model, calls,"
        " input_tokens, output_tokens, total_tokens) values (?, ?, ?, ?, ?, ?, ?)",
        (
            session_id,
            int(row["message_id"]) if row and row["message_id"] is not None else None,
            str(usage.get("model") or ""),
            int(usage.get("calls") or 1),
            usage.get("input_tokens"),
            usage.get("output_tokens"),
            usage.get("total_tokens"),
        ),
    )
