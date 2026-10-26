"""UI 会话的三张表：落一轮、把中断那轮补上、列表与 token 合计。

**全部离线**：连接是替身，它只按调用顺序把预先准备好的行交出去。SQL 本身在真机探针里
对着真 PG 跑过（见 CHANGELOG），这里钉的是编排：写几行、按什么顺序、更新的目标对不对。
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from rag.persistence.chat import (
    STATUS_ANSWERED,
    STATUS_INTERRUPTED,
    ChatStore,
)


class _Cursor:
    def __init__(self, rows: list[Any], rowcount: int | None = None) -> None:
        self._rows = list(rows)
        self.rowcount = rowcount if rowcount is not None else len(rows)

    async def fetchall(self) -> list[Any]:
        return list(self._rows)

    async def fetchone(self) -> Any:
        return self._rows[0] if self._rows else None


class _Conn:
    """按顺序回答：第 n 次 `execute` 拿第 n 组预置的行，多出来的都当空结果。

    预置的是**字典行** —— 真库那边连接池配的是 `dict_row`，替身得跟它一致，
    否则「按列名取」这件事在测试里根本验不到（这个错真机上炸过一次）。
    """

    def __init__(self, *answers: Any, rowcount: int = 0) -> None:
        self.statements: list[tuple[str, Any]] = []
        self._answers = list(answers)
        self._rowcount = rowcount

    async def execute(self, sql: str, params: Any = None) -> _Cursor:
        self.statements.append((" ".join(sql.split()), params))
        rows = self._answers.pop(0) if self._answers else []
        return _Cursor(rows, rowcount=self._rowcount if not rows else None)

    @property
    def sql(self) -> list[str]:
        return [statement for statement, _ in self.statements]


def _store(conn: _Conn) -> ChatStore:
    @contextlib.asynccontextmanager
    async def connect():
        yield conn

    return ChatStore(connect=connect)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_append_turn_writes_both_messages_and_the_token_row() -> None:
    """一轮 = human 一行 + ai 一行 + 一条 token 账。三件事一次做完，不留半截。"""
    conn = _Conn([{"last_seq": 4}], [{"message_id": 7}])

    seq = _run(
        _store(conn).append_turn(
            "s1",
            question="问什么",
            answer="答什么",
            mode="agent",
            thread_id="t1",
            usage={
                "model": "deepseek-chat",
                "calls": 3,
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
            },
        )
    )

    assert seq == 6, "上一轮到 4，这轮就是 5(human) 与 6(ai)"
    insert = next(sql for sql in conn.sql if sql.startswith("insert into chat_message"))
    assert "values (%s, %s, %s, %s, %s), (%s, %s, %s, %s, %s)" in insert, "两条消息一条语句"
    usage = next(sql for sql in conn.sql if sql.startswith("insert into token_usage"))
    assert "calls" in usage and "total_tokens" in usage


def test_an_interrupted_turn_still_gets_a_row() -> None:
    """中断那轮**必须落行**（content 空、status=interrupted）—— 不落的话重开这段会话
    就看不出「上次停在等人批准」，而载荷要现去 checkpointer 取。"""
    conn = _Conn([{"last_seq": 0}])

    _run(
        _store(conn).append_turn(
            "s1", question="写个文件", status=STATUS_INTERRUPTED, mode="agent", thread_id="t1"
        )
    )

    _, params = next(
        item for item in conn.statements if item[0].startswith("insert into chat_message")
    )
    assert params[9] == STATUS_INTERRUPTED, "ai 那行的 status"
    assert params[8] == "", "中断时没有答案正文"


def test_update_pending_fills_the_latest_unfinished_answer() -> None:
    """恢复那轮**更新**已有行，不是再插一行 —— 否则前端会看到两个答复。"""
    conn = _Conn([{"seq": 6}])

    ok = _run(_store(conn).update_pending("s1", answer="写好了 [1]", usage={"total_tokens": 5}))

    assert ok is True
    update = next(sql for sql in conn.sql if sql.startswith("update chat_message"))
    assert "content = %s" in update and "status = %s" in update
    _, params = next(item for item in conn.statements if item[0].startswith("update chat_message"))
    assert params[0] == "写好了 [1]" and params[2] == "s1" and params[3] == 6
    assert not any(sql.startswith("insert into chat_message") for sql in conn.sql)


def test_update_pending_does_nothing_when_nothing_is_waiting() -> None:
    conn = _Conn([])

    assert _run(_store(conn).update_pending("s1", answer="x")) is False
    assert not any(sql.startswith("update chat_message") for sql in conn.sql)


def test_list_hands_back_the_token_totals() -> None:
    """列表里就要有 token 合计，界面不用为此再来一次请求。"""
    conn = _Conn(
        [
            {
                "session_id": "s1",
                "title": "问了什么",
                "mode": "agent",
                "thread_id": "t1",
                "turns": 2,
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
                "calls": 3,
            }
        ]
    )

    rows = _run(_store(conn).list(limit=5))

    assert rows[0]["total_tokens"] == 30 and rows[0]["turns"] == 2
    sql, params = conn.statements[0]
    assert "left join" in sql and params == (5,), "limit 要走参数"


def test_get_returns_none_for_an_unknown_session() -> None:
    assert _run(_store(_Conn([])).get("nope")) is None


def test_delete_reports_whether_the_row_existed() -> None:
    assert _run(_store(_Conn(rowcount=1)).delete("s1")) is True
    assert _run(_store(_Conn(rowcount=0)).delete("s1")) is False


def test_the_ddl_has_no_schema_prefix() -> None:
    """三张表**不带 schema 前缀**：隔离靠连接串里的 `search_path`（与日志表同一个规矩）。
    带上前缀就等于把 schema 写死在代码里，换一个 schema 跑就得改代码。"""
    from rag.persistence.chat import SCHEMA

    ddl = "\n".join(SCHEMA)

    assert "create table if not exists chat_session" in ddl
    assert "create table if not exists chat_message" in ddl
    assert "create table if not exists token_usage" in ddl
    assert "rag." not in ddl
    assert "on delete cascade" in ddl, "删会话要能带走它的消息与 token 行"
    assert "add column if not exists owner" in ddl, (
        "`create table if not exists` 不给已有的表加列 —— 已经建过的库要靠这句 alter 补上"
    )


def test_owner_defaults_to_local_until_there_are_users() -> None:
    """用户系统还没做，`owner` 恒为 `'local'` —— 但**列现在就加上**，
    免得到时有数据了再迁。写进去，不按它过滤。"""
    conn = _Conn()

    _run(_store(conn).create(session_id="s1", title="标题"))

    sql, params = conn.statements[0]
    assert "owner" in sql and params[1] == "local"


def test_setup_runs_every_statement() -> None:
    conn = _Conn()

    _run(_store(conn).setup())

    from rag.persistence.chat import SCHEMA

    assert len(conn.sql) == len(SCHEMA)


@pytest.mark.parametrize("status", [STATUS_ANSWERED, STATUS_INTERRUPTED])
def test_status_constants_are_the_ones_written(status: str) -> None:
    conn = _Conn([{"last_seq": 0}])

    _run(_store(conn).append_turn("s1", question="q", status=status))

    _, params = next(
        item for item in conn.statements if item[0].startswith("insert into chat_message")
    )
    assert params[9] == status
