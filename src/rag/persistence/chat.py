"""UI 会话的 PG 存储：**一段对话 + 它的消息 + 每次问答的 token 消耗**（三张表）。

与 `rag sessions`（CLI，管 LangGraph checkpointer 的**残留 thread**）不是一回事：

* checkpointer 存的是 agent 的**机器记忆**（消息、中断状态），只有 agent 那条路有；
* 这三张表存的是**给人看的对话记录**，快问答与 agent 都有 —— 换机器、换浏览器都在。

两者靠 `chat_session.thread_id` 关联（快问答没有 thread，那一列就是空）。

三个决定：

1. **只留 human / ai 两条消息**（`chat_message` 一行一条）。工具消息、命中池、通道账本
   都不进这三张表 —— 它们是「这一次跑」的过程数据，要查有日志表。
2. **中断那轮照样落**：human 行 + 一条 `content` 为空、`status='interrupted'` 的 ai 行。
   载荷（它想写什么文件）**不存**，读的时候现去 checkpointer 取（`pending_actions`）——
   存两份没意义，而且写文件的正文可能很大。
3. **token 单开一张表**（`token_usage`），一次问答一行：按会话 / 按天 / 按模型都能聚合，
   要逐次调用的明细那是 `agent_run_log` 的事。两张表**不混**。

表建与写分开（与日志表同一个规矩）：DDL 由 `rag chats init` 显式触发，
写入失败**不影响问答**（调用方吞掉异常）。
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

SCHEMA = (
    """
create table if not exists chat_session (
    session_id  text primary key,
    owner       text        not null default 'local',
    title       text        not null default '',
    mode        text        not null default 'chat',
    thread_id   text,
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
)
""",
    # **建表语句只管新库**：`create table if not exists` 不会给已存在的表加列。
    # 所以 `owner` 这条既有建表里的定义、也有这一句 alter —— 已经建过的库靠它补上，
    # 重复跑也没事（`if not exists`）。迁移就应该是幂等的。
    "alter table chat_session add column if not exists owner text not null default 'local'",
    "create index if not exists chat_session_updated_idx on chat_session (updated_at desc)",
    """
create table if not exists chat_message (
    message_id bigserial primary key,
    session_id text        not null references chat_session(session_id) on delete cascade,
    seq        int         not null,
    role       text        not null,
    content    text        not null default '',
    status     text        not null default 'answered',
    created_at timestamptz not null default now(),
    unique (session_id, seq)
)
""",
    """
create table if not exists token_usage (
    usage_id      bigserial primary key,
    session_id    text        not null references chat_session(session_id) on delete cascade,
    message_id    bigint      references chat_message(message_id) on delete cascade,
    model         text        not null default '',
    calls         int         not null default 1,
    input_tokens  int,
    output_tokens int,
    total_tokens  int,
    created_at    timestamptz not null default now()
)
""",
    "create index if not exists token_usage_session_idx on token_usage (session_id)",
    "create index if not exists token_usage_created_idx on token_usage (created_at desc)",
)

# 消息状态：跑完了 / 停在闸门等人批准 / 人点了停止 / 出错了
STATUS_ANSWERED = "answered"
STATUS_INTERRUPTED = "interrupted"
STATUS_STOPPED = "stopped"
STATUS_ERROR = "error"

ROLE_HUMAN = "human"
ROLE_AI = "ai"


class ChatStore:
    """三张表的读写。**异步**（API 用），连接由调用方给：池或短连接。

    `pool` 与 `conninfo` 二选一：长驻进程（API）复用池，一次性命令（`rag chats`）
    走短连接。`connect` 是给测试塞替身留的口子。
    """

    def __init__(
        self,
        *,
        pool: Any = None,
        conninfo: str | None = None,
        connect: Callable[[], Any] | None = None,
    ) -> None:
        self.pool = pool
        self.conninfo = conninfo
        self._connect = connect

    @contextlib.asynccontextmanager
    async def _session(self) -> Any:
        if self._connect is not None:
            async with self._connect() as conn:
                yield conn
            return
        if self.pool is not None:
            async with self.pool.connection() as conn:
                yield conn
            return
        import psycopg

        conn = await psycopg.AsyncConnection.connect(self.conninfo, autocommit=True)
        try:
            yield conn
        finally:
            await conn.close()

    # ---- DDL ----

    async def setup(self) -> None:
        async with self._session() as conn:
            for statement in SCHEMA:
                await conn.execute(statement)

    # ---- 会话 ----

    async def create(
        self,
        *,
        session_id: str,
        owner: str = "local",
        title: str = "",
        mode: str = "chat",
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        """开一段（已存在就只碰 `updated_at`，标题与 owner 不动）。

        `owner` 现在恒为 `'local'`：鉴权拿掉之后没有「谁」这个概念了。
        用户系统那一批把注册用户 id 写进这一列，列表与详情按它过滤 ——
        **列现在就加上**，免得到时有数据了再迁。
        """
        async with self._session() as conn:
            await conn.execute(
                "insert into chat_session (session_id, owner, title, mode, thread_id)"
                " values (%s, %s, %s, %s, %s)"
                " on conflict (session_id) do update set updated_at = now()",
                (session_id, owner, title, mode, thread_id),
            )
        return {
            "session_id": session_id,
            "owner": owner,
            "title": title,
            "mode": mode,
            "thread_id": thread_id,
        }

    async def touch(
        self,
        session_id: str,
        *,
        title: str | None = None,
        mode: str | None = None,
        thread_id: str | None = None,
    ) -> None:
        """更新这一段：标题（首问定下之后不再覆盖）、模式、thread。"""
        async with self._session() as conn:
            await conn.execute(
                "update chat_session set updated_at = now(),"
                " title = coalesce(nullif(title, ''), %s),"
                " mode = coalesce(%s, mode),"
                " thread_id = coalesce(%s, thread_id)"
                " where session_id = %s",
                (title, mode, thread_id, session_id),
            )

    async def list(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """会话列表：轮数 + token 合计一起给（界面要显示，别让它再来一次）。"""
        async with self._session() as conn:
            cursor = await conn.execute(_LIST_SQL, (limit,))
            return [dict(row) for row in await cursor.fetchall()]

    async def get(self, session_id: str) -> dict[str, Any] | None:
        """一段会话的全部消息 + token 合计。**按 seq 排**，前端拿到就能直接渲染。"""
        async with self._session() as conn:
            cursor = await conn.execute(
                "select session_id, title, mode, thread_id, created_at, updated_at"
                " from chat_session where session_id = %s",
                (session_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            session = dict(row)
            cursor = await conn.execute(
                "select message_id, seq, role, content, status, created_at"
                " from chat_message where session_id = %s order by seq",
                (session_id,),
            )
            session["messages"] = [dict(item) for item in await cursor.fetchall()]
            # 每一轮的 token 用量**按消息挂回去**（`token_usage.message_id` 就是干这个的）：
            # 重开一段历史会话时，逐轮的账要跟着消息一起回来，不能只在当次看得见。
            cursor = await conn.execute(
                "select message_id, calls, input_tokens, output_tokens, total_tokens"
                " from token_usage where session_id = %s",
                (session_id,),
            )
            per_message = {
                int(row["message_id"]): dict(row)
                for row in await cursor.fetchall()
                if row["message_id"] is not None
            }
            for item in session["messages"]:
                item["usage"] = per_message.get(int(item["message_id"]))
            session["usage"] = await _usage_total(conn, session_id)
        return session

    async def delete(self, session_id: str) -> bool:
        """删一段。消息与 token 行靠外键级联一起走。"""
        async with self._session() as conn:
            cursor = await conn.execute(
                "delete from chat_session where session_id = %s", (session_id,)
            )
            return (cursor.rowcount or 0) > 0

    async def summary(self) -> dict[str, Any]:
        """总量：几段、几轮、累计 token —— token 那张表的第一个用处。"""
        async with self._session() as conn:
            cursor = await conn.execute(_SUMMARY_SQL)
            row = await cursor.fetchone()
        return dict(row) if row else {"sessions": 0, "turns": 0}

    # ---- 一轮问答 ----

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
        """落一轮：human 一行、ai 一行，外加一条 token 账。返回 ai 那行的 `seq`。

        中断时 `answer` 是空的、`status='interrupted'` —— 行照样要有，不然重开这段
        会话时看不出「上次停在等人批准」。
        """
        async with self._session() as conn:
            cursor = await conn.execute(
                # **按列名取**：连接池那边配的是 `dict_row`，逐位下标会在真库上炸
                # （KeyError: 0，实测）。所以每一条 select 都起别名。
                "select coalesce(max(seq), 0) as last_seq from chat_message where session_id = %s",
                (session_id,),
            )
            row = await cursor.fetchone()
            base = int((row["last_seq"] if row else 0) or 0)
            await conn.execute(
                "insert into chat_message (session_id, seq, role, content, status)"
                " values (%s, %s, %s, %s, %s), (%s, %s, %s, %s, %s)",
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
            await conn.execute(
                "update chat_session set updated_at = now(), mode = %s,"
                " thread_id = coalesce(%s, thread_id) where session_id = %s",
                (mode, thread_id, session_id),
            )
            if usage is not None:
                await _record_usage(conn, session_id, base + 2, usage)
        return base + 2

    async def update_pending(
        self,
        session_id: str,
        *,
        answer: str,
        status: str = STATUS_ANSWERED,
        usage: dict[str, Any] | None = None,
    ) -> bool:
        """把**最近一条还等着**（中断/被停）的 ai 行填上。

        恢复那轮不能用 `append_turn` —— 那一轮的行在中断时就已经落了，
        再插一行会让前端看到两个答复。一个会话同一时刻只可能停在一道闸门后面，
        所以「最近一条 status 不是 answered 的 ai 行」就是它。
        """
        async with self._session() as conn:
            cursor = await conn.execute(
                "select seq from chat_message where session_id = %s and role = %s"
                " and status <> %s order by seq desc limit 1",
                (session_id, ROLE_AI, STATUS_ANSWERED),
            )
            row = await cursor.fetchone()
            if row is None:
                return False
            seq = int(row["seq"] or 0)
            await conn.execute(
                "update chat_message set content = %s, status = %s"
                " where session_id = %s and seq = %s",
                (answer, status, session_id, seq),
            )
            await conn.execute(
                "update chat_session set updated_at = now() where session_id = %s",
                (session_id,),
            )
            if usage is not None:
                await _record_usage(conn, session_id, seq, usage)
        return True


_LIST_SQL = """
select s.session_id, s.title, s.mode, s.thread_id, s.created_at, s.updated_at,
       coalesce(m.turns, 0)      as turns,
       coalesce(u.tokens_in, 0)  as input_tokens,
       coalesce(u.tokens_out, 0) as output_tokens,
       coalesce(u.tokens_all, 0) as total_tokens,
       coalesce(u.calls, 0)      as calls
  from chat_session s
  left join (
       select session_id, count(*) filter (where role = 'ai') as turns
         from chat_message group by session_id
  ) m on m.session_id = s.session_id
  left join (
       select session_id, sum(input_tokens) as tokens_in, sum(output_tokens) as tokens_out,
              sum(total_tokens) as tokens_all, sum(calls) as calls
         from token_usage group by session_id
  ) u on u.session_id = s.session_id
 order by s.updated_at desc
 limit %s
"""

_SUMMARY_SQL = """
select (select count(*) from chat_session)                     as sessions,
       (select count(*) from chat_message where role = 'ai')   as turns,
       (select coalesce(sum(input_tokens), 0) from token_usage)  as input_tokens,
       (select coalesce(sum(output_tokens), 0) from token_usage) as output_tokens,
       (select coalesce(sum(total_tokens), 0) from token_usage)  as total_tokens
"""


async def _usage_total(conn: Any, session_id: str) -> dict[str, Any]:
    cursor = await conn.execute(
        "select coalesce(sum(input_tokens), 0)  as tokens_in,"
        " coalesce(sum(output_tokens), 0) as tokens_out,"
        " coalesce(sum(total_tokens), 0)  as tokens_all,"
        " coalesce(sum(calls), 0)         as calls"
        " from token_usage where session_id = %s",
        (session_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "calls": 0}
    return {
        "input_tokens": int(row["tokens_in"] or 0),
        "output_tokens": int(row["tokens_out"] or 0),
        "total_tokens": int(row["tokens_all"] or 0),
        "calls": int(row["calls"] or 0),
    }


async def _record_usage(conn: Any, session_id: str, seq: int, usage: dict[str, Any]) -> None:
    """记一条 token 账。`message_id` 靠 (session_id, seq) 反查 —— 调用方不必知道它。"""
    cursor = await conn.execute(
        "select message_id from chat_message where session_id = %s and seq = %s",
        (session_id, seq),
    )
    row = await cursor.fetchone()
    await conn.execute(
        "insert into token_usage (session_id, message_id, model, calls,"
        " input_tokens, output_tokens, total_tokens) values (%s, %s, %s, %s, %s, %s, %s)",
        (
            session_id,
            (int(row["message_id"]) if row and row["message_id"] is not None else None),
            str(usage.get("model") or ""),
            int(usage.get("calls") or 1),
            usage.get("input_tokens"),
            usage.get("output_tokens"),
            usage.get("total_tokens"),
        ),
    )


def build_chat_store(settings: Any, persistence: Any = None) -> ChatStore:
    """按配置建一个 ChatStore。有长驻 persistence 就用它的池。"""
    from rag.persistence.pg import build_persistence

    if persistence is not None and getattr(persistence, "available", False):
        return ChatStore(pool=persistence.pool)
    return ChatStore(conninfo=build_persistence(settings).conninfo())
