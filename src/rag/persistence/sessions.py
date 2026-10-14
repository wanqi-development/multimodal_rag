"""PG 里的会话账本：**看清楚有哪些会话**、**按需删掉指定的那些**。

LangGraph 的 checkpointer **没有自动清理**：每聊一轮就多几版 `checkpoints`，另加
`checkpoint_blobs` / `checkpoint_writes`（实测一轮 SSE 问答就有 40+ 版、100+ 行待写），
会话多了纯增长。代码里已经撞见过一次 —— 真机验证跑完六个会话，写完就得手动清。

**为什么不自动 TTL**：「等人批准」的那个中断也存在这里。按时间自动删，可能把
「昨天停下等批准、今天才回来看」的那次删掉，而人对此毫无感知 —— 那正是最不该静默
发生的事。所以只提供显式命令，且删之前还要 `--force`（见 CLI）。

**「还有人在等批准吗」怎么判**：HITL 的中断会以 `channel='__interrupt__'` 的行落在
`checkpoint_writes` 里（实测：假模型 + 真 PG 跑一次中断，见 CHANGELOG）。顺手用它把
「这个会话在等人」标出来 —— 删之前至少让人看见。

**顺序**：会话之间只按 `thread_id` 区分，`checkpoints` 的 `checkpoint_id` 是 langgraph
生成的 uuid6（时间有序），所以「最新一版」按它排就约等于按时间排，不用额外加时间戳字段。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg import sql

# 一个会话的账：版本数 + 最新一版（uuid6，字典序≈时间序）
_SUMMARY = """
select thread_id, count(*) as versions, max(checkpoint_id) as last_id
  from checkpoints
 group by thread_id
"""

# 待写的行数，以及其中有多少是「等人批准」的中断
_WRITES = """
select thread_id,
       count(*) as writes,
       count(*) filter (where channel = '__interrupt__') as interrupts
  from checkpoint_writes
 group by thread_id
"""

_DELETE_BY_THREAD = "delete from {table} where thread_id = any(%s)"
_DELETE_STORE = "delete from store where key = any(%s)"

# 删的顺序：先叶子（writes / blobs）再主表，最后是 store 里那次会话的引用表
_TABLES = ("checkpoint_writes", "checkpoint_blobs", "checkpoints")


@dataclass(frozen=True)
class SessionSummary:
    """一个会话在库里占了多少。`last_checkpoint_id` 是 uuid6，越大越新。"""

    thread_id: str
    versions: int
    writes: int
    interrupts: int
    last_checkpoint_id: str = ""

    @property
    def waiting_for_approval(self) -> bool:
        """这个会话正停在「等人批准」那一步 —— 删它等于把待批准的事一起扔掉。"""
        return self.interrupts > 0


class SessionStore:
    """会话账本。**同步**的（CLI 用）；连接由调用方给，测试可以塞替身。"""

    def __init__(self, conninfo: str, *, connect: Callable[..., Any] | None = None) -> None:
        self._conninfo = conninfo
        self._connect = connect or psycopg.connect

    def open(self):
        """一次连接（当上下文管理器用，退出时提交）。"""
        return self._connect(self._conninfo)

    def ensure_schema(self, schema_name: str) -> None:
        """建 schema（幂等）。**库自己只 `CREATE TABLE IF NOT EXISTS`，不建 schema** ——
        全新部署时不先建它，`checkpointer.setup()` 会直接报 schema 不存在。"""
        with self.open() as conn:
            conn.execute(
                sql.SQL("create schema if not exists {}").format(sql.Identifier(schema_name))
            )

    def list(self) -> list[SessionSummary]:
        """所有会话，**最新在前**。"""
        with self.open() as conn:
            rows = conn.execute(_SUMMARY).fetchall()
            counts = {row[0]: (row[1], row[2]) for row in conn.execute(_WRITES).fetchall()}
        out = [
            SessionSummary(
                thread_id=row[0],
                versions=int(row[1]),
                writes=int(counts.get(row[0], (0, 0))[0]),
                interrupts=int(counts.get(row[0], (0, 0))[1]),
                last_checkpoint_id=str(row[2] or ""),
            )
            for row in rows
        ]
        return sorted(out, key=lambda item: item.last_checkpoint_id, reverse=True)

    def prune(self, thread_ids: Sequence[str]) -> dict[str, int]:
        """删掉这些会话的**全部**残留（三张表 + store 里那次会话的引用表）。

        返回每张表删了几行 —— 打印出来才算「做完的事看得见」。
        """
        ids = list(dict.fromkeys(thread_ids))  # 去重且保序
        if not ids:
            return {}
        counts: dict[str, int] = {}
        with self.open() as conn:
            for table in _TABLES:
                counts[table] = conn.execute(_DELETE_BY_THREAD.format(table=table), (ids,)).rowcount
            counts["store"] = conn.execute(_DELETE_STORE, (ids,)).rowcount
        return counts


def build_session_store(settings: Any, secrets: Any = None) -> SessionStore:
    """按 PG 那份配置建账本（连接串带 schema，与 checkpointer 用的是同一条）。"""
    from rag.persistence.pg import build_persistence

    return SessionStore(build_persistence(settings, secrets).conninfo())


def select_prune_targets(
    sessions: Iterable[SessionSummary], *, keep: int | None = None
) -> list[SessionSummary]:
    """`--keep N` = 只保留最新（uuid6 序）的 N 个，其余都是要删的。"""
    if keep is None:
        return []
    ordered = sorted(sessions, key=lambda item: item.last_checkpoint_id, reverse=True)
    return ordered[keep:]
