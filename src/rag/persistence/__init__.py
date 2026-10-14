"""图的持久化（M5 起）：LangGraph 的 checkpointer 与 store 都落 PG。

`url` 留空就退回进程内（调用方拿到一个 `available=False` 的对象），
与 Redis / OCR 产线同一个约定。表结构与踩过的坑见 `pg.py` 的模块说明。
"""

from rag.persistence.chat import ChatStore, build_chat_store
from rag.persistence.pg import (
    PgPersistence,
    build_persistence,
    ensure_compatible_event_loop,
    selector_event_loop,
)
from rag.persistence.sessions import (
    SessionStore,
    SessionSummary,
    build_session_store,
    select_prune_targets,
)

__all__ = [
    "ChatStore",
    "PgPersistence",
    "SessionStore",
    "SessionSummary",
    "build_chat_store",
    "build_persistence",
    "build_session_store",
    "ensure_compatible_event_loop",
    "select_prune_targets",
    "selector_event_loop",
]
