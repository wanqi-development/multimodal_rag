"""LangGraph 的 PG 持久化：checkpointer（会话 + HITL 中断）与 store（跨会话记忆）。

**一个连接池，两个用途**（两者的构造函数都收 conn/pool，没必要开两份）。

三件实测撞出来的事，都写在代码里：

1. **schema 得我们自己建**：库只 `CREATE TABLE IF NOT EXISTS`，不建 schema；
   而且它不认 schema 参数 —— 隔离靠连接串里注入 `options=-c search_path=<schema>`。
2. **Windows 上 psycopg 的 async 跑不了默认的 `ProactorEventLoop`**（报
   `Psycopg cannot use the 'ProactorEventLoop' to run in async mode`）。而
   `AsyncPostgresSaver/Store` 是 async-only，所以**必须在建事件循环之前**换
   `SelectorEventLoop` —— 这就是 `ensure_compatible_event_loop()` 存在的原因，
   CLI 与 `create_app` 都要在最早的时机调它。
3. `url` 留空 = **不用 PG**，调用方拿到 None 就会退回进程内实现（与本项目其它
   「没配就不启用」的约定一致）。
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from typing import Any

from rag.config.schema import PostgresConfig, Settings


def ensure_compatible_event_loop() -> None:
    """Windows 上把事件循环策略换成 Selector。

    **只在配了 PG 时才需要**（没配就不会有 psycopg 的 async 调用），也只在 win32 上做。
    必须在任何事件循环被创建**之前**调用，否则设置不生效。
    """
    if sys.platform != "win32":
        return
    policy = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if policy is not None and not isinstance(asyncio.get_event_loop_policy(), policy):
        asyncio.set_event_loop_policy(policy())


def check_loop_compatible(loop: Any) -> None:
    """psycopg 的 async **不能**跑在 Windows 的 `ProactorEventLoop` 上（默认就是它）。

    为什么单独做一个检查而不是只靠 `ensure_compatible_event_loop()`：那个函数在事件循环
    **已经建好之后**调用是无效的（policy 只影响之后新建的循环）。而犯这个错时的表象是
    「连接池 10 秒超时」，从报错里完全看不出真正原因 —— 实测踩过。所以这里**当场说清**：
    是循环类型不对，以及该在哪里设。
    """
    if sys.platform != "win32":
        return
    if type(loop).__name__ == "ProactorEventLoop":
        raise RuntimeError(
            "psycopg 的 async 不能跑在 ProactorEventLoop 上（Windows 默认）。"
            "自己写脚本/CLI 时要在**创建事件循环之前**调用 "
            "rag.persistence.ensure_compatible_event_loop()"
            "（`rag agent` 与 create_app 已经这么做了）；"
            "用 uvicorn 起服务时那一招**不管用** —— uvicorn ≥0.36 会自己建 Proactor，"
            "必须把 loop 指到我们的工厂（`loop='rag.persistence:selector_event_loop'`，"
            "`rag serve` 已经这么传）。"
        )


def selector_event_loop() -> asyncio.AbstractEventLoop:
    """给 uvicorn 用的 loop 工厂：`rag serve` 通过
    `loop="rag.persistence:selector_event_loop"` 传进来。

    为什么非要有它（2026-10-05 真机实测）：**uvicorn 0.53 不再看事件循环策略** ——
    `uvicorn/loops/asyncio.py::asyncio_loop_factory(use_subprocess=False)` 在 Windows 上
    直接返回 `asyncio.ProactorEventLoop`，`Server.run()` 再把它交给 `asyncio_run`。
    所以 `ensure_compatible_event_loop()`（只改 policy）对 `rag serve` 完全无效：
    `rag serve` 会在 startup 时以
    「psycopg 的 async 不能跑在 ProactorEventLoop 上」直接崩掉。

    这里给 uvicorn 一个**显式工厂**：按策略建循环（Windows 上就是 Selector），
    uvicorn 的自定义 import 串路径会把它原样当工厂用（不传 `use_subprocess`）。
    """
    ensure_compatible_event_loop()
    return asyncio.new_event_loop()


class PgPersistence:
    """连接池 + saver + store 的持有者。**一个进程一份**。"""

    def __init__(self, cfg: PostgresConfig, password: str | None) -> None:
        self._cfg = cfg
        self._password = password or None
        self._pool: Any = None
        self.checkpointer: Any = None
        self.store: Any = None
        # 不可用的原因：`/health` 要原样报出来（没配 / 连不上 是两件事）
        self.reason: str = ""

    @property
    def configured(self) -> bool:
        return bool(self._cfg.url)

    @property
    def available(self) -> bool:
        return self.checkpointer is not None and self.store is not None

    @property
    def pool(self) -> Any:
        """连接池。**日志表也复用它**（一个进程一个池，别再开第二个）。"""
        return self._pool

    def conninfo(self) -> str:
        """带 schema 的连接串。密码在里面 —— **别打日志**。"""
        from psycopg.conninfo import make_conninfo

        return make_conninfo(
            self._cfg.url,
            password=self._password,
            options=f"-c search_path={self._cfg.schema_name}",
            connect_timeout=int(self._cfg.timeout_s),
        )

    async def connect(self) -> bool:
        """建池 + 建 saver/store。**不建表**：表由 `setup()` 显式建（那是一次写操作）。

        失败不抛：服务要以降级状态起来（退回进程内），`/health` 里说明原因。
        """
        if not self.configured:
            self.reason = "未配置（url 为空）"
            return False
        # **放在 try 外面**：循环类型不对是用法问题，不是「PG 连不上」这种外部状态 ——
        # 它应当直接抛出去，而不是被降级成一个含糊的 reason 藏在 /health 里。
        try:
            check_loop_compatible(asyncio.get_running_loop())
        except RuntimeError:
            raise
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            from langgraph.store.postgres import AsyncPostgresStore
            from psycopg.rows import dict_row
            from psycopg_pool import AsyncConnectionPool

            pool = AsyncConnectionPool(
                conninfo=self.conninfo(),
                min_size=self._cfg.pool_min,
                max_size=self._cfg.pool_max,
                kwargs={"autocommit": True, "row_factory": dict_row},
                open=False,
            )
            await pool.open(wait=True, timeout=self._cfg.timeout_s)
            self._pool = pool
            # saver 与 store **共用这个池**
            self.checkpointer = AsyncPostgresSaver(pool)
            self.store = AsyncPostgresStore(pool)
        except Exception as exc:  # noqa: BLE001 —— 连不上就降级，见模块说明
            self.reason = f"{type(exc).__name__}: {exc}"
            await self.close()
            return False
        self.reason = ""
        return True

    async def setup(self) -> None:
        """建表（幂等）。**这是一次写操作**，所以单独一个方法、由调用方显式触发。"""
        if not self.available:
            raise RuntimeError(f"PG 不可用：{self.reason}")
        await self.checkpointer.setup()
        await self.store.setup()

    async def ping(self) -> bool:
        if self._pool is None:
            return False
        try:
            async with self._pool.connection() as conn:
                await conn.execute("select 1")
            return True
        except Exception:  # noqa: BLE001
            return False

    async def close(self) -> None:
        if self._pool is not None:
            # 关池失败没什么可做的（进程要退了 / 已经断了），压掉
            with contextlib.suppress(Exception):
                await self._pool.close()
        self._pool = None
        self.checkpointer = None
        self.store = None


def build_persistence(settings: Settings, secrets: Any = None) -> PgPersistence:
    """按配置建一个（还没连的）持久化对象。"""
    if secrets is None:
        from rag.config import get_secrets

        secrets = get_secrets()
    return PgPersistence(settings.persistence.postgres, secrets.rag_pg_password)
