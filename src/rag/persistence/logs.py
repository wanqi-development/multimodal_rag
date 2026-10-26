"""日志进 PG：按 `request_id` / 类别 / 时间查（使用者 2026-10-08 定）。

**为什么不再用 grep**：文件一多就要跨文件扫，而且**没有索引**。实测现在全目录扫一遍
22.5 ms（5 个文件、690 KB、5,800 行），按 260 KB/天算，一年 95 MB —— 那时候线性扫是秒级，
还要加上几百个文件打开的开销。要求是"下拉就能筛"，这个量级撑不住。

**文件仍是唯一真相，这张表是从文件灌进来的索引**（2026-10-08 定的分工）：

* 写日志那条路**继续写文件**，只是**顺手**再塞一行进这里的队列；
* 落库失败**只 stderr 喊一句并丢掉这一批** —— 日志写不进库不该拖业务，这是 2026-10-06
  把日志从 PG 三张表搬到文件时的立场；库里那份丢了随时能从文件重建（所以这一版**不灌历史**，
  从新产生的行开始）；
* `/logs/tail` 保留不动：PG 挂了，日志页至少还能看文件尾巴。

四个口径（使用者逐条定的，写在这里免得以后又改回去）：

1. `request_id` **从入口就有**：HTTP 那条已经实现（`api/app.py` 的中间件），CLI 那条要补；
2. 不在请求里的行（库自己打的、测试进程、后台任务）**生成一个**，用 `solo-` 前缀 ——
   一眼看出"这条不属于任何请求"，点它也不会误匹配别的行；所以这一列 **not null**；
3. `run_id` 用 **`job_id`**：ingest 分多次请求跑（parse / 核对 / index），每进一次请求就新发
   一个 run 必然串不起来 —— 那不是缺一个 id，是 run 生错了地方。有 job 就用 job id，
   与页面上那排步骤灯同源；
4. 入口是 HTTP 还是 CLI 用 `source` 列区分，**不靠前缀**。

表建与写分开（`rag logs init` 显式建），与 chat 那三张表同一个规矩、同一个库。
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

SCHEMA = (
    """
create table if not exists log_line (
    id          bigserial primary key,
    ts          timestamptz not null,
    category    text        not null,
    level       text        not null default 'INFO',
    request_id  text        not null,
    run_id      text,
    source      text        not null default '',
    logger      text,
    status      int,
    duration_ms int,
    text        text        not null
)
""",
    # 主查询：拿一个 request_id 把整条链路按时间串出来
    "create index if not exists log_line_request on log_line (request_id, ts)",
    # 次查询：类别 + 时间（下拉筛"最近一小时的 system"）
    "create index if not exists log_line_category on log_line (category, ts desc)",
)

# 一次查询最多回多少行。**必须有上限**：一个 request_id 真出问题时会刷出几千行，
_SEARCH_LIMIT = 2000

# 一次批量插多少行 —— 与 handler 的 flush 阈值一致，改一处要改两处（写在这里提醒）
BATCH_SIZE = 200

_COLUMNS = (
    "ts",
    "category",
    "level",
    "request_id",
    "run_id",
    "source",
    "logger",
    "status",
    "duration_ms",
    "text",
)

# 同步与异步两条路共用同一段 SQL：两边列的顺序必须与 `_prepare` 的元组一致
_INSERT_SQL = (
    f"insert into log_line ({', '.join(_COLUMNS)}) values ({', '.join(['%s'] * len(_COLUMNS))})"
)


class LogStore:
    """`log_line` 的读写。**异步**（API 与后台 flush 任务用）。

    `pool` / `conninfo` / `connect` 三选一，与 `ChatStore` 同一套：长驻进程复用池，
    一次性命令走短连接，测试塞替身走 `connect`。
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
        from psycopg.rows import dict_row

        # **和池那条路给同一个形状**（`row_factory=dict_row`）：池是建的时候带的，
        # 短连接不显式指就没有 —— 少了它 `search()` 拿到的是 tuple，`dict(row)` 直接抛
        # （实测：`cannot convert dictionary update sequence element #0 to a sequence`）。
        conn = await psycopg.AsyncConnection.connect(
            self.conninfo, autocommit=True, row_factory=dict_row
        )
        try:
            yield conn
        finally:
            await conn.close()

    async def setup(self) -> None:
        async with self._session() as conn:
            for statement in SCHEMA:
                await conn.execute(statement)

    async def write(self, rows: Sequence[dict[str, Any]]) -> int:
        """批量插一批。返回写进去几行。

        **`request_id` 为空的行在这里补一个 `solo-`**（不是在调用方补）：漏补一行就会撞上
        `not null` 约束、整批插失败 —— 那种失败会静默丢掉一整批日志，代价远大于在这里兜一下。
        """
        prepared = self._prepare(rows)
        if not prepared:
            return 0
        async with self._session() as conn:
            cursor = await conn.cursor()
            await cursor.executemany(_INSERT_SQL, prepared)
        return len(prepared)

    def write_sync(self, rows: Sequence[dict[str, Any]]) -> int:
        """**同步**批量插：日志 handler 的后台线程用它（那里没有事件循环）。

        与 `write()` 共用同一套整理与同一段 SQL，所以两条路写出来的行必然一致 ——
        这也是为什么整理那步要抽出来（`_prepare`），而不是各写一遍。
        `conninfo` 那条路（CLI / handler）才有意义：池是异步的，同步侧用不了。
        """
        prepared = self._prepare(rows)
        if not prepared:
            return 0
        if not self.conninfo:
            raise RuntimeError("同步写入需要 conninfo（池只能异步用）")
        import psycopg

        with psycopg.connect(self.conninfo) as conn, conn.cursor() as cursor:
            cursor.executemany(_INSERT_SQL, prepared)
        return len(prepared)

    @staticmethod
    def _prepare(rows: Sequence[dict[str, Any]]) -> list[tuple[Any, ...]]:
        """整理成能直接 `executemany` 的形状（同步/异步两条路共用）。"""
        from rag.obs import new_id

        return [
            (
                row.get("ts") or datetime.now().astimezone(),
                str(row.get("category") or "system"),
                str(row.get("level") or "INFO"),
                str(row.get("request_id") or new_id("solo")),
                row.get("run_id") or None,
                str(row.get("source") or ""),
                row.get("logger") or None,
                row.get("status"),
                row.get("duration_ms"),
                str(row.get("text") or ""),
            )
            for row in rows
        ]

    async def search(
        self,
        *,
        request_id: str,
        category: str = "",
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """按 `request_id` 串一条链路，可选再加类别与时间窗。

        **`request_id` 必填**（使用者定的）：这个页面要回答的是"这一次请求到底发生了什么"，
        没有 id 就退化成"翻日志"，那正是文件 + grep 在做的事、也是上库要解决的问题。
        要按类别 + 时间翻，那是另一个入口（`/logs/tail` 已经能做）。
        """
        wanted = request_id.strip()
        if not wanted:
            raise ValueError("request_id 必填：这个查询要回答的是「这一次请求发生了什么」")
        sql = "select * from log_line where request_id = %s"
        args: list[Any] = [wanted]
        if category:
            sql += " and category = %s"
            args.append(category)
        if since is not None:
            sql += " and ts >= %s"
            args.append(since)
        if until is not None:
            sql += " and ts <= %s"
            args.append(until)
        sql += " order by ts limit %s"
        args.append(max(1, min(limit, _SEARCH_LIMIT)))
        async with self._session() as conn:
            cursor = await conn.execute(sql, tuple(args))
            rows = await cursor.fetchall()
        if rows and not isinstance(rows[0], dict):
            # 兜底：替身或别处塞进来的连接可能仍给 tuple，按列名补一次，别在这儿炸
            names = [item.name for item in (cursor.description or [])]
            return [dict(zip(names, row, strict=False)) for row in rows]
        return [dict(row) for row in rows]


def build_log_store(settings: Any, persistence: Any = None) -> LogStore:
    """按配置建日志表。**与 chat 那三张表同一个库**（`persistence.postgres.url`）。

    有 `persistence`（API 进程）就复用它的池；CLI 那种一次性命令走短连接。

    短连接这条走 `build_persistence(settings).conninfo()`（与 `build_chat_store` 同一套）：
    里面带着**密码与 `search_path`**，直接拿裸 url 会连不上、或者表建到 `public` 去。
    """
    pool = getattr(persistence, "pool", None) if persistence is not None else None
    if pool is not None:
        return LogStore(pool=pool)
    from rag.persistence.pg import build_persistence

    url = getattr(settings.persistence.postgres, "url", "")
    if not url:
        raise ValueError("没配 persistence.postgres.url：在 .env 里填 RAG_PG_URL（日志表与它同库）")
    return LogStore(conninfo=build_persistence(settings).conninfo())
