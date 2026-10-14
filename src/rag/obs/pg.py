"""把日志**顺手**再塞一份进 PG：查的时候按 `request_id` / 类别 / 时间拉，不用扫文件。

为什么要它、以及表里那些列为什么是那个形状，见 `persistence/logs.py` 的模块说明。
这里只讲**这个 handler 自己的取舍**：

1. **不许拖业务**。`emit()` 只做一件事：把算好的那一行塞进内存队列，然后立刻返回。
   真正的写库在一条后台守护线程里批量做 —— 磁盘满、PG 挂了、表没建，都只影响这份索引，
   业务照常。一次 `insert` 200 行比 200 次单行快两个数量级，而这个进程一天也就几千行，
   批量 + 定时已经足够。
2. **失败只喊一句**，然后**丢掉这一批**。不重试、不阻塞：库里那份随时能从文件重建
   （文件才是唯一真相，见 `obs/setup.py` 的落点表）。喊话按次数限流（前 3 次 + 每 100 次），
   不然 PG 停一小时会把 stderr 刷满，反而看不见别的东西。
3. **队列有界**：满了**丢最新的一条**并计数。日志是观测，不能因为下游慢就把 API 进程的
   内存吃光 —— 与 `obs/live.py` 对订阅者队列的态度一致。
4. **进程退出要把尾巴写完**：`logging.shutdown()` 会调 `flush()` 与 `close()`，
   这里把队列里剩下的先写完（有超时，失败就算了），不然短命令（CLI）最后那几行会丢。

**只挂在 root 上**：业务 logger 会冒泡上来，第三方库的日志也收得到，一条记录只过一次。
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
from collections.abc import Callable, Sequence
from typing import Any

from rag.obs.log import row_of
from rag.persistence.logs import BATCH_SIZE

# 攒多久写一次。0.25 秒是「页面上感觉是即时的」与「一次写多点」之间的折中：
# 一行日志被 flush 之前最多在内存里待这么久
INTERVAL_S = 0.25

# 队列上限。按一天几千行的量级，20k 行意味着下游挂掉之后还能撑一阵；
# 真正的意义是**有界**（见模块说明第 3 条）
QUEUE_MAX = 20_000

# 退出时等后台线程收尾的上限（秒）：CLI 不该因为 PG 慢而卡很久
JOIN_TIMEOUT_S = 5.0


class PgLogHandler(logging.Handler):
    """日志 → 内存队列 →（后台线程）批量 insert。

    `write` 是**同步**的批量写（`LogStore.write_sync`）：这个线程没有事件循环，
    用 async 那套要自己造 loop —— 实测 Windows 上 psycopg 的 async 只能跑在
    Selector 循环里，为一份日志索引去背这个约束不值得。
    """

    def __init__(
        self,
        write: Callable[[Sequence[dict[str, Any]]], Any],
        *,
        batch_size: int = BATCH_SIZE,
        interval: float = INTERVAL_S,
        queue_max: int = QUEUE_MAX,
    ) -> None:
        super().__init__()
        self._write = write
        self._batch_size = max(1, int(batch_size))
        self._interval = max(0.01, float(interval))
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max(1, int(queue_max)))
        self._stopping = threading.Event()
        # 计数：丢了就说清楚丢了多少（只喊不抛，见模块说明第 2 条）
        self._flushes = 0
        self._failed_flushes = 0
        self._lost_rows = 0
        self._dropped = 0
        self._thread = threading.Thread(target=self._run, name="rag-log-pg", daemon=True)
        self._thread.start()

    # ---- 调用方（logging）这一侧 ----

    def emit(self, record: logging.LogRecord) -> None:
        """**只入队，不写库**（见模块说明第 1 条）。"""
        try:
            row = row_of(record)
        except Exception:  # noqa: BLE001 —— 取字段失败也绝不能把业务带崩
            self.handleError(record)
            return
        if row is None:
            return
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            self._dropped += 1
            self._lost_rows += 1

    def flush(self) -> None:
        """把队列里剩下的写完（`logging.shutdown()` 会调它，见模块说明第 4 条）。

        写失败**不再往上抛**：这一刻通常进程正在退，抛出去只会变成一句没人读的
        `Exception ignored in …`。
        """
        rows = self._drain()
        if rows:
            self._flush(rows)

    def close(self) -> None:
        self._stopping.set()
        if self._thread.is_alive():
            self._thread.join(timeout=JOIN_TIMEOUT_S)
        # 线程可能是超时没停（PG 卡住），剩下的自己来一遍
        rows = self._drain()
        if rows:
            self._flush(rows)
        super().close()

    # ---- 后台线程这一侧 ----

    def _run(self) -> None:
        while True:
            try:
                first = self._queue.get(timeout=self._interval)
            except queue.Empty:
                if self._stopping.is_set():
                    return
                continue
            batch = [first]
            # 同一瞬间攒下的都一起写；到 `batch_size` 就收手（一次 insert 别太长）
            while len(batch) < self._batch_size:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            self._flush(batch)
            if self._stopping.is_set() and self._queue.empty():
                return

    def _drain(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        while True:
            try:
                rows.append(self._queue.get_nowait())
            except queue.Empty:
                return rows

    def _flush(self, rows: list[dict[str, Any]]) -> None:
        self._flushes += 1
        try:
            self._write(rows)
        except Exception as exc:  # noqa: BLE001 —— 见模块说明第 2 条
            self._failed_flushes += 1
            self._lost_rows += len(rows)
            # 限流喊话：前 3 次 + 之后每 100 次。PG 停掉时不该把 stderr 刷满
            if self._failed_flushes <= 3 or self._failed_flushes % 100 == 0:
                print(
                    f"[rag] 日志进库失败（第 {self._failed_flushes} 批，"
                    f"累计丢 {self._lost_rows} 行）：{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

    def stats(self) -> dict[str, int]:
        """给测试与排查用的小结：写了几批、失败几批、丢了几行。"""
        return {
            "flushes": self._flushes,
            "failed_flushes": self._failed_flushes,
            "lost_rows": self._lost_rows,
            "dropped": self._dropped,
            "pending": self._queue.qsize(),
        }


def build_pg_handler(settings: Any) -> PgLogHandler | None:
    """按配置建一个（已经启动后台线程的）handler。**没配 PG 或者关掉了就返回 None。**

    `settings.persistence.postgres.url` 为空 = 不用 PG（与其它「没配就不启用」的约定一致），
    这时只落文件，连 `psycopg` 都不 import。

    **统一开关选了 sqlite 时也不写库**：`persistence.backend=sqlite` 的意思是"关系型数据都落
    单机文件"，而日志索引只有 PG 这一份实现 —— 那就退回只落文件（文件仍是唯一真相）。
    """
    from rag.persistence.backend import resolve_backend

    try:
        if resolve_backend(settings) == "sqlite":
            return None
    except ValueError:
        # 配置自相矛盾（选了 pg 却没 url）：让各个 store 在第一次读写时去报错，
        # 日志这一份只落文件就好 —— 别让日志装配把整个进程拦在启动那一步。
        return None
    enabled = bool(getattr(settings.observability.logging, "postgres", True))
    postgres = getattr(settings.persistence, "postgres", None)
    if not enabled or postgres is None or not getattr(postgres, "url", ""):
        return None
    from rag.persistence.logs import build_log_store

    return PgLogHandler(build_log_store(settings).write_sync)
