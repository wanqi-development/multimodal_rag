"""日志的**进程内**实时缓冲：给页面的 `/logs/stream` 用。

为什么不让页面去 tail 文件：实测在 Windows 上，持着只读句柄会把 `RotatingFileHandler`
的轮转顶掉（`WinError 32`，而且那一条日志会整条丢掉）。所以实时这一层走内存 ——
logging handler 收一条 → 进环形缓冲 + 推给订阅者；**文件仍然是历史那一份**
（`/logs/tail` 读它）。

边界要说清：这里只有**本进程**写出来的日志。CLI（`rag ingest` / `rag ask`）在另一个
进程里，它的行要等 `/logs/tail` 或页面上手动刷新才看得到。

三条约束：

1. **推送不许影响业务**：`LiveHandler.emit` 里的一切异常都吞掉（`handleError` 按
   `raiseExceptions` 决定要不要喊），绝不能反过来把一次请求带崩；
2. **慢客户端拖不住后端**：每个订阅者一个有界队列（满了丢最旧的）；
3. **跨线程安全**：handler 可能从线程池里被调用（OCR / embedding），所以要
   `call_soon_threadsafe` 往订阅者所在的循环里投递。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from rag.obs.log import current_ids

# 环形缓冲留多少行：一屏几百行，2000 够往回翻「刚才那一下到底打了什么」
BUFFER = 2000

# 单个订阅者的队列上限：页面卡住时丢最旧的，绝不把 API 进程拖住
QUEUE = 512

# 业务日志的 logger → 页面上的类别名（其余都算 system，与文件落点一致）。
# agent 拆三类：一轮的汇总 / 每次模型调用 / 每次工具调用 —— 它们**同进 agent.log**。
_KIND_BY_LOGGER = {
    "rag.interface": "interface",
    "rag.channel": "channel",
    "rag.agent_run": "agent_run",
    "rag.agent_llm": "agent_llm",
    "rag.agent_tool": "agent_tool",
}

_LOCK = threading.Lock()
_RECORDS: deque[dict[str, Any]] = deque(maxlen=BUFFER)
_SUBSCRIBERS: set[asyncio.Queue] = set()
# 最近一次订阅所在的循环：从别的线程 publish 时用它来投递
_LOOP: asyncio.AbstractEventLoop | None = None


def kind_of(logger_name: str) -> str:
    """logger 名 → 页面类别（与四个文件的划分一致）。"""
    for prefix, kind in _KIND_BY_LOGGER.items():
        if logger_name == prefix or logger_name.startswith(f"{prefix}."):
            return kind
    return "system"


def recent(limit: int = 200) -> list[dict[str, Any]]:
    """最近 `limit` 条（按时间正序：老的在前，页面直接往下接）。"""
    with _LOCK:
        items = list(_RECORDS)
    return items[-limit:] if limit > 0 else items


@contextmanager
def subscribe() -> Iterator[asyncio.Queue]:
    """订阅实时流。退出时（客户端断开）自动注销。"""
    global _LOOP
    queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE)
    loop = asyncio.get_running_loop()
    with _LOCK:
        _SUBSCRIBERS.add(queue)
        _LOOP = loop
    try:
        yield queue
    finally:
        with _LOCK:
            _SUBSCRIBERS.discard(queue)


def publish(record: dict[str, Any]) -> None:
    """一条日志 → 环形缓冲 + 所有订阅者。**任何异常都不许冒出去**（调用方是 logging）。"""
    with _LOCK:
        _RECORDS.append(record)
        targets = list(_SUBSCRIBERS)
        loop = _LOOP
    if not targets:
        return
    for queue in targets:
        try:
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(_offer, queue, record)
            else:
                _offer(queue, record)
        except Exception:  # noqa: BLE001, S110 —— 观测不许影响业务，见模块说明
            pass


def _offer(queue: asyncio.Queue, record: dict[str, Any]) -> None:
    """投递给一个订阅者：满了先丢最旧的（页面卡住时后端照常跑）。"""
    if queue.full():
        with contextlib.suppress(asyncio.QueueEmpty):
            queue.get_nowait()
    with contextlib.suppress(asyncio.QueueFull):
        queue.put_nowait(record)


def reset() -> None:
    """清空缓冲与订阅者（测试用；生产路径没有调用点）。"""
    global _LOOP
    with _LOCK:
        _RECORDS.clear()
        _SUBSCRIBERS.clear()
        _LOOP = None


class LiveHandler(logging.Handler):
    """把每条日志按**我们那套格式**塞进环形缓冲。

    时间、级别、**来源**（logger 短名）都另给字段，正文只留 `ids + message` ——
    页面上一列一个东西，不用再去切字符串、也不会把短名重复显示两遍。
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            request_id, run_id = current_ids()
            publish(
                {
                    # 与文件里 `DATE_FORMAT` 同形，但**带毫秒** —— 只有秒的话，同一秒里
                    # 不同来源的行在页面上排不出先后（实时这条也一样）
                    "ts": f"{datetime.fromtimestamp(record.created).astimezone():%m-%d %H:%M:%S}"
                    f".{int(record.msecs):03d}",
                    "level": record.levelname,
                    "kind": kind_of(record.name),
                    "logger": record.name,
                    # 短名由 CorrelationFilter 填（它跑在 ShorterNames 之后，见 setup.py）
                    "source": getattr(record, "short", record.name.rsplit(".", 1)[-1]),
                    # 关联 id 单列出来：正文里不再重复一遍，页面上这两个值还要能点
                    # （点一下按它串出整条链路）
                    "request_id": request_id or "",
                    "run_id": run_id or "",
                    "text": self.format(record).strip(),
                }
            )
        except Exception:  # noqa: BLE001 —— 见模块说明第 1 条
            self.handleError(record)
