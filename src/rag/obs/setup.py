"""把日志接到 **stdlib `logging`** 上（2026-10-06 改的，之前是手写的一套）。

为什么必须用 `logging` 而不是自己写落点：**第三方库的日志一个都进不来**。
LangChain / LangGraph / pymilvus / httpx / uvicorn / transformers 全都走 stdlib logging ——
手写那套只认得我们自己产生的三件事，于是它们的异常只能在 stderr 上裸奔：
没有级别、不进文件、不参与轮转、也没有关联 id（实测在 `serve.err.log` 里见过
transformers 的进度条、pymupdf 的 DeprecationWarning、pymilvus 的整坨栈）。

落点长这样（2026-10-06 第二版：**业务账不再落 PG**，改成一类一个文件）：

    控制台(stderr)   全部日志，一个出口（进度与异常都要看得见，不按流分家）
    interface.log    `rag.interface`     一次 HTTP 请求一行
    agent.log        `rag.agent_run`     一轮问答的汇总一行
                     `rag.agent_llm`     每次模型调用一行（token 就在这一行）
                     `rag.agent_tool`    每次工具调用一行
    channel.log      `rag.channel`       一次查询 × 每一路一行
    system.log       其余（LangChain / pymilvus / httpx / uvicorn / 其他 rag.* 的系统日志）

**agent 的三类共用一个文件**：一轮的时间线要在同一个文件里连着读（`模型#1 → 工具#1 →
模型#2 …→ 汇总`），分成三个文件就断了。类别仍然分得开 —— 行里的 logger 短名就是类别
（见 `log.py` 的 `kind_of_line`）。

系统文件挂在 root 上但**排除**上面三个 logger —— 它们各有自己的文件，不在系统里再来一份。
三类业务日志的 `propagate` 保持默认的 True，所以它们同时也留在控制台里。

有一类库要单独处理：**自己装 handler 再把 `propagate` 关掉的**（pymilvus 实测就是：
一个裸 `StreamHandler(stderr)` + `propagate=False`）—— 它们一条都不会冒泡到 root。
`NOISY` 那几个 logger 因而**一律被接管**：摘掉库自带的 handler、关掉 propagate、
把 root 那一套原样挂上去。不这么做的话，`rag serve` 会漏掉整块 pymilvus 日志
（`configure_logging` 在 uvicorn 之前跑，那时 pymilvus 还没 import）。

三条约定（使用者 2026-10-06 定的）：

1. **控制台就一个出口**（stderr）；
2. 默认级别 **INFO**；
3. 第三方库压到 **WARNING** —— 但**影响了流程的异常要按 ERROR 记**（降级继续算 WARNING）。

`logging.raiseExceptions` 保持默认的 True：handler 自己故障（磁盘满、文件被占用）要看得见。
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

from rag.obs.live import LiveHandler
from rag.obs.log import current_ids, set_content
from rag.obs.pg import build_pg_handler

# 这些东西默认太吵：它们的每个连接、每个请求都打一条 INFO。压到 WARNING，
# 需要时按模块临时放开（`logging.getLogger("pymilvus").setLevel(logging.DEBUG)`）。
NOISY = (
    "httpx",
    "httpcore",
    "urllib3",
    "pymilvus",
    "transformers",
    "paddlex",
    "sentence_transformers",
    "asyncio",
    "urllib3.connectionpool",
)

# 时间戳**带毫秒**：四个文件的记录合并成一条流水时按时间排，只到秒的话同一秒里的
# 几条是并列的，稳定排序只能按「先读哪个文件」排 —— 看起来就是顺序乱（实测）。
FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-5s %(short)-16.16s%(ids)s %(message)s"
DATE_FORMAT = "%m-%d %H:%M:%S"

# 实时缓冲那一份：时间/级别/来源/关联 id 前端**都是独立字段**（见 `live.LiveHandler`），
# 这里只格式化「说了什么」—— 别再往正文里塞一遍 id，那会让页面上出现两份同样的东西
LIVE_FORMAT = "%(message)s"

# 业务日志的 logger（见模块说明）。**agent 拆三类但共用 agent.log** ——
# 一轮的时间线要在同一个文件里连着读，分成三个文件就断了。
BUSINESS_LOGGERS = (
    "rag.interface",
    "rag.channel",
    "rag.agent_run",
    "rag.agent_llm",
    "rag.agent_tool",
)

# 挂在我们自己 handler 上的记号：重复配置时按它摘掉（只摘我们的，不动别人的）
_MANAGED = "_rag_managed"


class CorrelationFilter(logging.Filter):
    """把 `request_id` / `run_id` 塞进**每一条** record。

    关键的不仅是「我们的日志带得上」，而是**第三方库的日志也带得上** —— 一次请求里
    LangGraph 内部、pymilvus 重试打印的东西，靠这个才能串成一条线。

    顺带把 `%(short)s`（logger 名的最后一段）补上：格式化字符串里没法做字符串运算。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        request_id, run_id = current_ids()
        parts = []
        if request_id:
            parts.append(f"req={request_id}")
        if run_id:
            parts.append(f"run={run_id}")
        record.ids = (" " + " ".join(parts)) if parts else ""
        record.short = record.name.rsplit(".", 1)[-1]
        return True


class OneLineFormatter(logging.Formatter):
    """**一条记录 = 一行**：把消息里的换行转义成 `\\n`。

    为什么必须有它：psycopg 的异常文本自带换行（`column … does not exist` +
    `LINE 1: insert …` + 定位用的 `^`），照原样写出去，文件里就多出几行**没有时间、
    没有级别**的续行 —— 页面上看是一条空行，读日志的人会以为出了两条问题。
    转义之后内容一个字符都不丢，而「一行一条」这个前提在四个文件里同时成立。

    控制台**不用**它：终端里栈要能正常换行才读得下去。
    """

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        return text.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n")


class ShorterNames(logging.Filter):
    """把 `uvicorn.error` / `uvicorn.access` 的短名统一成 `uvicorn`。

    不改的话日志里会写 `INFO  error  Application startup complete.` —— 看着像错误上报，
    其实只是 uvicorn 自己的两条通道（`error` 是它的运行信息、`access` 是请求行）。

    **必须排在 `CorrelationFilter` 前面**：短名是它算出来的，先改名才算得对。
    """

    _MAP = {"uvicorn.error": "uvicorn", "uvicorn.access": "uvicorn"}

    def filter(self, record: logging.LogRecord) -> bool:
        mapped = self._MAP.get(record.name)
        if mapped:
            record.name = mapped
        return True


def _filters(extra: tuple[logging.Filter, ...] = ()) -> tuple[logging.Filter, ...]:
    """每个 handler 都要过的那几道：先改名，再塞关联 id 与短名。"""
    return (ShorterNames(), CorrelationFilter(), *extra)


class ExcludeLoggers(logging.Filter):
    """系统文件用：业务日志已经有自己的文件，别在这里再来一份。"""

    def __init__(self, names: tuple[str, ...]) -> None:
        super().__init__()
        self._names = names

    def filter(self, record: logging.LogRecord) -> bool:
        return not any(
            record.name == name or record.name.startswith(f"{name}.") for name in self._names
        )


def _level(value: Any) -> int:
    """配置里写错级别要立刻报出来，别静默退到默认值。"""
    level = getattr(logging, str(value).upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"未知日志级别：{value!r}")
    return level


def _rotating(
    file_cfg: Any, formatter: logging.Formatter, extra_filters: tuple[logging.Filter, ...] = ()
) -> logging.Handler | None:
    """一个按大小轮转的文件 handler。`path` 留空 = 这一类不落文件。"""
    path_text = getattr(file_cfg, "path", "")
    if not path_text:
        return None
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        path,
        maxBytes=file_cfg.max_bytes or 0,
        backupCount=file_cfg.backups or 0,
        encoding="utf-8",
    )
    handler.setFormatter(formatter)
    for item in _filters(extra_filters):
        handler.addFilter(item)
    setattr(handler, _MANAGED, True)
    return handler


def _drop_managed(logger: logging.Logger) -> None:
    """把上次装的 handler 摘掉并**关掉**：不 close 会留着文件句柄（Windows 上锁文件）。"""
    for handler in list(logger.handlers):
        if getattr(handler, _MANAGED, False):
            logger.removeHandler(handler)
            handler.close()


def configure_logging(settings: Any) -> None:
    """按配置装好日志。**幂等**：重复调用不会挂两遍 handler，也不会留下旧文件句柄。

    装的是 root（第三方库的 logger 要冒泡到 root 才接得住）加三个业务 logger ——
    每个类别一个文件，见模块说明。顺手把 `content` 开关同步给 `log.py`。
    """
    cfg = settings.observability.logging
    root = logging.getLogger()
    # 控制台用普通 formatter（栈照旧换行读）；**文件与实时流用 OneLineFormatter**
    # （一条记录一行，见它的说明 —— 多行错误文本否则会变成「没有时间的空行」）
    formatter = logging.Formatter(FORMAT, datefmt=DATE_FORMAT)
    one_line = OneLineFormatter(FORMAT, datefmt=DATE_FORMAT)

    # 先拆上次装的（root + 三个业务 logger + 第三方那几个），再按现在的配置重建 ——
    # 配置里删掉的类别、上次挂到第三方 logger 上的那一套，都不会留下
    _drop_managed(root)
    for name in (*BUSINESS_LOGGERS, *NOISY):
        _drop_managed(logging.getLogger(name))

    set_content(bool(cfg.content))

    if cfg.console:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(formatter)
        for item in _filters():
            console.addFilter(item)
        setattr(console, _MANAGED, True)
        root.addHandler(console)

    system = _rotating(cfg.system, one_line, (ExcludeLoggers(BUSINESS_LOGGERS),))
    if system is not None:
        root.addHandler(system)

    # 进程内实时缓冲：页面 `/logs/stream` 读它（文件那一份是**历史**）。
    # 它也是「我们的 handler」，所以下面接管第三方 logger 时会一起挂过去。
    live = LiveHandler()
    live.setFormatter(OneLineFormatter(LIVE_FORMAT))
    for item in _filters():
        live.addFilter(item)
    setattr(live, _MANAGED, True)
    root.addHandler(live)

    # 落库那份**索引**（2026-10-08 加）：挂在 root 上，业务行冒泡上来、第三方库的行也收得到。
    # 它自己带队列与后台线程，写库失败只喊一句、绝不拖业务（见 `obs/pg.py` 的说明）。
    # 没配 PG（url 为空）或配置里关掉时返回 None —— 那种情况下日志只落文件，与从前一样。
    # **必须挂在这里**：下面接管第三方 logger 时会把 root 上这一套原样复制过去。
    pg = build_pg_handler(settings)
    if pg is not None:
        setattr(pg, _MANAGED, True)
        root.addHandler(pg)

    # 每类挂到自己的 logger 上；**同一个 path 只开一个 handler**（agent 那三类共用一个文件），
    # `propagate` 保持默认的 True，所以它们同时也进控制台。
    by_path: dict[str, logging.Handler] = {}
    for kind, file_cfg in (cfg.files or {}).items():
        path = str(getattr(file_cfg, "path", "") or "")
        if not path:
            continue
        if path not in by_path:
            handler = _rotating(file_cfg, one_line)
            if handler is None:
                continue
            by_path[path] = handler
        logging.getLogger(f"rag.{kind}").addHandler(by_path[path])

    levels = {name: logging.WARNING for name in NOISY}
    levels.update({name: _level(value) for name, value in (cfg.levels or {}).items()})
    for name, level in levels.items():
        logger = logging.getLogger(name)
        logger.setLevel(level)
        # **这些库自己会装 handler / 关掉 propagate**（pymilvus 实测：裸 StreamHandler(stderr)
        # + propagate=False），只挂 root 一条都收不到；而且它们往往是**在我们之后**才被
        # import 的（serve 先装日志、startup 才 import pymilvus），那时再判 propagate 已经晚了。
        # 所以这里一律接管：摘掉它自带的 handler（没有时间/关联 id/轮转的那份）、
        # 关掉 propagate、把 root 那一套原样挂过来。代价：库若在之后又给自己加回 handler，
        # 控制台上会多一份它的原始输出（诊断文件不受影响 —— 落点始终是我们这一套）。
        for handler in list(logger.handlers):
            if not getattr(handler, _MANAGED, False):
                logger.removeHandler(handler)
                handler.close()
        logger.propagate = False
        for handler in root.handlers:
            if getattr(handler, _MANAGED, False) and handler not in logger.handlers:
                logger.addHandler(handler)

    root.setLevel(_level(cfg.level))
    # uvicorn 自己那套 handler 会和我们重复一遍；`rag serve` 用 log_config=None 关掉它，
    # 这里只把它的级别收一下（access 的账我们有更全的接口日志）
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """统一前缀，顺手让 `rag` 这一支好按模块调级别。"""
    return logging.getLogger(name if name.startswith("rag") else f"rag.{name}")


def reset_logging() -> None:
    """把**我们装的** handler 全摘掉并关闭（root + 业务 logger + 被接管的第三方那几个）。

    给测试用：`logging` 的 handler 是**进程级**的，一个用例装过之后，后面的用例
    （以及它们触发的第三方日志）都会继续往同一个文件写 —— 用例之间要能互不打扰，
    更要保证不写进仓库里的 `data/logs/`（见 `tests/conftest.py`）。
    只摘我们自己的（`_MANAGED` 记号），别人的 handler 一根不动。
    """
    _drop_managed(logging.getLogger())
    for name in (*BUSINESS_LOGGERS, *NOISY):
        _drop_managed(logging.getLogger(name))
