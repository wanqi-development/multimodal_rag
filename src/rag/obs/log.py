"""日志：三类业务账（接口 / agent 运行 / 通道）+ 一套关联 id。

2026-10-06 改：**不再落 PG**。原先的三张表（`interface_log` / `agent_run_log` /
`channel_log`）连同写库那一套一起删了 —— 三类各落**自己的文件**（见 `setup.py` 的
`configure_logging`：`interface.log` / `agent.log` / `channel.log`，第三方库与其余系统日志
另有 `system.log`）。要给统计用的 token 账在**会话级**另有一套
（`persistence/chat.py` 的 `token_usage`），不从这条线上出。

四条硬约束：

1. **写日志失败不许影响业务**：这里只调 stdlib `logging`，不碰网络也不碰数据库。
   handler 自己的故障交给 stdlib 报出来 —— `logging.raiseExceptions` 保持默认的 True，
   别再关（关掉之后磁盘满、文件被占用这类事会一声不响，实测）。
2. **关联 id 走 contextvars**：接口进来的 `request_id`、agent/CLI 的 `run_id` 由入口设置，
   链路深处的通道日志直接读 —— 不用把 id 一层层往下传（传着传着就会漏）。
3. **内容默认不落**：问题/查询原文只在 `observability.logging.content: true` 时进日志行；
   默认只留 **长度 + sha1 前 16 位**（能对齐、能去重，又不留语料与用户原话）。
   **答案一律不入日志**（它是会话数据，在 chat 表与控制台输出里）。
   **例外是工具那一行**（2026-10-10 使用者定）：工具的入参与出参**永远给内容、只截断** ——
   「入参=query=19字 出参=12字」这种只报长度的写法，出了事根本没法分析（大模型可以这么
   回答，工具不行）。见 `TOOL_*_CHARS` 与 `record_agent_tool()`。
4. **密钥永不落**：这里只记「身份的 sha1 前 12 位」（复用限流那套 `identity_of`），
   不记 API key / 密码 / token 原文。

三类各一行什么：

- `interface`：谁在什么时候打了哪个端点、多久、什么结果；
- `agent_run`：这一轮 agent 干了什么（工具清单、步骤、召回、引用、token、状态、耗时）；
- `channel`：每个通道召回了多少、留下多少、多久、有没有挂 —— 「为什么这条没召回」的证据链。
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# 关联 id：接口请求 / 一次 agent 或 CLI 运行。链路深处直接读，不用层层传参
_REQUEST_ID: ContextVar[str | None] = ContextVar("rag_request_id", default=None)
_RUN_ID: ContextVar[str | None] = ContextVar("rag_run_id", default=None)
# 入口是哪一种：`http` / `cli`。**不靠 id 前缀猜** —— 两者发的都是 `req-`（同一个形状，
# 查询时不用分两套），"谁发起的"这一栏由它承担（见 `persistence/logs.py` 的表）。
_SOURCE: ContextVar[str] = ContextVar("rag_log_source", default="")


def digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def set_ids(*, request_id: str | None = None, run_id: str | None = None) -> None:
    """入口处设置关联 id（另一个参数为 None 表示不动它）。"""
    if request_id is not None:
        _REQUEST_ID.set(request_id)
    if run_id is not None:
        _RUN_ID.set(run_id)


def set_source(source: str) -> None:
    """入口是哪一种（`http` / `cli`）—— 落库那一列用它，不从 id 前缀反推。"""
    _SOURCE.set(str(source or ""))


def current_source() -> str:
    return _SOURCE.get()


def current_ids() -> tuple[str | None, str | None]:
    return _REQUEST_ID.get(), _RUN_ID.get()


# ---- 内容开关：问题/查询原文要不要进日志行 ----

_CONTENT = False


def set_content(enabled: bool) -> None:
    """要不要把问题/查询原文写进日志行（默认关，见 `LoggingConfig`）。"""
    global _CONTENT
    _CONTENT = bool(enabled)


# ---- 一行摘要 ----


def _ms(value: Any) -> str:
    return f"{value:.0f}ms" if isinstance(value, (int, float)) else "-"


def _flat(text: Any, width: int) -> str:
    """压成一行再截断：原文里的换行会把「一行一条」毁掉（用户粘的问题就是多行的）。"""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= width else flat[:width] + "…"


def one_line(kind: str, row: dict[str, Any]) -> str:
    """一条业务账 → 一行摘要。**只打摘要，不打原文**（除非 `content` 开着）。

    「这是哪一类」由 logger 名承担（`rag.interface` / `rag.channel` / …），
    所以这里不再自己加 `[接口]` 之类的方括号前缀。
    """
    if kind == "interface":
        line = (
            f"{row.get('method', '?')} {row.get('path', '?')} {row.get('status', '?')} "
            f"{_ms(row.get('duration_ms'))} {row.get('identity') or '-'}"
        )
        extras = []
        if row.get("response_bytes") is not None:
            extras.append(f"{row['response_bytes']}B")
        # 身份是 ip 的时候已经在上面那一栏了，别重复写一遍
        host = row.get("client_host")
        if host and not str(row.get("identity") or "").startswith("ip:"):
            extras.append(f"ip={host}")
        if extras:
            line = f"{line} {' '.join(extras)}"
    elif kind == "agent_run":
        tools = ",".join(str(item.get("name")) for item in (row.get("tools") or []))
        # content 关着时只给 hash 前 8 位（和通道那一行一个口径：能对齐，不留原话）
        ask = row.get("question") or str(row.get("question_hash") or "")[:8]
        # **不再重复 run_id**：`CorrelationFilter` 已经在每行前面打了 `run=…`
        line = (
            f"thread={row.get('thread_id') or '-'} "
            f"问={_flat(ask, 60)} 工具={tools or '无'} 步={row.get('steps')} "
            f"召回={row.get('retrieved')} 引用={row.get('cited')} "
            f"状态={row.get('status')} token={row.get('tokens_in')}/{row.get('tokens_out')} "
            f"{_ms(row.get('duration_ms'))}"
        )
    elif kind == "agent_llm":
        line = _llm_line(row)
    elif kind == "agent_tool":
        line = _tool_line(row)
    else:
        query = row.get("query") or str(row.get("query_hash") or "")[:8]
        line = (
            f"{_flat(query, 60)} {row.get('channel', '?')} 召回={row.get('candidates')} "
            f"入选={row.get('kept')} {_ms(row.get('latency_ms'))}"
        )
        filters = row.get("filters")
        if filters:
            line = f"{line} 过滤={','.join(f'{key}={value}' for key, value in filters.items())}"
    if row.get("error"):
        line = f"{line}  ⚠{_flat(row['error'], 120)}"
    return line


def _write(kind: str, row: dict[str, Any]) -> None:
    """所有 `record_*` 的唯一出口：一行摘要走 logging。

    三类各挂在自己的 logger（`rag.interface` / `rag.agent_run` / `rag.channel`）上，
    于是各进各的文件、同时也冒泡到控制台。**装 handler 是 `configure_logging` 的事**
    （CLI 根回调与 `serve` 都调）—— 没装的话 INFO 会被丢掉。

    顺手把这一行**原样挂在 record 上**（`extra={"rag_row": …}`）：日志进 PG 那个
    handler 直接取它，不用再从格式化好的字符串反解字段（反解迟早会对不上）。
    摘要是先算好再当 message 传的，所以文件/控制台里那一行一个字都没变。
    """
    text = one_line(kind, row)
    logging.getLogger(f"rag.{kind}").info(
        text,
        extra={
            "rag_row": {
                "category": kind,
                "text": text,
                "status": row.get("status"),
                "duration_ms": row.get("duration_ms"),
            }
        },
    )


# 第三方库与非业务 `rag.*` 那些行的正文：默认 `%(message)s`（不带时间/级别，
# 那两栏在库里是独立的列）
_PLAIN = logging.Formatter("%(message)s")


def _int_ms(value: Any) -> int | None:
    """耗时进库是 int 列（毫秒）。非数字一律当没给。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(round(float(value)))


def row_of(record: logging.LogRecord) -> dict[str, Any] | None:
    """一条 record → 一行落库字段。**谁都能进**：我们的业务行、第三方库、其余 `rag.*`。

    落库那份是**索引**（文件才是唯一真相），所以这里不挑食：

    * 业务行（`rag_row` 有值）：类别、正文、状态、耗时都用 `_write` 算好的那一份；
    * 其余 `rag.*`：归 `system`，正文是消息本身；
    * 第三方库：也归 `system`，`source` 记成 `library` —— 一次请求里 pymilvus 报的那句
      重试，靠同一个 `request_id` 就能和接口行串起来，这正是上库的用处。
    """
    extra = getattr(record, "rag_row", None)
    request_id, run_id = current_ids()
    entry = current_source()
    if extra:
        category = str(extra.get("category") or "system")
        text = str(extra.get("text") or "")
        status = extra.get("status")
        duration_ms = _int_ms(extra.get("duration_ms"))
        # 业务行的「谁发起的」看入口（http / cli）；都不在就归 task（后台任务、测试进程）
        source = entry or "task"
    else:
        category = "system"
        # 多行（psycopg 的 `LINE 1:` + `^`、栈）压成一行：库里一条就是一条，
        # 与文件那边 `OneLineFormatter` 的口径一致（内容一个字符都不丢）
        text = (
            _PLAIN.format(record).replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n")
        )
        status = None
        duration_ms = None
        source = (entry or "task") if record.name.startswith("rag.") else "library"
    return {
        "ts": datetime.fromtimestamp(record.created, tz=UTC),
        "category": category,
        "level": record.levelname,
        "request_id": request_id or "",
        "run_id": run_id,
        "source": source,
        "logger": record.name,
        "status": int(status) if isinstance(status, int) else None,
        "duration_ms": duration_ms,
        "text": text,
    }


# ---- agent 一轮里的时间线（模型调用 / 工具调用各一行）----

# 工具那一行的预览上限（2026-10-10）。**这两样不吃 `content` 开关**：工具干了什么、
# 拿回了什么，是排查一轮问答唯一的一手材料 —— 只给「多少字」等于没给。
#   `write_file` 的 content、`search_knowledge` 的返回都能很长，所以还是要截；
#   截多少按「一眼能认出这次调用干了什么」来定，别把整篇正文灌进日志行。
TOOL_ARG_CHARS = 200  # 单个参数值
TOOL_ARGS_CHARS = 400  # 整条入参串（参数多的时候）
TOOL_OUTPUT_CHARS = 600  # 出参


def _flatten(text: str) -> str:
    """预览压成一行：换行转义成 `\\n` 字面量（与文件那份 `OneLineFormatter` 同一口径）。

    工具出参天然是多行的（检索结果一条一段、报错带栈），不处理的话库里那份 `log_line.text`
    会带真换行 —— 同一个值在文件、实时流、库里长得不一样，查起来就会怀疑自己看错了源。
    """
    return text.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n")


def _clip(text: str, width: int) -> str:
    """超了才截，并在尾巴上写明原本有多长（截断本身也是信息）。"""
    return text if len(text) <= width else f"{text[:width]}…（共 {len(text)} 字）"


def _brief(value: Any) -> str:
    """入参里的一个值：**给值**，长的截断。数字/布尔/None 原样。"""
    if isinstance(value, (int, float, bool)) or value is None:
        return str(value)
    return _clip(_flatten(value if isinstance(value, str) else repr(value)), TOOL_ARG_CHARS)


def _args_brief(args: dict[str, Any] | None) -> str:
    """工具入参：`query=水位线是什么,top_k=5` 这种形状 —— **值就在行里**，只截断。"""
    if not args:
        return "—"
    line = ",".join(f"{key}={_brief(value)}" for key, value in args.items())
    return _clip(line, TOOL_ARGS_CHARS)


def _llm_line(row: dict[str, Any]) -> str:
    cache = row.get("cache_read")
    return (
        f"模型#{row.get('index')} {row.get('model') or '?'} "
        f"输入={row.get('tokens_in')} 输出={row.get('tokens_out')} 共={row.get('tokens_all')}"
        + (f" 缓存={cache}" if cache else "")
        + f" {_ms(row.get('duration_ms'))} 收尾={row.get('finish') or '?'}"
    )


def _tool_line(row: dict[str, Any]) -> str:
    preview = row.get("output") or "（空）"
    return (
        f"工具#{row.get('index')} {row.get('tool')} "
        f"入参={row.get('args') or '—'} 出参={preview} {_ms(row.get('duration_ms'))}"
    )


def record_agent_llm(
    *,
    run_id: str,
    index: int,
    usage: dict[str, Any] | None = None,
    model: str | None = None,
    duration_ms: float | None = None,
    finish: str | None = None,
) -> None:
    """**一次模型调用一行**（agent 时间线）。

    `usage` 是 v3 `message-finish` 里那一份（`input_tokens` / `output_tokens` /
    `total_tokens` / `input_token_details.cache_read`）。**输入是累积上下文** ——
    第 n 次的输入里含着前几次的全部内容，所以逐次相加不等于「这一轮花了多少」；
    要给统计用的口径仍看会话级的 `token_usage`。
    """
    usage = usage or {}
    details = usage.get("input_token_details") or {}
    _write(
        "agent_llm",
        {
            "run_id": run_id,
            "index": int(index),
            "model": model or "?",
            "tokens_in": usage.get("input_tokens"),
            "tokens_out": usage.get("output_tokens"),
            "tokens_all": usage.get("total_tokens"),
            "cache_read": details.get("cache_read"),
            "duration_ms": duration_ms,
            "finish": finish,
        },
    )


def record_agent_tool(
    *,
    run_id: str,
    index: int,
    tool: str,
    args: dict[str, Any] | None = None,
    output: Any = "",
    ok: bool = True,
    duration_ms: float | None = None,
) -> None:
    """**一次工具调用一行**（agent 时间线）。

    **入参、出参一律给内容、超长才截**（2026-10-10 使用者定：工具不能只报「多少字」）——
    这一行不受 `observability.logging.content` 管，理由见 `TOOL_ARG_CHARS` 上面那段。
    截断在尾部写明原本长度；换行转义成 `\\n` 字面量，三个落点（文件 / 实时流 / 库里那份）
    逐字一致。入参本来是 `write_file` 的 content 这种几万字的量，照样截到 `TOOL_ARG_CHARS`。
    """
    text = output if isinstance(output, str) else str(output)
    preview = _clip(_flatten(text), TOOL_OUTPUT_CHARS) if text else ""
    _write(
        "agent_tool",
        {
            "run_id": run_id,
            "index": int(index),
            "tool": tool,
            "args": _args_brief(args),
            "output": preview,
            "output_chars": len(text),
            "duration_ms": duration_ms,
            "error": None if ok else "工具返回错误（栈见 system.log 的 rag.agent.tool）",
        },
    )


def record_interface(
    *,
    method: str,
    path: str,
    status: int,
    duration_ms: float,
    identity: str | None = None,
    client_host: str | None = None,
    response_bytes: int | None = None,
    error: str | None = None,
) -> None:
    """一次 HTTP 请求的账。**同步**：只写一行日志，没有可等的东西。

    关联 id 从 contextvars 取（中间件已经 `set_ids`），所以这里不再接 `request_id`。
    """
    _write(
        "interface",
        {
            "method": method,
            "path": path,
            "status": int(status),
            "duration_ms": float(duration_ms),
            "identity": identity,
            "client_host": client_host,
            "response_bytes": response_bytes,
            "error": error,
        },
    )


def record_agent_run(
    *,
    run_id: str,
    question: str,
    thread_id: str | None = None,
    steps: int = 0,
    tools: list[dict[str, Any]] | None = None,
    retrieved: int = 0,
    cited: int = 0,
    status: str = "answered",
    tokens: tuple[int | None, int | None] = (None, None),
    duration_ms: float | None = None,
    error: str | None = None,
) -> None:
    """一轮 agent 的账。

    **答案不在这里**（它属会话数据）—— 只有问题那一栏，且受 `content` 管。
    """
    _write(
        "agent_run",
        {
            "run_id": run_id,
            "thread_id": thread_id,
            "question_hash": digest(question),
            "question_chars": len(question),
            "question": question if _CONTENT else None,
            "steps": int(steps),
            "tools": tools or [],
            "retrieved": int(retrieved),
            "cited": int(cited),
            "status": status,
            "tokens_in": tokens[0],
            "tokens_out": tokens[1],
            "duration_ms": duration_ms,
            "error": error,
        },
    )


def record_channels(
    *,
    query: str,
    channels: list[str],
    stats: list[Any],
    rerank: bool,
    hits: list[Any] | None = None,
    filters: dict[str, Any] | None = None,
) -> None:
    """一次查询的每路账本，一路一行。`stats` 是 `ChannelStats` 列表（含 rrf 与 rerank 两行）。"""
    top_score = max((float(hit.score) for hit in hits or []), default=None)
    hit_count = len(hits or [])
    for item in stats:
        _write(
            "channel",
            {
                "query_hash": digest(query),
                "query_chars": len(query),
                "query": query if _CONTENT else None,
                "channels": list(channels),
                "channel": str(getattr(item, "channel", "")),
                "candidates": int(getattr(item, "candidates", 0) or 0),
                "kept": int(getattr(item, "kept", 0) or 0),
                "latency_ms": getattr(item, "latency_ms", None),
                "error": getattr(item, "error", None),
                "rerank": bool(rerank),
                "top_score": top_score,
                "hit_count": hit_count,
                "filters": filters or None,
            },
        )


def new_id(prefix: str) -> str:
    """短 id（8 位）：够区分一次请求/一轮运行，日志里也好读。"""
    from uuid import uuid4

    return f"{prefix}-{uuid4().hex[:8]}"


def read_tail(path: str | Path, *, limit: int = 10, window: int = 262_144) -> list[str]:
    """读文件末尾的 `limit` 行（`rag logs tail` 用）。

    **开-读-关，不长期持有句柄**：Windows 上持着句柄会把 `RotatingFileHandler` 的轮转
    顶掉（实测 `WinError 32`，而且那条日志会整条丢掉），所以读到就关。
    文件比 `window` 字节还长时只读最后一截，避免为一个 tail 把 20MB 全读一遍。
    """
    target = Path(path)
    if not target.is_file():
        return []
    with target.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(0, os.SEEK_END)
        start = max(0, handle.tell() - window)
        handle.seek(start)
        if start:
            handle.readline()  # 丢掉被窗口截断的半个首行
        lines = handle.read().splitlines()
    return [line for line in lines[-limit:] if line.strip()]


# ---- 读日志文件：`rag logs tail` 与 `/logs/tail` **共用这一段** ----
#
# **一个文件可以被多个类别共用**：`agent.log` 同时装 agent_run / agent_llm / agent_tool
# （一轮的汇总、每次模型调用、每次工具调用）。所以「这一行属于哪一类」要看行里的
# logger 短名，不能只看文件。

KIND_BY_SHORT = {
    "interface": "interface",
    "channel": "channel",
    "agent_run": "agent_run",
    "agent_llm": "agent_llm",
    "agent_tool": "agent_tool",
}

# 按类别过滤时先在「最近这么多行」里找（见 `tail_items`）
FILTER_WINDOW = 2000


def kind_of_line(source: str, fallback: str) -> str:
    """行里的 logger 短名 → 类别；认不出来就用这个文件的主类别（第三方库都归 system）。"""
    return KIND_BY_SHORT.get(source, fallback)


def parse_line(line: str) -> tuple[str | None, str, str, str]:
    """`10-06 18:02:33.123 INFO  interface  req=… POST /search 200`
    → 时间 / 级别 / **来源**（logger 短名） / 其余。

    格式是我们自己定的（`obs/setup.py` 的 `FORMAT`），所以可以放心切；切不动就把整行
    当正文（宁可显示得糙，也不能因为一行格式怪就把它吞了）。

    毫秒是**可选**的：2026-10-07 之前写的行只有到秒（`18:02:33`），两种都要认。
    """
    parts = line.split(" ", 3)
    if (
        len(parts) == 4
        and re.fullmatch(r"\d\d-\d\d", parts[0])
        and re.fullmatch(r"\d\d:\d\d:\d\d(\.\d{3})?", parts[1])
    ):
        # 级别那一栏是 `%(levelname)-5s` 补过空格的；来源就是紧接着那个 logger 短名
        source, _, rest = parts[3].lstrip().partition(" ")
        return f"{parts[0]} {parts[1]}", parts[2], source, rest.lstrip()
    return None, "", "", line


# 行首的关联 id 是成对出现的（`CorrelationFilter` 打的：先是 req，后是 run，各自后面一个空格）
_IDS_PREFIX = re.compile(r"^(?:req=(?P<req>\S+) )?(?:run=(?P<run>\S+) )?")


def split_ids(rest: str) -> tuple[str, str, str]:
    """摘出行首的 `req=… run=…` → `(request_id, run_id, 正文)`。

    页面上这两个值要能单独显示、还要能点（点一下按这个 id 串出整条链路），所以不能让它们
    埋在正文里。都不在时原样返回（老行、第三方库的行没有 id）。
    """
    match = _IDS_PREFIX.match(rest)
    if match is None:
        return "", "", rest
    return match.group("req") or "", match.group("run") or "", rest[match.end() :]


def _stamp(ts: str | None) -> datetime | None:
    """`MM-DD HH:MM:SS` → 可比较的时间。**跨年也认**：读到「未来」就退回上一年
    （12-31 的日志在 01-01 打开时就是这个情形）。"""
    if not ts:
        return None
    year = datetime.now().year
    # 毫秒可有可无（老行只有到秒）；带上年份再解析，省得 3.15 起告警
    fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in ts else "%Y-%m-%d %H:%M:%S"
    try:
        parsed = datetime.strptime(f"{year}-{ts}", fmt)
    except ValueError:
        return None
    if parsed - datetime.now() > timedelta(days=1):
        parsed = parsed.replace(year=parsed.year - 1)
    return parsed


def tail_items(
    files: dict[str, str], *, kind: str = "all", limit: int = 200
) -> list[dict[str, Any]]:
    """读这些文件的尾巴，按时间合成一条流水。

    `files` 是「类别 → 路径」；**同一路径只读一次**，每行的类别由行里的 logger 短名决定。
    老格式留下的续行（多行异常文本）并进上一条，不另算一条。

    `limit` 是**最终要几行**。按类别过滤时会先在最近 `FILTER_WINDOW` 行里找 ——
    只读 `limit` 行的话，最新那几行可能全是别的类别（满屏 interface），会一条都找不到。
    """
    wanted = files if kind == "all" else {kind: files.get(kind, "")}
    window = limit if kind == "all" else max(limit, FILTER_WINDOW)
    picked: list[tuple[datetime | None, dict[str, Any]]] = []
    for path in dict.fromkeys(wanted.values()):
        if not path:
            continue
        # 这个文件的主类别：认不出短名的行（第三方库那种）都归它
        fallback = next(name for name, value in files.items() if value == path)
        for line in read_tail(path, limit=window):
            ts, level, source, text = parse_line(line)
            if ts is None and picked and picked[-1][1]["path"] == path:
                previous = picked[-1][1]
                previous["text"] = f"{previous['text']} ¶ {text.strip()}"
                continue
            request_id, run_id, body = split_ids(text)
            picked.append(
                (
                    _stamp(ts),
                    {
                        "kind": kind_of_line(source, fallback),
                        "ts": ts,
                        "level": level,
                        "source": source,
                        "request_id": request_id,
                        "run_id": run_id,
                        "text": body,
                        "path": path,
                    },
                )
            )
    picked.sort(key=lambda item: (item[0] is None, item[0] or datetime.min))
    return [item for _, item in picked if kind == "all" or item["kind"] == kind][-limit:]
