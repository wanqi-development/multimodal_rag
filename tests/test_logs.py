"""日志层：一行摘要的格式、关联 id、内容开关、密钥不落盘、四个文件的落点。

落点用**临时目录真跑一遍** `configure_logging`（不连 PG —— 2026-10-06 起三类业务日志
各落一个文件，写库那套连 `PgLogSink` 一起删了）。
"""

from __future__ import annotations

import io
import logging
import logging.handlers
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from rag.obs import log as obs
from rag.obs.log import record_agent_run, record_channels, record_interface


def _settings(tmp_path: Path, **overrides: Any) -> Any:
    """把四个文件都指到临时目录的配置（不碰仓库里的 data/logs）。"""
    from rag.config.schema import FileLogConfig, LoggingConfig, Settings

    logging_cfg = LoggingConfig(
        console=False,
        files={
            "interface": FileLogConfig(path=str(tmp_path / "interface.log")),
            "agent_run": FileLogConfig(path=str(tmp_path / "agent.log")),
            "agent_llm": FileLogConfig(path=str(tmp_path / "agent.log")),
            "agent_tool": FileLogConfig(path=str(tmp_path / "agent.log")),
            "channel": FileLogConfig(path=str(tmp_path / "channel.log")),
        },
        system=FileLogConfig(path=str(tmp_path / "system.log")),
    )
    for key, value in overrides.items():
        setattr(logging_cfg, key, value)
    settings = Settings()
    settings.observability.logging = logging_cfg
    return settings


@contextmanager
def _captured() -> Iterator[io.StringIO]:
    """用**我们自己的 formatter + CorrelationFilter** 把日志收进内存。

    这样断言的就是控制台/文件里那一行真正的样子（含 req/run），而不是 record 的裸消息。
    """
    from rag.obs.setup import DATE_FORMAT, FORMAT, CorrelationFilter

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(FORMAT, datefmt=DATE_FORMAT))
    handler.addFilter(CorrelationFilter())
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        yield stream
    finally:
        root.removeHandler(handler)


# ---- 一行摘要（格式即落点里看到的内容）----


def test_one_line_prints_one_readable_summary_per_kind() -> None:
    from rag.obs.log import one_line

    interface = one_line(
        "interface",
        {
            "method": "POST",
            "path": "/chat",
            "status": 200,
            "duration_ms": 4123.4,
            "identity": "ip:127.0.0.1",
        },
    )
    # 「这是哪一类」由 logger 名承担（`rag.interface` / `rag.channel` / …），
    # 消息里不再重复一个方括号前缀
    assert interface.startswith("POST /chat 200 4123ms ip:127.0.0.1")

    agent = one_line(
        "agent_run",
        {
            "run_id": "r-1",
            "thread_id": "t1",
            "question_hash": "8f2a1c9d0e",
            "status": "answered",
            "steps": 14,
            "retrieved": 36,
            "cited": 7,
            "tokens_in": 22030,
            "tokens_out": 501,
            "tools": [{"name": "search_knowledge"}],
            "duration_ms": 22904.7,
        },
    )
    # run_id 由 `CorrelationFilter` 打在行首（`run=…`），消息里不再重复一遍
    assert agent.startswith("thread=t1")
    # content 关着 → 问题那一栏只给 hash 前 8 位（和通道一行一个口径）
    assert "问=8f2a1c9d" in agent
    assert "工具=search_knowledge" in agent and "token=22030/501" in agent

    channel = one_line(
        "channel", {"query_hash": "8f2a1c9d0e", "channel": "rrf", "candidates": 20, "kept": 8}
    )
    # 没开 content 就只打 hash 前 8 位（原文不往控制台甩）
    assert channel.startswith("8f2a1c9d rrf 召回=20 入选=8")


def test_content_on_puts_the_question_on_one_line() -> None:
    """`content` 打开时打原文，但**换行要压掉** —— 用户粘的问题就是多行的，
    一行一条是这套日志能被 rg 和 tail 读的前提。"""
    from rag.obs.log import one_line

    line = one_line(
        "agent_run",
        {"run_id": "r-2", "question_hash": "h", "question": "第一行\n第二行 " + "长" * 80},
    )

    assert "\n" not in line
    assert "问=第一行 第二行" in line


def test_failures_are_marked_on_the_same_line() -> None:
    from rag.obs.log import one_line

    line = one_line("channel", {"channel": "graph", "error": "Milvus 连不上"})

    assert "⚠Milvus 连不上" in line


def test_configure_logging_installs_the_console_and_quiets_noisy_libraries(tmp_path: Path) -> None:
    """装 root logger（第三方库才会冒泡进来）；吵的库压到 WARNING；故障不许静默。"""
    from rag.obs.setup import configure_logging

    configure_logging(_settings(tmp_path, console=True))
    root = logging.getLogger()

    kinds = [type(item) for item in root.handlers]
    assert logging.StreamHandler in kinds, "控制台一个出口（stderr）"
    assert logging.handlers.RotatingFileHandler in kinds, "系统那一份也挂在 root 上"
    assert logging.getLogger("pymilvus").level == logging.WARNING
    assert root.level == logging.INFO, "默认 INFO"
    assert logging.raiseExceptions is True, "handler 自己的故障要看得见（以前被关掉了）"


def test_the_cli_callback_installs_the_handlers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**CLI 也得装 handler**。只有 serve 装的时候，`rag ask` 这类命令的日志行（INFO）
    会因为 root 上一个 handler 都没有被直接丢掉 —— 现在日志只有文件这一个落点，更不能漏。
    """
    from typer.testing import CliRunner

    import rag.cli.main as cli

    monkeypatch.setattr(cli, "load_settings", lambda *a, **k: _settings(tmp_path, console=True))

    result = CliRunner().invoke(cli.app, ["config"])

    assert result.exit_code == 0, result.output
    assert logging.getLogger().handlers, "根回调必须把 handler 装上"


def test_each_kind_gets_its_own_file_and_system_gets_the_rest(tmp_path: Path) -> None:
    """三类各落自己的文件，第三方库进 system.log，且**不在系统文件里重复**业务行。"""
    from rag.obs.setup import configure_logging

    configure_logging(_settings(tmp_path))

    logger = logging.getLogger("rag.interface")
    attached = [h for h in logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
    assert attached and attached[0].baseFilename.endswith("interface.log")
    assert logger.propagate is True, "还要能冒泡到 root（控制台）"

    record_interface(
        method="POST", path="/search", status=200, duration_ms=12.0, identity="ip:127.0.0.1"
    )
    record_agent_run(run_id="r-1", question="Flink 是什么")
    logging.getLogger("pymilvus.client").warning("重试第 3 次")

    assert "POST /search 200" in (tmp_path / "interface.log").read_text("utf-8")
    assert "状态=answered" in (tmp_path / "agent.log").read_text("utf-8")
    assert "POST /search 200" not in (tmp_path / "agent.log").read_text("utf-8"), "两类不串"
    system = (tmp_path / "system.log").read_text("utf-8")
    assert "重试第 3 次" in system, "第三方库的日志要进系统文件"
    assert "POST /search 200" not in system, "业务日志不在系统文件里再来一份"


def test_a_tool_line_gives_args_and_output_not_just_their_lengths(tmp_path: Path) -> None:
    """**工具行不给「输入多少字 / 输出多少字」**（2026-10-10 使用者定）。

    「入参=query=19字 出参=12字」这种行，出了事根本没法分析 —— 大模型可以这么回答，
    工具不行。所以这一行**不吃 `content` 开关**（默认就是关的），永远给值，长了才截。
    """
    from rag.obs.log import record_agent_tool
    from rag.obs.setup import configure_logging

    configure_logging(_settings(tmp_path))
    record_agent_tool(
        run_id="r-1",
        index=1,
        tool="search_knowledge",
        args={"query": "水位线是什么", "top_k": 5},
        output="命中 8 块，第一条是 1.3 节",
    )

    line = (tmp_path / "agent.log").read_text("utf-8")
    assert "入参=query=水位线是什么,top_k=5 出参=命中 8 块，第一条是 1.3 节" in line
    assert "字" not in line, "不该出现任何「N 字」"


def test_a_long_tool_output_is_clipped_and_says_how_long_it_really_was(tmp_path: Path) -> None:
    """截断是允许的，但**截了多少要写明** —— 不然「看着像一条短结果」会骗人。"""
    from rag.obs.log import TOOL_OUTPUT_CHARS, record_agent_tool
    from rag.obs.setup import configure_logging

    configure_logging(_settings(tmp_path))
    head = "正文" * (TOOL_OUTPUT_CHARS // 2)  # 正好到上限（用「重复字」拼，尾巴才好认）
    tail = "尾巴" * 300  # 超出去的那一截：**与头部不重样**，不然「截掉了」这条断言会自欺
    long = head + tail
    record_agent_tool(run_id="r-1", index=1, tool="search_knowledge", args={}, output=long)

    line = (tmp_path / "agent.log").read_text("utf-8")
    assert f"出参={head}" in line, "上限以内的字要原样在行里"
    assert f"（共 {len(long)} 字）" in line, "尾巴上要写原本多长"
    assert tail not in line, "超出上限的那一截不落"


def test_a_long_argument_is_clipped_and_newlines_become_literal_escapes() -> None:
    """入参同理（`write_file` 的 content 能好几万字）；**换行转义成 `\\n` 字面量** ——

    工具出参与参数天然是多行的，文件那份格式化只认一行，库里那份也得逐字一样。
    """
    from rag.obs.log import TOOL_ARG_CHARS, _args_brief

    assert _args_brief({"content": "第一行\n第二行"}) == "content=第一行\\n第二行"

    long = "x" * (TOOL_ARG_CHARS + 5)
    clipped = _args_brief({"q": long})
    assert clipped.startswith("q=" + long[:TOOL_ARG_CHARS]) and clipped.endswith(
        f"…（共 {len(long)} 字）"
    )
    assert "\n" not in clipped


def test_configure_logging_is_idempotent_and_closes_old_handlers(tmp_path: Path) -> None:
    """重复配置（serve 起两次、测试连着跑）不许挂两遍 handler，也不许留着旧文件句柄。"""
    from rag.obs.setup import _MANAGED, configure_logging

    def managed(logger: logging.Logger) -> list[logging.Handler]:
        """只数我们自己挂的 —— pytest 自己也会往 root 上挂 LogCaptureHandler。"""
        return [handler for handler in logger.handlers if getattr(handler, _MANAGED, False)]

    configure_logging(_settings(tmp_path))
    before = managed(logging.getLogger())
    configure_logging(_settings(tmp_path))
    after = managed(logging.getLogger())

    assert len(before) == len(after) == 2, "console=False 时 root 上是 system + 实时缓冲"
    assert all(item not in after for item in before), "旧的整批要换掉，不能叠加"
    assert all(getattr(item, "stream", None) is None for item in before), "旧 handler 要 close"
    assert len(managed(logging.getLogger("rag.interface"))) == 1


def test_the_formatter_puts_time_ids_and_a_short_logger_name_on_one_line() -> None:
    """控制台上那一行的样子：时间、级别、logger 名（末段）、关联 id、消息。

    这套格式**对第三方库的日志同样生效** —— 它们的 record 也要过同一个 formatter、
    也会带上 `request_id`（见 `CorrelationFilter`），这才是一次请求串成一条线的关键。
    """
    import logging

    from rag.obs.setup import DATE_FORMAT, FORMAT, CorrelationFilter

    obs.set_ids(request_id="req-1", run_id="run-2")
    record = logging.LogRecord(
        "pymilvus.client", logging.WARNING, "x.py", 1, "重试第 %d 次", (3,), None
    )
    assert CorrelationFilter().filter(record) is True

    line = logging.Formatter(FORMAT, datefmt=DATE_FORMAT).format(record)

    assert "WARNING" in line and "client" in line, "第三方库的 record 也走同一套格式"
    assert "req=req-1 run=run-2" in line, "关联 id 要跟着第三方库的日志一起出来"
    assert line.endswith("重试第 3 次"), "消息本身（含 %d 的实参）"


@pytest.fixture(autouse=True)
def _clean() -> None:
    """每个用例前后都把全局状态还原 —— 内容开关、关联 id、logging handler 都是模块级的。"""
    obs.set_content(False)
    obs._REQUEST_ID.set(None)
    obs._RUN_ID.set(None)
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    for handler in list(root.handlers):
        root.removeHandler(handler)
    yield
    obs.set_content(False)
    obs._REQUEST_ID.set(None)
    obs._RUN_ID.set(None)
    # 我们自己挂的 handler 可能落在 root、三个业务 logger、以及自己关掉 propagate 的
    # 第三方 logger（pymilvus）上 —— 逐个摘掉并 close，别把句柄留给下一个用例
    loggers = [root] + [
        item
        for item in logging.Logger.manager.loggerDict.values()
        if isinstance(item, logging.Logger)
    ]
    for logger in loggers:
        for handler in list(logger.handlers):
            if getattr(handler, "_rag_managed", False):
                logger.removeHandler(handler)
                handler.close()
    for handler in saved_handlers:
        root.addHandler(handler)
    root.setLevel(saved_level)


def test_three_kinds_share_the_same_correlation_ids() -> None:
    """接口 → agent → 通道 靠同一套 id 串起来：这是整套日志能被读的前提。"""
    from rag.contracts import ChannelStats

    obs.set_ids(request_id="req-abc", run_id="run-xyz")
    with _captured() as out:
        record_interface(
            method="POST", path="/agent", status=200, duration_ms=12.0, identity="ip:1"
        )
        record_agent_run(run_id="run-xyz", question="Flink 是什么", steps=3)
        record_channels(
            query="Flink 是什么",
            channels=["hybrid:milvus:c", "bm25:milvus:c"],
            stats=[
                ChannelStats(channel="hybrid:milvus:c", candidates=10, kept=2, latency_ms=11.0),
                ChannelStats(channel="bm25:milvus:c", candidates=10, kept=1, latency_ms=9.0),
            ],
            rerank=True,
        )
    lines = out.getvalue().splitlines()

    assert len(lines) == 4, "接口 1 行 + agent 1 行 + 通道 2 行（一路一条）"
    assert all("req=req-abc" in line and "run=run-xyz" in line for line in lines), lines
    assert "interface" in lines[0] and "agent_run" in lines[1] and "channel" in lines[2]


def test_content_is_off_by_default_and_hash_is_always_there() -> None:
    """内容默认不落（语料与用户原话属于数据），默认只给 hash 前 8 位；开了才打原文。"""
    question = "Flink 的 Save Points 是做什么用的？"

    with _captured() as out:
        record_agent_run(run_id="run-1", question=question)
    assert "Save Points" not in out.getvalue()
    assert obs.digest(question)[:8] in out.getvalue()

    obs.set_content(True)
    with _captured() as out:
        record_agent_run(run_id="run-2", question=question)
    assert "Save Points" in out.getvalue()


def test_api_keys_never_land_in_the_logs() -> None:
    """身份只记 **sha1 前 12 位**（复用限流那套）—— key 原文一个字都不许进日志。"""
    from rag.api.ratelimit import identity_of

    secret = "sk-super-secret-key"
    with _captured() as out:
        record_interface(
            method="GET",
            path="/search",
            status=200,
            duration_ms=1.0,
            identity=identity_of(secret, None),
        )

    text = out.getvalue()
    assert secret not in text
    assert "key:" in text


def test_channel_rows_carry_the_evidence_needed_to_explain_a_miss() -> None:
    """「为什么这条没召回」要看得到：每路召回多少、留多少、多久、有没有挂、过滤是什么。"""
    from rag.contracts import ChannelStats

    with _captured() as out:
        record_channels(
            query="Python asyncio",
            channels=["hybrid:milvus:c"],
            stats=[
                ChannelStats(
                    channel="hybrid:milvus:c",
                    candidates=10,
                    kept=0,
                    latency_ms=380.0,
                    error="MilvusException: 超时",
                )
            ],
            rerank=True,
            filters={"modality": "text"},
        )

    line = out.getvalue().strip()
    assert "召回=10" in line and "入选=0" in line and "380ms" in line
    assert "⚠MilvusException: 超时" in line
    assert "过滤=modality=text" in line


def test_read_tail_reads_the_last_lines(tmp_path: Path) -> None:
    """`rag logs tail` 读文件尾巴：够最近几行；文件不在也不是错误。"""
    from rag.obs.log import read_tail

    path = tmp_path / "x.log"
    path.write_text("\n".join(f"line {i}" for i in range(50)) + "\n", encoding="utf-8")

    assert read_tail(path, limit=3) == ["line 47", "line 48", "line 49"]
    assert read_tail(tmp_path / "missing.log", limit=3) == []


def test_the_live_buffer_keeps_recent_lines_and_feeds_subscribers() -> None:
    """实时那一层：handler 收一条 → 环形缓冲 + 每个订阅者都拿到。

    页面**不能**去 tail 文件（实测 Windows 上持句柄会把轮转顶掉），所以实时必须走内存。
    """
    import asyncio

    from rag.obs import live
    from rag.obs.setup import LIVE_FORMAT, CorrelationFilter

    live.reset()
    handler = live.LiveHandler()
    handler.setFormatter(logging.Formatter(LIVE_FORMAT))
    handler.addFilter(CorrelationFilter())

    async def scenario() -> dict:
        with live.subscribe() as queue:
            record = logging.LogRecord("rag.channel", logging.INFO, "x.py", 1, "召回=10", (), None)
            handler.handle(record)
            await asyncio.sleep(0.05)  # call_soon_threadsafe 要等下一轮循环才投递
            return queue.get_nowait()

    item = asyncio.run(scenario())

    assert item["kind"] == "channel" and item["level"] == "INFO"
    # 来源单独给字段，正文里不再重复一遍短名（页面上是五列，不是六个东西挤在正文里）
    assert item["source"] == "channel" and not item["text"].startswith("channel")
    assert "召回=10" in item["text"]
    assert live.recent(5)[-1]["text"] == item["text"]
    live.reset()


def test_uvicorn_logs_show_up_as_uvicorn_not_error(tmp_path: Path) -> None:
    """`uvicorn.error` / `uvicorn.access` 的短名是 `error` / `access` —— 看着像错误上报。
    统一显示成 `uvicorn`（页面的「来源」列与文件里的那一栏都跟着变）。"""
    from rag.obs.setup import configure_logging

    settings = _settings(tmp_path)
    configure_logging(settings)

    logging.getLogger("uvicorn.error").info("Application startup complete.")

    line = (tmp_path / "system.log").read_text("utf-8")
    assert "uvicorn" in line and "error" not in line, line


def test_records_never_span_two_lines_in_the_file() -> None:
    """多行消息（psycopg 的 `LINE 1:` + `^`）在文件里必须压成一行。

    不压的话文件里会多出「没有时间、没有级别」的续行 —— 页面上是一条空行，读的人会
    以为出了两条问题（真机上就是这么看到的）。
    """
    from rag.obs.setup import DATE_FORMAT, FORMAT, CorrelationFilter, OneLineFormatter

    record = logging.LogRecord(
        "rag.storage", logging.ERROR, "x.py", 1, "第一行\nLINE 1: insert\n   ^", (), None
    )
    CorrelationFilter().filter(record)

    line = OneLineFormatter(FORMAT, datefmt=DATE_FORMAT).format(record)

    assert "\n" not in line
    assert "第一行\\nLINE 1: insert" in line and line.endswith("^")


def test_the_three_agent_kinds_share_one_file_and_handler(tmp_path: Path) -> None:
    """agent 拆成 agent_run / agent_llm / agent_tool 三类，但**共用一个文件** ——
    一轮的时间线要在同一个文件里连得上（分成三个文件就断了）。"""
    from rag.obs.setup import configure_logging

    configure_logging(_settings(tmp_path))

    attached = [
        logging.getLogger(f"rag.{name}").handlers
        for name in ("agent_run", "agent_llm", "agent_tool")
    ]

    assert all(len(item) == 1 for item in attached), "每个类别挂一个 handler"
    assert attached[0][0] is attached[1][0] is attached[2][0], "三类共用一个 handler 实例"
    assert attached[0][0].baseFilename.endswith("agent.log")


def test_tail_items_reads_one_file_that_carries_several_kinds(tmp_path: Path) -> None:
    """一个文件混着几类时，类别由**行里的 logger 短名**判断（`--kind` 是按行过滤的）。"""
    from rag.obs.log import tail_items

    agent = tmp_path / "agent.log"
    agent.write_text(
        "10-06 19:00:01 INFO  agent_llm       模型#1 deepseek-flash 输入=100 输出=7\n"
        "10-06 19:00:02 INFO  agent_tool      工具#1 search_knowledge "
        "入参=query=水位线是什么,top_k=5 出参=命中 8 块\n"
        "10-06 19:00:03 INFO  agent_run       thread=t1 状态=answered token=100/7\n",
        encoding="utf-8",
    )
    files = {"agent_run": str(agent), "agent_llm": str(agent), "agent_tool": str(agent)}

    items = tail_items(files, limit=10)

    assert [item["kind"] for item in items] == ["agent_llm", "agent_tool", "agent_run"]
    assert items[0]["source"] == "agent_llm" and "模型#1" in items[0]["text"]
    only_tools = tail_items(files, kind="agent_tool", limit=10)
    assert [item["kind"] for item in only_tools] == ["agent_tool"]


def test_tail_items_orders_within_the_same_second_by_milliseconds(tmp_path: Path) -> None:
    """同一秒里来自**不同文件**的记录，靠毫秒才排得出先后 —— 只到秒时它们是并列的，
    稳定排序只能按「先读哪个文件」排，看起来就是顺序乱（使用者报的就是这个）。"""
    from rag.obs.log import parse_line, tail_items

    interface = tmp_path / "interface.log"
    channel = tmp_path / "channel.log"
    interface.write_text(
        "10-07 10:00:01.900 INFO  interface        req=r1 GET /health 200 1ms\n"
        "10-07 10:00:01.100 INFO  interface        req=r2 GET /health 200 2ms\n",
        encoding="utf-8",
    )
    channel.write_text(
        "10-07 10:00:01.500 INFO  channel          req=r3 rrf 召回=1 入选=1 1ms\n",
        encoding="utf-8",
    )

    items = tail_items({"interface": str(interface), "channel": str(channel)}, limit=10)

    assert [item["ts"] for item in items] == [
        "10-07 10:00:01.100",
        "10-07 10:00:01.500",
        "10-07 10:00:01.900",
    ]
    # 老格式（只有到秒）也得认出来，别当成「没有时间」的续行接到上一条上
    assert parse_line("10-07 10:00:02 INFO  channel  老格式一行")[0] == "10-07 10:00:02"
