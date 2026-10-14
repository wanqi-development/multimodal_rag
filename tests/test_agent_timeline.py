"""v3 事件流 → 界面事件 + agent 时间线日志。

**不连模型**：这里喂的是真机抓下来的 v3 事件形状（`messages` 的
`content-block-delta` / `message-finish`，`tools` 的 `tool-started` / `tool-finished`，
`values` 的状态快照），用一个只实现 `astream_events` / `aget_state` 的替身。
真实流式模型那条路（DeepSeek 之类的 `content-block-delta`）就是这么长的。
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from rag.agent.build import _stream_graph
from rag.config.schema import FileLogConfig, LoggingConfig, Settings
from rag.obs.setup import configure_logging

# ---- v3 信封与替身 ----


def _env(method: str, data: Any, seq: int = 1) -> dict[str, Any]:
    return {
        "type": "event",
        "method": method,
        "seq": seq,
        "params": {"namespace": [], "timestamp": 0, "data": data},
    }


def _msg(phase: str, **fields: Any) -> dict[str, Any]:
    """`method="messages"` 的那一条：`(内层事件, 元数据)` 二元组。"""
    return _env("messages", ({**fields, "event": phase}, {"langgraph_node": "model"}))


class _Stream:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events

    def __aiter__(self):
        async def walk():
            for item in self._events:
                yield item

        return walk()


class _Agent:
    """只实现我们用的两口：`astream_events`（v3）与 `aget_state`。"""

    def __init__(self, events: list[dict[str, Any]], *, state: Any = None) -> None:
        self._events = events
        self._state = state
        self.versions: list[str] = []

    async def astream_events(self, payload: Any, config: Any = None, version: str = "v2"):
        self.versions.append(version)
        return _Stream(self._events)

    async def aget_state(self, config: Any) -> Any:
        if self._state is None:
            raise RuntimeError("没有 checkpointer")
        return self._state


def _run_events(*, state: Any = None, output: str = "命中 8 块") -> list[dict[str, Any]]:
    """一轮最小可用的 v3 流水：正文两段 + 一次工具 + 第二次模型调用。"""
    return [
        _env("values", {"messages": [HumanMessage(content="Savepoint 是什么")]}),
        _msg("message-start", role="ai", id="lc-1"),
        _msg("content-block-delta", index=0, delta={"type": "text-delta", "text": "Savepoint 是"}),
        _msg("content-block-delta", index=0, delta={"type": "text-delta", "text": "快照"}),
        _msg(
            "message-finish",
            usage={
                "input_tokens": 4097,
                "output_tokens": 94,
                "total_tokens": 4191,
                "input_token_details": {"cache_read": 3963},
            },
            metadata={"model_name": "deepseek-flash", "finish_reason": "tool_calls"},
        ),
        _env(
            "tools",
            {
                "event": "tool-started",
                "tool_call_id": "c1",
                "tool_name": "search_knowledge",
                "input": {"query": "水位线是什么", "top_k": 5},
            },
        ),
        _env("tools", {"event": "tool-finished", "tool_call_id": "c1", "output": output}),
        _env(
            "values",
            {"messages": [HumanMessage(content="Savepoint 是什么"), AIMessage(content="答案 [1]")]},
        ),
        _msg("message-start", role="ai", id="lc-2"),
        _msg(
            "message-finish",
            usage={"input_tokens": 100, "output_tokens": 7, "total_tokens": 107},
            metadata={"model_name": "deepseek-flash", "finish_reason": "stop"},
        ),
    ]


def _collect(agent: _Agent) -> list[dict[str, Any]]:
    async def drain() -> list[dict[str, Any]]:
        from rag.agent.tools import CitationRegistry

        return [
            event
            async for event in _stream_graph(
                agent, {"messages": []}, "t1", CitationRegistry(), question="Savepoint 是什么"
            )
        ]

    return asyncio.run(drain())


# ---- 落点：临时目录里的四个文件 ----


@pytest.fixture
def logs(tmp_path: Path):
    settings = Settings()
    settings.observability.logging = LoggingConfig(
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
    configure_logging(settings)
    yield tmp_path
    for logger in [
        logging.getLogger(),
        *(
            item
            for item in logging.Logger.manager.loggerDict.values()
            if isinstance(item, logging.Logger)
        ),
    ]:
        for handler in list(logger.handlers):
            if getattr(handler, "_rag_managed", False):
                logger.removeHandler(handler)
                handler.close()


def _agent_log(tmp_path: Path) -> str:
    return (tmp_path / "agent.log").read_text("utf-8")


# ---- 用例 ----


def test_v3_stream_becomes_tokens_tools_and_timeline_lines(logs: Path) -> None:
    state = SimpleNamespace(
        values={
            "messages": [HumanMessage(content="Savepoint 是什么"), AIMessage(content="答案 [1]")]
        }
    )
    agent = _Agent(_run_events(), state=state)

    events = _collect(agent)

    assert agent.versions == ["v3"], "必须走 v3"
    assert [event["text"] for event in events if event["type"] == "token"] == [
        "Savepoint 是",
        "快照",
    ]
    assert events[-1]["type"] == "done" and events[-1]["answer"] == "答案 [1]"
    tool = next(event for event in events if event["type"] == "tool")
    assert tool["name"] == "search_knowledge"
    assert tool["args"] == {"query": "水位线是什么", "top_k": 5}

    text = _agent_log(logs)
    assert "模型#1 deepseek-flash 输入=4097 输出=94 共=4191 缓存=3963" in text
    assert "收尾=tool_calls" in text
    # **工具行不吃 content 开关**（2026-10-10 使用者定）：关着也把入参、出参写出来
    assert "工具#1 search_knowledge 入参=query=水位线是什么,top_k=5 出参=命中 8 块" in text
    assert "query=6字" not in text and "出参=6字" not in text, "工具行不该只报「多少字」"
    assert "模型#2 deepseek-flash 输入=100 输出=7 共=107" in text
    # 汇总行在最后一条（token 是逐次累加：4097+100 / 94+7）
    assert "状态=answered token=4197/101" in text.strip().splitlines()[-1]


def test_the_tool_line_shows_values_while_the_question_stays_gated(logs: Path) -> None:
    """工具行给值，但**问题原文仍被 `content` 挡着** —— 两者在同一份日志里对照着看。

    使用者 2026-10-10：「工具不能只报输入多少字、输出多少字」。改的就是工具这一行，
    `content` 那个开关管的是问题/查询原文，不是这里。
    """
    _collect(_Agent(_run_events(output="命中 8 块，第一条是 1.3 节")))

    text = _agent_log(logs)
    run_line = next(line for line in text.splitlines() if "agent_run" in line)

    assert "入参=query=水位线是什么,top_k=5" in text
    assert "出参=命中 8 块，第一条是 1.3 节" in text
    assert "问=Savepoint" not in run_line and "问=" in run_line, "问题原文仍只留 hash"


def test_v3_interrupt_comes_from_the_state_snapshot(logs: Path) -> None:
    """v3 的流里没有中断（实测）：收尾从 `state.tasks[].interrupts` 取。"""
    pending = {"action_requests": [{"name": "write_file", "args": {"path": "a.md"}}]}
    state = SimpleNamespace(
        values={"messages": [HumanMessage(content="写个报告")]},
        tasks=[SimpleNamespace(interrupts=[SimpleNamespace(value=pending)])],
    )

    events = _collect(_Agent(_run_events(), state=state))

    kinds = [event["type"] for event in events]
    assert kinds[-1] == "interrupt" and "done" not in kinds
    assert events[-1]["interrupt"] == pending
    assert "状态=interrupted" in _agent_log(logs)


def test_without_a_checkpointer_the_values_snapshot_still_gives_the_answer(logs: Path) -> None:
    """没接 checkpointer 时 `aget_state` 问不到 —— 用流里的 `values` 兜底（v2 靠 messages 流）。"""
    events = _collect(_Agent(_run_events(), state=None))

    assert events[-1]["type"] == "done"
    assert events[-1]["answer"] == "答案 [1]"


def _many_tools(count: int) -> list[dict[str, Any]]:
    """一份只有工具调用的流水（不产生正文）—— 用来把工具预算顶穿。"""
    events: list[dict[str, Any]] = [_env("values", {"messages": [HumanMessage(content="问题")]})]
    for index in range(count):
        events.append(
            _env(
                "tools",
                {
                    "event": "tool-started",
                    "tool_call_id": f"c{index}",
                    "tool_name": "get_section",
                    "input": {"doc_id": "d1", "heading": f"第{index}节"},
                },
            )
        )
        events.append(
            _env(
                "tools",
                {"event": "tool-finished", "tool_call_id": f"c{index}", "output": "正文"},
            )
        )
    return events


def test_the_run_stops_after_the_tool_budget(logs: Path) -> None:
    """工具调用到上限就**收手**，并在答案里说清是「到了上限」。

    这条线的由来（2026-10-08 实测）：一次问答里模型做了 8 次 `get_section`，一节一节翻材料，
    6 次模型往返、输入 token 12.3 万、61 秒。收手之后不再往下消费流 —— 后面每多要一次工具，
    都要把整段上下文再发一遍。**说清是「我们收的手」很重要**：不写那句，看起来就像模型答不出来。
    """
    from rag.agent.build import MAX_TOOL_CALLS

    events = _collect(_Agent(_many_tools(MAX_TOOL_CALLS + 3), state=None))

    tools = [event for event in events if event["type"] == "tool"]
    assert len(tools) == MAX_TOOL_CALLS, "到线就停，后面那三次不该再被消费"
    done = events[-1]
    assert done["type"] == "done"
    assert "上限" in done["answer"] and str(MAX_TOOL_CALLS) in done["answer"]
