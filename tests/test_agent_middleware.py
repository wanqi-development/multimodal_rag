"""工具出错时的包装中间件（2026-10-06 加的）。

**离线**：不建图、不调模型，直接喂一个会抛的 handler 给中间件 —— 要钉的就是
「异常变成一条能读的工具结果 + 一条 ERROR 日志」这一件事。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from rag.agent.middleware import (
    ToolCallBudgetMiddleware,
    ToolErrorMiddleware,
    ToolPolicy,
    ToolPolicyMiddleware,
)


class _Request:
    def __init__(self, name: str = "search_graph", args: Any = None) -> None:
        self.tool_call = {"name": name, "id": "call-1", "args": args or {"query": "Flink"}}


class _ModelRequest:
    def __init__(self, tools: list[Any], state: dict[str, Any] | None = None) -> None:
        self.tools = tools
        self.state = state or {}

    def override(self, **changes: Any) -> _ModelRequest:
        return _ModelRequest(changes.get("tools", self.tools), self.state)


def _tool(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


def test_readonly_policy_filters_the_model_visible_surface() -> None:
    tools = [
        _tool("search_knowledge"),
        _tool("read_file"),
        _tool("write_file"),
        _tool("delete"),
        _tool("execute"),
        _tool("task"),
    ]
    seen: list[str] = []

    def handler(request: Any) -> Any:
        seen.extend(tool.name for tool in request.tools)
        return "ok"

    out = ToolPolicyMiddleware(ToolPolicy.readonly()).wrap_model_call(
        _ModelRequest(tools),
        handler,
    )

    assert out == "ok"
    assert seen == ["search_knowledge", "read_file"]


def test_report_policy_allows_report_writes_but_not_shell_or_delete() -> None:
    tools = [
        _tool("write_file"),
        _tool("edit_file"),
        _tool("delete"),
        _tool("execute"),
        _tool("task"),
    ]

    async def handler(request: Any) -> Any:
        return [tool.name for tool in request.tools]

    out = asyncio.run(
        ToolPolicyMiddleware(ToolPolicy.report()).awrap_model_call(
            _ModelRequest(tools),
            handler,
        )
    )

    assert out == ["write_file", "edit_file"]


def test_policy_rejects_a_denied_tool_before_its_handler(caplog: Any) -> None:
    called = False

    async def handler(request: Any) -> Any:
        nonlocal called
        called = True
        return ToolMessage(content="不该执行", tool_call_id="call-1")

    with caplog.at_level("WARNING", logger="rag.agent.tool"):
        out = asyncio.run(
            ToolPolicyMiddleware(ToolPolicy.report()).awrap_tool_call(
                _Request(name="execute"),
                handler,
            )
        )

    assert called is False
    assert isinstance(out, ToolMessage)
    assert out.status == "error"
    assert "execute" in out.content and "不可用" in out.content
    assert "工具策略拒绝：tool=execute mode=report" in caplog.text


def test_policy_passes_an_allowed_tool_to_its_handler() -> None:
    sentinel = ToolMessage(content="正常结果", tool_call_id="call-1")
    called = False

    def handler(request: Any) -> Any:
        nonlocal called
        called = True
        return sentinel

    out = ToolPolicyMiddleware(ToolPolicy.readonly()).wrap_tool_call(
        _Request(name="search_knowledge"),
        handler,
    )

    assert called is True and out is sentinel


def test_task_is_visible_only_when_at_least_one_subagent_is_allowed() -> None:
    seen: list[str] = []

    def handler(request: Any) -> Any:
        seen.extend(tool.name for tool in request.tools)
        return "ok"

    ToolPolicyMiddleware(
        ToolPolicy.readonly(allowed_subagents=frozenset({"policy-document"}))
    ).wrap_model_call(_ModelRequest([_tool("task"), _tool("execute")]), handler)

    assert seen == ["task"]


def test_injected_agent_tools_extend_the_main_agent_allowlist() -> None:
    seen: list[str] = []

    def handler(request: Any) -> Any:
        seen.extend(tool.name for tool in request.tools)
        return "ok"

    ToolPolicyMiddleware(
        ToolPolicy.readonly(extra_tools=frozenset({"company_web_crawl"}))
    ).wrap_model_call(
        _ModelRequest([_tool("company_web_crawl"), _tool("execute")]),
        handler,
    )

    assert seen == ["company_web_crawl"]


def test_exact_subagent_policy_hides_default_filesystem_tools() -> None:
    seen: list[str] = []

    def handler(request: Any) -> Any:
        seen.extend(tool.name for tool in request.tools)
        return "ok"

    ToolPolicyMiddleware(
        ToolPolicy.exact(frozenset({"search_knowledge", "get_section"}))
    ).wrap_model_call(
        _ModelRequest(
            [
                _tool("search_knowledge"),
                _tool("get_section"),
                _tool("grep"),
                _tool("read_file"),
                _tool("EvidenceResult"),
            ]
        ),
        handler,
    )

    assert seen == ["search_knowledge", "get_section", "EvidenceResult"]


def test_tool_budget_uses_run_state_and_resets_for_each_task() -> None:
    budget = ToolCallBudgetMiddleware(2)
    assert budget._matches_tool_filter({"name": "search_knowledge"}) is True
    assert budget._matches_tool_filter({"name": "EvidenceResult"}) is False

    def call(index: int) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_knowledge",
                    "args": {"query": "q"},
                    "id": f"call-{index}",
                    "type": "tool_call",
                }
            ],
        )

    first = budget.after_model({"messages": [call(1)]}, None)
    assert first["run_tool_call_count"]["__all__"] == 1
    second = budget.after_model(
        {"messages": [call(2)], "run_tool_call_count": first["run_tool_call_count"]},
        None,
    )
    assert second["run_tool_call_count"]["__all__"] == 2
    blocked = budget.after_model(
        {"messages": [call(3)], "run_tool_call_count": second["run_tool_call_count"]},
        None,
    )
    assert blocked["run_tool_call_count"]["__all__"] == 3
    assert isinstance(blocked["messages"][0], ToolMessage)
    assert "工具预算已用完" in blocked["messages"][0].content

    seen: list[str] = []

    def model_handler(request: Any) -> Any:
        seen.extend(tool.name for tool in request.tools)
        return "ok"

    budget.wrap_model_call(
        _ModelRequest(
            [_tool("search_knowledge"), _tool("EvidenceResult")],
            {"run_tool_call_count": {"__all__": 2}},
        ),
        model_handler,
    )
    assert seen == ["EvidenceResult"]

    # 同一个 middleware 实例在第二次 task 的全新 state 中重新获得预算。
    assert (
        budget.wrap_model_call(
            _ModelRequest([_tool("search_knowledge"), _tool("EvidenceResult")], {}),
            model_handler,
        )
        == "ok"
    )
    assert seen == ["EvidenceResult", "search_knowledge", "EvidenceResult"]


def test_task_rejects_a_subagent_outside_the_registry() -> None:
    called = False

    def handler(request: Any) -> Any:
        nonlocal called
        called = True
        return ToolMessage(content="不该执行", tool_call_id="call-1")

    out = ToolPolicyMiddleware(
        ToolPolicy.readonly(allowed_subagents=frozenset({"policy-document"}))
    ).wrap_tool_call(
        _Request(name="task", args={"subagent_type": "general-purpose", "description": "查"}),
        handler,
    )

    assert called is False
    assert isinstance(out, ToolMessage) and out.status == "error"
    assert "general-purpose" in out.content and "policy-document" in out.content


def test_a_failing_tool_becomes_a_readable_tool_message(caplog: Any) -> None:
    """工具炸了不能把整轮问答带走：包成 `status="error"` 的工具结果交给模型，
    它还能换一条路走（换关键词、只用语料、如实说查不到）。"""

    async def boom(request: Any) -> Any:
        raise ConnectionError("Milvus 连不上")

    with caplog.at_level("ERROR", logger="rag.agent.tool"):
        out = asyncio.run(ToolErrorMiddleware().awrap_tool_call(_Request(), boom))

    assert isinstance(out, ToolMessage)
    assert out.status == "error"
    assert "search_graph" in out.content and "Milvus 连不上" in out.content
    assert "不要用同样的参数重试" in out.content, "要明确告诉它别原地重试"


def test_the_failure_is_logged_as_an_error_with_the_arguments(caplog: Any) -> None:
    """**影响到了这次问答的可信度 → ERROR**（OCR 那种换条路照样跑通的降级才是 WARNING）。
    参数要带上：不然事后只知道「它失败了」，不知道它当时在查什么。"""

    async def boom(request: Any) -> Any:
        raise TimeoutError("图谱 10s 超时")

    with caplog.at_level("ERROR", logger="rag.agent.tool"):
        asyncio.run(ToolErrorMiddleware().awrap_tool_call(_Request(args={"query": "水位线"}), boom))

    assert "工具 search_graph 执行失败" in caplog.text
    assert "TimeoutError" in caplog.text
    assert "query=水位线" in caplog.text
    assert caplog.records[0].exc_info is not None, "栈要留下来"


def test_a_healthy_tool_passes_through_untouched() -> None:
    """没出错就什么都别做 —— 中间件不该改变正常那一路的任何东西。"""
    sentinel = ToolMessage(content="正常结果", tool_call_id="call-1")

    async def fine(request: Any) -> Any:
        return sentinel

    assert asyncio.run(ToolErrorMiddleware().awrap_tool_call(_Request(), fine)) is sentinel


def test_long_arguments_are_truncated_in_the_log() -> None:
    """`write_file` 的 content 能有好几万字，整段塞进日志只是让日志没法看。"""
    from rag.agent.middleware import _preview

    preview = _preview({"content": "x" * 500})

    assert len(preview) == 200, "整段预览截到 200 字符"
