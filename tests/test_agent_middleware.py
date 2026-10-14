"""工具出错时的包装中间件（2026-10-06 加的）。

**离线**：不建图、不调模型，直接喂一个会抛的 handler 给中间件 —— 要钉的就是
「异常变成一条能读的工具结果 + 一条 ERROR 日志」这一件事。
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import ToolMessage

from rag.agent.middleware import ToolErrorMiddleware


class _Request:
    def __init__(self, name: str = "search_graph", args: Any = None) -> None:
        self.tool_call = {"name": name, "id": "call-1", "args": args or {"query": "Flink"}}


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
