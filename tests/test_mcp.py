"""MCP 工具发现、引用登记与可选服务降级。"""

from __future__ import annotations

import asyncio
import os

import pytest
from langchain_core.tools import StructuredTool

from rag.agent.mcp import McpRuntime, McpRuntimeError
from rag.config.schema import McpConfig, McpServerConfig
from rag.contracts import EvidenceLedger


def _time_tool(text: str = '{"success": true, "date": "2026-11-03"}') -> StructuredTool:
    async def current_time(**_: object) -> list[dict[str, str]]:
        return [{"type": "text", "text": text}]

    return StructuredTool.from_function(
        coroutine=current_time,
        name="get-current-time",
        description="获取当前时间",
        args_schema={
            "type": "object",
            "properties": {
                "timezone": {"type": "string"},
                "format": {"type": "string"},
            },
        },
    )


def _config(*, required: bool = False, tools: list[str] | None = None) -> McpConfig:
    return McpConfig(
        servers={
            "railway": McpServerConfig(
                enabled=True,
                required=required,
                url="http://mcp.test/mcp",
                allowed_tools=tools or ["get-current-time"],
            )
        }
    )


def test_mcp_tools_are_renamed_and_bound_to_the_request_ledger(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, connections, **_: object) -> None:
            self.connections = connections

        async def get_tools(self, *, server_name: str):
            assert server_name == "railway"
            return [_time_tool()]

    monkeypatch.setattr(
        "langchain_mcp_adapters.client.MultiServerMCPClient",
        FakeClient,
    )

    runtime = asyncio.run(McpRuntime.start(_config()))
    ledger = EvidenceLedger()
    tools = runtime.bind(ledger)

    assert runtime.available_tools == ("railway__get_current_time",)
    assert runtime.status()["servers"]["railway"]["ok"] is True
    assert [tool.name for tool in tools] == ["railway__get_current_time"]

    output = asyncio.run(tools[0].ainvoke({"timezone": "Asia/Shanghai"}))

    assert output.startswith("[1] mcp://railway/get-current-time")
    assert "2026-11-03" in output
    citation = ledger.items[1]
    assert citation.source_type == "mcp"
    assert citation.source_uri == "mcp://railway/get-current-time"


def test_mcp_output_is_explicitly_truncated(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        async def get_tools(self, *, server_name: str):
            return [_time_tool("x" * 2_000)]

    monkeypatch.setattr(
        "langchain_mcp_adapters.client.MultiServerMCPClient",
        FakeClient,
    )
    config = _config()
    config.servers["railway"].max_output_chars = 1_000

    runtime = asyncio.run(McpRuntime.start(config))
    output = asyncio.run(runtime.bind(EvidenceLedger())[0].ainvoke({}))

    assert "输出已截断" in output
    assert "另省略 1000 个字符" in output


def test_optional_mcp_missing_allowed_tool_disables_without_raising(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        async def get_tools(self, *, server_name: str):
            return [_time_tool()]

    monkeypatch.setattr(
        "langchain_mcp_adapters.client.MultiServerMCPClient",
        FakeClient,
    )

    runtime = asyncio.run(McpRuntime.start(_config(tools=["not-exposed"])))

    assert runtime.available_tools == ()
    assert runtime.status()["servers"]["railway"]["ok"] is False
    assert "not-exposed" in runtime.status()["servers"]["railway"]["detail"]


def test_optional_mcp_connection_failure_does_not_raise(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        async def get_tools(self, *, server_name: str):
            raise ConnectionError("server down")

    monkeypatch.setattr(
        "langchain_mcp_adapters.client.MultiServerMCPClient",
        FakeClient,
    )

    runtime = asyncio.run(McpRuntime.start(_config()))

    assert runtime.available_tools == ()
    status = runtime.status()["servers"]["railway"]
    assert status["ok"] is False
    assert "server down" in status["detail"]


def test_required_mcp_missing_allowed_tool_fails_fast(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        async def get_tools(self, *, server_name: str):
            return [_time_tool()]

    monkeypatch.setattr(
        "langchain_mcp_adapters.client.MultiServerMCPClient",
        FakeClient,
    )

    with pytest.raises(McpRuntimeError, match="not-exposed"):
        asyncio.run(McpRuntime.start(_config(required=True, tools=["not-exposed"])))


@pytest.mark.skipif(
    not os.environ.get("RAG_TEST_MCP_URL"),
    reason="设置 RAG_TEST_MCP_URL 后才运行真实 MCP 集成测试",
)
def test_live_streamable_http_server_exposes_the_seven_configured_tools() -> None:
    config = McpConfig(
        servers={
            "railway": McpServerConfig(
                enabled=True,
                url=os.environ["RAG_TEST_MCP_URL"],
                allowed_tools=[
                    "query-tickets",
                    "query-ticket-price",
                    "search-stations",
                    "query-transfer",
                    "get-train-route-stations",
                    "get-train-no-by-train-code",
                    "get-current-time",
                ],
            )
        }
    )

    runtime = asyncio.run(McpRuntime.start(config))

    assert len(runtime.available_tools) == 7
    assert runtime.status()["servers"]["railway"]["ok"] is True
