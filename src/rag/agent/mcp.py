"""只读 MCP 工具接入。

连接目录在启动时发现一次；真正的 MCP 工具调用由适配器按次建立
Streamable HTTP 会话。每次问答再生成一层薄包装，把工具结果登记到本次
``EvidenceLedger``，因此引用编号与文档、网页共用同一个空间。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from rag.config.schema import McpConfig, McpServerConfig
from rag.contracts import EvidenceLedger

logger = logging.getLogger("rag.agent.mcp")

_NON_IDENTIFIER = re.compile(r"[^a-z0-9_]+")


class McpRuntimeError(RuntimeError):
    """MCP 启动配置或必需服务不可用。"""


@dataclass(frozen=True)
class McpToolSpec:
    server: str
    remote_name: str
    exposed_name: str
    tool: BaseTool
    max_output_chars: int


@dataclass(frozen=True)
class McpServerStatus:
    ok: bool
    detail: str
    tools: tuple[str, ...] = ()


def _exposed_name(server: str, tool_name: str) -> str:
    """把 MCP 的连字符工具名变成跨模型稳定的标识符。"""
    raw = f"{server}__{tool_name}".lower().replace("-", "_")
    return _NON_IDENTIFIER.sub("_", raw).strip("_")


def _connection(config: McpServerConfig) -> dict[str, Any]:
    if config.transport != "streamable_http":
        raise McpRuntimeError(f"不支持的 MCP 传输方式：{config.transport}")
    return {
        "transport": "streamable_http",
        "url": config.url,
        "headers": dict(config.headers),
        "timeout": timedelta(seconds=config.connect_timeout_s),
        "sse_read_timeout": timedelta(seconds=config.read_timeout_s),
    }


def _content_text(value: Any) -> str:
    """把适配器返回的 MCP content blocks 收敛成模型能读的文本。"""
    if isinstance(value, str):
        return value
    content = getattr(value, "content", None)
    if content is not None:
        return _content_text(content)
    if isinstance(value, list):
        parts: list[str] = []
        for block in value:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            else:
                parts.append(json.dumps(block, ensure_ascii=False, default=str))
        return "\n".join(part for part in parts if part)
    return json.dumps(value, ensure_ascii=False, default=str)


def _bounded_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    dropped = len(text) - limit
    return f"{text[:limit]}\n\n（输出已截断：另省略 {dropped} 个字符。）"


def _evidence_tool(spec: McpToolSpec, registry: EvidenceLedger) -> StructuredTool:
    async def invoke(**kwargs: Any) -> str:
        raw = await spec.tool.ainvoke(kwargs)
        text = _bounded_text(_content_text(raw), spec.max_output_chars)
        args_json = json.dumps(kwargs, ensure_ascii=False, sort_keys=True, default=str)
        digest = hashlib.sha256(f"{args_json}\0{text}".encode()).hexdigest()[:16]
        source = f"mcp://{spec.server}/{spec.remote_name}"
        number = registry.issue(
            source_type="mcp",
            chunk_id=f"mcp:{spec.server}:{digest}",
            source_uri=source,
            title=f"{spec.server} MCP · {spec.remote_name}",
            snippet=text[:200],
            heading_path=spec.remote_name,
        )
        checked_at = datetime.now().astimezone().isoformat(timespec="seconds")
        return f"[{number}] {source}\n查询参数：{args_json}\n调用时间：{checked_at}\n{text}"

    return StructuredTool.from_function(
        coroutine=invoke,
        name=spec.exposed_name,
        description=spec.tool.description or f"MCP 工具 {spec.remote_name}",
        args_schema=spec.tool.args_schema,
        response_format="content",
    )


class McpRuntime:
    """进程级 MCP 目录，加上按请求绑定引用表的工具包装。"""

    def __init__(
        self,
        *,
        client: Any | None,
        specs: tuple[McpToolSpec, ...],
        statuses: dict[str, McpServerStatus],
    ) -> None:
        self._client = client
        self._specs = specs
        self._statuses = statuses

    @classmethod
    async def start(cls, config: McpConfig) -> McpRuntime:
        enabled = {name: server for name, server in config.servers.items() if server.enabled}
        if not enabled:
            return cls(client=None, specs=(), statuses={})

        # 延迟导入：完全没启用 MCP 时不让适配器进入 API 启动热路径。
        from langchain_mcp_adapters.client import MultiServerMCPClient

        connections = {name: _connection(server) for name, server in enabled.items()}
        client = MultiServerMCPClient(connections, tool_name_prefix=False)
        specs: list[McpToolSpec] = []
        statuses: dict[str, McpServerStatus] = {}
        seen_names: set[str] = set()

        for server_name, server in enabled.items():
            try:
                discovered = await client.get_tools(server_name=server_name)
            except Exception as exc:  # noqa: BLE001 —— 外部服务失败要落状态
                detail = f"{type(exc).__name__}: {exc}"
                statuses[server_name] = McpServerStatus(ok=False, detail=detail)
                if server.required:
                    raise McpRuntimeError(
                        f"必需的 MCP 服务 {server_name} 不可用：{detail}"
                    ) from exc
                logger.error("MCP 服务 %s 不可用，相关子 Agent 将不加载：%s", server_name, detail)
                continue

            by_remote = {tool.name: tool for tool in discovered}
            missing = [name for name in server.allowed_tools if name not in by_remote]
            if missing:
                detail = f"配置允许的工具不存在：{', '.join(missing)}"
                statuses[server_name] = McpServerStatus(ok=False, detail=detail)
                if server.required:
                    raise McpRuntimeError(f"MCP 服务 {server_name} {detail}")
                logger.error("MCP 服务 %s %s，相关子 Agent 将不加载", server_name, detail)
                continue

            exposed: list[str] = []
            for remote_name in server.allowed_tools:
                name = _exposed_name(server_name, remote_name)
                if name in seen_names:
                    raise McpRuntimeError(f"MCP 规范化工具名重复：{name}")
                seen_names.add(name)
                exposed.append(name)
                specs.append(
                    McpToolSpec(
                        server=server_name,
                        remote_name=remote_name,
                        exposed_name=name,
                        tool=by_remote[remote_name],
                        max_output_chars=server.max_output_chars,
                    )
                )
            statuses[server_name] = McpServerStatus(
                ok=True,
                detail=f"已发现 {len(discovered)} 个工具，开放 {len(exposed)} 个",
                tools=tuple(exposed),
            )

        logger.info(
            "MCP 工具目录已加载：servers=%s tools=%d",
            ",".join(name for name, status in statuses.items() if status.ok) or "无",
            len(specs),
        )
        return cls(client=client, specs=tuple(specs), statuses=statuses)

    @property
    def available_tools(self) -> tuple[str, ...]:
        return tuple(spec.exposed_name for spec in self._specs)

    def bind(self, registry: EvidenceLedger) -> list[BaseTool]:
        """给本次问答生成引用感知的工具；工具定义本身仍复用启动时的目录。"""
        return [_evidence_tool(spec, registry) for spec in self._specs]

    def status(self) -> dict[str, Any]:
        return {
            "available": bool(self._specs),
            "tools": list(self.available_tools),
            "servers": {
                name: {
                    "ok": status.ok,
                    "detail": status.detail,
                    "tools": list(status.tools),
                }
                for name, status in self._statuses.items()
            },
        }

    async def close(self) -> None:
        """适配器的工具按调用建会话，没有需要常驻释放的连接。"""
        self._client = None
