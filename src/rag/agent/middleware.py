"""Agent 中间件：**把工具的错误包起来，别让它把整轮问答带走**。

使用者 2026-10-06 定的：不管是图谱、联网还是别的要真的去执行工具的场景，
工具炸了要**包一层**再交给模型，同时**按 ERROR 记一条日志**。

为什么必须包：LangGraph 默认会把工具抛的异常直接往上抛 —— 一次「图谱连不上」就能让
整轮问答失败，而模型其实还有别的路可走（换关键词、只用语料、如实说这部分查不到）。
包成 `ToolMessage(status="error")` 交给它，这一轮就还能给出一句有用的答复。

为什么这些错误是 ERROR 而不是 WARNING：**它们影响到了这次问答的可信度** ——
本来该查图谱没查到，答案可能因此不完整。OCR 那种「换条路照样能跑通」的降级才是 WARNING。
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

logger = logging.getLogger("rag.agent.tool")

# 参数预览只留这么长：工具参数里可能有整篇正文（`write_file` 的 content）
ARGS_PREVIEW = 200


def _preview(args: Any) -> str:
    if not isinstance(args, dict):
        return str(args)[:ARGS_PREVIEW]
    parts = []
    for key, value in args.items():
        text = value if isinstance(value, str) else repr(value)
        parts.append(f"{key}={text[:ARGS_PREVIEW]}")
    return ", ".join(parts)[:ARGS_PREVIEW]


class ToolErrorMiddleware(AgentMiddleware):
    """工具抛异常 → 记 ERROR + 回一条能读的工具结果。

    `writer` 那张嘴留给「失败时到底该怎么跟模型说」，默认是「别重试同一个工具，
    换一种方式，或者如实说查不到」—— 不说清的话模型经常会拿同一个参数再试一遍。
    """

    def __init__(self, note: str = "") -> None:
        super().__init__()
        self.note = note or (
            "不要用同样的参数重试这个工具；换一种方式，或者如实告诉用户这部分暂时查不到。"
        )

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        call = getattr(request, "tool_call", None) or {}
        name = call.get("name") or "?"
        try:
            return await handler(request)
        except Exception as exc:  # noqa: BLE001 —— 就是要拦下所有工具异常
            return self._on_error(call, name, exc)

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        """同步那条路也要拦（同一个中间件两套入口，漏一个就等于没拦）。

        **不能拿 `asyncio.run` 代劳**：在已经跑着的事件循环里调它会直接报错，
        而这个入口本来就在没有循环的时候才走到。
        """
        call = getattr(request, "tool_call", None) or {}
        name = call.get("name") or "?"
        try:
            return handler(request)
        except Exception as exc:  # noqa: BLE001
            return self._on_error(call, name, exc)

    def _on_error(self, call: dict[str, Any], name: str, exc: Exception) -> ToolMessage:
        """两条路共用的善后：记 ERROR（带栈、带关联 id）+ 回一条能读的工具结果。"""
        logger.error(
            "工具 %s 执行失败：%s: %s（参数 %s）",
            name,
            type(exc).__name__,
            exc,
            _preview(call.get("args")),
            exc_info=True,
        )
        return ToolMessage(
            content=f"工具 {name} 执行失败：{type(exc).__name__}: {exc}。{self.note}",
            tool_call_id=str(call.get("id") or ""),
            status="error",
        )
