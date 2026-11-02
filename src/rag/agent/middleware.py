"""Agent 中间件：工具能力边界与错误兜底。

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
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.tool_call_limit import ToolCallLimitMiddleware
from langchain_core.messages import ToolMessage

logger = logging.getLogger("rag.agent.tool")

# 参数预览只留这么长：工具参数里可能有整篇正文（`write_file` 的 content）
ARGS_PREVIEW = 200

KNOWLEDGE_TOOLS = frozenset(
    {
        "search_knowledge",
        "search_relationships",
        "list_documents",
        "get_section",
        "web_search_tool",
        "web_fetch_tool",
    }
)
STRUCTURED_OUTPUT_TOOLS = frozenset({"EvidenceResult"})
READ_ONLY_FILE_TOOLS = frozenset({"ls", "read_file", "glob", "grep"})
REPORT_WRITE_TOOLS = frozenset({"write_file", "edit_file"})


@dataclass(frozen=True)
class ToolPolicy:
    """一次 agent 运行允许哪些工具。

    默认拒绝未登记的工具，避免 deepagents 升级后自动注入的 `task`、`execute` 等能力
    悄悄扩大工具面。策略管“有没有资格使用”，具体业务门控（如 WebSearchGate）仍由工具执行层管。
    """

    mode: str
    allowed: frozenset[str]
    allowed_subagents: frozenset[str] = frozenset()

    @classmethod
    def readonly(
        cls,
        *,
        allowed_subagents: frozenset[str] = frozenset(),
        extra_tools: frozenset[str] = frozenset(),
    ) -> ToolPolicy:
        allowed = KNOWLEDGE_TOOLS | READ_ONLY_FILE_TOOLS | extra_tools
        if allowed_subagents:
            allowed = allowed | {"task"}
        return cls(
            mode="readonly",
            allowed=allowed,
            allowed_subagents=allowed_subagents,
        )

    @classmethod
    def report(
        cls,
        *,
        allowed_subagents: frozenset[str] = frozenset(),
        extra_tools: frozenset[str] = frozenset(),
    ) -> ToolPolicy:
        allowed = KNOWLEDGE_TOOLS | READ_ONLY_FILE_TOOLS | REPORT_WRITE_TOOLS | extra_tools
        if allowed_subagents:
            allowed = allowed | {"task"}
        return cls(
            mode="report",
            allowed=allowed,
            allowed_subagents=allowed_subagents,
        )

    @classmethod
    def exact(cls, allowed: frozenset[str]) -> ToolPolicy:
        """子 Agent 精确白名单：没有列出的工具不可见、不可执行。"""
        return cls(mode="subagent", allowed=allowed | STRUCTURED_OUTPUT_TOOLS)

    def allows(self, name: str) -> bool:
        return name in self.allowed

    def allows_task(self, subagent: str) -> bool:
        return bool(subagent) and subagent in self.allowed_subagents


def _tool_name(tool: Any) -> str | None:
    """从 BaseTool 或 dict 形态的工具声明里取名字。"""
    name = tool.get("name") if isinstance(tool, dict) else getattr(tool, "name", None)
    return name if isinstance(name, str) else None


class ToolPolicyMiddleware(AgentMiddleware):
    """在模型可见面和工具执行面同时执行同一份 ToolPolicy。

    - `wrap_model_call`：裁掉不允许的工具，模型根本看不到；
    - `wrap_tool_call`：即使模型凭历史或幻觉发出禁用调用，也在 HITL 之前拒绝。
    """

    def __init__(self, policy: ToolPolicy) -> None:
        super().__init__()
        self.policy = policy

    def _visible(self, request: Any) -> Any:
        tools = [
            tool
            for tool in request.tools
            if (name := _tool_name(tool)) is not None and self.policy.allows(name)
        ]
        return request.override(tools=tools)

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        return handler(self._visible(request))

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        return await handler(self._visible(request))

    def _rejection(self, request: Any) -> ToolMessage | None:
        call = getattr(request, "tool_call", None) or {}
        name = str(call.get("name") or "")
        subagent = ""
        if self.policy.allows(name):
            if name != "task":
                return None
            args = call.get("args") if isinstance(call, dict) else None
            subagent = str((args or {}).get("subagent_type") or "")
            if self.policy.allows_task(subagent):
                return None
            reason = (
                f"子 Agent {subagent or '?'} 不在本次允许清单中。"
                f"可用：{', '.join(sorted(self.policy.allowed_subagents)) or '无'}。"
            )
        else:
            reason = f"工具 {name or '?'} 在当前 {self.policy.mode} 策略下不可用。"
        if name == "task":
            logger.warning(
                "子 Agent 策略拒绝：subagent=%s allowed=%s",
                subagent,
                ",".join(sorted(self.policy.allowed_subagents)),
            )
        else:
            logger.warning(
                "工具策略拒绝：tool=%s mode=%s",
                name or "?",
                self.policy.mode,
            )
        if name != "task":
            return ToolMessage(
                content=(reason + "请继续使用已开放的工具，不要重试这个调用。"),
                tool_call_id=str(call.get("id") or ""),
                name=name or None,
                status="error",
            )
        return ToolMessage(
            content=reason,
            tool_call_id=str(call.get("id") or ""),
            name=name or None,
            status="error",
        )

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        return self._rejection(request) or handler(request)

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        return self._rejection(request) or await handler(request)


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


class ToolCallBudgetMiddleware(ToolCallLimitMiddleware):
    """限制一个 Agent 实例内实际执行的工具数量。

    计数存放在 LangGraph 的 run state 中，因此每次子 Agent `task` 调用都会自动
    从 0 开始；达到预算后继续保留结构化输出工具，使模型基于已有证据收尾。
    """

    def __init__(
        self,
        max_calls: int,
        *,
        protected: frozenset[str] = STRUCTURED_OUTPUT_TOOLS,
        note: str = "",
    ) -> None:
        if max_calls < 1:
            raise ValueError("max_calls 必须大于 0")
        super().__init__(run_limit=max_calls, exit_behavior="continue")
        self.protected = protected
        self.max_calls = max_calls
        self.note = note or (
            "本次工具预算已用完。不要继续调用工具，基于已经取得的证据回答；资料不足的部分明确说明。"
        )

    def _matches_tool_filter(self, tool_call: Any) -> bool:
        """结构化输出不算业务工具；其余工具都计入 run 预算。"""
        return str(tool_call.get("name") or "") not in self.protected

    def _localize(self, result: dict[str, Any] | None) -> dict[str, Any] | None:
        if not result or not result.get("messages"):
            return result
        return {
            **result,
            "messages": [
                message.model_copy(update={"content": self.note})
                if isinstance(message, ToolMessage) and message.status == "error"
                else message
                for message in result["messages"]
            ],
        }

    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self._localize(super().after_model(state, runtime))

    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self._localize(await super().aafter_model(state, runtime))

    def _visible(self, request: Any) -> Any:
        state = getattr(request, "state", None) or {}
        counts = state.get("run_tool_call_count") or {}
        exhausted = int(counts.get("__all__", 0)) >= self.max_calls
        if not exhausted:
            return request
        tools = [tool for tool in request.tools if _tool_name(tool) in self.protected]
        return request.override(tools=tools)

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        return handler(self._visible(request))

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        return await handler(self._visible(request))
