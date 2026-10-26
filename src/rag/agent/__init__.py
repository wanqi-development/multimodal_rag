"""Agent 层（M5）：deepagents 编排，工具复用下层能力。

    agent = build_agent(settings, heavy)                      # 一次请求建一次（见 build.py）
    result = await run_agent(agent, "问题")                    # 跑一次
    ...                                                        # 中断了 → 人看 pending_actions
    result = await resume_agent(agent, thread_id=..., decisions=[{"type": "approve"}])

批次 A 是**只读**的（StateBackend + 四个知识工具）。批次 B 加上：给 workspace 才
允许报告写入，`write_file` / `edit_file` 先过 HITL 闸门；`delete` / `execute` 由
ToolPolicy 禁用。SSE 流式（`stream_agent`）；按会话存引用表
（`save_registry` / `load_registry`）。批准之后接着流式跑是 `stream_resume`，
事件与 `stream_agent` 完全一样。
"""

from rag.agent.build import (
    AgentResult,
    build_agent,
    default_workspace,
    load_registry,
    pending_actions,
    pending_interrupt,
    resume_agent,
    run_agent,
    save_registry,
    stream_agent,
    stream_resume,
)
from rag.agent.middleware import ToolPolicy, ToolPolicyMiddleware
from rag.agent.tools import CitationRegistry, WebSearchGate, build_tools
from rag.contracts import EvidenceLedger

__all__ = [
    "AgentResult",
    "CitationRegistry",
    "EvidenceLedger",
    "ToolPolicy",
    "ToolPolicyMiddleware",
    "WebSearchGate",
    "build_agent",
    "build_tools",
    "default_workspace",
    "load_registry",
    "pending_actions",
    "pending_interrupt",
    "resume_agent",
    "run_agent",
    "save_registry",
    "stream_agent",
    "stream_resume",
]
