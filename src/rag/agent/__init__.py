"""Agent 层：deepagents 主 Agent 按文件配置编排专业子 Agent。

    agent = build_agent(settings, heavy)                      # 一次请求建一次（见 build.py）
    result = await run_agent(agent, "问题")                    # 跑一次
    ...                                                        # 中断了 → 人看 pending_actions
    result = await resume_agent(agent, thread_id=..., decisions=[{"type": "approve"}])

主 Agent 负责意图识别、子 Agent 选择、证据融合和最终生成；网络搜索/网页抓取是
主 Agent 的直接工具。每个专业子 Agent 在 ``config/agents/*.yaml`` 中单独声明。

给 workspace 才允许报告写入，`write_file` / `edit_file` 先过 HITL 闸门；`delete` /
`execute` 由 ToolPolicy 禁用。SSE 流式（`stream_agent`）；按会话存引用表
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
from rag.agent.registry import AgentRegistry, load_agent_registry
from rag.agent.tools import CitationRegistry, WebSearchGate, build_tools
from rag.contracts import EvidenceLedger

__all__ = [
    "AgentResult",
    "AgentRegistry",
    "CitationRegistry",
    "EvidenceLedger",
    "ToolPolicy",
    "ToolPolicyMiddleware",
    "WebSearchGate",
    "build_agent",
    "build_tools",
    "default_workspace",
    "load_agent_registry",
    "load_registry",
    "pending_actions",
    "pending_interrupt",
    "resume_agent",
    "run_agent",
    "save_registry",
    "stream_agent",
    "stream_resume",
]
