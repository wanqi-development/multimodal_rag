"""Agent 层（M5 批次 B）：可写 workspace、HITL 闸门、中断恢复、流式、引用表。

**真图 + 假模型**：HITL 是 deepagents 的中间件，用替身图验不了它 —— 所以图是真建的，
只有模型是我们自己的假模型。不连 Milvus、不调线上、不花一分钱，走的是真协议
（中断载荷、`Command(resume=…)`、`__interrupt__`）。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import PrivateAttr

from rag.agent import (
    CitationRegistry,
    build_agent,
    default_workspace,
    load_registry,
    pending_actions,
    resume_agent,
    run_agent,
    save_registry,
    stream_agent,
    stream_resume,
)
from rag.config.schema import Settings

FILE = "/workspace/report.md"


class _ToolCallingModel(BaseChatModel):
    """第一轮要求写文件（撞闸门），看过工具结果之后就交答案。"""

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003, ANN201
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001, ANN003, ANN201
        if any(getattr(item, "type", "") == "tool" for item in messages):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="写好了 [1]"))])
        call = {
            "name": "write_file",
            "args": {"file_path": FILE, "content": "报告正文"},
            "id": "call-1",
        }
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content="我先写个文件", tool_calls=[call]))
            ]
        )


class _RecordingModel(BaseChatModel):
    """记录模型实际绑定的工具；把策略过滤后的可见面留在测试里。"""

    _seen_tools: list[list[str]] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-recording"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003, ANN201
        names = []
        for tool in tools:
            name = tool.get("name") if isinstance(tool, dict) else getattr(tool, "name", None)
            if isinstance(name, str):
                names.append(name)
        self._seen_tools.append(names)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001, ANN003, ANN201
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="答案"))])


class _ExecuteCallingModel(_ToolCallingModel):
    """第一条消息请求执行命令；策略应在 HITL 之前拒绝。"""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001, ANN003, ANN201
        tool_messages = [item for item in messages if getattr(item, "type", "") == "tool"]
        if tool_messages:
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content=f"工具结果：{tool_messages[-1].content}")
                    )
                ]
            )
        call = {"name": "execute", "args": {"command": "whoami"}, "id": "call-x"}
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content="我先执行命令", tool_calls=[call]))
            ]
        )


class _Heavy:
    """只满足 `build_agent` 的取用：模型给假的，检索那三件只要「有」。"""

    def __init__(self, model: BaseChatModel | None = None) -> None:
        self.index = object()
        self.embedder = object()
        self.reranker = None
        self.chat = model or _ToolCallingModel()
        self.generator = object()
        self.errors: dict[str, str] = {}


def _settings(tmp_path: Path) -> Settings:
    settings = Settings()
    settings.agent.workspace_dir = str(tmp_path / "workspace")
    return settings


def _agent(tmp_path: Path, workspace: Path | None) -> Any:
    graph, _ = build_agent(
        _settings(tmp_path),
        _Heavy(),
        checkpointer=InMemorySaver(),
        workspace=workspace,
    )
    return graph


def _run(agent: Any, question: str = "写个报告", thread_id: str = "t1"):
    return asyncio.run(run_agent(agent, question, registry=CitationRegistry(), thread_id=thread_id))


def _resume(agent: Any, decisions: list[dict[str, Any]], thread_id: str = "t1"):
    return asyncio.run(
        resume_agent(agent, thread_id=thread_id, decisions=decisions, registry=CitationRegistry())
    )


# ---- workspace 路由 ----（写盘能力由「给不给 workspace」决定）


def test_default_workspace_follows_the_config(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    assert default_workspace(settings) == Path(settings.agent.workspace_dir)

    settings.agent.workspace_writes = False

    assert default_workspace(settings) is None, "关掉写盘就该回到批次 A 的只读"


def test_the_gate_covers_every_policy_allowed_write_tool(tmp_path: Path, monkeypatch) -> None:
    """HITL 只接策略允许的写工具；delete / execute 在更前面的策略层直接拒绝。"""
    from deepagents.backends import CompositeBackend

    captured: dict = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return "graph"

    import deepagents

    monkeypatch.setattr(deepagents, "create_deep_agent", fake_create)
    workspace = tmp_path / "workspace"
    build_agent(_settings(tmp_path), _Heavy(), workspace=workspace)

    assert isinstance(captured["backend"], CompositeBackend), "/workspace/ 要路由到真磁盘"
    assert set(captured["interrupt_on"]) == {"write_file", "edit_file"}
    assert all(captured["interrupt_on"].values()), "允许的写工具都要挂闸门，没有例外"
    assert captured["memory"] is None, "没有 AGENTS.md 就不该挂一个不存在的路径"
    assert workspace.is_dir()


def test_without_a_workspace_nothing_is_gated_because_nothing_writes(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return "graph"

    import deepagents

    monkeypatch.setattr(deepagents, "create_deep_agent", fake_create)
    build_agent(_settings(tmp_path), _Heavy())

    assert captured["interrupt_on"] is None
    assert captured["memory"] is None


def test_compiled_graph_exposes_only_policy_tools(tmp_path: Path) -> None:
    readonly_model = _RecordingModel()
    readonly_graph, readonly_registry = build_agent(
        _settings(tmp_path),
        _Heavy(readonly_model),
    )
    asyncio.run(
        run_agent(
            readonly_graph,
            "只读问题",
            registry=readonly_registry,
            thread_id="readonly",
        )
    )

    readonly_tools = set(readonly_model._seen_tools[-1])
    assert {"search_knowledge", "read_file", "grep"} <= readonly_tools
    assert {"task", "write_file", "edit_file", "delete", "execute"}.isdisjoint(readonly_tools)

    report_model = _RecordingModel()
    report_graph, report_registry = build_agent(
        _settings(tmp_path),
        _Heavy(report_model),
        workspace=tmp_path / "workspace-report",
    )
    asyncio.run(
        run_agent(
            report_graph,
            "写报告",
            registry=report_registry,
            thread_id="report",
        )
    )

    report_tools = set(report_model._seen_tools[-1])
    assert {"write_file", "edit_file"} <= report_tools
    assert {"task", "delete", "execute"}.isdisjoint(report_tools)


def test_policy_denial_happens_before_hitl(tmp_path: Path) -> None:
    agent = build_agent(
        _settings(tmp_path),
        _Heavy(_ExecuteCallingModel()),
        checkpointer=InMemorySaver(),
        workspace=tmp_path / "workspace",
    )[0]

    result = _run(agent)

    assert result.status == "answered", "策略拒绝不该进入等人批准"
    assert result.interrupt is None
    assert "execute" in result.answer and "不可用" in result.answer


def test_agents_md_becomes_the_agents_memory(tmp_path: Path, monkeypatch) -> None:
    """workspace/AGENTS.md 存在才挂 —— 内容由人维护，是给模型看的约定。"""
    captured: dict = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return "graph"

    import deepagents

    monkeypatch.setattr(deepagents, "create_deep_agent", fake_create)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("报告写到 /workspace/。", encoding="utf-8")

    build_agent(_settings(tmp_path), _Heavy(), workspace=workspace)

    assert captured["memory"] == ["/workspace/AGENTS.md"]


def test_skills_are_mounted_from_the_workspace(tmp_path: Path, monkeypatch) -> None:
    """skills 从**随包发布**的那份读（`/skills`），只有给了 workspace 才挂得上。

    只读档（StateBackend）没有文件系统后端，挂了等于给模型看一个不存在的路径。
    """
    captured: dict = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return "graph"

    import deepagents

    monkeypatch.setattr(deepagents, "create_deep_agent", fake_create)
    build_agent(_settings(tmp_path), _Heavy(), workspace=tmp_path / "workspace")

    assert captured["skills"] == ["/skills"]
    assert "/skills/" in captured["backend"].routes, "两种 skills 都要挂在 backend 路由上"


def test_user_skills_take_precedence_over_the_shipped_ones(tmp_path: Path, monkeypatch) -> None:
    """包内的排前面、用户自己的排后面 —— 中间件「同名后者覆盖」，所以用户能覆盖包内的。"""
    captured: dict = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return "graph"

    import deepagents

    monkeypatch.setattr(deepagents, "create_deep_agent", fake_create)
    workspace = tmp_path / "workspace"
    (workspace / "skills").mkdir(parents=True)
    build_agent(_settings(tmp_path), _Heavy(), workspace=workspace)

    assert captured["skills"] == ["/skills", "/workspace/skills"]


def test_the_workspace_backend_skips_ripgrep(tmp_path: Path, monkeypatch) -> None:
    """上游按 locale(GBK) 解码 rg 的 UTF-8 输出 —— 工作区里有中文时 `grep` 必崩（真机实测）。

    我们的子类返回 `(None, False)`（库自己约定的「rg 不可用」），于是它走按 UTF-8 读文件的
    Python 实现。这里直接拿后端跑一次中文 grep：**不抛异常、能搜到**才算过。
    """
    captured: dict = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return "graph"

    import deepagents

    monkeypatch.setattr(deepagents, "create_deep_agent", fake_create)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "笔记.md").write_text("第一行\nFlink 支持窗口操作\n", encoding="utf-8")
    build_agent(_settings(tmp_path), _Heavy(), workspace=workspace)

    backend = captured["backend"].routes["/workspace/"]
    result = backend.grep("窗口", "/")

    assert result.error in (None, ""), f"grep 不该报错：{result.error}"
    # 匹配项的形状由上游定（这里两种都收）：只验「搜到了那行中文」
    assert any("Flink 支持窗口操作" in str(match) for match in result.matches)


# ---- 闸门本身 ----（真图跑一遍中断与恢复）


def test_a_write_stops_at_the_gate_and_touches_nothing(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    agent = _agent(tmp_path, workspace)

    result = _run(agent)

    assert result.status == "interrupted"
    assert [item["name"] for item in result.interrupt["action_requests"]] == ["write_file"]
    assert result.answer == "", "中断时没有答案 —— 最后那条 AI 消息只是「我要写文件」"
    assert list(workspace.rglob("*")) == [], "人没批准之前，一个字都不许落盘"


def test_pending_actions_reads_the_gate_from_another_process(tmp_path: Path) -> None:
    """恢复是**另一种调用**（CLI 是另一次进程）：手上只有 thread_id 也得能问出「等什么」。"""
    agent = _agent(tmp_path, tmp_path / "workspace")
    _run(agent)

    actions = asyncio.run(pending_actions(agent, "t1"))

    assert [item["name"] for item in actions] == ["write_file"]
    assert actions[0]["args"]["file_path"] == FILE


def test_approving_lets_it_write_and_finish(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    agent = _agent(tmp_path, workspace)
    _run(agent)

    result = _resume(agent, [{"type": "approve"}])

    assert result.status == "answered"
    assert result.answer == "写好了 [1]"
    assert (workspace / "report.md").read_text(encoding="utf-8") == "报告正文"


def test_rejecting_leaves_the_disk_alone(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    agent = _agent(tmp_path, workspace)
    _run(agent)

    result = _resume(agent, [{"type": "reject", "message": "不用写"}])

    assert result.status == "answered"
    assert not (workspace / "report.md").exists()


def test_without_a_workspace_it_writes_into_the_state_not_the_disk(tmp_path: Path) -> None:
    """只读档（批次 A 的行为）：工具照样「成功」，但落的是内存 state，磁盘没有新文件。"""
    agent = _agent(tmp_path, None)

    result = _run(agent)

    assert result.status == "answered"
    assert list(tmp_path.rglob("*.md")) == []


# ---- 引用表 ----（中断那一刻发的号，恢复之后还得接得上）


def test_the_citation_table_round_trips_through_the_store() -> None:
    store = InMemoryStore()
    registry = CitationRegistry()
    registry.issue(chunk_id="d1:0001", source_uri="a.pdf", snippet="甲")
    registry.issue(chunk_id="d1:0002", source_uri="b.pdf", snippet="乙")

    assert asyncio.run(save_registry(store, "t1", registry)) is True

    restored = asyncio.run(load_registry(store, "t1"))

    assert [(n, c.chunk_id) for n, c in restored.items.items()] == [(1, "d1:0001"), (2, "d1:0002")]
    assert restored.cited_in("结论 [2]")[0].source_uri == "b.pdf"


def test_an_empty_table_is_not_written_and_a_missing_one_is_empty() -> None:
    store = InMemoryStore()

    assert asyncio.run(save_registry(store, "t1", CitationRegistry())) is False
    assert asyncio.run(load_registry(store, "没存过")).items == {}
    assert asyncio.run(load_registry(None, "t1")).items == {}, "没 store 也不该炸"


# ---- 流式 ----（事件四种：token / tool / interrupt / done）


def _drain(agent: Any, question: str = "写个报告", thread_id: str = "t1") -> list[dict]:
    async def collect():
        return [
            event
            async for event in stream_agent(
                agent, question, registry=CitationRegistry(), thread_id=thread_id
            )
        ]

    return asyncio.run(collect())


def _drain_resume(agent: Any, decisions: list[dict], thread_id: str = "t1") -> list[dict]:
    async def collect():
        return [
            event
            async for event in stream_resume(
                agent, thread_id=thread_id, decisions=decisions, registry=CitationRegistry()
            )
        ]

    return asyncio.run(collect())


def test_the_stream_ends_with_done_and_the_cited_sources(tmp_path: Path) -> None:
    agent = _agent(tmp_path, None)

    events = _drain(agent)

    kinds = [event["type"] for event in events]
    assert kinds[-1] == "done" and "interrupt" not in kinds
    # **一个 token 事件都没有，这是对的**：这个假模型只实现了 `_generate`（非流式），
    # 而 v3 的 `messages` 流只在模型真的流式吐字时才有 —— 正文照旧从 `done.answer` 拿。
    # 真流式那条路（`content-block-delta` → token）由 tests/test_agent_timeline.py 钉住。
    assert not [event for event in events if event["type"] == "token"]
    assert events[-1]["answer"] == "写好了 [1]"
    assert events[-1]["thread_id"] == "t1"


def test_the_tool_event_carries_the_arguments_and_a_duration(tmp_path: Path) -> None:
    """工具事件不能只有个名字 —— 时间线上「它查了什么词 / 要写哪个文件」全靠参数，
    而「这一步值不值得等」靠耗时。参数在 AIMessage 的 `tool_calls` 上，不在 ToolMessage 上。"""
    agent = _agent(tmp_path, tmp_path / "workspace")
    assert _drain(agent)[-1]["type"] == "interrupt", "先撞一次写盘闸门"

    tool = next(
        event for event in _drain_resume(agent, [{"type": "approve"}]) if event["type"] == "tool"
    )

    assert tool["name"] == "write_file" and tool["status"] == "success"
    assert tool["args"] == {"file_path": FILE, "content": "报告正文"}
    assert isinstance(tool["ms"], int) and tool["ms"] >= 0


def test_the_args_preview_truncates_a_long_body() -> None:
    """`write_file` 的 content 能有好几万字：整包塞进 SSE 只是让事件变胖。"""
    from rag.agent.build import _args_preview

    assert _args_preview({"file_path": FILE}) == {"file_path": FILE}
    long_body = _args_preview({"content": "x" * 500})["content"]
    assert len(long_body) == 201 and long_body.endswith("…")
    assert _args_preview("模型偶尔给个字符串") == {}


def test_resume_streams_the_same_events_as_the_first_run(tmp_path: Path) -> None:
    """批准之后的答案照样是一段段生成的。事件形状必须与首次跑**完全一样**，
    否则客户端得为「批准后」再写一套解析。"""
    agent = _agent(tmp_path, tmp_path / "workspace")
    assert _drain(agent)[-1]["type"] == "interrupt", "先撞一次闸门"

    events = _drain_resume(agent, [{"type": "approve"}])

    kinds = [event["type"] for event in events]
    assert kinds[-1] == "done" and "interrupt" not in kinds
    assert events[-1]["answer"] == "写好了 [1]"
    tool = next(event for event in events if event["type"] == "tool")
    assert tool["name"] == "write_file" and tool["status"] == "success"


def test_the_stream_stops_at_the_gate(tmp_path: Path) -> None:
    agent = _agent(tmp_path, tmp_path / "workspace")

    events = _drain(agent)

    kinds = [event["type"] for event in events]
    assert kinds[-1] == "interrupt" and "done" not in kinds
    assert events[-1]["interrupt"]["action_requests"][0]["name"] == "write_file"
    assert events[-1]["thread_id"] == "t1"


def test_the_streaming_path_writes_the_agent_run_log(tmp_path: Path, monkeypatch) -> None:
    """**流式也要落运行日志**。界面走的就是这条：不落的话 agent.log 里查不到任何一次
    界面发起的运行（只有 CLI 那两条非流式路径有账）。"""
    seen: list[dict] = []

    def fake(**kwargs):  # noqa: ANN003, ANN201
        seen.append(kwargs)

    monkeypatch.setattr("rag.obs.record_agent_run", fake)
    _drain(_agent(tmp_path, None))

    assert len(seen) == 1
    assert seen[0]["question"] == "写个报告"
    assert seen[0]["status"] == "answered"
    assert seen[0]["thread_id"] == "t1"
    assert [item["name"] for item in seen[0]["tools"]] == ["write_file"], "工具摘要要带上"


def test_the_interrupted_stream_is_logged_as_interrupted(tmp_path: Path, monkeypatch) -> None:
    """停在闸门也是一次「运行」，而且是有结论的那种（它想干什么）。"""
    seen: list[dict] = []

    def fake(**kwargs):  # noqa: ANN003, ANN201
        seen.append(kwargs)

    monkeypatch.setattr("rag.obs.record_agent_run", fake)
    _drain(_agent(tmp_path, tmp_path / "workspace"))

    assert len(seen) == 1
    assert seen[0]["status"] == "interrupted"
    assert "write_file" in seen[0]["error"], "error 那一栏记的是「它想干什么」"


def test_the_stream_still_finishes_without_a_checkpointer(tmp_path: Path) -> None:
    """没接 checkpointer 时问不到图状态，就退回流里攒下来的正文 —— 流式不该因此断掉。"""
    graph, _ = build_agent(_settings(tmp_path), _Heavy())

    events = _drain(graph, thread_id=None)

    assert events[-1]["type"] == "done"
    assert "写好了 [1]" in events[-1]["answer"]
    assert events[-1]["thread_id"] is None, "没接 checkpointer 就没有会话 id"


def test_subagent_tokens_are_filtered_out_of_the_stream() -> None:
    """子 agent 跑在子图里，它的 token 会混在同一个 `messages` 流里 —— 不过滤就会把
    中间过程当成答案吐给用户。判据是路径（`checkpoint_ns`）有没有多段。"""
    from rag.agent.build import _is_main_model_chunk

    main = {"langgraph_node": "model", "checkpoint_ns": "model:abc"}
    sub = {"langgraph_node": "model", "checkpoint_ns": "tools:xyz|model:abc"}

    assert _is_main_model_chunk(main) is True
    assert _is_main_model_chunk(sub) is False
    assert _is_main_model_chunk({"langgraph_node": "model"}) is True, "没有 ns 就当前位置未知"
    assert _is_main_model_chunk({"langgraph_node": "tools"}) is False


def test_the_cli_decision_specs_are_translated() -> None:
    """CLI 的决定 → HITL 中间件的形状。放在这里是因为它与闸门是同一件事的两端。"""
    from rag.cli.main import _decisions

    assert _decisions(approve=True, reject=False, specs=[], count=2) == [
        {"type": "approve"},
        {"type": "approve"},
    ]
    assert _decisions(approve=False, reject=False, specs=["approve", "reject"], count=2) == [
        {"type": "approve"},
        {"type": "reject"},
    ]
    assert _decisions(
        approve=False, reject=False, specs=[f"edit:{json.dumps({'content': '改过'})}"], count=1
    ) == [{"type": "edit", "args": {"content": "改过"}}]


def test_the_cli_refuses_a_decision_count_that_does_not_match() -> None:
    """对不上就当场报错：别等中间件在模型那头抛一句没人看得懂的「decisions 数量不对」。"""
    import typer

    from rag.cli.main import _decisions

    with pytest.raises(typer.BadParameter, match="待批的操作有 2 条"):
        _decisions(approve=False, reject=False, specs=["approve"], count=2)
    with pytest.raises(typer.BadParameter, match="没有等人批准的操作"):
        _decisions(approve=True, reject=False, specs=[], count=0)
    with pytest.raises(typer.BadParameter, match="只用一种"):
        _decisions(approve=True, reject=False, specs=["approve"], count=1)
    with pytest.raises(typer.BadParameter, match="不认识的决定"):
        _decisions(approve=False, reject=False, specs=["随便"], count=1)
