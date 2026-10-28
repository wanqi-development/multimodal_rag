"""会话记录与 agent 端点。"""

from __future__ import annotations

import asyncio
import contextlib

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from rag.api.common import HEARTBEAT_S, _citations, _event
from rag.api.deps import get_components
from rag.api.schemas import (
    AgentRequest,
    AgentResponse,
    AgentResumeRequest,
    DeletedView,
    SessionDetailResponse,
    SessionListResponse,
    SessionMessageView,
    SessionUsageView,
    SessionView,
)

sessions_agent_router = APIRouter()


def _session_view(row: dict) -> SessionView:
    """列表一行 → 响应模型。token 合计单独收成一个对象，前端不用自己挑字段。"""
    return SessionView(
        session_id=row["session_id"],
        title=row.get("title") or "",
        mode=row.get("mode") or "chat",
        thread_id=row.get("thread_id"),
        turns=int(row.get("turns") or 0),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
        usage=SessionUsageView(
            input_tokens=int(row.get("input_tokens") or 0),
            output_tokens=int(row.get("output_tokens") or 0),
            total_tokens=int(row.get("total_tokens") or 0),
            calls=int(row.get("calls") or 0),
        ),
    )


def _chats(components):
    """会话存储。没初始化成功就明确回 503，别让人对着 500 猜。"""
    if components.chats is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "会话存储未初始化；查看 /health 或服务启动日志。",
        )
    return components.chats


@sessions_agent_router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(limit: int = 20, components=Depends(get_components)) -> SessionListResponse:
    """UI 会话列表。

    **和 `rag sessions` 不是一回事**：那个管的是 LangGraph checkpointer 的残留 thread，
    这里是我们自己的对话记录（快问答也有）。表结构见 `persistence/chat.py`。
    token 合计一起给 —— 列表上就要显示「这段花了多少」，别让界面再来一次。
    """
    rows = await _chats(components).list(limit=max(1, min(limit, 100)))
    return SessionListResponse(sessions=[_session_view(row) for row in rows])


@sessions_agent_router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session(session_id: str, components=Depends(get_components)) -> SessionDetailResponse:
    """一段会话的全部消息（已按 `seq` 排好）。刷新页面、换台机器都靠它。"""
    found = await _chats(components).get(session_id)
    if found is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"没有这段会话：{session_id}")
    messages = [SessionMessageView(**item) for item in (found.get("messages") or [])]
    # 有停在闸门的那一轮时**现去 checkpointer 取载荷** —— 中断载荷不进会话表，
    # 重开这段对话要能把批准卡片原样摆回来（使用者明确不接受的损失就是它）。
    if any(item.status == "interrupted" for item in messages):
        payload = await components.pending_interrupt(found.get("thread_id"))
        if payload:
            for item in messages:
                if item.status == "interrupted":
                    item.interrupt = payload
    return SessionDetailResponse(
        session_id=found["session_id"],
        title=found.get("title") or "",
        mode=found.get("mode") or "chat",
        thread_id=found.get("thread_id"),
        created_at=found.get("created_at"),
        updated_at=found.get("updated_at"),
        messages=messages,
        usage=SessionUsageView(**(found.get("usage") or {})),
    )


@sessions_agent_router.delete("/sessions/{session_id}", response_model=DeletedView)
async def delete_session(session_id: str, components=Depends(get_components)) -> DeletedView:
    """删一段会话。消息与 token 行靠外键级联一起走。"""
    if not await _chats(components).delete(session_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"没有这段会话：{session_id}")
    return DeletedView(deleted=session_id)


@sessions_agent_router.post("/agent", response_model=AgentResponse)
async def agent(body: AgentRequest, components=Depends(get_components)) -> AgentResponse:
    """对话式 agent（M5，deepagents）。

    与 `/chat` 的区别：`/chat` 走我们定好的直线或 QueryGraph；这里**让模型自己决定**
    要不要查、查几轮、派给哪个子 agent。代价是 LLM 调用次数多得多。

    批次 B：agent 落在 `data/workspace` 上（`agent.workspace_writes` 可关回只读），
    但**写类工具一律等人批准** —— 撞上闸门时返回 `status="interrupted"` 与
    `interrupt.action_requests`，客户端看完调 `/agent/resume` 放行或拒绝。
    多轮靠 `thread_id`（历史与中断状态都在 PG 的 checkpointer 里，重启还在）。

    `stream=true` 走 SSE，四种事件：`token` / `tool` / `interrupt` / `done`。
    """
    if body.stream:
        return StreamingResponse(
            _sse_events(
                components.stream_agent(
                    body.message,
                    thread_id=body.thread_id,
                    web_explicit=body.web,
                    session_id=body.session_id,
                )
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    result = await components.agent(
        body.message, thread_id=body.thread_id, web_explicit=body.web, session_id=body.session_id
    )
    return AgentResponse(
        answer=result["answer"],
        citations=_citations(result["citations"]),
        thread_id=result["thread_id"],
        steps=result["steps"],
        retrieved=result["retrieved"],
        status=result.get("status", "answered"),
        interrupt=result.get("interrupt"),
        session_id=result.get("session_id"),
    )


@sessions_agent_router.post("/agent/resume", response_model=AgentResponse)
async def agent_resume(
    body: AgentResumeRequest, components=Depends(get_components)
) -> AgentResponse:
    """批准 / 拒绝被中断的那次 tool 调用，接着跑。

    `decisions` 一条对一条 `action_requests`，**顺序要对**（中间件的约定）。
    放行之后可能又撞上下一道闸门：那就再返回一个 `interrupted`，客户端再来一次。

    `stream=true` 和 `/agent` 走**同一组事件**（`token` / `tool` / `interrupt` / `done`）——
    批准之后那段答案照样是一段段生成的，客户端不该为此再写一套解析。
    """
    if body.stream:
        return StreamingResponse(
            _sse_events(
                components.stream_resume(
                    body.thread_id, decisions=body.decisions, session_id=body.session_id
                )
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    result = await components.resume(
        body.thread_id, decisions=body.decisions, session_id=body.session_id
    )
    return AgentResponse(
        answer=result["answer"],
        citations=_citations(result["citations"]),
        thread_id=result["thread_id"],
        steps=result["steps"],
        retrieved=result["retrieved"],
        status=result.get("status", "answered"),
        interrupt=result.get("interrupt"),
        session_id=result.get("session_id"),
    )


async def _sse_events(source):
    """把 agent 事件源翻成 SSE。心跳与断线取消与 `/chat` 同一套（自己写，不引 sse_starlette）。

    事件类型：`token`（主 agent 的正文增量）、`tool`（某次工具调用结束）、
    `interrupt`（等人批准，流到此为止）、`done`（跑完，带引用）。

    **收的是一个异步迭代器，不是一个请求体**：`/agent` 与 `/agent/resume` 的起点不同
    （用户消息 vs `Command(resume=…)`），但下发的事件必须一模一样，
    所以「起点」留在调用方，这里只做翻译。
    """
    queue: asyncio.Queue = asyncio.Queue()

    async def produce() -> None:
        try:
            async for event in source:
                kind = event["type"]
                if kind == "token":
                    await queue.put(("token", event["text"]))
                elif kind == "tool":
                    await queue.put(("tool", event))
                elif kind == "interrupt":
                    # **thread_id 必须带上**：批准是「按会话找回中断状态」的，
                    # 客户端在中断这一刻就得知道该拿哪个 id 去调 /agent/resume。
                    payload = dict(event["interrupt"] or {})
                    payload["thread_id"] = event.get("thread_id")
                    payload["session_id"] = event.get("session_id")
                    await queue.put(("interrupt", payload))
                else:
                    await queue.put(("done", event))
        except Exception as exc:  # noqa: BLE001 —— 流里出什么事都要让客户端看到一句话
            await queue.put(("error", f"{type(exc).__name__}: {exc}"))

    task = asyncio.create_task(produce())
    try:
        while True:
            try:
                kind, payload = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_S)
            except TimeoutError:
                yield ": ping\n\n"
                continue
            if kind == "token":
                yield _event("token", {"text": payload})
            elif kind == "tool":
                yield _event("tool", payload)
            elif kind == "interrupt":
                yield _event("interrupt", payload)
                break
            elif kind == "done":
                yield _event(
                    "done",
                    {
                        "answer": payload["answer"],
                        "citations": [
                            item.model_dump() for item in _citations(payload["citations"])
                        ],
                        "thread_id": payload["thread_id"],
                        "session_id": payload.get("session_id"),
                        # token 用量一起给：界面上要显示「这一轮花了多少」
                        "usage": payload.get("usage"),
                    },
                )
                break
            else:
                yield _event("error", {"message": payload})
                break
    finally:
        if not task.done():
            task.cancel()
        with contextlib.suppress(Exception):
            await task
