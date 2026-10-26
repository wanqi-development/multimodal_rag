"""检索、图片与语音上传、普通问答端点。"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from rag.api.common import HEARTBEAT_S, UNPROCESSABLE, _citations, _event, _hits, _stats
from rag.api.deps import get_components
from rag.api.schemas import (
    ChatRequest,
    ChatResponse,
    ImageUploadView,
    SearchRequest,
    SearchResponse,
    TranscribeResponse,
)

search_chat_router = APIRouter()


@search_chat_router.post("/search/image", response_model=ImageUploadView, status_code=201)
async def upload_search_image(
    file: UploadFile = File(...), components=Depends(get_components)
) -> ImageUploadView:
    """上传一张图给「以图搜图」。**只落盘、只回路径** —— 检索仍由 `/search` 发起。

    为什么不合成一个 multipart 端点直接搜：`/search` 那条要过线上 embedding（按量计费），
    而「传了图、还没点搜」是常态。分开之后，重搜/换词/调 top_k 都不用重传图上。

    落点 `api.media_dir`（默认 `data/chat/media/`），**不是 `data/uploads/`** ——
    那里的 `.png/.jpg` 会被当成文档列出来走五步流程。上限 `api.media_max_image_mb`（默认 5MB）
    **在上游之前拦**：打过去就是一次按量的 embedding。
    """
    from rag.api.media import MediaError, save_image

    limit = int(float(components.settings.api.media_max_image_mb) * 1_000_000)
    # 多读 1 个字节就够判断「超了」，别把一张几十 MB 的图整个读进内存
    data = await file.read(limit + 1)
    try:
        path = save_image(
            components.settings,
            data,
            name=file.filename or "",
            content_type=file.content_type or "",
        )
    except MediaError as exc:
        code = (
            status.HTTP_413_CONTENT_TOO_LARGE if "上限" in str(exc) else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(code, str(exc)) from exc
    return ImageUploadView(path=str(path), name=path.name, bytes=len(data))


@search_chat_router.post("/search", response_model=SearchResponse)
async def search(body: SearchRequest, components=Depends(get_components)) -> SearchResponse:
    """纯检索（M2 那一整套：多通道 + 融合 + 重排 + 过滤 + 以图搜图）。"""
    image = _image(body.image_path)
    if not body.query.strip() and image is None:
        raise HTTPException(UNPROCESSABLE, "query 与 image_path 至少给一个")
    hits, stats = await components.search(
        query=body.query,
        top_k=body.top_k,
        filters=body.filters,
        channels=body.channels,
        rerank=body.rerank,
        image=image,
    )
    return SearchResponse(query=body.query, hits=_hits(hits), stats=_stats(stats))


def _image(raw: str | None) -> Path | None:
    if raw is None:
        return None
    path = Path(raw)
    if not path.is_file():
        raise HTTPException(UNPROCESSABLE, f"image_path 不存在：{path}")
    return path


@search_chat_router.post("/chat/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    file: UploadFile = File(...), components=Depends(get_components)
) -> TranscribeResponse:
    """语音转文字（对话页输入框那个麦克风）。**只转文字：不落盘、不落库、不进会话。**

    一次调用就完、体量也小，所以不走异步任务那套（与解析/入库不同）—— 请求里同步等结果。

    上限 `api.media_max_audio_mb`（默认 10MB）**在上游之前拦**：打上去了就是一次按量计费，
    哪怕上游最后回你 413。返回的文字由**前端放进输入框让人确认**，这里不代提问。
    """
    from rag.providers import build_asr
    from rag.providers.asr import AsrError, AsrNoSpeechError

    limit = int(components.settings.api.media_max_audio_mb * 1_000_000)
    # 多读 1 个字节就够判断「超了」，别把一个几百 MB 的东西整个读进内存
    data = await file.read(limit + 1)
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "音频是空的")
    if len(data) > limit:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"音频超过上限 {limit / 1e6:.0f}MB（要放大就改 api.media_max_audio_mb）",
        )
    try:
        client = build_asr(components.settings, components.secrets)
    except ValueError as exc:  # 缺 key / 缺地址：说清去哪填，别吐一个 500
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    try:
        # `AsrClient` 是同步的（httpx），丢线程池 —— 别把事件循环按住
        text = await asyncio.to_thread(
            client.transcribe,
            data,
            name=file.filename or "",
            content_type=file.content_type or "",
        )
    except AsrNoSpeechError as exc:
        # 「没识别到说话」是**这次输入**的问题（静音/没选中麦克风），不是上游挂了 → 422
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except AsrError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    return TranscribeResponse(text=text, model=client.model, chars=len(text))


@search_chat_router.post("/chat", response_model=None)
async def chat(body: ChatRequest, components=Depends(get_components)):
    """问答。`stream=true` 走 SSE（`retrieval` → `token`* → `done`），否则一次性 JSON。"""
    if not body.stream:
        result = await components.answer(body)
        return ChatResponse(
            answer=result["answer"],
            citations=_citations(result["citations"]),
            hits=_hits(result["hits"]),
            stats=_stats(result["stats"]),
            rounds=result["rounds"],
            grade=result["grade"],
            mode=result.get("mode") or "plain",
            usage=result.get("usage"),
            session_id=result.get("session_id"),
        )
    return StreamingResponse(
        _sse(components, body),
        media_type="text/event-stream",
        headers={
            # 关掉中间层缓冲，否则 token 会攒着一次性到（Nginx 尤其明显）
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _sse(components, body: ChatRequest):
    """三轮事件的生产者/消费者之间垫一个队列，才能**同时**做三件事：
    等下一个 token、按间隔发心跳、客户端断开时把生成任务停掉。

    自己写而不用 `sse_starlette`（本环境没装）：要的就是心跳与断线取消这两件 ——
    空闲连接会被代理掐，客户端关页面后模型还在烧钱。
    """
    queue: asyncio.Queue = asyncio.Queue()

    async def produce() -> None:
        try:
            async for event in components.stream_answer(body):
                kind = event["type"]
                if kind == "retrieval":
                    await queue.put(
                        (
                            "retrieval",
                            {
                                "hits": [item.model_dump() for item in _hits(event["hits"])],
                                "stats": [item.model_dump() for item in _stats(event["stats"])],
                                "rounds": event.get("rounds") or 1,
                                "grade": event.get("grade") or {},
                                "mode": event.get("mode") or "plain",
                            },
                        )
                    )
                elif kind == "token":
                    await queue.put(("token", event["text"]))
                else:
                    await queue.put(
                        (
                            "done",
                            {
                                "citations": _citations(event.get("citations")),
                                "usage": event.get("usage"),
                                "rounds": event.get("rounds") or 1,
                                "grade": event.get("grade") or {},
                                "mode": event.get("mode") or "plain",
                                # 会话 id 要带回给客户端，下一轮追问才知道接在哪一段后面
                                "session_id": event.get("session_id"),
                            },
                        )
                    )
        except Exception as exc:  # noqa: BLE001 —— 流里出什么事都要让客户端看到一句话
            await queue.put(("error", f"{type(exc).__name__}: {exc}"))

    task = asyncio.create_task(produce())
    try:
        while True:
            try:
                kind, payload = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_S)
            except TimeoutError:
                yield ": ping\n\n"  # SSE 注释行：客户端忽略，但能把空闲连接保活
                continue
            if kind == "token":
                yield _event("token", {"text": payload})
            elif kind == "done":
                yield _event(
                    "done",
                    {
                        "citations": [
                            item.model_dump() for item in (payload.get("citations") or [])
                        ],
                        "session_id": payload.get("session_id"),
                        "usage": payload.get("usage"),
                    },
                )
                break
            elif kind == "retrieval":
                yield _event("retrieval", payload)
            else:
                yield _event("error", {"message": payload})
                break
    finally:
        # 客户端断开（或我们出错）时把生产者停掉 —— 不然模型还在这儿一个 token 一个 token 地跑
        if not task.done():
            task.cancel()
        with contextlib.suppress(Exception):
            await task
