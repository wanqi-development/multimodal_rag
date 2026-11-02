"""主 Agent 对话页的非问答辅助端点。"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from rag.api.deps import get_components
from rag.api.schemas import TranscribeResponse

chat_router = APIRouter()


@chat_router.post("/chat/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    file: UploadFile = File(...), components=Depends(get_components)
) -> TranscribeResponse:
    """语音转文字。只回文字：不落盘、不落库、不进会话。"""
    from rag.providers import build_asr
    from rag.providers.asr import AsrError, AsrNoSpeechError

    limit = int(components.settings.api.media_max_audio_mb * 1_000_000)
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
    except ValueError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    try:
        text = await asyncio.to_thread(
            client.transcribe,
            data,
            name=file.filename or "",
            content_type=file.content_type or "",
        )
    except AsrNoSpeechError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except AsrError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    return TranscribeResponse(text=text, model=client.model, chars=len(text))
