"""HTTP 路由共享的响应转换与 SSE 报文工具。"""

from __future__ import annotations

import json

from fastapi import status

from rag.api.schemas import CitationView, HitView, StatsView
from rag.contracts import ScoredChunk

HEARTBEAT_S = 15.0

UNPROCESSABLE = (
    getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", None) or status.HTTP_422_UNPROCESSABLE_ENTITY
)


def _hits(chunks: list[ScoredChunk]) -> list[HitView]:
    return [
        HitView(
            chunk_id=hit.chunk.chunk_id,
            score=hit.score,
            sources=hit.sources,
            doc_id=hit.chunk.doc_id,
            modality=hit.chunk.modality,
            doc_type=hit.chunk.doc_type,
            source_uri=str(hit.chunk.metadata.get("source_uri", "")),
            heading_path=hit.chunk.heading_path,
            heading_numbers=hit.chunk.heading_numbers,
            page=hit.chunk.page,
            line=hit.chunk.line,
            assets=hit.chunk.assets,
            text=hit.chunk.text,
        )
        for hit in chunks
    ]


def _stats(rows) -> list[StatsView]:
    return [
        StatsView(
            channel=row.channel,
            candidates=row.candidates,
            kept=row.kept,
            latency_ms=row.latency_ms,
            error=row.error,
        )
        for row in rows or []
    ]


def _citations(rows) -> list[CitationView]:
    return [
        CitationView(
            index=item.index,
            source_type=item.source_type,
            chunk_id=item.chunk_id,
            source_uri=item.source_uri,
            url=item.url,
            title=item.title,
            snippet=item.snippet,
            heading_path=item.heading_path,
        )
        for item in rows or []
    ]


def _event(name: str, payload: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
