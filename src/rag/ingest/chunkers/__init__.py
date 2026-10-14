"""切块层：Document → Chunk。按配置里的 strategy 选实现，见 build_chunker。"""

from __future__ import annotations

from rag.ingest.chunkers.base import CHUNKERS, build_chunker, make_chunk, register_chunker
from rag.ingest.chunkers.markdown import MarkdownChunker  # noqa: F401  触发注册
from rag.ingest.chunkers.recursive import RecursiveChunker  # noqa: F401  触发注册
from rag.ingest.chunkers.rows import TableRowsChunker  # noqa: F401  触发注册
from rag.ingest.chunkers.semantic import MarkdownSemanticChunker  # noqa: F401  触发注册

__all__ = [
    "CHUNKERS",
    "MarkdownChunker",
    "MarkdownSemanticChunker",
    "RecursiveChunker",
    "TableRowsChunker",
    "build_chunker",
    "make_chunk",
    "register_chunker",
]
