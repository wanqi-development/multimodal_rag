"""markdown 感知切块：先按标题切，再把超长小节交给递归切块。

**标题不进正文**：``strip_headers=True`` 把标题从正文里剥掉，切完再以「标题路径」
的形式追加到末尾（见 base.with_heading_path）。这样正文是干净的内容本身，
而标题词照样进向量 —— 实测放末尾比放开头更能保住向量区分度，见
docs/milvus-schema-design.md §2.4。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag.config.schema import ChunkingConfig
from rag.contracts import Chunk, Document
from rag.ingest.chunkers.base import (
    make_chunk,
    register_chunker,
    sections_of,
    with_heading_path,
)
from rag.ingest.chunkers.recursive import SEPARATORS

if TYPE_CHECKING:
    from rag.contracts import Embedder


@register_chunker("markdown")
class MarkdownChunker:
    """按标题切块，标题路径进显式字段。

    比纯递归强的地方：一个小节不会被从中间切开（整节是一个语义单元），
    且每块都带着 ``heading_path``（给人看、给大模型看）与 ``heading_numbers``
    （位置派生的编号，给过滤用）。

    **标题不在正文里**：切分时剥掉，切完追加到末尾 —— 正文干净，标题词照样进向量。
    """

    def __init__(self, cfg: ChunkingConfig, embedder: Embedder | None = None) -> None:
        """embedder 用不上 —— 签名统一成 (cfg, embedder=None)，见 base.py 的说明。"""
        # 延迟 import：这个包的 __init__ 会顺带拉起 sentence_transformers / torch（实测 5.5 秒），
        # 见 base.py 里的说明
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self.name = f"markdown:{cfg.size}/{cfg.overlap}"
        # 兜底：一个小节本身超过 size 时还得切（大标题下塞几千字很常见）
        self._overflow = RecursiveCharacterTextSplitter(
            chunk_size=cfg.size,
            chunk_overlap=cfg.overlap,
            separators=SEPARATORS,
            keep_separator="end",
            length_function=len,
        )

    def split(self, docs: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in docs:
            index = 0
            # 层级从哪来由 sections_of 决定：人工对齐过的清单优先，否则按 markdown 标题
            for heading_path, heading_numbers, body in sections_of(doc):
                for piece in self._overflow.split_text(body):
                    text = with_heading_path(piece, heading_path)
                    if not text.strip():
                        continue
                    # 标题不再进 metadata：它已经是显式字段了（避免同一件事存两处）
                    chunks.append(
                        make_chunk(
                            doc,
                            index,
                            text,
                            heading_path=heading_path,
                            heading_numbers=heading_numbers,
                        )
                    )
                    index += 1
        return chunks
