"""递归切块：按分隔符优先级逐级降级。**分隔符只有换行**，越界部分按字符硬切。

见下面 SEPARATORS 的注释：降级阶梯越往下走，切口越接近句子中间，
所以这里刻意只保留「段落边界」这一级，而不是一路退到逗号与空格。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag.config.schema import ChunkingConfig
from rag.contracts import Chunk, Document
from rag.ingest.chunkers.base import make_chunk, register_chunker

if TYPE_CHECKING:
    from rag.contracts import Embedder

# **只按换行切**（2026-09-30 收窄）。之前是「段落 → 句末标点 → 逗号 → 空格」一路退下去，
# 退得太远反而在**句子中间**下刀：一个 1200 字的段落会被切成「…，」+「剩下的半句…」，
# 两块各自都不完整（dense 被截断，BM25 的 term 统计跟着失真）。
#
# 换行是文档里最硬的边界（一个段落就是一段话）。按它切不动的长段落宁可先留大块，
# 真嫌大再往下加「。」—— 而不是一上来就用逗号、空格去凑尺寸。
#
# 末尾那个空串不是「分隔符」，是 LangChain 约定的**按字符硬切**兜底：没有它，
# 一个没有任何换行的超长段落会整段成为一块（超过 Milvus 的 VARCHAR 8192 字节上限就写不进去）。
SEPARATORS = ["\n", ""]


@register_chunker("recursive")
class RecursiveChunker:
    """langchain 的 RecursiveCharacterTextSplitter，分隔符只留换行。

    length_function 用 len：按字符数切。中文 1 字≈1 token，与
    providers.embedding.dim 那边的窗口约束配套（见 ChunkingConfig 的说明）。
    """

    def __init__(self, cfg: ChunkingConfig, embedder: Embedder | None = None) -> None:
        """embedder 用不上 —— 签名统一成 (cfg, embedder=None)，见 base.py 的说明。"""
        # 延迟 import：那个包的 __init__ 会顺带拉起 sentence_transformers / torch（实测 5.5 秒）
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self.name = f"recursive:{cfg.size}/{cfg.overlap}"
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.size,
            chunk_overlap=cfg.overlap,
            separators=SEPARATORS,
            # 句末标点留在前一句末尾（"start"/True 都会把它推到下一块开头）
            keep_separator="end",
            length_function=len,
        )

    def split(self, docs: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in docs:
            index = 0
            for piece in self._splitter.split_text(doc.text):
                text = piece.strip()
                if not text:
                    continue
                chunks.append(make_chunk(doc, index, text))
                index += 1
        return chunks
