"""四级切块：header → 递归 → 语义兜底 → 按大小硬切。

每一级都在解决上一级解决不了的问题，顺序刻意从便宜排到贵：

| 级别 | 手段 | 解决什么 |
|---|---|---|
| 1 | markdown 标题（h1/h2/h3） | 最可靠的语义边界，顺带把标题路径写进元数据 |
| 2 | 递归（``\\n\\n`` / ``\\n`` / ``。``） | 小节内部的段落与句子边界，覆盖绝大多数情况 |
| 3 | 语义（句向量 + percentile 断点） | 递归切不动的：只有逗号、或根本没有句子边界 |
| 4 | 按字符硬切 | 语义断点**不是尺寸控制**；表格 HTML、无标点长串只能硬切 |

第 3 级是唯一贵的一步（每句话一次 embedding），所以只让它处理第 2 级剩下的残留 ——
正常语料里这部分很少。

**分隔符只保留前三个**（``\\n\\n`` / ``\\n`` / ``。``）。逗号、分号、空格切开会把句子
拦腰截断，碎片化比「块大一点」更伤检索；而缺了这些分隔符就切不动的文本，正好交给第 3 级。

**为什么不用 LangChain 的 ``SemanticChunker``**（ADR 22）：那个包已 sunset；它的默认分句
正则 ``(?<=[.?!])\\s+`` 对中文完全无效（整段算一句，直接原样返回）；同语料下还慢 2.5 倍，
且没有尺寸护栏（切完最大块超出阈值近一倍）。
"""

from __future__ import annotations

import asyncio
import math
import re
from typing import TYPE_CHECKING

from rag.config.schema import ChunkingConfig
from rag.contracts import Chunk, Document
from rag.ingest.chunkers.base import (
    make_chunk,
    register_chunker,
    sections_of,
    with_heading_path,
)

if TYPE_CHECKING:
    from rag.contracts import Embedder

# 只要前三个。多余的分隔符会把句子截断，碎片化更伤检索 ——
# 切不动的部分留给语义那一级。
SEPARATORS = ["\n\n", "\n", "。"]

# 语义那一级的**分句器**。这里要认逗号：递归切不动的一大类正是「只有逗号、没有句号」
# 的文本，分句器不认逗号就还是切不动。
SENTENCE_END = re.compile(r"(?<=[。！？；，])")


def _cosine(left: list[float], right: list[float]) -> float:
    """余弦相似度。不假设向量已归一化 —— 换 embedder 时不该悄悄算错距离。"""
    # strict=True：维度不一致就是出了真问题，宁可当场炸也不要截到较短的那个上
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)


def _percentile(values: list[float], pct: float) -> float:
    """线性插值分位数，与 numpy.percentile 默认口径一致。

    自己写是为了不引 numpy —— 它现在只是 torch / fastembed 的传递依赖，
    不该成为切块器的直接依赖。
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (pct / 100.0)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@register_chunker("markdown-semantic")
class MarkdownSemanticChunker:
    """按标题切、按语义兜底，最后保证不超 ``size``。"""

    def __init__(self, cfg: ChunkingConfig, embedder: Embedder | None = None) -> None:
        if embedder is None:
            # 不静默退化：缺 embedder 时偷偷只做硬切，会让人以为语义切分生效了
            raise ValueError(
                "chunker 'markdown-semantic' 需要 embedder（语义兜底那一步要算句向量）。"
                "prepare() 的 embedder 参数没传，或者配置里 strategy 写错了。"
            )
        self._cfg = cfg
        self._embedder = embedder
        self.name = f"markdown-semantic:{cfg.size}/{cfg.overlap}"
        # 延迟 import：那个包的 __init__ 会顺带拉起 sentence_transformers / torch（实测 5.5 秒）
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self._recursive = RecursiveCharacterTextSplitter(
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
                for piece in self._split_section(body):
                    text = with_heading_path(piece, heading_path)
                    if not text.strip():
                        continue
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

    def _split_section(self, text: str) -> list[str]:
        """第 2 级之后仍然超限的，才进第 3、4 级。"""
        out: list[str] = []
        for piece in self._recursive.split_text(text):
            out.extend(self._from_leftover(piece) if len(piece) > self._cfg.size else [piece])
        return out

    def _from_leftover(self, text: str) -> list[str]:
        sentences = [s for s in SENTENCE_END.split(text) if s.strip()]
        if len(sentences) < 3:
            # 连句子都分不出来（表格 HTML、无标点长串），语义那一步无从下手
            return self._hard_cut(text)
        vectors = asyncio.run(self._embedder.embed_documents(sentences)).dense
        return self._merge(sentences, self._breakpoints(vectors))

    def _breakpoints(self, vectors: list[list[float]]) -> set[int]:
        """相邻句的余弦距离超过 percentile 分位，就算一个语义断点。"""
        # 相邻两句：长度天然差一个，不能用 strict
        distances = [1.0 - _cosine(a, b) for a, b in zip(vectors, vectors[1:], strict=False)]
        if not distances:
            return set()
        threshold = _percentile(distances, self._cfg.semantic.percentile)
        return {i for i, distance in enumerate(distances) if distance > threshold}

    def _merge(self, sentences: list[str], breakpoints: set[int]) -> list[str]:
        """在断点处合并句子，带两条护栏。

        - ``min_size``：断点太密就不切 —— 实测裸 percentile 会切出 3 个字的块
        - ``size``：攒够上限就强制切 —— **语义断点不是尺寸控制**，没有这一条
          就保证不了块不超窗口（LangChain 那版正是栽在这里）
        """
        out: list[str] = []
        buffer = ""
        for i, sentence in enumerate(sentences):
            if buffer and len(buffer) + len(sentence) > self._cfg.size:
                out.append(buffer)
                buffer = ""
            buffer += sentence
            if i in breakpoints and len(buffer) >= self._cfg.semantic.min_size:
                out.append(buffer)
                buffer = ""
        if buffer:
            out.append(buffer)

        # 单句本身就超上限（表格 HTML 那种），只能硬切
        return [
            piece
            for merged in out
            for piece in (self._hard_cut(merged) if len(merged) > self._cfg.size else [merged])
        ]

    def _hard_cut(self, text: str) -> list[str]:
        """按字符切。最后一刀，保证「块不超 size」这条不出例外。"""
        return [text[i : i + self._cfg.size] for i in range(0, len(text), self._cfg.size)]
