"""三段流水线的**第二段**：Document → Chunk（切分 + 定位 + 图片描述）。

单独一个模块，是为了让它能被两处复用而不打架：

* ``ingest/__init__.py::prepare_paths`` —— 一口气跑完「解析 + 切分」的老入口；
* ``ingest/stages.py`` —— 按阶段跑，切分那一段要能从**磁盘上的解析产物**接着做。

它不碰 Milvus、不碰文件系统（除了读图片本体），纯函数式的：给 Document 列表，
还 Chunk 列表。三段式拆分能不能成立，全看这一步有没有副作用。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, NamedTuple

from rag.config.schema import Settings
from rag.contracts import Chunk, Document, Embedder
from rag.ingest.assets import position_at
from rag.ingest.chunkers import build_chunker
from rag.ingest.images import describe_images
from rag.ingest.loaders.base import SkippedFile

if TYPE_CHECKING:
    from rag.providers.vision import VisionClient

# 正文里的图片标记。剥掉它之后还剩什么，才是这一块真正的检索内容。
_MARKER = re.compile(r"!\[[^\]]*\]\(asset:[^)]+\)")
# 只有 markdown 分隔符/表格分隔行的正文 —— 一个字都没有（`| --- |` 这种）
_SEPARATOR_ONLY = re.compile(r"^[|\-\s*#=]+$")

# 空壳块过滤规则的版本。它进 ``pipeline_version``：规则一改，切出来的块集合就变了，
# 已灌的语料必须重切（不写进指纹的话增量会跳过所有文件，库里留着过滤前的块）。
SHELL_FILTER_VERSION = "1"


def build_chunks(
    documents: list[Document],
    settings: Settings,
    embedder: Embedder | None,
    vision: VisionClient | None,
) -> ChunkBuild:
    """切分 → 定位 → 图片描述 → 丢掉空壳块。"""
    chunks = chunk_documents(documents, settings, embedder)
    locate(chunks, documents)
    # 图片描述要继承所在块的页码/行号，所以排在 locate 之后（见 ingest/images.py）
    described, failed = describe_images(documents, chunks, settings, vision)
    # 空壳块要在**图片描述之后**丢：描述块的标题路径/页码是继承自含标记的那一块的，
    # 先丢掉那一块，描述就丢了位置。
    kept, shells = drop_shell_chunks([*chunks, *described])
    return ChunkBuild(kept, failed, len(described), shells)


class ChunkBuild(NamedTuple):
    """阶段2 的产出：块、失败的图、图片描述条数、丢掉的空壳块数。"""

    chunks: list[Chunk]
    failed: list[SkippedFile]
    described: int
    shells: int


def is_shell_chunk(chunk: Chunk) -> bool:
    """空壳块：剥掉标题路径与图片标记后，没有任何可检索的文字。

    实测来源（那份 10MB 的 Word 文档，现已不在语料里）：173 个正文块里 21 个
    正文不足 50 字，最短的只剩一张表的 `| --- |` 分隔行，还有一个正文里只有一个
    `![image](asset:…)` 标记。

    这类块没有任何检索价值，却会占召回名额、拉低引用质量；而它们携带的信息
    （图在哪、什么标题下）在图片描述块里已经完整保留了。**只丢「一个字都不剩」的**，
    短但真有内容的块一律保留 —— 这条线不能靠长度阈值划。
    """
    if chunk.modality == "image":
        return False  # 图片描述块本身就有内容，且是图片在库里的唯一入口
    text = _MARKER.sub("", _body_of(chunk)).strip()
    return not text or bool(_SEPARATOR_ONLY.fullmatch(text))


def drop_shell_chunks(chunks: list[Chunk]) -> tuple[list[Chunk], int]:
    """丢掉空壳块，返回 ``(留下的块, 丢了几个)``。

    序号**不重排**：`chunk_id = doc_id:序号` 要跨次灌库稳定，重排会让每次跑出来的
    chunk_id 都不一样。留下空号是好事 —— 一眼能看出「这块被过滤了」。
    """
    kept = [chunk for chunk in chunks if not is_shell_chunk(chunk)]
    return kept, len(chunks) - len(kept)


def chunk_documents(
    documents: list[Document], settings: Settings, embedder: Embedder | None
) -> list[Chunk]:
    """按每个文档**自己声明的**策略切，没声明才用全局配置。

    为什么需要按文档分发：一份语料里往往散文和表格混着（md 和 xlsx 一起灌）。
    散文按语义切，表格得按行切 —— 一个全局策略服务不了两边。loader 在
    ``Document.metadata["chunk_strategy"]`` 里写下切法，这里按它分组。
    """
    groups: dict[str, list[Document]] = {}
    for doc in documents:
        strategy = doc.metadata.get("chunk_strategy") or settings.indexing.chunk.strategy
        groups.setdefault(strategy, []).append(doc)

    chunks: list[Chunk] = []
    for strategy, group in groups.items():
        cfg = settings.indexing.chunk.model_copy(update={"strategy": strategy})
        chunks.extend(build_chunker(cfg, embedder).split(group))
    return chunks


def _body_of(chunk: Chunk) -> str:
    """块里那段**来自原文**的正文 —— 标题路径要剥掉。

    标题是我们贴上去的（``with_heading_path``），原文里没有它，带着它去找必然失败。
    """
    text = chunk.text
    if chunk.heading_path:
        prefix = f"{chunk.heading_path}\n\n"
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return text.strip()


def _squashed(text: str) -> tuple[str, list[int]]:
    """去掉所有空白字符，并记下每个保留字符在原文里的偏移。

    **为什么不能直接 find**：`MarkdownHeaderTextSplitter` 会**改写空白** —— 实测连续空行、
    行尾空格、CRLF 都被它归一化过，所以切出来的块正文不是原文的逐字子串，直接找会大面积落空
    （尤其是空白多的那几页）。

    去掉空白再匹配就绕开了这层改写；拿到压缩文本里的下标后，用这张映射表换回**原文**偏移，
    位置仍然精确。
    """
    chars: list[str] = []
    offsets: list[int] = []
    for index, char in enumerate(text):
        if not char.isspace():
            chars.append(char)
            offsets.append(index)
    return "".join(chars), offsets


def locate(chunks: list[Chunk], documents: list[Document]) -> None:
    """切完块之后，把每块的起始偏移映射回来源位置（页码 / 行号）。

    **不改 chunker**：切块器（含 LangChain 的）只返回字符串、不报偏移，让它们带上
    偏移要改一大圈。改成切完之后再定位，反而更简单。

    用**单调游标**处理 overlap：同一段文字会出现在相邻两块里，每次从上一块的起点往后找，
    就不会匹配到更早的那一份。
    """
    by_id = {doc.doc_id: doc for doc in documents}
    cursors: dict[str, int] = {}
    cache: dict[str, tuple[str, list[int], list[int]]] = {}

    for chunk in chunks:
        doc = by_id.get(chunk.doc_id)
        if doc is None or not doc.positions:
            continue
        body = _body_of(chunk)
        if not body:
            continue

        if chunk.doc_id not in cache:
            squashed, mapping = _squashed(doc.text)
            cache[chunk.doc_id] = (squashed, mapping, [p.offset for p in doc.positions])
        squashed, mapping, offsets = cache[chunk.doc_id]

        needle = "".join(body.split())
        if not needle:
            continue
        cursor = cursors.get(chunk.doc_id, 0)
        at = squashed.find(needle, cursor)
        if at < 0:
            # 理论上块文本一定出现在正文里。真找不到也不静默填一个假位置 ——
            # 位置留 0，调用方一眼能看出「这块没定位上」。
            continue
        cursors[chunk.doc_id] = at + 1
        position = position_at(doc.positions, offsets, mapping[at])
        chunk.page = position.page
        chunk.line = position.line
