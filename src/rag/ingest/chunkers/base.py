"""chunker 的公共部分：注册表 + chunk_id 生成。

与 providers 的约定一致：注册的类构造函数只收「它自己那段配置对象」。

例外是**语义切分**：它要算句向量，得多收一个 embedder。为了不破坏统一约定，
所有 chunker 的构造函数都统一成 ``(cfg, embedder=None)`` —— 用不上的忽略它，
和 ``build_loader(cfg, ocr)`` 是同一个路子。
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

from rag.config.schema import ChunkingConfig
from rag.contracts import Chunk, Chunker, Document, Embedder
from rag.ingest.assets import asset_uris_in
from rag.ingest.headings import MAX_LEVEL, HeadingTree, chapter_headings, scan_headings
from rag.ingest.outline import Section, sections

# 参与切分的标题级别：**到 h3 为止**（用户 2026-10-01 定的），也就是 `headings.MAX_LEVEL`。
# 更深的标题（h4+）既不是切分边界、也不进标题路径，留在正文里当内容 —— 切到 h4 会把
# 一节拆成一堆小碎块，一小节一小块反而不如「一节一块」看得全。
#
# 这条只管 docx / html / md 这些「结构本来就写在文件里」的来源；PDF 走人工对齐的
# `outline.tsv`（`sections_of` 优先用它），那边由 `outline.heading_lines` 守着同一个级别。
HEADERS = [(f"{'#' * level}", f"h{level}") for level in range(1, MAX_LEVEL + 1)]
_HEADING_LEVELS = tuple(range(1, MAX_LEVEL + 1))

CHUNKERS: dict[str, type] = {}


def register_chunker(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        existing = CHUNKERS.get(name)
        if existing is not None:
            raise ValueError(f"chunker 名称冲突: {name!r} 已被 {existing!r} 占用")
        CHUNKERS[name] = cls
        return cls

    return deco


def build_chunker(cfg: ChunkingConfig, embedder: Embedder | None = None) -> Chunker:
    """按配置里的 strategy 实例化。拼错策略名会立刻报错，不会静默退化。

    embedder 只有语义切分用得上；真要用了却没传，那个 chunker 自己会报错
    （见 semantic.MarkdownSemanticChunker），不会悄悄退化成只做硬切。
    """
    cls = CHUNKERS.get(cfg.strategy)
    if cls is None:
        raise KeyError(f"未注册的 chunker: {cfg.strategy!r}；已注册: {sorted(CHUNKERS)}")
    return cls(cfg, embedder)


def make_chunk(
    doc: Document,
    index: int,
    text: str,
    *,
    metadata: dict[str, Any] | None = None,
    heading_path: str = "",
    heading_numbers: str = "",
    modality: str = "text",
) -> Chunk:
    """chunk_id = ``doc_id:序号``，确定性生成。

    确定性是 upsert 能原地覆盖的前提：重新 ingest 同一份文档时 chunk_id 不变，
    旧块被覆盖；若用随机 id，每跑一次就往库里堆一份副本。

    metadata 里带上 source_uri：Chunk 没有这个字段，而 Milvus 里存的是块不是文档，
    引用要显示来源就得在块上带着它。doc 级的元数据（如 page_count）一并继承。

    这里顺手把「能从 text 直接看出来」的东西算成**显式字段**：
    ``assets``（这一块引用了哪几张图）与 ``content_hash``（去重用）。
    ``page`` / ``line`` 由 prepare 之后的 locate 步骤补 —— 那时候才知道偏移。
    """
    return Chunk(
        chunk_id=f"{doc.doc_id}:{index:04d}",
        doc_id=doc.doc_id,
        text=text,
        index=index,
        doc_type=str(doc.metadata.get("parser", "")),
        modality=modality,
        heading_path=heading_path,
        heading_numbers=heading_numbers,
        assets=asset_uris_in(text),
        content_hash=hashlib.sha1(text.encode("utf-8")).hexdigest()[:16],
        metadata={
            "source_uri": doc.source_uri,
            # 下划线开头的是管线内部用的（如 `_heading_levels`），不进库
            **{k: v for k, v in doc.metadata.items() if not k.startswith("_")},
            **(metadata or {}),
        },
    )


def sections_of(doc: Document) -> list[Section]:
    """一份文档要切成哪几节 —— 层级从哪来由它决定。

    * ``doc.metadata["_heading_levels"]`` 存在：**以人工核对过的清单为准**（``outline.py``），
      层级直接用它，不再靠 markdown 的 `#` 数量或编号去猜；切到第几级也听清单的
      （``_max_level``，不同文档可以不一样）。run_chunk 已经把「没核对的文档」挡在门外，
      所以走到这里的 Document 一定是核对过的。
    * 否则：用 markdown 标题切分器（``#`` 数量 + 编号推断，见 ``HeadingNumbering``），
      也就是 docx / html / md 这些「结构本来就写在文件里」的来源（没有清单时才会走，
      正常情况下所有来源都有清单）。
    """
    levels = doc.metadata.get("_heading_levels")
    if isinstance(levels, list) and levels:
        return sections(
            doc.text,
            [int(level) for level in levels],
            max_level=int(doc.metadata.get("_max_level", MAX_LEVEL)),
        )
    # **延迟 import**：`langchain_text_splitters` 的包 `__init__` 会把
    # sentence_transformers（进而 torch / transformers）一起拉进来 —— 实测 5.5 秒，
    # 而 `rag --help` / `rag sessions list` 这类命令根本用不到切分器。重依赖一律晚点拿。
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS, strip_headers=True)
    numbering = heading_numbering(doc)
    out: list[Section] = []
    for piece in splitter.split_text(doc.text):
        path, numbers = numbering.update(dict(piece.metadata))
        out.append(Section(path, numbers, piece.page_content))
    return out


class HeadingNumbering:
    """把「这一节出现了哪些层级的标题」翻成 ``(从根开始的路径, 层级编号)``。

    **2026-09-30 重写**：以前只按位置数（第 1 个 h1 是 1、它下面第 2 个 h2 是 1-2），
    在层级可靠的文档上没问题；但这些 OCR 出来的教材 PDF 里，章、节、列表项经常被
    标成同一级，位置派生的编号就变成了 ``4-17`` 这种谁也看不懂的东西，路径还会
    被同级标题挤断（不从根开始）。现在改成**按作者编号重建树**，见
    ``rag.ingest.headings`` 的模块说明。

    用法不变：按文档顺序把每个小节的标题元数据喂进来，拿回 ``(路径, 编号)``。
    """

    def __init__(self, chapters: set[str] | None = None) -> None:
        self._tree = HeadingTree(chapters=chapters or set())
        self._last: dict[int, str | None] = {}

    def update(self, metadata: dict[str, Any]) -> tuple[str, str]:
        """一节可能有**多个级别同时变化**（新的一节同时开了 h2 与 h3），
        按级别从浅到深喂进树里，返回最深那次的路径 —— 那才是这一节的位置。"""
        path, numbers = "", ""
        for level in sorted(_HEADING_LEVELS):
            title = metadata.get(f"h{level}")
            if title is None or title == self._last.get(level):
                continue
            self._last[level] = title
            path, numbers = self._tree.push(level, str(title))
        return path, numbers


def heading_numbering(doc: Document) -> HeadingNumbering:
    """给一份文档建编号器：先扫一遍它的标题行，认出哪些单段编号的标题是**章**。"""
    return HeadingNumbering(chapters=chapter_headings(scan_headings(doc.text)))


def with_heading_path(body: str, heading_path: str) -> str:
    """标题路径 + 正文 → 要 embedding 的那段文本。

    **标题不在正文里**：切分时先剥掉（``strip_headers=True``），切完再贴回去 ——
    贴的位置是**开头**，这样读到这一块的人（或模型）先知道它在哪一节里。

    ⚠️ 已知代价：Qwen3-Embedding 是 **last-token 池化**，末位内容对向量影响最大。
    实测同节 8 段正文的平均两两余弦（越低越好区分）：

    | 布局 | 平均两两余弦 |
    |---|---|
    | 标题在末尾 | **0.6830** |
    | 标题在开头（当前选择） | 0.8166 |
    | 不带标题 | 0.5899 |

    也就是**放开头会让同一节的块彼此更相似、区分度下降**。选开头是拿区分度换阅读顺序，
    见 docs/milvus-schema-design.md §2.4。
    """
    return f"{heading_path}\n\n{body.strip()}" if heading_path else body.strip()
