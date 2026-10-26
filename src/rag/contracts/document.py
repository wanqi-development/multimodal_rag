"""跨层数据模型。

本模块只依赖 pydantic（项目里处处都在用的基础原语），不依赖 langchain /
langgraph / milvus 等任何框架——这是「换实现不动上层」的前提。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field


class ImageAsset(BaseModel):
    """文档里的一张图。

    正文里留的是 ``marker``（短标记，位置就是原图在文中的位置），图片本体落在
    资产目录、由 ``asset_uri`` 指过去。切块后靠 marker 就能反查「这张图在哪一块、
    上边一段和下边一段是什么」—— 图片描述要带上下文，靠的就是它。
    """

    marker: str
    asset_uri: str
    # 页码。docx / html 这类没有页的概念，留空
    page: int | None = None
    source_name: str = ""


class Position(BaseModel):
    """文本偏移 → 来源位置。

    loader 在拼接正文时记下「哪一段来自哪儿」，切块之后就能反查每块在第几页第几行。
    按 offset 升序、**稀疏**存放 —— 只在页码或行号变化处打点。

    ``page`` / ``line`` 都是 1 起，0 表示**该格式没有这个概念**（html 没有行号、
    docx 没有页码）。语义随 ``doc_type`` 变，引用展示时按来源渲染成人话，
    不假装它们是统一的东西。
    """

    offset: int
    page: int = 0
    line: int = 0


class DocRecord(BaseModel):
    """文档注册表的一行：**每份文档一行**，不是每块一行。

    这是增量更新的唯一依据 —— 有了它才能回答「这份文件上次灌的是什么、变了没有」。
    与块表的 ``doc_id`` 一一对应。
    """

    doc_id: str
    source_uri: str
    # **文件字节**的 sha1[:16]。变更检测的钥匙：只读文件、不用解析，便宜到能对全语料跑一遍
    content_hash: str
    chunk_count: int = 0
    # Unix 时间戳
    ingested_at: int = 0
    # 切分策略 + 清洗规则 + embedding 模型的指纹。内容没变但切法变了，靠它强制重灌
    pipeline_version: str = ""
    # 这份文档**走到哪一步了**（见 ``DOC_STAGES``）。列表页只读它，不去翻产物。
    stage: str = ""
    # 最近一次索引写入的批次与状态：staging → active / failed。
    # batch 在写 Milvus 前落库，只有 Milvus 与注册表都写完才转 active；
    # 中途失败会留下可对账的 failed / staging 记录，而不是只靠猜。
    index_batch_id: str = ""
    index_status: str = ""
    indexed_at: int = 0
    index_error: str = ""
    # 最近一次索引写入的批次与状态：staging → active / failed。
    # batch 在写 Milvus 前落库，只有 Milvus 与注册表都写完才转 active；
    # 中途失败会留下可对账的 failed / staging 记录，而不是只靠猜。
    index_batch_id: str = ""
    index_status: str = ""
    indexed_at: int = 0
    index_error: str = ""


class IndexVersionRecord(BaseModel):
    """一批候选索引记录；旧 active 在新版本完成前保持可查。"""

    doc_id: str
    batch_id: str
    status: str
    content_hash: str = ""
    pipeline_version: str = ""
    chunk_count: int = 0
    created_at: int = 0
    completed_at: int = 0
    error: str = ""


# 一份文档在页面上那条流水线里的位置。**只前进**（重做会让它退回去）。
DOC_STAGES = ("uploaded", "parsed", "reviewed", "chunked", "indexed")
INDEX_STATUSES = ("", "staging", "active", "failed")
INDEX_VERSION_STATUSES = ("staging", "active", "failed", "superseded")
INDEX_STATUSES = ("", "staging", "active", "failed")


def stage_rank(stage: str) -> int:
    """``stage`` 在流水线里的位置；不认识的值当 ``-1``（比"刚上传"还早）。"""
    try:
        return DOC_STAGES.index(stage)
    except ValueError:
        return -1


def reached(stage: str, target: str) -> bool:
    """这份文档**已经走到** ``target`` 这一步没有（含更靠后的步骤）。"""
    return stage_rank(stage) >= stage_rank(target)


class Document(BaseModel):
    """解析后、切分前的文档单元。"""

    doc_id: str
    source_uri: str
    text: str
    mime: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    images: list[ImageAsset] = Field(default_factory=list)
    # 文本偏移 → 来源位置，见 Position 的说明
    positions: list[Position] = Field(default_factory=list)


class Chunk(BaseModel):
    """索引与检索的最小单元。

    除 ``text`` 外的这些字段都要写进 Milvus 的**显式列**（而不是塞在 JSON 里）——
    显式才能过滤。设计见 docs/milvus-schema-design.md。
    """

    chunk_id: str
    doc_id: str
    text: str
    index: int = 0
    parent_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # 来源类型：docx / html / pdf / xlsx / md / image
    doc_type: str = ""
    # 这块是正文、图片描述，还是表格记录
    modality: str = "text"
    # 内容归属的标题路径，形如 `第一章、概念 -> 1、什么是Agent？`
    heading_path: str = ""
    # 位置派生的层级编号 `1-2-3`；不解析作者写的编号
    heading_numbers: str = ""
    # 1 起；0 = 该格式没有页的概念（docx/html/xlsx）或定位失败
    page: int = 0
    # 1 起；0 = 没有行的概念。含义随 doc_type 变：PDF 页内行号 / xlsx 表内行号 / docx 段落号
    line: int = 0
    # 本块引用到的资产 uri（图片路径）
    assets: list[str] = Field(default_factory=list)
    # sha1(text)[:16]，去重用
    content_hash: str = ""


class Embeddings(BaseModel):
    """一次编码产出的向量表示。

    dense 与 sparse 与输入等长且一一对应。纯稠密实现把 sparse 留空即可。
    sparse 用 {term_id: weight} 表示，正好对上 Milvus 的稀疏向量格式。
    """

    dense: list[list[float]]
    sparse: list[dict[int, float]] | None = None


class ScoredChunk(BaseModel):
    """带分数的检索结果。

    channel 和 rank 是融合的必需输入：RRF 只看排名不看分数，
    且需要知道每条来自哪一路召回才能做加权和调试。

    ``sources`` 是**融合之后**才填的：这条命中被哪几路召回过（按贡献顺序去重）。
    单路检索时它是空的 —— ``channel`` 已经说明来源；融合之后 ``channel`` 变成 ``"rrf"``，
    真实来源只能靠它保留下来，否则「这条是 dense 捞的还是 bm25 捞的」查都查不到。
    """

    chunk: Chunk
    score: float
    channel: str
    rank: int
    sources: list[str] = Field(default_factory=list)


class ChannelStats(BaseModel):
    """单通道召回统计，用于调试「为什么这条没被召回」。"""

    channel: str
    candidates: int
    kept: int
    latency_ms: float | None = None
    error: str | None = None
    # 「只补位」的通道（见 GraphRetriever）与别路重复、被丢掉的候选数。
    # `candidates` 记的是**它自己召回了多少**（含丢掉的），这样「这一路在不在干活」
    # 和「补进来几条新的」两个问题都答得上。
    complement_dropped: int = 0


class RetrievalResult(BaseModel):
    """一次检索的完整产出。"""

    query: str
    chunks: list[ScoredChunk] = Field(default_factory=list)
    stats: list[ChannelStats] = Field(default_factory=list)


class Citation(BaseModel):
    """统一引用条目；语料块与网络结果共用这一种形状。"""

    index: int
    # corpus / web；以后接别的数据源也只扩展类型，不另造引用协议
    source_type: str = "corpus"
    chunk_id: str = ""
    # 人读的来源名；语料通常是相对路径，网络通常是 URL
    source_uri: str = ""
    # 网络来源的稳定定位；语料块留空
    url: str = ""
    title: str = ""
    snippet: str = ""
    # 这一块属于哪个章节（`标题1 -> 标题2`）。答案末尾那段来源清单要靠它 ——
    # 「引用了哪个文档的哪一章」比「引用了哪一块」对人有用。
    heading_path: str = ""


SOURCES_HEADER = "引用来源："


def format_sources(citations: Sequence[Citation]) -> str:
    """答案末尾那段来源清单（**带前导空行**；没有引用时返回空串）。

    三个约定：

    * **同一章节只列一行**，编号并在一起（`[3][7] 文档 · 章节`）—— 一个章节里命中三个块，
      列三行没人看得下去；但编号必须都在，否则正文里的 `[7]` 就成了悬空引用。
    * **编号沿用正文里的 `[n]`**，不重排。重排了正文就对不上。
    * 章节为空时只写文档名（图谱命中的行没有 heading_path）。
    """
    if not citations:
        return ""
    groups: dict[str, list[int]] = {}
    for item in sorted(citations, key=lambda row: row.index):
        if item.source_type == "web":
            label = item.title or item.url or item.source_uri or "网络来源"
            where = f"来自网络 · {label}"
            if item.url and item.url != label:
                where += f" · {item.url}"
        else:
            label = item.title or item.source_uri
            where = f"{label} · {item.heading_path}" if item.heading_path else label
        groups.setdefault(where, []).append(item.index)
    lines = [
        f"{''.join(f'[{number}]' for number in numbers)} {where}"
        for where, numbers in groups.items()
    ]
    return "\n\n" + SOURCES_HEADER + "\n" + "\n".join(lines)


class AssembledContext(BaseModel):
    """交给生成模型的上下文，含去重后的正文与引用表。"""

    text: str
    citations: list[Citation] = Field(default_factory=list)
    dropped: int = 0
    tokens_estimate: int = 0
