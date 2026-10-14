"""各层组件的协议定义。

上层只依赖这些 Protocol，不依赖具体实现。新增一路召回通道 =
写一个满足 Retriever 的类 + 在配置里加一行，融合层无需改动。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from rag.contracts.document import (
    AssembledContext,
    Chunk,
    Document,
    Embeddings,
    ScoredChunk,
)


@runtime_checkable
class Parser(Protocol):
    """把原始文档转成纯文本（+ 结构化元数据）。"""

    name: str

    def parse(self, doc: Document) -> list[Document]: ...


@runtime_checkable
class Chunker(Protocol):
    """把文档切成索引单元。"""

    name: str

    def split(self, docs: list[Document]) -> list[Chunk]: ...


@runtime_checkable
class Embedder(Protocol):
    """文本（可选带图）→ 向量。

    ``Embeddings.sparse`` 默认留空（None）：sparse 那条腿由 Milvus 内建 BM25
    在服务端从原文生成，不需要 embedding 模型参与。字段保留是为了支持
    "换成自产 sparse 的模型"这种变体（那时要配一个独立的 sparse 字段，
    不能与 BM25 function 的输出字段共用）。

    ``embed_documents`` 的 ``images`` 是**每条文本对应的图片路径**（没有图就是空列表）：
    多模态 backend（如线上的 qwen3-vl-embedding）会把图文放进同一个内容项、出一个融合向量；
    纯文本 backend **收到图必须报错**，不能静默忽略 —— 那样配置写错了会退化成一个
    看起来正常、实际丢掉了图片信息的向量空间。

    实现用一个 ``supports_images`` 类属性声明自己吃不吃图，装配层（灌库那一步）据此决定
    要不要传 ``images``：纯文本 backend 拿到 ``None``，多模态 backend 才收到真实的图片路径。
    """

    name: str
    supports_images: bool

    @property
    def dim(self) -> int: ...

    async def embed_documents(
        self, texts: list[str], images: list[list[Path]] | None = None
    ) -> Embeddings: ...

    async def embed_query(self, text: str, image: Path | None = None) -> Embeddings:
        """查询侧编码。``image`` 非空时是**图文查询**（以图搜图 / 图文搜）。

        与文档侧同一条立场：**纯文本 backend 收到图必须报错**，不能静默只编文字 ——
        那样「拿一张图去搜」会退化成一个用空文本搜出来的结果，看起来却像成功了。
        """
        ...


@runtime_checkable
class Index(Protocol):
    """向量/稀疏索引的写侧。"""

    name: str

    async def ensure_ready(self) -> None:
        """建表建索引，必须幂等。"""
        ...

    async def upsert(self, chunks: list[Chunk], embeddings: Embeddings) -> int: ...

    async def delete(self, chunk_ids: list[str]) -> int: ...


@runtime_checkable
class Retriever(Protocol):
    """一路召回通道。"""

    name: str

    async def retrieve(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
        image: Path | None = None,
    ) -> list[ScoredChunk]:
        """``image`` 是**以图搜图**那条路：查询本身可以只有图、没有文字。

        用不上的通道（纯词频那种）收到它应当**自己返回空**而不是报错 —— 图文查询里
        词频这一路没有输入，是正常情况，不是配置错误。
        """
        ...


@runtime_checkable
class Reranker(Protocol):
    """对候选做精排。"""

    name: str

    async def rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
        top_n: int,
    ) -> list[ScoredChunk]: ...


@runtime_checkable
class Generator(Protocol):
    """基于装配好的上下文生成带引用的答案。"""

    name: str

    async def generate(self, query: str, context: AssembledContext) -> str: ...
