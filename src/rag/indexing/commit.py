"""第三段：把切分产物写进 Milvus（+ 注册表）。**CLI 与 IndexGraph 共用这一段。**

以前这段逻辑写在 `cli/main.py` 的一个闭包里，只能被 CLI 用。抽出来是因为编排层要复用它 ——
「先删旧块再写、写块与写注册表要成对、缺产物的文档不许静默跳过」这几条规矩只该有一份实现。

三条规矩的来由（都在 §8.2 记过）：

1. **先按 doc_id 删旧块，再 upsert**：`chunk_id = doc_id:序号`，文档改短后多出来的旧序号
   不会被覆盖，会变成「检索得到、正文却已不存在」的孤儿；
2. **缺切分产物就整批不写**：只写一部分等于库里新旧混杂，比什么都不写更难查；
3. **删块与删注册行有顺序**：先删块、再删注册行。反过来的话中途失败会留下「有块没注册」的
   孤儿；现在这样只会留下「注册在、块没了」，下次跑会重做 —— 是安全的方向。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field

from rag.config.schema import Settings
from rag.contracts import Chunk, DocRecord, Embedder
from rag.indexing.milvus import MilvusIndex
from rag.ingest.assets import build_asset_store, image_paths_for
from rag.ingest.incremental import file_hash, pipeline_version
from rag.ingest.loaders.base import doc_id_for, rel_uri
from rag.ingest.stages import ArtifactStore


class CommitResult(BaseModel):
    """写库这一段的产出。``missing`` 非空时**什么都没写**（见模块说明第 2 条）。"""

    written: int = 0
    # 其中有多少条是图片描述块（CLI 的收尾提示要这个数）
    described: int = 0
    registered: list[DocRecord] = Field(default_factory=list)
    # 缺切分产物的文档（source_uri）—— 调用方要把它打出来并给出下一步命令
    missing: list[str] = Field(default_factory=list)


def index_fingerprint(version: str, store: ArtifactStore, path: Path, root: Path) -> str:
    """注册表里记的指纹 = 切法指纹 + **这一份的标题清单指纹**。

    清单是人工核对的：改一份清单只该重灌那一份，所以它必须进「这份文档变了没有」的比较。
    """
    return f"{version}+{store.outline_fingerprint(doc_id_for(path, root))}"


async def commit_chunks(
    settings: Settings,
    store: ArtifactStore,
    jobs: Sequence[tuple[Path, Path]],
    *,
    embedder: Embedder | None = None,
    index: MilvusIndex | None = None,
    registry: object | None = None,
) -> CommitResult:
    """``jobs`` 是 ``(文件路径, 语料根)`` 的列表（doc_id 与 source_uri 相对根算）。

    ``embedder`` / ``index`` / ``registry`` 可以注入（测试用）；不传就按配置建。
    """
    version = pipeline_version(settings)
    chunks: list[Chunk] = []
    records: list[DocRecord] = []
    missing: list[str] = []
    for path, root in jobs:
        doc_id = doc_id_for(path, root)
        cached = store.load_chunks(
            doc_id,
            source_hash=file_hash(path),
            pipeline=version,
            outline_hash=store.outline_fingerprint(doc_id),
        )
        if cached is None:
            missing.append(rel_uri(path, root))
            continue
        chunks.extend(cached)
        records.append(
            DocRecord(
                doc_id=doc_id,
                source_uri=rel_uri(path, root),
                content_hash=file_hash(path),
                chunk_count=len(cached),
                ingested_at=int(time.time()),
                # 与计划阶段用的是同一个串，否则下次跑会永远判成「变更」
                pipeline_version=index_fingerprint(version, store, path, root),
                # 注册表里那一列状态：走到这一步了
                stage="indexed",
            )
        )
    if missing:
        # 只写一部分会让库里新旧混杂 —— 整批不写，让调用方去补齐产物（见模块说明第 2 条）
        return CommitResult(missing=missing)

    from rag.indexing import build_index, build_registry

    index = index or build_index(settings)
    registry = registry or build_registry(settings)
    doc_ids = sorted({record.doc_id for record in records})
    batch_id = uuid4().hex[:12]
    # 先落 staging：若进程在 Milvus 写入中途退出，DB 会留下可对账的批次，而不是
    # 一份仍在宣称 active、实际块却已删/写了一半的假状态。
    await registry.begin_index(batch_id, records)
    try:
        await index.ensure_ready()
        if not chunks:
            await registry.complete_index(batch_id, records)
            return CommitResult(registered=records)

        embedder = embedder or _build_embedder(settings)
        # 两条路在这一行分岔：纯文本 backend 传 None；多模态 backend 给图片块附上图片本体，
        # 「图 + 描述」编成一个融合向量（正文块照旧只有文字）。
        images: list[list[Path]] | None = None
        if getattr(embedder, "supports_images", False):
            images, _ = image_paths_for(chunks, build_asset_store(settings))
        vectors = await embedder.embed_documents([chunk.text for chunk in chunks], images)

        if hasattr(index, "upsert_batch"):
            # 双版本：只写新批次，旧批次保留到活动指针切换之后。
            written = await index.upsert_batch(chunks, vectors, batch_id=batch_id)
        else:
            # 非版本化后端 / 测试替身：保留原来的覆盖语义。
            await index.delete_docs(doc_ids)
            written = await index.upsert(chunks, vectors)
        # 所有 Milvus 写成功后才把整批转 active；这一步在注册表一个事务里完成。
        await registry.complete_index(batch_id, records)
        return CommitResult(
            written=written,
            described=sum(1 for chunk in chunks if chunk.modality == "image"),
            registered=records,
        )
    except Exception as exc:
        with suppress(Exception):
            await registry.fail_index(
                batch_id,
                doc_ids,
                f"{type(exc).__name__}: {exc}",
            )
        raise


async def drop_docs(
    settings: Settings,
    store: ArtifactStore,
    deleted: Sequence[str],
    *,
    index: MilvusIndex | None = None,
    registry: object | None = None,
) -> int:
    """清掉语料里已经消失的文档：**先删块、再删注册行、最后清产物**。

    顺序不能反（见模块说明第 3 条）。单独成一个函数是因为它跟「写」是两件事：
    编排层按文件铺开写库时，删除这件事只该由调用方做一次，不该混在写里。
    """
    if not deleted:
        return 0
    from rag.indexing import build_index, build_registry

    index = index or build_index(settings)
    registry = registry or build_registry(settings)
    await index.ensure_ready()
    await index.delete_docs(list(deleted))
    await registry.delete(list(deleted))
    # 语料里删掉的文档，它的中间产物也跟着清 —— 否则 data/stages 只增不减
    for doc_id in deleted:
        store.drop(doc_id)
    return len(deleted)


def _build_embedder(settings: Settings) -> Embedder:
    # 延迟 import：providers 是个重包（一 import 就带出 fastembed / torch），
    # 而这个模块在「只算指纹、不写库」的路径上也会被 import。
    from rag.providers import build_embedder

    return build_embedder(settings)
