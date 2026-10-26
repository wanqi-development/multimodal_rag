"""文档索引对账：比较注册表 manifest、切分产物与 Milvus 实际块。"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from rag.contracts import DocRecord


class ReconcileIssue(BaseModel):
    doc_id: str
    source_uri: str = ""
    code: str
    message: str
    repaired: bool = False


class ReconcileReport(BaseModel):
    checked: int = 0
    issues: list[ReconcileIssue] = Field(default_factory=list)
    repaired: int = 0

    @property
    def ok(self) -> bool:
        return not self.issues


async def prune_inactive_versions(
    settings: Any,
    *,
    registry: Any,
    index: Any,
    older_than_seconds: int = 86400,
    dry_run: bool = False,
) -> list[str]:
    """删除不再承载活动批次的旧版本；活动 batch 永不在这里删。"""
    versions = await registry.versions()
    active = set((await registry.active_batches()).values())
    now = int(time.time())
    groups: dict[str, list[Any]] = {}
    for version in versions:
        groups.setdefault(version.batch_id, []).append(version)

    eligible: list[str] = []
    for batch_id, rows in groups.items():
        if not batch_id or batch_id in active:
            continue
        if any(row.status == "active" for row in rows):
            continue
        newest = max(max(row.created_at, row.completed_at) for row in rows)
        if now - newest >= older_than_seconds:
            eligible.append(batch_id)

    if not dry_run:
        for batch_id in eligible:
            await index.delete_batch(batch_id)
            await registry.delete_versions(batch_id)
    return sorted(eligible)


async def reconcile_indexes(
    settings: Any,
    *,
    registry: Any = None,
    index: Any = None,
    store: Any = None,
    repair: bool = False,
    doc_ids: set[str] | None = None,
) -> ReconcileReport:
    """对账；默认只读。`repair=True` 只修 manifest 状态，不重写向量。

    “可修”的条件是：切分产物完整、文件/切法指纹一致、Milvus 里的 chunk_id 与内容哈希
    全部一致。只有这种“数据其实已完成、只是 DB 状态落后”的情况才回填 active；
    真正的块缺失或内容不一致只报告，不做静默修复。
    """
    if registry is None or index is None or store is None:
        from rag.indexing import build_index, build_registry
        from rag.ingest.stages import build_artifact_store

        registry = registry or build_registry(settings)
        index = index or build_index(settings)
        store = store or build_artifact_store(settings)

    records = await registry.all()
    active_batches = (
        await registry.active_batches()
        if hasattr(registry, "active_batches")
        else {
            doc_id: record.index_batch_id
            for doc_id, record in records.items()
            if record.index_batch_id and record.index_status == "active"
        }
    )
    if doc_ids is not None:
        records = {doc_id: record for doc_id, record in records.items() if doc_id in doc_ids}

    report = ReconcileReport()
    repairs: list[DocRecord] = []
    for doc_id, record in sorted(records.items()):
        if record.stage != "indexed" and not record.index_status:
            continue
        report.checked += 1

        payload = store.read_payload(doc_id, "chunks.json")
        if payload is None:
            report.issues.append(
                ReconcileIssue(
                    doc_id=doc_id,
                    source_uri=record.source_uri,
                    code="artifact_missing",
                    message="注册表说已索引，但切分产物缺失或损坏",
                )
            )
            continue

        expected = payload.get("chunks") or []
        expected_ids = {
            str(chunk.get("chunk_id") or ""): str(chunk.get("content_hash") or "")
            for chunk in expected
        }
        expected_count = len(expected)
        expected_pipeline = (
            f"{payload.get('pipeline_version') or ''}+{payload.get('outline_hash') or ''}"
        )

        try:
            actual_chunks = await index.get_doc_chunks(doc_id)
        except Exception as exc:  # noqa: BLE001 —— 对账报告要比整批崩掉有用
            report.issues.append(
                ReconcileIssue(
                    doc_id=doc_id,
                    source_uri=record.source_uri,
                    code="milvus_query_failed",
                    message=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        actual_ids = {chunk.chunk_id: chunk.content_hash for chunk in actual_chunks}

        problems: list[ReconcileIssue] = []
        if payload.get("file_hash") != record.content_hash:
            problems.append(
                ReconcileIssue(
                    doc_id=doc_id,
                    source_uri=record.source_uri,
                    code="file_hash_mismatch",
                    message="切分产物的 file_hash 与注册表不一致",
                )
            )
        if expected_pipeline != record.pipeline_version:
            problems.append(
                ReconcileIssue(
                    doc_id=doc_id,
                    source_uri=record.source_uri,
                    code="pipeline_mismatch",
                    message="切分产物的 pipeline/outline 指纹与注册表不一致",
                )
            )
        if record.chunk_count != expected_count:
            problems.append(
                ReconcileIssue(
                    doc_id=doc_id,
                    source_uri=record.source_uri,
                    code="chunk_count_mismatch",
                    message=f"注册表 {record.chunk_count} 块，产物 {expected_count} 块",
                )
            )
        if actual_ids != expected_ids:
            missing = sorted(set(expected_ids) - set(actual_ids))
            extra = sorted(set(actual_ids) - set(expected_ids))
            changed = sorted(
                chunk_id
                for chunk_id in set(expected_ids) & set(actual_ids)
                if expected_ids[chunk_id] != actual_ids[chunk_id]
            )
            problems.append(
                ReconcileIssue(
                    doc_id=doc_id,
                    source_uri=record.source_uri,
                    code="chunks_mismatch",
                    message=(
                        f"Milvus 与产物不一致：缺 {len(missing)} / 多 {len(extra)} / "
                        f"内容变 {len(changed)}"
                    ),
                )
            )

        versioned = active_batches.get(doc_id) == record.index_batch_id
        if record.stage != "indexed":
            problems.append(
                ReconcileIssue(
                    doc_id=doc_id,
                    source_uri=record.source_uri,
                    code="stage_mismatch",
                    message=f"索引状态存在，但文档 stage={record.stage!r}",
                )
            )
        if versioned and record.index_status != "active":
            problems.append(
                ReconcileIssue(
                    doc_id=doc_id,
                    source_uri=record.source_uri,
                    code="status_not_active",
                    message=f"index_status={record.index_status or 'legacy'}",
                )
            )
        if not versioned and record.index_status == "active":
            problems.append(
                ReconcileIssue(
                    doc_id=doc_id,
                    source_uri=record.source_uri,
                    code="missing_batch_id",
                    message="index_status=active 但缺少活动 batch",
                )
            )

        if problems:
            report.issues.extend(problems)
            if (
                repair
                and versioned
                and not any(
                    problem.code
                    in {
                        "artifact_missing",
                        "file_hash_mismatch",
                        "pipeline_mismatch",
                        "chunk_count_mismatch",
                        "chunks_mismatch",
                        "milvus_query_failed",
                    }
                    for problem in problems
                )
            ):
                repairs.append(_active_record(record))
                report.repaired += 1
                for problem in problems:
                    problem.repaired = True
        elif versioned and record.index_status != "active" and repair:
            repairs.append(_active_record(record))
            report.repaired += 1

    if repairs:
        await registry.upsert(repairs)
    return report


def _active_record(record: DocRecord) -> DocRecord:
    return record.model_copy(
        update={
            "stage": "indexed",
            "index_batch_id": record.index_batch_id,
            "index_status": "active",
            "indexed_at": record.indexed_at or int(time.time()),
            "index_error": "",
        }
    )
