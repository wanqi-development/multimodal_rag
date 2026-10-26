"""索引 manifest 对账：只修“数据完成但状态落后”，不改坏数据。"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from typing import Any

from rag.contracts import Chunk, DocRecord, IndexVersionRecord
from rag.indexing.reconcile import prune_inactive_versions, reconcile_indexes
from rag.persistence.documents import SqliteDocumentStore


class _Store:
    def __init__(self, payload: dict[str, Any] | None) -> None:
        self.payload = payload

    def read_payload(self, doc_id: str, name: str) -> dict[str, Any] | None:
        assert (doc_id, name) == ("d1", "chunks.json")
        return self.payload


class _Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

    async def get_doc_chunks(self, doc_id: str) -> list[Chunk]:
        return self.chunks


class _Registry:
    def __init__(self, record: DocRecord) -> None:
        self.record = record
        self.updated: list[DocRecord] = []

    async def all(self) -> dict[str, DocRecord]:
        return {"d1": self.record}

    async def upsert(self, records) -> int:
        items = list(records)
        self.updated.extend(items)
        return len(items)


class _VersionRegistry(_Registry):
    def __init__(self, versions: list[IndexVersionRecord], active: dict[str, str]) -> None:
        super().__init__(_record())
        self.version_rows = versions
        self.active = active
        self.deleted: list[str] = []

    async def versions(self, doc_id: str | None = None) -> list[IndexVersionRecord]:
        return self.version_rows

    async def active_batches(self) -> dict[str, str]:
        return self.active

    async def delete_versions(self, batch_id: str) -> int:
        self.deleted.append(batch_id)
        return 1


class _BatchIndex:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def delete_batch(self, batch_id: str) -> None:
        self.deleted.append(batch_id)


def _payload() -> dict[str, Any]:
    return {
        "file_hash": "file-1",
        "pipeline_version": "pipe",
        "outline_hash": "outline",
        "chunks": [
            {"chunk_id": "d1:0000", "content_hash": "chunk-a"},
            {"chunk_id": "d1:0001", "content_hash": "chunk-b"},
        ],
    }


def _record(**over: Any) -> DocRecord:
    values = {
        "doc_id": "d1",
        "source_uri": "a.pdf",
        "content_hash": "file-1",
        "chunk_count": 2,
        "pipeline_version": "pipe+outline",
        "stage": "indexed",
        "index_batch_id": "batch-1",
        "index_status": "active",
    }
    values.update(over)
    return DocRecord(**values)


def _chunks() -> list[Chunk]:
    return [
        Chunk(chunk_id="d1:0000", doc_id="d1", text="甲", content_hash="chunk-a"),
        Chunk(chunk_id="d1:0001", doc_id="d1", text="乙", content_hash="chunk-b"),
    ]


def test_a_complete_active_index_has_no_issue() -> None:
    report = asyncio.run(
        reconcile_indexes(
            None,
            registry=_Registry(_record()),
            index=_Index(_chunks()),
            store=_Store(_payload()),
        )
    )

    assert report.checked == 1 and report.issues == [] and report.repaired == 0


def test_missing_milvus_chunks_are_reported_and_not_repaired() -> None:
    registry = _Registry(_record())
    report = asyncio.run(
        reconcile_indexes(
            None,
            registry=registry,
            index=_Index(_chunks()[:1]),
            store=_Store(_payload()),
            repair=True,
        )
    )

    assert [issue.code for issue in report.issues] == ["chunks_mismatch"]
    assert report.repaired == 0 and registry.updated == []


def test_staging_pointer_without_version_row_is_treated_as_legacy() -> None:
    registry = _Registry(_record(index_status="staging", index_batch_id="batch-2"))
    report = asyncio.run(
        reconcile_indexes(
            None,
            registry=registry,
            index=_Index(_chunks()),
            store=_Store(_payload()),
            repair=True,
        )
    )

    assert report.issues == []
    assert report.repaired == 0
    assert registry.updated == [], "没有版本表证据时不能把 staging 指针当成可修复版本"


def test_missing_artifact_is_reported_even_when_milvus_has_rows() -> None:
    registry = _Registry(_record())
    report = asyncio.run(
        reconcile_indexes(
            None,
            registry=registry,
            index=_Index(_chunks()),
            store=_Store(None),
            repair=True,
        )
    )

    assert [issue.code for issue in report.issues] == ["artifact_missing"]
    assert report.repaired == 0 and registry.updated == []


def test_registry_tracks_staging_active_and_failed_batches(tmp_path) -> None:
    store = SqliteDocumentStore(tmp_path / "documents.sqlite3")
    store.setup()
    store.sync_upsert(
        [
            DocRecord(
                doc_id="d1",
                source_uri="a.pdf",
                content_hash="file-1",
                chunk_count=2,
                pipeline_version="pipe",
                stage="chunked",
            )
        ]
    )

    candidate = _record(stage="chunked", index_batch_id="", index_status="")
    store.sync_begin_index("batch-1", [candidate])
    before = store.sync_get("d1")
    assert before is not None and before.index_batch_id == ""
    assert store.sync_versions("d1")[0].status == "staging"

    store.sync_complete_index("batch-1", [candidate])
    assert store.sync_get("d1").index_status == "active"
    assert store.sync_versions("d1")[0].status == "active"
    assert store.sync_active_batches() == {"d1": "batch-1"}

    store.sync_begin_index("batch-2", [candidate])
    store.sync_fail_index("batch-2", ["d1"], "写入中断")
    failed_active = store.sync_get("d1")
    assert failed_active is not None and failed_active.index_status == "active"
    assert failed_active.index_batch_id == "batch-1"
    failed_version = store.sync_versions("d1")[0]
    assert failed_version.status == "failed" and "写入中断" in failed_version.error


def test_an_old_registry_table_is_migrated_in_place(tmp_path) -> None:
    path = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute(
        "create table document_registry ("
        "doc_id text primary key, source_uri text not null default '',"
        "content_hash text not null default '', chunk_count integer not null default 0,"
        "ingested_at integer not null default 0, pipeline_version text not null default '',"
        "stage text not null default 'uploaded', updated_at text not null default '')"
    )
    conn.execute(
        "insert into document_registry "
        "(doc_id, source_uri, stage) values ('d1', 'a.pdf', 'indexed')"
    )
    conn.commit()
    conn.close()

    record = SqliteDocumentStore(path).sync_get("d1")

    assert record is not None
    assert record.index_status == "" and record.index_batch_id == ""


def test_an_active_pointer_without_version_row_falls_back_to_legacy(tmp_path) -> None:
    """升级前的 manifest active 只能算 legacy；不能把旧 collection 的文档排除掉。"""
    store = SqliteDocumentStore(tmp_path / "documents.sqlite3")
    store.setup()
    store.sync_upsert([_record(index_batch_id="old-manifest", index_status="active")])

    assert store.sync_active_batches() == {}


def test_prune_deletes_only_old_inactive_batches() -> None:
    registry = _VersionRegistry(
        [
            IndexVersionRecord(
                doc_id="d1",
                batch_id="active",
                status="active",
                created_at=1,
            ),
            IndexVersionRecord(
                doc_id="d1",
                batch_id="old",
                status="superseded",
                created_at=1,
            ),
            IndexVersionRecord(
                doc_id="d1",
                batch_id="new-old",
                status="failed",
                created_at=int(time.time()),
            ),
        ],
        active={"d1": "active"},
    )
    index = _BatchIndex()

    deleted = asyncio.run(
        prune_inactive_versions(
            None,
            registry=registry,
            index=index,
            older_than_seconds=3600,
        )
    )

    assert deleted == ["old"]
    assert index.deleted == ["old"] and registry.deleted == ["old"]
