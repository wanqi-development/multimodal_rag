"""文档注册表：每篇一行（走到哪一步 + 入库信息），2026-10-09 从 Milvus 搬到数据库。

钉三件事：**状态能一格格往前推**、**补数字不会把状态拽回去**、**列表只读那一列**
（不再逐份翻产物、不算哈希、不碰 Milvus）。
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from rag.api.app import Components, create_app
from rag.config.schema import Settings
from rag.config.settings import Secrets
from rag.contracts import DocRecord, reached
from rag.deployment import apply_deployment_profile
from rag.persistence.documents import SqliteDocumentStore


def _store(tmp_path: Path) -> SqliteDocumentStore:
    store = SqliteDocumentStore(tmp_path / "documents.sqlite3")
    store.setup()
    return store


def _settings(tmp_path: Path) -> Settings:
    settings = Settings()
    settings.deployment.profile = "local-lite"
    settings.deployment.state_dir = str(tmp_path)
    apply_deployment_profile(settings)
    return settings


def test_a_new_document_can_jump_straight_to_any_stage_and_is_then_reported(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)

    store.sync_set_stage("d1", "uploaded", source_uri="a.md")
    assert store.sync_stages() == {"d1": "uploaded"}
    assert reached(store.sync_stages()["d1"], "uploaded") and not reached(
        store.sync_stages()["d1"], "parsed"
    )

    store.sync_set_stage("d2", "chunked")  # 没有这一行也直接建出来（迁移就是这条路）

    assert store.sync_stages() == {"d1": "uploaded", "d2": "chunked"}
    assert store.sync_get("d2") is not None and store.sync_get("d2").stage == "chunked"
    assert store.sync_get("nope") is None


def test_the_content_hash_is_written_from_upload_onwards(tmp_path: Path) -> None:
    """内容哈希从**上传**那一步就该有值（2026-10-10 补）。

    以前只有入库（`upsert`）写它，于是「解析完但还没入库」的文档这一列是空串 —— 而
    `plan_incremental()` 判「这份变了没有」用的正是它：空值会被判成"变了"，批量 ingest
    时白白重跑一遍 OCR（PDF 就是重新过一遍产线，最慢也最贵）。
    """
    store = _store(tmp_path)

    store.sync_set_stage("d1", "uploaded", source_uri="手册-abc.md", content_hash="abc")
    assert store.sync_get("d1").content_hash == "abc"

    store.sync_set_stage("d1", "parsed")  # 推进时不带 hash
    row = store.sync_get("d1")
    assert row.content_hash == "abc" and row.stage == "parsed", "推进不能把已有的哈希抹掉"

    store.sync_set_stage("d1", "chunked", content_hash="def")
    assert store.sync_get("d1").content_hash == "def", "带了新值就更新"


def test_progressing_a_stage_keeps_the_source_uri_that_was_already_there(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.sync_set_stage("d1", "uploaded", source_uri="手册-abc.md")

    store.sync_set_stage("d1", "parsed")  # 推进时不带 source_uri

    row = store.sync_get("d1")
    assert row is not None and row.stage == "parsed" and row.source_uri == "手册-abc.md"


def test_upsert_writes_the_numbers_and_leaves_the_stage_alone_when_it_is_empty(
    tmp_path: Path,
) -> None:
    """`stage=""` = 只补数字。迁移与 `commit_chunks` 都靠这条不互相打架。"""
    store = _store(tmp_path)
    store.sync_set_stage("d1", "chunked", source_uri="a.md")

    store.sync_upsert(
        [
            DocRecord(
                doc_id="d1",
                source_uri="a.md",
                content_hash="hash",
                chunk_count=7,
                ingested_at=123,
                pipeline_version="v1+o1",
                stage="",
            )
        ]
    )

    row = store.sync_get("d1")
    assert row is not None
    assert (row.stage, row.chunk_count, row.content_hash) == ("chunked", 7, "hash")
    assert row.pipeline_version == "v1+o1" and row.ingested_at == 123

    # 带了 stage 就往前推（`commit_chunks` 走这条）
    store.sync_upsert(
        [DocRecord(doc_id="d1", source_uri="a.md", content_hash="hash", stage="indexed")]
    )
    assert store.sync_get("d1").stage == "indexed"


def test_deleting_rows_and_the_whole_table(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.sync_set_stage("d1", "indexed")
    store.sync_set_stage("d2", "parsed")

    assert store.sync_delete(["d1", "d2", ""]) == 2
    assert store.sync_stages() == {}

    store.sync_set_stage("d3", "parsed")
    store.sync_drop()
    assert store.sync_stages() == {}, "删表之后读回来是空的（再写会自己把表建回来）"


def test_the_list_reads_the_stage_column_instead_of_walking_artifacts(tmp_path: Path) -> None:
    """列表页的状态来自注册表那一列 —— 所以**只写库、不写产物**它也该看得见。"""
    settings = _settings(tmp_path)
    app = create_app(settings, Secrets(rag_api_key=None))
    app.state.components = Components(settings, Secrets(rag_api_key=None))
    client = TestClient(app)
    client.post(
        "/documents/upload",
        files=[("files", ("a.md", b"# t\n\nbody\n", "text/markdown"))],
    )
    doc_id = client.get("/documents").json()["documents"][0]["doc_id"]
    assert client.get("/documents").json()["documents"][0]["stages"] == {
        "parsed": False,
        "reviewed": False,
        "chunked": False,
        "indexed": False,
    }, "刚上传：四个灯都还没亮"

    # 直接改表（没有任何产物），列表要跟着变 —— 这就是「列表只读这一列」
    store = SqliteDocumentStore(settings.documents.sqlite_path)
    store.sync_set_stage(doc_id, "indexed")

    row = client.get("/documents").json()["documents"][0]
    assert row["stages"] == {"parsed": True, "reviewed": True, "chunked": True, "indexed": True}
    assert client.get("/documents").json()["indexed_known"] is True
