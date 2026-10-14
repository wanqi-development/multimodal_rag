"""上传件台账（2026-10-09 从 `_index.json` 挪进数据库）。

钉三件事：**每份上传各写各的行**（多 worker 不再互相覆盖）、**导入老 JSON 只补空位**、
以及最要紧的那条不变量 —— **`doc_id` 与老算法同值**（否则搬完就对不上产物与已入库的块）。
"""

from __future__ import annotations

import json
from pathlib import Path

from rag.api.uploads import BlobUploadStore, doc_id_for_name, import_index_file, safe_name
from rag.ingest.loaders.base import doc_id_for
from rag.persistence.uploads import SqliteUploadMetaStore
from rag.storage import LocalBlobStore


def _store(tmp_path: Path) -> SqliteUploadMetaStore:
    store = SqliteUploadMetaStore(tmp_path / "documents.sqlite3")
    store.setup()
    return store


def _row(doc_id: str, **over: object) -> dict:
    base: dict = {
        "doc_id": doc_id,
        "name": f"手册-{doc_id}.md",
        "display_name": "手册.md",
        "sha1": doc_id,
        "size": 12,
        "created_at": 100,
        "attachments": ["a_files/img01.png"],
    }
    base.update(over)
    return base


def test_each_upload_writes_its_own_row_and_reads_back(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert([_row("d1"), _row("d2", display_name="另一份.md")])

    assert sorted(store.all()) == ["d1", "d2"]
    assert store.get("d1")["attachments"] == ["a_files/img01.png"], (
        "数组按 JSON 文本存、读回来还是列表"
    )
    assert [row["doc_id"] for row in store.by_display("手册.md")] == ["d1"]
    assert store.delete(["d1"]) == 1 and store.get("d1") is None


def test_the_list_is_newest_first(tmp_path: Path) -> None:
    """列表按**导入时间倒序**（使用者 2026-10-10 报的「导入顺序和展示顺序不一致」）。

    以前按落盘名排 —— 而落盘名是 `<真名>-<内容哈希><后缀>`，等于**按真名字典序**，
    跟导入先后毫无关系。同一秒的按落盘名（稳定）；老 `_index.json` 补进来、
    没有时间戳的那些（`created_at=0`）落在最后。
    """
    meta = _store(tmp_path)
    meta.upsert(
        [
            _row("old", name="a-old.md", created_at=100),
            _row("legacy", name="m-legacy.md", created_at=0),
            _row("tie", name="b-tie.md", created_at=200),
            _row("new", name="z-new.md", created_at=200),
        ]
    )
    store = BlobUploadStore(
        LocalBlobStore(tmp_path / "up"), meta, root=tmp_path / "up", max_bytes=1000
    )

    assert [item.doc_id for item in store.list()] == ["tie", "new", "old", "legacy"]
    assert store.list()[0].created_at == 200, "最上面那条就是最新导入的"


def test_upsert_overwrites_but_only_if_absent_never_does(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert([_row("d1", display_name="旧名.md")])

    store.upsert([_row("d1", display_name="新名.md")])
    assert store.get("d1")["display_name"] == "新名.md", "正常写台账是覆盖"

    store.upsert([_row("d1", display_name="导入的名.md")], only_if_absent=True)
    assert store.get("d1")["display_name"] == "新名.md", "导入老 JSON 只补空位，绝不覆盖"


def test_the_doc_id_is_exactly_what_the_old_algorithm_produced(tmp_path: Path) -> None:
    """**这条不变量最关键**：两个后端（本地盘 / 对象存储）算出的 doc_id 必须一模一样，
    否则产物与已入库的块全对不上。"""
    name = "第一章 Apache Flink 概述-27a5b9410abaea5a.pdf"

    assert doc_id_for_name(name) == doc_id_for(tmp_path / name, tmp_path)


def test_importing_the_legacy_index_only_fills_the_gaps(tmp_path: Path) -> None:
    """老 `_index.json` 里没有落盘名 —— 能扫到文件就用文件（还带真实大小），
    扫不到才按「真名 + sha1」复原。"""
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    on_disk = "手册-aaaaaaaaaaaaaaaa.md"
    (uploads / on_disk).write_bytes(b"# t\n")
    legacy = {
        doc_id_for_name(on_disk): {
            "display_name": "手册.md",
            "sha1": "aaaaaaaaaaaaaaaa",
            "created_at": 7,
            "attachments": [],
        },
        "bbbbbbbbbbbbbbbb": {"display_name": "丢了.pdf", "sha1": "cccccccccccccccc"},
    }
    (uploads / "_index.json").write_text(json.dumps(legacy), encoding="utf-8")
    store = _store(tmp_path)

    assert import_index_file(store, uploads) == 2

    row = store.get(doc_id_for_name(on_disk))
    assert row is not None
    assert (row["name"], row["display_name"], row["size"]) == (on_disk, "手册.md", 4)
    gone = store.get("bbbbbbbbbbbbbbbb")
    assert gone is not None and gone["name"] == "丢了-cccccccccccccccc.pdf" and gone["size"] == 0
    assert import_index_file(store, uploads) == 2 and len(store.all()) == 2, "再导一次也不重复"


def test_safe_name_keeps_the_real_name_but_strips_path_traversal() -> None:
    """真名原样留着（中文、括号、空格），只挡「跑到上传目录外面」这一类。"""
    assert safe_name("../../etc/passwd") == "passwd"
    assert safe_name("第一章（概述） v2.pdf") == "第一章（概述） v2.pdf"
