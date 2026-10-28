"""人工目录：**纯元数据**的组织方式（虚拟目录，最多三层）。

两层用例：`FolderStore` 的规则（三层上限、同级重名、环、删目录的后果）走真文件，
HTTP 那几个端点走 `TestClient` —— 与 `test_api_documents.py` 同一套离线 fixture
（只碰临时目录，不连 Milvus / PG / 模型）。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag.api.app import Components, create_app
from rag.api.folders import FolderError, FolderStore
from rag.config.schema import Settings
from rag.config.settings import Secrets
from rag.deployment import apply_deployment_profile


@pytest.fixture
def store(tmp_path: Path) -> FolderStore:
    return FolderStore(tmp_path / "uploads")


# ---- FolderStore ----


def test_a_folder_survives_a_round_trip(store: FolderStore) -> None:
    root = store.create("Flink")
    child = store.create("第一章", root.id)

    # 换个实例读（就是「重启进程之后还在不在」）
    again = FolderStore(store.root)

    assert [(item.name, item.parent) for item in again.tree()] == [
        ("Flink", None),
        ("第一章", root.id),
    ]
    assert child.parent == root.id


def test_the_fourth_level_is_rejected(store: FolderStore) -> None:
    one = store.create("一")
    two = store.create("二", one.id)
    three = store.create("三", two.id)

    with pytest.raises(FolderError, match="最多 3 层"):
        store.create("四", three.id)

    assert len(store.tree()) == 3


def test_a_same_name_sibling_is_rejected(store: FolderStore) -> None:
    parent = store.create("Flink")
    store.create("第一章", parent.id)

    with pytest.raises(FolderError, match="已经有一个叫"):
        store.create("第一章", parent.id)

    # 不同父下面同名是可以的（两章 FLINK 文档各有一个「测试数据」很正常）
    other = store.create("Kafka")
    assert store.create("第一章", other.id).name == "第一章"


def test_an_empty_or_control_character_name_is_rejected(store: FolderStore) -> None:
    with pytest.raises(FolderError, match="不能是空的"):
        store.create("   ")
    with pytest.raises(FolderError, match="控制字符"):
        store.create("坏\u0007名字")
    with pytest.raises(FolderError, match="最多 60"):
        store.create("x" * 61)


def test_a_folder_cannot_move_into_its_own_child(store: FolderStore) -> None:
    outer = store.create("外")
    inner = store.create("内", outer.id)

    with pytest.raises(FolderError, match="搬到它自己"):
        store.update(outer.id, parent=inner.id, move=True)

    # 搬到顶层没问题
    moved = store.update(outer.id, parent=None, move=True)
    assert moved.parent is None


def test_a_move_that_would_exceed_three_levels_is_rejected(store: FolderStore) -> None:
    one = store.create("一")
    two = store.create("二", one.id)
    deep = store.create("深")
    deep_one = store.create("深一", deep.id)
    store.create("深二", deep_one.id)

    with pytest.raises(FolderError, match="最多 3 层"):
        store.update(deep.id, parent=two.id, move=True)

    assert store.update(deep.id, name="更深", move=False).name == "更深", "只改名不受影响"


def test_deleting_a_folder_with_children_is_rejected(store: FolderStore) -> None:
    outer = store.create("外")
    store.create("内", outer.id)

    with pytest.raises(FolderError, match="还有子目录"):
        store.delete(outer.id)

    assert len(store.tree()) == 2


def test_deleting_a_folder_sends_its_docs_back_to_unfiled(store: FolderStore) -> None:
    folder = store.create("Flink")
    store.assign(["doc1", "doc2"], folder.id)

    assert store.delete(folder.id) == 2, "返回「被移回未分类」的份数"

    assert store.tree() == []
    assert store.members() == {}, "归属跟着目录一起没了 —— 文档本身一个都没动"


def test_assign_moves_documents_between_folders(store: FolderStore) -> None:
    left = store.create("左")
    right = store.create("右")

    assert store.assign(["doc1", "doc2"], left.id) == 2
    assert store.members() == {"doc1": left.id, "doc2": left.id}
    # 已经在那儿的，再放一次不算改动
    assert store.assign(["doc1"], left.id) == 0
    assert store.assign(["doc1"], right.id) == 1
    assert store.assign(["doc1", "doc2"], None) == 2, "移回未分类"
    assert store.members() == {}


def test_assigning_to_an_unknown_folder_is_rejected(store: FolderStore) -> None:
    with pytest.raises(FolderError, match="没有这个目录"):
        store.assign(["doc1"], "f-nope")


def test_a_broken_file_means_no_folders(store: FolderStore) -> None:
    """附属信息坏了不该让整页打不开 —— 退回「一个目录都没有」。"""
    store.create("Flink")
    (store.root / "_folders.json").write_text("{ 这不是 json", encoding="utf-8")

    assert FolderStore(store.root).tree() == []
    assert FolderStore(store.root).members() == {}


# ---- HTTP ----


def _settings(tmp_path: Path) -> Settings:
    settings = Settings()
    settings.deployment.profile = "local-lite"
    settings.deployment.state_dir = str(tmp_path)
    apply_deployment_profile(settings)
    return settings


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    settings = _settings(tmp_path)
    app = create_app(settings, Secrets(rag_api_key=None))
    app.state.components = Components(settings, Secrets(rag_api_key=None))
    return TestClient(app)


def _upload(client: TestClient, name: str = "手册.md") -> str:
    body = client.post(
        "/documents/upload",
        files=[("files", (name, "# 第一章\n\n正文\n".encode(), "text/markdown"))],
    ).json()
    return body["uploads"][0]["doc_id"]


def test_the_list_carries_the_tree_and_each_documents_folder(client: TestClient) -> None:
    doc_id = _upload(client)

    body = client.get("/documents").json()

    assert body["folders"] == []
    assert body["documents"][0]["folder"] == "", "没归类就是空串（页面画进「未分类」）"

    folder = client.post("/documents/folders", json={"name": "Flink"}).json()
    assert client.put(
        "/documents/folders/members", json={"doc_ids": [doc_id], "folder_id": folder["id"]}
    ).json() == {"moved": 1}

    body = client.get("/documents").json()
    assert body["folders"] == [folder]
    assert body["documents"][0]["folder"] == folder["id"]


def test_folders_can_be_created_renamed_and_removed(client: TestClient) -> None:
    parent = client.post("/documents/folders", json={"name": "Flink"}).json()
    child = client.post(
        "/documents/folders", json={"name": "第一章", "parent": parent["id"]}
    ).json()

    renamed = client.patch(f"/documents/folders/{child['id']}", json={"name": "第一章 概述"})
    assert renamed.status_code == 200 and renamed.json()["parent"] == parent["id"], (
        "只改名：父目录不该被顺手搬到顶层"
    )
    moved = client.patch(f"/documents/folders/{child['id']}", json={"parent": None})
    assert moved.json()["parent"] is None

    assert client.delete(f"/documents/folders/{parent['id']}").status_code == 200
    assert [item["id"] for item in client.get("/documents").json()["folders"]] == [child["id"]]


def test_the_fourth_level_over_http_is_a_400(client: TestClient) -> None:
    level = None
    for index in range(3):
        level = client.post(
            "/documents/folders", json={"name": f"第{index + 1}层", "parent": level}
        ).json()["id"]

    res = client.post("/documents/folders", json={"name": "第4层", "parent": level})

    assert res.status_code == 400 and "最多 3 层" in res.json()["error"]["message"]


def test_deleting_a_folder_with_children_is_a_409(client: TestClient) -> None:
    outer = client.post("/documents/folders", json={"name": "外"}).json()
    client.post("/documents/folders", json={"name": "内", "parent": outer["id"]})

    res = client.delete(f"/documents/folders/{outer['id']}")

    assert res.status_code == 409 and "子目录" in res.json()["error"]["message"]


def test_assigning_an_unknown_document_or_folder_is_a_404(client: TestClient) -> None:
    assert (
        client.put(
            "/documents/folders/members", json={"doc_ids": ["nope"], "folder_id": None}
        ).status_code
        == 404
    )

    doc_id = _upload(client)
    assert (
        client.put(
            "/documents/folders/members", json={"doc_ids": [doc_id], "folder_id": "f-nope"}
        ).status_code
        == 404
    )


def test_deleting_a_document_forgets_its_folder(client: TestClient, tmp_path: Path) -> None:
    doc_id = _upload(client)
    folder = client.post("/documents/folders", json={"name": "Flink"}).json()
    client.put("/documents/folders/members", json={"doc_ids": [doc_id], "folder_id": folder["id"]})

    assert client.delete(f"/documents/{doc_id}?purge=false").status_code == 200

    # 目录还在，但台账里不留那份已经删掉的文档
    assert [item["id"] for item in client.get("/documents").json()["folders"]] == [folder["id"]]
    assert FolderStore(tmp_path / "uploads").members() == {}
