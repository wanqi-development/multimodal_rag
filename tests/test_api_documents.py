"""文档上传 → 解析 → 预览（页面五步里的前两步）。

真跑 stage（`.md` 走文本 loader，离线），但**只用临时目录**：不连 Milvus、不连 PG、
不调模型。解析那一步直接调 `run_parse_job`，不走 HTTP —— `TestClient` 不进 lifespan 时
每个请求一个 portal，后台任务会被连带取消（见下面注释）。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag.api.app import Components, create_app
from rag.config.schema import Settings
from rag.config.settings import Secrets


def _settings(tmp_path: Path) -> Settings:
    settings = Settings()
    settings.ingest.uploads.dir = str(tmp_path / "uploads")
    settings.ingest.stages.dir = str(tmp_path / "stages")
    settings.ingest.assets.dir = str(tmp_path / "assets")
    # 文档注册表（每篇走到哪一步）2026-10-09 起落库：测试也走真的 SQLite，
    # 但**必须指到临时目录** —— 默认路径是仓库里的 `data/documents.sqlite3`
    settings.documents.backend = "sqlite"
    settings.documents.sqlite_path = str(tmp_path / "documents.sqlite3")
    # 这些用例验的是「清单本身」，不是自动入库那一步：**把自动入库关掉**，
    # 否则每次提交核对都会去排一个真任务（那一步要建 embedder / 问 VLM）。
    # 自动那条单独有用例（`test_confirming_the_outline_starts_the_index_job` 等）。
    settings.api.index_after_review = False
    return settings


@pytest.fixture
def components(tmp_path: Path) -> Components:
    return Components(_settings(tmp_path), Secrets(rag_api_key=None))


@pytest.fixture
def client(components: Components) -> TestClient:
    app = create_app(components.settings, Secrets(rag_api_key=None))
    app.state.components = components
    # **不进 `with`**：那会跑 lifespan（连 Redis / PG / Milvus）。这几个端点只用本地
    # 文件与 stage 产物，离线就该能测。
    return TestClient(app)


def _upload(client: TestClient, name: str, text: str = "# 第一章\n\n正文\n"):
    return client.post(
        "/documents/upload", files=[("files", (name, text.encode("utf-8"), "text/markdown"))]
    )


def _stored(tmp_path: Path) -> list[Path]:
    """上传目录里**真正的文档**。`_index.json`（真名 / 来源）是附属信息，不算一份文档。"""
    return sorted(p for p in (tmp_path / "uploads").iterdir() if p.name != "_index.json")


def test_upload_lands_on_disk_and_lists_with_stage_flags(
    client: TestClient, tmp_path: Path
) -> None:
    body = _upload(client, "手册.md").json()

    assert len(body["uploads"]) == 1 and not body["skipped"]
    item = body["uploads"][0]
    assert item["display_name"] == "手册.md", "真名原样留着（页面显示这个）"
    assert item["name"].startswith("手册-") and item["name"].endswith(".md"), "落盘名带内容哈希"
    assert item["size"] > 0 and item["replaced"] is False
    assert (tmp_path / "uploads" / item["name"]).is_file(), "上传件落在配置的目录里"

    listed = client.get("/documents").json()["documents"]
    assert [row["doc_id"] for row in listed] == [item["doc_id"]]
    assert listed[0]["display_name"] == "手册.md" and listed[0]["name"] == item["name"]
    assert listed[0]["stages"] == {
        "parsed": False,
        "reviewed": False,
        "chunked": False,
        "indexed": False,
    }, "还没解析：四个状态都是 false"

    detail = client.get(f"/documents/{item['doc_id']}").json()
    assert detail["markdown"] == "" and detail["outline"] is None


def test_uploading_the_same_bytes_twice_is_idempotent(client: TestClient, tmp_path: Path) -> None:
    """同一份内容重传 = 同一个 doc_id（原地 upsert），盘上还是一份。"""
    first = _upload(client, "a.md").json()["uploads"][0]
    body = _upload(client, "a.md").json()
    second = body["uploads"][0]

    assert second["doc_id"] == first["doc_id"], "同名同内容 → 同一个落盘名 → 同一个 doc_id"
    assert second["replaced"] is True, "如实说「已经在了」"
    assert body["conflicts"] == [], "内容一样不算冲突，不该弹窗问"
    assert len(_stored(tmp_path)) == 1, "盘上只有一份"
    assert len(client.get("/documents").json()["documents"]) == 1


def test_a_changed_file_lands_beside_the_old_one_and_is_reported_as_a_conflict(
    client: TestClient, tmp_path: Path
) -> None:
    """同名、内容不同 → **两份文件**（名字里的哈希不同），并把旧的那份摆出来问用户。

    不自动覆盖是使用者定的：改了几个字就悄悄替掉旧版，人会以为「更新了」，
    其实旧版已经连同它的人工核对结果一起没了。
    """
    first = _upload(client, "a.md", "# 第一版\n").json()["uploads"][0]
    body = _upload(client, "a.md", "# 第二版\n").json()
    second = body["uploads"][0]

    assert second["doc_id"] != first["doc_id"] and not second["replaced"]
    assert second["display_name"] == first["display_name"] == "a.md"
    assert len(_stored(tmp_path)) == 2, "两版同时留着，谁也不覆盖谁"
    assert [row["doc_id"] for row in body["conflicts"]] == [first["doc_id"]]
    assert [row["display_name"] for row in body["conflicts"]] == ["a.md"]
    assert len(client.get("/documents").json()["documents"]) == 2


def test_deleting_a_document_purges_its_artifacts_too(
    client: TestClient, components: Components, tmp_path: Path, monkeypatch
) -> None:
    """页面上的「覆盖更新」就是这条：把旧版**连它的产物与已入库内容**一起清掉。"""
    import rag.indexing

    index = _Index()
    monkeypatch.setattr(rag.indexing, "build_index", lambda settings: index)
    doc_id = _parsed(client, components)
    assert (tmp_path / "stages" / doc_id).is_dir()
    assert client.delete("/documents/nope").status_code == 404

    body = client.delete(f"/documents/{doc_id}").json()

    assert body["deleted"] == doc_id
    assert index.dropped == [doc_id], "先摘库里的块，再清产物，最后删文件"
    assert not (tmp_path / "stages" / doc_id).exists(), "中间产物跟着清，不然只增不减"
    assert [row["doc_id"] for row in client.get("/documents").json()["documents"]] == []


def test_unsupported_and_oversized_files_are_reported_not_swallowed(
    client: TestClient, components: Components
) -> None:
    """被拒的要**说清原因**（后缀不支持 / 超过上限）—— 静默丢文件是最难查的那类问题。"""
    components.settings.ingest.uploads.max_mb = 0  # 任何非空文件都超限

    body = _upload(client, "表.xyz").json()
    assert body["uploads"] == [] and len(body["skipped"]) == 1
    assert "不支持的文件类型" in body["skipped"][0]

    body = _upload(client, "ok.md").json()
    assert body["uploads"] == [] and "超过上限" in body["skipped"][0]


def test_parse_produces_the_preview_and_the_outline(
    client: TestClient, components: Components
) -> None:
    """解析：落 `document.md` + `outline.tsv`，详情接口把它读出来（正文只读）。"""
    item = _upload(client, "手册.md").json()["uploads"][0]

    # 后台任务的编排在这里真实跑一遍（HTTP 那层只负责 spawn，见模块说明）
    components.ingest_slot = asyncio.Semaphore(1)
    job = components.jobs.spawn(kind="parse", request={"doc_id": item["doc_id"]})
    asyncio.run(components.run_parse_job(job, item["doc_id"]))

    assert job.status == "done", job.error
    assert job.result["parsed"] == 1

    detail = client.get(f"/documents/{item['doc_id']}").json()

    assert detail["stages"]["parsed"] is True
    assert detail["stages"]["reviewed"] is False, "没人工核对之前不许切分"
    assert "# 第一章" in detail["markdown"], "解析稿（人读版）"
    # 头部那段 `<!-- doc_id … metadata … -->` 是**产物**给人查 parse_key / 页数用的，
    # 不该跑到页面的解析稿里当正文显示（2026-10-10 使用者报的）
    assert not detail["markdown"].lstrip().startswith("<!--")
    assert "parse_key" not in detail["markdown"]
    assert detail["outline"] is not None
    assert [row["title"] for row in detail["outline"]["rows"]] == ["第一章"]
    assert detail["outline"]["confirmed"] is False


def test_unknown_document_is_404(client: TestClient) -> None:
    assert client.get("/documents/nope").status_code == 404
    assert client.post("/documents/nope/parse").status_code == 404


def test_jobs_list_is_newest_first(components: Components) -> None:
    """`GET /jobs` 只列本进程的任务（多 worker 那天要扩成扫 Redis，见 jobs.py）。"""
    app = create_app(components.settings, Secrets(rag_api_key=None))
    app.state.components = components
    client = TestClient(app)

    older = components.jobs.spawn(kind="ingest", request={})
    older.created_at = 100
    newer = components.jobs.spawn(kind="parse", request={})
    newer.created_at = 200
    asyncio.run(components.jobs.save(older))
    asyncio.run(components.jobs.save(newer))

    rows = client.get("/jobs?limit=5").json()["jobs"]

    assert [row["kind"] for row in rows] == ["parse", "ingest"], "新的在前"


# ---- 第三步：人工核对（只改层级，正文一个字都不许动）----

_MULTI = "# 第一章\n\n开头\n\n## 1.1 节\n\n中间\n\n### 1.1.1 小节\n\n结尾\n"


def _parsed(client: TestClient, components: Components, name: str = "手册.md") -> str:
    """上传 + 解析一份三级标题的文档，返回 doc_id。"""
    item = _upload(client, name, _MULTI).json()["uploads"][0]
    components.ingest_slot = asyncio.Semaphore(1)
    job = components.jobs.spawn(kind="parse", request={"doc_id": item["doc_id"]})
    asyncio.run(components.run_parse_job(job, item["doc_id"]))
    assert job.status == "done", job.error
    return item["doc_id"]


def test_outline_update_only_rewrites_the_checklist(
    client: TestClient, components: Components, tmp_path: Path
) -> None:
    """提交核对结果：层级改掉了、正文与大纲条目标题**一个字都没动**。"""
    doc_id = _parsed(client, components)
    before = (tmp_path / "stages" / doc_id / "document.md").read_bytes()

    # 三行写成 1/3/5，看它是不是原样落盘（跳级压平是切分那一步的事，写回不做手脚）
    body = client.put(
        f"/documents/{doc_id}/outline",
        json={
            "rows": [{"level": 2}, {"level": 2}, {"level": 3}],
            "max_level": 2,
            "confirmed": True,
        },
    ).json()

    assert body["outline"]["confirmed"] is True
    assert [row["level"] for row in body["outline"]["rows"]] == [2, 2, 3]
    assert [row["title"] for row in body["outline"]["rows"]] == ["第一章", "1.1 节", "1.1.1 小节"]
    assert body["outline"]["max_level"] == 2
    assert body["stages"]["reviewed"] is True

    tsv = (tmp_path / "stages" / doc_id / "outline.tsv").read_text(encoding="utf-8")
    assert "# 最大层级: 2" in tsv and "# 已核对: 是" in tsv
    assert (tmp_path / "stages" / doc_id / "document.md").read_bytes() == before, "正文没动"


def test_outline_update_refuses_to_add_or_drop_rows(
    client: TestClient, components: Components
) -> None:
    """清单只该改层级，不该增删行 —— 对不上要说清差多少，别猜。"""
    doc_id = _parsed(client, components)

    few = client.put(f"/documents/{doc_id}/outline", json={"rows": [{"level": 1}]})
    assert few.status_code == 400, "3 个标题行只提交 1 条"
    assert "1 条" in few.json()["error"]["message"] and "3 个" in few.json()["error"]["message"]

    out_of_range = client.put(
        f"/documents/{doc_id}/outline",
        json={"rows": [{"level": 7}] * 3},
    )
    assert out_of_range.status_code == 422, "层级只有 1..6"

    unknown = client.put("/documents/nope/outline", json={"rows": [], "confirmed": True})
    assert unknown.status_code == 404


def test_outline_update_needs_a_parse_first(client: TestClient, components: Components) -> None:
    """没解析就来核对：409 说清差哪一步（没有解析稿就没有可核对的标题行）。"""
    item = _upload(client, "还没解析.md").json()["uploads"][0]

    res = client.put(f"/documents/{item['doc_id']}/outline", json={"rows": [], "confirmed": True})

    assert res.status_code == 409 and "还没解析" in res.json()["error"]["message"]


def test_a_heading_can_be_marked_as_body(client: TestClient, components: Components) -> None:
    """**标为正文**：层级写 0，那一行不进标题路径、单独成节 —— 正文一个字都不动。

    这是「解析把一段普通文字认成了标题」的出口：删行会让清单与正文对不上（整份切不了），
    所以改的是它的**角色**。
    """
    doc_id = _parsed(client, components)

    saved = client.put(
        f"/documents/{doc_id}/outline",
        json={
            "rows": [{"level": 1}, {"level": 0}, {"level": 2}],
            "max_level": 3,
            "confirmed": True,
        },
    ).json()

    assert [row["level"] for row in saved["outline"]["rows"]] == [1, 0, 2]
    assert [row["title"] for row in saved["outline"]["rows"]] == [
        "第一章",
        "1.1 节",
        "1.1.1 小节",
    ], "标题原文照旧是只读的"


def test_index_refuses_until_the_outline_is_confirmed(
    client: TestClient, components: Components
) -> None:
    """「已核对」是硬门槛：没核对就 409，不进后台任务。"""
    doc_id = _parsed(client, components)

    blocked = client.post(f"/documents/{doc_id}/index")
    assert blocked.status_code == 409 and "还没核对" in blocked.json()["error"]["message"]

    client.put(
        f"/documents/{doc_id}/outline",
        json={
            "rows": [{"level": 1}, {"level": 2}, {"level": 3}],
            "max_level": 3,
            "confirmed": True,
        },
    )
    started = client.post(f"/documents/{doc_id}/index")

    assert started.status_code == 202
    assert started.json()["kind"] == "index" and started.json()["request"]["doc_id"] == doc_id
    assert client.post("/documents/nope/index").status_code == 404


def _jobs(client: TestClient) -> list[dict]:
    return client.get("/jobs").json()["jobs"]


def _index_jobs(client: TestClient) -> list[dict]:
    return [item for item in _jobs(client) if item["kind"] == "index"]


def _confirm(client: TestClient, doc_id: str, *, confirmed: bool = True):
    return client.put(
        f"/documents/{doc_id}/outline",
        json={
            "rows": [{"level": 1}, {"level": 2}, {"level": 3}],
            "max_level": 3,
            "confirmed": confirmed,
        },
    )


async def _noop_index_run(job, doc_id: str) -> None:
    """把真任务换掉：这些用例验的是「**有没有排上**」，不是切分本身对不对（那在别处）。"""
    return None


def test_confirming_the_outline_starts_indexing_right_away(
    client: TestClient, components: Components, monkeypatch
) -> None:
    """勾「已核对」提交 = **直接往下走**（`api.index_after_review`，默认开）。

    四步流程里最后那次点击没有新决策 —— 切分只认核对过的清单，而它刚被核对完。
    响应里带回那个任务，页面据此切到「拆分入库中…」，不用再让人点一次。
    """
    components.settings.api.index_after_review = True
    monkeypatch.setattr(components, "run_index_job", _noop_index_run)
    doc_id = _parsed(client, components)

    body = _confirm(client, doc_id).json()

    assert body["stages"]["reviewed"] is True
    job = body["index_job"]
    assert job and job["kind"] == "index" and job["request"]["doc_id"] == doc_id
    assert [item["id"] for item in _index_jobs(client)] == [job["id"]]


def test_saving_without_the_checkmark_does_not_start_indexing(
    client: TestClient, components: Components, monkeypatch
) -> None:
    """没勾「已核对」只是**存草稿**：不触发任何花钱的动作（切分/入库都按量计费）。"""
    components.settings.api.index_after_review = True
    monkeypatch.setattr(components, "run_index_job", _noop_index_run)
    doc_id = _parsed(client, components)

    body = _confirm(client, doc_id, confirmed=False).json()

    assert body["index_job"] is None
    assert _index_jobs(client) == []


def test_the_auto_index_can_be_switched_off(client: TestClient, components: Components) -> None:
    """`api.index_after_review: false` = 回到「手动点拆分并入库」。"""
    components.settings.api.index_after_review = False
    doc_id = _parsed(client, components)

    body = _confirm(client, doc_id).json()

    assert body["index_job"] is None
    assert _index_jobs(client) == []
    assert client.post(f"/documents/{doc_id}/index").status_code == 202, "手动那条路照旧"


def test_the_same_document_never_gets_two_index_jobs(
    client: TestClient, components: Components, monkeypatch
) -> None:
    """连着提交两次、或提交完再点「拆分并入库」，**只该有一个任务** —— 重复跑就是重复花钱。"""
    components.settings.api.index_after_review = True
    monkeypatch.setattr(components, "run_index_job", _noop_index_run)
    doc_id = _parsed(client, components)

    first = _confirm(client, doc_id).json()["index_job"]
    second = _confirm(client, doc_id).json()["index_job"]
    manual = client.post(f"/documents/{doc_id}/index").json()

    assert first and second and manual
    assert len({first["id"], second["id"], manual["id"]}) == 1, "三次都要指向同一个任务"
    assert len(_index_jobs(client)) == 1


class _Index:
    """替身索引：只记账（不连 Milvus）。"""

    def __init__(self) -> None:
        self.upserted: list[str] = []
        self.dropped: list[str] = []

    async def ensure_ready(self) -> None:
        return None

    async def delete_docs(self, doc_ids: list[str]) -> None:
        self.dropped.extend(doc_ids)

    async def upsert(self, chunks, vectors) -> int:
        self.upserted.extend(chunk.chunk_id for chunk in chunks)
        return len(chunks)


class _Embedder:
    supports_images = False
    name = "stub"
    dim = 2

    async def embed_documents(self, texts, images=None):
        from rag.contracts import Embeddings

        return Embeddings(dense=[[1.0, 0.0] for _ in texts])


def test_the_index_job_chunks_and_writes_this_document(
    client: TestClient, components: Components, tmp_path: Path, monkeypatch
) -> None:
    """第五步真跑一遍：切分 → 写 chunks 产物 → 灌库（索引与注册表都是替身）。"""
    from types import SimpleNamespace

    doc_id = _parsed(client, components)
    client.put(
        f"/documents/{doc_id}/outline",
        json={
            "rows": [{"level": 1}, {"level": 2}, {"level": 3}],
            "max_level": 2,
            "confirmed": True,
        },
    )
    components.index = _Index()
    components.heavy = SimpleNamespace(embedder=_Embedder(), errors={})
    components.ingest_slot = asyncio.Semaphore(1)
    job = components.jobs.spawn(kind="index", request={"doc_id": doc_id})

    asyncio.run(components.run_index_job(job, doc_id))

    assert job.status == "done", job.error
    assert job.result["chunks"] > 0 and job.result["written"] == job.result["chunks"]
    assert (tmp_path / "stages" / doc_id / "chunks.json").is_file()

    detail = client.get(f"/documents/{doc_id}").json()
    assert detail["stages"]["chunked"] is True

    # 再改一次层级：指纹变了，旧的切分产物**自动作废**（下次跑 index 会重切）
    client.put(
        f"/documents/{doc_id}/outline",
        json={
            "rows": [{"level": 1}, {"level": 1}, {"level": 2}],
            "max_level": 3,
            "confirmed": True,
        },
    )
    assert client.get(f"/documents/{doc_id}").json()["stages"]["chunked"] is False


# ---- 从链接导入（抓回来的东西当普通上传件落盘）----


def _fake_fetch(name: str = "官网手册.html", data: bytes = b"<h1>hi</h1>", **over):
    """替掉真的 `fetch_url`：**用例不碰网络**（真抓由 tests/test_fetch.py 离线覆盖）。"""
    from rag.ingest.fetch import Fetched

    def fake(url: str, **kwargs):
        return Fetched(
            name=name,
            data=data,
            content_type=over.get("content_type", "text/html"),
            final_url=over.get("final_url", url),
        )

    return fake


def test_a_fetched_link_lands_like_an_upload(
    client: TestClient, tmp_path: Path, monkeypatch
) -> None:
    """抓回来的东西与上传件完全同构：真名 + 哈希落盘，来源 URL 记着，后面五步一样走。"""
    import rag.ingest.fetch

    monkeypatch.setattr(rag.ingest.fetch, "fetch_url", _fake_fetch())
    body = client.post(
        "/documents/fetch",
        json={"url": "https://docs.example.com/page?id=7"},
    ).json()

    item = body["uploads"][0]
    assert item["display_name"] == "官网手册.html"
    assert item["name"].startswith("官网手册-") and item["name"].endswith(".html")
    assert item["source_url"] == "https://docs.example.com/page?id=7"
    assert (tmp_path / "uploads" / item["name"]).is_file()

    row = client.get("/documents").json()["documents"][0]
    assert row["display_name"] == "官网手册.html"
    assert row["source_url"] == "https://docs.example.com/page?id=7"
    assert row["content_type"] == "text/html"
    assert client.get(f"/documents/{item['doc_id']}").json()["source_url"] == item["source_url"]


def test_a_fetched_link_can_be_parsed_right_away(
    client: TestClient, components: Components, monkeypatch
) -> None:
    """抓下来就能直接解析 —— 这条是「网页上的文档」那个场景的正题。"""
    import rag.ingest.fetch

    monkeypatch.setattr(
        rag.ingest.fetch, "fetch_url", _fake_fetch(data="<h1>第一章</h1><p>正文</p>".encode())
    )
    doc_id = client.post("/documents/fetch", json={"url": "https://x.example/a"}).json()["uploads"][
        0
    ]["doc_id"]

    components.ingest_slot = asyncio.Semaphore(1)
    job = components.jobs.spawn(kind="parse", request={"doc_id": doc_id})
    asyncio.run(components.run_parse_job(job, doc_id))

    assert job.status == "done", job.error
    detail = client.get(f"/documents/{doc_id}").json()
    assert "# 第一章" in detail["markdown"]
    assert [row["title"] for row in detail["outline"]["rows"]] == ["第一章"]


def test_the_same_link_fetched_twice_is_idempotent(client: TestClient, monkeypatch) -> None:
    """同一个链接再抓一次：内容没变 → 同名同哈希 → 原地 upsert，不弹冲突。"""
    import rag.ingest.fetch

    monkeypatch.setattr(rag.ingest.fetch, "fetch_url", _fake_fetch())
    first = client.post("/documents/fetch", json={"url": "https://x.example/a"}).json()
    again = client.post("/documents/fetch", json={"url": "https://x.example/a"}).json()

    assert again["uploads"][0]["doc_id"] == first["uploads"][0]["doc_id"]
    assert again["uploads"][0]["replaced"] is True and again["conflicts"] == []


def test_a_changed_page_is_reported_as_a_conflict_not_silently_replaced(
    client: TestClient, monkeypatch
) -> None:
    """同一个链接的内容变了（页面更新了）→ 新版落盘、旧版摆出来问，不自动顶掉。"""
    import rag.ingest.fetch

    monkeypatch.setattr(rag.ingest.fetch, "fetch_url", _fake_fetch(data=b"<h1>v1</h1>"))
    old = client.post("/documents/fetch", json={"url": "https://x.example/a"}).json()["uploads"][0]
    monkeypatch.setattr(rag.ingest.fetch, "fetch_url", _fake_fetch(data=b"<h1>v2</h1>"))

    body = client.post("/documents/fetch", json={"url": "https://x.example/a"}).json()

    assert body["uploads"][0]["doc_id"] != old["doc_id"]
    assert [row["doc_id"] for row in body["conflicts"]] == [old["doc_id"]]


def test_a_custom_name_without_a_suffix_borrows_the_fetched_one(
    client: TestClient, monkeypatch
) -> None:
    """自己起名时不用管后缀：`官网手册` + 抓到的 `.html` → `官网手册.html`。"""
    import rag.ingest.fetch

    monkeypatch.setattr(rag.ingest.fetch, "fetch_url", _fake_fetch(name="raw.html"))
    item = client.post(
        "/documents/fetch", json={"url": "https://x.example/a", "name": "官网手册"}
    ).json()["uploads"][0]

    assert item["display_name"] == "官网手册.html"


def test_fetch_failures_keep_their_status_and_message(client: TestClient, monkeypatch) -> None:
    """远端不给 → 502 并把原因原样带出来；认不出类型 → 400。两者要分得开。"""
    import rag.ingest.fetch

    def boom(url: str, **kwargs):
        raise rag.ingest.fetch.FetchError("站点返回 403（https://x.example/a）", status=502)

    monkeypatch.setattr(rag.ingest.fetch, "fetch_url", boom)
    res = client.post("/documents/fetch", json={"url": "https://x.example/a"})

    assert res.status_code == 502 and "403" in res.json()["error"]["message"]


def test_a_fetched_page_keeps_its_images_on_disk(
    client: TestClient, tmp_path: Path, monkeypatch
) -> None:
    """页面里的图一起落到上传目录（同名 `_files/` 子目录里），删文档时跟着走。"""
    import rag.indexing
    import rag.ingest.fetch
    from rag.ingest.fetch import Fetched

    monkeypatch.setattr(rag.indexing, "build_index", lambda settings: _Index())
    folder = "官网手册-0f1e2d3c4b5a6978_files"
    monkeypatch.setattr(
        rag.ingest.fetch,
        "fetch_url",
        lambda url, **kwargs: Fetched(
            name="官网手册.html",
            data=f'<p>x</p><img src="{folder}/img01.png">'.encode(),
            content_type="text/html",
            final_url=url,
            attachments={f"{folder}/img01.png": b"PNG"},
        ),
    )
    item = client.post("/documents/fetch", json={"url": "https://x.example/a"}).json()["uploads"][0]

    assert (tmp_path / "uploads" / folder / "img01.png").read_bytes() == b"PNG"
    # 附属目录不是文档：清单里还是只有那一个 .html
    assert [row["doc_id"] for row in client.get("/documents").json()["documents"]] == [
        item["doc_id"]
    ]

    client.delete(f"/documents/{item['doc_id']}")

    assert not (tmp_path / "uploads" / folder).exists(), "删文档时图片跟着走，否则只增不减"


def test_notes_from_the_fetch_are_returned_to_the_page(client: TestClient, monkeypatch) -> None:
    """没抓到的图要**回给页面**（还有一条 WARNING 进 system.log，由日志那层验）。"""
    import rag.ingest.fetch
    from rag.ingest.fetch import Fetched

    monkeypatch.setattr(
        rag.ingest.fetch,
        "fetch_url",
        lambda url, **kwargs: Fetched(
            name="a.html",
            data=b"<p>x</p>",
            content_type="text/html",
            final_url=url,
            notes=["https://x.example/img.png：站点返回 403"],
        ),
    )
    body = client.post("/documents/fetch", json={"url": "https://x.example/a"}).json()

    assert body["notes"] == ["https://x.example/img.png：站点返回 403"]


# ---- 各阶段的摘要（页面上那条流程图的数字从这里来）----


def test_the_detail_reports_what_each_stage_produced(
    client: TestClient, components: Components, monkeypatch
) -> None:
    """解析/切分/入库三段各自的数字：字数、块数、产物大小、入库记录、对不对得上。"""
    from types import SimpleNamespace

    # 用 markdown 切法（真机 dev 档是 markdown-semantic）：只有它会把 heading_path 带进块，
    # 大纲那一栏才有东西可看。测试用的 `Settings()` 走的是 pydantic 默认（recursive）。
    components.settings.indexing.chunk.strategy = "markdown"
    doc_id = _parsed(client, components)
    client.put(
        f"/documents/{doc_id}/outline",
        json={
            "rows": [{"level": 1}, {"level": 2}, {"level": 3}],
            "max_level": 3,
            "confirmed": True,
        },
    )
    components.index = _Index()
    components.heavy = SimpleNamespace(embedder=_Embedder(), errors={})
    components.ingest_slot = asyncio.Semaphore(1)
    job = components.jobs.spawn(kind="index", request={"doc_id": doc_id})
    asyncio.run(components.run_index_job(job, doc_id))
    assert job.status == "done", job.error

    body = client.get(f"/documents/{doc_id}").json()

    parse = body["parse"]
    assert parse["chars"] > 0 and parse["parser"] == "text" and parse["stale"] is False
    assert parse["recorded_hash"] == parse["current_hash"] == body["sha1"]

    chunks = body["chunks"]
    assert chunks["blocks"] == job.result["chunks"] == chunks["text_blocks"]
    assert chunks["chars"] > 0 and chunks["max_chars"] >= chunks["avg_chars"] > 0
    assert len(chunks["sizes"]) == chunks["blocks"], "每块一个字数是画分布条用的"
    assert chunks["artifact_bytes"]["chunks.json"] > 0 and chunks["artifact_bytes"]["chunks.md"] > 0
    assert [row["heading_path"] for row in chunks["sections"]][:1] == ["第一章"]
    assert chunks["stale"] is False

    index = body["index"]
    assert index["indexed"] is True and index["chunk_count"] == chunks["blocks"]
    assert index["in_sync"] is True, "注册表说灌的块数 == 产物里的块数"
    assert (
        index["vector_dim"] > 0
        and index["vector_bytes"] == index["chunk_count"] * index["vector_dim"] * 4
    )


def test_a_stale_artifact_says_why_it_went_stale(
    client: TestClient, components: Components, tmp_path: Path
) -> None:
    """源文件改了 → 步骤灯回退，**并且说明是为什么**（不然会以为是自己点错了）。"""
    doc_id = _parsed(client, components)
    uploads = tmp_path / "uploads"
    target = next(p for p in uploads.iterdir() if p.is_file() and p.suffix == ".md")
    before = client.get(f"/documents/{doc_id}").json()
    assert before["stages"]["parsed"] is True and before["parse"]["stale"] is False

    # 原地换内容：文件名不变（doc_id 不变），但内容哈希变了 → 产物立刻作废
    target.write_text("# 换过的标题\n", encoding="utf-8")

    after = client.get(f"/documents/{doc_id}").json()
    assert after["stages"]["parsed"] is False, "源文件变了，步骤灯回退"
    assert after["parse"]["stale"] is True, "但要能说出**为什么**回退"
    assert after["parse"]["recorded_hash"] != after["parse"]["current_hash"]
    assert after["parse"]["chars"] > 0, "过期那份的摘要照样给出来（好让人对着看）"


def test_the_source_endpoint_serves_the_original_bytes(client: TestClient) -> None:
    """「上传」那一步要看原文：原样给字节，但**只认 uploads 里的 doc_id**。"""
    item = _upload(client, "手册.md", "# 第一章\n\n正文\n").json()["uploads"][0]

    res = client.get(f"/documents/{item['doc_id']}/source")

    assert res.status_code == 200
    assert res.text == "# 第一章\n\n正文\n"
    assert res.headers["content-type"].startswith("text/plain"), "md 要能直接显示，不能触发下载"
    assert res.headers["x-content-type-options"] == "nosniff"
    assert res.headers["content-security-policy"] == "sandbox"
    assert "inline" in res.headers["content-disposition"]
    assert client.get("/documents/nope/source").status_code == 404


def test_the_source_endpoint_keeps_html_as_html(client: TestClient) -> None:
    """HTML 保持 `text/html` —— 页面要用沙箱 iframe 渲染出版排，降级成纯文本就白搭。"""
    item = _upload(client, "页.html", "<h1>标题</h1>").json()["uploads"][0]

    res = client.get(f"/documents/{item['doc_id']}/source")

    assert res.headers["content-type"].startswith("text/html")
