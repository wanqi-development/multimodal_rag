"""三段流水线的中间产物：落盘、人读版、新鲜度。

全部本地 —— OCR / VLM 都用假的，不碰网络，也不碰 Milvus。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from rag.config.schema import (
    ChunkingConfig,
    IndexingConfig,
    IngestAssetsConfig,
    IngestConfig,
    IngestImagesConfig,
    IngestPdfConfig,
    IngestStagesConfig,
    Settings,
)
from rag.contracts import Embeddings
from rag.ingest.ocr import OcrPage
from rag.ingest.stages import ArtifactStore, build_artifact_store, parse_key, run_chunk, run_parse
from rag.storage import BlobNotFound

_SRC = "imgs/img001.png"
_PNG = b"\x89PNG\r\n\x1a\nfake-png"
_PAGE = f'# 第一章 概述\n\n上面这段讲的是作业图。\n\n<img src="{_SRC}" />\n\n下面这段讲的是并行度。'


class _FakeOcr:
    name = "paddleocr"  # 真实 client 都会报这个名字（工厂约定的一部分）

    def __init__(self, page: str = _PAGE) -> None:
        self._page = page
        self.calls = 0

    def parse(self, path: Path, *, file_type: int) -> list[OcrPage]:
        self.calls += 1
        return [OcrPage(page_index=0, markdown=self._page, images={_SRC: _PNG})]


class _StubEmbedder:
    @property
    def dim(self) -> int:
        return 2

    async def embed_documents(self, texts: list[str]) -> Embeddings:
        return Embeddings(dense=[[1.0, 0.0] for _ in texts])

    async def embed_query(self, text: str) -> Embeddings:
        return Embeddings(dense=[[1.0, 0.0]])


def _settings(
    tmp_path: Path, *, size: int = 1000, parser: str = "paddleocr", describe: bool = False
) -> Settings:
    return Settings(
        indexing=IndexingConfig(chunk=ChunkingConfig(strategy="markdown", size=size, overlap=0)),
        ingest=IngestConfig(
            pdf=IngestPdfConfig(parser=parser),
            assets=IngestAssetsConfig(dir=str(tmp_path / "assets")),
            images=IngestImagesConfig(
                describe=describe,
                cache_path=str(tmp_path / "cache.json"),
                max_images=0,
            ),
            stages=IngestStagesConfig(dir=str(tmp_path / "stages")),
        ),
    )


def _scan(tmp_path: Path) -> Path:
    """用 .png：它的 loader 必须拿到 OCR 客户端，正好用来证明「产物命中就不解析」。

    （不用 .pdf 是因为 PDF loader 会先用 pymupdf 验一遍是不是合法 PDF，
    那要求测试里塞一份真 PDF —— 与这条用例要验的事无关。）
    """
    target = tmp_path / "scan.png"
    target.write_bytes(_PNG)
    return target


def _confirm(tmp_path: Path, store: ArtifactStore, *, max_level: int | None = None) -> str:
    """模拟人工核对那一步：把清单头部的「已核对: 否」改成「是」（可顺便改最大层级）。

    切分**只认核对过的清单**（见 run_chunk），所以除专门验这道门槛的用例之外，
    每个用例的这一下都不能省 —— 它就是把真实流程里人工做的动作写下来。
    """
    doc_id = next(p.name for p in (tmp_path / "stages").iterdir())
    path = store.path_of(doc_id, "outline.tsv")
    text = path.read_text(encoding="utf-8").replace("# 已核对: 否", "# 已核对: 是")
    if max_level is not None:
        text = re.sub(r"^# 最大层级: \d+", f"# 最大层级: {max_level}", text, flags=re.M)
    path.write_text(text, encoding="utf-8")
    return doc_id


def _run_parse(tmp_path: Path, settings: Settings, ocr: _FakeOcr, path: Path, store: ArtifactStore):
    result = run_parse([path], tmp_path, settings, ocr, store)
    _confirm(tmp_path, store)  # 人是解析完之后核对的，这里跟上
    return result


def _run_chunk(tmp_path: Path, settings: Settings, store: ArtifactStore, path: Path, ocr=None):
    return run_chunk([path], tmp_path, settings, None, None, ocr, store)


# ---- 阶段1：解析产物 ----


def test_parse_writes_both_machine_and_human_readable_artifacts(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = build_artifact_store(settings)
    path = _scan(tmp_path)

    result = _run_parse(tmp_path, settings, _FakeOcr(), path, store)

    assert (result.parsed, result.reused) == (1, 0)
    doc_id = next(p.name for p in (tmp_path / "stages").iterdir())
    payload = json.loads(store.path_of(doc_id, "document.json").read_text(encoding="utf-8"))
    assert payload["kind"] == "document"
    assert payload["parse_key"] == parse_key(settings, ".pdf")
    assert payload["source_uri"] == "scan.png"
    assert payload["document"]["text"].startswith("# 第一章 概述")
    assert payload["document"]["images"][0]["asset_uri"].endswith("p0001_01.png")

    human = store.path_of(doc_id, "document.md").read_text(encoding="utf-8")
    assert human.startswith("<!--")
    assert "source_uri   scan.png" in human
    assert "# 第一章 概述" in human, "人读版要能看到解析出来的正文"


def test_parse_reuses_a_fresh_artifact_without_touching_the_file(tmp_path: Path) -> None:
    """产物新鲜 = 连文件都不读（这才叫缓存）。"""
    settings = _settings(tmp_path)
    store = build_artifact_store(settings)
    path = _scan(tmp_path)
    ocr = _FakeOcr()

    _run_parse(tmp_path, settings, ocr, path, store)
    second = _run_parse(tmp_path, settings, ocr, path, store)

    assert (second.parsed, second.reused) == (0, 1)
    assert ocr.calls == 1, "第二次不该再解析一遍"
    assert second.todo == []


def test_parse_redo_when_the_file_changes(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = build_artifact_store(settings)
    path = _scan(tmp_path)

    _run_parse(tmp_path, settings, _FakeOcr(), path, store)
    path.write_bytes(b"%PDF-1.4 fake but changed")
    result = _run_parse(tmp_path, settings, _FakeOcr(), path, store)

    assert (result.parsed, result.reused) == (1, 0), "文件变了就必须重解析"


def test_parse_redo_when_parse_settings_change(tmp_path: Path) -> None:
    """换了 PDF 解析方式（ocr ↔ pymupdf），产物必须失效 —— 出来的东西完全不同。"""
    store = build_artifact_store(_settings(tmp_path))
    path = _scan(tmp_path)
    _run_parse(tmp_path, _settings(tmp_path), _FakeOcr(), path, store)

    other = _settings(tmp_path, parser="pymupdf")
    result = _run_parse(tmp_path, other, _FakeOcr(), path, build_artifact_store(other))

    assert (result.parsed, result.reused) == (1, 0)
    assert parse_key(other, ".pdf") != parse_key(_settings(tmp_path), ".pdf")


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = build_artifact_store(settings)
    path = _scan(tmp_path)
    ocr = _FakeOcr()

    result = run_parse([path], tmp_path, settings, ocr, store, dry_run=True)

    assert result.todo == ["scan.png"], "dry-run 要把「将会处理哪些」摊出来"
    assert result.parsed == 0 and ocr.calls == 0
    assert not (tmp_path / "stages").exists(), "一行都没写"


# ---- 阶段2：切分产物 ----


def test_chunk_runs_off_the_cached_document(tmp_path: Path) -> None:
    """阶段2 只吃阶段1 的产物：连解析用的 OCR 客户端都可以没有。"""
    settings = _settings(tmp_path)
    store = build_artifact_store(settings)
    path = _scan(tmp_path)
    _run_parse(tmp_path, settings, _FakeOcr(), path, store)

    result = _run_chunk(tmp_path, settings, store, path, ocr=None)

    assert result.chunked == 1
    assert result.chunks, "块应当从产物里切出来"
    assert all(c.doc_id == result.chunks[0].doc_id for c in result.chunks)
    assert result.chunks[0].heading_path == "第一章 概述"
    # 页内第 3 行：标题行被 MarkdownHeaderTextSplitter 剥掉了，块从正文那一行起
    assert (result.chunks[0].page, result.chunks[0].line) == (1, 3)


def test_chunk_artifacts_have_both_readings(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = build_artifact_store(settings)
    path = _scan(tmp_path)
    _run_parse(tmp_path, settings, _FakeOcr(), path, store)
    _run_chunk(tmp_path, settings, store, path)
    doc_id = next(p.name for p in (tmp_path / "stages").iterdir())

    payload = json.loads(store.path_of(doc_id, "chunks.json").read_text(encoding="utf-8"))
    assert payload["kind"] == "chunks"
    assert payload["chunks"][0]["modality"] == "text"

    human = store.path_of(doc_id, "chunks.md").read_text(encoding="utf-8")
    assert "## [0000] text · p1 L3 · 章节=第一章 概述 · 编号=1 ·" in human
    assert "图=" in human, "块里引用了哪几张图也要摊出来"
    assert "上面这段讲的是作业图。" in human


def test_chunk_reuses_fresh_artifacts(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = build_artifact_store(settings)
    path = _scan(tmp_path)
    _run_parse(tmp_path, settings, _FakeOcr(), path, store)
    _run_chunk(tmp_path, settings, store, path)

    second = _run_chunk(tmp_path, settings, store, path)

    assert (second.chunked, second.reused) == (0, 1)
    assert second.chunks, "复用时也要把块带出来（阶段3 要用）"


def test_chunk_redo_when_the_chunking_changes(tmp_path: Path) -> None:
    """切法变了（这里是 chunk.size），阶段2 的产物必须失效，但**解析产物不用重做**。"""
    settings = _settings(tmp_path, size=1000)
    store = build_artifact_store(settings)
    path = _scan(tmp_path)
    _run_parse(tmp_path, settings, _FakeOcr(), path, store)
    _run_chunk(tmp_path, settings, store, path)

    other = _settings(tmp_path, size=60)
    ocr = _FakeOcr()
    result = _run_chunk(tmp_path, other, store, path, ocr=ocr)

    assert result.chunked == 1
    assert ocr.calls == 0, "解析产物还是新鲜的，不该重新解析"


def test_chunk_waits_for_the_review_when_there_is_no_list_yet(tmp_path: Path) -> None:
    """没先跑阶段1 就直接切：进「待人工核对」，并指路去跑 --stage parse。

    以前这里会现场补解析；加了「人工核对」这道门槛之后就不行了 —— 核对发生在两次命令
    之间，一次 run 里没法既生成清单又等人核对完再切。所以如实告诉用户下一步敲什么。
    """
    settings = _settings(tmp_path)
    store = build_artifact_store(settings)
    path = _scan(tmp_path)
    ocr = _FakeOcr()

    result = _run_chunk(tmp_path, settings, store, path, ocr=ocr)

    assert result.chunked == 0 and ocr.calls == 0, "连解析都不做：切之前先要人核对"
    assert [item.path for item in result.pending] == ["scan.png"]
    assert "stage parse" in result.pending[0].reason


def test_chunk_full_rechunks_but_does_not_reparse(tmp_path: Path) -> None:
    """`--stage chunk --full` = 重新切，不该顺带重新 OCR（解析产物是它的输入）。"""
    settings = _settings(tmp_path)
    store = build_artifact_store(settings)
    path = _scan(tmp_path)
    _run_parse(tmp_path, settings, _FakeOcr(), path, store)
    _run_chunk(tmp_path, settings, store, path)

    ocr = _FakeOcr()
    result = run_chunk([path], tmp_path, settings, None, None, ocr, store, full=True)

    assert result.chunked == 1, "full 要重新切"
    assert ocr.calls == 0, "但解析产物是新鲜的，不该重新解析"


def test_chunk_before_the_parse_stage_lists_the_file_as_pending(tmp_path: Path) -> None:
    """解析产物与清单都还没有：不能装作成功 —— 列进 pending，让 CLI 打出来。"""
    settings = _settings(tmp_path)
    store = build_artifact_store(settings)
    path = _scan(tmp_path)

    result = _run_chunk(tmp_path, settings, store, path, ocr=None)

    assert result.chunked == 0 and result.chunks == []
    assert result.pending and "已核对" in result.pending[0].reason


# ---- 坏产物与清理 ----


def test_broken_artifact_is_treated_as_missing(tmp_path: Path) -> None:
    """产物坏了 = 没有产物：切分不会拿坏产物硬切，而是等重新解析 + 人工核对。"""
    settings = _settings(tmp_path)
    store = build_artifact_store(settings)
    path = _scan(tmp_path)
    _run_parse(tmp_path, settings, _FakeOcr(), path, store)
    doc_id = next(p.name for p in (tmp_path / "stages").iterdir())
    (tmp_path / "stages" / doc_id / "document.json").write_text("{ 不是 json", encoding="utf-8")

    result = _run_chunk(tmp_path, settings, store, path, ocr=None)

    assert result.chunked == 0
    assert [item.path for item in result.pending] == ["scan.png"]

    # 它只是缓存：重解析 + 再核对一次就又能切了
    _run_parse(tmp_path, settings, _FakeOcr(), path, store)
    assert _run_chunk(tmp_path, settings, store, path).chunked == 1


def test_drop_removes_every_artifact_of_a_document(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = build_artifact_store(settings)
    path = _scan(tmp_path)
    _run_parse(tmp_path, settings, _FakeOcr(), path, store)
    _run_chunk(tmp_path, settings, store, path)
    doc_id = next(p.name for p in (tmp_path / "stages").iterdir())

    # document.json / document.md / outline.tsv / chunks.json / chunks.md
    assert store.drop(doc_id) == 5
    assert store.path_of(doc_id, "chunks.json") is None
    assert store.drop(doc_id) == 0, "再删一次是空操作"


def test_the_list_fingerprint_ignores_comment_wording(tmp_path: Path) -> None:
    """清单指纹只取「层级 + 最大层级」：改一句注释不该让整份文档重灌一遍。"""
    settings = _settings(tmp_path)
    store = build_artifact_store(settings)
    path = _scan(tmp_path)
    _run_parse(tmp_path, settings, _FakeOcr(), path, store)
    doc_id = next(p.name for p in (tmp_path / "stages").iterdir())
    outline_path = store.path_of(doc_id, "outline.tsv")
    before = store.outline_fingerprint(doc_id)

    outline_path.write_text(
        outline_path.read_text(encoding="utf-8").replace("核对三件事", "要核对的三件事"),
        encoding="utf-8",
    )
    assert store.outline_fingerprint(doc_id) == before, "注释措辞不影响切法"

    outline_path.write_text(
        outline_path.read_text(encoding="utf-8").replace("# 最大层级: 3", "# 最大层级: 2"),
        encoding="utf-8",
    )
    assert store.outline_fingerprint(doc_id) != before, "选了别的深度就得重切"


# ---- 后端语义必须一致（2026-10-10：切到 MinIO 之后连炸两处）----


class _BucketOnly:
    """像 MinIO 那样**没有本地副本**：`local_path()` 一律给 None。

    本地盘那份 `local_path()` 从来不抛、只是给一个不存在的路径，所以"拿路径当写入口"
    这类写法在本地是通的 —— 换到桶后端才暴露。这个假桶就是用来钉住这个差别的。
    """

    backend = "minio"
    name = "bucket:test"

    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}

    def put(self, key: str, data: bytes) -> None:
        self.data[key] = data

    def get(self, key: str) -> bytes:
        if key not in self.data:
            raise BlobNotFound(f"桶里没有这个键：{key}")
        return self.data[key]

    def exists(self, key: str) -> bool:
        return key in self.data

    def size(self, key: str) -> int:
        return len(self.get(key))

    def delete(self, key: str) -> bool:
        return self.data.pop(key, None) is not None

    def delete_prefix(self, prefix: str) -> int:
        hit = [key for key in self.data if key.startswith(prefix)]
        for key in hit:
            del self.data[key]
        return len(hit)

    def keys(self, prefix: str = "") -> list[str]:
        return sorted(key for key in self.data if key.startswith(prefix))

    def local_path(self, key: str) -> None:
        return None

    def get_text(self, key: str) -> str:
        return self.get(key).decode("utf-8")

    def put_text(self, key: str, text: str) -> None:
        self.put(key, text.encode("utf-8"))


def test_writing_artifacts_goes_through_keys_not_local_paths(tmp_path: Path) -> None:
    """写产物走**键**，不走 `path_of()`（2026-10-10 修）。

    以前 `save_document` / `save_chunks` 先把 `path_of()` 的结果当下标写进去，而桶后端
    的 `path_of()` 要先 HEAD 一次 —— 新键必然"不存在"，于是切到 MinIO 之后**解析产物
    一个都写不进去**（本地盘碰巧通，因为它的 local_path 直接给路径）。
    """
    settings = _settings(tmp_path)
    bucket = _BucketOnly()
    store = ArtifactStore(bucket)
    path = _scan(tmp_path)

    result = run_parse([path], tmp_path, settings, _FakeOcr(), store)

    assert result.parsed == 1
    doc_id = bucket.keys()[0].split("/")[0]
    assert bucket.keys() == sorted(
        f"{doc_id}/{name}" for name in ("document.json", "document.md", "outline.tsv")
    )


def test_a_missing_artifact_is_none_instead_of_an_error(tmp_path: Path) -> None:
    """「这份产物还没有」是一种正常状态，三种消费方式都得能往下走（2026-10-10 那个 500）。

    实测现场：`GET /documents/{id}` → `outline_fingerprint()` → `path_of()`，本地后端
    返回一个不存在的 Path（照常返回空指纹），MinIO 后端却抛 `BlobNotFound` → 整页 500。
    所以两个后端的"没有"必须都落成 None。
    """
    store = build_artifact_store(_settings(tmp_path))

    assert store.path_of("deadbeefdeadbeef", "outline.tsv") is None
    assert store.outline_fingerprint("deadbeefdeadbeef") == ""
    assert store.load_outline("deadbeefdeadbeef", source_hash="h", parse="p") is None
    assert store.ensure_outline("deadbeefdeadbeef", source_hash="h", parse="p") is False
    assert store.load_chunks("deadbeefdeadbeef", source_hash="h", pipeline="p") is None


# ---- 吃图的 embedding 必须有图片描述 ----

_DEEP_PAGE = (
    "# 第一章 概述\n\n"
    "第一章的正文。\n\n"
    "## 1.1 小节\n\n"
    "小节的正文。\n\n"
    "### 1.1.1 更深的一层\n\n"
    "更深那层的正文。\n"
)


def test_chunk_waits_until_the_human_confirms_the_list(tmp_path: Path) -> None:
    """人工核对是硬门槛：清单生成了但没点「已核对: 是」之前，这份文档不切。"""
    settings = _settings(tmp_path)
    store = build_artifact_store(settings)
    path = _scan(tmp_path)
    run_parse([path], tmp_path, settings, _FakeOcr(), store)  # 故意不核对

    result = run_chunk([path], tmp_path, settings, None, None, _FakeOcr(), store)

    assert result.chunked == 0 and result.chunks == []
    assert [item.path for item in result.pending] == ["scan.png"]
    assert "outline.tsv" in result.pending[0].reason, "要说清楚去改哪个文件"

    _confirm(tmp_path, store)

    assert _run_chunk(tmp_path, settings, store, path).chunked == 1, "核对完就能切了"


def test_the_selected_max_level_decides_how_deep_this_document_splits(tmp_path: Path) -> None:
    """最大层级从清单里读、按文档生效：选 2 就把三级标题并进二级那一节。"""
    settings = _settings(tmp_path)
    store = build_artifact_store(settings)
    path = _scan(tmp_path)
    run_parse([path], tmp_path, settings, _FakeOcr(page=_DEEP_PAGE), store)
    _confirm(tmp_path, store, max_level=2)

    result = run_chunk([path], tmp_path, settings, None, None, _FakeOcr(page=_DEEP_PAGE), store)

    assert [c.heading_path for c in result.chunks] == ["第一章 概述", "第一章 概述 -> 1.1 小节"]
    assert "### 1.1.1 更深的一层" in result.chunks[-1].text, "更深那层留在上一节的正文里"


# ---- 吃图的 embedding 必须有图片描述 ----


class _MultimodalEmbedder:
    """只声明「我吃图」，其余照抄 _StubEmbedder。"""

    supports_images = True

    @property
    def dim(self) -> int:
        return 2

    def __init__(self) -> None:
        self.sent: list[tuple[list[str], object]] = []

    async def embed_documents(self, texts: list[str], images=None) -> Embeddings:
        self.sent.append((list(texts), images))
        return Embeddings(dense=[[1.0, 0.0] for _ in texts])

    async def embed_query(self, text: str) -> Embeddings:
        return Embeddings(dense=[[1.0, 0.0]])


class _FakeVision:
    fingerprint = "fake-vl@2"

    def describe(self, data: bytes, *, before: str = "", after: str = "", name: str = "") -> str:
        return f"这张图画的是作业图。上文说：{before[:6]}"


def test_a_multimodal_backend_without_image_descriptions_is_rejected(tmp_path: Path) -> None:
    """吃图却没有描述 —— 图会一张都进不了库（标记编码前被剥掉），必须当场挡住。"""
    settings = _settings(tmp_path, describe=False)
    store = build_artifact_store(settings)
    path = _scan(tmp_path)

    with pytest.raises(ValueError, match="describe"):
        run_chunk([path], tmp_path, settings, _MultimodalEmbedder(), None, _FakeOcr(), store)


def test_a_multimodal_backend_keeps_the_description_as_the_text(
    tmp_path: Path,
) -> None:
    """融合用的文本就是 VLM 那段描述（+标题路径）—— 图片本体由灌库那一步再附上。"""
    settings = _settings(tmp_path, describe=True)
    store = build_artifact_store(settings)
    path = _scan(tmp_path)
    _run_parse(tmp_path, settings, _FakeOcr(), path, store)

    result = run_chunk(
        [path], tmp_path, settings, _MultimodalEmbedder(), _FakeVision(), _FakeOcr(), store
    )

    images = [c for c in result.chunks if c.modality == "image"]
    assert len(images) == 1
    assert "这张图画的是作业图" in images[0].text
    assert "上文说：上面这段讲的" in images[0].text, "描述要吃到上下文"
    assert images[0].assets == [images[0].metadata["image"]], "融合时要凭它找到图片本体"
