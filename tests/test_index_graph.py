"""IndexGraph：按文件铺开跑（并发 + 单文件失败隔离），最后统一写库。

写库那一段注入替身索引与替身注册表 —— 单测不该连 Milvus。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from rag.config.schema import (
    ChunkingConfig,
    IndexingConfig,
    IngestAssetsConfig,
    IngestConfig,
    IngestImagesConfig,
    IngestStagesConfig,
    Settings,
)
from rag.graphs.index import run_index_graph
from rag.ingest.ocr import OcrPage
from rag.ingest.stages import build_artifact_store

_SRC = "imgs/img001.png"
_PNG = b"\x89PNG\r\n\x1a\nfake-png"
_PAGE = f'# 第一章 概述\n\n上面这段讲的是作业图。\n\n<img src="{_SRC}" />\n\n下面这段讲的是并行度。'


class _Ocr:
    """替身 OCR：按文件名决定成功还是报错（验「单文件失败不牵连其余」）。"""

    name = "paddleocr"

    def __init__(self, broken: set[str] | None = None) -> None:
        self._broken = broken or set()
        self.calls: list[str] = []

    def parse(self, path: Path, *, file_type: int) -> list[OcrPage]:
        self.calls.append(path.name)
        if path.name in self._broken:
            raise RuntimeError(f"{path.name} 读不动")
        return [OcrPage(page_index=0, markdown=_PAGE, images={_SRC: _PNG})]


class _Index:
    """替身索引：只记账。"""

    def __init__(self) -> None:
        self.upserted: list[str] = []

    async def ensure_ready(self) -> None:
        return None

    async def delete_docs(self, doc_ids: list[str]) -> None:
        return None

    async def upsert(self, chunks, vectors) -> int:
        self.upserted.extend(chunk.chunk_id for chunk in chunks)
        return len(chunks)


class _VersionedIndex(_Index):
    def __init__(self) -> None:
        super().__init__()
        self.batches: list[tuple[str, list[str]]] = []

    async def delete_docs(self, doc_ids: list[str]) -> None:
        raise AssertionError("双版本写入不能先删旧块")

    async def upsert_batch(self, chunks, vectors, *, batch_id: str) -> int:
        ids = [chunk.chunk_id for chunk in chunks]
        self.batches.append((batch_id, ids))
        self.upserted.extend(ids)
        return len(chunks)


class _Registry:
    def __init__(self) -> None:
        self.records: list = []
        self.batches: list[tuple[str, list[str]]] = []
        self.failures: list[tuple[str, list[str], str]] = []

    async def begin_index(self, batch_id: str, records) -> None:
        items = list(records)
        self.batches.append((batch_id, [record.doc_id for record in items]))

    async def complete_index(self, batch_id: str, records) -> int:
        items = [
            record.model_copy(
                update={
                    "stage": "indexed",
                    "index_batch_id": batch_id,
                    "index_status": "active",
                }
            )
            for record in records
        ]
        self.records.extend(items)
        return len(items)

    async def fail_index(self, batch_id: str, doc_ids: list[str], error: str) -> None:
        self.failures.append((batch_id, list(doc_ids), error))

    async def upsert(self, records) -> None:
        self.records.extend(records)

    async def delete(self, doc_ids) -> None:
        return None


class _Embedder:
    supports_images = False
    name = "stub"
    dim = 2

    async def embed_documents(self, texts, images=None):
        from rag.contracts import Embeddings

        return Embeddings(dense=[[1.0, 0.0] for _ in texts])


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        indexing=IndexingConfig(chunk=ChunkingConfig(strategy="markdown", size=1000, overlap=0)),
        ingest=IngestConfig(
            assets=IngestAssetsConfig(dir=str(tmp_path / "assets")),
            images=IngestImagesConfig(describe=False, cache_path=str(tmp_path / "cache.json")),
            stages=IngestStagesConfig(dir=str(tmp_path / "stages")),
        ),
    )


def _scan(tmp_path: Path, name: str) -> Path:
    target = tmp_path / name
    target.write_bytes(_PNG)
    return target


def _confirm_all(tmp_path: Path) -> None:
    """把每份清单都标成「已核对」—— 闸门是切分的前置条件。"""
    for outline in (tmp_path / "stages").glob("*/outline.tsv"):
        outline.write_text(
            outline.read_text(encoding="utf-8").replace("# 已核对: 否", "# 已核对: 是"),
            encoding="utf-8",
        )


def _run(
    settings,
    files,
    tmp_path,
    *,
    ocr=None,
    concurrency=None,
    index=None,
    registry=None,
):
    store = build_artifact_store(settings)
    index = index or _Index()
    registry = registry or _Registry()
    result = asyncio.run(
        run_index_graph(
            settings,
            todo=files,
            root=tmp_path,
            ocr=ocr or _Ocr(),
            vision=None,
            store=store,
            embedder=_Embedder(),
            concurrency=concurrency,
            index=index,
            registry=registry,
        )
    )
    return result, index, registry, store


def test_each_file_is_parsed_and_chunked_on_its_own(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    files = [_scan(tmp_path, "a.png"), _scan(tmp_path, "b.png")]

    # 先跑一次把清单生成出来，只核对其中一份 —— 另一份应当进 pending
    result, _, _, _ = _run(settings, files, tmp_path)
    assert result.pending and len(result.pending) == 2, "两份都没核对 → 都不切"
    assert result.chunked == []

    outlines = sorted((tmp_path / "stages").glob("*/outline.tsv"))
    outlines[0].write_text(
        outlines[0].read_text(encoding="utf-8").replace("# 已核对: 否", "# 已核对: 是"),
        encoding="utf-8",
    )

    result, index, registry, _ = _run(settings, files, tmp_path)

    assert len(result.chunked) == 1, "核对过的那份才切"
    assert len(result.pending) == 1
    assert index.upserted, "切出来的块写进了库"
    assert len(registry.records) == 1
    record = registry.records[0]
    assert record.index_status == "active"
    assert record.index_batch_id
    assert registry.batches == [(record.index_batch_id, [record.doc_id])]


def test_one_broken_file_does_not_stop_the_others(tmp_path: Path) -> None:
    """一个坏文件只影响它自己 —— 这是按文件铺开的主要理由。"""
    settings = _settings(tmp_path)
    files = [_scan(tmp_path, "good.png"), _scan(tmp_path, "bad.png")]
    ocr = _Ocr(broken={"bad.png"})

    result, index, registry, _ = _run(settings, files, tmp_path, ocr=ocr)

    _confirm_all(tmp_path)
    result, index, registry, _ = _run(settings, files, tmp_path, ocr=ocr)

    assert [item.path for item in result.skipped] == ["bad.png"], "坏文件记进 skipped"
    assert "读不动" in result.skipped[0].reason
    assert result.chunked == ["good.png"], "好文件照常落库"
    assert index.upserted and len(registry.records) == 1


def test_nothing_to_do_is_not_an_error(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    result, _, _, _ = _run(settings, [], tmp_path)

    assert result.chunked == [] and result.written == 0


def test_versioned_write_keeps_the_old_batch_until_complete(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    files = [_scan(tmp_path, "a.png")]
    _run(settings, files, tmp_path)
    _confirm_all(tmp_path)
    index, registry = _VersionedIndex(), _Registry()

    result, _, _, _ = _run(settings, files, tmp_path, index=index, registry=registry)

    assert result.written > 0
    assert index.batches and registry.records[0].index_status == "active"
    assert registry.batches[0][0] == index.batches[0][0]


def test_a_missing_artifact_keeps_the_whole_batch_out_of_the_index(tmp_path: Path) -> None:
    """缺切分产物时**整批不写**（规矩来自 commit.py）：只写一部分会让库里新旧混杂。

    直接测 `commit_chunks`：经由图跑的路径会先把产物补出来（图就是为了不让人手工补），
    所以「产物缺失」这个分支要用最底层的写库函数来验。
    """
    from rag.indexing.commit import commit_chunks

    settings = _settings(tmp_path)
    store = build_artifact_store(settings)
    files = [_scan(tmp_path, "a.png")]  # 只落了源文件，没有解析/切分产物
    index, registry = _Index(), _Registry()

    result = asyncio.run(
        commit_chunks(
            settings,
            store,
            [(path, tmp_path) for path in files],
            embedder=_Embedder(),
            index=index,
            registry=registry,
        )
    )

    assert index.upserted == [], "一件都没写"
    assert registry.records == []
    assert result.missing == ["a.png"], "并且明确说出缺哪一份"
