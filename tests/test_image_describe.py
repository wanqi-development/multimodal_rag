"""图片描述：标记 → 上下文 → VLM → 一条 ``modality=image`` 的块。

VLM 用假客户端（不碰网络、不花钱）；OCR 那条路沿用 test_locate 里的假客户端。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.config.schema import (
    ChunkingConfig,
    IndexingConfig,
    IngestAssetsConfig,
    IngestConfig,
    IngestImagesConfig,
    ProvidersConfig,
    Settings,
    VisionProviderConfig,
)
from rag.contracts import Document, Embeddings, ImageAsset
from rag.ingest import prepare
from rag.ingest.chunkers import build_chunker
from rag.ingest.images import describe_images
from rag.ingest.ocr import OcrPage
from rag.providers.vision import VisionError

_SRC = "imgs/img_in_image_box_69_91_1085_722.png"
_PNG = b"\x89PNG\r\n\x1a\nfake-png-bytes"
_ABOVE = "上面这段讲的是 Flink 的作业图。"
_BELOW = "下面这段讲的是并行度。"
_PAGE = (
    "# 第一章 概述\n\n"
    f"{_ABOVE}\n\n"
    f'<div style="text-align: center;"><img src="{_SRC}" alt="Image" /></div>\n\n'
    f"{_BELOW}"
)


class _FakeOcr:
    name = "paddleocr"  # 真实 client 都会报这个名字（工厂约定的一部分）

    def __init__(self, *pages: tuple[str, dict[str, bytes]]) -> None:
        self._pages = list(pages)

    def parse(self, path: Path, *, file_type: int) -> list[OcrPage]:
        return [
            OcrPage(page_index=i, markdown=markdown, images=images)
            for i, (markdown, images) in enumerate(self._pages)
        ]


class _StubEmbedder:
    @property
    def dim(self) -> int:
        return 2

    async def embed_documents(self, texts: list[str]) -> Embeddings:
        return Embeddings(dense=[[1.0, 0.0] for _ in texts])

    async def embed_query(self, text: str) -> Embeddings:
        return Embeddings(dense=[[1.0, 0.0]])


class _FakeVision:
    """只实现 describe_images 要用的两个成员：fingerprint / describe。"""

    def __init__(
        self, *, model: str = "fake-vl", boom: str | None = None, boom_at: int | None = None
    ) -> None:
        self.calls: list[dict[str, str]] = []
        self._fingerprint = f"{model}@1"
        self._boom = boom
        self._boom_at = boom_at

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def describe(self, data: bytes, *, before: str = "", after: str = "", name: str = "") -> str:
        self.calls.append({"name": name, "before": before, "after": after, "size": str(len(data))})
        if self._boom and (self._boom_at is None or self._boom_at == len(self.calls) - 1):
            raise VisionError(self._boom)
        return f"这是 {name} 的描述。"


def _settings(
    tmp_path: Path,
    *,
    describe: bool = True,
    size: int = 1000,
    context_chars: int = 600,
    model: str = "fake-vl",
    max_images: int = 0,
) -> Settings:
    # markdown 策略才会带上标题路径（recursive 不管标题），而标题是描述块要继承的东西
    return Settings(
        providers=ProvidersConfig(vision=VisionProviderConfig(model=model)),
        indexing=IndexingConfig(chunk=ChunkingConfig(strategy="markdown", size=size, overlap=0)),
        ingest=IngestConfig(
            assets=IngestAssetsConfig(dir=str(tmp_path / "assets")),
            images=IngestImagesConfig(
                describe=describe,
                context_chars=context_chars,
                max_images=max_images,
                cache_path=str(tmp_path / "cache.json"),
            ),
        ),
    )


def _prepare(tmp_path: Path, settings: Settings, vision: object | None):
    scan = tmp_path / "scan.png"
    scan.write_bytes(_PNG)
    return prepare(scan, settings, _FakeOcr((_PAGE, {_SRC: _PNG})), _StubEmbedder(), vision)


def _image_chunks(chunks: list) -> list:
    return [c for c in chunks if c.modality == "image"]


# ---- 主链路 ----


def test_describe_off_adds_nothing(tmp_path: Path) -> None:
    """开关关着 = 一次请求都没有（VLM 按量计费）。"""
    prep = _prepare(tmp_path, _settings(tmp_path, describe=False), None)

    assert _image_chunks(prep.chunks) == []
    assert prep.images_described == 0
    assert prep.skipped == []


def test_each_image_becomes_its_own_chunk(tmp_path: Path) -> None:
    prep = _prepare(tmp_path, _settings(tmp_path), _FakeVision())
    described = _image_chunks(prep.chunks)

    assert len(described) == 1, "有标记的图应当生成一条独立记录"
    chunk = described[0]
    assert chunk.doc_id == prep.chunks[0].doc_id
    assert chunk.modality == "image"
    assert chunk.content_hash and chunk.chunk_id.endswith(f"{chunk.index:04d}")
    assert prep.images_described == 1
    assert prep.skipped == []


def test_description_block_points_at_the_image_itself(tmp_path: Path) -> None:
    """页/行取**标记自己**那行，标题路径继承所在块 —— 各取各自知道得最准的那份。"""
    prep = _prepare(tmp_path, _settings(tmp_path), _FakeVision())
    owner = next(c for c in prep.chunks if c.modality == "text")
    chunk = _image_chunks(prep.chunks)[0]

    assert chunk.heading_path == owner.heading_path == "第一章 概述"
    assert chunk.heading_numbers == owner.heading_numbers == "1"
    # 页内第 5 行是那个图片标记（块首在第 3 行）—— 能定位到行就不退化成块首
    assert (chunk.page, chunk.line) == (1, 5)
    assert owner.line == 3


def test_image_chunk_assets_point_at_the_file_on_disk(tmp_path: Path) -> None:
    prep = _prepare(tmp_path, _settings(tmp_path), _FakeVision())
    chunk = _image_chunks(prep.chunks)[0]

    assert len(chunk.assets) == 1
    assert (tmp_path / "assets" / chunk.assets[0]).read_bytes() == _PNG
    assert chunk.metadata["image"] == chunk.assets[0]


def test_body_is_heading_then_description_then_image_address(tmp_path: Path) -> None:
    """布局与正文块一致：标题贴开头，正文（这里是描述）在后面。"""
    prep = _prepare(tmp_path, _settings(tmp_path), _FakeVision())
    chunk = _image_chunks(prep.chunks)[0]
    uri = chunk.assets[0]
    heading, body = chunk.text.split("\n\n", 1)
    description, address = body.split("\n")

    assert heading == "第一章 概述"
    assert description == f"这是 {uri} 的描述。"
    assert address == f"![image](asset:{uri})"


def test_image_index_continues_after_the_text_chunks(tmp_path: Path) -> None:
    """不许改已有块的序号 —— 那会让前后两次灌库的 chunk_id 错位。"""
    prep = _prepare(tmp_path, _settings(tmp_path), _FakeVision())
    text_indexes = [c.index for c in prep.chunks if c.modality == "text"]
    image_indexes = [c.index for c in _image_chunks(prep.chunks)]

    assert image_indexes == [max(text_indexes) + 1]


# ---- 上下文 ----


def test_context_is_the_paragraph_above_and_below(tmp_path: Path) -> None:
    vision = _FakeVision()
    _prepare(tmp_path, _settings(tmp_path), vision)

    assert len(vision.calls) == 1
    assert _ABOVE in vision.calls[0]["before"]
    assert _BELOW in vision.calls[0]["after"]
    assert "第一章 概述" not in vision.calls[0]["before"], "标题行不是正文段落"


def test_context_is_clipped_towards_the_image(tmp_path: Path) -> None:
    """截断留靠近图片的那一头：远端那几句说明不了这张图在讲什么。"""
    vision = _FakeVision()
    _prepare(tmp_path, _settings(tmp_path, context_chars=8), vision)

    assert vision.calls[0]["before"] == _ABOVE[-8:]
    assert vision.calls[0]["after"] == _BELOW[:8]


def test_zero_context_chars_means_no_context(tmp_path: Path) -> None:
    vision = _FakeVision()
    _prepare(tmp_path, _settings(tmp_path, context_chars=0), vision)

    assert vision.calls[0]["before"] == ""
    assert vision.calls[0]["after"] == ""


def test_max_images_caps_the_run(tmp_path: Path) -> None:
    """先花几毛钱看效果：跑几张确认提示词对路，再决定全量。"""
    markdown = (
        f'第一段。\n\n<img src="{_SRC}" />\n\n第二段。\n\n<img src="imgs/second.png" />\n\n第三段。'
    )
    scan = tmp_path / "scan.png"
    scan.write_bytes(_PNG)
    vision = _FakeVision()

    prep = prepare(
        scan,
        _settings(tmp_path, max_images=1),
        _FakeOcr((markdown, {_SRC: _PNG, "imgs/second.png": _PNG})),
        _StubEmbedder(),
        vision,
    )

    assert len(vision.calls) == 1
    assert prep.images_described == 1


def test_max_images_does_not_count_failed_images(tmp_path: Path) -> None:
    """失败的不占名额 —— 占了等于「说好跑 3 张，却只看得到 1 张结果」。"""
    markdown = f'<img src="{_SRC}" />\n\n第一段。\n\n<img src="imgs/second.png" />\n\n第二段。'
    scan = tmp_path / "scan.png"
    scan.write_bytes(_PNG)

    prep = prepare(
        scan,
        _settings(tmp_path, max_images=1),
        _FakeOcr((markdown, {_SRC: _PNG, "imgs/second.png": _PNG})),
        _StubEmbedder(),
        _FakeVision(boom="第一张炸了", boom_at=0),
    )

    assert prep.images_described == 1, "第一张失败后，第二张仍要占那一个名额"
    assert len(prep.skipped) == 1


# ---- 缓存：改一段正文不该让几十张图全部重问 ----


def test_second_run_uses_the_cache(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    first = _prepare(tmp_path, settings, _FakeVision())
    cache = json.loads((tmp_path / "cache.json").read_text(encoding="utf-8"))

    # 第二次故意用一个「一调就炸」的客户端：描述还能出来，就证明没有真的去调
    second = _prepare(tmp_path, settings, _FakeVision(boom="不该被调用"))

    assert len(cache) == 1
    assert _image_chunks(second.chunks)[0].text == _image_chunks(first.chunks)[0].text
    assert second.skipped == []


def test_cache_key_follows_model_and_prompt_version(tmp_path: Path) -> None:
    """换了模型（或提示词版本）旧描述就不算数 —— 缓存键里带着它们。"""
    _prepare(tmp_path, _settings(tmp_path), _FakeVision(model="fake-vl"))
    vision = _FakeVision(model="other-vl")
    _prepare(tmp_path, _settings(tmp_path, model="other-vl"), vision)

    assert len(vision.calls) == 1, "换模型后应当重新问一次"
    assert len(json.loads((tmp_path / "cache.json").read_text(encoding="utf-8"))) == 2


# ---- 出错时的行为 ----


def test_one_bad_image_does_not_take_the_document_down(tmp_path: Path) -> None:
    """十几张图里挂一张，为它丢掉另外十几张的描述是更坏的结果。"""
    markdown = (
        "# 两台机器\n\n"
        "第一张图讲的是作业图。\n\n"
        f'<img src="{_SRC}" />\n\n'
        "第二张图讲的是并行度。\n\n"
        '<img src="imgs/second.png" />'
    )
    scan = tmp_path / "scan.png"
    scan.write_bytes(_PNG)
    prep = prepare(
        scan,
        _settings(tmp_path),
        _FakeOcr((markdown, {_SRC: _PNG, "imgs/second.png": _PNG})),
        _StubEmbedder(),
        # 只让第一次调用失败：第二张图照旧要有描述，文档也要照旧进库
        _FakeVision(boom="连不上 VLM", boom_at=0),
    )

    assert prep.chunks, "正文块照旧"
    assert len(_image_chunks(prep.chunks)) == 1, "第二张图不受影响"
    assert len(prep.skipped) == 1, "只有失败的那张记一条原因"
    assert all("连不上 VLM" in item.reason for item in prep.skipped)
    assert prep.documents == 1, "文档本身仍然进库"


def test_failed_image_is_named_by_its_asset_uri(tmp_path: Path) -> None:
    prep = _prepare(tmp_path, _settings(tmp_path), _FakeVision(boom="429 限流"))

    assert [item.reason for item in prep.skipped] == ["VLM 调用失败：429 限流"]
    assert prep.skipped[0].path in prep.chunks[0].assets


def test_missing_asset_file_is_reported_not_sent_to_the_model(tmp_path: Path) -> None:
    """资产不在了就说清楚，别去问 VLM 一张读不到的图。"""
    marker = "![image](asset:doc1/gone.png)"
    doc = Document(
        doc_id="doc1",
        source_uri="scan.pdf",
        text=f"{_ABOVE}\n\n{marker}\n\n{_BELOW}",
        images=[ImageAsset(marker=marker, asset_uri="doc1/gone.png", page=2)],
    )
    chunks = build_chunker(_settings(tmp_path).indexing.chunk).split([doc])
    vision = _FakeVision()

    described, failed = describe_images([doc], chunks, _settings(tmp_path), vision)

    assert described == []
    assert vision.calls == []
    assert "资产文件不存在" in failed[0].reason


def test_describe_on_without_a_client_is_an_error(tmp_path: Path) -> None:
    """开了开关却没有客户端 = 配置装配漏了，别静默退化成「一张都不描述」。"""
    doc = Document(doc_id="doc1", source_uri="s.pdf", text="正文。")
    chunks = build_chunker(_settings(tmp_path).indexing.chunk).split([doc])

    with pytest.raises(ValueError, match="build_vision"):
        describe_images([doc], chunks, _settings(tmp_path), None)


def test_broken_cache_file_is_ignored(tmp_path: Path) -> None:
    """缓存只是省钱的手段，不是数据 —— 坏了当作空的，别让整次 ingest 失败。"""
    (tmp_path / "cache.json").write_text("{ not json", encoding="utf-8")

    prep = _prepare(tmp_path, _settings(tmp_path), _FakeVision())

    assert prep.images_described == 1
    assert len(json.loads((tmp_path / "cache.json").read_text(encoding="utf-8"))) == 1
