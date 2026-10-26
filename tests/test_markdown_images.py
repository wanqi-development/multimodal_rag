"""markdown 里的图片：`![图](本地相对路径)` → 正文标记 + 图片本体落 assets。

为什么单独钉这一条（2026-10-10）：网页交给 Firecrawl 之后落盘件是 `.md`（见
`CHANGELOG.md` 同日的 Firecrawl 那条）。不换成标记的话，图**只在盘上**：assets →
VLM 描述 → 多模态 embedding 那条线全断，等于切到 Firecrawl 之后图用不上。
出口与 html / docx / OCR 三条路完全一致（`![image](asset:…)`）。

全程本地：图片从 markdown 文件所在目录读，loader 不联网。
"""

from __future__ import annotations

from pathlib import Path

from rag.config.schema import IngestAssetsConfig, IngestConfig
from rag.ingest.assets import markers_in
from rag.ingest.loaders import build_loader

_PNG = b"\x89PNG\r\n\x1a\nfake-png"


def _cfg(tmp_path: Path) -> IngestConfig:
    return IngestConfig(assets=IngestAssetsConfig(dir=str(tmp_path / "assets")))


def _load(tmp_path: Path, body: str, *, name: str = "page.md"):
    target = tmp_path / name
    target.write_text(body, encoding="utf-8")
    return build_loader(Path(name).suffix, _cfg(tmp_path)).load(target, root=tmp_path)


def _png(tmp_path: Path, name: str = "page-abc_files/img01.png") -> Path:
    target = tmp_path / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_PNG)
    return target


def test_a_local_image_becomes_a_marker_and_an_asset(tmp_path: Path) -> None:
    """抓取那条路把图下到 `<真名>-<哈希>_files/`，这里把它换进资产库。"""
    _png(tmp_path)

    doc = _load(
        tmp_path, "# 标题\n\n上面的一段。\n\n![图](page-abc_files/img01.png)\n\n下面的一段。"
    )

    before, after = doc.text.split("![image](asset:")
    assert "上面的一段。" in before and "下面的一段。" in after, "标记留在正文里的原位"
    assert "page-abc_files" not in doc.text, "原来的相对路径不该留在正文里"
    assert len(doc.images) == 1 and doc.metadata["image_count"] == 1
    asset = tmp_path / "assets" / doc.images[0].asset_uri
    assert asset.read_bytes() == _PNG, "图片本体落到了 assets 目录"
    assert doc.images[0].asset_uri.startswith(f"{doc.doc_id}/"), "按 doc_id 分目录"
    assert markers_in(doc.text) == [doc.images[0].marker]


def test_the_same_image_twice_saves_one_asset(tmp_path: Path) -> None:
    """同一张图出现两次只落一份 —— 多落一份就等于同一张图被 VLM 描述两遍（花钱）。"""
    _png(tmp_path)

    doc = _load(
        tmp_path, "![a](page-abc_files/img01.png)\n\n中间\n\n![a](page-abc_files/img01.png)"
    )

    assert len(doc.images) == 1
    assert doc.text.count("![image](asset:") == 2, "两处都要有标记"
    assert len(list((tmp_path / "assets").rglob("*.png"))) == 1


def test_a_missing_file_is_noted_not_fatal(tmp_path: Path) -> None:
    """图不在了只记账、不废掉整份文档 —— 与 html 那条同一个立场。"""
    doc = _load(tmp_path, "# 标题\n\n![图](page-abc_files/nope.png)\n")

    assert doc.images == []
    assert "本地没有这个文件" in doc.metadata["images_missing"][0]
    assert "page-abc_files/nope.png" in doc.text, "拿不到就别动它（删掉等于悄悄丢图）"


def test_a_path_outside_the_document_folder_is_refused(tmp_path: Path) -> None:
    """markdown 是不可信输入：`../..` 别让它读到语料目录外面去。"""
    outside = tmp_path.parent / "secret.png"
    outside.write_bytes(_PNG)
    try:
        doc = _load(tmp_path, "![x](../../secret.png)\n")
    finally:
        outside.unlink(missing_ok=True)

    assert doc.images == []
    assert "目录外面" in doc.metadata["images_missing"][0]


def test_external_urls_and_plain_text_files_are_left_alone(tmp_path: Path) -> None:
    """外链本机读不到（fetch 那条路本不该留外链）；`.txt` 里的 `![x](y)` 是数据，别动。"""
    remote = _load(tmp_path, "![x](https://example.com/a.png)\n")
    plain = _load(tmp_path, "![x](page-abc_files/img01.png)\n", name="data.txt")

    assert remote.images == [] and "https://example.com/a.png" in remote.text
    assert plain.images == [] and "page-abc_files/img01.png" in plain.text
    assert "image_count" not in plain.metadata


def test_the_loader_cleans_the_heading_anchors_it_reads(tmp_path: Path) -> None:
    """loader 出来就该是干净的：锚点、零宽字符都不留（2026-10-10）。

    页面把标题**当纯文本**显示（`{{ segment.text }}`），所以标题行里带什么就会原样露出来 ——
    清洗必须发生在解析这一步，而不是渲染那一步。
    """
    doc = _load(
        tmp_path,
        "## [\u200b](https://docs.example.com/a#step-1)  Step 1：装依赖\n\n正文。\n",
        name="page.md",
    )

    assert doc.text.startswith("## Step 1：装依赖")
    assert "docs.example.com" not in doc.text and "\u200b" not in doc.text


def test_the_loader_reports_the_document_title(tmp_path: Path) -> None:
    """文档标题（第一个一级标题）落在 `metadata["title"]`：将来列表/引用要显示文档名时用它。"""
    doc = _load(tmp_path, "# Deep Agents overview\n\n正文。\n")

    assert doc.metadata["title"] == "Deep Agents overview"


def test_a_document_without_a_title_has_no_title_field(tmp_path: Path) -> None:
    """没有一级标题就别硬编一个 —— 空字段比猜一个强。"""
    doc = _load(tmp_path, "## 只有二级标题\n\n正文。\n")

    assert "title" not in doc.metadata


def test_a_lone_image_becomes_a_shell_chunk_not_a_text_block(tmp_path: Path) -> None:
    """整篇只有一张图时，那块正文**没有可检索的文字**，该按空壳丢掉。

    这条钉的是「标记要**独立成段**」：把 `![图](…)` 留着不换（或只换里面的地址）会拼出
    `![图](![image](asset:…))`，`is_shell_chunk` 剥掉标记后还剩 `![图]()` → 判不出空壳，
    于是一个没有内容的块进库、占召回名额。图片在库里的入口是**描述块**（`describe_images`
    生成），不是这条空壳。
    """
    from rag.config.schema import Settings
    from rag.ingest.pipeline import build_chunks

    _png(tmp_path)
    target = tmp_path / "only.md"
    target.write_text("![图](page-abc_files/img01.png)\n", encoding="utf-8")
    settings = Settings()
    settings.ingest.assets = IngestAssetsConfig(dir=str(tmp_path / "assets"))
    doc = build_loader(".md", settings.ingest).load(target, root=tmp_path)

    built = build_chunks([doc], settings, None, None)

    assert doc.text.strip() == doc.images[0].marker, "标记独立成段、原语法不留"
    assert built.chunks == [] and built.shells == 1
