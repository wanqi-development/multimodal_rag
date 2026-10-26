"""HTML 里的图片：`<img>` → 正文标记 + 图片本体落 assets。

出口与 docx / OCR 两条路**完全一致**（``![image](asset:…)``），下游的图片描述（VLM）
与多模态 embedding 不必分来源。取不到的图只记账、不废掉整页 —— 网页里混着图标、徽标、
外链图是常态，一张 404 不该让整份文档 skipped。

全程本地：图片从 HTML 文件所在目录读，loader 不联网。
"""

from __future__ import annotations

from pathlib import Path

from rag.config.schema import IngestAssetsConfig, IngestConfig
from rag.ingest.assets import markers_in
from rag.ingest.loaders import build_loader

_PNG = b"\x89PNG\r\n\x1a\nfake-png"


def _cfg(tmp_path: Path) -> IngestConfig:
    return IngestConfig(assets=IngestAssetsConfig(dir=str(tmp_path / "assets")))


def _load(tmp_path: Path, body: str, *, name: str = "page.html"):
    target = tmp_path / name
    target.write_text(f"<html><body>{body}</body></html>", encoding="utf-8")
    return build_loader(".html", _cfg(tmp_path)).load(target, root=tmp_path)


def _png(tmp_path: Path, name: str = "imgs/a.png") -> Path:
    target = tmp_path / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_PNG)
    return target


def test_a_relative_image_becomes_a_marker_and_an_asset(tmp_path: Path) -> None:
    _png(tmp_path)

    doc = _load(tmp_path, '<p>上面的一段。</p><img src="imgs/a.png"><p>下面的一段。</p>')

    before, after = doc.text.split("![image](asset:")
    assert "上面的一段。" in before and "下面的一段。" in after, "标记留在正文里的原位"
    assert len(doc.images) == 1
    assert doc.metadata["image_count"] == 1
    asset = tmp_path / "assets" / doc.images[0].asset_uri
    assert asset.read_bytes() == _PNG, "图片本体落到了 assets 目录"
    assert doc.images[0].asset_uri.startswith(f"{doc.doc_id}/"), "按 doc_id 分目录"


def test_the_same_image_twice_saves_one_asset(tmp_path: Path) -> None:
    """同一张图出现两次只落一份 —— 多落一份就等于同一张图被 VLM 描述两遍（那是花钱的）。"""
    _png(tmp_path)

    doc = _load(tmp_path, '<img src="imgs/a.png"><p>x</p><img src="imgs/a.png">')

    assert len(markers_in(doc.text)) == 2, "两处标记都在"
    assert len(doc.images) == 1, "但图只有一份"
    assert len(list((tmp_path / "assets" / doc.doc_id).iterdir())) == 1


def test_a_lazy_loaded_image_is_taken_from_data_src(tmp_path: Path) -> None:
    _png(tmp_path)

    doc = _load(tmp_path, '<img src="placeholder.gif" data-src="imgs/a.png">')

    assert len(markers_in(doc.text)) == 1, "懒加载站点的图也要拿到"


def test_a_remote_image_is_noted_without_failing_the_page(tmp_path: Path) -> None:
    """外链的图（抓取时没下到本地）只记账：整页照常解析，缺哪张写在 metadata 里。"""
    doc = _load(tmp_path, '<p>正文还在。</p><img src="https://cdn.example.com/a.png">')

    assert "正文还在。" in doc.text and markers_in(doc.text) == []
    assert any("外链" in item for item in doc.metadata["images_missing"])


def test_a_missing_file_and_a_vector_image_are_noted(tmp_path: Path) -> None:
    doc = _load(tmp_path, '<img src="imgs/nope.png"><img src="imgs/logo.svg">')

    reasons = doc.metadata["images_missing"]
    assert any("本地没有这个文件" in item for item in reasons)
    assert any("不是能进库的图片格式" in item for item in reasons)


def test_an_image_path_cannot_escape_the_page_directory(tmp_path: Path) -> None:
    """HTML 是不可信的输入：`../../` 这种别让它读到语料目录外面去。"""
    outside = tmp_path.parent / "secret.png"
    outside.write_bytes(_PNG)
    page_dir = tmp_path / "docs"
    page_dir.mkdir()
    (page_dir / "page.html").write_text('<img src="../../secret.png">', encoding="utf-8")

    doc = build_loader(".html", _cfg(tmp_path)).load(page_dir / "page.html", root=page_dir)

    assert markers_in(doc.text) == []
    assert any("路径跑到" in item for item in doc.metadata["images_missing"])


def test_images_inside_the_navigation_are_not_taken(tmp_path: Path) -> None:
    """导航/页脚里的是图标，不是正文 —— 与文字那套「先清噪声」保持一致。"""
    _png(tmp_path)

    doc = _load(tmp_path, '<nav><img src="imgs/a.png"></nav><p>正文。</p>')

    assert markers_in(doc.text) == []
    assert "images_missing" not in doc.metadata, "连提都不提：它压根不算正文里的图"


def test_an_image_without_a_source_is_noted(tmp_path: Path) -> None:
    doc = _load(tmp_path, "<p>x</p><img>")

    assert any("既没有 src" in item for item in doc.metadata["images_missing"])
