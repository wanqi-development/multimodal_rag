"""dots.ocr 作为 PDF 解析后端：接口、页序、失败处理，以及它内联的图片。

不碰网络：把 dots_ocr 样例里的四个函数（渲染 / 推理 / 后处理 / 拼 markdown）都换成假的，
验的是我们这一层的逻辑 —— 页序、错误处理、data URI 图片落到资产目录。
"""

from __future__ import annotations

import base64
from pathlib import Path

import pymupdf
import pytest

from rag.config.schema import IngestAssetsConfig, IngestConfig, IngestDotsConfig, IngestPdfConfig
from rag.ingest import dots as dots_module
from rag.ingest.assets import AssetStore, extract_images
from rag.ingest.loaders.pdf import PdfLoader
from rag.ingest.ocr_client import build_ocr_engine
from rag.storage import LocalBlobStore

# 1x1 的 PNG，够验「base64 解出来、落盘、换成标记」
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)

_CELLS = [
    {"bbox": [10, 10, 100, 40], "category": "Section-header", "text": "# 1第一章 概述"},
    {"bbox": [10, 50, 300, 90], "category": "Text", "text": "正文第一段。"},
    {"bbox": [10, 100, 300, 200], "category": "Picture", "text": ""},
]


class _FakeOcrStack:
    """替掉 dots_ocr 样例里的四个函数。"""

    def __init__(
        self,
        *,
        pages: int = 2,
        response: str | None = "[]",
        json_failed: bool = False,
        fail_times: int = 0,
    ):
        self.pages = pages
        self.response = response
        self.json_failed = json_failed
        # 前 N 次调用返回空响应（模拟隧道抖动），用来验重试
        self.fail_times = fail_times
        self.seen_pages: list[int] = []
        self.model_names: list[str] = []
        self.calls = 0

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def load_images(path: str, dpi: int = 200) -> list[str]:
            return [f"page-{i}" for i in range(self.pages)]

        def fetch_image(image: object, *args: object, **kwargs: object) -> str:
            return str(image)

        def inference(image: object, prompt: str, **kwargs: object) -> str | None:
            self.calls += 1
            if self.calls <= self.fail_times:
                return None
            self.seen_pages.append(int(str(image).removeprefix("page-")))
            self.model_names.append(str(kwargs.get("model_name")))
            return self.response

        def post_process(
            response: str, prompt_mode: str, origin: object, page: object, **kwargs: object
        ):
            return (response if self.json_failed else list(_CELLS)), self.json_failed

        def to_md(image: object, cells: list, text_key: str = "text", **kwargs: object) -> str:
            return "\n\n".join(cell["text"] for cell in cells if cell.get("text"))

        monkeypatch.setattr(dots_module, "load_images_from_pdf", load_images)
        monkeypatch.setattr(dots_module, "fetch_image", fetch_image)
        monkeypatch.setattr(dots_module, "inference_with_vllm", inference)
        monkeypatch.setattr(dots_module, "post_process_output", post_process)
        monkeypatch.setattr(dots_module, "layoutjson2md", to_md)


def _real_pdf(tmp_path: Path, pages: int = 2) -> Path:
    """def 用真 PDF：loader 会先用 pymupdf 验一遍文件合不合法。"""
    target = tmp_path / "doc.pdf"
    with pymupdf.open() as pdf:
        for _ in range(pages):
            pdf.new_page().insert_text((72, 72), "hello")
        pdf.save(target)
    return target


# ---- 客户端 ----


def test_parse_returns_one_page_per_pdf_page_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack = _FakeOcrStack(pages=3)
    stack.install(monkeypatch)
    client = dots_module.build_dots(IngestDotsConfig(model="dots_ocr"))

    pages = client.parse(_real_pdf(tmp_path, 3), file_type=0)

    assert [page.page_index for page in pages] == [0, 1, 2]
    assert all("# 1第一章 概述" in page.markdown for page in pages)
    assert sorted(stack.seen_pages) == [0, 1, 2], "每一页都要调一次"
    assert set(stack.model_names) == {"dots_ocr"}


def test_empty_response_fails_the_document(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """服务连不上时 `inference_with_vllm` 返回 None —— 不能安静地出一份空正文。"""
    _FakeOcrStack(response=None).install(monkeypatch)
    client = dots_module.build_dots(IngestDotsConfig())

    with pytest.raises(dots_module.DotsOcrError, match="空响应"):
        client.parse(_real_pdf(tmp_path), file_type=0)


def test_transient_failure_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """单页抖动重试后就该过 —— 几百页的 PDF 不该因为一次抖动白跑。"""
    stack = _FakeOcrStack(pages=1, fail_times=1)
    stack.install(monkeypatch)
    client = dots_module.build_dots(IngestDotsConfig(max_retries=2))

    pages = client.parse(_real_pdf(tmp_path, 1), file_type=0)

    assert "# 1第一章 概述" in pages[0].markdown
    assert stack.calls == 2, "第一次失败、第二次成功"
    assert "重试" in capsys.readouterr().err


def test_retries_are_exhausted_before_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack = _FakeOcrStack(pages=1, fail_times=99)
    stack.install(monkeypatch)
    client = dots_module.build_dots(IngestDotsConfig(max_retries=1))

    with pytest.raises(dots_module.DotsOcrError, match="重试 1 次仍失败"):
        client.parse(_real_pdf(tmp_path, 1), file_type=0)

    assert stack.calls == 2, "第一次 + 重试一次"


def test_malformed_json_falls_back_to_plain_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """模型没按 JSON 输出时收下清洗后的文本（与样例的处理一致），并在 stderr 出声。"""
    _FakeOcrStack(response="一堆不是 JSON 的文字", json_failed=True).install(monkeypatch)
    client = dots_module.build_dots(IngestDotsConfig())

    pages = client.parse(_real_pdf(tmp_path, 1), file_type=0)

    assert pages[0].markdown == "一堆不是 JSON 的文字"
    assert "布局 JSON 解析失败" in capsys.readouterr().err


def test_unknown_prompt_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="prompt_mode"):
        dots_module.build_dots(IngestDotsConfig(prompt_mode="prompt_不存在"))


def test_missing_plugin_fails_with_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """核心包没装可选插件时，错误必须直接告诉人装哪一条。"""
    monkeypatch.setattr(
        dots_module, "_IMPORT_ERROR", ImportError("No module named rag_dots_parser")
    )

    with pytest.raises(dots_module.DotsOcrError, match="pip install -e plugins/rag-dots-parser"):
        dots_module.build_dots(IngestDotsConfig())


# ---- loader 串起来 ----


def test_pdf_loader_routes_to_dots_and_keeps_positions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _FakeOcrStack(pages=2).install(monkeypatch)
    cfg = IngestConfig(
        pdf=IngestPdfConfig(parser="dots"),
        assets=IngestAssetsConfig(dir=str(tmp_path / "assets")),
    )

    # 客户端由**工厂**按配置造（换 OCR 只改这一处配置），再交给 loader
    doc = PdfLoader(cfg, build_ocr_engine(cfg, "dots")).load(_real_pdf(tmp_path), root=tmp_path)

    assert doc.metadata["parser"] == "dots"
    assert doc.metadata["page_count"] == 2
    assert doc.text.startswith("# 1第一章 概述")
    # 每页逐行打点：第 1 页第 1 行起，第二页回到第 1 行
    assert [(p.page, p.line) for p in doc.positions][:2] == [(1, 1), (1, 2)]
    assert doc.positions[-1].page == 2


# ---- 它内联的图片 ----


def test_inline_data_uri_image_is_moved_to_the_asset_store(tmp_path: Path) -> None:
    """dots.ocr 把裁剪好的图内联成 `![](data:image/png;base64,…)`：落盘 + 换短标记。"""
    payload = base64.b64encode(_PNG).decode("ascii")
    markdown = f"上边一段。\n\n![](data:image/png;base64,{payload})\n\n下边一段。"
    store = AssetStore(LocalBlobStore(tmp_path))

    result = extract_images(markdown, {}, store=store, doc_id="doc1", page=3)

    assert "base64" not in result.text
    assert len(result.images) == 1
    asset = result.images[0]
    assert (tmp_path / asset.asset_uri).read_bytes() == _PNG
    assert asset.asset_uri == "doc1/p0003_01.png"
    assert result.text.split(asset.marker)[0].strip() == "上边一段。"


def test_broken_base64_is_reported(tmp_path: Path) -> None:
    markdown = "![](data:image/png;base64,这不是 base64!!!)"

    with pytest.raises(ValueError, match="解不开"):
        extract_images(markdown, {}, store=AssetStore(LocalBlobStore(tmp_path)), doc_id="d", page=1)
