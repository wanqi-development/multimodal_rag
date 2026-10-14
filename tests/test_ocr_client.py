"""OCR 客户端：请求形状、按页拼装、错误转成能照着修的中文报错。

不碰网络 —— 用 httpx 自带的 MockTransport 注入一个 handler。比 monkeypatch 掉
`post` 好：序列化、状态码、异常类型都走真实调用路径，替身只换掉最外面那一层。

跑的是真的 `parse` / `_pages`，只是把内部的 httpx.Client 换掉（见 _RecordingOcr）。
"""

from __future__ import annotations

import base64
import json
from functools import cached_property
from pathlib import Path
from typing import Any

import httpx
import pytest

from rag.config.schema import IngestOcrConfig
from rag.ingest.ocr import FILE_IMAGE, FILE_PDF, OcrClient, OcrUnavailable

BASE_URL = "http://ocr.test:8080"


class _RecordingOcr(OcrClient):
    """把 httpx.Client 换成走 MockTransport 的，其余逻辑一行不改。"""

    def __init__(self, cfg: IngestOcrConfig, handler: httpx.MockTransport) -> None:
        super().__init__(cfg)
        self._transport = handler

    @cached_property
    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, transport=self._transport)


class _Recorder:
    """记下发出去的请求体，按顺序吐出预设响应。"""

    def __init__(self, *responses: httpx.Response) -> None:
        self.bodies: list[dict[str, Any]] = []
        self.raw: list[httpx.Request] = []
        self._responses = list(responses)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.raw.append(request)
        self.bodies.append(json.loads(request.content))
        return self._responses.pop(0)


def _page(markdown: str, images: dict[str, str] | None = None) -> dict[str, Any]:
    # 真实的响应里每页还带 prunedResult / outputImages / exports，
    # 客户端只取 markdown.text，多带几个键确认它不会被多余的字段绊倒
    return {
        "markdown": {"text": markdown, "images": images or {}},
        "prunedResult": {"parsing_res_list": [{"block_label": "text", "block_content": "x"}]},
        "outputImages": None,
    }


def _ok(*pages: str, images: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "logId": "log-1",
            "errorCode": 0,
            "errorMsg": "Success",
            "result": {
                "layoutParsingResults": [
                    _page(p, images if i == 0 else None) for i, p in enumerate(pages)
                ]
            },
        },
    )


def _client(recorder: _Recorder, **cfg: Any) -> OcrClient:
    handler = httpx.MockTransport(recorder)
    return _RecordingOcr(IngestOcrConfig(base_url=BASE_URL, **cfg), handler)


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    target = tmp_path / "scan.pdf"
    target.write_bytes(b"%PDF-1.7 fake scan bytes")
    return target


# ---- 请求形状 ----


def test_request_carries_the_whole_file_as_base64(sample_pdf: Path) -> None:
    """整份送过去而不是逐页：跨页表格合并与标题重建都要求产线一次看到全部页。"""
    rec = _Recorder(_ok("第一页"))
    client = _client(rec)

    client.parse(sample_pdf, file_type=FILE_PDF)

    body = rec.bodies[0]
    assert base64.b64decode(body["file"]) == sample_pdf.read_bytes()
    assert body["fileType"] == FILE_PDF
    assert rec.raw[0].url.path == "/layout-parsing"


def test_cross_page_switches_are_sent_as_configured(sample_pdf: Path) -> None:
    """三个开关是一组：只开 merge/relevel 不开 restructure 服务端不认，
    所以配置里默认全 true，这里确认真的传出去了。"""
    rec = _Recorder(_ok("一页"))
    _client(rec).parse(sample_pdf, file_type=FILE_PDF)

    body = rec.bodies[0]
    assert body["restructurePages"] is True
    assert body["mergeTables"] is True
    assert body["relevelTitles"] is True


def test_visualization_images_are_never_requested(sample_pdf: Path) -> None:
    """可视化的中间图 RAG 用不上，白占带宽与时间 —— 没有开关，代码里就是关的。"""
    rec = _Recorder(_ok("一页"))
    _client(rec).parse(sample_pdf, file_type=FILE_PDF)

    assert rec.bodies[0]["visualize"] is False


def test_markdown_images_follow_the_config(sample_pdf: Path) -> None:
    """图片默认要（后面要挂 VLM 描述），但能关掉省带宽。"""
    rec = _Recorder(_ok("一页"))
    _client(rec).parse(sample_pdf, file_type=FILE_PDF)
    assert rec.bodies[0]["returnMarkdownImages"] is True

    off = _Recorder(_ok("一页"))
    _client(off, return_images=False).parse(sample_pdf, file_type=FILE_PDF)
    assert off.bodies[0]["returnMarkdownImages"] is False


def test_markdown_images_are_decoded_to_bytes(sample_pdf: Path) -> None:
    """图片本体在 markdown.images 里，键与 <img src> 逐字相同 —— 位置与数据都靠它。"""
    name = "imgs/img_in_image_box_1_2_3_4.jpg"
    rec = _Recorder(_ok(f'<img src="{name}" />', images={name: "aGVsbG8="}))

    pages = _client(rec).parse(sample_pdf, file_type=FILE_PDF)

    assert pages[0].images == {name: b"hello"}
    assert pages[0].markdown == f'<img src="{name}" />'


def test_switches_can_be_turned_off_by_config(sample_pdf: Path) -> None:
    """配置说了算，不在代码里写死。"""
    rec = _Recorder(_ok("一页"))
    _client(rec, restructure_pages=False, merge_tables=False, relevel_titles=False).parse(
        sample_pdf, file_type=FILE_PDF
    )

    body = rec.bodies[0]
    assert body["restructurePages"] is False
    assert body["mergeTables"] is False
    assert body["relevelTitles"] is False


def test_image_files_use_the_image_file_type(sample_pdf: Path) -> None:
    rec = _Recorder(_ok("图上的一行字"))
    _client(rec).parse(sample_pdf, file_type=FILE_IMAGE)

    assert rec.bodies[0]["fileType"] == FILE_IMAGE


# ---- 响应解析 ----


def test_pages_keep_their_order(sample_pdf: Path) -> None:
    """页序只能靠数组下标 —— 产线把 prunedResult 里的 page_index 剥掉了，
    响应里没有任何可信的页码字段。"""
    rec = _Recorder(_ok("第一页的内容", "第二页的内容", "第三页的内容"))
    pages = _client(rec).parse(sample_pdf, file_type=FILE_PDF)

    assert [p.page_index for p in pages] == [0, 1, 2]
    assert [p.markdown for p in pages] == ["第一页的内容", "第二页的内容", "第三页的内容"]


def test_page_without_markdown_becomes_empty_not_missing(sample_pdf: Path) -> None:
    """服务端可能给 null，别让 None 传到下游当正文用。"""
    response = httpx.Response(
        200,
        json={
            "logId": "L",
            "errorCode": 0,
            "result": {"layoutParsingResults": [{"markdown": None}, {}]},
        },
    )
    pages = _client(_Recorder(response)).parse(sample_pdf, file_type=FILE_PDF)

    assert [p.markdown for p in pages] == ["", ""]


# ---- 错误路径：报错要能照着修 ----


def test_error_envelope_surfaces_the_service_message(sample_pdf: Path) -> None:
    """失败响应没有 result，errorCode 就是 HTTP 状态码。要把服务自己的
    errorMsg 与 logId 带出来，否则只能对着一句 500 猜。"""
    response = httpx.Response(
        500,
        json={"logId": "log-9", "errorCode": 500, "errorMsg": "model server unreachable"},
    )
    with pytest.raises(OcrUnavailable) as exc:
        # 这条验的是**错误映射**，不是重试：重试会再发一次请求，把信号搅浑。
        # 「不可用会先原地重试」另有用例（test_ocr_fallback.py）。
        _client(_Recorder(response), retry_primary=0).parse(sample_pdf, file_type=FILE_PDF)

    message = str(exc.value)
    assert "log-9" in message
    assert "model server unreachable" in message


def _failing_client(exc: Exception) -> OcrClient:
    """一个在连接层就炸的客户端：handler 抛异常即等价于连不上 / 超时。"""

    def boom(request: httpx.Request) -> httpx.Response:
        raise exc

    return _RecordingOcr(IngestOcrConfig(base_url=BASE_URL), httpx.MockTransport(boom))


def test_unreachable_service_names_the_variable_to_fix(sample_pdf: Path) -> None:
    client = _failing_client(httpx.ConnectError("连接被拒绝"))

    with pytest.raises(OcrUnavailable) as exc:
        client.parse(sample_pdf, file_type=FILE_PDF)

    message = str(exc.value)
    assert "PADDLEOCR_BASE_URL" in message
    assert BASE_URL in message
    # 8118 是产线的下游，填错端口是最容易踩的坑，报错里得点到
    assert "8118" in message


def test_a_bad_request_is_not_marked_unavailable(sample_pdf: Path) -> None:
    """请求本身有问题时不该走兜底 —— 换一家也一样失败，白等一轮还把真正的原因盖掉。

    只有「服务不可用/自身故障」才是 OcrUnavailable（见 FallbackOcr）。
    """
    response = httpx.Response(
        400, json={"logId": "l", "errorCode": 400, "errorMsg": "unsupported file"}
    )

    with pytest.raises(RuntimeError) as exc:
        _client(_Recorder(response)).parse(sample_pdf, file_type=FILE_PDF)

    assert not isinstance(exc.value, OcrUnavailable)
    assert "unsupported file" in str(exc.value)


def test_timeout_is_reported_as_a_runtime_error(sample_pdf: Path) -> None:
    client = _failing_client(httpx.ReadTimeout("等太久了"))

    with pytest.raises(OcrUnavailable, match="PADDLEOCR_BASE_URL"):
        client.parse(sample_pdf, file_type=FILE_PDF)


def test_non_json_response_is_reported_with_a_body_preview(sample_pdf: Path) -> None:
    """网关拦下来时返回的是 HTML，别让它变成一句 json 解析异常。"""
    response = httpx.Response(502, text="<html><body>Bad Gateway</body></html>")
    with pytest.raises(OcrUnavailable, match="Bad Gateway"):
        _client(_Recorder(response), retry_primary=0).parse(sample_pdf, file_type=FILE_PDF)


def test_zero_pages_is_an_error_not_an_empty_document(sample_pdf: Path) -> None:
    """errorCode 是 0 却一页没有。宁可在这里报错，也别让它静默变成空正文 ——
    那样 skip 原因只能说「正文为空」，查不出是服务端的问题。"""
    response = httpx.Response(
        200, json={"logId": "L", "errorCode": 0, "result": {"layoutParsingResults": []}}
    )
    with pytest.raises(RuntimeError, match="0 页"):
        _client(_Recorder(response)).parse(sample_pdf, file_type=FILE_PDF)
