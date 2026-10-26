"""MinerU 远程 API 的客户端：上传 → 建任务 → 轮询 → 取产物 → 逐页组装。

全程用 ``httpx.MockTransport``，不碰真服务（也不花 GPU 时间）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from rag.config.schema import IngestMineruConfig
from rag.ingest.mineru_api import MineruClient, MineruError

_MD = "## 2第二章 Apache Flink 快速入门\n\n第一段正文。\n\n![](data:image/png;base64,AAAA)\n"

_MIDDLE: dict[str, Any] = {
    "pages": [
        {
            "page_idx": 0,
            "blocks": [
                {
                    "type": "paragraph_title",
                    "level": 2,
                    "bbox": [0.1, 0.1, 0.5, 0.2],
                    "content": [{"type": "text", "content": "2第二章 Apache Flink 快速入门"}],
                },
                {
                    "type": "text",
                    "bbox": [0.1, 0.3, 0.9, 0.5],
                    "content": [{"type": "text", "content": "第一段正文。"}],
                },
                {"type": "image", "bbox": [0.1, 0.6, 0.9, 0.9], "content": []},
                {
                    "type": "page_number",
                    "bbox": [0.5, 0.95, 0.6, 0.99],
                    "content": [{"type": "text", "content": "3"}],
                },
            ],
        },
        {
            "page_idx": 1,
            "blocks": [
                {
                    "type": "text",
                    "bbox": [0.1, 0.1, 0.9, 0.3],
                    "content": [{"type": "text", "content": "第二页正文。"}],
                }
            ],
        },
    ]
}


class _Server:
    """一个够用的假 MinerU：记下每一步，按真实响应形状回。"""

    def __init__(self, *, file_status: str = "completed", upload_id_only: bool = False):
        self.calls: list[tuple[str, str]] = []
        self.file_status = file_status
        self.upload_id_only = upload_id_only
        self.authorization: str | None = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append((request.method, path))
        self.authorization = request.headers.get("authorization")

        if path == "/v1/uploads" and request.method == "POST":
            return httpx.Response(200, json={"id": "upload_1", "status": "created"})
        if path == "/v1/uploads/upload_1/content":
            return httpx.Response(200, json={})
        if path == "/v1/uploads/upload_1/complete":
            body: dict[str, Any] = {"id": "upload_1", "status": "completed"}
            if not self.upload_id_only:
                body["file"] = {"id": "file-1", "filename": "doc.pdf"}
            return httpx.Response(200, json=body)
        if path == "/v1/parse/jobs" and request.method == "POST":
            self.job_body = json.loads(request.content)
            return httpx.Response(200, json={"job_id": "job_1", "status": "queued"})
        if path == "/v1/parse/jobs/job_1":
            status = self.file_status
            entry: dict[str, Any] = {"name": "doc.pdf", "page_range": "1-2", "status": status}
            if status == "completed":
                entry["output_files"] = {
                    "markdown": {"file_id": "out-md", "bytes": len(_MD)},
                    "middle_json": {"file_id": "out-json", "bytes": 100},
                }
            else:
                entry["error"] = {
                    "type": "engine_error",
                    "code": "parse_failed",
                    "message": "File upload_1 not found",
                }
            return httpx.Response(
                200,
                json={
                    "job_id": "job_1",
                    "status": status,
                    "files": [entry],
                    "progress": {"completed": 1, "failed": 0, "total": 1},
                },
            )
        if path == "/v1/files/out-md/content":
            return httpx.Response(200, text=_MD)
        if path == "/v1/files/out-json/content":
            return httpx.Response(200, content=json.dumps(_MIDDLE).encode())
        return httpx.Response(404, json={"error": {"message": f"no route {path}"}})


def _client(server: _Server, **cfg: Any) -> MineruClient:
    config = IngestMineruConfig(poll_interval_s=0.0, **cfg)
    return MineruClient(config, transport=httpx.MockTransport(server.handler))


def _pdf(tmp_path: Path) -> Path:
    target = tmp_path / "doc.pdf"
    target.write_bytes(b"%PDF-1.4 fake")  # 客户端只上传字节，不解析 PDF
    return target


# ---- 主流程 ----


def test_full_flow_returns_one_page_per_pdf_page(tmp_path: Path) -> None:
    server = _Server()

    pages = _client(server).parse(_pdf(tmp_path), file_type=0)

    assert [page.page_index for page in pages] == [0, 1]
    # 标题块带 `#`、页码块丢掉、正文保留
    assert pages[0].markdown.splitlines()[0] == "## 2第二章 Apache Flink 快速入门"
    assert "第一段正文。" in pages[0].markdown
    assert "第二页正文。" in pages[1].markdown
    assert "3" not in pages[0].markdown, "页码块不是内容"
    # 走的顺序就是样例里那五步
    assert [f"{m} {p}" for m, p in server.calls[:5]] == [
        "POST /v1/uploads",
        "PUT /v1/uploads/upload_1/content",
        "POST /v1/uploads/upload_1/complete",
        "POST /v1/parse/jobs",
        "GET /v1/parse/jobs/job_1",
    ]


def test_image_block_takes_the_next_inline_data_uri(tmp_path: Path) -> None:
    """图片本体只在整份 markdown 里（内联 data URI），按出现顺序回填到图片块的位置。"""
    pages = _client(_Server()).parse(_pdf(tmp_path), file_type=0)

    assert "![](data:image/png;base64,AAAA)" in pages[0].markdown
    # 落在图片块原来的位置：正文之后
    assert pages[0].markdown.index("第一段正文。") < pages[0].markdown.index("data:image")


def test_the_job_uses_the_file_id_not_the_upload_id(tmp_path: Path) -> None:
    """实测踩过：把 upload 的 id 当 file_id 传，任务直接报 "File upload_… not found"。"""
    server = _Server()

    _client(server).parse(_pdf(tmp_path), file_type=0)

    source = server.job_body["files"][0]["source"]
    assert source == {"type": "file_id", "file_id": "file-1"}
    assert server.job_body["tier"] == "standard"


def test_missing_file_id_in_complete_response_is_an_error(tmp_path: Path) -> None:
    server = _Server(upload_id_only=True)

    with pytest.raises(MineruError, match="file.id"):
        _client(server).parse(_pdf(tmp_path), file_type=0)


def test_page_range_and_api_key_are_passed(tmp_path: Path) -> None:
    server = _Server()

    _client(server, page_range="1-2", api_key="sk-x").parse(_pdf(tmp_path), file_type=0)

    assert server.job_body["files"][0]["page_range"] == "1-2"
    assert server.authorization == "Bearer sk-x"


# ---- 失败处理 ----


def test_failed_job_surfaces_the_server_error(tmp_path: Path) -> None:
    server = _Server(file_status="failed")

    with pytest.raises(MineruError, match="File upload_1 not found"):
        _client(server).parse(_pdf(tmp_path), file_type=0)


def test_partial_job_is_treated_as_failure(tmp_path: Path) -> None:
    """partial = 有的文件没成功；我们不接受「少几页」的结果。"""
    server = _Server(file_status="partial")

    with pytest.raises(MineruError, match="解析失败"):
        _client(server).parse(_pdf(tmp_path), file_type=0)


def test_five_hundred_is_retried_then_fails(tmp_path: Path) -> None:
    """远端是转发链路，5xx 抖动要重试；重试完仍失败就报出来（带上状态码）。"""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(502, text="bad gateway")

    client = MineruClient(
        IngestMineruConfig(poll_interval_s=0.0, max_retries=1),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(MineruError, match="HTTP 502"):
        client.parse(_pdf(tmp_path), file_type=0)

    assert calls["n"] == 2, "一次 + 重试一次"
