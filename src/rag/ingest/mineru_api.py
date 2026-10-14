"""MinerU 远程 API 的客户端（`MinerU API v1.0.0`），与另外两家 OCR 同接口。

**不用官方 SDK**：`from mineru.parser import MinerUApiParser` 需要 `mineru` 包，
而它只在另一个 Python 环境里装着（装进本项目环境会把 torch 那一串重依赖带进来）。
这里用 httpx 直接打 REST，几十行就够，也和其他 client 一个路子。

流程（每一步都实测过）::

    POST /v1/uploads                    → upload id
    PUT  /v1/uploads/{id}/content       → 传字节
    POST /v1/uploads/{id}/complete      → **file.id**（不是 upload 的 id！传错会报
                                          "File upload_… not found"）
    POST /v1/parse/jobs                 → job_id
    GET  /v1/parse/jobs/{job_id}        → 轮询到 completed / partial / failed
    GET  /v1/files/{file_id}/content    → 取产物（markdown / middle_json / zip）

**按页组装**：API 只回整份 markdown + 一份 `middle_json`，而我们的出口是「一页一份
markdown」（`OcrPage`）。所以这里用 `middle_json` 的 `pages[] → blocks[]` 逐页重建：
标题块加 `#`（层级来自块的 `level`，实测恒为 2 —— MinerU 不给层级，层级由人工清单决定）、
表格原样保留、页码块丢掉；图片按**出现顺序**与 markdown 里的 data URI 一一对上。
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from rag.config.schema import IngestConfig, IngestMineruConfig
from rag.ingest.ocr_client import OcrEngine, OcrPage, register_ocr

PARSER_NAME = "mineru"

# 整份 markdown 里内联的图片（MinerU 与 dots.ocr 一样，把裁好的图内联成 data URI）
_DATA_URI = re.compile(r"!\[[^\]]*\]\(data:image/[^)]+\)")

# 块类型 → 是否算标题，以及标题级别取哪
_TITLE_TYPES = {"doc_title", "title", "paragraph_title", "section_header"}
_SKIP_TYPES = {"page_number", "page_footer", "page_header", "discarded"}


class MineruError(RuntimeError):
    """MinerU 调用失败（上传/建任务/轮询/取产物，或服务返回 failed）。"""


class MineruClient:
    """MinerU 远程 API 的薄客户端。整份文档一次任务，页与页之间不用我们并发。"""

    def __init__(
        self, cfg: IngestMineruConfig, *, transport: httpx.BaseTransport | None = None
    ) -> None:
        self._cfg = cfg
        # 测试用的口子：换成 httpx.MockTransport 就能把整条流程跑在本地
        self._transport = transport

    @property
    def name(self) -> str:
        return PARSER_NAME

    @property
    def base_url(self) -> str:
        return self._cfg.base_url.rstrip("/")

    def parse(self, path: Path, *, file_type: int) -> list[OcrPage]:
        """``file_type`` 用不上：MinerU 按上传的 mime 自己判断。"""
        headers = (
            {"Authorization": f"Bearer {self._cfg.api_key}"} if self._cfg.api_key.strip() else None
        )
        from rag.net import trust_env_for

        with httpx.Client(
            base_url=self.base_url,
            timeout=self._cfg.timeout_s,
            headers=headers,
            # 本机 / 内网地址不看系统代理：代理够不着本机服务，回的是 502（见 rag/net.py）
            trust_env=trust_env_for(self.base_url),
            transport=self._transport,
        ) as client:
            file_id = self._upload(client, path)
            job_id = self._create_job(client, file_id)
            entry = self._wait(client, job_id)
            markdown = self._fetch_output(client, entry, "markdown")
            middle = self._fetch_output(client, entry, "middle_json")
        if middle is None:
            raise MineruError("MinerU 没回 middle_json，拿不到逐页结构（文本会失去页码定位）")
        return self._to_pages(markdown or "", middle)

    # ---- 四个 API 步骤 ----

    def _upload(self, client: httpx.Client, path: Path) -> str:
        data = path.read_bytes()
        created = self._request(
            client,
            "POST",
            "/v1/uploads",
            json={
                "filename": path.name,
                "bytes": len(data),
                "mime_type": "application/pdf",
                "purpose": "parse",
            },
        )
        upload_id = created["id"]
        # 直传内容（服务端也可以给 presigned url，本地转发这条路用直传更省事）
        resp = client.put(f"/v1/uploads/{upload_id}/content", content=data)
        if resp.status_code >= 400:
            raise MineruError(f"上传失败：HTTP {resp.status_code} {resp.text[:200]}")
        done = self._request(client, "POST", f"/v1/uploads/{upload_id}/complete", json={})
        # **要的是 file 的 id**：第一次把 upload 的 id 传进 parse/jobs，任务直接报
        # "File upload_… not found"
        file_id = (done.get("file") or {}).get("id")
        if not file_id:
            raise MineruError(f"上传响应里没有 file.id：{str(done)[:200]}")
        return file_id

    def _create_job(self, client: httpx.Client, file_id: str) -> str:
        entry: dict[str, Any] = {"source": {"type": "file_id", "file_id": file_id}}
        if self._cfg.page_range.strip():
            entry["page_range"] = self._cfg.page_range.strip()
        body = self._request(
            client,
            "POST",
            "/v1/parse/jobs",
            json={
                "files": [entry],
                "tier": self._cfg.tier,
                "ocr_mode": self._cfg.ocr_mode,
                "output_formats": ["markdown", "middle_json"],
            },
        )
        job_id = body.get("job_id")
        if not job_id:
            raise MineruError(f"建任务响应里没有 job_id：{str(body)[:200]}")
        return job_id

    def _wait(self, client: httpx.Client, job_id: str) -> dict[str, Any]:
        deadline = time.time() + self._cfg.timeout_s
        while True:
            state = self._request(client, "GET", f"/v1/parse/jobs/{job_id}")
            status = state.get("status")
            if status in ("completed", "partial", "failed", "canceled"):
                break
            if time.time() > deadline:
                raise MineruError(
                    f"MinerU 任务 {job_id} 超过 {self._cfg.timeout_s:.0f}s 还没完成"
                    f"（status={status}，tier={self._cfg.tier}）"
                )
            time.sleep(self._cfg.poll_interval_s)
        files = state.get("files") or [{}]
        entry = files[0]
        if entry.get("status") != "completed":
            error = entry.get("error") or {}
            raise MineruError(
                f"MinerU 解析失败：{error.get('code') or status} {error.get('message') or ''}"
                f"（tier={self._cfg.tier}，page_range={self._cfg.page_range or '全部'}）"
            )
        return entry

    def _fetch_output(self, client: httpx.Client, entry: dict[str, Any], kind: str) -> Any | None:
        ref = ((entry.get("output_files") or {}).get(kind)) or None
        if not ref:
            return None
        resp = client.get(f"/v1/files/{ref['file_id']}/content")
        if resp.status_code >= 400:
            raise MineruError(f"取 {kind} 产物失败：HTTP {resp.status_code} {resp.text[:200]}")
        if kind == "markdown":
            return resp.text
        try:
            return json.loads(resp.content)
        except ValueError as exc:
            raise MineruError(f"{kind} 产物不是合法 JSON：{exc}") from exc

    def _request(self, client: httpx.Client, method: str, url: str, **kwargs: Any) -> dict:
        """带上重试：远端是转发链路，偶发 5xx / 连接重置很常见。"""
        last: Exception | None = None
        for attempt in range(self._cfg.max_retries + 1):
            try:
                resp = client.request(method, url, **kwargs)
                if resp.status_code >= 400:
                    # 4xx/5xx 都带出来：远端是转发链路，5xx 多半是抖动（外层会重试）；
                    # 4xx 是请求本身有问题，重试也没用 —— 但原因要原样报出来
                    raise MineruError(
                        f"{method} {url} -> HTTP {resp.status_code} {resp.text[:200]}"
                    )
                return resp.json()
            except MineruError as exc:
                last = exc
            except (httpx.HTTPError, ValueError) as exc:
                last = exc
            if attempt < self._cfg.max_retries:
                print(
                    f"⚠ MinerU {method} {url} 失败（第 {attempt + 1} 次，"
                    f"{type(last).__name__}），重试：{last}",
                    file=sys.stderr,
                )
                time.sleep(self._cfg.poll_interval_s)
        raise MineruError(f"{method} {url} 重试 {self._cfg.max_retries} 次仍失败：{last}")

    # ---- 整份 markdown + middle_json → 逐页 markdown ----

    def _to_pages(self, markdown: str, middle: dict[str, Any]) -> list[OcrPage]:
        pages = middle.get("pages") or []
        if not pages:
            raise MineruError("middle_json 里没有 pages")
        images = _DATA_URI.findall(markdown)
        cursor = 0
        out: list[OcrPage] = []
        for page in pages:
            parts: list[str] = []
            for block in page.get("blocks") or []:
                kind = (block.get("type") or "").lower()
                if kind in _SKIP_TYPES:
                    continue
                if kind == "image":
                    # 图片本体只在整份 markdown 里（内联 data URI）：按出现顺序一一对上
                    if cursor < len(images):
                        parts.append(f"![]({images[cursor]})")
                        cursor += 1
                    continue
                text = _block_text(block)
                if not text:
                    continue
                if kind in _TITLE_TYPES:
                    level = int(block.get("level") or 2)
                    parts.append(f"{'#' * max(1, min(level, 6))} {text}")
                else:
                    parts.append(text)
            out.append(
                OcrPage(
                    page_index=int(page.get("page_idx") or len(out)),
                    markdown="\n\n".join(parts),
                )
            )
        out.sort(key=lambda page: page.page_index)
        # 页码从 0 起、按顺序重排，保证与产线/dots 两种出口一致
        return [
            OcrPage(page_index=index, markdown=page.markdown, images=page.images)
            for index, page in enumerate(out)
        ]


def _block_text(block: dict[str, Any]) -> str:
    """块文字：``content`` 里各子块的 ``content`` 拼起来（表格/公式原样保留）。"""
    pieces: list[str] = []
    for item in block.get("content") or []:
        if not isinstance(item, dict):
            continue
        value = item.get("content")
        if isinstance(value, str) and value.strip():
            pieces.append(value.strip())
    return "\n".join(pieces).strip()


@register_ocr("mineru")
def build_mineru_engine(ingest: IngestConfig, secrets: object | None = None) -> OcrEngine:
    """MinerU 远程 API 的工厂入口（页码/块级定位最规整，但不给标题层级）。"""
    return MineruClient(ingest.mineru)
