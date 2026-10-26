"""PaddleOCR-VL 产线服务的客户端。

打的是**产线服务**（paddleocr-vl-api，默认 8080），不是 vLLM 托管的
PaddleOCR-VL 模型（默认 8118）—— 后者只是产线的下游，单独打它只拿得到
逐块识别结果，拿不到版面分析与阅读顺序。

这里只做一件事：把一份 PDF 或一张图交给 ``/layout-parsing``，按页取回 markdown。
"""

from __future__ import annotations

import base64
import logging
from collections.abc import Callable, Mapping
from functools import cached_property
from pathlib import Path
from typing import Any

import httpx

from rag.config.schema import IngestCloudOcrConfig, IngestConfig, IngestOcrConfig
from rag.ingest.ocr_client import OcrEngine, OcrPage, register_ocr

logger = logging.getLogger("rag.ingest.ocr")

# fileType：产线按它决定怎么切页。0 覆盖 PDF 与多页 TIFF，1 是单张图像。
FILE_PDF = 0
FILE_IMAGE = 1

# 走 OCR 的文档在 Document.metadata["parser"] 里记这个名字。
# 与 pymupdf / auto 这些策略值区分开：策略是「怎么选的」，这个是「谁解析的」。
PARSER_NAME = "paddleocr"

_LAYOUT_PATH = "/layout-parsing"


class OcrUnavailable(RuntimeError):
    """OCR 服务**不可用或自身故障**（连不上 / 5xx / 服务端 errorCode）。

    只有这类错误才值得切到云端兜底 —— 请求本身有问题（4xx、参数写错）时换一家
    也一样失败，白等一轮还把真正的原因盖掉。
    """


class OcrClient:
    """产线服务的薄客户端。同步的 —— loader 这一层本来就是同步的。"""

    def __init__(self, cfg: IngestOcrConfig, *, warn: Callable[[str], None] | None = None) -> None:
        self._cfg = cfg
        self._warn = warn or _warn_default

    @property
    def name(self) -> str:
        return PARSER_NAME

    @property
    def base_url(self) -> str:
        return self._cfg.base_url or ""

    @cached_property
    def _client(self) -> httpx.Client:
        from rag.net import trust_env_for

        return httpx.Client(
            base_url=self.base_url,
            timeout=self._cfg.timeout_s,
            # 本机 / 内网地址不看系统代理：代理够不着本机服务，回的是 502（见 rag/net.py）
            trust_env=trust_env_for(self.base_url),
            # httpx 的 retries 只覆盖连接层失败（连不上 / 握手失败），不重试超时与 5xx：
            # 那些多半是服务真在处理，重试只会更慢。
            transport=httpx.HTTPTransport(retries=self._cfg.max_retries),
        )

    def parse(self, path: Path, *, file_type: int) -> list[OcrPage]:
        """整份文件交给产线，按页返回 markdown。

        整份而不是逐页：跨页表格合并与标题重建都要求产线一次看到全部页。

        **不可用时先原地重试**（`ingest.ocr.retry_primary`，默认一次）：实测那条转发在真实
        负载下 28% 失败（§16 风险表），一个 502 就把整份文档推给**按量计费**的云端不值得。
        重试同样会出声 —— 静默重试会让人以为产线一直很稳。
        """
        attempts = max(1, self._cfg.retry_primary + 1)
        last: OcrUnavailable | None = None
        for attempt in range(1, attempts + 1):
            try:
                return self._parse_once(path, file_type=file_type)
            except OcrUnavailable as exc:
                last = exc
                if attempt >= attempts:
                    break
                self._warn(f"本地产线不可用（{exc}），原地重试（第 {attempt + 1}/{attempts} 次）")
        assert last is not None  # 循环要么返回、要么带着 last 出来
        raise last

    def _parse_once(self, path: Path, *, file_type: int) -> list[OcrPage]:
        """打一次产线。重试与兜底的判断都在调用方，这里只管一次请求。"""
        payload: dict[str, Any] = {
            "file": base64.b64encode(path.read_bytes()).decode("ascii"),
            "fileType": file_type,
            # 跨页重构三件套是一组，只开后两个不生效
            "restructurePages": self._cfg.restructure_pages,
            "mergeTables": self._cfg.merge_tables,
            "relevelTitles": self._cfg.relevel_titles,
            # 开着才拿得到图片：产线会把 <img src="imgs/…"> 内联进 markdown（位置就是
            # 原图在文中的位置），图片数据在响应的 markdown.images 里回。关掉则两者都没有。
            "returnMarkdownImages": self._cfg.return_images,
            "visualize": False,
        }
        try:
            resp = self._client.post(_LAYOUT_PATH, json=payload)
        except httpx.HTTPError as exc:
            raise OcrUnavailable(self._unreachable(exc)) from exc
        return self._pages(resp)

    def _pages(self, resp: httpx.Response) -> list[OcrPage]:
        body = _json_or_none(resp)
        # 失败响应里没有 result，errorCode 就是 HTTP 状态码。先看信封：
        # 这样报出来的是服务自己的 errorMsg，而不是干巴巴的一句 500。
        if body is not None and body.get("errorCode"):
            error = (
                f"OCR 服务报错：errorCode={body.get('errorCode')} "
                f"errorMsg={body.get('errorMsg')!r} logId={body.get('logId')}"
            )
            # 服务端自己出问题才值得兜底；报文错（4xx）换一家也一样
            code = body.get("errorCode")
            if isinstance(code, int) and code >= 500:
                raise OcrUnavailable(error)
            raise RuntimeError(error)
        if body is None:
            message = f"OCR 服务没返回 JSON：HTTP {resp.status_code} {_preview(resp)}"
            # 网关挂掉时返回的是 HTML，状态码同样是 5xx —— 那也是「服务不可用」，
            # 不该因为解析不出 JSON 就降级成「报文有问题」
            if resp.status_code >= 500:
                raise OcrUnavailable(message)
            raise RuntimeError(message)
        if resp.is_error:
            if resp.status_code >= 500:
                raise OcrUnavailable(f"OCR 服务返回 HTTP {resp.status_code} {_preview(resp)}")
            raise RuntimeError(f"OCR 服务返回 HTTP {resp.status_code} {_preview(resp)}")

        items = (body.get("result") or {}).get("layoutParsingResults") or []
        pages = [
            OcrPage(
                page_index=i,
                markdown=(item.get("markdown") or {}).get("text") or "",
                images=_images(item),
            )
            for i, item in enumerate(items)
        ]
        if not pages:
            # errorCode 是 0 却一页没有。宁可在这里报错，也别让它静默变成空正文
            # —— 那样 skip 原因只能含糊地说「正文为空」，查不出是服务端的问题。
            raise RuntimeError("OCR 服务返回 0 页（errorCode=0 但 layoutParsingResults 为空）")
        return pages

    def _unreachable(self, exc: httpx.HTTPError) -> str:
        return (
            f"连不上 OCR 服务 {self.base_url}（{type(exc).__name__}: {exc}）。"
            f"确认 .env 的 PADDLEOCR_BASE_URL 指向已启动的 paddleocr-vl-api"
            f"（默认 8080，不是 vLLM 那个 8118）。"
        )


class CloudOcrClient:
    """PaddleOCR **云端 API**（`paddleocr` 自带的 SDK），接口与 OcrClient 一致。

    用在本地产线不可用时的兜底。与本地那条路的两处差别都在这里抹平：

    - 要 access token（本地不要）
    - 图片回的是 **URL**，本地回的是 base64 —— 这里下载成字节，让下游拿到的完全一致
    """

    def __init__(self, cfg: IngestCloudOcrConfig, token: str) -> None:
        self._cfg = cfg
        self._token = token

    @property
    def name(self) -> str:
        return PARSER_NAME

    @property
    def base_url(self) -> str:
        return self._cfg.base_url

    def parse(self, path: Path, *, file_type: int) -> list[OcrPage]:
        """file_type 用不上：云端自己按文件内容判断，那是本地产线的参数。"""
        # 延迟 import：paddleocr 拖着一堆重依赖，别让每条 CLI 命令都付这个钱
        # 注意路径：`paddleocr._api_client` 只暴露子模块，**不重导出类** ——
        # 写 `from paddleocr._api_client import PaddleOCRClient` 会 ImportError
        # （打桩测试掩盖过这个错，真调云端时才暴露）
        from paddleocr._api_client.client import PaddleOCRClient

        client = PaddleOCRClient(
            token=self._token,
            base_url=self._cfg.base_url,
            request_timeout=self._cfg.request_timeout_s,
            poll_timeout=self._cfg.poll_timeout_s,
        )
        try:
            result = client.parse_document(model=self._cfg.model, file_path=str(path))
        except Exception as exc:
            raise OcrUnavailable(f"云端 PaddleOCR 解析失败：{type(exc).__name__}: {exc}") from exc
        finally:
            client.close()

        pages = [
            OcrPage(
                page_index=i,
                markdown=page.markdown_text or "",
                images=self._download(page.markdown_images or {}),
            )
            for i, page in enumerate(result.pages)
        ]
        if not pages:
            raise OcrUnavailable("云端返回 0 页")
        return pages

    def _download(self, urls: Mapping[str, str]) -> dict[str, bytes]:
        """图片 URL → 字节。

        同时按**全名和 basename** 建索引：正文里的 `<img src>` 与 `markdown_images`
        的键是否逐字相同，取决于云端怎么拼这个键 —— 两种都认，少一处对不上就少一张图。
        """
        out: dict[str, bytes] = {}
        with httpx.Client(timeout=self._cfg.request_timeout_s) as client:
            for name, url in urls.items():
                data = client.get(url).content
                out[name] = data
                out.setdefault(Path(name).name, data)
        return out


def _warn_default(message: str) -> None:
    """OCR 的降级（产线不可用、原地重试、切云端兜底）→ **WARNING**。

    换条路照样能把文档解析出来，没到「影响这次问答可信度」那个程度 ——
    那种才按 ERROR 记（见 `agent/middleware.py`，使用者 2026-10-06 定的口径）。
    """
    logger.warning("%s", message)


class FallbackOcr:
    """先打**本地产线**；产线不可用才走**云端**。

    只在 ``OcrUnavailable`` 上兜底 —— 请求本身有问题时换一家也一样失败，
    白等一轮还把真正的原因盖掉。切换**必须出声**：悄悄换到按量计费的云端，
    用户会看着账单猜原因。
    """

    def __init__(
        self,
        primary: OcrClient,
        fallback: CloudOcrClient,
        *,
        warn: Callable[[str], None] | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._warn = warn or _warn_default

    @property
    def name(self) -> str:
        return PARSER_NAME

    @property
    def base_url(self) -> str:
        """CLI 打印用的仍是本地那条路的地址。"""
        return self._primary.base_url

    def parse(self, path: Path, *, file_type: int) -> list[OcrPage]:
        try:
            return self._primary.parse(path, file_type=file_type)
        except OcrUnavailable as exc:
            self._warn(
                f"本地产线不可用（{exc}），本次改走云端 PaddleOCR：{self._fallback.base_url}"
            )
            return self._fallback.parse(path, file_type=file_type)


def build_ocr(
    cfg: IngestOcrConfig,
    token: str | None = None,
    *,
    warn: Callable[[str], None] | None = None,
) -> OcrClient | FallbackOcr | None:
    """base_url 为空 = 不启用 OCR：整条链路上不会有任何网络请求。

    这里 ``or ""`` 之后再 strip，是因为 yaml 里写 ``${VAR:-}`` 时空值可能以 None
    （不带引号）或空串（带引号）传进来 —— 与 providers/chat.py 那个 base_url 同一个坑。

    配了 ``cloud_fallback`` 时包一层 FallbackOcr：产线不可用才走云端。
    **开了兜底却没给 token 会直接报错** —— 默默退回「没有兜底」等于让人以为有保险。

    token 不传就自己从 .env 取（``get_secrets``）：调用点大多是「只要一个能用的
    OCR 客户端」，不该为了一个可选兜底去记住传 token。显式传参的口子留着给测试。
    """
    if not (cfg.base_url or "").strip():
        return None
    primary = OcrClient(cfg, warn=warn)

    cloud = cfg.cloud_fallback
    if not cloud.enabled:
        return primary

    if token is None:
        from rag.config import get_secrets

        token = get_secrets().paddleocr_access_token
    if not (token or "").strip():
        raise ValueError(
            "ingest.ocr.cloud_fallback 开着，但没有 PADDLEOCR_ACCESS_TOKEN："
            "在 .env 里填上一个云端 access token，或者把 enabled 设成 false。"
        )
    return FallbackOcr(primary, CloudOcrClient(cloud, token or ""), warn=warn)


@register_ocr("paddleocr")
def build_paddle_engine(ingest: IngestConfig, secrets: object | None = None) -> OcrEngine | None:
    """PaddleOCR 产线（含云端兜底）的工厂入口。

    登记名 = `paddleocr`（规范名）。老名字 ``ocr`` 仍然收 —— 配置里写哪个都行，
    归一化在 `config.schema.normalize_pdf_parser()`。

    它和另外两家的差别在**谁给标题层级**：产线的 `relevelTitles` 只按版面标签猜，
    实测在这些教材 PDF 上不可用；要层级请用 `dots`，要页码/块级定位请用 `mineru`。
    """
    token = getattr(secrets, "paddleocr_access_token", None) if secrets else None
    return build_ocr(ingest.ocr, token)


def _images(item: dict[str, Any]) -> dict[str, bytes]:
    """把一页的 ``markdown.images`` 解成「文件名 → 字节」。

    产线回的是**裸 base64**（实测不带 ``data:`` 前缀），但这里两种都认：接口换个版本
    加上前缀的话，不该让整份文档解析失败。
    """
    raw = (item.get("markdown") or {}).get("images") or {}
    out: dict[str, bytes] = {}
    for name, value in raw.items():
        if not isinstance(value, str):
            continue
        payload = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
        out[name] = base64.b64decode(payload)
    return out


def _json_or_none(resp: httpx.Response) -> dict[str, Any] | None:
    try:
        body = resp.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def _preview(resp: httpx.Response, width: int = 200) -> str:
    return " ".join(resp.text.split())[:width]
