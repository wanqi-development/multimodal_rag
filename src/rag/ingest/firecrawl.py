"""Firecrawl（云端网页抓取）的薄客户端：**一个 REST 调用，换回干净正文**。

为什么要它（2026-10-10 定）：我们自己抓到的 HTML 里混着导航、页脚、"Copy page" 这类
chrome，而这类页面（mintlify / Next 那些）的导航往往**不是 `<nav>`**，只靠标签黑名单
一定漏。Firecrawl 的 `/scrape` 带 `onlyMainContent`，正文直接给 markdown —— 省掉我们
自己写启发式那一大摊。

三处刻意的选择：

* **走 REST（httpx），不装 `firecrawl-py`**：本项目每个第三方服务都是「自己的 httpx
  薄客户端」（dots / mineru / OCR 都一样），而且这样能顺手用上我们自己的
  `trust_env_for()`（本机/内网不走代理）与响应大小上限 —— 少一个依赖。
* **只要 markdown**：`html` / `screenshot` / `json` 那些格式我们下游一个都用不上，多要
  一种就多一份传输与费用。顺手开 `removeBase64Images`：不然正文里会塞进一坨 base64，
  既撑大落盘件、又会把图片当成"正文的一部分"。
* **只管网页**：PDF / 图片 / 附件一律**不走这里** —— 我们的 PDF 管线要的是**原件**
  （版面模型给标题层级、抽图、页码行号），Firecrawl 只给 markdown 会把这些全丢掉。
   这一条由 `fetch.py` 判类型后决定，这里不重复判断。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

# 云端接口。自建部署把 `api_url` 指到自己的实例即可（路径仍是 `/v2/scrape`）。
DEFAULT_API_URL = "https://api.firecrawl.dev"
SCRAPE_PATH = "/v2/scrape"

# 一次抓取最长等多久。云版要现场渲染，比我们自己的 15s 宽松得多。
DEFAULT_TIMEOUT_S = 60.0


class FirecrawlError(RuntimeError):
    """抓取失败。**消息就是给用户看的那句话**（与 `FetchError` 同一个立场）。"""


@dataclass(frozen=True)
class FirecrawlOptions:
    """接进取件层的那几样。key 只从 `.env`（Secrets）来，不写进 yaml。"""

    api_key: str
    api_url: str = DEFAULT_API_URL
    # 只要正文（导航 / 页脚 / 侧栏交给它去判）—— 这正是换 Firecrawl 的**主要理由**
    only_main_content: bool = True
    timeout_s: float = DEFAULT_TIMEOUT_S
    # 响应的字节上限（正文 + 元数据）。给足但仍然有上限：云端返回多大不由我们决定。
    max_bytes: int = 8_000_000


@dataclass(frozen=True)
class ScrapedPage:
    """抓回来的一页。`final_url` 是**重定向之后**的地址（图片的相对路径按它解析）。"""

    markdown: str
    final_url: str
    # 页面的 `<title>`。**注意它不是正文的 `<h1>`** —— 原始 HTML 里那份对应的是站点 header。
    # 曾经拿它往正文首行补过 `# <title>`（2026-10-10 撤了：等于往语料里塞站点名）。
    # 留着这个字段是因为它仍是"这份抓取来自哪个页面"的线索（排查、将来的元数据都可能用）。
    title: str = ""
    status_code: int | None = None
    # 云端的提示（比如"正文可能是空的"）——原样带出去，别吞
    warning: str = ""


def _read_capped(response: httpx.Response, *, max_bytes: int, url: str) -> bytes:
    """流式收、边收边数 —— 云端回多大不由我们决定，得有个上限。"""
    chunks: list[bytes] = []
    total = 0
    for block in response.iter_bytes():
        total += len(block)
        if total > max_bytes:
            raise FirecrawlError(f"Firecrawl 的响应超过上限 {max_bytes / 1e6:.0f}MB（{url}）")
        chunks.append(block)
    return b"".join(chunks)


def _error_for(status: int, body: str) -> str:
    """把 HTTP 状态翻成「下一步该怎么办」的那句话。"""
    detail = body.strip()[:200]
    if status in (401, 403):
        return (
            "Firecrawl 说这枚 key 不认（401/403）：检查 .env 里的 FIRECRAWL_API_KEY"
            f"（是不是复制少了、或者被撤销了）。{detail}"
        )
    if status == 402:
        return f"Firecrawl 账户额度/订阅不够（402）：去控制台看用量。{detail}"
    if status == 429:
        return f"Firecrawl 限流了（429）：等一会儿再抓，或调低并发。{detail}"
    if status >= 500:
        return f"Firecrawl 服务端出错（{status}），这一侧没错，稍后重试。{detail}"
    return f"Firecrawl 拒绝了这次请求（{status}）：{detail}"


def scrape_markdown(
    url: str, options: FirecrawlOptions, *, client: httpx.Client | None = None
) -> ScrapedPage:
    """抓一页，只要正文 markdown。失败抛 `FirecrawlError`（消息给人看）。

    `client` 是给测试的口子（`httpx.MockTransport`），与 `fetch.fetch_url` 同一套写法。
    """
    from rag.net import trust_env_for

    if not (options.api_key or "").strip():
        raise FirecrawlError(
            "没配 FIRECRAWL_API_KEY：在 .env 里填上 Firecrawl 的 key（fc-… 开头），"
            "或者把 ingest.uploads.fetch.backend 换回 http（本机抓取）。"
        )
    api_url = (options.api_url or DEFAULT_API_URL).rstrip("/")
    endpoint = f"{api_url}{SCRAPE_PATH}"
    payload: dict[str, Any] = {
        "url": url,
        "formats": ["markdown"],
        "onlyMainContent": bool(options.only_main_content),
        # 正文里塞 base64 的图既撑大落盘件、又会把图片混进文本 —— 明确不要
        "removeBase64Images": True,
    }
    owned = client is None
    client = client or httpx.Client(
        timeout=httpx.Timeout(options.timeout_s),
        # 云端是公网地址，该走代理就走；自建在内网时 trust_env_for 会自动关掉
        trust_env=trust_env_for(api_url),
    )
    try:
        try:
            with client.stream(
                "POST",
                endpoint,
                json=payload,
                headers={
                    "Authorization": f"Bearer {options.api_key.strip()}",
                    "Content-Type": "application/json",
                },
            ) as response:
                if response.status_code >= 400:
                    body = response.read().decode("utf-8", "replace")
                    raise FirecrawlError(_error_for(response.status_code, body))
                raw = _read_capped(response, max_bytes=options.max_bytes, url=url)
        except httpx.HTTPError as exc:
            raise FirecrawlError(
                f"连不上 Firecrawl（{endpoint}）：{type(exc).__name__}: {exc}"
            ) from exc
    finally:
        if owned:
            client.close()

    import json

    try:
        body = json.loads(raw)
    except ValueError as exc:
        raise FirecrawlError(f"Firecrawl 回了不是 JSON 的东西（{url}）：{exc}") from exc
    if not isinstance(body, dict) or not body.get("success", False):
        detail = (body or {}).get("error") if isinstance(body, dict) else None
        raise FirecrawlError(f"Firecrawl 说这次没成功（{url}）：{detail or body}")
    data = body.get("data") or {}
    markdown = data.get("markdown") or ""
    metadata = data.get("metadata") or {}
    if not markdown.strip():
        why = data.get("warning") or metadata.get("error") or "空内容"
        raise FirecrawlError(f"Firecrawl 没拿到正文（{url}）：{why}")
    return ScrapedPage(
        markdown=markdown,
        final_url=str(metadata.get("url") or metadata.get("sourceURL") or url),
        title=str(metadata.get("title") or ""),
        status_code=metadata.get("statusCode"),
        warning=str(data.get("warning") or ""),
    )
