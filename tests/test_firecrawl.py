"""Firecrawl 这条路：**全程离线**（MockTransport），不碰网络、不花钱。

真调用那一次按技能要求另外做（`backend=firecrawl` 的 smoke test），不进用例 ——
单元测试不该按次计费。
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from rag.ingest.fetch import fetch_url
from rag.ingest.firecrawl import FirecrawlError, FirecrawlOptions, scrape_markdown

PUBLIC = ["93.184.216.34"]


def _public(_host: str) -> list[str]:
    return list(PUBLIC)


def _options(**over) -> FirecrawlOptions:
    base = {"api_key": "fc-test", "api_url": "https://api.firecrawl.dev"}
    base.update(over)
    return FirecrawlOptions(**base)


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)


def _firecrawl_ok(
    markdown: str = "# 标题\n\n正文。", *, title: str = "示例页"
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST" and str(request.url).endswith("/v2/scrape")
        assert request.headers["authorization"] == "Bearer fc-test"
        body = json.loads(request.content)
        assert body["formats"] == ["markdown"] and body["onlyMainContent"] is True
        assert body["removeBase64Images"] is True, "别把 base64 图混进正文"
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": {
                    "markdown": markdown,
                    "metadata": {
                        "title": title,
                        "sourceURL": "https://example.com/a",
                        "url": "https://example.com/a",
                        "statusCode": 200,
                    },
                },
            },
            request=request,
        )

    return handler


def test_scrape_returns_the_markdown_and_metadata() -> None:
    with _client(_firecrawl_ok()) as client:
        page = scrape_markdown("https://example.com/a", _options(), client=client)

    assert page.markdown.startswith("# 标题")
    assert page.final_url == "https://example.com/a"
    assert page.title == "示例页" and page.status_code == 200


def test_a_missing_key_says_what_to_do() -> None:
    """没配 key 时要说清去哪儿配，而不是拿一枚空 token 去打上游换 401。"""
    with _client(_firecrawl_ok()) as client, pytest.raises(FirecrawlError) as info:
        scrape_markdown("https://example.com/a", _options(api_key="  "), client=client)

    assert "FIRECRAWL_API_KEY" in str(info.value)


@pytest.mark.parametrize(
    ("status", "expect"),
    [(401, "FIRECRAWL_API_KEY"), (402, "额度"), (429, "限流"), (503, "服务端")],
)
def test_http_errors_become_actionable_sentences(status: int, expect: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text='{"error": "nope"}', request=request)

    with _client(handler) as client, pytest.raises(FirecrawlError) as info:
        scrape_markdown("https://example.com/a", _options(), client=client)

    assert expect in str(info.value)


def test_an_empty_markdown_is_an_error_not_a_blank_document() -> None:
    """空正文要当场报，别落一份空文档进库 —— 那在检索里就是"这份文件什么都没有"。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"success": True, "data": {"markdown": "  ", "warning": "页面可能是空壳"}},
            request=request,
        )

    with _client(handler) as client, pytest.raises(FirecrawlError) as info:
        scrape_markdown("https://example.com/a", _options(), client=client)

    assert "空壳" in str(info.value)


def test_firecrawl_handles_html_pages_only_and_keeps_the_images() -> None:
    """网页走 Firecrawl（正文干净），**图片仍由我们下到本地**、改成相对路径。

    PDF 那条在下面单独验：它必须原样走本机（我们的 PDF 管线要原件）。
    """
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.method == "POST":  # Firecrawl
            return _firecrawl_ok("# 标题\n\n正文。\n\n![图](https://example.com/img.png)")(request)
        if str(request.url).startswith("https://example.com/img.png"):
            return httpx.Response(
                200, content=b"\x89PNG", headers={"Content-Type": "image/png"}, request=request
            )
        return httpx.Response(
            200,
            content="<html><body>原始 HTML</body></html>".encode(),
            headers={"Content-Type": "text/html"},
            request=request,
        )

    with _client(handler) as fc, _client(handler) as local:
        item = fetch_url(
            "https://example.com/a",
            max_bytes=100_000,
            client=local,
            resolver=_public,
            backend="firecrawl",
            firecrawl=_options(),
            firecrawl_client=fc,
        )

    assert item.name == "a.md", "落盘件是 markdown（标题就是 #，比 HTML 猜层级准）"
    assert item.content_type == "text/markdown"
    assert "原始 HTML".encode() not in item.data, "没用我们自己抓的那份 HTTP 原文"
    assert "正文。".encode() in item.data
    assert len(item.attachments) == 1, "图片下到了本地附件里"
    relative, blob = next(iter(item.attachments.items()))
    assert relative.endswith("_files/img01.png") and blob == b"\x89PNG"
    assert relative.encode() in item.data, "markdown 里的地址改成了本地相对路径"
    assert not item.notes


def _fetch_firecrawl(handler: Callable[[httpx.Request], httpx.Response]) -> object:
    """走一遍真实的取件路径：本机 GET（探类型）→ Firecrawl → 图片本地化。"""
    with _client(handler) as fc, _client(handler) as local:
        return fetch_url(
            "https://example.com/a",
            max_bytes=100_000,
            client=local,
            resolver=_public,
            backend="firecrawl",
            firecrawl=_options(),
            firecrawl_client=fc,
        )


def _html_then_firecrawl(markdown: str, *, title: str = "示例页"):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return _firecrawl_ok(markdown, title=title)(request)
        return httpx.Response(
            200,
            content="<html><body>原始 HTML</body></html>".encode(),
            headers={"Content-Type": "text/html"},
            request=request,
        )

    return handler


def test_the_page_title_is_not_pushed_into_the_body() -> None:
    """页面 `<title>` **不进正文**（2026-10-10 撤掉那次的"补标题"）。

    使用者核过原始 HTML：那份 title 对应的是**站点 header**，不是正文的 `<h1>` ——
    补进正文等于往语料里塞站点名（实测那份抓下来是 `Deep Agents overview - Docs by LangChain`）。
    """
    item = _fetch_firecrawl(_html_then_firecrawl("## 小节\n\n正文。\n", title="示例页"))

    text = item.data.decode("utf-8")
    assert text.startswith("## 小节"), text[:60]
    assert "示例页" not in text, "站点 header 的名字不许进语料"
    assert item.name == "a.md", "文件名不动（改了真名就换 doc_id）"


def test_the_leading_site_chrome_is_dropped() -> None:
    """开头两坨站点 chrome 仍然清掉（这是保留下来的那半）。"""
    body = (
        "> ## Documentation Index\n"
        "> Fetch the complete documentation index at: [/llms.txt](https://x/llms.txt)\n"
        "\n"
        "[Skip to main content](https://x/a#content)\n"
        "\n"
        "## 小节\n\n正文。\n"
    )
    item = _fetch_firecrawl(_html_then_firecrawl(body))

    text = item.data.decode("utf-8")
    assert text.startswith("## 小节"), text[:80]
    assert "llms.txt" not in text and "Skip to main content" not in text


def test_a_pdf_never_goes_to_firecrawl() -> None:
    """PDF 走本机原路：我们的 PDF 管线要的是**原件**（版面模型给层级、抽图、页码）。"""
    posts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            posts.append(str(request.url))
        return httpx.Response(
            200, content=b"%PDF-1.4", headers={"Content-Type": "application/pdf"}, request=request
        )

    with _client(handler) as fc, _client(handler) as local:
        item = fetch_url(
            "https://example.com/paper.pdf",
            max_bytes=100_000,
            client=local,
            resolver=_public,
            backend="firecrawl",
            firecrawl=_options(),
            firecrawl_client=fc,
        )

    assert item.data == b"%PDF-1.4" and item.name == "paper.pdf"
    assert posts == [], "一次 Firecrawl 调用都不该发生（那是有钱的）"


def test_an_unknown_backend_is_rejected_instead_of_silently_going_local() -> None:
    with _client(_firecrawl_ok()) as client, pytest.raises(ValueError, match="fetch.backend"):
        fetch_url("https://example.com/a", max_bytes=1000, client=client, backend="ceph")
