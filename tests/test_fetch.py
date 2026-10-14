"""从链接抓文档：**全程离线**（`httpx.MockTransport` + 假 resolver），不碰网络。

重点在 SSRF 那几条：私网 / 环回 / 链路本地，以及**重定向到私网**都必须被拦下来 ——
服务端替用户抓 URL，等于把这台机器当成探内网的代理。
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from rag.ingest.fetch import FetchError, fetch_url
from rag.ingest.loaders.base import data_hash

# example.com 的地址。写死在这儿，免得用例去问 DNS（离线也该跑得过）
PUBLIC = ["93.184.216.34"]


def _public(_host: str) -> list[str]:
    return list(PUBLIC)


def _fetch(
    handler: Callable[[httpx.Request], httpx.Response],
    url: str = "https://example.com/a.html",
    *,
    resolver: Callable[[str], list[str]] = _public,
    max_bytes: int = 10_000,
    **kwargs,
):
    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False) as client:
        return fetch_url(url, max_bytes=max_bytes, client=client, resolver=resolver, **kwargs)


def _ok(
    body: bytes = b"<h1>hi</h1>", headers: dict[str, str] | None = None
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers=headers or {}, request=request)

    return handler


def test_an_html_page_takes_its_name_from_content_disposition() -> None:
    """下载链接常常自带真名（RFC 2231 那种中文名也算），优先用它。"""
    item = _fetch(
        _ok(
            headers={
                "Content-Type": "text/html; charset=utf-8",
                "Content-Disposition": "attachment; filename*=UTF-8''%E6%89%8B%E5%86%8C.html",
            }
        ),
        "https://example.com/download?id=7",
    )

    assert item.name == "手册.html"
    assert item.content_type == "text/html", "charset 那一段要削掉"
    assert item.data == b"<h1>hi</h1>"
    assert item.final_url == "https://example.com/download?id=7"


def test_a_pdf_download_without_a_suffix_in_the_url_uses_the_content_type() -> None:
    """URL 里没有后缀（`/download?id=7`），就靠 Content-Type 反查 —— 它是唯一线索。"""
    item = _fetch(
        _ok(b"%PDF-1.4", headers={"Content-Type": "application/pdf"}),
        "https://example.com/download",
    )

    assert item.name == "download.pdf"


def test_the_url_last_segment_is_used_when_the_server_gives_no_filename() -> None:
    item = _fetch(
        _ok(headers={"Content-Type": "text/html"}), "https://example.com/docs/intro%20v2.html"
    )

    assert item.name == "intro v2.html", "URL 末段要解码（%20 不该留在文件名里）"


def test_a_relative_redirect_is_followed_and_the_final_url_is_recorded() -> None:
    """跟跳转要记下**最终**地址：下载链接跳到 CDN 是常态，出问题时要看得出跳去了哪。"""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/real/page.pdf"}, request=request)
        return httpx.Response(
            200, content=b"%PDF-1.4", headers={"content-type": "application/pdf"}, request=request
        )

    item = _fetch(handler, "https://example.com/start")

    assert item.final_url == "https://example.com/real/page.pdf"
    assert item.name == "page.pdf"


def test_a_redirect_to_a_private_address_is_refused() -> None:
    """**最有价值的那条**：公网域名把我们引到内网，每一跳都要重新查一次。"""

    def resolver(host: str) -> list[str]:
        return ["10.0.0.9"] if host == "internal.example.com" else list(PUBLIC)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302, headers={"location": "http://internal.example.com/secret"}, request=request
        )

    with pytest.raises(FetchError, match="非公网地址"):
        _fetch(handler, "https://example.com/start", resolver=resolver)


@pytest.mark.parametrize("host", ["127.0.0.1", "169.254.169.254", "[::1]"])
def test_loopback_and_link_local_hosts_are_refused(host: str) -> None:
    """环回与云元数据地址（169.254.169.254）是最经典的两个目标，直接挡在门前。"""
    # **不注入 resolver**：字面量 IP 由 getaddrinfo 自己就能解析，离线也跑得通
    with pytest.raises(FetchError, match="非公网地址"):
        fetch_url(f"http://{host}/x", max_bytes=1000)


def test_a_dns_name_pointing_at_a_private_ip_is_refused() -> None:
    """一个域名可以同时挂公网与内网 IP：**每个地址都要查**，有一个内网就拒绝。"""
    with pytest.raises(FetchError, match="非公网地址"):
        _fetch(_ok(), resolver=lambda host: [*PUBLIC, "192.168.1.10"])


def test_allowing_private_hosts_is_opt_in_for_self_hosted_sites() -> None:
    """自建文档站要抓内网，就把 `allow_private_hosts` 打开（默认关）。"""
    item = _fetch(
        _ok(headers={"Content-Type": "text/html"}),
        "http://docs.internal/x.html",
        resolver=lambda host: ["10.0.0.9"],
        allow_private_hosts=True,
    )

    assert item.name == "x.html"


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/a.pdf", "example.com/a"])
def test_non_http_schemes_are_refused(url: str) -> None:
    with pytest.raises(FetchError, match="http"):
        _fetch(_ok(), url)


def test_an_unsupported_extension_is_refused_by_name() -> None:
    """`.doc` 这种老格式不在名单里 —— 不能偷偷按 .docx 试。"""
    with pytest.raises(FetchError, match=r"\.doc"):
        _fetch(_ok(headers={"Content-Disposition": 'attachment; filename="old.doc"'}))


def test_an_unknown_type_says_what_the_server_gave() -> None:
    """认不出类型时要把线索写出来：URL 没后缀、服务器给的 Content-Type 是什么。"""
    with pytest.raises(FetchError, match="application/octet-stream"):
        _fetch(
            _ok(headers={"Content-Type": "application/octet-stream"}),
            "https://example.com/download",
        )


def test_oversized_downloads_are_cut_off() -> None:
    with pytest.raises(FetchError, match="超过上限"):
        _fetch(_ok(b"x" * 500, headers={"Content-Type": "text/html"}), max_bytes=100)


def test_remote_errors_report_the_status_code() -> None:
    """远端不给（403 / 限流 / 要登录）→ 502，并带上状态码，别吞成「解析失败」。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, request=request)

    with pytest.raises(FetchError) as info:
        _fetch(handler)

    assert info.value.status == 502
    assert "403" in str(info.value)


def test_too_many_redirects_stops_instead_of_looping() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "/next"}, request=request)

    with pytest.raises(FetchError, match="重定向"):
        _fetch(handler, max_redirects=2)


# ---- 页面里的图片：一起下下来，HTML 里的 src 改写成相对路径 ----

_PNG = b"\x89PNG\r\n\x1a\nfake-png"

_PAGE = (
    b"<html><body><nav><img src='/nav/logo.svg'></nav>"
    b"<h1>Hi</h1><p>text</p><img src='_images/a.png'>"
    b"<img src='/img/b.jpg'><img data-src='/img/lazy.png' src='/img/blank.gif'>"
    b"</body></html>"
)


def _image_handler(request: httpx.Request) -> httpx.Response:
    """一个「文档页 + 几张图」的假站点（相对路径按浏览器的语义解析）。"""
    if request.url.path == "/docs/page.html":
        return httpx.Response(
            200, content=_PAGE, headers={"Content-Type": "text/html"}, request=request
        )
    if request.url.path == "/docs/_images/a.png":
        return httpx.Response(
            200, content=_PNG, headers={"Content-Type": "image/png"}, request=request
        )
    if request.url.path == "/img/b.jpg":
        return httpx.Response(
            200, content=b"JPEG", headers={"Content-Type": "image/jpeg"}, request=request
        )
    if request.url.path == "/img/lazy.png":
        return httpx.Response(
            200, content=_PNG, headers={"Content-Type": "image/png"}, request=request
        )
    return httpx.Response(404, request=request)


def test_images_in_a_page_are_downloaded_and_the_src_is_rewritten() -> None:
    """**这条正是那个缺口**：抓 HTML 时页面里的图一起下，`src` 改成本地相对路径。"""
    item = _fetch(_image_handler, "https://example.com/docs/page.html")

    assert item.name == "page.html" and item.notes == []
    folder = f"page-{data_hash(_PAGE)}_files"
    assert sorted(item.attachments) == [
        f"{folder}/img01.png",
        f"{folder}/img02.jpg",
        f"{folder}/img03.png",
    ]
    assert item.attachments[f"{folder}/img01.png"] == _PNG
    text = item.data.decode("utf-8")
    assert f'src="{folder}/img01.png"' in text
    # 懒加载那张：真图在 `data-src` 里，`src` 是占位图 —— 两者都改成本地路径，
    # 否则 loader 先看到 `src`，读到的还是占位图
    assert f'src="{folder}/img03.png"' in text and f'data-src="{folder}/img03.png"' in text
    assert "blank.gif" not in text, "占位图不该留在 HTML 里"
    assert "/nav/logo.svg" in text, "导航里的图不抓（也不改写）—— 它不算正文"
    assert "logo.svg" not in "".join(item.attachments), "别把侧边栏图标也下下来"


def test_a_fetched_page_with_images_reaches_the_loader(tmp_path) -> None:
    """**端到端**：抓回来 → 落盘（HTML + 图片）→ loader 出图片标记 → 资产也落好了。

    这一段是「url 解析之后的图片没有拿到」那个问题的正面验收。
    """
    from rag.config.schema import IngestAssetsConfig, IngestConfig
    from rag.ingest.assets import markers_in
    from rag.ingest.loaders import build_loader

    item = _fetch(_image_handler, "https://example.com/docs/page.html")
    (tmp_path / item.name).write_bytes(item.data)
    for relative, blob in item.attachments.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)

    cfg = IngestConfig(assets=IngestAssetsConfig(dir=str(tmp_path / "assets")))
    doc = build_loader(".html", cfg).load(tmp_path / item.name, root=tmp_path)

    assert len(markers_in(doc.text)) == 3, "正文里的三张图都成了标记"
    assert doc.metadata["image_count"] == 3
    assert "images_missing" not in doc.metadata, "都拿到了，一条缺图记录都不该有"
    for image in doc.images:
        assert (tmp_path / "assets" / image.asset_uri).is_file()


def test_a_broken_image_is_noted_and_the_rest_still_come() -> None:
    """一张图 404 不该让整页抓取失败 —— 记下来、其余照抓。"""
    item = _fetch(_image_handler, "https://example.com/docs/page.html", max_images=1)

    assert len(item.attachments) == 1, "上限 1 就只下一张"
    assert any("max_images=1" in note for note in item.notes)


def test_an_image_that_is_not_really_an_image_is_noted() -> None:
    """服务器对着一张图回 `text/html`（登录页/防盗链）→ 不当图片收，记账。"""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/page.html":
            return httpx.Response(
                200,
                content=b'<p>x</p><img src="a.png">',
                headers={"Content-Type": "text/html"},
                request=request,
            )
        return httpx.Response(
            200,
            content=b"<html>login</html>",
            headers={"Content-Type": "text/html"},
            request=request,
        )

    item = _fetch(handler, "https://example.com/page.html")

    assert item.attachments == {}
    assert any("不是能进库的图片格式" in note for note in item.notes)
    assert b'<img src="a.png">' in item.data, "没抓到的图不动原文（loader 会记账）"


def test_a_non_html_download_has_no_image_pass() -> None:
    """PDF / Word 那些没有「页面里的图」这回事，照原样收一份文件。"""
    item = _fetch(_ok(b"%PDF-1.4", headers={"Content-Type": "application/pdf"}))

    assert item.attachments == {} and item.notes == [] and item.data == b"%PDF-1.4"
