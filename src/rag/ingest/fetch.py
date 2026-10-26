"""按 URL 抓一份文档：抓回来的东西**当成一份普通上传件**落盘。

为什么不把字节直接喂给解析器、而是先落成文件：下游整条链路（doc_id、stages 产物、
增量比对、Milvus upsert）全部按**文件路径**走。先落盘，两条入口（页面上传 / 链接导入）
就共用同一套东西，下游一行都不用改 —— 而且「重新抓一次」天然就是「重新传一份」。

**类型不定**（使用者 2026-10-07 定的）：链接可能是 HTML 文档页，也可能是 PDF / Word /
表格的下载链接。后缀按 `Content-Disposition` → URL 末段 → `Content-Type` 反查的顺序定
（见 `_name_for`），三者都定不出来就拒绝，不猜。

**抓 HTML 时连页面里的图片一起抓**（2026-10-07 加）：按 `loaders.html.content_image_urls`
（与 loader 同一套「哪些图算正文」的判定）把 `<img>` 逐个下下来，落在同级的
``<真名>-<页面内容哈希>_files/`` 目录里，并把 HTML 里的 `src` **改写成相对路径** ——
形态就是「浏览器另存为网页」。这样 loader 只从磁盘读图，不用联网（loader 一旦联网，
`rag ingest ./data/docs` 就会意外出网）。

目录名用**页面内容哈希**而不是 URL：同一链接内容变了，新旧两份 HTML 各带各的图
（那时页面通常也真换了图），互相不踩。落盘那份 HTML 是 BeautifulSoup 重新序列化过的
快照（编码统一成 utf-8）—— 它是给我们自己读的，不是给人对照原站的。

**只做静态抓取**：不跑 headless 浏览器，所以 JS 渲染的页面抓回来基本是空的 ——
那种情况由 `load_paths` 报「解析后正文为空」，不会静默成功；需要登录才能下载的链接会
拿到 302/403，也如实报出来（不做 cookie 注入）。编码不用在这里操心：几种文本 loader
都走 `read_text_any_encoding`（utf-8 → gb18030），中文站点的 GBK 系页面照样读得出来。

**SSRF 是这里唯一的新风险**：服务端替用户抓任意 URL，等于把这台机器当成探内网的代理
（`http://169.254.169.254/` 就是云上取凭据的经典目标）。三条对策缺一不可：

1. 只允许 http / https；
2. 每一跳的**主机解析出来的每个 IP** 都必须是公网地址（私网 / 环回 / 链路本地 / 保留段
   一律拒绝）；
3. **重定向自己走**（`follow_redirects=False` 手动跳），每跳都重查一遍 —— 交给 httpx
   自动跟的话，`http://evil/ → http://127.0.0.1:8000/` 就绕过去了。

本机自建服务（比如把内网的文档站也收进来）要放行就改
`ingest.uploads.fetch.allow_private_hosts`；默认是关的。
"""

from __future__ import annotations

import ipaddress
import re
import socket
from collections.abc import Callable
from dataclasses import dataclass, field
from email.message import Message
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup

from rag.ingest.cleaning import clean_markdown
from rag.ingest.firecrawl import FirecrawlError, FirecrawlOptions, scrape_markdown
from rag.ingest.loaders import LOADERS, decode_text, suffix_for_mime
from rag.ingest.loaders.base import data_hash, image_suffixes
from rag.ingest.loaders.html import IMG_SRC_ATTRS, content_image_urls, img_source

# 默认最多跟几次跳转。链子太长多半是登录/同意页在来回弹，跟下去没意义
MAX_REDIRECTS = 5

# 一页最多抓几张图 / 单张图多大。图片描述是**按量计费**的那一步（VLM），
# 这两条是钱的口子，默认值故意保守；要放开就改 `ingest.uploads.fetch`。
MAX_IMAGES = 20
MAX_IMAGE_BYTES = 8_000_000

# 有的站点直接挡空 User-Agent。写清楚自己是谁，不做伪装
_UA = "multimodal_rag/0.1 (+ingest)"

# 这些状态码带 Location，其余一律按错误处理
_REDIRECTS = (301, 302, 303, 307, 308)


@dataclass(frozen=True)
class Fetched:
    """抓回来的那一份。`name` 是**真名**（含后缀）；落盘时由 uploads 再加 `-hash`。

    `attachments` 是页面引用的图片：``{相对路径: 图片本体}``，与 `data` 里的 `src`
    一一对应，由调用方写到上传目录的同名子目录下。`notes` 是**没抓到的图**与原因，
    调用方要把它写进日志、回给页面 —— 悄悄少几张图是查不出来的。
    """

    name: str
    data: bytes
    content_type: str
    final_url: str
    attachments: dict[str, bytes] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Response:
    """一跳之后拿到的东西（页面与图片共用）。"""

    url: str
    data: bytes
    content_type: str
    disposition: str


class FetchError(ValueError):
    """抓取失败。

    `status` 是给 HTTP 层用的：**用户给的 URL 有问题 → 400，远端不给 → 502**。
    两种要分开 —— 页面上的提示语与排查方向完全不同（改链接 vs 找对方站点）。
    """

    def __init__(self, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def _resolve(host: str) -> list[str]:
    """主机名 → 全部地址（IPv4 + IPv6）。返回空列表 = 解析不出东西。"""
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    return sorted({info[4][0] for info in infos})


def _assert_public(url: str, *, resolver: Callable[[str], list[str]], allow_private: bool) -> None:
    """这一跳的主机必须是公网地址。**每个地址都要查**：一个域名可以同时挂公网与内网 IP。"""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise FetchError(f"只支持 http / https 链接，收到的是 {parts.scheme or '（没有协议）'!r}")
    host = parts.hostname
    if not host:
        raise FetchError(f"链接里没有主机名：{url!r}")
    if allow_private:
        return
    try:
        addresses = resolver(host)
    except OSError as exc:
        raise FetchError(f"解析不了主机名 {host!r}：{exc}", status=502) from exc
    if not addresses:
        raise FetchError(f"主机名 {host!r} 没解析出地址", status=502)
    for address in addresses:
        # is_global 一票否决：私网、环回、链路本地（含 169.254 与 fe80::）、
        # 保留段、运营商级 NAT（100.64/10）、组播全在里面
        if not ipaddress.ip_address(address).is_global:
            raise FetchError(f"{host!r} 指向非公网地址 {address}，拒绝抓取（防内网探测）")


def _read_capped(response: httpx.Response, *, max_bytes: int, url: str) -> bytes:
    """流式收，**边收边数**：不能先全下到内存再说 —— 那就等于没有上限。"""
    chunks: list[bytes] = []
    total = 0
    for block in response.iter_bytes():
        total += len(block)
        if total > max_bytes:
            raise FetchError(
                f"{url} 超过上限 {max_bytes / 1e6:.0f}MB，已中止下载"
                f"（要抓更大的文件就调大 ingest.uploads.max_mb）"
            )
        chunks.append(block)
    return b"".join(chunks)


def _disposition_name(header: str) -> str:
    """`Content-Disposition` 里的文件名。

    用 `email.message` 解而不是自己写正则：RFC 2231 那种
    `filename*=UTF-8''%E6%89%8B%E5%86%8C.pdf`（中文名的常见写法）它替我们处理好了。
    """
    if not header:
        return ""
    message = Message()
    message["content-disposition"] = header
    return (message.get_filename() or "").strip()


def _name_for(url: str, disposition: str, content_type: str) -> str:
    """真名 + 后缀。

    真名：`Content-Disposition` → URL 末段 → 主机名（下载链接常常是 `/download?id=7`，
    两个线索都没有，只能拿主机名兜底）。
    后缀：名字里带的那个优先；**带了但不支持就直接拒绝**（`.doc` 这种老格式不在名单里，
    不能偷偷按 `.docx` 试）；没带就用 `Content-Type` 反查；还认不出就报错，
    并把服务器给的 Content-Type 原样写出来 —— 那种「什么都没说清」的错最难查。
    """
    raw = _disposition_name(disposition) or Path(unquote(urlsplit(url).path)).name
    suffix = Path(raw).suffix.lower()
    stem = raw[: -len(suffix)] if suffix else raw
    if suffix and suffix not in LOADERS:
        raise FetchError(
            f"链接给的是一份 {suffix} 文件，这种格式不支持；支持：{'、'.join(sorted(LOADERS))}"
        )
    if not suffix:
        suffix = suffix_for_mime(content_type)
    if not suffix:
        raise FetchError(
            f"认不出这是什么文件：{url} 没有后缀，服务器给的 Content-Type 是 "
            f"{content_type or '（没给）'}。可以改用本地上传，或把 URL 指向带后缀的下载链接。"
        )
    stem = stem.strip() or urlsplit(url).hostname or "download"
    return f"{stem}{suffix}"


def fetch_url(
    url: str,
    *,
    max_bytes: int,
    timeout_s: float = 15.0,
    max_redirects: int = MAX_REDIRECTS,
    allow_private_hosts: bool = False,
    max_images: int = MAX_IMAGES,
    max_image_bytes: int = MAX_IMAGE_BYTES,
    client: httpx.Client | None = None,
    resolver: Callable[[str], list[str]] = _resolve,
    backend: str = "http",
    firecrawl: FirecrawlOptions | None = None,
    firecrawl_client: httpx.Client | None = None,
) -> Fetched:
    """抓一份文档。失败一律抛 `FetchError`（消息就是给用户看的那句话）。

    HTML 会**连页面里的图片一起抓**（见模块说明）；其它类型就是一份文件。
    `client` / `resolver` 是给测试的口子：`tests/test_fetch.py` 用 `httpx.MockTransport`
    加一个假的 resolver，整条路不碰网络。

    ``backend`` 决定**网页**怎么抓：

    * ``http``（默认）—— 本机 GET + BeautifulSoup 抽正文；
    * ``firecrawl`` —— 网页交给 Firecrawl 的 `/scrape`（带 `onlyMainContent`，正文干净），
      **图片仍由我们下到本地**（我们的图片管线要的是本体，见 `_localize_markdown_images`）。
      **PDF / 图片 / 附件不走 Firecrawl**：那几条路要的是**原件**（版面模型给标题层级、
      抽图、页码行号），只给 markdown 会把它们全丢掉。
    """
    if backend not in ("http", "firecrawl"):
        raise ValueError(f"未知的 ingest.uploads.fetch.backend: {backend!r}；可选 http / firecrawl")
    owned = client is None
    client = client or httpx.Client(
        timeout=httpx.Timeout(timeout_s),
        # **不让 httpx 自己跟跳转**：每一跳都要过一次 IP 检查（见模块说明第 3 条）
        follow_redirects=False,
        headers={"User-Agent": _UA},
    )
    try:
        got = _download(
            client,
            url.strip(),
            max_bytes=max_bytes,
            max_redirects=max_redirects,
            resolver=resolver,
            allow_private=allow_private_hosts,
        )
        name = _name_for(got.url, got.disposition, got.content_type)
        is_html = Path(name).suffix.lower() in (".html", ".htm")
        if is_html and backend == "firecrawl":
            return _fetch_via_firecrawl(
                got,
                name=name,
                options=firecrawl,
                firecrawl_client=firecrawl_client,
                client=client,
                max_images=max_images,
                max_image_bytes=max_image_bytes,
                max_redirects=max_redirects,
                resolver=resolver,
                allow_private=allow_private_hosts,
            )
        if not is_html:
            return Fetched(
                name=name, data=got.data, content_type=got.content_type, final_url=got.url
            )
        data, attachments, notes = _with_local_images(
            got,
            display=name,
            client=client,
            max_images=max_images,
            max_image_bytes=max_image_bytes,
            max_redirects=max_redirects,
            resolver=resolver,
            allow_private=allow_private_hosts,
        )
        return Fetched(
            name=name,
            data=data,
            content_type=got.content_type,
            final_url=got.url,
            attachments=attachments,
            notes=notes,
        )
    finally:
        if owned:
            client.close()


def _fetch_via_firecrawl(
    got: _Response,
    *,
    name: str,
    options: FirecrawlOptions | None,
    firecrawl_client: httpx.Client | None,
    client: httpx.Client,
    max_images: int,
    max_image_bytes: int,
    max_redirects: int,
    resolver: Callable[[str], list[str]],
    allow_private: bool,
) -> Fetched:
    """网页交给 Firecrawl：正文用它给的 markdown，**图片仍由我们下到本地**。

    为什么图还是自己下：图片管线的下游（assets → VLM 描述 → 多模态 embedding）要的是
    **图片本体**，而 markdown 里给的是绝对 URL —— 不落地就等于这条线断了。命名与 HTML
    那条路完全一致（`<真名>-<哈希>_files/imgNN.ext`），落盘/上传那一层不用改一个字。

    落盘件也从 `.html` 变成 `.md`：正文已经是 markdown（标题就是 `#`），走我们已有的
    markdown loader 比"HTML 抽完再猜层级"更准。
    """
    if options is None:
        raise FetchError(
            "fetch.backend=firecrawl，但没读进 Firecrawl 的配置"
            "（检查 ingest.uploads.fetch.firecrawl_api_url 与 .env 里的 FIRECRAWL_API_KEY）",
            status=500,
        )
    try:
        page = scrape_markdown(got.url, options, client=firecrawl_client)
    except FirecrawlError as exc:
        raise FetchError(f"Firecrawl 没抓成：{exc}", status=502) from exc

    # 开头那两坨站点 chrome（`> ## Documentation Index … llms.txt`、`[Skip to main content](…)`）
    # 在这里就先清掉：`clean_markdown()` 的规则只看**文档开头**，而这份 markdown 会被原样落盘
    # （人读的那份产物），留着它俩既脏又会被切块进检索。loader 那边还会再清一遍（幂等）。
    #
    # **不拿 `page.title` 往正文里补标题**（2026-10-10 定）：那个 title 是页面的 `<title>`，
    # 在原始 HTML 里对应的是**站点 header**，不是正文的 `<h1>` —— 补进正文等于往语料里塞站点名。
    markdown = clean_markdown(page.markdown)
    markdown_bytes = markdown.encode("utf-8")
    display = f"{Path(name).stem}.md"
    folder = f"{Path(display).stem}-{data_hash(markdown_bytes)}_files"
    text, attachments, notes = _localize_markdown_images(
        markdown,
        base=page.final_url or got.url,
        folder=folder,
        client=client,
        max_images=max_images,
        max_image_bytes=max_image_bytes,
        max_redirects=max_redirects,
        resolver=resolver,
        allow_private=allow_private,
    )
    if page.warning:
        notes.append(f"Firecrawl 提示：{page.warning}")
    return Fetched(
        name=display,
        data=text.encode("utf-8"),
        content_type="text/markdown",
        final_url=page.final_url or got.url,
        attachments=attachments,
        notes=notes,
    )


# markdown 里的图片两种写法都认：`![alt](url)` 与内联 `<img src="url">`。
# 云端已经把 base64 图去掉了（`removeBase64Images`），所以这里只处理普通 URL。
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\(\s*<?([^)\s>]+)>?")
_MD_IMG_TAG = re.compile(r"<img[^>]+src=[\"']([^\"']+)[\"']", re.IGNORECASE)


def _localize_markdown_images(
    text: str,
    *,
    base: str,
    folder: str,
    client: httpx.Client,
    max_images: int,
    max_image_bytes: int,
    max_redirects: int,
    resolver: Callable[[str], list[str]],
    allow_private: bool,
) -> tuple[str, dict[str, bytes], list[str]]:
    """把 markdown 里的图片下下来、把地址改成本地相对路径。

    返回 ``(新的 markdown, {相对路径: 图片本体}, 没抓到的原因)`` —— 与 HTML 那条路同一
    套出口（`_with_local_images`）。相对地址按 `base`（重定向之后的页面地址）解析。
    """
    sources: list[str] = []
    for raw in (*_MD_IMAGE.findall(text), *_MD_IMG_TAG.findall(text)):
        absolute = urljoin(base, raw.strip())
        if absolute.startswith(("data:", "asset:")) or absolute in sources:
            continue
        sources.append(absolute)
    if not sources:
        return text, {}, []

    mapping: dict[str, str] = {}
    attachments: dict[str, bytes] = {}
    notes: list[str] = []
    for source in sources:
        if len(mapping) >= max_images:
            notes.append(
                f"还有 {len(sources) - len(mapping)} 张图超过上限 max_images={max_images}，没下"
            )
            break
        try:
            got = _download(
                client,
                source,
                max_bytes=max_image_bytes,
                max_redirects=max_redirects,
                resolver=resolver,
                allow_private=allow_private,
            )
        except FetchError as exc:
            notes.append(f"{source}：{exc}")
            continue
        suffix = _image_suffix(got.content_type, source)
        if suffix is None:
            notes.append(
                f"{source}：不是能进库的图片格式（服务器给的是 {got.content_type or '（没给）'}）"
            )
            continue
        relative = f"{folder}/img{len(mapping) + 1:02d}{suffix}"
        mapping[source] = relative
        attachments[relative] = got.data

    out = text
    for source, relative in mapping.items():
        out = out.replace(source, relative)
    return out, attachments, notes


def _download(
    client: httpx.Client,
    url: str,
    *,
    max_bytes: int,
    max_redirects: int,
    resolver: Callable[[str], list[str]],
    allow_private: bool,
) -> _Response:
    """取一份内容（页面或图片）。**每一跳都过一次 IP 检查**，见模块说明第 3 条。"""
    current = url
    for _ in range(max_redirects + 1):
        _assert_public(current, resolver=resolver, allow_private=allow_private)
        try:
            with client.stream("GET", current) as response:
                if response.status_code in _REDIRECTS:
                    location = response.headers.get("location")
                    if not location:
                        raise FetchError(
                            f"{current} 返回 {response.status_code} 却没给 Location", status=502
                        )
                    current = str(response.url.join(location))
                    continue
                if response.status_code >= 400:
                    raise FetchError(
                        f"站点返回 {response.status_code}（{current}）"
                        f"；需要登录或对方限流时就是这种结果",
                        status=502,
                    )
                data = _read_capped(response, max_bytes=max_bytes, url=current)
                return _Response(
                    url=current,
                    data=data,
                    content_type=response.headers.get("content-type", "")
                    .split(";")[0]
                    .strip()
                    .lower(),
                    disposition=response.headers.get("content-disposition", ""),
                )
        except httpx.HTTPError as exc:
            raise FetchError(f"连不上 {current}：{type(exc).__name__}: {exc}", status=502) from exc
    raise FetchError(f"重定向超过 {max_redirects} 次（最后停在 {current}）", status=502)


def _with_local_images(
    page: _Response,
    *,
    display: str,
    client: httpx.Client,
    max_images: int,
    max_image_bytes: int,
    max_redirects: int,
    resolver: Callable[[str], list[str]],
    allow_private: bool,
) -> tuple[bytes, dict[str, bytes], list[str]]:
    """把页面里的图片下下来、把 HTML 里的 `src` 改写成相对路径。

    返回 ``(新的 HTML 字节, {相对路径: 图片本体}, 没抓到的原因)``。
    """
    text = decode_text(page.data)
    sources = content_image_urls(text, page.url)
    if not sources:
        return page.data, {}, []
    # 目录名带**页面内容哈希**：同一链接内容变了，两份 HTML 各带各的图，互不踩
    folder = f"{Path(display).stem}-{data_hash(page.data)}_files"
    mapping: dict[str, str] = {}
    attachments: dict[str, bytes] = {}
    notes: list[str] = []
    for source in sources:
        if len(mapping) >= max_images:
            notes.append(
                f"还有 {len(sources) - len(mapping)} 张图超过上限 max_images={max_images}，没下"
            )
            break
        try:
            got = _download(
                client,
                source,
                max_bytes=max_image_bytes,
                max_redirects=max_redirects,
                resolver=resolver,
                allow_private=allow_private,
            )
        except FetchError as exc:
            notes.append(f"{source}：{exc}")
            continue
        suffix = _image_suffix(got.content_type, source)
        if suffix is None:
            notes.append(
                f"{source}：不是能进库的图片格式（服务器给的是 {got.content_type or '（没给）'}）"
            )
            continue
        relative = f"{folder}/img{len(mapping) + 1:02d}{suffix}"
        mapping[source] = relative
        attachments[relative] = got.data
    return _rewrite_sources(text, mapping, page.url).encode("utf-8"), attachments, notes


def _image_suffix(content_type: str, source: str) -> str | None:
    """图片的后缀。两种情况分开：

    * `Content-Type` 是个**明确的非图片类型**（`text/html` 之类）→ 直接拒：那多半是
      防盗链或登录页返回的一页 HTML，当成图片收下来，VLM 那边只会拿到一张废图。
    * 没给类型、或给了 `application/octet-stream` 这种「我不知道」的通用类型 → 退回
      **URL 自己的后缀**：不少静态站就是这么发图的（实测 Spring 那几个文档站是相对路径
      + 正常类型，但对象存储回 octet-stream 很常见）。
    """
    if content_type and content_type != "application/octet-stream":
        from_mime = suffix_for_mime(content_type)
        return from_mime if from_mime in image_suffixes() else None
    from_url = Path(urlsplit(source).path).suffix.lower()
    return from_url if from_url in image_suffixes() else None


def _rewrite_sources(text: str, mapping: dict[str, str], base: str) -> str:
    """把命中的 `<img>` 的地址改成本地相对路径。

    用 BeautifulSoup 改属性再序列化（不是手写正则）：`>` 出现在属性值里的那种标签
    正则挡不住，而序列化出来的 HTML 反正是给我们自己的 loader 读的。

    **地址原本写在哪个属性，那个属性也要跟着改**：只改 `src` 的话，懒加载那张图的
    `data-src` 还挂着外链 —— 而 loader 读图时先看 `data-src`，等于白下了一张。
    """
    if not mapping:
        return text
    soup = BeautifulSoup(text, "html.parser")
    for node in soup.find_all("img"):
        source = img_source(node)
        if not source:
            continue
        local = mapping.get(urljoin(base, source))
        if local is None:
            continue
        node["src"] = local
        for attr in IMG_SRC_ATTRS:
            value = node.get(attr)
            if isinstance(value, str) and urljoin(base, value) in mapping:
                node[attr] = mapping[urljoin(base, value)]
    return str(soup)
