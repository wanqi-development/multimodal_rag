"""网络小工具：**本机 / 内网的服务不该被系统代理劫持**（2026-10-10）。

起因是实测出来的：机器上开着系统代理（Clash 那类，`ProxyServer=127.0.0.1:7897`），
而 httpx 与 openai SDK 的 `trust_env=True` **只认 `NO_PROXY` 环境变量**，不看 Windows
「跳过代理的地址」那份列表（系统的 ProxyOverride 里明明写了 `127.*` 与 `localhost`，
httpx 照样把请求发给代理）。于是打 `http://127.0.0.1:6006` 的 dots.ocr 请求被送到代理，
代理够不着本机服务，回一个 **502** —— 日志里就是一片「dots.ocr 这一页失败（第 2 次，
InternalServerError）：Error code: 502」，重试几十次、一页都解析不出来。

两道保险，各管一段：

1. `trust_env_for(url)` —— **我们自己的** httpx 客户端按地址决定要不要看环境代理
   （回环 / 内网 → 不看）。这一道不依赖机器设置，是首选。
2. `ensure_local_bypasses_proxy()` —— 进程启动时把回环 / 内网地址并进 `NO_PROXY`。
   这一道兜住**别人的客户端**：dots 那条走的是 vendored 代码里的 `OpenAI(...)`
   （`ingest/dots_ocr/model/inference.py`），我们不好往里塞 http_client。
"""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlsplit

# 名单类写法：`localhost` 这类名字 `ipaddress` 认不出来，得单独列
_LOOPBACK_NAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost"})

# 无论配置怎么写，这三个一定该绕过代理（回环地址的三种写法）
_ALWAYS_BYPASS = ("127.0.0.1", "localhost", "::1")


def host_of(url_or_host: str) -> str:
    """从 `http://host:port/...` 或裸 `host:port` 里取主机名（取不出来就给空串）。"""
    raw = (url_or_host or "").strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = f"//{raw}"
    try:
        return (urlsplit(raw).hostname or "").strip()
    except ValueError:  # 端口写坏了之类
        return ""


def is_local_host(host: str) -> bool:
    """回环 / 私有网段 / 链路本地 —— 这些地址上的服务只可能在本机或内网。"""
    name = (host or "").strip().strip("[]").lower()
    if not name:
        return False
    if name in _LOOPBACK_NAMES or name.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(name)
    except ValueError:
        return False  # 普通域名（云端 API）不猜，让代理正常发挥作用
    return address.is_loopback or address.is_private or address.is_link_local


def is_local_service(url_or_host: str) -> bool:
    return is_local_host(host_of(url_or_host))


def trust_env_for(url_or_host: str) -> bool:
    """这个地址要不要理会环境里的代理设置？

    `httpx.Client(trust_env=trust_env_for(base_url))`：本机 / 内网地址一律不看代理，
    公网地址照旧（国内的机器上，云端 API 常常正是靠那个代理才连得出去）。
    """
    return not is_local_service(url_or_host)


def configured_service_urls(settings: Any) -> list[str]:
    """配置里那些**可能是本机 / 内网服务**的地址（duck-typing，缺字段就跳过）。

    只挑会走 HTTP 客户端的那几个：OCR 产线（dots / PaddleOCR / MinerU）。Milvus 走 gRPC、
    MinIO 走自己的 urllib3（两者都不看环境代理），列进来只会让名单变长。
    """
    paths = (
        ("ingest", "dots", "base_url"),
        ("ingest", "mineru", "base_url"),
        ("ingest", "ocr", "base_url"),
    )
    urls: list[str] = []
    for path in paths:
        node: Any = settings
        for key in path:
            node = getattr(node, key, None)
            if node is None:
                break
        if isinstance(node, str) and node.strip():
            urls.append(node)
    return urls


def ensure_local_bypasses_proxy(hosts: Iterable[str] = ()) -> str:
    """把回环 / 内网地址并进 `NO_PROXY`（已设置的条目保留），返回合并后的名单。

    进程启动时调一次即可（httpx 在**建客户端**时读环境，所以要在建之前调）。
    """
    entries = list(_ALWAYS_BYPASS)
    for item in hosts:
        host = host_of(item)
        if host and is_local_host(host) and host not in entries:
            entries.append(host)

    current = os.environ.get("NO_PROXY", "") or ""
    known = {part.strip() for part in current.split(",") if part.strip()}
    merged = current.strip()
    for entry in entries:
        if entry in known:
            continue
        merged = f"{merged},{entry}" if merged else entry
        known.add(entry)
    if merged:
        # 大小写两个都写：不同库读的名字不一样（Windows 上 os.environ 不区分大小写，
        # 写两遍等价于写一遍，不会有副作用）
        os.environ["NO_PROXY"] = merged
        os.environ["no_proxy"] = merged
    return merged
