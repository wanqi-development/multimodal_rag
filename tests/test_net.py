"""本机 / 内网服务不能被系统代理劫持（2026-10-10）。

实测现场：机器开着系统代理，而 httpx / openai SDK 只认 `NO_PROXY`、不看 Windows
「跳过代理的地址」那份列表 —— 打 `http://127.0.0.1:6006` 的 dots.ocr 请求被送到代理，
代理够不着本机服务，回 502，一页 PDF 都解析不出来。
"""

from __future__ import annotations

import os

from rag.config.schema import Settings
from rag.net import (
    configured_service_urls,
    ensure_local_bypasses_proxy,
    host_of,
    is_local_host,
    is_local_service,
    trust_env_for,
)


def test_host_of_understands_urls_and_bare_hosts() -> None:
    assert host_of("http://127.0.0.1:6006/v1/models") == "127.0.0.1"
    assert host_of("127.0.0.1:8080") == "127.0.0.1"
    assert host_of("192.168.1.10:9000") == "192.168.1.10"
    assert host_of("https://api.deepseek.com/v1") == "api.deepseek.com"
    assert host_of("") == ""


def test_loopback_and_private_addresses_are_local() -> None:
    for value in (
        "http://127.0.0.1:6006",
        "http://localhost:8080",
        "http://[::1]:8080",
        "http://192.168.1.10:9000",
        "http://10.0.0.5:7860",
        "http://172.16.3.9:8080",
    ):
        assert is_local_service(value), value


def test_public_hosts_are_not_local() -> None:
    for value in (
        "https://api.deepseek.com/v1",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "https://hf-mirror.com",
    ):
        assert not is_local_service(value), value
        assert is_local_host(host_of(value)) is False


def test_trust_env_is_off_for_local_services_only() -> None:
    """本机服务的客户端不理会环境代理；云端 API 照旧（它常常正靠那个代理才连得上）。"""
    assert trust_env_for("http://127.0.0.1:8080") is False
    assert trust_env_for("http://192.168.1.10:6006") is False
    assert trust_env_for("https://api.deepseek.com/v1") is True


def test_the_bypass_list_is_merged_not_replaced(monkeypatch) -> None:
    """已经设过的条目要留着 —— 机器上可能有人为别的服务配过。"""
    monkeypatch.setenv("NO_PROXY", "example.com")

    merged = ensure_local_bypasses_proxy(["http://127.0.0.1:8080", "http://192.168.1.10:6006"])

    entries = merged.split(",")
    assert "example.com" in entries, "别人写的不许丢"
    for entry in ("127.0.0.1", "localhost", "::1", "192.168.1.10"):
        assert entry in entries, entry
    assert os.environ["NO_PROXY"] == merged, "写完要让 httpx 真的读得到"


def test_a_public_host_never_enters_the_bypass_list(monkeypatch) -> None:
    monkeypatch.delenv("NO_PROXY", raising=False)

    merged = ensure_local_bypasses_proxy(["https://api.deepseek.com/v1"])

    assert "deepseek" not in merged
    assert "127.0.0.1" in merged.split(","), "回环地址无论如何都要绕过代理"


def test_configured_urls_pulls_the_local_ocr_endpoints() -> None:
    settings = Settings()
    settings.ingest.dots.base_url = "http://127.0.0.1:6006"
    settings.ingest.mineru.base_url = "http://192.168.1.10:7000"
    settings.ingest.ocr.base_url = None  # 没配 = 不启用，不该出现在名单里

    urls = configured_service_urls(settings)

    assert urls == ["http://127.0.0.1:6006", "http://192.168.1.10:7000"]
