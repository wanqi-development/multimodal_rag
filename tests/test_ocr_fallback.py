"""产线挂了就切云端：切换的**条件**与**可见性**。

真调云端要 access token（按量计费），这里全部打桩 —— 验的是「什么时候切、
切之前有没有出声、以及云端的返回怎么映射成我们内部的 OcrPage」。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from rag.config.schema import IngestCloudOcrConfig, IngestOcrConfig, IngestPdfConfig
from rag.ingest.ocr import FallbackOcr, OcrClient, OcrPage, OcrUnavailable, build_ocr


class _Primary:
    """本地产线替身：按预设抛错或返回。"""

    base_url = "http://local:8080"

    def __init__(self, error: Exception | None = None) -> None:
        self.calls = 0
        self._error = error

    def parse(self, path: Path, *, file_type: int) -> list[OcrPage]:
        self.calls += 1
        if self._error:
            raise self._error
        return [OcrPage(page_index=0, markdown="本地结果")]


class _Cloud:
    base_url = "https://paddleocr.aistudio-app.com"

    def __init__(self) -> None:
        self.calls = 0

    def parse(self, path: Path, *, file_type: int) -> list[OcrPage]:
        self.calls += 1
        return [OcrPage(page_index=0, markdown="云端结果")]


def _fallback(error: Exception | None) -> tuple[FallbackOcr, _Primary, _Cloud, list[str]]:
    primary, cloud, seen = _Primary(error), _Cloud(), []
    return FallbackOcr(primary, cloud, warn=seen.append), primary, cloud, seen


# ---- 切换的条件 ----­


def test_unavailable_service_switches_to_the_cloud() -> None:
    fallback, primary, cloud, seen = _fallback(OcrUnavailable("连不上"))

    pages = fallback.parse(Path("x.pdf"), file_type=0)

    assert primary.calls == 1
    assert cloud.calls == 1
    assert pages[0].markdown == "云端结果"


def test_a_normal_error_does_not_switch() -> None:
    """请求本身有问题时换一家也一样失败，白等一轮还把真正的原因盖掉。"""
    fallback, _, cloud, seen = _fallback(RuntimeError("报文不对"))

    with pytest.raises(RuntimeError, match="报文不对"):
        fallback.parse(Path("x.pdf"), file_type=0)

    assert cloud.calls == 0
    assert seen == []


def test_a_healthy_line_never_touches_the_cloud() -> None:
    fallback, _, cloud, seen = _fallback(None)

    pages = fallback.parse(Path("x.pdf"), file_type=0)

    assert pages[0].markdown == "本地结果"
    assert cloud.calls == 0
    assert seen == []


# ---- 切云端之前先原地重试 ----（实测那条转发在真实负载下 28% 失败）


class _Flaky:
    """第一次不可用、第二次就好了的产线 —— 实测里最常见的形态（偶发 502）。"""

    def __init__(self, failures: int) -> None:
        self.calls = 0
        self._failures = failures

    def parse(self, path: Path, *, file_type: int) -> list[OcrPage]:
        self.calls += 1
        if self.calls <= self._failures:
            raise OcrUnavailable("502 Bad Gateway")
        return [OcrPage(page_index=0, markdown="本地重试成功")]


def _flaky(failures: int, *, retry_primary: int = 1) -> tuple[Any, list[str]]:
    seen: list[str] = []
    primary = _Flaky(failures)
    client = OcrClient(
        IngestOcrConfig(base_url="http://local:8080", retry_primary=retry_primary), warn=seen.append
    )
    client.__dict__["_parse_once"] = primary.parse  # 只替掉「发一次请求」那一步
    return client, seen


def test_a_flaky_line_is_retried_in_place_without_going_to_the_cloud() -> None:
    """一个 502 就把整份文档推给**按量计费**的云端不值得 —— 先原地再来一次。"""
    client, seen = _flaky(1)

    pages = client.parse(Path("x.pdf"), file_type=0)

    assert pages[0].markdown == "本地重试成功"
    assert len(seen) == 1 and "原地重试" in seen[0], "重试也要出声，别让人以为产线一直很稳"


def test_retrying_can_be_turned_off() -> None:
    client, seen = _flaky(1, retry_primary=0)

    with pytest.raises(OcrUnavailable, match="502"):
        client.parse(Path("x.pdf"), file_type=0)

    assert seen == []


def test_the_retry_is_exhausted_before_the_cloud_sees_anything() -> None:
    """重试到次数用完才轮到云端：`FallbackOcr` 外面那层不用改。"""
    primary = _Flaky(2)
    cloud, seen = _Cloud(), []
    client = OcrClient(IngestOcrConfig(base_url="http://local:8080", retry_primary=1))
    client.__dict__["_parse_once"] = primary.parse
    fallback = FallbackOcr(client, cloud, warn=seen.append)

    pages = fallback.parse(Path("x.pdf"), file_type=0)

    assert primary.calls == 2, "一次原请求 + 一次重试"
    assert cloud.calls == 1
    assert pages[0].markdown == "云端结果"
    assert any("云端" in item for item in seen)


# ---- 切换必须出声 ----­


def test_switching_warns_with_both_reason_and_destination() -> None:
    """悄悄换到按量计费的云端，用户会看着账单猜原因。"""
    fallback, _, _, seen = _fallback(OcrUnavailable("502 Bad Gateway"))

    fallback.parse(Path("x.pdf"), file_type=0)

    assert len(seen) == 1
    assert "502" in seen[0]
    assert "paddleocr.aistudio-app.com" in seen[0]


# ---- 装配 ----­


def _cfg(**cloud: Any) -> IngestOcrConfig:
    return IngestOcrConfig(
        pdf=IngestPdfConfig(),
        base_url="http://local:8080",
        cloud_fallback=IngestCloudOcrConfig(**cloud),
    )


def test_fallback_is_off_by_default() -> None:
    """它按量计费 —— 打开是个花钱的决定，不该悄悄发生。"""
    assert IngestCloudOcrConfig().enabled is False
    assert build_ocr(_cfg(token_ignored=None)) is not None
    assert not isinstance(build_ocr(_cfg()), FallbackOcr)


def test_enabling_the_fallback_without_a_token_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """默默退回「没有兜底」，等于让人以为有保险。"""
    # 本机 .env 里是配了 token 的，这里要模拟「没配」的情形。
    # 不能只把 get_secrets 换成 Secrets —— Secrets() 自己就会去读项目根的 .env，
    # 照样把真 token 读进来（这个坑 test_chat_provider.py 也踩过，见那里的 _empty_cwd）。
    monkeypatch.setattr(
        "rag.config.get_secrets",
        lambda: types.SimpleNamespace(paddleocr_access_token=None),
    )

    with pytest.raises(ValueError, match="PADDLEOCR_ACCESS_TOKEN"):
        build_ocr(_cfg(enabled=True), token=None)


def test_enabling_the_fallback_with_a_token_wraps_the_client() -> None:
    client = build_ocr(_cfg(enabled=True), token="tok")

    assert isinstance(client, FallbackOcr)


def test_no_base_url_means_no_ocr_at_all() -> None:
    assert build_ocr(IngestOcrConfig(base_url=None, cloud_fallback=IngestCloudOcrConfig())) is None


# ---- 云端结果 → 内部 OcrPage ----­


class _Page:
    def __init__(self, text: str, images: dict[str, str]) -> None:
        self.markdown_text = text
        self.markdown_images = images


class _Result:
    def __init__(self, pages: list[_Page]) -> None:
        self.pages = pages


class _FakeSdkClient:
    """替下 paddleocr 的 SDK 客户端，记下参数。"""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.closed = False
        _FakeSdkClient.last = self

    def parse_document(self, *, model: str, file_path: str) -> _Result:
        self.model = model
        self.file_path = file_path
        return _Result(
            [
                _Page("# 云端标题", {"imgs/a.jpg": "http://files/a.jpg"}),
                _Page("第二页", {}),
            ]
        )

    def close(self) -> None:
        self.closed = True


class _FakeHttp:
    def __init__(self, **_kwargs: Any) -> None:
        pass

    def __enter__(self) -> _FakeHttp:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def get(self, url: str) -> object:
        return types.SimpleNamespace(content=b"IMG", url=url)


def _install_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """桩必须挂在**与实现相同的路径**上。

    先前挂在 `paddleocr._api_client` 上、实现也写成从那里 import —— 两边一起错、
    测试照样绿，真调云端时才 ImportError。桩挂对路径，这类错才藏不住。
    """
    module = types.ModuleType("paddleocr._api_client.client")
    module.PaddleOCRClient = _FakeSdkClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr._api_client.client", module)
    monkeypatch.setattr("rag.ingest.ocr.httpx.Client", _FakeHttp)


def test_cloud_result_maps_onto_our_page_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """markdown_text / markdown_images 与本地那套几乎同名，映射要一一对上。"""
    _install_sdk(monkeypatch)
    from rag.ingest.ocr import CloudOcrClient

    pages = CloudOcrClient(IngestCloudOcrConfig(enabled=True), "tok").parse(
        Path("x.pdf"), file_type=0
    )

    assert [p.markdown for p in pages] == ["# 云端标题", "第二页"]
    assert [p.page_index for p in pages] == [0, 1]
    # 云端给 URL，这里下载成字节 —— 下游（extract_images）只认字节
    assert pages[0].images["imgs/a.jpg"] == b"IMG"
    assert pages[0].images["a.jpg"] == b"IMG", "basename 也要能查到，少一处就少一张图"
    assert pages[1].images == {}


def test_cloud_client_passes_token_and_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_sdk(monkeypatch)
    from rag.ingest.ocr import CloudOcrClient

    CloudOcrClient(
        IngestCloudOcrConfig(
            enabled=True, model="PaddleOCR-VL-1.6", request_timeout_s=11.0, poll_timeout_s=22.0
        ),
        "tok",
    ).parse(Path("x.pdf"), file_type=0)

    sdk = _FakeSdkClient.last
    assert sdk.kwargs["token"] == "tok"
    assert sdk.kwargs["request_timeout"] == 11.0
    assert sdk.kwargs["poll_timeout"] == 22.0
    assert sdk.model == "PaddleOCR-VL-1.6"
    assert sdk.file_path == "x.pdf"
    assert sdk.closed is True, "客户端要关掉，否则连接会攒着"


def test_empty_cloud_result_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """云端返回 0 页时宁可报错，也别让它静默变成空正文。"""
    _install_sdk(monkeypatch)
    from rag.ingest.ocr import CloudOcrClient

    monkeypatch.setattr(_FakeSdkClient, "parse_document", lambda self, **_: _Result([]))

    with pytest.raises(OcrUnavailable, match="0 页"):
        CloudOcrClient(IngestCloudOcrConfig(enabled=True), "tok").parse(Path("x.pdf"), file_type=0)


def test_the_real_sdk_exposes_the_client_at_the_import_path_we_use() -> None:
    """桩能掩盖 import 路径写错 —— 真 SDK 就装在环境里，顺手把这个路径钉住。

    这条是补的：先前实现写 `from paddleocr._api_client import PaddleOCRClient`、
    桩也挂在同一个错路径上，测试全绿，真调云端才 ImportError。

    前提是**真 SDK 装在这台机器上**（`paddleocr` 不在 `pyproject.toml` 的依赖里：
    它是本地产线/云端兜底那条路才要的东西）。没装就跳过 —— 不装还要它过，只能靠假
    SDK，那正是这条用例要防的事。CI 里就是跳过的（装上它是 100MB+ 的包，不值当）。
    """
    pytest.importorskip("paddleocr", reason="真 SDK 不在这个环境里（它不属于项目依赖）")
    from paddleocr._api_client.client import PaddleOCRClient

    assert PaddleOCRClient is not None
