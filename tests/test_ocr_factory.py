"""OCR 工厂：注册表、按配置选 client、换一家不用动下游。

这是「三种 OCR 做成配置」的核心约定 —— 三家 client 的出口都是 `list[OcrPage]`，
上层只认工厂给的那个对象。
"""

from __future__ import annotations

import pytest

from rag.config.schema import IngestConfig
from rag.ingest.ocr_client import OCR_ENGINES, OcrEngine, build_ocr_engine


def test_the_three_engines_are_registered() -> None:
    """规范名是 `paddleocr`（PaddleOCR 产线）；`ocr` 是老名字，归一化后才查注册表。"""
    assert {"paddleocr", "dots", "mineru"} <= set(OCR_ENGINES)
    assert "ocr" not in OCR_ENGINES, "老名字不该另外注册一份（两份实现迟早分家）"


def _configured() -> IngestConfig:
    """三家的地址都配上 —— 没配地址的 PaddleOCR 按设计返回 None。"""
    cfg = IngestConfig()
    cfg.ocr.base_url = "http://paddle.internal:8080"
    return cfg


@pytest.mark.parametrize("name", ["paddleocr", "dots", "mineru"])
def test_every_registered_engine_returns_a_conforming_client(name: str) -> None:
    """都实现同一个 Protocol（parse / base_url / name），上层才敢只认接口。"""
    engine = build_ocr_engine(_configured(), name)

    assert isinstance(engine, OcrEngine)
    assert engine.name  # 进 Milvus 的 doc_type
    assert engine.base_url


def test_an_engine_without_an_address_is_none() -> None:
    """PaddleOCR 产线没配地址 = 不启用：整条链路上一个请求都不发（不是错误）。"""
    assert build_ocr_engine(IngestConfig(), "paddleocr") is None


@pytest.mark.parametrize("name", ["paddleocr", "ocr"])
def test_the_old_name_still_builds_the_same_client(name: str) -> None:
    """`ocr` 是 PaddleOCR 的历史别名：老 `.env`（`RAG_PDF_PARSER=ocr`）不该因为改名就跑不起来。"""
    engine = build_ocr_engine(_configured(), name)

    assert engine is not None
    assert engine.name == "paddleocr", "报出去的名字要统一，否则 parse_key / 界面两种说法"


def test_the_client_follows_the_config() -> None:
    """换地址只改配置：工厂把配置里的地址交给对应的 client。"""
    cfg = _configured()
    cfg.dots.base_url = "http://dots.internal:9999"
    cfg.mineru.base_url = "http://mineru.internal:9999"

    assert build_ocr_engine(cfg, "dots").base_url == "http://dots.internal:9999"
    assert build_ocr_engine(cfg, "mineru").base_url == "http://mineru.internal:9999"
    assert build_ocr_engine(cfg, "dots").name == "dots"
    assert build_ocr_engine(cfg, "mineru").name == "mineru"
    assert build_ocr_engine(cfg, "paddleocr").name == "paddleocr"


def test_unknown_engine_is_rejected_with_the_options() -> None:
    """配置写错了当场报错并列出可选项，不静默退回默认值。"""
    with pytest.raises(KeyError) as excinfo:
        build_ocr_engine(IngestConfig(), "who")

    message = str(excinfo.value)
    assert "who" in message
    for name in ("paddleocr", "dots", "mineru"):
        assert name in message


def test_a_new_engine_can_be_added_by_registering() -> None:
    """加一家 = 注册一个 client，工厂与上层都不用改。"""
    from rag.ingest.ocr_client import register_ocr

    class _Fake:
        name = "fake"
        base_url = "http://fake"

        def parse(self, path, *, file_type):
            return []

    @register_ocr("fake-engine-for-test")
    def _build(ingest: IngestConfig, secrets: object | None = None) -> _Fake:
        return _Fake()

    try:
        engine = build_ocr_engine(IngestConfig(), "fake-engine-for-test")
        assert engine is not None and engine.name == "fake"
    finally:
        OCR_ENGINES.pop("fake-engine-for-test", None)


def test_pymupdf_is_allowed_through_as_no_engine() -> None:
    """`pymupdf` 不是 OCR，工厂要放行它、返回 None。

    它只在 `pdf.py` 里当本地降级路径用（直接读文字层，OCR 全挂时的替代、或排查「是不是
    OCR 读歪了」的对照），但建客户端那一步以前不认这个名字 —— 于是
    `ingest.pdf.parser = pymupdf` 会当场抛 `未注册的 OCR`，配置里写着的选项根本用不了。
    """
    assert build_ocr_engine(IngestConfig(), "pymupdf") is None
