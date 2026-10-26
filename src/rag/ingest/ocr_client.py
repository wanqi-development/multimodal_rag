"""OCR 的**统一契约 + 注册表 + 工厂**：换 OCR 只换配置，不碰下游一行代码。

三种 OCR（PaddleOCR 产线 / dots.ocr / MinerU）的实现差别很大，但对上层只有一件事要紧：

    parse(path, *, file_type) -> list[OcrPage]      # 一页一份 markdown

只要守住这个出口，**中间产物就是同一种格式**（每页 markdown + 图片），下游的切分 /
定位 / 图片描述 / 灌库全都不用知道用的是哪家。新增一家 = 写一个 client 再
``@register_ocr("名字")``，然后把它写进配置。

约定与 ``loaders`` / ``chunkers`` / ``embedders`` 完全一致：

* 注册的是**构建函数**（不是类）：不同家的 client 需要的是各自那段配置，
  由构建函数自己去取，工厂不关心；
* 名字对不上直接报错并列出所有已注册的名字 —— 配置写错了不该静默退回默认值；
* ``build_ocr_engine`` 允许返回 ``None``（该后端没配地址 = 不启用），
  调用方拿到 None 时按「这条链路上不发任何请求」处理。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from rag.config.schema import IngestConfig


class OcrPage(BaseModel):
    """一页的识别结果 —— **三种 OCR 共用的出口格式**。

    ``page_index`` 从 0 起、由数组下标推出：产线把 prunedResult 里的 page_index
    剥掉了，响应里没有任何可信的页码字段。

    ``images`` 是「文件名 → 图片数据」，键与 markdown 里 ``<img src="imgs/…">`` 的
    src 逐字相同 —— 靠它才能把图片放回文中的确切位置。有的后端（dots.ocr / MinerU）
    直接把图内联成 markdown 的 data URI，那时这里留空、由 ``assets.extract_images``
    从正文里解出来即可。
    """

    page_index: int
    markdown: str
    images: dict[str, bytes] = Field(default_factory=dict)


@runtime_checkable
class OcrEngine(Protocol):
    """OCR 客户端的统一接口。实现它就能被 ``build_ocr_engine`` 造出来。"""

    def parse(self, path: Path, *, file_type: int) -> list[OcrPage]: ...

    @property
    def base_url(self) -> str: ...

    @property
    def name(self) -> str:
        """进 ``Document.metadata["parser"]``（也就是 Milvus 的 doc_type）的名字。"""
        ...


# 名字 → 构建函数。构建函数收 (ingest 配置, 机密)，返回客户端或 None（没配地址）。
OCR_ENGINES: dict[str, Callable[..., OcrEngine | None]] = {}


def register_ocr(name: str) -> Callable[[Callable[..., OcrEngine | None]], Callable]:
    """把一个 OCR 客户端登记到名字上。重复登记直接报错，不静默覆盖。"""

    def deco(func: Callable[..., OcrEngine | None]) -> Callable:
        existing = OCR_ENGINES.get(name)
        if existing is not None:
            raise ValueError(f"OCR 名字冲突: {name!r} 已被 {existing!r} 占用")
        OCR_ENGINES[name] = func
        return func

    return deco


def build_ocr_engine(
    ingest: IngestConfig,
    name: str,
    secrets: object | None = None,
) -> OcrEngine | None:
    """按名字建一个 OCR 客户端。名字不认识就报错，并把可选的列出来。

    **`pymupdf` 单独放行、返回 None**：它不是 OCR，是 PDF loader 自己的本地降级路径
    （直接读文字层）。以前工厂不认这个名字，于是 `ingest.pdf.parser = pymupdf` 会在
    「建客户端」那一步就炸 —— 而 `pdf.py` 的说明里一直写着它可用（OCR 全挂时的降级、
    或排查「是不是 OCR 读歪了」的对照）。

    **老名字 `ocr` 也认**：它专指 PaddleOCR 那一家，规范名现在是 `paddleocr`
    （归一化在 `config.schema.normalize_pdf_parser`）。配置里两种写法都能跑。
    """
    from rag.config.schema import normalize_pdf_parser

    name = normalize_pdf_parser(name)
    if name == "pymupdf":
        return None
    builder = OCR_ENGINES.get(name)
    if builder is None:
        raise KeyError(f"未注册的 OCR: {name!r}；已注册: {sorted(OCR_ENGINES)}")
    return builder(ingest, secrets)
