"""接入层：读文件 → 切块，产出可送去 embedding 的 Chunk 列表。

    prep = prepare(Path("./data/docs"), settings, build_ocr(settings.ingest.ocr))
    prep.chunks      # 可入库的块
    prep.skipped     # 没进来的文件及原因

loader 与 chunker 分开，是因为两者的变化频率和选择依据不同：loader 按文件格式
选（后缀决定），chunker 按语料性质选（配置决定）。

**整条链路其实是三段**（2026-09-30 拆开，见 ``ingest/stages.py``）：

    1. 解析文件          loaders + OCR      → Document（落 data/stages/<doc_id>/document.*）
    2. 切分 + 图片描述    pipeline.build_chunks → Chunk（落 .../chunks.*）
    3. 灌库              embedding + Milvus → 只有这一段需要数据库

这里保留的 ``prepare()`` / ``prepare_paths()`` 是「1+2 一口气跑完」的老入口，
给测试与「跑一遍看看」用；真要分段跑（Milvus 挂着、或等机器空闲再灌）走 CLI 的
``rag ingest --stage parse|chunk|index``。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from rag.config.schema import Settings
from rag.contracts import Chunk, Embedder

# 下面两个 import 是为了**触发注册**：工厂（ocr_client）靠注册表找 client，
# 不 import 到它们，`build_ocr_engine` 就找不到 "dots" / "mineru"
from rag.ingest import dots as _dots  # noqa: F401
from rag.ingest import mineru_api as _mineru  # noqa: F401
from rag.ingest.images import describe_images
from rag.ingest.loaders import LoadReport, SkippedFile, discover, load_paths, load_source
from rag.ingest.ocr import build_ocr
from rag.ingest.ocr_client import OcrEngine, OcrPage, build_ocr_engine
from rag.ingest.pipeline import build_chunks, chunk_documents, drop_shell_chunks, locate

__all__ = [
    "IngestPrep",
    "LoadReport",
    "OcrEngine",
    "OcrPage",
    "SkippedFile",
    "build_chunks",
    "build_ocr",
    "build_ocr_engine",
    "chunk_documents",
    "describe_images",
    "drop_shell_chunks",
    "load_source",
    "locate",
    "prepare",
    "prepare_paths",
]

if TYPE_CHECKING:
    from rag.providers.vision import VisionClient


class IngestPrep(BaseModel):
    """加载 + 切块的产出。

    skipped 原样带上来而不是在这里丢掉：CLI 需要打印它，
    否则「哪个文件没进来、为什么」只有翻日志才知道。
    """

    chunks: list[Chunk] = Field(default_factory=list)
    documents: int = 0
    skipped: list[SkippedFile] = Field(default_factory=list)
    # 生成了几条图片描述块。CLI 要靠它说清「这次花了几次 VLM 调用」——
    # 那是按量计费的一步，不该只体现在账单上。
    images_described: int = 0
    # 丢掉了几个空壳块（剥掉标题与图片标记后没有任何可检索文字的块）。
    # 块不该静默消失，所以这个数要带出来让 CLI 打印。
    shells_dropped: int = 0


def prepare(
    source: Path,
    settings: Settings,
    ocr: OcrEngine | None,
    embedder: Embedder | None = None,
    vision: VisionClient | None = None,
) -> IngestPrep:
    """读一个 source（文件或目录）→ 切块。"""
    source = Path(source).resolve()
    root = source if source.is_dir() else source.parent
    return prepare_paths(discover(source), root, settings, ocr, embedder, vision)


def prepare_paths(
    paths: list[Path],
    root: Path,
    settings: Settings,
    ocr: OcrEngine | None,
    embedder: Embedder | None = None,
    vision: VisionClient | None = None,
) -> IngestPrep:
    """读**指定的这一批文件** → 切块。

    与 `prepare` 的区别只在「读哪些文件」：增量灌库只处理变更过的那几个，但**必须沿用
    语料根** —— doc_id 与 source_uri 都是相对根算的，换成各文件的父目录会让子目录里的
    文件算出另一个 doc_id，注册表就对不上了。

    ocr 是必填的（可以是 None）：它由 build_ocr 按 ingest.ocr.base_url 决定，没配地址
    就是 None、这条链路上不会有任何请求。做成必填而不是默认 None，是因为「配了但忘了
    传」会退化成没配 —— 那时扫描件的报错会让人去 .env 里找一个其实已经填好的值。
    批量调用时把同一个客户端传进来，连接池就复用了。

    embedder 只有 ``strategy: markdown-semantic`` 用得上（语义兜底要算句向量）。
    由调用方建好传进来，为的是和灌库那一步**共用同一个实例** —— 否则 1.2GB 的
    embedding 模型会被加载两次。策略用不上时忽略即可。

    vision 只有 ``ingest.images.describe`` 打开时才会被用到（那时它不能是 None）。
    由调用方按开关决定建不建 —— 开关关着就一次网络请求都没有（VLM 按量计费）。
    """
    report = load_paths(list(paths), settings.ingest, root=root, ocr=ocr)
    built = build_chunks(report.documents, settings, embedder, vision)
    return IngestPrep(
        chunks=built.chunks,
        documents=len(report.documents),
        skipped=[*report.skipped, *built.failed],
        images_described=built.described,
        shells_dropped=built.shells,
    )
