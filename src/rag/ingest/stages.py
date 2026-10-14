"""三段流水线的中间产物：落在本地、人机两读、带新鲜度校验。

为什么要把一条 ``rag ingest`` 拆成三段（用户 2026-09-30 提的两个理由，都成立）：

1. **人工可核对**：OCR 读歪了、标题层级错了、块切得太碎 —— 只有把中间结果落成文件
   才看得见；连上库再反查既慢，还得自己把块拼回原文。
2. **解耦时间与依赖**：解析与切分贵在 OCR / VLM，而**只有第三段依赖 Milvus**。
   库挂了、或想等机器空闲再灌，都不必把前面两步重做一遍。

落在 ``ingest.stages.dir``（默认 ``./data/stages``），按 doc_id 分目录：

    data/stages/<doc_id>/
        document.json   阶段1 的机器可读结果（Document：正文 / 位置 / 图片 / 元数据）
        document.md     同一份内容的人读版：解析出的 markdown + 一行头信息
        chunks.json     阶段2 的机器可读结果（Chunk 列表）
        chunks.md       人读版：一块一段，带序号 / 模态 / 页码行号 / 章节路径

``.json`` 给下一段读，``.md`` 给**人看**。两件事的诉求不同，合成一个文件要么机器读着
别扭、要么人看着费劲，文本量又不大，所以两份都留。

**新鲜度是这套东西的关键。** 文件一改、解析设置一改、切法一改，旧产物就不能再用。
所以每份产物都记 ``file_hash``，外加 ``parse_key``（阶段1）或 ``pipeline_version``
（阶段2），读取时逐个比对；对不上就当没缓存 —— 宁可让调用方重跑一遍，
也绝不拿一份对不上的中间结果去写库（写进去是脏数据，比慢一轮难查得多）。

这是**缓存**不是数据：整目录删掉只会让下次 ingest 从头跑，不丢任何已入库的东西。
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from rag.config.schema import Settings
from rag.contracts import Chunk, Document, Embedder
from rag.ingest.incremental import file_hash, pipeline_version
from rag.ingest.loaders import SkippedFile, load_paths
from rag.ingest.loaders.base import doc_id_for, loader_version, rel_uri
from rag.ingest.ocr import OcrClient
from rag.ingest.outline import Outline, normalize_levels, read_outline, render_outline
from rag.ingest.pipeline import build_chunks
from rag.storage import BlobNotFound, BlobStore

if TYPE_CHECKING:
    # 只在类型标注里用：providers 是个重包（一 import 就带出 fastembed / langchain），
    # 而这段代码在「只解析、不灌库」的场景下根本用不到 VLM
    from rag.providers.vision import VisionClient

# 产物格式的版本。**字段改了必须抬这个数**：旧产物会被判成「不认识」而重跑，
# 而不是被硬塞进新代码里，以奇怪的方式炸在下游。
SCHEMA_VERSION = 1

_FILES = (
    "document.json",
    "document.md",
    "outline.tsv",
    "chunks.json",
    "chunks.md",
    # 图谱抽取产物（M6）：语料里删了这份文件，它跟着一起清
    "graph.json",
)


def parse_key(settings: Settings, suffix: str = "") -> str:
    """阶段1 的指纹：**只包含影响解析结果的设置**。

    与 ``pipeline_version`` 分开是有意的：换了 embedding 模型不该让解析产物失效 ——
    那要重新 OCR，是这条链路上最慢、也最可能花钱的一步。

    ``suffix`` 决定用哪个 loader 的版本号：**按格式分开**，改 html 的清洗不该让
    PDF 产物失效（PDF 重解析要过 OCR / VLM，按量计费）。
    """
    ocr = settings.ingest.ocr
    parts = [
        f"parser:{settings.ingest.pdf.parser}",
        f"ocr:{ocr.restructure_pages}/{ocr.merge_tables}/{ocr.relevel_titles}/{ocr.return_images}",
    ]
    # loader 的行为改了（清洗、标题识别、表格转换……），那种格式的解析产物就该重做。
    # **只在版本不是基线 "1" 时才写进指纹**：凭空多一个组件会让所有老产物一次性失效，
    # 而 PDF 重解析要过 OCR / VLM（Kafka 那份 396 页花了十几分钟 GPU）。
    if (version := loader_version(suffix)) != "1":
        parts.append(f"loader:{version}")
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


class ArtifactStore:
    """``<doc_id>/<产物名>`` 的读写。**落本地盘还是 MinIO 由 `storage.BlobStore` 决定**
    （2026-10-09 起；单机默认本地盘，行为与以前逐字节一致）。

    键固定是 ``<doc_id>/<名字>``：local 下就是 `data/stages/<doc_id>/<名字>`，
    minio 下是 `<stages 前缀>/<doc_id>/<名字>` —— 两边同一个字符串，所以 doc_id 与
    产物内容都不必迁移。
    """

    def __init__(self, blobs: BlobStore) -> None:
        self._blobs = blobs

    @property
    def root(self) -> str:
        """给 CLI 打给人看的一行（`local:...` / `minio:...`）。"""
        return self._blobs.name

    @property
    def backend(self) -> str:
        return self._blobs.backend

    @staticmethod
    def _key(doc_id: str, name: str) -> str:
        return f"{doc_id}/{name}"

    def path_of(self, doc_id: str, name: str) -> Path | None:
        """**只认 Path 的消费方**（读大小、`read_text`、清单指纹）走这里。

        local：真路径；minio：读穿透到本地缓存后给缓存路径（桶是真相，缓存可丢）。
        **这份产物不存在 → None**（两个后端一个说法，见 `storage.local_path`）。

        **写入不要走这里**：`_write()` 收的是**键**（它自己 `put_text` 到桶/盘）。
        拿路径去当写入口，在 minio 下会先去 HEAD 一个还不存在的键（2026-10-10 修）。
        """
        return self._blobs.local_path(self._key(doc_id, name))

    # ---- 阶段1：解析产物 ----

    def save_document(self, doc: Document, *, source_hash: str, parse: str) -> None:
        self._write(
            self._key(doc.doc_id, "document.json"),
            {
                "kind": "document",
                "schema": SCHEMA_VERSION,
                "saved_at": int(time.time()),
                "file_hash": source_hash,
                "parse_key": parse,
                "source_uri": doc.source_uri,
                "document": doc.model_dump(mode="json"),
            },
        )
        self._write(self._key(doc.doc_id, "document.md"), render_document(doc, parse))
        # 标题层级清单：**人工对齐用**。PDF 走 OCR 之后层级只能靠猜（版面量过、分不开；
        # 编号形式又千变万化），所以这里出一份清单让人核对，切分以改完的清单为准。
        # 不动 document.*：清单只描述「哪一行是几级」，正文一个字都不改。
        self._write(self._key(doc.doc_id, "outline.tsv"), render_outline(doc.text))

    def read_payload(self, doc_id: str, name: str) -> dict | None:
        """读产物的**原始 payload**（`document.json` / `chunks.json`）。

        与 `load_document` / `load_chunks` 的区别：那两个要判「新鲜度」，对不上就返回 None；
        这里不管新鲜度、原样给出来 —— 页面得能说明**为什么**产物作废了（源文件变了？
        换了解析器？改了切法？），那就必须先看得到过期的那一份。
        """
        return self._read(self._key(doc_id, name))

    def load_document(self, doc_id: str, *, source_hash: str, parse: str) -> Document | None:
        """产物缺失 / 格式不认识 / 文件或解析设置变了 —— 一律返回 None（当作没缓存）。"""
        payload = self._read(self._key(doc_id, "document.json"))
        if not _matches(payload, kind="document", source_hash=source_hash):
            return None
        if payload.get("parse_key") != parse:
            return None
        return Document.model_validate(payload["document"])

    # ---- 标题层级清单（人工对齐） ----

    def load_outline(self, doc_id: str, *, source_hash: str, parse: str) -> Outline | None:
        """读人工核对过的清单。没生成过、文件或解析设置变了 → None（当作「还没核对」）。

        内容坏掉（层级列写成别的、最大层级写了个不可能的数）会抛 ``ValueError`` ——
        那是人填的东西，要说清楚是第几行，不能当成「没核对」混过去。
        """
        document = self._read(self._key(doc_id, "document.json"))
        if not _matches(document, kind="document", source_hash=source_hash):
            return None
        if document.get("parse_key") != parse:
            return None
        path = self.path_of(doc_id, "outline.tsv")
        if path is None or not path.is_file():
            return None
        return read_outline(path.read_text(encoding="utf-8"))

    def ensure_outline(self, doc_id: str, *, source_hash: str, parse: str) -> bool:
        """补齐缺失的清单（已解析过、产物还新鲜的文档）。返回是否补了。

            产物是分阶段加进来的：**加这一版之前解析的文档没有 outline.tsv**，
        为它重新 OCR 一遍纯属浪费 —— 从缓存里的 document 生成即可。
        """
        outline_path = self.path_of(doc_id, "outline.tsv")
        if outline_path is not None and outline_path.is_file():
            return False
        doc = self.load_document(doc_id, source_hash=source_hash, parse=parse)
        if doc is None:
            return False
        self._write(self._key(doc_id, "outline.tsv"), render_outline(doc.text))
        return True

    def save_outline(self, doc_id: str, text: str, *, source_hash: str, parse: str) -> None:
        """把**人工核定后**的清单写回去（页面第三步的提交）。

        写之前先确认 `document.json` 还对得上：清单是解析产物的从属，解析失效（换了文件、
        换了解析器）时写回去只会留下一条挂在旧解析上的清单，后面 `run_chunk` 会按它切。

        **只动 `outline.tsv` 一个文件** —— `document.json` / `document.md` 一个字都不改。
        「正文不允许修改」这条就落在这里：清单只描述「第几行是几级」，正文是解析出来的原文。
        """
        document = self._read(self._key(doc_id, "document.json"))
        if not _matches(document, kind="document", source_hash=source_hash):
            raise ValueError(f"这份文档还没有解析产物（或解析结果已过期）：{doc_id}")
        if document.get("parse_key") != parse:
            raise ValueError(f"这份文档的解析器已经变了，要重新解析才能改清单：{doc_id}")
        self._write(self._key(doc_id, "outline.tsv"), text)

    def outline_fingerprint(self, doc_id: str) -> str:
        """清单**语义**的指纹（没有清单就是空串），进「这份文档变了没有」的比较。

        人工核对过的清单必须进比较：改一份清单只该重灌那一份。但**只取决定切法的两样**
        —— 层级列表与最大层级；刻意不哈希文件原文：清单里有六七行是给人看的说明，
        改一句措辞也会让整份文档重切、重新 embedding，而切法一个字都没变。
        新增/删掉行会直接反映在层级列表上（条目数变了），所以照样会重灌。
        """
        path = self.path_of(doc_id, "outline.tsv")
        if path is None or not path.is_file():
            return ""
        try:
            outline = read_outline(path.read_text(encoding="utf-8"))
        except ValueError:
            # 清单坏了，算不出语义指纹 → 退回按原文算（反正它接下来会被 skipped 拦下）
            return hashlib.sha1(path.read_bytes()).hexdigest()[:16]
        payload = f"{outline.levels}|{outline.max_level}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    # ---- 阶段2：切分产物 ----

    def save_chunks(
        self,
        doc_id: str,
        chunks: list[Chunk],
        *,
        source_hash: str,
        source_uri: str,
        pipeline: str,
        outline_hash: str = "",
    ) -> None:
        self._write(
            self._key(doc_id, "chunks.json"),
            {
                "kind": "chunks",
                "schema": SCHEMA_VERSION,
                "saved_at": int(time.time()),
                "file_hash": source_hash,
                "pipeline_version": pipeline,
                # 人工对齐过的标题清单也算「切法的输入」：清单一改，块就该重切
                "outline_hash": outline_hash,
                "source_uri": source_uri,
                "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
            },
        )
        self._write(self._key(doc_id, "chunks.md"), render_chunks(chunks, source_uri))

    def load_chunks(
        self, doc_id: str, *, source_hash: str, pipeline: str, outline_hash: str = ""
    ) -> list[Chunk] | None:
        payload = self._read(self._key(doc_id, "chunks.json"))
        if not _matches(payload, kind="chunks", source_hash=source_hash):
            return None
        if (
            payload.get("pipeline_version") != pipeline
            or payload.get("outline_hash") != outline_hash
        ):
            return None
        return [Chunk.model_validate(item) for item in payload["chunks"]]

    # ---- 清理 ----

    # ---- 图谱抽取产物（M6）----

    def save_graph(
        self,
        doc_id: str,
        *,
        prompt: str,
        model: str,
        chunks: Sequence[Any],
        source_uri: str = "",
    ) -> Path | None:
        """写 `<doc_id>/graph.json`。

        **不带 file_hash**：图谱的输入是**切分产物**而不是原始文件（块边界变了才该重抽），
        所以缓存判据放在块级 —— 每块记 `text_hash`，另外用 `prompt` / `model` 两个版本号
        决定整篇要不要重抽（见 ingest/graph.py 的说明）。
        """
        self._write(
            self._key(doc_id, "graph.json"),
            {
                "kind": "graph",
                "schema": SCHEMA_VERSION,
                "saved_at": int(time.time()),
                "doc_id": doc_id,
                "source_uri": source_uri,
                "prompt": prompt,
                "model": model,
                "chunks": {
                    item.chunk_id: {
                        "text_hash": item.text_hash,
                        "entities": item.entities,
                        "relations": item.relations,
                        "error": item.error,
                    }
                    for item in chunks
                },
            },
        )
        # 写完再取路径：这时键一定在（minio 下也是），拿不到就是后端出了问题
        return self.path_of(doc_id, "graph.json")

    def load_graph(self, doc_id: str, *, prompt: str, model: str) -> dict | None:
        """读图谱产物。**提示词/模型对不上就当没有** —— 那是要重抽的信号。"""
        payload = self._read(self._key(doc_id, "graph.json"))
        if not payload or payload.get("kind") != "graph":
            return None
        if payload.get("prompt") != prompt or payload.get("model") != model:
            return None
        return payload

    def load_any_graph(self, doc_id: str) -> dict | None:
        """读图谱产物，**不校验版本** —— 灌库用（产物是上次抽的，模型名不必与当前一致）。"""
        payload = self._read(self._key(doc_id, "graph.json"))
        return payload if payload and payload.get("kind") == "graph" else None

    def drop(self, doc_id: str) -> int:
        """删掉一份文档的全部产物。语料里删了文件就跟着清，免得越积越多。

        按**前缀**一次收干净（`<doc_id>/`）—— local 是这样、minio 也是这样（对象存储
        本来就没有"空目录"这回事，所以这条在两个后端下语义一致）。下次要写产物时
        `_write` 会把位置建回来。
        """
        return self._blobs.delete_prefix(f"{doc_id}/")

    # ---- 读写 ----

    def _write(self, key: str, text: str | dict) -> None:
        body = (
            json.dumps(text, ensure_ascii=False, indent=2) + "\n"
            if isinstance(text, dict)
            else text
        )
        self._blobs.put_text(key, body)

    def _read(self, key: str) -> dict | None:
        try:
            payload = json.loads(self._blobs.get_text(key))
        except BlobNotFound:
            return None
        except (ValueError, OSError, UnicodeDecodeError):
            # 产物坏了当作没有：它只是缓存，重跑一次即可，不该让整批 ingest 失败
            return None
        if not isinstance(payload, dict) or payload.get("schema") != SCHEMA_VERSION:
            return None
        return payload


def build_artifact_store(settings: Settings) -> ArtifactStore:
    """按 `ingest.stages.backend` 建：`auto`（默认，听统一开关 `storage.backend`）→ 本地盘或桶。"""
    from rag.storage import build_blob_store

    cfg = settings.ingest.stages
    return ArtifactStore(
        build_blob_store(backend=cfg.backend, settings=settings, dir=cfg.dir, prefix=cfg.prefix)
    )


def _matches(payload: dict | None, *, kind: str, source_hash: str) -> bool:
    return bool(payload) and payload.get("kind") == kind and payload.get("file_hash") == source_hash


class ParseResult(BaseModel):
    """阶段1 的产出。"""

    documents: list[Document] = Field(default_factory=list)
    skipped: list[SkippedFile] = Field(default_factory=list)
    # 这次真的解析并落盘的文档数（复用已有产物、以及解析失败的都不计）
    parsed: int = 0
    reused: int = 0
    # 这次决定要处理哪些文件（source_uri）。--dry-run 时它就是「将会处理的东西」
    todo: list[str] = Field(default_factory=list)


class ChunkResult(BaseModel):
    """阶段2 的产出。"""

    chunks: list[Chunk] = Field(default_factory=list)
    skipped: list[SkippedFile] = Field(default_factory=list)
    described: int = 0
    # 丢掉的空壳块（剥掉标题与图片标记后没有任何可检索文字的块）
    shells: int = 0
    # 这次真的切分并落盘的文档数
    chunked: int = 0
    reused: int = 0
    todo: list[str] = Field(default_factory=list)
    # **待人工核对**的文档（清单没生成、或 `已核对:` 还不是「是」）。
    # 单列一个字段而不是塞进 skipped：它不是「出错」，是流程上还没走到这一步 ——
    # CLI 要把它和「失败」分开打，并且给出「改哪个文件」的指引。reason 里写的就是这句指引。
    pending: list[SkippedFile] = Field(default_factory=list)


def run_parse(
    paths: list[Path],
    root: Path,
    settings: Settings,
    ocr: OcrClient | None,
    store: ArtifactStore,
    *,
    full: bool = False,
    dry_run: bool = False,
) -> ParseResult:
    """阶段1：解析文件 → 落盘。**不碰 Milvus**，所以库挂着也能先跑。"""
    result = ParseResult()
    todo: list[Path] = []
    for path in paths:
        doc_id = doc_id_for(path, root)
        source_hash = file_hash(path)
        # 指纹按**格式**算：改了 html 的 loader 不该让 PDF 的产物失效
        parse = parse_key(settings, path.suffix)
        if not full and store.load_document(doc_id, source_hash=source_hash, parse=parse):
            # 产物新鲜：不重新解析，但**补一下可能缺的标题清单**（老产物没有它）
            store.ensure_outline(doc_id, source_hash=source_hash, parse=parse)
            continue
        todo.append(path)
    result.reused = len(paths) - len(todo)
    result.todo = [rel_uri(path, root) for path in todo]
    if dry_run:
        return result
    if not todo:
        return result

    from rag.ingest.assets import build_asset_store

    # 图片本体按 `ingest.assets.backend` 落（多 worker 时是 MinIO 桶）：html/docx 抽出来的图
    # 必须跟着走同一个后端，否则 worker A 写本地、worker B 读不到。
    report = load_paths(
        todo, settings.ingest, root=root, ocr=ocr, assets=build_asset_store(settings)
    )
    result.skipped = report.skipped
    by_id = {doc_id_for(path, root): path for path in todo}
    for doc in report.documents:
        result.documents.append(doc)
        path = by_id.get(doc.doc_id)
        if path is None:  # 理论上不会有；真出现也不静默丢产物
            result.skipped.append(
                SkippedFile(path=doc.source_uri, reason="解析产物未落盘：找不到对应的源文件")
            )
            continue
        store.save_document(
            doc, source_hash=file_hash(path), parse=parse_key(settings, path.suffix)
        )
        result.parsed += 1
    return result


def run_chunk(
    paths: list[Path],
    root: Path,
    settings: Settings,
    embedder: Embedder | None,
    vision: VisionClient | None,
    ocr: OcrClient | None,
    store: ArtifactStore,
    *,
    full: bool = False,
    dry_run: bool = False,
) -> ChunkResult:
    """阶段2：切分 + 定位 + 图片描述 → 落盘。**不碰 Milvus**。

    依赖阶段1 的产物。产物缺失时**现场补解析**（切分本来就要它，为这个再让用户敲一条
    命令纯属折腾，而且 ``--stage chunk`` 会顺手把解析产物也落上）；
    但它**不会**替阶段3 做任何事 —— 写库只有 ``--stage index`` 才发生。

    **人工核对是一道硬门槛**：任何文档的解析产物都要先由人核对标题清单
    （改层级、选最大层级、把 `# 已核对: 否` 改成 `是`），核对之前这份文档不切 ——
    它进 ``pending``，CLI 会连同「改哪个文件」一起打出来。产物已经切好、且没变的文档
    不受影响（不需要为了跑一次 ingest 重新核对一遍已经切过的语料）。
    """
    pipeline = pipeline_version(settings)
    _require_image_descriptions(settings, embedder)
    result = ChunkResult()
    reused: list[Chunk] = []
    todo: list[Path] = []
    outlines: dict[str, Outline] = {}

    for path in paths:
        doc_id = doc_id_for(path, root)
        # 人工核对过的标题清单：它是这份文档「切哪几节、切到第几级」的依据，
        # 所以内容哈希也进新鲜度比对（改过清单 → 这份文档要重切）
        try:
            outline = store.load_outline(
                doc_id, source_hash=file_hash(path), parse=parse_key(settings, path.suffix)
            )
        except ValueError as exc:
            result.skipped.append(
                SkippedFile(path=rel_uri(path, root), reason=f"标题清单有问题：{exc}")
            )
            continue
        if outline is not None:
            outlines[doc_id] = outline
        outline_hash = store.outline_fingerprint(doc_id)
        cached = (
            None
            if full
            else store.load_chunks(
                doc_id,
                source_hash=file_hash(path),
                pipeline=pipeline,
                outline_hash=outline_hash,
            )
        )
        if cached is None:
            if outline is None or not outline.confirmed:
                # 还没人工核对：不切，也不当成错 —— 但要说清楚去改哪个文件
                result.pending.append(
                    SkippedFile(path=rel_uri(path, root), reason=_review_hint(outline))
                )
            else:
                todo.append(path)
        else:
            result.reused += 1
            reused.extend(cached)
    result.todo = [rel_uri(path, root) for path in todo]
    if dry_run:
        return result

    if todo:
        documents, by_id = _documents_for(todo, root, settings, ocr, store, result)
        documents, rejected = _apply_outlines(documents, outlines)
        result.skipped.extend(rejected)
        built = build_chunks(documents, settings, embedder, vision)
        chunks = built.chunks
        result.skipped.extend(built.failed)
        result.described += built.described
        result.shells += built.shells
        grouped: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            grouped.setdefault(chunk.doc_id, []).append(chunk)
        for doc in documents:
            store.save_chunks(
                doc.doc_id,
                grouped.get(doc.doc_id, []),
                source_hash=file_hash(by_id[doc.doc_id]),
                source_uri=doc.source_uri,
                pipeline=pipeline,
                outline_hash=store.outline_fingerprint(doc.doc_id),
            )
            result.chunked += 1
        reused = [*reused, *chunks]  # 本次新切出来的也要带回去（阶段3 只看这个字段）

    result.chunks = reused
    return result


def _review_hint(outline: Outline | None) -> str:
    """「待人工核对」那一行要说清楚：去改哪个文件、改哪三处。"""
    if outline is None:
        return (
            "解析产物或标题清单还没有：先跑 --stage parse 生成清单，"
            "人工核对后把清单头部的「已核对:」改成「是」"
        )
    hint = (
        "标题清单还没人工核对：改 data/stages/<doc_id>/outline.tsv —— "
        "核对第一列的层级、在上面选好「最大层级」，再把「已核对: 否」改成「是」"
    )
    if outline.notes:
        hint += "（" + "；".join(outline.notes) + "）"
    return hint


def _require_image_descriptions(settings: Settings, embedder: Embedder | None) -> None:
    """吃图的 embedding 必须配图片描述 —— 少了它，等于整个库里的图都没进去。

    融合向量是「**描述文字 + 图片本体**」算出来的，而描述是 VLM 拿「上下文 + 图」写的
    （``images.py``）。描述关着时压根没有图片块：图只剩下正文里的 ``![image](asset:…)``
    标记，而标记在编码前会被剥掉 —— 结果是**一张图都没进向量空间**，配置却看起来一切正常。
    所以这里当场挡住，而不是让它悄悄退化成一个「只有文字」的库。
    """
    if embedder is None or not embedder.supports_images:
        return
    if settings.ingest.images.describe:
        return
    raise ValueError(
        "embedding backend 吃图，但 ingest.images.describe 关着："
        "图片要先由 VLM 拿「上下文 + 图」做描述，再用「描述 + 图」算融合向量。"
        "打开 ingest.images.describe（CLI 加 --describe），"
        "或把 providers.embedding.backend 换回纯文本那条。"
    )


def _apply_outlines(
    documents: list[Document], outlines: dict[str, Outline]
) -> tuple[list[Document], list[SkippedFile]]:
    """把人工核对过的清单挂到 Document 上（``_heading_levels`` / ``_max_level``）。

    走到这里的文档都是**核对过**的（没核对的在 run_chunk 里就进 pending 了）。
    清单与正文对不上（人增删了行）时**只跳过这一份**并说明原因：其余文件照常切。
    清单是人工的东西，出了错要指明是哪个文件、怎么修，而不是猜。
    """
    ok: list[Document] = []
    rejected: list[SkippedFile] = []
    for doc in documents:
        outline = outlines.get(doc.doc_id)
        if outline is None:
            ok.append(doc)
            continue
        try:
            levels, notes = normalize_levels(outline.levels)
            doc.metadata["_heading_levels"] = levels
            doc.metadata["_max_level"] = outline.max_level
        except ValueError as exc:
            rejected.append(SkippedFile(path=doc.source_uri, reason=f"标题清单有问题：{exc}"))
            continue
        if notes:
            rejected.append(
                SkippedFile(
                    path=doc.source_uri,
                    reason="标题清单已用，但有跳级：" + "；".join(notes[:3]),
                )
            )
        ok.append(doc)
    return ok, rejected


def _documents_for(
    todo: list[Path],
    root: Path,
    settings: Settings,
    ocr: OcrClient | None,
    store: ArtifactStore,
    result: ChunkResult,
) -> tuple[list[Document], dict[str, Path]]:
    """拿齐这批文件的 Document：优先用阶段1 的产物，缺了就现场解析并补上产物。

    **不看 --full**：解析产物是这一段的**输入**，`--stage chunk --full` 指的是
    「重新切一遍」，不是「连解析一起重做」—— 重新解析要过 OCR，最慢、也最可能花钱。
    解析产物自己会按 `parse_key` 判断新鲜度，真过期了这里自然会重做。
    """
    by_id = {doc_id_for(path, root): path for path in todo}
    documents: list[Document] = []
    for doc_id, path in by_id.items():
        parse = parse_key(settings, path.suffix)
        cached = store.load_document(doc_id, source_hash=file_hash(path), parse=parse)
        if cached is not None:
            documents.append(cached)
            continue
        from rag.ingest.assets import build_asset_store

        report = load_paths(
            [path], settings.ingest, root=root, ocr=ocr, assets=build_asset_store(settings)
        )
        result.skipped.extend(report.skipped)
        for doc in report.documents:
            documents.append(doc)
            store.save_document(doc, source_hash=file_hash(path), parse=parse)
    return documents, by_id


def render_document(doc: Document, parse: str) -> str:
    """人读版：头信息 + 解析出的正文原文。

    正文**原样**输出，不做任何排版加工 —— 要核对的就是「解析得对不对」，
    再套一层装饰反而看不清。
    """
    header = [
        "<!--",
        f"doc_id       {doc.doc_id}",
        f"source_uri   {doc.source_uri}",
        f"mime         {doc.mime or '-'}",
        f"parse_key    {parse}",
        f"metadata     {json.dumps(doc.metadata, ensure_ascii=False)}",
        f"图片 {len(doc.images)} 张，位置打点 {len(doc.positions)} 处（正文 {len(doc.text)} 字）",
    ]
    for image in doc.images:
        header.append(f"  {image.asset_uri}  page={image.page}  {image.source_name}")
    header.append("-->")
    return "\n".join(header) + "\n\n" + doc.text + "\n"


def render_chunks(chunks: list[Chunk], source_uri: str) -> str:
    """人读版：一块一段。要核对的是**块边界**，所以每块前面把定位信息摊开。"""
    kinds: dict[str, int] = {}
    for chunk in chunks:
        kinds[chunk.modality] = kinds.get(chunk.modality, 0) + 1
    summary = " / ".join(f"{name} {count}" for name, count in sorted(kinds.items())) or "（空）"
    lines = [
        "<!--",
        f"source_uri   {source_uri}",
        f"chunks       {len(chunks)}（{summary}）",
        "-->",
        "",
    ]
    for chunk in chunks:
        place = " · ".join(
            part
            for part in (
                _coords(chunk.page, chunk.line),
                f"章节={chunk.heading_path}" if chunk.heading_path else "",
                f"编号={chunk.heading_numbers}" if chunk.heading_numbers else "",
                f"图={','.join(chunk.assets)}" if chunk.assets else "",
            )
            if part
        )
        lines.append(f"## [{chunk.index:04d}] {chunk.modality} · {place} · {len(chunk.text)} 字")
        lines.append("")
        lines.append(chunk.text)
        lines.append("")
    return "\n".join(lines)


def _coords(page: int, line: int) -> str:
    """页码与行号合成一项：``p3 L12`` / 只有页 ``p3`` / 只有行 ``L12`` / 都没有则空。

    分开写成两项会出现 ``p1 · L3`` 这种读着断开的组合 —— 它们本来就是同一个位置的坐标。
    docx 这类没有页概念的，只剩 ``L7``（段落序号）。
    """
    return " ".join(x for x in (f"p{page}" if page else "", f"L{line}" if line else "") if x)
