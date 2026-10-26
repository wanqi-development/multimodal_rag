"""增量更新：判断哪些文件要重灌，哪些可以跳过。

**为什么要它**：解析（OCR）与 embedding 占了几乎全部代价，而它们都在「知道文件变没变」
之后才该发生。指向整个语料目录跑一次 `rag ingest`，如果每份文件都重解析一遍，
改一个字也要付全量的钱。这一层做的就是**在解析之前**把没变的挑出去。

设计见 docs/milvus-schema-design.md §4。
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from rag.config.schema import VISION_PROMPT_VERSION, Settings
from rag.contracts import DocRecord
from rag.ingest.chunkers.recursive import SEPARATORS as RECURSIVE_SEPARATORS
from rag.ingest.headings import MAX_LEVEL
from rag.ingest.headings import RULE_VERSION as HEADING_RULE_VERSION
from rag.ingest.loaders.base import doc_id_for
from rag.ingest.pipeline import SHELL_FILTER_VERSION

# 哪些策略用 `chunkers/recursive.py` 的那组分隔符 —— 见 pipeline_version 里的用法。
# markdown-semantic **不在表里**：它有自己的三级阶梯（semantic.SEPARATORS），
# 改那一组时要手动在 parts 里补一位（照 `cleaning:1` 的先例），否则已灌语料不会重切。
SEPARATORS_BY_STRATEGY: dict[str, tuple[str, ...]] = {
    "recursive": tuple(RECURSIVE_SEPARATORS),
    "markdown": tuple(RECURSIVE_SEPARATORS),  # 超长小节的递归兜底用的是同一组
}


def file_hash(path: Path) -> str:
    """文件字节的 sha1[:16] —— **变更检测的钥匙**。

    只读文件、不用解析，所以便宜到能对整个语料跑一遍。不选 mtime（备份/同步工具会改它）
    也不选 size（改名换字都会漏）。
    """
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def pipeline_version(settings: Settings) -> str:
    """切分与清洗这套「做法」的指纹。

    **内容没变、但结果该变**的情况：改了切分策略、改了清洗规则、换了 embedding 模型。
    这时文件哈希完全相同，增量会跳过它 —— 库里就留着用旧规则切出来的块。
    所以注册表要存这个指纹，对不上就强制重灌那一份。

    collection 名管「向量空间变了」，这里管「切法变了」。
    """
    chunk = settings.indexing.chunk
    embedding = settings.providers.embedding
    ocr = settings.ingest.ocr
    images = settings.ingest.images
    # 图片描述**也必须进指纹**：打开开关后文件一个字节都没变，增量会把它们全跳过，
    # 结果一张图都没描述、库里却「看起来是新的」。
    # 开关关着时只记 False —— 否则改一下 VL 模型名就会白白重灌整个语料。
    vision = f"vision:{images.describe}"
    if images.describe:
        # max_images 也要进：从「先跑 3 张」改成全量时，得让没描述的图重新排队
        vision += f"/{settings.providers.vision.model}/{VISION_PROMPT_VERSION}/{images.max_images}"
    parts = [
        f"chunk:{chunk.strategy}/{chunk.size}/{chunk.overlap}",
        f"semantic:{chunk.semantic.min_size}/{chunk.semantic.percentile}",
        # 标题层级的重建规则（chunkers 都用它算 heading_path / heading_numbers，清单也靠它）：
        # 规则一改，切出来的块就变了，必须让已灌语料重切。
        # **级别上限一起记**：从「到 h4」改成「只到 h3」不改规则号，但切法确实变了。
        f"headings:{HEADING_RULE_VERSION}/{MAX_LEVEL}",
        # 空壳块过滤：删掉哪些块也决定「库里该有哪些块」，规则一改就得重切
        f"shells:{SHELL_FILTER_VERSION}",
        f"pdf:{settings.ingest.pdf.parser}",
        f"ocr:{ocr.restructure_pages}/{ocr.merge_tables}/{ocr.relevel_titles}",
        f"embedding:{embedding.backend}/{embedding.model}/{embedding.dim}",
        vision,
        # 清洗规则的版本号：改了 cleaning.py 的规则就手动抬一下，否则老块不会重灌
        "cleaning:1",
    ]
    # 分隔符只对**真正用它**的策略追加。写死进去、或给别的策略补一个空占位，
    # 都会让指纹整体变化 → 已灌的语料被判定「切法变了」→ 全量重灌一遍
    # （重新解析、重新 OCR、重新问 VLM，最后两步都花钱）。所以宁可只覆盖一部分策略，
    # 也不要为了整齐而凭空触发重灌。用常量本身而不是手写版本号：改了它就自动失效。
    if separators := SEPARATORS_BY_STRATEGY.get(chunk.strategy):
        parts.append(f"separators:{separators!r}")
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


class IngestPlan(BaseModel):
    """一次 ingest 要做什么。全是路径 / 标识，不碰库。"""

    # 注册表里没有的
    added: list[Path] = Field(default_factory=list)
    # 有，但内容哈希或 pipeline 指纹对不上
    changed: list[Path] = Field(default_factory=list)
    # 有，且完全一致 —— **连解析都不做**
    skipped: list[Path] = Field(default_factory=list)
    # 注册表里有、语料里找不到文件的（只记 doc_id，文件本身已经没了）
    deleted: list[str] = Field(default_factory=list)

    @property
    def todo(self) -> list[Path]:
        return [*self.added, *self.changed]


def plan_incremental(
    files: list[Path],
    *,
    root: Path,
    registry: dict[str, DocRecord],
    pipeline: str | Callable[[Path], str],
    detect_deleted: bool = False,
) -> IngestPlan:
    """三态 diff（新增 / 变更 / 跳过），外加第四态：**文件被删除**。

    第四态是之前完全没处理的 —— 语料里删掉一份文件，库里的块会永远留着。
    只在**目录源**上做检测（单文件源看不出「别的文件没了」），判据是
    「注册表里的 doc_id 不在本次发现里，且它的路径在磁盘上也不存在了」。

    ``pipeline`` 可以是字符串（全局指纹），也可以是**按文件算**的函数 ——
    PDF 的标题清单是人工对齐的，一份文档的清单改了，只有那份文档该重灌，
    所以调用方要能把「这一份的清单哈希」也算进它的指纹里。
    """
    plan = IngestPlan()
    seen: set[str] = set()
    fingerprint = pipeline if isinstance(pipeline, str) else None

    for path in files:
        doc_id = doc_id_for(path, root)
        seen.add(doc_id)
        record = registry.get(doc_id)
        expected = fingerprint if fingerprint is not None else pipeline(path)  # type: ignore[operator]
        if record is None:
            plan.added.append(path)
        elif record.content_hash == file_hash(path) and record.pipeline_version == expected:
            plan.skipped.append(path)
        else:
            plan.changed.append(path)

    if detect_deleted:
        plan.deleted = [
            doc_id
            for doc_id, record in registry.items()
            if doc_id not in seen and not (root / record.source_uri).exists()
        ]
    return plan
