"""按行切块：一个非空行 = 一个 chunk。

表格数据与散文是两种形状。散文按语义切 —— 一个段落可能被切成几块，几段也可能并成一块；
而表格的**一行本身就是最小的完整语义单元**：两行并成一块会把两条记录混在一起，
一行被切两半则两边都不完整。所以表格不套用四级切分。

**谁走这条路不由全局配置决定，而是由 loader 声明。** 一份语料里往往散文和表格混着
（md 和 xlsx 一起灌），一个全局策略服务不了两边 —— loader 在自己的
``Document.metadata["chunk_strategy"]`` 里写下切法，``prepare()`` 按文档分组分发。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag.config.schema import ChunkingConfig
from rag.contracts import Chunk, Document
from rag.ingest.chunkers.base import make_chunk, register_chunker

if TYPE_CHECKING:
    from rag.contracts import Embedder


@register_chunker("table-rows")
class TableRowsChunker:
    """一个非空行 = 一个 chunk。不合并、不按标题切、不做语义兜底。"""

    def __init__(self, cfg: ChunkingConfig, embedder: Embedder | None = None) -> None:
        """签名统一成 ``(cfg, embedder=None)``，见 base.py 的说明。

        **不看 size**：一条记录再长也是它自己，按大小压它就把记录压坏了。
        """
        self._cfg = cfg
        self.name = "table-rows"

    def split(self, docs: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in docs:
            index = 0
            for line in doc.text.splitlines():
                text = line.strip()
                if not text:
                    continue
                # 记录形如 `客户清单：名称=甲公司，数量=512` —— 冒号前那段是工作表名，
                # 它同时就是这一行的「标题」。这个约定由 loaders/xlsx.py 维护。
                heading = text.split("：", 1)[0] if "：" in text else ""
                chunks.append(make_chunk(doc, index, text, heading_path=heading, modality="table"))
                index += 1
        return chunks
