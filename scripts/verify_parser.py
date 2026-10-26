"""验证接入层解析器：拿一份真文件走一遍 loader + chunker，看它到底解析出了什么。

    python scripts/verify_parser.py "Apache Kafka Documentation v2.5.pdf"

用 config 里的 ingest 配置（默认 `parser: ocr`，所有 PDF 统一走产线）——
与 `rag ingest` 同一条路径，不另开一套参数。诊断脚本，只打印，不做断言。
"""

import sys
import time
from pathlib import Path

from rag.config import load_settings
from rag.ingest import build_chunker, build_ocr
from rag.ingest.loaders import load_source
from rag.providers import build_embedder

path = Path(sys.argv[1] if len(sys.argv) > 1 else "Apache Kafka Documentation v2.5.pdf")
if not path.exists():
    sys.exit(f"文件不存在：{path.resolve()}")

settings = load_settings()
ingest = settings.ingest
ocr = build_ocr(ingest.ocr)
print(f"文件      {path.name}  {path.stat().st_size / 1024 / 1024:.1f} MB")
print(f"parser    {ingest.pdf.parser}（ocr = 统一走产线；pymupdf = 有损降级）")
print(f"OCR       {ocr.base_url if ocr else '未启用（纯 pymupdf）'}")
print()

# ---- 1. 走一遍 loader（与 rag ingest 同一条路径）----
start = time.perf_counter()
report = load_source(path, ingest, ocr)
load_s = time.perf_counter() - start
print(f"1. loader（{load_s:.2f}s）")
print(f"   成功          {len(report.documents)} 篇")
for item in report.skipped:
    print(f"   跳过          {item.path}：{item.reason}")
if not report.documents:
    sys.exit(1)

doc = report.documents[0]
print(f"   doc_id        {doc.doc_id}")
print(f"   source_uri    {doc.source_uri}")
print(f"   metadata      {doc.metadata}")
print(f"   正文字符数    {len(doc.text)}")

headings = [line for line in doc.text.splitlines() if line.lstrip().startswith("#")]
print(f"   以 # 开头的行  {len(headings)}")
print(f"   图片          {len(doc.images)} 张")
print()

# ---- 2. 正文抽样：开头 / 中间 / 结尾，看有没有解析出乱码或断层 ----
print("2. 正文抽样")
third = max(len(doc.text) // 3, 1)
for label, offset in (("开头", 0), ("中间", third), ("结尾", len(doc.text) - 600)):
    snippet = doc.text[offset : offset + 300].replace("\n", " ⏎ ")
    print(f"   [{label}] {snippet}")
print()

# ---- 3. 切块策略对比：标题层级能不能落到 chunk 上 ----
print("3. 切块")
embedder = build_embedder(settings)
for strategy in ("recursive", "markdown", "markdown-semantic"):
    cfg = settings.indexing.chunk.model_copy(update={"strategy": strategy})
    start = time.perf_counter()
    chunks = build_chunker(cfg, embedder).split(report.documents)
    elapsed = time.perf_counter() - start
    with_heading = sum(1 for c in chunks if any(k in c.metadata for k in ("h1", "h2", "h3")))
    sizes = [len(c.text) for c in chunks] or [0]
    print(
        f"   {strategy:<10} {len(chunks):>5} 块  {elapsed:.2f}s  "
        f"带标题的 {with_heading} 块  "
        f"长度 min={min(sizes)} avg={sum(sizes) // len(sizes)} max={max(sizes)}"
    )
    for chunk in chunks[:3]:
        tags = {k: v for k, v in chunk.metadata.items() if k in ("h1", "h2", "h3")}
        print(f"              {chunk.chunk_id}  {tags}  {chunk.text[:70]!r}")
