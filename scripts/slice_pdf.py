"""按页码切 PDF：给解析器 / 产线做小样本，别每次拿整份几百页去试。

    python scripts/slice_pdf.py "Apache Kafka Documentation v2.5.pdf" 1-4,57-60,118
    python scripts/slice_pdf.py in.pdf 8-10 -o probe.pdf

页码 1-based，逗号分隔，支持区间；页序按传入顺序，页码可重复。
默认输出 <源文件名>_slice.pdf。切片不重写页面内容，字体与连字缺陷原样带过去。
"""

import sys
from collections.abc import Iterable
from pathlib import Path

import pymupdf


def parse_page_spec(spec: str) -> list[int]:
    """把 "1-4,57-60,118" 解析成页码列表（1-based，保持书写顺序）。"""
    pages: list[int] = []
    for part in spec.replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        head, sep, tail = part.partition("-")
        if sep:
            pages.extend(range(int(head), int(tail) + 1))
        else:
            pages.append(int(part))
    return pages


def slice_pdf(source: Path, target: Path, pages: Iterable[int]) -> Path:
    """切出指定页（1-based）另存为新 PDF，页序按传入顺序，页码可重复。

    越界页码直接报错，不悄悄少切几页 —— 拿切片比对解析效果时，
    「源里少了一页」看起来会像「解析器漏了页」。
    """
    with pymupdf.open(source) as doc:
        total = doc.page_count
        indices: list[int] = []
        for page in pages:
            if not 1 <= page <= total:
                raise ValueError(f"页码越界：{page}（源文件共 {total} 页）")
            indices.append(page - 1)
        if not indices:
            raise ValueError("没指定任何页")
        # 空文档上 select() 会抛 "bad page number(s)"，逐页 insert_pdf 才行
        with pymupdf.open() as out:
            for index in indices:
                out.insert_pdf(doc, from_page=index, to_page=index)
            out.save(target)
    return target


if __name__ == "__main__":
    argv = sys.argv[1:]
    usage = "用法：slice_pdf.py <源.pdf> <页码，如 1-4,57-60,118> [-o 输出.pdf]"
    if not (len(argv) == 2 or (len(argv) == 4 and argv[2] in ("-o", "--out"))):
        sys.exit(usage)

    source = Path(argv[0])
    if not source.exists():
        sys.exit(f"源文件不存在：{source.resolve()}")
    target = Path(argv[3]) if len(argv) == 4 else source.with_name(f"{source.stem}_slice.pdf")

    pages = parse_page_spec(argv[1])
    result = slice_pdf(source, target, pages)

    with pymupdf.open(result) as out:
        size_kb = result.stat().st_size / 1024
        print(f"{source.name} -> {result.name}  共 {out.page_count} 页  {size_kb:.0f} KB")
    print(f"页码（1-based）：{pages}")
