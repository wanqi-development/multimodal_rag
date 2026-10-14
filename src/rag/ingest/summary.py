"""产物摘要：把「这次解析 / 切分做到了什么程度」算成能给页面看的数字。

**现算，不写进产物**：产物格式是数据，不是报表 —— 多维护一份统计就多一处会不一致的地方。
两份 json 读一次几十毫秒（实测一份 393 块的 `chunks.json` 是 402KB，1691 块的那份 1.7MB），
页面上完全够用。

这里**只吃 dict / list**：不碰文件系统、不碰 Milvus，所以既能被 API 用，也能被 CLI 用，
测试直接喂真实产物的 payload 就行。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def parse_stats(document: dict[str, Any]) -> dict[str, Any]:
    """`document.json` 里的 `document` 字段 → 解析结果摘要。

    `images_missing` 来自 loader 的 metadata：**取不到的图要一直看得见**，不能只在导入时
    闪一下提示条（少几张图会让检索结果悄悄变差）。
    """
    metadata = document.get("metadata") or {}
    return {
        "chars": len(document.get("text") or ""),
        "images": len(document.get("images") or []),
        "missing_images": [str(item) for item in (metadata.get("images_missing") or [])],
        "parser": str(metadata.get("parser") or ""),
        "mime": str(document.get("mime") or ""),
    }


def chunk_stats(chunks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """块列表 → 切分结果摘要（含画分布条用的 `sizes` 与大纲树）。

    `sizes` 是**每块的字数**，页面上接成一条宽度正比于字数的色条 —— 「平均 465、最大 1053」
    这种数字不如一眼看出切得匀不匀。
    """
    sizes = [len(chunk.get("text") or "") for chunk in chunks]
    image_blocks = sum(1 for chunk in chunks if chunk.get("modality") == "image")
    return {
        "blocks": len(chunks),
        "chars": sum(sizes),
        "avg_chars": round(sum(sizes) / len(sizes)) if sizes else 0,
        "max_chars": max(sizes) if sizes else 0,
        "text_blocks": len(chunks) - image_blocks,
        "image_blocks": image_blocks,
        "sizes": sizes,
        "sections": section_rollup(chunks),
    }


def section_rollup(chunks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 `heading_path` 聚合成大纲：**哪几节、各多少块多少字**。

    页面上要的是这个，不是 393 条块 —— 块级明细留给 `chunks.md`。
    顺序按**第一次出现**（也就是它在文档里的位置）；`heading_path` 为空的那些（第一个标题
    之前的正文）自成一类排在最前。

    每节还带一行 `head`（**第一块的开头几十个字**）：光看「（无标题）1 块 133 字」，没人
    判断得出那是什么 —— 给它一行原文，一眼就知道是正文还是页面导航的残留。
    """
    from rag.ingest.assets import MARKER_RE

    buckets: dict[tuple[str, str], dict[str, Any]] = {}
    for chunk in chunks:
        key = (str(chunk.get("heading_path") or ""), str(chunk.get("heading_numbers") or ""))
        bucket = buckets.setdefault(
            key,
            {
                "heading_path": key[0],
                "heading_numbers": key[1],
                "blocks": 0,
                "chars": 0,
                "head": "",
            },
        )
        text = str(chunk.get("text") or "")
        if not bucket["head"]:
            # 图片标记换成 [图片] 再压成一行：它是三十几个字符的资产路径，摆这儿只是噪声
            bucket["head"] = " ".join(MARKER_RE.sub("[图片]", text).split())[:90]
        bucket["blocks"] += 1
        bucket["chars"] += len(text)
    return list(buckets.values())


def vector_bytes(blocks: int, dim: int) -> int:
    """稠密向量的**理论**占用：每块 `dim` 个 float32。

    只是估算，页面上要标明：Milvus 那边还有索引、稀疏向量（服务端从 text 生成）与标量字段
    的账，这里给的是「向量本体」那一份。
    """
    return max(blocks, 0) * max(dim, 0) * 4
