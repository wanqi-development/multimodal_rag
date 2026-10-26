"""上下文装配：去重 → token 预算裁剪 → 引用编号。

编号在这里生成并贯穿到最终答案。生成模型只被允许引用资料里的 [n]，
所以引用表和正文必须出自同一次装配 —— 分两处算必然对不上。
"""

from __future__ import annotations

import re

from rag.contracts import AssembledContext, Citation, EvidenceLedger, ScoredChunk

# CJK 及全角标点：中文按 1 字 ≈ 1 token 估，拉丁文按 4 字符 ≈ 1 token。
# 故意估高一点 —— 估低了会超模型窗口被静默截断，估高只是少放几条。
_CJK = re.compile(r"[　-〿一-鿿＀-￯]")

SNIPPET_CHARS = 80


def estimate_tokens(text: str) -> int:
    cjk = len(_CJK.findall(text))
    latin = len(text) - cjk
    return cjk + max(1, latin // 4)


def _citation(index: int, scored: ScoredChunk) -> Citation:
    meta = scored.chunk.metadata
    snippet = " ".join(scored.chunk.text.split())[:SNIPPET_CHARS]
    return Citation(
        index=index,
        source_type="corpus",
        chunk_id=scored.chunk.chunk_id,
        source_uri=str(meta.get("source_uri", "")),
        snippet=snippet,
        # 答案末尾的「引用了哪个文档的哪一章」全靠它
        heading_path=scored.chunk.heading_path,
    )


def assemble(chunks: list[ScoredChunk], max_tokens: int) -> AssembledContext:
    """按召回顺序吃进候选，去重并裁到预算内。

    候选顺序就是召回排序，所以「先到先得」意味着被丢掉的总是排得更靠后的。
    遇到超预算的条目是**跳过继续**而不是就此停止：一条长块不该让后面那些
    放得下的块一起被浪费。
    """
    seen: set[str] = set()
    picked: list[ScoredChunk] = []
    ledger = EvidenceLedger()
    used = 0
    dropped = 0

    for scored in chunks:
        chunk_id = scored.chunk.chunk_id
        if chunk_id in seen:
            # 同一块被多路召回是 hybrid 的常态，去重后只保留排名最靠前的那次
            dropped += 1
            continue
        seen.add(chunk_id)

        citation = _citation(ledger.next_index, scored)
        cost = estimate_tokens(scored.chunk.text) + estimate_tokens(_render(citation, scored))
        if used + cost > max_tokens:
            dropped += 1
            continue

        used += cost
        picked.append(scored)
        number = ledger.issue(
            source_type=citation.source_type,
            chunk_id=citation.chunk_id,
            source_uri=citation.source_uri,
            title=citation.title,
            snippet=citation.snippet,
            heading_path=citation.heading_path,
        )
        assert number == citation.index

    return AssembledContext(
        text="\n\n".join(_render(c, s) for c, s in zip(ledger.records(), picked, strict=True)),
        citations=ledger.records(),
        dropped=dropped,
        tokens_estimate=used,
    )


def _render(citation: Citation, scored: ScoredChunk) -> str:
    """一块资料的呈现形式。编号和来源都带上，模型才有东西可引。"""
    return f"[{citation.index}] 来源：{citation.source_uri}\n{scored.chunk.text}"
