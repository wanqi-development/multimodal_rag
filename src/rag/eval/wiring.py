"""把评测接到**生产链路**上（与 `graphs/wiring.py` 同一个思路：接线收在一处）。

评测跑的必须是线上那条路，否则分数是假的。所以这里三个依赖都取自同一批重对象：

* `retrieve` —— `build_retriever`（多通道 + 融合 + 重排，与 `rag search` 完全一致）；
* `generate` —— `assemble` + `heavy.generator`（与 `rag ask` 的直线一致）；
* `ask`     —— `heavy.chat`（与图上 plan/grade 同一个模型、同一份连接池）。

**引用清单给判定模型的是片段全文**（不是 `assemble` 里那 80 字的 snippet）——
拿 80 字去判「这段原文支不支持这句话」是在赌运气。
"""

from __future__ import annotations

from typing import Any

from rag.contracts import ScoredChunk
from rag.eval.judge import AskFn
from rag.eval.runner import GenerateFn, RetrieveFn


def build_retrieve(settings: Any, heavy: Any) -> RetrieveFn:
    """检索闭包。**重对象来自 heavy**（embedder / reranker 只建一次，见 graphs/wiring 的教训）。"""
    from rag.retrieval import build_retriever

    async def retrieve(question: str, top_k: int) -> list[ScoredChunk]:
        retriever = build_retriever(settings, heavy.index, heavy.embedder, heavy.reranker)
        return await retriever.retrieve(question, top_k)

    return retrieve


def build_generate(settings: Any, heavy: Any) -> GenerateFn:
    """生成闭包：装配上下文 → 生成 → 把引用翻译成「编号 + 来源 + 片段全文」。"""
    from rag.retrieval import assemble

    max_tokens = settings.retrieval.assemble.max_tokens

    async def generate(question: str, hits: list[ScoredChunk]) -> tuple[str, list[dict[str, Any]]]:
        context = assemble(hits, max_tokens)
        answer = await heavy.generator.generate(question, context)
        by_id = {hit.chunk.chunk_id: hit.chunk for hit in hits}
        citations = [
            {
                "index": item.index,
                "chunk_id": item.chunk_id,
                "source_uri": item.source_uri,
                "text": (by_id[item.chunk_id].text if item.chunk_id in by_id else item.snippet),
            }
            for item in context.citations
        ]
        return answer, citations

    return generate


def build_ask(heavy: Any) -> AskFn:
    from rag.eval.judge import build_asker

    return build_asker(heavy)
