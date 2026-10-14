"""跨通道融合（L2）：把多路召回的候选并成一张榜。

**为什么用 RRF（Reciprocal Rank Fusion）而不是加权求和**：各通道的分数根本不在一个量纲上
—— dense 是 COSINE（0..1）、sparse 是 BM25（无上界的累加值）、以后的 web 又是另一种。
归一化要按语料调参，调出来的权重还容易过拟合；RRF 只用**名次**：

    score(d) = Σ_i 1 / (k + rank_i(d))

跨通道天然可比，也不需要谁先做过校准。Milvus 服务端融 dense+sparse 用的就是同一套
（`hybrid.fusion.strategy: rrf`），这里只是把它搬到客户端、扩到任意路通道。

两条实现上的约定：

* **rank 从 1 起**（第 1 名的贡献是 1/(k+1)）；输入里那个 ``rank`` 字段不参与计算 ——
  它是上一层的产物，按列表顺序重数才是融合该用的口径。
* 同一路里重复出现的 chunk_id **只算一次**（服务端返回重复行时不该被重复加权）。
"""

from __future__ import annotations

from collections.abc import Sequence

from rag.contracts import ScoredChunk


def rrf_fuse(
    ranked: Sequence[Sequence[ScoredChunk]],
    *,
    k: int = 60,
    sources: Sequence[str] | None = None,
) -> list[ScoredChunk]:
    """多路候选 → 一张榜。``ranked[i]`` 是第 i 路按名次排好的结果。

    ``sources`` 给每一路起个名字（进 ``ScoredChunk.sources``，供调试与加权用）；
    不传就用命中自带的 ``channel``。

    相同分数时按 chunk_id 排 —— 顺序必须是**确定**的，否则同一份数据两次检索能给出
    不同的榜，测试与排查都无从下手。
    """
    scores: dict[str, float] = {}
    picked: dict[str, ScoredChunk] = {}
    origin: dict[str, list[str]] = {}

    for index, hits in enumerate(ranked):
        given = sources[index] if sources is not None and index < len(sources) else ""
        # 没给名字就跟着命中自报的 channel；一路之内它是同一个值，取第一条即可
        name = given or (hits[0].channel if hits else "")
        seen: set[str] = set()
        for rank, hit in enumerate(hits, start=1):
            key = hit.chunk.chunk_id
            if key in seen:  # 同一路里的重复行只算一次
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            origin.setdefault(key, [])
            if name and name not in origin[key]:
                origin[key].append(name)
            picked.setdefault(key, hit)

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [
        ScoredChunk(
            chunk=picked[key].chunk,
            score=score,
            channel="rrf",
            rank=rank,
            sources=origin[key],
        )
        for rank, (key, score) in enumerate(ordered)
    ]
