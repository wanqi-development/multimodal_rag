"""检索侧的指标：**纯函数**，输入是「每个问题第一条相关结果的排名」。

为什么不在这里直接读检索结果：指标是最容易被写错、也最该被单测钉住的一块
（`rank` 从 1 还是 0 起、没命中算 0 还是算 inf）。所以这一层只认数字，不认识
`ScoredChunk`，`runner.py` 负责把命中变成排名。

四个指标各管一件事：

* `recall_at_k` —— 前 k 条里**有没有**相关材料（召回，回答「查得到吗」）；
* `reciprocal_rank`（MRR）—— 第一条相关材料排多前（回答「好找吗」）；
* `ndcg_at_k` —— 二元相关性下的排序质量（对「相关材料排在 3 还是 7」更敏感）；
* 命中率（rank==1 的比例）在报告里单独算 —— 它是最直观的那一个。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

# 没有相关结果时的排名：用它是为了让「没命中」在所有指标里都退化成 0，
# 而不是被当成「排得很靠后但还算命中」。
NO_HIT = 0


def recall_at_k(ranks: Iterable[int], k: int) -> float:
    """前 k 条里命中相关材料的比例。`ranks` 里 0（或 >k）都算没命中。"""
    values = list(ranks)
    if not values:
        return 0.0
    return sum(1 for rank in values if 0 < rank <= k) / len(values)


def reciprocal_rank(ranks: Iterable[int]) -> float:
    """MRR：`mean(1/rank)`；没命中记 0。"""
    values = list(ranks)
    if not values:
        return 0.0
    return sum(1.0 / rank if rank > 0 else 0.0 for rank in values) / len(values)


def ndcg_at_k(ranks: Iterable[int], k: int) -> float:
    """二元相关性下的 nDCG@k。

    每条问题只有「有 / 没有相关材料」两种，理想排序是相关材料排第一，所以
    IDCG 恒为 1（`1/log2(2)`），这里就简化成 DCG 本身 —— 但**要把这一点写出来**，
    否则读的人会以为漏算了 IDCG。
    """
    values = list(ranks)
    if not values:
        return 0.0
    total = 0.0
    for rank in values:
        if 0 < rank <= k:
            total += 1.0 / math.log2(rank + 1)
    return total / len(values)


def top1_rate(ranks: Iterable[int]) -> float:
    """第一条就是相关材料的比例 —— 报告里最直观的那一行。"""
    values = list(ranks)
    if not values:
        return 0.0
    return sum(1 for rank in values if rank == 1) / len(values)


def summarize(ranks: Sequence[int], *, k: int) -> dict[str, float]:
    """一次给全（报告里要一起打）。"""
    return {
        f"recall@{k}": recall_at_k(ranks, k),
        "mrr": reciprocal_rank(ranks),
        f"ndcg@{k}": ndcg_at_k(ranks, k),
        "top1": top1_rate(ranks),
    }
