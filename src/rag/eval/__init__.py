"""评测层（M6）：黄金集 → 指标 → 报告。

    cases = load_golden(Path("eval/golden.jsonl"))
    rows = await run_eval(cases, retrieve=..., generate=..., judge=...)
    payload = build_report(rows, meta={...})

**为什么先把这一层做出来**：M2/M3/M5 加了一堆可调参数（融合权重、`candidates_per_channel`、
rerank、1000 字切分、agent 提示词与 memory 措辞），在此之前一个数都没有 —— 改完只能靠
「感觉对了」。真机验证几轮已经证明这类直觉会翻车（memory 那句措辞就是一例）。

带 `judge` 的部分才调模型（花钱），检索那层用本地 embedding，随时能跑。
"""

from rag.eval.dataset import GoldenCase, is_relevant, load_golden
from rag.eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from rag.eval.report import build_report, render_markdown, write_report
from rag.eval.runner import CaseResult, first_relevant_rank, run_eval

__all__ = [
    "CaseResult",
    "GoldenCase",
    "build_report",
    "first_relevant_rank",
    "is_relevant",
    "load_golden",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "render_markdown",
    "run_eval",
    "write_report",
]
