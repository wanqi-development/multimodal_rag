"""与数据来源解耦的评测原语。

当前不维护固定黄金集，也不提供批量 `rag eval` 命令。保留以下能力，供以后对
任意检索结果快照或线上抽样结果做评测：

- 检索指标纯函数；
- 引用准确率 / 要点覆盖 / 拒答 judge；
- ragas faithfulness 适配器；
- JSON / Markdown 报告汇总。
"""

from rag.eval.judge import JudgeVerdict, judge_case, parse_verdict, used_citations
from rag.eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from rag.eval.models import EvaluationRow
from rag.eval.report import build_report, render_markdown, write_report

__all__ = [
    "EvaluationRow",
    "JudgeVerdict",
    "build_report",
    "judge_case",
    "ndcg_at_k",
    "parse_verdict",
    "recall_at_k",
    "reciprocal_rank",
    "render_markdown",
    "used_citations",
    "write_report",
]
