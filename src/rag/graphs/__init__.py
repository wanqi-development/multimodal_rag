"""编排层：LangGraph 的两张图 + 统一答案引擎。

* ``query.py`` —— QueryGraph：plan → retrieve → grade →（不达标就改写重查）→ assemble；
  非流式可再接 generate，流式由 AnswerEngine 在预检后逐 token 生成
* ``index.py`` —— IndexGraph：discover → 按文件 fan-out（parse → 人工核对闸门 → chunk）→ 写库

两者都**只做编排**：真正干活的还是 ingest / retrieval / providers 里那些函数与契约。
所以这一层可以被完整替换（比如换成 Temporal、Airflow）而不动下层 —— 这也是当初把
三层协议定成 Protocol 的用途。
"""

from rag.graphs.answer import (
    AnswerEngine,
    AnswerRun,
    PreparedAnswer,
    build_answer_engine,
)
from rag.graphs.index import IndexGraphResult, build_index_graph, run_index_graph
from rag.graphs.query import (
    QueryGraphResult,
    QueryPreparation,
    build_query_graph,
    build_query_preflight,
    run_query_graph,
    run_query_preflight,
)

__all__ = [
    "AnswerEngine",
    "AnswerRun",
    "IndexGraphResult",
    "PreparedAnswer",
    "QueryGraphResult",
    "QueryPreparation",
    "build_answer_engine",
    "build_index_graph",
    "build_query_graph",
    "build_query_preflight",
    "run_index_graph",
    "run_query_graph",
    "run_query_preflight",
]
