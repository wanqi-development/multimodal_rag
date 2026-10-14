"""multimodal_rag: 多模态 RAG 检索。

「多模态」的落点是接入层 + 存储层（多种模态的语料进来，统一成文本后检索），
查询侧不做跨模态。见 docs/architecture.md §1 与 ADR 19。

分层见 docs/architecture.md：
    contracts  纯协议层，零第三方依赖
    config     配置体系
    providers  多供应商适配
    ingest     接入：loader → parser → chunker
    indexing   索引：Milvus / 父块 / 图
    retrieval  检索：多通道 + 变换 + 融合 + 重排 + 装配
    graphs     LangGraph 编排
    agent      deepagents 对话式助手
    api / cli  服务入口
"""

__version__ = "0.1.0"

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
