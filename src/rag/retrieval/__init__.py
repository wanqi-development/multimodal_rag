"""检索层：把查询变成候选块，再把候选块装配成给模型的上下文。

    retriever = build_retriever(settings, index, embedder)
    hits = await retriever.retrieve(query, top_k)
    context = assemble(hits, settings.retrieval.assemble.max_tokens)

**两层融合要分清**（见 docs/architecture.md §3）：

* L1 —— Milvus 服务端把 dense + sparse 用 RRF 融成一路：**语义 + 关键词都在这一路里**；
* L2 —— 客户端再把各路**数据源**融一次，由 ``MultiChannelRetriever`` 负责。

**装配由配置驱动**（2026-10-08）：`retrieval.channels` 是一张表，每一路 = 一个数据源
（`name` / `type` / …）。只有一路时**原样返回它**（不包一层），保持单路行为与性能不变。
`type` 怎么找到适配器见 ``retrieval/plugins.py``（内置注册表 → entry points → 点路径）。

**谁在调这一段**：检索页、快问答（StateGraph 的 retrieve 节点）、agent 的
``search_knowledge`` 工具 —— 三处共用 ``build_retriever``，所以"一次工具调用 = 一次
融合检索"。关键词不再有独立通道（2026-10-08）：它就是 Milvus 那一路的 sparse 腿。
"""

from __future__ import annotations

from rag.config.schema import ChannelConfig, Settings
from rag.contracts import Embedder, Reranker, Retriever
from rag.indexing.milvus import MilvusIndex
from rag.retrieval.assembler import assemble, estimate_tokens
from rag.retrieval.channels import MilvusHybridRetriever
from rag.retrieval.fusion import rrf_fuse
from rag.retrieval.multichannel import MultiChannelRetriever, RetrievalError
from rag.retrieval.plugins import resolve_channel_type

__all__ = [
    "MilvusHybridRetriever",
    "MultiChannelRetriever",
    "RetrievalError",
    "assemble",
    "build_channel",
    "build_retriever",
    "estimate_tokens",
    "resolve_channel_type",
    "rrf_fuse",
]


def build_retriever(
    settings: Settings,
    index: MilvusIndex,
    embedder: Embedder,
    reranker: Reranker | None = None,
) -> Retriever:
    """**按配置里那张表装配**：每一路 = 一个数据源，多路就 L2 融合成一路。

    加一路的唯一动作是往 `retrieval.channels` 里加一项（`type` 决定用哪个适配器，
    插件化见 `retrieval/plugins.py`）—— 这里不再有"主通道"这个特殊概念。

    ``reranker`` 由调用方用 ``providers.build_reranker(settings)`` 建好传进来：
    检索层不 import providers（那是一大包 fastembed/torch），装配留给上层。
    ``retrieval.rerank.enabled`` 开着却没传 → **当场报错**，不静默跳过 ——
    「配了重排、实际没重排」是查不出效果的那种问题（与 image describe 同一个立场）。
    """
    cfg = settings.retrieval
    if cfg.rerank.enabled and reranker is None:
        raise ValueError(
            "retrieval.rerank.enabled 开着，但没有传 reranker："
            "调用方要用 providers.build_reranker(settings) 建一个传进来，"
            "或者把 retrieval.rerank.enabled 关掉。"
        )
    enabled = [item for item in cfg.channels if item.enabled]
    if not enabled:  # 配置校验也会拦，这里兜一层（settings 可能在代码里被改过）
        raise ValueError("retrieval.channels 里一路都没启用：检索没有数据源可用")
    channels = [build_channel(item, settings, index, embedder) for item in enabled]
    limits = {item.name: item.top_k for item in enabled if item.top_k}
    if len(channels) == 1 and reranker is None:
        # 单路（也是默认档）：原样返回，不包一层 —— 保持单路的行为与性能不变
        return channels[0]
    return MultiChannelRetriever(channels, cfg.fusion, reranker, limits=limits)


def build_channel(
    cfg: ChannelConfig,
    settings: Settings,
    index: MilvusIndex,
    embedder: Embedder | None = None,
) -> Retriever:
    """按配置里的**一项**建一路通道：`type` → 适配器（插件化），然后钉上这一路的名字。

    名字统一用配置里的 `name`（不再让适配器自己取一个）—— 日志、通道账本、RRF 的
    `sources`、`--channel` 全都看这一个名字，省得出现"两套身份"。
    """
    factory = resolve_channel_type(cfg.type)
    channel = factory(index=index, embedder=embedder, settings=settings, cfg=cfg)
    if not callable(getattr(channel, "retrieve", None)):
        raise TypeError(
            f"通道 {cfg.name!r}（type={cfg.type}）建出来的东西没有 retrieve()：{channel!r}"
            f"；适配器要实现 Retriever 协议（见 retrieval/plugins.py 的接口约定）"
        )
    channel.name = cfg.name
    if cfg.complement_only is not None:
        channel.complement_only = bool(cfg.complement_only)
    return channel
