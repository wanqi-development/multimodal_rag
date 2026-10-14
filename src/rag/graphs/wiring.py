"""把「生产环境怎么接图」这一层装配收在一处。

CLI 与 API 需要的是**同一套**接线：怎么建检索器、plan/grade 走哪家 chat、生成器是谁。
以前这套写在 `cli/main.py` 里，API 要用就得复制一份 —— 复制出来的两份迟早会走散
（比如「重对象只建一次」这条，M3 真机踩过：写错就是每轮重新加载一次 1GB 模型）。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from rag.config.schema import Settings
from rag.contracts import AssembledContext, ChannelStats, ScoredChunk

# 图要的四个外部依赖（见 graphs/query.py 的说明）
RetrieveFn = Callable[..., Awaitable[tuple[list[ScoredChunk], list[ChannelStats]]]]
GenerateFn = Callable[[str, AssembledContext], Awaitable[str]]
JsonFn = Callable[[str], Awaitable[str]]
QueryNodes = tuple[RetrieveFn, JsonFn, JsonFn, GenerateFn]


@dataclass
class Heavy:
    """一个进程里**只该有一份**的重对象。

    M3 真机踩过这个坑：检索闭包每次被调（每个子查询、每一轮改写）都重建一次 embedder
    与 reranker，输出里出现三次「Loading weights」——白等好几秒、还反复申请 1GB 级内存。
    所以「建一次」这件事做成**结构**：重对象的持有者由调用方（CLI 一次命令 / API 一次
    lifespan）建好，闭包只**用**它，没有地方可以偷偷重建。
    """

    settings: Settings
    index: Any
    embedder: Any
    reranker: Any | None
    chat: Any
    generator: Any
    errors: dict[str, str] = field(default_factory=dict)


def build_heavy(settings: Settings, index: Any, *, chat_name: str = "default") -> Heavy:
    """建重对象。**每处只该调一次**（CLI 一次命令 / API 一次启动）。

    单个模型建不起来时**不让整件事失败**：记进 `errors` 继续走 —— API 那边靠它给出
    「503 + 具体原因」，而不是整个服务起不来（`/health` 也会照实报）。
    """
    from rag.providers import build_chat, build_embedder, build_reranker

    errors: dict[str, str] = {}

    def _try(name: str, factory: Callable[[], Any], default: Any = None) -> Any:
        try:
            return factory()
        except Exception as exc:  # noqa: BLE001 —— 缺 key / 模型下不来都在这里
            errors[name] = f"{type(exc).__name__}: {exc}"
            return default

    chat = _try("chat", lambda: build_chat(settings, _secrets(), chat_name))
    return Heavy(
        settings=settings,
        index=index,
        embedder=_try("embedding", lambda: build_embedder(settings)),
        reranker=(
            _try("reranker", lambda: build_reranker(settings))
            if settings.retrieval.rerank.enabled
            else None
        ),
        chat=chat,
        # 生成器复用同一个 chat 实例（一份连接池、一份配置）
        generator=_try(
            "generator",
            lambda: _generator(chat) if chat is not None else _raise("chat 没建起来"),
        ),
        errors=errors,
    )


def build_retrieve(heavy: Heavy) -> RetrieveFn:
    """检索闭包。每个子查询**重新装配**一个检索器（计划可能指定了别的 channels）——
    `build_retriever` 只是接线，很便宜；重对象来自 `heavy`，不会重建。"""
    from rag.retrieval import build_retriever

    async def retrieve(
        query: str,
        top_k: int,
        *,
        filters: dict[str, Any] | None = None,
        channels: list[str] | None = None,
    ) -> tuple[list[ScoredChunk], list[ChannelStats]]:
        # 深拷贝一份再改：计划里指定的通道不该污染这次进程里别的调用
        local = heavy.settings.model_copy(deep=True)
        if channels:
            # 计划给的是**通道名**（`retrieval.channels[].name`）：列出来的开、其余关
            wanted = set(channels)
            for item in local.retrieval.channels:
                item.enabled = item.name in wanted
        retriever = build_retriever(local, heavy.index, heavy.embedder, heavy.reranker)
        hits = await retriever.retrieve(query, top_k, filters=filters or None)
        return hits, getattr(retriever, "last_stats", [])

    return retrieve


def build_json_caller(heavy: Heavy) -> JsonFn:
    """plan / grade 用的「prompt → 模型原文」。

    JSON 的解析放在图里（`graphs/query.py`），所以这一层与模型/网关无关。
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    system = SystemMessage("你是严谨的检索规划与评估助手，只输出要求的 JSON。")

    async def ask(prompt: str) -> str:
        reply = await heavy.chat.ainvoke([system, HumanMessage(prompt)])
        content = reply.content
        return content if isinstance(content, str) else str(content)

    return ask


def build_generate(heavy: Heavy, usage: dict[str, Any] | None = None) -> GenerateFn:
    """生成节点。`usage` 给了就把 token 用量记进那个容器（图不返回用量，
    而调用方要记 token 账）。"""

    async def generate(question: str, context: AssembledContext) -> str:
        return await heavy.generator.generate(question, context, usage=usage)

    return generate


def build_query_nodes(
    settings: Settings,
    index: Any,
    heavy: Heavy | None = None,
    *,
    usage: dict[str, Any] | None = None,
) -> QueryNodes:
    """图要的四个依赖一次给全（`heavy` 给了就复用，没给就现建 —— 只现建一次）。"""
    heavy = heavy or build_heavy(settings, index)
    caller = build_json_caller(heavy)
    return build_retrieve(heavy), caller, caller, build_generate(heavy, usage)


def _generator(chat: Any) -> Any:
    from rag.providers.generator import CitationGenerator

    return CitationGenerator(chat)


def _raise(message: str) -> Any:
    raise RuntimeError(message)


def _secrets():
    from rag.config import get_secrets

    return get_secrets()
