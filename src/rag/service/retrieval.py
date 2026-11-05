"""各入口共用的检索接线及相关性判断；重对象注入，检索器按调用创建。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from rag.config.schema import Settings
from rag.contracts import Embedder, Reranker, RetrievalResult
from rag.obs.log import record_retrieval
from rag.retrieval import build_retriever, configured_hyde
from rag.retrieval.multichannel import retrieve_channel_result

Relevance = Literal["relevant", "irrelevant", "unknown"]


class UnknownChannelError(ValueError):
    """请求包含未配置的通道；入口可映射为参数错误，不吞掉检索异常。"""


def resolve_channels(settings: Settings, channels: list[str] | None) -> set[str]:
    """解析本次启用的通道名，不修改配置。

    None 或空列表沿用配置；非空列表仅启用指定通道（可包含配置中已关闭的通道）。
    重复名称去重，任一未知名称都会拒绝整个请求，不静默使用剩余通道。

    Raises:
        UnknownChannelError: 请求包含配置中不存在的通道名。
    """
    configured = settings.retrieval.channels
    if not channels:
        return {item.name for item in configured if item.enabled}
    known = {item.name for item in configured}
    unknown = sorted(set(channels) - known)
    if unknown:
        raise UnknownChannelError(
            f"未知的通道 {unknown}；配置里有 {sorted(known)}（见 retrieval.channels）"
        )
    return set(channels)


def assess_relevance(result: RetrievalResult, threshold: float) -> Relevance:
    """只判断检索相关性，不代表答案充分、事实真实或联网授权。

    任何通道失败都代表检索不完整，无论候选分数高低均不能确认相关性。
    """
    if any(item.code == "channel_failed" for item in result.degradations):
        return "unknown"
    if not result.chunks:
        return "irrelevant"
    if result.score_kind == "rerank" and result.rerank_status == "succeeded":
        return "relevant" if max(hit.score for hit in result.chunks) >= threshold else "irrelevant"
    return "unknown"


class RetrievalService:
    """显式复用模型与索引；不保存 last_result、账本或其他查询状态。"""

    def __init__(
        self,
        settings: Settings,
        index: Any,
        embedder: Embedder,
        reranker: Reranker | None,
        chat: Any,
    ) -> None:
        self._settings = settings
        self._index = index
        self._embedder = embedder
        self._reranker = reranker
        self._chat = chat

    async def retrieve(
        self,
        query: str,
        top_k: int,
        *,
        filters: dict[str, Any] | None = None,
        channels: list[str] | None = None,
        image: Path | None = None,
        rerank: bool | None = None,
    ) -> RetrievalResult:
        """用有效配置的副本检索；旧插件只需实现 retrieve，异常保持抛出。"""
        local = self._settings.model_copy(deep=True)
        wanted = resolve_channels(local, channels)
        for item in local.retrieval.channels:
            item.enabled = item.name in wanted
        if rerank is not None:
            local.retrieval.rerank.enabled = rerank
        try:
            retriever = build_retriever(
                local,
                self._index,
                self._embedder,
                self._reranker,
                hyde=configured_hyde(local, self._chat),
            )
            result = await retrieve_channel_result(
                retriever,
                query,
                top_k,
                filters=filters,
                image=image,
            )
            if not callable(getattr(retriever, "retrieve_result", None)):
                result = result.model_copy(
                    update={
                        "rerank_status": "unknown"
                        if local.retrieval.rerank.enabled
                        else "disabled",
                    }
                )
        except Exception:
            record_retrieval(RetrievalResult(query=query), error="retrieval_failed")
            raise
        record_retrieval(result)
        return result
