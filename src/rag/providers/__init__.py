"""具体实现的存放处：按 backend 名字注册，上层只认 contracts 里的 Protocol。

约定：registry 里注册的类，构造函数都只收「它自己那段配置对象」。
这样装配层（build_* 函数）的写法对每个 backend 都一样。
"""

from __future__ import annotations

from rag.config.schema import Settings
from rag.config.settings import Secrets, get_secrets
from rag.contracts import EMBEDDERS, RERANKERS, Embedder, Reranker
from rag.providers import embedding as _embedding  # noqa: F401  触发注册
from rag.providers import rerank as _rerank  # noqa: F401  触发注册
from rag.providers.asr import build_asr
from rag.providers.chat import build_chat
from rag.providers.embedding import KEY_ENV as EMBEDDING_KEY_ENV
from rag.providers.generator import CitationGenerator, message_text
from rag.providers.vision import build_vision

__all__ = [
    "CitationGenerator",
    "build_asr",
    "build_chat",
    "build_embedder",
    "build_generator",
    "build_reranker",
    "build_vision",
    "message_text",
]


def build_embedder(settings: Settings, secrets: Secrets | None = None) -> Embedder:
    """按配置里的 backend 名实例化 embedder。

    本地两条路（fastembed / sentence-transformers）用不上 ``secrets``；
    多模态那条（qwen3-vl）走线上接口，要拿供应商的 key —— 与 ``build_vision`` 同一个约定。
    它由类属性 ``needs_api_key`` 自报，装配层照着一行判断即可，不用为每家写一个分支。
    """
    cfg = settings.providers.embedding
    cls = EMBEDDERS.get(cfg.backend)
    if cls is None:
        # 拼错 backend 名要当场说清楚有哪些可选，而不是拿 None 去调用换一个 TypeError
        raise KeyError(f"未注册的 embedding backend: {cfg.backend!r}；已注册: {sorted(EMBEDDERS)}")
    if getattr(cls, "needs_api_key", False):
        return cls(cfg, _embedding_key(cfg.provider, secrets))
    return cls(cfg)


def _embedding_key(provider: str, secrets: Secrets | None) -> str:
    """线上 embedding 的 key：按供应商名从 .env 取（与 chat / vision 同一套约定）。"""
    names = EMBEDDING_KEY_ENV.get(provider)
    if names is None:
        raise KeyError(f"未知的 embedding 供应商: {provider!r}；已知: {sorted(EMBEDDING_KEY_ENV)}")
    secrets = secrets or get_secrets()
    for name in names:
        key = getattr(secrets, name.lower(), None)
        if key:
            return key
    raise RuntimeError(f"缺少 {names[0]}：在项目根目录的 .env 里填上它。")


def build_generator(
    settings: Settings,
    secrets: Secrets | None = None,
    name: str = "default",
) -> CitationGenerator:
    """带引用的生成器。默认用 providers.chat.default 那个供应商。"""
    return CitationGenerator(build_chat(settings, secrets or get_secrets(), name))


def build_reranker(settings: Settings) -> Reranker:
    """按 ``providers.rerank.backend`` 建重排器（与 build_embedder 同一个形状）。

    两个 backend 都在本地跑（flashrank 的 ONNX / sentence-transformers 的 CrossEncoder），
    不调外部 API；但 flashrank 那条要一把 **HF token** 才拿得到模型（它自带的下载路
    既不带鉴权、文件名又过期），由类属性 ``needs_hf_token`` 自报 —— 与 embedder 的
    ``needs_api_key`` 同一套写法。
    """
    cfg = settings.providers.rerank
    cls = RERANKERS.get(cfg.backend)  # 未注册会抛 KeyError 并列出可选项
    if getattr(cls, "needs_hf_token", False):
        return cls(cfg, get_secrets().hf_token)
    return cls(cfg)
