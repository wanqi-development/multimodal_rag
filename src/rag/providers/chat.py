"""chat provider：配置里选一个名字，拿到一个 LangChain chat model。

多供应商切换 = 换配置里的 provider/model 并设好对应 key，不改代码。

国产供应商（deepseek / moonshot / zhipu / dashscope…）大多没有独立的 LangChain
集成包，但它们都提供 OpenAI 兼容的 /v1/chat/completions，所以走 openai 通道 +
各自的 base_url。实测 ``model_provider="deepseek"`` 会要求额外安装
``langchain-deepseek``，而 openai 通道无需任何附加依赖。
"""

from __future__ import annotations

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from rag.config.schema import ChatProviderConfig, Settings
from rag.config.settings import Secrets

# LangChain 有原生集成、不需要走 OpenAI 兼容通道的供应商
NATIVE_PROVIDERS = frozenset(
    {"openai", "anthropic", "google_genai", "google_vertexai", "mistralai", "groq", "ollama"}
)

# 供应商名 → 存 key 的环境变量名
KEY_ENV = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "dashscope": "DASHSCOPE_API_KEY",
    "zhipuai": "ZHIPUAI_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
}


def _require_key(cfg: ChatProviderConfig, secrets: Secrets) -> str:
    env_name = KEY_ENV.get(cfg.provider)
    if env_name is None:
        raise KeyError(
            f"未知的 chat provider: {cfg.provider!r}；已知: {sorted(KEY_ENV)}。"
            f"新增供应商时要在 providers/chat.py 的 KEY_ENV 里登记它的 key 变量名"
        )
    key = getattr(secrets, env_name.lower(), None)
    if not key:
        raise RuntimeError(
            f"缺少 {env_name}：在项目根目录的 .env 里填上它。"
            f"（当前用 providers.chat 里的哪个条目由参数决定，也可换一个已配好 key 的供应商）"
        )
    return key


def build_chat(
    settings: Settings,
    secrets: Secrets,
    name: str = "default",
) -> BaseChatModel:
    """按 ``providers.chat.<name>`` 构造 chat model。"""
    cfg = settings.providers.chat.get(name)
    if cfg is None:
        raise KeyError(f"配置里没有 providers.chat.{name}；已有: {sorted(settings.providers.chat)}")

    provider = cfg.provider
    if provider not in NATIVE_PROVIDERS:
        if not cfg.base_url:
            raise ValueError(
                f"{provider} 没有原生 LangChain 集成，走 OpenAI 兼容通道就必须给 base_url"
                f"（配置项 providers.chat.{name}.base_url）"
            )
        provider = "openai"

    extra: dict[str, Any] = {}
    if provider == "openai":
        # **流式也要拿到 token 用量**：OpenAI 兼容通道默认不在最后一个分片里带 usage，
        # 要显式带 `stream_options.include_usage`（`init_chat_model` 会把它透传给
        # `ChatOpenAI`）。只对这条通道开 —— 原生集成（anthropic 等）不认这个参数，
        # 传了会在构造时当场报错。
        extra["stream_usage"] = True

    return init_chat_model(
        model=cfg.model,
        model_provider=provider,
        api_key=_require_key(cfg, secrets),
        # 配置里写 ${VAR:-} 时，空值可能以空串形式传进来（yaml 里带引号写就会）。
        # 空串会被当成真实地址去请求，比不传更糟，所以在这层统一收敛成 None。
        base_url=cfg.base_url or None,
        temperature=cfg.temperature,
        timeout=cfg.timeout_s,
        max_retries=cfg.max_retries,
        **extra,
    )
