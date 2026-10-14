"""chat provider 选型：改配置里的 provider/model/base_url 就换供应商，不改代码。

不碰网络 —— 构造 client 不发请求，这里只验证我们交给 LangChain 的参数对不对。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rag.config.schema import ChatProviderConfig, ProvidersConfig, Settings
from rag.config.settings import Secrets
from rag.providers import chat as chat_module
from rag.providers.chat import build_chat


@pytest.fixture(autouse=True)
def _empty_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Secrets 的 env_file 是相对 cwd 的 ".env"：留在项目根会把本机真实 key 读进来，
    让「缺 key」这类用例失去意义。
    """
    monkeypatch.chdir(tmp_path)


class _Recorder:
    """替下 init_chat_model，把参数记下来。"""

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def __call__(self, **kwargs: Any) -> object:
        self.kwargs = kwargs
        return object()


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    rec = _Recorder()
    monkeypatch.setattr(chat_module, "init_chat_model", rec)
    return rec


def _settings(**chat: ChatProviderConfig) -> Settings:
    return Settings(providers=ProvidersConfig(chat=chat))


# ---- OpenAI 兼容通道 ----


def test_openai_compatible_provider_goes_through_the_openai_channel(
    recorded: _Recorder,
) -> None:
    """deepseek 之类没有独立集成包，靠 openai 通道 + 自己的 base_url。"""
    settings = _settings(
        default=ChatProviderConfig(
            provider="deepseek", model="deepseek-chat", base_url="https://api.deepseek.com/v1"
        )
    )
    build_chat(settings, Secrets(deepseek_api_key="k"))

    assert recorded.kwargs["model_provider"] == "openai"
    assert recorded.kwargs["model"] == "deepseek-chat"
    assert recorded.kwargs["base_url"] == "https://api.deepseek.com/v1"
    assert recorded.kwargs["api_key"] == "k"


def test_compatible_provider_without_base_url_is_rejected() -> None:
    """没有 base_url 就不知道该往哪发，这里拦住比等远端超时强。"""
    settings = _settings(
        default=ChatProviderConfig(provider="deepseek", model="deepseek-chat", base_url=None)
    )

    with pytest.raises(ValueError, match="base_url"):
        build_chat(settings, Secrets(deepseek_api_key="k"))


# ---- 原生集成通道 ----


def test_blank_base_url_is_passed_as_none(recorded: _Recorder) -> None:
    """yaml 里的 ${OPENAI_BASE_URL:-} 解析出来可能是空串。

    空串会被原生集成当成真实地址去请求，比不传更糟 —— 所以这一层必须收敛成 None，
    否则「留空即用官方地址」这条约定是假的。
    """
    settings = _settings(
        openai=ChatProviderConfig(provider="openai", model="gpt-4o-mini", base_url="")
    )
    build_chat(settings, Secrets(openai_api_key="k"), name="openai")

    assert recorded.kwargs["model_provider"] == "openai"
    assert recorded.kwargs["base_url"] is None


def test_native_provider_keeps_the_configured_gateway(recorded: _Recorder) -> None:
    settings = _settings(
        openai=ChatProviderConfig(
            provider="openai", model="gpt-4o-mini", base_url="https://gw.internal/v1"
        )
    )
    build_chat(settings, Secrets(openai_api_key="k"), name="openai")

    assert recorded.kwargs["base_url"] == "https://gw.internal/v1"


# ---- 选错名字时要说清楚 ----


def test_missing_key_names_the_variable_to_fill() -> None:
    settings = _settings(
        default=ChatProviderConfig(
            provider="deepseek", model="deepseek-chat", base_url="https://api.deepseek.com/v1"
        )
    )

    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        build_chat(settings, Secrets())


def test_unknown_chat_name_lists_what_is_available() -> None:
    settings = _settings(
        default=ChatProviderConfig(
            provider="deepseek", model="deepseek-chat", base_url="https://api.deepseek.com/v1"
        )
    )

    with pytest.raises(KeyError, match="default"):
        build_chat(settings, Secrets(deepseek_api_key="k"), name="fast")
