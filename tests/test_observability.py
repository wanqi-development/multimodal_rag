"""观测开关：LangSmith 按配置打开。

要验的是**行为**而不是「函数被调过」：环境变量真的设了没、关着的时候有没有闭嘴、
开关开着但缺 key 时**不能**当成打开（那正是「配置里开了、实际没报」的来源）。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag.config.schema import ObservabilityConfig, Settings
from rag.observability import configure_tracing

TRACING_VARS = (
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_PROJECT",
    "LANGSMITH_TRACING",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """这几个变量一旦漏出来会污染后面所有用例（并真的往 LangSmith 发数据）。"""
    for name in TRACING_VARS:
        monkeypatch.delenv(name, raising=False)


def _settings(*, on: bool, project: str = "multimodal_rag") -> Settings:
    settings = Settings()
    settings.observability = ObservabilityConfig(langsmith=on, project=project)
    return settings


def test_tracing_stays_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "leftover")

    status = configure_tracing(_settings(on=False), SimpleNamespace(langsmith_api_key="k"))

    assert status.startswith("关闭")
    assert "LANGSMITH_TRACING" not in __import__("os").environ


def test_tracing_sets_both_the_old_and_new_variable_names(monkeypatch) -> None:
    """LangChain 认 LANGCHAIN_*，langsmith 认 LANGSMITH_*：只设一套迟早撞上版本差异。"""
    import os

    status = configure_tracing(_settings(on=True), SimpleNamespace(langsmith_api_key="sk-1"))

    assert "已开" in status and "multimodal_rag" in status
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "sk-1"
    assert os.environ["LANGCHAIN_PROJECT"] == "multimodal_rag"


def test_switching_on_without_a_key_is_not_silently_off() -> None:
    """开关开着却没 key：要说清是哪一件缺，而不是安静地不报（最难查的状态）。"""
    status = configure_tracing(_settings(on=True), SimpleNamespace(langsmith_api_key=None))

    assert "LANGSMITH_API_KEY" in status
    assert "没开" in status
