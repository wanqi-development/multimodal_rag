"""通道类型的解析 = 插件化（2026-10-08）：内置注册表 / entry points / 点路径。

使用者点名要的是「Java 里加载不同 db 驱动」那一套：**接口在核心、实现在包里、
靠名字在运行时对上**。这里把三条来源与失败路径都钉住 —— 少一路数据源是静默故障，
那种问题最难查。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from rag.retrieval import plugins


def test_the_only_builtin_type_is_milvus() -> None:
    """核心的内置通道**只剩 `milvus`**（图谱那一路 2026-10-08 搬进了插件包，见 docs/plugins.md）。

    这条原先叫 `test_the_two_builtin_types_resolve`，同时断言 `neo4j` 也能解析 ——
    那是「本机装过 `pip install -e plugins/rag-channel-neo4j`」这个前提带来的巧合：
    核心的内置表里早就没有它了。CI（Linux、没装插件）第一次跑就把这个巧合抖了出来。
    `neo4j` 那半边由 `tests/test_graph_store.py` 覆盖，那边带着 `importorskip`。
    """
    assert callable(plugins.resolve_channel_type("milvus"))


def test_a_dotted_path_works_without_installing_anything() -> None:
    """开发期的逃生舱：`type` 直接写点路径就能装配（免安装、免改 pyproject）。"""
    factory = plugins.resolve_channel_type("rag.retrieval.channels.milvus_hybrid:build")

    assert callable(factory)
    assert factory is not None


def test_entry_points_are_discovered_under_our_group(monkeypatch: pytest.MonkeyPatch) -> None:
    """装好的插件包按 entry point 名字解析 —— `ServiceLoader` 那一套。"""
    seen: list[str | None] = []

    class _Meta:
        @staticmethod
        def entry_points(*, group: str | None = None) -> list[Any]:
            seen.append(group)
            return [
                SimpleNamespace(name="my_es", value="rag.retrieval.channels.milvus_hybrid:build")
            ]

    monkeypatch.setattr(plugins, "metadata", _Meta)
    plugins.resolve_channel_type.cache_clear()
    try:
        factory = plugins.resolve_channel_type("my_es")
    finally:
        plugins.resolve_channel_type.cache_clear()

    assert callable(factory)
    assert seen == [plugins.ENTRY_POINT_GROUP]


def test_an_unknown_type_lists_what_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """找不到就要说清**有什么**：内置的名单 + 入口点里声明的名单 + 怎么装。"""
    monkeypatch.setattr(plugins, "declare_entry_points", lambda: {"my_es": "x:y"})
    plugins.resolve_channel_type.cache_clear()
    try:
        with pytest.raises(KeyError) as caught:
            plugins.resolve_channel_type("nope")
    finally:
        plugins.resolve_channel_type.cache_clear()

    message = str(caught.value)
    assert "milvus" in message and "my_es" in message and "entry-points" in message


def test_a_broken_target_says_which_part_is_wrong() -> None:
    with pytest.raises(ValueError, match="包.模块:函数"):
        plugins.load_target("没有冒号的写法")
    with pytest.raises(ImportError, match="模块导不进来"):
        plugins.load_target("no_such_package_at_all.channel:build")
    with pytest.raises(AttributeError, match="里没有"):
        plugins.load_target("rag.retrieval.plugins:没有这个名字")
