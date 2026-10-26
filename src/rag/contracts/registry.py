"""插件注册表。

配置驱动装配的基础：实现方用装饰器登记名字，装配方按配置里的名字取。
重名直接报错而非静默覆盖——插件名冲突是那种要到生产才暴露的问题。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class Registry:
    """名字 → 实现的映射。"""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, Any] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def deco(obj: T) -> T:
            existing = self._items.get(name)
            if existing is not None:
                raise ValueError(f"{self.kind} 名称冲突: {name!r} 已被 {existing!r} 占用")
            self._items[name] = obj
            return obj

        return deco

    def get(self, name: str) -> Any:
        if name not in self._items:
            raise KeyError(f"未注册的 {self.kind}: {name!r}；已注册: {self.names()}")
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)


PARSERS = Registry("parser")
CHUNKERS = Registry("chunker")
EMBEDDERS = Registry("embedder")
RERANKERS = Registry("reranker")
# **通道类型**注册表：`type` 名 → 工厂。装配由配置里那张表驱动（`retrieval.channels`，
# 每一路 = 一个数据源），所以这里注册的是**适配器**，不是"这一路叫什么名字"。
# 第三方包不用改这里 —— 走 entry points（组名 `rag.channels`），见 `retrieval/plugins.py`。
CHANNEL_TYPES = Registry("通道类型")


def register_parser(name: str) -> Callable[[T], T]:
    return PARSERS.register(name)


def register_chunker(name: str) -> Callable[[T], T]:
    return CHUNKERS.register(name)


def register_embedder(name: str) -> Callable[[T], T]:
    return EMBEDDERS.register(name)


def register_reranker(name: str) -> Callable[[T], T]:
    return RERANKERS.register(name)


def register_channel_type(name: str) -> Callable[[T], T]:
    """把一路**适配器**登记到 `type` 名上（内置的两路在 `retrieval/channels/` 里注册）。"""
    return CHANNEL_TYPES.register(name)
