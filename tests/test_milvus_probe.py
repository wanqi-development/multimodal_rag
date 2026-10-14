"""Milvus 探活（`MilvusIndex.ping`）：**接不住就得在超时内抛出来**。

为什么值得一条用例：Milvus 停掉时 pymilvus 的重连会空转很久（实测 `rag ingest` 烧了
三十几分钟 CPU、数据一行没进，§16）。探活是那条路上唯一能把等待压到几秒的东西。
"""

from __future__ import annotations

import asyncio
import time

import pytest

from rag.config.schema import VectorIndexConfig
from rag.indexing.milvus import MilvusIndex


def _index() -> MilvusIndex:
    return MilvusIndex(VectorIndexConfig(collection="c1"), dim=4)


def _with_client(index: MilvusIndex, client: object) -> MilvusIndex:
    """`_client` 是 cached_property：往实例字典里塞就直接生效（不会真去连 Milvus）。"""
    index.__dict__["_client"] = client
    return index


class _Slow:
    """连不上时的样子：调用不报错，就是一直不返回。"""

    def has_collection(self, name: str) -> bool:
        time.sleep(1.0)
        return True


class _Ok:
    def __init__(self) -> None:
        self.asked: list[str] = []

    def has_collection(self, name: str) -> bool:
        self.asked.append(name)
        return True


def test_ping_gives_up_within_the_timeout() -> None:
    index = _with_client(_index(), _Slow())

    with pytest.raises(TimeoutError):
        asyncio.run(index.ping(timeout_s=0.05))


def test_ping_asks_the_cheapest_question_when_the_server_answers() -> None:
    """探活用 `has_collection`（一次读 RPC）——不是建表、不是 load。"""
    client = _Ok()
    index = _with_client(_index(), client)

    asyncio.run(index.ping(timeout_s=5))

    assert client.asked == ["c1"]
