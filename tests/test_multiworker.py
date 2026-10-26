"""多 worker 的共享状态：任务列表、灌库槽位、VLM 缓存。"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from rag.api.ingest_gate import IngestGate
from rag.api.jobs import JobStore
from rag.api.redis_backend import RedisBackend
from rag.ingest.images import _BlobDescriptionCache
from rag.storage import BlobNotFound


class _SharedRedis:
    available = True
    prefix = "rag:"

    def __init__(self) -> None:
        self.values: dict[str, dict[str, Any]] = {}
        self.sets: dict[str, str] = {}
        self.zsets: dict[str, dict[str, float]] = {}

    async def put(self, key: str, value: dict[str, Any], *, ttl_s: int) -> bool:
        self.values[key] = value
        return True

    async def get(self, key: str) -> dict[str, Any] | None:
        return self.values.get(key)

    async def zadd(self, key: str, member: str, *, score: float, ttl_s: int) -> bool:
        self.zsets.setdefault(key, {})[member] = score
        return True

    async def zrevrange(self, key: str, *, start: int, end: int) -> list[str]:
        rows = [
            member
            for member, _ in sorted(
                self.zsets.get(key, {}).items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]
        return rows[start : end + 1]

    async def set_if_absent(self, key: str, value: str, *, ttl_s: int) -> bool:
        if key in self.sets:
            return False
        self.sets[key] = value
        return True

    async def delete_if_value(self, key: str, value: str) -> bool:
        if self.sets.get(key) != value:
            return False
        del self.sets[key]
        return True

    async def ping(self) -> bool:
        return True


class _Blob:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    def get_text(self, key: str) -> str:
        if key not in self.data:
            raise BlobNotFound(key)
        return self.data[key]

    def put_text(self, key: str, text: str) -> None:
        self.data[key] = text


def test_job_list_sees_jobs_saved_by_another_worker() -> None:
    redis = _SharedRedis()
    first = JobStore(redis)  # type: ignore[arg-type]
    second = JobStore(redis)  # type: ignore[arg-type]
    job = second.spawn(kind="ingest", request={"paths": ["a.md"]})
    job.created_at = int(time.time())

    asyncio.run(second.save(job))
    rows = asyncio.run(first.list())

    assert [row.id for row in rows] == [job.id]


def test_redis_ingest_gate_is_shared_across_instances() -> None:
    redis = _SharedRedis()
    first = IngestGate(redis, limit=1, lease_s=60)  # type: ignore[arg-type]
    second = IngestGate(redis, limit=1, lease_s=60)  # type: ignore[arg-type]

    async def scenario() -> None:
        held = asyncio.Event()
        release = asyncio.Event()

        async def occupy() -> None:
            async with first.hold():
                held.set()
                await release.wait()

        task = asyncio.create_task(occupy())
        await held.wait()

        entered = asyncio.Event()

        async def wait_for_slot() -> None:
            async with second.hold():
                entered.set()

        waiter = asyncio.create_task(wait_for_slot())
        await asyncio.sleep(0.05)
        assert not entered.is_set(), "同一时刻只允许一个跨 worker 灌库任务"

        release.set()
        await task
        await asyncio.wait_for(waiter, timeout=2)
        assert entered.is_set()
        assert first.backend == second.backend == "redis"

    asyncio.run(scenario())


def test_scale_out_gate_refuses_to_fall_back_without_redis() -> None:
    redis = RedisBackend.__new__(RedisBackend)
    redis._client = None
    redis._cfg = type("Cfg", (), {"url": "", "key_prefix": "rag:"})()
    gate = IngestGate(redis, limit=1, lease_s=60, require_shared=True)

    async def try_enter() -> None:
        async with gate.hold():
            pass

    with pytest.raises(RuntimeError, match="Redis"):
        asyncio.run(try_enter())


def test_blob_description_cache_is_shared_by_key() -> None:
    blob = _Blob()
    first = _BlobDescriptionCache(blob, "image-cache")  # type: ignore[arg-type]
    second = _BlobDescriptionCache(blob, "image-cache")  # type: ignore[arg-type]
    key = "abc123:model@prompt"

    first.put(key, "共享描述")

    assert second.get(key) == "共享描述"
    assert "image-cache/abc123/model@prompt.json" in blob.data
