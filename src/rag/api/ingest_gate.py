"""灌库并发闸门：单机用 asyncio，多 worker 用 Redis 槽位租约。"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator

from rag.api.redis_backend import RedisBackend


class IngestGate:
    def __init__(
        self,
        redis: RedisBackend,
        *,
        limit: int,
        lease_s: int,
        require_shared: bool = False,
    ) -> None:
        self._redis = redis
        self._limit = max(int(limit), 1)
        self._lease_s = max(int(lease_s), 60)
        self._require_shared = require_shared
        self._local = asyncio.Semaphore(self._limit)

    @property
    def backend(self) -> str:
        return "redis" if self._redis.available else "local"

    @contextlib.asynccontextmanager
    async def hold(self) -> AsyncIterator[None]:
        if not self._redis.available:
            if self._require_shared:
                raise RuntimeError("multiworker 灌库闸门需要 Redis，但 Redis 当前不可用")
            async with self._local:
                yield
            return

        token = uuid.uuid4().hex
        acquired_key = ""
        while not acquired_key:
            for index in range(self._limit):
                key = f"{self._redis.prefix}ingest-slot:{index}"
                if await self._redis.set_if_absent(key, token, ttl_s=self._lease_s):
                    acquired_key = key
                    break
            if acquired_key:
                break
            if not await self._redis.ping():
                if self._require_shared:
                    raise RuntimeError("multiworker 灌库闸门失去 Redis，拒绝退回单进程")
                async with self._local:
                    yield
                return
            await asyncio.sleep(0.5)

        try:
            yield
        finally:
            await self._redis.delete_if_value(acquired_key, token)
