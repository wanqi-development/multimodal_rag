"""限流：每分钟每身份 N 次。

两条实现，接口一样：

* **有 Redis**：`INCR <前缀>rate:<身份>:<窗口>` + 首次设 60 秒 TTL。跨 worker 共享，
  这是选它而不是进程内计数的主要原因。
* **没有 Redis**：进程内字典。多 worker 时各算各的（实际额度 = N × worker 数），
  这一点在 `/health` 里会写明 —— 与其假装限住了，不如说清边界。

**身份不落明文 key**：API key 经过 sha1 取前 12 位。Redis 可能被别人看，
把用户的令牌原文当键名不合适。
"""

from __future__ import annotations

import hashlib
import threading
import time

from rag.api.redis_backend import RedisBackend

WINDOW_S = 60


def identity_of(api_key: str | None, client_host: str | None) -> str:
    """限流的身份：优先用 API key（同一把 key 的多个客户端共享额度），否则用来源 IP。"""
    if api_key:
        digest = hashlib.sha1(api_key.encode("utf-8")).hexdigest()[:12]
        return f"key:{digest}"
    return f"ip:{client_host or 'unknown'}"


class RateLimiter:
    def __init__(self, per_minute: int, redis: RedisBackend) -> None:
        self._per_minute = per_minute
        self._redis = redis
        self._hits: dict[tuple[str, int], int] = {}
        self._lock = threading.Lock()

    async def hit(self, identity: str) -> tuple[bool, int]:
        """记一次请求，返回 ``(放行?, 本窗口次数)``。``per_minute <= 0`` 表示不限。"""
        if self._per_minute <= 0:
            return True, 0
        window = int(time.time() // WINDOW_S)
        count = await self._redis.incr(
            f"{self._redis.prefix}rate:{identity}:{window}", ttl_s=WINDOW_S
        )
        if count is None:  # Redis 不可用 → 降级到进程内
            count = self._local(identity, window)
        return count <= self._per_minute, count

    def _local(self, identity: str, window: int) -> int:
        """进程内计数。顺手清掉过期的窗口，免得长跑的服务把它攒成内存泄漏。"""
        with self._lock:
            for stale in [key for key in self._hits if key[1] != window]:
                del self._hits[stale]
            key = (identity, window)
            self._hits[key] = self._hits.get(key, 0) + 1
            return self._hits[key]
