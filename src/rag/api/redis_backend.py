"""Redis 的可选接入（限流计数 + 任务状态）。

**为什么写成「可选 + 永远不抛」**：

* `redis` 客户端不在依赖里（用户环境不一定装）。没装就是没有 Redis —— 与「OCR 产线
  没配地址就不启用」同一个约定，不需要让 API 起不来。
* 这两样东西（配额计数、任务状态）**都不是数据本身**：Redis 挂了、慢了、超时了，
  正确的反应是**降级**（退回进程内）并让 `/health` 标红，而不是让整个问答挂掉。
  所以这层的每个方法失败都只返回 None/False，**从不抛异常**。

代价说清楚：降级期间限流计数与任务状态是**进程内**的 —— 多 worker 时各算各的、
重启就丢。单机开发场景可以接受，`/health` 里会明确写出来。
"""

from __future__ import annotations

import contextlib
import json
from typing import Any

from rag.config.schema import RedisConfig


class RedisBackend:
    """一个瘦封装：只暴露我们真正用到的四个动作。"""

    def __init__(self, cfg: RedisConfig, password: str | None = None) -> None:
        self._cfg = cfg
        self._password = password or None
        self._client: Any = None
        # 为什么不可用 —— `/health` 要原样报出来（「没配」「没装客户端」「连不上」是三件事）
        self.reason: str = "未配置" if not cfg.url else ""

    @property
    def configured(self) -> bool:
        return bool(self._cfg.url)

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def prefix(self) -> str:
        return self._cfg.key_prefix

    async def connect(self) -> bool:
        """lifespan 里调一次。失败**不抛**：服务要以降级状态起来。"""
        if not self.configured:
            return False
        try:
            from redis.asyncio import Redis
        except ImportError:
            self.reason = "未安装 redis 客户端（pip install redis）"
            return False
        try:
            client = Redis.from_url(
                self._cfg.url,
                password=self._password,
                socket_timeout=self._cfg.timeout_s,
                socket_connect_timeout=self._cfg.timeout_s,
                decode_responses=True,
            )
            await client.ping()
        except Exception as exc:  # noqa: BLE001 —— 连不上就降级，见模块说明
            self.reason = f"连不上：{type(exc).__name__}: {exc}"
            return False
        self._client = client
        self.reason = ""
        return True

    async def ping(self) -> bool:
        if self._client is None:
            return False
        try:
            await self._client.ping()
            return True
        except Exception:  # noqa: BLE001
            return False

    async def incr(self, key: str, *, ttl_s: int) -> int | None:
        """计数 +1 并确保有 TTL（限流窗口用）。失败返回 None。"""
        if self._client is None:
            return None
        try:
            # INCR 与 EXPIRE 不是原子的，但窗口键带时间戳、过期只是兜底：
            # 最坏情况是某个键没设上 TTL 而永久累加 —— 所以用 `expire` 的 NX 语义
            value = await self._client.incr(key)
            if value == 1:  # 第一次出现才设过期，避免每次请求都刷新 TTL
                await self._client.expire(key, ttl_s)
            return int(value)
        except Exception:  # noqa: BLE001
            return None

    async def put(self, key: str, value: dict[str, Any], *, ttl_s: int) -> bool:
        if self._client is None:
            return False
        try:
            await self._client.set(key, json.dumps(value, ensure_ascii=False), ex=ttl_s)
            return True
        except Exception:  # noqa: BLE001
            return False

    async def get(self, key: str) -> dict[str, Any] | None:
        if self._client is None:
            return None
        try:
            raw = await self._client.get(key)
        except Exception:  # noqa: BLE001
            return None
        if raw is None:
            return None
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    async def zadd(self, key: str, member: str, *, score: float, ttl_s: int) -> bool:
        if self._client is None:
            return False
        try:
            await self._client.zadd(key, {member: score})
            await self._client.expire(key, ttl_s)
            return True
        except Exception:  # noqa: BLE001
            return False

    async def zrevrange(self, key: str, *, start: int, end: int) -> list[str]:
        if self._client is None:
            return []
        try:
            rows = await self._client.zrevrange(key, start, end)
            return [str(row) for row in rows]
        except Exception:  # noqa: BLE001
            return []

    async def set_if_absent(self, key: str, value: str, *, ttl_s: int) -> bool:
        if self._client is None:
            return False
        try:
            return bool(await self._client.set(key, value, nx=True, ex=ttl_s))
        except Exception:  # noqa: BLE001
            return False

    async def delete_if_value(self, key: str, value: str) -> bool:
        if self._client is None:
            return False
        try:
            deleted = await self._client.eval(
                "if redis.call('get', KEYS[1]) == ARGV[1] "
                "then return redis.call('del', KEYS[1]) else return 0 end",
                1,
                key,
                value,
            )
            return bool(deleted)
        except Exception:  # noqa: BLE001
            return False

    async def close(self) -> None:
        if self._client is not None:
            # 关连接失败没什么可做的（进程都要退了），压掉就好
            with contextlib.suppress(Exception):
                await self._client.aclose()
            self._client = None
