"""灌库任务表：`POST /ingest` 提交后靠它查进度。

**为什么要有它**：OCR/VLM 是几十分钟级，HTTP 请求不能等（等就是超时）。
所以 `/ingest` 只登记一个任务就返回 202，真正的活在后台跑。

存两处：

* **进程内字典**（永远写）：`/jobs` 读它最快，也是 Redis 挂掉时的兜底；
* **Redis**（装了并连上就写）：跨 worker、跨重启能看到 —— 这是接 Redis 的主要理由。

读的优先级是「内存 → Redis」：内存里有就是最新的（同一个进程写的），没有再去 Redis 找
（可能是别的 worker 写的，也可能是本进程重启前的）。
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from rag.api.redis_backend import RedisBackend

JobStatus = Literal["queued", "running", "done", "failed"]

# 任务记录保留 1 天：够看「昨天那次灌库怎么了」，又不会无限攒
JOB_TTL_S = 86400


class Job(BaseModel):
    id: str
    kind: str
    status: JobStatus = "queued"
    # 请求原样留一份：排查「这次到底跑的是什么参数」时不用去翻日志
    request: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    created_at: int = 0
    updated_at: int = 0


class JobStore:
    def __init__(self, redis: RedisBackend) -> None:
        self._redis = redis
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def new_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def spawn(self, *, kind: str, request: dict[str, Any]) -> Job:
        """登记一个新任务（还没跑）。id 用随机 hex —— 不暴露「第几个任务」这种信息。"""
        now = int(time.time())
        return Job(
            id=self.new_id(),
            kind=kind,
            status="queued",
            request=request,
            created_at=now,
            updated_at=now,
        )

    async def save(self, job: Job) -> None:
        job.updated_at = int(time.time())
        with self._lock:
            self._jobs[job.id] = job
        stored = await self._redis.put(
            f"{self._redis.prefix}job:{job.id}",
            job.model_dump(mode="json"),
            ttl_s=JOB_TTL_S,
        )
        if stored:
            await self._redis.zadd(
                f"{self._redis.prefix}job:index",
                job.id,
                score=float(job.updated_at),
                ttl_s=JOB_TTL_S * 2,
            )

    async def get(self, job_id: str) -> Job | None:
        with self._lock:
            cached = self._jobs.get(job_id)
        if cached is not None:
            return cached
        payload = await self._redis.get(f"{self._redis.prefix}job:{job_id}")
        return Job.model_validate(payload) if payload else None

    async def list(self, *, limit: int = 20) -> list[Job]:
        """最近的任务，新的在前；Redis 可用时合并其他 worker 的任务。"""
        with self._lock:
            local = list(self._jobs.values())
        if not self._redis.available:
            return sorted(
                local,
                key=lambda job: (job.created_at, job.updated_at),
                reverse=True,
            )[:limit]

        ids = await self._redis.zrevrange(
            f"{self._redis.prefix}job:index",
            start=0,
            end=max(limit * 3, limit) - 1,
        )
        found: dict[str, Job] = {job.id: job for job in local}
        for job_id in ids:
            if job_id in found:
                continue
            payload = await self._redis.get(f"{self._redis.prefix}job:{job_id}")
            if payload:
                found[job_id] = Job.model_validate(payload)
        return sorted(
            found.values(),
            key=lambda job: (job.created_at, job.updated_at),
            reverse=True,
        )[:limit]
