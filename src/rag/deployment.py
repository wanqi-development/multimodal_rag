"""部署档位与共享后端校验。"""

from __future__ import annotations

from typing import Any

PROFILES = ("single-node", "server", "scale-out")


class DeploymentError(ValueError):
    """档位声明与后端配置矛盾。"""


def validate_deployment(settings: Any) -> str:
    """返回规范 profile；不合法时抛带修法的错误。"""
    profile = str(getattr(settings.deployment, "profile", "single-node") or "").strip().lower()
    if profile not in PROFILES:
        raise DeploymentError(f"未知 deployment.profile={profile!r}；可选：{', '.join(PROFILES)}")

    workers = int(getattr(settings.api, "workers", 1) or 1)
    if workers < 1:
        raise DeploymentError("api.workers 必须 >= 1")
    replicas = int(getattr(settings.deployment, "replicas", 1) or 1)
    if replicas < 1:
        raise DeploymentError("deployment.replicas 必须 >= 1")

    if profile == "single-node":
        return profile

    from rag.persistence.backend import resolve_backend

    database = resolve_backend(settings, getattr(settings.documents, "backend", "auto"))
    if database != "postgres":
        raise DeploymentError(
            f"deployment.profile={profile} 需要 PostgreSQL；当前关系库={database}。"
            "配置 RAG_PG_URL，并把 RAG_DB_BACKEND=postgres"
        )

    if profile == "server":
        if workers != 1:
            raise DeploymentError("server profile 只支持单 worker；多 worker 请切到 scale-out")
        return profile

    from rag.storage import resolve_storage_backend

    storage = resolve_storage_backend(settings, settings.storage.backend)
    if storage != "minio":
        raise DeploymentError(
            "deployment.profile=scale-out 需要共享文件存储；把 RAG_STORAGE_BACKEND 设为 minio"
        )
    if not settings.api.redis.url:
        raise DeploymentError(
            "deployment.profile=scale-out 需要 Redis（RAG_REDIS_URL）；"
            "否则任务列表与灌库闸门会退回单进程"
        )
    if max(workers, replicas) < 2:
        raise DeploymentError(
            "deployment.profile=scale-out 需要总副本数 >= 2："
            "单进程多 worker 设 api.workers，两个容器各一个 worker 设 deployment.replicas=2"
        )
    return profile
