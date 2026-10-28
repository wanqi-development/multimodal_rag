"""部署档位、profile 状态域派生与启动前硬校验。

三档是一个完整状态域，而不是三个后端开关：

* ``local-lite``：Milvus Lite + SQLite + 本地文件，单进程；
* ``docker-single``：Docker Milvus + 同实例 MinIO 的独立 bucket + SQLite，单进程；
* ``multiworker``：Docker Milvus + 独立 MinIO bucket + PostgreSQL + Redis，多进程。

同一档里的 SQLite、对象 bucket、Milvus collection、PG schema、Redis prefix、缓存、
日志与 workspace 必须属于同一命名空间；跨档不能隐式共享。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

PROFILES = ("local-lite", "docker-single", "multiworker")


class DeploymentError(ValueError):
    """档位声明与后端配置矛盾。"""


def canonical_profile(name: str) -> str:
    return (name or "").strip().lower()


def profile_state_dir(settings: Any, profile: str) -> Path:
    """返回一档的本地状态根。demo 可用 state_dir 显式覆盖。"""
    override = str(getattr(settings.deployment, "state_dir", "") or "").strip()
    if override:
        return Path(override).expanduser()
    return Path(settings.deployment.state_root).expanduser() / profile


def _path(root: Path, *parts: str) -> str:
    value = root.joinpath(*parts).as_posix()
    if root.is_absolute() or value.startswith("./"):
        return value
    return f"./{value}"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DeploymentError(message)


def _set_sqlite(settings: Any) -> None:
    settings.persistence.backend = "sqlite"
    settings.persistence.postgres.url = ""
    settings.documents.backend = "sqlite"
    settings.models.backend = "sqlite"
    settings.observability.logging.postgres = False


def _set_local_storage(settings: Any) -> None:
    settings.storage.backend = "local"
    settings.ingest.uploads.backend = "local"
    settings.ingest.stages.backend = "local"
    settings.ingest.assets.backend = "local"
    settings.ingest.images.cache_backend = "local"
    settings.api.media_backend = "local"


def _set_minio_storage(settings: Any, bucket: str) -> None:
    settings.storage.backend = "minio"
    settings.storage.minio.bucket = bucket
    settings.ingest.uploads.backend = "minio"
    settings.ingest.stages.backend = "minio"
    settings.ingest.assets.backend = "minio"
    settings.ingest.images.cache_backend = "minio"
    settings.api.media_backend = "minio"


def _set_postgres(settings: Any, schema_name: str) -> None:
    settings.persistence.backend = "postgres"
    settings.persistence.postgres.schema_name = schema_name
    settings.documents.backend = "postgres"
    settings.models.backend = "postgres"
    settings.observability.logging.postgres = True


def _set_runtime_paths(settings: Any, root: Path) -> None:
    """把所有本地可变状态绑定到同一个 profile 目录。"""
    settings.deployment.run_dir = _path(root, "run")
    settings.documents.sqlite_path = _path(root, "documents.sqlite3")
    settings.models.sqlite_path = _path(root, "model_settings.sqlite3")
    settings.persistence.chat_sqlite_path = _path(root, "chats.sqlite3")

    settings.ingest.uploads.dir = _path(root, "uploads")
    settings.ingest.stages.dir = _path(root, "stages")
    settings.ingest.assets.dir = _path(root, "assets")
    settings.api.media_dir = _path(root, "media")
    settings.ingest.images.cache_path = _path(root, "cache", "image_descriptions.json")
    settings.storage.minio.cache_dir = _path(root, "cache", "blobs")
    settings.agent.workspace_dir = _path(root, "workspace")

    log_names = {
        "interface": "interface.log",
        "channel": "channel.log",
        "agent_run": "agent.log",
        "agent_llm": "agent.log",
        "agent_tool": "agent.log",
    }
    for kind, item in settings.observability.logging.files.items():
        item.path = _path(root, "logs", log_names.get(kind, f"{kind}.log"))
    settings.observability.logging.system.path = _path(root, "logs", "system.log")


def _profile_collection(settings: Any, profile: str) -> str:
    return {
        "local-lite": settings.deployment.local_lite_collection,
        "docker-single": settings.deployment.docker_single_collection,
        "multiworker": settings.deployment.multiworker_collection,
    }[profile]


def _profile_bucket(settings: Any, profile: str) -> str:
    return {
        "docker-single": settings.deployment.docker_single_bucket,
        "multiworker": settings.deployment.multiworker_bucket,
    }.get(profile, "")


def apply_deployment_profile(settings: Any) -> str:
    """把 profile 的完整状态域写进配置，返回 canonical profile。"""
    profile = canonical_profile(getattr(settings.deployment, "profile", ""))
    _require(
        profile in PROFILES,
        f"未知 deployment.profile={settings.deployment.profile!r}；可选：{', '.join(PROFILES)}",
    )
    settings.deployment.profile = profile
    root = profile_state_dir(settings, profile)
    _set_runtime_paths(settings, root)

    settings.indexing.vector.backend = "milvus-lite" if profile == "local-lite" else "milvus"
    settings.indexing.vector.bm25.analyzer = "jieba" if profile == "local-lite" else "chinese"
    settings.indexing.vector.collection = _profile_collection(settings, profile)
    settings.indexing.vector.versions_collection = ""

    if profile == "local-lite":
        settings.indexing.vector.uri = _path(root, "milvus.db")
        settings.indexing.vector.token = None
        _set_sqlite(settings)
        _set_local_storage(settings)
        settings.storage.minio.bucket = ""
        settings.api.redis.url = ""
        settings.api.workers = 1
        settings.deployment.replicas = 1
        return profile

    if profile == "docker-single":
        _set_sqlite(settings)
        _set_minio_storage(settings, settings.deployment.docker_single_bucket)
        settings.api.redis.url = ""
        settings.api.workers = 1
        settings.deployment.replicas = 1
        return profile

    _set_postgres(settings, settings.deployment.multiworker_pg_schema)
    _set_minio_storage(settings, settings.deployment.multiworker_bucket)
    settings.api.redis.key_prefix = settings.deployment.multiworker_redis_prefix
    return profile


def validate_deployment(settings: Any, *, secrets: Any = None) -> str:
    """校验 profile 与完整状态域；返回 canonical profile。"""
    profile = canonical_profile(getattr(settings.deployment, "profile", ""))
    _require(
        profile in PROFILES,
        f"未知 deployment.profile={settings.deployment.profile!r}；可选：{', '.join(PROFILES)}",
    )
    settings.deployment.profile = profile

    workers = int(getattr(settings.api, "workers", 1) or 1)
    replicas = int(getattr(settings.deployment, "replicas", 1) or 1)
    _require(workers >= 1, "api.workers 必须 >= 1")
    _require(replicas >= 1, "deployment.replicas 必须 >= 1")

    buckets = {
        settings.deployment.docker_single_bucket,
        settings.deployment.multiworker_bucket,
    }
    collections = {
        settings.deployment.local_lite_collection,
        settings.deployment.docker_single_collection,
        settings.deployment.multiworker_collection,
    }
    _require(len(buckets) == 2, "docker-single / multiworker 必须使用不同 MinIO bucket")
    _require(
        settings.deployment.milvus_internal_bucket not in buckets,
        "应用 bucket 不得与 Milvus 内部 bucket 相同",
    )
    _require(len(collections) == 3, "三档必须使用不同 Milvus collection")

    from rag.persistence.backend import resolve_backend
    from rag.storage import resolve_storage_backend

    vector = settings.indexing.vector
    root = profile_state_dir(settings, profile)
    expected_collection = _profile_collection(settings, profile)
    _require(
        vector.collection == expected_collection,
        f"{profile} 的 collection 必须是 {expected_collection!r}，当前 {vector.collection!r}",
    )
    _require(
        settings.documents.sqlite_path == _path(root, "documents.sqlite3"),
        f"{profile} 的 documents SQLite 不在自己的状态目录",
    )
    if profile in {"local-lite", "docker-single"}:
        _require(
            settings.persistence.chat_sqlite_path == _path(root, "chats.sqlite3"),
            f"{profile} 的会话 SQLite 不在自己的状态目录",
        )
        _require(
            settings.models.sqlite_path == _path(root, "model_settings.sqlite3"),
            f"{profile} 的模型 SQLite 不在自己的状态目录",
        )

    if profile == "multiworker":
        _require(settings.persistence.backend == "postgres", "multiworker 必须使用 PostgreSQL")
        _require(
            bool(settings.persistence.postgres.url),
            "multiworker 缺少 PostgreSQL URL（RAG_PG_URL）",
        )
        _require(
            settings.persistence.postgres.schema_name == settings.deployment.multiworker_pg_schema,
            "multiworker 的 PostgreSQL schema 不在自己的命名空间",
        )
    try:
        database = resolve_backend(settings, getattr(settings.documents, "backend", "auto"))
    except ValueError as exc:
        raise DeploymentError(str(exc)) from exc
    storage = resolve_storage_backend(settings, settings.storage.backend)
    domain_storage = {
        "uploads": resolve_storage_backend(settings, settings.ingest.uploads.backend),
        "stages": resolve_storage_backend(settings, settings.ingest.stages.backend),
        "assets": resolve_storage_backend(settings, settings.ingest.assets.backend),
        "media": resolve_storage_backend(settings, settings.api.media_backend),
        "vision_cache": resolve_storage_backend(settings, settings.ingest.images.cache_backend),
    }

    if profile == "local-lite":
        _require(vector.backend == "milvus-lite", "local-lite 必须使用 milvus-lite")
        _require(
            vector.uri == _path(root, "milvus.db"),
            "local-lite 的 Milvus Lite 文件不在自己的状态目录",
        )
        _require(workers == 1 and replicas == 1, "local-lite 只能单 worker、单副本")
        _require(database == "sqlite", "local-lite 的关系库必须是 SQLite")
        _require(storage == "local", "local-lite 的文件后端必须是 local")
        _require(
            all(value == "local" for value in domain_storage.values()),
            f"local-lite 的所有文件域必须是 local；当前={domain_storage}",
        )
        _require(not settings.persistence.postgres.url, "local-lite 不能配置 PostgreSQL")
        _require(not settings.api.redis.url, "local-lite 不能配置 Redis")
        return profile

    expected_bucket = _profile_bucket(settings, profile)
    _require(vector.backend == "milvus", f"{profile} 必须使用 Docker Milvus")
    _require(
        vector.uri.startswith(("http://", "https://")),
        f"{profile} 的 Milvus URI 必须是服务地址",
    )
    _require(database == "sqlite" if profile == "docker-single" else "postgres", "关系库错误")
    _require(storage == "minio", f"{profile} 必须使用 MinIO")
    _require(
        settings.storage.minio.bucket == expected_bucket,
        f"{profile} 必须使用自己的 bucket {expected_bucket!r}",
    )
    _require(
        all(value == "minio" for value in domain_storage.values()),
        f"{profile} 的所有文件域必须是 minio；当前={domain_storage}",
    )
    _require(
        bool(settings.storage.minio.endpoint and settings.storage.minio.bucket),
        f"{profile} 缺少 MinIO endpoint / bucket",
    )
    if profile == "multiworker" and secrets is not None:
        _require(
            bool(secrets.minio_access_key and secrets.minio_secret_key),
            f"{profile} 缺少 MinIO 凭据（MINIO_ACCESS_KEY / MINIO_SECRET_KEY）",
        )

    if profile == "docker-single":
        _require(workers == 1 and replicas == 1, "docker-single 只能单 worker、单副本")
        _require(not settings.persistence.postgres.url, "docker-single 不能配置 PostgreSQL")
        _require(not settings.api.redis.url, "docker-single 不能配置 Redis")
        return profile

    _require(
        settings.api.redis.key_prefix == settings.deployment.multiworker_redis_prefix,
        "multiworker 的 Redis prefix 不在自己的命名空间",
    )
    _require(bool(settings.api.redis.url), "multiworker 缺少 Redis URL（RAG_REDIS_URL）")
    _require(
        max(workers, replicas) >= 2,
        "multiworker 需要总副本数 >= 2：单进程多 worker 设 api.workers，"
        "多个单 worker 容器设 deployment.replicas",
    )
    return profile
