"""部署档位：用不可满足的组合尽早失败。"""

from __future__ import annotations

import pytest

from rag.config.schema import Settings
from rag.deployment import DeploymentError, validate_deployment


def _scale_out() -> Settings:
    settings = Settings()
    settings.deployment.profile = "scale-out"
    settings.api.workers = 2
    settings.persistence.backend = "postgres"
    settings.persistence.postgres.url = "postgresql://rag@localhost/rag"
    settings.storage.backend = "minio"
    settings.api.redis.url = "redis://localhost:6379/0"
    return settings


def test_single_node_does_not_require_shared_services() -> None:
    assert validate_deployment(Settings()) == "single-node"


def test_server_requires_postgres_but_only_one_worker() -> None:
    settings = Settings()
    settings.deployment.profile = "server"
    with pytest.raises(DeploymentError, match="PostgreSQL"):
        validate_deployment(settings)

    settings.persistence.backend = "postgres"
    settings.persistence.postgres.url = "postgresql://rag@localhost/rag"
    assert validate_deployment(settings) == "server"

    settings.api.workers = 2
    with pytest.raises(DeploymentError, match="单 worker"):
        validate_deployment(settings)


def test_scale_out_requires_postgres_minio_redis_and_workers() -> None:
    settings = _scale_out()
    assert validate_deployment(settings) == "scale-out"

    settings.storage.backend = "local"
    with pytest.raises(DeploymentError, match="minio"):
        validate_deployment(settings)

    settings.storage.backend = "minio"
    settings.api.redis.url = ""
    with pytest.raises(DeploymentError, match="Redis"):
        validate_deployment(settings)

    settings.api.redis.url = "redis://localhost:6379/0"
    settings.api.workers = 1
    settings.deployment.replicas = 1
    with pytest.raises(DeploymentError, match="总副本数"):
        validate_deployment(settings)

    settings.deployment.replicas = 2
    assert validate_deployment(settings) == "scale-out", "两个单 worker 容器也是有效组合"
