"""部署档位：用不可满足的组合尽早失败。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag.config.schema import Settings, VectorIndexConfig
from rag.config.settings import load_settings
from rag.deployment import (
    DeploymentError,
    apply_deployment_profile,
    validate_deployment,
)


def _multiworker() -> Settings:
    settings = Settings()
    settings.deployment.profile = "multiworker"
    settings.api.workers = 2
    settings.persistence.postgres.url = "postgresql://rag@localhost/rag"
    settings.api.redis.url = "redis://localhost:6379/0"
    settings.storage.minio.endpoint = "http://localhost:9000"
    apply_deployment_profile(settings)
    return settings


def test_legacy_profile_names_are_rejected() -> None:
    for name in ("single-node", "scale-out"):
        settings = Settings()
        settings.deployment.profile = name
        with pytest.raises(DeploymentError, match="未知 deployment.profile"):
            apply_deployment_profile(settings)


def test_milvus_lite_switches_chinese_analyzer_to_supported_jieba() -> None:
    cfg = VectorIndexConfig(backend="milvus-lite", uri="./data/local-lite/milvus.db")
    assert cfg.bm25.analyzer == "jieba"


def test_one_environment_switch_selects_the_local_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_DEPLOYMENT_PROFILE", "local-lite")
    settings = load_settings("dev")

    assert settings.deployment.profile == "local-lite"
    assert settings.indexing.vector.backend == "milvus-lite"
    assert settings.indexing.vector.uri == "./data/local-lite/milvus.db"
    assert settings.persistence.backend == "sqlite"
    assert settings.storage.backend == "local"
    assert settings.api.workers == 1


def test_demo_profile_is_fully_embedded() -> None:
    settings = load_settings("demo")

    assert settings.deployment.profile == "local-lite"
    assert settings.indexing.vector.backend == "milvus-lite"
    assert settings.indexing.vector.uri == "./data/demo/milvus.db"
    assert settings.indexing.vector.collection == "rag_demo_qwen3_v1"
    assert settings.storage.backend == "local"
    assert settings.persistence.backend == "sqlite"
    assert settings.persistence.chat_sqlite_path == "./data/demo/chats.sqlite3"


def test_docker_single_uses_minio_and_sqlite() -> None:
    settings = Settings()
    settings.deployment.profile = "docker-single"
    settings.storage.minio.endpoint = "http://localhost:9000"
    apply_deployment_profile(settings)

    assert settings.deployment.profile == "docker-single"
    assert settings.indexing.vector.backend == "milvus"
    assert settings.persistence.backend == "sqlite"
    assert settings.storage.backend == "minio"
    assert settings.storage.minio.bucket == "rag-docker-single"
    assert settings.api.workers == 1
    assert validate_deployment(settings) == "docker-single"


def test_local_lite_forces_one_process_and_local_files() -> None:
    settings = Settings()
    settings.deployment.profile = "local-lite"
    settings.persistence.postgres.url = "postgresql://must-be-ignored"
    settings.api.redis.url = "redis://must-be-ignored"
    settings.api.workers = 8
    settings.storage.backend = "minio"

    assert apply_deployment_profile(settings) == "local-lite"
    assert settings.indexing.vector.backend == "milvus-lite"
    assert settings.indexing.vector.uri == "./data/local-lite/milvus.db"
    assert settings.indexing.vector.collection == "rag_local_lite_v1"
    assert settings.indexing.vector.bm25.analyzer == "jieba"
    assert settings.persistence.backend == "sqlite"
    assert settings.storage.backend == "local"
    assert settings.api.redis.url == ""
    assert settings.api.workers == 1
    assert validate_deployment(settings) == "local-lite"


def test_local_lite_rejects_a_remote_vector_uri() -> None:
    settings = Settings()
    settings.deployment.profile = "local-lite"
    apply_deployment_profile(settings)
    settings.indexing.vector.uri = "http://localhost:19530"

    with pytest.raises(DeploymentError, match="自己的状态目录"):
        validate_deployment(settings)


def test_multiworker_requires_all_shared_backends() -> None:
    settings = _multiworker()
    assert validate_deployment(settings) == "multiworker"
    assert settings.persistence.backend == "postgres"
    assert settings.storage.backend == "minio"
    assert settings.api.workers == 2

    settings.persistence.postgres.url = ""
    with pytest.raises(DeploymentError, match="PostgreSQL URL"):
        validate_deployment(settings)

    settings = _multiworker()
    settings.api.redis.url = ""
    with pytest.raises(DeploymentError, match="Redis URL"):
        validate_deployment(settings)

    settings = _multiworker()
    settings.storage.minio.bucket = ""
    with pytest.raises(DeploymentError, match="自己的 bucket"):
        validate_deployment(settings)

    settings = _multiworker()
    secrets = SimpleNamespace(minio_access_key=None, minio_secret_key=None)
    with pytest.raises(DeploymentError, match="MinIO 凭据"):
        validate_deployment(settings, secrets=secrets)


def test_multiworker_requires_at_least_two_replicas_or_workers() -> None:
    settings = _multiworker()
    settings.api.workers = 1
    settings.deployment.replicas = 1

    with pytest.raises(DeploymentError, match="总副本数"):
        validate_deployment(settings)

    settings.deployment.replicas = 2
    assert validate_deployment(settings) == "multiworker"


def test_local_lite_cannot_claim_multiple_workers() -> None:
    settings = Settings()
    settings.deployment.profile = "local-lite"
    apply_deployment_profile(settings)
    settings.api.workers = 2

    with pytest.raises(DeploymentError, match="单 worker"):
        validate_deployment(settings)
