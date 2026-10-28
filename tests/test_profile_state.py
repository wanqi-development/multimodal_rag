"""profile 状态域隔离与 marker 校验。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import rag.profile_state as state_module
from rag.config.schema import Settings
from rag.deployment import DeploymentError, apply_deployment_profile, validate_deployment
from rag.profile_state import (
    ProfileStateError,
    check_local_marker,
    ensure_local_marker,
    ensure_profile_state,
)


def _local(tmp_path: Path) -> Settings:
    settings = Settings()
    settings.deployment.profile = "local-lite"
    settings.deployment.state_dir = str(tmp_path / "local-lite")
    apply_deployment_profile(settings)
    return settings


def _docker(tmp_path: Path) -> Settings:
    settings = Settings()
    settings.deployment.profile = "docker-single"
    settings.deployment.state_dir = str(tmp_path / "docker-single")
    settings.storage.minio.endpoint = "localhost:9000"
    apply_deployment_profile(settings)
    return settings


def _multi(tmp_path: Path) -> Settings:
    settings = Settings()
    settings.deployment.profile = "multiworker"
    settings.deployment.state_root = str(tmp_path)
    settings.api.workers = 2
    settings.persistence.postgres.url = "postgresql://rag@localhost/rag"
    settings.api.redis.url = "redis://localhost:6379/0"
    settings.storage.minio.endpoint = "localhost:9000"
    apply_deployment_profile(settings)
    return settings


def test_three_profiles_have_distinct_state_namespaces(tmp_path: Path) -> None:
    local = _local(tmp_path)
    docker = _docker(tmp_path)
    multi = _multi(tmp_path)

    sqlite_paths = {
        local.documents.sqlite_path,
        docker.documents.sqlite_path,
        multi.documents.sqlite_path,
    }
    assert len(sqlite_paths) == 3
    assert local.persistence.chat_sqlite_path != docker.persistence.chat_sqlite_path
    assert local.models.sqlite_path != docker.models.sqlite_path
    assert {
        local.indexing.vector.collection,
        docker.indexing.vector.collection,
        multi.indexing.vector.collection,
    } == {
        "rag_local_lite_v1",
        "rag_docker_single_v1",
        "rag_multiworker_v1",
    }
    assert local.storage.minio.bucket == ""
    assert docker.storage.minio.bucket == "rag-docker-single"
    assert multi.storage.minio.bucket == "rag-multiworker"
    assert (
        len(
            {
                local.deployment.run_dir,
                docker.deployment.run_dir,
                multi.deployment.run_dir,
            }
        )
        == 3
    )
    assert (
        len(
            {
                str(Path(local.observability.logging.system.path).parent),
                str(Path(docker.observability.logging.system.path).parent),
                str(Path(multi.observability.logging.system.path).parent),
            }
        )
        == 3
    )


def test_profile_validation_rejects_shared_bucket(tmp_path: Path) -> None:
    settings = _docker(tmp_path)
    settings.deployment.multiworker_bucket = settings.deployment.docker_single_bucket

    with pytest.raises(DeploymentError, match="不同 MinIO bucket"):
        validate_deployment(settings)


def test_profile_validation_rejects_milvus_internal_bucket(tmp_path: Path) -> None:
    settings = _docker(tmp_path)
    settings.deployment.docker_single_bucket = settings.deployment.milvus_internal_bucket
    apply_deployment_profile(settings)

    with pytest.raises(DeploymentError, match="Milvus 内部 bucket"):
        validate_deployment(settings)


def test_local_marker_is_created_and_checked(tmp_path: Path) -> None:
    settings = _local(tmp_path)

    marker = ensure_local_marker(settings)
    assert marker["profile"] == "local-lite"
    assert marker["namespace_id"]
    assert check_local_marker(settings) == marker

    path = Path(settings.deployment.state_dir) / "state.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["vector_collection"] = "wrong"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProfileStateError, match="标记不匹配"):
        check_local_marker(settings)


def test_bucket_marker_is_created_and_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import minio.error

    class _Missing(Exception):
        code = "NoSuchKey"

    class _Client:
        def __init__(self) -> None:
            self.objects: dict[tuple[str, str], bytes] = {}
            self.buckets: set[str] = set()

        def bucket_exists(self, bucket: str) -> bool:
            return bucket in self.buckets

        def make_bucket(self, bucket: str, location: str | None = None) -> None:
            self.buckets.add(bucket)

        def get_object(self, bucket: str, key: str):
            if (bucket, key) not in self.objects:
                raise _Missing()

            class _Response:
                def __init__(self, data: bytes) -> None:
                    self._data = data

                def read(self) -> bytes:
                    return self._data

                def close(self) -> None:
                    return None

                def release_conn(self) -> None:
                    return None

            return _Response(self.objects[(bucket, key)])

        def put_object(self, bucket: str, key: str, data, *, length: int, content_type: str):
            self.objects[(bucket, key)] = data.read(length)

    client = _Client()
    monkeypatch.setattr(state_module, "_minio", lambda _settings: (client, None))
    monkeypatch.setattr(minio.error, "S3Error", _Missing)
    state_module._CHECKED_BUCKETS.clear()
    settings = _docker(tmp_path)

    marker = ensure_profile_state(settings)

    stored = json.loads(client.objects[("rag-docker-single", "_profile.json")])
    assert stored == marker
