"""profile 状态域标记：防止两个 profile 误用同一套状态。"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

LOCAL_STATE_FILE = "state.json"
BUCKET_STATE_KEY = "_profile.json"
STATE_VERSION = 1
_CHECKED_BUCKETS: set[tuple[str, str]] = set()


class ProfileStateError(RuntimeError):
    """profile、本地 marker 与 bucket marker 不一致。"""


def expected_state(settings: Any) -> dict[str, Any]:
    """当前配置应有的状态身份；同一 namespace 在所有进程中稳定。"""
    profile = settings.deployment.profile
    metadata = "sqlite" if profile in {"local-lite", "docker-single"} else "postgres"
    payload = {
        "schema_version": STATE_VERSION,
        "profile": profile,
        "vector_backend": settings.indexing.vector.backend,
        "vector_collection": settings.indexing.vector.collection,
        "metadata_backend": metadata,
        "metadata_sqlite": (
            str(Path(settings.documents.sqlite_path).resolve()) if metadata == "sqlite" else ""
        ),
        "minio_bucket": settings.storage.minio.bucket,
        "postgres_schema": (
            settings.persistence.postgres.schema_name if metadata == "postgres" else ""
        ),
        "redis_prefix": (settings.api.redis.key_prefix if metadata == "postgres" else ""),
    }
    if profile == "local-lite":
        payload["minio_bucket"] = ""
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    payload["namespace_id"] = digest
    return payload


def local_state_path(settings: Any) -> Path:
    from rag.deployment import profile_state_dir

    return profile_state_dir(settings, settings.deployment.profile) / LOCAL_STATE_FILE


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProfileStateError(f"状态标记读不出来：{path}（{exc}）") from exc
    if not isinstance(payload, dict):
        raise ProfileStateError(f"状态标记不是 JSON 对象：{path}")
    return payload


def _compare(path: str, found: dict[str, Any], expected: dict[str, Any]) -> None:
    mismatched = {
        key: (found.get(key), value) for key, value in expected.items() if found.get(key) != value
    }
    if mismatched:
        detail = "；".join(
            f"{key}: marker={old!r} expected={new!r}" for key, (old, new) in mismatched.items()
        )
        raise ProfileStateError(f"{path} 的 profile 标记不匹配：{detail}")


def _expected_path(settings: Any) -> Path:
    from rag.deployment import profile_state_dir

    return profile_state_dir(settings, settings.deployment.profile)


def ensure_local_marker(settings: Any) -> dict[str, Any]:
    expected = expected_state(settings)
    root = _expected_path(settings)
    root.mkdir(parents=True, exist_ok=True)
    path = root / LOCAL_STATE_FILE
    if path.exists():
        _compare(str(path), _read_json(path), expected)
        return expected
    path.write_text(
        json.dumps(expected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return expected


def check_local_marker(settings: Any) -> dict[str, Any] | None:
    expected = expected_state(settings)
    path = _expected_path(settings) / LOCAL_STATE_FILE
    if not path.exists():
        return None
    _compare(str(path), _read_json(path), expected)
    return expected


def _minio(settings: Any):
    from minio import Minio

    from rag.config import get_secrets

    cfg = settings.storage.minio
    secrets = get_secrets()
    return (
        Minio(
            cfg.endpoint,
            access_key=secrets.minio_access_key or "",
            secret_key=secrets.minio_secret_key or "",
            secure=cfg.secure,
            region=cfg.region or None,
        ),
        cfg,
    )


def check_bucket_marker(settings: Any) -> dict[str, Any] | None:
    expected = expected_state(settings)
    bucket = expected["minio_bucket"]
    if not bucket:
        return None
    from minio.error import S3Error

    client, _ = _minio(settings)
    try:
        response = client.get_object(bucket, BUCKET_STATE_KEY)
        try:
            found = json.loads(response.read().decode("utf-8"))
        finally:
            response.close()
            response.release_conn()
    except S3Error as exc:
        if exc.code in {"NoSuchKey", "NoSuchObject", "NoSuchBucket"}:
            return None
        raise ProfileStateError(f"读不到 bucket {bucket} 的 profile 标记：{exc}") from exc
    if not isinstance(found, dict):
        raise ProfileStateError(f"bucket {bucket} 的 profile 标记不是 JSON 对象")
    _compare(f"minio://{bucket}/{BUCKET_STATE_KEY}", found, expected)
    return expected


def ensure_bucket_marker(settings: Any) -> dict[str, Any] | None:
    expected = expected_state(settings)
    bucket = expected["minio_bucket"]
    if not bucket:
        return None
    from minio.error import S3Error

    cfg = settings.storage.minio
    key = (cfg.endpoint, bucket)
    if key in _CHECKED_BUCKETS:
        return expected
    client, _ = _minio(settings)
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket, location=cfg.region or None)
    try:
        response = client.get_object(bucket, BUCKET_STATE_KEY)
        try:
            found = json.loads(response.read().decode("utf-8"))
        finally:
            response.close()
            response.release_conn()
        if not isinstance(found, dict):
            raise ProfileStateError(f"bucket {bucket} 的 profile 标记不是 JSON 对象")
        _compare(f"minio://{bucket}/{BUCKET_STATE_KEY}", found, expected)
    except S3Error as exc:
        if exc.code not in {"NoSuchKey", "NoSuchObject"}:
            raise ProfileStateError(f"读不到 bucket {bucket} 的 profile 标记：{exc}") from exc
        data = (json.dumps(expected, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        client.put_object(
            bucket,
            BUCKET_STATE_KEY,
            io.BytesIO(data),
            length=len(data),
            content_type="application/json",
        )
    _CHECKED_BUCKETS.add(key)
    return expected


def ensure_profile_state(settings: Any) -> dict[str, Any]:
    expected = ensure_local_marker(settings)
    ensure_bucket_marker(settings)
    return expected
