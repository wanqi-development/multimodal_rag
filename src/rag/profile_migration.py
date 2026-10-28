"""把旧版共享状态显式导入一个隔离 profile；只复制，不删除源数据。"""

from __future__ import annotations

import asyncio
import io
import json
import sqlite3
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from rag.config import get_secrets, load_settings
from rag.deployment import apply_deployment_profile

TABLE_GROUPS = {
    "documents.sqlite3": (
        "document_registry",
        "index_versions",
        "upload_meta",
    ),
    "model_settings.sqlite3": (
        "model_setting",
        "model_setting_change",
    ),
    "chats.sqlite3": (
        "chat_session",
        "chat_message",
        "token_usage",
    ),
}


def _target_settings():
    settings = load_settings()
    if settings.deployment.profile != "docker-single":
        raise RuntimeError("legacy 导入当前只支持目标 profile=docker-single")
    apply_deployment_profile(settings)
    return settings


def _pg_connection(schema: str, secrets: Any):
    import os

    import psycopg
    from dotenv import dotenv_values

    values = dotenv_values(".env")
    url = os.environ.get("RAG_PG_URL") or values.get("RAG_PG_URL")
    if not url:
        raise RuntimeError("没有 RAG_PG_URL，无法读取 legacy PostgreSQL")
    password = secrets.rag_pg_password or values.get("RAG_PG_PASSWORD")
    return psycopg.connect(
        url,
        password=password or None,
        options=f"-c search_path={schema}",
    )


def _minio(settings: Any):
    from minio import Minio

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


def _object_names(client: Any, bucket: str) -> list[str]:
    return [
        item.object_name
        for item in client.list_objects(bucket, recursive=True)
        if (
            not item.is_dir
            and not item.object_name.endswith(".part")
            and item.object_name != "_profile.json"
        )
    ]


def _target_table_counts(settings: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for filename, tables in TABLE_GROUPS.items():
        path = Path(
            settings.documents.sqlite_path
            if filename == "documents.sqlite3"
            else settings.models.sqlite_path
            if filename == "model_settings.sqlite3"
            else settings.persistence.chat_sqlite_path
        )
        if not path.exists():
            counts.update({table: 0 for table in tables})
            continue
        conn = sqlite3.connect(path)
        try:
            for table in tables:
                try:
                    counts[table] = int(
                        conn.execute(f'select count(*) from "{table}"').fetchone()[0]
                    )
                except sqlite3.OperationalError:
                    counts[table] = 0
        finally:
            conn.close()
    return counts


def plan_legacy(source_bucket: str = "rag", source_schema: str = "rag") -> dict[str, Any]:
    settings = _target_settings()
    client, cfg = _minio(settings)
    if not client.bucket_exists(source_bucket):
        raise RuntimeError(f"源 bucket 不存在：{source_bucket}")
    source_objects = _object_names(client, source_bucket)
    target_objects = (
        _object_names(client, settings.storage.minio.bucket)
        if client.bucket_exists(settings.storage.minio.bucket)
        else []
    )
    with _pg_connection(source_schema, get_secrets()) as conn, conn.cursor() as cur:
        tables = {}
        for table in (
            *TABLE_GROUPS["documents.sqlite3"],
            *TABLE_GROUPS["model_settings.sqlite3"],
            *TABLE_GROUPS["chats.sqlite3"],
        ):
            cur.execute(f"select count(*) from {source_schema}.{table}")
            tables[table] = int(cur.fetchone()[0])
    return {
        "target_profile": settings.deployment.profile,
        "target_bucket": cfg.bucket,
        "source_bucket": source_bucket,
        "source_schema": source_schema,
        "source_objects": len(source_objects),
        "target_objects": len(target_objects),
        "target_tables": _target_table_counts(settings),
        "tables": tables,
    }


def _adapt(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _rows_signature(rows: list[dict[str, Any]], cols: list[str]) -> list[str]:
    return sorted(
        json.dumps([_adapt(row[name]) for name in cols], ensure_ascii=False, sort_keys=True)
        for row in rows
    )


def _copy_tables(settings: Any, schema: str, secrets: Any) -> dict[str, int]:
    from psycopg.rows import dict_row

    from rag.persistence import build_chat_store
    from rag.persistence.documents import build_document_store
    from rag.persistence.model_settings import build_model_setting_store
    from rag.persistence.uploads import build_upload_meta_store

    build_document_store(settings).setup()
    build_upload_meta_store(settings).setup()
    build_model_setting_store(settings).setup()
    chats = build_chat_store(settings)
    asyncio.run(chats.setup())

    copied: dict[str, int] = {}
    conn = _pg_connection(schema, secrets)
    conn.row_factory = dict_row
    try:
        for filename, tables in TABLE_GROUPS.items():
            sqlite_path = Path(
                settings.documents.sqlite_path
                if filename == "documents.sqlite3"
                else settings.models.sqlite_path
                if filename == "model_settings.sqlite3"
                else settings.persistence.chat_sqlite_path
            )
            target = sqlite3.connect(sqlite_path)
            target.row_factory = sqlite3.Row
            target.execute("pragma foreign_keys = off")
            try:
                target.execute("begin immediate")
                for table in tables:
                    with conn.cursor() as cur:
                        cur.execute(f"select * from {schema}.{table}")
                        rows = cur.fetchall()
                    if not rows:
                        copied[table] = 0
                        continue
                    cols = list(rows[0])
                    existing = [
                        dict(row) for row in target.execute(f'select * from "{table}"').fetchall()
                    ]
                    if existing:
                        if len(existing) != len(rows) or _rows_signature(
                            existing, cols
                        ) != _rows_signature(rows, cols):
                            raise RuntimeError(f"目标表 {table} 已有不同数据，拒绝覆盖")
                        copied[table] = 0
                        continue
                    placeholders = ", ".join("?" for _ in cols)
                    quoted = ", ".join(f'"{name}"' for name in cols)
                    target.executemany(
                        f'insert into "{table}" ({quoted}) values ({placeholders})',
                        [[_adapt(row[name]) for name in cols] for row in rows],
                    )
                    copied[table] = len(rows)
                target.execute("commit")
            except Exception:
                target.execute("rollback")
                raise
            finally:
                target.close()
    finally:
        conn.close()
    return copied


def copy_legacy(
    source_bucket: str = "rag",
    source_schema: str = "rag",
) -> dict[str, Any]:
    plan = plan_legacy(source_bucket, source_schema)
    settings = _target_settings()
    client, cfg = _minio(settings)
    target_bucket = cfg.bucket
    if not client.bucket_exists(target_bucket):
        client.make_bucket(target_bucket, location=cfg.region or None)
    source_names = _object_names(client, source_bucket)
    target_names = set(_object_names(client, target_bucket))
    extras = sorted(target_names - set(source_names))
    if extras:
        raise RuntimeError(
            f"目标 bucket 有源状态之外的对象，拒绝合并：{extras[:5]}（共 {len(extras)} 个）"
        )
    copied_objects = 0
    skipped_objects = 0
    for key in source_names:
        source_stat = client.stat_object(source_bucket, key)
        if key in target_names:
            target_stat = client.stat_object(target_bucket, key)
            if source_stat.size != target_stat.size:
                raise RuntimeError(f"同名对象大小不一致，拒绝覆盖：{key}")
            skipped_objects += 1
            continue
        response = client.get_object(source_bucket, key)
        try:
            data = response.read()
        finally:
            response.close()
            response.release_conn()
        client.put_object(
            target_bucket,
            key,
            io.BytesIO(data),
            length=len(data),
            content_type="application/octet-stream",
        )
        copied_objects += 1

    copied_tables = _copy_tables(settings, source_schema, get_secrets())
    from rag.profile_state import ensure_profile_state

    ensure_profile_state(settings)
    return {
        **plan,
        "copied_objects": copied_objects,
        "skipped_objects": skipped_objects,
        "copied_tables": copied_tables,
    }
