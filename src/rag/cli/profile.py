"""Profile 状态域查看、初始化与只读校验。"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from rag.config import load_settings
from rag.deployment import profile_state_dir
from rag.profile_migration import copy_legacy, plan_legacy
from rag.profile_state import (
    ProfileStateError,
    check_bucket_marker,
    check_local_marker,
    ensure_profile_state,
    expected_state,
)

profile_app = typer.Typer(
    help="部署 profile 的完整状态域（本地路径 / bucket / collection / schema / Redis）",
    no_args_is_help=True,
)


def _summary(settings) -> dict:
    root = profile_state_dir(settings, settings.deployment.profile)
    return {
        **expected_state(settings),
        "state_dir": root.as_posix(),
        "documents_sqlite": Path(settings.documents.sqlite_path).as_posix(),
        "models_sqlite": Path(settings.models.sqlite_path).as_posix(),
        "chats_sqlite": Path(settings.persistence.chat_sqlite_path).as_posix(),
        "uploads": settings.ingest.uploads.dir,
        "stages": settings.ingest.stages.dir,
        "assets": settings.ingest.assets.dir,
        "media": settings.api.media_dir,
        "workspace": settings.agent.workspace_dir,
        "run_dir": settings.deployment.run_dir,
        "log_dir": (root / "logs").as_posix(),
    }


@profile_app.command("show")
def profile_show() -> None:
    """打印当前 profile 的完整状态域。只读。"""
    typer.echo(json.dumps(_summary(load_settings()), ensure_ascii=False, indent=2))


@profile_app.command("check")
def profile_check() -> None:
    """校验本地 state.json 与 bucket _profile.json；缺失或冲突时退出 1。"""
    settings = load_settings()
    try:
        local = check_local_marker(settings)
        bucket = check_bucket_marker(settings)
    except ProfileStateError as exc:
        typer.secho(f"✘ {exc}", fg=typer.colors.RED)
        raise typer.Exit(1) from exc

    if local is None:
        typer.secho("✘ 本地 state.json 不存在；先运行 `rag profile init`。", fg=typer.colors.RED)
        raise typer.Exit(1)
    if settings.storage.minio.bucket and bucket is None:
        typer.secho(
            "✘ bucket _profile.json 不存在；先运行 `rag profile init`。",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)
    typer.secho(
        f"✓ {settings.deployment.profile} 状态域一致：{local['namespace_id']}",
        fg=typer.colors.GREEN,
    )


@profile_app.command("init")
def profile_init() -> None:
    """创建/确认本地 state.json 与 bucket _profile.json（幂等写操作）。"""
    settings = load_settings()
    try:
        state = ensure_profile_state(settings)
    except ProfileStateError as exc:
        typer.secho(f"✘ {exc}", fg=typer.colors.RED)
        raise typer.Exit(1) from exc
    typer.secho(
        f"✓ {settings.deployment.profile} 状态域已就绪：{state['namespace_id']}",
        fg=typer.colors.GREEN,
    )


@profile_app.command("import-legacy")
def profile_import_legacy(
    source_bucket: str = typer.Option("rag", "--source-bucket", help="旧应用 bucket"),
    source_schema: str = typer.Option("rag", "--source-schema", help="旧 PostgreSQL schema"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只统计，不写对象或数据库"),
) -> None:
    """把旧共享状态复制到当前 docker-single；源数据不删不改。"""
    try:
        plan = plan_legacy(source_bucket, source_schema)
        typer.echo(json.dumps(plan, ensure_ascii=False, indent=2))
        if dry_run:
            typer.secho("✓ dry-run：没有写入任何文件或数据库", fg=typer.colors.YELLOW)
            return
        result = copy_legacy(source_bucket, source_schema)
    except Exception as exc:  # noqa: BLE001 —— CLI 边界翻成明确退出码
        typer.secho(f"✘ 导入失败：{type(exc).__name__}: {exc}", fg=typer.colors.RED)
        raise typer.Exit(1) from exc
    typer.secho(
        f"✓ 已复制对象 {result['copied_objects']} 个、复用 "
        f"{result['skipped_objects']} 个，表 "
        f"{sum(result['copied_tables'].values())} 行；源数据未删。",
        fg=typer.colors.GREEN,
    )
