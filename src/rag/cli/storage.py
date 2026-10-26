"""存储迁移命令：把本地已有文件幂等推送到 MinIO。"""

from __future__ import annotations

from pathlib import Path

import typer

from rag.config import load_settings

storage_app = typer.Typer(
    help="文件落点：本地盘 / MinIO 桶（多 worker 用桶，见 storage.py）", no_args_is_help=True
)


@storage_app.command("migrate")
def storage_migrate(
    dry_run: bool = typer.Option(False, "--dry-run", help="只报数，一个字都不写"),
    only: str = typer.Option(
        "uploads,stages,assets,media", "--only", help="只搬这几个域（逗号分隔）"
    ),
) -> None:
    """把**本地盘上已有的**文件推上 MinIO 桶（幂等）。改动的是桶，本地文件不删。

    什么时候用：把统一开关 `storage.backend`（或某一域的 `backend`）从 `local` 改成
    `minio` **之后**。不改的话新上传的会进桶，而老文件还在本地 —— 页面就少看见它们
    （台账已经独立在库里，不会丢）。只搬**解析后**为 minio 的那几个域；`--dry-run`
    先看会搬多少。

    键与两个后端保持一致（`<域前缀>/<相对路径>`），所以搬完读到的还是同一批 `doc_id`。
    """
    from rag.api.uploads import INDEX_NAME
    from rag.storage import LocalBlobStore, build_blob_store, resolve_storage_backend

    settings = load_settings()
    wanted = {item.strip() for item in only.split(",") if item.strip()}
    plan = (
        (
            "uploads",
            settings.ingest.uploads.backend,
            settings.ingest.uploads.dir,
            settings.ingest.uploads.prefix,
        ),
        (
            "stages",
            settings.ingest.stages.backend,
            settings.ingest.stages.dir,
            settings.ingest.stages.prefix,
        ),
        (
            "assets",
            settings.ingest.assets.backend,
            settings.ingest.assets.dir,
            settings.ingest.assets.prefix,
        ),
        ("media", settings.api.media_backend, settings.api.media_dir, settings.api.media_prefix),
    )
    total = 0
    for name, backend, directory, prefix in plan:
        if name not in wanted:
            continue
        # 四个域默认写 auto = 听统一开关 `storage.backend`，所以这里得先解析再判
        resolved = resolve_storage_backend(settings, backend)
        if resolved != "minio":
            typer.secho(
                f"{name}：backend={resolved}，跳过（只搬配成 minio 的域）", fg=typer.colors.YELLOW
            )
            continue
        local = LocalBlobStore(directory)
        # 那份老台账（`_index.json`）与半截文件（`.part`）不是内容，别搬进桶
        all_keys = local.keys()
        keys = [
            key for key in all_keys if not key.endswith(".part") and Path(key).name != INDEX_NAME
        ]
        total += len(keys)
        if dry_run:
            typer.echo(f"{name}：将搬 {len(keys)} 个文件 → {prefix}/")
            continue
        target = build_blob_store(backend="minio", settings=settings, dir=directory, prefix=prefix)
        for key in keys:
            target.put(key, local.get(key))
        typer.secho(f"{name}：已搬 {len(keys)} 个 → {target.name}", fg=typer.colors.GREEN)
    head = "--dry-run（什么都没写）" if dry_run else "搬完"
    typer.secho(f"{head}：共 {total} 个文件", fg=typer.colors.GREEN)
