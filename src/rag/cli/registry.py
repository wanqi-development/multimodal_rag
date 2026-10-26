"""文档注册表命令：初始化、迁移、对账与旧版本清理。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from rag.cli.runtime import _probe_milvus, _run_async
from rag.config import load_settings

registry_app = typer.Typer(
    help="文档注册表（每篇一行：走到哪一步 + 入库信息）。**2026-10-09 起落数据库**（PG / SQLite）",
    no_args_is_help=True,
)


@registry_app.command("init")
def registry_init() -> None:
    """建 `document_registry` 表（幂等）。**写操作**，所以显式触发 —— 与
    `rag models init` / `rag logs init` / `rag chats init` 同一个规矩。"""
    from rag.persistence.documents import build_document_store

    settings = load_settings()
    store = build_document_store(settings)
    store.setup()
    typer.secho(
        f"文档注册表已就绪：{store.name}（backend={settings.documents.backend}）",
        fg=typer.colors.GREEN,
    )


@registry_app.command("migrate")
def registry_migrate(
    from_milvus: bool = typer.Option(
        False, "--from-milvus", help="再按旧 Milvus 注册表把「真的入库了」标出来（要 Milvus 活着）"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="只报数，一个字都不写"),
) -> None:
    """把**已经存在的**文档搬进新表（这一步不重灌、不调模型、不花钱）。

    为什么需要它：注册表从 Milvus 搬到数据库之后，老数据的"这份走到哪一步了"没有地方记 ——
    不搬的话页面会把明明做过解析/切分的文档显示成"什么都没做"，逼人重来一遍。

    两路来源，**都不依赖向量库**（除非显式加 `--from-milvus`）：

    * `data/uploads/` —— 每个上传件一行，状态至少是 `uploaded`；
    * `data/stages/<doc_id>/` —— 有 `document.md` → `parsed`、有 `outline.tsv` → `reviewed`、
      有 `chunks.json` → `chunked`（块数 / 文件哈希 / 切法指纹都从这份产物抄）。

    **`indexed` 只能问 Milvus**：产物齐不代表灌进去了。所以 `--from-stages` 走到
    `chunked` 为止（看起来像"没入库"，重跑 `/index` 是安全的幂等操作）；等向量库好了，
    再跑一次 `--from-milvus` 把真正入库的那些补标成 `indexed`。

    命令**幂等，且只前进不后退**：已经是 `indexed` 的行不会被 `chunked` 覆盖。
    """
    from rag.api.uploads import build_upload_store
    from rag.contracts import DocRecord, stage_rank
    from rag.ingest.stages import build_artifact_store
    from rag.persistence.documents import build_document_store

    settings = load_settings()
    store = build_document_store(settings)
    store.setup()
    current = store.sync_stages()

    # 1) 上传件：至少 `uploaded`（`sources` 存落盘名，注册表里 `source_uri` 记的就是它）
    planned: dict[str, str] = {}
    sources: dict[str, str] = {}
    for item in build_upload_store(settings).list():
        planned[item.doc_id] = "uploaded"
        sources[item.doc_id] = item.name

    # 2) 产物：按"最高走到哪一步"升级（**不读 Milvus**）
    artifacts = build_artifact_store(settings)
    payloads: dict[str, dict] = {}
    root = Path(settings.ingest.stages.dir)
    if root.is_dir():
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            doc_id = entry.name
            stage = planned.get(doc_id, "uploaded")
            if (entry / "document.md").is_file() or (entry / "document.json").is_file():
                stage = "parsed"
            if (entry / "outline.tsv").is_file():
                stage = "reviewed"
            payload = artifacts.read_payload(doc_id, "chunks.json")
            if payload is not None:
                stage = "chunked"
                payloads[doc_id] = payload
            planned[doc_id] = stage

    # 3) 可选：旧 Milvus 注册表 → 真正入库的那些（**只有向量库活着才拿得到**）
    records: dict[str, DocRecord] = {}
    if from_milvus:
        from rag.indexing.milvus.registry import DocRegistry

        try:
            records = asyncio.run(DocRegistry(settings.indexing.vector).all())
        except Exception as exc:  # noqa: BLE001 —— 库够不着就退回「只按产物搬」
            typer.secho(
                f"读不到旧 Milvus 注册表（{type(exc).__name__}: {str(exc)[:160]}）；"
                f"这次只按产物搬，入库状态等向量库好了再补。",
                fg=typer.colors.YELLOW,
            )
        for doc_id, record in records.items():
            planned[doc_id] = "indexed"
            sources.setdefault(doc_id, record.source_uri)

    counts: dict[str, int] = {}
    advanced = 0
    for doc_id, stage in sorted(planned.items()):
        # **只前进不后退**：已经是 indexed 的行不该被产物那一趟拽回 chunked
        if stage_rank(stage) <= stage_rank(current.get(doc_id, "")):
            continue
        counts[stage] = counts.get(stage, 0) + 1
        advanced += 1
        if dry_run:
            continue
        store.sync_set_stage(doc_id, stage, sources.get(doc_id, ""))
        record = records.get(doc_id)
        if record is None and doc_id in payloads:
            payload = payloads[doc_id]
            record = DocRecord(
                doc_id=doc_id,
                source_uri=str(payload.get("source_uri") or sources.get(doc_id, "")),
                content_hash=str(payload.get("file_hash") or ""),
                chunk_count=len(payload.get("chunks") or []),
                ingested_at=int(payload.get("saved_at") or 0),
                # 注册表那一列 = 切法指纹 + 这一份的清单指纹（见 commit.index_fingerprint）
                pipeline_version=f"{payload.get('pipeline_version') or ''}"
                f"+{payload.get('outline_hash') or ''}",
            )
        if record is not None:
            # `stage=""`：状态上面已经写好了，这里只补数字，别把状态再写一遍
            store.sync_upsert([DocRecord(**{**record.model_dump(), "stage": ""})])

    total = "、".join(f"{name} {count}" for name, count in sorted(counts.items()) if count)
    head = "--dry-run（什么都没写）：" if dry_run else "已写入："
    typer.secho(f"{head}{advanced} 份（{total or '无'}）", fg=typer.colors.GREEN)
    typer.echo(f"注册表：{store.name}（backend={settings.documents.backend}）")
    if not from_milvus:
        typer.secho(
            "入库状态没搬（那只能问 Milvus）。等向量库好了再跑 "
            "`rag registry migrate --from-milvus`，把真正入库的补标成 indexed。",
            fg=typer.colors.YELLOW,
        )


@registry_app.command("reconcile")
def registry_reconcile(
    doc_id: str | None = typer.Option(None, "--doc-id", help="只核对一份文档"),
    repair: bool = typer.Option(
        False,
        "--repair",
        help="只回填可证明完成的 manifest 状态；不改写向量或块内容",
    ),
) -> None:
    """核对注册表 manifest、切分产物与 Milvus 实际块；默认只读。"""
    from rag.indexing import build_index, build_registry
    from rag.indexing.reconcile import reconcile_indexes
    from rag.ingest.stages import build_artifact_store

    settings = load_settings()

    async def _run():
        index = build_index(settings)
        await _probe_milvus(index, settings)
        return await reconcile_indexes(
            settings,
            registry=build_registry(settings),
            index=index,
            store=build_artifact_store(settings),
            repair=repair,
            doc_ids={doc_id} if doc_id else None,
        )

    report = _run_async(_run(), settings)
    unresolved = [issue for issue in report.issues if not issue.repaired]
    typer.echo(
        f"检查 {report.checked} 份，问题 {len(report.issues)} 条，"
        f"已修复 {report.repaired} 份，未解决 {len(unresolved)} 条。"
    )
    for issue in unresolved:
        typer.echo(f"- {issue.doc_id} [{issue.code}] {issue.message}")
    if unresolved:
        raise typer.Exit(1)


@registry_app.command("prune")
def registry_prune(
    older_hours: int = typer.Option(24, "--older-hours", min=1, help="只删超过多少小时的旧批次"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只列出可清理批次，不删任何东西"),
) -> None:
    """删除不再承载活动 batch 的旧版本；活动版本永不在这里删除。"""
    from rag.indexing import build_index, build_registry
    from rag.indexing.reconcile import prune_inactive_versions

    settings = load_settings()
    batches = asyncio.run(
        prune_inactive_versions(
            settings,
            registry=build_registry(settings),
            index=build_index(settings),
            older_than_seconds=older_hours * 3600,
            dry_run=dry_run,
        )
    )
    head = "可清理" if dry_run else "已清理"
    typer.echo(f"{head} {len(batches)} 个旧批次")
    for batch_id in batches:
        typer.echo(f"- {batch_id}")
