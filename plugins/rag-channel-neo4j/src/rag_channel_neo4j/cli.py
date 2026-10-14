"""`rag graph …` 三条命令：抽实体关系 → 写 Neo4j → 排查（都在这一个插件包里）。

核心只按 entry point 把这个 Typer 子应用挂到 `rag` 下（组名 `rag.cli`）——
**核心不认识 Neo4j**，也不 import 这个包的任何东西。

2026-10-08 从 `rag.cli.main` 整段搬过来（那段原本占了 253 行）；命令名保持不变，
所以 `rag graph extract/load/stats/query` 的用法跟以前一样，另外新增 `rag graph build`
（抽 + 灌一条命令，给"已经灌过 Milvus、现在才想上图"的语料用）。
"""

from __future__ import annotations

import asyncio
from typing import Any

import typer

from rag.config import get_secrets, load_settings

# 插件**不 import 核心 CLI 的内部辅助**（`_run_async` / `_tracing` 那种私有函数），
# 所以这里自带两个同名的薄片，被搬过来的命令体可以原样用。


def _run_async(coro: Any, settings: Any = None) -> Any:
    """跑协程：出错打一行人话再退出 1（不吞栈里的类型）。"""
    try:
        return asyncio.run(coro)
    except Exception as exc:
        typer.secho(f"{type(exc).__name__}: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc


def _tracing(settings: Any) -> None:
    """按配置开 LangSmith trace（只在有话说的时候出声）。"""
    from rag.observability import configure_tracing

    status = configure_tracing(settings)
    if not status.startswith("关闭"):
        typer.secho(f"LangSmith：{status}", fg=typer.colors.CYAN)


# ---- 知识图谱（M6）----

app = typer.Typer(
    help="知识图谱（M6）：从切分产物抽实体与关系，先落产物再谈写 Neo4j",
    no_args_is_help=True,
)


@app.command("extract")
def graph_extract(
    doc_id: list[str] = typer.Option([], "--doc-id", help="对哪几篇抽（可重复）；与 --all 二选一"),
    all_docs: bool = typer.Option(False, "--all", help="对所有已切好的文档抽"),
    limit: int | None = typer.Option(None, "--limit", help="每篇只抽前 N 块（试水用）"),
    concurrency: int = typer.Option(1, "--concurrency", help="同时在跑的抽取数"),
    force: bool = typer.Option(False, "--force", help="忽略缓存整篇重抽（**会重复花钱**）"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只算账：几块要抽、几块能复用"),
) -> None:
    """从**切分产物**抽实体与关系，写 `data/stages/<doc_id>/graph.json`。

    输入是块，不是原文 —— 这样关系上的 `chunk_id` 才能直接映射回检索结果。
    钱只花在模型调用上，所以：默认**按块复用缓存**（正文没变不重抽）、`--dry-run` 先算账、
    `--force` 才是整篇重抽。抽不出来/调用失败都记进产物（`error`），不带走整批。
    """
    from rag.ingest.stages import build_artifact_store
    from rag.providers import build_chat
    from rag_channel_neo4j.extract import (
        chat_model_name,
        extract_document,
        load_staged_chunks,
        staged_doc_ids,
        summarize_plan,
    )

    if bool(doc_id) == all_docs:
        raise typer.BadParameter("--doc-id 与 --all 二选一")

    settings = load_settings()
    _tracing(settings)
    store = build_artifact_store(settings)
    targets = list(doc_id) or (staged_doc_ids(store) if all_docs else [])
    if not targets:
        typer.secho(
            "没有可抽的文档：给 --doc-id，或用 --all（得先有切分产物）。", fg=typer.colors.YELLOW
        )
        raise typer.Exit(code=1)

    chat = build_chat(settings, get_secrets(), settings.agent.chat)
    model = chat_model_name(chat)

    if dry_run:
        rows = summarize_plan(store, targets, limit=limit, model=model)
        for row in rows:
            if row.get("error"):
                typer.secho(f"  {row['doc_id']}：{row['error']}", fg=typer.colors.RED)
                continue
            typer.echo(
                f"  {row['doc_id']}：{row['chunks']} 块 → 要抽 {row['todo']}，"
                f"可复用 {row['reusable']}"
            )
        pending = sum(row.get("todo", 0) for row in rows)
        typer.secho(
            f"\n--dry-run：这次会调 {pending} 次模型，没有任何写入。", fg=typer.colors.YELLOW
        )
        return

    async def _run():
        out = []
        for target in targets:
            source_uri, chunks = load_staged_chunks(store, target)
            out.append(
                await extract_document(
                    target,
                    chunks,
                    chat=chat,
                    store=store,
                    source_uri=source_uri,
                    model_name=model,
                    concurrency=concurrency,
                    limit=limit,
                    force=force,
                )
            )
        return out

    try:
        results = _run_async(_run(), settings)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from None

    typer.echo(f"\n抽取完成（模型={model}，提示词版本={_graph_prompt_version()}）：")
    for item in results:
        note = f"，失败 {item.failed} 块" if item.failed else ""
        typer.echo(
            f"  {item.doc_id}：{item.chunks} 块"
            f"（新抽 {item.extracted} / 复用 {item.reused}{note}），"
            f"有内容的块 {item.chunks_with_content}，实体 {item.entities}，"
            f"关系 {item.relations}"
        )
    typer.secho(f"\n产物：{results[0].path.parent}", fg=typer.colors.GREEN)


def _graph_prompt_version() -> str:
    from rag_channel_neo4j.extract import PROMPT_VERSION

    return PROMPT_VERSION


@app.command("load")
def graph_load(
    doc_id: list[str] = typer.Option([], "--doc-id", help="灌哪几篇；与 --all 二选一"),
    all_docs: bool = typer.Option(False, "--all", help="把所有有图谱产物的文档都灌一遍"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只算账：每篇多少实体/关系"),
) -> None:
    """把抽取产物写进 Neo4j（M6）。

    **按文档幂等**：先删这篇文档的边、再写新的（与 Milvus「先删后写」同一个规矩），
    所以重灌一份不会留下上一次的边。节点是共享的（归一之后本来就该共用），不删。
    """
    from rag.ingest.stages import build_artifact_store
    from rag_channel_neo4j.store import build_graph_store

    if bool(doc_id) == all_docs:
        raise typer.BadParameter("--doc-id 与 --all 二选一")
    settings = load_settings()
    _tracing(settings)
    artifacts = build_artifact_store(settings)
    targets = list(doc_id) or (
        sorted(path.parent.name for path in artifacts.root.glob("*/graph.json") if path.is_file())
        if all_docs
        else []
    )
    if not targets:
        typer.secho(
            "没有可灌的图谱产物：给 --doc-id，或用 --all（先跑 `rag graph extract`）。",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=1)

    payloads: list[tuple[str, dict]] = []
    for target in targets:
        payload = artifacts.load_any_graph(target)
        if payload is None:
            typer.secho(f"  {target}：没有图谱产物，跳过", fg=typer.colors.YELLOW)
            continue
        payloads.append((target, payload))
    if not payloads:
        typer.secho("一条都没有，退出。", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
    if dry_run:
        for target, payload in payloads:
            entries = payload.get("chunks") or {}
            entities = sum(len(item.get("entities") or []) for item in entries.values())
            relations = sum(len(item.get("relations") or []) for item in entries.values())
            typer.echo(f"  {target}：{len(entries)} 块 → 实体 {entities}，关系 {relations}")
        typer.secho("\n--dry-run：没有写任何东西。", fg=typer.colors.YELLOW)
        return

    graph = build_graph_store(settings, get_secrets())

    async def _run():
        from rag_channel_neo4j.extract import load_staged_chunks

        await graph.ensure_schema()
        out = []
        for target, payload in payloads:
            # 章节/页码从**切分产物**补齐（graph.json 只存抽取结果，不重复存块的元数据）。
            # 对不上的 chunk_id = 图谱产物比切分产物旧（重切过），这条边直接丢掉并出声。
            staged = {chunk.chunk_id: chunk for chunk in load_staged_chunks(artifacts, target)[1]}
            entries: dict[str, dict] = {}
            stale = 0
            for chunk_id, item in (payload.get("chunks") or {}).items():
                chunk = staged.get(chunk_id)
                if chunk is None:
                    stale += 1
                    continue
                entries[chunk_id] = {
                    **item,
                    "heading_path": chunk.heading_path,
                    "page": chunk.page,
                }
            if stale:
                typer.secho(
                    f"  {target}：{stale} 块的图谱产物比切分产物旧（重切过？），已跳过 ——"
                    f"要覆盖它们先跑 `rag graph extract --doc-id {target}`",
                    fg=typer.colors.YELLOW,
                )
            out.append(
                await graph.load_document(
                    target, entries, source_uri=str(payload.get("source_uri") or "")
                )
            )
        return out

    results = _run_async(_run(), settings)
    typer.echo("")
    for item in results:
        typer.echo(
            f"  {item.doc_id}：实体 {item.entities}，关系 {item.relations}"
            f"（按字符构成丢了 {item.skipped} 条不像概念的）"
        )
    typer.secho(f"\n图库：{settings.retrieval.graph.uri}", fg=typer.colors.GREEN)


@app.command("build")
def graph_build(
    doc_id: list[str] = typer.Option([], "--doc-id", help="做哪几篇（可重复）；与 --all 二选一"),
    all_docs: bool = typer.Option(False, "--all", help="所有已切好的文档"),
    limit: int | None = typer.Option(None, "--limit", help="每篇只抽前 N 块（试水用）"),
    concurrency: int = typer.Option(1, "--concurrency", help="同时在跑的抽取数"),
    force: bool = typer.Option(False, "--force", help="忽略缓存整篇重抽（**会重复花钱**）"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只算账：几块要抽、几块能复用"),
) -> None:
    """**抽 + 灌一条命令**（`extract` 之后接着 `load`）。

    这条是给「**已经灌过 Milvus、现在才想上图**」的语料用的 —— 也是搬插件时最要紧的
    那条路：图是从**切分产物**（`data/stages/<doc_id>/chunks.json`）里抽的，而那份产物
    在灌库时就落好了。所以补图**不用重新解析、不用重新 embedding、不用重灌 Milvus**，
    只花抽取这一次 LLM 的钱（而且按块缓存，之前抽过的一块都不重花）。

    例：`rag graph build --all`（先 `--dry-run` 看一眼要花多少）。
    """
    graph_extract(
        doc_id=doc_id,
        all_docs=all_docs,
        limit=limit,
        concurrency=concurrency,
        force=force,
        dry_run=dry_run,
    )
    if dry_run:
        typer.echo("\n（--dry-run：只算账，没有写图；去掉它就真灌）")
        return
    typer.echo("")
    graph_load(doc_id=doc_id, all_docs=all_docs, dry_run=False)


@app.command("stats")
def graph_stats() -> None:
    """看一眼图里现在有什么：实体/关系总数 + 每篇文档各多少条。"""
    from rag_channel_neo4j.store import build_graph_store

    settings = load_settings()
    graph = build_graph_store(settings, get_secrets())

    async def _run():
        return await graph.stats()

    stats = _run_async(_run(), settings)
    typer.echo(f"实体 {stats.entities}，关系 {stats.relations}")
    for doc, count in stats.by_doc.items():
        typer.echo(f"  {doc}：{count} 条关系")


@app.command("query")
def graph_query(
    question: str = typer.Argument(..., help="问题（看它认出哪些实体、拉回哪些关系）"),
    hops: int | None = typer.Option(None, "--hops", help="覆盖 retrieval.graph.hops"),
    limit: int = typer.Option(12, "--limit", help="最多打几条关系"),
) -> None:
    """排查用：**不连 Milvus**，只看实体链接与邻居。"""
    from rag_channel_neo4j.store import build_graph_store

    settings = load_settings()
    graph = build_graph_store(settings, get_secrets())

    async def _run():
        keys = await graph.find_entities(question, limit=settings.retrieval.graph.max_entities)
        edges = await graph.neighbors(
            keys,
            hops=hops or settings.retrieval.graph.hops,
            limit=settings.retrieval.graph.edges_per_entity,
        )
        return keys, edges

    keys, edges = _run_async(_run(), settings)
    typer.echo(f"认出的实体（归一后）：{keys or '（没有）'}")
    for edge in edges[:limit]:
        typer.echo(
            f"  hop{edge['hop']} {edge['source']} --{edge['relation']}--> {edge['neighbor']}"
            f"  ← {edge['chunk_id']} {str(edge.get('heading_path') or '')[:34]}"
        )
    typer.echo(f"\n共 {len(edges)} 条关系（去重后）")
