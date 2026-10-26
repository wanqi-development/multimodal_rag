"""系统管理端点：健康、模型选型、供应商与日志。"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from rag.api.common import HEARTBEAT_S, UNPROCESSABLE, _event
from rag.api.deps import get_components
from rag.api.schemas import (
    DependencyView,
    HealthResponse,
    LogLineView,
    LogSearchResponse,
    LogTailResponse,
    ModelListView,
    ModelSlotUpdateRequest,
    ModelSlotView,
    ProviderListView,
    ProviderSettingView,
    ProviderUpdateRequest,
)
from rag.obs import live
from rag.obs.log import tail_items

system_router = APIRouter()


@system_router.get("/health", response_model=HealthResponse)
async def health(components=Depends(get_components)) -> HealthResponse:
    """**分开报**：Milvus / Redis / 模型各报各的，哪个挂了单独看得见。

    一个笼统的 `ok` 只能回答「能不能用」，回答不了「哪里坏了」—— 后者才是运维要看的。
    """
    settings = components.settings
    milvus = DependencyView(ok=False, detail=f"不可用：{components.index_error}")
    if components.index is not None:
        try:
            # 只读：`count()` 是一次查询，不建表、不写
            blocks = await components.index.count()
            milvus = DependencyView(
                ok=True, detail=f"{settings.indexing.vector.collection}：{blocks} 个块"
            )
        except Exception as exc:  # noqa: BLE001
            milvus = DependencyView(ok=False, detail=f"{type(exc).__name__}: {exc}")

    redis_ok = await components.redis.ping()
    redis = DependencyView(
        ok=redis_ok,
        detail=(
            f"已连接（前缀 {components.redis.prefix}）：限流与任务状态跨 worker 共享"
            if redis_ok
            else f"{components.redis.reason or '不可用'}；限流与任务状态已退回进程内"
        ),
    )

    persistence = components.persistence
    pg_ok = bool(persistence and await persistence.ping())
    postgres = DependencyView(
        ok=pg_ok,
        detail=(
            f"已连接（schema {components.settings.persistence.postgres.schema_name}）："
            f"多轮会话与中断状态落库"
            if pg_ok
            else f"{(persistence.reason if persistence else '未启用')}；多轮会话退回单轮"
        ),
    )

    heavy = components.heavy
    models = {
        "embedding": _loaded(heavy.embedder, "已加载", "未加载（首次用到时加载）"),
        "reranker": (
            _loaded(heavy.reranker, "已加载", "未加载（首次用到时加载）")
            if heavy.reranker is not None
            else DependencyView(ok=True, detail="未启用（retrieval.rerank.enabled=false）")
        ),
        "chat": DependencyView(
            ok="chat" not in heavy.errors, detail=heavy.errors.get("chat", "已建好")
        ),
    }
    degraded = (
        not milvus.ok or not redis_ok or not pg_ok or any(not view.ok for view in models.values())
    )
    ingest_gate = getattr(components, "ingest_gate", None)
    from rag.storage import resolve_storage_backend

    vision_cache = resolve_storage_backend(
        settings,
        settings.ingest.images.cache_backend,
    )
    return HealthResponse(
        status="degraded" if degraded else "ok",
        milvus=milvus,
        redis=redis,
        postgres=postgres,
        models=models,
        config={
            "deployment_profile": settings.deployment.profile,
            "workers": settings.api.workers,
            "replicas": settings.deployment.replicas,
            "collection": settings.indexing.vector.collection,
            "embedding": settings.providers.embedding.backend,
            "rerank": settings.retrieval.rerank.enabled,
            "graph": settings.graph.enabled,
            "channels": [item.name for item in settings.retrieval.channels if item.enabled],
            "rate_limit_per_min": settings.api.rate_limit_per_min,
            "ingest_gate": (ingest_gate.backend if ingest_gate is not None else "未启动"),
            "job_store": "redis" if redis_ok else "process",
            "vision_cache": vision_cache,
        },
    )


def _loaded(obj, yes: str, no: str) -> DependencyView:
    """模型加载状态。**只读实例字典** —— `getattr` 会触发 cached_property 的加载，
    而 `/health` 绝不该把 1GB 模型拉起来。"""
    if obj is None:
        return DependencyView(ok=False, detail="没建起来")
    # 用 getattr 取 __dict__：替身/带 __slots__ 的对象可能没有它
    state = getattr(obj, "__dict__", {})
    return DependencyView(ok=True, detail=yes if "_model" in state else no)


# ---- 日志（2026-10-06 起不进数据库：四个文件；agent 那三类共用一个）----


def _log_files(settings) -> dict[str, str]:
    """类别 → 文件路径。来源与 `configure_logging` 装的是同一份配置。"""
    cfg = settings.observability.logging
    files = {name: item.path for name, item in (cfg.files or {}).items()}
    files["system"] = cfg.system.path
    return files


@system_router.put("/models/{slot}", response_model=ModelListView)
async def set_model_slot(
    slot: str, body: ModelSlotUpdateRequest, components=Depends(get_components)
) -> ModelListView:
    """改一个槽位（写库 + **写一行审计**）。**重启后生效** —— 这一版不做热切。

    校验三件事：槽位名在 `MODEL_SLOTS` 里（否则 422）、`provider_key` 若给了就得是
    `providers.chat` 里有的键（否则 422，避免写进一个谁都认不出的名字）、key 若被
    `require_key` 要求就得在环境里配着（否则 422 —— 让人当场知道"还差一把 key"，
    而不是等重启后第一次调用才炸）。
    """
    from rag.config.schema import MODEL_SLOTS
    from rag.persistence.model_settings import build_model_setting_store

    if slot not in MODEL_SLOTS:
        raise HTTPException(UNPROCESSABLE, f"未知的槽位 {slot}；可选：{list(MODEL_SLOTS)}")
    settings = components.settings
    if (
        body.provider_key
        and slot.startswith("chat.")
        and body.provider_key not in settings.providers.chat
    ):
        raise HTTPException(
            UNPROCESSABLE,
            f"providers.chat 里没有 {body.provider_key!r}；可选：{list(settings.providers.chat)}",
        )
    store = build_model_setting_store(settings)
    # 写入前把表建好（幂等）：没跑过 `rag models init` 的机器上，设置页也能直接保存。
    await asyncio.to_thread(store.setup)
    await asyncio.to_thread(
        store.set,
        slot,
        provider_key=body.provider_key,
        model=body.model,
        params=body.params,
        actor=body.actor or "ui",
    )
    return await list_models(components)  # 复用只读那一条：回整份列表，前端不用再拉一次


@system_router.get("/models", response_model=ModelListView)
async def list_models(components=Depends(get_components)) -> ModelListView:
    """八个模型槽位当前的选型 + **来源** + key 是否就绪。**只读。**

    为什么要有它：模型选型散在 `.env` / yaml / 库里，改完又"重启才生效"，
    没有一处能回答"现在跑的到底是哪个模型、这个值谁定的"。这一版只做**读**：
    写入走 `rag models set`（并留审计），**热切是遗留项**（`docs/backlog.md` 第 2 条）。

    key 只报"配没配"（True/False/None），**不回值** —— 与 `Secrets` 那条硬边界一致。
    """
    from rag.config.schema import MODEL_SLOTS
    from rag.persistence.model_settings import build_model_setting_store

    settings = components.settings
    store = build_model_setting_store(settings)
    stored = await asyncio.to_thread(store.all) if hasattr(store, "all") else {}
    chat_keys = list(settings.providers.chat)
    secrets = getattr(components, "secrets", None)

    slots: list[ModelSlotView] = []
    for name in MODEL_SLOTS:
        row = stored.get(name)
        provider_key = str((row or {}).get("provider_key") or "")
        candidates = chat_keys if name.startswith("chat.") else []
        probe = provider_key or (chat_keys[0] if candidates else "")
        slots.append(
            ModelSlotView(
                slot=name,
                provider_key=provider_key,
                model=str((row or {}).get("model") or ""),
                params=dict((row or {}).get("params") or {}),
                source="库" if row else "未设置",
                version=(row or {}).get("version"),
                updated_at=str((row or {}).get("updated_at") or ""),
                updated_by=str((row or {}).get("updated_by") or ""),
                candidates=candidates,
                key_ready=_key_ready(secrets, probe) if candidates else None,
            )
        )
    return ModelListView(
        backend=settings.models.backend,
        storage=store.name,
        slots=slots,
        note="改选型用 `rag models set <槽位> …`；**重启后生效**（热切是遗留项）",
    )


def _key_ready(secrets: Any, provider: str) -> bool | None:
    """这把 key 在环境里配没配 —— **只看有没有，不回值**（key 从不进 Settings）。"""
    if not provider:
        return None
    return bool(getattr(secrets, f"{provider}_api_key", None)) if secrets is not None else None


# ---- 供应商（key / base_url / 模型清单）：2026-10-09 B 方案 ----


def _provider_view(
    key: str, settings: Any, stored: dict[str, Any] | None, secrets: Any, can_set_key: bool
) -> ProviderSettingView:
    """组装页面要的那一行。**绝不带 key**：只有 `has_key` 与指纹（sha256 前 8 位）。"""
    chat = settings.providers.chat.get(key)
    env_has = False
    if chat is not None and secrets is not None:
        env_has = bool(getattr(secrets, f"{chat.provider}_api_key", None))
    row = stored or {}
    return ProviderSettingView(
        provider_key=key,
        has_key=bool(row.get("has_key")) or env_has,
        api_key_fingerprint=str(row.get("api_key_fingerprint") or ""),
        base_url=str(row.get("base_url") or (chat.base_url if chat is not None else None) or ""),
        models=list(row.get("models") or []),
        updated_at=str(row.get("updated_at") or ""),
        updated_by=str(row.get("updated_by") or ""),
        can_set_key=can_set_key,
    )


@system_router.get("/providers", response_model=ProviderListView)
async def list_providers(components=Depends(get_components)) -> ProviderListView:
    """供应商清单 = `providers.chat` 里的家 ∪ 库里存过的家。**只读、不回 key。**

    `can_set_key` 看主密钥在不在（部署配置 `RAG_SECRET_KEY`，库里 `app_secret` 兜底）：
    不在就只能在页面上改 base_url / 模型清单，key 一律不收。
    """
    from rag.persistence.model_settings import build_model_setting_store
    from rag.persistence.secrets import load_master_key

    settings = components.settings
    store = build_model_setting_store(settings)
    stored = await asyncio.to_thread(store.providers)
    master = await asyncio.to_thread(load_master_key, store)
    secrets = getattr(components, "secrets", None)

    ordered = list(settings.providers.chat)
    ordered += [name for name in stored if name not in ordered]
    return ProviderListView(
        providers=[
            _provider_view(k, settings, stored.get(k), secrets, master is not None) for k in ordered
        ],
        note=(
            "库里的 key 只存密文，页面只回指纹（sha256 前 8 位）；主密钥放部署配置 "
            "RAG_SECRET_KEY（库里 app_secret 兜底）。换 model 走上面的「模型选型」。"
        ),
    )


@system_router.put("/providers/{provider_key}", response_model=ProviderListView)
async def set_provider(
    provider_key: str, body: ProviderUpdateRequest, components=Depends(get_components)
) -> ProviderListView:
    """写一家供应商（写库 + **写一行审计**）。key 三种语义：None=不动 / 空串=清除 / 给值=加密存。

    没配主密钥却想设 key → 422（当场告诉人"还差 RAG_SECRET_KEY"，而不是静默丢 key）。
    审计只记指纹，密文与明文都不落审计。
    """
    from rag.obs.log import current_ids
    from rag.persistence.model_settings import build_model_setting_store
    from rag.persistence.secrets import SecretError, load_master_key

    settings = components.settings
    store = build_model_setting_store(settings)
    master = await asyncio.to_thread(load_master_key, store)
    request_id, _ = current_ids()
    await asyncio.to_thread(store.setup)
    try:
        await asyncio.to_thread(
            store.set_provider,
            provider_key,
            base_url=body.base_url,
            models=body.models,
            api_key=body.api_key,
            master=master,
            actor=body.actor or "ui",
            request_id=request_id or "",
        )
    except (SecretError, ValueError) as exc:
        raise HTTPException(UNPROCESSABLE, str(exc)) from exc
    return await list_providers(components)  # 回整份列表，页面不用再拉一次


@system_router.get("/logs/tail", response_model=LogTailResponse)
async def logs_tail(
    kind: str = Query("all", description="all / interface / channel / system / agent_*"),
    limit: int = Query(200, ge=1, le=2000),
    components=Depends(get_components),
) -> LogTailResponse:
    """最近几行日志 —— **读文件**，所以 CLI 进程写出来的那些也在里面。

    与 `/logs/stream` 的分工：这里是**历史**（读 `data/logs/*.log`，开-读-关），
    那边是**实时**（进程内缓冲，只覆盖 API 自己）。两边合起来才是页面上看到的完整流水。
    """
    files = _log_files(components.settings)
    if kind != "all" and kind not in files:
        raise HTTPException(UNPROCESSABLE, f"没有这个类别：{kind}（可选：all、{'、'.join(files)}）")

    # 读文件、切行、判类别、合流水的活全在 `log.py` 的 `tail_items` 里 ——
    # `rag logs tail` 用的是同一个函数，两边不会各写一套解析。
    items = [LogLineView(**item) for item in tail_items(files, kind=kind, limit=limit)]
    return LogTailResponse(items=items, files=files)


def _log_store(components):
    """日志索引（PG）。**没配 PG 就没得查** —— 明确回 503 并指路 `/logs/tail`。

    为什么不做「没库就自动退回扫文件」：那会把两种语义混成一个接口 —— 扫文件是
    「最近 N 行里筛」，查库是「按 id 取整条链路」，用时看着一样、结果不一样才是坑。
    """
    if not components.settings.persistence.postgres.url:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "没配 PG（persistence.postgres.url 为空），日志只有文件那一份："
            "用「刷新」读 /logs/tail，或先跑 `rag logs init`。",
        )
    from rag.persistence.logs import build_log_store

    return build_log_store(components.settings, components.persistence)


def _moment(text: str, *, end: bool = False):
    """页面上传上来的时间：ISO 8601（`datetime-local` 那种不带时区的也认）。

    不带时区的按**本机时区**理解（页面上那个控件就是本地时间）；认不出来就 400 ——
    静默忽略时间条件会让人以为"筛过了"，实际看到的是全部。
    """
    from datetime import datetime

    stem = (text or "").strip()
    if not stem:
        return None
    try:
        parsed = datetime.fromisoformat(stem.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(UNPROCESSABLE, f"时间格式看不懂：{text!r}（要 ISO 8601）") from None
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    if end and parsed.hour == 0 and parsed.minute == 0 and parsed.second == 0:
        # 页面上只选到「日」时，`00:00` 应该是**那一整天**的末尾
        return parsed.replace(hour=23, minute=59, second=59, microsecond=999_999)
    return parsed


@system_router.get("/logs/search", response_model=LogSearchResponse)
async def logs_search(
    request_id: str = Query(..., min_length=1, description="必填：一次请求的关联 id"),
    category: str = Query(
        "", description="留空 = 全部；可选 interface / channel / agent_* / system"
    ),
    since: str = Query("", description="起始时间（ISO 8601，含）"),
    until: str = Query("", description="结束时间（ISO 8601，含）"),
    limit: int = Query(500, ge=1, le=2000),
    components=Depends(get_components),
) -> LogSearchResponse:
    """按 `request_id` 把一条链路从 PG 里拉出来（可选再按类别与时间收窄）。

    它是 2026-10-08 那次「日志上 PG」的查询口：文件一多就没法扫、更没有索引，
    而这里两个索引正好覆盖用法 —— `(request_id, ts)` 串链路、`(category, ts desc)`
    做「最近的 system」。

    **写入是异步批量的**（后台线程每 0.25 秒一波），所以刚发生的那一两行可能还没进库；
    页面上的「刷新」走的是文件（`/logs/tail`），永远是最新的 —— 两边互补。
    """
    files = _log_files(components.settings)
    store = _log_store(components)
    rows = await store.search(
        request_id=request_id.strip(),
        category=category.strip(),
        since=_moment(since),
        until=_moment(until, end=True),
        limit=limit,
    )
    items = [_log_row_view(row) for row in rows]
    note = (
        ""
        if items
        else (
            "库里没有这个 id 的行：历史不回溯（日志是 2026-10-08 才开始进库的，"
            "文件里那些老行不在库里），刚发生的一两行也可能还在批量队列里（每 0.25 秒一波）——"
            "要最新的一份就点「刷新」读文件。"
        )
    )
    return LogSearchResponse(
        request_id=request_id.strip(),
        category=category.strip(),
        items=items,
        files=files,
        note=note,
    )


def _log_row_view(row: dict) -> LogLineView:
    """库里的列 → 页面那一行。**与文件那一路同一个形状**，页面只写一套渲染。"""
    from datetime import datetime

    when = row.get("ts")
    if isinstance(when, datetime):
        local = when.astimezone()
        stamp = f"{local:%m-%d %H:%M:%S}.{int(local.microsecond / 1000):03d}"
    else:
        stamp = None
    logger = str(row.get("logger") or "")
    return LogLineView(
        kind=str(row.get("category") or "system"),
        ts=stamp,
        level=str(row.get("level") or ""),
        # 与文件那一路一致：这里放 logger 短名（`interface` / `pymilvus`…）
        source=logger.rsplit(".", 1)[-1],
        request_id=str(row.get("request_id") or ""),
        run_id=str(row.get("run_id") or ""),
        origin=str(row.get("source") or ""),
        text=str(row.get("text") or ""),
    )


@system_router.get("/logs/stream")
async def logs_stream(
    backfill: int = Query(200, ge=0, le=2000),
) -> StreamingResponse:
    """实时日志（SSE）：先补最近 `backfill` 行，再跟着推。

    **别改成「页面去 tail 文件」**：实测在 Windows 上，持着只读句柄会让
    `RotatingFileHandler` 的轮转失败（`WinError 32`），而且那一条日志会整条丢掉。
    实时这一层必须走内存（见 `obs/live.py`）；代价是只覆盖本进程 —— CLI 跑出来的
    日志要靠 `/logs/tail` 或页面手动刷新。
    """
    return StreamingResponse(
        _log_sse(backfill),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _log_sse(backfill: int):
    # `backfill=0` = 只要新行（页面上手动「清屏 / 从此刻开始跟」用的就是它）
    if backfill > 0:
        for item in live.recent(backfill):
            yield _event("log", item)
    with live.subscribe() as queue:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_S)
            except TimeoutError:
                yield ": ping\n\n"  # 注释行：保活，前端那套解析会忽略它
                continue
            yield _event("log", item)
