"""灌库、任务、文档工作流与资产端点。"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from rag.api.common import UNPROCESSABLE
from rag.api.deps import get_components
from rag.api.schemas import (
    ChunkSummaryView,
    ChunkView,
    DeletedView,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentStageView,
    DocumentView,
    FetchRequest,
    FolderAssignRequest,
    FolderChangeResponse,
    FolderCreateRequest,
    FolderUpdateRequest,
    FolderView,
    IndexSummaryView,
    IngestRequest,
    JobListView,
    JobView,
    OutlineRowView,
    OutlineUpdateRequest,
    OutlineView,
    ParseSummaryView,
    SectionSummaryView,
    UploadedView,
    UploadResponse,
)
from rag.contracts import reached

documents_router = APIRouter()


@documents_router.post("/ingest", response_model=JobView, status_code=status.HTTP_202_ACCEPTED)
async def ingest(body: IngestRequest, components=Depends(get_components)) -> JobView:
    """提交灌库 → 202 + job_id。真正的活在后台跑（OCR/VLM 是几十分钟级，HTTP 不能等）。"""
    for raw in body.paths:
        if not Path(raw).exists():
            raise HTTPException(UNPROCESSABLE, f"路径不存在：{raw}")
    job = components.jobs.spawn(kind="ingest", request=body.model_dump())
    await components.jobs.save(job)
    components.track(asyncio.create_task(components.run_ingest_job(job, body)))
    return JobView(**job.model_dump())


@documents_router.get("/jobs/{job_id}", response_model=JobView)
async def get_job(job_id: str, components=Depends(get_components)) -> JobView:
    job = await components.jobs.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"没有这个任务：{job_id}")
    return JobView(**job.model_dump())


# ---- 文档：上传 → 解析 → 人工核对标题 → 切分 → 入库（页面的那五步）----
#
# 两件事刻意分开：**上传只落盘**（快，不花 OCR/VLM 的钱），**解析是单独的 job**（慢，
# PDF 走 OCR/VLM 是分钟级）。这样页面能先看到文件、再逐个决定解不解析。


def _uploads(components):
    from rag.api.uploads import build_upload_store

    return build_upload_store(components.settings)


def _folders(components):
    """人工目录表（`_folders.json`）。**与上传件同一个落点**，所以这里各建一份。"""
    from rag.api.folders import build_folder_store

    return build_folder_store(components.settings)


def _stages(components):
    from rag.ingest.stages import build_artifact_store

    return build_artifact_store(components.settings)


def _assets(components):
    """图片资产（本地盘或 MinIO，看 `ingest.assets.backend`）。"""
    from rag.ingest.assets import build_asset_store

    return build_asset_store(components.settings)


# `data/stages/<doc_id>/document.md` 开头是一段 `<!-- … -->` 头信息（doc_id / source_uri /
# parse_key / 图片清单），给人查产物用的。**它不是正文**：左栏拿它当一段文字渲染、
# 右栏还把整块切成「第一节」，都是噪声（实测预览里就是这么冒出 172 个字符的）。
# 那些字段接口本来就单独返回了，所以这里剥掉。
_PREAMBLE_RE = re.compile(r"^\s*<!--.*?-->\s*", re.DOTALL)


def _markdown_of(components, doc_id: str) -> str:
    """解析稿正文（页面左栏渲染的就是它）。没这份文档 → 404；还没解析 → 409。"""
    if _uploads(components).path_of(doc_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"没有这份上传件：{doc_id}")
    path = _stages(components).path_of(doc_id, "document.md")
    if path is None or not path.is_file():
        raise HTTPException(status.HTTP_409_CONFLICT, "这份文档还没解析，先解析再来核对清单")
    return _PREAMBLE_RE.sub("", path.read_text(encoding="utf-8"), count=1)


def _doc_stages(
    components, doc_id: str, path, *, source_hash: str | None = None
) -> DocumentStageView:
    """按**本地产物**现算「走到哪一步」—— 注册表里没有这一行时的回退路径。

    正常情况状态在数据库那一列（`documents.py` 的 `stage`），列表与详情都读它；
    只有"库里有文件、注册表却没这一行"（迁移之前上传的、或行被清过）才走这里。
    `source_hash` 传进来就复用，不重复把整个文件再哈希一遍。
    """
    from rag.ingest.incremental import file_hash, pipeline_version
    from rag.ingest.stages import parse_key

    settings = components.settings
    store = _stages(components)
    parse = parse_key(settings, path.suffix)
    source_hash = source_hash or file_hash(path)
    parsed = store.load_document(doc_id, source_hash=source_hash, parse=parse) is not None
    outline = None
    if parsed:
        try:
            outline = store.load_outline(doc_id, source_hash=source_hash, parse=parse)
        except ValueError:
            outline = None  # 清单本身写坏了：当作「还没核对」，页面上会显示 notes
    reviewed = bool(outline and outline.confirmed)
    chunked = False
    if reviewed:
        chunked = (
            store.load_chunks(
                doc_id,
                source_hash=source_hash,
                pipeline=pipeline_version(settings),
                outline_hash=store.outline_fingerprint(doc_id),
            )
            is not None
        )
    return DocumentStageView(parsed=parsed, reviewed=reviewed, chunked=chunked)


def _doc_store(components):
    """文档注册表（PG / SQLite）。**2026-10-09 起不再走 Milvus**。

    见 `persistence/documents.py`：这层只是"每篇走到哪一步 + 入库信息"的台账，
    它不该让文档列表去等向量库（实测 Milvus 一抖就白等 9 秒）。
    """
    from rag.persistence.documents import build_document_store

    return build_document_store(components.settings)


async def _stages_map(components) -> dict[str, str] | None:
    """`doc_id → stage`：列表页要的**就是这一列**。**查不到回 None**（库够不着 ≠ 没数据）。"""
    try:
        return await _doc_store(components).stages()
    except Exception:  # noqa: BLE001 —— 只是状态未知，不该让整个列表接口 500
        return None


def _stage_view(stage: str) -> DocumentStageView:
    """一列状态 → 页面上那四个灯。**不翻产物、不算哈希**（那是详情页的活）。"""
    return DocumentStageView(
        parsed=reached(stage, "parsed"),
        reviewed=reached(stage, "reviewed"),
        chunked=reached(stage, "chunked"),
        indexed=reached(stage, "indexed"),
    )


async def _mark_stage(
    components, doc_id: str, stage: str, *, source_uri: str = "", content_hash: str = ""
) -> None:
    """把一份文档推进到 `stage`。**写不进去不打断主流程** —— 注册表是台账，文件才是本体。

    代价是：库抖的那一小会儿，列表上这份会退回"什么都没做"。这与"状态未知就说不知道"
    是同一个取舍（宁可少说，别把上传/核对直接判死），所以这里记一条 WARNING 而不是抛。
    """
    try:
        await _doc_store(components).set_stage(
            doc_id, stage, source_uri=source_uri, content_hash=content_hash
        )
    except Exception as exc:  # noqa: BLE001 —— 台账写失败不该让上传 / 核对 500
        _warn_ingest(f"文档状态没写进注册表：{doc_id} → {stage}（{type(exc).__name__}: {exc}）")


def _parse_summary(components, doc_id: str, *, parse: str, current_hash: str):
    """第二步的产物摘要。

    **过期的那份也给出来**：`load_document` 遇到哈希对不上会返回 None（步骤灯回退），
    但页面得说清**为什么**回退 —— 源文件变了，还是换了解析器。
    """
    from rag.ingest.summary import parse_stats

    payload = _stages(components).read_payload(doc_id, "document.json")
    if payload is None:
        return None
    recorded = str(payload.get("file_hash") or "")
    recorded_parse = str(payload.get("parse_key") or "")
    return ParseSummaryView(
        at=int(payload.get("saved_at") or 0),
        parse_key=recorded_parse,
        recorded_hash=recorded,
        current_hash=current_hash,
        stale=recorded != current_hash or recorded_parse != parse,
        **parse_stats(payload.get("document") or {}),
    )


def _chunk_summary(components, doc_id: str, *, current_hash: str, pipeline: str, outline_hash: str):
    """第四步「拆分」那半的摘要：块数 / 字数 / 分布 / 大纲 / 产物大小。"""
    from rag.ingest.summary import chunk_stats

    store = _stages(components)
    payload = store.read_payload(doc_id, "chunks.json")
    if payload is None:
        return None
    stats = chunk_stats(payload.get("chunks") or [])
    sizes: dict[str, int] = {}
    for name in ("chunks.json", "chunks.md"):
        artifact = store.path_of(doc_id, name)
        if artifact is not None and artifact.is_file():
            sizes[name] = artifact.stat().st_size
    return ChunkSummaryView(
        at=int(payload.get("saved_at") or 0),
        blocks=stats["blocks"],
        chars=stats["chars"],
        avg_chars=stats["avg_chars"],
        max_chars=stats["max_chars"],
        text_blocks=stats["text_blocks"],
        image_blocks=stats["image_blocks"],
        sizes=stats["sizes"],
        sections=[SectionSummaryView(**item) for item in stats["sections"]],
        artifact_bytes=sizes,
        recorded_hash=str(payload.get("file_hash") or ""),
        current_hash=current_hash,
        stale=(
            str(payload.get("file_hash") or "") != current_hash
            or str(payload.get("pipeline_version") or "") != pipeline
            or str(payload.get("outline_hash") or "") != outline_hash
        ),
    )


def _index_summary(components, record, *, blocks: int) -> IndexSummaryView:
    """第四步「入库」那半：拿注册表那一行说话（没有那一行 / 没走到 indexed = 没入库）。

    注册表在数据库里（`persistence/documents.py`），所以这一步**不再问 Milvus** ——
    向量库够不着的时候，文档页照样能把"这份灌了多少块、什么时候灌的"说清楚。
    """
    from rag.ingest.summary import vector_bytes

    dim = int(getattr(components.settings.providers.embedding, "dim", 0) or 0)
    if record is None or not reached(record.stage, "indexed"):
        return IndexSummaryView(vector_bytes=vector_bytes(blocks, dim), vector_dim=dim)
    return IndexSummaryView(
        indexed=True,
        chunk_count=record.chunk_count,
        ingested_at=record.ingested_at,
        pipeline_version=record.pipeline_version,
        # 注册表说灌了 N 块、产物里也是 N 块 → 对得上。对不上就是「没灌全」
        in_sync=record.chunk_count == blocks,
        vector_bytes=vector_bytes(blocks, dim),
        vector_dim=dim,
    )


def _uploaded(item) -> UploadedView:
    """`Upload` 描述 → 返回体。上传与抓取两条路共用一份，字段不会两边漂。"""
    return UploadedView(
        doc_id=item.doc_id,
        name=item.name,
        display_name=item.display_name,
        size=item.size,
        replaced=item.replaced,
        sha1=item.sha1,
        source_url=item.source_url,
        content_type=item.content_type,
        created_at=item.created_at,
    )


def _conflicts(previous: list, saved) -> list[UploadedView]:
    """同名、**内容不一样**的旧版本 → 交给页面去问。

    同内容的那一份不算冲突（`save` 已把它算进 `replaced`）：那只是重复上传，
    悄无声息地幂等掉才对。
    """
    return [_uploaded(old) for old in previous if old.sha1 != saved.sha1]


def _warn_ingest(message: str) -> None:
    """抓取没抓全（图没下来之类）：**按 WARNING 记**。

    它不影响这次请求成功，但少几张图会让检索结果悄悄变差 —— 所以除了回给页面，
    日志里也要留一条（`rag.*` 里非业务类别都进 `system.log`）。
    """
    import logging

    logging.getLogger("rag.ingest").warning("%s", message)


@documents_router.post("/documents/upload", response_model=UploadResponse, status_code=201)
async def upload_documents(
    files: list[UploadFile] = File(...), components=Depends(get_components)
) -> UploadResponse:
    """上传文档（multipart，可多份）。**只落盘、不解析。**

    落盘名带内容哈希（`<真名>-<sha1 前 16 位><扩展名>`），所以：

    * **同一份内容重传** = 同一个文件名 = 同一个 doc_id，原地 upsert，`replaced=true`，
      已解析/已核对/已切分的产物照旧有效（`file_hash` 没变）；
    * **同名不同内容** = 两份文件，互不覆盖；旧的那几份原样放进 `conflicts`，
      页面拿它问「覆盖更新还是保留两份」（覆盖 = `DELETE /documents/{旧 doc_id}`）；
    * 被拒的（后缀不支持 / 超过上限）连同原因放进 `skipped`，不静默丢。
    """
    from rag.api.uploads import UploadError

    store = _uploads(components)
    uploads: list[UploadedView] = []
    conflicts: list[UploadedView] = []
    skipped: list[str] = []
    for item in files:
        name = item.filename or ""
        data = await item.read()
        try:
            previous = store.by_display(name)
            saved = store.save(name, data)
        except UploadError as exc:
            skipped.append(f"{name or '?'}：{exc}")
            continue
        uploads.append(_uploaded(saved))
        conflicts.extend(_conflicts(previous, saved))
        # 台账第一格：这份上传件已经在盘上了（后面解析/核对/切分/入库各推一格）。
        # **内容哈希这一格就写上**：`plan_incremental` 靠它判"这份变了没有"，
        # 只在入库时写的话，解析完还没入库的文档会被当成变了、白重跑一遍 OCR。
        await _mark_stage(
            components, saved.doc_id, "uploaded", source_uri=saved.name, content_hash=saved.sha1
        )
    return UploadResponse(uploads=uploads, skipped=skipped, conflicts=conflicts)


@documents_router.post("/documents/fetch", response_model=UploadResponse, status_code=201)
async def fetch_document(body: FetchRequest, components=Depends(get_components)) -> UploadResponse:
    """从链接导入一份文档：抓回来**当普通上传件落盘**，后面五步一模一样。

    **同步执行**，不排后台任务 —— 抓一页是秒级，超时上限在
    `ingest.uploads.fetch.timeout_s`（默认 15s）。抓的是同步 httpx，所以丢线程池，
    别占着事件循环。

    类型不定（HTML 文档页 / PDF / Word 下载链接都行），后缀按
    `Content-Disposition` → URL 末段 → `Content-Type` 定；三者都定不出来就 400 并说明
    服务器给的 Content-Type 是什么。远端不给（403/超时/太大）一律 502，原因原样带出来。

    落盘之后与上传件完全同构：同名不同内容同样进 `conflicts`，由页面问要不要覆盖。

    抓的是 HTML 时，**页面里的图片会一起抓下来**（落在同级的 `<真名>-<页面哈希>_files/`
    目录，HTML 里的 `src` 改写成相对路径），解析时它们就成了正常的图片块。没抓到的图
    放进 `notes` 回给页面，同时写一条 `rag.ingest` 的 WARNING（进 `system.log`）。
    """
    from rag.api.uploads import UploadError
    from rag.ingest.fetch import FetchError, fetch_url

    cfg = components.settings.ingest.uploads
    store = _uploads(components)
    # 网页交给谁抓：`fetch.backend=http`（默认，本机）还是 `firecrawl`（云端，按量计费）。
    # **只有网页**会走 Firecrawl —— PDF / 图片那几条路要的是原件，见 ingest/fetch.py。
    fetch_options = None
    if cfg.fetch.backend == "firecrawl":
        from rag.ingest.firecrawl import FirecrawlOptions

        fetch_options = FirecrawlOptions(
            api_key=getattr(components.secrets, "firecrawl_api_key", None) or "",
            api_url=cfg.fetch.firecrawl_api_url,
            only_main_content=cfg.fetch.only_main_content,
            timeout_s=cfg.fetch.firecrawl_timeout_s,
        )
    try:
        fetched = await asyncio.to_thread(
            fetch_url,
            body.url,
            max_bytes=int(cfg.max_mb) * 1_000_000,
            timeout_s=cfg.fetch.timeout_s,
            max_redirects=cfg.fetch.max_redirects,
            allow_private_hosts=cfg.fetch.allow_private_hosts,
            max_images=cfg.fetch.max_images,
            max_image_bytes=int(cfg.fetch.max_image_mb) * 1_000_000,
            backend=cfg.fetch.backend,
            firecrawl=fetch_options,
        )
    except FetchError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    # 自己给了名字但不带后缀 → 按抓到的类型补上（"官网手册" + ".pdf"）
    name = body.name.strip() or fetched.name
    if body.name.strip() and not Path(name).suffix:
        name = f"{name}{Path(fetched.name).suffix}"
    try:
        previous = store.by_display(name)
        saved = store.save(
            name,
            fetched.data,
            source_url=body.url,
            final_url=fetched.final_url,
            content_type=fetched.content_type,
            attachments=fetched.attachments,
        )
    except UploadError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    if fetched.notes:
        # 图没抓全不能只回给页面：这类缺失会一路影响检索结果，日志里也要留一条
        _warn_ingest(f"{body.url}：{len(fetched.notes)} 张图没拿到 —— {'；'.join(fetched.notes)}")
    await _mark_stage(
        components, saved.doc_id, "uploaded", source_uri=saved.name, content_hash=saved.sha1
    )
    return UploadResponse(
        uploads=[_uploaded(saved)], conflicts=_conflicts(previous, saved), notes=fetched.notes
    )


@documents_router.delete("/documents/{doc_id}", response_model=DeletedView)
async def delete_document(
    doc_id: str,
    purge: bool = Query(True, description="连已入库的块与解析产物一起清掉"),
    components=Depends(get_components),
) -> DeletedView:
    """删掉一份上传件。**不可逆**（文件、已入库内容、解析产物都会没）。

    `purge=true`（默认）走 `indexing.commit.drop_docs`：先删 Milvus 里的块、再删注册行、
    最后清 `data/stages/<doc_id>/` 产物 —— 顺序不能反（见那个函数的说明），所以文件放在
    最后删。**清理失败就什么都不删**并返回 502：留着一份能重试的状态，比删了一半强。

    `purge=false` 只删盘上的文件、留着已入库的块（极少用：想让某份文档从页面消失、
    但检索结果先保持原样）。要彻底摘干净就别用它。

    页面上的「覆盖更新」就是先导入新版本、再对它旧的那一版调这个接口。
    """
    store = _uploads(components)
    if store.path_of(doc_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"没有这份上传件：{doc_id}")
    if purge:
        from rag.indexing.commit import drop_docs

        try:
            await drop_docs(
                components.settings, _stages(components), [doc_id], index=components.index
            )
        except Exception as exc:  # noqa: BLE001 —— 清库这一步失败要说清，且不动文件
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                f"清理已入库内容失败（Milvus / 注册表够不着？）：{type(exc).__name__}: {exc}。"
                f"文件没有删 —— 修好之后再用 purge=false 只删文件。",
            ) from exc
    store.delete(doc_id)
    # 归属跟着走：`_folders.json` 里留着悬空 doc_id，删掉的文档会以「未知文档」的
    # 身份一直挂在目录里 —— 那是记给人看的台账，不该留这种记录
    _folders(components).forget([doc_id])
    return DeletedView(deleted=doc_id)


@documents_router.get("/documents", response_model=DocumentListResponse)
async def list_documents(components=Depends(get_components)) -> DocumentListResponse:
    """上传件清单 + 每一份走到哪一步 + **人工目录树**。

    目录跟着清单一起回（`folders`）：页面画左栏那棵树要的就是这两份数据，
    分成两个请求只会多一次往返、还可能画出中间态。
    """
    store = _uploads(components)
    folders = _folders(components)
    members = folders.members()
    # **只读注册表那一列**：不翻每份的产物、不算哈希、不碰 Milvus。
    # 以前这里对每份文档都 `path_of`（O(N²) 扫目录）+ `file_hash` + 读三份 JSON，
    # 18 份就多花 0.37s；那些都是**详情页**该干的活。
    stages_map = await _stages_map(components)
    documents: list[DocumentView] = []
    for item in store.list():
        documents.append(
            DocumentView(
                doc_id=item.doc_id,
                name=item.name,
                display_name=item.display_name,
                size=item.size,
                sha1=item.sha1,
                stages=_stage_view((stages_map or {}).get(item.doc_id, "")),
                source_url=item.source_url,
                content_type=item.content_type,
                created_at=item.created_at,
                attachments=list(item.attachments),
                folder=members.get(item.doc_id, ""),
            )
        )
    # **说清「没入库」是查过之后的结论，还是根本没查成**（库够不着）—— 两者在页面上
    # 长得一样，但一个是事实、一个是不知道。
    return DocumentListResponse(
        documents=documents,
        indexed_known=stages_map is not None,
        folders=[
            FolderView(id=item.id, name=item.name, parent=item.parent) for item in folders.tree()
        ],
    )


# ---- 人工目录（只改归类，不碰文件 / doc_id / Milvus）----


@documents_router.post(
    "/documents/folders",
    response_model=FolderView,
    status_code=status.HTTP_201_CREATED,
)
async def create_folder(
    body: FolderCreateRequest, components=Depends(get_components)
) -> FolderView:
    """新建目录。`parent` 为空 = 顶层；**最多三层**，建第 4 层当场 400。"""
    from rag.api.folders import FolderError

    try:
        folder = _folders(components).create(body.name, body.parent)
    except FolderError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return FolderView(id=folder.id, name=folder.name, parent=folder.parent)


@documents_router.patch("/documents/folders/{folder_id}", response_model=FolderView)
async def update_folder(
    folder_id: str, body: FolderUpdateRequest, components=Depends(get_components)
) -> FolderView:
    """改名、搬家，或一起做。**没传 `parent` 就是只改名** —— 传 `null` 才是搬到顶层。

    两件事都当场校验：同级重名、搬进自己的子孙、搬完会超过三层，一律 400 说清原因。
    """
    from rag.api.folders import FolderError

    try:
        folder = _folders(components).update(
            folder_id,
            name=body.name,
            parent=body.parent,
            move="parent" in body.model_fields_set,
        )
    except FolderError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return FolderView(id=folder.id, name=folder.name, parent=folder.parent)


@documents_router.delete("/documents/folders/{folder_id}", response_model=FolderChangeResponse)
async def delete_folder(folder_id: str, components=Depends(get_components)) -> FolderChangeResponse:
    """删目录。里面的文档**回「未分类」**（不删文件）；还有子目录就 409，先清子目录。

    返回 `moved` = 有几份文档被移回了未分类 —— 页面拿它说清这一下影响了几份。
    """
    from rag.api.folders import FolderError

    try:
        moved = _folders(components).delete(folder_id)
    except FolderError as exc:
        # 有子目录 → 409（状态冲突：清掉子目录之后就能删），其余是输入问题 → 400
        code = status.HTTP_409_CONFLICT if "子目录" in str(exc) else status.HTTP_400_BAD_REQUEST
        raise HTTPException(code, str(exc)) from exc
    return FolderChangeResponse(moved=moved)


@documents_router.put("/documents/folders/members", response_model=FolderChangeResponse)
async def assign_folder(
    body: FolderAssignRequest, components=Depends(get_components)
) -> FolderChangeResponse:
    """把几份文档放进一个目录；`folder_id` 为空 = 移回「未分类」。返回 `moved`。

    **doc_id 要真的存在**：写进 `_folders.json` 的悬空 id 没人会去清（文档删掉时
    `forget` 只清盘上还在的那些），所以在这里就挡掉。
    """
    from rag.api.folders import FolderError

    store = _uploads(components)
    known = {item.doc_id for item in store.list()}
    unknown = [doc_id for doc_id in body.doc_ids if doc_id not in known]
    if unknown:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"没有这些上传件：{'、'.join(unknown[:5])}")
    if body.folder_id is not None and not any(
        item.id == body.folder_id for item in _folders(components).tree()
    ):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"没有这个目录：{body.folder_id}")
    try:
        moved = _folders(components).assign(body.doc_ids, body.folder_id)
    except FolderError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return FolderChangeResponse(moved=moved)


@documents_router.get("/documents/{doc_id}", response_model=DocumentDetailResponse)
async def get_document(doc_id: str, components=Depends(get_components)) -> DocumentDetailResponse:
    """一份文档的详情：**解析稿（只读）** + 标题清单 + 步骤状态 + 每个阶段「做到了什么」。

    页面上那条流程图与下面对应的阶段面板全靠它：`parse` / `chunks` / `index` 三份摘要
    都是**现算**的（读产物 json），不写进产物格式 —— 报表不是数据。

    过期的产物也照给（`stale=true`）：只报「没到那一步」的话，步骤灯会毫无理由地回退，
    看的人不知道是自己改坏了还是换了配置。
    """
    from rag.ingest.incremental import file_hash, pipeline_version
    from rag.ingest.outline import read_rows
    from rag.ingest.stages import parse_key

    store = _uploads(components)
    path = store.path_of(doc_id)
    if path is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"没有这份上传件：{doc_id}")
    described = store.find(doc_id)
    settings = components.settings
    artifacts = _stages(components)
    parse = parse_key(settings, path.suffix)
    source_hash = file_hash(path)
    # 详情页**按产物现算**（它本来就看得起这一步的代价）：源文件变了、清单改过、
    # 换了解析器，产物就对不上，灯要如实灭掉 —— 台账那列只说明"曾经做过"。
    record = await _doc_store(components).get(doc_id)
    stages = _doc_stages(components, doc_id, path, source_hash=source_hash)
    if record is not None:
        # 「入库」这一格**只有注册表说了算**：产物齐不代表真灌进去了（块在 Milvus 里）
        stages.indexed = reached(record.stage, "indexed")
    parse_summary = _parse_summary(components, doc_id, parse=parse, current_hash=source_hash)
    chunk_summary = _chunk_summary(
        components,
        doc_id,
        current_hash=source_hash,
        pipeline=pipeline_version(settings),
        outline_hash=artifacts.outline_fingerprint(doc_id),
    )
    index_summary = _index_summary(
        components, record, blocks=chunk_summary.blocks if chunk_summary else 0
    )
    markdown = ""
    outline: OutlineView | None = None
    if stages.parsed:
        md_path = artifacts.path_of(doc_id, "document.md")
        if md_path is not None and md_path.is_file():
            # 剥掉头部那段 `<!-- doc_id … metadata … -->`（**产物文件里不动**：它是给人查
            # parse_key / parser / 页数用的）。这段在核对那一步本来就剥了（`_markdown_of`），
            # 解析阶段的左栏却会把它当正文显示出来（2026-10-10 使用者报的）。
            markdown = _PREAMBLE_RE.sub("", md_path.read_text(encoding="utf-8"), count=1)
        tsv_path = artifacts.path_of(doc_id, "outline.tsv")
        if tsv_path is not None and tsv_path.is_file():
            text = tsv_path.read_text(encoding="utf-8")
            try:
                found = artifacts.load_outline(doc_id, source_hash=source_hash, parse=parse)
            except ValueError as exc:
                outline = OutlineView(notes=[str(exc)])
            else:
                outline = OutlineView(
                    rows=[
                        OutlineRowView(level=level, hint=hint, title=title)
                        for level, hint, title in read_rows(text)
                    ],
                    max_level=found.max_level if found else 3,
                    confirmed=bool(found and found.confirmed),
                    notes=list(found.notes) if found else [],
                )
    return DocumentDetailResponse(
        doc_id=doc_id,
        name=path.name,
        display_name=described.display_name if described else path.name,
        size=path.stat().st_size,
        sha1=described.sha1 if described else source_hash,
        source_uri=path.name,
        source_url=described.source_url if described else "",
        content_type=described.content_type if described else "",
        created_at=described.created_at if described else 0,
        markdown=markdown,
        outline=outline,
        stages=stages,
        parse=parse_summary,
        chunks=chunk_summary,
        index=index_summary,
    )


@documents_router.get("/documents/{doc_id}/source")
async def get_document_source(doc_id: str, components=Depends(get_components)):
    """原始上传件的字节（页面「上传」那一步要看原文）。

    两处刻意的选择：

    * **只认 `doc_id`**，路径由上传存储查出来（`path_of`）—— 不接受任何路径参数，
      所以不存在「顺着参数读到目录外面去」这条路。
    * **`text/*` 里只有 html 原样给，其余降级成 `text/plain`**：浏览器遇到
      `text/markdown` 是下载而不是显示，页面上要看的是内容；HTML 要保持 `text/html`
      才能在沙箱 iframe 里渲染出排版。

    抓回来的是**第三方页面**，所以除了 `nosniff` 还带一条 `CSP: sandbox`：
    前端用 `<iframe sandbox="">`（不带 allow-scripts），这里再兜一层。
    """
    from fastapi.responses import FileResponse

    from rag.ingest.loaders.base import mime_for

    path = _uploads(components).path_of(doc_id)
    if path is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"没有这份上传件：{doc_id}")
    mime = mime_for(path)
    if mime.startswith("text/") and mime != "text/html":
        mime = "text/plain; charset=utf-8"
    return FileResponse(
        path,
        media_type=mime,
        headers={
            "Content-Disposition": f"inline; filename*=UTF-8''{quote(path.name)}",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "sandbox",
        },
    )


@documents_router.post("/documents/{doc_id}/parse", response_model=JobView, status_code=202)
async def parse_document(doc_id: str, components=Depends(get_components)) -> JobView:
    """解析这一份上传件 → 202 + job_id（PDF 走 OCR/VLM 是分钟级，页面轮询 `/jobs/{id}`）。"""
    path = _uploads(components).path_of(doc_id)
    if path is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"没有这份上传件：{doc_id}")
    job = components.jobs.spawn(kind="parse", request={"doc_id": doc_id, "name": path.name})
    await components.jobs.save(job)
    components.track(asyncio.create_task(components.run_parse_job(job, doc_id)))
    return JobView(**job.model_dump())


async def _ensure_index_job(components, doc_id: str, path: Path) -> tuple[JobView, bool]:
    """排一个「切分 + 入库」任务；**同一份文档已经在跑就复用那一个**。

    返回 `(任务, 是不是这次新排的)`。两处调它：人点「拆分并入库」
    （`POST /documents/{id}/index`）、以及勾了「已核对」提交之后的自动那一跑
    （`api.index_after_review`）。复用是为了防重复干活、防重复花钱 —— 切分要给每张图问
    VLM、入库要算 embedding，都是按量计费的动作；连点两次按钮不该跑两遍。
    """
    for job in await components.jobs.list(limit=50):
        if job.kind != "index" or job.status not in ("queued", "running"):
            continue
        if str((job.request or {}).get("doc_id") or "") == doc_id:
            return JobView(**job.model_dump()), False
    job = components.jobs.spawn(kind="index", request={"doc_id": doc_id, "name": path.name})
    await components.jobs.save(job)
    components.track(asyncio.create_task(components.run_index_job(job, doc_id)))
    return JobView(**job.model_dump()), True


@documents_router.put("/documents/{doc_id}/outline", response_model=DocumentDetailResponse)
async def save_document_outline(
    doc_id: str, body: OutlineUpdateRequest, components=Depends(get_components)
) -> DocumentDetailResponse:
    """提交人工核对结果（页面第三步）。**只改层级，不碰正文。**

    条目数必须与解析稿里的标题行数一致：清单只该改层级、不该增删行（增删会让清单与正文
    对不上，`sections` 那边会直接抛）。对不上就 400 并说清差多少 —— 让它悄悄写下去，
    下一步切分会拿着一条错的路径走。

    没写「已核对: 是」时**照样写得进去**，只是 `run_chunk` 会拦住它（`/index` 也返回 409）。
    这样「改一半先存着」是允许的，只是切不了。

    **勾了「已核对」就直接往下走**（`api.index_after_review`，默认开）：提交即排一个
    「切分 + 入库」任务，响应里带回 `index_job`。四步流程里最后那次点击没有新决策 ——
    切分只认核对过的清单，而它刚被核对完。关掉那个开关就回到手动点。
    """
    from rag.ingest.incremental import file_hash
    from rag.ingest.outline import heading_lines, render_reviewed_outline
    from rag.ingest.stages import parse_key

    path = _uploads(components).path_of(doc_id)
    if path is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"没有这份上传件：{doc_id}")
    settings = components.settings
    artifacts = _stages(components)
    markdown = _markdown_of(components, doc_id)
    headings = heading_lines(markdown)
    if len(body.rows) != len(headings):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"清单提交了 {len(body.rows)} 条，解析稿里有 {len(headings)} 个标题行："
            f"清单只该改层级、不该增删行",
        )
    text = render_reviewed_outline(
        markdown,
        [row.level for row in body.rows],
        max_level=body.max_level,
        confirmed=body.confirmed,
    )
    try:
        artifacts.save_outline(
            doc_id, text, source_hash=file_hash(path), parse=parse_key(settings, path.suffix)
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    # 存下清单 = 走到「核对」这一步（确认与否是清单自己的字段，由 `/index` 那道门槛管）
    await _mark_stage(
        components, doc_id, "reviewed", source_uri=path.name, content_hash=file_hash(path)
    )
    detail = await get_document(doc_id, components)
    if body.confirmed and settings.api.index_after_review:
        job, fresh = await _ensure_index_job(components, doc_id, path)
        detail.index_job = job
        if fresh:
            import logging

            logging.getLogger("rag.ingest").info(
                "《%s》已核对，自动开始拆分并入库（%s）；要关掉这个自动步骤就把 "
                "api.index_after_review 设成 false",
                path.name,
                job.id,
            )
    return detail


@documents_router.post("/documents/{doc_id}/index", response_model=JobView, status_code=202)
async def index_document(doc_id: str, components=Depends(get_components)) -> JobView:
    """切分 + 灌 Milvus（页面第五步）→ 202 + job_id，进度靠 `GET /jobs` 轮询。

    **核对是硬门槛**：清单上没写「已核对: 是」直接 409，不进后台任务 —— 让人在页面上
    立刻看到差哪一步，比等 job 跑完再报失败清楚。
    """
    from rag.ingest.incremental import file_hash
    from rag.ingest.stages import parse_key

    path = _uploads(components).path_of(doc_id)
    if path is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"没有这份上传件：{doc_id}")
    settings = components.settings
    try:
        outline = _stages(components).load_outline(
            doc_id, source_hash=file_hash(path), parse=parse_key(settings, path.suffix)
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, f"标题清单有问题：{exc}") from exc
    if outline is None or not outline.confirmed:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "这份文档的标题清单还没核对（outline.tsv 里「已核对」要是「是」）",
        )
    job, _fresh = await _ensure_index_job(components, doc_id, path)
    return job


@documents_router.get("/jobs", response_model=JobListView)
async def list_jobs(
    limit: int = Query(20, ge=1, le=200), components=Depends(get_components)
) -> JobListView:
    """最近的任务，新的在前（页面靠它轮询进度）。"""
    rows = await components.jobs.list(limit=limit)
    return JobListView(jobs=[JobView(**row.model_dump()) for row in rows])


@documents_router.get("/assets/{path:path}")
async def get_asset(path: str, components=Depends(get_components)):
    """取原始资产（图片等）。前端「引用里点开那张图」用它。

    安全上有两件事必须做（这个端点是**直接把文件吐出去**）：

    1. **键里不许出现 `..` / 绝对路径**（现在是键 → 字节的读法，两个后端共用这条判据；
       以前那条 `resolve()` 之后必须还在根目录内，是本地盘时代的等价物）；
    2. 鉴权由 router 统一那道 `X-API-Key` 管（本文件顶部 `dependencies=[...]`）。

    **2026-10-09 起不再自己拼路径**：资产可能在 MinIO 桶里（`ingest.assets.backend=minio`），
    所以统一走 `AssetStore`。走内存而不是流式：资产是图，上限本来就是几 MB。
    """
    import mimetypes

    from fastapi.responses import Response

    from rag.storage import BlobNotFound

    key = path.strip("/")
    if not key or ".." in key.split("/"):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"没有这个资产：{path}")
    try:
        data = _assets(components).read(key)
    except BlobNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"没有这个资产：{path}") from None
    media_type = mimetypes.guess_type(key)[0] or "application/octet-stream"
    return Response(data, media_type=media_type)


@documents_router.get("/chunks/{chunk_id}", response_model=ChunkView)
async def get_chunk(chunk_id: str, components=Depends(get_components)) -> ChunkView:
    """按 id 取一个块的原文（前端点引用看原文用）。

    走 Milvus 的按主键查询（`index.get_chunks`），不重新检索、不 embed。
    """
    components._require("embedding")  # noqa: SLF001 —— 与其它端点同一个前置检查
    chunks = await components.index.get_chunks([chunk_id])
    if not chunks:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"没有这个块：{chunk_id}")
    chunk = chunks[0]
    return ChunkView(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        text=chunk.text,
        doc_type=chunk.doc_type,
        modality=chunk.modality,
        source_uri=str(chunk.metadata.get("source_uri", "")),
        heading_path=chunk.heading_path,
        heading_numbers=chunk.heading_numbers,
        page=chunk.page,
        line=chunk.line,
        assets=chunk.assets,
        index=chunk.index,
    )
