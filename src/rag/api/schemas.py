"""请求 / 响应模型。字段与语义尽量与 CLI 那几个选项一一对应。

刻意**不**在这里重复一遍校验之外的东西：真正的活儿全在 retrieval / ingest 那两层，
这一层只负责「把 HTTP 的输入变成那两层的入参、再把结果变成 JSON」。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class HitView(BaseModel):
    """一条命中。字段是检索层现成的那些，不额外加工。"""

    chunk_id: str
    score: float
    sources: list[str] = Field(default_factory=list)
    doc_id: str = ""
    modality: str = "text"
    doc_type: str = ""
    source_uri: str = ""
    heading_path: str = ""
    heading_numbers: str = ""
    page: int = 0
    line: int = 0
    assets: list[str] = Field(default_factory=list)
    text: str = ""


class ChunkView(BaseModel):
    """单个块的原文（前端「点开引用看原文」用）。比 `HitView` 少检索字段，多完整正文。"""

    chunk_id: str
    doc_id: str = ""
    text: str = ""
    doc_type: str = ""
    modality: str = "text"
    source_uri: str = ""
    heading_path: str = ""
    heading_numbers: str = ""
    page: int = 0
    line: int = 0
    assets: list[str] = Field(default_factory=list)
    index: int = 0


class StatsView(BaseModel):
    """一路的账本（`--stats` 打的那张表）。"""

    channel: str
    candidates: int = 0
    kept: int = 0
    latency_ms: float | None = None
    error: str | None = None


class SearchRequest(BaseModel):
    query: str = ""
    # 以图搜图：服务端本地路径（真正对外时这里会换成 multipart 上传，见 §12 的说明）
    image_path: str | None = None
    top_k: int = Field(10, ge=1, le=100)
    filters: dict[str, Any] | None = None
    # **只跑列出来的那几路**（名字取自 `retrieval.channels[].name`，如 `milvus` / `kg`）；
    # 给空或不给 = 走配置里启用的全部。名字不认识回 422。
    channels: list[str] | None = None
    rerank: bool | None = None


class SearchResponse(BaseModel):
    query: str
    hits: list[HitView]
    stats: list[StatsView] = Field(default_factory=list)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(8, ge=1, le=50)
    stream: bool = False
    filters: dict[str, Any] | None = None
    channels: list[str] | None = None
    rerank: bool | None = None
    graph: bool | None = None
    # 会话 id：给了就接在那一段后面，没给就开一段新的（标题取首问）
    session_id: str | None = None


class CitationView(BaseModel):
    index: int
    chunk_id: str
    source_uri: str
    snippet: str = ""


class ImageUploadView(BaseModel):
    """上传一张图给「以图搜图」。**只回路径** —— 检索由 `/search` 的 `image_path` 发起。

    `path` 是**服务端路径**（页面把它原样回传给 `/search`），`name` 是落盘名（带哈希）。
    """

    path: str
    name: str = ""
    bytes: int = 0


class TranscribeResponse(BaseModel):
    """语音转文字的结果。**只有文字** —— 音频本身不落盘、不落库、不进会话。

    由前端放进输入框让人确认之后才提问，所以这里返回的是「草稿」，不是一轮问答。
    """

    text: str
    # 真正用的模型名：换供应商/换型号时，页面上那句提示能说清是谁转的
    model: str = ""
    chars: int = 0


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationView] = Field(default_factory=list)
    hits: list[HitView] = Field(default_factory=list)
    stats: list[StatsView] = Field(default_factory=list)
    rounds: int = 1
    grade: dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = None


class IngestRequest(BaseModel):
    """提交灌库。路径是**服务端本地路径**（API 不收上传流，那是另一件事）。"""

    paths: list[str] = Field(min_length=1)
    stage: Literal["parse", "chunk", "index", "all"] = "all"
    full: bool = False
    describe: bool | None = None
    graph: bool | None = None


class AgentRequest(BaseModel):
    """对话式 agent（M5）：主 agent 自己决定查什么、派给哪个子 agent。"""

    message: str = Field(min_length=1)
    # 会话 id：给了就是多轮（历史落在 PG 的 checkpointer 里，重启也还在）
    thread_id: str | None = None
    # **用户明确要求联网**：放行网络搜索。默认只在「语料检索查过且什么都没拿到」时才允许
    # （网络答案无法溯源到语料，见 agent/tools.py 的 WebSearchGate）
    web: bool = False
    # `true` 走 SSE：`token` / `tool` / `interrupt` / `done` 四种事件。默认不下发中间过程
    stream: bool = False
    # 与 `/chat` 同一个语义：这一段对话落在哪个 `chat_session` 上
    session_id: str | None = None


class AgentResponse(BaseModel):
    answer: str
    # 只列**答案里真的引用到**的那些
    citations: list[CitationView] = Field(default_factory=list)
    thread_id: str | None = None
    # 走过的消息条数（含工具调用）与本次召回总数 —— 判断「它到底查了没有」
    steps: int = 0
    retrieved: int = 0
    # `answered`（跑完了）或 `interrupted`（停在等人批准那一步）
    status: str = "answered"
    # 中断时「它想干什么」：`{"action_requests": [...], "review_configs": [...]}`
    interrupt: dict[str, Any] | None = None
    session_id: str | None = None


class AgentResumeRequest(BaseModel):
    """人批准之后接着跑。

    `decisions` **照搬 HITL 中间件的形状**（`{"type": "approve"|"edit"|"reject"|"respond"}`），
    一条 `action_requests` 对应一条，顺序要对 —— 与其自创一套再翻译，不如让它只有一层。
    `edit` 还要带 `args`，`respond` 要带 `message`，由中间件自己校验。

    `stream` 与 `/agent` 走**同一组事件**：批准之后的答案照样是一段段生成的。
    """

    thread_id: str = Field(min_length=1)
    decisions: list[dict[str, Any]] = Field(min_length=1)
    stream: bool = False
    session_id: str | None = None


class JobView(BaseModel):
    id: str
    kind: str
    status: str
    request: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    created_at: int = 0
    updated_at: int = 0


class SessionUsageView(BaseModel):
    """一段会话的 token 合计（`token_usage` 那张表聚合出来的）。"""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    # 这一共发生了多少次模型调用（agent 一轮会有好几次，快问答是 1）
    calls: int = 0


class SessionView(BaseModel):
    """会话列表里的一行。**与 `rag sessions`（checkpointer 的残留 thread）无关。**"""

    session_id: str
    title: str = ""
    mode: str = "chat"
    thread_id: str | None = None
    turns: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None
    usage: SessionUsageView = Field(default_factory=SessionUsageView)


class SessionMessageView(BaseModel):
    """一条消息。只会有 `human` 与 `ai` 两种角色。

    `status` 说明这条 ai 消息是怎么结束的：`answered`（跑完）/ `interrupted`（停在等人
    批准）/ `stopped`（人点了停止）/ `error`。中断那轮 `content` 是空的 ——
    「它想干什么」要去 checkpointer 现取，不在这张表里。
    """

    seq: int
    role: str
    content: str = ""
    status: str = "answered"
    created_at: datetime | None = None
    # **只在读的时候现取**（`status=interrupted` 的那条 ai 消息）：
    # `{"action_requests": [...], "review_configs": [...]}`，或 null（已经批过了就没有了）。
    # 中断载荷不落我们自己的表 —— 写文件的正文可能几 KB，存两份没意义。
    interrupt: dict[str, Any] | None = None
    # 这一轮的 token 用量（历史轮次也带回来，不然刷新之后逐轮的账就看不见了）
    usage: SessionUsageView | None = None


class SessionListResponse(BaseModel):
    sessions: list[SessionView] = Field(default_factory=list)


class SessionDetailResponse(BaseModel):
    """一段会话的全部消息 + token 合计。消息**已按 seq 排好**，前端直接渲染。"""

    session_id: str
    title: str = ""
    mode: str = "chat"
    thread_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    messages: list[SessionMessageView] = Field(default_factory=list)
    usage: SessionUsageView = Field(default_factory=SessionUsageView)


class DeletedView(BaseModel):
    """删除类操作的统一返回（`/chunks` 那种 404 之外，至少说清删掉了谁）。"""

    deleted: str


class DependencyView(BaseModel):
    """一个依赖的状态。`ok=false` 时 `detail` 要说清是**没配**还是**连不上**。"""

    ok: bool
    detail: str = ""


class UploadedView(BaseModel):
    """上传 / 抓取成功的一份。`doc_id` 由「相对上传目录的路径」算出，与 ingest 同一套。

    `name` 是**落盘名**（带内容哈希），`display_name` 是**真名** —— 页面显示后者，
    前端不去猜「把 -hash 切掉」（真名本身就可能带 `-1a2b3c4d5e6f7a8b` 这种尾巴）。
    """

    doc_id: str
    name: str
    display_name: str = ""
    size: int
    # **同一份内容**已经在了（同名同哈希）—— 如实告诉用户，别让人以为新增了一份
    replaced: bool = False
    # 内容的 sha1[:16]，与 file_hash（解析产物过期判据）是同一个值
    sha1: str = ""
    # 链接导入才有：用户给的 URL 与跟完重定向之后真正的 URL
    source_url: str = ""
    content_type: str = ""
    # 落盘时间（老的、加哈希之前传的文件没有这条记录，为 0）
    created_at: int = 0


class UploadResponse(BaseModel):
    uploads: list[UploadedView] = Field(default_factory=list)
    # 被拒的文件（后缀不支持 / 太大），连原因一起给出来
    skipped: list[str] = Field(default_factory=list)
    # **同名但内容不一样**的旧版本：页面拿它问「覆盖更新还是保留两份」。
    # 新版本已经落盘了（名字带哈希，不会互相覆盖），这里只是把冲突摆出来
    conflicts: list[UploadedView] = Field(default_factory=list)
    # 「拿到了但没拿全」的话：页面里哪些图没抓下来、为什么。**不静默丢**
    notes: list[str] = Field(default_factory=list)


class FetchRequest(BaseModel):
    """从链接导入一份文档。

    `url` 指向的东西类型不定：可能是 HTML 文档页，也可能是 PDF / Word 的下载链接
    （后缀按 Content-Disposition → URL 末段 → Content-Type 定，见 `ingest/fetch.py`）。

    `name` 可选：想自己定个真名就填；**不带后缀也没关系**，会照抓到的类型补上。
    """

    url: str
    name: str = ""


class DocumentStageView(BaseModel):
    """这份文档走到哪一步了（页面上的步骤条用它）。"""

    parsed: bool = False
    reviewed: bool = False
    chunked: bool = False
    indexed: bool = False


class FolderView(BaseModel):
    """一个目录。`parent` 为空 = 顶层；最多三层（`folders.MAX_DEPTH`）。"""

    id: str
    name: str
    parent: str | None = None


class DocumentView(BaseModel):
    doc_id: str
    name: str
    # 真名（页面显示这个）；`name` 是带哈希的落盘名
    display_name: str = ""
    size: int
    # 内容哈希（与「解析产物是否过期」用的同一把钥匙）
    sha1: str = ""
    stages: DocumentStageView = Field(default_factory=DocumentStageView)
    # 链接导入才有
    source_url: str = ""
    content_type: str = ""
    created_at: int = 0
    # 附属文件（抓页面时一起下下来的图片）的相对路径 —— 左边那棵树按它展开
    attachments: list[str] = Field(default_factory=list)
    # 人工目录的 id；空串 = 还没归类（页面上的「未分类」）。
    # **目录只是页面上的组织**：doc_id / Milvus / 解析产物一个都不受影响
    folder: str = ""


class DocumentListResponse(BaseModel):
    documents: list[DocumentView] = Field(default_factory=list)
    # 入库状态是不是"问过 Milvus"了。false = Milvus 够不着，页面别把「没入库」当成结论
    indexed_known: bool = True
    # 目录树（**扁平一层**：`parent` 指上去，怎么排是页面的事）。与文档一起返回，
    # 免得页面为了画一棵树打两次请求
    folders: list[FolderView] = Field(default_factory=list)


class FolderCreateRequest(BaseModel):
    """新建目录。`parent` 为空就是建在顶层。"""

    name: str
    parent: str | None = None


class FolderUpdateRequest(BaseModel):
    """改名 / 搬家。**只改名字就别传 `parent`** —— 传 `null` 是「搬到顶层」。"""

    name: str | None = None
    parent: str | None = None


class FolderAssignRequest(BaseModel):
    """把几份文档放进一个目录；`folder_id` 为空 = 移回「未分类」。"""

    doc_ids: list[str] = Field(default_factory=list)
    folder_id: str | None = None


class FolderChangeResponse(BaseModel):
    """目录操作的收尾数字：这次有几份文档被挪了归属。"""

    moved: int = 0


class OutlineRowView(BaseModel):
    """标题清单的一行：`level` 是**人工核定后的层级**，`hint` 是按编号推断的初值（供对照）。"""

    level: int
    hint: int
    title: str


class OutlineView(BaseModel):
    rows: list[OutlineRowView] = Field(default_factory=list)
    max_level: int = 3
    confirmed: bool = False
    notes: list[str] = Field(default_factory=list)


class OutlineRowUpdate(BaseModel):
    """人工核定的一行：**只有层级**。

    标题是解析稿里的原文，不在可改范围里 —— 改它等于改正文，而正文一动
    `Document.positions`（偏移 → 页码/行号）就全错位了。所以这里连字段都不给。
    """

    level: int = Field(ge=0, le=6)


class OutlineUpdateRequest(BaseModel):
    """提交人工核对结果（页面第三步）。`confirmed` 就是「已核对」那一行。"""

    rows: list[OutlineRowUpdate] = Field(default_factory=list)
    max_level: int = Field(default=3, ge=1, le=6)
    confirmed: bool = False


class ParseSummaryView(BaseModel):
    """第二步（解析）做到了什么 —— 页面上「解析完成」那一屏用它。"""

    at: int = 0
    chars: int = 0
    images: int = 0
    # 取不到的图与原因：**要一直看得见**，不能只在导入时闪一下提示条
    missing_images: list[str] = Field(default_factory=list)
    parser: str = ""
    mime: str = ""
    parse_key: str = ""
    # 产物记的源文件哈希 vs 现在盘上那份的哈希
    recorded_hash: str = ""
    current_hash: str = ""
    # 对不上 = 产物过期（换了源文件 / 换了解析器），要从第二步重来 —— 这就是「步骤灯为什么
    # 悄悄回退了」的答案
    stale: bool = False


class SectionSummaryView(BaseModel):
    """大纲里的一节：按 `heading_path` 聚合出来的「多少块、多少字」。

    `head` 是这一节**第一块的开头几十个字** —— 没有它，「（无标题）1 块 133 字」这种行
    谁也不知道说的是什么（实测那是文档站顶部的导航残留）。
    """

    heading_path: str = ""
    heading_numbers: str = ""
    blocks: int = 0
    chars: int = 0
    head: str = ""


class ChunkSummaryView(BaseModel):
    """第四步（拆分并入库）里「拆分」那半 —— 页面上「切分完成」那一屏用它。"""

    at: int = 0
    blocks: int = 0
    chars: int = 0
    avg_chars: int = 0
    max_chars: int = 0
    text_blocks: int = 0
    image_blocks: int = 0
    # 每块的字数：页面接成一条宽度正比于字数的色条，看切得匀不匀
    sizes: list[int] = Field(default_factory=list)
    # 大纲：哪几节、各多少块多少字
    sections: list[SectionSummaryView] = Field(default_factory=list)
    # 产物大小（`chunks.json` / `chunks.md`）
    artifact_bytes: dict[str, int] = Field(default_factory=dict)
    recorded_hash: str = ""
    current_hash: str = ""
    # 切法 / 清单 / 源文件变了 → 产物作废
    stale: bool = False


class IndexSummaryView(BaseModel):
    """第四步里「入库」那半 —— 拿 Milvus 注册表那一行说话。"""

    indexed: bool = False
    # 注册表记的块数
    chunk_count: int = 0
    ingested_at: int = 0
    pipeline_version: str = ""
    # 注册表块数 == 产物块数？对不上就是「没灌全」
    in_sync: bool = False
    # 稠密向量的**理论**占用（块数 × dim × 4B）：只是估算，不含索引与稀疏向量
    vector_bytes: int = 0
    vector_dim: int = 0


class DocumentDetailResponse(BaseModel):
    """一份文档的详情：解析稿（**只读**）+ 标题清单 + 步骤状态。"""

    doc_id: str
    name: str
    display_name: str = ""
    size: int
    sha1: str = ""
    source_uri: str = ""
    source_url: str = ""
    content_type: str = ""
    created_at: int = 0
    markdown: str = ""
    outline: OutlineView | None = None
    stages: DocumentStageView = Field(default_factory=DocumentStageView)
    # 勾了「已核对」提交时自动排的那个「切分 + 入库」任务（`api.index_after_review`）。
    # 页面拿到就切到「拆分入库中…」跟着轮询，不用再让人点一次。
    index_job: JobView | None = None
    # 每个阶段「做到了什么程度」：上传阶段看上面的字段，其余三步各有一份摘要
    parse: ParseSummaryView | None = None
    chunks: ChunkSummaryView | None = None
    index: IndexSummaryView | None = None


class JobListView(BaseModel):
    jobs: list[JobView] = Field(default_factory=list)


class LogLineView(BaseModel):
    """一行日志。`ts` / `level` / `source` 是从行里切出来的，切不动就留空、整行进 `text`。

    `source` 是 **logger 短名**（`storage` / `ocr` / `uvicorn` / `pymilvus`…）——
    页面上单列「来源」，省得看的人不知道这条是谁说的。

    `request_id` / `run_id` 单独拎出来（行首那两段 `req=… run=…`）：页面上要能点 ——
    点一下按这个 id 把同一条链路（interface + channel + agent）串出来。
    `origin` 是**落库那份才有**的入口标记（`http` / `cli` / `task` / `library`），
    文件那一路留空（文件里没有这一栏）。
    """

    kind: str
    ts: str | None = None
    level: str = ""
    source: str = ""
    request_id: str = ""
    run_id: str = ""
    origin: str = ""
    text: str


class LogTailResponse(BaseModel):
    """最近几行日志。`files` 是类别 → 文件路径（页面上写清「读的是哪个文件」）。"""

    items: list[LogLineView] = Field(default_factory=list)
    files: dict[str, str] = Field(default_factory=dict)


class LogSearchResponse(BaseModel):
    """按 `request_id` 从 PG 查出来的一条链路（`/logs/search`）。

    `request_id` 是**必填**的：这个查询要回答的是「这一次请求到底发生了什么」。
    按类别 + 时间翻日志那是 `/logs/tail` 的活（读文件、没有索引，但也永远可用）。
    """

    request_id: str
    category: str = ""
    items: list[LogLineView] = Field(default_factory=list)
    # 页脚那行「读的是哪个文件」：查库这条路也给一份，页面不用分两套
    files: dict[str, str] = Field(default_factory=dict)
    note: str = ""


class ProviderSettingView(BaseModel):
    """一家供应商给页面看的形状（`GET /providers`）。

    **没有 key、也没有密文**：只有 `has_key` 与 `api_key_fingerprint`（sha256 前 8 位）——
    它能让页面上显示"已配置 (a1b2c3d4)"、换 key 时能确认"确实是另一把"，
    而明文与密文都不出后端。
    """

    provider_key: str
    has_key: bool = False
    api_key_fingerprint: str = ""
    base_url: str = ""
    # 这家可选模型清单：[{name, type}]（页面右侧那张表）
    models: list[dict[str, Any]] = Field(default_factory=list)
    updated_at: str = ""
    updated_by: str = ""
    # 能不能在页面上填 key：主密钥在就 True（部署配置 RAG_SECRET_KEY / 库里 app_secret）
    can_set_key: bool = False


class ProviderListView(BaseModel):
    """`GET/PUT /providers`：全部供应商 + 一句是否可填 key 的说明。"""

    providers: list[ProviderSettingView] = Field(default_factory=list)
    note: str = ""


class ProviderUpdateRequest(BaseModel):
    """`PUT /providers/{key}`：三种语义 —— `api_key` **None = 不动**、**空串 = 清除**、
    给了值 = 加密后存。`base_url` / `models` 同理（None = 沿用库里那份，空 = 显式清空）。"""

    api_key: str | None = None
    base_url: str | None = None
    models: list[dict[str, Any]] | None = None
    actor: str = ""


class ModelSlotUpdateRequest(BaseModel):
    """`PUT /models/{slot}` 的请求体。**不含 key** —— key 永远只在环境里。"""

    provider_key: str = ""
    model: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    actor: str = ""


class ModelSlotView(BaseModel):
    """**一个模型槽位当前是什么**（设置页那一行）。

    `source` 回答"这个值是哪层来的"：`库`（有人设过）或 `未设置`（沿用 `.env` / yaml）。
    `key_ready` 只看"那把 key 在环境里配没配"——**绝不回值**（`rag config` 会打印整棵配置树，
    所以 key 从不进 `Settings`；这里同理）。
    `hot_swappable` 这一版**全是 false**：热切是遗留项（见 `docs/backlog.md` 第 2 条）。
    """

    slot: str
    provider_key: str = ""
    model: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    source: str = "未设置"
    version: int | None = None
    updated_at: str = ""
    updated_by: str = ""
    # 候选：chat 那几个槽位 = `providers.chat` 的键名（界面上的下拉就靠它）
    candidates: list[str] = Field(default_factory=list)
    # True / False / None（None = 这个槽位没有"key"这个概念，比如本地 rerank）
    key_ready: bool | None = None
    hot_swappable: bool = False


class ModelListView(BaseModel):
    """`GET /models`：八个槽位 + 存储位置 + 一个提醒（改完要重启）。"""

    backend: str
    storage: str
    slots: list[ModelSlotView] = Field(default_factory=list)
    note: str = ""


class HealthResponse(BaseModel):
    """**分开报**：Milvus / Redis / 模型各自的状态，哪个挂了单独看得见。

    合成一个笼统的 `ok` 只能回答「能不能用」，回答不了「哪里坏了」——
    而后者才是运维时真正要看的。
    """

    status: Literal["ok", "degraded"]
    milvus: DependencyView
    redis: DependencyView
    # LangGraph 的 checkpointer + store（多轮会话与 HITL 中断靠它）
    postgres: DependencyView
    models: dict[str, DependencyView] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
