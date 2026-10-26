// 后端接口的薄封装。**只做三件事**：带 X-API-Key、拼 URL、把 SSE 拆成事件。
//
// 一个关键选择：**流式用 fetch 读，不用 EventSource** —— EventSource 不能自定义请求头，
// 而我们的 /chat 与 /agent 都要 X-API-Key。fetch + ReadableStream 还能顺手支持取消
// （浏览器关页面/点停止时 abort，后端那套断线取消正好接上）。

import type { S } from './generated/openapi'

type RequiredSchema<K extends keyof S> = Required<S[K]>

export type Hit = RequiredSchema<'HitView'>
export type Stats = RequiredSchema<'StatsView'>
export type Citation = RequiredSchema<'CitationView'>
export type Chunk = RequiredSchema<'ChunkView'>
export type SearchResponse = Omit<RequiredSchema<'SearchResponse'>, 'hits' | 'stats'> & {
  hits: Hit[]
  stats: Stats[]
}
export type ImageUpload = RequiredSchema<'ImageUploadView'>
export type TranscribeResult = RequiredSchema<'TranscribeResponse'>

// 图片 blob 缓存：同一个路径只拉一次。
const assetBlobs = new Map<string, Promise<string>>()

function headers(extra?: Record<string, string>): Record<string, string> {
  const base: Record<string, string> = { 'Content-Type': 'application/json' }
  return { ...base, ...(extra ?? {}) }
}

/** 失败时把后端的 `{"error": {"code", "message"}}` 捞出来给人看。 */
async function failure(res: Response): Promise<Error> {
  const body = await res.text()
  let message = body
  try {
    message = JSON.parse(body)?.error?.message ?? body
  } catch {
    /* 不是 JSON 就原样显示 */
  }
  return new Error(`HTTP ${res.status}：${message}`)
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { ...init, headers: headers(init?.headers as Record<string, string>) })
  if (!res.ok) throw await failure(res)
  const body = await res.text()
  if (body.trimStart().startsWith('<')) {
    // 拿到 HTML = 这个路径被**前端静态页**兜底了（服务端没有这条 API）——
    // 最常见的成因是后端还在跑**旧进程**：重启 `rag serve` 即可。别把
    // `Unexpected token '<'` 这种话甩给看页面的人。
    throw new Error(
      `接口 ${path} 返回了 HTML（不是 JSON）：后端可能还是旧进程，重启 \`rag serve\` 再试`,
    )
  }
  return JSON.parse(body) as T
}

/** 一个模型槽位（`GET /models`）。**key 只回状态不回值**（`key_ready`）。 */
export type ModelSlot = RequiredSchema<'ModelSlotView'>
export type ModelList = Omit<RequiredSchema<'ModelListView'>, 'slots'> & { slots: ModelSlot[] }

/** 模型选型总览。**只读**：改选型走 `rag models set`，重启生效（热切是遗留项）。 */
export function modelSettings(): Promise<ModelList> {
  return json('/models')
}

/** 改一个槽位（写库 + 审计）。回整份列表，页面不用再拉一次。**重启后生效**。 */
export function setModelSlot(
  slot: string,
  body: { provider_key?: string; model?: string; params?: Record<string, unknown>; actor?: string },
): Promise<ModelList> {
  return json(`/models/${encodeURIComponent(slot)}`, {
    method: 'PUT',
    body: JSON.stringify(body),
  })
}

/** 一家供应商（`GET /providers`）。**没有 key**：只有 `has_key` 与指纹（sha256 前 8 位）。 */
export type ProviderSetting = RequiredSchema<'ProviderSettingView'>
export type ProviderList = Omit<RequiredSchema<'ProviderListView'>, 'providers'> & {
  providers: ProviderSetting[]
}

/** 供应商清单。**只读、不回 key**：key 那一格只有"配没配"与指纹。 */
export function providers(): Promise<ProviderList> {
  return json('/providers')
}

/** 写一家供应商（写库 + 审计）。key 三态：不传=不动、空串=清除、给值=加密存。 */
export function setProvider(
  providerKey: string,
  body: {
    api_key?: string | null
    base_url?: string | null
    models?: Array<Record<string, unknown>> | null
    actor?: string
  },
): Promise<ProviderList> {
  return json(`/providers/${encodeURIComponent(providerKey)}`, {
    method: 'PUT',
    body: JSON.stringify(body),
  })
}

/** 依赖探活条目：`/health` 里 milvus / redis / postgres / models.* 都是这个形状。 */
export type DependencyView = RequiredSchema<'DependencyView'>

/** `/health` 的原样返回：分开报，哪个挂了单独看得见。 */
export type Health = Omit<
  RequiredSchema<'HealthResponse'>,
  'status' | 'models' | 'config'
> & {
  status: string
  models: Record<string, DependencyView>
  config: Record<string, unknown>
}

export function health(): Promise<Health> {
  return json('/health')
}

/** 一行日志。`ts` / `level` / `source` / 关联 id 都是后端切好的字段，`text` 只剩正文。 */
export type LogLine = RequiredSchema<'LogLineView'>

export type LogTail = Omit<RequiredSchema<'LogTailResponse'>, 'items'> & {
  items: LogLine[]
}

/** 最近几行日志。**读文件**，所以 CLI 进程（`rag ingest` / `rag ask`）写的也在里面。 */
export function logTail(kind = 'all', limit = 200): Promise<LogTail> {
  return json(`/logs/tail?kind=${encodeURIComponent(kind)}&limit=${limit}`)
}

export type LogSearch = Omit<RequiredSchema<'LogSearchResponse'>, 'items'> & {
  items: LogLine[]
}

/**
 * 按 `request_id` 从 PG 里拉一条链路（可选再按类别与时间收窄）。**request_id 必填**。
 *
 * 与 `logTail` 的分工：那边读文件尾巴（永远最新、但没有索引），这边查库
 * （一次请求的 interface + channel + agent 能串起来）。写入是异步批量的，
 * 刚发生的一两行可能还没进库 —— 要最新的就点「刷新」。
 */
export function logSearch(params: {
  request_id: string
  category?: string
  since?: string
  until?: string
  limit?: number
}): Promise<LogSearch> {
  const query = new URLSearchParams({ request_id: params.request_id })
  if (params.category) query.set('category', params.category)
  if (params.since) query.set('since', params.since)
  if (params.until) query.set('until', params.until)
  if (params.limit) query.set('limit', String(params.limit))
  return json(`/logs/search?${query.toString()}`)
}

/**
 * 实时日志（SSE）。**只覆盖 API 这个进程** —— CLI 那些行要靠 `logTail` 或手动刷新。
 *
 * 默认 `backfill=0`：历史由 `logTail` 一次拿好，这里只跟新行，免得同一行出现两次。
 */
export async function streamLogs(
  onLine: (line: LogLine) => void,
  signal?: AbortSignal,
  backfill = 0,
): Promise<void> {
  const res = await fetch(`/logs/stream?backfill=${backfill}`, { headers: headers(), signal })
  await readSse(res, (name, data) => {
    if (name === 'log') onLine(data as unknown as LogLine)
  })
}

export function search(body: {
  query: string
  top_k?: number
  rerank?: boolean
  channels?: string[]
  /** **以图搜图**：服务端路径，由 `uploadSearchImage` 给。给了它，`query` 可以留空。 */
  image_path?: string
}): Promise<SearchResponse> {
  return json('/search', { method: 'POST', body: JSON.stringify(body) })
}

/**
 * 上传一张图给「以图搜图」。**只上传、不搜** —— 返回服务端路径，页面再拿它去调 `/search`。
 *
 * 分开的理由：`/search` 要过线上 embedding（按量计费），而「传了图还没点搜」是常态；
 * 分开之后重搜、换词、调 top_k 都不用重传图上。与文档上传同理：
 * **不要手写 `Content-Type`**，boundary 得由浏览器填。
 */
export async function uploadSearchImage(
  file: File,
): Promise<ImageUpload> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch('/search/image', { method: 'POST', body: form })
  if (!res.ok) throw await failure(res)
  return (await res.json()) as ImageUpload
}

// ---- 文档：上传 → 解析 → 人工核对标题 → 切分 → 入库（页面那五步）----

/** 一份上传件走到哪一步（页面上那排步骤灯用它）。 */
export type DocumentStages = RequiredSchema<'DocumentStageView'>
export type DocumentRow = Omit<RequiredSchema<'DocumentView'>, 'stages'> & {
  stages: DocumentStages
}
export type DocumentList = Omit<
  RequiredSchema<'DocumentListResponse'>,
  'documents' | 'folders'
> & {
  documents: DocumentRow[]
  folders: Folder[]
}

/** 人工目录。**纯页面上的组织**：不改 doc_id、不动文件，最多三层。 */
export type Folder = RequiredSchema<'FolderView'>
export type FolderChange = RequiredSchema<'FolderChangeResponse'>
export type Deleted = RequiredSchema<'DeletedView'>
export type Uploaded = RequiredSchema<'UploadedView'>
export type UploadResult = Omit<
  RequiredSchema<'UploadResponse'>,
  'uploads' | 'conflicts'
> & {
  uploads: Uploaded[]
  conflicts: Uploaded[]
}

export type OutlineRow = RequiredSchema<'OutlineRowView'>
export type Outline = Omit<RequiredSchema<'OutlineView'>, 'rows'> & { rows: OutlineRow[] }

export type DocumentDetail = DocumentRow &
  Omit<
    RequiredSchema<'DocumentDetailResponse'>,
    'stages' | 'outline' | 'parse' | 'chunks' | 'index' | 'index_job'
  > & {
  stages: DocumentStages
  outline: Outline | null
  parse: ParseSummary | null
  chunks: ChunkSummary | null
  index: IndexSummary | null
  index_job: Job | null
}

/** 第二步（解析）的产物摘要。`stale=true` = 产物过期，步骤灯因此回退（这里说明原因）。 */
export type ParseSummary = RequiredSchema<'ParseSummaryView'>

/** 大纲里的一节：多少块、多少字。 */
export type SectionSummary = RequiredSchema<'SectionSummaryView'>

/** 第四步里「拆分」那半的结果。 */
export type ChunkSummary = Omit<
  RequiredSchema<'ChunkSummaryView'>,
  'sections'
> & {
  sections: SectionSummary[]
}

/** 第四步里「入库」那半：拿 Milvus 注册表那一行说话。 */
export type IndexSummary = RequiredSchema<'IndexSummaryView'>

export function documents(): Promise<DocumentList> {
  return json('/documents')
}

/** 新建目录。`parent` 为空 = 顶层；最多三层（后端挡的，这里只是先拦一下别多跑一趟）。 */
export function createFolder(name: string, parent: string | null = null): Promise<Folder> {
  return json('/documents/folders', {
    method: 'POST',
    body: JSON.stringify({ name, parent }),
  })
}

/** 改名 / 搬家。**只传 `name` 就是只改名** —— 传 `parent: null` 才是搬到顶层。 */
export function updateFolder(
  folderId: string,
  body: { name?: string; parent?: string | null },
): Promise<Folder> {
  return json(`/documents/folders/${encodeURIComponent(folderId)}`, {
    method: 'PATCH',
    body: JSON.stringify(body),
  })
}

/** 删目录。里面的文档回「未分类」；还有子目录时后端回 409。`moved` = 影响了几份文档。 */
export function deleteFolder(folderId: string): Promise<FolderChange> {
  return json(`/documents/folders/${encodeURIComponent(folderId)}`, { method: 'DELETE' })
}

/** 把几份文档放进一个目录；`folderId` 为空 = 移回「未分类」。 */
export function assignFolder(
  docIds: string[],
  folderId: string | null,
): Promise<FolderChange> {
  return json('/documents/folders/members', {
    method: 'PUT',
    body: JSON.stringify({ doc_ids: docIds, folder_id: folderId }),
  })
}

export function documentDetail(docId: string): Promise<DocumentDetail> {
  return json(`/documents/${encodeURIComponent(docId)}`)
}

/**
 * 原始上传件的字节（页面「上传」那一步要看原文）。
 *
 * **直接给 URL 而不是 fetch**：HTML 要进沙箱 iframe 让浏览器自己渲染排版，PDF 交给内置
 * 查看器，图片交给 `<img>` —— 这三条都要浏览器原生处理，走 JS 反而要自己实现一遍。
 * 只有纯文本才用 `fetch(sourceUrl(id)).then(r => r.text())` 取回来放进 `<pre>`。
 */
export function sourceUrl(docId: string): string {
  return `/documents/${encodeURIComponent(docId)}/source`
}

export type SourceKind = 'html' | 'pdf' | 'image' | 'text' | 'other'

/** 原文怎么展示：按后缀定（`content_type` 只有抓来的文件才有，上传件没有）。 */
export function sourceKindOf(name: string): SourceKind {
  const suffix = (name.split('.').pop() ?? '').toLowerCase()
  if (suffix === 'html' || suffix === 'htm') return 'html'
  if (suffix === 'pdf') return 'pdf'
  if (['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'].includes(suffix)) return 'image'
  if (['md', 'markdown', 'txt', 'log', 'rst', 'csv'].includes(suffix)) return 'text'
  return 'other'
}

/**
 * 上传（multipart）。**不要手写 `Content-Type`** —— boundary 得由浏览器填，
 * 这里绕开 `json()` 就是因为它固定塞了 `application/json`。
 */
export async function uploadDocuments(files: File[]): Promise<UploadResult> {
  const form = new FormData()
  for (const file of files) form.append('files', file)
  const res = await fetch('/documents/upload', { method: 'POST', body: form })
  if (!res.ok) throw await failure(res)
  return (await res.json()) as UploadResult
}

/**
 * 语音转文字（对话页那个麦克风）。**只转文字**：音频不落盘、不落库、不进会话 ——
 * 页面把它放进输入框让人改，改完再走 `/chat`。
 *
 * 前端录的是 16kHz 单声道 wav（见 `ChatView.vue` 的录音那一段）：不用 `MediaRecorder`
 * 是因为它默认出 webm/opus，而 ASR 那一侧对 webm 没有保证。
 * 与上传同理：**不要手写 `Content-Type`**，boundary 得由浏览器填。
 *
 * 这段音频**按量计费**（一次调用一次），所以只在这里发，且不做自动重试。
 */
export async function transcribeAudio(audio: Blob, name = 'take.wav'): Promise<TranscribeResult> {
  const form = new FormData()
  form.append('file', audio, name)
  const res = await fetch('/chat/transcribe', { method: 'POST', body: form })
  if (!res.ok) throw await failure(res)
  return (await res.json()) as TranscribeResult
}

/**
 * 从链接导入一份文档。抓回来的东西**当普通上传件落盘**，后面五步一模一样。
 *
 * 链接指向的可能是 HTML 文档页，也可能是 PDF / Word 的下载链接 —— 类型由后端按
 * `Content-Disposition` → URL 末段 → `Content-Type` 定。`name` 可选，不带后缀也行。
 */
export function fetchDocument(body: { url: string; name?: string }): Promise<UploadResult> {
  return json('/documents/fetch', { method: 'POST', body: JSON.stringify(body) })
}

/**
 * 删掉一份文档。`purge=true`（默认，也是页面用的）连 Milvus 里的块与解析产物一起清 ——
 * **不可逆**。页面上的「覆盖更新」就是先导入新版、再对旧版调它。
 */
export function deleteDocument(docId: string, purge = true): Promise<Deleted> {
  return json(`/documents/${encodeURIComponent(docId)}?purge=${purge}`, { method: 'DELETE' })
}

/** 解析一份上传件：202 + job，进度靠 `jobs()` 轮询。 */
export function parseDocument(docId: string): Promise<Job> {
  return json(`/documents/${encodeURIComponent(docId)}/parse`, { method: 'POST' })
}

/** 提交人工核对结果。**只有层级**：正文与标题原文不在请求体里。 */
export function saveOutline(
  docId: string,
  body: { rows: { level: number }[]; max_level: number; confirmed: boolean },
): Promise<DocumentDetail> {
  return json(`/documents/${encodeURIComponent(docId)}/outline`, {
    method: 'PUT',
    body: JSON.stringify(body),
  })
}

/** 切分 + 灌 Milvus：202 + job。清单没核对过会 409（后端挡的，不是这里）。 */
export function indexDocument(docId: string): Promise<Job> {
  return json(`/documents/${encodeURIComponent(docId)}/index`, { method: 'POST' })
}

export type Job = Omit<RequiredSchema<'JobView'>, 'status'> & {
  status: 'queued' | 'running' | 'done' | 'failed'
}

/** 最近的任务，新的在前；Redis 可用时包含其他 worker 提交的任务。 */
export function jobs(limit = 20): Promise<{ jobs: Job[] }> {
  return json(`/jobs?limit=${limit}`)
}

export function chunk(chunkId: string): Promise<Chunk> {
  return json(`/chunks/${encodeURIComponent(chunkId)}`)
}

/** 资产（图片）的可访问 URL。鉴权走 header 是拿不到 <img> 的，所以后端那边用同源 + 代理。 */
export function assetUrl(path: string): string {
  const clean = path.startsWith('asset:') ? path.slice('asset:'.length) : path
  return `/assets/${clean}`
}

/**
 * 取资产并转成 `blob:` URL。**`<img src="/assets/…">` 在配了 `RAG_API_KEY` 的部署下必然 401**：
 * 图片端点在鉴权后面，而 `<img>` 发不出 `X-API-Key`（真机验过，返回 401 application/json）。
 * 所以带 key 把图拉回来，再交给浏览器一个同源 blob: —— 图能显示，鉴权也没松。
 */
export function assetBlob(path: string): Promise<string> {
  const url = assetUrl(path)
  const cached = assetBlobs.get(url)
  if (cached) return cached
  const task = fetch(url, { headers: headers() }).then(async (res) => {
    if (!res.ok) throw new Error(`HTTP ${res.status}：取不到资产 ${path}`)
    return URL.createObjectURL(await res.blob())
  })
  // 失败的**不留在缓存里**：key 填对了、图补上了，重试就该能成
  task.catch(() => assetBlobs.delete(url))
  assetBlobs.set(url, task)
  return task
}

export interface ChatEvent {
  type: 'retrieval' | 'token' | 'done' | 'error'
  text?: string
  hits?: Hit[]
  stats?: Stats[]
  citations?: Citation[]
  session_id?: string | null
  usage?: Usage | null
  rounds?: number
  grade?: Record<string, unknown>
  mode?: string
  message?: string
}

/** 一条工具调用。`args` 是后端截过的预览（长正文会被截断），`ms` 是从「看到调用」到「看到结果」。 */
export interface ToolStep {
  name: string
  status: string
  args: Record<string, unknown>
  ms: number | null
}

/** 撞上 HITL 闸门时它想干的事。`args` 这条是**完整**的（批准要看全）。 */
export interface ApprovalRequest {
  name: string
  args: Record<string, unknown>
  description?: string
}

export interface Interrupt {
  action_requests: ApprovalRequest[]
  review_configs?: { action_name: string; allowed_decisions: string[] }[]
}

export interface AgentEvent {
  type: 'token' | 'tool' | 'interrupt' | 'done' | 'error'
  text?: string
  name?: string
  status?: string
  args?: Record<string, unknown>
  ms?: number | null
  /** `interrupt` 事件：中断载荷**摊平**在事件本身上（和 `tool` 一致，不嵌一层） */
  action_requests?: ApprovalRequest[]
  review_configs?: Interrupt['review_configs']
  answer?: string
  citations?: Citation[]
  thread_id?: string | null
  session_id?: string | null
  usage?: Usage | null
  message?: string
}

/** 一段会话或一轮问答的 token 用量。 */
export type Usage = RequiredSchema<'SessionUsageView'>

/** `GET /sessions` 里的一行。 */
export type SessionSummary = Omit<RequiredSchema<'SessionView'>, 'usage'> & {
  usage: Usage
}

/** `GET /sessions/{id}` 里的一条消息。只会有 human 与 ai 两种角色。 */
export type SessionMessage = Omit<
  RequiredSchema<'SessionMessageView'>,
  'role' | 'status' | 'interrupt' | 'usage'
> & {
  role: 'human' | 'ai'
  status: 'answered' | 'interrupted' | 'stopped' | 'error'
  interrupt?: Interrupt | null
  usage?: Usage | null
}

export type SessionDetail = Omit<
  RequiredSchema<'SessionDetailResponse'>,
  'messages' | 'usage'
> & {
  messages: SessionMessage[]
  usage: Usage
}

/** 会话列表。**服务端存的**，换台机器、换个浏览器都在。 */
export function sessions(limit = 20): Promise<{ sessions: SessionSummary[] }> {
  return json(`/sessions?limit=${limit}`)
}

/** 一段会话的全部消息（已按 seq 排好）。 */
export function session(sessionId: string): Promise<SessionDetail> {
  return json(`/sessions/${encodeURIComponent(sessionId)}`)
}

export async function deleteSession(sessionId: string): Promise<void> {
  const res = await fetch(`/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
    headers: headers(),
  })
  if (!res.ok) throw await failure(res)
}

/**
 * 把 SSE 响应拆成一个个事件。`/chat`、`/agent`、`/agent/resume` **共用这一段**。
 *
 * 用 `fetch` + `ReadableStream` 而不是 `EventSource`：后者不能自定义请求头，而这几个端点
 * 都要 `X-API-Key`；`fetch` 还能 abort（页面上的「停止」直接触发后端那套断线取消）。
 */
async function readSse(
  res: Response,
  onEvent: (name: string, data: Record<string, unknown>) => void,
): Promise<void> {
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}：${await res.text()}`)
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let index = buffer.indexOf('\n\n')
    while (index >= 0) {
      const raw = buffer.slice(0, index)
      buffer = buffer.slice(index + 2)
      const name = raw
        .split('\n')
        .find((line) => line.startsWith('event: '))
        ?.slice('event: '.length)
      const data = raw
        .split('\n')
        .filter((line) => line.startsWith('data: '))
        .map((line) => line.slice('data: '.length))
        .join('')
      if (name && data) onEvent(name, JSON.parse(data) as Record<string, unknown>)
      index = buffer.indexOf('\n\n')
    }
  }
}

/** 流式问答（`/chat`）：定制链路，单轮、无历史。 */
export async function streamChat(
  body: {
    question: string
    top_k?: number
    graph?: boolean
    rerank?: boolean
    session_id?: string | null
  },
  onEvent: (event: ChatEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch('/chat', {
    method: 'POST',
    headers: headers(),
    body: JSON.stringify({ ...body, stream: true }),
    signal,
  })
  await readSse(res, (name, data) => onEvent({ type: name as ChatEvent['type'], ...data }))
}

/** 流式 agent（`/agent`）：让模型自己决定查什么，多轮靠 `thread_id`。 */
export async function streamAgent(
  body: { message: string; thread_id?: string | null; web?: boolean; session_id?: string | null },
  onEvent: (event: AgentEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch('/agent', {
    method: 'POST',
    headers: headers(),
    body: JSON.stringify({ ...body, stream: true }),
    signal,
  })
  await readSse(res, (name, data) => onEvent({ type: name as AgentEvent['type'], ...data }))
}

/**
 * 批准 / 拒绝之后接着流（`/agent/resume`）。
 *
 * 事件与 `streamAgent` **完全一样** —— 后端那边刻意做成同一组，这里不用第二套解析。
 * `decisions` 一条对一条 `action_requests`，**顺序要对**（中间件的约定）。
 */
export async function resumeAgent(
  body: { thread_id: string; decisions: { type: string }[]; session_id?: string | null },
  onEvent: (event: AgentEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch('/agent/resume', {
    method: 'POST',
    headers: headers(),
    body: JSON.stringify({ ...body, stream: true }),
    signal,
  })
  await readSse(res, (name, data) => onEvent({ type: name as AgentEvent['type'], ...data }))
}
