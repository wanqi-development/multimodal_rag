// 极小的共享状态（不引 pinia）：外壳、导航、详情抽屉要跨页面共用这几件事。
import { reactive, ref } from 'vue'
import {
  chunk as fetchChunk,
  deleteSession as deleteSessionApi,
  health as fetchHealth,
  session as fetchSession,
  sessions as fetchSessions,
  type Chunk,
  type Citation,
  type Hit,
  type Interrupt,
  type SessionDetail,
  type SessionSummary,
  type Stats,
  type Usage,
} from './api'

// 设置页（原 'models'）2026-10-10 暂时下线：页面与接口都留着，只是不挂导航
// （见 docs/backlog.md 第 2 条）。要放回来就在这个联合类型里加回 'models'，
// 再把 NavRail 的条目与 App.vue 那条分支接上。
export type PageId = 'search' | 'chat' | 'documents' | 'logs'

export const page = ref<PageId>('search')
export const detail = ref<Chunk | null>(null)
export const detailError = ref('')
export const status = ref<{ ok: boolean; text: string; milvus?: boolean; postgres?: boolean; redis?: boolean }>({
  ok: false,
  text: '未连接',
})

export async function ping(): Promise<void> {
  try {
    const body = await fetchHealth()
    status.value = {
      ok: body.status === 'ok',
      text: body.status === 'ok' ? '已连接' : '降级',
      milvus: body.milvus?.ok,
      postgres: body.postgres?.ok,
      redis: body.redis?.ok,
    }
  } catch (exc) {
    status.value = { ok: false, text: (exc as Error).message }
  }
}

export async function openChunk(chunkId: string): Promise<void> {
  detailError.value = ''
  try {
    detail.value = await fetchChunk(chunkId)
  } catch (exc) {
    detailError.value = (exc as Error).message
  }
}

export function closeDetail(): void {
  detail.value = null
  detailError.value = ''
}

// ---- 对话（P2）----
//
// **放在这里而不是 ChatView 里**：外壳用的是 `<ChatView v-if>`，一切到「检索」再切回来
// 组件就卸载重建了 —— 状态留在组件里等于「点一下别的页，聊过的东西全没了」。

/** 走哪条路：`chat` = 定制链路（单轮、无历史）；`agent` = 让模型自己决定查什么。 */
export type Mode = 'chat' | 'agent'

export interface Turn {
  question: string
  answer: string
  hits: Hit[]
  stats: Stats[]
  citations: Citation[]
  error: string
  running: boolean
  /** 这一轮实际走的哪条路 —— 事后要说得清「刚才那次是 agent 查的」 */
  mode: Mode
  /** agent 给的会话 id，追问必须带回去；`/chat` 恒为 null */
  thread: string | null
  /** 撞上 HITL 闸门时它想干什么，等人在卡片上按批准/拒绝 */
  interrupt: Interrupt | null
  /** 证据不足 → 提示可以转 agent 再查一遍 */
  weak: boolean
  /** 这一轮的 token 用量（服务端在 `done` 事件里给的；历史轮次没有） */
  usage?: Usage | null
}

/**
 * 「证据不足」的判据：最高分低于这个数。
 *
 * **这是抄后端那个数**（`agent.web_min_corpus_score: 0.2`，实测「语料内有答案 0.9985 /
 * 语料外 0.0077~0.0978」），不另发明一个 —— 两处判据不一致时最难查。
 * 注意只在**开着重排**时这个分才可比（关掉时是 RRF 名次分）；默认是开着的。
 */
export const WEAK_SCORE = 0.2

export const mode = ref<Mode>((localStorage.getItem('rag_mode') as Mode) ?? 'chat')
export const turns = ref<Turn[]>([])
/** 本会话的 agent thread：非空 = 已经聊出历史了 */
export const thread = ref<string | null>(null)

// ---- 会话（P2 第三批）----
//
// **存服务端**（`/sessions`，PG 三张表）。浏览器这边一个字都不留 ——
// 否则「换台机器、换个浏览器，会话都在」就不成立，而那是把会话放进 PG 的全部理由。
// 前端只保留「当前在哪一段」这个 id，它本身就是个 UI 状态。

export const sessions = ref<SessionSummary[]>([])
/** 拉会话列表失败的原因。**空列表要说清是「真没有」还是「没拉到」**。 */
export const sessionsError = ref('')
/**
 * 会话列表取多少条。默认 20 —— 下拉里那个「加载更多」把它翻倍（上限 200）。
 *
 * **独立会话页不做了**（使用者 2026-10-08 定）：会话的列表 / 回放 / 继续 / 删除 / 每轮 token
 * 对话页本来就全都有，"翻得更远"这点差别用这一个按钮补掉就够，不必再开一页。
 */
export const sessionLimit = ref(20)
/** 当前这段对话的 id。**由服务端发**（第一轮的 `done` 事件带回来），不是本地编的。 */
export const sessionId = ref<string | null>(null)

/**
 * 刷会话列表。
 *
 * **拉失败时不清空、也不静默**：清空的话，一次抖动（后端重启、PG 抖一下）就把下拉变成
 * 「还没有历史」，而且再也不会自己恢复 —— 使用者看到的就是「顶上的会话按钮点出来是空的」。
 * 现在留着上一次的结果、把原因显示出来，并且**展开下拉时会自动重拉一次**（见 ChatView）。
 */
export async function refreshSessions(limit = sessionLimit.value): Promise<void> {
  try {
    sessions.value = (await fetchSessions(limit)).sessions
    sessionsError.value = ''
  } catch (exc) {
    sessionsError.value = (exc as Error).message
  }
}

/** 下拉底部的「加载更多」：上限翻倍再拉一次。到 200 就停（再多也不该塞进一个下拉）。 */
export async function loadMoreSessions(): Promise<void> {
  sessionLimit.value = Math.min(sessionLimit.value * 2, 200)
  await refreshSessions()
}

/** 把服务端的一段会话还原成界面的轮次。 */
function turnsOf(detail: SessionDetail): Turn[] {
  const out: Turn[] = []
  for (const message of detail.messages) {
    if (message.role === 'human') {
      out.push({
        question: message.content,
        answer: '',
        hits: [],
        stats: [],
        citations: [],
        error: '',
        running: false,
        mode: (detail.mode as Mode) ?? 'chat',
        thread: detail.thread_id,
        interrupt: null,
        weak: false,
      })
      continue
    }
    const last = out[out.length - 1]
    if (!last) continue
    last.answer = message.content
    last.usage = message.usage ?? null
    // 停在闸门的那一轮：**载荷是服务端读的时候现取的**（不进会话表），拿到了就把卡片摆回来
    last.interrupt = message.interrupt ?? null
    if (message.status === 'error') last.error = '这一轮出过错'
  }
  return out.map((turn) => reactive(turn))
}

export async function openSession(id: string): Promise<void> {
  const detail = await fetchSession(id)
  sessionId.value = detail.session_id
  thread.value = detail.thread_id
  mode.value = (detail.mode as Mode) ?? 'chat'
  turns.value = turnsOf(detail)
}

/** 打开最近一段（刷新之后「会话还在」靠它）。没有历史就什么都不做。 */
export async function openLatest(): Promise<void> {
  const [recent] = sessions.value
  if (recent) await openSession(recent.session_id)
}

export async function dropSession(id: string): Promise<void> {
  await deleteSessionApi(id)
  if (id === sessionId.value) {
    turns.value = []
    thread.value = null
    sessionId.value = null
    mode.value = 'chat'
  }
  await refreshSessions()
}

export function setMode(next: Mode): void {
  mode.value = next
  localStorage.setItem('rag_mode', next)
}

/**
 * 开一段新对话。
 *
 * 本地什么都不用存 —— 上一段服务端已经写好了；这里只是**把「当前在哪一段」清掉**，
 * 下一轮问答会给一个新的 `session_id`。
 */
export function newConversation(): void {
  turns.value = []
  thread.value = null
  sessionId.value = null
  setMode('chat')
}

// 主题：自动（跟随系统）/ 浅 / 深 —— 写 <html data-theme>，token 里的 light-dark() 自动切
export type Theme = 'auto' | 'light' | 'dark'
export const theme = ref<Theme>((localStorage.getItem('rag_theme') as Theme) ?? 'auto')

export function setTheme(next: Theme): void {
  theme.value = next
  localStorage.setItem('rag_theme', next)
  if (next === 'auto') document.documentElement.removeAttribute('data-theme')
  else document.documentElement.setAttribute('data-theme', next)
}

// ---- 界面偏好：两个栏能收起来、会话栏宽度可拖（都存 localStorage）----
//
// 这些是**纯界面偏好**，与后端无关，所以和主题放同一个地方。收起/展开只是为了腾地方
// （使用者 2026-10-08 提的），所以收起之后要留一个明显的按钮能把它叫回来。

function flag(key: string, fallback: boolean): boolean {
  const raw = localStorage.getItem(key)
  return raw === null ? fallback : raw === '1'
}

function persist(key: string, on: boolean): void {
  localStorage.setItem(key, on ? '1' : '0')
}

/** 全站左侧导航栏是否收起（收起 = 60px、只剩图标）。 */
export const railCollapsed = ref(flag('rag_rail_collapsed', false))

export function toggleRail(): void {
  railCollapsed.value = !railCollapsed.value
  persist('rag_rail_collapsed', railCollapsed.value)
}

/** 对话页里的会话栏是否收起（收起 = 整列让给对话区）。 */
export const sessionsCollapsed = ref(flag('rag_sessions_collapsed', false))

export function toggleSessionsPane(): void {
  sessionsCollapsed.value = !sessionsCollapsed.value
  persist('rag_sessions_collapsed', sessionsCollapsed.value)
}

/** 会话栏宽度（px）。上下限写死：太窄标题没法看、太宽把对话区挤没。 */
export const SESSIONS_MIN_W = 180
export const SESSIONS_MAX_W = 420

export const sessionsWidth = ref(
  Math.min(
    SESSIONS_MAX_W,
    Math.max(SESSIONS_MIN_W, Number(localStorage.getItem('rag_sessions_w')) || 240),
  ),
)

export function setSessionsWidth(px: number): void {
  sessionsWidth.value = Math.min(SESSIONS_MAX_W, Math.max(SESSIONS_MIN_W, Math.round(px)))
  localStorage.setItem('rag_sessions_w', String(sessionsWidth.value))
}
