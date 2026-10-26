<script setup lang="ts">
// 对话页：两条路共用一套外壳。
//
//   快问答（`/chat`）—— 我们写死的链路，单轮、无历史、便宜、几秒钟
//   深度 agent（`/agent`）—— 模型自己决定查几次、派给哪个子 agent，多轮、贵、慢
//
// **有历史就只能走 agent**：`/chat` 没有历史，接不上上下文就是接不上，这不是策略问题。
// 所以「切回快问答」在有历史时是禁用的，想换就得开新对话（不然用户被锁在最贵那条路上）。
import { computed, onMounted, onUnmounted, reactive, ref, watch } from 'vue'
import {
  History,
  Mic,
  MessagesSquare,
  PanelLeftClose,
  PanelLeftOpen,
  Plus,
  SearchCheck,
  Square,
  Trash2,
  TriangleAlert,
  Zap,
} from 'lucide-vue-next'
import {
  resumeAgent,
  streamAgent,
  streamChat,
  transcribeAudio,
  type AgentEvent,
  type ChatEvent,
} from '../api'
import ApprovalCard from '../components/ApprovalCard.vue'
import { citeText, renderMarkdown } from '../markdown'
import {
  dropSession,
  loadMoreSessions,
  mode,
  newConversation,
  openChunk,
  openSession,
  openLatest,
  refreshSessions,
  sessionsCollapsed,
  sessionId,
  sessionLimit,
  sessions,
  sessionsError,
  sessionsWidth,
  setSessionsWidth,
  SESSIONS_MAX_W,
  SESSIONS_MIN_W,
  setMode,
  status,
  thread,
  toggleSessionsPane,
  turns,
  WEAK_SCORE,
  type Mode,
  type Turn,
} from '../state'

const draft = ref('')
/** 明确要求联网（对应 `/agent` 的 `web=true`）。agent 默认只在语料查不到时才联网。 */
const web = ref(false)
/** 正在读某一段会话（取消息是异步的，空着的时候得让人知道是在读、不是没有）。 */
const loadingSession = ref(false)
/** 读会话失败的原因。**必须显示出来** —— 静默失败看起来就是「这一段没有历史」。 */
const historyError = ref('')
const box = ref<HTMLInputElement | null>(null)
/** 下拉里的「只看 agent」：会话多了之后，想找某一段 agent 会话比翻 60 条快。 */
const onlyAgent = ref(false)
const visibleSessions = computed(() =>
  onlyAgent.value ? sessions.value.filter((item) => item.mode === 'agent') : sessions.value,
)
let controller: AbortController | null = null
let scroller: HTMLElement | null = null

// ---- 语音输入：录一段 → 转文字 → **填进输入框，不自动发**（使用者 2026-10-08 定）----
// 三种模态里先做语音这一条：图片那条要动生成链路，另起一轮。
/** 前端一概按 16kHz 单声道录：ASR 要的就是这个，体积也压得住（≈32KB/s，10MB 够 5 分钟） */
const TARGET_RATE = 16000
/** 录满这么久自动停 —— 再长会撞上后端 10MB 那道线 */
const MAX_SECONDS = 300

const recording = ref(false)
const transcribing = ref(false)
const micError = ref('')
/** 录音中的那套东西（流 / 上下文 / 采到的 PCM）。`null` = 没在录。 */
let take: {
  stop: () => Uint8Array | null
} | null = null
let autoStop: number | null = null

/** 采到的 PCM → 16kHz 单声道 16bit wav。**不引库**：wav 就是 44 字节头 + 小端 PCM。 */
function wavBytes(samples: Float32Array, rate: number): Uint8Array {
  const view = new DataView(new ArrayBuffer(44 + samples.length * 2))
  const ascii = (at: number, value: string): void => {
    for (let i = 0; i < value.length; i += 1) view.setUint8(at + i, value.charCodeAt(i))
  }
  ascii(0, 'RIFF')
  view.setUint32(4, 36 + samples.length * 2, true)
  ascii(8, 'WAVE')
  ascii(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true) // PCM
  view.setUint16(22, 1, true) // 单声道
  view.setUint32(24, rate, true)
  view.setUint32(28, rate * 2, true) // 每秒字节数
  view.setUint16(32, 2, true) // 每个采样 2 字节
  view.setUint16(34, 16, true)
  ascii(36, 'data')
  view.setUint32(40, samples.length * 2, true)
  for (let i = 0; i < samples.length; i += 1) {
    const clipped = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(44 + i * 2, clipped < 0 ? clipped * 0x8000 : clipped * 0x7fff, true)
  }
  return new Uint8Array(view.buffer)
}

/** 等间隔抽稀到 16kHz。浏览器未必肯按我们请求的采样率给，所以编码前统一一次。 */
function resample(samples: Float32Array, from: number, to: number): Float32Array {
  if (from <= to) return samples
  const ratio = from / to
  const out = new Float32Array(Math.floor(samples.length / ratio))
  for (let i = 0; i < out.length; i += 1) out[i] = samples[Math.floor(i * ratio)]
  return out
}

function join(chunks: Float32Array[]): Float32Array {
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0)
  const out = new Float32Array(total)
  let at = 0
  for (const chunk of chunks) {
    out.set(chunk, at)
    at += chunk.length
  }
  return out
}

async function startRecording(): Promise<void> {
  micError.value = ''
  if (!navigator.mediaDevices?.getUserMedia) {
    // 只有 https / localhost 才给麦克风权限 —— 局域网 IP 打开页面时这里会说话
    micError.value = '这个浏览器不给录音（要 https 或 localhost 打开页面）'
    return
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    })
    const ctx = new AudioContext({ sampleRate: TARGET_RATE })
    const chunks: Float32Array[] = []
    const source = ctx.createMediaStreamSource(stream)
    // `createScriptProcessor` 已经标了废弃，但 `AudioWorklet` 要单独一个模块文件（或 Blob
    // URL）才能加载，为这一段录音不值当；实测浏览器都还稳定支持它。
    const processor = ctx.createScriptProcessor(4096, 1, 1)
    processor.onaudioprocess = (event) => {
      chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)))
    }
    source.connect(processor)
    processor.connect(ctx.destination)
    recording.value = true
    autoStop = window.setTimeout(() => void stopRecording(), MAX_SECONDS * 1000)
    take = {
      stop: () => {
        if (autoStop !== null) window.clearTimeout(autoStop)
        autoStop = null
        processor.disconnect()
        source.disconnect()
        stream.getTracks().forEach((track) => track.stop())
        const rate = ctx.sampleRate
        void ctx.close()
        const pcm = join(chunks)
        return pcm.length ? wavBytes(resample(pcm, rate, TARGET_RATE), TARGET_RATE) : null
      },
    }
  } catch (exc) {
    micError.value = `开不了麦克风：${(exc as Error).message}`
  }
}

async function stopRecording(): Promise<void> {
  const done = take
  take = null
  recording.value = false
  const wav = done?.stop() ?? null
  if (!wav) {
    micError.value = '没录到声音'
    return
  }
  transcribing.value = true
  try {
    const result = await transcribeAudio(new Blob([wav], { type: 'audio/wav' }), 'take.wav')
    // **只填进输入框，不自动发** —— 识别难免有错字，人改完再回车
    draft.value = draft.value.trim() ? `${draft.value.trim()} ${result.text}` : result.text
    box.value?.focus()
  } catch (exc) {
    micError.value = (exc as Error).message
  } finally {
    transcribing.value = false
  }
}

function toggleRecording(): void {
  if (recording.value) void stopRecording()
  else void startRecording()
}

/**
 * 有历史就锁死 agent —— **但 PG 报挂了就不锁**：那时历史根本取不回来，
 * 继续锁着只会让每个问题都 503，还得让用户自己想到去点「新对话」。
 */
const lockedToAgent = computed(() => thread.value !== null && status.value.postgres !== false)
const effectiveMode = computed<Mode>(() => (lockedToAgent.value ? 'agent' : mode.value))

function scrollToEnd(): void {
  scroller?.scrollTo({ top: scroller.scrollHeight, behavior: 'smooth' })
}

function begin(question: string): { turn: Turn; signal: AbortSignal } {
  // **必须 `reactive()` 之后再 push**：push 进数组的是这个对象本身，而直接改**原始对象**
  // 不会触发渲染（响应式的依赖是挂在代理上的）。不这么做，现象是「发出去就一直生成中、
  // 答案一个字都不显示」—— 真机踩过，而且只在「有别的 ref 恰好变了」时才偶然刷新一下。
  const turn = reactive<Turn>({
    question,
    answer: '',
    hits: [],
    stats: [],
    citations: [],
    error: '',
    running: true,
    mode: effectiveMode.value,
    thread: null,
    interrupt: null,
    weak: false,
  })
  turns.value.push(turn)
  draft.value = ''
  const ctl = new AbortController()
  controller = ctl
  scrollToEnd()
  return { turn, signal: ctl.signal }
}

function finish(turn: Turn): void {
  turn.running = false
  controller = null
  // 轮次是**服务端落的库**（问答结束时它自己写），这里只把列表刷新一下
  void refreshSessions()
  scrollToEnd()
}

/** 中断事件是**摊平**的（与 tool 一样），不是嵌在 `interrupt` 里 —— 这里收成一个对象。 */
function onAgent(turn: Turn, event: AgentEvent): void {
  if (event.type === 'token') {
    turn.answer += event.text ?? ''
    scrollToEnd()
  } else if (event.type === 'interrupt') {
    turn.interrupt = event.action_requests
      ? { action_requests: event.action_requests, review_configs: event.review_configs }
      : null
    if (event.thread_id) remember(turn, event.thread_id)
  } else if (event.type === 'done') {
    // **以 done 里的答案为准**：流里攒下来的是主 agent 一路「边想边说」的碎片，
    // 最后那条 AI 消息才是答案（CLI 打的也是它）。不覆盖的话，答案框里是中间话术的拼接。
    if (event.answer) turn.answer = event.answer
    turn.citations = event.citations ?? []
    turn.usage = event.usage ?? null
    if (event.thread_id) remember(turn, event.thread_id)
    if (event.session_id) sessionId.value = event.session_id
  } else if (event.type === 'error') {
    turn.error = event.message ?? 'agent 跑挂了'
  }
}

function onChat(turn: Turn, event: ChatEvent): void {
  if (event.type === 'retrieval') {
    turn.hits = event.hits ?? []
    turn.stats = event.stats ?? []
  } else if (event.type === 'token') {
    turn.answer += event.text ?? ''
    scrollToEnd()
  } else if (event.type === 'done') {
    turn.citations = event.citations ?? []
    turn.usage = event.usage ?? null
    if (event.session_id) sessionId.value = event.session_id
    turn.weak = isWeak(turn)
  } else if (event.type === 'error') {
    turn.error = event.message ?? '生成失败'
  }
}

/** 会话 id 是本会话的锚：下一轮追问、以及中断后的批准都靠它。 */
function remember(turn: Turn, id: string): void {
  turn.thread = id
  thread.value = id
}

/**
 * 证据够不够。判据是「一条都没召回」或「最高分低于 WEAK_SCORE」——
 * 那个数与后端的联网闸门是同一个（见 state.ts 的说明）。
 */
function isWeak(turn: Turn): boolean {
  if (turn.error) return false
  const top = turn.hits[0]
  return !top || top.score < WEAK_SCORE
}

async function ask(question?: string): Promise<void> {
  const text = (question ?? draft.value).trim()
  if (!text || controller) return
  const { turn, signal } = begin(text)
  try {
    if (turn.mode === 'agent') {
      await streamAgent(
        { message: text, thread_id: thread.value, web: web.value, session_id: sessionId.value },
        (event) => onAgent(turn, event),
        signal,
      )
    } else {
      await streamChat(
        { question: text, session_id: sessionId.value },
        (event) => onChat(turn, event),
        signal,
      )
    }
  } catch (exc) {
    if ((exc as Error).name !== 'AbortError') turn.error = (exc as Error).message
  } finally {
    finish(turn)
  }
}

/** 批准 / 拒绝。批准之后那段答案照样是流式的 —— 后端刻意与 `/agent` 做成同一组事件。 */
async function decide(turn: Turn, kind: string): Promise<void> {
  const pending = turn.interrupt
  if (!pending || !turn.thread || controller) return
  const decisions = pending.action_requests.map(() => ({ type: kind }))
  turn.interrupt = null
  turn.running = true
  const ctl = new AbortController()
  controller = ctl
  try {
    await resumeAgent(
      { thread_id: turn.thread, decisions, session_id: sessionId.value },
      (event) => onAgent(turn, event),
      ctl.signal,
    )
  } catch (exc) {
    if ((exc as Error).name !== 'AbortError') turn.error = (exc as Error).message
  } finally {
    finish(turn)
  }
}

/**
 * 证据不足时人工升级。**不自动切** —— 自动升级等于替用户决定多花三五倍的钱和几十秒，
 * 而「证据不足」这个判断本身也会看走眼；让他按一下。
 */
async function escalate(turn: Turn): Promise<void> {
  if (controller) return
  turn.weak = false
  setMode('agent')
  await ask(turn.question)
}

function stop(): void {
  controller?.abort()
  controller = null
}

function startOver(): void {
  stop()
  newConversation()
}

/**
 * 拖动会话栏右边那条把手调宽度。
 *
 * 把 mousemove / mouseup 挂在 **window** 上（而不是在把手上做 capture）：拖动时指针很容易
 * 滑出那条几像素宽的把手，挂在 window 才不会中途丢事件。松手时才落一次 localStorage ——
 * 拖动过程里每像素都写没有必要。
 */
function startResize(event: MouseEvent): void {
  event.preventDefault()
  const startX = event.clientX
  const startWidth = sessionsWidth.value
  const onMove = (move: MouseEvent): void => {
    setSessionsWidth(startWidth + move.clientX - startX)
  }
  const onUp = (): void => {
    window.removeEventListener('mousemove', onMove)
    window.removeEventListener('mouseup', onUp)
    setSessionsWidth(sessionsWidth.value)
  }
  window.addEventListener('mousemove', onMove)
  window.addEventListener('mouseup', onUp)
}

/**
 * 展开会话栏时重拉一次列表：页面挂着的时候，别人（CLI、另一个标签页）聊出来的新会话
 * 得能看见。收起时不用拉 —— 列表根本不在屏幕上。
 */
watch(sessionsCollapsed, (collapsed) => {
  if (!collapsed) void refreshSessions()
})

async function reopen(id: string): Promise<void> {
  if (controller) return
  historyError.value = ''
  loadingSession.value = true
  try {
    await openSession(id)
    scrollToEnd()
  } catch (exc) {
    // 拉不到就留着当前这段：总好过把屏幕清空；但**得说出来**
    historyError.value = `读这一段会话失败：${(exc as Error).message}`
  } finally {
    loadingSession.value = false
  }
}

/** 「3 分钟前」这种人话。会话列表里要看的是「哪段是刚才聊的」。 */
function when(stamp: string | null): string {
  if (!stamp) return ''
  const at = Date.parse(stamp)
  if (Number.isNaN(at)) return ''
  const minutes = Math.round((Date.now() - at) / 60000)
  if (minutes < 1) return '刚刚'
  if (minutes < 60) return `${minutes} 分钟前`
  const hours = Math.round(minutes / 60)
  return hours < 24 ? `${hours} 小时前` : `${Math.round(hours / 24)} 天前`
}

function askAboutChunk(event: Event): void {
  const item = (event as CustomEvent).detail as { heading_path?: string; chunk_id: string }
  draft.value = `关于「${item.heading_path || item.chunk_id}」这一段，具体讲了什么？`
  box.value?.focus()
}

// 引用清单是**服务端拼在答案正文最后一段**的（`引用来源：` 起头，见后端的
// `format_sources`）。这里只负责把它切出来折进折叠框 —— **数据仍然只有一条正文**，
// 不额外存第二份。用 `lastIndexOf`：万一正文里真出现了这几个字，也只切最后那一段。
const SOURCES_MARK = '\n\n引用来源：\n'

function answerBody(answer: string): string {
  const at = answer.lastIndexOf(SOURCES_MARK)
  return at < 0 ? answer : answer.slice(0, at)
}

function answerSources(answer: string): string {
  const at = answer.lastIndexOf(SOURCES_MARK)
  return at < 0 ? '' : answer.slice(at + SOURCES_MARK.length).trim()
}

/**
 * 答案的 markdown 渲染。**只在跑完之后渲染**（`turn.running` 为假）：
 *
 * * 流式期间逐 token 重解析 markdown 会卡界面，而且半张表格渲染出来也是花的 ——
 *   那段时间就走纯文本（`white-space: pre-wrap`），和人一边想一边说一样；
 * * 跑完才解析一次，并用 `WeakMap` 按「正文内容」缓存 —— 后续重渲染（比如下一轮
 *   在流式）不会把前面几轮的 markdown 又解析一遍。
 */
const htmlCache = new WeakMap<Turn, { body: string; html: string }>()

/** 这一轮**能点开**的引用编号：只有当轮拿到过 citations 的才有映射（历史里的没有）。 */
function citesOf(turn: Turn): number[] {
  return turn.citations.map((item) => item.index)
}

function answerHtml(turn: Turn): string {
  const body = answerBody(turn.answer)
  const cached = htmlCache.get(turn)
  if (cached && cached.body === body) return cached.html
  const html = renderMarkdown(body, citesOf(turn))
  htmlCache.set(turn, { body, html })
  return html
}

/** 末尾那份「引用来源」清单：同样是 `[n] 文档 · 章节`，编号也做成可点。 */
function sourcesHtml(turn: Turn): string {
  return citeText(answerSources(turn.answer), citesOf(turn))
}

/**
 * 点答案里的 `[2]` → 右侧抽屉看原文（复用检索页那套 `openChunk`）。
 *
 * 事件用**委托**：这段 HTML 是 `v-html` 塞进去的，Vue 绑不上它里面的元素。
 * 从历史恢复的那几轮没有引用映射（服务端只存正文与来源清单，不存 chunk 对应关系），
 * 那些 `[n]` 压根不会被渲染成按钮 —— 不做点了没反应的按钮。
 */
function onCite(turn: Turn, event: MouseEvent): void {
  const target = (event.target as HTMLElement | null)?.closest?.('[data-cite]')
  if (!(target instanceof HTMLElement)) return
  const index = Number(target.dataset.cite)
  const cite = turn.citations.find((item) => item.index === index)
  if (cite) void openChunk(cite.chunk_id)
}

onMounted(() => {
  scroller = document.getElementById('chat-scroll')
  window.addEventListener('rag:focus-search', () => box.value?.focus())
  window.addEventListener('rag:ask-about-chunk', askAboutChunk)
  // 打开就把最近一段接回来 —— 会话在服务端，「刷新之后还在」靠这一步
  void (async () => {
    await refreshSessions()
    loadingSession.value = true
    try {
      await openLatest()
    } catch (exc) {
      // 这一句以前没有：失败时页面停在空态，看着像「历史没了」
      historyError.value = `读最近一段会话失败：${(exc as Error).message}`
    } finally {
      loadingSession.value = false
      scrollToEnd()
    }
  })()
})
onUnmounted(() => {
  window.removeEventListener('rag:ask-about-chunk', askAboutChunk)
})
</script>

<template>
  <header class="main-head">
    <div class="seg" role="group" aria-label="问答模式">
      <button
        :class="{ on: effectiveMode === 'chat' }"
        :disabled="lockedToAgent"
        :title="
          lockedToAgent
            ? '本会话已有历史：快问答没有上下文，接不上。想换就点「新对话」。'
            : '定制链路：单轮、几秒钟、便宜'
        "
        @click="setMode('chat')"
      >
        快问答
      </button>
      <button
        :class="{ on: effectiveMode === 'agent' }"
        title="模型自己决定查几次、派给哪个子 agent：慢、贵，但能多轮追问"
        @click="setMode('agent')"
      >
        深度 agent
      </button>
    </div>

    <button class="btn ghost" title="清空记录与上下文，模式也重置回快问答" @click="startOver()">
      <Plus :size="16" aria-hidden="true" />新对话
    </button>
    <!-- 会话列表**默认就展示**（左列），所以这里不再放「会话」按钮；收起/展开在那一列自己头上 -->
    <button
      v-if="effectiveMode === 'agent'"
      class="btn"
      :class="{ primary: web }"
      title="明确要求联网。默认只在语料里查不到时才允许联网（网络答案无法溯源到语料）。"
      @click="web = !web"
    >
      <Zap :size="16" aria-hidden="true" />{{ web ? '已允许联网' : '允许联网' }}
    </button>
  </header>

  <!-- 页面内的**左列会话栏**（使用者 2026-10-08 定：不要顶部那个按钮，竖着放左边）。
       收起只影响这一列；宽度可以拖（`pane-grip`），两个偏好都记在 localStorage。 -->
  <div
    class="chat-split"
    :class="{ 'pane-off': sessionsCollapsed }"
    :style="{ '--sessions-w': `${sessionsWidth}px` }"
  >
    <aside class="sessions-pane" aria-label="会话列表">
      <div class="pane-head">
        <History :size="14" aria-hidden="true" />
        <span>会话</span>
      <span v-if="sessions.length" class="hint">{{ sessions.length }} 段</span>
        <button
          class="icon-btn"
          :title="'收起会话栏（腾地方）'"
          aria-label="收起会话栏"
          @click="toggleSessionsPane"
        >
          <PanelLeftClose :size="14" aria-hidden="true" />
        </button>
      </div>
      <div class="pane-body">
        <p v-if="sessionsError" class="banner err">读会话列表失败：{{ sessionsError }}</p>
        <p v-if="!sessions.length" class="hint">还没有历史 —— 聊过一轮就会出现在这里。</p>
        <template v-else>
          <label class="hint">
            <input v-model="onlyAgent" type="checkbox" />
            只看 agent
          </label>
          <p v-if="!visibleSessions.length" class="hint">
            没有 agent 会话（这 {{ sessions.length }} 段都是快问答）。
          </p>
          <!-- 一行 = 「选这段」按钮 + 「删这段」按钮**并列**。不能把删除按钮嵌进选择按钮里：
               那是非法 HTML，内层的点击在部分浏览器里不触发。 -->
          <div
            v-for="item in visibleSessions"
            :key="item.session_id"
            class="session-row"
            :class="{ on: item.session_id === sessionId }"
          >
            <button type="button" class="session-pick" @click="reopen(item.session_id)">
              <b>{{ item.title }}</b>
              <span class="hint">
                {{ item.mode === 'agent' ? '深度 agent' : '快问答' }} · {{ item.turns }} 轮 ·
                {{ item.usage.total_tokens }} token · {{ when(item.updated_at) }}
              </span>
            </button>
            <button
              type="button"
              class="icon-btn session-del"
              title="删掉这一段"
              @click.stop="dropSession(item.session_id)"
            >
              <Trash2 :size="13" aria-hidden="true" />
            </button>
          </div>
          <button
            v-if="sessions.length >= sessionLimit && sessionLimit < 200"
            class="btn ghost small"
            @click="loadMoreSessions"
          >
            加载更多
          </button>
          <span v-else-if="sessions.length < sessionLimit" class="hint">（已经到底了）</span>
        </template>
      </div>
    </aside>
    <span
      class="pane-grip"
      role="separator"
      aria-label="拖动调整会话栏宽度"
      title="拖动调整宽度"
      @mousedown="startResize"
    ></span>
    <!-- 收起之后只剩这一条：点它把会话栏叫回来（列表默认是展开的，所以平时看不到它） -->
    <button
      v-if="sessionsCollapsed"
      class="pane-reopen"
      title="展开会话栏"
      aria-label="展开会话栏"
      @click="toggleSessionsPane"
    >
      <PanelLeftOpen :size="14" aria-hidden="true" />
    </button>
    <div class="chat-col">
  <div id="chat-scroll" class="main-body">
    <p v-if="historyError" class="banner err">
      <TriangleAlert :size="16" aria-hidden="true" />{{ historyError }}
    </p>

    <div v-if="loadingSession" class="empty">
      <b>正在读取会话…</b>
      从服务端取这一段的消息（历史在 PG 里，不在这台浏览器上）。
    </div>
    <div v-else-if="!turns.length && sessionId" class="empty">
      <b>这一段没有历史消息</b>
      服务端确实返回了这段会话，但一条消息都没有 —— 多半是那几轮落库失败过（去日志页看
      <code>rag.storage</code>）。也可以点「新对话」重开一段。
    </div>
    <div v-else-if="!turns.length" class="empty">
      <b>问语料里的东西</b>
      快问答走定制链路：单轮、几秒钟。要接着上文追问、要它自己挑路子、要查图谱或联网，
      切「深度 agent」。
    </div>

    <div class="thread">
      <template v-for="(turn, index) in turns" :key="index">
        <div class="msg user">{{ turn.question }}</div>

        <div v-if="turn.mode === 'agent'" class="chips">
          <span class="chip on">深度 agent</span>
          <span v-if="turn.thread" class="chip">会话 {{ turn.thread }}</span>
        </div>

        <details v-if="turn.stats.length" class="tools" open>
          <summary>
            <SearchCheck :size="16" aria-hidden="true" />
            检索证据（{{ turn.hits.length }} 条命中 / {{ turn.stats.length }} 路通道）
          </summary>
          <ol>
            <li v-for="row in turn.stats" :key="row.channel">
              <code>{{ row.channel }}</code>
              <span class="hint">
                召回 {{ row.candidates }} → 入选 {{ row.kept
                }}<template v-if="row.latency_ms"> · {{ row.latency_ms.toFixed(0) }}ms</template>
              </span>
            </li>
          </ol>
        </details>

        <ApprovalCard
          v-if="turn.interrupt"
          :interrupt="turn.interrupt"
          :busy="turn.running"
          @decide="decide(turn, $event)"
        />

        <p v-if="turn.error" class="banner err">
          <TriangleAlert :size="16" aria-hidden="true" />{{ turn.error }}
        </p>

        <div v-if="turn.answer || turn.running" class="msg ai">
          <!-- 流式：纯文本 + 光标（不解析 markdown）；跑完：渲染一次 markdown -->
          <template v-if="turn.running">
            {{ answerBody(turn.answer) }}<span class="caret"></span>
          </template>
          <div v-else class="md" v-html="answerHtml(turn)" @click="onCite(turn, $event)"></div>
        </div>

        <!-- 引用清单由服务端拼在答案正文的最后一段。这里把它**折起来**：
             平时不占地方，想看「引用了哪一章」再展开。存的还是一整条正文，没有第二份数据。 -->
        <details v-if="answerSources(turn.answer)" class="sources-fold">
          <summary>引用来源（{{ answerSources(turn.answer).split('\n').length }} 条）</summary>
          <pre v-html="sourcesHtml(turn)" @click="onCite(turn, $event)"></pre>
        </details>

        <div v-if="turn.usage" class="chips">
          <span class="chip">
            本轮 {{ turn.usage.total_tokens }} token · {{ turn.usage.calls }} 次模型调用
            （入 {{ turn.usage.input_tokens }} / 出 {{ turn.usage.output_tokens }}）
          </span>
        </div>

        <div v-if="turn.weak" class="banner warn">
          <TriangleAlert :size="16" aria-hidden="true" />
          <span>
            语料里没找到足够证据（最高分
            {{ turn.hits.length ? turn.hits[0].score.toFixed(4) : '无命中' }}，判据 &lt;
            {{ WEAK_SCORE }}）。
          </span>
          <button class="btn" @click="escalate(turn)">让 agent 再查一遍</button>
        </div>
      </template>
    </div>
  </div>

  <!-- 输入框放**底部**（对话是这样的：内容往上长，手一直在下面） -->
  <footer class="composer-bar">
    <div class="composer">
      <!-- 语音：录一段 → 转文字 → **填进输入框，不自动发**（识别难免有错字，人改完再回车） -->
      <button
        class="btn"
        :class="{ danger: recording }"
        :disabled="transcribing || !!controller"
        :title="recording ? '停止录音并转文字' : '按住说话，松开转文字'"
        :aria-label="recording ? '停止录音' : '录音'"
        @click="toggleRecording()"
      >
        <component :is="recording ? Square : Mic" :size="16" aria-hidden="true" />
        {{ transcribing ? '识别中…' : recording ? '停止' : '语音' }}
      </button>
      <div class="input">
        <MessagesSquare :size="16" aria-hidden="true" />
        <input
          ref="box"
          v-model="draft"
          aria-label="问题"
          :placeholder="transcribing ? '正在把语音转成文字…' : '问点什么…（Enter 发送，/ 聚焦）'"
          @keyup.enter="ask()"
        />
      </div>
      <button class="btn primary" :disabled="!!controller" @click="ask()">
        {{ controller ? '生成中…' : '发送' }}
      </button>
      <button v-if="controller" class="btn danger" @click="stop()">
        <Square :size="16" aria-hidden="true" />停止
      </button>
    </div>
    <p v-if="micError" class="banner err mic-error">{{ micError }}</p>
  </footer>
    </div>
  </div>
</template>
