<script setup lang="ts">
// 日志页：四类合成一条流水 + 按 request_id 查整条链路。
//
// **三条来源，各管一段**：
//
//   /logs/tail     读 `data/logs/*.log`（CLI 写的也在里面、重启后还在）—— 历史
//   /logs/stream   进程内缓冲（只覆盖 API 自己，但延迟是零）—— 实时
//   /logs/search   查 PG 的索引（`log_line` 表）—— 按 request_id 串一条链路
//
// 后端刻意不让页面去 tail 文件：实测 Windows 上持句柄会把轮转顶掉、丢行。
// **查库与看流水是两个模式**：查库时不再把实时行插进来 —— 两套来源混着显示，
// 看到的那一串就不是"这一次请求发生了什么"了。切回实时就用「回到实时」。
//
// **筛选长在对应列的表头上**（2026-10-08 使用者定）：每列自己的条件从那一列头上的小箭头
// 点开 —— 时间 / 级别 / 类别 / request_id / 正文各管各的，不在顶部横条里挤一排控件。
// 能进 SQL 的只有三个（`request_id` 必填 + 类别 + 时间，见 `/logs/search`）；
// 级别与关键字是**在已经加载出来的行里筛**（文件那条路没有索引），作用范围只到这一屏。
import { computed, nextTick, onMounted, onUnmounted, ref } from 'vue'
import { ChevronDown, Info, Pause, Play, RefreshCw, Search, Trash2 } from 'lucide-vue-next'
import { logSearch, logTail, streamLogs, type LogLine } from '../api'

// 类别名**统一用后端那套英文标识**（`all` 只是「不过滤」）。agent 拆三类：
// agent_run = 一轮汇总、agent_llm = 一次模型调用、agent_tool = 一次工具调用。
const KINDS = ['all', 'interface', 'channel', 'agent_run', 'agent_llm', 'agent_tool', 'system'] as const
// 级别筛选（在已加载的行里筛，不是查库）
const LEVELS = ['INFO', 'WARNING', 'ERROR', 'DEBUG'] as const
// 类别解释：原来是一整段说明挂在页面上，现在收进「类别」筛选里 system 那一项的小图标
// （悬浮才出来）。单挑 system 是因为只有它看不出是什么；其余几个名字本身就是答案。
const KIND_HELP = [
  '第三方库（LangChain / pymilvus / uvicorn / httpx…）',
  '以及我们自己的非业务日志（rag.storage / rag.ocr…）',
  '其余：interface 一次 HTTP 请求 · channel 每一路召回 · agent_run 一轮汇总 ·' +
    ' agent_llm 一次模型调用 · agent_tool 一次工具调用',
]
// 页面上最多留 2000 行：跟后端那个环形缓冲一个量级，再多就不是「看日志」而是「存日志」了
const LIMIT = 2000

// 表头上的筛选（每一列一个）
const kind = ref<string>('all') // 类别：同时管「读哪个文件」与「查库带哪个 category」
const level = ref('') // 级别：已在屏上的行里筛
const keyword = ref('') // 正文：已在屏上的行里筛
const query = ref('') // request_id（查库必填）
const since = ref('')
const until = ref('')
// 哪个表头的筛选浮层开着（同时只开一个：都打开会互相压住）
const openFilter = ref('')
const paused = ref(false)
const connected = ref(false)
const error = ref('')
const files = ref<Record<string, string>>({})
const lines = ref<LogLine[]>([])
const buffered = ref<LogLine[]>([])
const follow = ref(true)
const list = ref<HTMLElement | null>(null)
// 查库那一路的状态：`dbMode` 打开时列表里是库里的行，实时行不再往里插
const dbMode = ref(false)
const dbNote = ref('')
const searching = ref(false)
let abort: AbortController | null = null

const shown = computed(() => {
  const stem = keyword.value.trim().toLowerCase()
  const want = level.value.toUpperCase()
  return lines.value.filter((line) => {
    if (want && line.level.toUpperCase() !== want) return false
    if (!stem) return true
    return `${line.kind} ${line.source} ${line.ts ?? ''} ${line.level} ${line.request_id} ${line.text}`
      .toLowerCase()
      .includes(stem)
  })
})

const summary = computed(() => {
  const counts: Record<string, number> = {}
  for (const line of lines.value) counts[line.kind] = (counts[line.kind] ?? 0) + 1
  return KINDS.filter((name) => name !== 'all' && counts[name])
    .map((name) => `${name} ${counts[name]}`)
    .join(' · ')
})

/** 查库时能选的类别（`all` 只是「不过滤」，不是一个类别）。 */
const SEARCH_KINDS = KINDS.filter((name) => name !== 'all')

/** 底部那行「读的是哪个文件」：**按路径去重**（agent 的三类共用一个 agent.log）。 */
const logFiles = computed(() => {
  const seen = new Set<string>()
  const out: { kind: string; path: string }[] = []
  for (const [kind, path] of Object.entries(files.value)) {
    if (!path || seen.has(path)) continue
    seen.add(path)
    out.push({ kind, path })
  }
  return out
})

function append(items: LogLine[]): void {
  if (!items.length) return
  lines.value.push(...items)
  const extra = lines.value.length - LIMIT
  if (extra > 0) lines.value.splice(0, extra)
  if (follow.value) void stick()
}

async function stick(): Promise<void> {
  await nextTick()
  const box = list.value
  if (box) box.scrollTop = box.scrollHeight
}

function onScroll(): void {
  const box = list.value
  if (!box) return
  // 离底 40px 内算「跟着看」；往上翻就松开，别把人拽回去
  follow.value = box.scrollHeight - box.scrollTop - box.clientHeight < 40
}

function onLine(line: LogLine): void {
  // 查库模式下不接实时行：那会把另一条请求的行混进这一条链路里（见文件头的说明）
  if (dbMode.value) return
  if (kind.value !== 'all' && line.kind !== kind.value) return
  if (paused.value) {
    buffered.value.push(line)
    if (buffered.value.length > LIMIT) buffered.value.splice(0, buffered.value.length - LIMIT)
    return
  }
  append([line])
}

async function reload(): Promise<void> {
  error.value = ''
  dbNote.value = ''
  try {
    const body = await logTail(kind.value, 200)
    files.value = body.files
    lines.value = body.items
    follow.value = true
    void stick()
  } catch (exc) {
    error.value = (exc as Error).message
  }
}

async function connect(): Promise<void> {
  abort?.abort()
  const controller = new AbortController()
  abort = controller
  // 断了就自己接回来：SSE 被中间层掐、后端重启都会走到这儿，不能让人对着一条静止的流水发呆
  while (!controller.signal.aborted) {
    try {
      connected.value = true
      await streamLogs(onLine, controller.signal)
    } catch (exc) {
      if (controller.signal.aborted) break
      error.value = (exc as Error).message
    }
    connected.value = false
    await new Promise((resolve) => setTimeout(resolve, 2000))
  }
}

async function switchKind(next: string): Promise<void> {
  kind.value = next
  // 选完就收起来（使用者报的：选完浮层还杵在那儿，得再点一次才关）
  openFilter.value = ''
  dbMode.value = false
  dbNote.value = ''
  // 切类别要重新拉一次：不然看到的是上一个类别留下的行
  lines.value = []
  await reload()
}

/** 表头上那个小箭头：点一下开/关这一列的筛选浮层。 */
function toggle(name: string): void {
  openFilter.value = openFilter.value === name ? '' : name
}

/** 级别是个单选：点完就收（与类别一致，别让浮层赖着不走）。 */
function pickLevel(name: string): void {
  level.value = name
  openFilter.value = ''
}

function clearTime(): void {
  since.value = ''
  until.value = ''
  openFilter.value = ''
}

function clearKeyword(): void {
  keyword.value = ''
  openFilter.value = ''
}

/** 查一条链路：`request_id` 必填（这个口子要回答的是「这次请求怎么了」）。 */
async function runSearch(): Promise<void> {
  const wanted = query.value.trim()
  if (!wanted) {
    error.value = '先填一个 request_id —— 行里那个 req-… 直接点一下就填进来了'
    return
  }
  searching.value = true
  error.value = ''
  try {
    const body = await logSearch({
      request_id: wanted,
      // 类别跟着表头那个筛选走：`all` 只是「不过滤」，不是真的类别
      category: kind.value === 'all' ? '' : kind.value,
      since: since.value,
      until: until.value,
      limit: LIMIT,
    })
    files.value = body.files
    if (!body.items.length && !dbMode.value) {
      // 空手而归时**不进查库模式、也不清空这一屏**：库里那份是异步批量写的（每 0.25 秒一波），
      // 刚发生的一两行常常还没进去 —— 把整屏擦掉看起来像页面坏了（使用者报的就是这个）。
      dbNote.value = body.note || '库里还没有这几行'
      openFilter.value = ''
      return
    }
    dbMode.value = true
    lines.value = body.items
    dbNote.value = body.note
    follow.value = true
    await stick()
    openFilter.value = '' // 查完收起来，把版面还给列表
  } catch (exc) {
    error.value = (exc as Error).message
  } finally {
    searching.value = false
  }
}

/** 点某一行里的 `req-…`：填进框里并立刻查一次。 */
async function searchBy(requestId: string): Promise<void> {
  query.value = requestId
  await runSearch()
}

async function backToLive(): Promise<void> {
  dbMode.value = false
  dbNote.value = ''
  lines.value = []
  await reload()
}

function togglePause(): void {
  paused.value = !paused.value
  if (!paused.value) {
    append(buffered.value)
    buffered.value = []
  }
}

function clear(): void {
  lines.value = []
  buffered.value = []
}

function levelClass(level: string): string {
  return level.toLowerCase()
}

onMounted(async () => {
  await reload()
  connected.value = true
  await connect()
})
onUnmounted(() => abort?.abort())
</script>

<template>
  <header class="main-head">
    <span class="hint">
      筛选在表头上：点每列标题后的小箭头（时间 / 级别 / 类别 / request_id / 正文）
    </span>
    <div class="head-actions">
      <button class="btn" @click="togglePause">
        <component :is="paused ? Play : Pause" :size="14" aria-hidden="true" />
        {{ paused ? `继续${buffered.length ? ` (+${buffered.length})` : ''}` : '暂停' }}
      </button>
      <button class="btn ghost" title="重新读一遍文件（CLI 写的行靠它）" @click="reload">
        <RefreshCw :size="14" aria-hidden="true" />刷新
      </button>
      <button class="btn ghost" title="只清空这一屏" @click="clear">
        <Trash2 :size="14" aria-hidden="true" />清屏
      </button>
      <button v-if="dbMode" class="btn primary" title="退出查库模式，回到实时流水" @click="backToLive">
        回到实时
      </button>
    </div>
  </header>

  <div class="main-body" @click="openFilter = ''">
    <div class="chips">
      <span class="chip" :class="{ on: connected }">
        {{ dbMode ? '查库模式（不收实时行）' : connected ? '实时已连接' : '实时未连接' }}
      </span>
      <span class="chip">{{ lines.length }} 行{{ summary ? ` · ${summary}` : '' }}</span>
      <span v-if="dbMode && query.trim()" class="chip on">req={{ query.trim() }}</span>
      <span v-if="!follow" class="hint right">已松开跟随（滚到底自动恢复）</span>
    </div>

    <p v-if="error" class="banner err">{{ error }}</p>
    <p v-if="dbNote" class="banner warn">{{ dbNote }}</p>

    <div ref="list" class="panel logs" @scroll.passive="onScroll">
      <div v-if="!shown.length" class="empty">
        <b>还没有日志</b>
        跑一次检索或问答就有了。想看 CLI 那边（`rag ingest` / `rag ask`）的，点上面的「刷新」。
      </div>
      <div class="log-head">
        <div class="head-cell">
          <span class="head-label">时间</span>
          <button
            class="caret"
            :class="{ on: since || until }"
            title="按时间筛（查库）"
            @click.stop="toggle('time')"
          >
            <ChevronDown :size="12" aria-hidden="true" />
          </button>
          <div v-if="openFilter === 'time'" class="filter-pop" @click.stop>
            <label>起始<input v-model="since" type="datetime-local" aria-label="起始时间" /></label>
            <label>结束<input v-model="until" type="datetime-local" aria-label="结束时间" /></label>
            <p class="hint">时间只在查库这条路上生效；文件那条没有索引，只读最近几行</p>
            <div class="pop-actions">
              <button
                class="btn small primary"
                :disabled="searching || !query.trim()"
                @click="runSearch"
              >
                按时间查
              </button>
              <button class="btn small ghost" @click="clearTime">清除</button>
            </div>
          </div>
        </div>

        <div class="head-cell">
          <span class="head-label">级别</span>
          <button
            class="caret"
            :class="{ on: level }"
            title="按级别筛（这一屏）"
            @click.stop="toggle('level')"
          >
            <ChevronDown :size="12" aria-hidden="true" />
          </button>
          <div v-if="openFilter === 'level'" class="filter-pop" @click.stop>
            <button class="pop-item" :class="{ on: !level }" @click="pickLevel('')">全部级别</button>
            <button
              v-for="name in LEVELS"
              :key="name"
              class="pop-item"
              :class="{ on: level === name }"
              @click="pickLevel(name)"
            >
              {{ name }}
            </button>
            <p class="hint">在已加载的行里筛</p>
          </div>
        </div>

        <div class="head-cell"><span class="head-label">来源</span></div>

        <div class="head-cell">
          <span class="head-label">类别</span>
          <button
            class="caret"
            :class="{ on: kind !== 'all' }"
            title="按类别筛"
            @click.stop="toggle('kind')"
          >
            <ChevronDown :size="12" aria-hidden="true" />
          </button>
          <div v-if="openFilter === 'kind'" class="filter-pop" @click.stop>
            <button
              class="pop-item"
              :class="{ on: kind === 'all' }"
              @click="switchKind('all')"
            >
              全部类别
            </button>
            <div
              v-for="name in SEARCH_KINDS"
              :key="name"
              class="pop-item"
              :class="{ on: kind === name }"
              role="button"
              tabindex="0"
              @click="switchKind(name)"
              @keyup.enter="switchKind(name)"
            >
              <span>{{ name }}</span>
              <!-- 「system 是什么」只有它需要解释：悬停（或键盘聚焦）这个图标才出来 -->
              <span
                v-if="name === 'system'"
                class="kind-help"
                tabindex="0"
                aria-label="system 是什么"
                @click.stop
              >
                <Info :size="12" aria-hidden="true" />
                <span class="tip">
                  <b>system</b>
                  <span v-for="item in KIND_HELP" :key="item" class="tip-line">{{ item }}</span>
                </span>
              </span>
            </div>
            <p class="hint">切类别会重新读一遍文件；查库时它也是 category 条件</p>
          </div>
        </div>

        <div class="head-cell">
          <span class="head-label">request_id</span>
          <button class="caret" :class="{ on: dbMode }" title="按 request_id 查" @click.stop="toggle('req')">
            <ChevronDown :size="12" aria-hidden="true" />
          </button>
          <div v-if="openFilter === 'req'" class="filter-pop wide" @click.stop>
            <input
              v-model="query"
              class="req-input"
              placeholder="req-1a2b3c4d（必填）"
              aria-label="按 request_id 查"
              @keyup.enter="runSearch"
            />
            <p class="hint">
              库里那份是异步批量写的（每 0.25 秒一波）；刚发生的一两行点「刷新」看文件
            </p>
            <div class="pop-actions">
              <button
                class="btn small primary"
                :disabled="searching || !query.trim()"
                @click="runSearch"
              >
                <Search :size="12" aria-hidden="true" />{{ searching ? '查中…' : '查这条链路' }}
              </button>
              <button v-if="dbMode" class="btn small ghost" @click="backToLive">回到实时</button>
            </div>
          </div>
        </div>

        <div class="head-cell">
          <span class="head-label">正文</span>
          <button
            class="caret"
            :class="{ on: keyword }"
            title="在正文里找关键字"
            @click.stop="toggle('text')"
          >
            <ChevronDown :size="12" aria-hidden="true" />
          </button>
          <div v-if="openFilter === 'text'" class="filter-pop wide" @click.stop>
            <input
              v-model="keyword"
              class="req-input"
              placeholder="关键字（这一屏里筛）"
              aria-label="过滤日志正文"
              @keyup.enter="openFilter = ''"
            />
            <div class="pop-actions">
              <button class="btn small ghost" @click="clearKeyword">清除</button>
            </div>
          </div>
        </div>
      </div>
      <div v-if="shown.length" class="logs-list">
        <div
          v-for="(line, index) in shown"
          :key="`${index}-${line.ts}-${line.text.slice(0, 24)}`"
          class="log-line"
          :class="levelClass(line.level)"
        >
          <span class="log-time">{{ line.ts ?? '—' }}</span>
          <span class="log-level">{{ line.level || '·' }}</span>
          <span class="log-source" :title="line.source">{{ line.source || '·' }}</span>
          <span class="log-kind">{{ line.kind }}</span>
          <span class="log-req">
            <button
              v-if="line.request_id"
              class="req"
              :class="{ on: dbMode && line.request_id === query.trim() }"
              :title="`按 ${line.request_id} 串出整条链路`"
              @click="searchBy(line.request_id)"
            >
              {{ line.request_id }}
            </button>
          </span>
          <span class="log-msg">
            <span v-if="line.origin" class="origin" title="这条是谁发起的">{{ line.origin }}</span>
            <span class="msg-text">{{ line.text }}</span>
          </span>
        </div>
      </div>
    </div>

    <p class="hint">
      实时那条只覆盖 API 这个进程（CLI 写的行点「刷新」读文件）；每行的 request_id 可以点 ——
      点一下就把这次请求的 interface + channel + agent 串成一条。读的是
      <code v-for="item in logFiles" :key="item.path">{{ item.kind }}={{ item.path }}</code>
    </p>
  </div>
</template>
