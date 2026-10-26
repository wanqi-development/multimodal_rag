<script setup lang="ts">
// 文档页：上传 / 从链接导入 → 解析 → **核对标题层级** → 切分 → 入库（页面那五步）。
//
// 三条纪律写死在这里：
//
// 1. **正文只读**。页面没有任何入口改正文 —— 能改的只有「这一行是几级」。正文一改，
//    偏移 → 页码/行号的映射就全错位，而那份映射是「点引用看原文」的根。要改内容，
//    改源文件重传。
// 2. **层级编辑嵌在文档里**（不是另开一张表）：标题行就地一个层级下拉 + 「标为正文」，
//    其余段落照常渲染。这样「第几行是几级」看得见上下文，也才对得上解析稿本身。
// 3. **慢步骤一律 202 + 轮询 `/jobs`**。PDF 走 OCR/VLM 是分钟级，卡在一个请求里既看不到
//    进度、又会被中间层掐断。**右边的对比不算慢步骤**：它只做字符串切分，零成本、不落盘。
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import {
  Check,
  ChevronDown,
  ChevronRight,
  FileText,
  Folder as FolderIcon,
  FolderInput,
  FolderPlus,
  Link2,
  Maximize2,
  Minimize2,
  Pencil,
  RefreshCw,
  Scissors,
  Trash2,
  Upload,
  X,
} from 'lucide-vue-next'
import ChunkReport from '../components/ChunkReport.vue'
import FlowSteps, { type StepState } from '../components/FlowSteps.vue'
import SourceView from '../components/SourceView.vue'
import {
  assignFolder,
  createFolder,
  deleteDocument,
  deleteFolder,
  documentDetail,
  documents as documentList,
  fetchDocument,
  indexDocument,
  jobs as fetchJobs,
  parseDocument,
  saveOutline,
  sourceUrl,
  updateFolder,
  uploadDocuments,
  type DocumentDetail,
  type DocumentRow,
  type Folder,
  type Job,
  type OutlineRow,
  type Uploaded,
  type UploadResult,
} from '../api'
import { markdownSegments, renderMarkdown } from '../markdown'

const docs = ref<DocumentRow[]>([])
/** Milvus 够不着时「入库状态」是不知道，不能当成没入库（后端用 indexed_known 说明） */
const indexedKnown = ref(true)
const error = ref('')
/** 上传被拒 / 导入结果这类「说清楚就行」的结果，和报错分开放 */
const notice = ref('')
/** **同名但内容不一样**的旧版本：摆出来等用户选「覆盖更新 / 保留两份」 */
const conflicts = ref<Uploaded[]>([])
const loading = ref(false)
const link = ref('')
const fetching = ref(false)
const currentId = ref<string | null>(null)
const detail = ref<DocumentDetail | null>(null)
/** 清单的编辑副本：提交前动的是它，`detail.outline` 保持「服务端给的那一份」好比对 */
const draft = ref<OutlineRow[]>([])
const maxLevel = ref(3)
const confirmed = ref(false)
const busy = ref('')
const jobNote = ref('')
/** 核对步骤的「全屏对比」：把工具栏 + 两栏铺满窗口（层级编辑与提交都在里面） */
const reviewFull = ref(false)
/** 核对左栏「只看标题」：把正文段收起来，方便一行行过标题（长文档翻起来快很多）。 */
const bodyCollapsed = ref(false)
/**
 * 上传 / 抓取**进行中**的动态提示（整句；空串 = 没在跑）。
 *
 * 为什么要有它：这两个入口都在**左列顶栏**，而详情面板可能停在别的步骤上 —— 点完「上传」之后
 * 页面看着没反应（PDF 落盘+抓取要几秒），最容易被当成"没点上"。现在一开始就把详情切到
 * 「上传 / 导入」这一步，并在那里转一个动态提示。
 */
const inflow = ref('')
/** 本进程最近的任务：流程图靠它认出「这份文档正在解析 / 正在入库」（刷新页面也认得出） */
const jobs = ref<Job[]>([])
let lastRunningJob: string | null = null
let jobTimer: number | null = null
/**
 * 轮询间隔（使用者 2026-10-10 定：**5 秒**）。
 *
 * 定时器**只在有任务在跑时存在** —— 空闲时一次都不该调（见 `ensurePolling`）。
 */
const JOB_POLL_MS = 5000
/** 上一次拉详情的时间（毫秒）：拿它和任务的 updated_at 比，决定要不要重拉 */
let detailAt = 0
/** 点回已完成的节点看那一步的结果；`null` = 跟着「当前该做的」那一步 */
const pinnedStep = ref<number | null>(null)
let alive = true

const LEVELS = [1, 2, 3, 4, 5, 6]

/** 主流程四步。**不拆成五个**：后端「拆分」与「入库」是一个 job、不报子阶段，
 *  分开画就只能靠猜 —— 宁可标一个「拆分并入库中…」。 */
const STEPS = [
  { key: 'upload', label: '上传 / 导入', hint: '文件或链接进来、落在盘上' },
  { key: 'parse', label: '解析', hint: '读成正文 + 标题清单（PDF 走 OCR，分钟级）' },
  { key: 'review', label: '核对标题', hint: '改层级、标正文，和未切分前左右对比' },
  { key: 'index', label: '拆分并入库', hint: '切块 → 算向量 → 写 Milvus' },
] as const

/** 页面上一律显示真名：落盘名带内容哈希，是给排查用的，不是给人看的。 */
function displayName(row: { display_name?: string; name: string }): string {
  return row.display_name || row.name
}

/** 链接导入的那些：列一行「来源域名」，好和上传件区分开。 */
function sourceHost(row: DocumentRow): string {
  if (!row.source_url) return ''
  try {
    return new URL(row.source_url).host
  } catch {
    return row.source_url.slice(0, 40)
  }
}

const current = computed(() => docs.value.find((row) => row.doc_id === currentId.value) ?? null)
const saved = computed(() => detail.value?.outline ?? null)

/** 这份文档正在跑的任务（没有就是 null）。排队中也算「进行中」。 */
const runningJob = computed<Job | null>(() => {
  const docId = currentId.value
  if (!docId) return null
  return (
    jobs.value.find(
      (job) =>
        (job.status === 'queued' || job.status === 'running') &&
        String((job.request as Record<string, unknown>).doc_id ?? '') === docId,
    ) ?? null
  )
})

/**
 * **本进程还有任务在跑吗**（不限于当前这份文档）。
 *
 * 轮询要不要继续看的是它，不是 `runningJob` —— 用户切到另一份文档时，`runningJob`
 * 会是 null，但别的文档那条任务还在跑，这时候停表就再也发现不了它结束了。
 */
const anyRunning = computed(() =>
  jobs.value.some((job) => job.status === 'queued' || job.status === 'running'),
)

/**
 * 四步各自完成了没有。
 *
 * **列表那一份只作兜底、详情那一份优先**：详情里的 `index.indexed` 是**按 doc_id 单查注册表**
 * 拿到的，而列表里的 `stages.indexed` 要先把整张注册表读一遍 —— 那一步失败（Milvus 抖一下）
 * 就会让所有文档显示成「没入库」，页面上看起来像是根本没做。两个信号取"有一个说做了就算做了"。
 */
const flowDone = computed<boolean[]>(() => {
  const row = current.value
  const detailRow = detail.value
  if (!row) return [false, false, false, false]
  const indexed = Boolean(detailRow?.index?.indexed) || Boolean(row.stages.indexed)
  return [
    true, // 能出现在清单里就说明已经落盘了
    detailRow?.stages.parsed ?? row.stages.parsed,
    detailRow?.stages.reviewed ?? row.stages.reviewed,
    indexed,
  ]
})

/** 当前该做的那一步（4 = 四步都完了）。 */
const stageIndex = computed(() => {
  const next = flowDone.value.findIndex((done) => !done)
  return next === -1 ? flowDone.value.length : next
})

/** 流程图每个节点的状态：已到=蓝、进行中=半蓝半灰转圈、当前=蓝圈灰底、未到=灰。 */
const stepStates = computed<StepState[]>(() =>
  STEPS.map((step, index) => {
    if (runningJob.value?.kind === step.key) return 'running'
    if (flowDone.value[index]) return 'done'
    return index === stageIndex.value ? 'current' : 'todo'
  }),
)

/** 下方面板显示哪一步：点了节点就跟着点，否则跟着「当前该做的」。 */
const panelStep = computed(() => pinnedStep.value ?? Math.min(stageIndex.value, STEPS.length - 1))

const dirty = computed(() => {
  const out = saved.value
  if (!out) return false
  if (out.max_level !== maxLevel.value || out.confirmed !== confirmed.value) return true
  return out.rows.some((row, index) => row.level !== draft.value[index]?.level)
})

/**
 * 解析稿切成「标题行」与「正文段」两类块，**按行切**。
 *
 * 判定必须与后端的 `outline.heading_lines` **同一套规则** —— 含**跳过代码围栏**：
 * 代码块里的一行 `# Run the agent`（注释）不是标题。两边不一致的话，左边第 n 个标题行就会
 * 对到清单的第 m 条，那种错位最难查（而且标题行数会比清单多）。
 * 规则实现在 `markdown.ts` 的 `markdownSegments()`：与渲染那边共用同一份围栏判据。
 */
const segments = computed(() => markdownSegments(detail.value?.markdown ?? ''))

/** 渲染过的正文段缓存：改层级不该让整篇 markdown 重解析一遍。 */
const mdCache = new Map<string, string>()

function mdOf(text: string): string {
  const hit = mdCache.get(text)
  if (hit !== undefined) return hit
  const html = renderMarkdown(text)
  if (mdCache.size > 200) mdCache.clear()
  mdCache.set(text, html)
  return html
}

/** 「标为正文」/「当回标题」。层级 0 = 这一行不是标题（见 `outline.py`）。 */
function toggleBody(index: number): void {
  const row = draft.value[index]
  if (!row) return
  row.level = row.level === 0 ? row.hint || 1 : 0
}

function levelOf(index: number): number {
  return draft.value[index]?.level ?? 1
}

/**
 * 「切到第几级」这一档会**并进上一节**的标题行数。
 *
 * 为什么要有这个数：这个档位只影响**切分**（`outline.sections` 只把 `1 <= level <= max_level`
 * 的行当边界），核对界面本身不会因为它变 —— 于是"选了深度像没生效"。这里当场把影响算出来。
 */
const mergedCount = computed(
  () => draft.value.filter((row) => row.level > maxLevel.value && row.level > 0).length,
)

/** 这份已经入库了：解析 / 核对 / 入库的入口一律关掉（删除、覆盖更新、查看类的不受影响）。 */
const locked = computed(() => Boolean(current.value?.stages.indexed))

function stagesOf(row: DocumentRow | null): { label: string; done: boolean }[] {
  const stages = row?.stages
  return [
    { label: '已解析', done: Boolean(stages?.parsed) },
    { label: '已核对', done: Boolean(stages?.reviewed) },
    { label: '已切分', done: Boolean(stages?.chunked) },
    { label: '已入库', done: Boolean(stages?.indexed) },
  ]
}

async function reload(silent = false): Promise<void> {
  loading.value = true
  try {
    const body = await documentList()
    docs.value = body.documents
    indexedKnown.value = body.indexed_known
    folders.value = body.folders
    if (!silent) error.value = ''
  } catch (exc) {
    error.value = (exc as Error).message
  } finally {
    loading.value = false
  }
  if (currentId.value) await open(currentId.value, true)
}

// ---- 人工目录（左栏那棵树）----
// **纯页面上的组织**：后端只动 `data/uploads/_folders.json`，不改 doc_id、不挪文件、
// 不碰 Milvus，也不影响检索。最多三层（后端挡的，这里跟着显示）。
const folders = ref<Folder[]>([])
/** 展开的目录。**默认全展开** —— 文档都藏在折叠的目录里，页面看着就像空的 */
const openFolders = ref<Set<string>>(new Set())
/** 勾选的文档：批量「移动到…」用 */
const picked = ref<Set<string>>(new Set())
/** 「移动到…」的下拉开着没有 */
const moveOpen = ref(false)
/** 正在输入的那一格：`parent` = 新建在哪个目录下，`edit` = 在改哪个目录的名字 */
const naming = ref<{ parent: string | null; edit: string | null } | null>(null)
const draftName = ref('')
/** 上传 / 导入时的目标目录（空 = 未分类） */
const targetFolder = ref('')
/** 正在拖的那几份文档（拖一条 = 一条；拖的那条是勾选的就整批一起拖） */
const dragging = ref<string[]>([])
/** 拖着东西路过的那个目录（空串 = 未分类那一格）——只用来画高亮 */
const dragOver = ref('')

type TreeRow =
  | { kind: 'folder'; folder: Folder; depth: number; count: number }
  | { kind: 'doc'; doc: DocumentRow; depth: number }
  | { kind: 'input'; depth: number }
  | { kind: 'label'; text: string }

/** 子目录，按名字排（后端给的是「建的时候的顺序」，展示按名字更像文件树）。 */
function childrenOf(parent: string | null): Folder[] {
  return folders.value
    .filter((item) => (item.parent ?? null) === parent)
    .sort((a, b) => a.name.localeCompare(b.name, 'zh'))
}

function docsIn(folderId: string): DocumentRow[] {
  return docs.value.filter((row) => (row.folder || '') === folderId)
}

/** 这个目录连子目录一共几份 —— 折叠着的时候也要看得出里面有多少东西。 */
function countIn(folderId: string): number {
  return (
    docsIn(folderId).length +
    childrenOf(folderId).reduce((total, child) => total + countIn(child.id), 0)
  )
}

/** 摊平成一列来渲染：目录、文档、正在输入的那一格，都带缩进深度。 */
const treeRows = computed<TreeRow[]>(() => {
  const rows: TreeRow[] = []
  const walk = (parent: string | null, depth: number): void => {
    for (const folder of childrenOf(parent)) {
      // 正在改名的这一格**就地变成输入框**（连它的子项一起先不画：这是编辑中的瞬态）
      if (naming.value?.edit === folder.id) {
        rows.push({ kind: 'input', depth })
        continue
      }
      rows.push({ kind: 'folder', folder, depth, count: countIn(folder.id) })
      // 「在里面新建子目录」的输入框：挂在它下面、缩进一格
      if (naming.value && naming.value.edit === null && naming.value.parent === folder.id) {
        rows.push({ kind: 'input', depth: depth + 1 })
      }
      if (!openFolders.value.has(folder.id)) continue
      for (const doc of docsIn(folder.id)) rows.push({ kind: 'doc', doc, depth: depth + 1 })
      walk(folder.id, depth + 1)
    }
    if (parent !== null) return
    if (naming.value && naming.value.edit === null && naming.value.parent === null) {
      rows.push({ kind: 'input', depth: 0 })
    }
    // 有目录才打这个头。**没归类的文档为 0 时也留着**：它同时是「拖回未分类」的落点，
    // 有时候才是最需要的（刚把最后一份拖走，正想拖回来）
    if (folders.value.length) rows.push({ kind: 'label', text: '未分类' })
    for (const doc of docsIn('')) rows.push({ kind: 'doc', doc, depth: 0 })
  }
  walk(null, 0)
  return rows
})

const unfiled = computed(() => docsIn('').length)

/** 「移动到…」的候选：未分类 + 所有目录（按树序摊平、带缩进）。 */
const moveTargets = computed(() => [
  { id: '', name: '未分类', depth: 0 },
  ...flatten(null, 0),
])

function flatten(parent: string | null, depth: number): { id: string; name: string; depth: number }[] {
  return childrenOf(parent).flatMap((folder) => [
    { id: folder.id, name: folder.name, depth },
    ...flatten(folder.id, depth + 1),
  ])
}

function toggleFolder(folderId: string): void {
  const next = new Set(openFolders.value)
  if (next.has(folderId)) next.delete(folderId)
  else next.add(folderId)
  openFolders.value = next
}

function rowKey(row: TreeRow, index: number): string {
  if (row.kind === 'folder') return `f-${row.folder.id}`
  if (row.kind === 'doc') return `d-${row.doc.doc_id}`
  return `x-${index}`
}

/** 缩进：目录与文档都按深度往里挪一格 */
function indent(depth: number): { paddingLeft: string } {
  return { paddingLeft: `${8 + depth * 16}px` }
}

function startCreate(parent: string | null): void {
  naming.value = { parent, edit: null }
  draftName.value = ''
  moveOpen.value = false
}

function startRename(folder: Folder): void {
  naming.value = { parent: folder.parent ?? null, edit: folder.id }
  draftName.value = folder.name
}

function cancelNaming(): void {
  naming.value = null
  draftName.value = ''
}

async function commitNaming(): Promise<void> {
  const here = naming.value
  const name = draftName.value.trim()
  cancelNaming()
  if (!here || !name) return
  error.value = ''
  try {
    if (here.edit) await updateFolder(here.edit, { name })
    else await createFolder(name, here.parent)
    await reload(true)
  } catch (exc) {
    error.value = (exc as Error).message
  }
}

async function removeFolder(folder: Folder): Promise<void> {
  const inside = countIn(folder.id)
  const tail = inside ? `里面的 ${inside} 份文档会回到「未分类」，文件不会被删。` : ''
  if (!window.confirm(`删掉目录「${folder.name}」？${tail}`)) return
  error.value = ''
  try {
    await deleteFolder(folder.id)
    await reload(true)
  } catch (exc) {
    error.value = (exc as Error).message
  }
}

function togglePick(docId: string): void {
  const next = new Set(picked.value)
  if (next.has(docId)) next.delete(docId)
  else next.add(docId)
  picked.value = next
}

/** 把这几份文档挪进一个目录（`folderId` 为空串 = 未分类）。下拉与拖拽走同一条。 */
async function moveDocs(ids: string[], folderId: string): Promise<void> {
  if (!ids.length) return
  error.value = ''
  try {
    const result = await assignFolder(ids, folderId || null)
    notice.value = result.moved
      ? `已把 ${result.moved} 份文档放进「${folderId ? folderName(folderId) : '未分类'}」。`
      : '这些文档已经在那儿了。'
    picked.value = new Set()
    await reload(true)
  } catch (exc) {
    error.value = (exc as Error).message
  }
}

async function movePicked(folderId: string): Promise<void> {
  moveOpen.value = false
  await moveDocs([...picked.value], folderId)
}

// ---- 拖拽：把文档行拖到目录行上就归进去 ----
// 用浏览器原生的 HTML5 拖拽（`draggable` + dragstart/dragover/drop），不引第三方库：
// 这里只有「拖一份文档、放到一个目录上」这一种动作，落点判定交给 DOM 就够了。

/** 拖起来的这几份：拖的那条**在勾选里就整批一起拖**，否则只拖它自己。 */
function startDrag(event: DragEvent, docId: string): void {
  dragging.value = picked.value.has(docId) ? [...picked.value] : [docId]
  // 有些浏览器不带 dataTransfer 的拖动会被当成没有内容、光标是「禁止」
  event.dataTransfer?.setData('text/plain', dragging.value.join(','))
  if (event.dataTransfer) event.dataTransfer.effectAllowed = 'move'
}

function endDrag(): void {
  dragging.value = []
  dragOver.value = ''
}

/** 放下。`folderId` 为空串 = 放进「未分类」那一格。 */
async function dropOn(folderId: string): Promise<void> {
  const ids = dragging.value
  endDrag()
  await moveDocs(ids, folderId)
}

function folderName(folderId: string): string {
  return folders.value.find((item) => item.id === folderId)?.name ?? folderId
}

/** 上传 / 导入落盘之后放进选中的目录（没选就不用调接口）。 */
async function fileInto(body: UploadResult): Promise<void> {
  const ids = body.uploads.map((item) => item.doc_id)
  if (!targetFolder.value || !ids.length) return
  try {
    await assignFolder(ids, targetFolder.value)
  } catch (exc) {
    // 文件已经在盘上了 —— 只是没归成类，说清楚让人手动挪一次就行
    error.value = `已经落盘，但没能放进目录：${(exc as Error).message}`
  }
}

/** 选中一份：拉详情 + 把清单拷进编辑副本。 */
async function open(docId: string, silent = false): Promise<void> {
  currentId.value = docId
  pinnedStep.value = null
  reviewFull.value = false
  try {
    const body = await documentDetail(docId)
    detail.value = body
    detailAt = Date.now()
    draft.value = (body.outline?.rows ?? []).map((row) => ({ ...row }))
    maxLevel.value = body.outline?.max_level ?? 3
    confirmed.value = body.outline?.confirmed ?? false
    if (!silent) {
      error.value = ''
      notice.value = ''
      jobNote.value = ''
    }
  } catch (exc) {
    error.value = (exc as Error).message
  }
}

/**
 * 拉一次任务列表。两个用途：给流程图供应「进行中」（**刷新页面之后照样认得出来**），
 * 以及在某条任务落地时把阶段状态与各步摘要刷新一遍 —— 不然得让人手动点刷新。
 *
 * **只拉一次**，"要不要接着拉"交给外面的 `pollJobs` 决定（这里有早返回，写在里面会漏）。
 */
async function pollOnce(): Promise<void> {
  try {
    jobs.value = (await fetchJobs(50)).jobs
  } catch {
    return // 轮询失败不报错：后端重启一下，下一轮就回来了
  }
  const now = runningJob.value
  if (lastRunningJob !== null && now?.id !== lastRunningJob) {
    // 刚才那条落地了：刷新阶段状态与各步摘要；失败就把原因摆出来
    const finished = jobs.value.find((job) => job.id === lastRunningJob)
    lastRunningJob = null
    jobNote.value = ''
    if (finished?.status === 'failed') error.value = finished.error ?? '任务失败'
    await reload(true)
    return
  }
  if (now) lastRunningJob = now.id

  // **不依赖「抓到结束的那一瞬间」**：浏览器会把后台标签页的定时器降频到一分钟一次，
  // 那一瞬间错过了就永远停在旧状态（实测就是这么来的：任务早跑完了，按钮还亮着、
  // 结果也不出来）。所以这里不问"有没有变化"，只问一件事 ——
  // **这份文档最近一次任务的时间，比我手上这份详情的还新吗？** 是就重拉。
  const docId = currentId.value
  if (!docId || now) return
  const newest = jobs.value
    .filter((job) => String((job.request as Record<string, unknown>).doc_id ?? '') === docId)
    .reduce((latest, job) => Math.max(latest, job.updated_at * 1000), 0)
  if (newest > detailAt) {
    const finished = jobs.value.find((job) => job.updated_at * 1000 === newest)
    if (finished?.status === 'failed') error.value = finished.error ?? '任务失败'
    await reload(true)
  }
}

/**
 * 轮询**只在有任务在跑时存在**（使用者 2026-10-10 定）。
 *
 * 以前是一个常驻的 2.5 秒定时器，页面开着就一直打 `/jobs` —— 没任务时纯属空转。
 * 现在：挂载时查一次（用来发现"刷新前就在跑的任务"）、发起任务时立刻起表、
 * 任务全部落地就停表。**空闲时 0 次调用**。
 */
function ensurePolling(): void {
  if (!alive || jobTimer !== null) return
  // 后台标签页不轮询：浏览器会把定时器降频，轮了也不准；切回来时 `onVisible` 会补一次
  if (document.visibilityState !== 'visible') return
  jobTimer = window.setInterval(() => {
    if (alive) void pollJobs()
  }, JOB_POLL_MS)
}

function stopPolling(): void {
  if (jobTimer !== null) {
    window.clearInterval(jobTimer)
    jobTimer = null
  }
}

async function pollJobs(): Promise<void> {
  await pollOnce()
  // 还有任务在跑就接着轮；全落地了就停 —— **结束那一下不再多调**（结果已经在
  // `pollOnce` 里刷过了）。任务本身不会"凭空出现"，所以停表之后不必再探。
  if (anyRunning.value) ensurePolling()
  else stopPolling()
}

async function onPick(event: Event): Promise<void> {
  const input = event.target as HTMLInputElement
  const files = Array.from(input.files ?? [])
  // 清空：同一个文件再传一次也得触发 change
  input.value = ''
  if (!files.length) return
  notice.value = ''
  inflow.value = `正在上传 ${files.length} 个文件…`
  pinnedStep.value = 0 // 先把详情切到「上传 / 导入」这一步：不然点了上传，页面看着没反应
  try {
    const body = await uploadDocuments(files)
    applyResult(body)
    await fileInto(body)
    error.value = ''
    // 传完把面板切到**这一份**（`open()` 会清掉上面那个 pin；新文档停在 `uploaded`，
    // 于是自然落在第一步 —— 正好看到刚进来的原文）
    const first = body.uploads[0]
    if (first) await open(first.doc_id)
  } catch (exc) {
    error.value = (exc as Error).message
    pinnedStep.value = null // 失败就别把面板钉在第一步
  } finally {
    inflow.value = ''
  }
  await reload(true)
}

/** 从链接导入。类型不定（HTML 文档页 / PDF / Word 下载链接都行），后端负责判定。 */
async function importLink(): Promise<void> {
  const url = link.value.trim()
  if (!url || fetching.value) return
  fetching.value = true
  notice.value = ''
  inflow.value = `正在抓取 ${url}…`
  pinnedStep.value = 0 // 先把详情切到「上传 / 导入」这一步：抓取要几秒，别让人对着别的步骤等
  try {
    const body = await fetchDocument({ url })
    applyResult(body)
    await fileInto(body)
    error.value = ''
    link.value = ''
    // 抓完切到抓回来的那一份（`open()` 会清掉上面那个 pin，新文档停在 `uploaded` → 第一步）
    const first = body.uploads[0]
    if (first) await open(first.doc_id)
  } catch (exc) {
    error.value = (exc as Error).message
    pinnedStep.value = null // 失败就别把面板钉在第一步
  } finally {
    fetching.value = false
    inflow.value = ''
  }
  await reload(true)
}

function applyResult(body: UploadResult): void {
  const lines = body.uploads.map(
    (item) =>
      `${item.replaced ? '内容一致，已是最新' : '已导入'}：${displayName(item)}（${item.doc_id}）`,
  )
  conflicts.value = body.conflicts
  const parts = [...lines, ...body.skipped]
  // 图没抓全要说出来：它不挡这次导入，但会让检索结果悄悄变差
  if (body.notes.length) parts.push(`${body.notes.length} 张图没拿到 —— ${body.notes.join('；')}`)
  if (conflicts.value.length) parts.push(`${conflicts.value.length} 份同名但内容不同 —— 见下面的提示`)
  notice.value = parts.join('；')
}

/** 「覆盖更新」：把旧版本连它的产物与已入库内容一起清掉（不可逆，所以按钮是红的）。 */
async function replaceOld(): Promise<void> {
  const olds = conflicts.value.slice()
  conflicts.value = []
  busy.value = 'delete'
  try {
    for (const item of olds) await deleteDocument(item.doc_id)
    notice.value = `已覆盖 ${olds.length} 份旧版本（文件、解析产物与已入库内容都清了）`
    error.value = ''
    if (currentId.value && olds.some((item) => item.doc_id === currentId.value)) {
      currentId.value = null
      detail.value = null
      draft.value = []
    }
  } catch (exc) {
    error.value = (exc as Error).message
  } finally {
    busy.value = ''
  }
  await reload(true)
}

async function remove(row: DocumentRow): Promise<void> {
  const name = displayName(row)
  if (!window.confirm(`删掉《${name}》？文件、解析产物与已入库内容都会清掉，不可逆。`)) return
  busy.value = 'delete'
  try {
    await deleteDocument(row.doc_id)
    if (currentId.value === row.doc_id) {
      currentId.value = null
      detail.value = null
      draft.value = []
    }
    notice.value = `已删除：${name}`
    error.value = ''
  } catch (exc) {
    error.value = (exc as Error).message
  } finally {
    busy.value = ''
  }
  await reload(true)
}

/**
 * 跑一个慢步骤：拿到 job 之后**轮询到它落地**。
 *
 * 不 await 在同一个请求里等 —— 解析几分钟，中间层（代理/浏览器）早就掐了。
 */
async function run(kind: 'parse' | 'index'): Promise<void> {
  const docId = currentId.value
  if (!docId) return
  busy.value = kind
  jobNote.value = ''
  try {
    const job = kind === 'parse' ? await parseDocument(docId) : await indexDocument(docId)
    jobNote.value = `${kind === 'parse' ? '解析' : '拆分并入库'}已排队（${job.id}）…`
    // 立刻把节点切到「进行中」，不等下一次轮询（那要五秒）
    jobs.value = [...jobs.value, job]
    lastRunningJob = job.id
    // 起表：这一条任务要跟到落地（空闲时表是停着的）
    ensurePolling()
  } catch (exc) {
    error.value = (exc as Error).message
  } finally {
    busy.value = ''
  }
}

async function commit(): Promise<void> {
  const docId = currentId.value
  if (!docId) return
  busy.value = 'save'
  try {
    const body = await saveOutline(docId, {
      rows: draft.value.map((row) => ({ level: row.level })),
      max_level: maxLevel.value,
      confirmed: confirmed.value,
    })
    detail.value = body
    draft.value = (body.outline?.rows ?? []).map((row) => ({ ...row }))
    maxLevel.value = body.outline?.max_level ?? 3
    confirmed.value = body.outline?.confirmed ?? false
    error.value = ''
    // 勾了「已核对」时后端会**顺手把切分 + 入库排上**（`api.index_after_review`）——
    // 拿到那个任务就切到「拆分入库中…」并开始轮询，四步流程里不用再点一次
    const auto = body.index_job
    if (auto) {
      jobs.value = [...jobs.value.filter((job) => job.id !== auto.id), auto]
      lastRunningJob = auto.id
      jobNote.value = `拆分并入库已开始（${auto.id}）…`
      ensurePolling()
      notice.value = '清单已保存，已开始拆分并入库'
    } else {
      notice.value = confirmed.value
        ? '清单已保存，可以切分入库了'
        : '清单已保存（还没标「已核对」，暂时切不了）'
    }
    await reload(true)
  } catch (exc) {
    error.value = (exc as Error).message
  } finally {
    busy.value = ''
  }
}

function sizeText(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

/** 两步之间的耗时。**都是产物里的时间戳**（落盘 / 解析 / 切分 / 入库各记了一个）——
 *  「这一步花了多久」是排查时最先想看的东西之一，而它一直躺在产物里没人用。 */
function elapsed(from: number, to: number): string {
  if (!from || !to || to < from) return '—'
  const seconds = to - from
  if (seconds < 90) return `${seconds} 秒`
  if (seconds < 5400) return `${Math.round(seconds / 60)} 分`
  return `${(seconds / 3600).toFixed(1)} 小时`
}

onMounted(() => {
  void reload()
  // **唯一一次"无条件"查询**：用来发现「刷新页面之前就在跑的任务」。之后按需起表。
  void pollJobs()
  // 切回这个标签页时立刻刷一遍：后台标签页的定时器会被浏览器降频，回来时先对齐一次
  document.addEventListener('visibilitychange', onVisible)
  window.addEventListener('keydown', onReviewKey)
})
onUnmounted(() => {
  alive = false
  stopPolling()
  document.removeEventListener('visibilitychange', onVisible)
  window.removeEventListener('keydown', onReviewKey)
  document.body.style.overflow = ''
})

function onVisible(): void {
  if (!alive) return
  if (document.visibilityState === 'visible') {
    // 回到前台先补一次；有任务会自己把表起回来，没有就保持停着
    void pollJobs()
  } else {
    // 切后台就停：轮了也不准（定时器被降频），白打后端
    stopPolling()
  }
}

/** 全屏对比时按 Esc 退出；同时锁住背景滚动（不然滚轮会带着后面的页面一起动）。 */
watch(reviewFull, (on) => {
  document.body.style.overflow = on ? 'hidden' : ''
})

function onReviewKey(event: KeyboardEvent): void {
  if (event.key === 'Escape' && reviewFull.value) reviewFull.value = false
}
</script>

<template>
  <header class="main-head">
    <strong>文档</strong>
    <span class="chip">{{ docs.length }} 份</span>
    <label class="btn primary">
      <Upload :size="14" aria-hidden="true" />上传
      <input type="file" multiple class="hidden-file" @change="onPick" />
    </label>
    <!-- 上传 / 导入的落点目录：选「未分类」就不用调那个接口 -->
    <label class="input folder-pick" title="上传 / 导入的文档放进哪个目录">
      <FolderIcon :size="14" aria-hidden="true" />
      <select v-model="targetFolder" aria-label="上传到哪个目录">
        <option value="">未分类</option>
        <option v-for="item in moveTargets.slice(1)" :key="item.id" :value="item.id">
          {{ '　'.repeat(item.depth) }}{{ item.name }}
        </option>
      </select>
    </label>
    <div class="input link-input">
      <Link2 :size="14" aria-hidden="true" />
      <input
        v-model="link"
        aria-label="从链接导入"
        placeholder="粘贴文档链接：HTML 页面 / PDF / Word 下载链接都行"
        @keyup.enter="importLink"
      />
    </div>
    <button class="btn" :disabled="fetching || !link.trim()" @click="importLink">
      {{ fetching ? '抓取中…' : '从链接导入' }}
    </button>
    <button class="btn ghost" :disabled="loading" @click="reload()">
      <RefreshCw :size="14" aria-hidden="true" />刷新
    </button>
  </header>

  <!-- 点空白处收起「移动到…」的浮层（与日志页表头筛选同一个行为：选完/点别处都收） -->
  <div class="main-body" @click="moveOpen = false">
    <p v-if="error" class="banner err">{{ error }}</p>
    <p v-if="notice" class="banner warn">{{ notice }}</p>

    <div v-if="conflicts.length" class="banner warn conflicts">
      <span>
        <b>同名，但内容不一样</b>{{ conflicts.map((item) => displayName(item)).join('、') }}：
        新的一份已经进来了。「覆盖更新」会连旧版的解析产物与已入库内容一起清掉（不可逆），
        保留两份则两版并存、各自走一遍五步。
      </span>
      <button class="btn danger" :disabled="Boolean(busy)" @click="replaceOld">覆盖更新</button>
      <button class="btn ghost" @click="conflicts = []">保留两份</button>
    </div>

    <div class="cols">
      <div class="panel doc-list">
        <header>
          文档（{{ docs.length }}）
          <span class="hint" title="按导入时间倒序；同一秒导入的按文件名">· 新的在前</span>
          <span v-if="!indexedKnown" class="hint" title="/health 里看 Milvus">· Milvus 够不着</span>
          <button class="btn ghost small" title="新建一个顶层目录" @click="startCreate(null)">
            <FolderPlus :size="13" aria-hidden="true" />新建目录
          </button>
        </header>

        <div v-if="picked.size" class="pick-bar">
          <span>已选 {{ picked.size }} 份</span>
          <div class="move-wrap" @click.stop>
            <button class="btn small" @click="moveOpen = !moveOpen">
              <FolderInput :size="13" aria-hidden="true" />移动到…
            </button>
            <ul v-if="moveOpen" class="move-menu">
              <li v-for="item in moveTargets" :key="item.id || 'root'">
                <button :style="indent(item.depth)" @click="movePicked(item.id)">{{ item.name }}</button>
              </li>
            </ul>
          </div>
          <button class="btn ghost small" @click="picked = new Set()">
            <X :size="13" aria-hidden="true" />取消
          </button>
        </div>

        <div v-if="!docs.length" class="empty">
          <b>还没有文档</b>
          先「上传」一份：PDF / Word / Markdown / 表格都行。
        </div>
        <ul v-else class="doc-rows tree">
          <li v-for="(row, index) in treeRows" :key="rowKey(row, index)" class="tree-item">
            <!-- 正在输入名字的那一格：新建与改名共用 -->
            <div v-if="row.kind === 'input'" class="folder-row" :style="indent(row.depth)">
              <input
                v-model="draftName"
                class="folder-input"
                :aria-label="naming?.edit ? '目录的新名字' : '新目录的名字'"
                placeholder="目录名"
                @keyup.enter="commitNaming"
                @keyup.esc="cancelNaming"
                @blur="commitNaming"
              />
            </div>

            <!-- 「未分类」那一格同时是**拖回未分类**的落点 -->
            <div
              v-else-if="row.kind === 'label'"
              class="tree-label"
              :class="{ 'drop-on': dragOver === 'root' }"
              @dragover.prevent="dragOver = 'root'"
              @drop.prevent="dropOn('')"
            >
              {{ row.text }}
            </div>

            <!-- 目录：点行首的箭头收放，右边是新建子目录 / 改名 / 删除；文档可以直接拖进来 -->
            <div
              v-else-if="row.kind === 'folder'"
              class="folder-row"
              :class="{ 'drop-on': dragOver === row.folder.id }"
              :style="indent(row.depth)"
              @dragover.prevent="dragOver = row.folder.id"
              @drop.prevent="dropOn(row.folder.id)"
            >
              <button
                class="twist"
                :aria-label="openFolders.has(row.folder.id) ? '收起' : '展开'"
                @click="toggleFolder(row.folder.id)"
              >
                <component
                  :is="openFolders.has(row.folder.id) ? ChevronDown : ChevronRight"
                  :size="13"
                  aria-hidden="true"
                />
              </button>
              <FolderIcon :size="14" aria-hidden="true" class="folder-icon" />
              <span class="folder-name" :title="row.folder.name">{{ row.folder.name }}</span>
              <span class="hint">{{ row.count }}</span>
              <span class="folder-ops">
                <button
                  v-if="row.depth < 2"
                  title="在里面新建子目录"
                  @click="startCreate(row.folder.id)"
                >
                  <FolderPlus :size="12" aria-hidden="true" />
                </button>
                <button title="改名" @click="startRename(row.folder)">
                  <Pencil :size="12" aria-hidden="true" />
                </button>
                <button title="删除这个目录（里面的文档回到未分类）" @click="removeFolder(row.folder)">
                  <Trash2 :size="12" aria-hidden="true" />
                </button>
              </span>
            </div>

            <!-- 文档：勾选框是给「移动到…」用的，点别处照旧打开它 -->
            <div
              v-else
              class="doc-line"
              :class="{
                on: row.doc.doc_id === currentId,
                dragging: dragging.includes(row.doc.doc_id),
              }"
              :style="indent(row.depth)"
              draggable="true"
              @dragstart="startDrag($event, row.doc.doc_id)"
              @dragend="endDrag"
            >
              <input
                type="checkbox"
                class="pick"
                :checked="picked.has(row.doc.doc_id)"
                :aria-label="`选中 ${displayName(row.doc)}`"
                @click.stop
                @change="togglePick(row.doc.doc_id)"
              />
              <span class="doc-main" @click="open(row.doc.doc_id)">
                <span class="doc-name" :title="displayName(row.doc)">
                  {{ displayName(row.doc) }}
                </span>
                <span class="hint">
                  {{ sizeText(row.doc.size) }}
                  <template v-if="sourceHost(row.doc)"> · 链接 {{ sourceHost(row.doc) }}</template>
                </span>
                <span class="doc-stages">
                  <i
                    v-for="stage in stagesOf(row.doc)"
                    :key="stage.label"
                    :class="{ done: stage.done }"
                    :title="stage.label"
                    >{{ stage.label }}</i
                  >
                </span>
              </span>
            </div>
          </li>
        </ul>
      </div>

      <div class="grow">
        <div v-if="!current" class="empty">
          <b>左边选一份</b>
          选中之后这里会走完：上传 / 导入 → 解析 → 核对标题 → 拆分并入库。
          <!-- 还没选中文档时（比如刚点完上传）也要看得见"正在上传/抓取" -->
          <p v-if="inflow" class="banner warn"><span class="dots">{{ inflow }}</span></p>
        </div>

        <template v-else>
          <!-- 上：主流程。点已完成的节点可以回去看那一步的结果 -->
          <div class="panel">
            <header class="doc-head">
              <b>{{ displayName(current) }}</b>
              <span>{{ sizeText(current.size) }}</span>
              <span class="mono">{{ current.doc_id }}</span>
              <template v-if="current.source_url">
                · 来源
                <a :href="current.source_url" target="_blank" rel="noreferrer">原链接</a>
              </template>
              <button
                class="btn danger small right"
                :disabled="Boolean(busy)"
                @click="remove(current)"
              >
                <Trash2 :size="14" aria-hidden="true" />删除这份
              </button>
            </header>
            <FlowSteps
              :steps="[...STEPS]"
              :states="stepStates"
              :active="panelStep"
              @pick="(index) => (pinnedStep = index)"
            />
            <p v-if="jobNote" class="hint flow-note">
              <span class="dots">{{ jobNote }}</span>
            </p>
          </div>

          <!-- 下：当前那一步的展示与操作 -->
          <div class="panel">
            <header>{{ STEPS[panelStep].label }} · {{ STEPS[panelStep].hint }}</header>
            <p v-if="locked" class="hint">
              这一份<b>已入库</b>：解析 / 核对 / 入库的入口都关了（要改就先删掉这份或者覆盖更新）。
            </p>

            <!-- 第一步：上传 / 导入 —— 看原文 -->
            <div v-if="panelStep === 0" class="stage-body">
              <!-- 上传 / 抓取进行中：面板被切到这一步，这里转起来（左列顶栏那两个入口点完
                   最怕"看着没反应" —— PDF 抓取要几秒） -->
              <p v-if="inflow" class="banner warn"><span class="dots">{{ inflow }}</span></p>
              <div class="stat-row">
                <div class="stat"><b>{{ sizeText(current.size) }}</b><span>文件大小</span></div>
                <div class="stat"><b class="mono">{{ current.sha1 }}</b><span>内容哈希</span></div>
                <div class="stat">
                  <b>{{
                    current.created_at ? new Date(current.created_at * 1000).toLocaleString() : '—'
                  }}</b>
                  <span>落盘时间</span>
                </div>
                <div class="stat">
                  <b>{{ current.content_type || '上传件（没记类型）' }}</b><span>来源类型</span>
                </div>
              </div>
              <p v-if="current.source_url" class="hint">
                从链接抓来的，图片一起下在
                <code>data/uploads/&lt;同名&gt;_files/</code>：<a
                  :href="current.source_url"
                  target="_blank"
                  rel="noreferrer"
                  >{{ current.source_url }}</a
                >
              </p>
              <p class="hint">
                原文（<b>只读</b>）—— 要改内容请改源文件后重传：正文一动，引用里的页码/行号
                就全错位了。
              </p>
              <SourceView :doc-id="current.doc_id" :name="current.name" />
            </div>

            <!-- 第二步：解析 —— 进行中转圈，完成后给结果 -->
            <div v-else-if="panelStep === 1" class="stage-body">
              <p v-if="stepStates[1] === 'running'" class="banner warn">
                <span class="dots">解析中<i>.</i><i>.</i><i>.</i></span>
                PDF 走 OCR / 版面模型是分钟级 —— 先去干别的，回来点这个节点看结果。
              </p>
              <template v-else-if="current.stages.parsed">
                <div class="stat-row">
                  <div class="stat"><b>{{ detail?.parse?.chars ?? 0 }}</b><span>正文字数</span></div>
                  <div class="stat"><b>{{ detail?.parse?.images ?? 0 }}</b><span>图片</span></div>
                  <div class="stat"><b>{{ detail?.parse?.parser || '—' }}</b><span>解析器</span></div>
                  <div class="stat"><b>{{ detail?.parse?.mime || '—' }}</b><span>MIME</span></div>
                  <div class="stat">
                    <b>{{
                      detail?.parse?.at ? new Date(detail.parse.at * 1000).toLocaleString() : '—'
                    }}</b>
                    <span>解析于</span>
                  </div>
                  <div class="stat">
                    <b>{{ elapsed(current.created_at, detail?.parse?.at ?? 0) }}</b>
                    <span>耗时（从落盘算起）</span>
                  </div>
                </div>
                <p v-if="detail?.parse?.missing_images?.length" class="banner warn">
                  有 {{ detail.parse.missing_images.length }} 张图没拿到（外链 / 防盗链 / 不是栅格图）：
                  <span class="hint">{{ detail.parse.missing_images.join('；') }}</span>
                </p>
                <div class="panel">
                  <header>解析稿（只读）</header>
                  <div class="doc-preview md">
                    <template v-for="(segment, index) in segments" :key="index">
                      <h4 v-if="segment.kind === 'heading'" class="plain-heading">
                        {{ segment.text }}
                      </h4>
                      <div v-else v-html="mdOf(segment.text)"></div>
                    </template>
                  </div>
                </div>
              </template>
              <template v-else>
                <p v-if="detail?.parse?.stale" class="banner warn">
                  <b>产物已过期</b>：源文件的哈希对不上了（产物记的是
                  <code>{{ detail.parse.recorded_hash }}</code>，现在是
                  <code>{{ detail.parse.current_hash }}</code>）—— 重新解析一次。
                </p>
                <p v-else class="empty">
                  <b>还没解析</b>
                  点下面的「解析这份」。解析本身不花钱；PDF 走 OCR / VLM 才是花钱的那一步。
                </p>
              </template>
              <div class="doc-actions">
                <!-- 这一步做完了就把入口**置灰**：亮着的按钮会让人以为「还得再点一次」。
                     真要重跑用右边那个小号的 —— 它才是「故意再跑一遍」的入口。 -->
                <button
                  v-if="!current.stages.parsed"
                  class="btn primary"
                  :disabled="Boolean(busy) || stepStates[1] === 'running' || locked"
                  @click="run('parse')"
                >
                  <FileText :size="14" aria-hidden="true" />
                  解析这份
                </button>
                <template v-else>
                  <button class="btn" disabled>
                    <Check :size="14" aria-hidden="true" />已解析
                  </button>
                  <button
                    class="btn ghost small"
                    :disabled="Boolean(busy) || stepStates[1] === 'running' || locked"
                    title="源文件变了、或者换了 OCR 设置时才需要"
                    @click="run('parse')"
                  >
                    重新解析
                  </button>
                </template>
                <span class="hint">
                  源文件没变的话解析产物会直接复用（不重复花钱）
                </span>
              </div>
            </div>

            <!-- 第三步：核对 —— 左右对比（左边就地改层级） -->
            <div v-else-if="panelStep === 2" class="stage-body">
              <p v-if="!current.stages.parsed" class="empty">
                <b>先解析</b>
                没有解析稿就没有可核对的标题行。
              </p>

              <template v-else>
              <!-- 「全屏对比」把这一整块（工具栏 + 两栏）铺满窗口：**左边的层级编辑照样能用**，
                   提交按钮也在，不用为了保存退出来。`position: fixed` 覆盖，不改 DOM ——
                   下拉里已选的层级、未提交的改动都不会因为进出全屏而丢。 -->
              <div class="review" :class="{ full: reviewFull }">
              <div class="doc-actions">
                <label class="hint">
                  切到第几级为止
                  <select v-model.number="maxLevel" :disabled="locked">
                    <option v-for="level in LEVELS" :key="level" :value="level">{{ level }}</option>
                  </select>
                  <span class="hint">（比它深的行不单独成节，正文并进上一节）</span>
                  <span v-if="mergedCount" class="hint merge-note">
                    当前会合并 {{ mergedCount }} 行
                  </span>
                </label>
                <label class="hint">
                  <input v-model="confirmed" type="checkbox" :disabled="locked" />
                  已核对（不勾就不许切分）
                </label>
                <button
                  class="btn primary"
                  :disabled="Boolean(busy) || !dirty || locked"
                  @click="commit"
                >
                  <Check :size="14" aria-hidden="true" />
                  {{ busy === 'save' ? '保存中…' : dirty ? '提交核对结果' : '没有改动' }}
                </button>
                <button class="btn ghost small" @click="reviewFull = !reviewFull">
                  <component :is="reviewFull ? Minimize2 : Maximize2" :size="14" aria-hidden="true" />
                  {{ reviewFull ? '退出全屏（Esc）' : '全屏对比' }}
                </button>
                <span v-if="dirty" class="hint right">有未提交的改动</span>
              </div>
              <p v-for="note in saved?.notes ?? []" :key="note" class="hint">{{ note }}</p>

              <div class="editor-cols">
                <div class="edit-pane">
                  <div class="pane-head">
                    解析稿（<b>只读</b>）· 点标题行上的控件改层级；要改正文请改源文件后重传
                    <button
                      class="btn ghost small"
                      type="button"
                      :aria-pressed="bodyCollapsed"
                      :title="
                        bodyCollapsed
                          ? '把正文段也显示出来'
                          : '只留标题行，正文段收起来（长文档过标题快很多）'
                      "
                      @click="bodyCollapsed = !bodyCollapsed"
                    >
                      {{ bodyCollapsed ? '显示正文' : '只看标题' }}
                    </button>
                  </div>
                  <div class="pane-body md">
                    <template v-for="(segment, index) in segments" :key="index">
                      <!--
                        标题行：**按 markdown 的样式渲染**（h1..h6 的字号/粗细跟着「当前选的层级」
                        走，标成正文的按正文渲染），控件收在行尾 —— 左边看起来就该是那份文档，
                        而不是一张表单。层级下拉 + 标为正文，见 outline.py 的 0 = 算正文。
                      -->
                      <div
                        v-if="segment.kind === 'heading' && draft[segment.index]"
                        class="md-head"
                        :class="`lvl-${levelOf(segment.index)}`"
                      >
                        <span class="md-head-text">{{ segment.text }}</span>
                        <span class="md-head-tools">
                          <span
                            v-if="draft[segment.index].hint !== levelOf(segment.index)"
                            class="hint"
                            :title="`解析给的初值 ${draft[segment.index].hint} 级`"
                            >初值 {{ draft[segment.index].hint }}</span
                          >
                          <!-- 比「切到第几级」深：它会并进上一个小节（光看层级数字看不出来） -->
                          <span
                            v-if="levelOf(segment.index) > maxLevel && levelOf(segment.index) > 0"
                            class="tag-merge"
                            title="比「切到第几级」深：不单独成节，正文并进上一个小节"
                            >并入上一节</span
                          >
                          <select
                            v-model.number="draft[segment.index].level"
                            :disabled="locked"
                            :aria-label="`「${segment.text}」的层级`"
                          >
                            <option v-for="level in LEVELS" :key="level" :value="level">
                              {{ level }} 级
                            </option>
                            <option :value="0">正文</option>
                          </select>
                          <button
                            class="btn ghost small"
                            :disabled="locked"
                            @click="toggleBody(segment.index)"
                          >
                            {{ levelOf(segment.index) === 0 ? '当回标题' : '标为正文' }}
                          </button>
                        </span>
                      </div>
                      <div
                        v-else-if="segment.kind === 'body'"
                        v-show="!bodyCollapsed"
                        v-html="mdOf(segment.text)"
                      ></div>
                    </template>
                    <div v-if="!draft.length" class="empty">
                      <b>解析稿里一个标题行都没有</b>
                      切分会整篇当成一段（路径为空）。想让章节进路径，就在源文件里把标题写成
                      <code>#</code> 开头。
                    </div>
                  </div>
                </div>

                <!-- 右栏：**原文**（只读）。PDF 由浏览器内置查看器渲染 —— 它本来就是**上下滑
                     不分页**；其它格式整份看。**不做映射**（使用者 2026-10-08 定：不点左边
                     跳右边、不画高亮），所以这里就是一份原文，没有额外交互。 -->
                <div class="edit-pane">
                  <div class="pane-head">
                    原文（只读）
                    <a class="hint right" :href="sourceUrl(current.doc_id)" target="_blank" rel="noreferrer">
                      新窗口打开
                    </a>
                  </div>
                  <div class="pane-body source-pane">
                    <SourceView :doc-id="current.doc_id" :name="current.name" />
                  </div>
                </div>
              </div>
              </div>
              </template>
            </div>

            <!-- 第四步：拆分并入库 —— 进行中转圈，完成后给块数 / 分布 / 大纲 / 入库账 -->
            <div v-else class="stage-body">
              <p v-if="stepStates[3] === 'running'" class="banner warn">
                <span class="dots">拆分入库中<i>.</i><i>.</i><i>.</i></span>
                切块之后每一块都要算一次向量（这一条会写进 Milvus），几百块要等一会儿。
              </p>
              <template v-else-if="detail?.chunks">
                <p v-if="!current.stages.reviewed" class="banner warn">
                  清单还没标「已核对」—— 切分只认核对过的那一份，先去「核对标题」把那一步做完。
                </p>
                <ChunkReport
                  :chunks="detail.chunks"
                  :index="detail.index"
                  :since="detail.parse?.at ?? 0"
                />
              </template>
              <p v-else class="empty">
                <template v-if="flowDone[3]">
                  <b>已入库，但读不到切分产物</b>
                  <code>data/stages/{{ current.doc_id }}/chunks.json</code> 不在了 ——
                  点「刷新」再看；还在就重新入库一次。
                </template>
                <template v-else-if="!indexedKnown">
                  <b>入库状态未知</b>
                  Milvus 够不着，所以「没入库」这个结论下不了（<code>/health</code> 里看依赖）。
                  拆分结果如果已经有了，会显示在上面。
                </template>
                <template v-else>
                  <b>还没拆分</b>
                  拆分与入库是同一条任务：切块 → 算向量 → 写 Milvus → 记注册表。
                </template>
              </p>

              <div class="doc-actions">
                <!-- 同上：做完了就置灰，重跑走小号入口 -->
                <button
                  v-if="!flowDone[3]"
                  class="btn primary"
                  :disabled="
                    Boolean(busy) ||
                    stepStates[3] === 'running' ||
                    !current.stages.parsed ||
                    !saved?.confirmed ||
                    dirty
                  "
                  :title="
                    !saved?.confirmed
                      ? '清单还没标「已核对」'
                      : dirty
                        ? '清单有未提交的改动'
                        : '切块并写进 Milvus'
                  "
                  @click="run('index')"
                >
                  <Scissors :size="14" aria-hidden="true" />
                  拆分并入库
                </button>
                <template v-else>
                  <button class="btn" disabled>
                    <Check :size="14" aria-hidden="true" />已拆分并入库
                  </button>
                  <button
                    class="btn ghost small"
                    :disabled="
                      Boolean(busy) ||
                      stepStates[3] === 'running' ||
                      !saved?.confirmed ||
                      dirty ||
                      locked
                    "
                    title="改了层级清单、或者想重灌这一份时才需要"
                    @click="run('index')"
                  >
                    重新入库
                  </button>
                </template>
                <span class="hint">
                  向量那一步是<b>花时间</b>的那一步（几百块就是几百次 embedding 调用）
                </span>
              </div>
            </div>

            <p class="hint stage-foot">
              产物落在 <code>data/stages/{{ current.doc_id }}/</code>
              （<code>document.md</code> 解析稿 · <code>outline.tsv</code> 清单 ·
              <code>chunks.md</code> 切分结果）。上传件在 <code>data/uploads/</code>。
            </p>
          </div>
        </template>
      </div>
    </div>
  </div>
</template>
