<script setup lang="ts">
// 「拆分并入库」这一步的结果。
//
// 数字之外有两样专门给人一眼看的东西：
//   * **块大小分布条** —— 每块一条宽度正比于字数的色条接成一长条。平均/最大这种数字
//     不如一张图直观：一眼就能看出有没有「一小节切成几十个碎块」或者「一大块几千字」。
//   * **大纲树** —— 按 heading_path 聚合的「哪几节、各多少块多少字」。393 条块列出来
//     没人看得完，节级才是能核对的东西。
import { computed, ref } from 'vue'
import { AlertTriangle, Check, ChevronDown, ChevronRight } from 'lucide-vue-next'
import type { ChunkSummary, IndexSummary } from '../api'

const props = defineProps<{
  chunks: ChunkSummary
  index: IndexSummary | null
  /** 上一步（解析）的时间戳：好算「切分这一步花了多久」 */
  since?: number
}>()

/** 分布条里单块最多占多少宽度：太长的块按上限截，否则一条能顶掉整行。 */
const widest = computed(() => Math.max(props.chunks.max_chars, 1))

/**
 * 大纲摊平成「带层级的行」。
 *
 * 每一节的 `heading_path` 是**从根到它的完整路径**（`章 -> 节 -> 小节`，由后端按人工核对的
 * 清单拼出来），所以直接按 ` -> ` 拆就能还原成一棵树：同一串前缀归到同一个父节点。
 * 摊平而不是递归组件，是因为要按文档顺序渲染，摊平之后顺序天然就对了。
 */
/** 树里的一个节点（渲染前）。 */
interface OutlineNode {
  key: string
  label: string
  /** 作者写的编号（`2` / `2.1`）；容器层没有 */
  heading_numbers: string
  blocks: number
  chars: number
  head: string
  /** 这一层自己有没有实打实的节（没有就是被更深的路径「带出来」的容器） */
  hasSelf: boolean
  /** 子项（含自己）合计：折叠状态下靠它看出整章多大 */
  subBlocks: number
  subChars: number
  children: OutlineNode[]
}

/** 摊平成一行之后（多出深度与展开状态）。 */
interface OutlineRow extends Omit<OutlineNode, 'children'> {
  depth: number
  hasChildren: boolean
  open: boolean
}

/** 展开着哪些节点。**默认空 = 全折叠**，只露最外层 —— 一键展开 668 节那种大树没法看。 */
const openNodes = ref<Set<string>>(new Set())

function toggleNode(key: string): void {
  const next = new Set(openNodes.value)
  if (next.has(key)) next.delete(key)
  else next.add(key)
  openNodes.value = next
}

const tree = computed<OutlineNode[]>(() => {
  const nodes = new Map<string, OutlineNode>()
  const roots: OutlineNode[] = []
  for (const [index, section] of props.chunks.sections.entries()) {
    const segments = section.heading_path ? section.heading_path.split(' -> ') : []
    if (!segments.length) {
      // 路径为空 = 第一个标题之前的正文（页面导航的残留常在这里），单独一类
      roots.push({
        key: `root-${index}`,
        label: '（第一个标题之前的正文）',
        heading_numbers: '',
        blocks: section.blocks,
        chars: section.chars,
        head: section.head,
        hasSelf: true,
        subBlocks: 0,
        subChars: 0,
        children: [],
      })
      continue
    }
    let parent: OutlineNode | null = null
    let prefix = ''
    for (const segment of segments) {
      prefix = prefix ? `${prefix} -> ${segment}` : segment
      let node = nodes.get(prefix)
      if (!node) {
        node = {
          key: prefix,
          label: segment,
          heading_numbers: '',
          blocks: 0,
          chars: 0,
          head: '',
          hasSelf: false,
          subBlocks: 0,
          subChars: 0,
          children: [],
        }
        nodes.set(prefix, node)
        if (parent) parent.children.push(node)
        else roots.push(node)
      }
      parent = node
    }
    if (parent) {
      parent.heading_numbers = section.heading_numbers
      parent.blocks = section.blocks
      parent.chars = section.chars
      parent.head = section.head
      parent.hasSelf = true
    }
  }
  // 自底向上算「含子项」合计：折叠状态下，父节点那一行也要能看出整章多大
  const rollup = (node: OutlineNode): void => {
    let blocks = node.hasSelf ? node.blocks : 0
    let chars = node.hasSelf ? node.chars : 0
    for (const child of node.children) {
      rollup(child)
      blocks += child.subBlocks
      chars += child.subChars
    }
    node.subBlocks = blocks
    node.subChars = chars
  }
  for (const node of roots) rollup(node)
  return roots
})

/** 真正渲染出来的行：只走进**已展开**的节点。 */
/** 「含子项 N 块 / M 字」默认不显示：一行信息量不大，却让整棵树变吵（2026-10-10 使用者定）。 */
const showPeek = ref(false)

const outlineRows = computed<OutlineRow[]>(() => {
  const out: OutlineRow[] = []
  const walk = (list: OutlineNode[], depth: number): void => {
    for (const node of list) {
      const open = openNodes.value.has(node.key)
      out.push({ ...node, depth, hasChildren: node.children.length > 0, open })
      if (node.children.length && open) walk(node.children, depth + 1)
    }
  }
  walk(tree.value, 0)
  return out
})

function barWidth(chars: number): string {
  const ratio = Math.min(chars / widest.value, 1)
  return `${Math.max(ratio * 100, 0.6)}%`
}

function sizeText(bytes: number): string {
  if (!bytes) return '—'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

/** 两步之间的耗时（时间戳都在产物里，见 DocumentsView 的 elapsed 说明）。 */
function elapsed(from: number, to: number): string {
  if (!from || !to || to < from) return '—'
  const seconds = to - from
  if (seconds < 90) return `${seconds} 秒`
  if (seconds < 5400) return `${Math.round(seconds / 60)} 分`
  return `${(seconds / 3600).toFixed(1)} 小时`
}
</script>

<template>
  <div class="stage-body">
    <div class="stat-row">
      <div class="stat"><b>{{ chunks.blocks }}</b><span>块</span></div>
      <div class="stat"><b>{{ chunks.text_blocks }}</b><span>正文块</span></div>
      <div class="stat"><b>{{ chunks.image_blocks }}</b><span>图片块</span></div>
      <div class="stat"><b>{{ chunks.chars }}</b><span>字符合计</span></div>
      <div class="stat"><b>{{ chunks.avg_chars }}</b><span>平均块长</span></div>
      <div class="stat"><b>{{ chunks.max_chars }}</b><span>最大块长</span></div>
      <div class="stat">
        <b>{{ sizeText(chunks.artifact_bytes['chunks.json'] ?? 0) }}</b><span>chunks.json</span>
      </div>
      <div v-if="index?.vector_bytes" class="stat">
        <b>{{ sizeText(index.vector_bytes) }}</b>
        <span>向量估算（{{ index.vector_dim }} 维）</span>
      </div>
    </div>

    <p class="hint">
      切分于 {{ chunks.at ? new Date(chunks.at * 1000).toLocaleString() : '—' }}
      （用了 {{ elapsed(since ?? 0, chunks.at) }}）·
      产物的源文件哈希 {{ chunks.recorded_hash }}
      <template v-if="chunks.stale">· <b class="bad">产物已过期</b>（源文件或切法变了）</template>
    </p>

    <div class="panel">
      <header>块大小分布（每块一条，宽度 ∝ 字数）</header>
      <div class="bars">
        <span
          v-for="(size, index2) in chunks.sizes"
          :key="index2"
          class="bar-item"
          :style="{ width: barWidth(size) }"
          :title="`第 ${index2 + 1} 块：${size} 字`"
        ></span>
      </div>
    </div>

    <div class="panel">
      <header>
        章节（{{ chunks.sections.length }} 节）
        <span class="hint">按层级排成树，<b>默认折叠</b> —— <b>点标题</b>就能展开/收起</span>
        <button
          class="btn ghost small right"
          type="button"
          :aria-pressed="showPeek"
          @click="showPeek = !showPeek"
        >
          {{ showPeek ? '隐藏含子项' : '显示含子项' }}
        </button>
      </header>
      <table class="outline-table chapter-tree">
        <thead>
          <tr>
            <th class="tree-col"></th>
            <th>编号</th>
            <th>章节</th>
            <th>块</th>
            <th>字</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="section in outlineRows" :key="section.key">
            <td class="tree-col">
              <!-- 箭头只做**指示**：真正的开关是右边那一格的「点标题」 -->
              <span v-if="section.hasChildren" class="caret" aria-hidden="true">
                <component :is="section.open ? ChevronDown : ChevronRight" :size="14" />
              </span>
            </td>
            <td class="mono">{{ section.heading_numbers || '—' }}</td>
            <td class="tree-name">
              <!-- 每一级一条竖线：光靠 18px 的 padding，三层的缩进看着几乎一样 -->
              <span
                v-for="level in section.depth"
                :key="level"
                class="tree-guide"
                aria-hidden="true"
              ></span>
              <button
                v-if="section.hasChildren"
                type="button"
                class="tree-toggle"
                :aria-expanded="section.open"
                :title="section.open ? '收起这一节' : '展开这一节'"
                @click="toggleNode(section.key)"
              >
                <span class="chapter-name" :class="{ 'is-child': section.depth > 0 }">
                  {{ section.label }}
                </span>
              </button>
              <span v-else class="chapter-name" :class="{ 'is-child': section.depth > 0 }">
                {{ section.label }}
              </span>
              <!-- 折叠时父节点那一行看不出整章多大 —— 但默认不显示，点「显示含子项」才出 -->
              <div v-if="showPeek && section.hasChildren" class="hint section-peek">
                含子项 {{ section.subBlocks }} 块 / {{ section.subChars }} 字
              </div>
            </td>
            <td class="mono">{{ section.hasSelf ? section.blocks : '—' }}</td>
            <td class="mono">{{ section.hasSelf ? section.chars : '—' }}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="panel">
      <header>入库（Milvus 注册表）</header>
      <div class="stage-body">
        <p v-if="!index?.indexed" class="banner warn">
          注册表里没有这一份 —— 还没入库，或者 Milvus 够不着（这两种在这里分不出来）。
        </p>
        <p v-else-if="index.in_sync" class="banner ok">
          <Check :size="14" aria-hidden="true" />
          对得上：注册表记 {{ index.chunk_count }} 块，产物也是 {{ chunks.blocks }} 块
        </p>
        <p v-else class="banner err">
          <AlertTriangle :size="14" aria-hidden="true" />
          <b>对不上</b>：注册表记 {{ index.chunk_count }} 块，产物是 {{ chunks.blocks }} 块
          —— 上一次入库可能没跑完，重跑一次「拆分并入库」。
        </p>
        <p v-if="index?.indexed" class="hint">
          入库于 {{ index.ingested_at ? new Date(index.ingested_at * 1000).toLocaleString() : '—' }}
          （用了 {{ elapsed(chunks.at, index.ingested_at) }}）
          · 切法指纹 <code>{{ index.pipeline_version }}</code>
        </p>
      </div>
    </div>

    <p class="hint">
      块级明细（每块的正文与页码/行号）在
      <code>data/stages/&lt;doc_id&gt;/chunks.md</code> —— 393 块那种量级，列在页面上也没人
      看得完，所以这里只给到节一级。
    </p>
  </div>
</template>
