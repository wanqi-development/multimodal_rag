<script setup lang="ts">
// 原文展示：按类型交给最合适的那一种方式，**只读**。
//
// * HTML → `<iframe sandbox="">`：抓来的是第三方页面，不带 `allow-scripts` 就把脚本与表单
//   全关掉；后端那一侧还带了 `CSP: sandbox` 与 `nosniff`。
// * DOCX → `docx-preview` 以阅读视图渲染：正文按容器宽度连续重排，不做 A4 纸张整体缩放；
//   复杂分页/页眉页脚不承诺与桌面 Word 像素一致。
// * PDF → 浏览器内置查看器（整份、可翻页可搜索）。要看**高亮**就得上按页渲染那条路，
//   见 DocumentsView 里核对步骤的页面视图 —— 插件里叠不了我们的框。
// * 图片 → `<img>`；纯文本 → 取回来放 `<pre>`（截断，大文件不整份进 DOM）。
import { computed, nextTick, ref, watch } from 'vue'
import { sourceKindOf, sourceUrl } from '../api'

const props = defineProps<{ docId: string; name: string; maxChars?: number }>()

const kind = computed(() => sourceKindOf(props.name))
const text = ref('')
const error = ref('')
const loading = ref(false)
const docxRoot = ref<HTMLElement | null>(null)
let loadSeq = 0

// PDF 交给浏览器自带查看器。**保留它自己的工具条**（翻页 / 缩放 / 下载都在上面），
// 只把左侧的书签栏（navpanes）收起来、并按页宽铺满 —— 2026-10-10 使用者定。
// 参数是「PDF 打开参数」，Chrome / Edge 认；Firefox 忽略它们，不会因此打不开。
const pdfUrl = computed(() => `${sourceUrl(props.docId)}#zoom=page-width&navpanes=0`)

async function loadText(): Promise<void> {
  text.value = ''
  error.value = ''
  if (kind.value !== 'text') return
  try {
    const res = await fetch(sourceUrl(props.docId))
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const body = await res.text()
    const limit = props.maxChars ?? 20000
    text.value = body.length > limit ? `${body.slice(0, limit)}\n…（已截断）` : body
  } catch (exc) {
    error.value = (exc as Error).message
  }
}

async function loadDocx(): Promise<void> {
  const seq = ++loadSeq
  loading.value = false
  error.value = ''
  if (kind.value !== 'docx') return
  try {
    loading.value = true
    const res = await fetch(sourceUrl(props.docId))
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const blob = await res.blob()
    await nextTick()
    if (seq !== loadSeq || !docxRoot.value) return
    const { renderAsync } = await import('docx-preview')
    if (seq !== loadSeq || !docxRoot.value) return
    docxRoot.value.innerHTML = ''
    await renderAsync(blob, docxRoot.value, undefined, {
      className: 'docx',
      inWrapper: true,
      breakPages: false,
      ignoreWidth: true,
      ignoreHeight: true,
      renderHeaders: false,
      renderFooters: false,
      renderComments: true,
      renderChanges: true,
      useBase64URL: true,
    })
  } catch (exc) {
    if (seq === loadSeq) error.value = (exc as Error).message || 'DOCX 预览失败'
  } finally {
    if (seq === loadSeq) loading.value = false
  }
}

watch(
  () => [props.docId, props.name],
  async () => {
    loadSeq += 1
    await Promise.all([loadText(), loadDocx()])
  },
  { immediate: true },
)

</script>

<template>
  <div class="source-box">
    <iframe
      v-if="kind === 'html'"
      :src="sourceUrl(docId)"
      sandbox=""
      referrerpolicy="no-referrer"
      title="原文"
    ></iframe>
    <iframe v-else-if="kind === 'pdf'" :src="pdfUrl" title="原文"></iframe>
    <div v-else-if="kind === 'docx'" class="docx-source">
      <p v-if="loading" class="docx-state" aria-live="polite">正在渲染 Word 原文…</p>
      <div v-else-if="error" class="docx-state">
        <b>Word 原文预览失败</b>
        <span>{{ error }}</span>
        <a :href="sourceUrl(docId)" target="_blank" rel="noreferrer">下载原文件</a>
      </div>
      <div ref="docxRoot" class="docx-preview"></div>
    </div>
    <img v-else-if="kind === 'image'" :src="sourceUrl(docId)" alt="原文" />
    <pre v-else-if="kind === 'text'" class="pre">{{ text || error || '读取中…' }}</pre>
    <p v-else class="empty">
      <b>这个格式浏览器里看不了</b>
      解析结果在「解析」那一步看。
      <a :href="sourceUrl(docId)" target="_blank" rel="noreferrer">新窗口打开</a>
    </p>
  </div>
</template>
