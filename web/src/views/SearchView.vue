<script setup lang="ts">
// 检索页：查询 + 参数 → 命中卡片 / 骨架 / 空态 + 通道账本；点卡片在右侧看原文。
import { onMounted, onUnmounted, ref } from 'vue'
import { Image as ImageIcon, Search, TriangleAlert, X } from 'lucide-vue-next'
import { search, uploadSearchImage, type Hit, type Stats } from '../api'
import HitCard from '../components/HitCard.vue'
import LedgerTable from '../components/LedgerTable.vue'
import { openChunk } from '../state'

// **不预填、也不自动检索**：挂载就跑一次的话，一打开页面就是一条写死的示例查询 ——
// 既花了本地的重排时间，又让人以为「页面自己会查东西」。要查就自己按一下。
const query = ref('')
const topK = ref(8)
const rerank = true
const loading = ref(false)
const error = ref('')
const hits = ref<Hit[]>([])
const stats = ref<Stats[]>([])
const selected = ref('')
const searched = ref(false)
const box = ref<HTMLInputElement | null>(null)
const fileInput = ref<HTMLInputElement | null>(null)

// ---- 图片检索 ----
// 图**先传到服务端**（`/search/image` 回路径），再拿路径调 `/search`。分开两步是有意的：
// `/search` 要过线上 embedding（按量计费），而「传了图还没点搜」是常态 —— 分开之后
// 重搜、换词、调 top_k 都不用重传图上。
/** 服务端路径（搜的时候交给 `/search`），为空 = 纯文字检索 */
const imagePath = ref('')
/** 本地预览用的 blob URL（服务端的图不方便直接从浏览器取，也没必要为此开接口） */
const imageUrl = ref('')
const imageName = ref('')
const uploading = ref(false)
let uploadGeneration = 0

function resetImageState(): void {
  if (imageUrl.value) URL.revokeObjectURL(imageUrl.value)
  imageUrl.value = ''
  imagePath.value = ''
  imageName.value = ''
}

function clearImage(): void {
  uploadGeneration += 1
  uploading.value = false
  resetImageState()
}

async function onPickImage(event: Event): Promise<void> {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  input.value = '' // 同一张图再选一次也得触发 change
  if (!file) return
  await sendImage(file)
}

/** 粘贴或文件选择进来的图走同一条：截个图直接搜是最常用的一种。 */
function onPaste(event: ClipboardEvent): void {
  if (uploading.value || loading.value) return
  const file = Array.from(event.clipboardData?.files ?? [])[0]
  if (!file || !file.type.startsWith('image/')) return
  event.preventDefault()
  void sendImage(file)
}

async function sendImage(file: File): Promise<void> {
  const generation = ++uploadGeneration
  resetImageState()
  imageUrl.value = URL.createObjectURL(file)
  imageName.value = file.name || '粘贴的图片'
  uploading.value = true
  error.value = ''
  try {
    const body = await uploadSearchImage(file)
    if (generation !== uploadGeneration) return
    imagePath.value = body.path
    imageName.value = file.name || '粘贴的图片'
  } catch (exc) {
    if (generation !== uploadGeneration) return
    resetImageState()
    error.value = (exc as Error).message
  } finally {
    if (generation === uploadGeneration) uploading.value = false
  }
}

async function run(): Promise<void> {
  // **文字与图至少给一个**：只有图也能搜（后端那句校验是同一个口径）
  if (!query.value.trim() && !imagePath.value) return
  if (loading.value || uploading.value) return
  loading.value = true
  error.value = ''
  searched.value = true
  try {
    const body = await search({
      query: query.value,
      top_k: topK.value,
      rerank,
      ...(imagePath.value ? { image_path: imagePath.value } : {}),
    })
    hits.value = body.hits
    stats.value = body.stats
    selected.value = body.hits[0]?.chunk_id ?? ''
  } catch (exc) {
    error.value = (exc as Error).message
    hits.value = []
    stats.value = []
  } finally {
    loading.value = false
  }
}

function move(step: number): void {
  if (!hits.value.length) return
  const index = hits.value.findIndex((hit) => hit.chunk_id === selected.value)
  const next = Math.min(Math.max(index + step, 0), hits.value.length - 1)
  selected.value = hits.value[next].chunk_id
  document.getElementById(`hit-${selected.value}`)?.scrollIntoView({ block: 'nearest' })
}

function onKey(event: KeyboardEvent): void {
  const target = event.target as HTMLElement
  if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') return
  if (event.key === 'j') move(1)
  else if (event.key === 'k') move(-1)
  else if (event.key === 'o' && selected.value) void openChunk(selected.value)
}

function focusBox(): void {
  box.value?.focus()
  box.value?.select()
}

onMounted(() => {
  window.addEventListener('keydown', onKey)
  window.addEventListener('rag:focus-search', focusBox)
})
onUnmounted(() => {
  window.removeEventListener('keydown', onKey)
  window.removeEventListener('rag:focus-search', focusBox)
  clearImage()
})
</script>

<template>
  <!-- 顶部与结果区都接 paste：焦点在输入框或结果区时都能直接 Ctrl+V 附图。 -->
  <header class="main-head" @paste="onPaste">
    <form class="search-form" @submit.prevent="run()">
      <div class="input search-composer" :class="{ 'has-image': imageUrl }">
        <Search :size="16" aria-hidden="true" />
        <div v-if="imageUrl" class="search-attachment">
          <img :src="imageUrl" :alt="imageName" />
          <span class="search-attachment-name" aria-live="polite">
            {{ uploading ? '图片上传中…' : imageName }}
          </span>
          <button type="button" class="attachment-remove" title="移除图片" aria-label="移除图片" @click="clearImage()">
            <X :size="13" aria-hidden="true" />
          </button>
        </div>
        <input
          ref="box"
          v-model="query"
          aria-label="检索词或图片"
          :placeholder="imageUrl ? '可补充文字描述这张图' : '输入关键词，或按 Ctrl+V 粘贴图片'"
        />
        <button
          type="button"
          class="attachment-pick"
          title="添加图片（也可以直接按 Ctrl+V 粘贴）"
          aria-label="添加图片"
          :disabled="uploading || loading"
          @click="fileInput?.click()"
        >
          <ImageIcon :size="16" aria-hidden="true" />
        </button>
        <input
          ref="fileInput"
          type="file"
          accept="image/*"
          class="hidden-file"
          @change="onPickImage"
        />
        <select v-model.number="topK" class="topk-select" aria-label="返回结果数量">
          <option v-for="n in [3, 5, 8, 12, 20]" :key="n" :value="n">Top-k {{ n }}</option>
        </select>
      </div>
      <button
        class="btn primary search-submit"
        type="submit"
        :disabled="loading || uploading || (!query.trim() && !imagePath)"
      >
        {{ loading ? '检索中…' : '检索' }}
      </button>
    </form>
  </header>

  <div class="main-body" @paste="onPaste">
    <div class="chips">
      <span class="hint right">j / k 选择 · o 打开原文 · Esc 关闭</span>
    </div>

    <p v-if="error" class="banner err"><TriangleAlert :size="16" aria-hidden="true" />{{ error }}</p>

    <div v-if="loading" class="hits" aria-hidden="true">
      <article v-for="i in 3" :key="i" class="hit">
        <div class="skel" style="height: 34px"></div>
        <div><div class="skel" style="width: 55%"></div><div class="skel" style="width: 92%; margin-top: 8px"></div></div>
        <div class="skel" style="width: 30px"></div>
      </article>
    </div>

    <div v-else-if="hits.length" class="hits">
      <HitCard
        v-for="hit in hits"
        :id="`hit-${hit.chunk_id}`"
        :key="hit.chunk_id"
        :hit="hit"
        :selected="hit.chunk_id === selected"
        @click="selected = hit.chunk_id; openChunk(hit.chunk_id)"
      />
    </div>

    <div v-else-if="searched" class="empty">
      <b>没有命中</b>
      换个说法或放宽 Top-k；如果搜的是图片，可以补充文字描述后再试。
    </div>

    <div v-else class="empty">
      <b>输入检索词，或粘贴一张图片</b>
      直接返回相关原文片段。
      要成段结论、要接着上文追问，就切「对话」。
    </div>

    <LedgerTable v-if="stats.length" :stats="stats" />
  </div>
</template>
