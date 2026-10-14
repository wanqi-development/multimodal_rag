<script setup lang="ts">
// 右侧原文抽屉：引用点开就落在这里（原文 + 图片 + 几个下一步动作）。
import { computed } from 'vue'
import { Copy, ExternalLink, MessageSquarePlus, X } from 'lucide-vue-next'
import { assetBlob } from '../api'
import AssetImage from './AssetImage.vue'
import { closeDetail, detail, detailError, page } from '../state'

const location = computed(() => {
  const item = detail.value
  if (!item) return ''
  const parts = [item.source_uri, item.heading_path, item.page ? `p${item.page}` : '', item.line ? `L${item.line}` : '']
  return parts.filter(Boolean).join(' · ')
})

async function copy(): Promise<void> {
  if (!detail.value) return
  await navigator.clipboard.writeText(detail.value.text)
}

function askAboutIt(): void {
  const item = detail.value
  if (!item) return
  page.value = 'chat'
  window.dispatchEvent(new CustomEvent('rag:ask-about-chunk', { detail: item }))
}

/** 原图开新标签页。走 blob: 才带得上 key —— 直接开 `/assets/…` 会 401。 */
async function openOriginal(path: string): Promise<void> {
  try {
    window.open(await assetBlob(path), '_blank', 'noopener')
  } catch (exc) {
    detailError.value = (exc as Error).message
  }
}
</script>

<template>
  <aside v-if="detail || detailError" class="detail" aria-label="原文">
    <header>
      <div>
        <strong>{{ detail?.heading_path || detail?.chunk_id || '读不到这一块' }}</strong>
        <p class="meta">{{ location || detailError }}</p>
      </div>
      <button class="icon-btn" aria-label="关闭" @click="closeDetail()">
        <X :size="16" aria-hidden="true" />
      </button>
    </header>

    <template v-if="detail">
      <div v-for="path in detail.assets" :key="path" class="thumb">
        <AssetImage :path="path" />
        <button class="link" @click="openOriginal(path)">{{ path }}</button>
      </div>
      <pre class="body-text">{{ detail.text }}</pre>
      <div class="actions">
        <button class="btn" @click="copy()"><Copy :size="16" aria-hidden="true" />复制</button>
        <button v-if="detail.assets.length" class="btn" @click="openOriginal(detail.assets[0])">
          <ExternalLink :size="16" aria-hidden="true" />打开原图
        </button>
        <button class="btn" @click="askAboutIt()">
          <MessageSquarePlus :size="16" aria-hidden="true" />就此追问
        </button>
      </div>
      <p class="hint">chunk_id {{ detail.chunk_id }} · 模态 {{ detail.modality }}</p>
    </template>
  </aside>
</template>
