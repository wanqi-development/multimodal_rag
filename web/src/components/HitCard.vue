<script setup lang="ts">
// 一条命中。分数用"分数条"表达（纯数字看不出相对强弱），来源用「文档 · 章节 › 页」面包屑。
import type { Hit } from '../api'
import AssetImage from './AssetImage.vue'

defineProps<{ hit: Hit; selected?: boolean }>()
</script>

<template>
  <article
    class="hit"
    :class="{ selected }"
    tabindex="0"
    :aria-label="`命中 ${hit.chunk_id}`"
  >
    <div class="score">
      <b>{{ hit.score.toFixed(4) }}</b>
      <div class="bar"><i :style="{ width: `${Math.max(2, hit.score * 100)}%` }"></i></div>
    </div>
    <div>
      <div class="path">
        <b>{{ hit.source_uri }}</b>
        <template v-if="hit.heading_path"> · {{ hit.heading_path }}</template>
      </div>
      <p class="snippet">{{ hit.text.slice(0, 180) }}</p>
      <div v-if="hit.assets?.length" style="display: flex; gap: 6px; margin-top: 8px">
        <AssetImage
          v-for="path in hit.assets.slice(0, 3)"
          :key="path"
          class="thumb-sm"
          :path="path"
          style="height: 56px; border: 1px solid var(--border); border-radius: var(--r-sm)"
        />
      </div>
    </div>
    <div class="meta-col">
      <span v-if="hit.page" class="tag">p{{ hit.page }}</span>
      <span v-if="hit.line" class="tag">L{{ hit.line }}</span>
      <span class="tag" :class="{ image: hit.modality !== 'text' }">{{ hit.modality }}</span>
    </div>
  </article>
</template>
