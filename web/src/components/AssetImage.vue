<script setup lang="ts">
// 一张语料图片。**不能直接写 `<img src="/assets/…">`** —— 那个端点在鉴权后面，
// `<img>` 又发不出 `X-API-Key`，配了 key 的部署下会 401（实测）。这里统一走 assetBlob()。
import { ref, watchEffect } from 'vue'
import { assetBlob } from '../api'

const props = defineProps<{ path: string; alt?: string }>()

const src = ref('')
const failed = ref(false)

watchEffect(() => {
  const path = props.path
  src.value = ''
  failed.value = false
  assetBlob(path)
    .then((url) => {
      // 换图/换 key 之后旧结果可能才回来，丢掉这种过期的
      if (props.path === path) src.value = url
    })
    .catch(() => {
      if (props.path === path) failed.value = true
    })
})
</script>

<template>
  <img v-if="src" :src="src" :alt="alt ?? path" />
  <span v-else-if="failed" class="asset-note">图片取不到</span>
  <span v-else class="asset-note">读取中…</span>
</template>
