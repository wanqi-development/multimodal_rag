<script setup lang="ts">
// 应用外壳（"共用一套"）：左侧导航 + 主区 + 右侧原文抽屉。
// 各页面只关心自己的主区内容，导航与抽屉由这里统一提供。
import { onMounted, onUnmounted } from 'vue'
import ChatView from './views/ChatView.vue'
import DocumentsView from './views/DocumentsView.vue'
import LogsView from './views/LogsView.vue'
import SearchView from './views/SearchView.vue'
import NavRail from './components/NavRail.vue'
import DetailPane from './components/DetailPane.vue'
import { closeDetail, detail, page, ping, setTheme, theme } from './state'

function onKey(event: KeyboardEvent): void {
  const target = event.target as HTMLElement
  const typing = target.tagName === 'INPUT' || target.tagName === 'TEXTAREA'
  if (event.key === 'Escape') {
    closeDetail()
    return
  }
  if (!typing && event.key === '/') {
    event.preventDefault()
    window.dispatchEvent(new CustomEvent('rag:focus-search'))
  }
}

onMounted(() => {
  setTheme(theme.value)
  void ping()
  window.addEventListener('keydown', onKey)
})
onUnmounted(() => window.removeEventListener('keydown', onKey))
</script>

<template>
  <div class="app">
    <NavRail />
    <main class="main">
      <SearchView v-if="page === 'search'" />
      <ChatView v-else-if="page === 'chat'" />
      <DocumentsView v-else-if="page === 'documents'" />
      <LogsView v-else-if="page === 'logs'" />
    </main>
    <DetailPane v-if="detail" />
  </div>
</template>
