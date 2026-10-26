<script setup lang="ts">
// 左侧导航：只有这一份（"共用一套" 的骨架），各页面自己不再画导航。
import {
  Box,
  Files,
  MessagesSquare,
  ScrollText,
  Search,
  Sidebar,
  SidebarClose,
  type LucideIcon,
} from 'lucide-vue-next'
import {
  page,
  railCollapsed,
  setTheme,
  status,
  theme,
  toggleRail,
  type PageId,
  type Theme,
} from '../state'

const ICONS: Record<string, LucideIcon> = {
  search: Search,
  'messages-square': MessagesSquare,
  files: Files,
  'scroll-text': ScrollText,
}

// 只有做完的页面才挂在这儿。**评测** 2026-10-08 定下来不做管理台页面（要跑评测走
// `rag eval`，见 docs/architecture.md §13）—— 所以这里没有它，别再往回加占位项。
// **设置页**（模型选型）2026-10-10 暂时下线，条目撤掉；页面与接口留着，要做时再加回来
// （见 docs/backlog.md 第 2 条）。
const items: { id: PageId; label: string; icon: string; soon?: string }[] = [
  { id: 'search', label: '检索', icon: 'search' },
  { id: 'chat', label: '对话', icon: 'messages-square' },
  { id: 'documents', label: '文档', icon: 'files' },
  { id: 'logs', label: '日志', icon: 'scroll-text' },
]

const themes: { id: Theme; label: string }[] = [
  { id: 'auto', label: '自动' },
  { id: 'light', label: '浅' },
  { id: 'dark', label: '深' },
]

function pill(ok: boolean | undefined, label: string): { cls: string; text: string } {
  if (ok === undefined) return { cls: 'pill', text: `${label} ?` }
  return ok ? { cls: 'pill ok', text: label } : { cls: 'pill bad', text: label }
}
</script>

<template>
  <aside class="rail" :class="{ collapsed: railCollapsed }">
    <!-- 收起/展开放在**最上面**（使用者 2026-10-08：原来在底部，位置不对） -->
    <div class="brand">
      <Box :size="16" aria-hidden="true" />
      <span>multimodal_rag</span>
      <button
        class="icon-btn rail-toggle"
        :title="railCollapsed ? '展开导航栏' : '收起导航栏（腾地方）'"
        :aria-label="railCollapsed ? '展开导航栏' : '收起导航栏'"
        @click="toggleRail"
      >
        <!-- 图标表示「点下去会变成什么」：展开态显示 sidebar-close、收起态显示 sidebar -->
        <component :is="railCollapsed ? Sidebar : SidebarClose" :size="15" aria-hidden="true" />
      </button>
    </div>

    <nav class="nav" aria-label="主导航">
      <button
        v-for="item in items"
        :key="item.id"
        :class="{ active: page === item.id }"
        :aria-label="item.label"
        :title="item.label"
        @click="page = item.id"
      >
        <component :is="ICONS[item.icon]" :size="16" aria-hidden="true" />
        <span class="label">{{ item.label }}</span>
        <span v-if="item.soon" class="soon">{{ item.soon }}</span>
      </button>
    </nav>

    <div class="rail-foot">
      <div class="pills" aria-label="依赖状态">
        <span :class="pill(status.milvus, 'Milvus').cls">{{ pill(status.milvus, 'Milvus').text }}</span>
        <span :class="pill(status.postgres, 'PG').cls">{{ pill(status.postgres, 'PG').text }}</span>
        <span :class="pill(status.redis, 'Redis').cls">{{ pill(status.redis, 'Redis').text }}</span>
      </div>
      <div class="theme" role="group" aria-label="主题">
        <button
          v-for="item in themes"
          :key="item.id"
          :class="{ on: theme === item.id }"
          @click="setTheme(item.id)"
        >
          {{ item.label }}
        </button>
      </div>
    </div>
  </aside>
</template>
