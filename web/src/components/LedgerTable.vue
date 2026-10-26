<script setup lang="ts">
// 通道账本：每路召回多少、留下多少、多久、有没有挂。"为什么这条没召回" 就看这张表。
import type { Stats } from '../api'

defineProps<{ stats: Stats[] }>()

const SLOW_MS = 5000
</script>

<template>
  <div class="panel">
    <header>通道账本</header>
    <table class="data">
      <thead>
        <tr>
          <th>通道</th>
          <th class="num">召回</th>
          <th class="num">入选</th>
          <th class="num">耗时</th>
          <th>备注</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in stats" :key="row.channel" :class="{ slow: (row.latency_ms ?? 0) > SLOW_MS }">
          <td>{{ row.channel }}</td>
          <td class="num">{{ row.candidates }}</td>
          <td class="num">{{ row.kept }}</td>
          <td class="num">{{ row.latency_ms === null ? '—' : row.latency_ms.toFixed(0) + 'ms' }}</td>
          <td>
            <span v-if="row.error" class="tag warn">{{ row.error }}</span>
            <span v-else-if="(row.latency_ms ?? 0) > SLOW_MS" class="tag warn">偏慢：重排是大头</span>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>
