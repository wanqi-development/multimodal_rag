<script setup lang="ts">
// HITL 批准卡片：它要动磁盘 / 执行命令，**没有人的一句「可以」，一个字都不会写**。
//
// 按钮按后端给的 `review_configs[*].allowed_decisions` 渲染 —— 硬编码一个 approve 可能给出
// 中间件根本不接受的选项（那不是「点了没反应」，是用错协议）。
// 第一版只有「批准全部 / 拒绝全部」（与 CLI 的 `--approve` / `--reject` 一致）；
// 逐条 edit / respond 要内嵌编辑器，等真需要再说。
import { computed } from 'vue'
import { ShieldAlert } from 'lucide-vue-next'
import type { Interrupt } from '../api'

const props = defineProps<{ interrupt: Interrupt; busy?: boolean }>()
const emit = defineEmits<{ decide: [kind: string] }>()

/** 每个动作都允许这个决定，才能一键全批。后端没给清单就按最保守的来。 */
function everyAllows(kind: string): boolean {
  const configs = props.interrupt.review_configs
  if (!configs?.length) return kind === 'approve' || kind === 'reject'
  return configs.every((config) => config.allowed_decisions?.includes(kind))
}

const canApprove = computed(() => everyAllows('approve'))
const canReject = computed(() => everyAllows('reject'))

function argsText(args: Record<string, unknown> | undefined): string {
  return JSON.stringify(args ?? {})
}
</script>

<template>
  <div class="approval">
    <p class="approval-head">
      <ShieldAlert :size="16" aria-hidden="true" />
      它要动磁盘 / 执行命令，等你批准（{{ interrupt.action_requests.length }} 条）
    </p>
    <ol>
      <li v-for="(request, index) in interrupt.action_requests" :key="index">
        <code>{{ request.name }}</code>
        <span class="args" :title="argsText(request.args)">{{ argsText(request.args) }}</span>
      </li>
    </ol>
    <div class="actions">
      <button v-if="canApprove" class="btn primary" :disabled="busy" @click="emit('decide', 'approve')">
        批准，继续跑
      </button>
      <button v-if="canReject" class="btn danger" :disabled="busy" @click="emit('decide', 'reject')">
        拒绝
      </button>
      <span class="hint">批了就落盘；拒了就什么都不写，它会换个说法回你。</span>
    </div>
  </div>
</template>
