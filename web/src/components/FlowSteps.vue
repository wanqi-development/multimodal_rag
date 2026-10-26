<script setup lang="ts">
// 主流程那条流程图（像审批流那样）。
//
// 四种状态，颜色就是全部信息：
//   done    已完成 —— 蓝色实心
//   running 进行中 —— **半蓝半灰的圆环，转起来**（一眼看出「卡在哪一步在跑」）
//   current 当前该做的 —— 蓝圈灰底（不转，和「在跑」区分开）
//   todo    还没到 —— 灰
//
// 节点可点：点已完成的节点能回去看那一步的结果（下面板跟着换），不改任何状态。
export type StepState = 'done' | 'running' | 'current' | 'todo'

defineProps<{
  steps: { label: string; hint?: string }[]
  states: StepState[]
  /** 下方面板正显示哪一步（默认是「当前该做的」那一步） */
  active: number
}>()

const emit = defineEmits<{ (event: 'pick', index: number): void }>()
</script>

<template>
  <ol class="flow">
    <li v-for="(step, index) in steps" :key="step.label" class="flow-item">
      <button
        type="button"
        class="flow-node"
        :class="[states[index], { active: index === active }]"
        :title="step.hint ?? step.label"
        @click="emit('pick', index)"
      >
        <span class="ring" :class="states[index]" aria-hidden="true"></span>
        <span class="flow-label">{{ step.label }}</span>
      </button>
      <span
        v-if="index < steps.length - 1"
        class="flow-line"
        :class="{ on: states[index] === 'done' }"
        aria-hidden="true"
      ></span>
    </li>
  </ol>
</template>
