<script setup lang="ts">
// 设置页（2026-10-09）：两个标签 —— **模型选型** 与 **存储**（使用者定：只留这两个）。
//
// 三处硬边界：① 页面**永远拿不到 key**（接口只回 `has_key` 与指纹），没有主密钥
// （部署配置 RAG_SECRET_KEY）就禁填；② 这一版**不热切**：保存后必须重启 `rag serve`
// 才生效（热切是 `docs/backlog.md` 第 2 条）；③ 槽位表回答"谁在用"，不叫"默认模型"。
import { computed, onMounted, ref } from 'vue'
import {
  modelSettings,
  providers,
  setModelSlot,
  setProvider,
  type ModelList,
  type ModelSlot,
  type ProviderList,
  type ProviderSetting,
} from '../api'

const TABS = [
  { id: 'models', label: '模型选型' },
  { id: 'storage', label: '存储' },
] as const
const tab = ref<'models' | 'storage'>('models')

const modelData = ref<ModelList | null>(null)
const providerData = ref<ProviderList | null>(null)
const selected = ref('')
const error = ref('')
const notice = ref('')
const savingSlot = ref('')
const savingProvider = ref(false)

interface SlotDraft {
  provider_key: string
  model: string
}
interface ProviderDraft {
  api_key: string
  base_url: string
}
const slotDraft = ref<Record<string, SlotDraft>>({})
const providerDraft = ref<Record<string, ProviderDraft>>({})

const currentProvider = computed<ProviderSetting | null>(
  () => providerData.value?.providers.find((p) => p.provider_key === selected.value) ?? null,
)
const currentDraft = computed<ProviderDraft>(
  () => providerDraft.value[selected.value] ?? { api_key: '', base_url: '' },
)
const providerChanged = computed(() => {
  const p = currentProvider.value
  const d = providerDraft.value[selected.value]
  if (!p || !d) return false
  return d.base_url !== p.base_url || d.api_key !== ''
})

function keyText(value: boolean | null): string {
  return value === null ? '—' : value ? '已配置' : '未配置'
}
function slotChanged(slot: ModelSlot): boolean {
  const item = slotDraft.value[slot.slot]
  return !!item && (item.provider_key !== slot.provider_key || item.model !== slot.model)
}
function keyPlaceholder(p: ProviderSetting): string {
  if (!p.can_set_key) return '部署配置里没有 RAG_SECRET_KEY'
  return p.api_key_fingerprint ? '已存一把，留空即不改' : 'sk-…'
}
function seedDrafts(list: ProviderSetting[]): void {
  providerDraft.value = Object.fromEntries(
    list.map((p) => [p.provider_key, { api_key: '', base_url: p.base_url }]),
  )
  if (!selected.value && list.length) selected.value = list[0].provider_key
}
async function load(): Promise<void> {
  error.value = ''
  try {
    const [models, provs] = await Promise.all([modelSettings(), providers()])
    modelData.value = models
    slotDraft.value = Object.fromEntries(
      models.slots.map((s) => [s.slot, { provider_key: s.provider_key, model: s.model }]),
    )
    providerData.value = provs
    seedDrafts(provs.providers)
  } catch (exc) {
    error.value = (exc as Error).message
  }
}
async function saveSlot(slot: string): Promise<void> {
  savingSlot.value = slot
  error.value = ''
  notice.value = ''
  try {
    modelData.value = await setModelSlot(slot, slotDraft.value[slot])
    notice.value = `${slot} 已写入（含审计）：重启 \`rag serve\` 后生效`
  } catch (exc) {
    error.value = (exc as Error).message
  } finally {
    savingSlot.value = ''
  }
}
async function saveProvider(): Promise<void> {
  const p = currentProvider.value
  const d = providerDraft.value[selected.value]
  if (!p || !d) return
  savingProvider.value = true
  error.value = ''
  notice.value = ''
  try {
    const body: { base_url: string; api_key?: string } = { base_url: d.base_url }
    if (d.api_key) body.api_key = d.api_key
    providerData.value = await setProvider(p.provider_key, body)
    seedDrafts(providerData.value.providers)
    notice.value = `${p.provider_key} 已保存：重启 \`rag serve\` 后生效`
  } catch (exc) {
    error.value = (exc as Error).message
  } finally {
    savingProvider.value = false
  }
}
async function clearKey(): Promise<void> {
  const p = currentProvider.value
  if (!p) return
  savingProvider.value = true
  error.value = ''
  notice.value = ''
  try {
    providerData.value = await setProvider(p.provider_key, { api_key: '' })
    seedDrafts(providerData.value.providers)
    notice.value = `${p.provider_key} 的库内 key 已清除（回到部署配置那份）`
  } catch (exc) {
    error.value = (exc as Error).message
  } finally {
    savingProvider.value = false
  }
}
onMounted(load)
</script>

<template>
  <header class="main-head">
    <strong>设置</strong>
    <div class="seg" role="group" aria-label="设置分类">
      <button v-for="t in TABS" :key="t.id" :class="{ on: tab === t.id }" @click="tab = t.id">
        {{ t.label }}
      </button>
    </div>
    <span v-if="modelData" class="hint">存储 {{ modelData.storage }}（backend {{ modelData.backend }}）</span>
  </header>

  <div class="main-body">
    <p v-if="error" class="banner err">{{ error }}</p>
    <p v-if="notice" class="banner ok">{{ notice }}</p>

    <template v-if="tab === 'models'">
      <p v-if="modelData" class="banner warn">{{ modelData.note }}</p>

      <section class="split">
        <aside class="panel col-side">
          <div class="panel-head">供应商</div>
          <ul class="prov-list">
            <li
              v-for="p in providerData?.providers ?? []"
              :key="p.provider_key"
              :class="{ on: p.provider_key === selected }"
              @click="selected = p.provider_key"
            >
              <span class="pname">{{ p.provider_key }}</span>
              <span class="tag" :class="{ ok: p.has_key }">
                {{ p.has_key ? '已配置' : '未配置' }}
              </span>
            </li>
          </ul>
        </aside>

        <div v-if="currentProvider" class="panel col-main">
          <div class="panel-head">
            <span>{{ currentProvider.provider_key }}</span>
            <span v-if="currentProvider.api_key_fingerprint" class="fp mono">
              指纹 {{ currentProvider.api_key_fingerprint }}
            </span>
          </div>

          <div class="field">
            <label>API-Key</label>
            <input
              v-model="currentDraft.api_key"
              type="password"
              autocomplete="new-password"
              :disabled="!currentProvider.can_set_key"
              :placeholder="keyPlaceholder(currentProvider)"
            />
            <p v-if="!currentProvider.can_set_key" class="hint">
              部署配置里没有 <code>RAG_SECRET_KEY</code>：不许在页面上填 key（仍可改 base_url）。
            </p>
            <p v-else class="hint">留空即不动；填新值 = 加密存库，页面永不回显。</p>
          </div>

          <div class="field">
            <label>base_url</label>
            <input v-model="currentDraft.base_url" placeholder="（沿用部署配置）" />
          </div>

          <div class="actions">
            <button
              class="btn small"
              :disabled="savingProvider || !providerChanged"
              @click="saveProvider"
            >
              {{ savingProvider ? '保存中…' : '保存' }}
            </button>
            <button
              class="btn small"
              :disabled="savingProvider || !currentProvider.api_key_fingerprint"
              @click="clearKey"
            >
              清除库内 key
            </button>
          </div>

          <div class="field">
            <label>模型清单</label>
            <table v-if="currentProvider.models.length" class="table mini">
              <thead>
                <tr><th>name</th><th>type</th></tr>
              </thead>
              <tbody>
                <tr v-for="(m, i) in currentProvider.models" :key="i">
                  <td class="mono">{{ m.name ?? '—' }}</td>
                  <td>{{ m.type ?? '—' }}</td>
                </tr>
              </tbody>
            </table>
            <p v-else class="hint">这家没登记模型清单。</p>
          </div>
        </div>
      </section>

      <h4 class="section-title">槽位：谁在用</h4>
      <div v-if="modelData" class="panel">
        <table class="table settings-table">
          <thead>
            <tr>
              <th>槽位</th><th>来源</th><th>谁在用</th><th>model</th>
              <th>key</th><th class="text-end">版本</th><th>操作</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="s in modelData.slots" :key="s.slot">
              <td class="mono">{{ s.slot }}</td>
              <td>{{ s.source }}</td>
              <td>
                <select
                  v-if="s.candidates.length"
                  v-model="slotDraft[s.slot].provider_key"
                  class="cell"
                >
                  <option value="">（沿用原处）</option>
                  <option v-for="c in s.candidates" :key="c" :value="c">{{ c }}</option>
                </select>
                <span v-else class="mono">{{ s.provider_key || '—' }}</span>
              </td>
              <td>
                <input
                  v-model="slotDraft[s.slot].model"
                  class="cell mono"
                  placeholder="（沿用原处）"
                />
              </td>
              <td :class="{ bad: s.key_ready === false }">{{ keyText(s.key_ready) }}</td>
              <td class="text-end tabular-nums">{{ s.version ?? '—' }}</td>
              <td>
                <button
                  class="btn small"
                  :disabled="!slotChanged(s) || savingSlot === s.slot"
                  @click="saveSlot(s.slot)"
                >
                  {{ savingSlot === s.slot ? '保存中…' : '保存' }}
                </button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <p v-if="!modelData && !error" class="hint">读设置中…</p>
      <p class="hint">
        优先级 <b>库 &gt; .env &gt; yaml</b>：来源列写"未设置"的槽位就沿用原处那份配置；
        留空保存即"回到沿用"。key 只看配没配 —— 页面与接口都不持有它的值。
      </p>
    </template>

    <template v-else>
      <div v-if="modelData" class="panel">
        <table class="table settings-table">
          <tbody>
            <tr>
              <th>当前后端</th><td class="mono">{{ modelData.backend }}</td>
              <td class="hint">auto = 配了 PG 就用 PG，否则单机 SQLite</td>
            </tr>
            <tr>
              <th>存储位置</th><td class="mono">{{ modelData.storage }}</td>
              <td class="hint">
                四张表：<code>model_setting</code>（选型）/
                <code>model_setting_change</code>（审计）/
                <code>provider_setting</code>（供应商 key 密文）/
                <code>app_secret</code>（主密钥兜底位）
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <p class="hint">
        换库改配置文件：<code>models.backend</code>（auto / sqlite / postgres）与
        <code>models.sqlite_path</code>，重启后生效；建表用 <code>rag models init</code>。
      </p>
    </template>
  </div>
</template>

<style scoped>
/* 两栏：左边供应商一列（窄），右边这家的详情 */
.split { display: grid; grid-template-columns: 220px 1fr; gap: 12px; align-items: start; }
.col-side { padding: 0; overflow: hidden; }
.col-side .panel-head, .col-main .panel-head {
  padding: 10px 12px; border-bottom: 1px solid var(--border); font-weight: 600;
  display: flex; align-items: center; justify-content: space-between; gap: 8px;
}
.col-main { padding: 0 12px 12px; }
.prov-list { list-style: none; margin: 0; padding: 4px; }
.prov-list li {
  display: flex; align-items: center; gap: 8px; padding: 8px 10px;
  border-radius: var(--r-sm); cursor: pointer;
}
.prov-list li:hover { background: rgba(0, 0, 0, 0.03); }
.prov-list li.on { background: var(--accent-weak); }
.prov-list .pname { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; }
.tag { font-size: 12px; padding: 1px 6px; border-radius: 999px; background: #eee; color: var(--muted); }
.tag.ok { background: var(--accent-weak); color: var(--accent); }
.fp { font-size: 12px; color: var(--muted); font-weight: 400; }
.field { margin-top: 12px; display: flex; flex-direction: column; gap: 4px; }
.field label { font-size: 13px; color: var(--muted); }
.field input {
  height: 32px; padding: 0 8px; box-sizing: border-box;
  border: 1px solid var(--border); border-radius: var(--r-sm);
  background: var(--surface); color: inherit; font: inherit;
}
.field input:focus { outline: 0; border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-weak); }
.field input:disabled { background: #f4f4f4; color: var(--muted); }
.actions { margin-top: 12px; display: flex; gap: 8px; }
.section-title { margin: 18px 0 8px; font-size: 14px; color: var(--muted); font-weight: 600; }
.mini { width: 100%; }
.mini th, .mini td { text-align: left; padding: 4px 8px; }

/* 列宽写死，表头与内容共用同一条网格 —— 「对齐」这件事靠固定列宽，不靠 padding 凑 */
.settings-table { table-layout: fixed; width: 100%; }
.settings-table th:nth-child(1) { width: 118px; }
.settings-table th:nth-child(2) { width: 64px; }
.settings-table th:nth-child(3) { width: 170px; }
.settings-table th:nth-child(4) { width: 30%; }
.settings-table th:nth-child(5) { width: 72px; }
.settings-table th:nth-child(6) { width: 56px; }
.settings-table th:nth-child(7) { width: 96px; }
.settings-table th, .settings-table td { vertical-align: middle; }
.settings-table tbody th { text-align: left; color: var(--muted); font-weight: 400; width: 120px; }
.cell {
  width: 100%; box-sizing: border-box; height: 30px; padding: 0 8px;
  border: 1px solid var(--border); border-radius: var(--r-sm);
  background: var(--surface); color: inherit; font: inherit;
}
.cell.mono { font: var(--mono); }
.cell:focus { outline: 0; border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-weak); }
.bad { color: var(--danger); }
</style>
