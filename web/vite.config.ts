import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 后端在 127.0.0.1:8000。**开发期用代理，不加 CORS**：
// 后端所有端点都在根路径下（/search、/chat、/agent…），所以这里逐个列出来。
// 生产由 FastAPI 直接托管 `web/dist`（见 api/app.py 的 mount），那时不需要代理。
const BACKEND = process.env.RAG_API ?? 'http://127.0.0.1:8000'
const API_PATHS = [
  '/health',
  '/search',
  '/chat',
  '/ingest',
  '/documents',
  '/jobs',
  '/agent',
  '/assets',
  '/chunks',
  '/logs',
  // **新增后端路径时这里一定要同步加**：漏了的话，dev 模式下 Vite 会把它当成前端路由接走
  // —— 第一次回 index.html、第二次带 If-None-Match 就回 **304**，页面上看到的是
  // 「HTTP 304」（实测踩过：`/sessions` 漏了，对话页的会话下拉一直报 304）。
  // 前缀能覆盖子路径（`/agent` 管 `/agent/resume`、`/logs` 管 `/logs/tail`），所以只要列前缀。
  '/sessions',
  // 模型选型的只读接口（设置页用它）
  '/models',
  // 供应商设置（key / base_url / 模型清单，设置页用它）
  '/providers',
]

export default defineConfig({
  plugins: [vue()],
  server: {
    // **显式绑 IPv4**：Vite 默认只监听 localhost，在 Windows 上会落成 `::1`，
    // 于是 `127.0.0.1:5173` 连不上（脚本里印的就是这个地址，真机踩过）。
    host: '127.0.0.1',
    port: 5173,
    proxy: Object.fromEntries(API_PATHS.map((path) => [path, { target: BACKEND, changeOrigin: true }])),
  },
  build: {
    outDir: 'dist',
    // **不要用默认的 `assets`**：后端有个 `/assets/{path}` 端点（取语料图片），
    // 而它挂在鉴权后面 —— 浏览器的 <script>/<link> 带不了 X-API-Key，会直接 401。
    // 换成 `static` 就各走各的：静态页不需要 key，数据端点全都要。
    assetsDir: 'static',
  },
})
