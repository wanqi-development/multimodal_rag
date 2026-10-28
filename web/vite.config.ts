import { fileURLToPath } from 'node:url'
import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'

// 后端在 127.0.0.1:8000。**开发期用代理，不加 CORS**：
// 后端所有端点都在根路径下（/search、/chat、/agent…），所以这里逐个列出来。
// 生产由 FastAPI 直接托管 `web/dist`（见 api/app.py 的 mount），那时不需要代理。
const ROOT_ENV_DIR = fileURLToPath(new URL('../', import.meta.url))
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

export default defineConfig(({ mode }) => {
  // 直接 `cd web && pnpm dev` 也读取仓库根目录的 .env；启动脚本注入的真实
  // 环境变量优先级更高。
  const env = loadEnv(mode, ROOT_ENV_DIR, '')
  const backend = process.env.RAG_API ?? env.RAG_API ?? 'http://127.0.0.1:8001'
  const webHost = process.env.RAG_WEB_HOST ?? env.RAG_WEB_HOST ?? '127.0.0.1'
  const rawPort = process.env.RAG_WEB_PORT ?? env.RAG_WEB_PORT ?? '5174'
  const webPort = Number.parseInt(rawPort, 10)
  if (!Number.isInteger(webPort) || webPort < 1 || webPort > 65535) {
    throw new Error(`RAG_WEB_PORT 不是合法端口：${rawPort}`)
  }
  return {
    plugins: [vue()],
    server: {
      // **显式绑 IPv4**：Vite 默认只监听 localhost，在 Windows 上会落成 `::1`。
      // host/port 由 RAG_WEB_HOST / RAG_WEB_PORT 统一提供。
      host: webHost,
      port: webPort,
      // 端口被占时直接失败，不要悄悄切到 5175 让脚本和实际地址再次漂移
      strictPort: true,
      proxy: Object.fromEntries(
        API_PATHS.map((path) => [path, { target: backend, changeOrigin: true }]),
      ),
    },
    build: {
      outDir: 'dist',
      // **不要用默认的 `assets`**：后端有个 `/assets/{path}` 端点（取语料图片），
      // 而它挂在鉴权后面 —— 浏览器的 <script>/<link> 带不了 X-API-Key，会直接 401。
      // 换成 `static` 就各走各的：静态页不需要 key，数据端点全都要。
      assetsDir: 'static',
    },
  }
})
