# multimodal_rag

[![CI](https://github.com/wanqi-development/multimodal_rag/actions/workflows/ci.yml/badge.svg)](https://github.com/wanqi-development/multimodal_rag/actions/workflows/ci.yml)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**多模态** RAG 检索系统：语料以多种模态进来（文本 / 数字版 PDF / 扫描件 / 整页图片 / DOCX /
XLSX / HTML），统一解析成带元数据的文本后入库检索；交付形态是 **FastAPI 服务 + Vue 3 工作台 +
Typer CLI + 对话式 agent**。

> 命名：仓库与发行包叫 `multimodal_rag`，导入包与 CLI 入口叫 `rag`
> （历史上从 `multi-mode-rag` 改过来，见 [docs/architecture.md](docs/architecture.md) §21）。

## 实测结果

| 口径 | 数字 |
|---|---|
| 检索（22 题 · 默认含精排 · top-k 8） | recall@8 **86.4%** · MRR **75.6%** · nDCG@8 78.2% · top1 **68.2%** |
| 检索（同一批 · 关掉精排） | 81.8% · 54.7% · 61.1% · 45.5% —— 精排 **+20.9pp MRR / +22.7pp top1** |
| 生成（12 题带判据） | 引用准确率 **90.0%**（macro）· 要点覆盖率 94.4% · 漏拒 0 · 误拒 0 |
| A/B：BM25 独立通道（27 题） | nDCG@8 75.58% → **75.36%**（关掉反而更好）→ 该通道删除 |
| A/B：Neo4j 图谱通道（24 题） | 逐题名次变化 **0 条** → 默认关闭，不全量灌图 |

口径、逐题清单、以及工程侧实测（精排耗时、PDF 解析选型、embedding 吞吐……）见
[**docs/results.md**](docs/results.md)；「为什么这么选」的论证过程见
[docs/architecture.md](docs/architecture.md)。

## 60 秒 demo

自带三篇语料，**一分钱不花**（本地 embedding + 本地精排，不调 VLM / OCR / 云端），
也不碰你已有的数据（独立 collection + 独立的 `data/demo/`）：

```powershell
scripts\demo.cmd                 # Windows：解析 → 切分入库 → 检索 / 问答 / agent → 一个该被拒答的
scripts\demo.cmd -DryRun         # 只看它要做什么，一个字都不写
```

```bash
./scripts/linux/demo.sh          # Linux / macOS
```

首次灌库实测约 **38 秒**（3 篇 → 14 块），之后每条查询约 20 秒。步骤、真实输出、
以及页面版录屏的分镜见 [**docs/demo.md**](docs/demo.md)。

<!-- 录屏接上之后取消下面这行的注释（成图放 docs/assets/demo.gif）：
![60 秒 demo](docs/assets/demo.gif)
-->

## agent 层

对话式 agent 是这个项目最厚的一块（`src/rag/agent/`，见 architecture §11）。

| 环节 | 做法，以及为什么 |
|---|---|
| 工具面 | 只有 **4 个**工具：`search_knowledge`（**一次调用 = 一整轮融合检索**：所有启用通道并发 + RRF 融合 + 精排，`modality` 用来只看某类块）/ `list_documents` / `get_section` / `web_search_tool`。**刻意不做 subagent** —— 通道的并发与融合本来就在检索层做完，让模型分头调、再用文字拼结果，等于把 RRF 的分数换成模型的印象 |
| 网络兜底 | 只当兜底，而且靠**机制**管、不靠"不给它工具"：`WebSearchGate` 在语料没查过、或语料已有结果时直接拒绝这次调用并回理由 |
| 人在环（HITL） | 四个写类工具（`write_file` / `edit_file` / `delete` / `execute`）各挂一道闸门；中断与恢复走 LangGraph 原生协议（`Command(resume={"decisions": [...]})`），**跨进程**可用（CLI 一次进程、HTTP 一次请求都行）。CLI 中断退出码是 `3`，与真失败（`1`）分开，脚本里能一句话分辨 |
| 流式 | `stream_mode=["messages", "updates"]` → SSE 四类事件 `token` / `tool` / `interrupt` / `done`；靠 `checkpoint_ns` 把子图 token 滤掉，不把中间话术当答案吐给用户 |
| 引用编号 | `CitationRegistry` 发全局递增的 `[n]`：工具返回的每条命中拿号并登记，答案里的 `[n]` 直接查表；返回给调用方时只列答案**真的引用到**的那些（召回 20 条只用 3 条时，全塞回去就看不出哪条支撑哪句话）。引用表按会话存 PG —— 答案是中断恢复**之后**才生成的，用的还是中断前发的号 |
| 记忆与技能 | PG checkpointer（会话 / HITL 中断）+ store（跨会话记忆）；`workspace/AGENTS.md` 挂成 memory；skills 按需加载，两个源（包内 `src/rag/agent/skills/` + 用户目录，同名以用户为准） |
| 编排 | LangGraph 两张图，都**只做编排**：`QueryGraph`（plan → 并行通道 → 融合 → 精排 → **评分不够就改写重查**）与 `IndexGraph`（解析 / 切分 / 入库三段断点续跑） |

## 架构

```
入口层   FastAPI（/search /chat /ingest /documents /logs /agent …）│ Typer CLI │ deepagents agent
编排层   LangGraph：QueryGraph（在线，带自纠错循环）│ IndexGraph（离线灌库）
检索层   主通道 hybrid(dense + sparse，Milvus 服务端 RRF)
         增强：多路通道并发 → 客户端 RRF 融合 → cross-encoder 精排 → 装配（去重 / token 预算 / 引用编号）
索引层   Milvus(dense + 内建 BM25) │ 关系库（注册表 / 台账 / 选型 / 日志索引）│ 文件（本地盘 / MinIO）
接入层   loader → 解析（或 OCR 产线）→ 清洗 → 标题核对 → 切分（含 VLM 图片描述）
基础层   contracts（零第三方依赖的纯协议层）│ config（YAML + .env 插值）│ providers │ 观测 │ 评测
```

**两层融合要分清**：L1 是 Milvus 服务端把 dense 与 sparse 用 RRF 合成一路；L2 是客户端再把
hybrid 与图谱 / web 这类**别的数据源**合一次。

**「多模态」的落点是接入层 + 存储层，查询侧不做跨模态**：检索与生成都是「文本 query 对文本块」，
图像不作为 query，所以不需要 CLIP 那类模型。每种格式一个 loader；扫描件与整页图片交 OCR 产线
（PaddleOCR-VL / dots.ocr / MinerU 三选一，可切云端兜底）；文档内嵌图抽出来单独存（本地盘或
MinIO 桶），正文里只留 `![](asset:<doc_id>/<名>)` 短标记，图片另有一条 `modality=image` 的
VLM 描述块进检索。

## 一条链路长什么样

```
上传 / 从链接导入
 └─ 解析（loader，或 OCR 产线）            → 文档 + 图片资产
     └─ 人工核对大纲（页面上改标题层级、修正正文）
         └─ 切分（每张图一次 VLM 描述）     → 块（含图片块）
             └─ 写 Milvus（dense + sparse）
                 └─ 检索 / 问答 / agent（引用能回原文）
```

三段流水线能分开跑：`rag ingest --stage parse|chunk|index`。中间产物落在
`data/stages/<doc_id>/`：一份给人看的 `.md`（页码行号、章节路径、块边界都摊在上面），
一份给下一段读的 `.json`。前两段不碰 Milvus —— 库挂了也能先把最贵的 OCR / VLM 做完。

## 快速开始

### 1. 环境

Python **3.13**（`pyproject.toml` 锁了 `>=3.13,<3.14`；langgraph 官方声明封顶 3.13）。

```bash
conda create -n multi_modal_rag python=3.13
conda activate multi_modal_rag
pip install -e . --group dev     # 含 pytest / ruff；要跑评测再加 --extra eval
rag --help                       # CLI 入口
```

### 2. 配置

```bash
cp .env.example .env             # Windows: Copy-Item .env.example .env
```

`.env` 已被 gitignore，永不提交。**最少**要在里面填一个 chat 供应商的三件套
（如 `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL_NAME`），否则
`rag ask` 与 agent 起不来；向量化默认走本地模型，不用填。其余每一项
`.env.example` 里都写了「不填会怎样」。

配置分两层：`config/base.yaml` 声明结构与默认值，`config/<profile>.yaml` 覆盖
（`RAG_PROFILE`，默认 `dev`），**机器相关与机密一律放 `.env`**（yaml 里只写
`${VAR:-默认}` 这种插值）。两个"统一开关"值得先知道：

| 开关 | 取值 | 管什么 |
|---|---|---|
| `RAG_DB_BACKEND` | `auto`（默认）/ `sqlite` / `postgres` | 文档注册表 + 上传件台账 + 模型选型 + 日志索引 |
| `RAG_STORAGE_BACKEND` | `local`（默认）/ `minio` | 上传件 + 中间产物 + 原始资产 + 会话图片 |

想看最终生效的是什么，用 `rag config`。

### 3. 依赖服务

| 服务 | 必需？ | 起法 |
|---|---|---|
| Milvus（etcd + MinIO + milvus） | **要**（灌库与检索都靠它） | `cd deploy && docker compose up -d` |
| OCR 产线 | 扫描件才要 | `deploy/paddleocr-vl.yaml`，按 `RAG_PDF_PARSER` 选一个（dots / mineru / paddleocr） |
| PostgreSQL | 不要（单机走 SQLite） | 多 worker / 多机才配 `RAG_PG_URL` |
| MinIO | 不要（单机走本地盘） | 多 worker 才把 `RAG_STORAGE_BACKEND` 切 `minio` |
| Redis / Neo4j | 不要 | 分别是限流+任务状态、知识图谱通道 |

### 4. 本地模型

默认走本地推理，权重放 `.models/`（已 gitignore，不入库）：

| 用途 | 目录 | 怎么来 |
|---|---|---|
| embedding（默认） | `.models/Qwen3-Embedding-0.6B` | 从 **ModelScope** 下（hf-mirror 拉这个会卡在 0 字节，实测） |
| 重排（默认开） | `.models/bge-reranker-base` | 同上；不要重排就把 `RAG_RERANK` 设成 `false` |

也可以用 `EMBEDDING_MODEL` / `RAG_RERANK_MODEL` 指到别处，或者把 embedding 换成线上的
`qwen3-vl-embedding`（图 + 文同一个向量空间，按量计费）—— 见 `.env.example` 的 embedding 那一段。

### 5. 把服务起来

```powershell
scripts\start.cmd              # Windows：后端 → 等 /docs 就绪 → 前端
./scripts/start.sh             # Linux / macOS（只起后端加 --backend）

# 调后端时不想拖一个 Vite 起来：
scripts\windows\backend.cmd    ./scripts/linux/backend.sh
scripts\windows\frontend.cmd   ./scripts/linux/frontend.sh
```

**前端会等后端 `/docs` 能应答之后再起** —— 后端要加载 embedding 等重对象，几十秒是常态；
等就绪的上限默认 120 秒，可用 `RAG_BACKEND_WAIT_S` 或 `-WaitSeconds N` 调整。

起来之后：页面 <http://127.0.0.1:5173>（热更新），接口文档 <http://127.0.0.1:8000/docs>。
生产用 `cd web && pnpm.cmd build`（或 `npm.cmd run build`）再 `rag serve` —— 它会自动把
`web/dist` 挂到 `/`，一个进程就够。

### 6. 灌第一批语料

页面上是四步：**上传 / 导入 → 解析 → 核对大纲 → 切分并入库**。命令行等价：

```bash
rag ingest ./data/docs               # 解析 → 切分 → 写 Milvus（默认增量；--full 全量重灌）
rag ingest ./data/docs --stage parse # 只解析：不碰 Milvus，先把 OCR 跑完
rag ingest ./data/docs --dry-run     # 只看会做什么，一个字都不写
rag search "Flink 的 Savepoint 是什么"  # 检索，打印 chunk 与来源
rag ask    "Flink 的 Savepoint 是什么"  # 检索 + 生成带 [n] 引用的答案
rag agent  "Flink 的 Savepoint 是什么"  # 对话式 agent（多轮加 --thread-id）
```

## 常用命令

`rag --help` 是权威清单。按用途分：

| 命令 | 干什么 |
|---|---|
| `rag config` | 打印最终生效的配置（含每个 `${}` 插值的结果）—— 排查"我改的没生效"先看它 |
| `rag ingest <路径>` | 灌库：解析 → 切分 + 图片描述 → 写 Milvus |
| `rag search` / `rag ask` / `rag agent` | 检索 / 带引用问答 / 对话式 agent |
| `rag serve` | 起 FastAPI 服务（有 `web/dist` 就一并托管页面） |
| `rag eval` | 跑黄金集（`--judge` 再加生成侧判定，花钱） |
| `rag registry` / `rag storage` / `rag models` | 文档注册表 / 文件落点（含把本地已有文件推上桶）/ 模型选型 |
| `rag logs` / `rag chats` / `rag sessions` | 日志 / UI 会话记录 / LangGraph 会话 |
| `rag graph` | 知识图谱通道（**由插件提供**，默认关） |

## 目录结构

```
src/rag/
  config/       YAML + .env 的配置体系（base.yaml 声明结构，值放 .env）
  contracts/    核心抽象：Document / Chunk / Hit 与各层协议（零第三方依赖）
  ingest/       接入层：loader、OCR、清洗、标题、切分器、三阶段流水线
  indexing/     索引层：Milvus collection 与写入
  retrieval/    检索层：多通道、融合、重排、组装、插件加载
  graphs/       编排层：LangGraph（灌库图 / 查询图）
  agent/        对话式 agent：工具、中间件、时间线（deepagents）
  api/          FastAPI 服务（文档 / 检索 / 问答 / 日志 / 会话 / 模型选型）
  cli/          Typer CLI（`rag`）
  persistence/  SQLite 与 PostgreSQL 两套实现（注册表、台账、选型、日志）
  obs/          日志：四个文件 + 可选 PG 索引 + 实时流
  providers/    chat / vision / asr / embedding / rerank / OCR 的适配
  storage.py    文件落点门面：本地盘 或 MinIO 桶，键的写法两边一致
web/            Vue 3 + TypeScript + Vite 前端（没引 UI 组件库）
plugins/        两个扩展点（检索通道 / CLI 命令）的现成样板：rag-channel-neo4j
config/         base.yaml 与 dev.yaml（值放 .env）
deploy/         Milvus 与 OCR 产线的 compose
eval/           黄金集（跑出来的报告落在 eval/reports/，不进库）
tests/          pytest（915 个用例；需要真 Milvus / OCR 的集成用例默认跳过）
```

> `src/rag/ingest/dots_ocr/` 是 **dots.ocr 官方仓库整包拿过来的第三方样例代码**（含它自己的
> tools 与演示脚本），不是本仓库写的；我们只通过 `ingest/dots.py` 这个薄客户端调它，
> 所以它被排除在 lint / format 之外（见 `pyproject.toml` 的 `extend-exclude`）。

## 已知边界

这些是**已经知道、目前有意不做**的部分，不是"还没做完忘了写"：

* **图谱通道默认关**：860 行代码与两篇文档的图保留着，但 A/B 实测在这份语料上没有可测量的
  提升（见上表），所以先不全量灌图 —— 要让图通道有用，得先有真正需要跨文档多跳的问题。
* **隔离的 `web` 检索通道没接**：agent 侧有 web 搜索工具（带闸门），检索层还没有 web 适配器。
* **存储侧还差一块**：页图与表格片段未落地，资产也没有清理动作（文档改短后旧图会留在盘上）。
* **历史会话里的 `[n]` 点不开**：只有当轮引用可点（`[n] → chunk_id` 的映射来自当轮事件，不入库）。
* **模型选型的界面暂时下线**：视图、接口与用例都还在，只是导航里撤了入口；且选型**重启生效**。
* **黄金集是 30 行的小集合**：够用来做 A/B 对照与回归，不够用来宣称绝对水平。

## 开发

```bash
python -m pytest              # 全量用例
python -m ruff check          # 静态检查
python -m ruff format --check

cd web && pnpm.cmd dev        # 前端开发模式（热更新）
cd web && pnpm.cmd build      # 出 web/dist，交给 rag serve 托管
```

## 文档

| 文件 | 内容 |
|---|---|
| [docs/results.md](docs/results.md) | 实测数字汇总：口径、逐题清单、工程侧实测（每张表都指向论证出处） |
| [docs/architecture.md](docs/architecture.md) | 主文档：分层设计、配置体系、每个决策的理由（ADR）与验证清单 |
| [docs/milvus-schema-design.md](docs/milvus-schema-design.md) | collection / 字段 / 索引为什么是现在这样 |
| [docs/plugins.md](docs/plugins.md) | "按名字找实现"的插件约定（通道 + CLI 两个扩展点） |

## 许可

MIT，见 [LICENSE](LICENSE)。
