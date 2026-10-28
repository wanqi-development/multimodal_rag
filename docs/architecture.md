# multi-modal-rag 技术方案

> 状态：定稿（2026-09-21）
> 技术栈：Python + LangChain + LangGraph + deepagents + Milvus

---

## 1. 项目定位与范围

一个**多模态** RAG 检索项目：语料以多种模态进入，在接入层统一成文本，再走上层的检索链路。

**「多模态」的落点只有接入层 + 存储层：**

| 层 | 多模态体现在哪 |
|---|---|
| 接入层 | 每种模态一个 loader，出口统一是「带元数据的文本」。已实现：md / txt 族、PDF（parser 可配置，默认 `dots` 可选插件；PaddleOCR-VL / MinerU 可切换，ADR 17/18/23）、docx / html / xlsx（代码解析）、图片（OCR + VLM 描述，§7.4）。计划：csv / 代码 / web 抓取 |
| 存储层 | 文本块与向量进 Milvus（dense + 内建 BM25 sparse，§8）。原始资产：**内嵌图已落地**（本体进 `ingest.assets.dir`，正文留标记，图片另有 `modality=image` 的描述块，§7.4）；页面图像与表格片段仍未落地，见 §21 |

**非目标：查询侧不做跨模态。** 检索与生成一律是文本 query 对文本块 —— 图像不作为 query，
也就不需要 CLIP 那类多模态 embedding 与另一套向量空间。将来若要加，属于新决策，要走 ADR（§17）。

工程结构上另有两个维度，它们是**实现结构**，不是「多模态」的含义：

- **纵向流水线分层**：文档接入 → 解析切分 → 索引 → 检索 → 生成 → 评测，每层组件可插拔替换
- **横向多路检索通道**：dense + sparse 混合检索为主通道，BM25 / 图谱 / Web 为可选叠加通道

交付形态（三个都要）：

1. **FastAPI 服务** — 对外 HTTP 接口
2. **Typer CLI** — 灌库、检索、评测的调试入口
3. **deepagents 对话式 Agent** — 规划 + 多路子 agent 并行检索

**核心设计原则：`contracts/` 是纯协议层，零第三方依赖。** 上层组件只依赖协议不依赖实现。换向量库、换 embedding 模型、加一路检索通道，都只动一个文件。

---

## 2. 环境与依赖决策

### 2.1 Python 版本

> **2026-09-27 补注**：这一节记的是当时的机器状态。miniconda 后来从
> `D:\development_software\miniconda3` 搬到了 `D:\developmentBinary\miniconda3`，
> 原文提到的 `D:\development_software\python3.4.17\python.exe` 已不存在（那个目录现在堆的是安装包）。
>
> **搬家的副作用，值得记一笔**：pip 生成的 `rag.exe` 控制台脚本把解释器路径**烧进了 exe 内部**的
> shebang。搬完之后它仍指向 `D:\development_software\miniconda3\envs\...\python.exe` —— 路径不存在，
> 于是 `rag` 命令**静默失败（exit 1，一个字都不输出）**，而
> `python -c "from rag.cli.main import app; app()"` 完全正常。**这个症状不该怀疑代码**，
> `pip install -e . --no-deps` 重装一次即可（2026-09-27 已随改名一起修好）。

**3.13 兼容性：全绿，无任何硬阻塞。**

逐一核实了方案里全部依赖的 wheel 文件（2026-09-21 读 PyPI，比 classifier 可靠）：

| 包 | 3.13 | 3.14 | 形态 |
|---|---|---|---|
| langgraph 1.2.11 | ✅ | ⚠️ | **纯 Python 轮子**；classifier 封顶 3.13 |
| ├ xxhash 4.0.1 | ✅ | ✅ | 编译型，有 cp314 wheel |
| ├ orjson 3.12.0 | ✅ | ✅ | 编译型，有 cp314 wheel |
| └ zstandard 0.25.0 | ✅ | ✅ | 编译型，有 cp314 wheel |
| langchain 1.4.2 / core / classic / openai | ✅ | ✅ | 纯 Python |
| langchain-community 0.4.2 | ✅（classifier 未维护） | ✅ | 纯 Python |
| deepagents 0.7.15 | ✅ | ✅ | 声明 ≥3.11 |
| pymilvus 3.0.2 | ✅ | ✅ | `py3-none-any` 纯 Python |
| milvus-lite 3.2.1 | ✅ | ✅ | `py3-none-any`（运行时行为另说，见 §14） |
| fastapi / typer / pydantic-settings / uvicorn | ✅ | ✅ | 纯 Python |
| fastembed 0.8.0 | ✅ | ✅ | 纯 Python + onnxruntime |
| onnxruntime 1.30.0 | ✅ | ✅ | 编译型，cp313 / cp314 都有 wheel |
| torch 2.14.0 | ✅ | ✅ | 编译型，cp310–cp314 都有 wheel |
| pymupdf 1.28.2 | ✅ | ✅ | **`cp310-abi3` 轮子，前向兼容** |
| FlashRank / rank-bm25 / ragas / langsmith | ✅ | ✅ | 纯 Python，classifier 未维护但不影响安装 |

**关键发现：3.14 的"风险"基本是误报。** langgraph 本身是纯 Python 轮子，它依赖的三个编译型包（xxhash / orjson / zstandard）全部提供 cp314 wheel —— 所以 langgraph 在 3.14 下能装能 import，classifier 只是滞后。

**那为什么仍然选 3.13？** 因为**声明支持的边界**比"能否装上"更重要：

- langgraph 官方 classifier 封顶 3.13，而它是整个项目的编排基座。跑在未声明支持的版本上，一旦遇到怪异 bug，等于在无人走过的路上调试核心框架，上游也未必受理
- 其余所有包在两个版本上无差别
- 切 3.13 的成本只有一条命令

**决策：用 3.13。**

```bash
conda create -n multi_modal_rag python=3.13
conda activate multi_modal_rag
pip install -e . --group dev
```

3.14 作为备选保留，但没必要为省一条命令去赌核心框架。

### 2.2 关键依赖选型

| 层 | 选型 | 为何不用常见的替代 |
|---|---|---|
| 环境与包管理 | **conda**（建环境）+ **pip**（装包） | 本机已装 miniconda，conda 本身就解决多 Python 版本共存；装包用 `pip install -e . --group dev`（PEP 735 依赖组），没必要再叠一层工具 |
| 向量库 | **Milvus**（用户指定首选） | — |
| Milvus 客户端 | **pymilvus 直连** | 不用 langchain-milvus：wrapper 不暴露 sparse 字段 + `hybrid_search`，且无 3.14 支持 |
| 稀疏表示 | Milvus 内建 **BM25 Function**（服务端从原文生成） | 初稿的「一个模型双输出」落空；Milvus 的 BM25 function 不需要稀疏编码模型、不需要 torch，中文分词交给服务端 analyzer。见 ADR 13 |
| Embedding | 默认 **Qwen3-Embedding-0.6B**（1024d，纯文本）；可选 **qwen3-vl-embedding**（2560d，线上，图+文同一空间） | 换掉 bge-small-zh 是因为它窗口只有 512 token，块一超就被**静默截断**。Qwen3-Embedding 窗口 32768，chunk 才谈得上按语义放宽。代价是依赖 torch。两种方式见 §8.3。见 ADR 21（并取代 ADR 14） |
| Rerank | flashrank(ONNX) | 避开 FlagEmbedding（无 3.14 元数据） |
| 图存储 | networkx 起步 → Neo4j | 先跑通再上库 |
| 编排 | LangGraph 1.2 | 用户指定 |
| Agent | deepagents 0.7 | 用户指定 |
| 评测 | ragas + 自建 | 引用准确率 ragas 不直接给 |
| 观测 | LangSmith | 调 agent 行为必备 |

### 2.3 LangChain 1.x 模块归属（重要）

LangChain 1.x 重排过模块归属，网上大量 0.x 教程的 import 路径会直接报错：

- `langchain` 1.4.2 — `create_agent` 等核心
- `langchain-classic` 1.0.8 — 传统 retriever / chain（**不是** `langchain.retrievers`）
- `langchain-text-splitters` 1.1.2 — 切分器
- ~~`langchain-community`~~ — **不使用**。M0 实测该包已抛 sunset 警告（官方停止维护，建议迁往独立集成包）。本项目 `ingest/loaders/` 自研，PDF 用 pymupdf、docx 用 python-docx、md/csv/txt 用标准库，不需要它

M0 装完依赖后先用实际 import 验证一遍再往下写。

---

## 3. 整体架构

```
┌───────────────────── 入口层 ─────────────────────┐
│  FastAPI(/ingest /search /chat)  │  Typer CLI     │
│  deepagents 对话式助手                            │
└───────────────────────┬──────────────────────────┘
┌───────────────────────▼──────────────────────────┐
│  LangGraph 编排层   IndexGraph │ QueryGraph       │
└───────────────────────┬──────────────────────────┘
┌───────────────────────▼──────────────────────────┐
│  检索层                                          │
│  主通道: hybrid(dense + sparse) ← Milvus 服务端融合│
│  可选通道（数据源）: graph │ web（未接）          │
│  增强: query transform → 跨通道融合(RRF) → rerank │
│        → 上下文装配(去重/预算/引用编号)            │
└───────────────────────┬──────────────────────────┘
┌───────────────────────▼──────────────────────────┐
│  索引层  Milvus(dense+sparse) │ 父块存储 │ 图存储  │
└───────────────────────┬──────────────────────────┘
┌───────────────────────▼──────────────────────────┐
│  接入层  loader → parser → chunker → 元数据增强   │
└───────────────────────┬──────────────────────────┘
┌───────────────────────▼──────────────────────────┐
│  基础层  Contracts │ Config │ Providers │ 观测 │ 评测│
└──────────────────────────────────────────────────┘
```

**两层融合**要分清：

- **L1 — Milvus 服务端**：dense + sparse 用 `RRFRanker` 融合，一次 `hybrid_search` 调用完成
- **L2 — 客户端**：hybrid 结果再和 graph / web 这类**别的数据源**做 RRF

---

## 4. 目录结构

```
multi-modal-rag/
├── pyproject.toml
├── README.md
├── docs/architecture.md            # 本文档
├── config/
│   ├── base.yaml
│   └── dev.yaml / prod.yaml.example
├── deploy/
│   └── docker-compose.yml          # etcd + MinIO + milvus standalone
├── src/rag/
│   ├── contracts/                  # ★ 地基：纯协议，零依赖
│   │   ├── document.py             # Document / Chunk / ScoredChunk / RetrievalResult
│   │   ├── protocols.py            # Parser/Chunker/Embedder/Index/Retriever/Reranker
│   │   └── registry.py             # @register_channel("hybrid") 插件注册
│   ├── config/                     # pydantic-settings，env 覆盖 yaml
│   ├── providers/
│   │   ├── chat.py                 # init_chat_model 封装（重试/超时/降级）
│   │   ├── embedding.py            # 独立注册表（embedding 供应商集合 ≠ chat）
│   │   └── rerank.py               # flashrank / API 可切
│   ├── ingest/
│   │   ├── loaders/                # pdf docx html md csv code web
│   │   ├── parsers/                # 表格抽取、代码块保留、OCR 兜底
│   │   └── chunkers/               # recursive / markdown-aware / parent-child / semantic
│   ├── indexing/
│   │   ├── milvus/
│   │   │   ├── schema.py           # ★ 显式 schema + 幂等建索索引
│   │   │   └── client.py           # load/release 生命周期
│   │   ├── parent_store/           # small-to-big 的父块
│   │   └── graph/
│   ├── retrieval/
│   │   ├── channels/               # milvus_hybrid.py graph.py（bm25 2026-10-08 删）
│   │   ├── transform/              # hyde / multi_query / decompose
│   │   ├── fusion/                 # rrf / weighted
│   │   ├── rerank/
│   │   └── assembler.py            # 去重 + token 预算 + 引用编号
│   ├── graphs/
│   │   ├── index_graph.py
│   │   ├── query_graph.py
│   │   └── state.py
│   ├── agent/                      # deepagents: main / subagents / tools / backend
│   ├── api/
│   ├── cli/
│   ├── eval/
│   └── observability/
└── tests/
    ├── unit/
    ├── integration/
    └── conformance/                # ★ 供应商一致性测试
```

`tests/conformance/` 容易被忽略但很关键：多供应商切换时，用同一组用例验证「工具调用、结构化输出、长上下文截断行为」是否一致。没有它，换供应商就是在赌。

**M1 结束时的实际状态：** 上面是目标布局，已落地的是 `contracts/`（全部）、`config/`、
`providers/{chat,embedding,generator}.py`、`ingest/loaders/{base,text,pdf}.py`、
`ingest/chunkers/{base,recursive,markdown}.py`、`indexing/milvus/{schema,client}.py`、
`retrieval/channels/milvus_hybrid.py`、`retrieval/assembler.py`、`cli/main.py`。

尚未落地：`ingest/parsers/`、`indexing/{parent_store,graph}/`、`retrieval/{transform,fusion,rerank}/`、
`graphs/`、`api/`、`agent/`、`eval/`、`observability/`。

`tests/` 目前是平铺一层（M1 的用例按层分文件）。`unit/` + `integration/` + `conformance/`
三层拆分等到用例多到需要分流再做 —— 现在就拆只会多出两层空目录。

---

## 5. 核心抽象（contracts）

所有检索通道实现同一协议，这是「多路通道融合」能成立的前提
（注意：这是**检索策略**的多路，与「多模态」不是一回事，见 §1）：

```python
class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    channel: str          # 来源通道，融合和调试都要用
    rank: int             # 通道内排名，RRF 需要

class Retriever(Protocol):
    name: str
    async def retrieve(
        self, query: str, top_k: int, filters: dict | None = None
    ) -> list[ScoredChunk]: ...
```

融合层只认 `list[ScoredChunk]`，不关心它来自 Milvus、BM25 还是图谱。**新增一路召回 = 写一个类 + 加一行配置。**

其他协议：

```python
class Parser(Protocol):     def parse(self, doc: Document) -> list[Document]: ...
class Chunker(Protocol):    def split(self, docs: list[Document]) -> list[Chunk]: ...
class Embedder(Protocol):
    async def embed_documents(self, texts: list[str]) -> Embeddings: ...  # dense 必给，sparse 留空
    async def embed_query(self, text: str) -> Embeddings: ...
class Index(Protocol):
    async def ensure_ready(self) -> None: ...   # 建表建索引，必须幂等
    async def upsert(self, chunks: list[Chunk], vectors: Embeddings) -> int: ...
    async def delete(self, chunk_ids: list[str]) -> int: ...
class Reranker(Protocol):
    async def rerank(self, query: str, chunks: list[ScoredChunk], top_n: int) -> list[ScoredChunk]: ...
```

**引用不再按数据源分协议**（2026-10-23）：`Citation` 同时承载语料块和网页结果，
用 `source_type` 区分 `corpus / web`，保留 `chunk_id / url / title / heading_path`。
`EvidenceLedger` 是唯一的发号、去重、答案引用筛选与快照序列化入口；旧
`CitationRegistry` 只是兼容别名。普通装配、Agent 工具与会话恢复都走这一套。

`Embeddings.sparse` 字段保留，但当前恒为 `None`：dense 那条腿由 fastembed 出，sparse 那条腿由
Milvus 服务端从原文生成，Python 侧根本不经手。字段留着是为了将来换成「自带 sparse 输出的模型」
时不动协议 —— 那时要另配一个 sparse 字段，**不能**复用 BM25 function 的输出字段。

---

## 6. 配置体系

```yaml
# config/base.yaml（M1 现状，省去注释）
providers:
  chat:
    # 每家三件套都从 .env 来：<PROVIDER>_API_KEY / _BASE_URL / _MODEL_NAME。
    # 下面写的默认值只在 .env 没给时兜底（见下「base_url / model 一律走插值」）
    default:   { provider: deepseek,  model: ${DEEPSEEK_MODEL_NAME:-deepseek-chat},     base_url: ${DEEPSEEK_BASE_URL:-https://api.deepseek.com/v1} }
    openai:    { provider: openai,    model: ${OPENAI_MODEL_NAME:-gpt-4o-mini},         base_url: ${OPENAI_BASE_URL:-} }
    dashscope: { provider: dashscope, model: ${DASHSCOPE_MODEL_NAME:-qwen-plus},        base_url: ${DASHSCOPE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1} }
    zhipuai:   { provider: zhipuai,   model: ${ZHIPUAI_MODEL_NAME:-glm-4-flash},       base_url: ${ZHIPUAI_BASE_URL:-https://open.bigmodel.cn/api/paas/v4} }
    moonshot:  { provider: moonshot,  model: ${MOONSHOT_MODEL_NAME:-moonshot-v1-8k},    base_url: ${MOONSHOT_BASE_URL:-https://api.moonshot.cn/v1} }
    # openai 有原生集成，base_url 留空 = 官方地址；走 OpenAI 兼容通道的
    # （deepseek / dashscope / zhipuai / moonshot）必须有地址，否则 build_chat 直接抛错
  embedding:
    # sentence-transformers（纯文本）| qwen3-vl（图+文同一空间），见 §8.3
    backend: ${EMBEDDING_BACKEND:-sentence-transformers}
    model: ${EMBEDDING_MODEL:-.models/Qwen3-Embedding-0.6B}   # 本地路径：hf-mirror 拉不动，走 ModelScope
    provider: ${EMBEDDING_PROVIDER:-dashscope}                # 只有 qwen3-vl 用得上
    endpoint: ${EMBEDDING_ENDPOINT:-https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding}
    dim: ${EMBEDDING_DIM:-1024}  # dense 维度的唯一来源，Milvus collection 的 dim 由它推导（qwen3-vl 是 2560）
    batch_size: 32
    hf_endpoint: ${HF_ENDPOINT:-https://hf-mirror.com}   # huggingface.co 直连不通
    hf_disable_xet: true                                 # 镜像没有 HF 的 Xet 存储接口
  rerank:
    backend: flashrank
    model: ms-marco-MiniLM-L-12-v2
    top_n: 8

indexing:
  vector:
    backend: milvus
    uri: ${MILVUS_URI:-http://localhost:19530}
    collection: rag_qwen3_v1     # 名字带 embedding 指纹：换模型必须换名字重新灌
    # 这里不写 dim：维度由 providers.embedding.dim 推导，两处各写一遍必然漂移
    indexes:
      dense:  { type: IVF_FLAT,              metric: COSINE }
      sparse: { type: SPARSE_INVERTED_INDEX, metric: BM25 }   # 内建 BM25 要求 BM25，不是 IP
    bm25: { enabled: true, input_field: text, output_field: sparse, analyzer: chinese }
  chunk:
    strategy: recursive        # recursive | markdown
    size: 400                  # 字符数，不是 token 数（见下）
    overlap: 60

retrieval:
  hybrid:
    backend: milvus
    dense:  { field: dense,  limit: 20, weight: 1.0 }
    sparse: { field: sparse, limit: 20, weight: 1.0 }
    fusion: { strategy: rrf, k: 60 }   # rrf | weighted（rrf 下权重必须都是 1.0，见 ADR 16）
  optional_channels: []        # 按 query 类型和语料启用，M2 起
  transform: { multi_query: 0, hyde: false }
  rerank:    { enabled: true, top_n: 8 }
  assemble:  { max_tokens: 6000 }

agent:
  chat: ${AGENT_CHAT:-default}   # 填 providers.chat 的键名，不是 "provider:model" 字符串
  subagents: []
  workspace_dir: ./data/workspace
```

**`chunk.size` 是字符数，不是 token 数。** 中文里 1 字 ≈ 1.07 token（实测：550 字 = 512 token）。
旧模型 bge-small-zh-v1.5 的窗口只有 512 token，所以当时的 `size: 400` 是**贴着窗口**留的余量；
现在换到窗口 32768 的 Qwen3-Embedding-0.6B，尺寸约束松了，`size` 可以按语义粒度来定。

**两个来源，优先级明确：** 真实环境变量 > `.env` > yaml 里的默认值。

需要留意的是：pydantic-settings 只把 `.env` 读进 `Secrets` 模型，**不会写进 `os.environ`**，
而 yaml 的 `${VAR:-default}` 插值要靠 `os.environ`。所以配置层额外单独读一份 `.env` 作兜底
（`config/settings.py: _dotenv()`），否则 `.env` 里设的 `MILVUS_URI` 会被静默忽略、悄悄回落到默认值。

**约定：机器相关的值只进 `.env`（已 gitignore），不写进要提交的 yaml。**
yaml 里的默认值保持通用（如 `http://localhost:19530`），换机器只改 `.env`。`.env.example` 是字段清单模板。

**`base_url` 与 `model` 一律走插值。** 每家 chat 供应商在 `.env` 里配三件套，与
`providers.chat.*` 一一对应：`<供应商>_API_KEY`（机密，只有它进 `Secrets`）、`<供应商>_BASE_URL`、
`<供应商>_MODEL_NAME`。换自建网关 / 代理（one-api、new-api 之类）或换模型名，都只改 `.env`，
不用动要提交的配置；yaml 里写的默认值只是 `.env` 没给时的兜底。

`base_url` 留空表示用官方地址 —— 但只对 LangChain 有原生集成的供应商（openai）成立，
走 OpenAI 兼容通道的（deepseek / dashscope / zhipuai / moonshot）没有地址就不知道该往哪发，
`build_chat` 会直接抛错而不是等远端超时。

`model` 是**唯一**不能靠默认值蒙的地方：默认值只是占位，必须按自己账号下的模型列表核对一次。
写错要到第一次 `rag ask` 才以远端 400 暴露，那时报的是远端错误，不容易联想到是配置写错了。

**agent 用哪家模型也是同一个机制**：`agent.chat` 填的是 `providers.chat` 的**键名**，不是
`"provider:model"` 字符串 —— 后者会绕开三件套，`"deepseek:deepseek-chat"` 还需要额外装
`langchain-deepseek`。指到某个条目，就用那家的 key / base_url / model。

这里有个坑：`${OPENAI_BASE_URL:-}` 解析出来可能是**空串**而不是 `None`（yaml 里带引号写就是），
空串会被原生集成当成真实地址去请求，比不传更糟。所以 `providers/chat.py` 在传给 LangChain 之前
统一收敛成 `cfg.base_url or None`。

---

## 7. 接入层 Ingest

**接入层是多模态的第一落点**：每种模态各自进 loader，出口统一成「带元数据的文本」，
好让下面的 chunker / embedding 完全不感知模态差异。目前哪种模态走哪条路见 §7.1。

**Loaders**（2026-09-29 更新）：**pdf 统一走 OCR 产线**；docx / html / xlsx 走代码解析
（结构写在文件里，不该送去识别）；md / txt 族走标准库；图片走 OCR。
csv / 代码 / web 抓取留给后续。

**pdf 为什么不再用 pymupdf 抽正文**：pymupdf 只能拿纯文本，标题层级、表格、图片全没有，
而下游 `markdown-semantic` 切分器的第一级就是 markdown 标题、图片描述要的是正文里的图片标记 ——
两者在纯文本上都是空转。实测同一份数字版 PDF：pymupdf 路径 **0/20** 个块带标题路径、
0 个图片标记；OCR 路径 **36/36**、1 个图片标记。`parser: pymupdf` 保留为**有损降级开关**。

**四条路的出口必须是同一种格式（markdown）**，否则切分器要按来源分支 ——
docx 的 `Heading 1`、html 的 `<h1>`、xlsx 的表名、OCR 的 `relevelTitles` 全都落到 `#`。
唯一的例外是表格：OCR 那条路保留 HTML（产线会给合并单元格，竖线表表达不了），
代码解析的三条路用竖线表（本来就解析不出 span，竖线表更省 token）。

**已知问题：dots.ocr 会漏标题、层级也会给浅**（使用者 2026-10-08 报；**漏标题这半已修**，
层级那半按下面说的留给人工）。
现象的严重性不在"层级标错"，而在**那一行压根不在 `outline.tsv` 里**：清单要求与正文的
标题行一一对应、而且只能改层级不能增删行，所以**事后没法在页面上把它补成标题**
（现在的「标为正文」是反向操作）。后果是那一节不会成为切分边界、也不进标题路径。

**已复现的例子**（使用者给的文件：`第一章 Apache Flink 概述.pdf`，产物
`data/stages/3481c48bcb813900`；`1.3 Flink核心特性` 那一节，正文第 173–194 行）：

```
L173  ## 1.3 Flink核心特性            ← 标题（正确）
L177  1. 批流一体化                   ← **正文**（dots 没标）
L181  2. 同时支持高吞吐、低延迟、高性能  ← **正文**（没标）
L185  3. 支持事件时间(Event Time)概念   ← **正文**（没标）
L189  ## 4. 支持有状态计算             ← 标了，但层级是 2（子项应为 3）
      ## 5. … / ## 6. … / … / ## 9. 多层级API
```

两个毛病叠在一起：**同一种排版、同一个编号序列，只从第 4 条开始标**（前三条漏了），
而且**层级整体偏浅**（`##` 而不是 `###`）。切分结果因此是：`1.3` 的子节编号从 `1-3-4`
起跳（`1-3-1`…`1-3-3` 不存在），前三段内容并进了 `1.3` 自己的正文。

**另外三条判据在我这边都是阴性**（记下来免得重复走）：

> 补一句（2026-10-08 当天）：使用者报的「三条显示成 `1. / 1. / 1.`」**不是这条问题的表现**，
> 是我们自己渲染器的有序列表编号被重置（`<ol>` 被正文打断后浏览器各自从 1 数起），
> 已单独修掉（`markdown.ts` 带上首项编号）。这条已知问题说的是**它们压根不是标题**。

| 判据 | 结果 |
|---|---|
| 「像编号标题却没被标成标题」的正文行 | ⚠️ **判据其实是对的，是我当时判读错了**：扫出来的 `1. 批流一体化` 这些正是漏标的标题，我却当成"编号列表项"排除了 |
| **编号序列缺口**（有 1.2 与 1.4 就该有 1.3） | ❌ 5 份产物上 0 处 —— 因为漏的是**同一节内部的子项**（缺 1.3.1–1.3.3），跨节编号并没有缺口 |
| 「`#` 后面没空格」（我们的正则是 `^(#{1,6})\s+`，`##2.1 xx` 会被当正文） | ❌ 只有 4 行，全是代码块里的 shell 注释（`#!/bin/bash`、`#Step 1`） |

**丢在哪一步**：这两条都在 **dots 那一侧**（它输出的 markdown 里就没有 `#`、层级也给浅了），
不是我们的 loader 没认出来（我们的正则会认 `## …`，只是 dots 没给）。所以修法只能从解析这一端
想办法。

**修法：解析后加一层「编号序列兜底」**（2026-10-08 落地，
`headings.promote_numbered_headings`，接在 `loaders.base.pages_to_text` **拼完整份之后**）。
判据是结构性的两条，**都要**成立：

1. 一串从 `1.` 开始、编号连续的裸编号行，**紧后面那个标题**的编号正好接在它们后面
   （`1./2./3.` 后面是 `4.`）；
2. 那个标题**自己也在一个同级、编号连续的标题串里**（它后面同级的标题编号接着走）。

第 2 条不是想当然，是拿**全库 19 份解析稿**量出来的：只有第 1 条时 Kafka 那份有 6 行被误伤
——`1. sudo /usr/sbin/kadmin.local …` 后面碰巧挨着一个 `## 3. Make sure all hosts…`，
可它们是 `## 2.` 底下的**操作步骤**，后面并没有 `## 4.`（下一节跳回
`## 2. Configuring Kafka Brokers`）。两条都加上：全库只动 **6 行**、全在两章 Flink 里，
Kafka 快速开始那 318 行命令一个不碰。

**补出来的行按兄弟标题声明的级别补**，所以还是 `##`（与 4.–9. 一致）—— 「层级给浅」那半
仍然留给人工在清单里翻，第二列「编号推断」正好把这层意思标着（`2` vs `3`）。没顺手也纠掉它，
是因为「声明级别 vs 编号推断谁优先」是全库性的策略改动，会连带改掉别的文档的初值。

兜底**只往行首插 `#` 和空格，一个字不改、一行不增删** —— 这是它能放在拼完之后的前提：逐行
打点按「每页占几行」重算，页码/行号映射照旧精确。用例
`test_a_promoted_heading_keeps_the_page_mapping` 钉着这条，因为 `1./2./3.` 与 `4.`
实测**跨页**（第 9 页 / 第 10 页）—— 逐页做根本看不见「编号接得上」这个判据。

**要吃到这版得重解析**：解析产物按 loader 版本认新旧（`LOADER_VERSIONS`）。这一版**没有**
抬 `pdf` 的版本号 —— 抬了会把 13 份「已核对: 是」的清单一起作废（重解析会重写 `outline.tsv`），
所以哪些文档值得重跑由使用者定。新上传的 PDF 直接就是新行为。

**Parsers**：扫描件 OCR 兜底已实现；表格抽取、代码块保留留给后续
**Chunkers**：

| 策略 | 用途 |
|---|---|
| recursive | 默认，`RecursiveCharacterTextSplitter` |
| markdown-aware | md/html，按标题层级切，保留结构元数据 |
| **markdown-semantic** | **默认**。header → 递归（只要 `\n\n`/`\n`/`。`）→ 语义兜底 → 按大小硬切。见 §7.2 |
| **table-rows** | **一个非空行 = 一个 chunk**。表格数据专用，见 §7.3 |
| parent-child | 小块检索、父块生成（small-to-big） |
| semantic | 按语义断点切，需 embedding，可选 |

### 7.6 标题层级一律走人工核对（2026-10-02；之前只对 PDF）

**层级是切分的唯一依据，而它没有一条可靠的自动来源。** PDF 走 OCR 之后只能靠猜，实测：

| 信号 | 实测结论 |
|---|---|
| 版面（标题块的 bbox 高度） | ❌ 与真实层级**完全重叠**（已知第 3 层的比值 0.30–0.52、第 4 层 0.00–0.49、第 5 层 0.00–0.92）；块高是**多行块的整块高度**，折一行就翻倍 |
| 版面标签（label） | ❌ 38 个标题全是 `paragraph_title`；`header` 还误标了 `env.execute();` 这种代码块 |
| 标题里的编号 | ⚠️ 能救一部分（`2.1`/`2.2.1`），但写法千变万化（`一、`/`(1)`/`（一）`/`①`），而且 **Word 的编号是渲染时生成的**，段落文本里根本没有 |
| 目录页 | ⚠️ 强信号，但不是每份文档都有 |

docx/html/md 看着「结构写在文件里」，其实也不可靠：`#` 的个数只是作者的排版习惯，
同一层内容有时用 `##` 有时用 `####`（Kafka 官方那份 HTML 里 `Tags:` 是 h5，
而它其实是小节标题）。所以 2026-10-02 起**所有格式统一走同一道流程**：

```
解析 → 出一份清单 → 人核对（改层级、选最大层级、喊「已核对」）→ 才允许切分
```

```
data/stages/<doc_id>/outline.tsv

# 最大层级: 3          ← 这份文档切到第几级为止（每份可以不同）
# 已核对: 否            ← 人核对完改成「是」；没改之前 --stage chunk 会跳过这份文档
层级	编号推断	标题
1	2	2第二章 Apache Flink 快速入门
2	3	2.1 Flink 开发环境准备
3	2	Flink版本
```

三处各管一件事：

| 写什么 | 管什么 |
|---|---|
| 第一列 `层级` | **只有它算数**：这行是几级标题。第二列是按作者编号推断的结果，供对照 |
| `# 最大层级: N` | 这份文档**切到第几级**：比 N 深的行不单独成节、不进标题路径，正文并进上一个小节（§7.2）。想表达「这是子项、别单独切一块」就把那一行写深一级 |
| `# 已核对:` | **切分的开关**。没改成「是」之前，这份文档进 `pending`，CLI 打出「待人工核对」并指路；不报错、也不硬切 |

**这一步是「人工修正」，不是「人工重来」**：第一列的初值由程序按编号推断
（`outline.initial_levels`），人只改不对的那几行。自动推断再准也要留这道核对 ——
但核对几十行和从零填几十行是两件事，这决定了这条路能不能长期用下去。

两个实现要点：

1. **不改正文**。清单只改「层级」，而哪些行是标题一行都没变 —— 所以按行号切 section，
   正文分片仍是原文的**逐字子串**，页码/行号映射（`positions`）照旧精确。实测：把
   `Flink版本` 从 3 级改成 2 级，路径立刻变成 `第二章 -> Flink版本`，而所有块的
   `p1 L5` / `p3 L3` 一个没动。
2. **清单进指纹**。注册表里记的是「切法指纹 + **这一份的清单指纹**」：改一份清单只让
   那一份重灌（`plan_incremental` 的 `pipeline` 参数支持按文件算）。
   跳级（`1 → 3`）会被压平成 `1 → 2` 并在 CLI 里提示，保证路径仍然逐级下探。
   指纹取的是**语义**（层级列表 + 最大层级），不是文件原文：清单里有六七行是给人看的
   说明，改一句措辞不该让整份文档重切、重新 embedding。

清单一改，切分产物自动失效重切（`chunks.json` 里记了 `outline_hash`）；
清单条目数与正文标题数对不上时**只跳过那一份**并说明原因，不猜；
`# 最大层级:` 写了个不可能的数（0、7、abc）直接报错并指出行号 —— 那是明确的笔误，
猜一个数字继续跑等于悄悄换了切法。

已切好、且没变的语料**不需要**为了跑一次 ingest 重新核对一遍：核对只在「这份文档要重新
切」的时候才拦人。

**勾了「已核对」提交 = 直接往下切分并入库**（2026-10-10 定，页面第三步）：`PUT
/documents/{id}/outline` 在 `confirmed=true` 且 `api.index_after_review`（默认开）时，
顺手排一个「切分 + 入库」任务并把 job 放进响应（页面据此切到「拆分入库中…」）。
理由：四步流程里最后那次点击**没有新决策** —— 切分只认核对过的清单，而它刚被核对完；
上一版要求人先「提交核对」再点「拆分并入库」，中间那个中间态（保存成功、流程没动）
正是最容易被误判成"没生效"的地方。两点约束：只有**真的勾了**才触发（切分要给每张图
问 VLM、入库要算 embedding，都按量计费）；同一份文档已经有 index 任务在排队/在跑时
**复用那一个**，不重复排（连点两次不该跑两遍）。想回到手动点，把
`api.index_after_review` 设成 false。

### 7.5 三段流水线与中间产物（2026-09-30）

一条 `rag ingest` 以前是一口气跑完的，中间什么都不留。现在拆成**三段**，产物落本地：

| 段 | 做什么 | 要不要 Milvus | 产物 |
|---|---|---|---|
| 1 `--stage parse` | loader + OCR → `Document` | ❌ | `document.json` / `document.md` |
| 2 `--stage chunk` | 切分 + locate + 图片描述 → `Chunk` | ❌ | `chunks.json` / `chunks.md` |
| 3 `--stage index` | embedding + 写库 + 更新注册表 | ✅ | —（写进 Milvus） |

产物在 `ingest.stages.dir`（默认 `./data/stages/<doc_id>/`）。`.json` 给下一段读，
`.md` 给**人**看：每块一行标题写着「序号 · 模态 · 第几页第几行 · 章节路径 · 编号 ·
引用了哪几张图 · 多少字」，下面是块正文 —— 解析读歪了、标题层级错了、块切得太碎，
打开就能看见，不用连库反查。

**落点是 `local` 还是 `minio` 由 `ingest.stages.backend` 定，调用方不该看出区别**
（2026-10-10 补）：写产物一律走**键**（`_write(_key(doc_id, name), …)`），
`path_of()` 只用来读，而且**"这份产物不存在"必须给 `None` —— 两个后端一个说法**。
这两条都是踩出来的：本地盘的 `local_path()` 从来不抛（给一个不存在的路径），
桶后端却要先 HEAD 一次，于是「拿 `path_of()` 当写入口」在本地通、切到 MinIO 就崩，
而详情页在"还没有 `outline.tsv`"时必 500。同一类"后端语义不一致"以后按这条规矩收。

**四个落点的后端由一个统一开关收口**（2026-10-10 加，与 `persistence.backend` 同一个
形状）：`storage.backend`（base.yaml 里 `${RAG_STORAGE_BACKEND:-local}`，值放 `.env`）
管上传件 / 中间产物 / 原始资产 / 会话图片四个域；那四个域各自的 `backend` 默认写 `auto`
（= 听统一开关），想单独钉死某一个域（比如产物进桶、上传件留本地盘）就写回 `local` /
`minio`。优先级是「域显式写的 > `storage.backend` > `local`」，判据的唯一实现在
`rag/storage.py` 的 `resolve_storage_backend()`（`build_blob_store()` 开头调它）。
**默认 `local` 是刻意的**：单机部署不带 MinIO、不改配置，行为与从前逐字节一致。

**桶那边读穿透缓存的命中判据是 ETag，不是文件大小**（2026-10-10 改）：
本地副本旁边放一个 `<文件>.etag`，HEAD 回来的 ETag 一变就重新拉。为什么不能用大小：
`outline.tsv` 是**人就地编辑**的产物，`已核对: 否 → 是`、层级 `2 → 3` 这类改动
**字节数完全相同** —— 按大小比，缓存永远算命中，于是「人工核对结果不生效」
（实测：桶里 `confirmed=True`、应用读到 `False`）。原来的理由"键里带内容哈希
（上传件名、`doc_id`），所以大小一致≈内容一致"只对**上传件**成立，对这份清单不成立。

**为什么值这两个理由（用户 2026-09-30 提的，都成立）：**

1. **人工可核对**：中间结果不落盘时，「切得对不对」只能靠最终检索效果倒推。
2. **解耦时间与依赖**：只有第三段碰 Milvus。库挂了、或想等机器空闲再灌，
   前两段先跑完放着即可 —— 实测 2026-09-30 那天 Milvus 停了两次，
   解析与切分（最贵的 OCR + VLM）一次都没白做。

**新鲜度是这套东西的关键**：文件一改、解析设置一改、切法一改，旧产物就不能再用。
每份产物都记 `file_hash` + `parse_key`（阶段1）或 `pipeline_version`（阶段2），
读取时逐个比对，对不上就当没缓存 —— 宁可重跑一遍，也绝不拿对不上的中间结果去写库。
`parse_key` 与 `pipeline_version` **故意分开**：换 embedding 模型不该让解析产物失效
（那要重新 OCR，最慢也最可能花钱）。

分两处实现：`ingest/pipeline.py` 是阶段2 的核心（纯函数，给 Document 还 Chunk，
不碰库也不碰磁盘），`ingest/stages.py` 管产物的读写与新鲜度；CLI 只负责按 `--stage` 调它们。
`--stage all` 与分开跑的三段走的是**同一组函数**，不会各走各的。

### 7.4 图片描述（VLM，2026-09-30）

**图片块里只有 `![image](asset:…)` 这个标记 —— 检索到它，模型什么也看不到。**
实测过一次：问「两种方式的对比」，答案写的是「具体对比内容以**图片**形式呈现（img003），
文字部分未展开说明」。它没有编，但也答不了 —— 那张表整个是图。所以要给每张图一条
**自己的记录**（`modality=image`），把图变成可检索的文本。

位置在**切块与 locate 之后**（`ingest/images.py::describe_images`），因为它要的东西
都在那一步之后才有：

```
loader → chunker → _locate（页码/行号）→ describe_images → Milvus
```

**块正文的布局与正文块一致**（标题贴开头，见 §7.2），只是「内容」换成了图片描述：

```
<标题1> -> <标题2>
（空行）
<图片描述>
![image](asset:<doc_id>/p0003_01.jpg)
```

- **上下文**：把「上边一段 + 下边一段」一起喂给 VLM（段落粒度，`assets.context_for_marker`，
  会跳过相邻图片的标记）。光有图不够 —— 一张架构图脱离「上面那段在讲什么」说不清。
  各截 `ingest.images.context_chars`（默认 600），**留靠近图片的那一头**。
- **位置**：`page` / `line` 取**标记自己**的偏移（图在第几行就是第几行），
  `heading_path` / `heading_numbers` 继承含这张图的那一块 —— 标题层级只有切块那一步知道。
  实测：Flink 第一章那两张图分别落在 **第 1 页第 15 行**与**第 2 页第 1 行**，
  而它们**所在块都是从第 1 页第 5 行开始的**（一节跨页，两个标记落在同一个块里）——
  若图省事继承块首，第二张图会被报成「第 1 页第 5 行」，**页都是错的**。
- **模型**：`providers.vision`，默认 DashScope 的 `qwen-vl-max`，走 OpenAI 兼容通道
  （与 `providers/chat.py` 同一套三件套约定，变量名带 VL 后缀：`DASHSCOPE_VL_API_KEY`
  / `_BASE_URL` / `_MODEL_NAME`；key 留空回退到 `DASHSCOPE_API_KEY`）。
  实现是 `providers/vision.py` 里一个几十行的 httpx 客户端，不引 LangChain 的 chat 模型 ——
  调用点只有一处、不需要流式与工具调用，多一层封装只会让「请求长什么样」更难看清。
- **默认关闭 + 两道成本闸门**：`ingest.images.describe: false`（每张图一次调用、按量计费）；
  `max_images`（0 = 不限）用来「先跑几张看效果再全量」。
- **缓存**：键 = `sha1(图片字节)[:16] + 模型@提示词版本`，落 `ingest.images.cache_path`。
  改一段正文不该让几十张图全部重问 VLM。提示词版本是 `config/schema.py` 里的
  `VISION_PROMPT_VERSION` —— 改了提示词就抬一位，缓存与 `pipeline_version` 同时失效。
- **单张图失败不牵连整份文档**：那一张记进 `skipped`（带原因、CLI 逐条打印），
  其余照旧。十几张图里挂一张，为它丢掉另外十几张是更坏的结果。
- `pipeline_version` 里带上了这个开关（§4.3）：文件一个字节没改、只是打开了开关时，
  增量必须重灌，否则一张图都不会被描述，而库里「看起来是新的」。
- **上下文怎么用：模型判断、代码排版**（提示词 v2，2026-10-01）。提示词要求模型
  **第一行只写一个词**（`相关` / `不相关`），判断「图上/下方的段落是否在讲这张图」；
  相关就在描述末尾加一句以「结合上文：」开头的话，把上文那件事概括进来（检索要能命中），
  不相关就一个字都不提。**那一行判断不吃进描述**，由 `clean_description()` 摘掉；
  判为不相关时，代码再把「结合上文：…」那一句切掉。
  为什么要代码来收这一刀：实测 qwen-vl-max 判了「不相关」也照样补一句（有一次把
  pandas 的操作步骤原样抄进了一张 Flink 架构图的描述里）—— 那些词一旦进库就是
  **假的检索入口**。模型只做它做得动的判断，排版与取舍交给代码。
  切完会空的话**留着不切**：一张图没有任何描述，比多一句上文更坏。

> 图片块的向量怎么算有**两种方式**（描述转文本 / 图+文一起编码），见 §8.3。
> 描述文字**两种方式都要** —— 后者是把描述与图片本体放进同一个内容项，不是不要描述。

**实测（2026-09-30，Flink 第一章 + qwen-vl-max，限 2 张）**：OCR 产线报 500 走了云端兜底，
11 页 → 34 块（含 2 条图片描述）。第 2 页那张图被读成「橙色管道状结构…虚线分为三段，
每段标注 `bounded stream`…下方时间轴 `past`/`now`/`future`…另一条箭头标注 `unbounded stream`」
—— 图里的英文标签一个不差。

**不编造这一点单独验过一次**：拿一张与 Flink 无关的插画（正文站的吉祥物图），故意配上一段
「Flink 由 JobManager 与 TaskManager 组成」的上下文。它照实描述了插画（含读出的
`Status: Optimized` / `Build Success` 等界面文字），并**指出这张图与背景不符** ——
没有顺着上下文编一个架构图出来。上下文只影响「这图在讲哪件事」的判断，不会被当成图中内容。

### 7.3 表格按行切（`table-rows`，2026-09-29）

**表格和散文是两种形状。** 散文按语义切 —— 一段可能被切成几块，几段也可能并成一块；
而表格的**一行本身就是最小的完整语义单元**：两行并成一块会把两条记录混在一起，
一行被切两半则两边都不完整。所以表格不套用四级切分。

两条配套的设计：

1. **loader 声明切法**，不由全局配置决定。一份语料里往往散文和表格混着（md 和 xlsx
   一起灌），一个全局策略服务不了两边 —— `Document.metadata["chunk_strategy"]` 写下切法，
   `prepare()` 按文档分组分发。
2. **每条记录必须自描述**。`| 甲公司 | 512 |` 不知道列名就检索不到，所以表名与列名
   焊进每条记录 —— `客户清单：名称=甲公司，数量=512`。代价是每行重复一遍列名，但表格的
   块本来就不该按大小压。

`table-rows` **不看 `size`**：一条记录再长也是它自己。

### 7.2 四级切分（`markdown-semantic`，2026-09-29）

顺序刻意从便宜排到贵，每一级只处理上一级解决不了的：

| 级别 | 手段 | 解决什么 |
|---|---|---|
| 1 | markdown 标题 h1/h2/h3 | 最可靠的语义边界，顺带给出标题路径 |
| 2 | 递归，分隔符**只有** `\n\n` / `\n` / `。` | 小节内的段落与句子边界 |
| 3 | 语义：句向量 + 相邻余弦距离的 percentile 断点 | 递归切不动的（只有逗号、没有句子边界） |
| 4 | 按字符硬切 | 语义断点**不是尺寸控制**；表格 HTML、无标点长串 |

三条设计要点，都是实测撞出来的：

1. **分隔符只留前三个。** 逗号/分号/空格切开会把句子拦腰截断，碎片化比「块大一点」更伤检索。
   代价是去掉末尾的 `""` 之后递归那级**不再保证不超限**，所以第 3、4 级不是可选项。
2. **语义那一级必须有两条护栏。** `min_size`：裸 percentile 会切出 3 个字的块（实测撞到过
   `'但是，'`）；`size` 上限：一段语义自洽的长文本它就是不断，没有上限就保证不了块不超窗口
   （LangChain `SemanticChunker` 正是栽在这里，切完最大块到阈值的近两倍）。
3. **它在这份语料上是零调用。** 8 页真实 PDF 走 OCR、size 压到 250，语义那级仍然是
   0 批 / 0 句 —— 递归用 `。` 和换行就全解决了。贵的那一步只处理残留，这是设计意图，
   也意味着它平时不花钱。
4. **只切到三级**（2026-10-01）：`headings.MAX_LEVEL = 3`。比它深的标题**不是边界** ——
   不单独成节、不进标题路径，正文并进上一个小节。这条对**人工对齐的清单**尤其重要：
   在 `outline.tsv` 里把某一行写成更深一级，就是「这是小节里的子项、别单独切一块」的
   表达方式（Flink 第一章那两个子项、第二章的列表步骤都是这么标的），**不用删行** ——
   删行会让清单条目数与正文标题数对不上。同一个数在三处生效：`chunkers.base.HEADERS`
   （docx/html/md 那条 markdown 路径）、`outline.sections`（PDF 与人工清单那条路）、
   以及 `pipeline_version` 的指纹（改了它会自动重切已灌语料）。

**元数据增强**：来源路径、页码、标题层级、时间、文档类型、权限标签 —— 后三个都要参与 filter 和引用展示。

**已实现的部分：**

- loader 已实现六条路：md/txt 族（`.md .markdown .txt .log .rst .csv`，utf-8 → gb18030 试解码）、
  **pdf（统一走 OCR 产线）**、图片（`.png .jpg .jpeg .tif .tiff .bmp`，走 OCR）、
  **docx（python-docx）**、**html/htm（bs4）**、**xlsx/xlsm（openpyxl）**。csv / 代码 / web 抓取留给后续。
  docx/html/xlsx 这条线的意义是「结构明明白白写在文件里，不该送去识别」——
  docx 的 `Heading 1`、html 的 `<h1>`、xlsx 的表名各自落到 markdown 的 `#`

  HTML 那条路有四条**实测撞出来**的规则（`loaders/html.py`）：兜底容器 ——
  `div`/`section`/`article` 在内部没有块级元素时也算一段，否则现代前端那种
  「只有 div 没有 p」的段落会整段丢掉；`<pre>` 转围栏代码块，否则命令行全没；
  跳过嵌套 `li`，否则外层已吞了子项文字、子项再出一遍就是同一段进库三遍；
  文字提取**不用** `get_text(" ")` —— 它会在行内标签边界插空格，把「不要把」
  拆成「不要 把」，中文就检索不到了

  2026-09-29 拿两篇真实网页（Confluent 文档，用浏览器保存下来的）验证后又补了三条：

  - **正文区提取**：优先 `<main>`、其次 `<article>`、都没有才退回 `<body>`。
    真实文档页的页头/搜索框/面包屑/侧栏常占去**一半**文字量（实测 2,852 字符的正文
    被整页扫成 5,265）；且顺序必须是「先选内容区、再在区内清噪声」，
    反过来的话套着正文的 `<form>` 会把正文一起干掉
  - **跳过 `<li>` 里的 `<p>`**：li 的文字已经含了它，否则同一段进库两遍
  - **`cleaning.py::strip_invisible()` 清不可见字符**：零宽字符（U+200B）与
    **私有使用区**（图标字体的字形码点，U+E000–U+F8FF 等）。Confluent 每个标题尾部
    都挂着一个 U+F0C1（Font Awesome 的 `fa-link`），会跟着进标题路径让精确匹配失效。
    OCR 那条路也接上了

  另外 `discover()` 会跳过浏览器「保存网页」生成的 `<页面名>_files/` 目录 ——
  不跳的话，一份文档页会带出 4 张图标 PNG（还会被当成图片送去 OCR）与 4 个跟踪 HTML。

  **docx 那条路补充（2026-09-29 拿一份 10MB 真实 Word 验证）**：

  - **`p.style` 可能是 `None`** —— 那份文档 890 个段落全是（整篇没用样式），
    直接取 `.name` 会把整份文档变成 skipped。要 `getattr(paragraph.style, "name", "")`
  - **图片用与 PDF 同一套机制**抽取（本体进资产目录、正文留 `![image](asset:…)` 原位标记），
    那份文档有 **45 张内容图**，不取就是静默消失
  - **无样式文档的标题靠直接格式推断**：整段加粗 **且** 比正文大 **且** ≤60 字，
    按字号排名给层级。正文基准字号取「按字符数最多」的那个。那条 60 字上限来自实测 ——
    正文里最长的加粗段有 **4307 字**，那是强调不是标题。**作者用了样式就信样式，完全不猜**
- chunker 有 `recursive`（分隔符 `\n\n → \n → 。！？；， → 空格`）、`markdown`（按 h1/h2/h3 切）、
  以及默认的 **`markdown-semantic`**（四级，见 §7.2）。三者的标题路径都写进 chunk metadata。
- **OCR 产出的 markdown 入库前过一次 `ingest/cleaning.py`**：产线的表格是内联 HTML，
  每个格子上都挂着完整的内联样式（占约 80% 字符），剥掉版式属性、只在 `<table>` / `<tr>`
  边界补换行。不补换行的话递归切块器没有断点可下刀，会从**标签中间**切开表格 ——
  实测出现过整块只剩 `style='text-align: center; …'` 而没有任何单元格值的 chunk。
  PDF 与图片两条 OCR 路径共用 `loaders/base.py::pages_to_text()` 这一个入口
- **图片在同一个入口里就位**：`ingest/assets.py` 把产线内联在正文里的图片（`<img
  src="imgs/…">`，位置就是原图在文中的位置）取出来 —— 本体写进 `ingest.assets.dir`，
  正文里换成 `![image](asset:<doc_id>/p0003_01.jpg)` 短标记。base64 绝不留在正文里：
  一张图几十万字符，会把 chunk 预算整个吃掉。切块后 `context_for_marker()` 按标记给出
  「上边一段 / 下边一段」（段落粒度、会跳过相邻图片的标记），图片描述靠的就是它
- **`doc_id = sha1(相对路径)[:16]`**，不是内容哈希。内容哈希会让「改一个字」换一个 doc_id，
  旧块留在库里成为永远检索得到的幽灵；路径哈希则让重新灌库原地覆盖。
- `chunk_id = f"{doc_id}:{序号:04d}"`。

### 7.1 扫描件：PaddleOCR-VL 产线服务（M2.5 接入，ADR 17/18）

pymupdf 只能抽**文字层**。纯扫描件抽出来是空的，混合型 PDF 里那几页扫描页会静默
检索不到，多栏/表格/公式的顺序也是乱的。所以扫描件走 OCR。

**打的不是 8118。** GPU 主机上那个 8118 是 vLLM 托管的 `PaddleOCR-VL-1.6-0.9B`，
**只是产线的下游 VLM**，只会「看图说话」，不给版面分析。完整的「文档→markdown」是官方那个
产线服务 `paddleocr-vl-api`（`POST /layout-parsing`，默认 **8080**）：它切页、做版面分析、
定阅读顺序、拼 markdown，再逐块回调 8118。地址进 `.env` 的 `PADDLEOCR_BASE_URL`，
**留空 = 不启用 OCR** —— 客户端就是 `None`，整条链路上一个请求都不发，
没部署 GPU 服务的机器照常跑纯 pymupdf。

**路由由 `ingest.pdf.parser` 决定，走的是 OCR 工厂**（`ingest/ocr_client.py`，`auto` 已移除）。
工厂按名字从注册表里建 client；**三家 client 的出口是同一种格式**（`list[OcrPage]`：一页一份
markdown + 图片），所以换一家只改配置，loader 及其下游一行不动：

| 值 | 谁 | 特点 |
|---|---|---|
| `mineru` | MinerU 远程 API（`ingest.mineru`，默认 127.0.0.1:6006） | 「上传 → 建任务 → 轮询 → 取产物」；`middle_json` 给逐页逐块 + 归一化 bbox（**定位最规整**）；**不给标题层级**、慢（约 27 秒/页，大文档用 `page_range` 圈范围） |
| `dots` | dots.ocr（`ingest.dots`，vLLM 6006） | 端到端版面模型，`Section-header` 块直接给出 `#`/`##`/`###` —— **标题层级只有它给得对**；图裁剪后内联成 data URI |
| `paddleocr` | PaddleOCR 产线（`ingest.ocr`，8080） | 整份一次调用；`relevelTitles` 只按版面标签猜层级，实测在教材 PDF 上不可用。**规范名**：老配置里写的 `ocr` 仍然认（`config.schema.normalize_pdf_parser` 归一化），值在进配置时就统一成 `paddleocr` —— 免得指纹/界面出现两种说法 |
| `pymupdf` | 本机文字层（不是 OCR） | 有损降级：标题/表格/图片全丢，只在 OCR 都挂了或做对照时用 |

**加一家新的 OCR** = 写一个 client（实现 `parse / base_url / name` 三样）+
`@register_ocr("名字")`，然后改配置 —— 工厂与上层都不用动。没配地址的引擎（如
`ingest.ocr.base_url` 为空）返回 `None`：这条链路上一个请求都不发。

**为什么不再探测**：探测存在的前提是「数字版走 pymupdf 更划算」，但实测这个前提不成立 ——
pymupdf 拿到的是纯文本，标题层级、表格、图片一个都没有，下游的 markdown 切分与图片描述
都会空转（同一份数字版 PDF：pymupdf 0/20 个块带标题路径，OCR 36/36）。
整份送还有一个额外好处：跨页表格合并与标题重建本来就要求产线一次看到全部页。

加密 PDF 在送产线之前用 pymupdf 拦一道（`needs_pass`），否则产线只会回一个看不出所以然的错。

**产线挂了怎么办：切云端兜底**（2026-09-29）。`ingest.ocr.cloud_fallback` 打开后，
`build_ocr` 会包一层 `FallbackOcr`：本地产线报**不可用**（连不上 / 5xx / 服务端
errorCode ≥ 500）才走 PaddleOCR 云端 API（`https://paddleocr.aistudio-app.com`，
`paddleocr` 自带的 SDK）。

**产线的下游 VLM 在云上，这条转发链路是最脆的一环**（2026-09-30 排查）。本机 8080 上的
产线只做版面分析，识别全部转给 `http://127.0.0.1:6006/v1` —— 而 **6006 不是本机进程**，
它由 `AutoDL.exe`（AutoDL-SSH-Tools）从云 GPU 实例转发过来（`/v1/models` 里
`root=/root/.paddlex/…` 这个 Linux 路径就是线索；本机 python 进程里没有它）。

实测这条链路的失败率：**产线 access log 累计 34 次 200 / 13 次 500（28%）**，
而空闲时探活 `/v1/models` 12/12 成功、中位 51ms —— 说明不是「连不上」，
而是**真实识别请求（大图 + 长输出、并且产线会并发下发多个块）在这条转发上被切断**。
产线日志里两种症状都指向它：`Exception from the 'vlm' worker: Connection error.`（16 次）
与 `Error code: 502`（10 次，转发/网关层回的）。

对我们的影响：每次失败都会**切云端兜底**（本地一次 500 = 一次按量计费的云端调用），
以及多等一轮。可做的两件事：① 这条转发换成带保活的持久 SSH 隧道
（`ssh -L 6006:127.0.0.1:6006 -o ServerAliveInterval=30 -o ServerAliveCountMax=3`），
或在 AutoDL 客户端里重连；② 切云端之前**先原地重试本地产线** ——
隧道抖动是瞬时的，本机重试几乎免费，而云端兜底每次都在计费。
**②已实现**（2026-10-05）：`ingest.ocr.retry_primary`（默认 1 次）写在 `OcrClient.parse`
里，所以开不开云端兜底都受益；重试**也会出声**（静默重试会让人以为产线一直很稳）。

三个刻意的取舍：

1. **默认关闭**。云端按量计费，打开是个花钱的决定，不该悄悄发生。
2. **只在 `OcrUnavailable` 上兜底**。请求本身有问题（4xx、报文错）时换一家也一样失败，
   白等一轮还把真正的原因盖掉。为此 `ocr.py` 把「服务不可用」与「请求有问题」分成了
   两种异常 —— 注意 502 + HTML body 也算前者（网关挂掉时返回的正是 HTML）。
3. **切换必须出声**（stderr 打一行，带原因和目的地）。悄悄换到按量计费的云端，
   用户会看着账单猜原因。

开了却没配 `PADDLEOCR_ACCESS_TOKEN` 会**直接报错**，而不是默默退回「没有兜底」——
那等于让人以为有保险。云端给的是图片 **URL**（本地给 base64），`CloudOcrClient`
会下载成字节，让下游拿到的完全一致。

**2026-09-29 真调验证**：配上 token 后实测通（3 页 / 14.5s，拿回真 markdown 与真标题）；
**云端的 `markdown_images` 键与正文 `<img src>` 逐字相同**，和本地那套是同一个约定，
不需要额外适配。把产线指到死端口做端到端：打印警告 → 切云端 → 整条 ingest 跑完，
图片照常落盘、正文里照常有标记。

**同日晚些时候，兜底在真实场景派上了用场**：`rag ingest ./data/docs` 时本地产线报 500，
`FallbackOcr` 自动切到云端，灌库照常完成（6 篇 → 202 个 chunk）。

踩到一个坑值得记：实现时把 import 写成 `from paddleocr._api_client import PaddleOCRClient`，
而那个包**只暴露子模块、不重导出类**（正确路径是 `paddleocr._api_client.client`）。
打桩测试当时也挂在同一个错路径上，两边一起错、测试全绿 —— **只有真调用能抓住这类错**。

**为什么整份，而不是逐页**：只有一次把整份 PDF 交给产线，才拿得到
`mergeTables`（跨页表格合并）与 `relevelTitles`（把版面标题块重建成 `#/##/###`），
而重建出来的标题正好喂给现有的 markdown chunker（见下方「切块策略」）。
逐页拼装拿不到这两样；官方文档那条两阶段路（`/layout-parsing` → 攒 `pages` →
`/restructure-pages`）则会把多页拍平成一份 markdown，丢掉页归属。我们走的是
「一次调用 + `restructurePages: true`」，拼装自己做（按序 `markdown.text` 用 `\n\n` 连接）——
合并后的表格与页归属都能留下。注意 `prunedResult` 里的 `page_index` 被服务端剥掉了，
**页码只能从 `layoutParsingResults[]` 的数组下标推**。

**`auto` 撞上扫描页但没配地址时是报错，不是降级。** 混在里面的扫描页若被
pymupdf 悄悄跳过，用户会以为整份都进库了 —— 与 skipped 机制「不静默跳过」的立场一致。
报错信息给出两条出路：去 `.env` 配地址，或显式把 `parser` 改成 `pymupdf` 接受部分解析。

**落在 loader 层，不在 Parser 协议上。** `Parser.parse(doc: Document)` 拿不到原始文件，
而「整份交给产线」需要文件本身，所以 OCR 是 loader 的分支（`loaders/pdf.py`、`loaders/image.py`），
`Parser` 协议继续空着。

**切块策略**：`relevelTitles` 的收益要落到块上，前提是 **`indexing.chunk.strategy: markdown`**。
默认值仍是 `recursive` —— 换成 `markdown` 会改变现有语料的分块边界，那是另一个决定，
不该混在接入层改动里。语料以文档型 PDF 为主时建议改过去。
（`MarkdownHeaderTextSplitter` 对纯文本输入也安全：切不出标题就是一整段，再由 overflow splitter 按 size 切。）

---

## 8. 索引层：Milvus

> **字段设计的当前状态见 [milvus-schema-design.md](milvus-schema-design.md)（设计稿，未实现）。**
> 下面这段是 M0/M1 的既有实现；设计稿要把图片路径、标题路径、页码/行号补成显式字段，
> 并加上文档级指纹与增量更新。

**存储层是多模态的第二落点。** 文本块的检索索引在 Milvus（就是下面这套 schema）；原始资产
（页面图像 / 内嵌图 / 表格片段）需要另一处留存，供引用展示与回查 —— **尚未落地**，见 §21。
注意 `deploy/docker-compose.yml` 里的 MinIO 是 Milvus 自己的对象存储依赖（存 segment / 日志），
不是我们的资产仓；真要复用它，得先把归属与生命周期讲清楚。

Milvus 与 Qdrant 最大的操作差异：**schema 必须显式定义，没有自动建表**。

```python
from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker

# —— 建库（幂等）——
schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
schema.add_field("chunk_id",    DataType.VARCHAR, is_primary=True, max_length=64)
schema.add_field("doc_id",      DataType.VARCHAR, max_length=64)
schema.add_field("chunk_index", DataType.INT64)
schema.add_field("source_uri",  DataType.VARCHAR, max_length=1024)
schema.add_field("metadata",    DataType.JSON)      # 块级附加信息，M2 的 filter 从这里取
schema.add_field("text",        DataType.VARCHAR, max_length=8192,
                 enable_analyzer=True, analyzer_params={"type": "chinese"})
schema.add_field("dense",       DataType.FLOAT_VECTOR, dim=settings.dim)
schema.add_field("sparse",      DataType.SPARSE_FLOAT_VECTOR)
# ★ sparse 不来自 embedding 模型：服务端从 text 自动生成
schema.add_function(Function(name="bm25_text", function_type=FunctionType.BM25,
                             input_field_names=["text"], output_field_names=["sparse"]))

index_params = client.prepare_index_params()
index_params.add_index("dense",  index_type="IVF_FLAT",              metric_type="COSINE")
index_params.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
# 带 index_params 建表时 Milvus 会自动 load；对已存在的表必须显式 load
if not client.has_collection(settings.collection):
    client.create_collection(settings.collection, schema=schema, index_params=index_params)
client.load_collection(settings.collection)

# —— hybrid 检索（服务端融合）——
reqs = [
    AnnSearchRequest(data=[dense_q], anns_field="dense",
                     param={"metric_type": "COSINE"}, limit=20),
    AnnSearchRequest(data=[query],   anns_field="sparse",   # ★ 传查询原文，不是稀疏向量
                     param={"metric_type": "BM25"},   limit=20),
]
hits = client.hybrid_search(collection, reqs, ranker=RRFRanker(60), limit=10)
```

注意点：

- `load` / `release` 是 Milvus 特有的生命周期，封装进 `indexing/milvus/client.py`；
  建库逻辑必须幂等（重复执行不报错、不丢数据）
- **dense 与 sparse 的索引参数必须分开给**：一份参数套所有字段套不上
  （前者 `FLOAT_VECTOR` 系 + `COSINE`，后者 `SPARSE_INVERTED_INDEX` + `BM25`）
- **换 embedding 模型必须全量重灌** → collection 名带 embedding 指纹，双写切换
- `enable_dynamic_field=False` 是有意的：字段类型与长度上限决定了能写什么、能怎么过滤，
  靠动态字段兜住一切只会把问题推迟到灌数据那一刻，且症状是「写入失败」而非「schema 缺字段」

### 8.1 M0 实测结论（2026-09-21，Milvus 3.0.2）

在真实 Milvus（`192.168.x.x:19530`，server 3.0.2 / pymilvus 3.0.2）上建临时 collection 验证，
**dense + sparse + `hybrid_search` + `RRFRanker` 全部可用**，即本方案的检索主通道成立：

| 检索 | 结果 |
|---|---|
| dense only | 5 条命中（IVF_FLAT / COSINE，dim 1024） |
| sparse only | 5 条命中，目标文档排 **#1**（SPARSE_INVERTED_INDEX / IP） |
| hybrid (RRFRanker) | 目标排 #1，且结果里**同时含 dense 独有与 sparse 独有的候选** → 融合确实生效 |

若融合是坏的，hybrid 只会复刻 dense 的结果；两路候选都进最终列表，说明服务端融合工作正常。

**实测踩到的三个坑（写 hybrid 通道时会直接撞上）：**

1. **主键按字段名取，不是 `"id"`。** 返回的 `Hit` 主键 key 是 schema 里的字段名（如 `chunk_id`），
   `hit["id"]` 会抛 `KeyError`。`pymilvus` 3.x 的行为。
2. **写完立刻查需要 `flush()` + `consistency_level="Strong"`。** 只 `insert` 不 `flush`，
   或查询用默认 Bounded 一致性，会**静默返回 0 条**（不报错，最容易误判成"功能不支持"）。
   生产上可按延迟要求放宽，但灌库后的自检要用 Strong。
3. **sparse 是精确 term 匹配，不是模糊匹配。** query 的 term 与文档无重叠就是 0 命中。
   验证 sparse 通路时 query 必须复用真实文档的 term —— 第一版测试两边各自随机撒词（词表 3 万），
   命中率为 0，看起来像"Milvus 不支持 sparse"，实际是测试数据构造错误。

### 8.2 M1 实测结论（2026-09-22，Milvus 3.0.2）

M1 把索引层真正写出来时又撞到四条，全部体现在 `indexing/milvus/` 的代码与守卫里：

1. **`VARCHAR` 的 `max_length` 按 UTF-8 字节算，不是字符数。** 写 9000 个汉字报的是
   `length: 27000, max length: 8192`。一个汉字 3 字节，所以 8192 字节只放得下约 2700 汉字
   —— 对默认 `chunk.size=400` 有 6 倍余量，但真要调大 chunk.size 得一起抬这个上限。
   `client.py:_row()` 在本地就按字节拦住并提示调小 `chunk.size`，不让它变成服务端的报错。
2. **带 `index_params` 建表会自动 load**，此时再 `load_collection` 一次**不报错**。
   所以 `ensure_ready()` 无条件调 `load_collection`，省掉一次状态判断。
3. **`upsert` 就地覆盖**（行数不变、正文更新）—— 幂等灌库由此成立。
   但 `chunk_id = doc_id:序号` 意味着**文档改短后多出来的旧序号块不会被覆盖**，
   会成为检索得到、正文却已不存在的孤儿。所以 `rag ingest` 是「先按 doc_id 删旧块，再 upsert」。
4. **`RRFRanker(k)` 只有 k，不支持按腿加权**（`WeightedRanker(*weights)` 才是加权那个）。
   配置里 `dense.weight` / `sparse.weight` 在 `strategy: rrf` 下会被静默忽略 ——
   改成 `rrf` + 非 1.0 权重时直接报错，把「权重没生效」变成一句能看懂的提示（ADR 16）。

另外两条属于 embedding 侧、但直接约束索引层：

- **fastembed 的向量已是 L2 归一化**（实测范数恰好 1.0），所以 COSINE 与 IP 排序等价，
  配置里的 `COSINE` 是有效的；换模型后这个前提要重验。
- **`huggingface.co` 直连不通**，必须同时设 `HF_ENDPOINT=https://hf-mirror.com`
  **和** `HF_HUB_DISABLE_XET=1`（镜像没有 HF 的 Xet `/xet-read-token/` 接口）。
  `huggingface_hub` 只读环境变量、不读我们的 YAML，所以 `providers/embedding.py` 在
  构造模型前把配置值写进 `os.environ`。首次下载 bge-small-zh-v1.5 约 500 秒（~0.85 MB/s）。

### 8.3 两种向量化方式（`providers.embedding.backend`，2026-10-01）

图片进库有两条路，**只改配置、不改代码**，出口都是同一个 `Embeddings`：

| backend | 图片怎么进向量 | 向量空间 | 代价 |
|---|---|---|---|
| `sentence-transformers`（默认） | 先由 VLM 写成**描述文字**，和正文一样编码；**图本身不参与** | 纯文本 | 本地，不花钱（除 VLM 描述） |
| `qwen3-vl` | 图片块把「图片本体 + 描述文字」放进**同一个内容项**，服务端融合成**一个向量** | 图与文**同一个**空间 | 线上按量计费；2560 维；语义兜底也要调它 |

两个关键点：

1. **正文块也走同一个模型。** 「图+文同一空间」只有在两边同源时才成立 —— 图片块送
   `{"text": …, "image": …}`，正文块送 `{"text": …}`，查询送 `{"text": …}`。换成
   多模态 backend 之后，纯文字的检索与以文搜图仍然是同一套向量比对，不需要分支。
   这是「以图搜图 / 以文搜图 / 以文搜文」四向检索的地基。
2. **正文块不附图**，哪怕它的正文里带 `![image](asset:…)` 标记。那张图在同一次切分里
   已经有自己的图片块（§7.4），再附一次是把同一张图编两遍，还会把一块好端端的正文
   向量拽向图片。`ingest/assets.py:image_paths_for` 按「`modality=image` 才带图」构造
   `images`，返回的列表与块**等长、按下标对齐**。

接口与实测（`POST /api/v1/services/embeddings/multimodal-embedding/multimodal-embedding`）：

- **走原生端点，不是 OpenAI 兼容通道** —— 兼容通道对 `qwen3-vl-embedding` 直接 404。
- `input.contents` 是**批量**语义：列表里每一项出一个向量。一个内容项同时带 `text` 与
  `image` → 出 **1 个融合向量**（这就是「图+文一起」）。纯 `{image}` 也支持，留给后面的
  以图搜图。试过 `{items: […]}` 与列表套列表，都是 400。
- 响应 `output.embeddings[]` 带 **`index`**：**按它对序，不假设返回顺序**。
- 图片用 data URI 传。正文里的 `![image](asset:…)` 标记在编码前**剥掉** —— 它是记账用的，
  不是内容（`_IMAGE_MARKER`）。一个内容项只接受一张图（多张只取第一张）。

切过去时**必须同时改两处**，两处都有守卫：

1. `dim` 1024 → **2560**。忘了的话第一次调用就报 —— `embedding.py` 拿返回向量长度与配置比，
   不等就抛 `EmbeddingError`。等 Milvus 报要等向量都花钱编完，那时才说就既贵又晚。
2. **换 collection 名**（Milvus 建好表就改不了维度）。漏了会在 `upsert` 时被
   `client.py` 的维度校验挡住。

中间产物不用手动清：`pipeline_version` 里带着 `embedding.backend/model/dim`，换 backend
会让已切产物自动失效 → 重切重灌（§7.5）。

**以图搜图已接**（2026-10-03）：`embed_query(text, image=None)` 与文档侧完全对称 ——
图和词放进同一个内容项，只给图也行（此时**不带空的 `text`**，就是实测过的 `{"image": …}` 形态）。
纯文本 backend 拿到查询图**报错**，不静默退化成文字查询。纯图的走法见 §9.5。

### 8.4 首次灌库到多模态那条路（2026-10-01）

dev 档已切到 `qwen3-vl-embedding`（2560 维），collection 换成 `rag_dev_qwen3vl_v1`
（旧表 `rag_dev_qwen3_v2` 是纯文本那条路留下的 1024 维；Milvus 建好表就改不了维度，
只能新建）。dev.yaml 里写的是 `${EMBEDDING_BACKEND:-qwen3-vl}` 这种形式，
**`.env` 仍然优先** —— 换机器、换回纯文本都不用动 yaml。

`ingest.images.describe` 在 dev 档打开：多模态那条**必须有描述**，融合向量的文字部分
就是它（关着 `run_chunk` 直接报错，见 §7.4）。

服务端两个批次上限都是真机撞出来的（各撞一次 400）：**纯文本批 ≤ 20**、
**含图的批 ≤ 10**。`providers/embedding.py` 按内容分档切批 —— 图片块只占少数，
不该让正文跟着降到 10 个一批。

### 8.5 文档注册表搬进数据库（2026-10-09）

**背景**：文档页那两条接口（列表与详情）为了回答「这份走到哪一步了 / 入库了没」，
每次都去问 Milvus 的注册表 collection。Milvus 一抖，实测（两张表卡在 `Loading 50%`、
querynode 的 DML channel 不可服务）每次查询要**等 9013ms 才回 503**，文档页就白等 9 秒 ——
而 `GET /health`、列表、详情三条路径**同时都是 9.1–9.5s**，正是同一个等待。

**决定**（使用者 2026-10-09）：「解析的 document 文档列表其实可以走 PG 的，走 milvus 意义不大」。
注册表是 O(文档数) 的关系型数据，本来就该在关系库里；Milvus 只留向量本体与检索。

| 项 | 之前 | 现在 |
|---|---|---|
| 落点 | Milvus collection `<collection>_docs` | 表 `document_registry`（PG；单机 SQLite `<state_dir>/documents.sqlite3`） |
| 状态 | 靠 `_docs` 里有没有那一行 | 表里一列 `stage`：`uploaded → parsed → reviewed → chunked → indexed` |
| 列表页 | 逐份 `path_of`（O(N²)）+ `file_hash` + 读三份产物 JSON | **只读 `stage` 一列**（一次 `select`），不碰产物、不算哈希 |
| 详情页 | 现算产物摘要 + 查 Milvus | 现算产物摘要（灯如实反映产物是否过期）+ 读表里那一行 |

实现落在 `persistence/documents.py`（`SqliteDocumentStore` / `PgDocumentStore` +
`build_document_store`）。**选库统一走一个开关**（2026-10-10）：`persistence.backend`
（base.yaml 里 `${RAG_DB_BACKEND:-auto}`，值放 `.env`）管三份关系型数据 —— 文档注册表 +
上传件台账（`documents.backend` 那一组，两者共用一个）、模型选型（`models.backend`）、
日志索引（`observability.logging.postgres`）。优先级是「某一类显式写的 > 统一开关 > auto」，
`auto` 的含义没变（配了 `RAG_PG_URL` 用 pg，否则 sqlite）；选 pg 却没配 url 会当场报错。
判据的唯一实现在 `persistence/backend.py` 的 `resolve_backend()`。
**LangGraph 的会话状态 / HITL 中断不在此列**：它只有 PG 一个实现，由 `postgres.url` 决定
（留空 = 进程内，重启即丢）。
**对外形状不变**：`build_registry(settings)` 仍是那四个调用点
（`commit_chunks` / CLI / agent 的 `list_documents` / 文档接口）的入口，换的是它背后落在哪。
同步实现（`sync_*`）给 CLI 与测试，异步门面（`all/get/upsert/delete/stages/set_stage`）
给 API —— 与原来那个 Milvus 版同名同义。

**五处状态写入**：上传/抓取 → `uploaded`；解析 job 完成 → `parsed`；存清单 → `reviewed`；
切分完成 → `chunked`；`commit_chunks` 写注册行 → `indexed`。写不进去只记 WARNING
（台账挂了不该让上传/解析整批失败），代价是那一小会儿列表上这份会退回"什么都没做"。

**`content_hash` 从上传那一步就写**（2026-10-10 补）：以前只有 `upsert`（入库）写它，
于是「解析完、还没入库」的文档这一列是空串 —— 而 `plan_incremental()` 判「这份变了没有」
用的正是它（`incremental.py:139`），空值会被判成"变了"，批量 `rag ingest <目录>` 时
白白重跑一遍 OCR（PDF 就是重新过一遍产线）。现在 `set_stage()` 也收这个值
（**空值不覆盖已有值**），上传 / 解析 / 核对 / 切分各推一格时一起带上。

老数据用 `rag registry migrate` 搬：`--from-stages`（默认）扫 `data/uploads/` 与
`data/stages/<doc_id>/` 的产物，能判到 `chunked`；`indexed` 只有 Milvus 知道，所以
`--from-milvus` 单独一趟（向量库好了再跑，命令**只前进不后退**）。2026-10-09 实测：23 份
全部搬到 `chunked`。旧的 `<collection>_docs` collection 从此没人读写，
确认不再需要之后手动 drop 掉即可（**别在迁移前 drop** —— `--from-milvus` 还要用它）。

### 8.6 索引批次 manifest 与对账（2026-10-23）

`document_registry` 新增四个索引状态列，另有每批次一行的 `index_versions`：

| 列 | 含义 |
|---|---|
| `index_batch_id` | **当前活动版本**；旧数据为空，表示仍在 legacy collection |
| `index_status` | 新版本完成切活动时为 `active`；旧数据可为空 |
| `indexed_at` | 这一批真正完成的时间 |
| `index_error` | 最近一次活动版本错误；通常为空 |

`index_versions` 的每一行是 `(doc_id, batch_id)`，状态只允许
`staging / active / failed / superseded`。staging 版本不会改活动指针；失败版本也不会。

**双 collection 路由**：

| collection | 写 | 读 |
|---|---|---|
| 旧 `<collection>` | 升级后只读 legacy | 只用于没有活动版本的 doc；已有版本化的 doc 会被 filter 排除 |
| `<collection>_versions` | 新批次唯一写入点 | 只读 registry 当前活动 batch；staging / superseded 全部排除 |

版本化 schema 多两个字段：`index_batch_id` 与 `logical_chunk_id`。物理主键使用
`batch_id:logical_chunk_id`，两代块可以并存；检索输出仍还原成稳定的逻辑 chunk_id，
引用/详情接口不需要知道物理版本。

`commit_chunks()` 的顺序是：

1. 写 `index_versions(staging)`；旧活动版本保持可查；
2. embedding + 只写 `<collection>_versions`，**不删除任何旧批次**；
3. 关系库一个事务里把新版本转 active、旧版本转 superseded、注册表活动指针指向新 batch。

切换前查询只命中旧 batch；提交事务后查询只命中新 batch。中途失败只会留下 failed 候选，
旧版本完全不受影响。SQLite / PG 老表在 `setup()` 时原地补列和建 `index_versions`。

`rag registry reconcile` 是只读对账入口，比较：

- 切分产物的 `file_hash`、切法指纹与注册表；
- manifest 的记录块数与产物块数；
- Milvus 中该 `doc_id` 的 `chunk_id + content_hash` 集合；
- `stage / index_status` 是否完整。

`--repair` **只回填可证明完成的 manifest 状态**（例如数据都已写完、进程在转 active
前退出）。它会逐项报告，不会静默修数据：

- 产物缺失、文件/切法指纹不符、Milvus 块缺失/多出/内容变化 → 只报告；
- staging / legacy 但产物与 Milvus 完全一致 → 可回填 active；
- 真正的修复仍走重新 ingest。

`rag registry prune --older-hours 24` 只删除不承载任何活动 batch 的 failed / superseded
批次；活动 batch 永远不会被这个命令删除。先用 `--dry-run` 可以只看清单。

**实测（2026-10-23）**：真实 Milvus 的 legacy collection 有 245 块；探针用其中一份文档
的 100 个已有块/向量写入临时版本化 collection，验证 schema、活动 batch filter、
hybrid 检索命中与清理，最终 `versions_cleaned=true`，legacy 未改。探针不调用
embedding / VLM。

**边界**：尚未跑真实 `commit_chunks` 全链路重灌和 registry 活动指针切换的端到端验收。
查询每次还要从关系库读活动 batch 图，并把 batch/doc 集合拼进 Milvus filter；
超大语料要再验证表达式长度与 scalar filter 性能。

---

## 9. 检索层

### 9.1 通道

| 通道 | 说明 | 默认 |
|---|---|---|
| `hybrid` | dense + sparse，Milvus 服务端 RRF 融合。**语义与关键词都在这一路里** | ✅ 主通道 |
| ~~`bm25`~~ | **2026-10-08 已删**：它就是 hybrid 那条稀疏腿（同索引、同 analyzer、同一次打分），单开一路只是把同一份证据在 L2 里投两票。2026-10-07 的 27 题实测见下 | —— |
| `graph` | 实体关系、多跳（**已实现** 2026-10-05，Neo4j）：`rag graph extract` 抽实体关系 → `rag graph load` 写图 → 检索时实体链接 + 1~2 跳邻居，再把关系挂着的 `chunk_id` 换回正文（见 §9.6） | **插件**（`plugins/rag-channel-neo4j`，`type: neo4j`）：装了包 + 在 `retrieval.channels` 里加一项才参与 |
| `web` | 语料库外兜底。搜索走**智谱的 web 搜索**（`web_search_tool`，2026-10-05 接入），复用 `ZHIPUAI_API_KEY`，不额外要一家供应商的 key | 可选，**agent 侧已可用**（主 agent 直接有这个工具 + `WebSearchGate` 闸门）；检索侧还没有适配器（要加就得走 §9.2 的插件） |

**装配方式：一张配置表**（2026-10-08 起）。`retrieval.channels` 里每一项 = 一路数据源：

```yaml
retrieval:
  channels:
    - {name: milvus, type: milvus, enabled: true}          # 语义 + 关键词都在这一路里
    - {name: kg, type: neo4j, enabled: false, complement_only: true, top_k: 10}
```

* **`name`** 是这一路的身份：日志、通道账本、RRF 的 `sources`、`--channel` 与 `/search` 的
  `channels` 参数全都用它（重名会被配置校验拦下 —— 两套身份是最容易出错的地方）；
* **`enabled`** 是参不参与检索的**唯一**开关；
* **`complement_only`** 覆盖适配器默认值（只补位：与别路重复的候选丢掉，只贡献新块）；
* **`top_k`** 覆盖这一路的候选数（**默认全部路一样多** —— 给得多那一侧票更多，实测踩过）；
* **`options`** 由适配器自己解释（内置的 `milvus` 认 `dense_limit` / `sparse_limit`）。

只有一路时 `build_retriever` **原样返回它**（不包一层），保持单路的行为与性能不变。

### 9.2 通道怎么"零代码"加一路：插件化

使用者的原话：「加通道需要一个 python 包，实现这个接口，然后负责加载，这个用了 Java 中
加载不同 db 驱动的设计」—— 对，这就是 **`ServiceLoader` 那一套**。Python 里的对应物是
**entry points**，实现落在 `retrieval/plugins.py`：

```
type 名 ──► ① 内置注册表 CHANNEL_TYPES（milvus / neo4j）
            ├─► ② entry points（组名 rag.channels）= 装好的插件包
            └─► ③ 点路径 "包.模块:build" = 开发期免安装的逃生舱
```

**接口只有一个函数**：

```python
def build(*, index, embedder, settings, cfg) -> Retriever:
    # 返回的东西要有 name / last_stats，以及 retrieve(query, top_k, filters, image)
```

`index` / `embedder` 可能是 `None`（纯外部服务、纯标量的通道用不上），`cfg` 是配置里
**那一项**（`name` / `type` / `enabled` / `complement_only` / `top_k` / `options`）——
插件自己的参数写在 `options` 里，核心代码不认识它们、也不需要认识。

**装法**（在自己的包里声明入口点，等价于 `META-INF/services/java.sql.Driver`）：

```toml
[project.entry-points."rag.channels"]
my_es = "my_pkg.channel:build"
```

`pip install` 之后配置里写 `type: my_es` 就能用 —— **核心代码一行不改**。
开发期还没打包时，`type` 直接写点路径（`type: "my_pkg.channel:build"`）。

**加载失败一律当场报错**（包没装、入口点名写错、模块导不进来、对象不可调用），并把
"内置的有哪些、入口点里声明的有哪些"一起列出来 —— 静默少一路数据源是最难查的那类问题。

**样板就在仓库里：`plugins/rag-channel-neo4j`**（2026-10-08 把图谱通道从核心里搬了出去）。
它现在是个**独立分发的包**：核心的内置类型只剩 `milvus`，`type: neo4j` 是靠 entry point
找到 `rag_channel_neo4j:build` 的。装 / 卸 / 用的完整命令见那个包的 `README.md`；
装卸两次就能看清这条路的边界：

```bash
pip install -e plugins/rag-channel-neo4j   # 装上 → 配置里 type: neo4j 立即可用
pip uninstall rag-channel-neo4j            # 卸掉 → 配置里写 neo4j 当场报「未知的通道类型」
```

**2026-10-08 全搬完了**：图库客户端（`store.py`）、实体关系抽取（`extract.py`）、
`rag graph extract/load/build/stats/query` 五条命令（`cli.py`）也都在插件包里 ——
**核心不再有任何 Neo4j 代码**。命令靠**第二个扩展点**挂上去：entry point 组 `rag.cli`，
核心只有 `cli/main.py::_mount_plugin_commands()` 那 15 行（模块级挂 + 插件坏了只警告）。

**第三个扩展点是 PDF parser：`plugins/rag-dots-parser`**（2026-10-23 外置）。
dots.ocr 官方样例约 1981 行，原先躺在核心包的 `ingest/dots_ocr/`；现在作为独立包
`rag-dots-parser` 安装，核心的 `ingest/dots.py` 只保留可选 import 与薄客户端。
装上后默认 `pdf.parser: dots` 照常工作，卸掉后核心仍能启动，直到真的选择 dots 时才报：

```bash
pip install -e plugins/rag-dots-parser   # 默认 PDF 链路需要
```

第三方 vendor 目录仍不按本仓库风格改写，但排除范围跟着包移动；CI 会装这个包，
因此默认解析链路和插件缺失提示各有测试覆盖。

**已入库语料补图不用重灌**：图是从**切分产物**（`data/stages/<doc_id>/chunks.json`，灌库时
就落好的）抽的，`rag graph build --all` 一条命令搞定，只花抽取这一次 LLM 的钱。
完整的插件说明（工作原理、怎么写一个、什么时候不该做插件）见 **`docs/plugins.md`**。

**learned sparse ≠ BM25，不能互相替代。** 稀疏向量权重是模型学的、能做 term expansion，对「语义相近但用词不同」召回更好；BM25 是纯统计，对精确串匹配反而更稳。

**bm25 通道为什么默认关（2026-10-07 实测）**：它和 hybrid 的稀疏腿是**同一份**
Milvus 内建 BM25（同索引、同 `chinese` analyzer、同一次打分），所以不可能召回 hybrid
看不到的文档；它实际做的只是把同一份证据在 L2 的 RRF 里**投两票**，外加把服务端 RRF
截断掉的稀疏命中捞回来一点。27 题检索层对照（`rag eval --channel hybrid,bm25` vs
`--channel hybrid`，同一索引、同一配置）：

| 指标 | 开 bm25 | 关 bm25 |
|---|---|---|
| recall@8 | 87.5% | 87.5% |
| top1 | 62.5% | 62.5% |
| MRR | 71.70% | **71.43%** |
| nDCG@8 | 75.58% | **75.36%** |
| 名次有变化的题 | 2 / 27（一升一降） | |

单条查询的账本更直观（Kafka 日志保留时间那题）：`hybrid 10→入选 7`、`bm25 10→入选 5`，
两路重叠 4 条 —— **bm25 真正独有的只有 1 条，而那条是个跑题的 Flink 块**；
正确答案 `log.retentionminutes` 反倒被挤到第 8（关掉 bm25 回到第 7）。
代价那边也没占到便宜：多一次 Milvus 往返（实测 142ms）。

**但 M1 的结论是：本项目暂时只有 BM25 可用。** fastembed 拿不到任何 learned sparse 模型
（也没有 bge-m3），所以「hybrid 主通道」的 sparse 腿由 Milvus 内建 BM25 承担（ADR 13）。
这不是把 BM25 说成 learned sparse 的等价物 —— 是承认这一版的 hybrid 实际是 **dense + BM25**。
将来若换成能自产 sparse 的模型（比如走 FlagEmbedding 的 bge-m3），要另配一个 sparse 字段，
**不能**复用 BM25 function 的输出字段，届时 §9.1 这张表才谈得上「learned sparse 通道」。

### 9.2 增强环节

- **transform**：HyDE / multi-query 展开 / 问题分解
- **fusion**：跨通道 RRF（L2 层，客户端）—— **已实现**（2026-10-03，`retrieval/fusion.py`）
- **rerank**：本地 cross-encoder（flashrank / sentence-transformers 两个 backend），
  把融合后的 20 条候选重排 —— **已实现**（2026-10-03，`providers/rerank.py`，见 §9.4）
- **assemble**：按 chunk_id 去重 → token 预算裁剪 → **EvidenceLedger 引用编号**
  （编号在此生成，贯穿到最终答案）

**M1 已实现 assemble**（`retrieval/assembler.py`），**L2 融合已在 2026-10-03 落地**
（`retrieval/fusion.py` + `retrieval/multichannel.py`），**重排也在同一天落地**
（`providers/rerank.py`）；`transform` 还是配置里的空位。
装配阶段有三条不显眼但会直接决定答案质量的规则：

1. **同一块被多路召回时只保留排名最靠前的那次** —— hybrid 下这是常态，不去重会让同一条资料
   在上下文里占掉多份预算，还会让模型以为它是两条独立证据。
2. **token 超预算的条目是「跳过继续」而不是「就此停止」** —— 一条长块不该让后面那些放得下的
   块一起被浪费。
3. **编号在这里生成，和正文出自同一次装配** —— 分两处算必然对不上，而模型只被允许引用
   上下文里的 `[n]`，引用表错了整段引用就都是假的。

token 估算没有 tokenizer（不值得为估算引一个依赖）：CJK 码点约 1 token / 字，拉丁约 4 字符
1 token，并且**故意估高** —— 估低了会超窗口被静默截断，估高只是少放几条。

### 9.3 L2 融合（`retrieval/fusion.py` + `retrieval/multichannel.py`，2026-10-03）

主通道与 `optional_channels` 一起跑，客户端合成一张榜：

```
hybrid（dense+sparse，服务端 RRF）┐
bm25（sparse 腿单拎出来）         ├─► 客户端 RRF ─► 最终 top_k ─► 装配
（以后 graph / web）              ┘
```

三个实现决定，各有理由：

1. **只用名次，不用分数**。各通道的分数量纲根本不同（COSINE 0..1 / BM25 无上界 / 以后
   web 的另一种），归一化要按语料调参、还容易过拟合。RRF 是 `Σ 1/(k + rank)`，天然可比
   —— 与 Milvus 服务端融 dense+sparse 用的是同一套。`retrieval.fusion.k`（默认 60）
   是这里唯一要调的参数。
2. **并发跑；单路失败不拖垮整次查询**：一路挂了记进 stats、当作没召回，其余照常出结果。
   **但全挂是真失败**，原样抛出第一个异常 —— 返回空列表会让上层以为「就是没命中」，
   然后照常生成一个没有依据的答案。
3. **融合后 `channel` 变成 `rrf`，真实来源记在新增的 `ScoredChunk.sources`**，否则
   「这条是 dense 捞的还是 bm25 捞的」在结果里查不到。`--stats` 打印每路的召回数 / 入选数 /
   耗时 / 错误（用的是一直没派上用场的 `ChannelStats`）。

**候选池一路一样大**：每路都要 `max(top_k, retrieval.fusion.candidates_per_channel)` 条
（默认 **10**），融合去重后按 `top_k` 交货。这一条是真机撞出来的 —— 第一版 hybrid 只给
`top_k`=5 条、bm25 按自己的 `sparse.limit` 给 20 条，**票数是 1:4**；现在由融合层统一发话，
通道不再自作主张。同时这个池子就是下一层 rerank 的输入：精排得有得挑才有意义。
（20 → 10 是 2026-10-03 定的：重排一次 26 条要 6–8 秒，池子减半、耗时基本减半。）
真机验过（同一条查询）：池子 20 → 重排 26 条 6.2–7.7 秒；池子 10 → 重排 14 条 **4.9 秒**，
正确答案的名次与分数**一模一样**（都是 #2 / 0.8113）。

### 9.4 重排：为什么不用 Milvus 的 reranker（2026-10-03）

Milvus 3.0.2 确实自带重排能力（schema 上的 `FunctionType.RERANK`），但我们**用自己的**
（`providers/rerank.py`）。三条理由，与模型强弱无关：

1. **作用域**（决定性）：候选来自两路，重排必须吃**融合后的池子**。Milvus 的 reranker
   跟着一次 search 走，只看得到那一次检索的候选 —— 跨不了通道（除非把 bm25 塞回同一次
   检索，那等于放弃「独立通道」的设计）。
2. **换模型不改表结构**：Milvus 的 rerank 是写在 collection schema 上的 function，
   换模型要重建 collection；我们换 `providers.rerank.backend` 一行就行。
3. **依赖与费用**：flashrank 已在依赖里、本地 ONNX、零调用费；`pymilvus[model]` 要额外
   extras，线上重排 API 按量计费。

反过来该用 Milvus 那个的场景：候选池特别大不想搬数据、想复用 Milvus 侧 GPU、
或团队要统一在 Milvus 侧运维模型。现在都不成立（池子 20 条，CPU 足够）。

实现要点：

| 点 | 说明 |
|---|---|
| 两个 backend | `sentence-transformers`（**默认**，CrossEncoder + `BAAI/bge-reranker-base`，模型走 ModelScope 下到 `.models/`）/ `flashrank`（ONNX 依赖已在，但**它的模型本机拿不到**：硬编码直连 huggingface.co、不看 `HF_ENDPOINT`，镜像只做 302 跳回、那边匿名 401，ModelScope 也没这个仓库 —— 有 HF token 才用得上） |
| 位置 | 融合**之后**、交货之前（`MultiChannelRetriever._rerank`） |
| 交几条 | **调用方的 `top_k`**。配置里刻意**没有** `rerank.top_n` —— 两个数必然打架（`-k 20` 被悄悄截成 8） |
| 失败怎么办 | 退回融合顺序 + 记进 `--stats`（模型没下下来不该让整次查询失败）；空结果同样算失败 |
| 分数 | 重排后 `score` 换成相关性分数、`channel` 变 `rerank`，`sources` 保留（召回来源不因重排丢失） |
| 开关 | `retrieval.rerank.enabled`，**默认开**（2026-10-03 定）；临时关：`$env:RAG_RERANK="false"` |
| 位置前提 | 开着重排却没传 reranker → `build_retriever` 当场报错，不静默跳过 |

### 9.5 三个调试/对比用的 CLI 选项（2026-10-03）

`rag search` 与 `rag ask` 都有：

| 选项 | 干什么 | 典型用法 |
|---|---|---|
| `--filter key=value`（可重复） | metadata 过滤（`filter_expr` 那条管道终于有人用了）。`a,b` 当 `in`；纯数字变 int、`true/false` 变 bool | `--filter modality=image`、`--filter page=3,5` |
| `--channel`（逗号分隔） | 覆盖「跑哪些通道」：**列出来的那几路开、其余关**（名字取自 `retrieval.channels[].name`）。写 `milvus` 就是「只要 Milvus 那一路」（语义 + 关键词都在它里面） | `--channel milvus`（默认等价）、`--channel milvus,kg`（带上图谱那一路） |
| `--rerank/--no-rerank` | 覆盖 `retrieval.rerank.enabled`。关掉时**连模型都不加载** | `--no-rerank`（看纯召回的顺序） |
| `--image PATH` | **以图搜图**。检索词可以省略（只给图），也可以图 + 词一起给（这时词是「想找什么样的图」的补充说明） | `rag search --image shot.png`、`rag search "架构图" --image shot.png` |

名字/写法错了一律**当场报错并给出例子**：静默忽略等于「你以为关了 bm25，其实没关」，
这种「结果变了但看不出为什么」最难查。解析逻辑抽成了纯函数（`_parse_filters` /
`_apply_channels`），所以不连库也能单测。

**纯图查询在各通道里的行为**（写在代码里，不靠默认值蒙）：

| 通道 | 纯图查询时 |
|---|---|
| `hybrid`（dense + sparse） | **只走 dense**：没有查询词，稀疏腿无事可做，硬走 hybrid 等于拿空串去请求 BM25 |
| `bm25` | 直接返回空 —— 没有词就没有词频，这是正常情况不是错误，融合层把它当作「这一路没召回」 |

`ask` 暂时没有 `--image`：生成那一步还是要一句问题，图文提问等界面那一步一起做。

### 9.6 图谱通道（Neo4j，2026-10-05 落地）

**存储选 Neo4j**（用户 2026-10-05 定），不塞进 Milvus：这一路的价值就在多跳遍历，
而「实体→关系→实体→关系」的遍历是图数据库的活。部署走 SSH 隧道（`bolt://127.0.0.1:7687`），
与 PG / Redis 同一个习惯；密码进 `.env`（`NEO4J_PASSWORD`），地址与用户名在 yaml。

四步链路，每步都能单独停下看：

| 步 | 命令 | 产物 / 效果 | 花钱吗 |
|---|---|---|---|
| ① 抽取 | `rag graph extract --doc-id X [--limit N] [--dry-run] [--force]` | `data/stages/<doc_id>/graph.json`：逐块 `{entities, relations, evidence}` | **是**（每块一次 chat） |
| ② 写图 | `rag graph load --doc-id X` | Neo4j：`(:Entity {key, names, types, docs})` + `[:REL {relation, evidence, chunk_id, doc_id, heading_path, page}]` | 否 |
| ③ 检索 | `optional_channels: [..., graph]` | 实体链接 → 1~hops 邻居 → 映射回块 → 交给已有的 L2 融合 | 否 |
| ④ 排查 | `rag graph stats` / `rag graph query "问题"` | 图里有多少、某个问题能认出哪些实体、拉回哪些关系 | 否 |

**两条判据是实测出来的**（这两条决定图里有多少节点）：

1. **归一键**：同一概念在原文里有多种写法 —— 实测同一张图里 `Flink`×24 与 `Apache Flink`×4
   是两个节点，第二章还有 `StreamExecutionEnvironment` 与
   `org.apache.flink.api.java.ExecutionEnvironment`。节点身份用 `key`（NFKC + 小写 + 去组织前缀
   + 命名空间取类名 + 去掉 Maven 的 `${...}`），**原文写法全部留在 `names` 里**（不丢信息）。
   不合并就等于把同一个概念的边拆到几个节点上，一跳查询必然漏。
2. **按字符构成过滤**，不按长度：实测「超长实体名」里大多数是**对的** Java 类全名，
   真正该丢的是含 `${}`/花括号/换行的 Maven 坐标，以及 OCR 把点读成空格的路径
   （`org.apache.flink api.java ExecutionEnvironment`）。按长度砍会把好的砍掉。

**写出与删除**：`load` 按文档幂等 —— 先删这篇的边、再写新的（与 Milvus「先删后写」同一个规矩）；
节点不删（归一之后本来就该被多份文档共用，`docs` 属性记着它出现在哪些文档里）。
**关系键**是 `doc|chunk|source|relation|target` 的哈希，所以同一块里
`Flink --支持--> 窗口` 与 `Apache Flink --支持--> 窗口` 归一到同一条边，账也按这个口径记。

**检索侧的打分**只有两条：跳数越近越值钱（第 n 跳 1/n）；**一条边的两端都被问句认出来就翻倍**
（问句里同时出现两个实体时，「这两个之间的关系」才是有信息量的那条）。同一块被多条关系命中取
**最大值**而不是求和 —— 求和会让热门实体把同一块堆成虚高的分数，融合时压过所有别的通道。
实体链接是**纯子串匹配**（零额外模型调用）：问题里出现图里的实体名（或它的归一键）就认。

**降级**：Neo4j 连不上/查不动就**直接抛**，由 `MultiChannelRetriever` 记进 `stats.error`
并当作「这一路没召回」，其余通道照常 —— 与 Milvus、重排的降级是同一个立场（`--stats` 里
看得见是哪一路坏了）。

**这一批的边界**：只做 1~2 跳召回，**不做社区摘要/全局问答**，也**不做实体消歧**
（同名不同义靠 `docs` 属性区分；后续要更细的消歧，得在 `names` 上再加一层来源约束）。
当前只灌了 Flink 两章（354 实体 / 426 关系）。**A/B 结果：在现有黄金集上没有可测量的提升
（27 条里名次变化 0 条），所以全量灌图暂缓** —— 原因与下一步见 §13.3。

**它声明了 `complement_only = True`**（2026-10-05 加的）：融合层会把它与 hybrid/bm25
重复的候选丢掉，只让它贡献新块。理由与实测见 §13.3 —— 同一批块再投一票改不了名次，
而这条通道每次查询要花 ~350ms。**默认不启用**（`dev.yaml` 的 `optional_channels` 是空的）。

**真机三方对比（2026-10-03，`rag_dev_qwen3vl_v1`，26 条候选池）**：

| 查询 | 重排关 | bge-reranker-base | flashrank（MiniLM-L-12） |
|---|---|---|---|
| `Flink 的架构图里画了哪些数据源` | 真正那张架构图（1.1.1 图片块，描述里写着交易/日志/IOT/用户点击四类数据源）**不在前 5** | 它升到 **#2**（0.81）；#1 仍是「2.3 数据源分为有界和无界」 | **掉出前 5**（前 5 全是正文），#1 变成「1.1.2 Flink前身」 |
| `2.2.1 IDEA Project 创建及配置` | 前 5 全在同一节（3 正文 + 2 图片） | 仍全在同一节（4 图片 + 1 正文），分数 0.999x **饱和** | 仍全在同一节（5 图片），分数 0.9995 **饱和** |
| 26 条候选耗时 | — | **6.2–7.7 秒** | **1.4–1.9 秒** |

结论很清楚：**flashrank 在中文语料上没有区分度** —— 分数一片 0.9995，排序等于退化成
tie-break（融合顺序），第一个查询上还把正确答案挤掉了；它唯一的优势是快 3–4 倍
（英文 MS MARCO 训的模型，对中文本来就不该有期待）。bge 慢得多但真能挑出东西。
默认因此是 **bge（`sentence-transformers`）**，并且 `retrieval.rerank.enabled` 默认**关**。

`filter_expr()` 只支持等值/in 且对值做转义，不支持的类型直接报错：过滤值来自调用方，
直接拼表达式等于把注入面交给上层。注意块级元数据在 JSON 字段里，过滤要写成
`metadata["h1"] == "..."` 这种下标形式，而不是 `h1 == "..."`。

---

## 10. 编排层：LangGraph（2026-10-03 落地）

两张图都在 `src/rag/graphs/`，而且都**只做编排** —— 真正干活的还是 ingest / retrieval /
providers 里那些函数。所以这一层可以整体替换而不动下层，这也是当初把三层协议定成
Protocol 的用途。总开关 `graph.enabled`（默认关，`--graph/--no-graph` 可覆盖单次）。

### 10.1 QueryGraph（在线检索，带自我纠错）

```
plan ──► retrieve ──► grade ──┬──► assemble ──┬──► generate ──► END
  ▲                            │               └──► END（流式预检）
  └──── 资料不够（轮数没用完） ─┘
```

`AnswerEngine` 把生成前部分编译成预检图：非流式与流式都先跑完
`plan → retrieve → grade/replan → assemble`，再分别选择 `generate` 或
`generator.astream`。因此 `stream=true` 不再绕过检索纠错。

| 节点 | 干什么 |
|---|---|
| `plan` | 一次 JSON 输出：`{sub_queries[], filters{}, channels[], reason}`。决定改写、要不要拆多查询、要不要过滤、走哪几路 |
| `retrieve` | 每个子查询并行跑 `build_retriever`（M2 那一整套）；多个子查询的结果再用**同一个 `rrf_fuse`** 合一次 —— 多查询与多通道在数学上没有区别 |
| `grade` | 一次 JSON 输出：`{score, missing}`。低于 `graph.relevance_threshold` → 回 `plan` 重来，最多 `max_corrective_rounds` 轮 |
| `assemble` | 形成 `PreparedAnswer`；普通问答和流式问答从这里接同一个生成入口 |
| `generate` | 非流式时接 `CitationGenerator.generate`；流式时由 `AnswerEngine.stream` 逐块输出 |

**与初稿那张图的两处偏离，都有理由**：

1. **`transform/fanout/fuse/rerank` 收进 `retrieve` 一个节点**。从 M2 起，多通道融合与重排
   就是 `MultiChannelRetriever` 内部的事；编排层再拆开等于把封好的东西摊开、还多几层状态。
2. **`grade` 判「资料够不够」，放在 `generate` 之前**。判资料只看候选，不必先生成一版 ——
   省一次 LLM 调用，也避免「拿自己刚生成的答案给自己打分」这种自我循环。
   答案层面的核查（引用是否落在上下文里）属于 M6 评测。

两条**行为上的取舍**：

* **零召回也会改写重查**：问法与语料用词对不上是零召回最常见的原因，换种说法重来正是这条
  循环最值钱的地方。轮数用尽仍没资料时照样进生成 —— 那时上下文是空的，
  `CitationGenerator` 会回「没有检索到相关资料」，即老实拒答（比硬拒绝稳：评分模型自己也会判错）。
* **plan/grade 要 JSON，不用 `with_structured_output`**：那依赖网关的 tool calling / JSON mode
  支持，而我们四家 chat 供应商都走 OpenAI 兼容通道，支持程度不好赌。解析器容忍 ```json 围栏
  与前后废话，解析不出来**直接报错**（用错计划比报错难查）。计划还会被**收敛**：
  子查询最多 3 条、编出来的通道名滤掉、filters 不是字典就当空 —— 模型输出不可信。

### 10.2 IndexGraph（离线灌库）

```
START ─►（按文件 Send 派发）─┬─► [文件 A: 解析 → 闸门 → 切分] ─┐
                            ├─► [文件 B: …]                 ─┼─► write ─► END
                            └─► [文件 C: …]                 ─┘
```

* **fan-out 挂在条件边上**（`add_conditional_edges(START, fanout, ["file"])`）——
  `Send` 是派发指令不是状态，当普通节点返回会报 `InvalidUpdateError`（实测撞过）。
* **同步的解析/切分丢进线程池**（`asyncio.to_thread`）：`run_parse`/`run_chunk` 内部是阻塞的
  OCR / VLM / 模型推理，直接在协程里调会把事件循环堵死，fan-out 就变成假并发。
* 并发上限 `graph.index_concurrency`（默认 1）用信号量控制 —— 每个文件都可能打外部服务。
* **不引 sqlite checkpointer**：「断点续跑」我们早就有 —— `data/stages/` 的产物 + 注册表让
  重跑天然幂等。LangGraph 在这里负责的是**这一次运行的铺开方式**（并发、失败隔离、进度），
  不是持久化。
* **图片描述缓存加了锁**：它是所有文件共用的一个 json，两个线程「读→加一条→写回」会互相
  覆盖（症状是描述生成过却还去问 VLM，白花钱）。现在按文件路径加锁，写之前还会重新读一遍
  再合并。

写库那一段（含「先删后写」「缺产物整批不写」「删块在删注册行之前」三条规矩）抽到了
`indexing/commit.py`，CLI 与 IndexGraph 共用一份实现 —— 规矩只该有一处。

### 10.3 还没做

* ~~**流式输出**：`generate` 目前一次性返回~~ —— **已落地**：`generator.astream` +
  `/chat`、`/agent` 的 SSE（见 §11.5 与 §12.3）。这条 2026-10-12 从内部待办清单里清掉。
* **web 检索通道**：agent 侧已可用（`web_search_tool` + `WebSearchGate`，§11.6），
  检索层还没有适配器 —— 要加就得走 §9.2 的插件。
* **把图谱 / web 写进 LangGraph 的节点**：现在两者都不在图里 —— 图谱是检索层的一路通道
  （插件提供），web 是 agent 的一个工具。要不要变成图节点属于新决策，不在现有计划里。

---

## 11. Agent 层：deepagents（2026-10-04 落地批次 A + 批次 B）

**批次 A 是只读的**：工具 + 主 agent + `rag agent` + `POST /agent`。
**批次 B 打开写盘，但写盘要人批**：workspace 路由到真磁盘，`write_file` / `edit_file`
挂 HITL 闸门；`delete` / `execute` 由 ToolPolicy 禁用、不会进入审批。另有
中断恢复（`--resume` / `POST /agent/resume`）和 SSE 流式。

**2026-10-08：一个 agent + 四个知识工具，项目自定义 subagent 全部撤掉。** 使用者定了两条：
「一个智能体走通道融合最后出结果就行了」「工具层面只需要 `search_knowledge` /
`get_section` / `list_documents`（网络搜索工具还是要的）」。于是五个 subagent
（vector / keyword / image / graph / web-searcher）全删，`search_keywords`、
`search_images`、`search_graph` 三个工具也删 —— 通道的并发与融合本来就在检索层
（`MultiChannelRetriever`）做完，让模型分头调、再用文字拼两路结果，等于把 RRF 的分数
换成模型的印象；而网络那条**本来就是靠闸门管的**（`WebSearchGate`：语料没查过、
或语料已有结果 → 拒绝），不需要"主 agent 拿不到这个工具"来实现。

### 11.1 现状

| 组件 | 是什么 |
|---|---|
| 知识工具（`agent/tools.py`）**四个** | `search_knowledge`（**一次调用 = 一次融合检索**：所有启用的通道并发 + 融合 + 重排，`modality` 用来只看某类块）/ `list_documents`（有哪些文档）/ `get_section`（取某几节原文）/ `web_search_tool`（网络，带闸门） |
| 项目自定义 subagent | **没有**（2026-10-08 撤）。即使 deepagents 自动注入 general-purpose / `task`，ToolPolicy 也会把它从模型可见面和执行面同时裁掉 |
| 工具策略 | `ToolPolicyMiddleware`：`readonly` 只允许知识工具 + 只读文件工具；`report` 另允许 `write_file` / `edit_file`；`task` / `delete` / `execute` 默认拒绝 |
| 主 agent | `create_deep_agent(model=build_chat(agent.chat), tools, subagents=[], backend=StateBackend(), checkpointer=PG, store=PG)` |
| CLI / API | `rag agent "问题" [--thread-id] [--no-writes]` / `rag agent --resume --thread-id <会话> --approve` / `POST /agent {message, thread_id, stream}` / `POST /agent/resume {thread_id, decisions}` |

### 11.2 三个关键决定

1. **通道查询只有一条路：`search_knowledge`**（2026-10-08 定）。关键词、图片、图谱都不再
   单独开工具：关键词是主通道里的 sparse 腿、图片靠 `modality` 过滤、图谱是一路**通道**
   由检索层融合。判据很实在 —— 工具一多，模型就会分头调、再用文字合并结果，
   而"哪条支撑哪句话"是靠 RRF 的分数与引用编号钉住的，不能交给模型的印象。
   （代价要说清：撤掉 `mode="isolated"` 的 subagent 之后，中间结果会累积在主上下文里；
   换来的是"一次工具调用 = 一次融合检索"这个形状。）
2. **引用编号由我们发**（`EvidenceLedger`）：工具返回的每条命中拿一个**全局递增的 `[n]`**，
   同时登记 `n → Citation`；网页结果也登记到同一张表。答案里的 `[n]` 直接查表，同一个来源
   复用同一个号；返回给调用方时**只列答案里真的引用到的那些**（召回了 20 条只用 3 条时，
   全塞回去会让「哪条支撑哪句话」看不出来）。
3. **每请求建图、但不重建模型**：工具要闭包住这次的引用表，所以图不能全局共享；
   而 embedder / reranker / chat 都在 `graphs/wiring.py::build_heavy` 里只建一次（M3 的教训）。
4. **提示词只钉稳定契约**（2026-10-23）：主提示词负责证据优先、检索内容不可信、工具边界、
   引用、冲突 / 部分回答和输出顺序；工具参数细节留在工具 docstring，格式例子留在 skill。
   尤其明确 `task` 不用于委派检索，多节原文必须一次 `get_section` + `headings` 批量取，
   低分检索反馈只指向 `web_search_tool`，不再提已删除的 `web-researcher`。
   `tests/test_agent_tools.py` 钉住这几条，防止提示词以后在重构中被悄悄改薄。
5. **工具权限用中间件，不靠提示词自觉**（2026-10-23）：`ToolPolicyMiddleware` 集成
   `AgentMiddleware`，模型请求阶段按允许列表裁掉工具，执行阶段再拒绝越权调用。后者防的是
   历史消息、缓存或模型幻觉绕过“它看不见”；两处共用同一个 `ToolPolicy`，测试覆盖编译后
   真实图，而不只检查传给 `create_deep_agent` 的参数。

### 11.3 只读与多轮

* **只读**：backend 用 `StateBackend`（写也只落在内存 state 里，**磁盘一个字不动**）；
  `execute` 也无从谈起（没有 shell 后端）。
* **多轮**：`thread_id` + **PG 的 checkpointer**（§12.5）。给了 `thread_id` 但 PG 没连上时
  **直接报 503 并说明原因** —— 否则调用方会以为「历史在，只是模型忘了」。

```python
agent = create_deep_agent(
    model=build_chat(settings, secrets, settings.agent.chat),   # 用哪家见 agent.chat（.env 的 AGENT_CHAT）
    tools=build_tools(settings, heavy, registry, gate=gate),
    subagents=[],                           # 不再声明项目自己的检索 subagent
    system_prompt=SYSTEM_PROMPT,            # 证据、安全、冲突与输出契约
    backend=CompositeBackend(default=StateBackend(), routes={
        "/workspace/": FilesystemBackend(root_dir="./data/workspace"),  # 报告落盘
    }),
    memory=["/workspace/AGENTS.md"],         # 文件存在时才挂
    interrupt_on={"write_file": True, "edit_file": True},
    checkpointer=checkpointer,
    store=store,
)
```

**两个必须知道的注意事项：**

1. `model=None` 已在 deepagents 0.5.3 弃用（默认 `claude-sonnet-4-6`），1.0.0 会移除参数可空性。多供应商场景下**必须显式传 model**。
2. 传的是**构建好的 model 对象**，不是 `"provider:model"` 字符串 —— 后者会绕开三件套。

**取舍：不用 Anthropic（ADR 20）。** deepagents 的 harness 特性（prompt caching、`HarnessProfile`、
interrupt profile）确实对 Anthropic 支持最好，但价格也是最贵的 —— 使用者明确不用。所以 agent 层
与批处理统一走国产模型：`agent.chat` 指向 `providers.chat` 的哪个条目，由 `.env` 的 `AGENT_CHAT`
决定（默认 `default`，即 deepseek）。

**框架注入的兜底**：deepagents 自动注入的 general-purpose / 文件工具先经过 ToolPolicy
白名单。`task` 始终不可见、也不可执行；文件工具按 `readonly` / `report` 档位裁剪。
引用格式已经按 skill 加载；图谱只作为检索通道，不再做 subagent。

### 11.5 批次 B：可写 workspace + HITL + 流式（2026-10-04）

**开写盘的唯一开关是「给不给 workspace」**（`build_agent(..., workspace=...)`）：

* **不给** → `StateBackend`，与批次 A 一模一样：工具照样「成功」，但写的落点是内存
  state，**磁盘一个字不动**；
* **给了** → `CompositeBackend(default=StateBackend(), routes={"/workspace/":
  FilesystemBackend(root_dir=workspace)})`，`interrupt_on` 只给策略允许的
  `write_file` / `edit_file` 各挂一道闸门。`delete` / `execute` 更外层就被 ToolPolicy
  拒绝，因此不会弹审批；权限判断与人工确认是两层，不能拿 HITL 冒充授权。

`/workspace/` 是**虚拟根**：模型看到的路径是 `/workspace/report.md`，实际落在
`agent.workspace_dir`（默认 `./data/workspace`）下 —— 实测确认过（写 `/workspace/sub/x.md`
落成 `<workspace>/sub/x.md`）。`workspace/AGENTS.md` 存在时挂成 memory
（不存在就不挂，免得给模型一个不存在的路径）。

**中断与恢复**（协议是 LangGraph 原生的，离线用假模型验过）：

```text
中断载荷   {"action_requests": [{name, args, description}],
           "review_configs": [{"action_name", "allowed_decisions": [...]}]}
恢复       Command(resume={"decisions": [{"type": "approve"|"edit"|"reject"|"respond"}, …]})
```

* **一条 `action_requests` 对应一条 `decision`，顺序要对**。`--approve` / `--reject` 是
  「一键全批」：条数从图状态里数出来（`pending_actions`），不靠人报数；`--decision` 是逐个，
  条数与待批数对不上**当场报错** —— 否则中间件会在模型那头抛一句没人看得懂的错。
* 中断后 CLI **退出码 3**（区别于 1 = 真失败）：脚本里能一句话分辨「要人管」与「出错了」。
* 中断状态在 `aget_state(...).tasks[*].interrupts` 里，**`values` 里没有它**（实测）。
  所以「恢复」可以在**另一个进程**里做（CLI 是另一次进程、HTTP 是另一次请求）。
* **引用表按会话存一份**（`save_registry` / `load_registry`，落在 PG 的 store，
  命名空间 `("rag_agent", "citations")`）。原因：答案是**恢复之后**才生成的，它里面的 `[n]`
  用的是中断前发的号；不存，恢复后的答案会显示成「一条引用都没有」。取不到就给空表 ——
  引用丢了不该让恢复直接失败。

**流式**（`stream_agent`，`stream_mode=["messages", "updates"]`）四种事件：
`token` / `tool` / `interrupt` / `done`。两个实测细节：中断在 updates 里是
`__interrupt__` 这个键（值是个 tuple）；**子 agent 的 token 会混在同一个 `messages` 流里**，
不过滤就会把子 agent 的中间话术当成答案吐给用户 —— 判据是路径 `checkpoint_ns` 有没有多段
（主图一段、子图用 `|` 拼多段）。

**memory 的措辞会决定它敢不敢动手**（2026-10-05 真机踩到）：第一版 `AGENTS.md` 里写
「写盘 / 执行命令一律先等人批准」，模型把它读成「我不该发起这个调用」—— 于是它只输出一段
「我准备写…请确认」，一个工具都不调，闸门根本没机会出现。**闸门是在调用之后拦，不是在
调用之前**，所以 memory 必须写成「直接调用，系统有人工闸门会在落盘前停下来等人批准；
不要因为要等批准就自己放弃调用」。

### 11.4 实测（批次 A）

* Windows 上 psycopg 的 async 跑不了 `ProactorEventLoop` —— **必须在建循环之前**换
  `SelectorEventLoop`。第一版把这件事放在 `PgPersistence.connect()` 里做（那时循环已经建好，
  等于没设），表象是「连接池 10 秒超时」，从报错里完全看不出原因。现在改成
  **当场检查并说清修法**（`check_loop_compatible`），CLI 与 `create_app` 都在最早的时机设好。
* 单测里改全局 `asyncio` 策略**必须打桩**：第一版只打桩了 get、`set_event_loop_policy`
  是真调的，于是全局策略被换成假对象、`monkeypatch` 也还原不了 —— **后面 13 条用例一起挂**。

---

### 11.6 M6 尾巴：网络工具与 skills（2026-10-05；2026-10-08 收敛）

**`web_search_tool`**：使用者自研的智谱 web 搜索接进来了 —— key 从 `Secrets.zhipuai_api_key`、
base_url 从 `providers.chat.zhipuai.base_url`（**机密不进 yaml / Settings**：
`rag config` 会把整棵配置树打出来）。真机调过一次，返回带链接的真实结果。
**2026-10-08 起它直接挂主 agent**，不再藏在一个 `web-researcher` subagent 后面：
「网络只当兜底」这条现在是 `WebSearchGate` 这个**机制**在管（语料没查过、或语料已有结果
→ 工具直接拒绝并回理由），不靠"主 agent 手上没有这个工具"。
**Web 结果也进入 `EvidenceLedger`**（2026-10-23）：每条结果拿到与语料块同一编号空间的
`[n]`，`source_type=web`、`title` 和 `url` 一起返回；前端点击时打开原链接，CLI 的来源清单
明确写「来自网络」。不再出现“正文有编号、调用方却只有一段纯文本”的第二套来源格式。

**图谱（`search_graph` 工具已删）**：§9.6 的知识图谱现在只以**一路通道**的身份出现
（`type: neo4j`，`complement_only`：只补别人没找到的块），由检索层融合。
原来那个 `graph-explorer` subagent 与 `search_graph` 工具在 2026-10-08 一并撤掉，
理由与「知识工具只有四个」是同一条。它历史上的实测结论保留在这儿：**真机那轮模型没选它**
（问「Flink 和 Spark 在定位上有什么不同」，它走了 `list_documents → search_keywords →
search_knowledge → get_section`，11 步、引用 12 条，一次图谱都没查，答案是对的），
与 §13.3 的 A/B 一致 —— 这份语料上关系信息本来就在正文里，所以图谱通道默认不开。

**skills（按需加载）**：`create_deep_agent(skills=[...])` 挂两个源 —— 包内的
`src/rag/agent/skills/`（→ `/skills`）与用户自己的 `data/workspace/skills/`
（→ `/workspace/skills`，同名以用户为准）。每个 skill 是一个目录 + `SKILL.md`
（frontmatter 的 `name` 必须等于目录名，写错**静默不加载**，所以有用例拦）。现在两个：

| skill | 什么时候读 |
|---|---|
| `citation-format` | 输出带论断的答案/报告前：引用规矩、拒答怎么写（**拒答时别列无关引用** —— 实测那条把引用准确率从 90% 拖到 69%） |

（原来还有 `graph-query` 那个 skill，2026-10-08 随 `search_graph` 工具一起删了。）

skills 是从 **backend** 读的（中间件只认 backend 路径），所以要挂进 CompositeBackend 的路由；
没给 workspace（只读档）就不挂 —— 挂了等于给模型一个不存在的路径。

**顺带修掉一个上游 bug**：`FilesystemBackend._ripgrep_search` 用
`subprocess.Popen(text=True)` 且**没给 encoding**，Windows 上按 GBK 解码 rg 的 UTF-8 输出 ——
工作区里有中文（AGENTS.md / skills / 中文文件名）时，agent 一 `grep` 就
`UnicodeDecodeError: 'gbk' codec can't decode byte 0xa4`，整轮问答直接崩（真机实测）。
修法：我们自己的 `FilesystemBackend` 子类返回 `(None, False)`（上游约定的「rg 不可用」），
让它走按 UTF-8 读文件的 Python 回退。工作区只有几十个文件，性能无所谓；上游修了就能删。

## 12. 服务层

**FastAPI**（2026-10-03 落地，`src/rag/api/`；`rag serve` 启动）：

| 端点 | 说明 | 状态 |
|---|---|---|
| `GET /health` | Milvus / Redis / 模型**分开报**，哪个挂了单独看得见 | ✅ |
| `POST /search` | 纯检索：M2 那一整套（多通道 + 融合 + 重排 + 过滤 + 以图搜图） | ✅ |
| `POST /chat` | 问答；`stream=true` 走 **SSE**（`retrieval` → `token`* → `done`） | ✅ |
| `POST /ingest` | 提交文档 → **202 + job_id**（后台跑 IndexGraph） | ✅ |
| `GET /jobs/{id}` | 任务状态 / 结果 / 失败原因 | ✅ |
| `GET /jobs` | 最近的任务（新的在前），页面靠它轮询进度 | ✅ |
| `POST /documents/upload` | 上传（multipart，可多份）：**只落盘、不解析**，被拒的带原因回 | ✅ |
| `GET /documents` | 上传件清单 + 每份走到哪一步（解析/核对/切分/入库） | ✅ |
| `GET /documents/{doc_id}` | 详情：解析稿（**只读**）+ 标题清单 + 步骤状态 | ✅ |
| `GET /documents/{doc_id}/source` | 原始上传件的字节（页面「上传」那一步看原文） | ✅ |
| `POST /documents/{doc_id}/parse` | 解析这一份 → **202 + job_id** | ✅ |
| `PUT /documents/{doc_id}/outline` | 提交人工核对结果：**只改层级，不碰正文** | ✅ |
| `POST /documents/{doc_id}/index` | 切分 + 灌 Milvus → 202 + job_id（清单没核对就 409） | ✅ |
| `POST /documents/fetch` | 从链接导入：类型不定（HTML / PDF / Word…），抓 HTML 时**连页面里的图一起抓** | ✅ |
| `DELETE /documents/{doc_id}` | 删一份文档：默认连 Milvus 的块与 stages 产物一起清（不可逆） | ✅ |
| `POST /documents/folders` | 新建人工目录（`{name, parent}`，**最多三层**） | ✅ |
| `PATCH /documents/folders/{id}` | 目录改名 / 搬家（没传 `parent` 就是只改名） | ✅ |
| `DELETE /documents/folders/{id}` | 删目录：里面的文档回「未分类」；有子目录 409 | ✅ |
| `PUT /documents/folders/members` | 把几份文档放进一个目录 / 移回未分类（`{doc_ids, folder_id}`） | ✅ |
| `POST /agent` | deepagents 对话式入口；`stream=true` 走 SSE（`token`/`tool`/`interrupt`/`done`） | ✅ |
| `POST /agent/resume` | 批准 / 拒绝被中断的写操作，接着跑（`{thread_id, decisions}`） | ✅ |
| `GET /chunks/{id}` | 按 id 取一个块的原文（前端「点开引用看原文」） | ✅ |
| `GET /assets/{path}` | 取原始资产（图片）；**只认 assets 根目录底下、解析之后仍在根内**的路径 | ✅ |

### 12.1 三个设计决定

1. **服务层做薄**：它只做三件事 —— 建一次重对象、翻请求/响应、翻状态码。真正的活在
   retrieval / ingest / graphs 那几层。所以它才换得掉（`api/app.py::Components` 是唯一的
   状态，注入替身就能把 HTTP 这层单独测完）。
2. **重对象在 lifespan 里只建一次**（`graphs/wiring.py::build_heavy`）：embedder / reranker /
   chat / index。M3 真机踩过「每轮重建 → 一次问答加载三遍 1GB 模型」；服务里按请求建更致命。
3. **单个组件建不起来不让服务起不来**：记进 `heavy.errors`，用到的端点返 503 并带上原因
   （`{"error": {"code", "message"}}`），`/health` 照实报。整个服务因为一个缺 key 起不来
   是最难受的失败方式。

### 12.2 鉴权与限流

* **`X-API-Key`**（`.env` 的 `RAG_API_KEY`）。**没配 key 时只允许 127.0.0.1** ——
  默认放行等于裸奔，默认值必须是安全的那一侧。比较用 `hmac.compare_digest`。
* **限流**每分钟每身份 N 次（`api.rate_limit_per_min`），超了 429 + `Retry-After`。
  身份优先按 API key（同一把 key 的多个客户端共享额度）、否则按来源 IP；
  **key 原文不进键名**（sha1 前 12 位）。
* **Redis 是可选的**（`api.redis.url` 留空就没有）：有它时限流计数与任务状态**跨 worker 共享**，
  没有或连不上就**退回进程内**并在 `/health` 里写明「限流与任务状态已退回进程内」。
  两条都不是数据本身，所以选择降级（fail-open）而不是拒绝服务。

### 12.3 SSE 与流式

**没引 `sse_starlette`**（2026-10-03 定的：自己写）：我们在意的只有两件事 —— **心跳**（空闲连接会被
代理掐，常见 60 秒）与**断线取消**（客户端关页面后模型还在烧钱），这两件都在
`routes.py::_sse` 里显式做了：生产者把事件塞进队列，消费者 `wait_for` 超时就发一行
`: ping`；`finally` 里 `task.cancel()` 停掉生成。

真机验过：28 个 token 块逐块到达（不是攒完一次性发），`done` 事件带引用表。

**流式也走生成前 QueryGraph**（2026-10-23）：`stream=true` 会先完成同一套
`plan → retrieve → grade/replan → assemble`，随后才发出 `retrieval` 事件并开始
`generator.astream`。检索纠错发生在第一个 token 之前，不需要把已发出的 token 回滚；
`stream=true` 与 `stream=false` 只差最后一步的生成方式。

`/agent` 的流式（批次 B）复用同一套心跳与断线取消，事件换成
`token` / `tool` / `interrupt` / `done` —— agent 不是一条直线，中间会调工具、也可能停在
等人批准那一步，所以「工具结果」与「中断」都要能推给客户端。

### 12.4 已知取舍

* **`/ingest` 的路径是服务端本地路径**：不收上传流（multipart 上传是另一件事）。
* **多 worker 的共享状态已接上**（2026-10-23）：任务列表走 Redis 索引、灌库并发走
  Redis 槽位租约、VLM 描述缓存在 MinIO 下按 key 共享。模型仍是每 worker 一份 ——
  多 worker 会按 worker 数增加 embedding / rerank 的内存，这是当前明确代价。
* **`/ingest` 与检索同进程争资源**：`api.max_concurrent_ingest`（默认 1）限住同时在跑的
  灌库任务；重活仍建议走 CLI。
* **任务进度是粗粒度的**（排队/运行中/完成/失败 + 结果摘要），不是逐文件百分比。

### 12.5 持久化：LangGraph 的 checkpointer 与 store 走 PG（2026-10-04）

两条用途各自解决一个明确的问题：

| 组件 | 存什么 | 为什么不用内存 |
|---|---|---|
| **checkpointer** | agent 的会话状态、**HITL 中断** | 中断状态要**重启后仍在** —— 内存 saver 一重启，「等人批准」的那个中断就没了 |
| **store** | 跨会话记忆（`StoreBackend` / `memory`） | 记忆本来就该比进程活得久 |

配置写在 `.env`（机密与非机密分开）：`RAG_PG_URL` / `RAG_PG_PASSWORD`（机密，不写进 URL）/
`RAG_PG_SCHEMA` / `RAG_PG_POOL_MIN` / `RAG_PG_POOL_MAX`；yaml 里是 `persistence.postgres.*`。
**`url` 留空就退回进程内**（`InMemorySaver` / `InMemoryStore`）—— 与 Redis、OCR 产线同一个约定。

**schema 得我们自己管**（实测两件事）：

1. 那个库**只建表、不建 schema**（`setup()` 里全是 `CREATE TABLE IF NOT EXISTS`）→
   要先 `CREATE SCHEMA IF NOT EXISTS rag`；
2. 它的构造函数**不认 schema 参数**（`search_path` 一处都没有）→ 隔离靠连接串里注入
   `options=-c search_path=rag`，实测生效。

已建的表（6 张，全部 `IF NOT EXISTS`；`store_vectors` 只在给 store 配向量索引时才建）：
`checkpoints` / `checkpoint_blobs` / `checkpoint_writes` / `checkpoint_migrations` /
`store` / `store_migrations`。saver 与 store **共用一个连接池**（两者构造函数都收 conn/pool）。

> ⚠️ **Windows 上 psycopg 的 async 跑不了 `ProactorEventLoop`**（默认就是它）：
> `Psycopg cannot use the 'ProactorEventLoop' to run in async mode`。而
> `AsyncPostgresSaver/Store` 是 async-only，所以**只要接了 PG，入口就必须换
> `SelectorEventLoop`**（在创建循环之前设 policy）。
>
> **但 uvicorn 不吃这一套**（2026-10-05 真机实测）：uvicorn ≥0.36 不再看事件循环策略 ——
> `uvicorn/loops/asyncio.py::asyncio_loop_factory(use_subprocess=False)` 在 Windows 上
> **直接返回 `asyncio.ProactorEventLoop`**，`Server.run()` 再把它交给 `asyncio_run`。
> 于是 `ensure_compatible_event_loop()` 对 `rag serve` 完全无效，服务会在 startup 崩在
> 「psycopg 的 async 不能跑在 ProactorEventLoop 上」。**修法是给 uvicorn 一个显式 loop
> 工厂**：`rag.persistence.selector_event_loop`（先设策略、再 `new_event_loop()`），
> `rag serve` 里 `uvicorn.run(..., loop="rag.persistence:selector_event_loop")` —— uvicorn
> 对自定义 import 串会把它原样当工厂用（不传 `use_subprocess`），`--reload` 也照走。

**CLI**：

```bash
rag ingest ./data/docs --collection rag_v1
rag ask "问题" --show-sources
rag search "关键词" --channel hybrid --top-k 10
rag eval --dataset eval/golden.jsonl
rag serve
rag sessions list                     # 看有哪些会话（含「⏸ 等人批准」）
rag sessions prune --keep 5           # 只保留最新 5 个；--dry-run 先看要删什么
```

**CLI 的模块级 import 只留 typer 与配置**（2026-10-05）：重依赖（`rag.ingest` /
`rag.graphs` / `rag.indexing` / `rag.providers` / `rag.retrieval`）**在命令体里才 import**。
实测模块级全量 import 是 **12.0 秒**（`-X importtime`：`langchain_text_splitters` →
`sentence_transformers` → `torch` 一条链 5.5 秒；`rag.indexing.commit` 那条 9.2 秒），
于是 `rag --version` / `rag --help` / `rag sessions list` 这些用不到模型与切分器的命令
也得先等十几秒。下沉之后：**import 0.4 秒**、`rag sessions list` **24.6 秒 → 1.1 秒**、
`rag ingest --stage parse --dry-run` 4.2 秒。切分器那三处也照此办理
（`langchain_text_splitters` 的包 `__init__` 会顺带拉起 torch）。新增命令请照这个来，
`tests/test_cli_startup.py` 在子进程里钉住了这条规矩。

**命令组按领域分文件**（2026-10-23）：`cli/main.py` 只留核心流水线与命令注册，
模型选型、注册表、存储迁移分别在 `cli/models.py`、`cli/registry.py`、`cli/storage.py`；
Milvus 探活、错误翻译和异步执行抽到 `cli/runtime.py`。子模块只导出 `Typer` 实例，
不反向 import `main.py`，因此没有循环依赖，命令名、参数与帮助文本保持不变。

**API 路由同样按领域组合**：`api/routes.py` 现在是不到 30 行的聚合入口；
系统管理、检索问答、文档工作流、会话与 agent 分别在 `*_routes.py`。共享的响应转换与
SSE 报文在 `api/common.py`，限流只挂在聚合 router 一处，避免每个子路由各写一遍。
这种拆法让 OpenAPI 继续由同一个 `create_app()` 生成，路径与 schema 不随文件边界变化。

### 12.6 会话残留的清理：`rag sessions`（2026-10-05）

LangGraph 的 checkpointer **没有自动清理**：每聊一轮就多几版 `checkpoints`，另加
`checkpoint_blobs` / `checkpoint_writes`（实测一轮 SSE 问答就有 40+ 版、100+ 行待写）。
代码里已经撞见过一次 —— 真机验证跑完六个会话，写完得手工清。

**不做自动 TTL**：「等人批准」的那个中断也存这里。按时间自动删，可能把「昨天停下等批准、
今天才回来看」的那次删掉，而人毫无感知 —— 那正是最不该静默发生的事。所以只给显式命令：

* `rag sessions list` —— 每个会话的版本数、待写行数、**有没有在等人批准**、最新一版 id；
* `rag sessions prune --thread-id X` / `--keep N`（二选一）+ `--dry-run`；
* 默认**跳过**停在等人批准那一步的会话，要一起删得加 `--force`；id 写错直接报错退出
  （静默删掉 0 行最糟：人会以为已经清干净了）。

两个实现细节：

1. **「在等人批准」怎么判** —— HITL 的中断会以 `channel='__interrupt__'` 的行落在
   `checkpoint_writes` 里（实测：假模型 + 真 PG 跑一次中断）。顺手用它标记出来。
2. **「最新一版」怎么排** —— `checkpoint_id` 是 langgraph 生成的 **uuid6**（时间有序），
   字典序 ≈ 时间序，所以 `--keep N` 不用额外加时间戳字段。

顺带补上了一个缺口：**建表以前没有入口**（表是手工建的）。`rag sessions init` 建 schema
（库自己只 `CREATE TABLE IF NOT EXISTS`、不建 schema）+ 建表，幂等。

真机验证（2026-10-05）：造一个真停在中断的会话 → `list` 标出 `⏸ 等人批准` →
不带 `--force` 时跳过、`--dry-run` 不删、`--force` 删掉（6 行 writes / 1 行 blobs /
4 版 checkpoints）。

---

### 12.7 前端：`web/`（Vue 3 + TS + Vite，P1 + 工作台外壳已落地）

使用者定：**用 JS、选 Vue**。P1 = 检索 + 流式问答 + 引用点开看原文/图片。
方向定：**共用一套** —— 一套 token、一套组件，各页面只写主区内容（从三套 mockup 里选了「工作台」）。

```
web/                 Vue 3 + TypeScript + Vite（pnpm 管理；**不引 UI 组件库**，先跑通）
  src/styles/tokens.css  设计 token（明暗用 light-dark()，<html data-theme> 三态）
  src/styles/app.css     外壳与组件样式（颜色一律引用 token）
  src/state.ts           极简共享状态：page / detail / status / theme / apiKey（不引 pinia）
  src/api.ts             接口薄封装：带 X-API-Key、拼 URL、把 SSE 拆成事件、取图转 blob
  src/generated/openapi.ts  后端 OpenAPI schema 生成的 TS 类型（不手改）
  src/components/        NavRail / HitCard / LedgerTable / DetailPane / AssetImage
  src/views/             SearchView / ChatView / DocumentsView / LogsView（四个页面，**没有占位页**）
  src/App.vue            外壳装配：左导航 + 主区 + 右抽屉，主区按 page 切
  vite.config.ts     开发期把后端那几条路径代理到 127.0.0.1:8000（**不加 CORS**）
```

**API 类型只有一个来源**（2026-10-23）：`scripts/generate_openapi_types.py`
从 FastAPI OpenAPI 生成 `web/src/generated/openapi.ts`；`api.ts` 只保留类型别名与
SSE/UI 专用事件。CI 跑 `--check`，后端 schema 改了却没重新生成会直接失败。

**外壳的意义**：导航与详情抽屉只写一遍。加一个页面 = 在 `state.ts` 的 `PageId` 里加一项、
在 `NavRail` 的 `items` 里加一行、写一个 view；不用再动外壳。**管理台就这四页**
（检索 / 对话 / 文档 / 日志，日志页见 §13.4）：会话页 2026-10-08 决定不做（并进对话页），
评测页 2026-10-08 决定不做（留在 CLI，见 §13）—— 两者都不是「还没做」，导航里也不再留占位项；
原先那个 `PlaceholderView.vue` 随之删除。

跑法（`web/README.md` 有详细版）：`rag serve` 起后端，`cd web && pnpm.cmd dev` 起前端；
生产 `pnpm.cmd build` 后 `rag serve` **自动把 `web/dist` 挂到 `/`**（一个进程就够）。

**一条命令起两个**（`scripts/start.ps1` / `scripts/start.sh`，Windows / Linux 各一个）：

```powershell
scripts\start.cmd            # Windows（.cmd 只是包一层 -ExecutionPolicy Bypass）
./scripts/start.sh           # Linux / macOS
```

两个脚本行为一致：读 `.env` 的地址端口 → **按端口判断是不是已经在跑** → 有在跑的**问一句 y/n**
—— **y = 把占着端口的进程杀掉再启动，n = 放弃启动**（什么都不起）——
然后两个各起一份。`-DryRun` / `--dry-run` 只看它要做什么（不杀、不起）。

为什么 y 是「杀掉再重启」而不是「跳过已经在跑的那个」：跳过的话你面对的可能是**旧代码**
（这个项目上真踩过好几次，还以为自己重启过了）。要么干净重来，要么干脆别动。
杀之前一定先问；杀完**等端口真的空出来**再起 —— 进程退出与端口释放之间有一小段，
不等的话 `Start-Process` 会直接因为「地址已被占用」失败。`PID ≤ 4` 或脚本自己一律拒绝杀。

Windows 版还多一个 `-Backend`（只起后端，与 `start.sh --backend` 对齐）。
**踩过的坑**：参数 `$Backend` 与「解析出来的后端入口」变量**不能同名** ——
PowerShell 变量名大小写不敏感，撞上之后赋值哈希表会报
「无法将 Hashtable 转换成 SwitchParameter」（实测）。所以那个变量叫 `$backendEntry`。
Linux 版把日志落当前 profile 的 `<state_dir>/logs/`、PID 记 `<state_dir>/run/`
（例如停：`kill $(cat data/docker-single/run/backend.pid)`）；Windows 版各自开一个窗口，
窗口里的输出**同时 tee 进当前 profile 的 `<state_dir>/logs/backend-startup.log`**，
而且窗口**不再自动关**（`-NoExit`）—— 进程秒退的时候，栈既留在窗口里也留在文件里
（2026-10-10 加：那次配置校验失败一闪而过，什么都没抓到，见下面「启动失败那份日志」）。

写这两个脚本踩的坑（都挺典型，值得记）：

1. **`python` 不一定是装了本项目那个解释器**：直接 `python -m rag` 会因为
   import 不到 `rag` 一闪就退，窗口都来不及看。所以脚本按
   `RAG_PYTHON`（环境变量或 `.env`）→ PATH 上的 `rag` → 能 `import rag` 的 `python` →
   conda 各环境 依次解析，**并把最终用的入口打出来**；一个都找不到就明说怎么修。
   本机已把 `RAG_PYTHON` 写进 `.env`（`.env` 不进版本库；`.env.example` 里留了说明）。
   **入口是 `-m rag`，别写成 `-m rag.cli.main`**（2026-10-10 修）：后者会让 runpy 打一句
   `RuntimeWarning: 'rag.cli.main' found in sys.modules …`（`rag/cli/__init__.py` 里
   import 过它），而脚本把子进程 stderr 原样透出来 → PowerShell 把那一行渲染成一整片
   红字报错。正式的 `-m rag` 走 `src/rag/__main__.py`；`rag/cli/__init__.py` 也改成惰性
   出口，谁还敲老命令都不会再踩到。
2. **Windows PowerShell 5.1 按系统 ANSI（这台是 GBK）读没有 BOM 的 `.ps1`**：
   中文注释会被读成乱码、把引号吃掉 → `ParserError`。所以 `start.ps1` **存成 UTF-8 with BOM**
   （改这个文件时注意别把 BOM 弄丢）。
3. **`.cmd` 只能是 ASCII**：cmd.exe 按 OEM 码页读批处理，中文 `REM` 会被当命令执行
   （实测报「'程」），所以 `start.cmd` 里全是英文注释。
4. **Vite 默认只监听 `localhost`，在 Windows 上落成 `::1`** → 脚本印的 IPv4 地址
   连不上（`ConnectError`）。现在 `vite.config.ts` 与启动脚本共用
   `RAG_WEB_HOST` / `RAG_WEB_PORT`，默认显式绑 `127.0.0.1:5174`；
   `strictPort` 保证端口被占时直接失败，不会悄悄漂到下一个端口。
5. bash 版的 y/n 判断**先规整输入**（去空白与 CR）：管道喂进去的 `y\r` 不规整就匹配不上，
   表现成「明明是 y 却取消了」。

三个刻意的决定：

1. **流式用 `fetch` + `ReadableStream` 读 SSE，不用 `EventSource`** —— 后者不能自定义请求头，
   而 `/chat`、`/agent` 都要 `X-API-Key`；用 `fetch` 还能 abort（页面上的「停止」直接触发
   后端那套断线取消）。
2. **静态页不鉴权，数据端点全鉴权**：浏览器加载 `<script>`/`<img>` 时带不了 header，
   所以页面与构建产物是公开的（本地工具，无敏感内容），而 `/search`、`/chat`、
   `/chunks`、`/assets` 都走 router 上那道 `X-API-Key`（没配 key 时只允许本机）。
   **这条的代价是图片得自己取**：`<img src="/assets/…">` 一律 401，所以有 `AssetImage` ——
   带 key `fetch` 回来转 `blob:`，按路径缓存、key 一换整批作废（`interface.log` 里能同时看到
   同一张图的 401 `ip:…` 与 200 `key:…` 两条）。
3. **前端产物目录不能叫 `assets`**：后端有个 `/assets/{path}` 端点（取语料图片）挂在鉴权后面，
   浏览器的 `<script src="/assets/index-xxx.js">` 会**撞上它并拿到 401**（真机踩过）。
   所以 `vite.config.ts` 把 `assetsDir` 改成 `static`。
4. **API Key 在 `state.ts` 模块加载时就交给 api 层**，不能放在 `App.vue` 的 `onMounted`：
   子组件的 `onMounted` 比父组件先跑，检索页一挂载就自动检索 —— 等父组件才设 key，
   配了 `RAG_API_KEY` 的部署里**刷新页面第一眼必然是红色 401**（真机复现）。

踩过的另外两个坑：**pnpm 12 默认不许依赖跑构建脚本**（供应链防护），esbuild 的 postinstall
要用 `pnpm approve-builds --all` 放行（配置落在 `web/pnpm-workspace.yaml`，进版本库）；
**`vite.config.ts` 不该进 `vue-tsc` 的检查范围**（它跑在 Node 里、要 `@types/node`），
`tsconfig.json` 只 include `src/**`。

真机验收（2026-10-05）：`GET /` 200（前端页）、`/static/*.js|css` 200（不再 401）、
`GET /chunks/{id}` 200（正文 475 字 + 资产列表）、`GET /assets/3481c48bcb813900/p0007_01.png`
200（`image/png`，83KB）、`/assets/../.env` → 404（穿越被拦）。

改版后的真机验收（2026-10-05，Chrome + `rag serve` 托管 `web/dist`）：页面起来后三个依赖灯
全绿；检索「Flink 的 Save Points 是做什么用的」**8 条命中、无错误条**；命中卡里的图是 `blob:`
且**真实解码**（1383×416 / 1463×340 / 1465×737，`naturalWidth` 非 0 才算数）；
点开原文抽屉：标题层级 + `p11 · L1` + 916 字正文 + 两张缩略图 + 「打开原图」。

后面几批：**P2 已完成** —— 第一批（两条路 + HITL 批准）见 §12.8，
第二批（会话留在本地 + 流式 agent 落运行日志）见 §12.9；
下一步 **P3** 管理台（日志页、文档页、任务页都做完了，**会话页 2026-10-08 决定不做** ——
会话的列表/回放/继续/删除/每轮 token 对话页的下拉本来就有，见 `web/README.md`；
**评测页也 2026-10-08 决定不做** —— 它是一次性批处理、要过按量计费的线上模型，留在
`rag eval` CLI 里，见 §13）；**P4** any-to-any 的检索侧图片输入已经落地：
`POST /search/image` 负责 multipart 上传，检索框统一承载文字和图片附件；仍未做的是
把图片直接送进对话或 agent。检索框内的 Top-k 是唯一的数量选择入口，内部通道组合与
重排状态不暴露给用户；重排固定开启。

### 12.8 对话页：两条路、以及 HITL 批准（2026-10-05，P2 第一批）

**`/chat` 与 `/agent` 差在哪**（这决定了 UI 该给几个按钮）：

|  | `/chat` | `/agent` |
| --- | --- | --- |
| 谁决定查什么 | 我们（`plan → retrieve → grade →（不够就改写重查）→ assemble → generate`） | 模型自己，按需派子 agent |
| 对话轮次 | **单轮，没有历史**（请求里没有 `thread_id`） | 多轮，`thread_id` + PG checkpointer |
| 联网 / 写文件 | 都不行 | 行（联网有闸门、写文件必须人批准） |
| 一次问答的模型调用 | 1 次（开 QueryGraph 是 2~3 次） | 主 agent ≥2 次，派子 agent 每次都加 |
| 引用编号 | `assemble` 时发的，与 `hits` 一一对应 | 工具发号，只回「答案里真引用了」的 |
| 流式 | token 上屏；**流式时不叠 QueryGraph** | token + 工具事件 + 中断 |

注意 `/chat` 返回里的 `rounds` / `grade` 是 **QueryGraph 的重查轮次与「资料够不够」的评分**，
不是对话轮次 —— 别当成它能多轮。

**「什么时候切 agent」没有分数阈值**，这是刻意的：

1. 低分只说明「语料里没有」，而这件事后端已经管了（`agent.web_min_corpus_score: 0.2`，
   agent 内部的 `WebSearchGate` 自动执行）—— 前端再判一次是重复；
2. **多跳问题第一次检索的分数往往很高**（问题里的词能命中某一段，但答不全）——
   分数阈值恰恰抓不到真正需要 agent 的那类问题；
3. 只有开重排时这个分才可比（默认开），关掉时是 RRF 名次分。

所以落到三件事上：**手动开关**（默认 `/chat`）+ **一条硬规则**（本会话有历史就锁死 agent）
+ **一条事后提示**（快问答答完若最高分 < 0.2 或一条都没召回，给一个「让 agent 再查一遍」
的按钮，**点了才升** —— 自动升级是替用户决定多花三五倍的钱和几十秒）。

**后端为此补的三处**：

- `tool` 事件带 `args`（字符串截到 200 字）与 `ms`：参数在 `AIMessage.tool_calls` 上、
  不在 `ToolMessage` 上，靠 `tool_call_id` 对上；`ms` 的口径是「看到调用 → 看到结果」，
  含调度与子 agent 的时间，**不是工具内部计时**（对「这步值不值得等」更有意义）
- `/agent/resume` 支持 `stream=true`，事件与 `/agent` **完全一样**：
  批准之后那段答案照样是一段段生成的，客户端不该为「批准后」再写一套解析。
  已知小差异：恢复那次的 `args` 可能为空（那次调用的 AIMessage 在上一次运行里），
  但参数在中断载荷 `action_requests` 里已经有了一份
- **中断事件必须带 `thread_id`**：批准是「按会话找回中断状态」的，客户端在中断那一刻
  拿不到 id 就没法批准（原来 `_sse_events` 只透传了 `interrupt` 的内容）

**前端两个坑**（都值得记）：

1. **`turns.value.push(turn)` 之后再改 `turn` 不会触发渲染** —— push 进数组的是原始对象，
   而响应式的依赖挂在代理上。现象是「答案一个字不显示、`生成中…` 永远不结束」，
   而且只在别的 ref 恰好变了时才偶然刷新一次（`thread.value` 被赋值那一刻整页重渲染，
   所以看起来是"半好"的）。改法：`reactive()` 之后再 push。
2. 对话状态**必须放在共享状态里**：外壳是 `<ChatView v-if>`，切到「检索」再切回来组件会
   卸载重建 —— 状态留在组件里等于「点一下别的页，聊过的东西全没了」。

真机验收（2026-10-05，Chrome + 8001 端口新起的服务，真调模型）：agent 问答的时间线带
参数与耗时、`done` 带引用、答案框是最终答案；HITL 全流程「写文件 → 中断 → 拒绝」之后
`data/workspace/uicheck.md` **确认没有生成**，「批准」那次 `savepoints.md` 落盘 1803 字节。

### 12.9 会话留在本地（2026-10-06，P2 第二批）

会话列表**只存这台浏览器**（`localStorage`，key `rag_sessions`）：跨设备、带状态的列表
要等 P3 的 `/sessions` 接口，现在先不重复做。

```ts
{ id, title, updatedAt, mode, thread, turns }
```

四个决定：

1. **存盘前裁剪**：去掉命中池与通道账本（8 条命中 × 几百字正文，又大又只在当次可看），
   问答 / 引用 / 工具时间线留着 —— 重开对话要看得懂它当时干了什么。
   上限 20 段 × 每段 40 轮，配额满了静默放弃（不该把页面搞崩）。
2. **未批完的中断一起存**：重开时批准卡片还在，能接着批（中断载荷在 PG 的 checkpointer 里）。
3. **打开页面自动接回最近一段**，否则「刷新后会话还在」只是句口号；
   「新对话」必须能把当前这段先存下来再清空 —— 有历史就只能走 agent，
   不给出口用户就被锁在最贵那条路上（PG 挂了更是每个问题都 503）。
4. **会话 id 与 agent 的 `thread_id` 是两回事**：快问答没有 thread，但它也是一段对话，
   需要自己的 id。存进列表的轮次要**过一遍 `reactive()`** —— 见 §12.8 那个坑。

顺手补的两处：**流式 agent 也写运行日志**（`stream_agent` / `stream_resume` 收尾时补记，
跑完与停在闸门两种都算）—— 在这之前界面发起的每次运行在日志里都查不到（当时落 PG 的
`agent_run_log`，2026-10-06 起落 `agent.log`，2026-10-27 以后跟随当前 profile
落到 `<state_dir>/logs/agent.log`，见 §13.4）；
导航按钮补 `aria-label` / `title` —— 窄窗口下收起时只剩图标，按钮没有可访问名称。

真机验收（2026-10-06）：agent 问答 8 步时间线 + 7 条引用，答案分「能确定 / 可推断 / 没有」
三段；刷新后会话被接回来；新建+删除会话刷新后仍一致；PG 里查到
`thread=9434f8cfd986 steps=14 retrieved=36 cited=7 status=answered`。

### 12.10 会话落库：三张表（2026-10-06；2026-10-26 增加本地 SQLite）

**为什么不是 localStorage**：使用者要「换台机器、换台浏览器，会话都在」。
浏览器本地存储做不到，而且多实例部署时每个实例看到的会话都不一样。

**先分清两个「会话」**（名字撞得厉害，这段是给未来的自己看的）：

| 叫法 | 表 | 存什么 | 谁在用 |
| --- | --- | --- | --- |
| UI 会话 | `chat_session` / `chat_message` / `token_usage` | **给人看的对话记录**（human/ai 两条消息） | 界面、`rag chats` |
| agent thread | LangGraph 的 `checkpoints` 等 | **机器的记忆**（消息、中断状态） | agent 多轮、HITL |

只有 agent 那条路才有 thread；快问答没有，所以 `chat_session.thread_id` 可以是空。
两者靠这一列关联。`rag sessions` 那个 CLI 管的是**后者的残留清理**，与前者无关。

```
chat_session   一段对话：title / mode / thread_id / created_at / updated_at
chat_message   一条消息：(session_id, seq) 唯一，role ∈ {human, ai}，带 status
token_usage    一次问答一行：model / calls / input_tokens / output_tokens / total_tokens
```

四个决定：

1. **只留 human / ai 两条消息**。工具消息、命中池、通道账本都不进这三张表 ——
   它们是「这一次跑」的过程数据，要查有日志文件（`agent.log` / `channel.log`，见 §13.4）。
2. **中断那轮照样落**（`content` 空 + `status='interrupted'`），但**载荷不存**：
   写文件的参数里可能是一整篇正文，存两份没意义。读的时候现去 checkpointer 取
   （`pending_actions`），取不到就把状态降级 —— 已经批过的自然就没有卡片了。
3. **token 单开一张表**，不与对话记录混。粒度是「一次问答」：按会话 / 按天 / 按模型
   都能聚合；agent 一轮里发生几次模型调用，用 `calls` 记下来。
4. **存储随 profile 选**：local-lite / docker-single 使用
   `persistence.chat_sqlite_path`（默认 `data/<profile>/chats.sqlite3`），启动时自动建表并启用
   WAL；multiworker 使用 PG，仍可显式跑 `rag chats init`。写入失败不影响问答。

**一个真机才炸出来的坑**：连接池配的是 `dict_row`，`fetchone()` 返回字典。
第一版按逐位下标取，单测全绿（替身返回元组）而真库直接 `KeyError: 0` ——
现在每条 select 都起别名、按列名取，替身也改成字典行。

真机验收（2026-10-06，真 PG）：落一轮 + 落一轮中断 → 列表读到 `turns=2 /
total_tokens=1500 / calls=4` → `update_pending` 把中断那行补成 `answered`
（token 累加）→ 删会话后 `summary` 全归零（级联真的带走了消息与 token 行）。

本地 SQLite 回归（2026-10-26）：重建 store 后列表与详情仍在；
`created_at` 解析为带时区时间；中断行可由 `update_pending` 原地补齐；
删除会话后消息与 token 行都被外键级联清空。

### 12.11 问答落库：谁写、token 从哪来、引用怎么写（2026-10-06）

**服务端在问答结束时写**，客户端不回传任何东西：`/chat` 与 `/agent`（含 `resume`）
收一个可选 `session_id`，响应与 `done` 事件把它带回来。换台机器、换个浏览器，
会话都在 —— 这是把会话交给服务端关系库而不是浏览器存储的全部理由。

三个真机上才看清的点：

1. **先落库再发 `done`**。`_sse_events` 收到 `done` 就 break 并取消生产者任务，
   落库写在 `yield done` 之后会被取消 —— 第一次真机验证就是这么丢了一轮
   （列表里 `turns` 还是 1）。
2. **token 必须按轮算，不能按 thread 算**。`aget_state` / `ainvoke` 给的是整个 thread
   的消息，直接累加会把历史轮次重复计进去。非流式按「跑之前的消息条数」切片，
   流式**一边跑一边从分片里累加**（`rag/usage.py` 一处定义）。
3. **流式要显式开 usage**：OpenAI 兼容通道默认不在最后一个分片里带 `usage`，
   要 `stream_usage=True`。只对这条通道开 —— 原生集成不认这个参数。

**引用怎么写**（使用者定的）：不画成界面元素，而是在答案末尾列「引用了哪个文档的哪一章」。
`contracts.format_sources()` 一处定义、CLI 与界面共用，**拼在生成器与 agent 收尾那一层**
（不是前端，也不是路由层）：同一章节只列一行、编号并在一起，编号**沿用正文里的 `[n]`**——
去重时把编号丢了就成了悬空引用。`Citation` 为此加了 `heading_path`。

中断那轮照样落（`status=interrupted`、正文空），批准之后 `update_pending` **补同一行**；
落库失败绝不影响问答（与日志同一个立场）。

真机验收（2026-10-06）：`/chat` 落 `turns=1 / 1541 tokens`，同一 `session_id` 上的
`/agent` 追问落成 `turns=2 / mode=agent`，这一轮新增 `22030 / 501` token、`calls=4`。

### 12.12 前端不再自己存会话（2026-10-06，P2 收尾）

`state.ts` 里那套 localStorage 会话（key、裁剪、上限）**整段删掉**，只留
「当前在哪一段」这个 `sessionId` —— 它本身就是个 UI 状态，不是数据。

```
打开对话页   GET /sessions → GET /sessions/{id}（最近一段）→ 渲染
问一轮       POST /chat | /agent，body 里带 session_id
            done / interrupt 事件带回新的 session_id 与本轮 token 用量
新对话       只清本地状态；下一轮问答会给一个新的 session_id
```

三个细节：

1. **引用不画成界面元素**（使用者定的）：清单已经在答案正文最后一段，前端不再渲染
   引用按钮；每一轮下面显示「本轮 N token · M 次模型调用」。
2. **中断载荷在读的时候现取**：`GET /sessions/{id}` 发现某轮 `status=interrupted` 时，
   去 checkpointer 调 `pending_interrupt` 把载荷取回来 —— 会话表里不存它。
   取不到（已经批过了）就当没有，卡片自然不显示。
3. **逐轮的账要跟着消息回来**：`ChatStore.get` 按 `message_id` 把 `token_usage` 挂回
   每一条消息。不这么做的话，刷新之后每一轮的用量就看不见了 —— 只在当次有效，
   而那正好是统计功能最没用的形态。

真机验收（2026-10-06）：全新标签页打开 → 自动接回服务端会话（3 轮、61649 token），
逐轮显示 `1541 / 22531 / 37577`，引用按钮 0 个。

### 12.13 三条补丁：跨模式上下文、日志落点、不再用 API Key（2026-10-06）

**跨模式上下文**。`/chat` 是无状态的，历史只在我们自己的表里；切到 agent 时 thread 是空的
→ agent 只看得到当前这一句。改法：**第一次切到 agent 时把已有问答当初始消息灌进新 thread**
（`_history_for` + `run_agent` 的 `history`）。**已经有 thread 就不能再灌**（会重复）。
只取最近 8 条、跳过正文为空的；代价是历史进 prompt，本轮 token 会涨。

**日志落点**：这一版是「控制台 + 文件 + PG」三处；当天晚些又改成**三类各落一个文件、
不再写库**（见 §13.4），这里只留历史。**2026-10-08 又加回一张 `log_line`**，但只当索引、
文件仍是唯一真相（见 §13.4 末）。内容仍受 `logging.content` 管 —— 默认只打 hash。

**鉴权拿掉了**（使用者定：不再用 `RAG_API_KEY`，「注册后即可使用」）。
`require_api_key` 留着没删，用户系统那批在这里换成 token 校验。`chat_session.owner`
现在就加上（默认 `'local'`，只写不过滤），并有配套的幂等 `alter … add column if not exists`
—— `create table if not exists` 不给已有的表加列，这条是给已建过的库用的。

### 12.14 日志改用 stdlib `logging` + 工具错误中间件（2026-10-06）

**日志全部交给 stdlib `logging`**（之前是自己写的一套，代价是第三方库的日志完全不可见）：
一套 handler / formatter / 级别 / 轮转，我们自己的和第三方库的都在里面。装 **root**
logger（第三方库要冒泡到 root 才接得住），`StreamHandler`(stderr) + 按类别各一个
`RotatingFileHandler`，`CorrelationFilter` 把 `request_id`/`run_id` 注进**每一条 record
（含第三方库的）**。业务账（接口 / agent 运行 / 通道）也走这条线 —— 一行一条，
落点见 §13.4（**2026-10-06 起不进 PG**）。

三条约定：控制台**就一个出口**（stderr，进度与异常都要看得见）、默认 **INFO**、
第三方库压到 **WARNING**。`rag serve` 在 uvicorn 之前装日志并 `log_config=None`
（否则 uvicorn 自己再配一遍 handler，一个进程里两套格式）；**每个 CLI 命令都装**
（根回调里 `configure_logging`）—— 少了这一步，INFO 会因为没有 handler 被直接丢掉。

**工具错误中间件**（`agent/middleware.py`）：工具抛异常 → ERROR（带栈、工具名、参数预览）
+ 一条 `ToolMessage(status="error")` 交给模型继续，而不是让整轮问答挂掉。
挂在 `create_deep_agent(middleware=[...])`，`awrap_tool_call` / `wrap_tool_call` 两个入口都拦。
最实际的收益是**参数校验失败**（模型把参数写错时 LangChain 抛 `ValidationError`，以前整轮挂）。

**级别口径**（使用者定的）：*影响到了这次问答的可信度* → ERROR（工具失败、会话落库失败）；
*换条路照样跑通* → WARNING（OCR 产线切云端兜底）。

### 12.15 对话页显示的两处调整（2026-10-06）

**1. 不再展示工具调用步骤。** 原来每条 agent 回复下面挂一个「工具调用（n 步）」的时间线；
现在去掉 —— 对看答案的人来说那是中间过程，而且 6 次模型 × 7 次工具的清单比答案还长。
**过程信息没有丢**：`agent.log` 现在就是一条时间线（`模型#n` / `工具#n` 各一行，见 §13.4），
要查「它查了什么、哪一步慢」去日志页看。前端也不再收 `tool` 事件的字段
（SSE 里还有，留给日志与后续页面；`ToolTimeline.vue` 一并删了）。

**2. 答案按 markdown 渲染，但只在跑完之后渲染。** 模型写的是 markdown（标题、列表、表格、
粗体、代码），纯文本贴着看很难受；但**流式期间逐 token 重解析会卡界面**（半张表格渲染出来
也是花的）。所以：流式那一段仍走纯文本 + 光标，`done` 之后再解析一次，并用 `WeakMap` 按正文
缓存（下一轮流式时不会把前面几轮的 markdown 再解析一遍）。

渲染器是**手写的**（`web/src/markdown.ts`，零依赖，约 150 行）：本项目前端不引 UI 库，
而为多认几种边角语法去联网装 `marked` 不划算。做法是**先转义再拼标签** —— 进来的是模型
输出（不可信），`<script>` 之类只会变成文字；链接只放行 `http(s)://`。要换成
`marked` + `DOMPurify` 只需替换 `renderMarkdown` 这一个函数。

**一个例外：内联 HTML 表格**（2026-10-08 加）。PDF 走 OCR 时产线给的表格是 HTML
（`cleaning.py` 有意保留：竖线表表达不了 `rowspan`/`colspan`），而「先转义」的渲染器会把
它显示成标签文字 —— 实测 dots 解析的四份 PDF 里 `<table>` 计数 1/1/78/0、markdown 表行
全 0，而 html 那几份都是正常的竖线表。所以整块 `<table>…</table>` 交给 `safeTable`：
先用 `DOMParser` 解析成 DOM，**再按白名单重建**（只留表格结构标签 + `rowspan/colspan/align`，
其余标签只取文字、属性值也转义）。这不是"放行一段 HTML"，输入是用户文档经 OCR 出来的
文本、不可信；顺带也覆盖了解析稿预览与模型偶尔吐 HTML 表格的问答答案。

**「把表格转成竖线表」这条后端路线是明确不走的**（使用者 2026-10-08 定）：合并单元格在
竖线表里表达不了（压平或拆列都有损），而且改清洗规则要 bump `LOADER_VERSIONS["pdf"]`、
让已灌的 PDF 全部作废重解析。清洗那一步已经给表格行边界补了换行，切块切在行之间，
HTML 留在正文里并不影响切分质量 —— 缺口只在显示端。

**3. 引用编号可以点了**（2026-10-07）。正文里的 `[n]` 与末尾「引用来源」清单里的编号都做成
按钮，点了在右侧抽屉看原文（复用检索页那套 `openChunk`）。点击用**事件委托** ——
这段 HTML 是 `v-html` 塞进去的，Vue 绑不上它里面的元素。

**只有「当轮」的引用能点**：`[n] → chunk_id` 的映射来自这一轮 `done` 事件里的 citations，
**不入库**（`chat_message` 只存正文；来源清单是正文的最后一段）。所以从历史恢复的那几轮，
`[n]` 保持纯文本 —— 渲染时按「有没有映射」决定做不做成按钮，不做点了没反应的东西。
要让历史里也能点，得给会话表加一列存 citations（下一步的选项）。

**4. 读会话失败不再静默**（2026-10-07）。挂载时那段「自动接回最近一段」原来**没有 catch**：
拉取失败就停在空态，看着正是「点会话没有会话历史」。现在有加载态（"正在读取会话…"）、
带原因的失败横幅，以及「这一条会话一条消息都没有」的明确提示（那种多半是那几轮落库失败过）。

### 12.16 文档页：上传 → 解析 → 核对 → 切分入库（2026-10-07，P3 第一批）

页面那五步落成六个端点（表在 §12），前端是 `web/src/views/DocumentsView.vue`（左列上传件、
右侧工作区）。**任务页（`JobsView.vue`）2026-10-08 删掉了**：它只剩「看 CLI 灌库任务」这点
独特价值（文档页的流程图已经按 doc_id 显示进行中与失败），而 `JobStore` 是进程内 + Redis TTL，
重启即空、当不了排查用；`GET /jobs` 接口保留（文档页流程图靠它）。四个决定值得记下来：

**1. 上传只落盘，解析是另一个 job。** 上传是秒级、免费的；解析（PDF 走 OCR + 版面模型）
是分钟级、按量计费的。绑在一起就意味着「传错一个文件也得先把钱花了」，而且页面上看不到
文件清单。所以：`POST /documents/upload` 落地即返回，`POST /documents/{id}/parse` 是
**202 + job_id**，进度靠 `GET /jobs` 轮询。

**2. 慢步骤一律下线程池。** `run_parse` / `run_chunk` 与 OCR 引擎都是**同步**实现，直接在
事件循环里 await 会把整个 API 卡住几分钟 —— 那时页面连 `/jobs` 都轮询不到（自己把自己锁死）。
`Components._parse_one` / `run_index` 里两处 `asyncio.to_thread` 就是为这个；`run_ingest`
那两个 `run_parse` / `run_chunk` 也一起补上了（原来漏了，是这一批顺带修的真 bug）。

**3. 「修正」= 只改层级，正文一个字都不许动**（使用者 2026-10-07 定的）。
`PUT /documents/{doc_id}/outline` 的请求体里只有 `level`、`max_level`、`confirmed` ——
**连标题字段都没有**，因为标题是解析稿里的原文。这不是省事，是必须：正文一改，
`Document.positions`（偏移 → 页码/行号）就全错位，而那份映射是「点引用看原文」的根。
落盘的 `ArtifactStore.save_outline` 也只写 `outline.tsv` 一个文件，写入前先确认
`document.json` 还对得上（换了文件 / 换了解析器就 409，不留下一条挂在旧解析上的清单）。
要改文章内容，改源文件重新上传。

**4. 核对是硬门槛，且两道。** `POST /documents/{id}/index` 先看清单是不是「已核对: 是」，
不是就 **409**（页面立刻知道差哪一步，比等 job 失败清楚）；`run_chunk` 自己也会把没核对的
挡进 `pending` —— 不靠单点。层级改动会让 `outline_fingerprint`（= `levels|max_level` 的
sha1）变化，`chunks.json` 记着这个指纹，于是**改了层级，旧切分产物自动作废**，下次
`index` 会重切；页面上的「已切分」灯也跟着灭。

### 12.17 从链接导入：一个链接，类型不定（2026-10-07，P3 第二批）

使用者要的第二个场景：**文档来源不止「上传文件」**，还有「给一个链接让后端去拿」。
链接指向的东西**类型不定** —— 可能是一张文档页面（HTML），也可能是 PDF / Word 的下载链接。

**抓回来先落成文件**，之后与上传件**完全同构**（`src/rag/ingest/fetch.py` +
`POST /documents/fetch`）。这不是图省事：下游整条链路（doc_id、stages 产物、增量比对、
Milvus upsert）全部按**文件路径**走，先落盘就一行下游代码都不用改，而且「重新抓一次」
天然等于「重新传一份」。

四个决定：

**0（2026-10-10 补）：网页可以交给 Firecrawl 抓。** `ingest.uploads.fetch.backend`
（`${RAG_FETCH_BACKEND:-http}`，**默认 http = 本机抓**）切成 `firecrawl` 时，**网页**改由
Firecrawl 的 `/scrape`（`onlyMainContent`）给 markdown —— 导航、页脚、"Copy page" 这类
chrome 不进正文（实测同一页：本机 HTML 抽出来 `Copy page` 出现 4 次，Firecrawl 0 次）。
三条边界：① **PDF / 图片 / 附件仍走本机**，那几条路要的是**原件**（版面模型给标题层级、
抽图、页码行号），只给 markdown 会把这些全丢掉；② **图片仍由我们下到本地并改写相对路径**
（`_localize_markdown_images`），否则 assets → VLM → 多模态 embedding 那条线就断了；
③ 云端**按量计费**，所以默认关、切过去是显式决定，key 只从 `.env` 读。实现走 REST +
自己的 httpx 薄客户端（不装 `firecrawl-py`），与 dots / mineru / OCR 同一套写法。

**1. 后缀按「`Content-Disposition` → URL 末段 → `Content-Type` 反查」定**，三者都定不出来
才拒（400，并把服务器给的 Content-Type 原样写出来）。反查用的是 `loaders/base.py` 里
`_MIME_BY_SUFFIX` 生成的**反向表**（`suffix_for_mime`），不另立一份名单 —— 和
`uploads.supported()` 去问 `LOADERS` 是同一个立场，加了新格式只改一处。名字带了后缀但
不在名单里（`.doc` 那种老 Word）**直接拒绝**，不能偷偷按 `.docx` 试。

**2. SSRF 是这里唯一的新风险**，三条对策缺一不可：只允许 http/https；每一跳主机解析出的
**每个** IP 都必须是公网地址（私网 / 环回 / 链路本地 / 保留段一律拒绝，
`169.254.169.254` 就是云上取凭据的经典目标）；**重定向自己走**（`follow_redirects=False`）
并且每跳重查 —— 交给 httpx 自动跟，`http://evil/ → http://127.0.0.1:8000/` 就绕过去了。
自建内网文档站要抓就开 `ingest.uploads.fetch.allow_private_hosts`（默认关）。

**3. 抓取同步做、丢线程池**（`asyncio.to_thread`）：抓一页是秒级，超时 15s
（`fetch.timeout_s`）—— 为它排一个后台任务，反而要另找地方看结果。httpx 是同步的，
直接在事件循环里跑会把 API 卡住（与 §12.16 第 2 条同一个坑）。

**4. 落盘名带内容哈希，真名另存**（使用者 2026-10-07 定）：
`<真名去扩展名>-<sha1 前 16 位><扩展名>`，哈希就是 `ingest.incremental.file_hash` 那一个
（**文件名里的哈希 = 解析产物是否过期的钥匙**，不是另发明的一个）。于是：

* 同一份内容重传 / 重抓 = 同一个文件名 = 同一个 doc_id，**静默幂等**（`replaced=true`），
  已解析 / 已核对的产物照旧有效；
* **同名不同内容 = 两份文件**，谁也不盖谁 —— 旧的那几份放进响应里的 `conflicts`，
  页面问一句「覆盖更新还是保留两份」。不自动覆盖是故意的：改了几个字就悄悄替掉旧版，
  人会以为「更新了」，其实旧版连同它的人工核对结果已经没了；
* 「覆盖更新」= `DELETE /documents/{旧 doc_id}`，走 `indexing.commit.drop_docs`
  （**先删 Milvus 的块 → 再删注册行 → 清 stages 产物 → 最后删文件**，顺序不能反）。
  **不可逆**，失败时什么都不删并回 502 —— 留一个能重试的状态，比删了一半强。

真名与来源 URL 记在 `<dir>/_index.json`（`display_name` / `sha1` / `source_url` /
`final_url` / `created_at`），页面显示真名而**不去猜「把 -hash 切掉」**（真名本身就可能带
`-1a2b3c4d5e6f7a8b` 这种尾巴）。这份索引是附属信息：删了只会让老文件退回「展示名 =
文件名」，所以读写都按「坏了就当没有」处理。加哈希**之前**传的那批文件不改名
（改名会换 doc_id、把已入库的块变成孤儿），它们的展示名就是文件名。

已知边界两条：**只做静态抓取**，不跑 headless 浏览器（JS 渲染的页面抓回来基本是空的，
由 `load_paths` 报「解析后正文为空」）；**需要登录才能下载**的链接会拿到 302/403，
如实报出来（不做 cookie 注入）。

**5. 抓 HTML 时连页面里的图片一起抓**（2026-10-07 补，使用者报「url 解析之后的图片没有
拿到」）。当时实测：抓下来的三个文档页里有 11 / 4 / 5 个 `<img>`，而解析稿里的图片标记是
**0 个**、`data/assets/` 下也没有 HTML 文档的目录 —— 两处缺口叠在一起：`fetch` 只下 HTML
本体，`HtmlLoader` 又从不处理 `<img>`。

现在：`ingest/fetch.py` 按 `loaders.html.content_image_urls`（**与 loader 同一套「哪些图
算正文」判定**：同样先选 `main`/`article`、同样先清掉 `nav`/`footer`）把 `<img>` 逐个下下来，
落在同级的 ``<真名>-<页面内容哈希>_files/`` 目录，并把 HTML 里的 `src` 改写成相对路径 ——
形态就是「浏览器另存为网页」，`discover()` 本来就跳过 `*_files/`，不会被当成独立文档。
目录用**页面内容哈希**：同一链接内容变了，新旧两份 HTML 各带各的图，互不踩。
每张图复用同一套 SSRF 守卫（**每一跳**都查）、超时、单图上限，另有 `fetch.max_images`
（默认 20）—— 图片描述是 VLM 调用的**按量计费**那一步，这两个是钱的口子。

`HtmlLoader` 一侧把 `<img>` 换成 ``![image](asset:…)`` 标记（`store.save` + `make_marker`，
与 docx / OCR **同一套出口**），下游的 VLM 描述与多模态 embedding 不必分来源。三处细节：

* **先看 `data-src` 这类懒加载属性，再看 `src`** —— 懒加载站点往 `src` 里塞的是 1×1 占位图；
  改写时**地址原本写在哪个属性就改哪个**，只改 `src` 的话 loader 读到的还是占位图。
* **取不到的图只记账、不废掉整页**：网页里混着图标、徽标、外链图，一张 404 不该让整份
  文档 skipped（与 docx / OCR 那条「响应里本该有这张图」的严格报错不同）。原因进
  `Document.metadata["images_missing"]`，同时由 `/documents/fetch` 回给页面 + 写一条
  `rag.ingest` 的 WARNING（进 `system.log`）。
* **不做 SVG**：一页十几个内联 `<svg>` 加两个 logo，`image_suffixes()`（从 `_MIME_BY_SUFFIX`
  生成）里没有 `.svg` —— VLM 描述与多模态 embedding 都是按栅格图设计的。跳过的原因照实写出来。

实机验收（2026-10-07，`https://docs.spring.io/spring-ai/reference/1.0/concepts.html`）：
抓下 7 张正文图（合计 ~2.4MB，`img01.jpg`…`img07.jpg`），解析稿里 7 个图片标记、`assets` 下
7 份资产；只有两个 Algolia 的 svg logo 被跳过并写明原因。

花钱的口径：抓取本身不花钱，但抓回来是什么决定后面花不花 —— HTML / Word / 表格是本地
解析，PDF 走 `ingest.pdf.parser`（默认 dots，打本机那个 vLLM 服务），图片走 OCR。
哪天把 dots / OCR 换成云端服务，同样的操作就开始按量计费（配置注释里也写了这句）。

### 12.18 层级核对：就地改 + 左右对比（2026-10-07，P3 第三批）

使用者提的两点：**标题层级要嵌在文档里改、并且能「删掉」这一级（标为正文）**，以及
**要和未切分前对比，左右两栏**。这一批把 §12.16 那张清单表换掉了。

**1. 第一列多了一个值 `0` = 这一行不是标题，算正文。** 解析（尤其 OCR）常把一段普通文字
认成标题，人一眼就知道那不是 —— 但**正文一个字都不能改**（改了偏移 → 页码/行号全错位），
所以改的是它的**角色**：写 0 的行不进标题路径、不单独成节，那行原文（含 `#`）留在正文里。
这与「比 `max_level` 深的行」是同一种处理，区别只是人事先明说了它压根不是标题。

三处实现要点：

* `parse_outline` 认 **0..6**（其余照旧报错并指行号），`render_outline` 的头部注释里写明
  0 是什么意思 —— CLI 那边读同一份文件，不另搞一套；
* `normalize_levels` 让 0 **原样穿过**，既不参与压平、也不改变「上一条是几级」——
  否则 `1, 0, 2` 会被算成 `1, 1, 1`（0 被 `max(raw,1)` 顶成 1 级，下一条又被压进 2 级）；
* `sections()` 的边界条件从 `level <= max_level` 改成 `1 <= level <= max_level`。

删行是**不能**选的：清单必须与正文的标题行一一对应，少一行整份文档就切不了（那条校验
是故意的）。标 0 才是「只改角色、不动正文」。

**2. 前端两栏**（`DocumentsView.vue`）：左边解析稿、右边试切结果。层级编辑**嵌在文档里**
（标题行就地一个层级下拉 + 「标为正文」按钮，正文段落照常渲染），原来那张清单表撤掉 ——
它和这个只是同一件事的两种呈现，而表里看不到上下文。

**右栏的「试切预览」2026-10-08 撤掉了**（使用者定），换成**原文**（同一天做完）：
左边点一行、右边跳到原文那一块。

配套的 `POST /documents/{id}/preview`（按**草稿**层级试切、纯本地零成本、不落盘）也跟着
**删了**（2026-10-08，同一天定的）：页面不再调它之后就没有消费方了，留着只会让人以为还在用。
它当年不跑**真** chunker 的理由留在 `git log` 里 —— `indexing.chunk.strategy` 是
`markdown-semantic`，语义兜底要算句向量（线上模型、按量计费），给预览花这个钱不划算。
要看真的切法，跑完第五步看 `chunks.md`。

**「点左边 → 右边定位」试过、当天又整个删掉了**（2026-10-08，使用者两次改主意后的定论）：
**不做映射** —— 右栏就是**原文**：PDF 交给浏览器内置查看器（**上下连续滚动、不分页**），
其余格式整份看。做过的那些（下面 1–3 条）连同接口、模块、配置、测试一起删了 ——
`/documents/{id}/page/{n}`、`/documents/{id}/locate`、`ingest/pdf_pages.py`、
`ingest.page_view` 配置、大纲行上的页码字段，全部不再存在。

**「全屏对比」（2026-10-08 加）**：使用者说对比框小。工具栏上一个按钮把**工具栏 + 两栏**
一起铺满窗口（再点一次或 Esc 退出）。实现上刻意**只换定位、不改 DOM**
（`.review.full { position: fixed; inset: 0 }`）—— 左边下拉里已选的层级与没提交的改动都不会
因为进出全屏而丢，**提交按钮也在全屏里**（不用为了保存退出来）；全屏时两栏各自撑满高度、
各自内部滚动，并把 `source-box` 与 iframe 平时那个 `56vh` 的定高放开（不然原文会被压住），
同时锁住背景滚动。

留个记录，将来真要捡回来时不必重新踩一遍（**其中两条是真机踩出来的坑，值得先看一眼**）：

1. **自己渲染页面**（`ingest/pdf_pages.py` + `GET /documents/{id}/page/{n}`）。为什么不把 PDF
   丢给浏览器内置查看器：那是个插件，页面上没有能叠高亮框的 DOM。渲染结果按**源文件哈希**
   缓存（`data/cache/pages/<原名前 12 字>-<sha1>/p0003.png`）。
   两条尺寸纪律是**真机踩出来的**：大开本的扫描件在 144 DPI 下 PNG 无损耗存下来一张
   6.5MB / 18MB，点一下就下十几兆 —— 所以**长边封顶 1800 像素**，且 **PNG 压不下去就转
   JPEG**（文字页仍是无损 PNG）。复测：6570KB → 575KB、18172KB → 232KB。
2. **定位到「块」**（`GET /documents/{id}/locate`，`locate_block`）。三级兜底：
   `get_text("blocks")` 里哪一块含这段文字 → `search_for` 的行级命中并成外接矩形 →
   都没有就 `found=false`。**块级不是行级**是使用者定的（比逐行稳，也够回答问题）；
   比之前先把空白压掉，否则 OCR 的 `2.1 Flink` 与文字层的 `2.1Flink` 互相认不出。
3. **大纲每行带原文页码**：拿 `document.json` 里「正文偏移 → 页」的打点反查。
   **偏移在 `doc.text` 上数**，不在 `document.md` 上 —— md 前面多一段 `<!-- … -->` 头信息，
   拿它算会整体偏掉。坐标一律归一化（0..1），渲染 DPI 与显示尺寸变了都不用改前端。

**一条要记住的限制**：**扫描件定位不到**。那类 PDF 没有文字层（真机实测三份扫描件首页
文字层是 0 字符），既搜不到文字也没有块坐标 —— 页面只会跳到那一页并在横幅上说明。
有文字层的那份实测精确定位成功（框住标题块，`y 0.058-0.087`）。

**记一条优化项**（使用者 2026-10-08 定：先搁置）：**扫描件要能高亮，得在解析时把 OCR 的
bbox 存进产物** —— `Position` 加坐标字段，`dots` 的响应里本来就带 bbox（见 `dots.py` 的
说明），只是 loader 现在只留了 page/line。代价：产物格式变了，**已灌 PDF 全部作废、需要
重解析**（那几份扫描件多是 1 页，秒级；Kafka 那份 396 页另说）。当前状态：不做映射，
所以对正在用的功能没有影响。

左边**按 markdown 的样子渲染**（2026-10-07 使用者要求）：标题行的字号/粗细跟着**当前选的
层级**走（h1..h6），标成正文（0）的按正文渲染（灰、常规字重），层级下拉与「标为正文」收在
行尾、平时半透明。改的原因是：原来每行都是一个带虚框的表单行、字号还一样，看不出层级 ——
左边本来就该是那份文档本身。

一个必须说清的实现细节：**左边是按行切段渲染的，不是「整篇渲染完再从 DOM 里数 h1/h2」**。
后者会错位：代码块里的一行 `# 注释` 在后端算标题行（`heading_lines` 是逐行正则），
在渲染结果里却是代码 —— DOM 序号与清单序号就对不上了。所以前端用**同一条正则**
（`^(#{1,6})\s+(.*)$` 且标题非空）把解析稿切成「标题行 / 正文段」，标题行渲染成带控件的行，
正文段才交给 `renderMarkdown`。

顺带两处：`document.md` 顶部那段 `<!-- … -->` 头信息（doc_id / source_uri / parse_key /
图片清单）在 API 这一层剥掉 —— 它不是正文，留着会让右栏多出「第一节：doc_id 2e1139…」；
解析稿里的 `![image](asset:…)` 标记现在直接渲染成图（鉴权那层 2026-10-06 拿掉了，所以能
直连 `/assets/…`）。

实机验收（`data/stages/5c9999efeb5bf8f9`，Spring AI 的 concepts 页、11 个标题行）：
`max_level=3` → 12 节 / 15103 字；`max_level=2` → 9 节 / 15185 字（深的那几行并进上一节，
所以节更少、正文合计更多）；把第 2 行标成 0 → 11 节，那行原文出现在第 1 个标题节的正文里。

### 12.19 文档页改版：主流程 + 阶段面板（2026-10-07，P3 第四批）

使用者要的：右栏**上面是主流程**（像审批流那样，没到=灰、到了=蓝、进行中=半蓝半灰的圆环）、
**下面是当前那一步的展示与操作**（上传看原文、解析转圈、核对左右对比、拆分入库给块数）。

**上：流程图四节点**（上传/导入 → 解析 → 核对标题 → 拆分并入库）。**不拆成五个**：
后端的「拆分」与「入库」是**一个 job、不报子阶段**，分开画就只能靠猜。三种状态的颜色
就是全部信息（CSS `conic-gradient` 做半蓝半灰 + `@keyframes` 转，不引 UI 库）；点已完成的
节点能回去看那一步的结果，不改任何状态。

「进行中」得有**后端依据**，否则刷新页面就丢了：页面每 2.5 秒拉一次 `GET /jobs`，按
`request.doc_id` 认出这份文档正在跑哪一步。任务落地时顺手把阶段状态与摘要刷新一遍 ——
不用让人手动点刷新。任务失败的原因也从那一条里取出来摆到横幅上。

**下：四个阶段面板**，操作按钮就在对应面板底部（原来那排按钮被拆开了）：

| 阶段 | 进行中 | 完成 |
|---|---|---|
| 上传 / 导入 | 抓取中… | 大小 · 哈希 · 落盘时间 · 来源 · **原文** |
| 解析 | 循环「解析中…」 | 字数 · 图片 · 解析器 · MIME · 解析于 · 耗时 · 解析稿 |
| 核对标题 | — | §12.18 那套左右对比 |
| 拆分并入库 | 循环「拆分入库中…」 | 块数 · 分布 · 大纲 · 产物大小 · 入库账 |

**原文怎么给**：新端点 `GET /documents/{doc_id}/source`。两处刻意的选择：**只认 `doc_id`**
（路径由上传存储查出来，不接受任何路径参数，所以不存在「顺着参数读到目录外面去」）；`text/*`
里**只有 html 原样给**，其余降级成 `text/plain`（浏览器遇到 `text/markdown` 是下载而不是
显示）。HTML 走 **`<iframe sandbox="">`**（不带 `allow-scripts`，脚本与表单全关掉）+ 响应
头的 `CSP: sandbox` 与 `nosniff` —— 抓回来的是第三方页面，这是唯一有安全面的地方。
PDF 交给浏览器内置查看器，图片给 `<img>`，纯文本前端取回来放 `<pre>`（截断到 20000 字）。

**各阶段的数字是现算的**（`ingest/summary.py` 的纯函数：`parse_stats` / `chunk_stats` /
`section_rollup` / `vector_bytes`），读两份产物 json，**不写进产物格式** —— 报表不是数据，
多存一份就多一处会不一致的地方。详情接口因此多了 `parse` / `chunks` / `index` 三份摘要。

其中两样是**专门给人一眼看的**：**块大小分布条**（每块一条宽度正比字数的色条接成一长条 ——
「平均 740、最大 1037」不如一张图直观，碎块与巨块一眼就看得出来）和**章节树**（按
`heading_path` 聚合的「哪几节、各多少块多少字」；393 条块列出来没人看得完）。

章节树的做法：`heading_path` 本来就是**从根到该节的完整路径**（`章 -> 节 -> 小节`，
由人工核对过的清单拼出来），按 ` -> ` 拆开、同一串前缀归到同一个父节点，渲染时按深度
缩进 —— 前端摊平成「带层级的行」再渲染（按文档顺序，不用递归组件）。
**默认全折叠**（2026-10-07 使用者定的），点行首箭头展开：668 节那种树默认铺满一屏没法看。
注意实测多数文档只有一个根节点，所以默认那屏只露 **1 行**（那一行还可能是个"自己没有
正文"的容器，块/字显示 `—`）。所以**带子节点的行另外写一行「含子项 N 块 / M 字」**
（2026-10-08 使用者定）—— 折叠状态下靠它看出整章多大。
两处边界：**父节点自己没有正文是正常的**（标题下面直接跟更深的标题时就是这样，实测那份
Kafka 2.5 PDF 的 668 节里有不少），那种行显示 `—` 而不是 0；路径为空的那一节
（第一个标题之前的正文）单独作为根节点。

**过期的产物照样给**（`stale` + 记的哈希 vs 现在的哈希）：只报「没到那一步」的话，步骤灯
会毫无理由地回退，看的人不知道是自己改坏了还是换了配置。同理，解析阶段那张
`images_missing` 清单**常驻**在面板上，不是导入时闪一下就没。

顺带修了一个真错：`/documents/{id}/index` 的 job 结果里 `chunks` 报的是**文档数**不是块数
（`run_chunk` 的 `chunked` 计数器数的是文档）—— 393 块的文档在任务记录里显示「1」。现在
`chunks` 是块数，另外补了一个 `documents`。

实机验收（真上传件 + 真 Milvus）：`concepts.html` 解析 15669 字 / 7 图 → 切分 29 块
（正文 22 / 图 7）/ 21472 字 / 平均 740 / 最大 1037 / `chunks.json` 49KB → 注册表 29 块
**对得上**、向量估算 0.3MB（2560 维）；原文接口 200 `text/html` 60703 字节；
`getting-started.html` 停在「已解析未核对」，流程图第三节点亮着。

**三个坑（2026-10-07 当天踩到并修掉，值得记）：**

1. **别靠「抓到任务状态变化的那一瞬间」来刷新页面。** 原来的实现等 `running → done`
   的跳变才重拉详情 —— 而**浏览器会把后台标签页的定时器降频到一分钟一次**
   （`interface.log` 里 `/jobs` 的间隔从 2.5 秒变成整 60 秒，就是这个）。用户切到别的窗口
   等任务跑完，回来看到的就是旧状态：按钮还亮着、结果不出来。现在改成**幂等的判断**：
   每次轮询比「这份文档最近一次任务的时间戳 vs 手上这份详情的时间戳」，新了就重拉；
   另外 `visibilitychange` 时立刻刷一次。
2. **入库状态有两个信号源，别混成一样。** 列表那一份要先把整张注册表读一遍
   （失败 → 所有文档都显示成「没入库」），详情里那份是按 `doc_id` 单查的。
   现在两者「有一个说做了就算做了」，列表接口另给 `indexed_known`，
   够不着时页面明说「状态未知」而不是下结论。
3. **做完的阶段要把入口置灰。** 亮着的按钮会让人以为「还得再点一次」。
   主按钮完成后变成不可点的「✓ 已解析」，重跑走旁边小号的「重新解析」——
   把「已经做完」和「故意再跑一遍」这两件事分开。

### 12.20 文档页的人工目录（2026-10-08）

使用者要的：文档页按目录管理，**最多三层**，已有的文档也能放进对应目录。

**决定 1：目录是虚拟的 —— 磁盘上不建目录、文件一个都不挪。** 这一条不是偏好，是
`doc_id` 的算法逼出来的：`doc_id = sha1(相对路径)[:16]`（`loaders.base.doc_id_for`），
把 `第一章…-1a2b.pdf` 挪进 `Flink/` 之后 doc_id 立刻变 —— `data/stages/<doc_id>` 的解析
产物、Milvus 里那批块、`_index.json` 的记录会一起变孤儿。另外上传目录顶层已经被「抓来的
图片」占着（`*_files/` 子目录），而 `LocalUploadStore.list()` 只扫顶层文件。

**决定 2：落点在 `data/uploads/_folders.json`**，跟上传件放在一起：

```json
{ "folders": [{"id": "f0a1", "name": "Flink", "parent": null}],
  "members": {"<doc_id>": "f0a1"} }
```

与 `_index.json` **分开**：那份是「一个 doc_id 一条」，塞进去要么破坏形状，要么挪一次
目录就重写整份索引。丢了当没有（目录全没、文档全回未分类），与 `_index.json` 同一个立场。
以后换文件服务器（MinIO / 共享盘）时，这份元数据跟着文件走，不用另建库。

**决定 3：三条约束**（使用者定）：最多三层（建第 4 层、把子树搬到会超三层的位置，都当场
400 说清）；一份文档只在一个目录里（不做多归属、不做标签）；同级不能重名。

**决定 4：删目录时里面的文档回「未分类」，不删文件**；**还有子目录就直接 409**，要人先
把子目录清掉 —— 一次点击连着子树一起删，等于悄悄改掉看不全的一大批归属。文件的删除只有
一个入口（删文档那个）。

**决定 5：目录树跟着 `GET /documents` 一起回**（`folders` 字段 + 每份文档的 `folder`），
不单开一个 `GET /documents/folders`：页面画左栏要的就是这两份数据，分两次请求只会多一次
往返、还可能画出中间态。写操作 4 个端点见 §12 开头那张表。

**决定 6：页面**（`DocumentsView.vue` 左栏）：树**默认全展开**（文档都藏在折叠的目录里，
页面看着就像空的）；目录行 hover 出「新建子目录 / 改名 / 删除」；移动文档走**勾选 + 「移动
到…」下拉**（一次能挪一批），**也可以直接把文档行拖到目录行上**（2026-10-08 补的，使用者
要的）。拖拽用浏览器原生的 HTML5 那套（`draggable` + `dragstart/dragover/drop`），不引第三方
库：这里只有「拖文档、放进目录」一种动作，落点判定交给 DOM 就够。**整批拖**：拖的那条在
勾选里就整批一起走。落点有两类 —— 目录行，以及「未分类」那一格（所以**没归类的文档为 0
时那一格也留着**，刚把最后一份拖走、想拖回来的时候才用得上）。**拖拽目录本身去改父子关系
没做**：后端支持，但使用者要的是「拖文件进目录」，先把这条路走通。
顶部那条上传/导入的旁边有个「放进哪个目录」，落盘之后顺手归类（没选就不调那个接口）。

**没做的**：拖目录改父子关系；跨目录的多归属；「只在某个目录里检索」（那是检索侧加过滤，与组织方式
无关）；**并发写保护** —— 现在读-改-写没有锁，与 `_index.json` 一样只有一个进程在写，
多 worker 那天要靠文件服务器的原子操作或换库。

---

### 12.21 多模态输入（一）：语音 → 文字（2026-10-08）

使用者要的三种输入模态：**语音 / 文字 / 图片**，先说清「语音」这条 —— 文字本来就有。

**交互定死的一处**：语音先用 ASR 转成文字、**填进输入框、由人确认**，再由人按回车提交。
不是「说完直接发」—— 识别难免有错字，直接发出去的错误答案要花一整轮才纠正得回来。

**ASR 选型**：DashScope 兼容模式 + `qwen-audio-3.0-asr-flash`（一次性识别，不是
`*-realtime-*` 那族流式）。两件事值得记：

1. **`qwen-vl-max` 干不了这件事** —— 它是视觉语言模型（图 + 文字），不接受音频输入。
   语音在 DashScope 是另一族模型。这一点是选型时最容易踩的坑：名字里都有 qwen、都走同一个
   `/chat/completions`，看起来像同一件事，其实不是。
2. **型号名是查过 `GET /compatible-mode/v1/models` 的清单才写的**。`deepseek-flash` 那次
   就是名字写错，一直拖到第一次真实调用才以 400 暴露 —— 配置项里那句注释就是记这个教训。

**端点与 vision 不是同一条**（这里走过一次弯路，值得记）：`vision` 走 OpenAI 兼容的
`/compatible-mode/v1/chat/completions`，而 ASR 走 DashScope **原生**的
`/api/v1/services/aigc/multimodal-generation/generation`。第一版照 vision 抄了兼容通道，
真调用回 `{"message":"format is empty","type":"UNSUPPORTED_FORMAT"}` —— 兼容通道读
`input_audio.format`，原生端点把 `format` / `sample_rate` 放在 `parameters` 里。
改完再拿 **1 秒静音**打过去，回 `ASR_RESPONSE_HAVE_NO_WORDS`：**这就是通了**（模型真的
去听了，只是没听到人声），也顺带说明 `input_audio.data` 收 base64 的 data URI。

那个「没听到人声」的码单独归一类（`AsrNoSpeechError` → 422）：静音、麦克风没选中、
离得太远都会走到它，它不是「上游挂了」（502）。

**响应形状也不是 chat 那一套**（第二次真调用才暴露）：ASR 回来的是

```json
{"output": {"output": {"request_id": "…",
   "sentence": {"begin_time": 880, "end_time": 3440, "sentence_id": 1,
                "text": "Spring AI是什么？怎么用。",
                "words": [{"begin_time": 880, "text": "Spring"}]}}}}
```

**两层 `output` + `sentence.text`**；`words` 是逐字时间戳，那些 `text` 各是一个词 ——
顺着递归往下捞就会拼出一地碎片，所以 `_sentences_text` 认到带 `text` 的那一层就停。
现在 `_extract_text` 三种形状都认（chat 的 `choices`、ASR 的 `sentence`、裸 `text`），
长录音分句时按顺序拼起来。

**这三条合起来的教训**：同一个供应商、同一个 `/chat/completions` 家族、名字里都有 qwen，
端点和响应体仍可能完全不同 —— 我连着照 vision 抄错了两次（先端点、后解析）。
`qwen-audio-3.0-asr-flash` 的真实响应已经收进用例当固定装置，不再靠猜。

**前端为什么不录 webm**：`MediaRecorder` 默认出 `webm/opus`，而 ASR 那一侧对 webm 没有
保证。所以改成自己采 PCM、编 **16kHz 单声道 wav**（`ChatView.vue` 的 `wavBytes` /
`resample`，不引第三方库）：wav 是所有 ASR 的一等公民，体积也可控（≈32KB/s，10MB ≈ 5 分钟）。
浏览器未必肯按请求的采样率给，所以编码前按**真实采样率**抽稀一次 —— wav 头里写的永远是
真实值，这一点错了整段音频都会变调。

**边界**：音频**不落盘、不落库、不进会话**（它只是「打出这段字」的一种输入方式）；
上限 `api.media_max_audio_mb`（默认 10MB）**在上游之前拦**，打上去就是一次按量计费；
ASR 的 `max_retries` 默认 **0**，失败多半是格式或时长的问题，重试只是把钱花三遍。

**图片那条还没做**：定的方向是「让对话模型直接吃图」，但现在 `providers.chat.default`
是 deepseek（不吃图），带图的轮次得切到能看图的模型 —— 要动 `ChatRequest`、
`CitationGenerator` 与 agent 那条路。记在仓库内部的待办清单里（那份不发布）。

---

### 12.22 agent 省着翻材料：提示词 + 工具调用上限（2026-10-08）

使用者问「一次请求怎么调了这么多次工具」，查 `agent.log` 得到一次真实画像
（`req-c509ebfa`）：**12 次工具 + 6 次模型往返、61.6 秒、输入 token 12.3 万**。

**不是重复调用，是 agent 的形状**：它是个循环 —— 模型决定一批工具 → 执行 → 结果塞回上下文
→ 再决定下一批。工具次数 = 模型分了几批。那一次分成 6 批：3 次检索（两条通道 + 一次换关键词
重试）→ `list_documents` → 然后 **`get_section` ×2 / ×3 / ×3**，最后才作答。

**那 8 次逐节读是模型自己的选择**，不是工具粒度的问题（`get_section` 每次 1–6 毫秒）。
它在「把相关章节读全再答」，而一批只要 2–3 节。

**时间花在两处，不在工具上**：

1. **首次检索 48 秒** —— 进程冷启动加载 embedding / rerank 模型。拿同一天的几次对照：
   新进程里的第一次检索分别是 10.7 / 39.5 / 48 秒，而**同进程里的第二次只要 4.5 秒**。
   所以「刚重启完就问，第一问特别慢」是必然，不是模型抽风。
2. **6 次模型往返** —— 每次都把整段上下文重发（缓存命中约 2.2 万）。

**两条改动**：

* **提示词**（`SYSTEM_PROMPT` 规则 6）：检索结果里已经带着每块的正文，先据此作答；
  真缺某段再用 `get_section`，并且**一轮里把缺的几节一起要**。这是日常那条路的主要杠杆。
* **`MAX_TOOL_CALLS = 12`**（`agent/build.py` 的模块常量，与 `HISTORY_MESSAGES` 同一种写法）：
  到顶就**收手**，不再往下消费事件流 —— 后面每多要一次工具都要重发一遍上下文。收尾时在
  答案里写明「工具调用到了上限」，**这一点是必须的**：不写那句，用户看到一段短答案会以为
  是模型答不出来，而不是我们主动停的。同时往 `system.log` 记一条 warning，事后能查。

**工具粒度也改了**（同一天，使用者定的）：`get_section` 加了 `headings` 入参，
**一次最多要 5 节**。原来一次只能要一节 —— 那 8 次逐节读正是被这个粒度逼出来的。
三个细节：同一次调用里**同一个块只回一遍**（`2.2` 与 `2.2.1` 会同时命中它）；
没找到的那几节**明说没找到**（不说的话模型会以为是空节）；要超过 5 节也告诉它
「剩下的没取」。**5 这个上限是「敢让它一次多要」的前提**：不限的话它会把整章一次拖进来，
那比多跑两轮更贵。工具说明里也写清了「确知缺哪几段就一次要全」—— 得给它一个这么用的理由。

**没收手的那半边**：快问答与 agent 的分工没动 —— 单跳问题本该走 `/chat`，只有多跳推理
或要写文件才值得上 agent。页面现在是「有历史就锁死 agent」，所以续着聊就一直在 agent 那条。

---

## 13. 评测与观测（2026-10-05 落地）

### 13.1 三层结构

```
eval/golden.jsonl        一行一个问题 + 判据（人维护，进 git）
src/rag/eval/            dataset（校验）→ runner（跑）→ judge（花钱的判定）→ report（两份报告）
eval/reports/*.json|md   每次跑的报告（.gitignore 已忽略）
```

```bash
rag eval                                # 只跑检索：本地 embedding，不花钱
rag eval --judge                        # 加生成侧判定：每个问题一次模型调用（花钱）
rag eval --label no-rerank --no-rerank  # A/B：改前缀，对比两份 JSON
```

**评测跑的是生产链路本身**（`build_retriever` + `assemble` + `heavy.generator` +
`heavy.chat`，见 `eval/wiring.py`）—— 另写一条「简化版检索」等于给自己发一张假成绩单。

### 13.2 指标

| 层 | 指标 | 回答的问题 | 花钱 |
|---|---|---|---|
| 检索 | `recall@k` / `mrr` / `nDCG@k` / `top1` | 材料找得到吗、好找吗、排序好不好 | 否 |
| 生成 | **引用准确率** | 答案里的 `[n]` 指的那段原文，是否真支持该编号所在的那句话 | 是 |
| 生成 | 要点覆盖率 | `answer_points` 里被讲到几条 | 是 |
| 生成 | 漏拒 / 误拒 | 语料外问题该拒没拒 / 不该拒却拒了（分开数，修法相反） | 是 |
| — | 判不了 | 模型输出解析不出来。**不算通过**，单列 | — |

**引用准确率必须自己写**：ragas 的 faithfulness 看的是「答案是否忠于上下文」，而我们要的
是更窄的一件事 —— 那个编号指的片段能不能支撑它所在的那句话。两者对「引用了无关片段」的
容忍度不一样，而后者才是引用能不能点开的依据。**两者互补，不是替代**：自研的看「引用站不站得住」，
ragas 的看「答案有没有超出上下文」。

**ragas 已经装并接上（2026-10-05，`--ragas` 开关）**，但有一个必须先垫的坑：ragas 0.4.3
（当前最新）在 `ragas/llms/base.py` 里 `from langchain_community.chat_models.vertexai import
ChatVertexAI`（还有 `llms.vertexai.VertieAI`），而我们这套 `langchain-community` 0.4.2
**已经把那两个模块移走了** —— 直接 `import ragas` 就是 `ModuleNotFoundError`。
它只是把这两个名字再导出、做类型判断，我们不用 Vertex，所以在 `eval/ragas_backend.py` 里
塞两个占位类进 `sys.modules` 即可（上游补上就能删）。另外**必须给异步 client**：
ragas 内部走 `agenerate`，传同步 `OpenAI` 会报「Cannot use agenerate() with a synchronous client.」

真机实测（deepseek-chat 当裁判，同一段上下文）：

| 答案 | faithfulness |
|---|---|
| 「Save Points 把任务快照存在存储介质上…」（有依据） | **1.0** |
| 「Save Points 会把数据写到 HDFS，每小时自动执行一次」（编的） | **0.0** |

评测里它是**可选后端**：`rag eval --judge --ragas` 才多算这一项（每个问题再加一次调用），
拒答/没有上下文那类不适用，报告里单独标「N 条适用」。

### 13.3 基线（2026-10-05，`rag_dev_qwen3vl_v1`，1691 块）

`rag eval --label baseline --top-k 8`，22 条检索条目：

| recall@8 | MRR | nDCG@8 | top1 |
|---|---|---|---|
| **86.4%** | 75.6% | 78.2% | 68.2% |

同一批问题、关掉重排（`--label no-rerank --no-rerank`）：

| 配置 | recall@8 | MRR | nDCG@8 | top1 |
|---|---|---|---|---|
| 有重排（默认） | **86.4%** | **75.6%** | **78.2%** | **68.2%** |
| 无重排 | 81.8% | 54.7% | 61.1% | 45.5% |

**重排在这套语料上是 +20.9pp MRR、+22.7pp top1**（召回只差 4.6pp）—— 也就是说它主要不是
「多找到几条」，而是**把对的那条提到第一**。这是评测骨架交出的第一份有用结论：M2 那批
把 `retrieval.rerank.enabled` 默认打开的取舍，现在有数了。

生成侧（`rag eval --label judge-final --judge`，12 条带判据的条目）：

| 指标 | 值 |
|---|---|
| 引用准确率（按问题平均，macro） | **90.0%** |
| 引用准确率（按引用条数汇总，micro） | **69.2%**（共判 26 条引用） |
| 要点覆盖率 | 94.4% |
| 漏拒 / 误拒 | **0 / 0**（3 条语料外全部正确拒答） |
| 判不了 | 0 |

**两种引用口径要一起看。** macro 与 micro 差 21pp，原因很具体：一条语料外的问题正确拒答了，
但它顺手把查过的 8 条无关材料列成了引用 —— 那 8 条不支持任何论断，把 micro 拖下来。
于是有两件事可做（都还没做）：① 生成提示词里写明**拒答时不要列无关引用**；
② 报告里对拒答条目单独看（现在漏拒/误拒本来就是单列的）。

**这里还修了一次指标的错**：第一版把 `assemble` 发出去的**所有**编号都算进引用准确率的分母，
而答案常常只用了其中几条 —— 那等于在问「你有没有把材料引用完」，跑出来 76.0%。改成只看
**答案里真的引用到的**之后是 90.0%。判据写歪比指标难看危险得多，这一条同样适用。

**图谱通道的 A/B：跑了，没有可测量的提升**（2026-10-05）。同口径对照（黄金集 24 条检索项，
含为它加的两条关系类问题 `flink-vs-spark` / `flink-dependencies`）：

| 配置 | recall@8 | MRR | nDCG@8 | top1 |
|---|---|---|---|---|
| `hybrid + bm25`（基线） | 87.5% | 71.7% | 75.6% | 62.5% |
| **`hybrid + bm25 + graph`** | 87.5% | 71.7% | 75.6% | 62.5% |

**逐题 diff：27 条里名次变化 0 条**（含 11 条 Flink 题 —— 图里正好只有这两篇文档）。
`--stats` 显示这一路**确实在工作**：每次召回 10 条候选、各有 1 条进了最终 5 条，
耗时约 400ms（对比 rerank 的 16s 可忽略）。

三条原因，按重要性排：

1. **带回来的块和 hybrid 命中的是同一批** —— 图只是给它们再投一票，而 RRF 的票数变化
   不足以改变名次（rerank 之后更是如此）；
2. **覆盖面**：图里只有 Flink 两章（354 实体 / 426 关系），另外 3 篇（Kafka 1208 块等）
   一条边都没有 —— 14 条 Kafka/Docker 问题这一路直接空转；
3. **问题类型**：这份黄金集大多是「找一段文本」，而图通道的长处是**跨文档的多跳关系**，
   现有语料里这种问题很少（两条关系类题也只在单篇内部）。

**决定：不全量灌图**（按事先说好的口径）。这 860 行代码与 Flink 两章的图保留 ——
`rag graph query` 已经是排查实体/关系的现成工具，将来 M6 的 `graph-explorer` subagent
也要用它；但要让它影响检索结果，得先解决上面第 1 条（见下）。

**下一步的两个候选**（都还没做）：

* **只补位**：图通道只返回 hybrid/bm25 没命中的块（过滤掉重复候选）—— 假设是「同一批块
  再投票没有信息量，只有新块才有」，改动小、一次免费 A/B 就能验；
* **灌全量 + 跨文档问题**：Kafka 那 1208 块灌图约几块钱，然后在黄金集里加真正需要多跳的
  问题（「A 文档说的 X 和 B 文档说的 Y 是什么关系」）—— 验的是图通道的本命场景。

**候选 1（只补位）已经做了，也没用**（用户定「先做 A」，同日）：通道可以声明
`complement_only = True`，融合层会把它与别路重复的候选丢掉，只让它贡献新块
（`ChannelStats.complement_dropped` 记着丢了几条）。

```text
--stats 实测：graph 召回 10 → 丢重复 5 → 补进 5 条新块 → 入选 0
评测（同一批 24 条）：87.5% / 71.7% / 75.6% / 62.5%  —— 与基线逐字节相同
逐题 diff：名次变化 0 条；**没有任何图通道来源的块进过 top-5**
```

**这说明原假设不成立**：问题不在「重复投票稀释了名次」，而在**图补进来的新块本身不够竞争** ——
rerank 拿同一批 hybrid 候选就把它们比下去了。根因还是那两条：图只覆盖 2/5 篇文档，
而这些问题大多是单跳查找（图的「新块」恰好是 hybrid 有意没排高的邻居）。

于是判断收敛到一件事：**要么按候选 2 去验图通道的本命场景（跨文档多跳），要么就承认
这份语料上它没用**。当前 `graph` **不在任何默认配置里**（`dev.yaml` 的
`optional_channels` 是空的），要跑得显式 `--channel hybrid,graph`；每次查询它多花
~300-400ms，所以默认关着是对的。

三条真 miss（报告里带「它召回了什么」，便于分辨是检索问题还是期望写歪）：

- 「Kafka 快速开始里的 Step 7 是做什么的」→ 召回 `Step 8: Use Kafka Streams`（步骤号差一位）；
- 「Kafka 2.5 的安全章节包含哪些小节」→ 召回文档总目录（目录页列了所有章名，语义上更"像"）；
- 「listener.name 这种前缀是干什么用的」→ 召回 `8. KAFKA CONNECT -> REST API` 等。

**第一次跑还暴露了两条我自己写歪的期望**（已改，并记在 golden.jsonl 的 `notes` 里）：
`listener.name` 我指到了 `control.plane.listener.name`（那是控制器专用监听器，不是通用前缀）；
日志保留时间我按 heading 找 `log.retention`，而 OCR 把配置名拼成了 `log.retentionminutes`。
**这两条是事实错误，不是拿指标迁就系统** —— 判据写歪比指标难看危险得多。

### 13.4 日志：三类各落一个文件；再顺手进一张 PG 索引表（2026-10-08）

**两次改动的顺序**：2026-10-06 之前是「控制台 + 文件 + PG 三张表」，当天改成**三类各落
一个文件**（那三张表连同写库那套一起删了）；2026-10-08 使用者要求**按 `request_id` 筛、
`request_id` 还要能点**，于是又加回**一张**表（`log_line`），但**定位变了** ——
**文件仍是唯一真相，库里那份只是索引**（见本节末尾）。

使用者定的三部分，**共用一套关联 id**（`request_id` / `run_id`，走 contextvars）——
没有这套 id，三份文件就是三堆互不相干的流水：

```
接口请求  req-06fc5982 ─┬─→ channel.log（4 行：hybrid / bm25 / rrf / cross-encoder，同 id）
                        └─→ agent.log（同 id；CLI 里则是 run-xxxx）
```

| 类别 | 文件（当前 profile 的 `<state_dir>/logs/` 下） | 一行是什么 | 行上有什么 |
|---|---|---|---|
| `interface` | `interface.log` | 一次 HTTP 请求 | `method / path / status / 耗时 / 身份[ / 响应字节 / ip=]`，失败再带 `⚠error` |
| `channel` | `channel.log` | 一次查询 × 每一路通道 | `（查询或 hash 前 8）/ channel / 召回 / 入选 / 耗时 [ / 过滤 ]`，失败再带 `⚠error` |
| `agent_run` | `agent.log` | 一轮问答的**汇总** | `thread / 问（原文或 hash 前 8）/ 工具 / 步 / 召回 / 引用 / 状态 / token / 耗时` |
| `agent_llm` | `agent.log` | **一次模型调用** | `模型#n / 模型名 / 输入 / 输出 / 共 / 缓存 / 耗时 / 收尾` |
| `agent_tool` | `agent.log` | **一次工具调用** | `工具#n / 工具名 / 入参（值）/ 出参（值）/ 耗时`（失败带 `⚠`）；**入参出参恒给内容**，见下面第 3 条 |
| `system` | `system.log` | 其余：LangChain / pymilvus / httpx / uvicorn / 其他 `rag.*` | 同一套格式，带 `req=` / `run=`（**不含**上面几类，不重复落） |

**另有一份 `<state_dir>/logs/startup-error.log`**：它**不是**业务日志的一个类别（`rag logs tail --kind`
里没有它，也不进 PG），而是 `rag` 每次**启动读配置失败**时的现场 —— 完整栈按时间追加，
由 `cli/main.py` 的 `_record_startup_error()` 写。它只读启动前已可用的
`RAG_DEPLOYMENT_PROFILE` / `RAG_STATE_ROOT` 来选当前状态域，因此即使完整配置校验失败，
也不会把两个 profile 的失败栈混在同一个文件里。为什么非单独有一份不可：配置坏掉是
唯一一类恰好发生在 `configure_logging` **之前**的失败，那一刻四个文件一个 handler 都
没装，栈只到 stderr；而 `rag serve` 被 `scripts/windows/backend.ps1` 起在新窗口里，
窗口一关就什么都不剩。实测 2026-10-10：`config/dev.yaml` 的 `ingest.images` 成了空节点
（底下只剩注释 = YAML 的 `null`），服务起不来、日志全空，只能靠手工敲 `rag config`
才看出是 pydantic 校验失败。

**类别名一律用这套英文标识**（页面、`rag logs tail --kind`、`/logs/tail?kind=` 一致）；
`all` 只是「不过滤」。**agent 的三类共用一个文件**（`agent.log`）：一轮的时间线要在同一个
文件里连着读（`模型#1 → 工具#1 → 模型#2 … → 汇总`），分成三个文件就断了。类别靠
**行里的 logger 短名**分（`obs/log.py` 的 `kind_of_line`），所以 `--kind agent_tool`
是**按行**过滤的、不是按文件；过滤时在最近 2000 行里找（`FILTER_WINDOW`），
否则最新那几行恰好是别的类别时会一条都找不到。

四份各 20MB × 3 个备份（`observability.logging.files` / `.system`），上限合计 320MB；
控制台（stderr）是第五个出口，全部日志在那里也看得见。装 handler 是 `configure_logging`
的事：**CLI 根回调**与 `rag serve` 都调 —— 少了这一步 CLI 的 INFO 会因为没有 handler
被直接丢掉（2026-10-06 修，有用例钉住）。但**配置读不出来时它自己也装不上**，
那种失败走上面那份 `startup-error.log`。

**一条记录 = 一行**：四个文件与实时流都过 `OneLineFormatter`（换行转义成 `\n`，内容一个
字符不丢）。psycopg 的异常文本自带换行（`LINE 1: insert …` + 定位的 `^`），照原样写出去
就会多出几行**没有时间、没有级别**的续行 —— 页面上是一条空行（真机上就是这么看到的）。
控制台**不用**这个 formatter：终端里栈要能正常换行读。

**时间戳带毫秒**（`FORMAT` 里是 `%(asctime)s.%(msecs)03d`，`10-07 11:10:58.876`）：
四份文件的记录要按时间合并成一条流水，只到秒的话同一秒里的几条是并列的 —— 稳定排序
只能按「先读哪个文件」排，看起来就是顺序乱（使用者报的正是这个）。读的时候**两种都认**
（2026-10-07 之前的行只有到秒），实时流那条 `ts` 也补齐到毫秒。

**`agent.log` 是一轮一条时间线**（2026-10-06 起）：`模型#n` / `工具#n` 各一行，最后一条是
这一轮的汇总。数据来自 **v3 事件流**（`await agent.astream_events(..., version="v3")`，
信封是 `{type, method, seq, params:{namespace, timestamp, data}}`；`run_agent` / `resume_agent`
也走同一条，所以 CLI 与界面记出来的账一样）：

* `method="messages"` 的 `content-block-delta` 给正文增量、`message-finish` 给**这次调用的**
  token（`usage.input_tokens / output_tokens / total_tokens`，以及
  `input_token_details.cache_read` —— 真机上 6 次调用有 5 次命中缓存）；
* `method="tools"` 的 `tool-started` 给工具名与**结构化入参**、`tool-finished` 给输出
  （是 `str(ToolMessage)`，所以只截断、只判 `status='error'`，不硬解结构）。

**v3 的三个坑**（都实测过）：

1. **调用方式不一样**：v1/v2 直接 `async for`，v3 要先 `await` 拿到流；
   Pregel 上它自己是 experimental（引擎会发 `LangChainBetaWarning`）。
2. **流里没有中断**：v2 的 `updates` 里那个 `__interrupt__` 在 v3 中不存在 ——
   中断只在 `state.tasks[*].interrupts`（`aget_state` 取，要 checkpointer）。
   所以收尾统一读一次状态：`_interrupt_from_snapshot`。
3. **非流式模型没有 `messages` 事件**（只有 `values`）：那种模型不会吐 token，
   正文从 `values` 快照兜底 —— 没有 checkpointer 时它是唯一来源。

口径提醒：逐次 `输入=` 是**当次调用的完整上下文**，第 6 次的 14147 里含着前 5 次的内容，
所以汇总之和不是「这一轮真实消耗」；要给统计用的 token 仍看会话级的 `token_usage`。

**有一类库要单独处理**：自己装 handler 再把 `propagate` 关掉的（pymilvus 实测：裸
`StreamHandler(stderr)` + `propagate=False`），一条都不会冒泡到 root —— 只挂 root 的话
系统文件里会缺一整块。`NOISY` 那几个 logger 因而被**无条件接管**：摘掉库自带的 handler、
关掉 propagate、把 root 那一套原样挂上去。为什么不按 `propagate` 值判断：`rag serve`
是先装日志、startup 才 import pymilvus，等它自己配完再判断已经晚了。这个洞是全量跑测试
才暴露的 —— 单跑那条用例时 pymilvus 还没被 import，它是「通过」的。

**四条硬约束**（都写进了 `obs/log.py`）：

1. **写日志不影响业务**：只调 stdlib `logging`，不碰网络也不碰数据库。
   `logging.raiseExceptions` 保持默认的 **True** —— handler 自己的故障（磁盘满、文件被占用）
   要看得见；关掉之后这些会一声不响（实测，以前那行就是关掉的）。
2. **关联 id 走 contextvars**：接口中间件设 `request_id` 与 `run_id`，链路深处的通道日志
   与 agent 日志直接读 —— 不用把 id 一层层往下传（传着传着就会漏）。
3. **内容默认不落**（`observability.logging.content`）：问题/查询原文里有语料正文与用户原话，
   属于数据；默认只记 **sha1 前 16 位 + 长度**，行里给 hash 前 8 位（能对齐、能去重、
   不留原话）。**答案一律不入日志**（它是会话数据，在 chat 表与控制台输出里）。
   **工具那一行是例外**（2026-10-10 使用者定：「工具的调用，输入输出应该展示出来，可以
   截断，不能展示输入多少字，输出多少字，这怎么分析问题」）：`agent_tool` 的入参与出参
   **永远给内容、超长才截**，不吃这个开关 —— 上限是 `obs/log.py` 的 `TOOL_ARG_CHARS`(200) /
   `TOOL_ARGS_CHARS`(400) / `TOOL_OUTPUT_CHARS`(600)，截了在尾巴上写 `…（共 N 字）`。
   换行按上面那条规矩转义成 `\n` 字面量，所以三个落点（文件 / 实时流 / 库里那份）逐字一致。
4. **密钥永不落**：身份只记限流那套 `identity_of` 的结果（`key:<sha1 前 12 位>` / `ip:...`）。

**通道日志记在 `MultiChannelRetriever.retrieve` 里**（不是调用点）：那是唯一同时握着
「每一路账本」与「最终结果」的地方，放调用点记一定会漏（调用点有五个）。

`rag logs tail -n N [--kind channel]` **直接读文件尾巴**（开-读-关，不连库）。
`rag serve` 里 **uvicorn 的 access log 关掉了** —— 接口日志自己记（带 request_id），
再叠一份只会重复。

**token 统计不在日志上**：要给统计用的 token 账在**会话级**（`persistence/chat.py` 的
`token_usage`，一次问答一行，见 §12.10）；`agent.log` 里的 `token=in/out` 只是过程信息。

**页面的三个端点**（P3 的日志页，2026-10-06 起两个、2026-10-08 加第三个）：

* `GET /logs/tail?kind=&limit=` —— **历史**：读上面那四个文件的尾巴（开-读-关、带类型
  过滤、四类按时间合成一条流水），所以 CLI 进程写出来的行也在里面；
* `GET /logs/stream?backfill=` —— **实时**：SSE，先补最近几行再跟着推。数据来自进程内
  环形缓冲（`obs/live.py`：一个 logging handler 写 deque + 推订阅者），只覆盖 API 这个
  进程；页面把两者拼起来，CLI 那些行靠「刷新」或 `tail`。
* `GET /logs/search?request_id=&category=&since=&until=&limit=` —— **查索引**（PG 的
  `log_line` 表）。`request_id` **必填**：这个口子要回答的是「这一次请求到底发生了什么」，
  没有 id 就退化成翻日志 —— 那正是文件 + grep 在做的事。没配 PG 时回 503 并指路
  `/logs/tail`（**不做「没库就自动退回扫文件」**：两种语义混成一个接口，用时看着一样、
  结果不一样才是坑）。

页面上一行是五列：**时间 · 级别 · 来源 · 类别 · 正文**。**来源**是 logger 短名
（`storage` / `ocr` / `uvicorn` / `pymilvus`…），历史那条从行里切、实时那条由 handler 直接
给字段 —— 单列出来是因为「系统」这一类里混着好几个来源，挤在正文里看不出是谁说的。
`uvicorn.error` / `uvicorn.access` 的短名统一改写成 `uvicorn`：那两条本来叫 `error` /
`access`，`INFO error Application startup complete.` 会被当成错误上报。

**行首那两段关联 id 单拎出来**（`obs/log.py` 的 `split_ids`，实时那条由 handler 给字段）：
页面上它单独占**一列**（`request_id`），值**可点**，点一下就把这次请求的 interface +
channel + agent 串成一条。埋在正文里就只能让人复制粘贴，而那正是这个页面要省掉的动作。

**筛选长在对应列的表头上**（2026-10-08 使用者定）：每列标题后面一个小箭头，点开是这一列
自己的条件（时间 / 级别 / 类别 / request_id / 正文），有条件的箭头点亮 —— 不点亮的话
「日志怎么变少了」看不出来是筛过的。顶部横条里只留暂停 / 刷新 / 清屏 / 回到实时。
**能进 SQL 的只有三个**（`request_id` + 类别 + 时间，见 `/logs/search`）；级别与关键字是
**在已经加载出来的行里筛**（文件那条路没有索引），浮层里明说了"在已加载的行里筛"。
类别解释原来是一整段说明挂在页面上，现在收进**类别筛选里 `system` 那一项的小图标**
（悬浮或键盘聚焦才出来）：只有 `system` 看不出是什么，其余几个名字本身就是答案。

**三个当场踩出来的坑**（都是使用者试出来的，都记在这儿免得改回去）：

1. **级联的类名别叫 `.info`**：行上的级别类名就是 `levelClass(level)` → `info` / `warning` /
   `error`。给别的元素起 `.info` 会与 `.log-line` 同优先级，而后出现的
   `.info{display:inline-flex}` 把 `.log-line{display:grid}` 顶掉 —— 整张表从 grid 变成
   按内容排的 flex，列宽全漂（实测表头 `201/331/401/505/587/711` vs 行 `201/336/372/443/514/611`）。
   改叫 `.kind-help` 后表头与每一行的六列 x 完全一致。**量表用无头浏览器报几何，不要靠肉眼看截图**
   （这台机器上 `--dump-dom` 吐不出东西，是让页面把量好的 JSON POST 回本地服务再读的）。
2. **池与短连接的行形状要一致**：`LogStore` 有两条连接路 —— 池（建的时候带了
   `row_factory=dict_row`）与短连接（`AsyncConnection.connect`）。短连接不显式要 `dict_row`
   就返回 tuple，`search()` 里的 `dict(row)` 会抛
   `cannot convert dictionary update sequence element #0 to a sequence`（页面上表现为
   "点了 request_id 什么也没有"）。现在两条路都要 `dict_row`，`search()` 另按列名兜一次 tuple。
3. **空结果不要把整屏擦掉**：历史不回溯（文件里的老行不在库里），加上写入是 0.25 秒一批，
   点一个 request_id 常常查不到。这时**保持当前列表不动**、只挂一条写清原因的说明 ——
   空屏看起来像页面坏了。

#### 2026-10-08：`log_line` —— 文件是唯一真相，PG 只是索引

**为什么回头加库**：要求从「看最近的流水」变成「下拉就能按 id / 类别 / 时间筛」，而
**文件一多就要跨文件扫、而且没有索引**。实测全目录扫一遍 22.5 ms（5 个文件 / 690 KB /
5,800 行），按 260 KB/天算一年 95 MB —— 那时线性扫是**秒级**，还要加几百个文件打开的
开销。使用者拍板：上 PG。

**表**（`persistence/logs.py`，`rag logs init` 建，与 chat 那三张表同库同 schema）：
`log_line (id, ts, category, level, request_id, run_id, source, logger, status, duration_ms, text)`，
两个索引正好覆盖两种用法 —— `(request_id, ts)` 串链路、`(category, ts desc)` 做「最近的
system」。**不灌历史**（使用者定），从新产生的行开始。

**四条口径**（都是使用者逐条定的）：

1. **`request_id` 从入口就有**：HTTP 那条是中间件发（`api/app.py`），**CLI 也发一个**
   （根回调里 `new_id("req")`）—— 两边**同一个形状**，查询不用分两套；
2. **不在请求里的行补一个 `solo-`**：库自己打的、测试进程、后台任务。补在
   **`LogStore.write` 里而不是调用方** —— 漏补一行会撞 `not null` 让**整批**插失败，
   那种失败是静默丢掉一整批日志；
3. **`run_id` 用 `job_id`**：一次「把这份文档入库」分几次请求跑（parse / 核对 / index），
   每进一次请求就新发一个 run **必然串不起来** —— 不是缺一个 id，是 run 生错了地方。
   三个后台任务（`run_parse_job` / `run_index_job` / `run_ingest_job`）进来就
   `set_ids(run_id=job.id)`，与页面上那排步骤灯同源。**实测**：`asyncio.create_task`
   会继承发起请求的 context（父协程里的 `request_id` 带得进来），而任务里改的
   `run_id` **不会漏回父协程** —— 这正是要的行为；
4. **入口是 HTTP 还是 CLI 用 `source` 列区分**，不从 id 前缀反推（两边都是 `req-`）。
   第三方库的行归 `system` 且 `source=library`：一次请求里 pymilvus 报的那句重试，
   靠同一个 `request_id` 就能和接口行串起来 —— 这正是上库的用处。

**写路径**：`obs/log.py` 的 `_write` 先把摘要算好（当 message，文件/控制台那一行一个字
没变），再顺手把结构化那一份挂在 record 的 `extra={"rag_row": …}` 上；`obs/pg.py` 的
`PgLogHandler` 挂在 **root** 上，`emit()` **只入队**，后台守护线程每 0.25 秒或攒够 200 行
批量 insert（同步写 —— 那个线程没有事件循环，而 Windows 上 psycopg 的 async 只能跑在
Selector 循环里，为一份索引背这个约束不值得）。

**写不进去不影响业务**：失败只 stderr 喊一句然后**丢掉这一批**（喊话限流：前 3 次 +
之后每 100 次），队列有界（满了丢最新的）；进程退出时 `logging.shutdown()` 会调
`flush()`/`close()`，把尾巴写完。库里那份丢了随时能从文件重建。

**为什么文件不能省**（2026-10-06 那三条现在仍然成立）：

* 日志的写者和页面的服务端是**同一个进程**（`rag serve` 单 worker），要动态展示走内存
  推送即可（那是最新的一份，零延迟）；
* 机器上**文件是跨进程共享的**（CLI 跑出来的日志也在里面），比只覆盖 API 进程的内存缓冲全。
  代价与坑：**别长期持有文件句柄**去 tail —— 实测在 Windows 上会把 `RotatingFileHandler`
  的轮转顶掉（`WinError 32`，而且那一条日志会整条丢掉）；读的时候开-读-关。
* 而 PG 在本项目里是**可选依赖**（`persistence.postgres.url` 留空是合法配置）：
  没配、连不上、表没建，日志都得照常写、页面都得照常看 —— `observability.logging.postgres`
  关掉或没配 url 时**连 handler 都不装**，行为与从前完全一样。

于是分工是：**文件是唯一真相**（跨进程、可读、可 `rg`），**库只是它的索引**（按 id /
类别 / 时间查得快）。两边都要，取舍是「多写一份」这点开销 —— 一天几千行。

真机验证：2026-10-05 的 PG 版验证过三类同源、id 串得起来；改成文件后
`pytest tests/test_logs.py` 用临时目录真跑一遍 `configure_logging`，钉住四个文件互不串、
系统文件不含业务行、重复配置不留旧句柄。加索引之后 2026-10-08 实测：`rag logs init`
建表 → 发三条探针（同一 `req-probe01`）→ `rag logs search req-probe01` 三行原样回来
（`source=cli`、`run_id=job-probe01`）；用例在 `tests/test_log_pg.py`
（批量与退出收尾、失败只喊不抛、队列满丢新行、`solo-` 兜底、端点 422/503/映射）。

### 13.5 观测：LangSmith

`observability.langsmith: true` + `.env` 的 `LANGSMITH_API_KEY` → `configure_tracing()`
把 `LANGCHAIN_*` 与 `LANGSMITH_*` **两套环境变量都设上**（LangChain 认前者、langsmith 认后者，
只设一套迟早撞版本差异）。CLI 在真干活前打一行状态，**关着的时候闭嘴**（默认档就是关的，
每次刷「未开」会让人习惯性忽略，配错时就没人看见了）。

调 agent 行为没有 trace 是盲调 —— M6 批次 B（图谱）之前，这一层先准备好。

---

## 14. 部署

Milvus standalone 需要三个组件：**etcd + MinIO + milvus**。

### 14.0 部署档位（2026-10-26 改为三档 canonical profile）

| profile | 向量库 | 关系库 | 文件 / bucket | Redis | workers |
|---|---|---|---|---|
| `local-lite` | `rag_local_lite_v1`（Lite 文件） | `data/local-lite/*.sqlite3` | `data/local-lite/**` | 不允许 | 严格 1 |
| `docker-single` | `rag_docker_single_v1` | `data/docker-single/*.sqlite3` | `rag-docker-single` | 不允许 | 严格 1 |
| `multiworker` | `rag_multiworker_v1` | PG schema `rag_multiworker` | `rag-multiworker` | **必需** | `max(workers, replicas) >= 2` |

profile 只接受 `local-lite` / `docker-single` / `multiworker` 三个 canonical 名称。
`load_settings()` 会先按 profile 强制基础设施组合，再在 `create_app()` 二次校验；
不满足组合直接报错。

`local-lite` 是本机零服务档：Milvus Lite 仍由 `pymilvus` 调用，但数据写在
`RAG_STATE_ROOT/local-lite/milvus.db`（默认 `./data/local-lite/milvus.db`），
不启动 Docker 或网络端口。
实测 Milvus Lite 3.2.2 支持 dense/sparse、BM25 Function 与 `hybrid_search`，但 analyzer
只接受 `standard` / `jieba`，Docker Milvus 默认的 `chinese` 会在建表时直接报错；
因此 `local-lite` 自动改用 `jieba`。Lite 的数据库锁是进程级的：同一数据库不能一边跑
API 一边另起 CLI 灌库，切换或清理通常要停进程。

`docker-single` 保留真实 Milvus 服务端的 hybrid 行为。Docker Milvus standalone 本身
已经依赖 MinIO，因此应用复用同一 MinIO 实例的 `rag-docker-single` bucket，不新增
另一个对象存储进程；关系库仍用独立的 SQLite 文件。
`multiworker` 的 PostgreSQL / Redis / MinIO / Docker Milvus 是四项硬要求，缺一项就拒绝启动。
Milvus 内部 bucket `milvus-bucket` 与两个应用 bucket 完全分离。
UI 对话记录也跟随状态域：本地两档用各自目录下的 `chats.sqlite3`
（启动自动建表、WAL），multiworker 继续用 PG
`chat_session / chat_message / token_usage`。这样本地对话页不再返回“没配 PG”的 503，
普通快问答与会话列表重启后都还在；LangGraph Agent 的 checkpointer 仍是另一层。

每个 profile 还会写状态 marker：

- 本地：`<state_dir>/state.json`
- MinIO：bucket 根下 `_profile.json`
- marker 记录 profile、namespace id、collection、元数据后端、bucket/schema/Redis prefix；
  任一项不一致都拒绝启动。
`rag profile show` 查看解析结果，`rag profile check` 只读校验，
`rag profile import-legacy` 可把旧 `rag` bucket + PG schema 显式复制到 docker-single，
源数据不删不改。
启动脚本也复用 `rag profile show` 得到的 `run_dir` / `log_dir`：PID、Linux 的
`backend.out.log` / `web.out.log` 与 Windows 的 `backend-startup.log` 都跟随当前 profile，
不再写共享的 `data/run`、`data/logs`。
启动时还会实际探活：Redis `PING`、PostgreSQL 连接与建 pool、MinIO bucket 检查（不存在则自建）；
任一失败都直接终止 startup，而不是把 `/health` 标红后继续半降级运行。
`rag serve --workers N` 多 worker 时用 import-string 工厂让每个子进程重读配置；
两个单 worker 容器通过 `deployment.replicas=2` 表达总副本数。

**真实验收（2026-10-23）**：两个独立进程分别监听 18011 / 18012，`workers=1`、
`replicas=2`，共享远端 PostgreSQL / Redis / MinIO；两边 `/health` 均为 `ok`，
且 `ingest_gate=redis / job_store=redis / vision_cache=minio`。Docker 镜像与容器级
验收仍未做，见 backlog #29。

多 worker 的共享状态：

- `JobStore`：任务 JSON + Redis sorted-set 索引，列表能看见其他 worker；
- `IngestGate`：Redis `SET NX` 槽位 + 租约，finally 按 token 释放；multiworker 下 Redis
  丢失会拒绝退回单进程；
- VLM 描述缓存：multiworker 强制改用 MinIO，一个图片哈希一个 key，不存在 JSON 合并竞争；
- 限流：沿用原 Redis 计数。

```yaml
# deploy/docker-compose.yml（结构示意）
services:
  etcd:
    image: quay.io/coreos/etcd:<pin>
    command: etcd -advertise-client-urls=http://etcd:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
  minio:
    image: minio/minio:<pin>
    command: minio server /minio_data --console-address ":9001"
  standalone:
    image: milvusdb/milvus:<pin>
    command: ["milvus", "run", "standalone"]
    depends_on: [etcd, minio]
```

> 版本 tag 以 Milvus 官方 compose 文件为准，不要照抄本文档的占位符。

**本地开发方案**：**直连虚拟机上的 Milvus**（`192.168.x.x:19530`，server v3.0.2），
地址写在 `.env` 的 `MILVUS_URI` 里。已实测该实例支持 dense+sparse+hybrid，核心检索链路本地可完整验证。
`deploy/docker-compose.yml` 保留，用于需要本地独立实例的场景（版本已与虚拟机对齐 v3.0.2）。

**为什么不改用 Milvus Lite**：hybrid 是主线，而 Lite 是否支持 sparse vector 始终未经确认。
现在已有真实 Milvus 可直接用，没必要再引入一套能力存疑的运行时。
（"Lite 到底支不支持"这个问题仍未验证，只是已经不需要去回答了。）

### 14.1 OCR 产线服务（GPU 主机侧）

本项目代码不装 `paddlepaddle`（ADR 17），扫描件靠 GPU 主机上的产线服务解析。
要让 OCR 真正跑起来，主机上需要：

> **2026-09-29 更正 —— 产线不必是 Docker，也不必在 GPU 主机上。** 实测：vLLM 侧的
> `PaddleOCR-VL-1.6-0.9B` 在**本机 127.0.0.1:6006**（下游 VLM），而产线本身可以用
> PaddleX 自带的 serving 在本机起，**不需要 Docker、不需要本机 GPU**：
>
> ```powershell
> # 1) 导出官方流水线配置（非交互必须给 --save_path，否则它会卡在 input()）
> python -m paddlex --get_pipeline_config PaddleOCR-VL --save_path .
> # 2) 改两处后（见下）启动
> python -m paddlex --serve --pipeline deploy/paddleocr-vl.yaml --host 127.0.0.1 --port 8080
> ```
>
> `deploy/paddleocr-vl.yaml` 相对官方 dump 有三处必需改动：
> `VLRecognition.genai_config.backend: native → vllm-server`、`server_url` 指向 vLLM、
> `model_name` 必须**与 vLLM 实际服务的 model id 逐字一致** —— 写错不是本地报错，而是
> vLLM 回 404 `The model ... does not exist.`（官方 dump 给的是 `PaddleOCR-VL-0.9B`，
> 实测服务的是 `PaddleOCR-VL-1.6-0.9B`）。另外把布局模型换成已缓存的 `PP-DocLayoutV3`，
> 否则首次启动会卡在拉取 `PP-DocLayoutV2`。
>
> 产线暴露 `POST /layout-parsing`、`POST /restructure-pages`、`GET /health`，
> 信封与 `ingest/ocr.py` 期望的完全一致 —— **客户端一行不用改**。

1. 用官方 compose 起 `paddleocr-vl-api`（默认 **8080**）。它和 8118 那个
   `paddleocr-vlm-server` 是 compose 里的**两个容器、两个端口**。
2. 让产线**复用已经在跑的** 8118，别重复起一个 VLM：
   ```yaml
   VLRecognition:
     genai_config:
       backend: vllm-server
       server_url: http://localhost:8118/v1
   ```
3. 设页数上限，防长文档打爆：`Serving.extra.max_num_input_imgs: 200`。
4. 方向矫正 / 去扭曲（`useDocOrientationClassify` / `useDocUnwarping`）需要先开
   `Serving.use_doc_preprocessor`，**默认配置下调这两个参数会直接报错**。本轮不默认开。
5. 本机 `.env` 填 `PADDLEOCR_BASE_URL`，指向主机 8080。

服务没起来不影响别的：不配地址就是纯 pymupdf。

---

## 15. 里程碑

| 阶段 | 内容 | 工期 |
|---|---|---|
| **M0** 骨架 ✅ | Python 3.13 环境、目录、contracts 协议层、config（含 `.env` 插值）、CLI 空壳、**Milvus 连通性 + hybrid 能力实测** | 已完成 |
| **M1** 最小闭环 ✅ | PDF/MD 接入 → chunk → bge-small-zh-v1.5 embedding → Milvus 建库 → **dense + 内建 BM25 的 hybrid 检索** → 带引用生成 → `rag ingest` / `rag search` / `rag ask` | 已完成（2026-09-22） |
| **M2** 检索增强 | 跨通道 RRF（L2）+ rerank + metadata filter 打通 + 独立 bm25 通道 | 2–3 天 |
| **M2.5** 接入层 OCR | 扫描件与混合型 PDF 走 PaddleOCR-VL 产线服务（`auto` 逐页探测 → 整份送 OCR）+ 图片文件接入。**只碰 ingest**，不动检索/生成/编排 | 1–1.5 天 |
| **M3** LangGraph 编排 | IndexGraph 断点续跑 + QueryGraph（plan→并行→融合→rerank→generate→grade 循环）+ 流式 | 3–4 天 |
| **M4** API | FastAPI 五个端点 + SSE + 鉴权限流 | 2 天 |
| **M5** deepagents | 主 agent + 四个知识工具 + ToolPolicy + CompositeBackend + memory + skills + HITL | 3–4 天 |
| **M6** 图谱 + 评测 | 实体关系抽取、graph 通道、ragas + 引用准确率、LangSmith、CI | 3–4 天 |

**M1 结束就应该能 demo** —— 一个能问问题、带引用的最小 RAG。后面每阶段都是可独立停下的增量，不必全做完才有价值。

**M2 缩小了**：初稿把「sparse 字段 + `hybrid_search`」列为 M2，但 M1 为了让检索链路一开始就是
真能跑的（而不是先只做 dense、回头再接 sparse），把 hybrid 一起做了。所以 M2 现在只剩「跨通道融合」
—— 也就是 L2：把 hybrid 的结果与 graph / web 这类别的数据源再合一次 —— 以及 rerank、metadata filter 的接线。

---

## 16. 风险与对策

| 风险 | 说明 | 对策 |
|---|---|---|
| **Python 版本** | 3.13 / 3.14 安装均无障碍（langgraph 是纯 Python 轮子，其编译型依赖 xxhash/orjson/zstandard 都有 cp314 wheel）；已定 **3.13** | conda 建 `multi_modal_rag`（3.13）隔离环境 |
| Milvus Lite 能力未知 | 若不支持 sparse/hybrid，本地开发链路受限 | 搭虚拟机时验证；默认走 docker standalone |
| 图谱抽取贵 | LLM 逐 chunk 抽实体关系，慢且烧钱 | 设为可选通道，只对指定语料开，结果缓存 |
| 换 embedding = 全量重灌 | collection 必须重建 | collection 名带 embedding 指纹，双写切换 |
| ragas 兼容性 | ragas 0.4 对 langchain 1.x 需实测 | M1 就搭最小 eval 冒烟 |
| 供应商行为差异 | 结构化输出/tool calling/长上下文各家不一 | `tests/conformance/` 一致性套件 |
| Milvus 运维 | 三容器 + load/release 生命周期 | 本地 compose 与虚拟机保持同版本 |
| **OCR 服务是外部单点，且下游 VLM 隔着一条转发的公网链路** | 产线在 GPU 主机上，而它的识别全靠转发到 AutoDL 云实例上的 vLLM。实测这条转发在真实负载下 **28% 失败**（34 次 200 / 13 次 500，见 §7.1），失败即切云端兜底（按量计费），大 PDF 因此经常走云端 | 转发换成带保活的持久 SSH 隧道、或在 AutoDL 客户端里重连；`FallbackOcr` 切云端前先原地重试一次（尚未实现）。不配 `PADDLEOCR_BASE_URL` = 纯 pymupdf，链路完全不受影响；单文件失败带原因计入 skipped，不中断整批（见 §7.1） |
| **长文档整份 OCR 的耗时** | 一次 `/layout-parsing` 把整份 PDF 送过去，几百页能跑到分钟级甚至超时 | `ingest.ocr.timeout_s` 默认 600s（chat 那个 120s 不够）；服务端 `max_num_input_imgs` 限页 |
| **本机 embedding 的吞吐随块长变化很大** | 实测（2026-09-30，16 逻辑核、CPU）：**满 1000 字**的块只有 0.32 块/秒（10 块 31.1s），而真实语料（中位 565 字、42% 不到 500 字）**整条链路 243 块约 4 分钟**。「33.6 条/秒」那个旧数字是**短句批处理**的口径（语义切分那一步），不能用来估灌库时间 | 估时间用「块数 × 1~3s」；要提速：把 embedding 服务化（像 OCR 那样打到 GPU 机器）、换 ONNX/int8、或调大 batch |
| **Milvus 不可达时 CLI 不快速失败** | 实测：虚拟机上的 Milvus 停了，`rag ingest` 在写库那一步空转重连三十几分钟（纯 CPU 烧、零 I/O），数据一行没进 | **已实现**（2026-10-05）：`MilvusIndex.ping()` + `indexing.vector.probe_timeout_s`（默认 5 秒）。`ingest` / `search` / `ask` / `agent` 与 `serve` 的 startup 都先探活，接不住就一行话退出。**探在解析/OCR 之前** —— 否则那几十分钟的 OCR 也白花。实测：连接被拒 2.0 秒就返回（`Milvus 连不上：…`，退出码 1）；「假死」（不拒也不回）由 5 秒上限兜住 |

---

## 17. 决策记录（ADR）

| # | 决策 | 理由 | 被否方案 |
|---|---|---|---|
| 1 | 流水线分层 + 多检索模式，两者都要 | 用户明确选择 | 只做单一维度 |
| 2 | 多供应商可切换 | 用户要求 | 锁定单一厂商 |
| 3 | 三形态交付（API + CLI + Agent） | 用户要求 | 二选一 |
| 4 | 向量库用 Milvus | 用户指定首选，自行在虚拟机搭建 | Qdrant（初稿推荐） |
| 5 | 向量库客户端用 pymilvus 直连 | langchain-milvus 的 wrapper 不暴露 sparse + `hybrid_search`，且无 3.14 支持 | langchain-milvus |
| 6 | hybrid(dense+sparse) 为主通道，BM25 降级可选 | learned sparse 有 term expansion，召回更好；但精确串匹配 BM25 仍更稳 | BM25 作第二主通道 |
| 7 | ~~embedding 用 bge-m3 经 fastembed ONNX~~ **已由 14 取代** | 一个模型出 dense+sparse，且不需要 torch | FlagEmbedding（无 3.14 元数据） |
| 8 | ~~agent 层用 Claude，批处理用国产模型~~ **已由 20 取代** | 拿满 harness 能力同时压批量成本 | 全平权多供应商 |
| 9 | Python 用 3.13 | 全部依赖实测支持；langgraph 官方声明封顶 3.13，核心框架不跑在未声明支持的版本上 | 3.14（能装，但 langgraph 未声明） |
| 10 | 评测含自建「引用准确率」 | ragas 不直接提供该指标 | 只用 ragas |
| 11 | 本地开发直连虚拟机上的 Milvus | 已实测支持 hybrid，本地可验证核心链路；Lite 能力存疑且无必要 | Milvus Lite |
| 12 | 机器相关配置只进 `.env`，yaml 默认值保持通用 | 避免私有地址进提交、避免测试随本机环境漂 | 把地址写进 base.yaml 默认值 |
| 13 | sparse 腿改用 **Milvus 内建 BM25 Function**，不用 learned sparse | fastembed 里既没有 bge-m3、也拿不到任何中文 learned sparse 模型；内建 BM25 不需要稀疏编码模型、不需要 torch，中文分词由服务端 analyzer 做。见 §8.2 | FlagEmbedding 自造 sparse（多一个模型 + torch 依赖）；rank_bm25 在 Python 进程内算（要自己管词表与内存） |
| 14 | ~~embedding 换 **BAAI/bge-small-zh-v1.5**（512d，仅 dense）~~ **已由 21 取代** | 中文语料实测区分干净（相关 0.75–0.93 / 无关 0.24–0.38），95MB、CPU 可跑、不需要 torch，首次下载约 8 分钟。**但窗口只有 512 token，块一大就被静默截断** | 继续找 fastembed 的 bge-m3（不存在）；bge-large-zh（更大更慢，M1 用不上） |
| 15 | `doc_id` 取**相对路径哈希**，重新灌库先按 doc_id 删旧块 | 路径哈希让重复 ingest 原地覆盖、不产生幽灵块；但 `chunk_id = doc_id:序号` 使文档改短后旧序号块成为孤儿，故必须「先删后写」 | 内容哈希（改一个字就换 doc_id，旧块全变幽灵）；只 upsert 不删（文档变短即残留） |
| 16 | 融合策略由**检索层**构造；配置里 `rrf` + 非 1.0 权重**直接报错** | `RRFRanker` 只接受 k、不支持按腿加权；静默忽略配置里的权重比报错更坏 —— 用户会以为权重生效了 | 让索引层自己决定 ranker；静默忽略权重 |
| 17 | 扫描件 OCR 走**服务化产线**（GPU 主机上的 `paddleocr-vl-api`），本项目只写 HTTP 客户端 | 主机已有 GPU、已在跑 8118；本机没有 NVIDIA 驱动。产线要把重依赖（paddlepaddle + torch 系）和 GPU 需求引进来，客户端代码却只有几十行。地址按 ADR 12 进 `.env`，不配就是纯 pymupdf | 本机装 `paddleocr`（实测有 cp313-win wheel，但把重依赖和 GPU 需求引进主环境）；调 8118 那个 VLM（它只是产线的下游，单独打它拿不到版面分析与阅读顺序） |
| 18 | 混合型 PDF **整份**走 OCR（`auto` 探测到任一扫描页即整份送） | 只有整份进产线才拿得到跨页表格合并与标题重建，重建出的 `#/##/###` 正好喂 markdown chunker；也避免了「逐页拼装」时的顺序与页界问题。代价是其中的数字页也在 GPU 上重跑一遍 | 按比例阈值判定（阈值以下的零散扫描页静默丢失，与 skipped「不静默跳过」的立场冲突）；逐页拼装（拿不到 `mergeTables` / `relevelTitles`）；官方两阶段 `/restructure-pages`（把多页拍平成一份 markdown，丢掉页归属） |

---
| 19 | 「多模态」的范围限定为**接入层 + 存储层**，查询侧不做跨模态 | 用户明确：多模态 = 多种**输入**模态，检索仍是文本 query 对文本块。这样不必引入 CLIP 类多模态 embedding，也不必维护第二套向量空间与 collection。接入侧的多模态已在 M2.5 落地（扫描件/图片 OCR），存储侧的原始资产留存待做（§21） | 查询侧跨模态（图像当 query）；把「多模态」读成多路检索通道的「多模式」——原文的措辞，已纠正 |
| 20 | **不用 Anthropic**：agent 层与批处理统一走国产模型 | 使用者明确「Anthropic 太贵了，不用」。deepagents 的 harness 特性对 Anthropic 支持最好，但价差由账单决定 —— 本项目默认不付。`agent.chat` 指 `providers.chat` 的条目，切换只改 `.env` 的 `AGENT_CHAT` | 沿用 ADR 8（agent 用 Claude）；为省 token 而全平权多供应商 |
| 21 | embedding 换成 **Qwen3-Embedding-0.6B**（1024d），接受 torch 依赖 | bge-small-zh 的 512 token 窗口是硬约束：实测中文 **550 字就到 512 token**，块稍大就被静默截断（不报错，后半截根本不进向量）。Qwen3-Embedding 窗口 **32768**，尺寸约束基本解除，chunk 才谈得上按语义放宽。代价：引入 torch（fastembed 那条 ONNX 路不再默认），collection 必须重建、全量重灌（512 → 1024 维） | 守着 bge-small 只把 chunk 压小；为省依赖改用别的 ONNX 大窗口模型（fastembed 库里没有） |
| 22 | 语义切分**自研**，不用 LangChain 的 `SemanticChunker` | 三条实测理由：① `langchain-experimental` 官方已 **sunset**（与当初否掉 `langchain-community` 同一条）；② 它的默认分句正则 `(?<=[.?!])\s+` **对中文完全无效** —— 整段算一句，直接原样返回，等于空操作，必须自己塞 regex；③ 同语料同模型下它比自研慢 2.5 倍（16.3s vs 6.5s），且**没有尺寸护栏**（切完最大块 799 字，超出阈值近一倍） | 用 `SemanticChunker` + 自定义中文分句正则；改用别的语义切分库 |
| 23 | **PDF 统一走 OCR**（`auto` 探测移除）；**结构明确写在文件里的格式走代码解析** | ① 探测的前提是「数字版走 pymupdf 更划算」，实测不成立：pymupdf 只拿得到纯文本，标题/表格/图片全无，下游 markdown 切分与图片描述都空转（同一份数字版 PDF：pymupdf 0/20 带标题，OCR 36/36）。② docx / html / xlsx 的标题与表格是**明文写在文件里**的，识别它没有意义、也不该花 GPU。四条路的出口统一成 markdown，否则切分器要按来源分支。`parser: pymupdf` 降级成**有损开关**（产线挂了或做对照时用） | 保留 `auto` 探测 + 加阈值（我实测过它的误检：shell 代码块被判成表格、页眉 logo 被判成图）；让 pymupdf 也产 markdown（它的 `h1..h6` 是字号启发式猜的，同一页 37 个「标题」里 8 个是对的） |
| 24 | 产线不可用时**切 PaddleOCR 云端 API** 兜底；默认关闭、只对 `OcrUnavailable` 生效、切换必须出声 | ① 默认关闭：云端按量计费，打开是花钱的决定。② 只兜「服务不可用」（连不上/5xx/errorCode≥500）——请求本身有问题时换一家也一样失败，还盖掉真正原因；为此把这两类拆成不同异常。③ 出声：悄悄换到按量计费的云端，用户只能对着账单猜。开了没配 token 直接报错，不假装有保险 | 静默兜底；对所有异常都兜底；默认打开 | 
| 25 | 图片描述走 **VLM（默认 DashScope `qwen-vl-max`）**，产出 `modality=image` 的独立块；默认关闭 + `max_images` 闸门 + 按**图片内容哈希**缓存 | 图片内容在库里必须可检索：实测「两种方式的对比」问到图 3 时，模型只能说「以图片形式呈现，文字未展开」—— 它没编，但也答不了。三件套（key/base_url/model）沿用 `providers.chat` 那套约定（变量名带 VL 后缀），实现用几十行 httpx 而不是 LangChain chat 模型。缓存是必需的而不是优化：改一段正文会让文档重灌，没有缓存就是几十张图重烧一遍。默认关闭与 ADR 24 同一个立场 | 让 chat 模型兼职看图（qwen-plus 这类不认识 `image_url`）；把描述塞进正文块里（图片会污染它所在正文块的向量，且 `modality` 失去了意义）；不缓存，每次重灌全部重问 | 

---

## 18. M0 验证清单（结项 2026-09-21）

| # | 验证项 | 状态 | 结论 |
|---|---|---|---|
| 1 | 建 3.13 环境并装齐依赖 | ✅ | 108 个包，18/18 关键 import 通过 |
| 2 | LangChain 1.x 实际 import 路径 | ✅ | `langchain-classic` 提供传统 retriever；`langchain-community` 已 sunset，**不用**（见 §2.3） |
| 3 | `fastembed` 的 bge-m3 同时产出 dense + sparse | ❌ | **前提不成立**：fastembed 里没有 bge-m3，也拿不到任何 learned sparse 模型。改用 bge-small-zh-v1.5（仅 dense）+ Milvus 内建 BM25（ADR 13/14） |
| 4 | `pymilvus` 建带 sparse 字段的 collection + `hybrid_search` | ✅ | 在真实 Milvus 3.0.2 上跑通，见 §8.1 |
| 5 | Milvus Lite 是否支持 sparse + hybrid | — | 已无必要：改用虚拟机上的真实 Milvus |
| 6 | `flashrank` 加载 rerank 模型 | ⏳ | 仍未验证；rerank 归 M2，不在 M1 链路上 |
| 7 | `.env` 里的部署参数能被 yaml 插值看到 | ✅ | 发现并修复了一个静默失效的坑，见 §6 |

---

## 19. M1 验证清单（结项 2026-09-22）

| # | 验证项 | 状态 | 结论 |
|---|---|---|---|
| 1 | bge-small-zh-v1.5 经 fastembed 出 512d dense 向量 | ✅ | 维度自检通过；向量已 L2 归一化（范数 1.0），故 COSINE 与 IP 排序等价 |
| 2 | 走 HF 镜像下载模型 | ✅ | 必须同时设 `HF_ENDPOINT` + `HF_HUB_DISABLE_XET=1`，见 §8.2 |
| 3 | md / txt / pdf 三种语料接入 + 切块 | ✅ | gb18030 回退解码、CRLF 归一化、隐藏目录跳过、不支持的后缀计入 skipped |
| 4 | Milvus 内建 BM25：建表带 function + 中文 analyzer | ✅ | `sparse` 写入会被拒（它是 function 的输出），查询传原文即可命中 |
| 5 | 索引层 18 项集成测试（真 Milvus） | ✅ | `RAG_TEST_MILVUS=1 python -m pytest tests/test_milvus_index.py` |
| 6 | `rag ingest` 端到端灌库 | ✅ | 3 篇 md/txt/pdf → 3 chunk → 写入 dev collection，重复执行幂等 |
| 7 | `rag search` 三路召回正确排序 | ✅ | 两轮查询均把对应文档排到 #1（含 metadata 过滤） |
| 8 | `rag ask` 生成段 | ⏳ | **未实测**：`.env` 里还没有任何 chat provider 的 API key。检索→装配→引用→构造 model 这一段已走通（缺 key 时报的是我们自己的中文提示，不是 LangChain 的晦涩异常） |
| 9 | 单测 + lint | ✅ | 82 项用例（64 单测 + 18 集成）、`ruff check` / `ruff format --check` 干净 |
| 10 | `rag eval` / `rag serve` | — | 仍是空壳，分别属 M6 / M4 |

**两件待办（需要用户决定或提供）：**

1. `.env` 里补一个 chat provider 的 key，才能把 `rag ask` 的生成段跑通并肉眼验证引用质量。
2. `providers.chat.default.model` 已从 `deepseek-flash` 改回 `deepseek-chat` —— 前者不是有效的
   DeepSeek 模型名，写错要到第一次 `rag ask` 才以远端 400 暴露。这个值**只在本机能连 DeepSeek
   网络时可验证**，请按自己账号下的模型列表确认一次。

---

## 20. M2.5 验证清单（待跑）

不配 `PADDLEOCR_BASE_URL` 的机器上，第 1–5 项已可跑，且**不应发出任何网络请求**；
第 6–7 项需要 §14.1 的产线服务起来。

| # | 验证项 | 状态 | 结论 |
|---|---|---|---|
| 1 | 纯数字 PDF 仍走 pymupdf，OCR 一次都不调 | ✅ | `tests/test_ingest_ocr.py`：断言假客户端的调用记录为空 |
| 2 | 空白尾页不被当成扫描页 | ✅ | 同上：无文字**且**无图的页不算触发，否则数字版 PDF 会整份重跑 GPU |
| 3 | 任一扫描页 → 整份送 OCR，页序按数组下标拼回 | ✅ | 同上 + `tests/test_ocr_client.py` |
| 4 | `auto` 撞上扫描页但没配地址 → 报错并给出两条出路；不静默降级 | ✅ | 报错含 `PADDLEOCR_BASE_URL` 与 `pymupdf` |
| 5 | 标题层级只在 `chunk.strategy: markdown` 下落进 chunk metadata | ✅ | 同一份 OCR markdown 分别用两种策略切块对比 |
| 6 | 真扫描件能拿回非空中文 markdown | ✅ | `RAG_TEST_OCR=1 python -m pytest tests/test_ocr_service.py` 通过（2026-09-29，产线跑在本机 8080）。语料由 pymupdf 现画再渲染成位图，真的没有文字层 |
| 7 | 表格页还原成可解析的表格 + 单元格值正确 | ✅ | 同上，两条用例均通过。形态是**内联 HTML** `<table><tr><td>`（不是竖线语法），单元格值准确 |
| 8 | 端到端：扫描件灌库后能检索到、引用 `source_uri` 正确 | ⏳ | `rag ingest ./data/docs --parser ocr` 后 `rag search "..."` |
| 9 | 现有 md/txt/pdf 语料重灌结果不变（零回归） | ✅ | 默认 `parser: auto` + 无 OCR 地址时走的就是原来的 pymupdf 路径 |
| 10 | 单测 + lint | ✅ | 126 项用例（106 单测 + 20 集成）、`ruff check` / `ruff format --check` 干净 |

---

## 21. 定位与命名修正（2026-09-27）

原文把项目写成「多模式」，并把「多模块」定义成「纵向分层 + 横向多通道」两个维度 —— **这是错的**。
「多模态」指的是**多种输入模态**，落点在接入层 + 存储层（ADR 19）。

### 已改

| 项 | 内容 |
|---|---|
| 定位 | §1 重写；§5 / §7 / §8 里指代检索通道的「多模式融合」改称「多路通道」—— 检索策略不是多模态 |
| 项目名 | `multi-mode-rag` → **`multi-modal-rag`**：`pyproject.toml`（name + description）、`README.md` 标题、本文档标题、`src/rag/__init__.py` docstring、`src/rag/cli/main.py`（help / `--version` / 回调 docstring）、`src/rag/config/schema.py` 与 `config/base.yaml` / `config/dev.yaml` 的 LangSmith project、`.idea` 模块名与 SDK、`uv.lock` 的根包名 |
| 环境名 | README 与 §2.1 的 `multi_model_rag` → **`multi_modal_rag`**；本地环境已 `conda rename` 改过来，并重装 editable 包清掉了旧名残留（`pip list` 里现在只有 `multi-modal-rag` 一条） |
| 顺带修掉 | `rag.exe` 里烧死的解释器路径 —— miniconda 搬家后 `rag` 一直静默 exit 1，见 §2.1 补注 |

原来三个词根混着用：项目名 `multi-mode-rag`（**mode**）、目录名 `multi_mode_rag`（**mode**）、
环境名 `multi_model_rag`（**model**）。现在统一到 **modal**。

### 尚未做

| # | 事项 | 说明 |
|---|---|---|
| 1 | 仓库目录改名 | 目录仍是 `D:\python-project\multi_mode_rag`。**不影响运行** —— 装的包名是 `rag`，import 与 CLI 都不带项目名。改不改都行，但要和 IDE 的 `last_opened_file_path` 一起看 |
| 2 | 存储层的原始资产留存 | 「多模态」的存储侧落点。**图片这一半已落地**：本体落 `ingest.assets.dir`（默认 `./data/assets`，按 doc_id 分目录），正文里留 `![image](asset:<doc_id>/p0003_01.jpg)` 原位标记，切块后靠标记能反查「这张图在哪一块、上下两段是什么」（`ingest/assets.py`，2026-09-29）；图片内容另有一条 `modality=image` 的 **VLM 描述块**（`ingest/images.py` + `providers/vision.py`，2026-09-30，见 §7.4）。**页图与表格片段仍未落地**；资产也没有清理动作，文档改短后旧图会留在盘上 |

第 2 项是这次定位修正暴露出来的真空白：**接入层的多模态（M2.5）已经落地，存储层还只有文本。**

## 22. 密钥与模型选型的存储边界（2026-10-09，B 方案）

模型选型进配置这件事，绕不开一个"key 放哪"的问题。使用者 2026-10-09 拍板走 **B 方案**：
**页面上能填 API key，但库里只存密文；主密钥 `RAG_SECRET_KEY` 放部署配置**。

### 主密钥在哪、丢了会怎样

| 位置 | 角色 |
|---|---|
| 部署配置 `RAG_SECRET_KEY`（环境变量 / `.env`） | **主密钥的正位**。运维在部署时给，`base64` 或 `hex` 编 32 字节均可 |
| 库里 `app_secret` 表（`name='rag_secret_key'`） | **兜底位**。应用**只读**；部署里没给、但库里写了一份时用它 |
| 两处都没有 | 页面**禁用**填 key（`can_set_key=false`）；`PUT` 带 `api_key` 回 422 并点名"差一把 `RAG_SECRET_KEY`" |

**主密钥丢失 = 库里所有密文都解不回来**（AES-256-GCM，无后门）。所以它必须进密码管理，
而不是随手写进某个临时脚本。换主密钥会让旧密文全部失效——这也是刻意的：解密失败只报原因，
不静默回退。

### 库里存什么、页面能看什么

* **`provider_setting` 表**：`provider_key`（主键）/ `api_key_cipher` / `api_key_fingerprint`
  / `base_url` / `models`(JSON) / `updated_at` / `updated_by`。key 与"这家能出哪些模型"、
  base_url 同表（使用者定的）。
* **密文格式 `v1:nonce:ct`**（`persistence/secrets.py`）：密钥 32 字节、nonce 12 字节、
  **AAD = `provider_key`** —— 换个 provider 名字解密必然失败（防"把 A 家的密文抄到 B 家"）。
* **指纹 = `sha256(key)` 前 8 位十六进制**：页面拿它显示"已配置 (a1b2c3d4)"、换 key 时确认
  "确实是另一把"，但反推不出明文。
* **页面/接口永不回 key**：`GET /providers` 只回 `has_key` 与 `api_key_fingerprint`，
  连密文都不出后端（`providers()` 故意不带 `api_key_cipher`）。
* **审计只记指纹**：`model_setting_change` 里 `slot='provider:<name>'`，旧→新都写成
  `key=<指纹 或 ->`，明文与密文都不落审计。

### 两个接口与设置页

| 端点 | 语义 |
|---|---|
| `GET /providers` | 供应商清单 = `providers.chat` 的键 ∪ 库里存过的 `provider_key`；每行 `has_key` / 指纹 / base_url / 模型清单 / `can_set_key` |
| `PUT /providers/{key}` | `api_key` 三态：**不传 = 不动**、**空串 = 清除**（回到部署配置那份）、**给值 = 加密存**；`base_url` / `models` 同理；写库 + 写审计，回整份清单 |
| `GET /models` / `PUT /models/{slot}` | 八个槽位的"谁在用"（见 §6）；key 只报 `key_ready`（布尔/None） |

两处 `PUT` 在写之前都会 `store.setup()`（**幂等建表**）——没跑过 `rag models init` 的机器上，
设置页也能直接保存，不必先手敲一条 CLI。这四张表是 `model_setting` / `model_setting_change`
/ `provider_setting` / `app_secret`，SQLite 与 PG 同名同列，换库不改代码。

## 23. 文档分工与命名约定（2026-10-12）

这个项目里"文档"不止一份，而且**给谁看**不一样。2026-10-12 重排过一次 README，
顺手把分工写死在这里，免得下次又往同一份文件里堆。

### 一份文件对一类读者

| 文件 | 读者 | 放什么 | 不放什么 |
|---|---|---|---|
| `README.md` | **第一次见到这个项目的人**（评审、面试官、想跑起来的人） | 一句话定位、能力一览（agent 层单独一节）、**实测数字**、架构图、怎么跑、已知边界 | 论证过程、逐题的调研、踩坑细节 |
| `docs/results.md` | 想知道"到底跑出什么数"的人 | **只有数字与口径**，每张表标出它在本文档的出处 | 论证（只说"为什么这么选"在本文档哪一节） |
| `docs/architecture.md`（本文档） | 要改这个仓库的人 | 决策与理由（ADR）、验证清单、踩过的坑、**为什么不是另一种做法** | 待办清单 |
| `docs/plugins.md` | 要接新数据源或可选解析器的人 | 通道、CLI 命令与 parser 扩展包的约定与写法 | — |
| `docs/milvus-schema-design.md` | 要动 collection 的人 | 字段与索引为什么是现在这样 | — |
| `docs/backlog.md` | **只给自己**（已 gitignore，不发布） | 还没做的事、已知问题、内网地址与本地细节 | — |
| `CHANGELOG.md` | **只给自己**（已 gitignore） | 逐阶段的变更记录 | — |

### 三条规矩

1. **数字只有一份正本**：`docs/results.md`。同一个指标不要在 README、本文档、CHANGELOG 里各写一遍
   —— 对不上时没人知道该信哪个。README 里那张表是"摘要 + 指路"，改了数字要三处一起对（README /
   results / 本文档对应小节）。
2. **README 的文档表只列发布得出去的文件**。`backlog.md` 与 `CHANGELOG.md` 被 gitignore
   （含内网地址与本机路径），所以它们**不出现在 README 的链接表里** —— 否则对外就是一堆死链。
3. **已知边界要写进 README**。"有意不做"和"还没做完"对读者是两件事，README 的「已知边界」一节
   负责说清前者，backlog 负责记后者。

### 命名约定（对外统一口径）

| 场合 | 用哪个 | 备注 |
|---|---|---|
| 仓库名 / 发行包名（`pyproject.toml` 的 `name`） | `multimodal_rag` | 下划线、全小写 |
| 导入包名 / CLI 入口 | `rag` | `import rag…`、`rag --help`、`python -m rag` |
| 环境名（conda） | `multi_modal_rag` | 历史遗留的 `model/mode` 混用见 §21 |

§21 记的是当时那次更名（`multi-mode-rag` → `multi-modal-rag`）的现场；实际落盘的名字以本表为准。
**改仓库名/发行包名要连带改**：`pyproject.toml`、README 标题、本文档标题、`rag config` 打的版本串、
以及装完之后的 editable 路径 —— 2026-10-12 这一轮**没有**做这个改名（值不值得做见 backlog）。

设置页（`web/src/views/SettingsView.vue`）照 RAGFlow 的「模型设置」骨架：左列供应商
（`has_key` 打标），右侧这家的 API-Key 输入 + base_url + 模型清单表 + 保存 / 清除库内 key；
下面是槽位表，回答"这一类活现在是谁在干"（**不叫"默认模型"**——槽位本身就是"确定性"，
和 RAGFlow 那句"设为默认 chat 模型"是同一件事的两种说法）。

**这张页 2026-10-10 暂时从导航撤下**（使用者：「页面上的设置项，暂时不用，可以先撤掉」）：
只去掉入口（`NavRail` / `App.vue` / `PageId` 三处），视图与上面两个端点全部保留。要做回来
就把 `'models'` 加回 `PageId`、在 `NavRail` 的条目与 `App.vue` 的分支上各接一处（那两处都
留了注释指引），再 `npm.cmd run build` 重出 `web/dist`（后端把 `web/dist` 挂在 `/` 上，
不重新构建页面不会变）。撤下的原因是这一版**不热切**（改完要重启 `rag serve`），
验证期用它反而多一层解释成本。
