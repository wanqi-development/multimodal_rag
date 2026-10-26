# 60 秒 demo：自带语料，一分钱不花

这个 demo 的目的是**让别人在几分钟内看到这东西真的能跑** —— 灌进去三篇文档，问几个问题，
看引用能不能回到原文。语料在 `demo/corpus/`（三篇我们自己的短文档，约 4500 字，切出 **14 块**），
答案是我们**故意埋**在里面的，所以每次跑的结果都可复现。

## 为什么单独开一个档（`config/demo.yaml`）

因为默认那套 **dev 档是花钱的**，而且会碰你自己的数据：

| | dev 档（默认） | demo 档 |
|---|---|---|
| embedding | 线上 `qwen3-vl-embedding`（2560 维，**按量计费**） | 本地 `Qwen3-Embedding-0.6B`（1024 维，免费） |
| 图片描述 | 开（每张图一次 VLM，**按量计费**） | 关 |
| collection | `rag_dev_qwen3vl_v1`（你真实的语料） | `rag_demo_qwen3_v1`（独立，可随时删） |
| 本地落点 | `data/…` | 全在 `data/demo/` 下 |
| 关系库 | PG（若配了） | 单机 SQLite（`data/demo/documents.sqlite3`） |

demo 档里那几个值**写死、不吃 `.env` 插值**：本机 `.env` 里恰好把 `EMBEDDING_BACKEND` 指到线上
也不会让 demo 悄悄花钱。

## 前置

* Milvus 起着（`deploy/docker-compose.yml`）；
* `.models/` 下有两个本地模型（README「本地模型」一节有下载方式）；
* 问句那一半要一个 chat 供应商的 key（`.env` 三件套），与 `rag ask` 的要求一样。

## 跑起来

```powershell
scripts\demo.cmd                 # Windows
scripts\demo.cmd -DryRun         # 只打印要做什么，一个字都不写
scripts\demo.cmd -Reset          # 清掉上次的 demo 数据再重灌（collection + data/demo/）
```

```bash
./scripts/linux/demo.sh
./scripts/linux/demo.sh --dry-run
./scripts/linux/demo.sh --reset
```

它会依次做四件事：

1. `rag ingest demo/corpus --stage parse` —— 只解析，**按设计不碰 Milvus**；
2. **替人把「已核对」勾上** —— 这个仓库有意要求切分前有人核对过标题层级
   （[architecture §7.6](architecture.md)）。demo 语料是我们自己写的、标题本来就准，所以脚本
   替人勾这一声；**真语料别这么干**。它只改清单头部那一行；
3. `rag ingest demo/corpus` —— 切分 + 入库（复用第 1 步的产物）；
4. 三个问题 + 一个**该被拒答**的问题。

## 实测（2026-10-12，本机，Milvus 在远端虚拟机上）

| 步骤 | 耗时 | 备注 |
|---|---|---|
| 解析（冷，3 篇 md） | **4.0 秒** | 不碰 Milvus |
| 切分 + 入库（冷，14 块） | **34.3 秒** | 大头是加载 1.2GB 的本地 embedding |
| **首次灌库合计** | **约 38 秒** | |
| 重复灌库（增量） | **3.9 秒** | 输出是「扫描 3 篇 → 跳过 3 篇，没有需要处理的文件」 |
| 每次检索 | **19.5 秒** | Milvus 那一路 9.6–10.5s + 精排 3.7–4.3s + 进程内加载模型 |
| 每次问答 | **20.1–21.9 秒** | 同上，多一次 chat 调用 |
| **`-Reset` 全流程**（清数据 → 解析 → 核对 → 入库 → 四个问题） | **140.2 秒** | 退出码 0；这就是"从零跑一遍"的真实感受 |

**为什么每条查询要十几秒**：Milvus 在远端虚拟机上（`.env` 的 `MILVUS_URI`），一次 hybrid 检索
实测 **约 10 秒**；本地精排 bge-reranker-base 约 **4 秒**。而且每个 `rag` 命令都是新进程，
每次都要重新加载一遍本地模型。页面版省得掉"重新加载模型"那一段（后端进程常驻），
但 Milvus 那 10 秒省不掉 —— **录屏时建议只走「一次检索 + 一次问答 + 点开引用」，别塞满四个问题。**

## 真实输出（节选）

检索「Aurora 网关默认给每个租户多少 QPS？」—— 第 1 条就是答案所在的那一块，分数 0.9999：

```text
命中 10 个 chunk：
   1. 0.9999 [milvus]  ece7503498ccb959:0002 01-aurora-overview.md  Aurora 网关：产品概览 -> 默认配额
      …（命令行只打块正文的前一段）…
   2. 0.7949 [milvus]  a9327b72d5a4931d:0002 02-rate-limit-runbook.md  限流与提额运维手册 -> 临时提额
      …（同上）
```

问答「2026-08 那次断电事故的根因是什么，哪个版本修的？」（**跨节**取信息，带引用）：

```text
根因：策略快照按“先写临时文件、再改名”的方式落盘，但改名之前没有 fsync；断电时文件系统里留下
一个长度正确、内容全是零的块，重启后 JSON 解析必然失败，`policy-engine` 启动时报快照解析失败，
随后回退到空策略并正常启动，导致那段时间所有请求都被放行 [1]。

修复版本：2026.08.3 [1]。
```

语料外的问题「Aurora 网关支持哪些 Kubernetes 版本？」—— **拒答而不是硬编**（三篇里一个字都
没提 Kubernetes）：

```text
已有的资料无法确定。资料中没有提到 Aurora 网关与 Kubernetes 版本相关的任何信息。
```

## 录屏分镜（页面版）

页面版要先把档位切到 demo，否则页面是 dev 档、看不到这份语料：

```powershell
$env:RAG_PROFILE = 'demo'
scripts\start.cmd
```

```bash
RAG_PROFILE=demo ./scripts/start.sh
```

分镜（约 60–90 秒；每条查询十几秒，所以**别剪掉等待**，那正是真实速度）：

| 时间 | 画面 | 要点 |
|---|---|---|
| 0–5s | 打开 <http://127.0.0.1:5173> | 左侧导航 + 依赖灯（Milvus / 模型）全绿 |
| 5–25s | 检索页输入「Aurora 网关默认给每个租户多少 QPS？」 | 命中卡第 1 条就是「默认配额」，分数 0.9999 |
| 25–40s | 点开详情抽屉 | 能看到标题层级、页码行号与正文原文 —— **引用能回到原文**是重点 |
| 40–75s | 对话页问「2026-08 那次断电事故的根因是什么，哪个版本修的？」 | 流式答案 + `[1]` 引用可点回原文 |
| 75–90s（可选） | 再问一个语料外的 | 它拒答，而不是编一个答案 |

成图放 `docs/assets/demo.gif`，README 顶部引它。

## 跑完怎么清

demo 的数据只在两处，删掉不影响你别的语料：

* collection `rag_demo_qwen3_v1` —— `scripts\demo.cmd -Reset`（或 `./scripts/linux/demo.sh --reset`）；
* 本地文件 `data/demo/` —— 直接删目录（`-Reset` 也会一起删）。

**`-Reset` 为什么两处都要删**：只删 collection 不够。增量判断看的是**文档注册表**
（`data/demo/documents.sqlite3` 里记着「这篇已入库」），留着它再跑 ingest 会输出
「跳过 3 篇」，collection 还是空的（实测踩到，2026-10-12）。

那两条删除命令都带**安全绳**：删 collection 前断言名字必须以 `_demo_qwen3_v1` 结尾，
删目录前断言路径就是 `<仓库>/data/demo` —— 档位万一没生效，它当场拒绝，而不是把你的
`rag_dev_qwen3vl_v1` 或 `data/stages/` 删掉。
