# 插件：通道与命令怎么"零代码"扩展

**一句话**：核心只认识「**按名字找实现**」这套机制，不认识任何具体的外部服务。
要接一个新数据源（Elasticsearch、第二个 Milvus collection、别家的图库……），写一个 Python
包实现**一个函数**、在它的 `pyproject.toml` 里声明一行入口点、`pip install` ——
核心代码一行不改。

原型是使用者点名的 **Java 加载不同 DB 驱动**：接口在核心（`java.sql.Driver`）、实现在驱动
包里、靠 `META-INF/services` 在运行时对上。Python 的对应物是 **entry points**
（`importlib.metadata`）；我们用它，另外留了一条点路径的逃生舱给开发期。

## 1. 两个扩展点

| 扩展点 | entry point 组名 | 实现什么 | 装完的效果 |
|---|---|---|---|
| **通道** | `rag.channels` | `build(*, index, embedder, settings, cfg) -> Retriever` | 配置里能写 `type: <入口点名>` |
| **命令** | `rag.cli` | 一个 `typer.Typer()` 实例 | `rag` 下多出一组子命令（名字 = 入口点名） |

两者都是**运行时按名字查**：装了就出现、卸了就消失（不是"配置写了却静默不生效"）；
查不到时当场报错，并把**内置的**与**已装插件声明的**名单一起列出来；核心的 import 图里
没有插件 —— 插件坏了只影响它自己那条路。

## 2. 通道的解析顺序

```
type 名 ──► ① 内置注册表 CHANNEL_TYPES（核心自带：milvus）
            ├─► ② entry points（组名 rag.channels）= 装好的插件包
            └─► ③ 点路径 "包.模块:build"      = 开发期免安装的逃生舱
```

实现在 `src/rag/retrieval/plugins.py::resolve_channel_type`，结果按名字缓存
（`functools.cache`），一个进程只查一次。

## 3. 通道接口

```python
def build(*, index, embedder, settings, cfg) -> Retriever:
    ...
```

| 入参 | 是什么 |
|---|---|
| `index` | Milvus 索引。**可能是 `None`**（纯外部服务用不上），用到时才在 `retrieve` 里判 |
| `embedder` | 多模态 embedder，同样可能是 `None` |
| `settings` | 整棵 `Settings`（插件自己的参数块也在这里） |
| `cfg` | 配置里那一项 `ChannelConfig`：`name / type / enabled / complement_only / top_k / options` |

返回的对象要满足 `Retriever` 协议（`src/rag/contracts/protocols.py`）：
`name` + `async retrieve(query, top_k, filters=None, image=None) -> list[ScoredChunk]`。
三条要点：

1. **返回 `ScoredChunk`**（块 + 分数 + 通道名）。分数只要自己内部可比 —— 跨通道融合用 RRF，
   只看名次不看分数（`docs/architecture.md` §3）。
2. **失败就抛**：融合层把它记进 `stats.error` 并当作没召回，其余通道照常；
   自己吞掉错误会让 `--stats` 里看不出这条路到底有没有工作。
3. **`complement_only = True` = 只补位**：融合层会丢掉它与别路重复的候选，只留新块。
   异构源（图谱、外部搜索）大多该这么开 —— 重复投票不改变名次，只白花时间。
   配置里的 `complement_only` 能覆盖它。

## 4. 写一个插件（完整例子）

目录：

```
plugins/rag-channel-elasticsearch/
├── pyproject.toml
└── src/rag_channel_es/
    ├── __init__.py      # 导出 build（入口点指的就是它）
    └── channel.py
```

`pyproject.toml`：

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "rag-channel-elasticsearch"
version = "0.1.0"
requires-python = ">=3.11"
# 不写 `dependencies = ["rag"]`：核心没上 PyPI，写了会去网上找、白白失败。

[project.entry-points."rag.channels"]
elasticsearch = "rag_channel_es:build"

[tool.setuptools.packages.find]
where = ["src"]
```

`channel.py`：

```python
from pathlib import Path
from typing import Any

from rag.contracts import ScoredChunk


class EsRetriever:
    complement_only = True

    def __init__(self, endpoint: str, index_name: str) -> None:
        self.name = "es"
        self._endpoint, self._index = endpoint, index_name

    async def retrieve(self, query: str, top_k: int, filters=None, image=None) -> list[ScoredChunk]:
        ...        # 查 ES，把命中的 chunk_id 映射成 ScoredChunk
        return hits


def build(*, index, embedder, settings, cfg):
    options = cfg.options or {}
    return EsRetriever(options.get("endpoint", ""), options.get("index", "rag_chunks"))
```

> 外部索引常常只有 `chunk_id`、没有正文，用核心那份块表换回正文即可：
> `await index.get_chunks([...])` —— `plugins/rag-channel-neo4j/channel.py` 就是这么做的。

装、配、跑：

```bash
pip install -e plugins/rag-channel-elasticsearch
```

```yaml
retrieval:
  channels:
    - {name: milvus, type: milvus, enabled: true}
    - {name: es, type: elasticsearch, enabled: true, complement_only: true, top_k: 10,
       options: {endpoint: "http://127.0.0.1:9200", index: rag_chunks}}
```

```bash
rag search "Flink 的检查点是什么" --channel milvus,es    # 只在这一次里带上它
```

卸掉 `pip uninstall rag-channel-elasticsearch` 之后，配置里那句 `type: elasticsearch`
会在下一次检索时**当场**报：`未知的通道类型 'elasticsearch'；内置：['milvus']；…`。

## 5. 命令也是插件

同一个包里再声明一个入口点即可，`app` 是一个 `typer.Typer()`：

```toml
[project.entry-points."rag.cli"]
es = "rag_channel_es.cli:app"
```

核心在 `src/rag/cli/main.py::_mount_plugin_commands` 里挂上去（名字 = 入口点名 →
`rag es …`）。两条规矩：**模块级挂**（Click 先解析子命令再进回调，挂晚了认不出来）、
**插件坏了不拖垮 CLI**（import 失败只打一行黄字警告）。

## 6. 现成样板：`plugins/rag-channel-neo4j`

Neo4j 那一整套（图库客户端 `store.py`、实体关系抽取 `extract.py`、命令 `cli.py`、
检索通道 `channel.py`）2026-10-08 全部从核心搬进了插件包，演示三件事：

1. **通道**：`type: neo4j` 靠 entry point 找到 `rag_channel_neo4j:build`；
2. **命令**：`rag graph extract / load / build / stats / query` 全来自插件
   （核心只有 `_mount_plugin_commands` 那 15 行）；
3. **已入库数据的补图**：图是从**切分产物**（`data/stages/<doc_id>/chunks.json`）抽的，
   那份产物在灌库时就落好了 —— 所以补图**不用重新解析、不用重新 embedding、不用重灌
   Milvus**，只花抽取这一次 LLM 的钱（按块缓存，抽过的不重花）：

```bash
rag graph build --all --dry-run     # 先算账：几块要抽、几块能复用
rag graph build --all               # 抽 + 写 Neo4j
```

## 7. 什么时候该做插件（而不是改核心）

* 接**新的数据源 / 外部服务** → 插件；
* 加"同一个库的另一种搜法" → **都不是** —— 那不是通道（通道 = 数据源），
  该是那一路内部的实现细节（例：关键词已经是 Milvus 通道里的 sparse 腿）；
* 改的是**核心契约**（`ScoredChunk` 的形状、融合算法、`Retriever` 协议）→ 改核心。

## 8. 现在还没有的

* `options` **不做 schema 校验**：插件自己的参数自己解释（写错在它那一层报）。
* **版本/兼容性没有握手**：插件 import 的是核心公开模块（`rag.contracts` / `rag.config` /
  `rag.indexing`…），核心改这些签名时插件要跟着改；多插件分发那一步得给入口点加
  `api_version` 之类的约定。
* **没有插件注册表/市场**：现在就是"仓库内目录 + `pip install -e`"。
