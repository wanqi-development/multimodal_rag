# rag-channel-neo4j —— 插件样板（通道 + 命令）

multimodal_rag 的 **Neo4j 插件**：图库客户端、实体关系抽取、`rag graph …` 四条命令、
检索通道**全在这个包里**（2026-10-08 从核心里搬出来）。核心只认识「按名字找实现」这套
机制，**不认识 Neo4j 本身**；要写自己的插件，看 `docs/plugins.md`。

## 装 / 卸

```bash
pip install -e plugins/rag-channel-neo4j      # 装（前提：同一个环境已装 multimodal_rag）
pip uninstall rag-channel-neo4j               # 卸
```

装完会多出两样东西：配置里的 `type: neo4j`，以及 `rag graph …` 这一组命令。

## 用：当作检索通道

在 `config/*.yaml` 的 `retrieval.channels` 里加一项（`name` 随你取，日志/账本/`--channel`
都用它）：

```yaml
retrieval:
  channels:
    - {name: milvus, type: milvus, enabled: true}
    - name: kg
      type: neo4j
      enabled: true
      complement_only: true            # 只补位：与别的路重复的候选丢掉
      top_k: 10
      options: {hops: 2}               # 覆盖 retrieval.graph 那一份参数（可选）
```

或者临时开一路做对比：`rag search "Flink 和 Spark 的关系" --channel milvus,kg`。

## 用：灌图与排查

```bash
rag graph build --all --dry-run     # 先算账：几块要抽、几块能复用
rag graph build --all               # 抽（写 data/stages/<doc_id>/graph.json）+ 写 Neo4j
rag graph stats                     # 图里现在有什么
rag graph query "Flink 和 Spark 的关系"   # 只看实体链接与邻居，不连 Milvus
```

**已经灌过 Milvus 的语料不用重灌**：图是从**切分产物**抽的（`data/stages/<doc_id>/chunks.json`），
那份产物在灌库时就落好了 —— `rag graph build` 不会再解析、不会再 embedding，只花抽取这一次
LLM 的钱（按块缓存，抽过的不重花）。想重抽某几篇：`--doc-id <id>`；缓存全丢：`--force`。

## 它凭什么被找到

`pyproject.toml` 里这两行（Java 里对应 `META-INF/services/java.sql.Driver`）：

```toml
[project.entry-points."rag.channels"]
neo4j = "rag_channel_neo4j:build"

[project.entry-points."rag.cli"]
graph = "rag_channel_neo4j.cli:app"
```

装完之后：核心的 `retrieval/plugins.py` 按 `type: neo4j` 查到 `build`；
`rag.cli.main` 把 `rag_channel_neo4j.cli:app` 挂成 `rag graph …`。
**没装这个包时**，配置里写 `type: neo4j` 会当场报错（列出内置的与已装的），
`rag graph` 这个命令也不存在 —— 不会静默少一路。开发期不想装包，`type` 也可以直接写点路径
`type: "rag_channel_neo4j:build"`。

> `__init__.py` 用 `__getattr__` **懒导出** `build`：核心 CLI 在模块级挂插件命令，
> 这里一旦直接 import `channel` 就会把 pymilvus 拖进 `rag --help`（有用例钉这条）。

## 接口

一个函数，返回任何满足 `Retriever` 协议的对象（`name` / `last_stats` /
`async retrieve(query, top_k, filters, image)`）：

```python
def build(*, index, embedder, settings, cfg) -> Retriever: ...
```

`index` / `embedder` 可能是 `None`；`cfg` 是配置里那一项，插件自己的参数读 `cfg.options`。
