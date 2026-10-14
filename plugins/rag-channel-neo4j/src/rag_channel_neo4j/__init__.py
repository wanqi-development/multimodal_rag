"""multimodal_rag 的 **Neo4j 通道插件**（`type: neo4j`）。

这个包是"零代码加一路通道"的样板：核心不认识 Neo4j 的通道，它按 **`type` 名**通过
entry points 找到这里（`[project.entry-points."rag.channels"] neo4j = …:build`，
等价于 Java 驱动包的 `META-INF/services/java.sql.Driver`）。

装：`pip install -e plugins/rag-channel-neo4j`（前提是同一个环境里已装 multimodal_rag）。
用：在 `config/*.yaml` 的 `retrieval.channels` 里加一项

```yaml
- name: kg
  type: neo4j
  enabled: true
  complement_only: true     # 只补位（插件自己的默认也是 true）
  top_k: 10
  options: {uri: "bolt://127.0.0.1:7687", hops: 2}
```

卸掉这个包，核心再配 `type: neo4j` 就会当场报"未知的通道类型"并列出内置的与已装的 ——
**通道的增删是包的事，不是核心代码的事**。
"""

from __future__ import annotations

from typing import Any

__all__ = ["GraphRetriever", "build"]


def __getattr__(name: str) -> Any:
    """**懒导出**：`rag_channel_neo4j.build` 被取用时才去 import `channel`。

    为什么必须懒：核心 CLI 在**模块级**挂插件命令（`rag.cli` 入口点），
    而 `import rag_channel_neo4j.cli` 会先执行这个包的 `__init__` ——
    在这里 import `channel` 就等于把 pymilvus 一起拉进 `rag --help`，
    而"CLI 启动不许碰重依赖"是本项目最在意的一条（`tests/test_cli_startup.py` 钉着）。
    """
    if name in __all__:
        from rag_channel_neo4j import channel

        return getattr(channel, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
