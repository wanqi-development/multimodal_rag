"""**通道类型的解析 = 插件化**（2026-10-08 定，使用者点名"插件化"）。

    type 名 ──► ① 内置注册表 CHANNEL_TYPES
                ├─► ② entry points（组名 `rag.channels`）—— Java `ServiceLoader` 那一套
                └─► ③ 点路径 `包.模块:函数`（开发期还没打包时的逃生舱）

**接口约定**（第三方包只要实现这一个工厂函数）：

```python
def build(*, index, embedder, settings, cfg) -> Retriever:
    # 返回一路 Retriever：retrieve(query, top_k, filters, image) -> list[ScoredChunk]
    ...
```

`index` / `embedder` 可能是 None（纯外部服务、纯标量的通道用不上），`cfg` 是配置里
**那一项**（`name` / `type` / `enabled` / `complement_only` / `top_k` / `options`）——
插件自己的参数就写在那一项的 `options` 里，核心代码不认识它们，也不需要认识。

**装法（与 `META-INF/services` 同一个设计）**：在自己的包里声明入口点，

```toml
[project.entry-points."rag.channels"]
my_es = "my_pkg.channel:build"
```

`pip install` 之后，配置里写 `type: my_es` 就能用 —— **核心代码一行不改**。
这和 Java 里驱动包声明 `META-INF/services/java.sql.Driver`、`DriverManager` 按名字找
驱动的做法是同一个思路：**接口在核心，实现在包里，靠名字在运行时对上**。

**加载失败一律当场报错**（包没装、入口点名字写错、模块导不进来、对象不可调用），
并把"内置的有哪些、入口点里声明的有哪些"一起列出来 —— 与项目其它注册表同一个立场：
配了却不生效是最难查的一类问题，而"静默当作没有这一路"会让检索悄悄少一路数据源。
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from importlib import import_module, metadata
from typing import Any

# 第三方包声明入口点用的组名（配置里的 `type` 就是入口点名）
ENTRY_POINT_GROUP = "rag.channels"


def declare_entry_points() -> dict[str, str]:
    """已经装好的包在 `rag.channels` 组里声明了什么（名字 → `包.模块:函数`）。

    **不写进注册表**：这是"环境里有什么"的快照，解析时按需查，免得 import 期就扫一遍
    所有已装包（CLI 的启动速度是靠懒加载换来的）。
    """
    return {item.name: item.value for item in metadata.entry_points(group=ENTRY_POINT_GROUP)}


def load_target(target: str) -> Callable[..., Any]:
    """`包.模块:函数` → 那个对象。写错要**说清是哪一段错了**（模块找不到 / 函数没有）。"""
    module_name, separator, attribute = target.partition(":")
    if not separator or not attribute:
        raise ValueError(
            f"通道插件要写成「包.模块:函数」，收到的是 {target!r}（例：my_pkg.channel:build）"
        )
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise ImportError(f"通道插件 {target!r} 的模块导不进来（{module_name}）：{exc}") from exc
    try:
        factory = getattr(module, attribute)
    except AttributeError as exc:
        raise AttributeError(f"通道插件 {target!r}：{module_name} 里没有 {attribute!r}") from exc
    if not callable(factory):
        raise TypeError(f"通道插件 {target!r} 指向的 {attribute!r} 不可调用")
    return factory


@cache
def resolve_channel_type(type_name: str) -> Callable[..., Any]:
    """`type` 名 → 工厂。三条来源见模块说明；找不到就把**有什么**列清楚再报错。"""
    import rag.retrieval.channels  # noqa: F401  触发内置两路的注册
    from rag.contracts import CHANNEL_TYPES

    try:
        return CHANNEL_TYPES.get(type_name)
    except KeyError:
        builtin = list(CHANNEL_TYPES.names())

    if ":" in type_name:  # ③ 开发期的点路径（免安装）
        return load_target(type_name)

    declared = declare_entry_points()
    if type_name in declared:  # ② 装好的插件包
        return load_target(declared[type_name])

    raise KeyError(
        f"未知的通道类型 {type_name!r}。内置：{builtin}；"
        f"已装插件在 {ENTRY_POINT_GROUP!r} 组里声明的：{sorted(declared)}。"
        f"要加一路新数据源：写一个包实现 build(*, index, embedder, settings, cfg)，"
        f'并在它的 pyproject 里声明 [project.entry-points."{ENTRY_POINT_GROUP}"]，'
        f'或者先用点路径 `type: "包.模块:build"` 试。'
    )
