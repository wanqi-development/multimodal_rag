"""观测：把 LangSmith trace 按配置打开（architecture §13）。

为什么单独有这么一层：LangChain / LangGraph 只在**环境变量**存在时才上报，而我们的开关在
`observability.langsmith` + `.env` 的 `LANGSMITH_API_KEY` 里。不显式搬这一次，就会出现
「配置里开了、实际一条 trace 都没有」—— 那是最难查的一类状态（既不报错，也没效果）。

返回值是**给人看的一句话**（开了 / 为什么没开），CLI 在有话说的时候打出来：
关着的时候闭嘴（默认档就是关的，不该每次刷屏），开着或配错了才出声。

两套环境变量都设：`LANGCHAIN_*` 是 LangChain 的老名字，`LANGSMITH_*` 是新名字。
只设一套迟早会撞上版本差异（本项目 langsmith 0.14 / langchain 1.x 混着用）。
"""

from __future__ import annotations

import os
from typing import Any


def configure_tracing(settings: Any, secrets: Any = None) -> str:
    """按配置打开 trace。返回状态串（**不抛异常**：观测挂了不该拖死干活的那条路）。"""
    if not settings.observability.langsmith:
        return "关闭（observability.langsmith=false）"
    if secrets is None:
        from rag.config import get_secrets

        secrets = get_secrets()
    key = getattr(secrets, "langsmith_api_key", None)
    project = settings.observability.project
    if not key:
        return "没开：observability.langsmith=true，但 .env 里没有 LANGSMITH_API_KEY"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = key
    os.environ["LANGCHAIN_PROJECT"] = project
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = key
    os.environ["LANGSMITH_PROJECT"] = project
    return f"已开：project={project}"
