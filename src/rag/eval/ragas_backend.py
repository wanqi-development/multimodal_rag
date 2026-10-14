"""ragas 作为**可选**的评测后端（2026-10-05 装并跑通）。

两件事值得先写清楚：

1. **它导不进来，得垫一下**：ragas 0.4.3（当前最新）在 `ragas/llms/base.py` 里
   `from langchain_community.chat_models.vertexai import ChatVertexAI`
   （还有 `llms.vertexai` 的 `VertexAI`），
   而我们的 `langchain-community` 0.4.2 **已经把那两个模块移走了** —— 直接 `import ragas`
   就是 `ModuleNotFoundError`。ragas 只是把这两个名字再导出（`__all__`）与做类型判断，
   我们不用 Vertex，所以塞两个占位类进 `sys.modules` 即可。上游补上就能删。
2. **它是可选的**：默认不装也不影响任何东西（懒导入 + `--ragas` 才用）。
   我们自研的**引用准确率**仍然单独算 —— ragas 的 faithfulness 看的是「答案是否忠于上下文」，
   比「那个 `[n]` 指的那段原文是否支撑那句话」宽，两者互补而不是替代。

真机实测（deepseek-chat，同一段上下文）：

    有依据的答案 → faithfulness = 1.0
    编的答案     → faithfulness = 0.0
"""

from __future__ import annotations

import sys
import types
from collections.abc import Awaitable, Callable
from typing import Any

# 需要一个 async client：ragas 内部走 `agenerate`，传同步 client 会抛
# 「Cannot use agenerate() with a synchronous client.」（实测）。
FaithfulnessFn = Callable[[str, str, list[str]], Awaitable[float]]

_SHIMS = (
    ("langchain_community.chat_models.vertexai", "ChatVertexAI"),
    # 注意拼写：上游导的是 `VertexAI`（LLM 那个），模块在 llms 下
    ("langchain_community.llms.vertexai", "VertexAI"),
)


def ensure_ragas_importable() -> None:
    """补上 ragas 需要、而新版 langchain-community 已经移走的两个模块名。"""
    for module_name, symbol in _SHIMS:
        if module_name in sys.modules:
            continue
        try:
            __import__(module_name)
            continue
        except ImportError:
            pass
        module = types.ModuleType(module_name)
        setattr(module, symbol, type(symbol, (), {}))
        sys.modules[module_name] = module


def build_faithfulness(settings: Any, secrets: Any) -> FaithfulnessFn:
    """建一个 `faithfulness(问题, 答案, 上下文) -> 分数` 的异步函数（懒导入 ragas）。"""
    ensure_ragas_importable()
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections.faithfulness import Faithfulness

    name = getattr(settings.agent, "chat", "default")
    cfg = settings.providers.chat[name]
    key = getattr(secrets, f"{cfg.provider}_api_key", None)
    if not key:
        raise RuntimeError(
            f"providers.chat.{name} 是 {cfg.provider}，但 .env 里没有对应的 key："
            f"ragas 的判定要用它当裁判模型。"
        )
    llm = llm_factory(
        cfg.model,
        provider="openai",  # 国产供应商都走 OpenAI 兼容通道（与 providers/chat.py 同一个道理）
        client=AsyncOpenAI(api_key=key, base_url=cfg.base_url),
    )
    metric = Faithfulness(llm=llm)

    async def faithfulness(question: str, answer: str, contexts: list[str]) -> float:
        if not contexts:
            # 没有上下文可判（拒答那类）：ragas 无从判断，交给调用方当「没这项」
            raise ValueError("没有 retrieved_contexts，faithfulness 无从判起")
        result = await metric.ascore(
            user_input=question, response=answer, retrieved_contexts=contexts
        )
        return float(result.value)

    return faithfulness
