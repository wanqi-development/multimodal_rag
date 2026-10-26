"""生成：拿装配好的上下文回答问题，并带上引用编号。

引用编号由 retrieval/assembler 生成，这里只负责让模型"照着编号引用"。
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from rag.contracts import AssembledContext, format_sources
from rag.usage import absorb_usage

SYSTEM_PROMPT = """你是严谨的资料问答助手。规则：
1. 只依据「资料」部分回答，不要使用资料之外的知识；
2. 每条论断后用 [编号] 标注依据，编号必须来自资料里的方括号编号，可以引多个；
3. 资料不足以回答时，直接说明已有的资料无法确定，不要猜测或用常识补全；
4. 不要编造资料里没有的数字、名称或来源；
5. 用中文回答，简洁直接，不要复述问题、不要空话开场。
"""

USER_TEMPLATE = """资料：

{context}

问题：{query}"""

NO_CONTEXT_REPLY = "没有检索到相关资料，无法回答。"


class CitationGenerator:
    """把 AssembledContext 交给 chat model，产出带引用的答案。"""

    def __init__(self, model: BaseChatModel, name: str = "citation") -> None:
        self._model = model
        self.name = name

    async def generate(
        self, query: str, context: AssembledContext, *, usage: dict[str, Any] | None = None
    ) -> str:
        """一次性生成。

        `usage` 是**调用方传进来的容器**：给了就把这次调用的 token 用量填进去。
        不用实例属性 —— `heavy.generator` 是跨请求共用的单例，两个并发请求会互相盖掉。
        """
        if not context.text.strip():
            # 一条都没召回到时还去问模型，只会换回一段看起来合理的幻觉
            return NO_CONTEXT_REPLY
        reply = await self._model.ainvoke(
            [
                SystemMessage(SYSTEM_PROMPT),
                HumanMessage(USER_TEMPLATE.format(context=context.text, query=query)),
            ]
        )
        absorb_usage(usage, reply)
        # 来源清单由装配阶段定稿，让模型复述一遍只会错 —— 这里直接拼上去
        return message_text(reply) + format_sources(context.citations)

    async def astream(
        self, query: str, context: AssembledContext, *, usage: dict[str, Any] | None = None
    ):
        """逐块产出答案文本（M4 的 SSE 用）。

        **没有召回到任何资料时不流式**，直接产出那句「没有检索到相关资料」——
        与 `generate` 同一个立场：没依据就不要让模型开口，流式也一样。

        为什么不做「先流式、最后再补引用」：引用编号是装配阶段生成的（`[n]`），
        模型在流里已经带着它们，所以 `done` 事件只需要把引用表一起交出去。
        """
        if not context.text.strip():
            yield NO_CONTEXT_REPLY
            return
        messages = [
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage(USER_TEMPLATE.format(context=context.text, query=query)),
        ]
        async for piece in self._model.astream(messages):
            absorb_usage(usage, piece)
            text = message_text(piece)
            if text:
                yield text
        # 来源清单作为**最后一块**发出去：它就像答案的最后一段，客户端不用为它做任何事
        tail = format_sources(context.citations)
        if tail:
            yield tail


def message_text(reply: Any) -> str:
    """把回复取成字符串。

    部分模型返回的 content 是内容块列表（如 [{"type": "text", "text": ...}]），
    直接 str() 会拿到 Python 字面量而不是答案原文。
    """
    content = reply.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)
