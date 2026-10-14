"""生成层：提示词装配、空上下文兜底、回复取文本。

这里不碰网络：模型换成录制用的替身，只验证我们交给它的东西对不对。
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from rag.contracts import AssembledContext, Citation
from rag.providers import CitationGenerator, message_text
from rag.providers.generator import NO_CONTEXT_REPLY, SYSTEM_PROMPT


class _RecordingModel:
    """记下收到的消息，返回固定答复。"""

    def __init__(self, reply: str = "稀疏向量由内置 BM25 生成 [1]。") -> None:
        self.calls: list[list[Any]] = []
        self._reply = reply

    async def ainvoke(self, messages: list[Any], **kwargs: Any) -> AIMessage:
        self.calls.append(list(messages))
        return AIMessage(self._reply)


def _context(text: str = "[1] 来源：a.md\n甲") -> AssembledContext:
    return AssembledContext(
        text=text,
        citations=[Citation(index=1, chunk_id="c1", source_uri="a.md", snippet="甲")],
        tokens_estimate=8,
    )


# ---- 空上下文 ----


async def test_no_context_answers_without_asking_the_model() -> None:
    """一条都没召回到时还去问模型，只会换回一段看起来合理的幻觉。"""
    model = _RecordingModel()
    answer = await CitationGenerator(model).generate("问点什么", AssembledContext(text=""))

    assert answer == NO_CONTEXT_REPLY
    assert model.calls == []


async def test_whitespace_only_context_is_treated_as_empty() -> None:
    model = _RecordingModel()
    answer = await CitationGenerator(model).generate("问点什么", AssembledContext(text="  \n "))

    assert answer == NO_CONTEXT_REPLY
    assert model.calls == []


# ---- 提示词 ----


async def test_generate_sends_system_prompt_then_context_and_question() -> None:
    model = _RecordingModel()
    await CitationGenerator(model).generate("稀疏向量怎么生成？", _context())

    (messages,) = model.calls
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == SYSTEM_PROMPT
    assert isinstance(messages[1], HumanMessage)
    assert "[1] 来源：a.md\n甲" in messages[1].content
    assert "稀疏向量怎么生成？" in messages[1].content


async def test_context_reaches_the_model_with_citation_numbers_intact() -> None:
    """编号是模型唯一能引的东西，改一个字都会让答案里的 [n] 对不上引用表。"""
    model = _RecordingModel()
    text = "[1] 来源：a.md\n甲\n\n[2] 来源：b.pdf\n乙"
    await CitationGenerator(model).generate("q", _context(text))

    assert text in model.calls[0][1].content


def test_system_prompt_pins_the_citation_contract() -> None:
    """这几条是「带引用」的全部实现方式：换模型就靠提示词约束，不是靠后处理。"""
    assert "[编号]" in SYSTEM_PROMPT
    assert "只依据" in SYSTEM_PROMPT


# ---- 回复取文本 ----


def test_message_text_returns_string_content_as_is() -> None:
    assert message_text(AIMessage("直接是字符串")) == "直接是字符串"


def test_message_text_flattens_content_blocks() -> None:
    """部分模型的 content 是块列表，直接 str() 会拿到 Python 字面量。"""
    reply = AIMessage(
        content=[
            {"type": "text", "text": "第一段"},
            {"type": "tool_use", "id": "t1", "name": "x", "input": {}},
            {"type": "text", "text": "第二段"},
        ]
    )

    assert message_text(reply) == "第一段第二段"


def test_message_text_keeps_bare_string_blocks() -> None:
    assert message_text(AIMessage(content=["甲", {"type": "text", "text": "乙"}])) == "甲乙"


# ---- 与真实 BaseChatModel 的接缝 ----


async def test_generator_drives_a_real_base_chat_model() -> None:
    """替身只证明我们发了什么，这条证明 BaseChatModel 那条路本身走得通。"""
    model = FakeMessagesListChatModel(responses=[AIMessage("按资料 [1]，走 Milvus 内建 BM25。")])
    answer = await CitationGenerator(model).generate("稀疏向量怎么生成？", _context())

    # **来源清单是拼上去的**：答案末尾要能看出「引用了哪个文档的哪一章」，
    # 由装配阶段定稿，不让模型复述（复述会飘）。
    assert answer.startswith("按资料 [1]，走 Milvus 内建 BM25。")
    assert answer.endswith("引用来源：\n[1] a.md")
