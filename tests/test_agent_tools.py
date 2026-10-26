"""Agent 层（M5 批次 A）：工具、引用登记、装配、跑一次。

全部用替身：不连 Milvus、不加载模型、不调线上 —— 这一层要能单独验。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rag.agent.build import SYSTEM_PROMPT, build_agent, run_agent
from rag.agent.tools import (
    CitationRegistry,
    WebSearchGate,
    build_tools,
    build_web_search_tool,
)
from rag.config.schema import (
    ChunkingConfig,
    IndexingConfig,
    IngestAssetsConfig,
    IngestConfig,
    IngestImagesConfig,
    IngestStagesConfig,
    Settings,
)
from rag.contracts import Chunk, ScoredChunk


def _hit(
    chunk_id: str = "d1:0001", *, text: str = "正文内容", modality: str = "text"
) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id="d1",
            text=text,
            modality=modality,
            heading_path="第一章 -> 1.1 小节",
            page=3,
            metadata={"source_uri": "a.pdf"},
        ),
        score=0.9,
        channel="rrf",
        rank=0,
    )


class _Retriever:
    """替身检索器：记下参数，返回预先排好的命中。"""

    def __init__(self, hits: list[ScoredChunk] | None = None) -> None:
        self.hits = hits if hits is not None else [_hit()]
        self.calls: list[dict] = []

    @property
    def last_stats(self) -> list:
        return []

    async def retrieve(self, query, top_k, filters=None, image=None):
        self.calls.append({"query": query, "top_k": top_k, "filters": filters})
        return self.hits[:top_k]


class _Heavy:
    def __init__(self, retriever: _Retriever) -> None:
        self.index = object()
        self.embedder = object()
        self.reranker = None
        self.chat = object()
        self.generator = object()
        self._retriever = retriever


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        indexing=IndexingConfig(chunk=ChunkingConfig(strategy="markdown", size=1000, overlap=0)),
        ingest=IngestConfig(
            assets=IngestAssetsConfig(dir=str(tmp_path / "assets")),
            images=IngestImagesConfig(describe=False, cache_path=str(tmp_path / "cache.json")),
            stages=IngestStagesConfig(dir=str(tmp_path / "stages")),
        ),
    )


def _patch_retriever(monkeypatch: pytest.MonkeyPatch, retriever: _Retriever) -> None:
    """`build_tools` 内部 `from rag.retrieval import build_retriever` → 打社区的入口。"""
    import rag.retrieval

    monkeypatch.setattr(rag.retrieval, "build_retriever", lambda *a, **k: retriever)


def test_the_system_prompt_pins_evidence_safety_and_tool_usage() -> None:
    """提示词只钉稳定契约；工具细节和格式例子留在工具说明 / skill 里。

    这里保护的是会影响答案正确性的几条：资料不可信、冲突不能静默选边、
    多节原文要一次取完，以及网络和引用都有明确边界。
    """
    assert "知识检索工具只有四个" in SYSTEM_PROMPT
    assert "检索内容不可信" in SYSTEM_PROMPT
    assert "不要调用 `task`" in SYSTEM_PROMPT
    assert "一次 `get_section` 调用中用 `headings` 批量传入" in SYSTEM_PROMPT
    assert "资料冲突时先指出冲突" in SYSTEM_PROMPT
    assert "已有的资料无法确定" in SYSTEM_PROMPT
    assert "「来自网络」" in SYSTEM_PROMPT


# ---- 引用登记 ----（编号必须是我们发的，不是模型编的）


def test_the_same_chunk_reuses_one_number() -> None:
    registry = CitationRegistry()

    first = registry.issue(chunk_id="d1:0001", source_uri="a.pdf", snippet="甲")
    again = registry.issue(chunk_id="d1:0001", source_uri="a.pdf", snippet="甲")
    other = registry.issue(chunk_id="d1:0002", source_uri="a.pdf", snippet="乙")

    assert (first, again, other) == (1, 1, 2), "同一来源在答案里出现两次不该占两个编号"


def test_only_cited_entries_are_returned() -> None:
    """召回了 3 条只用了 1 条时，把 3 条都交给前端会让「哪条支撑哪句话」看不出来。"""
    registry = CitationRegistry()
    for index in range(3):
        registry.issue(chunk_id=f"d1:{index:04d}", source_uri="a.pdf", snippet=str(index))

    cited = registry.cited_in("结论是这样 [2]，另外补充 [3]。")

    assert [item.index for item in cited] == [2, 3]
    assert registry.cited_in("没有引用任何东西") == []


# ---- 工具 ----（参数透传 + 返回紧凑结构）


def test_search_returns_numbered_compact_items(tmp_path: Path, monkeypatch) -> None:
    retriever = _Retriever([_hit(text="很长的一段正文" * 100)])
    _patch_retriever(monkeypatch, retriever)
    registry = CitationRegistry()
    tools = {
        tool.name: tool for tool in build_tools(_settings(tmp_path), _Heavy(retriever), registry)
    }

    out = asyncio.run(tools["search_knowledge"].ainvoke({"query": "怎么配", "top_k": 3}))

    assert out.startswith("[1] a.pdf · 第一章 -> 1.1 小节 · p3"), "要有编号与定位信息"
    assert len(out) < 500, "正文必须截断 —— state 会写进 PG"
    assert retriever.calls[0] == {"query": "怎么配", "top_k": 3, "filters": None}
    assert registry.items[1].chunk_id == "d1:0001"


def test_search_says_so_when_nothing_was_found(tmp_path: Path, monkeypatch) -> None:
    """空结果要明说「没查到」—— 返回空字符串会被模型当成「查到了但没内容」。"""
    retriever = _Retriever([])
    _patch_retriever(monkeypatch, retriever)
    tools = {
        t.name: t for t in build_tools(_settings(tmp_path), _Heavy(retriever), CitationRegistry())
    }

    out = asyncio.run(tools["search_knowledge"].ainvoke({"query": "不存在的东西"}))

    assert "没有检索到" in out


def test_image_search_forces_the_image_modality(tmp_path: Path, monkeypatch) -> None:
    """看图片走 `search_knowledge(modality="image")` —— 图与文同一个空间，不再有独立工具。"""
    retriever = _Retriever([_hit(modality="image")])
    _patch_retriever(monkeypatch, retriever)
    tools = {
        t.name: t for t in build_tools(_settings(tmp_path), _Heavy(retriever), CitationRegistry())
    }

    out = asyncio.run(tools["search_knowledge"].ainvoke({"query": "架构图", "modality": "image"}))

    assert retriever.calls[0]["filters"] == {"modality": "image"}
    assert out.startswith("[1]")


def test_the_tool_list_is_exactly_four(tmp_path: Path) -> None:
    """工具只有四个（2026-10-08 定）：通道查询只有 `search_knowledge` 一条路。

    关键词并进 Milvus 那一路，图片靠 `modality` 过滤，图谱是一路**通道** ——
    它们都不该再出现在工具清单里（多了就等于让模型分头调、再用文字拼两路结果）。
    """
    tools = build_tools(_settings(tmp_path), _Heavy(_Retriever([])), CitationRegistry())

    assert sorted(t.name for t in tools) == [
        "get_section",
        "list_documents",
        "search_knowledge",
        "web_search_tool",
    ]


def _sections_doc(tmp_path: Path, count: int = 4) -> Settings:
    """写一份 `d1` 的切分产物：第一章 -> 1.1 … 1.<count>，每块一个字。"""
    settings = _settings(tmp_path)
    doc_dir = tmp_path / "stages" / "d1"
    doc_dir.mkdir(parents=True)
    marks = "甲乙丙丁戊己庚辛"
    (doc_dir / "chunks.json").write_text(
        json.dumps(
            {
                "chunks": [
                    {
                        "chunk_id": f"d1:{index:04d}",
                        "heading_path": f"第一章 -> 1.{index}",
                        "text": marks[index - 1] * 20,
                    }
                    for index in range(1, count + 1)
                ],
                "source_uri": "a.pdf",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return settings


def test_get_section_can_take_several_sections_at_once(tmp_path: Path) -> None:
    """**一次要多节**（2026-10-08 加）。分几轮要，每轮都要把整段上下文重发一次模型 ——
    实测一次问答里 8 轮逐节读把输入 token 顶到了 12.3 万。"""
    settings = _sections_doc(tmp_path)

    out = asyncio.run(
        _tool(settings, "get_section").ainvoke({"doc_id": "d1", "headings": ["1.1", "1.2", "1.3"]})
    )

    assert "[1]" in out and "[2]" in out and "[3]" in out, "三节各发一个引用号"
    assert "甲" in out and "乙" in out and "丙" in out
    assert "丁" not in out, "没要的那节不该出现"
    assert "没找到" not in out


def test_get_section_reports_the_ones_it_could_not_find(tmp_path: Path) -> None:
    """没找到的要说出来 —— 不说的话，模型会以为「这几节是空的」。"""
    settings = _sections_doc(tmp_path)

    out = asyncio.run(
        _tool(settings, "get_section").ainvoke({"doc_id": "d1", "headings": ["1.1", "9.9"]})
    )

    assert "甲" in out and "没找到" in out and "9.9" in out


def test_get_section_counts_one_chunk_once(tmp_path: Path) -> None:
    """`2.2` 与 `2.2.1` 会同时命中同一个块 —— 同一次调用里只回一遍。"""
    settings = _settings(tmp_path)
    doc_dir = tmp_path / "stages" / "d1"
    doc_dir.mkdir(parents=True)
    (doc_dir / "chunks.json").write_text(
        json.dumps(
            {
                "chunks": [
                    {"chunk_id": "d1:0001", "heading_path": "二 -> 2.2 -> 2.2.1", "text": "甲甲"}
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    out = asyncio.run(
        _tool(settings, "get_section").ainvoke({"doc_id": "d1", "headings": ["2.2", "2.2.1"]})
    )

    assert out.count("甲甲") == 1 and out.count("[") == 1


def test_get_section_caps_how_many_it_takes_at_once(tmp_path: Path) -> None:
    """上限是「敢让模型一次多要」的前提：不限的话它会把整章拖进来，反而顶爆上下文。"""
    from rag.agent.tools import MAX_SECTIONS

    settings = _sections_doc(tmp_path, count=MAX_SECTIONS + 2)
    wanted = [f"1.{index}" for index in range(1, MAX_SECTIONS + 3)]

    out = asyncio.run(_tool(settings, "get_section").ainvoke({"doc_id": "d1", "headings": wanted}))

    assert f"一次最多取 {MAX_SECTIONS} 节" in out


def test_get_section_reads_the_artifacts(tmp_path: Path) -> None:
    """「把那一节原文给我」：只读切分产物，不动任何东西。"""
    settings = _settings(tmp_path)
    doc_dir = tmp_path / "stages" / "d1"
    doc_dir.mkdir(parents=True)
    (doc_dir / "chunks.json").write_text(
        json.dumps(
            {
                "chunks": [
                    {"chunk_id": "d1:0001", "heading_path": "第一章 -> 1.1", "text": "甲" * 50},
                    {"chunk_id": "d1:0002", "heading_path": "第二章 -> 2.1", "text": "乙" * 50},
                ],
                "source_uri": "a.pdf",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    out = asyncio.run(_tool(settings, "get_section").ainvoke({"doc_id": "d1", "heading": "2.1"}))

    assert "[1]" in out and "第二章 -> 2.1" in out and "乙" in out
    assert "甲" not in out, "只该回匹配到的那一节"


def test_get_section_without_heading_lists_the_outline(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    doc_dir = tmp_path / "stages" / "d1"
    doc_dir.mkdir(parents=True)
    (doc_dir / "chunks.json").write_text(
        json.dumps(
            {"chunks": [{"chunk_id": "d1:0001", "heading_path": "第一章 -> 1.1", "text": "甲"}]}
        ),
        encoding="utf-8",
    )

    out = asyncio.run(_tool(settings, "get_section").ainvoke({"doc_id": "d1"}))

    assert "章节" in out and "第一章 -> 1.1" in out


def test_get_section_reports_a_missing_artifact(tmp_path: Path) -> None:
    out = asyncio.run(_tool(_settings(tmp_path), "get_section").ainvoke({"doc_id": "nope"}))

    assert "没有" in out and "nope" in out


# ---- 网络工具 ----（使用者自研，M6 接进来：key 从 Secrets、base_url 从 providers）
#
# 图谱那三个用例随 `search_graph` 工具一起删了（2026-10-08）：图谱现在是**一路通道**
# （Neo4j，`complement_only`），它由检索层融合 —— 通道本身的行为在
# `test_graph_store.py` 里验（实体链接、邻居、映射回块），不再有"agent 工具"这一层。


def test_web_search_reports_a_missing_key_instead_of_raising(tmp_path: Path, monkeypatch) -> None:
    """没配 key 是**配置问题**，不是运行时异常：返回一句话，让 agent 换别的路。"""
    import rag.config

    monkeypatch.setattr(rag.config, "get_secrets", lambda: SimpleNamespace(zhipuai_api_key=None))
    from rag.agent.tools import build_web_search_tool

    out = build_web_search_tool(_settings(tmp_path)).invoke({"query": "Flink 1.16 发布时间"})

    assert "ZHIPUAI_API_KEY" in out


def test_web_search_takes_the_base_url_from_the_zhipu_chat_provider(tmp_path: Path) -> None:
    """base_url 从 `providers.chat.zhipuai` 取（非机密走 yaml 插值），key 从 Secrets 取。"""
    from rag.agent.tools import build_web_search_tool
    from rag.config.schema import ChatProviderConfig

    settings = _settings(tmp_path)
    settings.providers.chat["zhipuai"] = ChatProviderConfig(
        provider="zhipuai", model="glm-4-flash", base_url="http://zhipu.test/v4"
    )
    seen: dict = {}

    class _FakeClient:
        def __init__(self, **kwargs) -> None:
            seen.update(kwargs)

        class web_search:  # noqa: N801 —— 模仿 SDK 的嵌套结构
            @staticmethod
            def web_search(**kwargs):  # noqa: ANN003, ANN202
                seen["call"] = kwargs
                raise RuntimeError("到这儿就够了：只验客户端怎么建的")

    monkeypatch = pytest.MonkeyPatch()
    import zai

    monkeypatch.setattr(zai, "ZhipuAiClient", _FakeClient)
    try:
        out = build_web_search_tool(settings, SimpleNamespace(zhipuai_api_key="zk")).invoke(
            {"query": "q"}
        )
    finally:
        monkeypatch.undo()

    assert seen["api_key"] == "zk" and seen["base_url"] == "http://zhipu.test/v4"
    assert "搜索失败" in out, "这一轮故意让 SDK 抛，验的是客户端参数"


def test_web_results_enter_the_same_citation_ledger(tmp_path: Path) -> None:
    """网页结果也拿统一编号，不能再只回一段无法被前端 / 评测识别的纯文本。"""
    from rag.agent.tools import build_web_search_tool

    registry = CitationRegistry()
    result = SimpleNamespace(
        title="Flink 官方文档",
        content="Savepoint 是作业状态的一致快照。",
        link="https://example.test/flink/savepoint",
        media="Apache Flink",
        publish_date="2026-10-01",
        refer="flink-doc",
    )

    class _FakeClient:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            self.kwargs = kwargs

        class web_search:  # noqa: N801 —— 模仿 SDK 的嵌套结构
            @staticmethod
            def web_search(**kwargs):  # noqa: ANN003, ANN202
                return SimpleNamespace(search_result=[result])

    monkeypatch = pytest.MonkeyPatch()
    import zai

    monkeypatch.setattr(zai, "ZhipuAiClient", _FakeClient)
    try:
        out = build_web_search_tool(
            _settings(tmp_path),
            SimpleNamespace(zhipuai_api_key="zk"),
            gate=WebSearchGate(explicit=True),
            registry=registry,
        ).invoke({"query": "Flink Savepoint"})
    finally:
        monkeypatch.undo()

    assert out.startswith("[1] 来自网络 · Flink 官方文档")
    assert registry.items[1].source_type == "web"
    assert registry.items[1].url == "https://example.test/flink/savepoint"
    assert registry.items[1].title == "Flink 官方文档"


# ---- 网络闸门 ----（用户定的规则：只在语料查空、或用户明确要求时才允许联网）


def test_the_gate_refuses_before_the_corpus_was_even_searched() -> None:
    """「还没查就想联网」要拦住 —— 而且是机制拦，不是靠提示词嘱咐。"""
    gate = WebSearchGate()

    assert gate.allowed is False
    assert "先用 search_knowledge" in gate.refusal()


def test_the_gate_opens_when_the_corpus_came_up_empty() -> None:
    gate = WebSearchGate()

    gate.note_corpus(found=False)

    assert gate.allowed is True


def test_the_gate_stays_shut_when_the_corpus_had_answers() -> None:
    gate = WebSearchGate()

    gate.note_corpus(found=True)

    assert gate.allowed is False
    assert "以语料为准" in gate.refusal()
    # 空手而归之后又查到东西：状态是**或**关系，一旦有过就算有
    gate.note_corpus(found=False)
    assert gate.allowed is False


def test_an_explicit_request_opens_the_gate_even_with_corpus_hits() -> None:
    gate = WebSearchGate(explicit=True)
    gate.note_corpus(found=True)

    assert gate.allowed is True


def test_the_search_tools_report_to_the_gate(tmp_path: Path, monkeypatch) -> None:
    """语料工具**自己**上报「查过没有」—— 空手而归也算查过（不然闸门永远开不了）。"""
    gate = WebSearchGate()
    retriever = _Retriever([])  # 什么都查不到
    _patch_retriever(monkeypatch, retriever)

    asyncio.run(
        _tool(_settings(tmp_path), "search_knowledge", gate=gate).ainvoke({"query": "没有的东西"})
    )

    assert gate.corpus_attempted is True and gate.corpus_found is False
    assert gate.allowed is True, "查空之后就该放行网络"


def test_low_scores_count_as_no_corpus_content(tmp_path: Path, monkeypatch) -> None:
    """语义检索**永远**返回 top-k —— 判「语料里有没有」只能看分数。

    实测（bge-reranker-base）：语料内有答案 0.9985，语料外 0.0077 / 0.0978。
    这条钉住两件事：低分**放行联网**，并且**把话说破**（不然模型会拿边角料硬答）。
    """
    gate = WebSearchGate()
    retriever = _Retriever([_hit(text="毫不相干的内容")])
    retriever.hits[0].score = 0.0077
    _patch_retriever(monkeypatch, retriever)

    out = asyncio.run(
        _tool(_settings(tmp_path), "search_knowledge", gate=gate).ainvoke(
            {"query": "Python asyncio"}
        )
    )

    assert gate.corpus_attempted and gate.corpus_found is False
    assert gate.allowed is True, "低分 = 语料里没有，放行网络"
    assert "最高分只有 0.008" in out and "web_search_tool" in out
    assert "web-researcher" not in out, "已删除的 subagent 不能再出现在工具反馈里"


def test_high_scores_keep_the_gate_shut(tmp_path: Path, monkeypatch) -> None:
    gate = WebSearchGate()
    retriever = _Retriever([_hit(text="正好是答案")])
    retriever.hits[0].score = 0.99
    _patch_retriever(monkeypatch, retriever)

    out = asyncio.run(
        _tool(_settings(tmp_path), "search_knowledge", gate=gate).ainvoke({"query": "Flink"})
    )

    assert gate.corpus_found is True and gate.allowed is False
    assert "最高分" not in out, "命中了就别絮叨"


def test_the_web_tool_obeys_the_gate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("rag.config.get_secrets", lambda: SimpleNamespace(zhipuai_api_key=None))
    blocked = build_web_search_tool(_settings(tmp_path), gate=WebSearchGate())
    opened = build_web_search_tool(_settings(tmp_path), gate=WebSearchGate(explicit=True))

    assert "现在不能用网络搜索" in blocked.invoke({"query": "q"})
    # 放行之后才轮到「没配 key」那条 —— 说明闸门已经让路了
    assert "ZHIPUAI_API_KEY" in opened.invoke({"query": "q"})


# ---- skills ----（随包发布的那几个，格式错了要在这里就拦住，而不是等到运行时）


def test_the_shipped_skills_parse_and_are_named_after_their_directory() -> None:
    """skills 的规矩（上游定的）：`SKILL.md` 要有 frontmatter，`name` 必须等于目录名。

    写错不会报错，只会**静默不加载** —— 那种「配了却没生效」正是这个项目最烦的一类问题。
    """
    from deepagents.middleware import skills as sk

    from rag.agent.build import SKILLS_DIR

    paths = sorted(SKILLS_DIR.glob("*/SKILL.md"))
    assert paths, "随包的 skills 一个都没有？"
    for path in paths:
        meta = sk._parse_skill_metadata(  # noqa: SLF001 —— 上游没导出公开的解析函数
            path.read_text(encoding="utf-8"), str(path), path.parent.name
        )
        assert meta, f"{path} 解析失败（frontmatter 或 name 不对）"
        assert meta["name"] == path.parent.name


def _tool(settings: Settings, name: str, *, gate: WebSearchGate | None = None):
    return {
        t.name: t
        for t in build_tools(settings, _Heavy(_Retriever()), CitationRegistry(), gate=gate)
    }[name]


# ---- 装配与跑一次 ----（用假模型建真图；deepagents 支持）


def test_the_agent_has_no_subagents_and_keeps_the_web_tool(tmp_path: Path, monkeypatch) -> None:
    """**没有 subagent**（2026-10-08 使用者定）：一个 agent + 四个工具。

    通道融合在检索层做完，不需要派子 agent 分头查；网络那条**靠闸门管**
    （`WebSearchGate`），也不需要"主 agent 拿不到这个工具"来实现。
    """
    captured: dict = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return "graph"

    import deepagents

    monkeypatch.setattr(deepagents, "create_deep_agent", fake_create)
    graph, registry = build_agent(_settings(tmp_path), _Heavy(_Retriever()))

    assert graph == "graph" and isinstance(registry, CitationRegistry)
    assert captured["subagents"] == [], "一个 subagent 都不留"
    assert captured["system_prompt"] == SYSTEM_PROMPT, "提示词契约必须真的装进 agent"
    assert sorted(t.name for t in captured["tools"]) == [
        "get_section",
        "list_documents",
        "search_knowledge",
        "web_search_tool",
    ], "网络工具回主 agent —— 闸门是机制（WebSearchGate），不是靠不给工具"
    # skills 挂在 workspace 上：只读档（没给 workspace）不该挂一个不存在的路径
    assert captured["skills"] is None


def _env(method: str, data: Any, seq: int = 1) -> dict:
    """v3 的事件信封（`{type, method, seq, params: {…, data}}`）。"""
    return {
        "type": "event",
        "method": method,
        "seq": seq,
        "params": {"namespace": [], "timestamp": 0, "data": data},
    }


class _V3Stream:
    def __init__(self, events: list[dict]) -> None:
        self._events = events

    def __aiter__(self):
        async def walk():
            for item in self._events:
                yield item

        return walk()


class _FakeGraph:
    """替身图：按 **v3 事件流**回一段带引用的答案（不跑真模型）。

    `run_agent` 现在和界面走同一条路（`_stream_graph` → `astream_events(version="v3")`），
    所以替身要给 `astream_events`；`aget_state` 模拟 checkpointer 的「这一轮之前有什么」
    （没接 checkpointer 时抛 —— 真图就是这样）。
    """

    def __init__(self, *, checkpointer: object | None = None) -> None:
        self.configs: list = []
        self.checkpointer = checkpointer
        self.messages: list[Any] = []

    async def astream_events(self, payload: Any, config: Any = None, version: str = "v2"):
        from langchain_core.messages import AIMessage, HumanMessage

        self.configs.append(config)
        self.messages = [
            HumanMessage(content=str((payload.get("messages") or [{}])[-1].get("content") or "")),
            AIMessage(content="结论 [1]"),
        ]
        return _V3Stream([_env("values", {"messages": self.messages})])

    async def ainvoke(self, payload: Any, config: Any = None) -> dict:
        """老入口留着（真图也有）：直接给最终状态。"""
        await self.astream_events(payload, config)
        return {"messages": self.messages}

    async def aget_state(self, config: Any) -> Any:
        if self.checkpointer is None:
            raise RuntimeError("没有 checkpointer")
        from types import SimpleNamespace

        return SimpleNamespace(values={"messages": list(self.messages)}, tasks=[])


def test_run_agent_returns_the_answer_and_the_cited_sources() -> None:
    graph = _FakeGraph()
    registry = CitationRegistry()
    registry.issue(chunk_id="d1:0001", source_uri="a.pdf", snippet="正文")
    registry.issue(chunk_id="d1:0002", source_uri="a.pdf", snippet="没被引用")

    result = asyncio.run(run_agent(graph, "问题", registry=registry, thread_id="t1"))

    # 答案末尾会拼上来源清单（`引用来源：`）—— 使用者定的口径：引用不画成界面元素，
    # 而是在问答结束处说明引用了哪个文档的哪一章
    assert result.answer.startswith("结论 [1]")
    assert result.answer.endswith("引用来源：\n[1] a.pdf")
    assert [item.chunk_id for item in result.citations] == ["d1:0001"], "没用到的来源不该出现"
    assert result.retrieved == 2 and result.steps == 2 and result.thread_id == "t1"
    assert graph.configs == [{"configurable": {"thread_id": "t1"}}]


def test_run_agent_without_a_thread_has_no_config() -> None:
    graph = _FakeGraph()

    result = asyncio.run(run_agent(graph, "问题", registry=CitationRegistry()))

    assert graph.configs == [None], "没接 checkpointer 就不该带 config"
    assert result.thread_id is None


def test_a_checkpointer_forces_a_thread_id_even_for_one_shot_questions() -> None:
    """**langgraph 的硬要求**：接了 checkpointer 就必须给 thread_id，否则直接抛
    `Checkpointer requires one or more of the following 'configurable' keys: thread_id…`
    （实测在「问一个语料外的问题」那条上撞到的）。

    所以单轮也要生成一个 —— 顺带这次问答事后还能接着聊，返回结果里带着它。
    """
    graph = _FakeGraph(checkpointer=object())

    result = asyncio.run(run_agent(graph, "问题", registry=CitationRegistry()))

    assert result.thread_id, "自动生成了一个会话 id"
    assert graph.configs == [{"configurable": {"thread_id": result.thread_id}}]


def test_an_explicit_thread_id_wins() -> None:
    graph = _FakeGraph(checkpointer=object())

    result = asyncio.run(run_agent(graph, "接着问", registry=CitationRegistry(), thread_id="t9"))

    assert result.thread_id == "t9"
    assert graph.configs == [{"configurable": {"thread_id": "t9"}}]
