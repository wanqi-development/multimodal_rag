"""图谱抽取（M6）：解析容错、按块缓存、失败隔离。

全部离线：模型是假的，产物落在 tmp_path 的 ArtifactStore 上（真读写，验的是缓存判据）。
这几条里最值钱的是**缓存**那几条 —— 抽错了要能重跑，但**重跑不该重复花钱**。

**实现在插件包里**（`plugins/rag-channel-neo4j`，2026-10-08 从核心搬出去）：
没装插件就跳过这个文件。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag.contracts import Chunk
from rag.ingest.stages import ArtifactStore
from rag.storage import LocalBlobStore

_extract = pytest.importorskip(
    "rag_channel_neo4j.extract",
    reason="图谱抽取是插件包（pip install -e plugins/rag-channel-neo4j）",
)
PROMPT_VERSION = _extract.PROMPT_VERSION
chat_model_name = _extract.chat_model_name
extract_document = _extract.extract_document
parse_extraction = _extract.parse_extraction
summarize_plan = _extract.summarize_plan
text_hash = _extract.text_hash

REPLY = json.dumps(
    {
        "entities": [{"name": "Flink", "type": "技术"}, {"name": "窗口", "type": "概念"}],
        "relations": [
            {"source": "Flink", "relation": "支持", "target": "窗口", "evidence": "支持窗口操作"}
        ],
    },
    ensure_ascii=False,
)


class _FakeChat:
    """假模型：按块给回复，或按回调给（用来制造失败与不同内容）。"""

    model_name = "fake-model"

    def __init__(self, reply: str | object) -> None:
        self._reply = reply
        self.calls: list[object] = []

    async def ainvoke(self, messages):  # noqa: ANN001, ANN201
        self.calls.append(messages)
        if callable(self._reply):
            return SimpleNamespace(content=self._reply(messages))
        return SimpleNamespace(content=self._reply)


def _chunks(count: int = 3) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"d1:{index:04d}",
            doc_id="d1",
            index=index,
            text=f"第 {index} 块：Flink 支持窗口操作",
            heading_path="第一章 -> 1.1",
        )
        for index in range(1, count + 1)
    ]


def _store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(LocalBlobStore(tmp_path / "stages"))


# ---- 解析容错 ----（模型爱套围栏、爱多说话）----


def test_parse_survives_fences_and_coerces_types() -> None:
    raw = (
        '好的：\n```json\n{"entities": [{"name": " Flink ", "type": null}, "垃圾"],\n'
        '"relations": [{"source": "Flink", "target": "窗口", "relation": "支持"}]}\n```'
    )

    entities, relations, error = parse_extraction(raw)

    assert error == ""
    assert entities == [{"name": "Flink", "type": ""}], "去空白、丢非字典项"
    assert relations[0]["target"] == "窗口"


def test_parse_drops_self_loops_and_halves() -> None:
    """自环与半截关系在图里没有信息量，还污染邻居查询。"""
    raw = json.dumps(
        {
            "entities": [{"name": "甲", "type": "x"}, {"name": "甲", "type": "y"}],
            "relations": [
                {"source": "甲", "target": "甲", "relation": "是"},
                {"source": "甲", "target": "", "relation": "是"},
                {"source": "甲", "target": "乙", "relation": "依赖"},
            ],
        }
    )

    entities, relations, _ = parse_extraction(raw)

    assert entities == [{"name": "甲", "type": "x"}], "同名实体只留第一次"
    assert [(r["source"], r["target"]) for r in relations] == [("甲", "乙")]


def test_parse_truncates_to_the_limits() -> None:
    raw = json.dumps(
        {
            "entities": [{"name": f"e{index}", "type": "x"} for index in range(20)],
            "relations": [
                {"source": f"e{index}", "target": f"e{index + 1}", "relation": "r"}
                for index in range(20)
            ],
        }
    )

    entities, relations, _ = parse_extraction(raw, max_entities=3, max_relations=2)

    assert len(entities) == 3 and len(relations) == 2


def test_unparseable_output_is_an_error_not_an_empty_extraction() -> None:
    """「解析不出来」与「这段没东西可抽」要分得开：前者要人看提示词。"""
    entities, relations, error = parse_extraction("我觉得这段没什么")

    assert entities == [] and relations == []
    assert "没有 JSON" in error


# ---- 缓存：重跑不重复花钱 ----（这一条最值钱）


def test_second_run_reuses_every_chunk(tmp_path: Path) -> None:
    store, chunks = _store(tmp_path), _chunks()
    first = _FakeChat(REPLY)

    once = asyncio.run(
        extract_document("d1", chunks, chat=first, store=store, model_name="fake-model")
    )
    second = _FakeChat(REPLY)
    twice = asyncio.run(
        extract_document("d1", chunks, chat=second, store=store, model_name="fake-model")
    )

    assert (once.extracted, once.reused) == (3, 0)
    assert (twice.extracted, twice.reused) == (0, 3), "第二次一次模型调用都不该有"
    assert second.calls == []
    # 每块 2 实体 + 1 关系，3 块 → 6 / 3；复用要真的把内容带回来
    assert twice.entities == 6 and twice.relations == 3


def test_only_the_changed_chunk_is_re_extracted(tmp_path: Path) -> None:
    store, chunks = _store(tmp_path), _chunks()
    asyncio.run(
        extract_document("d1", chunks, chat=_FakeChat(REPLY), store=store, model_name="fake-model")
    )
    chunks[1].text = "第 2 块改过了：Flink 也支持批处理"

    chat = _FakeChat(REPLY)
    result = asyncio.run(
        extract_document("d1", chunks, chat=chat, store=store, model_name="fake-model")
    )

    assert (result.extracted, result.reused) == (1, 2)
    assert len(chat.calls) == 1


def test_a_new_prompt_or_model_invalidates_the_whole_document(tmp_path: Path) -> None:
    """提示词/模型一变，旧产物不能再用 —— 否则改了提示词却看不出效果，最容易被骗。"""
    store, chunks = _store(tmp_path), _chunks()
    asyncio.run(
        extract_document("d1", chunks, chat=_FakeChat(REPLY), store=store, model_name="fake-model")
    )

    chat = _FakeChat(REPLY)
    result = asyncio.run(
        extract_document("d1", chunks, chat=chat, store=store, model_name="另一个模型")
    )

    assert result.extracted == 3 and result.reused == 0


def test_a_broken_chunk_does_not_take_down_the_batch(tmp_path: Path) -> None:
    def reply(messages):  # noqa: ANN001, ANN202
        text = messages[-1].content
        if "第 2 块" in text:
            return "模型今天不想干活"
        return REPLY

    result = asyncio.run(
        extract_document(
            "d1", _chunks(), chat=_FakeChat(reply), store=_store(tmp_path), model_name="fake-model"
        )
    )

    assert result.extracted == 3 and result.failed == 1
    assert result.chunks_with_content == 2


def test_force_re_extracts_everything(tmp_path: Path) -> None:
    store, chunks = _store(tmp_path), _chunks()
    asyncio.run(
        extract_document("d1", chunks, chat=_FakeChat(REPLY), store=store, model_name="fake-model")
    )

    chat = _FakeChat(REPLY)
    result = asyncio.run(
        extract_document("d1", chunks, chat=chat, store=store, model_name="fake-model", force=True)
    )

    assert result.extracted == 3 and result.reused == 0
    assert len(chat.calls) == 3


# ---- 产物与账 ----（人要看，dry-run 要算得准）


def test_the_artifact_records_versions_and_per_chunk_hashes(tmp_path: Path) -> None:
    store, chunks = _store(tmp_path), _chunks(2)
    asyncio.run(
        extract_document(
            "d1",
            chunks,
            chat=_FakeChat(REPLY),
            store=store,
            model_name="fake-model",
            source_uri="a.pdf",
        )
    )

    payload = json.loads(store.path_of("d1", "graph.json").read_text(encoding="utf-8"))

    assert payload["kind"] == "graph" and payload["prompt"] == PROMPT_VERSION
    assert payload["model"] == "fake-model" and payload["source_uri"] == "a.pdf"
    assert payload["chunks"]["d1:0001"]["text_hash"] == text_hash(chunks[0].text)
    assert store.load_graph("d1", prompt=PROMPT_VERSION, model="别的") is None


def test_dry_run_accounting_matches_what_would_be_spent(tmp_path: Path) -> None:
    store, chunks = _store(tmp_path), _chunks(3)
    # dry-run 算的是「对切分产物再抽一遍」，所以产物得先在（图谱产品不算输入）
    store.save_chunks("d1", chunks, source_hash="h1", source_uri="a.pdf", pipeline="p1")
    asyncio.run(
        extract_document("d1", chunks, chat=_FakeChat(REPLY), store=store, model_name="fake-model")
    )

    fresh = summarize_plan(store, ["d1"], limit=None, model="fake-model")
    other = summarize_plan(store, ["d1"], limit=None, model="另一个模型")

    assert fresh[0] == {"doc_id": "d1", "chunks": 3, "reusable": 3, "todo": 0}
    assert other[0]["todo"] == 3


def test_chat_model_name_has_a_fallback() -> None:
    """拿不到真名也要能区分「换了模型」—— 退回类名比空串安全。"""
    assert chat_model_name(_FakeChat(REPLY)) == "fake-model"
    assert chat_model_name(object()) == "object"
