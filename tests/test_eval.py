"""评测层（M6）：黄金集校验、指标、判定解析、跑一遍。

全部离线：检索与生成都用替身，判定用假模型 —— 这一层不能因为「要连库/要花钱」而没法测。
指标是最容易写错的一块（rank 从 1 还是 0 起、没命中算 0 还是算无限远），所以单独钉住。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from rag.contracts import Chunk, ScoredChunk
from rag.eval import (
    GoldenCase,
    build_report,
    load_golden,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    render_markdown,
    run_eval,
)
from rag.eval.dataset import is_relevant, parse_case
from rag.eval.judge import parse_verdict

GOLDEN = Path("eval/golden.jsonl")


def _hit(chunk_id: str, *, doc_id: str = "d1", heading: str = "第一章 -> 1.1", score=0.9):
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=f"{chunk_id} 的正文",
            heading_path=heading,
            metadata={"source_uri": "a.pdf"},
        ),
        score=score,
        channel="rrf",
        rank=0,
    )


# ---- 指标 ----（rank 的口径：1 起、0 = 没命中）----


def test_recall_counts_only_within_k() -> None:
    ranks = [1, 3, 0, 8]

    assert recall_at_k(ranks, 3) == 0.5, "排第 3 的算命中，排第 8 与没命中都不算"
    assert recall_at_k(ranks, 8) == 0.75
    assert recall_at_k([], 5) == 0.0


def test_missing_hits_pull_the_average_down() -> None:
    """没命中记 0（而不是丢掉这一条）—— 丢掉会让指标变成「命中的那些有多准」。"""
    assert reciprocal_rank([1, 0]) == 0.5
    assert reciprocal_rank([0, 0]) == 0.0


def test_ndcg_prefers_relevant_material_ranked_higher() -> None:
    assert ndcg_at_k([1], 8) == pytest.approx(1.0)
    assert ndcg_at_k([2], 8) < ndcg_at_k([1], 8)
    assert ndcg_at_k([0], 8) == 0.0


# ---- 黄金集 ----（人维护的文件，写错要当场拦住）----


def test_the_repo_golden_set_loads_and_covers_the_basics() -> None:
    """仓库里那份黄金集本身要能读、要够用（覆盖检索 / 生成 / 拒答三类）。"""
    cases = load_golden(GOLDEN)

    assert len(cases) >= 20
    assert sum(1 for case in cases if case.is_retrieval_case) >= 15
    assert any(case.must_refuse for case in cases), "语料外的问题至少要有几条"
    assert any(case.answer_points for case in cases), "要有带要点清单的条目"
    assert len({case.id for case in cases}) == len(cases)


def test_a_case_without_any_criteria_is_rejected() -> None:
    with pytest.raises(ValueError, match="至少要有一种判据"):
        parse_case({"id": "x", "question": "问题"})


def test_a_refusal_case_must_not_expect_a_hit() -> None:
    """语料外的问题写「期望命中」是自相矛盾 —— 拦下来。"""
    with pytest.raises(ValueError, match="must_refuse"):
        parse_case(
            {
                "id": "x",
                "question": "语料外的问题",
                "must_refuse": True,
                "expected_doc_ids": ["d1"],
            }
        )


def test_a_single_string_where_a_list_belongs_is_accepted_as_one_item() -> None:
    """手写 JSONL 时把列表写成字符串是最常见的手误，别让它变成逐字拆分。"""
    case = parse_case({"id": "x", "question": "q", "expected_doc_ids": "d1"})

    assert case.expected_doc_ids == ("d1",)


def test_duplicate_ids_and_comments_in_the_file(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    path.write_text(
        "# 注释行\n"
        '{"id": "a", "question": "q1", "expected_doc_ids": ["d1"]}\n'
        "\n"
        '{"id": "a", "question": "q2", "expected_doc_ids": ["d1"]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="id 重复"):
        load_golden(path)


def test_relevance_uses_the_narrowest_criterion_first() -> None:
    case = GoldenCase(id="x", question="q", expected_chunk_ids=("d1:0001",))

    assert is_relevant(case, chunk_id="d1:0001", doc_id="d1", heading_path="任何时候")
    assert not is_relevant(case, chunk_id="d1:0002", doc_id="d1", heading_path="任何时候")

    loose = GoldenCase(id="y", question="q", expected_doc_ids=("d1",), expected_headings=("1.1",))
    assert is_relevant(loose, chunk_id="d1:0009", doc_id="d1", heading_path="第一章 -> 1.1 小节")
    assert not is_relevant(loose, chunk_id="d1:0009", doc_id="d1", heading_path="第二章 -> 2.1")
    assert not is_relevant(loose, chunk_id="d2:0009", doc_id="d2", heading_path="1.1")


# ---- 判定解析 ----（模型很爱套围栏、把 true 写成字符串）----


def test_verdict_survives_code_fences_and_string_booleans() -> None:
    verdict = parse_verdict(
        "好的，结果如下：\n```json\n"
        '{"citations": [{"index": "[1]", "supported": "true"}], '
        '"points": [{"point": "甲", "covered": "false"}], "refused": false}\n```'
    )

    assert verdict.error == ""
    assert verdict.citations[0]["index"] == 1
    assert verdict.citations[0]["supported"] is True
    assert verdict.points[0]["covered"] is False
    assert verdict.citation_accuracy == 1.0
    assert verdict.point_coverage == 0.0


def test_unparseable_output_is_an_error_not_a_pass() -> None:
    """静默按「通过」处理等于把评测变成安慰剂。"""
    verdict = parse_verdict("我觉得还不错")

    assert verdict.error and verdict.citations == []
    assert verdict.citation_accuracy is None, "一条没判 → None（不是 1.0）"


# ---- 跑一遍 ----（替身检索 + 假判定）----


def _cases() -> list[GoldenCase]:
    return [
        GoldenCase(id="r-hit", question="命中", expected_doc_ids=("d1",)),
        GoldenCase(id="r-miss", question="不命中", expected_doc_ids=("d9",)),
        GoldenCase(id="refuse", question="语料外", must_refuse=True),
    ]


def test_run_eval_scores_retrieval_and_keeps_the_denominator() -> None:
    async def retrieve(question: str, top_k: int):
        return [] if question == "不命中" else [_hit("d1:0001")]

    rows = asyncio.run(run_eval(_cases(), retrieve=retrieve, top_k=5))
    report = build_report(rows, top_k=5, meta={"label": "t"})

    assert [row.rank for row in rows] == [1, 0, 0]
    assert report["summary"]["retrieval_cases"] == 2, "没命中的也要进分母"
    assert report["summary"]["retrieval"]["recall@5"] == 0.5
    assert report["summary"]["judged_cases"] == 0, "没给 generate/ask 就不该判"


def test_run_eval_judges_only_the_answer_cases() -> None:
    asked: list[str] = []

    async def retrieve(question: str, top_k: int):
        return [_hit("d1:0001")]

    async def generate(question: str, hits):
        return "答案 [1]", [{"index": 1, "source_uri": "a.pdf", "text": "片段"}]

    async def ask(prompt: str) -> str:
        asked.append(prompt)
        return json.dumps(
            {"citations": [{"index": 1, "supported": True}], "points": [], "refused": True}
        )

    rows = asyncio.run(run_eval(_cases(), retrieve=retrieve, generate=generate, ask=ask, top_k=5))
    report = build_report(rows, top_k=5)

    assert len(asked) == 1, "只有 must_refuse 那条要判（另外两条没有要点也不要求拒答）"
    assert report["summary"]["judged_cases"] == 1
    assert report["summary"]["refusal_missed"] == 0, "该拒的拒了"
    assert render_markdown(report).startswith("# 评测报告")


def test_one_broken_case_does_not_take_down_the_round() -> None:
    """单条失败（模型抽风、片段为空）不能带走整轮评测 —— 记 error 继续跑。"""

    async def retrieve(question: str, top_k: int):
        return [_hit("d1:0001")]

    async def generate(question: str, hits):
        raise RuntimeError("模型抽风")

    async def ask(prompt: str) -> str:
        return "{}"

    rows = asyncio.run(run_eval(_cases(), retrieve=retrieve, generate=generate, ask=ask, top_k=5))
    report = build_report(rows, top_k=5)

    assert rows[2].error.startswith("RuntimeError")
    assert report["summary"]["errors"] == 1
    assert report["summary"]["retrieval_cases"] == 2, "别的条目照常算"


def test_generate_and_ask_must_come_together() -> None:
    async def retrieve(question: str, top_k: int):
        return []

    async def generate(question: str, hits):
        return "答案", []

    with pytest.raises(ValueError, match="要么都给"):
        asyncio.run(run_eval(_cases(), retrieve=retrieve, generate=generate))


def test_only_the_citations_the_answer_used_are_judged() -> None:
    """装配给了 8 条编号，答案只用了 [2] —— 判定只看 [2]。

    第一次 `--judge` 跑出来 76% 就是栽在这儿：把「没用到的那些」也算进分母，等于在问
    「你有没有把材料引用完」，而引用准确率要回答的是「你做出的引用站得住吗」。
    """
    context = [
        {"index": index, "source_uri": "a.pdf", "text": f"片段{index}"} for index in (1, 2, 3)
    ]
    prompts: list[str] = []

    async def retrieve(question: str, top_k: int):
        return [_hit("d1:0001")]

    async def generate(question: str, hits):
        return "结论这样 [2]", context

    async def ask(prompt: str) -> str:
        prompts.append(prompt)
        return json.dumps({"citations": [{"index": 2, "supported": True}], "points": []})

    rows = asyncio.run(run_eval(_cases(), retrieve=retrieve, generate=generate, ask=ask, top_k=5))
    judged = [row for row in rows if row.verdict is not None][0]

    assert [item["index"] for item in judged.citations] == [2]
    assert judged.context_size == 3, "装配进上下文的条数仍要记下来（看它有没有超用）"
    assert "片段1" not in prompts[0] and "片段2" in prompts[0]


def test_citation_accuracy_is_reported_both_ways() -> None:
    """macro（按问题平均）与 micro（按引用条数）要分开看：一条问题塞 8 条假引用时，
    它们能差出 30 个百分点（实测 90% vs 66.7%）。"""
    cases = [
        GoldenCase(id="one-good", question="甲", must_refuse=True),
        GoldenCase(id="eight-bad", question="乙", must_refuse=True),
    ]

    async def retrieve(question: str, top_k: int):
        return [_hit("d1:0001")]

    async def generate(question: str, hits):
        # 「甲」只引用 1 条（对的），「乙」把 8 条无关材料都列成引用（错的）
        width = 1 if question == "甲" else 8
        marks = "".join(f"[{index}]" for index in range(1, width + 1))
        return f"答 {marks}", [
            {"index": index, "source_uri": "a.pdf", "text": f"片段{index}"}
            for index in range(1, width + 1)
        ]

    async def ask(prompt: str) -> str:
        good = "片段1" in prompt and "片段2" not in prompt
        return json.dumps(
            {
                "citations": (
                    [{"index": 1, "supported": True}]
                    if good
                    else [{"index": index, "supported": False} for index in range(1, 9)]
                ),
                "points": [],
            }
        )

    rows = asyncio.run(run_eval(cases, retrieve=retrieve, generate=generate, ask=ask, top_k=5))
    summary = build_report(rows, top_k=5)["summary"]

    assert summary["citation_accuracy"] == 0.5, "两个问题：(1.0 + 0.0) / 2"
    assert summary["citation_accuracy_micro"] == pytest.approx(1 / 9), "9 条引用里 1 条对"
    assert summary["judged_citations"] == 9
