"""报告：JSON（机器读）+ Markdown（人读）。

为什么两份：JSON 是给「下次跑完 diff 一下」用的（指标变没变、哪条掉了），
Markdown 是给人在 PR/聊天里看的 —— 只留 JSON 等于每次都要写脚本才看得懂，
只留 Markdown 等于没法比较。

报告里**一定要打出「没命中的那些**召回的是什么**」**：期望落空有两种完全不同的原因 ——
检索没找着（该调检索），或者黄金集的期望写歪了（该改判据）。不打出召回内容，这两种
分不出来，最后会变成「指标掉了但不知道为什么」。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from rag.eval.metrics import summarize
from rag.eval.runner import CaseResult


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def build_report(
    rows: Sequence[CaseResult],
    *,
    meta: dict[str, Any] | None = None,
    top_k: int = 8,
) -> dict[str, Any]:
    """汇总。检索指标只统计**有检索判据**的条目；语料外的问题只进拒答那一项。"""
    # **没命中的也进分母**：按 rank/top 过滤会把「一条都没召回」的条目悄悄剔掉，
    # 指标就变成「命中的那些有多准」——那不是我们要回答的问题
    retrieval_rows = [row for row in rows if row.retrieval_case]
    ranks = [row.rank for row in retrieval_rows]

    judged = [row for row in rows if row.verdict is not None]
    citation_scores = [
        row.verdict.citation_accuracy for row in judged if row.verdict.citation_accuracy is not None
    ]
    point_scores = [
        row.verdict.point_coverage for row in judged if row.verdict.point_coverage is not None
    ]
    # 两种口径都要看：**按问题平均**（macro）不让人数多的问题主导，
    # **按引用条数汇总**（micro）才反映「你看到的引用里有多少是假的」。
    # 实测差得很远：一条拒答把 8 条无关材料列成引用，macro 90% / micro 66.7%。
    judged_citations = [
        item for row in judged for item in row.verdict.citations if "supported" in item
    ]
    micro = (
        sum(1 for item in judged_citations if item["supported"]) / len(judged_citations)
        if judged_citations
        else None
    )
    # ragas 的 faithfulness（可选后端）：拒答/没上下文那类不适用，只对算出来的求平均
    faithfulness_scores = [row.faithfulness for row in judged if row.faithfulness is not None]
    # 漏拒（该拒没拒）与误拒（不该拒却拒了）分开数：它们的修法完全相反
    missed_refusal = [row for row in judged if row.must_refuse and not row.verdict.refused]
    false_refusal = [row for row in judged if not row.must_refuse and row.verdict.refused]

    return {
        "meta": {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            **(meta or {}),
        },
        "summary": {
            "cases": len(rows),
            "retrieval_cases": len(retrieval_rows),
            "judged_cases": len(judged),
            "retrieval": summarize(ranks, k=top_k) if retrieval_rows else {},
            "citation_accuracy": _mean(citation_scores),
            "citation_accuracy_micro": micro,
            "judged_citations": len(judged_citations),
            "point_coverage": _mean(point_scores),
            "ragas_faithfulness": _mean(faithfulness_scores),
            "ragas_cases": len(faithfulness_scores),
            "refused": sum(1 for row in judged if row.verdict.refused),
            "refusal_missed": len(missed_refusal),
            "refusal_false": len(false_refusal),
            "errors": sum(1 for row in rows if row.error or (row.verdict and row.verdict.error)),
        },
        "rows": [
            {
                "id": row.id,
                "question": row.question,
                "tags": list(row.tags),
                "retrieval_case": row.retrieval_case,
                "must_refuse": row.must_refuse,
                "rank": row.rank,
                "answer": row.answer,
                "context_size": row.context_size,
                "citations": row.citations,
                "faithfulness": row.faithfulness,
                "top": row.top,
                "error": row.error,
                "verdict": (
                    None
                    if row.verdict is None
                    else {
                        "citation_accuracy": row.verdict.citation_accuracy,
                        "point_coverage": row.verdict.point_coverage,
                        "refused": row.verdict.refused,
                        "citations": row.verdict.citations,
                        "points": row.verdict.points,
                        "error": row.verdict.error,
                    }
                ),
            }
            for row in rows
        ],
    }


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def render_markdown(report: dict[str, Any]) -> str:
    """人读版。整体数字在前，逐条明细在后。"""
    meta, summary = report["meta"], report["summary"]
    lines = [
        f"# 评测报告 {meta.get('label', '')}".rstrip(),
        "",
        f"- 时间：{meta.get('generated_at')}",
        f"- 数据集：{meta.get('dataset')}（{summary['cases']} 条，其中检索 "
        f"{summary['retrieval_cases']} 条、生成 {summary['judged_cases']} 条）",
        f"- 配置：{json.dumps(meta.get('config') or {}, ensure_ascii=False)}",
        "",
        "## 总体",
        "",
    ]
    retrieval = summary.get("retrieval") or {}
    if retrieval:
        lines.append("| 指标 | 值 |")
        lines.append("|---|---|")
        for name, value in retrieval.items():
            lines.append(f"| {name} | {_percent(value)} |")
        lines.append("")
    else:
        lines += ["（没有检索判据的条目）", ""]
    lines += [
        f"- 引用准确率：按问题平均 {_percent(summary.get('citation_accuracy'))}；"
        f"按引用条数汇总 {_percent(summary.get('citation_accuracy_micro'))}"
        f"（共判 {summary.get('judged_citations')} 条引用）",
        f"- 要点覆盖率：{_percent(summary.get('point_coverage'))}",
        f"- ragas faithfulness：{_percent(summary.get('ragas_faithfulness'))}"
        f"（{summary.get('ragas_cases')} 条适用）",
        f"- 拒答：判定为拒答 {summary.get('refused')} 条；"
        f"**漏拒 {summary.get('refusal_missed')} 条**（该拒没拒）、"
        f"误拒 {summary.get('refusal_false')} 条",
        f"- **判不了 / 出错：{summary.get('errors')} 条**（不算通过）",
        "",
        "## 逐条",
        "",
        "| # | 问题 | 首条命中排名 | 引用准确率 | 要点 | 说明 |",
        "|---|---|---|---|---|---|",
    ]
    for index, row in enumerate(report["rows"], 1):
        verdict = row.get("verdict") or {}
        note = row.get("error") or verdict.get("error") or ""
        if not note:
            misses = [item for item in row["top"] if not item["relevant"]]
            note = "全部命中" if row["top"] and not misses else ""
        lines.append(
            f"| {index} | {row['question'][:40]} | {row['rank'] or '—'} | "
            f"{_percent(verdict.get('citation_accuracy'))} | "
            f"{_percent(verdict.get('point_coverage'))} | {note[:60]} |"
        )
    lines.append("")

    # 只看**有检索判据**的条目：语料外的问题 rank 恒为 0，混进来会看着像检索失败
    misses = [row for row in report["rows"] if row["retrieval_case"] and row["rank"] == 0]
    if misses:
        lines += ["## 没命中的那些，召回的是什么", ""]
        for row in misses:
            lines.append(f"- **{row['question'][:50]}**")
            for item in row["top"][:3]:
                lines.append(
                    f"    - #{item['rank']} `{item['chunk_id']}` {item['heading_path'][:60]}"
                    f"（{item['score']}）"
                )
        lines.append("")
    return "\n".join(lines)


def write_report(report: dict[str, Any], *, out_dir: Path, label: str) -> tuple[Path, Path]:
    """落盘：`<label>-<时间>.json` + 同名 `.md`。返回两个路径。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    stem = f"{label}-{stamp}" if label else stamp
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    # 引用片段可能很长：报告是给人看的，`indent` 让 diff 也能读
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path
