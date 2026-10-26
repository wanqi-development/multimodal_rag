"""黄金集：读写 + 校验 + 「什么算答对」的判据。

一条记录 = 一个问题 + 判据。判据分两层写，因为**代价差一个数量级**：

* **检索层（免费）**：命中哪些 doc / 章节就算相关 —— 本地 embedding，随时能跑；
* **生成层（花钱）**：答案覆盖了哪些要点、引用是否真支持论断、语料外该不该拒答。

所以 `--judge` 是开关而不是默认：默认只跑检索那层（详见 `runner.py`）。

判据的写法（`is_relevant` 的顺序也就是优先级）：

1. 给了 `expected_chunk_ids` → 只认这几个块，最精确；
2. 给了 `expected_doc_ids` + `expected_headings` → doc 对上、且章节路径里含任一片段；
3. 只给 `expected_doc_ids` → 该文档的任一块都算（最宽松，适合「这份文档里应该有」）。

`expected_headings` 用**子串**匹配章节路径（如 `1.1.1`、`Transaction`）—— 章节路径是
`A -> B` 拼起来的，写全了又长又容易被 OCR 的空格坑（实测 `1.1.1 Flink的定义` 与
`1.1.1Flink的定义` 都出现过），子串是这里唯一稳的写法。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GoldenCase:
    """一个问题 + 判据。字段都可选，但**至少要有一条判据**（校验会拦）。"""

    id: str
    question: str
    # 检索侧
    expected_chunk_ids: tuple[str, ...] = ()
    expected_doc_ids: tuple[str, ...] = ()
    expected_headings: tuple[str, ...] = ()
    # 生成侧（只有 --judge 才用得上）
    answer_points: tuple[str, ...] = ()
    must_refuse: bool = False
    tags: tuple[str, ...] = ()
    notes: str = ""

    @property
    def is_retrieval_case(self) -> bool:
        return bool(self.expected_chunk_ids or self.expected_doc_ids or self.expected_headings)

    @property
    def is_judged_case(self) -> bool:
        """要不要调模型判：有要点、或者要求拒答。"""
        return bool(self.answer_points) or self.must_refuse


def _strings(raw: Any, field_name: str, case_id: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):  # 单值写成字符串是常见手误，别让它变成逐字拆分
        return (raw,)
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError(f"{case_id}: {field_name} 要是字符串列表，收到 {raw!r}")
    return tuple(raw)


def parse_case(raw: dict[str, Any], *, where: str = "?") -> GoldenCase:
    """一行 JSON → `GoldenCase`。缺字段、写错类型都当场报错并指明是哪一行。"""
    case_id = str(raw.get("id") or raw.get("question") or where)
    question = str(raw.get("question") or "").strip()
    if not question:
        raise ValueError(f"{where}: 缺 question")
    case = GoldenCase(
        id=case_id,
        question=question,
        expected_chunk_ids=_strings(raw.get("expected_chunk_ids"), "expected_chunk_ids", case_id),
        expected_doc_ids=_strings(raw.get("expected_doc_ids"), "expected_doc_ids", case_id),
        expected_headings=_strings(raw.get("expected_headings"), "expected_headings", case_id),
        answer_points=_strings(raw.get("answer_points"), "answer_points", case_id),
        must_refuse=bool(raw.get("must_refuse", False)),
        tags=_strings(raw.get("tags"), "tags", case_id),
        notes=str(raw.get("notes") or ""),
    )
    if not (case.is_retrieval_case or case.is_judged_case):
        raise ValueError(f"{where}: 至少要有一种判据（expected_* / answer_points / must_refuse）")
    if case.must_refuse and case.is_retrieval_case:
        # 语料外的问题**不该命中语料**：给它写期望命中就是自相矛盾
        raise ValueError(f"{where}: must_refuse 的条目不该再写 expected_*（它本来就该查不到）")
    return case


def load_golden(path: Path) -> list[GoldenCase]:
    """读 JSONL。空行与 `#` 注释行跳过（人维护的文件，允许留注释）。"""
    cases: list[GoldenCase] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            raw = json.loads(text)
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: 不是合法 JSON：{exc}") from None
        case = parse_case(raw, where=f"{path.name}:{number}")
        if case.id in seen:
            raise ValueError(f"{path}:{number}: id 重复：{case.id}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise ValueError(f"{path}: 一条都没有")
    return cases


def is_relevant(case: GoldenCase, *, chunk_id: str, doc_id: str, heading_path: str) -> bool:
    """这一块算不算「答对这条问题所需的材料」。判据顺序见模块说明。"""
    if case.expected_chunk_ids:
        return chunk_id in case.expected_chunk_ids
    if case.expected_doc_ids and doc_id not in case.expected_doc_ids:
        return False
    if case.expected_headings:
        return any(head in (heading_path or "") for head in case.expected_headings)
    return bool(case.expected_doc_ids)
