"""生成侧的判定（**花钱的那一半**）：引用是否真的支持论断、答案有没有覆盖要点、该不该拒答。

三件事**合成一次调用**：逐条引用判 N 次，24 个问题的评测就是几百次调用；把论断、引用清单、
要点清单一起给模型、让它逐条回一个 JSON，一个问题一次调用就够（deepseek 上这是几毛钱级）。

为什么引用准确率必须自己写（而不是等 ragas）：ragas 的 faithfulness 看的是「答案是否
忠于上下文」，而我们要的是**更窄**的一件事 —— 答案里那个 `[n]` 指的片段，是否真的支撑
它所在的那句话。它们对「引用了无关片段」的容忍度不一样，后者才是引用能不能点开的依据。

模型的输出只当**数据**看：解析不出来就记 `error` 并把这条判成「判不了」，绝不当成「通过」。
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from rag.eval.dataset import GoldenCase

AskFn = Callable[[str], Awaitable[str]]

SYSTEM = "你是严格的检索评测员。只输出要求的 JSON，不要解释、不要复述规则。"

# 引用片段给模型看的长度：够判「支持 / 不支持」，又不至于把上下文吃光
EVIDENCE_CHARS = 800

PROMPT = """【问题】{question}

【答案】
{answer}

【引用清单】（答案里的 [n] 对应的原文片段）
{citations}

【要点清单】（答案应当覆盖到的点）
{points}

逐条判定，规则：
1. 每条引用：**答案里用到该编号的那句话**是否真的被这段原文支持。原文没提到、只沾边、
   或者答案明显扩展过头（原文没说那么满），都算 false。别因为「大方向对」就给 true。
2. 每个要点：答案是否讲到（同义即可）。
3. refused：答案是否明确说「资料里没有 / 无法确定」，而不是硬答。

只输出这个 JSON：
{{"citations": [{{"index": 1, "supported": true, "why": "短原因"}}],
 "points": [{{"point": "要点原文", "covered": true}}],
 "refused": false}}"""


@dataclass
class JudgeVerdict:
    """一次判定的结果。解析失败时 `error` 非空、其余为空 —— **不要**当成通过。"""

    citations: list[dict[str, Any]] = field(default_factory=list)
    points: list[dict[str, Any]] = field(default_factory=list)
    refused: bool = False
    error: str = ""
    raw: str = ""

    @property
    def citation_accuracy(self) -> float | None:
        """判过的引用里「真被支持」的比例。一条引用都没有时给 None（不是 0，也不是 1）。"""
        judged = [item for item in self.citations if "supported" in item]
        if not judged:
            return None
        return sum(1 for item in judged if item["supported"]) / len(judged)

    @property
    def point_coverage(self) -> float | None:
        judged = [item for item in self.points if "covered" in item]
        if not judged:
            return None
        return sum(1 for item in judged if item["covered"]) / len(judged)


def _render_citations(citations: list[dict[str, Any]]) -> str:
    if not citations:
        return "（答案里没有引用编号）"
    return "\n".join(
        f"[{item.get('index')}] {item.get('source_uri', '')}\n"
        f"{(item.get('text') or '')[:EVIDENCE_CHARS]}"
        for item in citations
    )


def _render_points(points: tuple[str, ...]) -> str:
    if not points:
        return "（没有要点清单，这项略过）"
    return "\n".join(f"- {point}" for point in points)


def parse_verdict(text: str) -> JudgeVerdict:
    """从模型输出里抠出那个 JSON。

    模型很爱在 JSON 外面套 ```json 围栏或加一句「好的」，所以取**第一个 `{` 到最后一个 `}`**
    再解析；还失败就记 error（`judge_case` 会把这条标成判不了）。
    """
    verdict = JudgeVerdict(raw=text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        verdict.error = "输出里没有 JSON"
        return verdict
    try:
        payload = json.loads(text[start : end + 1])
    except ValueError as exc:
        verdict.error = f"JSON 解析失败：{exc}"
        return verdict
    if not isinstance(payload, dict):
        verdict.error = "JSON 顶层不是对象"
        return verdict
    verdict.citations = [item for item in payload.get("citations") or [] if isinstance(item, dict)]
    verdict.points = [item for item in payload.get("points") or [] if isinstance(item, dict)]
    verdict.refused = bool(payload.get("refused", False))
    # 逐条抠类型：模型把 true 写成 "true"、把 index 写成 "[1]" 都见过
    for item in verdict.citations:
        item["index"] = _as_index(item.get("index"))
        supported = item.get("supported")
        item["supported"] = supported if isinstance(supported, bool) else _as_bool(supported)
    for item in verdict.points:
        covered = item.get("covered")
        item["covered"] = covered if isinstance(covered, bool) else _as_bool(covered)
    return verdict


def _as_index(value: Any) -> int | None:
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else None


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "yes", "是", "1"}


async def judge_case(
    ask: AskFn, case: GoldenCase, *, answer: str, citations: list[dict[str, Any]]
) -> JudgeVerdict:
    """判一条问题。`ask` 是「prompt → 模型原文」，这样这一层能注入假模型单测。"""
    prompt = PROMPT.format(
        question=case.question,
        answer=answer or "（空答案）",
        citations=_render_citations(citations),
        points=_render_points(case.answer_points),
    )
    reply = await ask(prompt)
    verdict = parse_verdict(reply)
    if verdict.error:
        # 判不了就要说出来：静默按「通过」处理等于把评测变成安慰剂
        verdict.error = f"{verdict.error}（原始输出前 120 字：{reply[:120]!r}）"
    return verdict


def build_asker(heavy: Any) -> AskFn:
    """生产用的 `ask`：走 heavy.chat（与图上 plan/grade 同一个模型、同一份连接池）。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    system = SystemMessage(SYSTEM)

    async def ask(prompt: str) -> str:
        reply = await heavy.chat.ainvoke([system, HumanMessage(prompt)])
        content = reply.content
        return content if isinstance(content, str) else str(content)

    return ask
