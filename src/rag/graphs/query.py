"""QueryGraph：带自我纠错的检索问答（M3）。

```
plan ──► retrieve ──► grade ──┬──► assemble ──► generate ──► END
  ▲                            │
  └──── 资料不够（轮数没用完） ─┘
```

**和文档里那张图的区别，以及为什么**：文档画的是
`plan→transform→fanout→fuse→rerank→assemble→generate→grade`。两处改动：

1. **fuse/rerank 收进 `retrieve` 一个节点**。从 M2 起，多通道融合与重排就是
   `MultiChannelRetriever` 内部的事（`build_retriever` 直接给你一路「检索器」）。
   编排层再把它拆成节点，等于把已经封好的东西摊开、还多几层状态传递。
2. **grade 判的是「资料够不够」，不是「答案对不对」**，而且放在 generate **之前**。
   判资料只需要看候选，不用先生成一遍 —— 省一次 LLM 调用，也避免「用生成的答案给自己
   打分」这种自我循环。答案层面的核查（引用准确性）属于 M6 评测。

**plan 与 grade 都要求模型输出 JSON**，不用 `with_structured_output`：那依赖网关的
tool calling / JSON mode 支持，而我们的供应商是 deepseek / dashscope / zhipuai / moonshot
四家走 OpenAI 兼容通道，支持程度不好赌。解析器对「带 ```json 围栏」「前后有废话」都容忍。
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from rag.config.schema import Settings
from rag.contracts import AssembledContext, ChannelStats, Citation, ScoredChunk
from rag.retrieval import assemble, rrf_fuse

# 计划里最多几条子查询。拆得越多，每次问答的 embedding 调用越多（按量计费），
# 所以默认只允许拆到 3 —— 再多就该让人自己拆成几个问题了。
MAX_SUB_QUERIES = 3

PLAN_PROMPT = """你是检索规划器。根据用户的问题，输出这次检索的计划。只输出 JSON：

{{"sub_queries": ["..."], "filters": {{}}, "channels": [], "reason": "一句话"}}

规则：
1. sub_queries：默认**只给 1 条**（把问题改写成更适合检索的样子，保留专有名词、编号、
   代码符号，去掉「请问」「帮我看看」这类废话）。只有当问题里明显有**多个互不相干的
   子问题**时才拆成多条，最多 {max_sub} 条。
2. filters：只在问题里**明说**了下面这些字段时才填，否则给空对象 {{}}：
   - modality：image / text / table（问「图里」时给 image）
   - doc_type：pdf / docx / html / xlsx / image
   - page：页码（整数）
   - heading_path：某章节的路径。**用模糊匹配**，写法 {{"like": "%关键词%"}}，
     例如问「第 2.1 节」→ {{"like": "%2.1%"}}。
   拿不准就给 {{}} —— 过滤条件是硬过滤，填错会**一条都召不回来**。
3. channels：**默认留空**（`[]`）—— 那表示走配置里启用的全部通道。列出来的通道是
   **数据源**的名字（从 {available} 里选），只有当问题明显需要某个特定来源的证据时
   才写它（比如关系类问题写图谱那一路）；写错的名字会被滤掉，但别浪费这一栏。
4. 只输出 JSON，不要解释、不要 Markdown 围栏。

上一轮检索被判为「不够」时的反馈（第一轮为空）：{notes}"""

GRADE_PROMPT = """你在判断「检索到的资料够不够回答用户的问题」。只输出 JSON：

{{"score": 0.0, "missing": "缺什么，一句话；不缺就空字符串"}}

评分口径：
- 0.8~1.0：资料直接回答了问题；
- 0.4~0.8：部分相关，可能不够完整；
- 0.0~0.4：基本无关。

**只看资料与问题的相关性，不要预先回答问题**，也不要因为资料少就扣分 —— 少但切题
就是高分。"""

# 检索函数的形状：问题 + 条数 + 计划里的 filters/channels → 候选 + 账本。
# 生产实现是 ``build_retriever`` 那一套；测试里换成替身就能把图跑起来（不连库）。
RetrieveFn = Callable[..., Awaitable[tuple[list[ScoredChunk], list[ChannelStats]]]]
JsonFn = Callable[[str], Awaitable[dict[str, Any]]]


class QueryState(TypedDict, total=False):
    """图的状态。每个节点只回自己改的那几个键（LangGraph 会合并）。"""

    question: str
    top_k: int
    round: int
    notes: str
    plan: dict[str, Any]
    hits: list[ScoredChunk]
    stats: list[ChannelStats]
    context: AssembledContext
    grade: dict[str, Any]
    answer: str
    citations: list[Citation]


class QueryGraphResult(TypedDict):
    answer: str
    hits: list[ScoredChunk]
    citations: list[Citation]
    rounds: int
    stats: list[ChannelStats]
    grade: dict[str, Any]


def parse_json_object(text: str) -> dict[str, Any]:
    """从模型输出里抠出那个 JSON 对象。

    容忍三种常见形态：纯净 JSON、带 ```json 围栏、前后夹了废话（取第一个 `{` 到最后一个 `}`）。
    解析不出来**报错**而不是给个默认值 —— 计划错了会静默查到错的东西，比报错难查得多。
    """
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"^```[a-zA-Z]*\s*", "", body)
        body = re.sub(r"\s*```$", "", body)
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"模型没有输出 JSON：{text[:200]!r}")
    payload = json.loads(body[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError(f"模型输出的不是 JSON 对象：{text[:200]!r}")
    return payload


def normalise_plan(raw: dict[str, Any], *, available: list[str]) -> dict[str, Any]:
    """把模型给的计划收敛成「一定安全」的形状。

    模型偶尔会多给几条子查询、或者编一个不存在的通道名。这里**收敛而不报错**：
    计划是给检索用的，多一条子查询只是多花一次 embedding，编个通道名却会让整次查询失败。
    通道名按可用列表过滤，全被过滤掉就退回主通道。
    """
    queries = [str(item).strip() for item in raw.get("sub_queries") or [] if str(item).strip()]
    channels = [str(item).strip() for item in raw.get("channels") or [] if str(item).strip()]
    filters = raw.get("filters")
    return {
        "sub_queries": queries[:MAX_SUB_QUERIES] or [""],
        # 留空 = 走配置里启用的那几路（不是"没计划"）；编出来的名字要滤掉，不然整次查询会挂
        "channels": [name for name in channels if name in available],
        "filters": filters if isinstance(filters, dict) else {},
        "reason": str(raw.get("reason") or ""),
    }


def build_query_graph(
    settings: Settings,
    *,
    retrieve: RetrieveFn,
    plan_llm: JsonFn,
    grade_llm: JsonFn,
    generate: Callable[[str, AssembledContext], Awaitable[str]],
    channels: list[str] | None = None,
) -> Any:
    """组图。所有外部依赖（检索、两个 LLM 调用、生成）都是**传进来的**。

    这样做不是为了好看：图本身要能被单测跑起来，而单测不该连 Milvus、也不该调线上模型。
    """
    cfg = settings.graph
    # 计划里能选的**通道名**（配置里的 `retrieval.channels[].name`）；空 = 不限制
    available = list(channels or [])
    max_rounds = max(cfg.max_corrective_rounds, 0)

    async def plan_node(state: QueryState) -> QueryState:
        prompt = PLAN_PROMPT.format(
            max_sub=MAX_SUB_QUERIES, available=available, notes=state.get("notes", "")
        )
        raw = parse_json_object(await _json_text(plan_llm, prompt, state["question"]))
        return {"plan": normalise_plan(raw, available=available)}

    async def retrieve_node(state: QueryState) -> QueryState:
        plan = state.get("plan") or {}
        queries = plan.get("sub_queries") or [state["question"]]
        filters = plan.get("filters") or None
        picked = plan.get("channels") or None
        ranked: list[list[ScoredChunk]] = []
        stats: list[ChannelStats] = []
        for sub_query in queries:
            hits, channel_stats = await retrieve(
                sub_query or state["question"],
                state["top_k"],
                filters=filters,
                channels=picked,
            )
            ranked.append(hits)
            for row in channel_stats:
                stats.append(_tag(row, sub_query))
        # 多个子查询的结果用**同一个 RRF** 合一次 —— 融合层本来就是这个职责，
        # 多查询展开与多通道召回在数学上没有区别。
        merged = (
            rrf_fuse(ranked, k=settings.retrieval.fusion.k)
            if len(ranked) > 1
            else (ranked[0] if ranked else [])
        )
        return {"hits": merged[: state["top_k"]], "stats": stats}

    async def grade_node(state: QueryState) -> QueryState:
        hits = state.get("hits") or []
        if not hits:
            # 零召回不用问模型 —— 它只会把「没有资料」再判一遍，白花一次调用
            return {"grade": {"score": 0.0, "missing": "一条都没召回到"}}
        if not state.get("question", "").strip():
            return {"grade": {"score": 1.0, "missing": ""}}
        context = _excerpt(hits)
        raw = parse_json_object(
            await _json_text(
                grade_llm, GRADE_PROMPT, f"问题：{state['question']}\n\n资料：\n{context}"
            )
        )
        try:
            score = float(raw.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        return {"grade": {"score": score, "missing": str(raw.get("missing") or "")}}

    def after_grade(state: QueryState) -> str:
        grade = state.get("grade") or {}
        enough = float(grade.get("score") or 0.0) >= cfg.relevance_threshold
        rounds_left = state.get("round", 0) < max_rounds
        return "assemble" if enough or not rounds_left else "plan"

    async def plan_again(state: QueryState) -> QueryState:
        """回炉：把「缺什么」喂给下一轮 plan，并把轮次 +1。"""
        return {
            "round": state.get("round", 0) + 1,
            "notes": (state.get("grade") or {}).get("missing") or "上一轮资料不够",
        }

    async def assemble_node(state: QueryState) -> QueryState:
        context = assemble(state.get("hits") or [], settings.retrieval.assemble.max_tokens)
        return {"context": context, "citations": context.citations}

    async def generate_node(state: QueryState) -> QueryState:
        context = state.get("context")
        if context is None:  # 理论上到不了；真到不了也别让 State 缺键炸在生成那一步
            context = AssembledContext(text="")
        return {"answer": await generate(state["question"], context)}

    graph = StateGraph(QueryState)
    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("plan_again", plan_again)
    graph.add_node("assemble", assemble_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade", after_grade, {"plan": "plan_again", "assemble": "assemble"}
    )
    graph.add_edge("plan_again", "plan")
    graph.add_edge("assemble", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


async def run_query_graph(
    graph: Any,
    question: str,
    *,
    top_k: int = 8,
) -> QueryGraphResult:
    """跑一遍图，把状态收敛成调用方要的那几样。"""
    state = await graph.ainvoke({"question": question, "top_k": top_k, "round": 0})
    return QueryGraphResult(
        answer=state.get("answer", ""),
        hits=state.get("hits") or [],
        citations=state.get("citations") or [],
        rounds=state.get("round", 0) + 1,
        stats=state.get("stats") or [],
        grade=state.get("grade") or {},
    )


async def _json_text(llm: JsonFn, instruction: str, payload: str) -> str:
    """两个 LLM 节点的共同形状：系统指令 + 具体内容 → 模型原文（解析交给上层）。"""
    return await llm(f"{instruction}\n\n{payload}")


def _tag(row: ChannelStats, sub_query: str) -> ChannelStats:
    """给账本标上是哪个子查询的 —— 多查询时否则分不清哪一路是谁的。"""
    if not sub_query:
        return row
    return row.model_copy(update={"channel": f"{row.channel}@{sub_query[:16]}"})


def _excerpt(hits: list[ScoredChunk], limit: int = 400) -> str:
    """给 grade 看的资料摘要：每条截断，避免把整池候选塞进 prompt。"""
    return "\n\n".join(
        f"[{index}] {hit.chunk.text[:limit]}" for index, hit in enumerate(hits, start=1)
    )
