"""知识图谱抽取（M6）：逐 chunk 抽实体与关系，**先落产物、再谈写库**。

三条约束，都是这个项目已经踩过的教训：

1. **产物先行**：先写 `data/stages/<doc_id>/graph.json`，再谈写 Neo4j。抽错了要能人看、
   能重跑；而且**重跑不重复花钱** —— 缓存粒度是**单块**（正文哈希 + 提示词/模型版本），
   改了一段就只重抽那一段，而不是整篇。
2. **每块一次调用**：一次把这块的实体与关系抽完（不逐实体问），并**限制条数** ——
   噪声与成本一起压住。目录页、图片描述这类块本来就没东西可抽，交白卷是对的。
3. **只给这一段 + 它的章节路径**：不给全文。跨块的关系不用我们操心 —— 实体名相同，
   写进图之后自然就连上了。

产物形状（一篇一份）：

    {"kind": "graph", "schema": 1, "doc_id": ..., "prompt": "graph-extract-v1",
     "model": "deepseek-chat", "source_uri": ...,
     "chunks": {"<chunk_id>": {"text_hash": ..., "entities": [...], "relations": [...],
                               "error": ""}}}

提示词或模型一变（产物里的 `prompt` / `model` 对不上），整篇重抽；只有正文变，才按块重抽。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag.contracts import Chunk

# 改提示词就要改这个版本号：它进产物，用来判断缓存还能不能用
PROMPT_VERSION = "graph-extract-v1"
# 每块最多抽多少：噪声与成本一起压住（超出的截掉，不是报错）
MAX_ENTITIES = 8
MAX_RELATIONS = 6
# 给模型看的正文上限：够抽出这一段讲什么，又不至于把上下文吃光
MAX_TEXT_CHARS = 1200
# 关系证据的长度上限（原文里的一小段）
EVIDENCE_CHARS = 60

SYSTEM = "你是知识图谱抽取器。只输出要求的 JSON，不要解释、不要复述规则。"

PROMPT = """【章节】{heading}
【正文】
{text}

抽取这段里的实体与关系：
1. entities（最多 {max_entities} 个）：`name` 用**原文里的写法**（别改写、别翻译），
   `type` 给一个词（技术 / 组件 / 配置项 / 概念 / 指标 …）。只抽这段真的讲到的，不要补常识。
2. relations（最多 {max_relations} 条）：两端都必须是上面 entities 里的 name；
   `relation` 用简短短语（如「依赖」「是…的一部分」「用于」）；
   `evidence` 给原文里支撑它的一小段（≤{evidence} 字）。

没有可抽的就给空数组。只输出：
{{"entities": [{{"name": "", "type": ""}}],
 "relations": [{{"source": "", "relation": "", "target": "", "evidence": ""}}]}}"""


@dataclass
class ChunkGraph:
    """一块的抽取结果。`error` 非空 = 这次没抽成（不影响别的块）。"""

    chunk_id: str
    text_hash: str
    entities: list[dict[str, Any]] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""


@dataclass
class ExtractResult:
    """一篇文档的抽取账：调了几次模型、复用了几块、抽到多少东西。"""

    doc_id: str
    chunks: int = 0
    extracted: int = 0
    reused: int = 0
    failed: int = 0
    entities: int = 0
    relations: int = 0
    chunks_with_content: int = 0
    path: Path | None = None


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _as_str(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _entities(raw: Any, limit: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        name = _as_str(item.get("name"))
        if not name or name in seen:
            continue
        seen.add(name)
        out.append({"name": name, "type": _as_str(item.get("type"))})
        if len(out) >= limit:
            break
    return out


def _relations(raw: Any, limit: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        source, target = _as_str(item.get("source")), _as_str(item.get("target"))
        if not source or not target or source == target:
            # 自环与半截关系都是噪声：图里没有信息量，还污染邻居查询
            continue
        out.append(
            {
                "source": source,
                "relation": _as_str(item.get("relation")),
                "target": target,
                "evidence": _as_str(item.get("evidence"))[:EVIDENCE_CHARS],
            }
        )
        if len(out) >= limit:
            break
    return out


def parse_extraction(
    raw: str, *, max_entities: int = MAX_ENTITIES, max_relations: int = MAX_RELATIONS
) -> tuple[list, list, str]:
    """模型输出 → (entities, relations, error)。

    与评测那边同一个立场：**解析不出来就是解析不出来**（记 error），不要当成「没抽到」——
    两者处理方式不同：前者要人看一眼提示词，后者本来就该是空的（目录页）。
    """
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return [], [], "输出里没有 JSON"
    try:
        payload = json.loads(raw[start : end + 1])
    except ValueError as exc:
        return [], [], f"JSON 解析失败：{exc}"
    if not isinstance(payload, dict):
        return [], [], "JSON 顶层不是对象"
    return (
        _entities(payload.get("entities"), max_entities),
        _relations(payload.get("relations"), max_relations),
        "",
    )


async def extract_chunk(chat: Any, chunk: Chunk) -> tuple[list, list, str]:
    """抽一块。返回 (entities, relations, error) —— **不抛异常**，异常也记成 error。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    prompt = PROMPT.format(
        heading=chunk.heading_path or "（无章节）",
        text=chunk.text[:MAX_TEXT_CHARS],
        max_entities=MAX_ENTITIES,
        max_relations=MAX_RELATIONS,
        evidence=EVIDENCE_CHARS,
    )
    try:
        reply = await chat.ainvoke([SystemMessage(SYSTEM), HumanMessage(prompt)])
    except Exception as exc:  # noqa: BLE001 —— 单块失败不该带走整批
        return [], [], f"{type(exc).__name__}: {exc}"
    content = reply.content
    return parse_extraction(content if isinstance(content, str) else str(content))


def chat_model_name(chat: Any) -> str:
    """给缓存用的模型标识：拿不到真名就退回类名（够区分「换了模型」）。"""
    for attr in ("model_name", "model"):
        value = getattr(chat, attr, None)
        if isinstance(value, str) and value:
            return value
    return type(chat).__name__


def load_staged_chunks(store: Any, doc_id: str) -> tuple[str, list[Chunk]]:
    """读已经切好的产物（`<doc_id>/chunks.json`）。

    图谱抽取的输入是**切分产物**，不是原始文件 —— 这样它跟「块」是同一套边界，
    关系上的 `chunk_id` 才能直接映射回检索结果。
    """
    path = store.path_of(doc_id, "chunks.json")
    if path is None or not path.is_file():
        raise FileNotFoundError(
            f"没有 {doc_id} 的切分产物（{path}）：先跑 `rag ingest <路径> --stage chunk`"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks = [Chunk.model_validate(item) for item in payload.get("chunks") or []]
    return str(payload.get("source_uri") or doc_id), chunks


def staged_doc_ids(store: Any) -> list[str]:
    """所有切好的文档（`--all` 用）。"""
    root = Path(store.root)
    if not root.is_dir():
        return []
    return sorted(
        path.name for path in root.iterdir() if path.is_dir() and (path / "chunks.json").is_file()
    )


def _cached_entries(store: Any, doc_id: str, *, model: str) -> dict[str, dict]:
    payload = store.load_graph(doc_id, prompt=PROMPT_VERSION, model=model) or {}
    entries = payload.get("chunks") if isinstance(payload, dict) else {}
    return entries if isinstance(entries, dict) else {}


def summarize_plan(
    store: Any, doc_ids: Sequence[str], *, limit: int | None, model: str
) -> list[dict[str, Any]]:
    """`--dry-run` 要看的账：每篇几块、其中几块能复用（不会重复花钱）。

    `model` 要**真给**（`chat_model_name(chat)`）：空串会被产物里的模型名挡住，
    于是全部算成「要抽」—— 那这个 dry-run 就是在骗人。
    """
    out: list[dict[str, Any]] = []
    for doc_id in doc_ids:
        try:
            _, chunks = load_staged_chunks(store, doc_id)
        except (FileNotFoundError, ValueError) as exc:
            out.append({"doc_id": doc_id, "error": str(exc)})
            continue
        todo = list(chunks[:limit] if limit else chunks)
        entries = _cached_entries(store, doc_id, model=model)
        reusable = sum(
            1
            for chunk in todo
            if (entries.get(chunk.chunk_id) or {}).get("text_hash") == text_hash(chunk.text)
        )
        out.append(
            {
                "doc_id": doc_id,
                "chunks": len(todo),
                "reusable": reusable,
                "todo": len(todo) - reusable,
            }
        )
    return out


async def extract_document(
    doc_id: str,
    chunks: Sequence[Chunk],
    *,
    chat: Any,
    store: Any,
    source_uri: str = "",
    model_name: str = "",
    concurrency: int = 1,
    limit: int | None = None,
    force: bool = False,
) -> ExtractResult:
    """抽一篇。**按块复用缓存**：正文哈希对得上就不重抽（省的是真金白银）。"""
    todo = list(chunks[:limit] if limit else chunks)
    previous = {} if force else _cached_entries(store, doc_id, model=model_name)

    entries: dict[str, ChunkGraph] = {}
    pending: list[Chunk] = []
    result = ExtractResult(doc_id=doc_id, chunks=len(todo))
    for chunk in todo:
        digest = text_hash(chunk.text)
        cached = previous.get(chunk.chunk_id) or {}
        if cached.get("text_hash") == digest:
            entries[chunk.chunk_id] = ChunkGraph(
                chunk_id=chunk.chunk_id,
                text_hash=digest,
                entities=cached.get("entities") or [],
                relations=cached.get("relations") or [],
                error=cached.get("error") or "",
            )
            result.reused += 1
        else:
            pending.append(chunk)

    slots = asyncio.Semaphore(max(1, concurrency))

    async def one(chunk: Chunk) -> None:
        async with slots:
            entities, relations, error = await extract_chunk(chat, chunk)
        entries[chunk.chunk_id] = ChunkGraph(
            chunk_id=chunk.chunk_id,
            text_hash=text_hash(chunk.text),
            entities=entities,
            relations=relations,
            error=error,
        )

    if pending:
        await asyncio.gather(*(one(chunk) for chunk in pending))

    # 统计与落盘都按**切分顺序**：产物给人看，顺序要稳定才能 diff
    ordered = [entries[chunk.chunk_id] for chunk in todo if chunk.chunk_id in entries]
    result.extracted = len(pending)
    result.failed = sum(1 for item in ordered if item.error)
    result.entities = sum(len(item.entities) for item in ordered)
    result.relations = sum(len(item.relations) for item in ordered)
    result.chunks_with_content = sum(1 for item in ordered if item.entities or item.relations)
    result.path = store.save_graph(
        doc_id,
        prompt=PROMPT_VERSION,
        model=model_name,
        source_uri=source_uri,
        chunks=[item for item in ordered],
    )
    return result


__all__ = [
    "EVIDENCE_CHARS",
    "MAX_ENTITIES",
    "MAX_RELATIONS",
    "MAX_TEXT_CHARS",
    "PROMPT_VERSION",
    "ChunkGraph",
    "ExtractResult",
    "chat_model_name",
    "extract_chunk",
    "extract_document",
    "load_staged_chunks",
    "parse_extraction",
    "staged_doc_ids",
    "summarize_plan",
    "text_hash",
]
