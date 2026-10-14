"""IndexGraph：把「解析 → 人工核对闸门 → 切分 → 写库」按**文件**铺开（M3）。

```
START ─► fanout ─┬─► [文件 A: 解析 → 闸门 → 切分] ─┐
                 ├─► [文件 B: 解析 → 闸门 → 切分] ─┼─► write ─► END
                 └─► [文件 C: …]                 ─┘
```

**它买到的是什么，以及不是什么**（说清楚，免得重复造）：

* **买到**：文件级并发（外部的 OCR / VLM / embedding 是瓶颈，串行跑等于把 GPU 晾着）、
  单个文件失败只影响它自己（其余照常落库）、每个文件的进度都能单独报出来。
* **不是**：断点续跑。那个我们**早就有** —— `data/stages/<doc_id>/` 的产物 + 注册表
  让重跑天然幂等，跑第二遍自动跳过做完的部分。所以这里**不引 sqlite checkpointer**
  （那要多一个依赖），LangGraph 只负责「这一次运行怎么铺开」。

闸门（`outline.tsv` 的「已核对」）仍走 `run_chunk` 里那道，不在这里重复实现：
没核对的文件会进 `pending`，CLI 照旧把「改哪个文件」打出来。
"""

from __future__ import annotations

import asyncio
import operator
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from rag.config.schema import Settings
from rag.contracts import Embedder
from rag.indexing.commit import commit_chunks
from rag.ingest.loaders import SkippedFile
from rag.ingest.loaders.base import rel_uri
from rag.ingest.stages import ArtifactStore, run_chunk, run_parse


class IndexState(TypedDict, total=False):
    """图的状态。

    ``Annotated[list, operator.add]`` 是**累加型**：每个文件节点回的列表会并进去，
    而不是互相覆盖 —— 这是 fan-out 之后把结果汇起来的唯一方式。
    """

    root: Path
    todo: list[Path]
    parsed: Annotated[list[str], operator.add]
    chunked: Annotated[list[Path], operator.add]
    pending: Annotated[list[SkippedFile], operator.add]
    skipped: Annotated[list[SkippedFile], operator.add]
    described: Annotated[int, operator.add]
    shells: Annotated[int, operator.add]
    written: int
    registered: int
    missing: list[str]


class IndexGraphResult(BaseModel):
    parsed: list[str] = Field(default_factory=list)
    chunked: list[str] = Field(default_factory=list)
    pending: list[SkippedFile] = Field(default_factory=list)
    skipped: list[SkippedFile] = Field(default_factory=list)
    described: int = 0
    shells: int = 0
    written: int = 0
    registered: int = 0
    missing: list[str] = Field(default_factory=list)


def build_index_graph(
    settings: Settings,
    *,
    ocr: Any,
    vision: Any,
    store: ArtifactStore,
    embedder: Embedder | None = None,
    concurrency: int | None = None,
    index: Any = None,
    registry: Any = None,
) -> Any:
    """组图。``concurrency`` 是**同时在跑的文件数**，默认取 ``graph.index_concurrency``。

    为什么要有上限：每个文件节点都可能去调 OCR / VLM / 线上 embedding，全都是外部服务。
    无限并发只会把它们打到限流，然后整批一起变慢。默认 1（等于以前的行为），
    想快就往上加 —— 但要跟上游的承载能力匹配。
    """
    limit = concurrency or max(settings.graph.index_concurrency, 1)
    semaphore = asyncio.Semaphore(limit)

    def fanout(state: IndexState) -> list[Send]:
        """按文件派发。

        **必须是条件边的路由函数，不能是普通节点** —— 普通节点的返回值要能并进 state，
        而 `Send` 是「派发指令」不是状态（langgraph 1.x 会报
        `InvalidUpdateError: Expected dict, got [Send(...)]`）。
        """
        return [Send("file", {"root": state["root"], "path": path}) for path in state["todo"]]

    async def file_node(state: dict[str, Any]) -> dict[str, Any]:
        path: Path = state["path"]
        root: Path = state["root"]
        async with semaphore:
            # `run_parse` / `run_chunk` 是**同步阻塞**的（OCR、VLM、模型推理都在里面）。
            # 直接在协程里调它们会把事件循环堵死，fan-out 就变成假的并发 ——
            # 丢进线程池才是真的并行（与 embedding / 图片描述那两处同一个做法）。
            parsed, chunked = await asyncio.to_thread(
                _process_one, path, root, settings, ocr, vision, embedder, store
            )
        return {
            "parsed": [rel_uri(path, root)] if parsed.parsed else [],
            "chunked": [path] if chunked.chunked else [],
            "pending": chunked.pending,
            "skipped": [*parsed.skipped, *chunked.skipped],
            "described": chunked.described,
            "shells": chunked.shells,
        }

    async def write_node(state: IndexState) -> dict[str, Any]:
        root = state["root"]
        jobs = [(path, root) for path in state.get("chunked") or []]
        # 同一个 embedder 传下去：它可能在切分那段（语义兜底）已经加载过一遍，
        # 这里再建一个就是两份模型常驻内存（本地那条路一个就是 1GB 量级）。
        result = await commit_chunks(
            settings, store, jobs, embedder=embedder, index=index, registry=registry
        )
        return {
            "written": result.written,
            "registered": len(result.registered),
            "missing": result.missing,
        }

    graph = StateGraph(IndexState)
    graph.add_node("file", file_node)
    graph.add_node("write", write_node)
    graph.add_conditional_edges(START, fanout, ["file"])
    graph.add_edge("file", "write")
    graph.add_edge("write", END)
    return graph.compile()


def _process_one(
    path: Path, root: Path, settings: Settings, ocr: Any, vision: Any, embedder: Any, store: Any
) -> tuple[Any, Any]:
    """一个文件的全部同步工作：解析 → （闸门 + 切分）。给线程池调用，所以是模块级函数。"""
    parsed = run_parse([path], root, settings, ocr, store)
    chunked = run_chunk([path], root, settings, embedder, vision, ocr, store)
    return parsed, chunked


async def run_index_graph(
    settings: Settings,
    *,
    todo: Sequence[Path],
    root: Path,
    ocr: Any,
    vision: Any,
    store: ArtifactStore,
    embedder: Embedder | None = None,
    concurrency: int | None = None,
    index: Any = None,
    registry: Any = None,
) -> IndexGraphResult:
    """跑一遍：把 ``todo`` 里的文件按文件铺开处理，最后统一写库。"""
    if not todo:
        return IndexGraphResult()
    graph = build_index_graph(
        settings,
        ocr=ocr,
        vision=vision,
        store=store,
        embedder=embedder,
        concurrency=concurrency,
        index=index,
        registry=registry,
    )
    state = await graph.ainvoke({"root": root, "todo": list(todo), "parsed": []})
    return IndexGraphResult(
        parsed=state.get("parsed") or [],
        chunked=[rel_uri(path, root) for path in state.get("chunked") or []],
        pending=state.get("pending") or [],
        skipped=state.get("skipped") or [],
        described=state.get("described") or 0,
        shells=state.get("shells") or 0,
        written=state.get("written") or 0,
        registered=state.get("registered") or 0,
        missing=state.get("missing") or [],
    )
