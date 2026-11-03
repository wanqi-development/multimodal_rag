"""给 Agent 用的工具，全部是下层能力的薄包装，不新写检索逻辑。

2026-10-30 起按证据来源拆给不同子 Agent：

* 关键词 —— 就是 Milvus 那一路里的 sparse 腿（dense + sparse 在服务端融合），
  不再有一条"只走词频"的独立通道；将来换 Elasticsearch 是**换通道**，不是加工具；
* 图片 —— 图与文由同一个多模态 embedding 编进同一空间，检索时走的是同一条通道，
  要看图块就给 `search_knowledge(modality="image")`；
* 图谱 —— `search_relationships` 只开图谱通道，交给 relationship-graph 子 Agent；
* 网络 —— 搜索与网页抓取留在主 Agent，统一引用编号，仍受同一个闸门约束。

三条设计约束，都来自前面的实测：

1. **返回紧凑结构**（编号 + 定位 + 截断正文）。agent 的 state 会被序列化进 PG
   （`checkpoints` / `checkpoint_blobs`），一次返回 8 块千字正文既吃上下文也吃库。
2. **引用编号由我们发**，不是模型编。每次工具调用返回的条目拿到**全局递增**的 `[n]`，
   同时把 `n → Citation` 记进 `EvidenceLedger`；最终答案里的 `[n]` 直接查这张表 ——
   否则模型自己编的编号与我们的引用表必然对不上。
3. **所有证据工具只读**。文件工具由 deepagents backend 提供；ToolPolicy 先决定
   哪些能力可见，给了 workspace 时 `write_file` / `edit_file` 再由 HITL 拦住。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool, tool

from rag.config.schema import Settings
from rag.contracts import EvidenceLedger

# 每条命中给模型看的正文上限：够判断相关性，又不至于把上下文吃光
SNIPPET_CHARS = 320
# get_section 一次最多回多少字
SECTION_CHARS = 2000
# get_section 一次最多取几节。**有这个上限才敢让模型一次多要**：不限的话它会把整章
# 一次拖进来，反而把上下文顶爆（实测一次问答里 8 次逐节读 = 输入 token 12.3 万）。
MAX_SECTIONS = 5


CitationRegistry = EvidenceLedger


@dataclass
class WebSearchGate:
    """网络搜索与网页抓取的闸门。

    只有用户明确允许时才执行。内部证据不足时闸门只记录事实并提示模型先询问用户，
    不自动联网。

    为什么做成闸门而不是写进提示词：这个项目一贯的立场是「能力靠机制，不靠模型自觉」——
    嘱咐它「记得先查语料」是赌它听话；而网络答案是**不可溯源到语料**的，混进回答里就破坏了
    「每句话都能点开」这条底线（§9.6 / §11.6 都是这个立场）。

    「查过没有」由内部证据工具自己记（`note_corpus`），用于提示模型该先问用户；
    它不会改变 `allowed` 的结果。
    """

    explicit: bool = False
    corpus_attempted: bool = False
    corpus_found: bool = False
    corpus_unknown: bool = False

    def note_corpus_unknown(self) -> None:
        """有候选但相关性未确认，不等同于无资料，也不授予联网权限。"""
        self.corpus_attempted = True
        self.corpus_unknown = True

    def note_corpus(self, *, found: bool) -> None:
        """语料检索工具上报一次结果（`found=False` 也是有用信息）。"""
        self.corpus_attempted = True
        self.corpus_found = self.corpus_found or found

    @property
    def allowed(self) -> bool:
        return self.explicit

    def refusal(self) -> str:
        """被拦住时给模型的话：**说清为什么，以及下一步该干什么**。"""
        if not self.corpus_attempted:
            return (
                "用户还没有明确允许联网。先调用内部子 Agent 取证；"
                "内部证据不足时，在答案里询问用户是否允许联网，不要直接搜索或抓取网页。"
            )
        if self.corpus_found:
            return (
                "内部已找到相关资料或读到原文，但不代表事实已核验。"
                "如果用户要求核对网络，仍须先取得明确联网授权。"
            )
        if self.corpus_unknown:
            return "内部返回候选，相关性未确认。可继续取证；联网仍须先询问用户并取得明确授权。"
        return "内部证据不足。先询问用户是否允许联网，用户确认后才能调用网络工具。"


def build_tools(
    settings: Settings,
    heavy: Any,
    registry: CitationRegistry,
    *,
    gate: WebSearchGate | None = None,
) -> list[Any]:
    """建这一批工具。``heavy`` 里带着 embedder / reranker（**不在这里重建模型**）。"""
    from rag.service.retrieval import RetrievalService, assess_relevance

    service = RetrievalService(
        settings,
        heavy.index,
        heavy.embedder,
        heavy.reranker,
        heavy.chat,
    )

    async def _search(
        query: str,
        top_k: int,
        *,
        modality: str | None = None,
        channels: list[str] | None = None,
    ) -> str:
        if channels and not any(item.name in channels for item in settings.retrieval.channels):
            return f"没有可用的检索通道：{', '.join(sorted(channels))}。"
        filters = {"modality": modality} if modality else None
        result = await service.retrieve(query, top_k, filters=filters, channels=channels)
        hits = result.chunks
        relevance = assess_relevance(result, settings.agent.web_min_corpus_score)
        best = max((float(hit.score) for hit in hits), default=0.0)
        if gate is not None:
            if relevance == "unknown":
                gate.note_corpus_unknown()
            else:
                gate.note_corpus(found=relevance == "relevant")
        degradation = (
            "\n\n（检索降级：" + "、".join(item.code for item in result.degradations) + "。）"
            if result.degradations
            else ""
        )
        if not hits:
            # 空结果与有候选但相关性未知分开描述。
            return f"没有检索到与「{query}」相关的内容。" + degradation
        lines: list[str] = []
        for hit in hits:
            chunk = hit.chunk
            number = registry.issue(
                chunk_id=chunk.chunk_id,
                source_uri=str(chunk.metadata.get("source_uri", "")),
                snippet=chunk.text[:80],
                heading_path=chunk.heading_path,
            )
            where = " · ".join(
                part
                for part in (
                    str(chunk.metadata.get("source_uri", "")),
                    chunk.heading_path,
                    f"p{chunk.page}" if chunk.page else "",
                    f"模态={chunk.modality}" if chunk.modality != "text" else "",
                )
                if part
            )
            lines.append(f"[{number}] {where}\n{chunk.text[:SNIPPET_CHARS]}")
        text = "\n\n".join(lines)
        if relevance == "unknown":
            text += (
                "\n\n（返回候选，相关性未确认；可换关键词或读取原文继续取证，"
                "联网仍需用户明确授权。）"
            )
        elif relevance == "irrelevant":
            # 只有本次成功重排后的分数才与阈值比较。
            text += (
                f"\n\n（提示：以上结果的最高分只有 {best:.3f}，"
                f"低于「算作相关」的阈值 {settings.agent.web_min_corpus_score} —— "
                "内部证据很可能不足：可以换关键词再试；仍不足时请询问用户是否允许联网，"
                "不要自行调用 web_search_tool。）"
            )
        return text + degradation

    @tool
    async def search_knowledge(query: str, top_k: int = 8, modality: str | None = None) -> str:
        """检索文档语料：多通道并发 → 融合 → 重排，一次调用就是文档侧全部召回。

        语义（融合向量）、关键词（Milvus 内建 BM25）以及文档类附加通道都在这一次调用里
        融合完。关系图谱由 `search_relationships` 单独负责，不混进文档子 Agent。

        Args:
            query: 检索词。把问题改写成适合检索的样子，保留专有名词与编号。
            top_k: 返回几条，默认 8，最多 20。
            modality: 只看某种模态：`image`（图片描述块）/ `text` / `table`。
                问「图里画了什么」时给 `image`。
        """
        channels = [
            item.name for item in settings.retrieval.channels if item.enabled and item.name != "kg"
        ]
        return await _search(
            query,
            max(1, min(top_k, 20)),
            modality=modality,
            channels=channels,
        )

    @tool
    async def search_relationships(query: str, top_k: int = 10) -> str:
        """只查询知识图谱通道，返回实体关系、路径及其原始证据。

        Args:
            query: 包含人物、部门、组织或关系问题。保留完整姓名和部门名。
            top_k: 最多返回几条关系证据，默认 10，最多 20。
        """
        return await _search(
            query,
            max(1, min(top_k, 20)),
            channels=["kg"],
        )

    @tool
    async def list_documents() -> str:
        """列出语料里有哪些文档、各有多少块。**先看地图再查**，避免拿着问题瞎试。"""
        from rag.indexing import build_registry

        records = await build_registry(settings).all()
        if not records:
            return "语料库是空的（还没灌过数据）。"
        lines = [
            f"- {record.source_uri}（doc_id={doc_id}，{record.chunk_count} 块）"
            for doc_id, record in sorted(records.items(), key=lambda kv: -kv[1].chunk_count)
        ]
        return "语料库里的文档：\n" + "\n".join(lines)

    @tool
    async def get_section(doc_id: str, heading: str = "", headings: list[str] | None = None) -> str:
        """把某一节/某份文档的**原文**取出来。检索片段不够用时用它看全貌。

        **一次可以要多节**（最多 5 节）：确知缺哪几段时就一次要全 —— 分几轮要，每轮都要
        把整段上下文再发一次模型，那是这条链路上最贵的部分。

        Args:
            doc_id: 文档 id（先用 `list_documents` 拿）。
            heading: 章节路径里的关键词（如 `2.2.1`）。留空则返回该文档所有章节的标题清单。
            headings: 想一次要的多节（章节关键词列表，如 `["2.2.1", "2.2.2"]`）。
                与 `heading` 可同时给；两个都留空就是看章节清单。
        """
        return _section(settings, registry, doc_id, heading, headings=headings, gate=gate)

    return [
        search_knowledge,
        search_relationships,
        list_documents,
        get_section,
        # key 由它自己从 Secrets 取（见函数说明）：build_tools 不该被塞一堆机密
        build_web_search_tool(settings, gate=gate, registry=registry),
        build_web_fetch_tool(settings, gate=gate, registry=registry),
    ]


def build_web_fetch_tool(
    settings: Settings,
    *,
    gate: WebSearchGate | None = None,
    registry: CitationRegistry | None = None,
) -> BaseTool:
    """抓取一个公开网页并只返回正文摘要。

    复用 ingest 的 URL 安全校验：只允许 http/https、每一跳都拒绝非公网地址。
    网页抓取不落上传目录，也不触发 OCR / VLM；它只为主 Agent 提供当轮证据。
    """

    @tool
    async def web_fetch_tool(url: str) -> str:
        """读取一个公开网页的正文，用于核对搜索结果是引用内容。

        只能在用户明确允许联网后使用；优先抓网络搜索返回的链接，不要枚举站点。

        Args:
            url: 完整的 http/https 地址。
        """
        if gate is not None and not gate.allowed:
            return f"现在不能抓取网页：{gate.refusal()}"

        from bs4 import BeautifulSoup

        from rag.ingest.fetch import FetchError, fetch_url
        from rag.ingest.loaders.base import decode_text

        max_bytes = max(1, int(settings.ingest.uploads.max_mb)) * 1_000_000
        try:
            fetched = await asyncio.to_thread(
                fetch_url,
                url,
                max_bytes=max_bytes,
                timeout_s=15.0,
                max_images=0,
            )
        except FetchError as exc:
            return f"网页抓取失败：{exc}"
        except Exception as exc:  # noqa: BLE001
            return f"网页抓取失败：{type(exc).__name__}: {exc}"

        suffix = fetched.name.rsplit(".", 1)[-1].lower() if "." in fetched.name else ""
        if suffix in {"html", "htm"}:
            soup = BeautifulSoup(fetched.data, "html.parser")
            for node in soup(["script", "style", "noscript"]):
                node.decompose()
            title = (soup.title.get_text(" ", strip=True) if soup.title else "") or fetched.name
            text = soup.get_text("\n", strip=True)
        elif suffix in {"md", "markdown", "txt"}:
            title = fetched.name
            text = decode_text(fetched.data)
        else:
            return (
                f"抓到了 {fetched.name}，但它不是可直接阅读的网页正文。"
                "PDF、Office 或图片文件请走文档导入流程。"
            )

        text = text.strip()
        if not text:
            return "网页抓取成功，但没有提取到正文（可能需要登录或依赖 JavaScript 渲染）。"
        ledger = registry or EvidenceLedger()
        number = ledger.issue_web(
            url=fetched.final_url,
            title=title,
            snippet=text[:80],
        )
        return f"[{number}] 来自网络 · {title}\n链接：{fetched.final_url}\n{text[:6000]}"

    return web_fetch_tool


def _section(
    settings: Settings,
    registry: CitationRegistry,
    doc_id: str,
    heading: str,
    *,
    headings: list[str] | None = None,
    gate: WebSearchGate | None = None,
) -> str:
    """读切分产物里的原文。**只读**，不动任何东西。

    `heading` 与 `headings` 一起认（一次最多 `MAX_SECTIONS` 节）：**分几轮要，每轮都要把
    整段上下文重发一次模型** —— 那是这条链路上最贵的部分（实测一次 8 轮逐节读 = 输入 token
    12.3 万）。
    """
    from rag.ingest.stages import build_artifact_store

    store = build_artifact_store(settings)
    path = store.path_of(doc_id, "chunks.json")
    if path is None or not path.is_file():
        if gate is not None:
            gate.note_corpus(found=False)
        return f"没有 {doc_id} 的切分产物（先灌库，或 doc_id 写错了）。"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunks = payload.get("chunks") or []
    except (OSError, ValueError) as exc:
        return f"读 {doc_id} 的产物失败：{type(exc).__name__}: {exc}"

    wanted = [item.strip() for item in [heading, *(headings or [])] if item and item.strip()]
    # 去重但保序：同一个关键词给两遍不该取两遍
    wanted = list(dict.fromkeys(wanted))
    if not wanted:
        # 没给章节就给标题清单：让模型先看清结构，再决定看哪一节
        titles = [chunk["heading_path"] for chunk in chunks if chunk.get("heading_path")]
        unique = list(dict.fromkeys(titles))
        return f"{doc_id} 的章节（{len(chunks)} 块）：\n" + "\n".join(f"- {t}" for t in unique[:60])

    parts: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()
    truncated = 0
    for needle in wanted[:MAX_SECTIONS]:
        matched = [chunk for chunk in chunks if needle in (chunk.get("heading_path") or "")]
        if not matched:
            missing.append(needle)
            continue
        if len(matched) > 5:
            truncated += 1
        for chunk in matched[:5]:
            # **同一次调用里同一个块只回一遍**：`2.2` 与 `2.2.1` 会同时命中它
            if chunk["chunk_id"] in seen:
                continue
            seen.add(chunk["chunk_id"])
            number = registry.issue(
                chunk_id=chunk["chunk_id"],
                source_uri=payload.get("source_uri", doc_id),
                snippet=(chunk.get("text") or "")[:80],
                heading_path=str(chunk.get("heading_path") or ""),
            )
            parts.append(
                f"[{number}] {chunk['heading_path']}\n{(chunk.get('text') or '')[:SECTION_CHARS]}"
            )
    if not parts:
        if gate is not None:
            gate.note_corpus(found=False)
        return (
            f"{doc_id} 里没有章节路径包含「{'、'.join(missing)}」。"
            f"用 get_section(doc_id) 看章节清单。"
        )
    if gate is not None:
        gate.note_corpus(found=True)
    notes: list[str] = []
    if truncated:
        # 原来的口径保留：一节里块特别多时只给前 5 块，且要说出来
        notes.append(f"（有 {truncated} 节匹配到超过 5 块，那些节只给了前 5 块）")
    if missing:
        # 没找到的也要说：不说的话模型会以为「这几节是空的」
        notes.append(f"（没找到：{'、'.join(missing)}；用 get_section(doc_id) 看章节清单）")
    if len(wanted) > MAX_SECTIONS:
        notes.append(f"（一次最多取 {MAX_SECTIONS} 节，剩下 {len(wanted) - MAX_SECTIONS} 个没取）")
    tail = ("\n\n" + "\n".join(notes)) if notes else ""
    return "\n\n".join(parts) + tail


def build_web_search_tool(
    settings: Settings,
    secrets: Any = None,
    *,
    gate: WebSearchGate | None = None,
    registry: EvidenceLedger | None = None,
) -> BaseTool:
    """智谱 web 搜索工具（使用者自研，M6 接进来）。

    两处**取值的来源**（2026-10-05 定）：key 从 `Secrets.zhipuai_api_key`（机密只走 .env，
    绝不出现在 yaml 或 `Settings` 里 —— `rag config` 会把整棵配置树打出来）；
    base_url 从 `providers.chat.zhipuai.base_url`（非机密，走 `${ZHIPUAI_BASE_URL:-…}` 插值）。
    两者在 `.env` 里都已经有了。

    智谱 SDK（`zai-sdk`）**懒加载**：即使它缺失/损坏，也不影响后端启动。

    `gate` 是那道网络闸门（见 `WebSearchGate`）：只有用户明确允许联网时才执行。
    """
    if secrets is None:
        from rag.config import get_secrets

        secrets = get_secrets()
    # 用 getattr 兜住「providers.chat 里没配 zhipuai」的情形：那就退到 SDK 默认地址
    zhipu = settings.providers.chat.get("zhipuai")
    base_url = zhipu.base_url if zhipu is not None else None
    api_key = getattr(secrets, "zhipuai_api_key", None)

    @tool
    def web_search_tool(query: str) -> str:
        """用网络搜索引擎检索信息，用于回答一般性/知识性问题。

        只有在用户明确允许联网后才能调用。内部证据不足时，先询问用户。

        Args:
            query: 搜索查询语句。
        """
        if gate is not None and not gate.allowed:
            return f"现在不能用网络搜索：{gate.refusal()}"
        if not api_key:
            return "搜索失败：未配置 ZHIPUAI_API_KEY（请在 .env 中配置）。"

        try:
            from zai import ZhipuAiClient
            from zai.core._errors import (
                APIConnectionError,
                APIResponseError,
                APIStatusError,
                APITimeoutError,
                ZaiError,
            )
        except ImportError as exc:
            return f"搜索失败：无法导入智谱 SDK（{exc}）。请安装 `pip install zai-sdk`。"

        client = ZhipuAiClient(
            api_key=api_key,
            base_url=base_url or None,
        )
        try:
            response = client.web_search.web_search(
                search_engine="search_pro",
                search_query=query,
            )
        except APITimeoutError:
            return "搜索失败：请求超时，请稍后重试。"
        except APIConnectionError:
            return "搜索失败：无法连接搜索服务，请检查网络或 GLM_BASE_URL。"
        except APIStatusError as exc:
            return f"搜索失败：API 状态错误（HTTP {exc.status_code}）。"
        except APIResponseError as exc:
            return f"搜索失败：响应错误（{exc.message}）。"
        except ZaiError as exc:
            return f"搜索失败：{exc}。"
        except Exception as exc:  # noqa: BLE001
            return f"搜索失败：{type(exc).__name__}: {exc}。"

        result = response.search_result
        if not result:
            return "搜索无结果。"

        # 实际返回 list[SearchResultResp]。网络结果和语料块登记进同一张证据表，
        # 模型用同一个 [n] 引用，前端与评测也拿到同一种来源形状。
        ledger = registry or EvidenceLedger()
        parts = []
        for item in result:
            title = item.title or "(无标题)"
            content = (item.content or "").strip()
            meta = "、".join(x for x in (item.media, item.publish_date) if x)
            number = ledger.issue_web(url=str(item.link or ""), title=str(title), snippet=content)
            link = f"\n链接：{item.link}" if item.link else ""
            head = f"[{number}] 来自网络 · {title}" + (f"（{meta}）" if meta else "")
            parts.append(f"{head}\n{content}{link}")
        return "\n\n".join(parts)

    return web_search_tool
