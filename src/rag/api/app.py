"""FastAPI 应用：`create_app(settings)`。

服务层刻意做薄 —— 它只负责三件事：**建一次重对象**、把 HTTP 请求翻成下层入参、
把异常翻成状态码。真正的活儿在 retrieval / ingest / graphs 那几层。

两条来自真机教训的硬约束：

1. **重对象只在 lifespan 里建一次**（`graphs/wiring.py::build_heavy`）。M3 踩过
   「每次检索重建 embedder/reranker → 一次问答加载三遍 1GB 模型」；服务里按请求建
   更致命（内存直接翻倍）。
2. **单个模型建不起来不让服务起不来**：记进 `heavy.errors`，用到的端点返 503 并带上原因，
   `/health` 照实报。整个服务因为一个缺 key 起不来是最难受的失败方式。
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from rag.api.ingest_gate import IngestGate
from rag.api.jobs import JobStore
from rag.api.ratelimit import RateLimiter
from rag.api.redis_backend import RedisBackend
from rag.api.routes import router
from rag.config.schema import Settings
from rag.config.settings import Secrets, get_secrets
from rag.contracts import ChannelStats, ScoredChunk
from rag.graphs import run_index_graph
from rag.graphs.wiring import build_heavy


class ServiceUnavailable(RuntimeError):
    """服务起不来 / 依赖挂了 → 503。带上「是哪一件、为什么」。"""


def _warn_storage(message: str) -> None:
    """存储写失败：业务照常（会话记录是附属品），但**按 ERROR 记** ——
    它已经影响到功能了（那段会话没落到库里），不能只当噪声。"""
    import logging

    logging.getLogger("rag.storage").error("%s", message)


# 给新 thread 铺垫时最多带几条历史消息（一轮 = human + ai 两条）。
# 整段塞进去既贵又没必要，还容易把上下文撑爆。
HISTORY_MESSAGES = 8


class Components:
    """服务级单例。一个进程一份。"""

    def __init__(self, settings: Settings, secrets: Secrets) -> None:
        self.settings = settings
        self.secrets = secrets
        self.redis = RedisBackend(settings.api.redis, secrets.rag_redis_password)
        self.jobs = JobStore(self.redis)
        self.limiter = RateLimiter(settings.api.rate_limit_per_min, self.redis)
        self.index: Any = None
        self.heavy: Any = None
        self.persistence: Any = None
        # UI 会话的存储（chat_session / chat_message / token_usage）。
        # local 档用 SQLite，multiworker 用 PostgreSQL。
        self.chats: Any = None
        self.index_error: str = ""
        self.ingest_gate: IngestGate | None = None
        self.background: set[asyncio.Task] = set()

    # ---- 生命周期 ----

    async def startup(self) -> None:
        await self.redis.connect()
        profile = self.settings.deployment.profile
        if profile == "multiworker" and not self.redis.available:
            raise RuntimeError(f"multiworker 要求 Redis 可用：{self.redis.reason}")
        self.ingest_gate = IngestGate(
            self.redis,
            limit=self.settings.api.max_concurrent_ingest,
            lease_s=self.settings.deployment.ingest_lease_s,
            require_shared=profile == "multiworker",
        )
        from rag.persistence import build_persistence

        self.persistence = build_persistence(self.settings, self.secrets)
        await self.persistence.connect()
        if profile == "multiworker" and not self.persistence.available:
            raise RuntimeError(f"multiworker 要求 PostgreSQL 可用：{self.persistence.reason}")
        if profile in {"docker-single", "multiworker"}:
            from rag.storage import build_blob_store

            files = build_blob_store(backend="minio", settings=self.settings)
            try:
                await asyncio.to_thread(files.ping)
            except Exception as exc:  # noqa: BLE001 —— 必选项不可用就拒绝启动
                raise RuntimeError(f"{profile} 要求 MinIO 可用：{exc}") from exc
        from rag.persistence import build_chat_store

        self.chats = build_chat_store(self.settings, self.persistence)
        if getattr(self.chats, "backend", "") == "sqlite":
            try:
                await self.chats.setup()
            except Exception as exc:  # noqa: BLE001 —— 本地会话库是核心功能，不能静默缺失
                raise RuntimeError(f"SQLite 会话库初始化失败：{exc}") from exc
        from rag.profile_state import ensure_profile_state

        try:
            await asyncio.to_thread(ensure_profile_state, self.settings)
        except Exception as exc:  # noqa: BLE001 —— 状态域不一致必须阻止启动
            raise RuntimeError(f"profile 状态域校验失败：{exc}") from exc
        try:
            from rag.indexing import build_index

            self.index = build_index(self.settings)
            # 先探活（有超时）再 ensure_ready（它建表 + load，大表 load 十几秒是正常的，
            # 所以那一步不设超时）。Milvus 挂着时这样能在几秒内把 /health 变成 degraded，
            # 而不是让 uvicorn 卡在 startup 上 —— 探活为什么必要见 §16 那条三十几分钟的空转。
            await self.index.ping(self.settings.indexing.vector.probe_timeout_s)
            await self.index.ensure_ready()
        except Exception as exc:  # noqa: BLE001 —— Milvus 挂了也要能起来看 /health
            if profile == "multiworker":
                raise RuntimeError(f"multiworker 要求 Milvus 可用：{exc}") from exc
            self.index_error = f"{type(exc).__name__}: {exc}"
            # **影响整个流程的第三方异常按 ERROR 记**：Milvus 挂着时检索与问答全不可用，
            # 这不是「某条日志吵」，是功能没了（uvicorn 起来但 /health degraded）
            import logging

            logging.getLogger("rag.api").error(
                "Milvus 不可用，检索与问答都会降级：%s", self.index_error
            )
        self.heavy = build_heavy(self.settings, self.index)
        # 文档注册表（每篇一行：走到哪一步）建表，幂等。**放在 Milvus 那一段之外**：
        # 文档页的状态不该因为向量库挂着就记不下来（2026-10-09 从 Milvus 搬出来就是为了这个）。
        try:
            from rag.persistence.documents import build_document_store

            await asyncio.to_thread(build_document_store(self.settings).setup)
        except Exception as exc:  # noqa: BLE001 —— 建表失败也不该挡住服务启动
            import logging

            logging.getLogger("rag.api").error("文档注册表建表失败：%s", exc)

    async def shutdown(self) -> None:
        for task in list(self.background):
            task.cancel()
        if self.background:
            await asyncio.gather(*self.background, return_exceptions=True)
        if self.persistence is not None:
            await self.persistence.close()
        await self.redis.close()

    def track(self, task: asyncio.Task) -> None:
        """留住后台任务：不持有引用的话可能被 GC 掉，任务就静默消失了。"""
        self.background.add(task)
        task.add_done_callback(self.background.discard)

    # ---- 参数整理 ----

    def _local(self, channels: list[str] | None, rerank: bool | None) -> Settings:
        """把这次请求的覆盖项落到一份**拷贝**上 —— 请求之间不能互相污染。"""
        local = self.settings.model_copy(deep=True)
        if channels:
            # 语义：**列出来的那几路开、其余关**（名字取自 retrieval.channels[].name）
            known = {item.name for item in local.retrieval.channels}
            unknown = [name for name in channels if name not in known]
            if unknown:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                    f"未知的通道 {unknown}；配置里有 {sorted(known)}（见 retrieval.channels）",
                )
            wanted = set(channels)
            for item in local.retrieval.channels:
                item.enabled = item.name in wanted
        if rerank is not None:
            local.retrieval.rerank.enabled = rerank
        return local

    def _require(self, *names: str) -> None:
        if self.index is None:
            raise ServiceUnavailable(f"Milvus 不可用：{self.index_error}")
        broken = [
            f"{name}（{self.heavy.errors[name]}）" for name in names if name in self.heavy.errors
        ]
        if broken:
            raise ServiceUnavailable("组件建不起来：" + "；".join(broken))

    # ---- 检索与问答 ----

    async def search(
        self,
        *,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
        channels: list[str] | None = None,
        rerank: bool | None = None,
        image: Any = None,
    ) -> tuple[list[ScoredChunk], list[ChannelStats]]:
        self._require("embedding")
        from rag.retrieval import build_retriever

        local = self._local(channels, rerank)
        retriever = build_retriever(local, self.index, self.heavy.embedder, self.heavy.reranker)
        hits = await retriever.retrieve(query, top_k, filters=filters or None, image=image)
        return hits, getattr(retriever, "last_stats", [])

    async def answer(self, body) -> dict[str, Any]:
        """一次完整问答；流式与非流式共用 AnswerEngine 的检索预检。"""
        local = self._local(body.channels, body.rerank)
        if body.graph is not None:
            local.graph.enabled = body.graph
        self._require("embedding", "generator")
        from rag.graphs import build_answer_engine

        # token 用量靠调用方传容器收集（`heavy.generator` 是跨请求单例，不能挂实例属性）
        usage: dict[str, Any] = {}
        engine = build_answer_engine(local, self.index, self.heavy)
        result = await engine.answer(
            body.question,
            body.top_k,
            filters=body.filters or None,
            usage=usage,
        )
        out = {
            "answer": result.answer,
            "hits": result.hits,
            "stats": result.stats,
            "citations": result.citations,
            "rounds": result.rounds,
            "grade": result.grade,
            "mode": result.mode,
            "usage": result.usage,
        }
        sid = await self._open_session(body.session_id, title=body.question[:30], mode="chat")
        await self._save_turn(
            sid, question=body.question, answer=out["answer"], mode="chat", usage=usage
        )
        out["session_id"] = sid
        return out

    async def stream_answer(self, body):
        """流式问答：产出 `retrieval` → `token`* → `done` 三种字典。

        `graph=true` 时先跑完整的生成前预检，再开始流式生成；检索纠错不会因为
        `stream=true` 被绕过。
        """
        local = self._local(body.channels, body.rerank)
        if body.graph is not None:
            local.graph.enabled = body.graph
        self._require("embedding", "generator")
        from rag.graphs import build_answer_engine

        engine = build_answer_engine(local, self.index, self.heavy)
        sid = await self._open_session(body.session_id, title=body.question[:30], mode="chat")
        async for event in engine.stream(
            body.question,
            body.top_k,
            filters=body.filters or None,
        ):
            if event["type"] == "done":
                # **先落库再发出去**：SSE 收到 done 就取消生产者，yield 之后再写会被取消。
                await self._save_turn(
                    sid,
                    question=body.question,
                    answer=str(event.get("answer") or ""),
                    mode="chat",
                    usage=event.get("usage") or {},
                )
                event = {**event, "session_id": sid}
            yield event

    # ---- 会话落库 ----

    async def _open_session(self, session_id: str | None, *, title: str, mode: str) -> str | None:
        """这次的会话 id：给了就沿用，没给就开一段（标题取首问）。

        会话存储没建起来时返回 None —— 问答照常，只是不落库。写失败也一样
        （只在 stderr 喊一句）。
        """
        if self.chats is None:
            return None
        from rag.obs import new_id

        sid = session_id or new_id("c")
        try:
            await self.chats.create(session_id=sid, title=title, mode=mode)
        except Exception as exc:  # noqa: BLE001
            _warn_storage(f"会话建不起来（{type(exc).__name__}）：{exc}")
            return None
        return sid

    async def _save_turn(
        self,
        session_id: str | None,
        *,
        question: str,
        answer: str,
        mode: str,
        status: str = "answered",
        thread_id: str | None = None,
        usage: dict[str, Any] | None = None,
        resume: bool = False,
    ) -> None:
        """把一轮落进去。**失败不影响问答**（与日志同一个立场）。"""
        if session_id is None or self.chats is None:
            return
        from rag.usage import usage_totals

        totals = usage_totals(usage)
        try:
            if resume:
                # 恢复那轮的行在中断时就落了，这里是**补上**它，不是再插一行
                filled = await self.chats.update_pending(
                    session_id, answer=answer, status=status, usage=totals
                )
                if filled:
                    return
            await self.chats.append_turn(
                session_id,
                question=question,
                answer=answer,
                mode=mode,
                status=status,
                thread_id=thread_id,
                usage=totals,
            )
        except Exception as exc:  # noqa: BLE001
            _warn_storage(f"会话落库失败（{type(exc).__name__}）：{exc}")

    def _build_agent(self, *, web_explicit: bool = False):
        """建图 + 引用表（工具要闭包住这一张，所以每请求建一次 —— 模型不重建）。"""
        from rag.agent import build_agent, default_workspace

        return build_agent(
            self.settings,
            self.heavy,
            checkpointer=(self.persistence.checkpointer if self.persistence else None),
            store=(self.persistence.store if self.persistence else None),
            workspace=default_workspace(self.settings),
            web_explicit=web_explicit,
        )

    def _agent_persistence_check(self, thread_id: str | None) -> None:
        """多轮 / 中断恢复**都必须有 checkpointer** —— 没连上 PG 就明说，
        别让调用方以为「历史丢了」或者「它怎么不等人批准」。"""
        if thread_id and not (self.persistence and self.persistence.available):
            reason = self.persistence.reason if self.persistence else "未启用"
            raise ServiceUnavailable(f"给了 thread_id 但 PG 不可用（{reason}），多轮会话需要它")

    async def agent(
        self,
        message: str,
        *,
        thread_id: str | None = None,
        web_explicit: bool = False,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """对话式 agent（M5 批次 B：可写但要人批准 + 多轮）。"""
        self._require("embedding", "generator", "chat")
        self._agent_persistence_check(thread_id)
        from rag.agent import run_agent

        graph, registry = self._build_agent(web_explicit=web_explicit)
        # thread 是新的才铺垫历史；已经有 thread 的话历史在 checkpointer 里，再塞一遍就重复了
        history = await self._history_for(session_id) if thread_id is None else None
        result = await run_agent(
            graph, message, registry=registry, thread_id=thread_id, history=history
        )
        out = await self._agent_result(result, registry, graph)
        out["session_id"] = await self._record_agent_turn(
            session_id, question=message, out=out, usage=result.usage
        )
        return out

    async def resume(
        self,
        thread_id: str,
        *,
        decisions: list[dict[str, Any]],
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """人批准之后接着跑。**引用表要取回来** —— 否则恢复后的答案一条引用都显示不出来
        （`[n]` 是中断前发的号，见 `save_registry` 的说明）。"""
        self._require("embedding", "generator", "chat")
        self._agent_persistence_check(thread_id)
        from rag.agent import load_registry, resume_agent

        store = self.persistence.store if self.persistence else None
        graph, _ = self._build_agent()
        registry = await load_registry(store, thread_id)
        result = await resume_agent(
            graph, thread_id=thread_id, decisions=decisions, registry=registry
        )
        out = await self._agent_result(result, registry, graph)
        sid = await self._record_agent_turn(
            session_id, question="", out=out, usage=result.usage, resume=True
        )
        out["session_id"] = sid
        return out

    async def _record_agent_turn(
        self,
        session_id: str | None,
        *,
        question: str,
        out: dict[str, Any],
        usage: dict[str, Any] | None,
        resume: bool = False,
    ) -> str | None:
        sid = await self._open_session(session_id, title=question[:30], mode="agent")
        await self._save_turn(
            sid,
            question=question,
            answer=out.get("answer") or "",
            mode="agent",
            status=str(out.get("status") or "answered"),
            thread_id=out.get("thread_id"),
            usage=usage,
            resume=resume,
        )
        return sid

    async def _history_for(self, session_id: str | None) -> list[dict[str, Any]] | None:
        """给新 thread 的铺垫：这个会话已有的问答。

        **只在 thread 还是空的时候用**：一段会话可能先用快问答聊了几轮（那条路没有
        checkpointer），切到 agent 时 thread 是新的 —— 不把历史灌进去，agent 就只看得到
        当前这一句（使用者提的第一个问题就是它）。

        只取最近 `HISTORY_MESSAGES` 条、且跳过正文为空的（中断那轮的回答是空的，
        塞进 prompt 只会添乱）。
        """
        if not session_id or self.chats is None:
            return None
        try:
            found = await self.chats.get(session_id)
        except Exception as exc:  # noqa: BLE001 —— 拿不到历史就只问当前这一句，别把问答挡了
            _warn_storage(f"取会话历史失败（{type(exc).__name__}）：{exc}")
            return None
        messages = [
            item
            for item in ((found or {}).get("messages") or [])
            if (item.get("content") or "").strip()
        ][-HISTORY_MESSAGES:]
        if not messages:
            return None
        return [
            {
                "role": "assistant" if item.get("role") == "ai" else "user",
                "content": item["content"],
            }
            for item in messages
        ]

    async def pending_interrupt(self, thread_id: str | None) -> dict[str, Any] | None:
        """这会话还等着批准什么 —— **现去 checkpointer 取**，我们自己的表里不存它。

        取不到（已经批过了、PG 挂了、图都建不起来）就返回 None：调用方按「没有待批」处理。
        """
        if not thread_id or not (self.persistence and self.persistence.available):
            return None
        from rag.agent import pending_interrupt

        try:
            graph, _ = self._build_agent()
            return await pending_interrupt(graph, thread_id)
        except Exception as exc:  # noqa: BLE001
            _warn_storage(f"取中断载荷失败（{type(exc).__name__}）：{exc}")
            return None

    async def _agent_result(self, result, registry, graph) -> dict[str, Any]:
        """收敛成返回体。**中断时把引用表存一份**，好让恢复那次还能用同一套编号。"""
        from rag.agent import save_registry

        if result.status == "interrupted" and result.thread_id:
            await save_registry(
                self.persistence.store if self.persistence else None,
                result.thread_id,
                registry,
            )
        return {
            "answer": result.answer,
            "citations": result.citations,
            "thread_id": result.thread_id,
            "steps": result.steps,
            "retrieved": result.retrieved,
            "status": result.status,
            "interrupt": result.interrupt,
        }

    async def stream_agent(
        self,
        message: str,
        *,
        thread_id: str | None = None,
        web_explicit: bool = False,
        session_id: str | None = None,
    ):
        """流式 agent：产出 `token` / `tool` / `interrupt` / `done` 四种字典。"""
        self._require("embedding", "generator", "chat")
        self._agent_persistence_check(thread_id)
        from rag.agent import save_registry, stream_agent

        sid = await self._open_session(session_id, title=message[:30], mode="agent")
        graph, registry = self._build_agent(web_explicit=web_explicit)
        history = await self._history_for(session_id) if thread_id is None else None
        answer = ""
        interrupted = False
        usage: dict[str, Any] = {}
        final_thread: str | None = thread_id
        async for event in stream_agent(
            graph, message, registry=registry, thread_id=thread_id, history=history
        ):
            if event["type"] == "interrupt" and event.get("thread_id"):
                await save_registry(
                    self.persistence.store if self.persistence else None,
                    event["thread_id"],
                    registry,
                )
                interrupted = True
            if event.get("thread_id"):
                final_thread = event["thread_id"]
            if event["type"] == "done":
                answer = event.get("answer") or ""
                usage = event.get("usage") or {}
            if event["type"] in ("done", "interrupt"):
                # **先落库再发出去**：`_sse_events` 收到 done 就 break 并取消生产者任务，
                # 落库写在 yield 之后会被取消掉（真机上就是这么丢了一轮）。
                await self._save_turn(
                    sid,
                    question=message,
                    answer=answer,
                    mode="agent",
                    status="interrupted" if interrupted else "answered",
                    thread_id=final_thread,
                    usage=usage,
                )
            # session_id 由这一层发出去（生成器不知道它）—— 客户端要拿它续下一轮
            yield {**event, "session_id": sid}

    async def stream_resume(
        self,
        thread_id: str,
        *,
        decisions: list[dict[str, Any]],
        session_id: str | None = None,
    ):
        """批准之后接着流。**事件与 `stream_agent` 一样**，这里只负责把引用表接回来。

        `load_registry` 那一步不能省：`[n]` 是中断前发的号，不接回来恢复后的答案
        就一条引用都显示不出来（见 `save_registry` 的说明）。
        """
        self._require("embedding", "generator", "chat")
        self._agent_persistence_check(thread_id)
        from rag.agent import load_registry, save_registry, stream_resume

        sid = await self._open_session(session_id, title="", mode="agent")
        store = self.persistence.store if self.persistence else None
        graph, _ = self._build_agent()
        registry = await load_registry(store, thread_id)
        answer = ""
        interrupted = False
        usage: dict[str, Any] = {}
        async for event in stream_resume(
            graph, thread_id=thread_id, decisions=decisions, registry=registry
        ):
            if event["type"] == "interrupt" and event.get("thread_id"):
                await save_registry(store, event["thread_id"], registry)
                interrupted = True
            if event["type"] == "done":
                answer = event.get("answer") or ""
                usage = event.get("usage") or {}
            if event["type"] in ("done", "interrupt"):
                # 同上：先落库再发出去
                await self._save_turn(
                    sid,
                    question="",
                    answer=answer,
                    mode="agent",
                    status="interrupted" if interrupted else "answered",
                    thread_id=thread_id,
                    usage=usage,
                    resume=True,
                )
            yield {**event, "session_id": sid}

    # ---- 灌库 ----

    async def run_parse_job(self, job, doc_id: str) -> None:
        """解析一份上传件（页面第二步）。状态流转与 `run_ingest_job` 一致。

        **`_parse_one` 是同步的**（`run_parse` 与 OCR 引擎都是同步实现），整段要丢线程池：
        不然解析这几分钟里事件循环被占着，API 不响应、页面连 `/jobs/{id}` 都轮询不到。
        """
        from rag.obs import set_ids

        # 后台任务的 `run_id` 用 **job id**（页面那排步骤灯同源）：一次「把这份文档入库」
        # 分几次请求跑（parse / 核对 / index），各进一次请求就新发一个 run 必然串不起来
        set_ids(run_id=job.id)
        job.status = "running"
        await self.jobs.save(job)
        try:
            assert self.ingest_gate is not None
            async with self.ingest_gate.hold():
                job.result = await asyncio.to_thread(self._parse_one, doc_id)
            # 内容哈希这一格在**上传**时就写过了；这里再带一次是为了老行（升级前建的）
            # 能补上 —— `plan_incremental` 判「这份变了没有」用的就是它
            await self.mark_stage(
                doc_id, "parsed", content_hash=str((job.result or {}).get("sha1") or "")
            )
            job.status = "done"
        except Exception as exc:  # noqa: BLE001 —— 后台任务，失败写进任务记录
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
        await self.jobs.save(job)

    async def mark_stage(
        self, doc_id: str, stage: str, *, source_uri: str = "", content_hash: str = ""
    ) -> None:
        """把一份文档推进到 `stage`（写文档注册表）。**写不进去不打断主流程**。

        注册表是台账：它挂了不该让解析/入库整批失败，所以这里只记一条 WARNING。
        与 `routes._mark_stage` 同一个口径（那边是 HTTP 侧的两个入口）。
        """
        import logging

        from rag.persistence.documents import build_document_store

        try:
            await build_document_store(self.settings).set_stage(
                doc_id, stage, source_uri=source_uri, content_hash=content_hash
            )
        except Exception as exc:  # noqa: BLE001 —— 台账写失败不该让后台任务失败
            logging.getLogger("rag.ingest").warning(
                "文档状态没写进注册表：%s → %s（%s: %s）", doc_id, stage, type(exc).__name__, exc
            )

    def _parse_one(self, doc_id: str) -> dict[str, Any]:
        """**同步**地解析一份上传件（由 `run_parse_job` 丢进线程池）。"""
        from rag.api.uploads import build_upload_store
        from rag.ingest import build_ocr_engine
        from rag.ingest.incremental import file_hash
        from rag.ingest.stages import build_artifact_store, run_parse

        local = self.settings.model_copy(deep=True)
        uploads = build_upload_store(local)
        path = uploads.path_of(doc_id)
        if path is None:
            raise FileNotFoundError(f"没有这份上传件：{doc_id}")
        store = build_artifact_store(local)
        ocr = build_ocr_engine(local.ingest, local.ingest.pdf.parser)
        report = run_parse([path], uploads.root, local, ocr, store)
        return {
            "name": path.name,
            # 内容哈希跟着一起回：`run_parse_job` 要用它补注册表那一列
            "sha1": file_hash(path),
            "parsed": report.parsed,
            "reused": report.reused,
            "skipped": [f"{item.path}: {item.reason}" for item in report.skipped],
        }

    async def run_index_job(self, job, doc_id: str) -> None:
        """切分 + 入库一份上传件（页面第五步）。状态流转与 `run_parse_job` 一致。"""
        from rag.obs import set_ids

        set_ids(run_id=job.id)  # 与 `run_parse_job` 同一个口径：run_id = job id
        job.status = "running"
        await self.jobs.save(job)
        try:
            assert self.ingest_gate is not None
            async with self.ingest_gate.hold():
                job.result = await self.run_index(doc_id)
            job.status = "done"
        except Exception as exc:  # noqa: BLE001 —— 后台任务，失败写进任务记录
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
        await self.jobs.save(job)

    async def run_index(self, doc_id: str) -> dict[str, Any]:
        """一份上传件：切分 → 灌 Milvus。

        调用方（路由）已经挡过「清单没核对」这一道，这里不再判断 —— 但 `run_chunk` 自己
        也会把没核对的挡进 `pending`，那道硬门槛不靠单点。
        """
        from rag.api.uploads import build_upload_store
        from rag.indexing.commit import commit_chunks
        from rag.ingest import build_ocr_engine
        from rag.ingest.incremental import file_hash
        from rag.ingest.stages import build_artifact_store, run_chunk

        local = self.settings.model_copy(deep=True)
        self._require("embedding")
        uploads = build_upload_store(local)
        path = uploads.path_of(doc_id)
        if path is None:
            raise FileNotFoundError(f"没有这份上传件：{doc_id}")
        store = build_artifact_store(local)
        ocr = build_ocr_engine(local.ingest, local.ingest.pdf.parser)
        # 切分是同步的（OCR 引擎、图片描述），下线程池 —— 与 run_ingest 同一条理由
        chunked = await asyncio.to_thread(
            run_chunk, [path], uploads.root, local, self.heavy.embedder, _vision(local), ocr, store
        )
        # 先记「切完了」，再让 `commit_chunks` 把它推到 `indexed` —— 顺序反了会把
        # 已经写好的 `indexed` 拽回 `chunked`（台账只该往前）
        await self.mark_stage(doc_id, "chunked", source_uri=path.name, content_hash=file_hash(path))
        committed = await commit_chunks(
            local, store, [(path, uploads.root)], embedder=self.heavy.embedder, index=self.index
        )
        return {
            "name": path.name,
            # **块数**，不是「几份文档」：`chunked.chunked` 数的是文档（run_chunk 里那份
            # 计数器），早先这里把它当块数报了 —— 393 块的文档在任务页上显示「1」。
            "chunks": len(chunked.chunks),
            "documents": chunked.chunked + chunked.reused,
            "written": committed.written,
            "pending": [item.reason for item in chunked.pending],
            "missing": committed.missing,
            "skipped": [f"{item.path}: {item.reason}" for item in chunked.skipped],
        }

    async def run_ingest_job(self, job, body) -> None:
        """后台任务的包装：状态流转 + 并发闸门 + 失败落账。

        失败**不抛给调用方**（它是后台任务），而是写进任务记录 —— `/jobs/{id}` 看得到
        「哪一步、什么错」，这比翻日志快。
        """
        from rag.obs import set_ids

        set_ids(run_id=job.id)  # 与 `run_parse_job` 同一个口径：run_id = job id
        job.status = "running"
        await self.jobs.save(job)
        try:
            assert self.ingest_gate is not None  # startup 里建的；None 只可能是没跑 lifespan
            async with self.ingest_gate.hold():
                job.result = await self.run_ingest(body)
            job.status = "done"
        except Exception as exc:  # noqa: BLE001
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
        await self.jobs.save(job)

    async def run_ingest(self, body) -> dict[str, Any]:
        """跑一次灌库。**只跑 `--stage all` 那条路**（后台任务里再分段手工调度没有意义）。"""
        from pathlib import Path as _Path

        from rag.indexing.commit import commit_chunks, drop_docs, index_fingerprint
        from rag.ingest import build_ocr_engine
        from rag.ingest.incremental import pipeline_version, plan_incremental
        from rag.ingest.loaders import discover
        from rag.ingest.stages import build_artifact_store

        local = self.settings.model_copy(deep=True)
        if body.describe is not None:
            local.ingest.images.describe = body.describe
        if body.graph is not None:
            local.graph.enabled = body.graph
        self._require("embedding")

        store = build_artifact_store(local)
        ocr = build_ocr_engine(local.ingest, local.ingest.pdf.parser)
        vision = _vision(local)
        # 路径 → 语料根：与 CLI 同一条规则（目录当根、单文件用它的父目录）
        sources = [(_Path(raw).resolve(), _Path(raw).resolve()) for raw in body.paths]
        summary: dict[str, Any] = {"written": 0, "registered": 0, "pending": [], "skipped": []}
        for source, root in sources:
            root = root if root.is_dir() else root.parent
            files = discover(source)
            registry = _registry(local)
            records = {} if body.full else await registry.all()
            plan = plan_incremental(
                files,
                root=root,
                registry=records,
                pipeline=lambda path, root=root: index_fingerprint(
                    pipeline_version(local), store, path, root
                ),
                detect_deleted=source.is_dir() and not body.full,
            )
            if plan.deleted:
                await drop_docs(local, store, plan.deleted)
            if not plan.todo:
                continue
            if local.graph.enabled:
                result = await run_index_graph(
                    local,
                    todo=plan.todo,
                    root=root,
                    ocr=ocr,
                    vision=vision,
                    store=store,
                    embedder=self.heavy.embedder,
                    index=self.index,
                    registry=registry,
                )
                summary["written"] += result.written
                summary["registered"] += result.registered
                summary["pending"].extend(item.path for item in result.pending)
                summary["skipped"].extend(f"{item.path}: {item.reason}" for item in result.skipped)
                continue
            from rag.ingest.stages import run_chunk, run_parse

            # **两步都下线程池**：它们内部是同步的（OCR 引擎、切分、图片描述），
            # 在事件循环里直接跑会把整个 API 卡住几分钟 —— 页面连 `/jobs/{id}` 都轮询不到。
            parsed = await asyncio.to_thread(run_parse, plan.todo, root, local, ocr, store)
            await self._mark_paths(plan.todo, root, "parsed", skip=parsed.skipped)
            chunked = await asyncio.to_thread(
                run_chunk, plan.todo, root, local, self.heavy.embedder, vision, ocr, store
            )
            await self._mark_paths(
                plan.todo, root, "chunked", skip=[*chunked.pending, *chunked.skipped]
            )
            summary["pending"].extend(item.path for item in chunked.pending)
            summary["skipped"].extend(f"{item.path}: {item.reason}" for item in chunked.skipped)
            committed = await commit_chunks(
                local, store, [(path, root) for path in plan.todo], embedder=self.heavy.embedder
            )
            summary["written"] += committed.written
            summary["registered"] += len(committed.registered)
            summary["missing"] = committed.missing
        return summary

    async def _mark_paths(self, paths, root, stage: str, *, skip) -> None:
        """批量推进一批文件的状态（按路径算 doc_id）。`skip` 里的那些**跳过** ——
        台账不能替没做成的文件报"做完了"。"""
        from rag.ingest.incremental import file_hash
        from rag.ingest.loaders import doc_id_for

        skipped = {item.path for item in skip}
        for path in paths:
            if str(path) in skipped:
                continue
            await self.mark_stage(
                doc_id_for(path, root), stage, source_uri=path.name, content_hash=file_hash(path)
            )


def _vision(settings: Settings):
    if not settings.ingest.images.describe:
        return None
    from rag.providers import build_vision

    return build_vision(settings)


def _registry(settings: Settings):
    from rag.indexing import build_registry

    return build_registry(settings)


def create_app(settings: Settings, secrets: Secrets | None = None) -> FastAPI:
    """建应用。`settings` 由调用方给（CLI 从 yaml 读、测试直接构造）。"""
    from rag.deployment import apply_deployment_profile, validate_deployment

    apply_deployment_profile(settings)
    validate_deployment(settings, secrets=secrets)
    if settings.persistence.postgres.url:
        # Windows 上 psycopg 的 async 需要 Selector 循环，而它**必须在建循环之前**设。
        # `create_app` 是 CLI 与 TestClient（启动 portal 之前）都会先走到的地方，放这里能兜住
        # 这两条路。**但它兜不住 uvicorn**：uvicorn ≥0.36 自己按 loop 参数建循环、不看策略，
        # 所以 `rag serve` 另外传了 `loop="rag.persistence:selector_event_loop"`
        # （实测 2026-10-05）。
        from rag.persistence import ensure_compatible_event_loop

        ensure_compatible_event_loop()
    secrets = secrets or get_secrets()
    # **在建任何 HTTP 客户端之前**把回环 / 内网地址并进 NO_PROXY：开着系统代理的机器上，
    # 打本机 OCR 产线的请求会被送到代理并回 502（实测，见 rag/net.py）
    from rag.net import configured_service_urls, ensure_local_bypasses_proxy

    ensure_local_bypasses_proxy(configured_service_urls(settings))
    # 观测：按配置打开 LangSmith trace（关闭时是空操作，见 observability.py）
    from rag.observability import configure_tracing

    configure_tracing(settings, secrets)
    components = Components(settings, secrets)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        await components.startup()
        try:
            yield
        finally:
            await components.shutdown()

    app = FastAPI(title="multimodal_rag", lifespan=lifespan)
    app.state.components = components
    app.include_router(router)
    _install_request_logging(app)
    _install_error_handlers(app)
    _mount_web(app)
    return app


def create_app_from_env() -> FastAPI:
    """uvicorn 多 worker 的 import-string 工厂：每个子进程从环境重读配置。"""
    from rag.config import load_settings
    from rag.obs import configure_logging

    settings = load_settings()
    configure_logging(settings)
    return create_app(settings)


def _mount_web(app: FastAPI) -> None:
    """有前端产物就把 `web/dist` 挂到 `/`（`pnpm build` 之后 `rag serve` 一个进程就够）。

    挂载放在最后：API 那几条路由先匹配，剩下的才轮到静态页。**没有 dist 就不挂** ——
    开发期走 Vite dev server（5173 → 代理到 8000），后端不该因为缺前端产物起不来。
    """
    from pathlib import Path

    from fastapi.staticfiles import StaticFiles

    repo_root = Path(__file__).resolve().parents[3]
    dist = repo_root / "web" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="web")


def _install_request_logging(app: FastAPI) -> None:
    """**接口日志**（三类日志里的第一类）：一次请求一行，并在这里设关联 id。

    关联 id 是这套日志的骨架：`request_id` 由这里生成、`run_id` 也在这里设，
    于是链路深处的通道日志（`MultiChannelRetriever`）与 agent 运行日志
    （`run_agent`）都能读到同一个 id —— 不用把 id 一层层往下传。

    记的东西刻意克制：方法 / 路径 / 状态 / 耗时 / 身份（**sha1 前 12 位，不落 key 原文**）
    / 来源 IP / 响应大小 / 错误。请求体与响应体属于数据，只在 `logging.content` 打开时
    才由各自的记录点落盘。
    """
    import time

    from rag.api.ratelimit import identity_of
    from rag.obs import new_id, record_interface, set_ids, set_source

    @app.middleware("http")
    async def _log_request(request: Request, call_next):  # noqa: ANN001, ANN202
        request_id = new_id("req")
        set_ids(request_id=request_id, run_id=new_id("run"))
        # 「这一行是谁发起的」——落库那一列用 `source`，**不从 id 前缀反推**
        # （HTTP 与 CLI 发的都是 `req-`，形状一致，查询不用分两套）
        set_source("http")
        host = request.client.host if request.client else None
        identity = identity_of(request.headers.get("x-api-key"), host)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:  # noqa: BLE001 —— 记一笔再原样抛给错误处理器
            record_interface(
                method=request.method,
                path=request.url.path,
                status=500,
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
                identity=identity,
                client_host=host,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        record_interface(
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
            identity=identity,
            client_host=host,
            response_bytes=_body_size(response),
        )
        return response


def _body_size(response: Any) -> int | None:
    """响应大小。流式响应（SSE）没有 `body`，返回 None —— 别为此把流读掉。"""
    body = getattr(response, "body", None)
    return len(body) if isinstance(body, (bytes, bytearray)) else None


def _install_error_handlers(app: FastAPI) -> None:
    """所有错误统一成 `{"error": {"code", "message"}}`。

    统一形状是为了让客户端只写一处解析；状态码则保留语义（401/404/422/429/503）。
    """

    @app.exception_handler(HTTPException)
    async def _http(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            {"error": {"code": exc.status_code, "message": exc.detail}},
            status_code=exc.status_code,
            headers=exc.headers,
        )

    @app.exception_handler(ServiceUnavailable)
    async def _unavailable(request: Request, exc: ServiceUnavailable) -> JSONResponse:
        return JSONResponse({"error": {"code": 503, "message": str(exc)}}, status_code=503)

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        # 兜底：连没预料到的异常也要是 JSON，不然客户端拿到一坨 HTML 栈
        return JSONResponse(
            {"error": {"code": 500, "message": f"{type(exc).__name__}: {exc}"}},
            status_code=500,
        )
