"""服务层：鉴权、限流、五个端点、SSE 事件顺序。

用 `TestClient` + **替身 Components**：不起 Milvus、不加载模型、不调线上 —— HTTP 这一层
要能单独验，否则每次改服务都要等模型加载十几秒。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from rag.api.app import Components, create_app
from rag.api.jobs import JobStore
from rag.api.ratelimit import RateLimiter, identity_of
from rag.api.redis_backend import RedisBackend
from rag.config.schema import ApiConfig, RedisConfig, Settings
from rag.config.settings import Secrets
from rag.contracts import ChannelStats, Chunk, Citation, ScoredChunk
from rag.deployment import apply_deployment_profile


def _hit(chunk_id: str = "d1:0001") -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id="d1",
            text="正文内容",
            modality="image",
            metadata={"source_uri": "a.pdf"},
        ),
        score=0.9,
        channel="rrf",
        rank=0,
        sources=["hybrid:milvus:c"],
    )


class _Redis(RedisBackend):
    """不真连的 Redis：`available`/`reason` 可指定，其余走父类的降级路径。"""

    def __init__(self, *, ok: bool) -> None:
        super().__init__(RedisConfig(url="redis://127.0.0.1:6379/0", key_prefix="t:"))
        self._ok = ok
        self.reason = "" if ok else "连不上：ConnectionError"

    @property
    def available(self) -> bool:
        return self._ok

    async def ping(self) -> bool:
        return self._ok


class _Index:
    def __init__(self) -> None:
        self.boom = False

    async def count(self) -> int:
        if self.boom:
            raise RuntimeError("Milvus 连不上")
        return 1691

    async def get_chunks(self, chunk_ids: list[str]):
        from rag.contracts import Chunk

        self.asked_chunks = list(chunk_ids)
        if chunk_ids and chunk_ids[0] == "missing":
            return []
        return [
            Chunk(
                chunk_id=chunk_ids[0],
                doc_id="d1",
                text="这一块的完整正文",
                modality="image",
                heading_path="第一章 -> 1.1",
                page=3,
                metadata={"source_uri": "a.pdf"},
                assets=["3481c48bcb813900/p0001_01.jpg"],
            )
        ]


class _Persistence:
    """替身 PG 持久化：默认「已连上」，可切成不可用。"""

    def __init__(self, *, ok: bool = True) -> None:
        self._ok = ok
        self.reason = "" if ok else "OperationalError: 连不上"
        self.checkpointer = object() if ok else None
        self.store = object() if ok else None

    @property
    def available(self) -> bool:
        return self._ok

    async def ping(self) -> bool:
        return self._ok


class _Components:
    """替身 Components：只实现路由真正用到的那几个成员。"""

    def __init__(self, settings: Settings, *, index_ok: bool = True, redis_ok: bool = True) -> None:
        self.settings = settings
        self.secrets = Secrets(rag_api_key=None)
        self.redis = _Redis(ok=redis_ok)
        self.jobs = JobStore(self.redis)
        self.limiter = RateLimiter(settings.api.rate_limit_per_min, self.redis)
        self.index = _Index() if index_ok else None
        self.index_error = "" if index_ok else "MilvusException: 拒绝连接"
        self.persistence = _Persistence()
        self.heavy = _Heavy()
        self.ingest_gate = None
        self.background: set = set()
        self.searches: list[dict] = []
        self.ingested: list[Any] = []
        self.agent_calls: list[dict] = []
        self.resumes: list[dict] = []
        self.chats = _Chats()

    async def search(self, **kwargs):
        self.searches.append(kwargs)
        return [_hit()], [ChannelStats(channel="hybrid:milvus:c", candidates=10, kept=1)]

    async def answer(self, body):
        return {
            "answer": "答案 [1]",
            "hits": [_hit()],
            "stats": [],
            "citations": [
                Citation(index=1, chunk_id="d1:0001", source_uri="a.pdf", snippet="正文")
            ],
            "rounds": 1,
            "grade": {"score": 0.9},
        }

    async def stream_answer(self, body):
        yield {"type": "retrieval", "hits": [_hit()], "stats": []}
        for piece in ("你", "好"):
            yield {"type": "token", "text": piece}
        yield {
            "type": "done",
            "citations": [
                Citation(index=1, chunk_id="d1:0001", source_uri="a.pdf", snippet="正文")
            ],
        }

    async def run_ingest(self, body):
        self.ingested.append(body)
        return {"written": 3, "registered": 1, "pending": [], "skipped": []}

    def _require(self, *names: str) -> None:
        """替身总是「组件齐备」——真实实现会按 index/heavy.errors 判并抛 503。"""
        return None

    async def run_ingest_job(self, job, body) -> None:
        """与真实实现同样的三步：running → 干活 → done（失败落账）。"""
        job.status = "running"
        await self.jobs.save(job)
        job.result = await self.run_ingest(body)
        job.status = "done"
        await self.jobs.save(job)

    async def agent(
        self,
        message: str,
        *,
        thread_id: str | None = None,
        web_explicit: bool = False,
        session_id: str | None = None,
    ) -> dict:
        self.agent_calls.append({"message": message, "thread_id": thread_id, "web": web_explicit})
        return {
            "answer": "答案 [1]",
            "citations": [
                Citation(index=1, chunk_id="d1:0001", source_uri="a.pdf", snippet="正文")
            ],
            "thread_id": thread_id,
            "steps": 4,
            "retrieved": 3,
            "status": "answered",
            "interrupt": None,
        }

    async def resume(
        self, thread_id: str, *, decisions: list[dict], session_id: str | None = None
    ) -> dict:
        self.resumes.append({"thread_id": thread_id, "decisions": decisions})
        return {
            "answer": "写好了 [1]",
            "citations": [
                Citation(index=1, chunk_id="d1:0001", source_uri="a.pdf", snippet="正文")
            ],
            "thread_id": thread_id,
            "steps": 6,
            "retrieved": 3,
            "status": "answered",
            "interrupt": None,
        }

    async def stream_agent(
        self,
        message: str,
        *,
        thread_id: str | None = None,
        web_explicit: bool = False,
        session_id: str | None = None,
    ):
        yield {"type": "token", "text": "正在"}
        yield {"type": "tool", "name": "search_knowledge", "status": "success"}
        yield {
            "type": "done",
            "answer": "答案 [1]",
            "citations": [
                Citation(index=1, chunk_id="d1:0001", source_uri="a.pdf", snippet="正文")
            ],
            "thread_id": thread_id,
            "usage": {"calls": 2, "input_tokens": 30, "output_tokens": 9, "total_tokens": 39},
        }

    async def stream_resume(
        self, thread_id: str, *, decisions: list[dict], session_id: str | None = None
    ):
        """与 `stream_agent` 同一组事件 —— 这是契约，不是巧合。"""
        self.resumes.append({"thread_id": thread_id, "decisions": decisions})
        yield {"type": "token", "text": "写"}
        yield {
            "type": "tool",
            "name": "write_file",
            "status": "success",
            "args": {"file_path": "/workspace/r.md"},
            "ms": 12,
        }
        yield {
            "type": "done",
            "answer": "写好了 [1]",
            "citations": [
                Citation(index=1, chunk_id="d1:0001", source_uri="a.pdf", snippet="正文")
            ],
            "thread_id": thread_id,
        }

    async def stream_agent_to_the_gate(
        self,
        message: str = "",
        *,
        thread_id: str | None = None,
        web_explicit: bool = False,
        session_id: str | None = None,
    ):
        """走到闸门前停下 —— 用来看中断事件里到底带了什么（签名与 stream_agent 一致）。"""
        yield {"type": "token", "text": "我先写个文件"}
        yield {
            "type": "interrupt",
            "interrupt": {
                "action_requests": [
                    {"name": "write_file", "args": {"file_path": "/workspace/r.md"}}
                ],
                "review_configs": [{"action_name": "write_file", "allowed_decisions": ["approve"]}],
            },
            "thread_id": thread_id or "t1",
        }

    def track(self, task) -> None:
        self.background.add(task)

    async def pending_interrupt(self, thread_id: str | None) -> dict | None:
        """替身：假装 checkpointer 里还挂着一条待批（真实现会去 `aget_state` 取）。"""
        if not thread_id:
            return None
        return {
            "action_requests": [{"name": "write_file", "args": {"file_path": "/workspace/r.md"}}],
            "review_configs": [{"action_name": "write_file", "allowed_decisions": ["approve"]}],
        }


class _Chats:
    """会话存储的替身。只验 HTTP 这一层，不真连 PG。"""

    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.created: list[dict] = []
        self.turns: list[dict] = []
        self.pending: list[dict] = []
        self.rows = [
            {
                "session_id": "s1",
                "title": "Save Points 是做什么的",
                "mode": "agent",
                "thread_id": "t1",
                "turns": 2,
                "input_tokens": 1200,
                "output_tokens": 300,
                "total_tokens": 1500,
                "calls": 6,
                "created_at": None,
                "updated_at": None,
            }
        ]

    async def list(self, *, limit: int = 20) -> list[dict]:
        return self.rows

    async def create(self, *, session_id: str, title: str = "", mode: str = "chat", thread_id=None):
        self.created.append(
            {"session_id": session_id, "title": title, "mode": mode, "thread_id": thread_id}
        )
        return {"session_id": session_id, "title": title, "mode": mode, "thread_id": thread_id}

    async def append_turn(self, session_id: str, **kwargs: Any) -> int:
        self.turns.append({"session_id": session_id, **kwargs})
        return len(self.turns) * 2

    async def update_pending(self, session_id: str, **kwargs: Any) -> bool:
        self.pending.append({"session_id": session_id, **kwargs})
        return True

    async def get(self, session_id: str) -> dict | None:
        if session_id != "s1":
            return None
        return {
            "session_id": "s1",
            "title": "Save Points 是做什么的",
            "mode": "agent",
            "thread_id": "t1",
            "created_at": None,
            "updated_at": None,
            "messages": [
                {
                    "seq": 1,
                    "role": "human",
                    "content": "Save Points 是做什么的",
                    "status": "answered",
                },
                {
                    "seq": 2,
                    "role": "ai",
                    "content": "先说结论 [1]",
                    "status": "answered",
                    "usage": {
                        "calls": 1,
                        "input_tokens": 10,
                        "output_tokens": 4,
                        "total_tokens": 14,
                    },
                },
                {"seq": 3, "role": "human", "content": "写个文件", "status": "answered"},
                {"seq": 4, "role": "ai", "content": "", "status": "interrupted"},
            ],
            "usage": {
                "input_tokens": 1200,
                "output_tokens": 300,
                "total_tokens": 1500,
                "calls": 5,
            },
        }

    async def delete(self, session_id: str) -> bool:
        self.deleted.append(session_id)
        return session_id == "s1"


class _Heavy:
    """替身重对象。`embedder`/`chat` 给个普通对象（表示「建好了」），
    `reranker=None` 表示这次没启用重排。"""

    def __init__(self) -> None:
        self.errors: dict[str, str] = {}
        self.embedder = object()
        self.reranker = None
        self.chat = object()
        self.generator = object()


def _settings(**over) -> Settings:
    settings = Settings(api=ApiConfig(**over))
    settings.deployment.profile = "local-lite"
    return settings


def _real_components(settings: Settings | None = None) -> Components:
    """**真的 `Components`**（只是把存储换成替身）。

    落库那几步的编排在 `Components` 自己身上（`_open_session` / `_save_turn`），
    用替身类验不到 —— 而那段编排正是「会话存 PG」的全部逻辑。
    构造它不连任何东西（连接都在 startup 里），所以离线跑得动。
    """
    item = Components(settings or _settings(), Secrets(rag_api_key=None))
    item.chats = _Chats()
    return item


def _client(components: _Components, *, secrets: Secrets | None = None) -> TestClient:
    """把替身塞进 app.state —— 路由通过 `get_components` 取的就是它。"""
    app = create_app(components.settings, secrets or Secrets(rag_api_key=None))
    app.state.components = components
    # 走 TestClient 的上下文管理器会执行 lifespan（会真去连 Redis/建模型），
    # 所以这里**不进** with 块：我们只验 HTTP 层，生命周期另有用例覆盖。
    return TestClient(app)


LOCAL = {"client": ("127.0.0.1", 12345)}


# ---- 鉴权 ----


def test_the_api_is_open_now_that_the_key_is_gone() -> None:
    """2026-10-06 定的：**不再用 `RAG_API_KEY`**（「注册后即可使用」，用户系统是后一批）。

    两条都要钉住，不然「拿掉了」只做了一半：

    * **本机以外的来源**不再被 401（以前没配 key 时只放行本机）；
    * **配了 key 也不校验** —— `.env` 里那行现在没用了。

    限流还在（挂在 router 上），身份退回按 IP 算。
    """
    components = _Components(_settings())
    remote = TestClient(
        _client(components).app,
        client=("10.0.0.9", 1234),  # type: ignore[arg-type]
    )
    assert remote.get("/health").status_code == 200, "外网来源不再被挡"

    components.secrets = Secrets(rag_api_key="s3cret")
    client = _client(components)
    assert client.get("/health").status_code == 200, "配了 key 也不校验"
    assert client.get("/health", headers={"X-API-Key": "wrong"}).status_code == 200


# ---- 限流 ----


def test_the_rate_limit_returns_429_with_retry_after() -> None:
    components = _Components(_settings(rate_limit_per_min=2))
    client = _client(components)

    assert client.get("/health").status_code == 200
    assert client.get("/health").status_code == 200
    third = client.get("/health")

    assert third.status_code == 429
    assert third.headers["retry-after"] == "60"
    assert "超过限额" in third.json()["error"]["message"]


def test_zero_rate_limit_means_no_limit() -> None:
    components = _Components(_settings(rate_limit_per_min=0))
    client = _client(components)

    for _ in range(5):
        assert client.get("/health").status_code == 200


def test_identity_prefers_the_key_over_the_ip() -> None:
    """同一把 key 的多个客户端共享额度；没带 key 才按 IP 算。"""
    assert identity_of("abc", "1.2.3.4") == identity_of("abc", "5.6.7.8")
    assert identity_of(None, "1.2.3.4") != identity_of(None, "5.6.7.8")
    assert "abc" not in identity_of("abc", None), "key 原文不该出现在键名里"


# ---- /health：分开报 ----


def test_health_reports_each_dependency_separately() -> None:
    client = _client(_Components(_settings()))

    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["milvus"]["ok"] and "1691" in body["milvus"]["detail"]
    assert body["redis"]["ok"]
    assert body["postgres"]["ok"] and "多轮会话" in body["postgres"]["detail"]
    assert set(body["models"]) == {"embedding", "reranker", "chat"}
    assert (
        body["models"]["reranker"]["detail"].startswith("未启用")
        or body["models"]["reranker"]["ok"]
    )


def test_health_is_degraded_when_milvus_or_redis_is_down() -> None:
    """哪个挂了单独看得见 —— 合成一个 ok 只能回答「能不能用」。"""
    broken_index = _client(_Components(_settings(), index_ok=False))
    body = broken_index.get("/health").json()
    assert body["status"] == "degraded"
    assert not body["milvus"]["ok"] and "拒绝连接" in body["milvus"]["detail"]

    shared = _settings()
    shared.deployment.profile = "multiworker"
    shared.api.workers = 2
    shared.persistence.postgres.url = "postgresql://rag@localhost/rag"
    shared.api.redis.url = "redis://localhost:6379/0"
    shared.storage.minio.endpoint = "http://localhost:9000"
    apply_deployment_profile(shared)
    shared_secrets = Secrets(
        rag_api_key=None,
        minio_access_key="test-access",
        minio_secret_key="test-secret",
    )
    no_redis = _client(_Components(shared, redis_ok=False), secrets=shared_secrets)
    body = no_redis.get("/health").json()
    assert body["status"] == "degraded"
    assert not body["redis"]["ok"]
    assert "退回进程内" in body["redis"]["detail"], "降级要说清降成了什么"

    no_pg = _Components(shared)
    no_pg.persistence = _Persistence(ok=False)
    body = _client(no_pg, secrets=shared_secrets).get("/health").json()
    assert body["status"] == "degraded"
    assert not body["postgres"]["ok"] and "单轮" in body["postgres"]["detail"]


def test_local_lite_health_does_not_treat_absent_services_as_degraded() -> None:
    settings = _settings()
    settings.deployment.profile = "local-lite"
    components = _Components(settings, redis_ok=False)
    components.persistence = _Persistence(ok=False)

    body = _client(components).get("/health").json()

    assert body["status"] == "ok"
    assert body["redis"]["ok"] and "未启用" in body["redis"]["detail"]
    assert body["postgres"]["ok"] and "未启用" in body["postgres"]["detail"]


def test_health_never_triggers_model_loading() -> None:
    """`/health` 只看「加载了没有」—— 它不该把 1GB 模型拉起来。"""
    components = _Components(_settings())

    class _Exploding:
        def __getattr__(self, name):  # pragma: no cover - 触发就是失败
            raise AssertionError("health 不该访问模型的属性")

    components.heavy.embedder = _Exploding()
    client = _client(components)

    assert client.get("/health").status_code == 200


# ---- /search ----


def test_search_returns_hits_and_passes_the_options_through() -> None:
    components = _Components(_settings())
    client = _client(components)

    body = client.post(
        "/search",
        json={
            "query": "怎么配",
            "top_k": 3,
            "filters": {"modality": "image"},
            "channels": ["hybrid"],
        },
    ).json()

    assert body["hits"][0]["chunk_id"] == "d1:0001"
    assert body["hits"][0]["modality"] == "image"
    assert body["stats"][0]["channel"] == "hybrid:milvus:c"
    call = components.searches[0]
    assert (call["query"], call["top_k"]) == ("怎么配", 3)
    assert call["filters"] == {"modality": "image"} and call["channels"] == ["hybrid"]


def test_search_needs_a_query_or_an_image() -> None:
    client = _client(_Components(_settings()))

    empty = client.post("/search", json={"query": "  "})

    assert empty.status_code == 422
    assert "至少给一个" in empty.json()["error"]["message"]


def test_search_rejects_a_missing_image_path() -> None:
    client = _client(_Components(_settings()))

    resp = client.post("/search", json={"query": "", "image_path": "no/such.png"})

    assert resp.status_code == 422 and "不存在" in resp.json()["error"]["message"]


def test_search_rejects_an_oversized_top_k() -> None:
    client = _client(_Components(_settings()))

    assert client.post("/search", json={"query": "x", "top_k": 999}).status_code == 422


# ---- /chat ----


def test_chat_returns_the_answer_with_citations() -> None:
    client = _client(_Components(_settings()))

    body = client.post("/chat", json={"question": "怎么配"}).json()

    assert body["answer"] == "答案 [1]"
    assert body["citations"][0]["chunk_id"] == "d1:0001"
    assert body["grade"] == {"score": 0.9}


def test_chat_stream_emits_retrieval_then_tokens_then_done() -> None:
    """SSE 的事件顺序是契约：客户端靠它决定什么时候能渲染来源、什么时候结束。"""
    client = _client(_Components(_settings()))

    with client.stream("POST", "/chat", json={"question": "怎么配", "stream": True}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        text = "".join(resp.iter_text())

    events = [
        line.removeprefix("event: ") for line in text.splitlines() if line.startswith("event: ")
    ]
    assert events == ["retrieval", "token", "token", "done"]
    assert '"text": "你"' in text and '"text": "好"' in text
    assert '"chunk_id": "d1:0001"' in text, "done 事件里要带引用"


# ---- /ingest + /jobs ----


def test_ingest_accepts_a_job_and_reports_progress(tmp_path) -> None:
    target = tmp_path / "doc.pdf"
    target.write_bytes(b"%PDF-1.4")
    components = _Components(_settings())
    client = _client(components)

    accepted = client.post("/ingest", json={"paths": [str(target)]})
    assert accepted.status_code == 202
    job_id = accepted.json()["id"]

    job = client.get(f"/jobs/{job_id}").json()
    assert job["request"]["paths"] == [str(target)]


def test_ingest_rejects_a_missing_path() -> None:
    client = _client(_Components(_settings()))

    resp = client.post("/ingest", json={"paths": ["no/such/dir"]})

    assert resp.status_code == 422 and "路径不存在" in resp.json()["error"]["message"]


def test_an_unknown_job_is_404() -> None:
    client = _client(_Components(_settings()))

    assert client.get("/jobs/nope").status_code == 404


# ---- /chunks 与 /assets：前端「点开引用看原文/图片」要用的两个 ----


def test_a_chunk_can_be_fetched_by_id() -> None:
    """引用只给 chunk_id；要展开原文就得按 id 取 —— 一次 Milvus 主键查询，不重新检索。"""
    components = _Components(_settings())

    body = _client(components).get("/chunks/d1:0001").json()

    assert body["text"] == "这一块的完整正文"
    assert body["source_uri"] == "a.pdf" and body["page"] == 3
    assert body["assets"] == ["3481c48bcb813900/p0001_01.jpg"]
    assert components.index.asked_chunks == ["d1:0001"]


def test_an_unknown_chunk_is_404() -> None:
    assert _client(_Components(_settings())).get("/chunks/missing").status_code == 404


def test_assets_are_served_from_the_assets_dir(tmp_path) -> None:
    settings = _settings()
    settings.deployment.state_dir = str(tmp_path)
    apply_deployment_profile(settings)
    assets = Path(settings.ingest.assets.dir)
    (assets / "doc1").mkdir(parents=True)
    (assets / "doc1" / "p1.jpg").write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")

    response = _client(_Components(settings)).get("/assets/doc1/p1.jpg")

    assert response.status_code == 200 and response.content.startswith(b"\xff\xd8")


def test_assets_refuse_to_escape_their_root(tmp_path) -> None:
    """这个端点是**直接把磁盘文件吐出去**的，所以只认 assets 根目录底下的路径。

    `../` 与符号链接都能绕开字符串前缀判断，所以判据必须是「解析之后的位置」。
    """
    settings = _settings()
    settings.deployment.state_dir = str(tmp_path)
    apply_deployment_profile(settings)
    root = Path(settings.ingest.assets.dir)
    root.mkdir(parents=True)
    (tmp_path / "secret.txt").write_text("不该被读到", encoding="utf-8")
    (root / "ok.txt").write_text("可以读", encoding="utf-8")

    client = _client(_Components(settings))

    assert client.get("/assets/ok.txt").status_code == 200
    assert client.get("/assets/../secret.txt").status_code == 404
    assert client.get("/assets/..%2Fsecret.txt").status_code == 404
    assert client.get("/assets/").status_code == 404, "目录不该被列出来"


# ---- 占位 ----


def test_agent_answers_with_citations() -> None:
    """M5 起 `/agent` 是真端点：主 agent 自己决定查什么，返回答案 + 被引用到的来源。"""
    components = _Components(_settings())
    client = _client(components)

    resp = client.post("/agent", json={"message": "Flink 架构图里画了哪些数据源"})
    body = resp.json()

    assert resp.status_code == 200
    assert body["answer"] == "答案 [1]"
    assert body["citations"][0]["chunk_id"] == "d1:0001"
    assert body["steps"] == 4 and body["retrieved"] == 3
    assert components.agent_calls == [
        {"message": "Flink 架构图里画了哪些数据源", "thread_id": None, "web": False}
    ]


def test_agent_passes_the_thread_id_through() -> None:
    """多轮靠 thread_id —— 它必须原样传下去（历史在 PG 里）。"""
    components = _Components(_settings())

    body = (
        _client(components).post("/agent", json={"message": "接着上面", "thread_id": "t1"}).json()
    )

    assert body["thread_id"] == "t1"
    assert components.agent_calls[0]["thread_id"] == "t1"


def test_agent_passes_the_explicit_web_request_through() -> None:
    """`web=true` = **用户明确要求联网**：闸门放行网络搜索（默认只在语料查空时放行）。"""
    components = _Components(_settings())

    _client(components).post("/agent", json={"message": "上网查一下", "web": True})
    _client(components).post("/agent", json={"message": "问问语料"})

    assert [call["web"] for call in components.agent_calls] == [True, False]


def test_agent_needs_a_message() -> None:
    client = _client(_Components(_settings()))

    assert client.post("/agent", json={}).status_code == 422


def test_agent_reports_an_interrupt_instead_of_an_empty_answer() -> None:
    """撞上 HITL 闸门时**不能**回一个空答案就算完 —— 状态与「它想干什么」都要给出来，
    客户端才知道该调 /agent/resume 还是该收工。"""
    components = _Components(_settings())

    async def stopped(message, *, thread_id=None, web_explicit=False, session_id=None):
        return {
            "answer": "",
            "citations": [],
            "thread_id": "t1",
            "steps": 3,
            "retrieved": 4,
            "status": "interrupted",
            "interrupt": {
                "action_requests": [
                    {"name": "write_file", "args": {"file_path": "/workspace/r.md"}}
                ],
                "review_configs": [{"action_name": "write_file", "allowed_decisions": ["approve"]}],
            },
        }

    components.agent = stopped  # type: ignore[method-assign]
    body = _client(components).post("/agent", json={"message": "写个报告"}).json()

    assert body["status"] == "interrupted" and body["answer"] == ""
    assert body["interrupt"]["action_requests"][0]["name"] == "write_file"


def test_agent_resume_passes_the_decisions_through() -> None:
    components = _Components(_settings())

    body = (
        _client(components)
        .post(
            "/agent/resume",
            json={"thread_id": "t1", "decisions": [{"type": "approve"}]},
        )
        .json()
    )

    assert body["answer"] == "写好了 [1]" and body["status"] == "answered"
    assert components.resumes == [{"thread_id": "t1", "decisions": [{"type": "approve"}]}]


def test_agent_resume_needs_a_thread_and_a_decision() -> None:
    """恢复是**按会话**找回中断状态的：少一个 id、少一个决定都不该放过去。"""
    client = _client(_Components(_settings()))

    assert (
        client.post("/agent/resume", json={"decisions": [{"type": "approve"}]}).status_code == 422
    )
    assert (
        client.post("/agent/resume", json={"thread_id": "t1", "decisions": []}).status_code == 422
    )


def test_agent_stream_emits_tokens_tools_and_done() -> None:
    """SSE 版：token（正文增量）、tool（某次工具调用结束）、done（收尾 + 引用）。"""
    client = _client(_Components(_settings()))

    with client.stream("POST", "/agent", json={"message": "随便问问", "stream": True}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        text = "".join(resp.iter_text())

    events = [
        line.removeprefix("event: ") for line in text.splitlines() if line.startswith("event: ")
    ]
    assert events == ["token", "tool", "done"]
    assert '"text": "正在"' in text
    assert '"name": "search_knowledge"' in text
    assert '"chunk_id": "d1:0001"' in text, "done 事件里要带引用"
    assert '"total_tokens": 39' in text, "done 事件也要带这一轮的 token 用量"


def test_agent_resume_stream_emits_the_same_events_as_agent() -> None:
    """批准之后接着流：事件与 `/agent` **一模一样**（含工具的参数与耗时），
    客户端不该为「批准后」再写一套解析。"""
    components = _Components(_settings())

    with _client(components).stream(
        "POST",
        "/agent/resume",
        json={"thread_id": "t1", "decisions": [{"type": "approve"}], "stream": True},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        text = "".join(resp.iter_text())

    events = [
        line.removeprefix("event: ") for line in text.splitlines() if line.startswith("event: ")
    ]
    assert events == ["token", "tool", "done"]
    assert '"file_path": "/workspace/r.md"' in text, "工具事件要带参数"
    assert '"ms": 12' in text, "工具事件要带耗时"
    assert components.resumes == [{"thread_id": "t1", "decisions": [{"type": "approve"}]}]


def test_the_interrupt_event_carries_the_thread_id() -> None:
    """中断事件不带 thread_id 的话，客户端**根本没法批准** —— 恢复是按会话找回状态的。"""
    components = _Components(_settings())
    components.stream_agent = components.stream_agent_to_the_gate  # type: ignore[method-assign]

    with _client(components).stream(
        "POST", "/agent", json={"message": "写个报告", "stream": True}
    ) as resp:
        text = "".join(resp.iter_text())

    events = [
        line.removeprefix("event: ") for line in text.splitlines() if line.startswith("event: ")
    ]
    assert events == ["token", "interrupt"]
    assert '"thread_id": "t1"' in text
    assert '"name": "write_file"' in text, "「它想干什么」也要给出来"


# ---- 会话端点（P2 第三批：对话记录落 PG）----
# 与 `rag sessions`（checkpointer 的残留 thread）**不是一回事**，命名别搞混。


def test_sessions_list_carries_the_token_totals() -> None:
    """列表上就要显示「这段花了多少 token」，别让界面为每个会话再来一次请求。"""
    body = _client(_Components(_settings())).get("/sessions").json()

    assert len(body["sessions"]) == 1
    item = body["sessions"][0]
    assert item["session_id"] == "s1" and item["turns"] == 2
    assert item["usage"] == {
        "input_tokens": 1200,
        "output_tokens": 300,
        "total_tokens": 1500,
        "calls": 6,
    }


def test_session_detail_returns_messages_in_order() -> None:
    body = _client(_Components(_settings())).get("/sessions/s1").json()

    assert [item["seq"] for item in body["messages"]] == [1, 2, 3, 4]
    assert body["messages"][1]["role"] == "ai"
    assert body["messages"][3]["status"] == "interrupted", "中断那轮的状态要带出来"
    assert body["usage"]["total_tokens"] == 1500 and body["usage"]["calls"] == 5
    assert body["messages"][3]["interrupt"]["action_requests"][0]["name"] == "write_file", (
        "中断载荷是**读的时候现取**的（不进会话表）—— 重开这段对话要能把批准卡片摆回来"
    )
    assert body["messages"][1]["usage"]["total_tokens"] == 14, (
        "逐轮的 token 账要跟着消息一起回来 —— 不然刷新之后每一轮的用量就看不见了"
    )


def test_an_unknown_session_is_a_404() -> None:
    assert _client(_Components(_settings())).get("/sessions/nope").status_code == 404
    assert _client(_Components(_settings())).delete("/sessions/nope").status_code == 404


def test_deleting_a_session_reports_what_went() -> None:
    components = _Components(_settings())

    body = _client(components).delete("/sessions/s1").json()

    assert body == {"deleted": "s1"} and components.chats.deleted == ["s1"]


def test_sessions_say_so_when_storage_was_not_initialized() -> None:
    """会话存储没初始化时明确回 503，而不是让人对着 500 猜哪里坏了。"""
    components = _Components(_settings())
    components.chats = None

    resp = _client(components).get("/sessions")

    assert resp.status_code == 503
    assert "会话存储未初始化" in resp.json()["error"]["message"]


# ---- 问答落库：这是「换台机器、换台浏览器，会话都在」的全部依据 ----


def test_a_question_opens_a_session_titled_by_the_first_question() -> None:
    components = _real_components()

    sid = asyncio.run(components._open_session(None, title="Save Points 是做什么的", mode="chat"))

    assert sid and sid.startswith("c-"), "没给 id 就开一段新的"
    assert components.chats.created[0]["title"] == "Save Points 是做什么的"


def test_a_given_session_id_is_reused_instead_of_opening_a_new_one() -> None:
    components = _real_components()

    sid = asyncio.run(components._open_session("s1", title="追问", mode="agent"))

    assert sid == "s1"
    assert components.chats.created[0]["session_id"] == "s1"


def test_a_turn_lands_with_its_token_usage() -> None:
    components = _real_components()

    asyncio.run(
        components._save_turn(
            "s1",
            question="问什么",
            answer="答什么",
            mode="agent",
            thread_id="t1",
            usage={"calls": 3, "input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        )
    )

    turn = components.chats.turns[0]
    assert turn["question"] == "问什么" and turn["mode"] == "agent"
    assert turn["usage"]["total_tokens"] == 30 and turn["usage"]["calls"] == 3
    assert components.chats.pending == [], "不是恢复那轮，不该走 update_pending"


def test_a_resumed_turn_fills_the_pending_row_instead_of_appending() -> None:
    """恢复那轮的行在中断时就落了 —— 再插一行会让前端看到两个答复。"""
    components = _real_components()

    asyncio.run(
        components._save_turn("s1", question="", answer="写好了", mode="agent", resume=True)
    )

    assert components.chats.pending[0]["answer"] == "写好了"
    assert components.chats.turns == []


def test_storage_failure_never_breaks_the_answer() -> None:
    """存储是附属品：PG 抖动不该让一次问答 500（与日志同一个立场）。"""
    components = _real_components()

    class _Broken(_Chats):
        async def append_turn(self, session_id: str, **kwargs: Any) -> int:
            raise RuntimeError("PG 挂了")

    components.chats = _Broken()

    asyncio.run(components._save_turn("s1", question="问", answer="答", mode="chat"))
    assert True, "走到这里就是没抛出去"


def test_a_stopped_or_missing_usage_writes_no_token_row() -> None:
    """没拿到用量时**不写 0**：0 和「不知道」在统计里是两回事。"""
    components = _real_components()

    asyncio.run(components._save_turn("s1", question="问", answer="答", mode="chat", usage={}))

    assert components.chats.turns[0]["usage"] is None


def test_a_new_agent_thread_is_seeded_with_the_chat_history() -> None:
    """一段会话先用快问答聊了几轮，再切到 agent：thread 是**新的**，得把已有问答灌进去。

    不灌的话 agent 只看得到当前这一句 —— 快问答那条路没有 checkpointer，历史只在我们
    自己的会话表里。
    """
    components = _real_components()

    history = asyncio.run(components._history_for("s1"))

    assert history is not None
    assert [item["role"] for item in history] == ["user", "assistant", "user"]
    assert history[1]["content"].startswith("先说结论"), "ai 的正文要按 assistant 给"
    assert all(item["content"].strip() for item in history), "正文为空的那轮（中断）要跳过"


def test_history_is_none_when_there_is_nothing_to_seed() -> None:
    components = _real_components()

    assert asyncio.run(components._history_for(None)) is None, "没有会话就没有历史"
    assert asyncio.run(components._history_for("nope")) is None, "查不到的会话也一样"


# ---- 日志（2026-10-06：不进数据库，四个文件 + 进程内实时缓冲）----


def _log_settings(tmp_path) -> Settings:
    """把四个日志文件都指到 tmp（别去碰仓库里的 data/logs）。"""
    settings = _settings()
    settings.deployment.state_dir = str(tmp_path)
    apply_deployment_profile(settings)
    cfg = settings.observability.logging
    cfg.console = False
    return settings


def _write(path, *lines: str) -> None:
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def test_logs_tail_merges_the_four_files_in_time_order(tmp_path) -> None:
    """`/logs/tail` 读文件并把四类按时间合成一条流水 —— CLI 进程写的行也在里面。"""
    settings = _log_settings(tmp_path)
    _write(
        settings.observability.logging.files["interface"].path,
        "10-06 18:00:01 INFO  interface        req=r1 run=x POST /search 200 12ms",
    )
    _write(
        settings.observability.logging.files["channel"].path,
        "10-06 18:00:00 INFO  channel          召回=10 入选=2 380ms",
    )
    _write(
        settings.observability.logging.system.path,
        "10-06 18:00:02 WARNING client           重试第 3 次",
    )

    body = _client(_Components(settings)).get("/logs/tail?limit=10").json()

    assert [item["kind"] for item in body["items"]] == ["channel", "interface", "system"], (
        "按时间合并，不是按文件顺序"
    )
    line = body["items"][1]
    assert line["ts"] == "10-06 18:00:01" and line["level"] == "INFO"
    assert line["source"] == "interface", "来源单独一列，正文里不再重复短名"
    assert "POST /search 200" in line["text"]
    assert body["files"]["channel"].endswith("channel.log")


def test_logs_tail_honours_the_kind_filter_and_rejects_unknown_kinds(tmp_path) -> None:
    settings = _log_settings(tmp_path)
    _write(
        settings.observability.logging.files["channel"].path,
        "10-06 18:00:00 INFO  channel          召回=10 入选=2 380ms",
    )
    client = _client(_Components(settings))

    body = client.get("/logs/tail?kind=channel").json()
    assert len(body["items"]) == 1 and body["items"][0]["kind"] == "channel"

    # 类别写错要明确报错，不能悄悄给一份空列表（页面上会看成「没有日志」）
    assert client.get("/logs/tail?kind=nope").status_code == 422


def test_logs_tail_keeps_unparseable_lines_instead_of_dropping_them(tmp_path) -> None:
    """切不动时间/级别的行（比如第三方库自己格式化的那种）也要原样给出来。"""
    settings = _log_settings(tmp_path)
    _write(settings.observability.logging.system.path, "一整行没有时间也没有级别")

    body = _client(_Components(settings)).get("/logs/tail?kind=system").json()

    item = body["items"][0]
    assert item["ts"] is None and item["level"] == ""
    assert item["text"] == "一整行没有时间也没有级别"


def test_logs_tail_folds_continuation_lines_into_the_previous_record(tmp_path) -> None:
    """旧文件里那些「没有时间的续行」要并进上一条 —— 页面上不能出现半条记录。"""
    settings = _log_settings(tmp_path)
    _write(
        settings.observability.logging.system.path,
        "10-06 19:33:20 ERROR storage          req=r1 会话建不起来：缺列",
        "LINE 1: insert into chat_session (session_id, owner, ...",
        "                                              ^",
    )

    body = _client(_Components(settings)).get("/logs/tail?kind=system").json()

    assert len(body["items"]) == 1, "续行不算独立记录"
    item = body["items"][0]
    assert item["ts"] == "10-06 19:33:20" and item["level"] == "ERROR"
    assert "LINE 1:" in item["text"] and item["text"].rstrip().endswith("^")


def test_logs_stream_backfills_then_follows_the_live_buffer() -> None:
    """`/logs/stream` 先补最近几行、再跟着推 —— 实时走内存，不去 tail 文件。

    这里**直接驱动那两条协程**，不走 TestClient 的流式传输：那是一条永不结束的 SSE，
    用 `client.stream` 读会在关闭响应时卡在 portal 线程里（实测挂住）。要验的是
    「补历史 + 推新行」这两件事，驱动生成器就够，而且跑得快。
    """
    from rag.api import system_routes
    from rag.obs import live

    live.reset()
    live.publish(
        {
            "ts": "2026-10-06T18:00:00+08:00",
            "level": "INFO",
            "kind": "channel",
            "logger": "rag.channel",
            "text": "channel          召回=1",
        }
    )

    async def scenario() -> tuple[str, str]:
        response = await system_routes.logs_stream(backfill=5)
        assert response.media_type == "text/event-stream"
        backfill = await anext(response.body_iterator)

        # 订阅之后再发一条：应当被推出来（而不是等 15 秒心跳）
        gen = system_routes._log_sse(backfill=0)
        pending = asyncio.create_task(anext(gen))
        await asyncio.sleep(0.05)
        live.publish(
            {
                "ts": "2026-10-06T18:00:01+08:00",
                "level": "WARNING",
                "kind": "system",
                "logger": "pymilvus.client",
                "text": "跟着来的",
            }
        )
        followed = await asyncio.wait_for(pending, timeout=2)
        return str(backfill), str(followed)

    backfill, followed = asyncio.run(scenario())

    assert backfill.startswith("event: log") and "召回=1" in backfill
    assert followed.startswith("event: log") and "跟着来的" in followed
    live.reset()
