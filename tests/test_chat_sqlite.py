"""本地 profile 的 UI 会话：SQLite 落库、重启、API 列表。"""

from __future__ import annotations

import asyncio
import sqlite3
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from rag.api.app import Components, create_app
from rag.config.schema import Settings
from rag.config.settings import Secrets
from rag.deployment import apply_deployment_profile
from rag.persistence import ChatStore, SqliteChatStore, build_chat_store


def _store(path: Any) -> SqliteChatStore:
    store = SqliteChatStore(path)
    asyncio.run(store.setup())
    return store


def test_sqlite_chat_store_round_trips_and_survives_reopen(tmp_path) -> None:
    path = tmp_path / "chats.sqlite3"
    store = _store(path)

    asyncio.run(store.create(session_id="c1", title="第一问", mode="chat"))
    asyncio.run(
        store.append_turn(
            "c1",
            question="第一问",
            answer="第一答",
            usage={
                "model": "local",
                "calls": 1,
                "input_tokens": 3,
                "output_tokens": 5,
                "total_tokens": 8,
            },
        )
    )

    reopened = _store(path)
    rows = asyncio.run(reopened.list())
    assert rows[0]["session_id"] == "c1"
    assert rows[0]["turns"] == 1
    assert rows[0]["input_tokens"] == 3 and rows[0]["output_tokens"] == 5
    assert rows[0]["created_at"].endswith("Z")

    found = asyncio.run(reopened.get("c1"))
    assert found is not None
    assert [(item["role"], item["content"]) for item in found["messages"]] == [
        ("human", "第一问"),
        ("ai", "第一答"),
    ]
    assert found["messages"][1]["usage"]["total_tokens"] == 8
    assert found["usage"] == {
        "input_tokens": 3,
        "output_tokens": 5,
        "total_tokens": 8,
        "calls": 1,
    }


def test_sqlite_chat_store_fills_interrupted_turn_and_cascades_delete(tmp_path) -> None:
    store = _store(tmp_path / "chats.sqlite3")
    asyncio.run(store.create(session_id="c2", mode="agent"))
    asyncio.run(
        store.append_turn(
            "c2",
            question="写报告",
            mode="agent",
            status="interrupted",
            thread_id="t1",
        )
    )

    assert asyncio.run(
        store.update_pending(
            "c2",
            answer="已批准并完成",
            usage={"calls": 1, "total_tokens": 7},
        )
    )
    found = asyncio.run(store.get("c2"))
    assert found is not None
    assert found["messages"][1]["status"] == "answered"
    assert found["messages"][1]["content"] == "已批准并完成"
    assert found["thread_id"] == "t1"

    assert asyncio.run(store.delete("c2"))
    assert asyncio.run(store.get("c2")) is None
    with sqlite3.connect(tmp_path / "chats.sqlite3") as conn:
        assert conn.execute("select count(*) from chat_message").fetchone()[0] == 0
        assert conn.execute("select count(*) from token_usage").fetchone()[0] == 0


def test_build_chat_store_picks_sqlite_for_local_and_pg_for_multiworker(
    tmp_path,
) -> None:
    local = Settings()
    local.deployment.state_dir = str(tmp_path)
    apply_deployment_profile(local)
    assert isinstance(build_chat_store(local), SqliteChatStore)

    shared = Settings()
    shared.deployment.profile = "multiworker"
    shared.api.workers = 2
    shared.persistence.postgres.url = "postgresql://rag@localhost/rag"
    shared.api.redis.url = "redis://localhost:6379/0"
    shared.storage.minio.endpoint = "http://localhost:9000"
    apply_deployment_profile(shared)
    assert isinstance(build_chat_store(shared), ChatStore)


def test_sessions_endpoint_reads_the_local_store(tmp_path) -> None:
    settings = Settings()
    settings.deployment.profile = "local-lite"
    settings.deployment.state_dir = str(tmp_path)
    apply_deployment_profile(settings)
    store = _store(settings.persistence.chat_sqlite_path)
    asyncio.run(store.create(session_id="c3", title="对话", mode="chat"))
    asyncio.run(store.append_turn("c3", question="问", answer="答"))

    secrets = Secrets(rag_api_key=None)
    components = Components(settings, secrets)
    components.chats = store
    app = create_app(settings, secrets)
    app.state.components = components
    client = TestClient(app)

    listed = client.get("/sessions")
    assert listed.status_code == 200, listed.text
    assert listed.json()["sessions"][0]["session_id"] == "c3"

    detail = client.get("/sessions/c3")
    assert detail.status_code == 200, detail.text
    assert detail.json()["messages"][1]["content"] == "答"


def test_components_startup_initializes_local_chat_store(tmp_path, monkeypatch) -> None:
    import rag.api.app as app_module
    import rag.indexing

    settings = Settings()
    settings.deployment.profile = "local-lite"
    settings.deployment.state_dir = str(tmp_path)
    apply_deployment_profile(settings)

    class _Index:
        async def ping(self, timeout_s: float) -> None:
            return None

        async def ensure_ready(self) -> None:
            return None

    monkeypatch.setattr(rag.indexing, "build_index", lambda _settings: _Index())
    monkeypatch.setattr(
        app_module,
        "build_heavy",
        lambda _settings, _index: SimpleNamespace(errors={}, embedder=None, reranker=None),
    )

    components = Components(settings, Secrets(rag_api_key=None))
    asyncio.run(components.startup())
    try:
        assert isinstance(components.chats, SqliteChatStore)
        assert asyncio.run(components.chats.list()) == []
    finally:
        asyncio.run(components.shutdown())
