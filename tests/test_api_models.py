"""`GET /models`：八个槽位 + 来源 + key 就绪状态（只读接口，2026-10-09 第③步）。"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from rag.api.app import Components, create_app
from rag.config.schema import Settings
from rag.config.settings import Secrets
from rag.deployment import apply_deployment_profile
from rag.persistence.model_settings import SqliteModelSettingStore


def _client(tmp_path: Path, settings: Settings) -> TestClient:
    components = Components(settings, Secrets(rag_api_key=None))
    app = create_app(settings, Secrets(rag_api_key=None))
    app.state.components = components
    return TestClient(app)


def _settings(tmp_path: Path) -> Settings:
    settings = Settings()
    settings.deployment.profile = "local-lite"
    settings.deployment.state_dir = str(tmp_path)
    apply_deployment_profile(settings)
    settings.models.backend = "sqlite"
    return settings


def test_all_slots_are_listed_and_unset_means_following_the_env(tmp_path: Path) -> None:
    body = _client(tmp_path, _settings(tmp_path)).get("/models").json()

    assert [item["slot"] for item in body["slots"]] == [
        "chat.fast",
        "chat.agent",
        "chat.judge",
        "rerank",
        "embedding",
        "vision",
        "asr",
        "ocr",
    ]
    assert all(item["source"] == "未设置" for item in body["slots"]), (
        "还没人设过 → 沿用 .env / yaml"
    )
    assert all(item["hot_swappable"] is False for item in body["slots"]), "热切是遗留项"
    assert body["backend"] == "sqlite" and body["storage"].startswith("sqlite:")


def test_a_slot_set_in_the_store_shows_up_with_its_source(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = SqliteModelSettingStore(settings.models.sqlite_path)
    store.setup()
    store.set("chat.agent", provider_key="dashscope", model="qwen-plus", actor="tester")

    body = _client(tmp_path, settings).get("/models").json()
    agent = next(item for item in body["slots"] if item["slot"] == "chat.agent")

    assert agent["source"] == "库" and agent["provider_key"] == "dashscope"
    assert (
        agent["model"] == "qwen-plus" and agent["version"] == 1 and agent["updated_by"] == "tester"
    )
    assert agent["candidates"] == ["default"], "chat 槽位的候选来自 providers.chat 的键名"


def test_put_writes_the_slot_and_records_an_audit_row(tmp_path: Path) -> None:
    """页面上的「保存」走这条：写库 + 写审计，回整份列表（前端不用再拉一次）。"""
    settings = _settings(tmp_path)
    store = SqliteModelSettingStore(settings.models.sqlite_path)
    store.setup()
    client = _client(tmp_path, settings)

    body = client.put(
        "/models/chat.agent",
        json={"provider_key": "default", "model": "qwen-plus", "actor": "tester"},
    ).json()

    agent = next(i for i in body["slots"] if i["slot"] == "chat.agent")
    assert (agent["source"], agent["model"], agent["version"]) == ("库", "qwen-plus", 1)
    assert agent["updated_by"] == "tester"
    [change] = store.changes(slot="chat.agent")
    assert change["actor"] == "tester" and "qwen-plus" in change["new_value"]


def test_put_rejects_an_unknown_slot_and_an_unknown_provider_key(tmp_path: Path) -> None:
    client = _client(tmp_path, _settings(tmp_path))

    assert client.put("/models/chat.nope", json={"model": "x"}).status_code == 422
    bad = client.put("/models/chat.agent", json={"provider_key": "nope", "model": "x"})
    assert bad.status_code == 422 and "providers.chat" in bad.json()["error"]["message"]


def test_the_api_never_returns_a_key(tmp_path: Path) -> None:
    """只报"配没配"：`key_ready` 是布尔，响应里**一个字符的 key 都不许有**。"""
    settings = _settings(tmp_path)
    store = SqliteModelSettingStore(settings.models.sqlite_path)
    store.setup()
    store.set("chat.fast", provider_key="dashscope", model="qwen-plus")
    client = _client(tmp_path, settings)
    client.app.state.components.secrets = Secrets(rag_api_key=None, dashscope_api_key="sk-secret")

    raw = client.get("/models").text

    assert "sk-secret" not in raw
    slot = next(
        item for item in client.get("/models").json()["slots"] if item["slot"] == "chat.fast"
    )
    assert slot["key_ready"] is True, "配了 key 只报 True"
