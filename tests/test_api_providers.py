"""`GET/PUT /providers`：设置页填 key 的那条路（2026-10-09，B 方案）。

钉住三件事：**页面永远拿不到 key**（只有 `has_key` 与指纹）、**没主密钥就拒收 key**
（回 422 并点名 RAG_SECRET_KEY）、**没跑过 init 的机器也能直接保存**（写入前建表）。
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

from rag.api.app import Components, create_app
from rag.config.schema import Settings
from rag.config.settings import Secrets
from rag.persistence import secrets as S

KEY = "sk-live-xyz"


def _settings(tmp_path: Path) -> Settings:
    settings = Settings()
    settings.models.backend = "sqlite"
    settings.models.sqlite_path = str(tmp_path / "models.sqlite3")
    return settings


def _secrets(**kwargs) -> Secrets:
    return Secrets(rag_api_key=None, **kwargs)


def _client(settings: Settings, secrets: Secrets) -> TestClient:
    app = create_app(settings, secrets)
    app.state.components = Components(settings, secrets)
    return TestClient(app)


def _master(monkeypatch) -> None:
    """给这一轮塞一把主密钥（部署配置那条路：RAG_SECRET_KEY）。"""
    monkeypatch.setenv("RAG_SECRET_KEY", base64.b64encode(os.urandom(32)).decode())


def test_without_a_master_key_the_page_cannot_take_a_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("RAG_SECRET_KEY", raising=False)
    settings = _settings(tmp_path)
    secrets = _secrets()

    raw = _client(settings, secrets).get("/providers").text
    body = json.loads(raw)

    assert [p["provider_key"] for p in body["providers"]] == list(settings.providers.chat)
    assert all(p["can_set_key"] is False for p in body["providers"]), "没主密钥 → 页面禁填"
    # 就算 .env 里真配了 key（has_key 会 True），响应也只说"配没配"，不回值
    assert "v1:" not in raw, "库里密文不出后端"
    for field in ("deepseek_api_key", "openai_api_key", "dashscope_api_key"):
        value = getattr(secrets, field, None)
        if value:
            assert value not in raw


def test_put_stores_a_ciphertext_and_returns_only_a_fingerprint(
    tmp_path: Path, monkeypatch
) -> None:
    _master(monkeypatch)
    client = _client(_settings(tmp_path), _secrets())

    raw = client.put(
        "/providers/dashscope",
        json={"api_key": KEY, "base_url": "https://gw/v1", "actor": "tester"},
    ).text

    assert KEY not in raw, "响应里一个字符的 key 都不许有"
    row = next(p for p in json.loads(raw)["providers"] if p["provider_key"] == "dashscope")
    assert row["has_key"] is True and row["api_key_fingerprint"] == S.fingerprint(KEY)
    assert row["base_url"] == "https://gw/v1" and row["can_set_key"] is True
    # **主库 + WAL 一起扫**：落的是密文、明文不落盘（WAL 模式，只扫主库会假阴性）
    blob = b"".join(p.read_bytes() for p in tmp_path.glob("models.sqlite3*"))
    assert KEY.encode() not in blob and b"v1:" in blob


def test_a_key_without_a_master_key_is_rejected_but_base_url_still_saves(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("RAG_SECRET_KEY", raising=False)
    client = _client(_settings(tmp_path), _secrets())

    bad = client.put("/providers/openai", json={"api_key": KEY})

    assert bad.status_code == 422 and "RAG_SECRET_KEY" in bad.json()["error"]["message"], (
        "当场点名缺哪把主密钥"
    )
    ok = client.put("/providers/openai", json={"base_url": "https://gw/v1"})
    row = next(p for p in ok.json()["providers"] if p["provider_key"] == "openai")
    assert ok.status_code == 200 and row["base_url"] == "https://gw/v1"
    assert row["api_key_fingerprint"] == "", "被拒的那把 key 不许留下任何痕迹"


def test_an_empty_key_clears_it_and_a_brand_new_provider_can_be_added(
    tmp_path: Path, monkeypatch
) -> None:
    _master(monkeypatch)
    client = _client(_settings(tmp_path), _secrets())
    client.put("/providers/dashscope", json={"api_key": KEY})

    cleared = client.put("/providers/dashscope", json={"api_key": ""}).json()
    row = next(p for p in cleared["providers"] if p["provider_key"] == "dashscope")
    assert row["api_key_fingerprint"] == "", "空串 = 显式清除（回到部署配置那份）"

    # 库里存过的家即使不在 providers.chat 里，也要被列出来（允许新增自建网关）
    added = client.put(
        "/providers/custom-gw",
        json={"base_url": "https://custom/v1", "models": [{"name": "m1", "type": "chat"}]},
    ).json()
    keys = [p["provider_key"] for p in added["providers"]]
    custom = next(p for p in added["providers"] if p["provider_key"] == "custom-gw")
    assert "custom-gw" in keys
    assert custom["base_url"] == "https://custom/v1"
    assert custom["models"] == [{"name": "m1", "type": "chat"}]
