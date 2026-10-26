"""模型选型的存储 + 审计（2026-10-09 第②步）。

只测**单机 SQLite** 那条路（真读写一个临时文件）：PG 那份的 SQL 与它几乎一样，
差别只在自增列与占位符，等有真库时按 `rag models init/list/set` 走一遍即可。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.persistence.model_settings import SqliteModelSettingStore


def _store(tmp_path: Path) -> SqliteModelSettingStore:
    store = SqliteModelSettingStore(tmp_path / "models.sqlite3")
    store.setup()
    return store


def test_a_fresh_store_has_nothing_and_setting_one_writes_an_audit_row(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.all() == {}

    result = store.set("chat.agent", provider_key="dashscope", model="qwen-plus", actor="tester")

    assert result["version"] == 1 and result["old_value"] == ""
    row = store.all()["chat.agent"]
    assert (row["provider_key"], row["model"], row["version"]) == ("dashscope", "qwen-plus", 1)
    [change] = store.changes(slot="chat.agent")
    assert change["actor"] == "tester" and change["old_value"] == ""
    assert json.loads(change["new_value"])["model"] == "qwen-plus"


def test_setting_the_same_slot_again_bumps_the_version_and_keeps_the_old_value(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.set("rerank", provider_key="sentence-transformers", model="bge-reranker-base")

    result = store.set("rerank", provider_key="jev", model="jev-reranker")

    assert result["version"] == 2, "版本号要递增（将来热切靠它判断新旧）"
    assert "bge-reranker-base" in result["old_value"], "旧值要留在审计里"
    assert [c["slot"] for c in store.changes()] == ["rerank", "rerank"], "两次变更两行"


def test_an_unknown_slot_is_rejected_at_write_time(tmp_path: Path) -> None:
    """槽位名是库里主键与界面上的键：写错一个就是"界面上永远看不到的那一行"。"""
    store = _store(tmp_path)

    with pytest.raises(ValueError, match="未知的模型槽位"):
        store.set("chat.unknown", model="x")


def test_params_round_trip_as_a_dict(tmp_path: Path) -> None:
    store = _store(tmp_path)

    store.set("embedding", model="qwen3-vl-embedding", params={"dim": 2560})

    assert store.all()["embedding"]["params"] == {"dim": 2560}
