"""日志的**索引那一份**：一行怎么变成库里的一行、批量写、失败不拖业务、查询口。

文件仍是唯一真相（见 `tests/test_logs.py`）。这里验的是 2026-10-08 加的那条支路：
每一行顺手再进一次 PG，页面才能按 `request_id` / 类别 / 时间拉 —— 文件一多就没法扫。
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag.api.app import Components, create_app
from rag.config.schema import Settings
from rag.config.settings import Secrets
from rag.obs import log as obs
from rag.obs.log import row_of, split_ids, tail_items
from rag.obs.pg import PgLogHandler, build_pg_handler


@pytest.fixture(autouse=True)
def _clean_ids() -> None:
    """关联 id 是 contextvar 里的全局状态，每个用例前后都还原。"""
    obs._REQUEST_ID.set(None)
    obs._RUN_ID.set(None)
    obs.set_source("")
    yield
    obs._REQUEST_ID.set(None)
    obs._RUN_ID.set(None)
    obs.set_source("")


def _record(
    name: str = "rag.interface",
    *,
    level: int = logging.INFO,
    message: str = "x",
    row: dict[str, Any] | None = None,
) -> logging.LogRecord:
    record = logging.LogRecord(name, level, "probe.py", 1, message, (), None)
    if row is not None:
        record.rag_row = row
    return record


# ---- 一行 → 库里的字段 ----


def test_a_business_row_carries_category_status_and_duration() -> None:
    obs.set_ids(request_id="req-1", run_id="job-2")
    obs.set_source("http")

    row = row_of(
        _record(
            row={
                "category": "interface",
                "text": "POST /chat 200 12ms",
                "status": 200,
                "duration_ms": 12.4,
            }
        )
    )

    assert row is not None
    assert row["category"] == "interface"
    assert row["request_id"] == "req-1" and row["run_id"] == "job-2"
    assert row["source"] == "http", "入口是 HTTP，不从 id 前缀反推"
    assert row["logger"] == "rag.interface"
    assert row["status"] == 200 and row["duration_ms"] == 12, "耗时列是 int 毫秒"
    assert row["text"] == "POST /chat 200 12ms"
    assert row["ts"].tzinfo is not None, "时间要带时区（列是 timestamptz）"


def test_a_library_row_lands_as_system_with_its_entry_marked() -> None:
    """第三方库的行也收：一次请求里 pymilvus 报的那句重试，靠同一个 request_id 串得上。"""
    obs.set_ids(request_id="req-1")
    obs.set_source("http")

    row = row_of(_record("pymilvus.client", level=logging.WARNING, message="重试\n第 3 次"))

    assert row is not None
    assert row["category"] == "system" and row["source"] == "library"
    assert "\n" not in row["text"] and "\\n" in row["text"], "多行压成一行（与文件一个口径）"
    assert row["request_id"] == "req-1", "库里的行也跟着这次请求"


def test_a_row_outside_any_request_has_no_id_and_gets_one_at_write_time() -> None:
    row = row_of(_record(row={"category": "channel", "text": "rrf 召回=1"}))
    assert row is not None and row["request_id"] == "", "补 solo- 是写入口的事（下面有用例）"
    assert row["source"] == "task", "不在请求里、也不是第三方库 → task"


def test_the_write_side_fills_a_solo_id_so_the_column_stays_not_null() -> None:
    from rag.persistence.logs import LogStore

    prepared = LogStore._prepare([{"category": "system", "text": "库自己打的"}])

    assert len(prepared) == 1 and len(prepared[0]) == 10, "列数与 _COLUMNS 对齐"
    assert str(prepared[0][3]).startswith("solo-")


def test_search_takes_dict_rows_and_tuple_rows_both() -> None:
    """池那条路给 dict（建池时带了 `row_factory`），短连接那条路也给 dict ——
    但替身或别处塞进来的连接可能还是 tuple，`search()` 两条都得吃得下。

    实测踩过：少了 `row_factory` 时 `dict(row)` 抛
    `cannot convert dictionary update sequence element #0 to a sequence`，
    `/logs/search` 直接 500（页面看起来就是「点了 request_id 啥也没有」）。
    """
    import asyncio
    import contextlib

    from rag.persistence.logs import _COLUMNS, LogStore

    class _Cursor:
        description = [type("Col", (), {"name": name}) for name in _COLUMNS]

        def __init__(self, rows: list[Any]) -> None:
            self._rows = rows

        async def fetchall(self) -> list[Any]:
            return self._rows

    class _Conn:
        def __init__(self, rows: list[Any]) -> None:
            self._rows = rows

        async def execute(self, sql: str, args: tuple) -> _Cursor:
            return _Cursor(self._rows)

    def store_with(rows: list[Any]) -> LogStore:
        conn = _Conn(rows)

        @contextlib.asynccontextmanager
        async def connect():
            yield conn

        return LogStore(connect=connect)

    as_dict = {name: name for name in _COLUMNS}
    for rows in ([tuple(_COLUMNS)], [as_dict]):
        found = asyncio.run(store_with(rows).search(request_id="req-1"))
        assert found == [as_dict], f"两种行形状都要能认：{rows}"


# ---- handler：批量、失败不抛、退出把尾巴写完 ----


def test_the_handler_writes_in_batches_and_flushes_the_tail_on_close() -> None:
    written: list[dict[str, Any]] = []
    handler = PgLogHandler(written.extend, interval=0.05, batch_size=10)

    for index in range(3):
        handler.emit(_record(row={"category": "channel", "text": f"第 {index} 行"}))

    handler.close()

    assert [item["text"] for item in written] == ["第 0 行", "第 1 行", "第 2 行"]
    assert handler.stats()["lost_rows"] == 0 and handler.stats()["pending"] == 0


def test_a_failed_flush_only_shouts_and_never_raises(capsys: pytest.CaptureFixture) -> None:
    """PG 挂着 / 表没建的时候，日志写不进去**不该拖业务**，只喊一句。"""

    def boom(rows: list[dict[str, Any]]) -> None:
        raise RuntimeError("表不存在")

    handler = PgLogHandler(boom, interval=0.05)
    handler.emit(_record(row={"category": "channel", "text": "一条"}))
    handler.close()  # 不该抛

    stats = handler.stats()
    assert stats["failed_flushes"] >= 1 and stats["lost_rows"] >= 1
    assert "日志进库失败" in capsys.readouterr().err


def test_a_full_queue_drops_rows_instead_of_blocking() -> None:
    """队列有界：下游卡住时**丢新行**，绝不把调用方（业务线程）堵在那儿。"""
    gate = threading.Event()

    def slow(rows: list[dict[str, Any]]) -> None:
        gate.wait(2.0)

    handler = PgLogHandler(slow, interval=0.01, batch_size=1, queue_max=1)
    for index in range(5):
        handler.emit(_record(row={"category": "channel", "text": f"第 {index} 行"}))
    time.sleep(0.3)

    assert handler.stats()["dropped"] >= 1
    gate.set()
    handler.close()


def test_no_pg_url_or_a_switch_off_means_no_handler() -> None:
    settings = Settings()
    assert build_pg_handler(settings) is None, "没配 PG 就只落文件"

    settings.persistence.postgres.url = "postgresql://rag@127.0.0.1:5432/x"
    settings.observability.logging.postgres = False
    assert build_pg_handler(settings) is None, "配置里关掉就不装"


# ---- 行首那两段关联 id 要能单独拿出来（页面上要能点）----


def test_split_ids_takes_the_prefix_away_from_the_message() -> None:
    assert split_ids("req=r1 run=run-9 GET /health 200") == ("r1", "run-9", "GET /health 200")
    assert split_ids("只有正文") == ("", "", "只有正文")
    assert split_ids("run=run-1 没有 req") == ("", "run-1", "没有 req")


def test_tail_items_lifts_the_ids_and_leaves_them_out_of_the_text(tmp_path: Path) -> None:
    path = tmp_path / "interface.log"
    path.write_text(
        "10-07 10:00:01.100 INFO  interface        req=r1 run=run-9 GET /health 200 1ms\n",
        encoding="utf-8",
    )

    item = tail_items({"interface": str(path)}, limit=1)[0]

    assert item["request_id"] == "r1" and item["run_id"] == "run-9"
    assert item["text"] == "GET /health 200 1ms", "正文里不再重复一遍 id"


# ---- 查询端点 ----


def _client(settings: Settings) -> TestClient:
    components = Components(settings, Secrets(rag_api_key=None))
    components.persistence = None
    app = create_app(settings, Secrets(rag_api_key=None))
    app.state.components = components
    return TestClient(app)


def test_search_requires_a_request_id() -> None:
    settings = Settings()
    settings.persistence.postgres.url = "postgresql://rag@127.0.0.1:5432/x"

    answer = _client(settings).get("/logs/search")

    assert answer.status_code == 422, "request_id 必填：这个口子回答的是「这次请求怎么了」"


def test_search_says_503_when_there_is_no_pg() -> None:
    """没配 PG → 明确 503 并指路 `/logs/tail`，而不是让人对着 500 猜。"""
    answer = _client(Settings()).get("/logs/search", params={"request_id": "req-1"})

    assert answer.status_code == 503
    assert "logs/tail" in answer.json()["error"]["message"]


def test_search_maps_the_rows_into_the_same_shape_as_the_file_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import UTC, datetime

    class _Store:
        async def search(self, **kwargs: Any) -> list[dict[str, Any]]:
            assert kwargs["request_id"] == "req-1" and kwargs["category"] == "channel"
            return [
                {
                    "ts": datetime(2026, 10, 8, 2, 0, 1, 500_000, tzinfo=UTC),
                    "category": "channel",
                    "level": "INFO",
                    "request_id": "req-1",
                    "run_id": "run-2",
                    "source": "http",
                    "logger": "rag.channel",
                    "status": None,
                    "duration_ms": 11,
                    "text": "rrf 召回=10 入选=8",
                }
            ]

    import rag.persistence.logs as logs

    monkeypatch.setattr(logs, "build_log_store", lambda *a, **k: _Store())
    settings = Settings()
    settings.persistence.postgres.url = "postgresql://rag@127.0.0.1:5432/x"

    body = (
        _client(settings)
        .get("/logs/search", params={"request_id": "req-1", "category": "channel"})
        .json()
    )

    assert body["request_id"] == "req-1"
    item = body["items"][0]
    assert item["kind"] == "channel" and item["source"] == "channel", "来源仍是 logger 短名"
    assert item["request_id"] == "req-1" and item["run_id"] == "run-2"
    assert item["origin"] == "http", "库里那份多一个入口标记"
    assert item["text"] == "rrf 召回=10 入选=8"
    assert item["ts"].startswith("10-08 "), "时间按本机时区显示，与文件那一路同形"
