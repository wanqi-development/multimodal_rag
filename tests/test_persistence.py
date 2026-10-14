"""PG 持久化：连接串拼装、ProactorEventLoop 守卫、没配就退回进程内。

全部离线：不真连 PG（连接本身在真机探针里验过），这里只钉住「拼装与守卫」这两件
容易写错、又不容易从报错里看出原因的地方。
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

from rag.config.schema import PostgresConfig, Settings
from rag.persistence import pg as persistence


def test_conninfo_carries_the_schema_and_never_leaks_it_in_repr() -> None:
    """schema 只能从连接串进（库不认 schema 参数），密码也在这里 —— 所以**别打日志**。"""
    item = persistence.PgPersistence(
        PostgresConfig(url="postgresql://rag@127.0.0.1:5432/ragdb", schema_name="rag2"),
        "s3cret",
    )

    conninfo = item.conninfo()

    assert "search_path=rag2" in conninfo
    assert "password=s3cret" in conninfo
    assert "rag2" in conninfo and "ragdb" in conninfo


def test_an_unconfigured_persistence_is_not_an_error() -> None:
    """url 留空 = 不用 PG（退回进程内），不该抛、也不该去连。"""
    item = persistence.PgPersistence(PostgresConfig(url=""), None)

    assert item.configured is False
    assert asyncio.run(item.connect()) is False
    assert "未配置" in item.reason


class ProactorEventLoop:  # 名字就是判据：psycopg 看的是循环类型
    pass


class SelectorEventLoop:
    pass


def test_the_proactor_guard_says_what_to_do(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows 上 psycopg 的 async 跑不了 Proactor —— 而它的表象是「连接池 10 秒超时」，
    从报错里根本看不出原因。所以这里必须当场说清是什么问题、该怎么修（实测踩过）。"""
    # 换掉 pg 模块看到的那份 sys，别去动全局的 sys.platform
    monkeypatch.setattr(persistence, "sys", SimpleNamespace(platform="win32"))

    with pytest.raises(RuntimeError, match="ensure_compatible_event_loop"):
        persistence.check_loop_compatible(ProactorEventLoop())

    persistence.check_loop_compatible(SelectorEventLoop())  # 别的循环不管


def test_the_guard_does_nothing_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(persistence, "sys", SimpleNamespace(platform="linux"))

    persistence.check_loop_compatible(ProactorEventLoop())  # 不该抛


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="验的是 Windows 专有的 Selector 策略类（asyncio.WindowsSelectorEventLoopPolicy）",
)
def test_ensure_compatible_event_loop_switches_the_policy(monkeypatch) -> None:
    """它只换策略、不建循环 —— 所以**必须在建循环之前**调，晚了就没用。

    ⚠️ 这条用例里 `set_/get_event_loop_policy` 都打了桩：第一版只打桩 get，
    `set_event_loop_policy` 是**真**调的，于是全局策略被换成假对象、而 monkeypatch
    还原不了它（那不是它改的），**后面十几条用例一起挂**。测试留下真实副作用，
    比测试失败难查得多。
    """
    monkeypatch.setattr(persistence, "sys", SimpleNamespace(platform="win32"))
    switched: list[object] = []
    monkeypatch.setattr(persistence.asyncio, "get_event_loop_policy", lambda: object())
    monkeypatch.setattr(
        persistence.asyncio, "set_event_loop_policy", lambda policy: switched.append(policy)
    )

    persistence.ensure_compatible_event_loop()

    assert len(switched) == 1, "应当换一次策略"
    assert isinstance(switched[0], asyncio.AbstractEventLoopPolicy)
    assert Settings().persistence.postgres.schema_name == "rag"


def test_the_loop_factory_never_hands_uvicorn_a_proactor_loop() -> None:
    """uvicorn ≥0.36 不看事件循环策略：`Server.run()` 直接把 loop 工厂的返回值交给
    `asyncio_run`，而它在 Windows 上给的是 `ProactorEventLoop`（实测 2026-10-05，
    `rag serve` 因此起不来）。所以给 uvicorn 的必须是我们这个工厂。

    ⚠️ 这条用例会真的换一次全局策略（那正是它要验的事），**所以结束时必须还原** ——
    测试留下真实副作用比测试失败难查得多（这一课批次 A 已经踩过一次）。
    """
    policy_before = asyncio.get_event_loop_policy()
    loop = persistence.selector_event_loop()
    try:
        assert type(loop).__name__ != "ProactorEventLoop"
        assert loop.is_running() is False
        if sys.platform == "win32":
            assert "Selector" in type(loop).__name__, "Windows 上要的是 Selector 循环"
    finally:
        loop.close()
        asyncio.set_event_loop_policy(policy_before)
