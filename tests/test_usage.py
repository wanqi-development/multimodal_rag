"""token 用量累加：两个调用点（生成器、agent 流）共用一套口径。

**不编数字**是这个模块的全部立场：拿不到 `usage_metadata` 就返回 None，
宁可这一轮没有数字，也不要拿字符数估一个填进统计表。
"""

from __future__ import annotations

from types import SimpleNamespace

from rag.usage import absorb_usage, usage_pair, usage_totals


def _chunk(**meta: object) -> SimpleNamespace:
    return SimpleNamespace(usage_metadata=meta or None)


def test_usage_accumulates_across_calls() -> None:
    """agent 一轮里有好几次模型调用，而 `usage_metadata` 通常只挂在最后一块分片上 ——
    两处叠起来只有累加才对。"""
    usage: dict = {}

    absorb_usage(usage, _chunk(input_tokens=10, output_tokens=2, total_tokens=12))
    absorb_usage(usage, _chunk(input_tokens=5, output_tokens=7, total_tokens=12))

    assert usage == {
        "calls": 2,
        "input_tokens": 15,
        "output_tokens": 9,
        "total_tokens": 24,
    }


def test_a_chunk_without_usage_is_ignored() -> None:
    usage: dict = {}

    absorb_usage(usage, SimpleNamespace(content="没有 usage"))
    absorb_usage(usage, _chunk())
    absorb_usage(None, _chunk(input_tokens=1))

    assert usage == {}, "拿不到就什么都不记，别编"


def test_totals_fall_back_to_input_plus_output() -> None:
    """有的网关只回 in/out、不回 total。"""
    assert usage_totals({"calls": 1, "input_tokens": 3, "output_tokens": 4}) == {
        "calls": 1,
        "input_tokens": 3,
        "output_tokens": 4,
        "total_tokens": 7,
    }


def test_nothing_recorded_means_no_row() -> None:
    """**一次都没拿到就返回 None** —— 落库时字段留空，而不是写一排 0
    （0 和「不知道」在统计里是两回事）。"""
    assert usage_totals(None) is None
    assert usage_totals({}) is None
    assert usage_totals({"input_tokens": 3}) is None, "没有 calls 就等于没记到"
    assert usage_pair({}) is None
    assert usage_pair({"calls": 1, "input_tokens": 3, "output_tokens": 4}) == (3, 4)
