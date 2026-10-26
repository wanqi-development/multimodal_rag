"""token 用量累加。**一个地方定义，两处用**（`providers/generator.py` 与 `agent/build.py`）。

为什么是累加而不是赋值：一次模型调用的 `usage_metadata` 通常只挂在**最后一块**
流式分片上；而 agent 一轮里会有好几次模型调用（主 agent + 各子 agent）。
两处叠起来，只有累加才对。

拿不到用量时**什么都不做** —— 不是所有网关都回 usage。宁可这一轮没有数字，
也不要拿字符数瞎估一个填进去：「token 消耗统计」的价值全在准。
"""

from __future__ import annotations

from typing import Any

# 这三个就是我们要的账
_FIELDS = ("input_tokens", "output_tokens", "total_tokens")


def absorb_usage(usage: dict[str, Any] | None, message: Any) -> None:
    """把一条消息上的 `usage_metadata` 累加进 `usage`。`usage` 是调用方给的容器。"""
    if usage is None:
        return
    meta = getattr(message, "usage_metadata", None) or {}
    if not meta:
        return
    usage["calls"] = int(usage.get("calls") or 0) + 1
    for key in _FIELDS:
        value = meta.get(key)
        if value is not None:
            usage[key] = int(usage.get(key) or 0) + int(value)


def usage_totals(usage: dict[str, Any] | None) -> dict[str, Any] | None:
    """收成一个能直接落库的字典。**一次都没拿到就返回 None**（落库时字段留空）。"""
    if not usage or not usage.get("calls"):
        return None
    total = int(usage.get("total_tokens") or 0)
    if not total:
        total = int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
    return {
        "calls": int(usage.get("calls") or 1),
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "total_tokens": total,
    }


def usage_pair(usage: dict[str, Any] | None) -> tuple[int | None, int | None] | None:
    """`(input, output)` 两个数的形状（agent 日志行里的 `token=in/out`），拿不到就是 None。

    **这不是统计口径**：要给统计用的 token 账在会话级（`persistence/chat.py` 的
    `token_usage`），日志行上的 token 只是过程信息。
    """
    totals = usage_totals(usage)
    return None if totals is None else (totals["input_tokens"], totals["output_tokens"])
