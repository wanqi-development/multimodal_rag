"""CLI 共用运行时：Milvus 探活、错误翻译与异步执行。"""

from __future__ import annotations

import asyncio

import typer


async def _probe_milvus(index, settings) -> None:
    """开工前探一次活；接不住就一行话 + 退出 1。"""
    timeout = settings.indexing.vector.probe_timeout_s
    uri = settings.indexing.vector.uri
    try:
        await index.ping(timeout_s=timeout)
    except TimeoutError:
        typer.secho(f"Milvus 探活超时（{timeout:g} 秒没响应）：{uri}", fg=typer.colors.RED)
        typer.secho(
            "  先确认服务起来了（虚拟机上的 etcd + MinIO + milvus），或检查 .env 里的 MILVUS_URI；"
            "机器慢就把 indexing.vector.probe_timeout_s 调大。",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=1) from None
    except Exception as exc:  # noqa: BLE001 —— 连不上交给 _explain 翻成人话，别的原样抛
        _explain(exc, settings)


def _explain(exc: Exception, settings) -> None:
    """把「连不上 Milvus」说成人话，而不是甩一屏 pymilvus 的调用栈。"""
    from pymilvus.exceptions import MilvusException

    if not isinstance(exc, MilvusException) or "Fail connecting" not in str(exc):
        raise exc
    uri = settings.indexing.vector.uri
    typer.secho(f"Milvus 连不上：{uri}", fg=typer.colors.RED)
    typer.secho(
        "  先确认服务起来了（虚拟机上那套 etcd + MinIO + milvus），或检查 .env 里的 MILVUS_URI。",
        fg=typer.colors.YELLOW,
    )
    raise typer.Exit(code=1) from None


def _run_async(coro, settings):
    """跑协程，并把「连不上 Milvus」翻译成人话（其它异常原样抛出）。"""
    try:
        return asyncio.run(coro)
    except Exception as exc:  # noqa: BLE001 —— 不匹配的会由 _explain 原样抛出
        _explain(exc, settings)
