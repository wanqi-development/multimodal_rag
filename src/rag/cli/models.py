"""模型选型命令：槽位读取、写入与审计。"""

from __future__ import annotations

import typer

from rag.config import load_settings

models_app = typer.Typer(
    help="模型选型（槽位 → 谁）：存库 + 审计。**这一版写入后重启生效**（热切是遗留项）",
    no_args_is_help=True,
)


@models_app.command("init")
def models_init() -> None:
    """建两张表（幂等）：`model_setting`（当前选型）与 `model_setting_change`（审计）。

    **这是一次写操作**，所以要显式触发。库的选法见 `models.backend`：
    `auto`（默认）= 配了 PG 用 PG，否则单机 SQLite（`models.sqlite_path`）。
    """
    from rag.persistence.model_settings import build_model_setting_store

    settings = load_settings()
    store = build_model_setting_store(settings)
    store.setup()
    typer.secho(
        f"模型选型表已就绪：{store.name}（backend={settings.models.backend}）",
        fg=typer.colors.GREEN,
    )


@models_app.command("list")
def models_list() -> None:
    """八个槽位 + 当前选的谁 + **这份值是哪层来的**。只读。"""
    from rag.config.schema import MODEL_SLOTS
    from rag.persistence.model_settings import build_model_setting_store

    settings = load_settings()
    stored = build_model_setting_store(settings).all()
    typer.echo(f"{'槽位':<14}{'来源':<10}{'provider_key':<16}{'model':<34}{'版本':>4}  更新时间")
    for slot in MODEL_SLOTS:
        row = stored.get(slot)
        if row is None:
            typer.echo(f"{slot:<14}{'未设置':<10}{'-':<16}{'（沿用 .env / yaml）':<34}{'-':>4}  -")
            continue
        typer.echo(
            f"{slot:<14}{'库':<10}{row['provider_key'] or '-':<16}"
            f"{(row['model'] or '-'):<34}{row['version']:>4}  {row['updated_at']}"
        )
    typer.echo("\n优先级：**库 > .env > yaml 默认**（库里没有的槽位就走原处那份配置）")


@models_app.command("set")
def models_set(
    slot: str = typer.Argument(..., help="槽位名，如 chat.agent / rerank"),
    provider_key: str = typer.Option(
        "", "--provider-key", help="chat 指 providers.chat 的键名；rerank 指 backend"
    ),
    model: str = typer.Option("", "--model", help="模型名或本地路径"),
    actor: str = typer.Option("", "--actor", help="谁改的（写进审计）"),
) -> None:
    """写一个槽位（写库 + **写一行审计**）。**重启服务才生效** —— 这一版不做热切。"""
    from rag.persistence.model_settings import build_model_setting_store

    settings = load_settings()
    store = build_model_setting_store(settings)
    result = store.set(slot, provider_key=provider_key, model=model, actor=actor or "cli")
    typer.secho(
        f"{result['slot']} → provider_key={provider_key or '-'} model={model or '-'}"
        f"（版本 {result['version']}，已记审计）",
        fg=typer.colors.GREEN,
    )
    typer.secho("重启 `rag serve` 后生效（这一版不做热切）。", fg=typer.colors.YELLOW)
