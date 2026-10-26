"""「这一份关系型数据落哪个库」的**唯一判据**（2026-10-10 使用者定）。

以前是三处各判一套：`documents.backend`（文档注册表 + 上传件台账共用）、`models.backend`
（模型选型）、日志那份看 `observability.logging.postgres` —— 想整体从 PG 切回单机 SQLite
得改好几处。现在统一成一个开关 `persistence.backend`（base.yaml 里声明成
`${RAG_DB_BACKEND:-auto}`，**值放在 `.env`**）：

    pg / postgres —— 用 PostgreSQL（前提：配了 `RAG_PG_URL`）
    sqlite        —— 用单机文件（`data/documents.sqlite3` / `data/model_settings.sqlite3`）
    auto（默认）  —— 配了 `RAG_PG_URL` 就用 pg，否则 sqlite（**与从前逐字一致**）

优先级：**某一类的显式设置 > 统一开关 > auto**。

* `documents.backend` 写 `sqlite` / `postgres` 时听它（上传件台账与文档注册表共用这一个）；
* `models.backend` 同理；
* 两处留 `auto`（默认）就往上问 `persistence.backend`。

**不在这个开关里的**：LangGraph 的会话状态 / HITL 中断只有一个实现（PG），它由
`persistence.postgres.url` 决定 —— 留空就退回**进程内**（重启即丢）。要彻底单机，
把 `RAG_PG_URL` 也清掉。
"""

from __future__ import annotations

from typing import Any

SQL_BACKENDS = ("sqlite", "postgres")
# 别名：写 `pg` 也认（日常都这么叫）
_ALIASES = {"pg": "postgres", "postgresql": "postgres"}


def _normalize(value: str) -> str:
    key = (value or "auto").strip().lower()
    key = _ALIASES.get(key, key)
    if key not in (*SQL_BACKENDS, "auto"):
        raise ValueError(f"未知的数据库开关 {value!r}：可选 auto / sqlite / postgres（写 pg 也认）")
    return key


def resolve_backend(settings: Any, want: str | None = None) -> str:
    """定这一份用哪个库。返回 `"sqlite"` 或 `"postgres"` —— **不会再是 auto**。

    `want` 是**某一类自己的**设置（`documents.backend` / `models.backend`）；不传或写
    `auto` 就往上问统一开关。选了 postgres 却没配 url → 当场报错（与从前一致：
    不静默退回，免得你以为在用 PG、其实写进了本地文件）。
    """
    postgres = getattr(getattr(settings, "persistence", None), "postgres", None)
    has_url = bool(getattr(postgres, "url", ""))
    choice = _normalize(want or "auto")
    if choice == "auto":
        unified = getattr(getattr(settings, "persistence", None), "backend", "auto")
        choice = _normalize(unified or "auto")
    if choice == "auto":
        return "postgres" if has_url else "sqlite"
    if choice == "postgres" and not has_url:
        raise ValueError(
            "选了 PostgreSQL，但没配 `persistence.postgres.url`（.env 里的 RAG_PG_URL）："
            "要么把 url 填上，要么把开关改成 sqlite / auto。"
        )
    return choice
