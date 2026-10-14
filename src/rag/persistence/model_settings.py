"""模型选型的**存储与审计**（2026-10-09 第②步）。

使用者定死的四条口径（见 `docs/backlog.md` 第 2 条）：

1. **不是"必须 PG"，是"必须有数据库"**：单机版 SQLite（`data/model_settings.sqlite3`，
   WAL + 短事务），多 worker 版 PG；`models.backend: auto|sqlite|postgres`，
   `auto` = 连得上 PG 就用 PG，否则 SQLite。
2. **审计要**：每次变更落一行 `model_setting_change`（谁 / 何时 / 旧值→新值 / request_id）。
3. **优先级**：库 > `.env` > yaml 默认（取值来源由调用方负责报出来，这里只存库那份）。
4. **这一版不热切**：写完**重启生效**（界面上也会这么写）。热切与跨进程广播是遗留项。

三处刻意的选择：

* **同步接口**（`sqlite3` 与 psycopg 的同步连接）：写配置是极低频的操作，调用方是 CLI 与
  启动期 —— 为它引一套 async 池不划算，也与 `LogStore.write_sync` 同一个路子。
* **时间戳存 ISO 字符串、params 存 JSON 文本**：两种库都能直接放，DDL 不必分家太多；
  真要做时间范围查询时再按库换成 `timestamptz`（现在没这个需求）。
* **`slot` 校验在写入侧**（必须在 `MODEL_SLOTS` 里）：槽位名是库里的主键与界面上的键，
  写错一个就是"界面上永远看不到的那一行"。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag.config.schema import MODEL_SLOTS

# SQLite 的 DDL（INTEGER PRIMARY KEY 自增）
_SQLITE_SCHEMA = (
    """
    create table if not exists model_setting (
        slot        text primary key,
        provider_key text not null default '',
        model       text not null default '',
        params      text not null default '{}',
        version     integer not null default 1,
        updated_at  text not null,
        updated_by  text not null default ''
    )
    """,
    """
    create table if not exists model_setting_change (
        id         integer primary key autoincrement,
        slot       text not null,
        old_value  text not null default '',
        new_value  text not null default '',
        actor      text not null default '',
        at         text not null,
        request_id text not null default ''
    )
    """,
)

# PG 的 DDL（bigserial + 与 SQLite 同名同列，换库不用改代码）
_PG_SCHEMA = (
    """
    create table if not exists model_setting (
        slot        text primary key,
        provider_key text not null default '',
        model       text not null default '',
        params      text not null default '{}',
        version     integer not null default 1,
        updated_at  text not null,
        updated_by  text not null default ''
    )
    """,
    """
    create table if not exists model_setting_change (
        id         bigserial primary key,
        slot       text not null,
        old_value  text not null default '',
        new_value  text not null default '',
        actor      text not null default '',
        at         text not null,
        request_id text not null default ''
    )
    """,
)

DEFAULT_SQLITE_PATH = "./data/model_settings.sqlite3"

# **供应商表：key 与模型、base_url 存在一起**（使用者 2026-10-09 定）。
# 全是文本列，所以 SQLite 与 PG 共用同一段 DDL。
_PROVIDER_SCHEMA = (
    """
    create table if not exists provider_setting (
        provider_key        text primary key,
        api_key_cipher      text not null default '',
        api_key_fingerprint text not null default '',
        base_url            text not null default '',
        models              text not null default '[]',
        updated_at          text not null default '',
        updated_by          text not null default ''
    )
    """,
    # 主密钥的自举位：部署时由运维写入（应用**只读**）。配置里先给 RAG_SECRET_KEY 则优先用它。
    """
    create table if not exists app_secret (
        name       text primary key,
        value      text not null,
        updated_at text not null default ''
    )
    """,
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _check_slot(slot: str) -> str:
    name = slot.strip()
    if name not in MODEL_SLOTS:
        raise ValueError(f"未知的模型槽位 {slot!r}；可选：{list(MODEL_SLOTS)}")
    return name


def _row_to_dict(row: Any, columns: list[str]) -> dict[str, Any]:
    item = dict(zip(columns, row, strict=False))
    try:
        item["params"] = json.loads(item.get("params") or "{}")
    except ValueError:
        item["params"] = {}
    return item


class SqliteModelSettingStore:
    """单机版：一个文件、WAL、短事务。**多进程读没问题**（CLI 与 serve 可以同时开着）。"""

    backend = "sqlite"
    # 占位符：SQLite 用 `?`、PG 用 `%s` —— 下面的提供商方法两种库共用一份代码
    placeholder = "?"

    def __init__(self, path: str | Path = DEFAULT_SQLITE_PATH) -> None:
        self.path = Path(path)
        self.name = f"sqlite:{self.path}"

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.execute("pragma journal_mode=wal")
        return conn

    def setup(self) -> None:
        with self._connect() as conn:
            for statement in (*_SQLITE_SCHEMA, *_PROVIDER_SCHEMA):
                conn.execute(statement)

    def all(self) -> dict[str, dict[str, Any]]:
        columns = ["slot", "provider_key", "model", "params", "version", "updated_at", "updated_by"]
        try:
            with self._connect() as conn:
                rows = conn.execute(f"select {', '.join(columns)} from model_setting").fetchall()
        except sqlite3.OperationalError:
            # **表还没建**（没跑 `rag models init`）不是错误：语义是"一个槽位都没设过"，
            # 于是全部沿用 .env / yaml。这里 500 掉会让设置页在没初始化的机器上打不开。
            return {}
        return {row[0]: _row_to_dict(row, columns) for row in rows}

    def set(
        self,
        slot: str,
        *,
        provider_key: str = "",
        model: str = "",
        params: dict[str, Any] | None = None,
        actor: str = "",
        request_id: str = "",
    ) -> dict[str, Any]:
        name = _check_slot(slot)
        payload = json.dumps(params or {}, ensure_ascii=False, sort_keys=True)
        at = _now()
        with self._connect() as conn:
            old = conn.execute(
                "select provider_key, model, params, version from model_setting where slot = ?",
                (name,),
            ).fetchone()
            version = int(old[3]) + 1 if old else 1
            old_value = (
                json.dumps(
                    {"provider_key": old[0], "model": old[1], "params": old[2]}, ensure_ascii=False
                )
                if old
                else ""
            )
            conn.execute(
                "insert into model_setting"
                " (slot, provider_key, model, params, version, updated_at, updated_by)"
                " values (?, ?, ?, ?, ?, ?, ?)"
                " on conflict(slot) do update set"
                " provider_key=excluded.provider_key, model=excluded.model,"
                " params=excluded.params, version=excluded.version,"
                " updated_at=excluded.updated_at, updated_by=excluded.updated_by",
                (name, provider_key, model, payload, version, at, actor),
            )
            conn.execute(
                "insert into model_setting_change"
                " (slot, old_value, new_value, actor, at, request_id) values (?, ?, ?, ?, ?, ?)",
                (
                    name,
                    old_value,
                    json.dumps(
                        {"provider_key": provider_key, "model": model, "params": payload},
                        ensure_ascii=False,
                    ),
                    actor,
                    at,
                    request_id,
                ),
            )
        return {"slot": name, "version": version, "updated_at": at, "old_value": old_value}

    def changes(self, *, slot: str = "", limit: int = 50) -> list[dict[str, Any]]:
        sql = "select slot, old_value, new_value, actor, at, request_id from model_setting_change"
        args: list[Any] = []
        if slot:
            sql += " where slot = ?"
            args.append(_check_slot(slot))
        sql += " order by id desc limit ?"
        args.append(max(1, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(args)).fetchall()
        return [
            _row_to_dict(row, ["slot", "old_value", "new_value", "actor", "at", "request_id"])
            for row in rows
        ]

    # ---- 供应商：key（密文）与模型清单、base_url 同表 ----

    _PROVIDER_COLUMNS = (
        "provider_key",
        "api_key_cipher",
        "api_key_fingerprint",
        "base_url",
        "models",
        "updated_at",
        "updated_by",
    )

    def secret(self, name: str) -> str:
        """读 `app_secret` 里的一行（主密钥的自举位；**应用只读**）。表没建就当没有。"""
        try:
            conn = self._connect()
            with conn:
                row = conn.execute(
                    f"select value from app_secret where name = {self.placeholder}", (name,)
                ).fetchone()
        except Exception:  # noqa: BLE001 —— 表没建 / 库还没起：都当"没配"
            return ""
        return str(row[0]) if row else ""

    def providers(self) -> dict[str, dict[str, Any]]:
        """每家一行。**不回密文**（页面只用 `has_key` 与指纹）。"""
        columns = self._PROVIDER_COLUMNS
        try:
            conn = self._connect()
            with conn:
                rows = conn.execute(f"select {', '.join(columns)} from provider_setting").fetchall()
        except Exception:  # noqa: BLE001 —— 表没建 = 一家都没配
            return {}
        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            item = dict(zip(columns, row, strict=False))
            try:
                item["models"] = json.loads(item.get("models") or "[]")
            except ValueError:
                item["models"] = []
            item["has_key"] = bool(item.pop("api_key_cipher", ""))
            out[str(item["provider_key"])] = item
        return out

    def set_provider(
        self,
        provider_key: str,
        *,
        base_url: str | None = None,
        models: list[dict[str, Any]] | None = None,
        api_key: str | None = None,
        master: bytes | None = None,
        actor: str = "",
        request_id: str = "",
    ) -> dict[str, Any]:
        """写一家供应商。

        `api_key` 的三种语义：**None = 不动 key**（只改 base_url / 模型清单）；
        **空串 = 清除 key**（回到部署配置那份）；**给了值 = 加密后存**（没主密钥就报错，
        绝不落明文）。审计只记**指纹**，不记密文、更不记明文。
        """
        from rag.persistence.secrets import SecretError, encrypt_key, fingerprint

        name = provider_key.strip()
        if not name:
            raise ValueError("provider_key 不能为空")
        # **原始行**（含密文）：`providers()` 故意不带密文，所以这里自己查一次。
        # 语义：`None` = 这一项不动、沿用库里那份；空串 = 显式清空。
        known: dict[str, Any] = {}
        try:
            conn = self._connect()
            with conn:
                row = conn.execute(
                    f"select api_key_cipher, api_key_fingerprint, base_url, models"
                    f" from provider_setting where provider_key = {self.placeholder}",
                    (name,),
                ).fetchone()
            if row:
                known = {
                    "api_key_cipher": row[0] or "",
                    "api_key_fingerprint": row[1] or "",
                    "base_url": row[2] or "",
                    "models": json.loads(row[3] or "[]"),
                }
        except Exception:  # noqa: BLE001 —— 表还没建 = 全新一家
            known = {}
        cipher = str(known.get("api_key_cipher") or "")
        mark = str(known.get("api_key_fingerprint") or "")
        if api_key == "":
            cipher, mark = "", ""
        if api_key:
            if master is None:
                raise SecretError(
                    "部署配置里没有 RAG_SECRET_KEY：页面不能收 API key"
                    "（要么在部署配置里放主密钥，要么只改 base_url / 模型清单）"
                )
            cipher = encrypt_key(api_key, master=master, provider_key=name)
            mark = fingerprint(api_key)
        at = _now()
        columns = self._PROVIDER_COLUMNS
        values = (
            name,
            cipher,
            mark,
            str(known.get("base_url") or "") if base_url is None else base_url,
            json.dumps(models or known.get("models") or [], ensure_ascii=False),
            at,
            actor,
        )
        placeholders = ", ".join([self.placeholder] * len(columns))
        updates = ", ".join(f"{column}=excluded.{column}" for column in columns[1:])
        conn = self._connect()
        with conn:
            conn.execute(
                f"insert into provider_setting ({', '.join(columns)}) values ({placeholders})"
                f" on conflict ({columns[0]}) do update set {updates}",
                values,
            )
            conn.execute(
                f"insert into model_setting_change"
                f" (slot, old_value, new_value, actor, at, request_id)"
                f" values ({', '.join([self.placeholder] * 6)})",
                (
                    f"provider:{name}",
                    f"key={known.get('api_key_fingerprint') or '-'}",
                    f"key={mark or '-'}",
                    actor,
                    at,
                    request_id,
                ),
            )
        return {"provider_key": name, "has_key": bool(cipher), "fingerprint": mark}


class PgModelSettingStore(SqliteModelSettingStore):
    """多 worker 版：与 chat / 日志表同库。SQL 与 SQLite 那份**几乎一样**，
    只有自增列类型与占位符不同（`%s`），所以这里只覆盖连接与 DDL，其余复用。"""

    backend = "postgres"
    placeholder = "%s"

    def __init__(self, conninfo: str) -> None:
        self.conninfo = conninfo
        self.name = "postgres:model_setting"

    def _connect(self) -> Any:  # pragma: no cover —— 需要真库才跑到，用例在 SQLite 那侧
        import psycopg

        return psycopg.connect(self.conninfo, autocommit=True)

    def setup(self) -> None:
        with self._connect() as conn, conn.cursor() as cursor:
            for statement in (*_PG_SCHEMA, *_PROVIDER_SCHEMA):
                cursor.execute(statement)

    def all(self) -> dict[str, dict[str, Any]]:
        columns = ["slot", "provider_key", "model", "params", "version", "updated_at", "updated_by"]
        from psycopg.errors import UndefinedTable

        try:
            with self._connect() as conn, conn.cursor() as cursor:
                cursor.execute(f"select {', '.join(columns)} from model_setting")
                rows = cursor.fetchall()
        except UndefinedTable:
            # **表还没建**（没跑 `rag models init`）不是错误 —— 语义是"一个槽位都没设过"，
            # 于是全部沿用 .env / yaml。这里 500 掉会让设置页/CLI 在没初始化的机器上用不了。
            return {}
        return {row[0]: _row_to_dict(row, columns) for row in rows}

    def set(self, slot: str, **kwargs: Any) -> dict[str, Any]:
        name = _check_slot(slot)
        params = kwargs.get("params") or {}
        payload = json.dumps(params, ensure_ascii=False, sort_keys=True)
        actor = kwargs.get("actor", "")
        request_id = kwargs.get("request_id", "")
        at = _now()
        with self._connect() as conn, conn.cursor() as cursor:
            cursor.execute(
                "select provider_key, model, params, version from model_setting where slot = %s",
                (name,),
            )
            old = cursor.fetchone()
            version = int(old[3]) + 1 if old else 1
            old_value = (
                json.dumps(
                    {"provider_key": old[0], "model": old[1], "params": old[2]},
                    ensure_ascii=False,
                )
                if old
                else ""
            )
            cursor.execute(
                "insert into model_setting"
                " (slot, provider_key, model, params, version, updated_at, updated_by)"
                " values (%s, %s, %s, %s, %s, %s, %s)"
                " on conflict (slot) do update set"
                " provider_key=excluded.provider_key, model=excluded.model,"
                " params=excluded.params, version=excluded.version,"
                " updated_at=excluded.updated_at, updated_by=excluded.updated_by",
                (
                    name,
                    kwargs.get("provider_key", ""),
                    kwargs.get("model", ""),
                    payload,
                    version,
                    at,
                    actor,
                ),
            )
            cursor.execute(
                "insert into model_setting_change"
                " (slot, old_value, new_value, actor, at, request_id)"
                " values (%s, %s, %s, %s, %s, %s)",
                (
                    name,
                    old_value,
                    json.dumps(
                        {
                            "provider_key": kwargs.get("provider_key", ""),
                            "model": kwargs.get("model", ""),
                            "params": payload,
                        },
                        ensure_ascii=False,
                    ),
                    actor,
                    at,
                    request_id,
                ),
            )
        return {"slot": name, "version": version, "updated_at": at, "old_value": old_value}

    def changes(self, *, slot: str = "", limit: int = 50) -> list[dict[str, Any]]:
        sql = "select slot, old_value, new_value, actor, at, request_id from model_setting_change"
        args: list[Any] = []
        if slot:
            sql += " where slot = %s"
            args.append(_check_slot(slot))
        sql += " order by id desc limit %s"
        args.append(max(1, int(limit)))
        with self._connect() as conn, conn.cursor() as cursor:
            cursor.execute(sql, tuple(args))
            rows = cursor.fetchall()
        return [
            _row_to_dict(row, ["slot", "old_value", "new_value", "actor", "at", "request_id"])
            for row in rows
        ]


def build_model_setting_store(settings: Any, *, backend: str | None = None) -> Any:
    """按**选库开关**建（见 `rag/persistence/backend.py`）：`models.backend` 显式写了就听它，
    留 `auto`（默认）就往上问 `persistence.backend`（.env 的 `RAG_DB_BACKEND`）。

    **不主动探活**：`auto` 只看"配没配 url" —— 配置读取发生在启动期与 CLI，
    这里多一次网络探活只会让"没网也能起服务"这条变难；真连不上时错误发生在
    第一次读写，而那一步本来就有明确的报错。
    """
    from rag.persistence.backend import resolve_backend

    want = backend or getattr(settings.models, "backend", "auto")
    if resolve_backend(settings, want) == "postgres":
        from rag.persistence.pg import build_persistence

        return PgModelSettingStore(build_persistence(settings).conninfo())
    sqlite_path = getattr(settings.models, "sqlite_path", "") or DEFAULT_SQLITE_PATH
    return SqliteModelSettingStore(sqlite_path)
