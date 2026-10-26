"""配置加载。

两条来源，职责分明：

1. ``config/*.yaml`` —— 结构化项目配置，支持 ``${VAR:-default}`` 插值。
   先读 base.yaml，再用 ``RAG_PROFILE``（默认 dev）对应的文件深合并覆盖。
2. 环境变量 / ``.env`` —— 两类值都放这里，都不进配置文件：
   - 机密（API key / token）：由 pydantic-settings 读进 ``Secrets``；
   - 部署参数（MILVUS_URI / *_BASE_URL 等）：由 ``${VAR:-default}`` 插值读走。
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag import ROOT_DIR
from rag.config.schema import Settings

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class Secrets(BaseSettings):
    """机密。只从环境变量 / .env 读，不出现在 yaml 里。

    字段名 = ``<供应商>_API_KEY`` 环境变量名的小写（pydantic-settings 大小写不敏感）。
    web 搜索不单列 —— 用智谱的 web 搜索，复用 ``zhipuai_api_key``（见 docs/architecture.md §9）。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    deepseek_api_key: str | None = None
    openai_api_key: str | None = None
    dashscope_api_key: str | None = None
    # 看一眼 VLM 的 key。留空回退到 dashscope_api_key —— 同一个账号同一把 key。
    # 单列出来是为了「VL 换个账号/网关」时不用连 chat 的 key 一起换。
    dashscope_vl_api_key: str | None = None
    # 语音转文字（ASR）的 key。同上：留空回退到 dashscope_api_key
    dashscope_asr_api_key: str | None = None
    zhipuai_api_key: str | None = None
    moonshot_api_key: str | None = None
    milvus_token: str | None = None
    langsmith_api_key: str | None = None
    # 本地产线挂掉时的兜底：PaddleOCR 云端 API 的 access token。
    # 从 PaddleOCR 的 AI Studio 平台申请，配了才谈得上兜底（见 IngestCloudOcrConfig）。
    paddleocr_access_token: str | None = None
    # HuggingFace token。**只有 flashrank 那条路要它**：flashrank 自己下模型时既不带鉴权、
    # 用的又是个已经 404 的旧文件名（见 providers/rerank.py），所以由我们带 token 去下。
    # 放在 .env 里（跟其它机密一样），别写进 yaml。
    hf_token: str | None = None
    # API 服务的访问令牌（`X-API-Key`）。**留空则只允许 127.0.0.1** ——
    # 没配 key 就放行所有人等于裸奔，默认值必须是安全的那一侧。
    rag_api_key: str | None = None
    # Redis 密码（走 SSH 隧道那台）。URL 里也可以写，但单独一个变量更不容易被日志带出去。
    rag_redis_password: str | None = None
    # PostgreSQL 密码（LangGraph 的 checkpointer 与 store）。同上：URL 里不写密码。
    rag_pg_password: str | None = None
    # Neo4j 密码（知识图谱，M6）。地址与用户名在 yaml（非机密），密码只从 .env 来。
    neo4j_password: str | None = None
    # Firecrawl（**云端**网页抓取）的 key：`ingest.uploads.fetch.backend=firecrawl` 时才用。
    # 按量计费，所以默认那条路（http，本机抓）不碰它。地址不是机密，在 yaml 里。
    firecrawl_api_key: str | None = None
    # MinIO（多 worker 模式的文件落点）。地址与桶名在 yaml，**密钥只从 .env 来**。
    minio_access_key: str | None = None
    minio_secret_key: str | None = None


@lru_cache(maxsize=1)
def _dotenv() -> dict[str, str]:
    """``.env`` 的内容，作为插值兜底。

    pydantic-settings 只把 ``.env`` 读进 ``Secrets`` 模型，不会写进 ``os.environ``，
    而 yaml 里的 ``${VAR:-default}`` 需要看得见它 —— 所以这里单独读一份。
    """
    path = Path(".env")
    if not path.is_file():
        return {}
    return {k: v for k, v in dotenv_values(path).items() if v is not None}


def _expand_env(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        var, default = match.group(1), match.group(2)
        # 优先级：真实环境变量 > .env > yaml 里的默认值
        value = os.environ.get(var)
        if value is None:
            value = _dotenv().get(var)
        if value is not None:
            return value
        return default if default is not None else ""

    return _ENV_REF.sub(repl, text)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return yaml.safe_load(_expand_env(path.read_text(encoding="utf-8"))) or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(out.get(key), dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def config_dir() -> Path:
    return Path(os.environ.get("RAG_CONFIG_DIR") or ROOT_DIR / "config")


def load_settings(profile: str | None = None, directory: Path | None = None) -> Settings:
    directory = directory or config_dir()
    # 这里不能通过.env文件来配置 profile，因为还没加载.env文件， 需要通过环境变量来配置
    profile = profile or os.environ.get("RAG_PROFILE", "dev")
    data = _read_yaml(directory / "base.yaml")
    data = _deep_merge(data, _read_yaml(directory / f"{profile}.yaml"))
    settings = Settings.model_validate(data)
    from rag.deployment import validate_deployment

    validate_deployment(settings)
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


@lru_cache(maxsize=1)
def get_secrets() -> Secrets:
    return Secrets()
