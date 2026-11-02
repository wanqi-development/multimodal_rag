"""按 YAML 文件发现并编译子 Agent。

每个 ``*.yaml`` 是一个完整 Agent 定义。这里只读取静态能力声明，真正的工具和中间件
仍由代码注册表按名称解析 —— 配置文件不能 import 任意 Python。
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from rag.agent.contracts import EvidenceResult
from rag.config.schema import Settings

logger = logging.getLogger("rag.agent.registry")

_NAME = re.compile(r"^[a-z][a-z0-9-]*$")


def _middleware_registry() -> dict[str, Any]:
    """YAML 只能写符号 ID；这里决定部署代码真正允许哪些中间件。"""
    from rag.agent.middleware import ToolErrorMiddleware

    return {"tool-error": ToolErrorMiddleware}


class AgentDefinition(BaseModel):
    """一个子 Agent 的静态配置。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    name: str
    description: str = Field(min_length=1)
    enabled: bool = True
    system_prompt: str = Field(min_length=1)
    tools: list[str] = Field(default_factory=list)
    middleware: list[str] = Field(default_factory=list)
    model: Literal["inherit"] = "inherit"
    max_tool_calls: int = Field(6, ge=1)
    requires_channels: list[str] = Field(default_factory=list)
    response_schema: Literal["EvidenceResult"] = "EvidenceResult"
    source: Path

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not _NAME.fullmatch(value):
            raise ValueError("name 只能使用小写字母、数字和连字符，且必须以字母开头")
        return value

    @field_validator("tools", "middleware", "requires_channels")
    @classmethod
    def _unique_names(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("列表里不能有重复项")
        return value


@dataclass(frozen=True)
class AgentRegistry:
    """启动时加载的一份不可变 Agent 清单。"""

    definitions: tuple[AgentDefinition, ...]
    fingerprint: str

    def enabled(self) -> tuple[AgentDefinition, ...]:
        return tuple(item for item in self.definitions if item.enabled)

    def prompt_catalog(self, definitions: Iterable[AgentDefinition] | None = None) -> str:
        rows = list(definitions if definitions is not None else self.enabled())
        if not rows:
            return "当前没有可用的专业子 Agent，只能使用主 Agent 的直接工具。"
        return "\n".join(f"- `{item.name}`：{item.description}" for item in rows)

    def build_subagents(
        self,
        settings: Settings,
        *,
        model: Any,
        tools: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
        """把可用文件配置编译成 deepagents 原生 ``SubAgent``。

        缺少通道或工具的可选 Agent 不加载；已启用文件的工具名写错则立即失败，避免
        “看起来加载了、运行时才发现没有能力”。
        """
        channel_names = {item.name for item in settings.retrieval.channels if item.enabled}
        middleware_specs = _middleware_registry()
        from rag.agent.middleware import ToolCallBudgetMiddleware, ToolPolicy, ToolPolicyMiddleware

        specs: list[dict[str, Any]] = []
        names: list[str] = []
        for item in self.definitions:
            if not item.enabled:
                logger.info("子 Agent 已按配置禁用：%s", item.name)
                continue
            missing_channels = set(item.requires_channels) - channel_names
            if missing_channels:
                logger.info(
                    "子 Agent %s 暂不加载：缺少通道 %s",
                    item.name,
                    ", ".join(sorted(missing_channels)),
                )
                continue
            missing_tools = set(item.tools) - set(tools)
            if missing_tools:
                raise ValueError(f"子 Agent {item.name} 引用了未注册工具：{sorted(missing_tools)}")
            missing_middleware = set(item.middleware) - set(middleware_specs)
            if missing_middleware:
                raise ValueError(
                    f"子 Agent {item.name} 引用了未注册中间件：{sorted(missing_middleware)}"
                )
            specs.append(
                {
                    "name": item.name,
                    "description": item.description,
                    "system_prompt": item.system_prompt,
                    "tools": [tools[name] for name in item.tools],
                    "middleware": [
                        ToolPolicyMiddleware(ToolPolicy.exact(frozenset(item.tools))),
                        ToolCallBudgetMiddleware(item.max_tool_calls),
                        *(middleware_specs[name]() for name in item.middleware),
                    ],
                    "model": model,
                    "mode": "isolated",
                    "response_format": EvidenceResult,
                }
            )
            names.append(item.name)
        return specs, tuple(names)


def _parse_file(path: Path) -> AgentDefinition:
    try:
        meta = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{path} 的 YAML 不合法：{exc}") from exc
    if not isinstance(meta, dict):
        raise ValueError(f"{path} 的顶层必须是对象")
    meta["source"] = path
    try:
        return AgentDefinition.model_validate(meta)
    except ValidationError as exc:
        raise ValueError(f"{path} 配置不合法：{exc}") from exc


def load_agent_registry(directory: str | Path) -> AgentRegistry:
    """加载目录下全部 ``*.yaml``；空目录也可启动，但不会有专业子 Agent。"""
    root = Path(directory)
    if root.is_dir():
        legacy = sorted(root.glob("*.agent.md"))
        if legacy:
            names = "、".join(path.name for path in legacy)
            raise ValueError(f"子 Agent 只支持 YAML，发现旧 Markdown 配置：{names}")
    files = sorted(root.glob("*.yaml")) if root.is_dir() else []
    definitions: list[AgentDefinition] = []
    middleware_names = set(_middleware_registry())
    seen: set[str] = set()
    digest = hashlib.sha256()
    for path in files:
        definition = _parse_file(path)
        missing_middleware = set(definition.middleware) - middleware_names
        if missing_middleware:
            raise ValueError(
                f"子 Agent {definition.name} 引用了未注册中间件：{sorted(missing_middleware)}"
            )
        if definition.name in seen:
            raise ValueError(f"子 Agent 名称重复：{definition.name}（{path}）")
        seen.add(definition.name)
        definitions.append(definition)
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return AgentRegistry(tuple(definitions), digest.hexdigest()[:12])
