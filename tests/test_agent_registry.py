"""子 Agent 文件配置：解析、能力过滤与编译。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from rag.agent.contracts import EvidenceResult
from rag.agent.middleware import ToolCallBudgetMiddleware, ToolPolicyMiddleware
from rag.agent.registry import load_agent_registry
from rag.config.schema import Settings


def _agent(
    path: Path,
    name: str,
    *,
    tools: list[str] | None = None,
    middleware: list[str] | None = None,
    channels: list[str] | None = None,
):
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": name,
                "description": f"{name} 的能力",
                "enabled": True,
                "system_prompt": f"你是 {name}。",
                "tools": tools or [],
                "middleware": middleware or [],
                "model": "inherit",
                "max_tool_calls": 4,
                "response_schema": "EvidenceResult",
                "requires_channels": channels or [],
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_each_file_is_loaded_and_compiled_to_a_deepagents_subagent(tmp_path: Path) -> None:
    _agent(
        tmp_path / "docs.yaml",
        "docs",
        tools=["search_knowledge"],
        middleware=["tool-error"],
    )
    registry = load_agent_registry(tmp_path)
    tool = SimpleNamespace(name="search_knowledge")

    specs, names = registry.build_subagents(
        Settings(),
        model=object(),
        tools={"search_knowledge": tool},
    )

    assert names == ("docs",)
    assert specs[0]["name"] == "docs"
    assert specs[0]["tools"] == [tool]
    assert specs[0]["response_format"] is EvidenceResult
    assert specs[0]["system_prompt"] == "你是 docs。"
    assert specs[0]["mode"] == "isolated"
    assert len(specs[0]["middleware"]) == 3, "精确白名单 + 预算 + YAML 中间件"
    assert isinstance(specs[0]["middleware"][0], ToolPolicyMiddleware)
    assert isinstance(specs[0]["middleware"][1], ToolCallBudgetMiddleware)
    assert specs[0]["middleware"][1].max_calls == 4


def test_agent_is_skipped_when_its_optional_store_is_not_enabled(tmp_path: Path) -> None:
    _agent(
        tmp_path / "graph.yaml",
        "graph",
        tools=["search_relationships"],
        channels=["kg"],
    )
    registry = load_agent_registry(tmp_path)

    specs, names = registry.build_subagents(
        Settings(),
        model=object(),
        tools={"search_relationships": SimpleNamespace(name="search_relationships")},
    )

    assert specs == [] and names == ()


def test_enabled_agent_with_an_unknown_tool_fails_fast(tmp_path: Path) -> None:
    _agent(tmp_path / "business.yaml", "business", tools=["missing_tool"])
    registry = load_agent_registry(tmp_path)

    with pytest.raises(ValueError, match="missing_tool"):
        registry.build_subagents(Settings(), model=object(), tools={})


def test_duplicate_agent_names_are_rejected(tmp_path: Path) -> None:
    _agent(tmp_path / "a.yaml", "same", tools=[])
    _agent(tmp_path / "b.yaml", "same", tools=[])

    with pytest.raises(ValueError, match="名称重复"):
        load_agent_registry(tmp_path)


def test_unknown_middleware_fails_fast(tmp_path: Path) -> None:
    _agent(tmp_path / "docs.yaml", "docs", middleware=["not-registered"])

    with pytest.raises(ValueError, match="not-registered"):
        load_agent_registry(tmp_path)


def test_mode_field_is_rejected_because_isolated_is_fixed(tmp_path: Path) -> None:
    path = tmp_path / "docs.yaml"
    _agent(path, "docs")
    path.write_text(path.read_text("utf-8") + "\nmode: fork\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mode"):
        load_agent_registry(tmp_path)


def test_legacy_markdown_agent_files_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "docs.agent.md").write_text("old", encoding="utf-8")

    with pytest.raises(ValueError, match="只支持 YAML"):
        load_agent_registry(tmp_path)
