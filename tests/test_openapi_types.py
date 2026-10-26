"""前端 API 类型必须由 OpenAPI 生成，不能手改出漂移。"""

from __future__ import annotations

from scripts.generate_openapi_types import TARGET, build_types


def test_generated_openapi_types_are_current() -> None:
    assert TARGET.read_text(encoding="utf-8") == build_types()
