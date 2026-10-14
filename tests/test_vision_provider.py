"""VLM 供应商：请求长什么样、失败怎么报、key 从哪来。

全部走 ``httpx.MockTransport``，不碰网络、不花钱。
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from rag.config.schema import ProvidersConfig, Settings, VisionProviderConfig
from rag.config.settings import Secrets
from rag.providers.vision import (
    VisionClient,
    VisionError,
    build_vision,
    clean_description,
    mime_of,
)

# 1x1 的 png 头，够判魔数就行
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)


@pytest.fixture(autouse=True)
def _empty_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Secrets 的 env_file 是相对 cwd 的 ".env"：留在项目根会把本机真实 key 读进来。"""
    monkeypatch.chdir(tmp_path)


def _client(
    handler: Any,
    *,
    model: str = "qwen-vl-max",
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
) -> VisionClient:
    cfg = VisionProviderConfig(model=model, base_url=base_url)
    client = VisionClient(cfg, "sk-test")
    return _install(client, handler)


def _install(client: VisionClient, handler: Any) -> VisionClient:
    """把 client 内部的 httpx.Client 换成走 MockTransport 的那个。

    ``_client`` 是 cached_property，它先查实例字典 —— 直接塞进去即可生效。
    连同 client 自己那把 key 一起带上，这样「key 从哪来」那几条用例才有意义。
    """
    client._client = httpx.Client(
        base_url=client.base_url,
        timeout=5,
        headers={"Authorization": f"Bearer {client._api_key}"},
        transport=httpx.MockTransport(handler),
    )
    return client


def _reply(text: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


# ---- 请求形态 ----


def test_request_carries_model_image_data_uri_and_context() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return _reply("一张架构图。")

    client = _client(handler)
    text = client.describe(
        _PNG, before="上边讲的是作业图。", after="下边讲的是并行度。", name="doc1/p0001_01.png"
    )

    assert text == "一张架构图。"
    assert seen["url"].endswith("/chat/completions")
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["model"] == "qwen-vl-max"
    system, user = seen["body"]["messages"]
    assert system["role"] == "system"
    blocks = user["content"]
    assert "上边讲的是作业图。" in blocks[0]["text"]
    assert "下边讲的是并行度。" in blocks[0]["text"]
    assert blocks[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert base64.b64decode(blocks[1]["image_url"]["url"].split(",", 1)[1]) == _PNG


def test_base_url_keeps_single_slash() -> None:
    """配置文件里带不带尾斜杠都该打到同一个路径上。"""
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return _reply("x")

    _client(handler, base_url="https://gw.internal/v1/").describe(_PNG)

    assert seen["url"] == "https://gw.internal/v1/chat/completions"


def test_content_blocks_are_joined() -> None:
    """有的网关把 content 回成块数组而不是字符串。"""
    client = _client(
        lambda request: httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "前半"},
                                {"type": "text", "text": "后半"},
                            ]
                        }
                    }
                ]
            },
        )
    )

    assert client.describe(_PNG) == "前半后半"


# ---- 失败要报得清楚 ----


def test_http_error_reports_the_server_message() -> None:
    """模型名写错/欠费都在 error.message 里，比状态码有用得多。"""
    client = _client(
        lambda request: httpx.Response(
            400, json={"error": {"message": "Model not exist.", "code": "InvalidParameter"}}
        )
    )

    with pytest.raises(VisionError, match="Model not exist."):
        client.describe(_PNG)


def test_empty_choices_is_an_error() -> None:
    """空描述写进库 = 「这张图没内容」，比报错难查得多。"""
    client = _client(lambda request: httpx.Response(200, json={"choices": []}))

    with pytest.raises(VisionError, match="choices"):
        client.describe(_PNG)


def test_empty_description_is_an_error() -> None:
    client = _client(lambda request: _reply("   "))

    with pytest.raises(VisionError, match="空描述"):
        client.describe(_PNG)


def test_non_json_response_is_an_error() -> None:
    client = _client(lambda request: httpx.Response(502, text="<html>bad gateway</html>"))

    with pytest.raises(VisionError, match="没返回 JSON"):
        client.describe(_PNG)


# ---- 图片类型 ----


def test_mime_is_sniffed_by_magic_bytes_then_suffix() -> None:
    assert mime_of(_PNG) == "image/png"
    assert mime_of(b"\xff\xd8\xff\xe0jpeg") == "image/jpeg"
    assert mime_of(b"II*\x00tiff") == "image/tiff"
    # 魔数不认识就按后缀；后缀也不认识就当 jpeg
    assert mime_of(b"who-knows", "a.bmp") == "image/bmp"
    assert mime_of(b"who-knows", "a") == "image/jpeg"


# ---- 装配：key 与地址 ----


def _settings(**vision: Any) -> Settings:
    return Settings(providers=ProvidersConfig(vision=VisionProviderConfig(**vision)))


def test_vl_key_falls_back_to_the_chat_key() -> None:
    """DashScope 的 key 是账号级的：同一个 qwen-plus 与 qwen-vl-max 共用一把。"""
    client = build_vision(_settings(), Secrets(dashscope_api_key="sk-chat"))

    assert client.model == "qwen-vl-max"
    assert client.fingerprint.startswith("qwen-vl-max@")


def test_vl_key_wins_over_the_chat_key() -> None:
    """单列 VL 变量是为了「VL 换个账号」时不用连 chat 的 key 一起换。"""
    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return _reply("x")

    client = build_vision(
        _settings(), Secrets(dashscope_api_key="sk-chat", dashscope_vl_api_key="sk-vl")
    )
    _install(client, handler).describe(_PNG)

    assert seen["auth"] == "Bearer sk-vl"


def test_missing_key_fails_loudly() -> None:
    with pytest.raises(RuntimeError, match="DASHSCOPE_VL_API_KEY"):
        build_vision(_settings(), Secrets())


def test_missing_base_url_fails_loudly() -> None:
    """VLM 走 OpenAI 兼容通道，没有地址就没法发请求 —— 别等运行到一半才发现。"""
    with pytest.raises(ValueError, match="base_url"):
        build_vision(_settings(base_url=""), Secrets(dashscope_api_key="k"))


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(KeyError, match="vision provider"):
        build_vision(_settings(provider="who"), Secrets(dashscope_api_key="k"))


# ---- 判断行 + 「结合上文」那一句的取舍（模型判断、代码排版）----


def test_the_decision_line_is_stripped_from_the_description() -> None:
    """判断行是说给我们看的，不是给检索用的 —— 留在正文里只会多几个无意义词。"""
    text = "相关\n\n架构图，画的是 Flink 的数据处理流程。\n\n结合上文：Flink 把批处理当作有界流。"

    assert clean_description(text) == (
        "架构图，画的是 Flink 的数据处理流程。\n\n结合上文：Flink 把批处理当作有界流。"
    )


def test_an_irrelevant_context_sentence_is_cut_off() -> None:
    """实测 qwen-vl-max 判「不相关」也还是会补一句 —— 那一句会把上文词带进库里。

    带进去就是**假的检索入口**：问 pandas 会命中一张 Flink 架构图。所以按标记切掉。
    """
    text = "不相关\n\n架构图，画的是 Flink 的数据处理流程。\n\n结合上文：本章讲用 pandas 读取 CSV。"

    assert clean_description(text) == "架构图，画的是 Flink 的数据处理流程。"


def test_a_description_without_a_decision_line_is_left_alone() -> None:
    """模型没按格式来时**原样保留**：宁可多留一句，也别按猜测去切内容。"""
    text = "这张图与上下文相关，画的是 Flink 的架构图。"

    assert clean_description(text) == text


def test_a_cut_that_would_empty_the_description_falls_back() -> None:
    """判断行后面只有「结合上文」时不能返回空串 —— 那会让整张图没有描述。

    两害相权：描述里多一句上文，比「这张图什么描述都没有」轻。
    """
    text = "不相关\n\n结合上文：本章讲用 pandas 读取 CSV。"

    assert clean_description(text) == "结合上文：本章讲用 pandas 读取 CSV。", "切没了就留着"


def test_describe_applies_the_cleanup() -> None:
    client = _client(lambda _request: _reply("相关\n\n架构图。\n\n结合上文：讲的是作业图。"))

    assert client.describe(_PNG, before="上边讲的是作业图。") == (
        "架构图。\n\n结合上文：讲的是作业图。"
    )
