"""embedding provider：三条后端的选择、Qwen3-Embedding 的查询前缀、多模态那条的图文融合。

不加载真模型（那是 1.2GB + 十几秒）—— 用替身验「我们交给 SentenceTransformer
的参数对不对」。真模型的端到端编码在 scripts 里手工验过。
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path
from typing import Any

import httpx
import pytest

from rag.config.schema import EmbeddingProviderConfig
from rag.providers.embedding import (
    EmbeddingError,
    FastEmbedDenseEmbedder,
    QwenVlMultimodalEmbedder,
    SentenceTransformerEmbedder,
)

QWEN_PROMPTS = {
    "query": "Instruct: Given a web search query, retrieve relevant passages\nQuery:",
    "document": "",
}


class _FakeModel:
    """替下 SentenceTransformer：记下参数，吐固定维度的零向量。"""

    def __init__(self, *, dim: int = 1024, prompts: dict[str, str] | None = None) -> None:
        self.prompts = prompts if prompts is not None else dict(QWEN_PROMPTS)
        self.calls: list[dict[str, Any]] = []
        self._dim = dim

    def encode(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        self.calls.append({"texts": list(texts), **kwargs})
        return [[0.0] * self._dim for _ in texts]


def _install(monkeypatch: pytest.MonkeyPatch, model: _FakeModel) -> _FakeModel:
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = lambda _name: model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return model


def _cfg(**over: Any) -> EmbeddingProviderConfig:
    base: dict[str, Any] = {
        "backend": "sentence-transformers",
        "model": ".models/Qwen3-Embedding-0.6B",
        "dim": 1024,
    }
    return EmbeddingProviderConfig(**{**base, **over})


def test_documents_and_queries_use_different_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Qwen3-Embedding 是非对称的：query 要带 task instruction，文档不带。

    把两边当成一样会白丢召回 —— 而且不报错，只是效果差，最难发现的那种。
    """
    model = _install(monkeypatch, _FakeModel())
    emb = SentenceTransformerEmbedder(_cfg())

    import asyncio

    asyncio.run(emb.embed_documents(["文档一"]))
    asyncio.run(emb.embed_query("问题"))

    # calls[0] 是加载时的维度自检，真正的编码从 -2 / -1 看
    assert model.calls[-2]["prompt_name"] == "document"
    assert model.calls[-1]["prompt_name"] == "query"
    assert model.calls[-2]["texts"] == ["文档一"]
    assert model.calls[-1]["texts"] == ["问题"]


def test_vectors_are_normalized_before_hitting_milvus(monkeypatch: pytest.MonkeyPatch) -> None:
    """索引是 COSINE；归一化是 Qwen3-Embedding 的用法要求。"""
    model = _install(monkeypatch, _FakeModel())

    import asyncio

    asyncio.run(SentenceTransformerEmbedder(_cfg()).embed_documents(["x"]))

    assert model.calls[-1]["normalize_embeddings"] is True


def test_a_model_without_prompts_is_not_given_a_prompt_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """换个不自带 prompt 的模型也要能跑，不该因为取不到名字就崩。"""
    model = _install(monkeypatch, _FakeModel(prompts={}))

    import asyncio

    asyncio.run(SentenceTransformerEmbedder(_cfg()).embed_query("x"))

    assert "prompt_name" not in model.calls[-1]


def test_dim_mismatch_is_rejected_before_anything_is_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """配置维度与模型实际维度不符时当场报错 —— 等到灌完库才发现就晚了。"""
    _install(monkeypatch, _FakeModel(dim=8))

    import asyncio

    with pytest.raises(ValueError, match="维度不匹配"):
        asyncio.run(SentenceTransformerEmbedder(_cfg()).embed_documents(["x"]))


def test_batch_size_comes_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _install(monkeypatch, _FakeModel())

    import asyncio

    asyncio.run(SentenceTransformerEmbedder(_cfg(batch_size=7)).embed_documents(["x"]))

    assert model.calls[-1]["batch_size"] == 7


def test_both_backends_are_registered() -> None:
    """registry 里两条路都在，切后端只改配置。"""
    from rag.contracts import EMBEDDERS

    assert "fastembed" in EMBEDDERS
    assert "sentence-transformers" in EMBEDDERS
    assert "qwen3-vl" in EMBEDDERS
    assert FastEmbedDenseEmbedder(EmbeddingProviderConfig()).dim == 1024


# ---- 多模态那条路（qwen3-vl-embedding）：图与文进同一个向量空间 ----


def _vl_cfg(**over: Any) -> EmbeddingProviderConfig:
    base: dict[str, Any] = {
        "backend": "qwen3-vl",
        "model": "qwen3-vl-embedding",
        "dim": 2560,
        "endpoint": "https://example.test/multimodal-embedding",
    }
    return EmbeddingProviderConfig(**{**base, **over})


def _serve(embedder: QwenVlMultimodalEmbedder, dim: int, *, reverse: bool = False) -> list[Any]:
    """把 embedder 的 HTTP 客户端换成假服务，返回它收到的请求列表。

    假服务给第 i 条内容吐 ``[i, 0, 0, …]`` —— 第 0 维就是下标，正好用来验「按 index 对序」。
    """
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        contents = json.loads(request.content)["input"]["contents"]
        items = [
            {"index": i, "embedding": [float(i)] + [0.0] * (dim - 1)} for i in range(len(contents))
        ]
        if reverse:
            # 真实服务不保证返回顺序 —— 对序必须靠响应里的 index，不能靠列表顺序
            items.reverse()
        return httpx.Response(200, json={"output": {"embeddings": items}})

    embedder._make_client = lambda: httpx.AsyncClient(  # type: ignore[method-assign]
        transport=httpx.MockTransport(handle)
    )
    return seen


def test_image_and_text_are_sent_as_one_content_item(tmp_path: Path) -> None:
    """图片块要「图 + 描述」一起送 —— 服务端融合成一个向量，这才有以图搜图。"""
    image = tmp_path / "p0001_01.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n fake")
    embedder = QwenVlMultimodalEmbedder(_vl_cfg(), "sk-test")
    seen = _serve(embedder, 2560)

    asyncio.run(embedder.embed_documents(["正文块", "架构图：上游是 Kafka"], [[], [image]]))

    contents = json.loads(seen[0].content)["input"]["contents"]
    assert contents[0] == {"text": "正文块"}  # 文字块只有文字，不带图
    assert contents[1]["text"] == "架构图：上游是 Kafka"
    assert contents[1]["image"].startswith("data:image/png;base64,")


def test_vectors_are_ordered_by_the_index_field() -> None:
    """响应里的 ``index`` 是唯一可信的顺序 —— 直接按列表顺序取会错位到别的块上。"""
    embedder = QwenVlMultimodalEmbedder(_vl_cfg(), "sk-test")
    _serve(embedder, 2560, reverse=True)

    vectors = asyncio.run(embedder.embed_documents(["甲", "乙", "丙"])).dense

    assert [vec[0] for vec in vectors] == [0.0, 1.0, 2.0]


def test_image_marker_is_stripped_before_encoding() -> None:
    """``![image](asset:…)`` 是记账用的，不是内容：留着会把 asset 这类词喂进向量。"""
    embedder = QwenVlMultimodalEmbedder(_vl_cfg(), "sk-test")
    seen = _serve(embedder, 2560)

    asyncio.run(embedder.embed_documents(["描述\n![image](asset:doc/p0001_01.png)"]))

    text = json.loads(seen[0].content)["input"]["contents"][0]["text"]
    assert text == "描述"


def test_dim_mismatch_is_caught_at_the_first_call() -> None:
    """忘了把 dim 改成 2560 时立刻报错 —— 等 Milvus 报要等向量都花钱编完。"""
    embedder = QwenVlMultimodalEmbedder(_vl_cfg(), "sk-test")
    _serve(embedder, 1024)

    with pytest.raises(EmbeddingError, match="2560"):
        asyncio.run(embedder.embed_documents(["x"]))


def test_query_goes_through_the_same_model() -> None:
    """查询与文档必须来自**同一个**向量空间，所以查询也走这个端点。"""
    embedder = QwenVlMultimodalEmbedder(_vl_cfg(), "sk-test")
    seen = _serve(embedder, 2560)

    vector = asyncio.run(embedder.embed_query("怎么配 Kafka")).dense

    assert json.loads(seen[0].content)["input"]["contents"] == [{"text": "怎么配 Kafka"}]
    assert len(vector[0]) == 2560


def test_an_image_query_is_sent_as_one_fused_content_item(tmp_path: Path) -> None:
    """以图搜图：图和词进同一个内容项 —— 与文档侧的图片块完全对称，所以两边可比。"""
    image = tmp_path / "photo.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n fake")
    embedder = QwenVlMultimodalEmbedder(_vl_cfg(), "sk-test")
    seen = _serve(embedder, 2560)

    asyncio.run(embedder.embed_query("找类似的架构图", image))

    content = json.loads(seen[0].content)["input"]["contents"][0]
    assert content["text"] == "找类似的架构图"
    assert content["image"].startswith("data:image/png;base64,")


def test_a_pure_image_query_sends_only_the_image(tmp_path: Path) -> None:
    """只有图没有词时**不带空的 text**：实测过的形态就是 `{"image": …}`。"""
    image = tmp_path / "photo.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n fake")
    embedder = QwenVlMultimodalEmbedder(_vl_cfg(), "sk-test")
    seen = _serve(embedder, 2560)

    asyncio.run(embedder.embed_query("", image))

    content = json.loads(seen[0].content)["input"]["contents"][0]
    assert set(content) == {"image"}


def test_a_text_only_backend_refuses_an_image_query() -> None:
    """纯文本 backend 拿到查询图必须报错：静默只编文字 = 「以图搜图」悄悄变成「空文本搜」。"""
    with pytest.raises(ValueError, match="不支持图文一起编码"):
        asyncio.run(
            FastEmbedDenseEmbedder(_cfg(backend="fastembed")).embed_query(
                "找类似的图", Path("photo.png")
            )
        )


def test_two_separate_event_loops_both_work() -> None:
    """httpx 的 AsyncClient 绑创建它的那个事件循环 —— 客户端绝不能跨 ``asyncio.run`` 缓存。

    实测撞到的：切分阶段的语义兜底自己起循环（每段残留各一次），灌库那一步再起一个，
    缓存的客户端到第二次调用就报 "Event loop is closed"。
    """
    embedder = QwenVlMultimodalEmbedder(_vl_cfg(), "sk-test")
    _serve(embedder, 2560)

    first = asyncio.run(embedder.embed_documents(["甲"])).dense
    second = asyncio.run(embedder.embed_documents(["乙"])).dense

    assert len(first[0]) == 2560
    assert len(second[0]) == 2560


def test_the_server_side_batch_limit_is_respected() -> None:
    """实测服务端硬上限 20 —— 配置里写 32（本地那条路的默认）也不能超发。

    真机上这一条是灌库那一步撞出来的：整批 32 条直接 400
    `batch size is invalid, it should not be larger than 20`，一个块都没写进去。
    """
    embedder = QwenVlMultimodalEmbedder(_vl_cfg(batch_size=32), "sk-test")
    seen = _serve(embedder, 2560)

    asyncio.run(embedder.embed_documents([f"文{i}" for i in range(45)]))

    sizes = [len(json.loads(request.content)["input"]["contents"]) for request in seen]
    assert sizes == [20, 20, 5]


def test_a_batch_carrying_images_is_capped_at_ten(tmp_path: Path) -> None:
    """含图的批上限更低（10）—— 也是真机撞出来的：`image batch size can should be [1, 10]`。

    正文块占绝大多数，所以只把**含图的那几批**压到 10，别拖累纯文本的吞吐。
    """
    image = tmp_path / "p0001_01.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n fake")
    embedder = QwenVlMultimodalEmbedder(_vl_cfg(batch_size=32), "sk-test")
    seen = _serve(embedder, 2560)
    images = [[image] if index % 5 == 0 else [] for index in range(25)]

    asyncio.run(embedder.embed_documents([f"文{index}" for index in range(25)], images))

    batched = [json.loads(request.content)["input"]["contents"] for request in seen]
    assert [len(batch) for batch in batched] == [10, 10, 5]
    for batch in batched:
        if any("image" in item for item in batch):
            assert len(batch) <= 10


def test_text_only_backend_refuses_images() -> None:
    """配置写成多模态、embedding 却是纯文本的 —— 必须报错，不能静默丢图。"""
    with pytest.raises(ValueError, match="不支持图文一起编码"):
        asyncio.run(
            FastEmbedDenseEmbedder(_cfg(backend="fastembed")).embed_documents(
                ["x"], [[Path("a.png")]]
            )
        )


def test_build_embedder_needs_the_key_for_the_online_backend() -> None:
    """线上那条要 key；缺了要说清缺哪个，而不是等远端 401。"""
    from rag.config.schema import Settings
    from rag.providers import build_embedder

    settings = Settings()
    settings.providers.embedding = _vl_cfg()

    class _NoKey:
        dashscope_api_key = None

    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
        build_embedder(settings, _NoKey())  # type: ignore[arg-type]

    class _Key:
        dashscope_api_key = "sk-test"

    embedder = build_embedder(settings, _Key())  # type: ignore[arg-type]
    assert isinstance(embedder, QwenVlMultimodalEmbedder)
    assert embedder.supports_images is True


def test_an_unknown_backend_says_what_is_available() -> None:
    """backend 名写错要给一句能照着改的话，不是 `None(cfg)` 的 TypeError。"""
    from rag.config.schema import Settings
    from rag.providers import build_embedder

    settings = Settings()
    settings.providers.embedding.backend = "qwen3v1"  # 打错一个字符的典型

    with pytest.raises(KeyError, match="已注册"):
        build_embedder(settings)
