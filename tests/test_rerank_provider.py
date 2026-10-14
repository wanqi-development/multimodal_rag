"""重排 provider：两个 backend 的调用形态、分数写回、模型延迟加载。

不加载真模型（要下载）—— 用替身验「我们交给它什么、拿回来怎么排」。
"""

from __future__ import annotations

import asyncio
import os
import sys
import types
from pathlib import Path
from typing import Any

import httpx
import pytest

from rag.config.schema import RerankProviderConfig, Settings
from rag.contracts import Chunk, ScoredChunk
from rag.providers import build_reranker
from rag.providers.rerank import (
    CrossEncoderReranker,
    FlashrankReranker,
    RerankError,
    _ensure_flashrank_model,
)


def _hit(chunk_id: str, *, text: str | None = None) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(chunk_id=chunk_id, doc_id="d1", text=text or f"正文 {chunk_id}"),
        score=0.03,
        channel="rrf",
        rank=0,
        sources=["hybrid:milvus:c"],
    )


def _cfg(**over: Any) -> RerankProviderConfig:
    return RerankProviderConfig(**{"backend": "flashrank", "model": "m", **over})


def _cache(tmp_path: Path, model: str = "m") -> str:
    """造一个「模型已经下好了」的缓存目录 —— 否则 provider 会真去下载。

    flashrank 只认 `cache_dir/<模型名>/` 存在与否，里面的文件在替身路径下没人读。
    """
    (tmp_path / model).mkdir(parents=True, exist_ok=True)
    return str(tmp_path)


class _FakeFlashrank:
    """替下 flashrank.Ranker：按传入顺序倒着给分，方便验排序有没有生效。"""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.calls: list[dict[str, Any]] = []
        self.fail = False

    def rerank(self, request: Any) -> list[dict]:
        """签名**照抄真库**：收一个 RerankRequest，没有 top_n、返回全部候选。

        第一版替身照我们「以为」的签名（`rerank(query, passages, top_n=…)`）写，
        结果单测全绿、真机一跑就 `TypeError: unexpected keyword argument 'top_n'`。
        """
        if self.fail:
            raise RuntimeError("ONNX 跑不动")
        passages = list(request.passages)
        self.calls.append({"query": request.query, "passages": passages})
        # 越靠后的候选给越高分 → 结果应当是**倒序**，这样一眼看得出重排生效了
        return [{"id": item["id"], "score": float(index)} for index, item in enumerate(passages)]


class _FakeRerankRequest:
    def __init__(self, *, query: str, passages: list[dict[str, Any]]) -> None:
        self.query = query
        self.passages = passages


def _install_flashrank(monkeypatch: pytest.MonkeyPatch) -> tuple[_FakeFlashrank, Any]:
    """替下 flashrank 这个**包**（含 Config / Ranker 两个子模块）。

    真 flashrank 是包，我们的 `_mirror_url()` 会去改 `flashrank.Config.model_url` 与
    `flashrank.Ranker.model_url` —— 假模块也得有这两处，否则测出来的成功是假的。
    返回 (假 Ranker, 假 Ranker 子模块)，后者用来检查 URL 有没有被改到。
    """
    model = _FakeFlashrank()
    package = types.ModuleType("flashrank")
    package.__path__ = []  # type: ignore[attr-defined]  —— 让它像个包，子模块才 import 得到
    upstream = "https://huggingface.co/prithvida/flashrank/resolve/main/{}.zip"
    config = types.ModuleType("flashrank.Config")
    config.model_url = upstream  # type: ignore[attr-defined]
    ranker = types.ModuleType("flashrank.Ranker")
    ranker.model_url = upstream  # type: ignore[attr-defined]
    ranker.Ranker = lambda **kwargs: (setattr(model, "kwargs", kwargs), model)[1]  # type: ignore[attr-defined]
    ranker.RerankRequest = _FakeRerankRequest  # type: ignore[attr-defined]
    package.Ranker = ranker.Ranker  # type: ignore[attr-defined]
    package.RerankRequest = _FakeRerankRequest  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "flashrank", package)
    monkeypatch.setitem(sys.modules, "flashrank.Config", config)
    monkeypatch.setitem(sys.modules, "flashrank.Ranker", ranker)
    return model, ranker


class _FakeCrossEncoder:
    def __init__(self, model: str, **kwargs: Any) -> None:
        self.model = model
        self.kwargs = kwargs
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.pairs = list(pairs)
        # 越靠前的候选给越高分 → 顺序保持不变，用来验「没被无谓地打乱」
        return [float(len(pairs) - index) for index, _ in enumerate(pairs)]


def _install_cross_encoder(monkeypatch: pytest.MonkeyPatch) -> _FakeCrossEncoder:
    model = _FakeCrossEncoder("")
    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = lambda name, **kwargs: _FakeCrossEncoder(name, **kwargs)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return model


# ---- flashrank ----


def test_flashrank_gets_the_query_and_every_candidate_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ranker, _ = _install_flashrank(monkeypatch)
    cache = _cache(tmp_path)
    reranker = FlashrankReranker(_cfg(cache_dir=cache))

    result = asyncio.run(reranker.rerank("怎么配", [_hit("a"), _hit("b")], 2))

    call = ranker.calls[0]
    assert call["query"] == "怎么配"
    assert [item["text"] for item in call["passages"]] == ["正文 a", "正文 b"]
    assert ranker.kwargs == {"model_name": "m", "cache_dir": cache}
    # 替身把顺序倒过来：b 原先排第 2，重排后应当第 1
    assert [hit.chunk.chunk_id for hit in result] == ["b", "a"]


def test_rerank_scores_and_channel_replace_the_retrieval_ones(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """到了这一步，RRF 那点分数已经没有意义 —— 交下去的相关性分数，channel 也要改。"""
    _install_flashrank(monkeypatch)
    reranker = FlashrankReranker(_cfg(cache_dir=_cache(tmp_path)))

    result = asyncio.run(reranker.rerank("q", [_hit("a"), _hit("b")], 2))

    assert result[0].channel == "rerank"
    assert [hit.rank for hit in result] == [0, 1]
    assert result[0].sources == ["hybrid:milvus:c"], "来源要留着，重排不改召回信息"
    assert result[0].score != 0.03


def test_top_n_cuts_the_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_flashrank(monkeypatch)
    reranker = FlashrankReranker(_cfg(cache_dir=_cache(tmp_path)))

    result = asyncio.run(reranker.rerank("q", [_hit("a"), _hit("b"), _hit("c")], 2))

    assert len(result) == 2


def test_equal_scores_keep_the_fused_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """量化后的 ONNX 分数经常成片相等 —— 没有这条 tie-break 会变成随机顺序。"""

    class _Flat(_FakeFlashrank):
        def rerank(self, request):  # type: ignore[no-untyped-def]
            return [{"id": item["id"], "score": 1.0} for item in request.passages]

    monkeypatch.setattr(_FakeFlashrank, "rerank", _Flat.rerank)
    _install_flashrank(monkeypatch)

    reranker = FlashrankReranker(_cfg(cache_dir=_cache(tmp_path)))

    result = asyncio.run(reranker.rerank("q", [_hit("a"), _hit("b")], 2))

    assert [hit.chunk.chunk_id for hit in result] == ["a", "b"]


def test_an_empty_candidate_list_never_touches_the_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """没有候选就别去加载模型 —— 第一次调用要把模型下下来，空跑一次是白花钱白等。"""
    ranker, _ = _install_flashrank(monkeypatch)
    reranker = FlashrankReranker(_cfg(cache_dir=str(tmp_path / "空目录")))

    assert asyncio.run(reranker.rerank("q", [], 5)) == []
    assert ranker.calls == []


def test_a_model_failure_is_reported_as_rerank_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ranker, _ = _install_flashrank(monkeypatch)
    ranker.fail = True
    reranker = FlashrankReranker(_cfg(cache_dir=_cache(tmp_path)))

    with pytest.raises(RerankError, match="ONNX 跑不动"):
        asyncio.run(reranker.rerank("q", [_hit("a")], 1))


def test_the_hf_mirror_is_exported_before_the_model_is_loaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """直连 huggingface.co 不通 —— 加载前必须先把镜像与 Xet 开关落进环境变量。"""
    _install_flashrank(monkeypatch)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)

    ranker = FlashrankReranker(_cfg(hf_endpoint="https://mirror.test", cache_dir=_cache(tmp_path)))
    assert ranker._ranker is not None, "读 _ranker 就是触发加载那一下"

    assert os.environ["HF_ENDPOINT"] == "https://mirror.test"
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"


def test_a_missing_model_without_a_token_says_what_to_do(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """模型不在、又没有 token：要说清「去 .env 配 HF_TOKEN」，而不是抛一个 401。"""

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise httpx.ConnectError("没网")

    monkeypatch.setattr(httpx, "stream", _boom)

    with pytest.raises(RerankError, match="HF_TOKEN"):
        _ensure_flashrank_model(tmp_path, "m", "https://mirror.test", None)


def test_a_present_model_is_never_downloaded_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """模型目录在就返回它 —— 这是「下过一次就不再联网」的保证。"""
    cache = Path(_cache(tmp_path))

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("不该发请求")

    monkeypatch.setattr(httpx, "stream", _boom)

    assert _ensure_flashrank_model(cache, "m", "https://mirror.test", "hf_x") == cache / "m"


# ---- sentence-transformers（CrossEncoder）----


def test_cross_encoder_gets_query_passage_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_cross_encoder(monkeypatch)
    reranker = CrossEncoderReranker(_cfg(backend="sentence-transformers", max_length=256))

    result = asyncio.run(reranker.rerank("怎么配", [_hit("a"), _hit("b")], 2))

    assert reranker._model.pairs == [("怎么配", "正文 a"), ("怎么配", "正文 b")]
    assert reranker._model.kwargs == {"max_length": 256}, "窗口要显式给，不能靠默认值"
    assert [hit.chunk.chunk_id for hit in result] == ["a", "b"]


# ---- 装配 ----


def test_build_reranker_picks_the_configured_backend() -> None:
    """默认是 sentence-transformers：flashrank 的模型在本机网络下拿不到（见模块 docstring）。"""
    assert isinstance(build_reranker(Settings()), CrossEncoderReranker)

    settings = Settings()
    settings.providers.rerank.backend = "flashrank"

    assert isinstance(build_reranker(settings), FlashrankReranker)


def test_both_backends_are_registered() -> None:
    from rag.contracts import RERANKERS

    assert RERANKERS.names() == ["flashrank", "sentence-transformers"]


def test_an_unknown_backend_lists_what_is_available() -> None:
    settings = Settings()
    settings.providers.rerank.backend = "cohere"

    with pytest.raises(KeyError, match="已注册"):
        build_reranker(settings)
