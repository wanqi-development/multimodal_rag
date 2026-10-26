"""重排（精排）：把融合后的候选池按「查询 ↔ 正文」的相关性重新排一遍。

**为什么重排放在客户端、而不是用 Milvus 的 rerank**（2026-10-03 定的）：

* **作用域**：我们的候选来自两路（hybrid + bm25），重排必须吃 **L2 融合后的池子**。
  Milvus 的 reranker 跟着一次 search 走，只看得到那一次检索的候选 —— 跨不了通道。
* **换模型不改表结构**：Milvus 的 rerank 是写在 collection schema 上的 function，
  换个模型就得重建 collection（我们刚被「建好表就改不了维度」教育过一次）；
  这里换模型只改一行配置。
* 依赖与费用：flashrank 已在依赖里、本地 ONNX、零调用费；`pymilvus[model]` 要额外
  extras，线上重排 API 按量计费。

两个 backend 都是**本地交叉编码器**，都要把「query + 正文」成对喂进去 ——
这点和 embedding 截然不同：embedding 是两边各自编码再算距离，交叉编码器能看到两者的
交互，所以更准、也更慢。慢是它的代价，也是它只该作用在 20 条候选上的原因。
"""

from __future__ import annotations

import asyncio
import zipfile
from functools import cached_property
from pathlib import Path
from typing import Any

import httpx

from rag.config.schema import RerankProviderConfig
from rag.contracts import ScoredChunk, register_reranker
from rag.providers.hf import export_hf_env


class RerankError(RuntimeError):
    """重排失败（模型下不下来 / 加载失败 / 推理报错）。"""


def _ensure_flashrank_model(
    cache_dir: Path, model: str, endpoint: str | None, token: str | None
) -> Path:
    """模型不在就自己下一份（带 token），返回模型目录。

    仓库里的包名就是 ``<模型名>.zip``，解开后正好是 ``cache_dir/<模型名>/`` ——
    flashrank 看到这个目录存在就跳过它自己那套（已经 404 的）下载。
    下载先走配置里的镜像、再退回 huggingface.co：镜像对未缓存文件只做 302，
    而 302 之后 httpx 不会把 Authorization 带过去，所以**直连那一步必须自己带 token**。
    """
    model_dir = cache_dir / model
    if model_dir.is_dir():
        return model_dir
    name = f"{model}.zip"
    hosts = [h for h in (endpoint, "https://huggingface.co") if h]
    errors: list[str] = []
    for host in hosts:
        url = f"{host.rstrip('/')}/prithivida/flashrank/resolve/main/{name}"
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            with httpx.stream(
                "GET", url, headers=headers, follow_redirects=True, timeout=300.0
            ) as resp:
                resp.raise_for_status()
                blob = b"".join(resp.iter_bytes(1 << 16))
            archive_path = cache_dir / name
            archive_path.write_bytes(blob)
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(cache_dir)
            archive_path.unlink()
            return model_dir
        except Exception as exc:  # noqa: BLE001  —— 换下一个源再试
            errors.append(f"{url} → {type(exc).__name__}: {exc}")
    hint = "" if token else "（没有 HF_TOKEN：这个仓库匿名访问会被拒，在 .env 里配上）"
    raise RerankError(f"flashrank 模型 {model} 下不下来{hint}：" + "；".join(errors))


@register_reranker("flashrank")
class FlashrankReranker:
    """flashrank：ONNX 交叉编码器，CPU 上几十毫秒跑完 20 条。

    **模型由我们自己下**（2026-10-03 实测后定的）—— 它自带那条下载路走不通，两个原因：

    1. 地址硬编码直连 huggingface.co、不看 `HF_ENDPOINT`，而且 `requests.get` **不带鉴权头**
       —— 匿名打那个仓库是 401；
    2. 更要命的是 **0.2.x 里的文件名已经过期**：`Config.py` 写的是
       `flashrank-MiniLM-L-12-v2_Q.onnx.zip`，仓库里真正的包叫 `ms-marco-MiniLM-L-12-v2.zip`
       —— 带上 token 去下也是 404。

    所以我们自己按 `<模型名>.zip` 下（带 `HF_TOKEN`）、解到它认的目录结构里；它看到
    `cache_dir/<模型名>/` 存在就不会再联网。默认缓存目录它写死在 `/tmp`（重启就没），
    所以缓存目录必须显式指到项目内。
    """

    # 需要 HuggingFace token 才能取模型（装配层按配置里的 .env 取值传进来）
    needs_hf_token = True

    def __init__(self, cfg: RerankProviderConfig, token: str | None = None) -> None:
        self._cfg = cfg
        self._token = token
        self.name = f"flashrank:{cfg.model}"

    @cached_property
    def _ranker(self) -> Any:
        # 延迟加载：构造 reranker 不该触发下载（几万到几十万字节的 zip）
        export_hf_env(endpoint=self._cfg.hf_endpoint, disable_xet=self._cfg.hf_disable_xet)
        _ensure_flashrank_model(
            Path(self._cfg.cache_dir), self._cfg.model, self._cfg.hf_endpoint, self._token
        )
        from flashrank import Ranker

        return Ranker(model_name=self._cfg.model, cache_dir=self._cfg.cache_dir)

    async def rerank(self, query: str, chunks: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        if not chunks:
            return []
        passages = [{"id": index, "text": hit.chunk.text} for index, hit in enumerate(chunks)]
        try:
            ranked = await asyncio.to_thread(self._score, query, passages)
        except Exception as exc:
            raise RerankError(f"flashrank 重排失败：{type(exc).__name__}: {exc}") from exc
        return _reorder(chunks, ranked, top_n)

    def _score(self, query: str, passages: list[dict[str, Any]]) -> list[tuple[int, float]]:
        """flashrank 是同步且吃 CPU 的，调用方用 to_thread 包起来。

        注意它的 API 形态：`rerank(request: RerankRequest)` —— **收一个对象、不收
        (query, passages) 两个参数，也没有 top_n**（返回的是全部候选，按分数排好）。
        这条是第一次真机跑才发现的：替身是照着我们**以为**的签名写的，所以单测全绿。
        """
        from flashrank import RerankRequest

        hits = self._ranker.rerank(RerankRequest(query=query, passages=passages))
        return [(int(hit["id"]), float(hit["score"])) for hit in hits]


@register_reranker("sentence-transformers")
class CrossEncoderReranker:
    """SentenceTransformers 的 CrossEncoder —— 用来上**中文**重排模型（bge-reranker 一类）。

    与 flashrank 那条路的差别只有模型来源：这里 ``model`` 可以是本地目录
    （推荐，走 ``.models/``，和 embedding 同一条约定），也可以是 HF id。
    """

    def __init__(self, cfg: RerankProviderConfig) -> None:
        self._cfg = cfg
        self.name = f"cross-encoder:{cfg.model}"

    @cached_property
    def _model(self) -> Any:
        export_hf_env(endpoint=self._cfg.hf_endpoint, disable_xet=self._cfg.hf_disable_xet)
        from sentence_transformers import CrossEncoder

        return CrossEncoder(self._cfg.model, max_length=self._cfg.max_length)

    async def rerank(self, query: str, chunks: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        if not chunks:
            return []
        pairs = [(query, hit.chunk.text) for hit in chunks]
        try:
            scores = await asyncio.to_thread(self._predict, pairs)
        except Exception as exc:
            raise RerankError(f"CrossEncoder 重排失败：{type(exc).__name__}: {exc}") from exc
        return _reorder(chunks, list(enumerate(scores)), top_n)

    def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [float(score) for score in self._model.predict(pairs)]


def _reorder(
    chunks: list[ScoredChunk], ranked: list[tuple[int, float]], top_n: int
) -> list[ScoredChunk]:
    """按重排分数重排、截到 ``top_n``，并**把分数写回** ``ScoredChunk.score``。

    分数写回是有意的：到了这一步，检索分数（RRF 那个 0.03 量级）已经没有意义，
    交下去的是相关性分数。``channel`` 也换成 ``rerank``，免得下游以为这还是召回分。
    同分按原来的名次排 —— 重排模型的分数经常成片相等（量化后的 ONNX 尤其明显），
    没有这条会变成随机顺序。
    """
    order = sorted(ranked, key=lambda item: (-item[1], item[0]))[:top_n]
    return [
        ScoredChunk(
            chunk=chunks[index].chunk,
            score=score,
            channel="rerank",
            rank=rank,
            sources=chunks[index].sources,
        )
        for rank, (index, score) in enumerate(order)
    ]
