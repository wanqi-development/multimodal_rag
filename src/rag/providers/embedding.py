"""embedding provider：dense 编码。三条路可切。

只做 dense。sparse 那条腿由 Milvus 内建 BM25 在服务端从原文生成
（见 config 的 ``indexing.vector.bm25``），所以这里拿不到也不需要 sparse ——
fastembed 的 sparse 模型只有英文基座，中文语料用不上。

| backend | 说明 |
|---|---|
| ``fastembed`` | ONNX Runtime，不引入 torch；窗口小（bge-small-zh 只有 512 token） |
| ``sentence-transformers`` | 依赖 torch；可用窗口大的模型（现用 Qwen3-Embedding-0.6B，32768） |
| ``qwen3-vl`` | **多模态**（线上 DashScope 的 ``qwen3-vl-embedding``，2560 维）：**
文本**和**图文**都能编，进同一个向量空间 |

两条路都把配置里的 hf_* 落到 ``os.environ``：huggingface_hub 只认环境变量，
读不到我们的 yaml。
"""

from __future__ import annotations

import asyncio
import base64
import re
from functools import cached_property
from pathlib import Path

import httpx
from fastembed import TextEmbedding

from rag.config.schema import EmbeddingProviderConfig
from rag.contracts import Embeddings, register_embedder
from rag.providers.hf import export_hf_env

# 走多模态 backend 时，正文里的图片标记要**剥掉再编码**：`![image](asset:…)` 是记账用的，
# 不是内容 —— 留着会白占几十个字符，还会把「asset」这类词喂进向量。
_IMAGE_MARKER = re.compile(r"!\[[^\]]*\]\(asset:[^)]+\)")

# 线上 embedding 的 key：与 chat/vision 同一套约定，按供应商名取环境变量
KEY_ENV = {"dashscope": ("DASHSCOPE_API_KEY",)}


def _reject_unexpected_images(backend: str, images: list[list[Path]] | None) -> None:
    """纯文本 backend 收到图就报错 —— **不静默忽略**。

    忽略的后果很隐蔽：配置写成多模态、embedding 却是纯文本的，图片信息全丢，
    但链路看起来一切正常（库里就是一堆「只有描述文字」的向量），要到检索时才发觉。
    """
    if images and any(paths for paths in images):
        raise ValueError(
            f"embedding backend {backend!r} 不支持图文一起编码，但收到了图片。"
            f"要用多模态 embedding 就把 providers.embedding.backend 换成 qwen3-vl。"
        )


def _query_images(image: Path | None) -> list[list[Path]] | None:
    """「一张查询图」装成 ``_reject_unexpected_images`` 认的形状（没有图就是 None）。"""
    return [[image]] if image is not None else None


def _export_hf_env(cfg: EmbeddingProviderConfig) -> None:
    """embedding 这边的入口；实现在 ``providers/hf.py``（rerank 用的是同一个）。

    镜像不支持 HF 的 Xet 存储接口，不关掉会去请求 /xet-read-token/ 然后失败 ——
    这层细节由公共函数统一处理。
    """
    export_hf_env(endpoint=cfg.hf_endpoint, disable_xet=cfg.hf_disable_xet)


@register_embedder("fastembed")
class FastEmbedDenseEmbedder:
    """fastembed dense embedder。"""

    supports_images = False

    def __init__(self, cfg: EmbeddingProviderConfig) -> None:
        self._cfg = cfg
        self.name = f"fastembed:{cfg.model}"

    @property
    def dim(self) -> int:
        return self._cfg.dim

    @cached_property
    def _model(self) -> TextEmbedding:
        # 延迟到首次使用时再加载：构造 embedder 不该触发几百 MB 的模型加载，
        # 否则跑一个只想看看配置的 CLI 命令也要等下载。
        _export_hf_env(self._cfg)
        model = TextEmbedding(model_name=self._cfg.model)

        # 配置的 dim 与模型实际维度必须一致。不一致会建出一个维度对不上的
        # collection，而症状要到检索时才显形（那时候已经灌完数据了）。
        # 在这里当场挡掉。
        actual = len(next(iter(model.embed(["维度自检"]))))
        if actual != self._cfg.dim:
            raise ValueError(
                f"embedding 维度不匹配：config 里 dim={self._cfg.dim}，"
                f"但 {self._cfg.model} 实际输出 {actual} 维。"
                f"改 providers.embedding.dim，并重建 collection。"
            )
        return model

    def _encode(self, texts: list[str], *, query: bool) -> list[list[float]]:
        call = self._model.query_embed if query else self._model.embed
        return [[float(x) for x in vec] for vec in call(texts, batch_size=self._cfg.batch_size)]

    async def embed_documents(
        self, texts: list[str], images: list[list[Path]] | None = None
    ) -> Embeddings:
        _reject_unexpected_images(self.name, images)
        # fastembed 是同步且吃 CPU 的，扔进线程池，别堵住事件循环
        dense = await asyncio.to_thread(self._encode, texts, query=False)
        return Embeddings(dense=dense)

    async def embed_query(self, text: str, image: Path | None = None) -> Embeddings:
        # 纯文本这条只做文字 —— 给了图宁可报错，也别把「以图搜图」悄悄变成「用空文本搜」
        _reject_unexpected_images(self.name, _query_images(image))
        dense = await asyncio.to_thread(self._encode, [text], query=True)
        return Embeddings(dense=dense)


class EmbeddingError(RuntimeError):
    """线上 embedding 调用失败（连不上 / 鉴权 / 报文不对 / 返回条数对不上）。"""


@register_embedder("qwen3-vl")
class QwenVlMultimodalEmbedder:
    """**多模态** embedding：DashScope 原生的 multimodal-embedding 接口（qwen3-vl-embedding）。

    与另外两条路的关键差别：

    * **同一个向量空间里既有文本、也有图文**。文本块送 ``{"text": …}``；
      图片块送 ``{"text": 描述…, "image": dataURI}`` —— 后者在服务端**融合成一个向量**，
      所以「以文搜图」「以图搜文」都能做（纯图不带文字也支持，传 ``{"image": …}``）。
    * 走**原生端点**（``/api/v1/services/embeddings/multimodal-embedding/…``），
      不是 OpenAI 兼容通道 —— 实测兼容通道对这个模型直接 404
      （``Unsupported model for OpenAI compatibility mode``）。
    * ``input.contents`` 是**批量**语义：列表里每一项出一个向量，响应里带 ``index``
      可按它对序（不假设返回顺序）。
    * 维度 **2560**：换到这个 backend 必须同步改 ``providers.embedding.dim``，
      并且**换 collection 名**（Milvus 建好 schema 就不能改）。
    """

    supports_images = True
    # 构造它要多一把线上 key（装配层按 ``cfg.provider`` 去 .env 取）。见 build_embedder。
    needs_api_key = True
    # 服务端硬上限（都是实测撞出来的，各撞了一次 400）：
    #   * 纯文本批：≤ 20 —— `batch size is invalid, it should not be larger than 20,
    #     cur text batch:32`；
    #   * 含图的批：≤ 10 —— `image batch size can should be [1, 10]`。
    # 配置里的 batch_size 是给**本地**那条路调的，这里按内容分档取小值兜住：
    # 正文块占绝大多数，不该被图片的低上限连累成 10 个一批。
    MAX_TEXT_BATCH = 20
    MAX_IMAGE_BATCH = 10

    def __init__(self, cfg: EmbeddingProviderConfig, api_key: str) -> None:
        self._cfg = cfg
        self._api_key = api_key
        self.name = f"qwen3-vl:{cfg.model}"

    @property
    def dim(self) -> int:
        return self._cfg.dim

    def _make_client(self) -> httpx.AsyncClient:
        """每次调用**新建**一个客户端，别缓存。

        实测撞到的：httpx 的 ``AsyncClient`` 绑在**创建它的那个事件循环**上，而这条链路上
        ``asyncio.run`` 不止被调一次 —— 切分阶段的语义兜底自己起一个循环（每处理一段残留
        还各起一个），灌库那一步再起一个。缓存下来的客户端到第二个循环就是「Event loop
        is closed」，而且症状出现在**第二次调用**，最难往这上头想。
        """
        return httpx.AsyncClient(
            timeout=self._cfg.timeout_s,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )

    async def embed_documents(
        self, texts: list[str], images: list[list[Path]] | None = None
    ) -> Embeddings:
        contents = [
            _content(text, (images[index] if images and index < len(images) else []))
            for index, text in enumerate(texts)
        ]
        dense = await self._embed(contents)
        if len(dense) != len(texts):
            raise EmbeddingError(
                f"返回 {len(dense)} 条向量，但提交了 {len(texts)} 条内容 —— 数量对不上"
            )
        return Embeddings(dense=dense)

    async def embed_query(self, text: str, image: Path | None = None) -> Embeddings:
        """查询走同一个模型 —— 两边的向量必须来自同一个空间。

        给一张图就是**以图搜图**：图和文字放进同一个内容项，与文档侧的图片块完全对称
        （文档侧是「描述 + 图」，查询侧是「查询词 + 图」）。只给图不给字也行（``text=""``）——
        这正是当初选「图与文同一个空间」那条 embedding 的理由，否则图查询根本无处可去。
        """
        return Embeddings(dense=await self._embed([_content(text, [image] if image else [])]))

    async def _embed(self, contents: list[dict[str, str]]) -> list[list[float]]:
        """分批提交（``batch_size`` 条一批），按响应里的 ``index`` 对回原顺序。"""
        out: list[list[float]] = []
        # 客户端在**这一次调用**内复用（连接池给同一次调用的各批次共享），出了这一次就关 ——
        # 见 _make_client 里记的那个「事件循环已经关了」。
        async with self._make_client() as client:
            for batch in self._batches(contents):
                payload = {"model": self._cfg.model, "input": {"contents": batch}}
                try:
                    resp = await client.post(self._cfg.endpoint, json=payload)
                except httpx.HTTPError as exc:
                    raise EmbeddingError(
                        f"连不上 embedding 服务 {self._cfg.endpoint}（{type(exc).__name__}: {exc}）"
                    ) from exc
                out.extend(_vectors(resp))
        if out and len(out[0]) != self._cfg.dim:
            # 切到这个 backend 时最容易漏的两件事：dim（这里挡）与 collection 名（Milvus 挡）。
            # 在**第一次调用之后**就报出来 —— 等 Milvus 报要等到向量都花钱编完，那时才说
            # 「维度不对」既贵又晚。
            raise EmbeddingError(
                f"{self._cfg.model} 实际输出 {len(out[0])} 维，"
                f"但 providers.embedding.dim 写的是 {self._cfg.dim}。"
                f"改 dim，并换一个 collection 名 —— Milvus 建好表就改不了维度。"
            )
        return out

    def _batches(self, contents: list[dict[str, str]]) -> list[list[dict[str, str]]]:
        """按内容切批：**含图的批不超过 10 条，纯文本批不超过 20 条**。

        一次调用里图片是零散分布的（图片块只占少数），所以不能只看配置里的 batch_size ——
        批里只要有一张图，整批就得按图片那档来。
        """
        out: list[list[dict[str, str]]] = []
        batch: list[dict[str, str]] = []
        has_image = False
        for item in contents:
            # 这一条带图 → 当前批的容量降到图片档（已经装多了就先收走再开一批）
            limit = (
                self.MAX_IMAGE_BATCH
                if has_image or "image" in item
                else min(self._cfg.batch_size, self.MAX_TEXT_BATCH)
            )
            if len(batch) >= limit:
                out.append(batch)
                batch, has_image = [], False
            batch.append(item)
            has_image = has_image or "image" in item
        if batch:
            out.append(batch)
        return out


def _content(text: str, images: list[Path]) -> dict[str, str]:
    """一条内容项：**文本 + 图**（有图就一起给，服务端融合成一个向量）。

    正文里的图片标记在这里剥掉 —— 它是记账用的（哪个 asset 属于哪个块），不是内容。

    **多张图只取第一张**：这个接口的一个内容项只接受一张图（试过 `items` 形式，400）。
    实践上不会出现多张 —— 调用方只给 `modality=image` 的块附图，而那种块的 assets 恰好一张。
    """
    body = _IMAGE_MARKER.sub("", text).strip()
    # 纯图查询（没词）时**不要**带 `text: ""`：实测过的形态是 `{"image": …}`，
    # 空文本会不会被服务端收下没有验证过，没必要去赌。
    item: dict[str, str] = {"text": body} if body or not images else {}
    if images:
        item["image"] = _data_uri(images[0])
    return item


def _data_uri(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".") or "png"
    mime = {"jpg": "jpeg"}.get(suffix, suffix)
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{data}"


def _vectors(resp: httpx.Response) -> list[list[float]]:
    if resp.status_code >= 400:
        raise EmbeddingError(f"embedding 服务返回 HTTP {resp.status_code}：{resp.text[:300]}")
    try:
        body = resp.json()
    except ValueError as exc:
        raise EmbeddingError(f"embedding 服务没返回 JSON：{resp.text[:200]}") from exc
    entries = ((body.get("output") or {}).get("embeddings")) or []
    if not entries:
        raise EmbeddingError(f"embedding 响应里没有 output.embeddings：{str(body)[:200]}")
    # 响应带 index：按它对序，不假设返回顺序
    ordered = sorted(entries, key=lambda item: item.get("index", 0))
    return [[float(x) for x in (item.get("embedding") or [])] for item in ordered]


@register_embedder("sentence-transformers")
class SentenceTransformerEmbedder:
    """SentenceTransformer 的 dense 编码（现用于 Qwen3-Embedding-0.6B）。

    与 fastembed 那条路的差别不只是换个模型：

    - **查询要带指令前缀。** Qwen3-Embedding 这类模型是非对称的：query 前面加一句
      task instruction，文档不加。模型自己带着 prompt 定义，所以这里按名字取、
      不把指令硬编码 —— 换模型时它会跟着变，取不到就不加（别的模型也照样能跑）。
    - **依赖 torch**，比 ONNX 重得多：首次 import + 加载要十几秒，所以模型在
      ``cached_property`` 里延迟加载，且 ``sentence_transformers`` 是运行时才
      import 的 —— 否则每条 CLI 命令都要先付这个钱。
    - 窗口大得多，chunk 可以放宽到几千 token。
    """

    supports_images = False

    def __init__(self, cfg: EmbeddingProviderConfig) -> None:
        self._cfg = cfg
        self.name = f"sentence-transformers:{cfg.model}"

    @property
    def dim(self) -> int:
        return self._cfg.dim

    @cached_property
    def _model(self) -> object:
        _export_hf_env(self._cfg)
        # 延迟 import：torch 的导入本身就值十几秒
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self._cfg.model)
        actual = len(model.encode(["维度自检"], normalize_embeddings=True)[0])
        if actual != self._cfg.dim:
            raise ValueError(
                f"embedding 维度不匹配：config 里 dim={self._cfg.dim}，"
                f"但 {self._cfg.model} 实际输出 {actual} 维。"
                f"改 providers.embedding.dim，并重建 collection。"
            )
        return model

    def _prompt_name(self, *, query: bool) -> str | None:
        """模型自带的 prompt 名；没定义就返回 None（不加前缀）。"""
        prompts = getattr(self._model, "prompts", None) or {}
        name = "query" if query else "document"
        return name if name in prompts else None

    def _encode(self, texts: list[str], *, query: bool) -> list[list[float]]:
        kwargs: dict[str, object] = {
            "batch_size": self._cfg.batch_size,
            # 与 Milvus 的 COSINE 一致；Qwen3-Embedding 的用法也要求归一化
            "normalize_embeddings": True,
        }
        if name := self._prompt_name(query=query):
            kwargs["prompt_name"] = name
        vectors = self._model.encode(texts, **kwargs)
        return [[float(x) for x in vec] for vec in vectors]

    async def embed_documents(
        self, texts: list[str], images: list[list[Path]] | None = None
    ) -> Embeddings:
        _reject_unexpected_images(self.name, images)
        dense = await asyncio.to_thread(self._encode, texts, query=False)
        return Embeddings(dense=dense)

    async def embed_query(self, text: str, image: Path | None = None) -> Embeddings:
        # 同 fastembed：纯文本这条路遇到图就报错，别静默退化成文字查询
        _reject_unexpected_images(self.name, _query_images(image))
        dense = await asyncio.to_thread(self._encode, [text], query=True)
        return Embeddings(dense=dense)
