"""配置树的数据模型。

对应 config/*.yaml 的结构。providers 下按名字分组（default / fast / smart ...），
这样"多供应商可切换"就是配置里选一个 key，而不是改代码。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ChatProviderConfig(BaseModel):
    """一个 chat 供应商实例。

    api_key 一律从环境变量取（见 config.settings.Secrets），不落配置文件。
    """

    provider: str
    model: str
    base_url: str | None = None
    temperature: float = 0.0
    timeout_s: float = 120.0
    max_retries: int = 2


class EmbeddingProviderConfig(BaseModel):
    """dense embedding。

    sparse 不再由 embedding 模型产出 —— 改由 Milvus 内建 BM25 在服务端从原文生成
    （见 Bm25Config）。所以这里只管 dense。

    默认是 Qwen3-Embedding-0.6B（1024 维，窗口 32768 token）。换它是因为原来那条
    fastembed + bge-small-zh 的路窗口只有 512 token，块稍微一大就被模型静默截断。
    ``model`` 给的是**本地路径**：hf-mirror 拉这个模型会卡在 0 字节，实测走 ModelScope
    下到 ``.models/`` 才可靠；要换成 HF id，设 ``EMBEDDING_MODEL`` 覆盖即可。

    hf_* 两项是为了国内网络：huggingface.co 直连不通，必须走镜像，
    且要关掉 Xet（镜像不支持 HF 的 Xet 存储接口）。
    """

    # fastembed | sentence-transformers | qwen3-vl（见 providers/embedding.py 的对比表）
    # qwen3-vl 是**多模态**那条路：走线上 DashScope 的原生 multimodal-embedding 接口，
    # 文本与「图文」进同一个 2560 维空间（换它必须同步改 dim 与 collection 名）。
    backend: str = "sentence-transformers"
    model: str = ".models/Qwen3-Embedding-0.6B"
    # 只有 qwen3-vl 用得上：供应商名（决定 key 取哪个环境变量）与原生端点
    provider: str = "dashscope"
    endpoint: str = (
        "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
        "multimodal-embedding/multimodal-embedding"
    )
    timeout_s: float = 120.0
    # 这里是 dense 维度的唯一来源；Milvus collection 的 dim 由它推导，不要在两处各写一遍
    dim: int = 1024
    batch_size: int = 32
    hf_endpoint: str | None = "https://hf-mirror.com"
    hf_disable_xet: bool = True


class RerankProviderConfig(BaseModel):
    """重排模型。

    只做**精排**：融合后的候选池（20 条量级）按「查询-正文」相关性重排，再按调用方要的
    条数交货。**没有 top_n 这个旋钮** —— 交几条由调用方的 top_k 说了算，两处各写一个数
    必然打架（`-k 20` 被悄悄截成 8 这种）。

    两个 backend 都是**本地跑**（`providers/rerank.py`）：

    * ``flashrank``：ONNX，依赖已在（几十 MB 的 zip，从 HF 社区仓下），英文 MS MARCO 训的；
    * ``sentence-transformers``：CrossEncoder，可指向**中文**重排模型（torch 已在），
      模型路径与 embedding 那条路同一个约定（本地目录优先，走项目内 ``.models/``）。

    首次使用要下模型，所以缓存目录指到项目内（flashrank 默认写 ``/tmp``，重启就没）
    —— 与 embedding 一样，先设 HF 镜像环境变量再加载。
    """

    # 默认走 sentence-transformers：flashrank 的模型在本机网络下拿不到（见 providers/rerank.py
    # 里那段的实测记录），而 bge-reranker-base 能从 ModelScope 下到、且中英都能用。
    backend: str = "sentence-transformers"
    model: str = ".models/bge-reranker-base"
    cache_dir: str = ".models/flashrank"
    # CrossEncoder 的输入上限（query + 正文）。chunk ≤1000 字 ≈ 700 token，
    # MiniLM 系只有 512 窗口 —— 超了会被静默截断，所以显式给一个数、不靠默认值。
    max_length: int = 512
    hf_endpoint: str | None = "https://hf-mirror.com"
    hf_disable_xet: bool = True


# 图片描述**提示词**的版本号。它同时进 pipeline_version（决定要不要重灌）与描述
# 缓存键（决定描述还算不算数）—— 改了 providers/vision.py 里的提示词就抬一位，
# 否则「提示词明明改了，描述却还是老的」会是那种查不出来的怪事。
#
# 常量放在 config 而不是 providers/vision.py：ingest 层也要读它，而 providers
# 是个重包（一 import 就带出 fastembed / langchain）。
#
# v2（2026-10-01）：改成**先判上下文相关不相关** —— 相关就把上下文那件事提取进描述，
# 不相关就只按图描述；第一句必须点明这张图在讲什么。多模态那条 embedding 把
# 「描述 + 图片」融合成一个向量，检索入口就是这段文字。
VISION_PROMPT_VERSION = "2"


class VisionProviderConfig(BaseModel):
    """看图说话的模型（VLM）。**与 chat 是两套模型**：chat 那几家多半不能看图。

    走 OpenAI 兼容通道 + 三件套（key / base_url / model_name 全在 .env），
    与 ``providers.chat.*`` 同一套约定：

    * ``DASHSCOPE_VL_API_KEY``（留空回退到 ``DASHSCOPE_API_KEY`` —— 同一个账号
      同一把 key，只是给「VL 走另一个账号」留个口子）
    * ``DASHSCOPE_VL_BASE_URL``
    * ``DASHSCOPE_VL_MODEL_NAME``

    **这里的默认值必须是字面量**：pydantic 不做 ``${VAR:-default}`` 插值，
    插值只发生在读 config/*.yaml 那一步。写成 ``${...}`` 的话，yaml 没写这一项时
    拿到的就是一串字面量，会被当成真实地址去请求。
    """

    provider: str = "dashscope"
    model: str = "qwen-vl-max"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    timeout_s: float = 120.0
    max_retries: int = 2


class AsrProviderConfig(BaseModel):
    """语音转文字（ASR）。**与 ``VisionProviderConfig`` 同一套三件套约定**（key / base_url /
    model_name 都在 .env），但**端点不是同一个** —— 见下。

    * ``DASHSCOPE_ASR_API_KEY``（留空回退到 ``DASHSCOPE_API_KEY`` —— 同一个账号同一把 key）
    * ``DASHSCOPE_ASR_BASE_URL``
    * ``DASHSCOPE_ASR_MODEL_NAME``

    **走 DashScope 原生端点**（`/api/v1/services/aigc/multimodal-generation/generation`），
    不是 vision 那条 OpenAI 兼容的 `/compatible-mode/v1/chat/completions`。这是拿真调用换来的：
    照兼容通道抄过去，上游回 `{"message":"format is empty","type":"UNSUPPORTED_FORMAT"}` ——
    兼容通道读 `input_audio.format`，而原生端点把 `format` / `sample_rate` 放在 `parameters` 里。
    所以这里默认的 `base_url` 是**根域**（`https://dashscope.aliyuncs.com`），路径由客户端拼。
    账号若走百炼工作空间域名（`https://<workspace-id>.cn-beijing.maas.aliyuncs.com`），
    把 `DASHSCOPE_ASR_BASE_URL` 设成它即可。

    **型号名是查过账号里的清单才定的**（2026-10-08）：``qwen-audio-3.0-asr-flash`` 确实在
    ``GET /compatible-mode/v1/models`` 里。这一条值得单独写下来 —— ``deepseek-flash`` 那次
    就是名字写错，一直拖到第一次调用才以 400 暴露。

    **不能用 ``qwen-vl-max`` 干这件事**：它是视觉语言模型（图 + 文字），不接受音频输入；
    语音在 DashScope 是另一族模型（``qwen-audio-*-asr-*`` / ``qwen3-asr-flash-*`` /
    ``fun-asr-flash-*`` / ``*-realtime-*``）。「边说边出字」要 ``*-realtime-*`` 那族，
    现在选的是一次性识别这条。
    """

    provider: str = "dashscope"
    model: str = "qwen-audio-3.0-asr-flash"
    # **根域**，不是 compatible-mode：原生端点要自己拼 `/api/v1/...`（见类说明）
    base_url: str = "https://dashscope.aliyuncs.com"
    # 一次一段音频，比 chat 慢：留足余量
    timeout_s: float = 120.0
    # **默认不重试**：识别失败多半是格式/时长的原因，重试只会把钱花三遍
    max_retries: int = 0


class ProvidersConfig(BaseModel):
    chat: dict[str, ChatProviderConfig] = Field(
        default_factory=lambda: {
            "default": ChatProviderConfig(provider="deepseek", model="deepseek-chat")
        }
    )
    embedding: EmbeddingProviderConfig = Field(default_factory=EmbeddingProviderConfig)
    rerank: RerankProviderConfig = Field(default_factory=RerankProviderConfig)
    vision: VisionProviderConfig = Field(default_factory=VisionProviderConfig)
    asr: AsrProviderConfig = Field(default_factory=AsrProviderConfig)


class FieldIndexConfig(BaseModel):
    """单个向量字段的索引参数。"""

    type: str
    metric: str


class Bm25Config(BaseModel):
    """Milvus 内建 BM25。

    服务端用 analyzer 把 input_field 的原文切词，自动写入 output_field 的 sparse 向量。
    好处：不需要 sparse embedding 模型、不需要 torch，中文分词由服务端完成。
    """

    enabled: bool = True
    input_field: str = "text"
    output_field: str = "sparse"
    analyzer: str = "chinese"


class VectorIndexConfig(BaseModel):
    backend: str = "milvus"
    uri: str = "http://localhost:19530"
    token: str | None = None
    collection: str = "rag_v1"
    # 双版本：旧 collection 只读，新批次写 <collection>_versions。
    # 版本化表的 schema 多一个显式 index_batch_id，不能复用旧 collection。
    versions_enabled: bool = True
    versions_collection: str = ""
    # 开工前探活的超时（秒）。为什么需要它：Milvus 停掉时 pymilvus 的重连会**空转很久**
    # —— 实测 `rag ingest` 在写库那步烧了三十几分钟 CPU、数据一行没进（§16）。
    # 探活接不住（5 秒没响应）就直接退出，别让人对着一个不动的终端猜。
    probe_timeout_s: float = 5.0
    # 这里不写 dim：向量维度由 providers.embedding.dim 推导，两处各写一遍必然漂移
    # dense 与 sparse 的 index 参数不同：前者 IVF_FLAT/COSINE，后者 SPARSE_INVERTED_INDEX/BM25
    indexes: dict[str, FieldIndexConfig] = Field(
        default_factory=lambda: {
            "dense": FieldIndexConfig(type="IVF_FLAT", metric="COSINE"),
            "sparse": FieldIndexConfig(type="SPARSE_INVERTED_INDEX", metric="BM25"),
        }
    )
    bm25: Bm25Config = Field(default_factory=Bm25Config)


class SemanticChunkConfig(BaseModel):
    """语义兜底的参数。只对 ``strategy: markdown-semantic`` 生效。

    两条护栏都是实测撞出来的：只有 percentile 断点时切出过 **3 个字的块**（``'但是，'``），
    那在检索里等于没有内容；而语义断点**不是尺寸控制** —— 一段语义自洽的长文本它就是不断，
    所以合并时必须有尺寸上限。
    """

    # 小于它就不切：断点太密时宁可让块大一点，也不要微块
    min_size: int = 80
    # 相邻句余弦距离超过这个分位数，就算一个语义断点
    percentile: float = 90.0


class ChunkingConfig(BaseModel):
    """切块参数。

    size 是**字符数**不是 token 数：中文里 1 字约等于 1.07 token（实测 550 字 = 512 token）。
    旧模型 bge-small-zh-v1.5 的窗口只有 512 token，所以当时 size=400 是贴着窗口留的余量；
    现在换成窗口 32768 的 Qwen3-Embedding-0.6B，尺寸约束松了，size 按语义粒度定。
    """

    strategy: str = "recursive"
    size: int = 1000
    overlap: int = 60
    semantic: SemanticChunkConfig = Field(default_factory=SemanticChunkConfig)


class IndexingConfig(BaseModel):
    vector: VectorIndexConfig = Field(default_factory=VectorIndexConfig)
    chunk: ChunkingConfig = Field(default_factory=ChunkingConfig)


# PDF 走哪个解析后端（`ingest.pdf.parser` 的全部取值）。
#   paddleocr —— PaddleOCR-VL 产线（**规范名**；`ocr` 是历史别名，两个都收）
#   dots      —— dots.ocr（vLLM）：给标题层级
#   mineru    —— MinerU 远程 API：页码/块级定位最规整
#   pymupdf   —— 不是 OCR，是本地文字层提取（有损降级）
# 别在这儿加"要不要 OCR"那种布尔开关：所有 PDF 都整份送出去，不探测是不是扫描件。
PDF_PARSERS = ("paddleocr", "dots", "mineru", "pymupdf")

# 老 `.env` 与老配置里写的是 `ocr`（三个字母）。它专指 PaddleOCR 那一家产线，
# 和 `dots` / `mineru` 并列时既别扭又像"要不要 OCR"的开关，所以规范名改成 `paddleocr`。
# **别名照收**：老配置不该因为一次改名就跑不起来。
PDF_PARSER_ALIASES = {"ocr": "paddleocr"}


def normalize_pdf_parser(name: str) -> str:
    """别名 → 规范名。认不出来的原样返回（由调用方报错时列出可选值）。"""
    key = (name or "").strip().lower()
    return PDF_PARSER_ALIASES.get(key, key)


class IngestPdfConfig(BaseModel):
    """PDF 该走哪条解析路径（取值见 `PDF_PARSERS`）。

    `pymupdf` 是**有损的降级开关**：产线不可用、或要拿它做对照时用。其余三个都是
    "整份 PDF 交给谁"—— 所有 PDF 都整份送出去，不再探测是不是扫描件。理由见
    loaders/pdf.py 的开头：pymupdf 只能拿到纯文本，标题层级、表格、图片全都没有，
    下游的 markdown 切分与图片描述都会空转。

    **值在这里就归一化**（`ocr` → `paddleocr`）：这样分发、界面、以及 `parse_key()`
    里的指纹只有一种说法 —— 指纹混着两个名字写，会让同一份文档因为"写法不同"被判成
    换了配置、白重跑一遍 OCR。
    """

    # 与 `config/base.yaml` 的默认值一致（那里是 `${RAG_PDF_PARSER:-dots}`）：
    # schema 默认只在配置里整段缺了 `ingest.pdf` 时才用得上
    parser: str = "dots"

    @field_validator("parser")
    @classmethod
    def _canonical_parser(cls, value: str) -> str:
        return normalize_pdf_parser(value)


class IngestCloudOcrConfig(BaseModel):
    """本地产线挂掉时的兜底：**PaddleOCR 云端 API**（`paddleocr` 自带的 SDK）。

    与本地那条路的两处差别：要 access token（走 `.env`，机密不进 yaml）；
    按量计费，所以**默认关闭** —— 打开它是个花钱的决定，不该悄悄发生。

    打开却没配 token 会直接报错，而不是默默退回「没有兜底」。
    """

    enabled: bool = False
    model: str = "PaddleOCR-VL-1.6"
    base_url: str = "https://paddleocr.aistudio-app.com"
    request_timeout_s: float = 300.0
    # 云端是「提交任务 + 轮询」，整份几百页的 PDF 要留足时间
    poll_timeout_s: float = 900.0


class IngestOcrConfig(BaseModel):
    """PaddleOCR-VL 产线服务。

    注意打的是**产线服务**（paddleocr-vl-api，默认 8080），不是那个 vLLM 托管的
    PaddleOCR-VL 模型（默认 8118）—— 8118 只是产线的下游，单独打它拿不到版面分析。

    base_url 为空 = 不启用 OCR（返回 None），这条链路上没有任何网络请求，
    扫描件会照旧计入 skipped 并给出指路的原因。这样没部署 GPU 服务的机器也能跑。
    """

    base_url: str | None = None
    # 整份 PDF 走一次 OCR 比单页慢得多，别沿用 chat 那个 120s
    timeout_s: float = 600.0
    max_retries: int = 1
    # 产线**不可用时**先原地重试几次，再考虑云端兜底。实测那条转发在真实负载下 28% 失败
    # （§16），一个 502 就把整份文档推给按量计费的云端不值得。0 = 不重试。
    retry_primary: int = 1
    # 让产线把图片内联进 markdown（<img src="imgs/…">）并在响应的 markdown.images
    # 里回图片数据。开着才能把图片放进文本流的位置上、后面挂 VLM 描述。
    # 代价：响应会大很多（11 页实测 949KB，其中绝大部分是 base64）。
    return_images: bool = True
    # 跨页重构的三件套：只 merge_tables / relevel_titles 不开 restructure_pages 是不生效的，
    # 所以这三个是一组，默认全开
    restructure_pages: bool = True
    merge_tables: bool = True
    relevel_titles: bool = True
    cloud_fallback: IngestCloudOcrConfig = Field(default_factory=IngestCloudOcrConfig)


class IngestDotsConfig(BaseModel):
    """dots.ocr：**另一个 PDF 解析后端**（vLLM 上的 dots.ocr，默认 127.0.0.1:6006）。

    与 PaddleOCR 产线的分工是「谁负责给层级」：产线的 relevelTitles 只能按版面标签猜，
    实测在这些教材 PDF 上不可用；dots.ocr 是端到端的版面模型，`Section-header` 块直接
    给出 `#`/`##`/`###`，还带 bbox。所以 PDF 的 `parser: dots` 是现在的默认。

    走 OpenAI 兼容通道，与 ``providers.vision`` 同一套约定：
    ``DOTS_OCR_BASE_URL`` / ``DOTS_OCR_MODEL_NAME``（本地服务不要 key，样例里给的是 "0"）。
    """

    base_url: str = "http://127.0.0.1:6006"
    model: str = "dots_ocr"
    # 页面渲染 DPI。200 是样例默认值，再高对识别帮助不大、传输量却翻倍
    dpi: int = 200
    # 页之间并发。一页一次 vLLM 调用，串行会把几十页的 PDF 拖成十几分钟
    num_thread: int = 16
    # 单页失败重试几次。**长文档必需**：几百页里只要有一页失败，整份就白跑了
    # （本方法故意不吞错 —— 少几页而文档照常入库是最难查的错）。
    max_retries: int = 2
    # 样例里的模式：版面 + 识别 + 层级（prompt_layout_all_en）
    prompt_mode: str = "prompt_layout_all_en"
    temperature: float = 0.1
    top_p: float = 1.0
    max_completion_tokens: int = 16384


class IngestMineruConfig(BaseModel):
    """MinerU 的**远程 API**（`MinerU API v1.0.0`，默认 127.0.0.1:6006）。

    流程是「上传 → 建任务 → 轮询 → 取产物」（见 ``ingest/mineru_api.py``），
    与另外两家的差别：

    * **页码与块级定位最规整**：产物里的 `middle_json` 是 `pages[] → blocks[]`，
      每块带 `type`（text / paragraph_title / image / table / page_number）+ 归一化 bbox；
    * **不给标题层级**：`paragraph_title` 的 `level` 实测恒为 2（全平），所以清单初值
      要靠编号推断（见 ``ingest/outline.py`` 的自适应）；
    * **慢**：standard 档实测热态约 27 秒/页（396 页的 Kafka 要几小时）——
      大文档用 ``basic`` 档，或按 ``page_range`` 只解析需要的章节。
    """

    base_url: str = "http://127.0.0.1:6006"
    # 本地部署不要 key；网关/云端部署才填（空 = 不发 Authorization 头）
    api_key: str = ""
    # flash（最快，纯文字层）| basic（轻量本地模型）| standard（默认，VLM）| advanced
    tier: str = "standard"
    # auto（自动判断文字层）| txt（只用文字层）| ocr（强制 OCR）
    ocr_mode: str = "auto"
    # 只解析这些页（1 起，含端点，如 "1-5,8"；空 = 整份）。大文档靠它控制代价
    page_range: str = ""
    # 单次任务最长等多久（秒）。standard 档约 27 秒/页，整份大 PDF 要给足
    timeout_s: float = 3600.0
    poll_interval_s: float = 3.0
    # 上传与轮询失败重试几次（远端服务偶发 5xx / 连接重置）
    max_retries: int = 2


class IngestAssetsConfig(BaseModel):
    """原始资产（目前只有图片）的落盘位置。

    `backend=auto`（默认）听统一开关 `storage.backend`：本地盘或桶（键 = `assets/<doc_id>/<名>`）。
    两种后端下**正文里的标记都不动**（标记只带相对 uri）。按 doc_id 分目录，重灌同名覆盖。
    """

    # auto（默认）= 听统一开关 `storage.backend`；要单独钉死这个域就写 local / minio
    backend: str = "auto"  # auto | local | minio；见 storage.py 的 resolve_storage_backend
    prefix: str = "assets"  # minio 桶内的前缀
    dir: str = "./data/assets"


class IngestStagesConfig(BaseModel):
    """三段流水线的中间产物落在哪（见 ``ingest/stages.py``）。

    拆成「解析 → 切分+描述 → 灌库」有两个用处：

    1. **人工可核对**：OCR 读歪了、标题层级错了、块切得太碎 —— 只有中间结果落了盘才看得见；
    2. **解耦时间与依赖**：前两段只吃本地文件与模型（不碰 Milvus），
       第三段才写库。Milvus 挂了、或想等机器空闲再灌，都不必把前面重做一遍。

    这是**缓存**不是数据：删掉只会让下一次 ingest 重跑一遍，不会丢任何已入库的东西。
    """

    backend: str = "auto"  # auto（听统一开关）| local | minio
    prefix: str = "stages"
    dir: str = "./data/stages"


class IngestFetchConfig(BaseModel):
    """从链接抓文档（页面上的「从链接导入」）。

    **抓取本身不花钱**，但抓回来是什么决定后面花不花：HTML / Word / 表格是本地解析，
    PDF 走 `ingest.pdf.parser`（默认 dots，打本机那个 vLLM 服务），图片走 OCR。
    哪天把 dots / OCR 换成云端服务，同样的操作就开始按量计费了。
    """

    # 一次抓取的超时。抓一页是秒级，给到 15s 是给慢的下载链接留余量
    timeout_s: float = 15.0
    # 最多跟几次重定向（每一跳都要过一次 IP 检查，见 ingest/fetch.py）
    max_redirects: int = 5
    # **默认关**：开了就等于允许后端去抓内网地址（自建文档站才需要），
    # 而这是把服务端当内网代理的口子
    allow_private_hosts: bool = False
    # 抓 HTML 时最多下几张图。**这是钱的口子**：每张图都要过一次 VLM 描述（按量计费），
    # 所以默认给个保守值；正文里的图通常远少于这个数
    max_images: int = 20
    # 单张图上限（MB）。比整份文件的上限小得多：网页里的图大多是几十到几百 KB，
    # 一张几十 MB 的原图多半是误抓
    max_image_mb: int = 8
    # 网页用谁来抓：
    #   http（**默认**）—— 本机 GET，然后自己抽正文。不花钱，但导航/页脚/"Copy page"
    #     这类 chrome 得靠我们自己的规则剔（现代前端的导航常常不是 `<nav>`）。
    #   firecrawl —— 交给 Firecrawl 的 `/scrape`（`onlyMainContent`），正文干净得多；
    #     **按量计费**，所以默认不切；key 只从 `.env`（FIRECRAWL_API_KEY）读。
    #   **只管网页**：PDF / 图片 / 附件一律走本机（那几条路要的是原件，见 ingest/fetch.py）。
    backend: str = "http"  # http | firecrawl
    # 云端接口。自建部署改成自己的地址（路径仍是 /v2/scrape）；不是机密
    firecrawl_api_url: str = "https://api.firecrawl.dev"
    # 只要正文（导航 / 页脚 / 侧栏交给它判）—— 换 Firecrawl 的主要理由
    only_main_content: bool = True
    # 云端要现场渲染，比我们自己的 15s 宽松：一次 scrape 给它 60s
    firecrawl_timeout_s: float = 60.0


class IngestUploadsConfig(BaseModel):
    """页面上传的落点（**只放上传件，不放产物**）。

    先落本地盘、**保留不删**（使用者 2026-10-07 定）。将来换「文件服务器」时，
    改的是 `api/uploads.py` 里 `UploadStore` 的实现 —— 调用方拿的是一份描述
    （doc_id / 真名 / 落盘名 / 相对路径），要读文件时走 `path_of()`，所以多 worker 也能接。

    扁平存 `<dir>/<真名>-<sha1 前 16 位><扩展名>`（使用者 2026-10-07 定）：哈希就是
    `ingest.incremental.file_hash` 那一个，所以**同名同内容 = 同一个 doc_id = 原地 upsert**
    （Milvus 里不留旧块），**同名不同内容 = 两份文件**，由页面问用户「覆盖更新还是保留两份」。
    真名与来源 URL 记在 `<dir>/_index.json`，页面显示真名。
    """

    backend: str = "auto"  # auto（听统一开关）| local | minio
    prefix: str = "uploads"
    dir: str = "./data/uploads"
    # 单个文件上限（MB）：挡掉误传的大文件，别让一次上传把 worker 的内存吃满
    max_mb: int = 64
    # 「从链接导入」那一套（`api/routes.py` 的 `/documents/fetch`）
    fetch: IngestFetchConfig = Field(default_factory=IngestFetchConfig)


class IngestImagesConfig(BaseModel):
    """图片的 VLM 描述。

    **默认关闭**：每张图一次 VLM 调用、按量计费 —— 打开它是个花钱的决定，
    不该在一份默认配置里悄悄发生（与 cloud_fallback 同一个立场）。

    打开后：每张图配它**切块后的上下文**（上边一段 + 下边一段）问 VLM，
    描述作为独立块进库（``modality=image``）。
    """

    describe: bool = False
    # 上下文各取多少字符。两段加起来太长会稀释图片本身的信号
    context_chars: int = 600
    # 单次最多描述几张图，0 = 不限制。**先花几毛钱看效果**的口子：
    # 打开 describe 之前先用它跑几张，确认提示词与模型对路再全量。
    max_images: int = 0
    # auto（听 storage.backend）| local | minio。scale-out 下自动走桶，避免每个 worker 一份缓存。
    cache_backend: str = "auto"
    cache_prefix: str = "image-cache"
    # 描述缓存的落盘位置。**按图片内容哈希 + 模型 + 提示词版本做键** ——
    # 改一段正文不该让几十张图全部重问一遍 VLM（那是整条链路里最贵的一步）。
    cache_path: str = "./data/cache/image_descriptions.json"


class IngestConfig(BaseModel):
    pdf: IngestPdfConfig = Field(default_factory=IngestPdfConfig)
    ocr: IngestOcrConfig = Field(default_factory=IngestOcrConfig)
    dots: IngestDotsConfig = Field(default_factory=IngestDotsConfig)
    mineru: IngestMineruConfig = Field(default_factory=IngestMineruConfig)
    assets: IngestAssetsConfig = Field(default_factory=IngestAssetsConfig)
    images: IngestImagesConfig = Field(default_factory=IngestImagesConfig)
    stages: IngestStagesConfig = Field(default_factory=IngestStagesConfig)
    uploads: IngestUploadsConfig = Field(default_factory=IngestUploadsConfig)


class ChannelLegConfig(BaseModel):
    """hybrid 通道的一条腿（dense 或 sparse）。"""

    field: str
    limit: int = 20
    weight: float = 1.0


class FusionConfig(BaseModel):
    strategy: str = "rrf"
    k: int = 60
    # 每一路要多少候选（L2 融合用）。**所有通道要一样的数**，否则某一路的下手机会更多 ——
    # 实测撞到过：hybrid 只给 top_k=5 条、bm25 按自己的 limit 给 20 条，20 那一侧的票多 4 倍。
    # 取 max(top_k, 这个值)：调用方要得比它多时听调用方的。
    # 重排一跑就是 6–8 秒（26 条候选、CPU），减半池子直接减半耗时。
    candidates_per_channel: int = 10


class HybridChannelConfig(BaseModel):
    dense: ChannelLegConfig = Field(default_factory=lambda: ChannelLegConfig(field="dense"))
    sparse: ChannelLegConfig = Field(default_factory=lambda: ChannelLegConfig(field="sparse"))
    fusion: FusionConfig = Field(default_factory=FusionConfig)


class TransformConfig(BaseModel):
    """查询变换。multi_query=0 表示关闭。"""

    multi_query: int = 0
    hyde: bool = False


class RetrievalRerankConfig(BaseModel):
    """重排开关。交几条给下游由调用方的 top_k 决定（见 RerankProviderConfig）。"""

    # **默认开**（2026-10-03 定的）：实测 bge 重排在中文语料上真能挑出东西 —— 同一查询，
    # 关着重排时「真正那张架构图」不在前 5，开了之后升到 #2。代价是每次查询多 6–8 秒
    # （26 条候选、CPU）。临时关掉：`$env:RAG_RERANK="false"`（base.yaml 里走插值）。
    enabled: bool = True


class AssembleConfig(BaseModel):
    max_tokens: int = 6000


class GraphChannelConfig(BaseModel):
    """图谱通道（`type: neo4j`）**自己的参数**（M6）。

    **这里没有 `enabled` 开关**：这一路参不参与检索，由 `retrieval.channels` 里有没有
    那一项决定 —— 两个开关迟早会传错一个（workspace 那次也是这么定的）。
    要开第二路同类源（比如另一个图库），在那**一项**的 `options` 里写参数。

    `uri` / `user` 不是机密，放 yaml（走 SSH 隧道时就是 `bolt://127.0.0.1:7687`）；
    密码只从 `.env` 的 `NEO4J_PASSWORD` 来（见 config/settings.py 的 Secrets）。
    """

    uri: str = "bolt://127.0.0.1:7687"
    user: str = "neo4j"
    # 最多走几跳：1 跳 = 邻居关系，2 跳 = 「邻居的邻居」。跳数一涨，召回面与噪声一起涨，
    # 所以默认 2 而不是 3 —— 多跳的真实收益要用 rag eval 量，别凭感觉加。
    hops: int = 2
    # 问题里最多认几个实体、每个实体最多取几条边：都是给「问句很长 + 热门实体」兜底的，
    # 不设上限的话「Flink」一个词就能拉回几百条边，融合层直接被它淹掉
    max_entities: int = 8
    edges_per_entity: int = 50
    # 探活/查询超时：Neo4j 挂了要**快速降级**（这一路返回空 + 记 stats.error），不带崩检索
    timeout_s: float = 10.0


class ChannelConfig(BaseModel):
    """**一路通道 = 一个数据源**（2026-10-08 定）。装配由 `RetrievalConfig.channels` 这张表驱动。

    * `type` 决定用哪个**适配器**（插件化，见 `retrieval/plugins.py`）：
      内置 `milvus` / `neo4j`；第三方包的 entry point 名字；开发期也可以直接写点路径
      `包.模块:函数`。
    * `enabled` 是这一路参不参与检索的**唯一**开关 —— 不再有"名字列表 + 注册表"两处。
    * `complement_only` 覆盖适配器的默认值：**只补位**（与别路重复的候选丢掉，只贡献
      别人没找到的块）。异构源大多该这么开。
    * `top_k` 覆盖这一路的候选数（默认用 `retrieval.fusion.candidates_per_channel`）。
    * `options` 由适配器自己解释（内置的 milvus 认 `dense_limit` / `sparse_limit`）。
    """

    name: str
    type: str
    enabled: bool = True
    complement_only: bool | None = None
    top_k: int | None = None
    options: dict[str, Any] = Field(default_factory=dict)


def _default_channels() -> list[ChannelConfig]:
    """默认只有一路：**Milvus**（多模态融合向量 + 内建 BM25，服务端融合）。

    「通道 = 数据源」这条别在这儿打折：语义与关键词**同在这一路里**，不拆成两路；
    图谱要用来再加一项（`type: neo4j`）。
    """
    return [ChannelConfig(name="milvus", type="milvus")]


class RetrievalConfig(BaseModel):
    """检索配置。**通道 = 数据源**（2026-10-08 定）。

    * `channels` —— 这张表就是全部通道：加一路 = 加一项（同类源只改配置；
      全新类型要装一个实现了通道接口的包，见 `docs/architecture.md` §9.2）；
    * `hybrid` / `graph` —— 那两路适配器**自己的参数**（dense/sparse 上限、Neo4j 地址与跳数…）。
      参数留在这里是因为它有类型校验；要开**第二路同类源**时用那一项自己的 `options` 覆盖。
    * `fusion` —— **L2 融合**：跨通道的 RRF。与 `hybrid.fusion`（L1，Milvus 服务端融
      dense+sparse）是两回事，别混。
    """

    channels: list[ChannelConfig] = Field(default_factory=_default_channels)
    hybrid: HybridChannelConfig = Field(default_factory=HybridChannelConfig)
    graph: GraphChannelConfig = Field(default_factory=GraphChannelConfig)
    transform: TransformConfig = Field(default_factory=TransformConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    rerank: RetrievalRerankConfig = Field(default_factory=RetrievalRerankConfig)
    assemble: AssembleConfig = Field(default_factory=AssembleConfig)

    @field_validator("channels")
    @classmethod
    def _sane_channels(cls, items: list[ChannelConfig]) -> list[ChannelConfig]:
        """名字不能重复、类型不能空、**至少留一路启用**。

        重名会让 `--channel` 指不明白、还会让 RRF 的 `sources` 撞在一起；全禁用则等于
        检索没有数据源 —— 这两种都是"配了却不生效"那类最难查的问题，所以在配置这一步拦。
        """
        names = [item.name.strip() for item in items]
        if any(not name for name in names):
            raise ValueError("通道的 name 不能为空（日志、账本、--channel 都用它）")
        duplicated = {name for name in names if names.count(name) > 1}
        if duplicated:
            raise ValueError(f"通道名字重复：{sorted(duplicated)}（名字要能唯一定位一路）")
        if any(not item.type.strip() for item in items):
            raise ValueError("每个通道都要给 type（用哪个适配器）")
        if not any(item.enabled for item in items):
            raise ValueError("至少要启用一路通道，否则检索没有数据源")
        return items


class RedisConfig(BaseModel):
    """Redis（走 SSH 隧道连虚拟机那台）。

    两个用途：**限流计数**与**任务状态**。两条都是「多 worker 时需要共享」的东西，
    所以有 Redis 就用它、没有就退回进程内实现（`url` 留空 = 不用）——
    与本项目对「没配地址就不启用」的一贯做法一致（如 OCR 产线、云端兜底）。

    ``key_prefix`` 是必须的：这台 Redis 上可能还有别人的键，我们只用
    ``<前缀>rate:*`` 与 ``<前缀>job:*`` 两个命名空间。
    """

    # 留空 = 退回进程内实现（限流各 worker 各算、任务状态只在内存里）
    url: str = ""
    key_prefix: str = "rag:"
    # 操作与探活超时（秒）。Redis 慢不该拖垮 API —— 超时按「不可用」处理
    timeout_s: float = 2.0


class PostgresConfig(BaseModel):
    """LangGraph 的持久化后端（checkpointer + store 都用它）。

    两条用途各自解决一个明确的问题：

    * **checkpointer** —— agent 的会话状态与 **HITL 中断**。落 PG 的意义在于：中断状态
      **重启也能恢复**（内存 saver 一重启，等人批准的那个中断就没了）。
    * **store** —— 跨会话记忆（`StoreBackend` / `memory`）。

    ``url`` 留空 = 不用 PG，退回进程内（`InMemorySaver` / `InMemoryStore`）——
    与 Redis、OCR 产线同一个约定：没配就不启用，不报错、不假死。

    **schema 是我们自己管的**：库只建表、不建 schema（`PostgresSaver.setup()` 里
    只有 `CREATE TABLE IF NOT EXISTS`），而且它不认 schema 参数 —— 隔离靠连接串里的
    `options=-c search_path=<schema>`。
    """

    url: str = ""
    # 独立 schema：langgraph 那几张力表都建在这里，和别人的表隔离。
    # ⚠️ 字段名不能叫 `schema` —— pydantic 的 BaseModel 有这个属性名，会报
    # `Field name "schema" shadows an attribute in parent`（实测）。
    schema_name: str = "rag"
    # 连接池（saver 与 store **共用一套**，别开两个）
    pool_min: int = 1
    pool_max: int = 5
    # 连接与建表超时（秒）。PG 慢不该让服务起不来
    timeout_s: float = 10.0


class ApiConfig(BaseModel):
    """FastAPI 服务层。"""

    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    # 每分钟每个 key/IP 的请求上限；0 = 不限
    rate_limit_per_min: int = 60
    # 同时在跑的 ingest 任务数。重活（OCR/VLM/重排）与检索同进程争资源，
    # 默认只让一个 ingest 跑，避免把检索拖垮
    max_concurrent_ingest: int = 1
    # 对话页输入框能收的媒体上限（MB）。音频按「16kHz 单声道 wav ≈ 32KB/s」算，
    # 10MB ≈ 5 分钟；图片 5MB 是给手机直出照片留的余量（转 base64 之后还要 +33%）。
    # 超了**当场拒**，别等发到上游才以 400 回来 —— 那时候已经花了一次调用的钱。
    media_max_audio_mb: float = 10.0
    media_max_image_mb: float = 5.0
    # 图片落到哪。**不能指向 `ingest.uploads.dir`** —— 那是文档语料根，`.png/.jpg`
    # 在 loader 注册表里，丢进去会被当成一份文档列出来走五步流程。
    media_dir: str = "./data/chat/media"
    # 会话图片（语音不落盘：ASR 在内存里转完就丢）。多 worker 时改成 minio。
    media_backend: str = "auto"  # auto（听 storage.backend 那个统一开关）| local | minio
    media_prefix: str = "media"
    # 勾了「已核对」提交之后，**直接排切分 + 入库**（页面第三步提交即往下走）。
    # 为什么：四步流程里最后那次点击没有新决策 —— 切分只认核对过的清单，而它刚被核对完。
    # 关掉就回到「手动点拆分并入库」。
    # **注意代价**：切分要给每张图问 VLM、入库要算 embedding，都是**按量计费**的动作；
    # 所以只有 `confirmed=true`（人明确勾了「已核对」）才会触发。
    index_after_review: bool = True
    redis: RedisConfig = Field(default_factory=RedisConfig)


class DeploymentConfig(BaseModel):
    """部署形态。profile 约束必需的共享后端，具体值仍在各配置块里。"""

    profile: str = "single-node"  # single-node | server | scale-out
    # 外部容器/进程副本数；容器里通常 workers=1、replicas=2
    replicas: int = 1
    # 分布式灌库槽位的租约。正常释放走 finally；进程崩溃时靠它回收。
    ingest_lease_s: int = 86400


class PersistenceConfig(BaseModel):
    """持久化：**关系型的那几份数据落哪个库**（2026-10-10 统一成一个开关）。

    `backend` 只管三份「SQL 数据」：文档注册表 + 上传件台账（`documents.backend` 那一组）、
    模型选型（`models.backend`）、以及日志索引那份（`observability.logging.postgres`）。
    取值见 `rag.persistence.backend.resolve_backend`：

        pg / postgres —— PostgreSQL（要配 `RAG_PG_URL`）
        sqlite        —— 单机文件（`data/*.sqlite3`）
        auto（默认）  —— 配了 url 就用 pg，否则 sqlite（与从前一致）

    **LangGraph 的会话状态与 HITL 中断不在此列**：它只有 PG 一个实现，由
    `postgres.url` 决定（留空 = 进程内，重启即丢）。要彻底单机就把 `RAG_PG_URL` 清掉。
    """

    # 统一开关：base.yaml 里是 `${RAG_DB_BACKEND:-auto}`，值放 .env
    backend: str = "auto"

    postgres: PostgresConfig = Field(default_factory=PostgresConfig)


class GraphConfig(BaseModel):
    """QueryGraph 的自我纠错循环参数。"""

    max_corrective_rounds: int = 2
    relevance_threshold: float = 0.5
    # 编排总开关：关掉就走 M2 那条直线（`ask` 里 retrieve → assemble → generate）。
    # **默认关**，与 rerank 那次同一个流程：先落地、跑完 A/B、再决定默认值。
    # 临时开：`$env:RAG_GRAPH="true"`（base.yaml 里走插值）。
    enabled: bool = False
    # IndexGraph 同时在跑的文件数。每个文件节点都可能去调 OCR / VLM / 线上 embedding，
    # 全是外部服务 —— 无限并发只会把它们打到限流。默认 1 等于以前的行为。
    index_concurrency: int = 1


class AgentConfig(BaseModel):
    """deepagents 主 agent 的选型。

    ``chat`` 是 ``providers.chat`` 里的**键名**，不是 ``"provider:model"`` 字符串：
    后者会绕开我们的三件套 —— deepseek / zhipuai 这些没有独立 LangChain 集成包，
    写 ``"deepseek:deepseek-chat"`` 会要求额外装 ``langchain-deepseek``。指到
    ``providers.chat`` 的某个条目，model / base_url / api_key 就都由 .env 管，
    构建时走 ``build_chat()``。
    """

    # deepagents 0.5.3 起 model=None 已弃用，必须显式给一个能构建出来的 model
    chat: str = "default"
    subagents: list[str] = Field(default_factory=list)
    workspace_dir: str = "./data/workspace"
    # 给不给 agent 一个**真能落盘**的 workspace（批次 B 的功能）。
    # true => backend 的 /workspace/ 指向 workspace_dir，写类工具全部挂 HITL 闸门；
    # false => 回到批次 A 的只读 StateBackend，磁盘一个字都不动。
    workspace_writes: bool = True
    # 网络搜索闸门：语料检索的**最高分**低于这个值就算「语料里没有相关内容」，放行联网。
    # 实测（2026-10-05，bge-reranker-base）：语料内有答案 **0.9985**，语料外 **0.0077 / 0.0978**。
    # 注意：这个判据只在**开了重排**时成立（关掉重排时分数是 RRF 的名次分，量纲不同、不可比）——
    # 关着重排时闸门退化成「只有用户明确要求才放行」。
    web_min_corpus_score: float = 0.2


class FileLogConfig(BaseModel):
    """一个日志文件：路径 + 按大小轮转。**路径留空 = 这一类不落文件。**"""

    path: str = ""
    max_bytes: int = 20_000_000
    backups: int = 3


def _default_log_files() -> dict[str, FileLogConfig]:
    """业务日志的类别 → 文件（键是类别名，对应 logger `rag.<类别>`）。

    **agent 的三类是同一个文件**：一轮的时间线要在同一个文件里连着读；类别靠行里的
    logger 短名分（见 `obs/log.py` 的 `kind_of_line`）。
    """
    agent = "./data/logs/agent.log"
    return {
        "interface": FileLogConfig(path="./data/logs/interface.log"),
        "channel": FileLogConfig(path="./data/logs/channel.log"),
        "agent_run": FileLogConfig(path=agent),
        "agent_llm": FileLogConfig(path=agent),
        "agent_tool": FileLogConfig(path=agent),
    }


class LoggingConfig(BaseModel):
    """日志（三类，各落一个文件；见 `obs/log.py` 的说明）。

    **内容默认不落**：问题/查询原文里有语料正文与用户原话，属于数据而不是诊断信息。
    默认只记 hash 与长度（能对齐、能去重）；要排查时把 `content` 打开。
    开了之后原文落在**日志文件**上。

    **工具那一行是例外（2026-10-10 使用者定）**：`agent_tool` 的入参与出参**永远给内容、
    只截断**（上限见 `obs/log.py` 的 `TOOL_*_CHARS`），不受 `content` 管 —— 「入参=query=19字
    出参=12字」这种只报长度的写法，出了事没法分析。`content` 仍然管问题 / 查询原文。

    `postgres` 是**查的那一份**（2026-10-08 加）：文件仍是唯一真相，库里那份是从每行顺手
    再写一份的**索引**，用来按 `request_id` / 类别 / 时间拉。关掉它、或没配
    `persistence.postgres.url`，日志就只落文件（行为与从前完全一样）。
    """

    # 问题 / 查询原文要不要进日志行（工具那一行不吃它，见类说明）
    content: bool = False
    # 顺带把每一行写一份进 PG（`log_line` 表，与 chat 三张表同库）。
    # 写失败只 stderr 喊一句、**绝不影响业务**（见 `obs/pg.py`）。
    postgres: bool = True
    # 控制台默认开：看不见的日志等于没有日志
    console: bool = True
    # 默认 INFO；DEBUG 会把 HTTP 客户端与 LangChain 的调用细节全刷出来
    level: str = "INFO"
    # 业务日志的类别 → 文件；键就是类别名（`interface` / `channel` /
    # `agent_run` / `agent_llm` / `agent_tool`）。**同一个 path 只开一个 handler** ——
    # agent 那三类共用 agent.log。
    files: dict[str, FileLogConfig] = Field(default_factory=_default_log_files)
    # 系统日志：第三方库（LangChain / pymilvus / uvicorn…）与我们自己的非业务 rag.*
    # （**不含**上面三类，避免同一行落两份）
    system: FileLogConfig = Field(
        default_factory=lambda: FileLogConfig(path="./data/logs/system.log")
    )
    # 太吵的库默认压到 WARNING（见 `obs/setup.py` 的 NOISY），需要时在这里按模块覆盖
    levels: dict[str, str] = Field(default_factory=dict)


class ObservabilityConfig(BaseModel):
    langsmith: bool = False
    project: str = "multimodal_rag"
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


class ModelSlotConfig(BaseModel):
    """**一个模型槽位当前选了谁**（2026-10-09 定的"选型进配置"，第一步：只立结构）。

    三个字段的语义要说清，不然后面接存储时容易各理解一套：

    * `provider_key` —— 用哪一家/哪个后端。chat 那几个槽位指 `providers.chat` 的**键名**；
      rerank 指 `providers.rerank.backend`（`sentence-transformers` / `flashrank`）；
      embedding / vision / asr / ocr 目前各只有一条路，留空表示"沿用原处那份配置"。
    * `model` —— 模型名或本地路径；**留空 = 沿用原处那份配置里的值**（不改行为）。
    * `params` —— 这一槽位的参数覆盖（温度、`max_length`、超时…），留空 = 不覆盖。

    **这一步不接存储、不改任何调用方**：先把"槽位"这个概念立起来，
    让 `rag config` 能打印出来、后面再加 SQLite/PG 那份真相与界面（见 `docs/backlog.md` 第 2 条）。
    """

    provider_key: str = ""
    model: str = ""
    params: dict[str, Any] = Field(default_factory=dict)


def _default_model_slots() -> dict[str, ModelSlotConfig]:
    """八个槽位，默认**全部留空** —— 空就是"沿用今天各处那份配置"，所以行为一字不变。

    槽位名一旦定下就是库里的主键与界面上的键，**改名字要迁移**，所以这里写死一遍：
    `chat.fast`（快问答）/ `chat.agent`（agent 主模型）/ `chat.judge`（eval 与 ragas）/
    `rerank` / `embedding` / `vision` / `asr` / `ocr`。
    """
    return {name: ModelSlotConfig() for name in MODEL_SLOTS}


# 槽位的**唯一清单**（顺序即界面上展示的顺序）。加槽位 = 加一行 + 各处接上。
MODEL_SLOTS = (
    "chat.fast",
    "chat.agent",
    "chat.judge",
    "rerank",
    "embedding",
    "vision",
    "asr",
    "ocr",
)


class ModelsStorageConfig(BaseModel):
    """模型选型（见 `ModelSlotConfig`）。

    `backend` 决定**那份"当前选型"存在哪**：`auto`（默认）= 连得上 PG 就用 PG，
    否则用单机 SQLite；也可以强制 `sqlite` / `postgres`。
    **这一版还没接存储**（第一步只立结构），字段先摆在这儿，
    接的时候按 `docs/backlog.md` 第 2 条那四条口径来（审计要、优先级 库 > .env > yaml）。
    """

    backend: str = "auto"
    # 单机版那份 SQLite 放哪（`auto` 又没配 PG 时用它；与 data/logs、data/stages 同级）
    sqlite_path: str = "./data/model_settings.sqlite3"
    slots: dict[str, ModelSlotConfig] = Field(default_factory=_default_model_slots)


class MinioConfig(BaseModel):
    """对象存储（MinIO）。**多 worker 模式**下上传件 / 产物 / 图片 / 会话图片放这里。

    单机不配它也能跑 —— 统一开关 `storage.backend` 默认是 `local`（本地盘），
    只有把它（或某一域）改成 `minio` 才会读这一段。这与 Redis / OCR 产线同一个约定：
    没配就不启用，不报错、不假死。

    **地址与桶名在 yaml，密钥在 `.env`**（`MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY`）——
    与 `persistence.postgres` 那条规矩一致：机密不进配置文件。
    """

    endpoint: str = ""
    bucket: str = "rag"
    # 自建 MinIO 多半是 http；走 https 才置 true
    secure: bool = False
    region: str = ""
    # 只认 Path 的消费方（loader / PyMuPDF / OCR）走这里的读穿透缓存：桶是真相，缓存是副本
    cache_dir: str = "./data/cache/blobs"


class StorageConfig(BaseModel):
    """文件落点。**每个域一个开关**（`ingest.uploads` / `ingest.stages` / `ingest.assets` /
    `api.media` 各写自己的 `backend`），所以单机与多 worker 是同一份代码、同一套配置结构。

    **一个统一开关管四个域**（2026-10-10）：`storage.backend`（base.yaml 里声明成
    `${RAG_STORAGE_BACKEND:-local}`，值放 `.env`）。那四个域的 `backend` 默认写 `auto`
    （= 听统一开关），想单独钉死某一个域（比如产物进桶、上传件留本地盘）就把它写回
    `local` / `minio`。优先级：**域显式写的 > `storage.backend` > `local`**，判据的唯一实现在
    `rag/storage.py` 的 `resolve_storage_backend()`。
    """

    # 统一开关：local（默认）| minio
    backend: str = "local"

    minio: MinioConfig = Field(default_factory=MinioConfig)


class DocumentsConfig(BaseModel):
    """文档注册表（每篇一行：走到哪一步 + 入库信息）落在哪。

    2026-10-09 使用者定：**从 Milvus 搬到数据库** —— 文档列表要回答"这份到哪一步了"，
    那是 O(文档数) 的关系型数据，不该去问向量库（Milvus 一抖，实测查询等 9 秒才 503，
    页面就白等）。`backend` 与 `models.backend` 同一套选法：
    `auto`（默认）= 配了 PG 就用 PG，否则单机 SQLite；也可以强制 `sqlite` / `postgres`。
    """

    backend: str = "auto"
    # 单机版那份 SQLite 放哪（`auto` 又没配 PG 时用它；与 data/logs、data/stages 同级）
    sqlite_path: str = "./data/documents.sqlite3"


class Settings(BaseModel):
    """全量配置树的根。"""

    storage: StorageConfig = Field(default_factory=StorageConfig)
    documents: DocumentsConfig = Field(default_factory=DocumentsConfig)
    models: ModelsStorageConfig = Field(default_factory=ModelsStorageConfig)

    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
