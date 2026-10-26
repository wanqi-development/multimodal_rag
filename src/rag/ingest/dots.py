"""dots.ocr 客户端：PDF 的另一个解析后端（vLLM 上的 dots.ocr）。

**与 PaddleOCR 那条线的关系**：产线的 `relevelTitles` 只按版面标签猜标题层级，
实测在这些教材 PDF 上基本不可用（章、节、列表项标成同一级，路径会断）；
dots.ocr 是端到端的版面模型，输出的布局块**自带 markdown 层级**
（`Section-header` 块的 `text` 直接就是 `# 1第一章…` / `## 1.1 …` / `### 1.1.1 …`），
另外每个块都带 `bbox`。所以这一版把它接成 PDF 的解析后端。

接口与 ``ingest/ocr.py::OcrClient`` **完全一致**（``parse(path, file_type) -> list[OcrPage]``），
所以 loader 与下游（切分 / 定位 / 图片描述 / 灌库）一行都不用改。

实现复用了可选插件 ``plugins/rag-dots-parser`` 里的 dots.ocr 官方样例，只做三件事：

1. 按页渲染 PDF（复用 `load_images_from_pdf`）；
2. 逐页调用 vLLM（复用 `inference_with_vllm` + `post_process_output`），**页之间并发** ——
   一页一次调用、几十页串行会很慢，样例里也是用线程池；
3. 用 `layoutjson2md` 把布局块拼成 markdown（图片会内联成 `data:` base64，
   由 ``ingest/assets.py`` 挪到资产目录、正文里换成短标记）。
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

from rag.config.schema import IngestConfig, IngestDotsConfig
from rag.ingest.ocr_client import OcrEngine, OcrPage, register_ocr

try:
    from rag_dots_parser.vendor.dots_ocr.model.inference import inference_with_vllm
    from rag_dots_parser.vendor.dots_ocr.utils.doc_utils import load_images_from_pdf
    from rag_dots_parser.vendor.dots_ocr.utils.format_transformer import layoutjson2md
    from rag_dots_parser.vendor.dots_ocr.utils.image_utils import fetch_image
    from rag_dots_parser.vendor.dots_ocr.utils.layout_utils import post_process_output
    from rag_dots_parser.vendor.dots_ocr.utils.prompts import dict_promptmode_to_prompt

    _IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # 核心包不安装可选的 dots parser 也能启动
    inference_with_vllm = None
    load_images_from_pdf = None
    layoutjson2md = None
    fetch_image = None
    post_process_output = None
    dict_promptmode_to_prompt: dict[str, str] = {}
    _IMPORT_ERROR = exc

PARSER_NAME = "dots"


class DotsOcrError(RuntimeError):
    """dots.ocr 调用失败（服务连不上 / 某页返回空 / 页数对不上）。"""


class DotsOcrClient:
    """vLLM 上的 dots.ocr。同步接口，页内并发。"""

    def __init__(self, cfg: IngestDotsConfig) -> None:
        if _IMPORT_ERROR is not None:
            raise DotsOcrError(
                "dots.ocr 解析器是可选插件；先运行 `pip install -e plugins/rag-dots-parser`"
            ) from _IMPORT_ERROR
        self._cfg = cfg
        parts = urlsplit(cfg.base_url)
        if not parts.hostname:
            raise ValueError(f"ingest.dots.base_url 不是合法地址：{cfg.base_url!r}")
        self._protocol = parts.scheme or "http"
        self._host = parts.hostname
        self._port = parts.port or (443 if self._protocol == "https" else 80)
        if cfg.prompt_mode not in dict_promptmode_to_prompt:
            raise ValueError(
                f"未知的 ingest.dots.prompt_mode: {cfg.prompt_mode!r}；"
                f"可选 {sorted(dict_promptmode_to_prompt)}"
            )

    @property
    def name(self) -> str:
        return PARSER_NAME

    @property
    def base_url(self) -> str:
        return self._cfg.base_url.rstrip("/")

    @property
    def model(self) -> str:
        return self._cfg.model

    def parse(self, path: Path, *, file_type: int) -> list[OcrPage]:
        """``file_type`` 用不上：dots.ocr 的输入就是一张张页面图（本方法只吃 PDF）。"""
        images = load_images_from_pdf(str(path), dpi=self._cfg.dpi)
        if not images:
            raise DotsOcrError(f"{path.name} 渲染出 0 页")
        workers = max(1, min(len(images), self._cfg.num_thread))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            markdowns = list(pool.map(self._parse_page, images))
        return [
            OcrPage(page_index=index, markdown=markdown) for index, markdown in enumerate(markdowns)
        ]

    def _parse_page(self, image: object) -> str:
        """一页 → markdown。失败会重试几次；重试完仍失败就抛（整份失败）。

        **不吞页**：少几页而文档照常入库，是最难查的那种错。但也不能一页失败就丢掉
        整份 —— 几百页的 PDF 过一条不稳的隧道，单次抖动很常见（实测这条 AutoDL 转发
        在真实负载下有过 28% 的失败率）。所以这里重试，重试完还不行才让整份失败。
        """
        last: Exception | None = None
        for attempt in range(self._cfg.max_retries + 1):
            try:
                return self._parse_page_once(image)
            except Exception as exc:  # noqa: BLE001 —— 见下：重试用完才让整份失败
                # 这里刻意接住所有异常：调用是「一页一次网络请求」，超时/连接重置/
                # 报文异常都只该让**这一页**重试，不该让另外 395 页白跑。
                last = exc
                if attempt < self._cfg.max_retries:
                    print(
                        f"⚠ dots.ocr 这一页失败（第 {attempt + 1} 次，{type(exc).__name__}），"
                        f"重试：{exc}",
                        file=sys.stderr,
                    )
        raise DotsOcrError(f"重试 {self._cfg.max_retries} 次仍失败：{type(last).__name__}: {last}")

    def _parse_page_once(self, image: object) -> str:
        prompt = dict_promptmode_to_prompt[self._cfg.prompt_mode]
        page = fetch_image(image)
        response = inference_with_vllm(
            page,
            prompt,
            protocol=self._protocol,
            ip=self._host,
            port=self._port,
            temperature=self._cfg.temperature,
            top_p=self._cfg.top_p,
            max_completion_tokens=self._cfg.max_completion_tokens,
            model_name=self._cfg.model,
        )
        if not response:
            raise DotsOcrError(
                f"dots.ocr 返回空响应（{self.base_url}，模型 {self._cfg.model}）。"
                f"确认服务在跑、模型名对得上。"
            )
        cells, json_failed = post_process_output(response, self._cfg.prompt_mode, image, page)
        if json_failed:
            # 模型没按 JSON 输出：与样例一致，把清洗后的文本原样当 markdown（至少不丢内容）
            print("⚠ dots.ocr 这一页的布局 JSON 解析失败，按纯文本收下", file=sys.stderr)
            return str(cells)
        return layoutjson2md(image, cells, text_key="text")


def build_dots(cfg: IngestDotsConfig) -> DotsOcrClient:
    return DotsOcrClient(cfg)


@register_ocr("dots")
def build_dots_engine(ingest: IngestConfig, secrets: object | None = None) -> OcrEngine:
    """dots.ocr 的工厂入口（vLLM 的 OpenAI 兼容接口，本地服务不要 key）。

    它给的是**标题层级**（`Section-header` 块自带 `#`/`##`/`###`），实测在教材类 PDF 上
    基本可用，人只需修少数几行；页码靠页序推（一页一次调用）。
    """
    return build_dots(ingest.dots)


def prompt_modes() -> Sequence[str]:
    """给配置文档/报错用的可选 prompt 列表。"""
    return sorted(dict_promptmode_to_prompt)
