"""看图说话的模型（VLM）：图片字节 + 上下文 → 一段可检索的描述。

与 ``providers/chat.py`` 的区别只有两个：走的是**多模态消息**（``image_url``
里塞 base64 data URI），以及它用的是 ``DASHSCOPE_VL_*`` 那套三件套。其余照抄：
国产供应商都提供 OpenAI 兼容的 ``/v1/chat/completions``，所以这里用 httpx 直接
打它，不引 LangChain 的 chat 模型 —— 图片描述的调用点只有一处、也不需要流式与
工具调用，引入一层封装只会让「请求到底长什么样」更难看清。

**它按量计费**（每张图一次调用），所以开关默认关（``ingest.images.describe``），
且描述要按图片内容哈希缓存 —— 见 ``ingest/images.py``。
"""

from __future__ import annotations

import base64
from functools import cached_property
from typing import Any

import httpx

from rag.config.schema import VISION_PROMPT_VERSION, Settings, VisionProviderConfig
from rag.config.settings import Secrets

# 供应商名 → 取 key 的环境变量，按优先级排列（前面为空就退到后面）。
# 回退到 chat 那把 key 是因为 DashScope 的 key 是账号级的：同一个 qwen-plus 与
# qwen-vl-max 共用一把；单列 VL 变量只是给「VL 换账号」留个口子。
VISION_KEY_ENV: dict[str, tuple[str, ...]] = {
    "dashscope": ("DASHSCOPE_VL_API_KEY", "DASHSCOPE_API_KEY"),
    "openai": ("OPENAI_VL_API_KEY", "OPENAI_API_KEY"),
    "zhipuai": ("ZHIPUAI_VL_API_KEY", "ZHIPUAI_API_KEY"),
    "moonshot": ("MOONSHOT_VL_API_KEY", "MOONSHOT_API_KEY"),
}

_CHAT_PATH = "/chat/completions"

SYSTEM_PROMPT = """你是文档解析助手，负责为文档中的插图写一段**可被检索到的描述**。

**第一步：判断。** 给到的上下文（图上/下方的段落）是否在讲这张图？
输出**第一行**只写一个词：`相关` 或 `不相关`（拿不准就写 `不相关`）。这一行不要写别的。

（空一行）

**第二步：描述。** 只按图本身写，从头到尾**不许出现任何只有上文才知道的名字** ——
图上画的是 Flink 的图标，就不能写成「Kafka 图标」，哪怕上文通篇在讲 Kafka。
按下面的顺序，用连续的短句，不要 markdown 标题、不要分点符号：

1. 图是什么类型（架构图/流程图/时序图/表格/折线图/柱状图/截图/公式/照片等）；
2. 图里有哪些元素、它们之间的关系与流向；
3. 图中文字：表格转成 markdown 表格；代码与公式保留原样；其余文字逐条转述 ——
   标题、标签、字段名、数值都是检索的主要入口，必须尽量原样写出来。

**第三步：** 第一步判断为「相关」时，在描述最后**再加一句**，以「结合上文：」开头，
把上文说的那件事概括进来（该说的名字、结论要写出来，这部分也要能被检索到）。
判断为「不相关」就到此为止：**一个字都不要提上下文**，也不要写「与上文无关」这类话 ——
描述要进检索库，写进去的每个词都会变成命中入口。

底线：只写图上**确实有**的东西，看不清、或图上没有的一律不写，不要用常识补全，\
不要推测作者意图；颜色、数值、大小这类细节拿不准就不写（写错比不写更糟）。

长度控制在 400 字以内（表格可以超）。"""

# 描述正文里「来自上下文」那一句的标记。判断为「不相关」时，代码按它把这一句**切掉** ——
# 提示词里说了「不相关就别写」，但实测 qwen-vl-max 仍会补一句（甚至把上文原样抄进来）。
# 排版这种事交给代码做才靠得住，模型只负责「判断」这一件它做得动的事。
_CONTEXT_MARK = "结合上文"

# 判断行短得像一个词才认；正文里的句子（「这张图与上下文相关。」）不算，别错切。
_DECISION_MAX = 8
_NEGATIVE = ("不相关", "无关")
_POSITIVE = ("相关",)

# 按魔数判类型，判不出再按后缀。data URI 的 mime 写错时有些网关会直接 400，
# 而资产文件名是我们自己起的，不值得在这里赌。
_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
    (b"BM", "image/bmp"),
)

_SUFFIX_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".bmp": "image/bmp",
}


class VisionError(RuntimeError):
    """VLM 调用失败（连不上 / 鉴权失败 / 报文不对 / 返回空描述）。"""


def mime_of(data: bytes, name: str = "") -> str:
    """判断图片类型：先魔数，再后缀，都不认就当作 jpeg。"""
    for magic, mime in _MAGIC:
        if data.startswith(magic):
            return mime
    dot = name.rfind(".")
    return _SUFFIX_MIME.get(name[dot:].lower(), "image/jpeg") if dot >= 0 else "image/jpeg"


class VisionClient:
    """同步的 VLM 客户端。切块这一层本来就是同步的，不必为它引 async。"""

    def __init__(self, cfg: VisionProviderConfig, api_key: str) -> None:
        self._cfg = cfg
        self._api_key = api_key

    @property
    def base_url(self) -> str:
        return self._cfg.base_url.rstrip("/")

    @property
    def model(self) -> str:
        return self._cfg.model

    @property
    def fingerprint(self) -> str:
        """进缓存键：模型或提示词一变，旧描述就不该再被复用。"""
        return f"{self._cfg.model}@{VISION_PROMPT_VERSION}"

    @cached_property
    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            timeout=self._cfg.timeout_s,
            headers={"Authorization": f"Bearer {self._api_key}"},
            # 与 OCR 客户端同一个立场：httpx 的 retries 只管连接层（连不上/握手失败），
            # 超时与 5xx 不重试 —— 那多半是服务真在处理，重试只会更慢、还可能重复计费。
            transport=httpx.HTTPTransport(retries=self._cfg.max_retries),
        )

    def describe(self, data: bytes, *, before: str = "", after: str = "", name: str = "") -> str:
        """一张图（字节）+ 上下文 → 描述文本。"""
        payload: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _user_text(before, after)},
                        {"type": "image_url", "image_url": {"url": _data_uri(data, name)}},
                    ],
                },
            ],
        }
        try:
            resp = self._client.post(_CHAT_PATH, json=payload)
        except httpx.HTTPError as exc:
            raise VisionError(
                f"连不上 VLM {self.base_url}（{type(exc).__name__}: {exc}）。"
                f"确认 .env 里的 DASHSCOPE_VL_BASE_URL 与网络。"
            ) from exc
        return _content(resp, model=self._cfg.model)


def _user_text(before: str, after: str) -> str:
    """上下文拼在图片前面 —— 与切块时「标题贴开头」同一个理由：先读到它在讲哪件事。"""
    parts = ["请描述这张图。"]
    if before:
        parts.append(f"图上方的段落：{before}")
    if after:
        parts.append(f"图下方的段落：{after}")
    return "\n".join(parts)


def _data_uri(data: bytes, name: str) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_of(data, name)};base64,{encoded}"


def _content(resp: httpx.Response, *, model: str) -> str:
    body = _json_or_none(resp)
    if body is None:
        raise VisionError(f"VLM({model}) 没返回 JSON：HTTP {resp.status_code} {_preview(resp)}")
    if resp.is_error:
        # 报文里的 error.message 比状态码有用得多（模型名写错、欠费、限流都在这里）
        error = body.get("error")
        detail = error.get("message") if isinstance(error, dict) else str(error or "")
        raise VisionError(f"VLM({model}) 返回 HTTP {resp.status_code}：{detail or _preview(resp)}")
    choices = body.get("choices") or []
    if not choices:
        raise VisionError(f"VLM({model}) 的响应里没有 choices：{_preview(resp)}")
    text = _text_of((choices[0].get("message") or {}).get("content"))
    if not text:
        # 空描述写进库等于「这张图没内容」，比报错更难查
        raise VisionError(f"VLM({model}) 返回了空描述：{_preview(resp)}")
    return clean_description(text)


def clean_description(text: str) -> str:
    """把判断行摘掉；判断为「不相关」时，再把结尾那句「结合上文：…」切掉。

    为什么要代码来收这一刀：提示词里写明了「不相关就一个字都不要提上下文」，实测
    qwen-vl-max 还是会补一句（有几次甚至把上文原文抄进描述）。那些词一旦进库就变成
    **假的检索入口** —— 问 pandas 会命中一张 Flink 架构图。所以模型只负责判断（它做得动），
    排版与取舍由代码执行。

    取不到判断行时**原样保留**：宁可多留一句，也不要按猜测去切用户的描述。
    """
    lines = text.split("\n")
    head = next((i for i, line in enumerate(lines) if line.strip()), None)
    if head is None:
        return text
    decision = lines[head].strip().strip("*`# ：:")
    related = None
    if len(decision) <= _DECISION_MAX and any(word in decision for word in _NEGATIVE):
        related = False
    elif len(decision) <= _DECISION_MAX and any(word in decision for word in _POSITIVE):
        related = True
    if related is None:
        return text
    body = "\n".join(lines[head + 1 :]).strip()
    if not body:
        # 判断行之后什么都没写 —— 那是模型的输出有问题，别把整条描述也丢掉
        return text
    if not related:
        cut = body.find(_CONTEXT_MARK)
        if cut >= 0:
            # 切完就空了（模型只写了一句「结合上文」）→ 留着，别返回空描述：
            # 一张图没有任何描述文字，比起描述里多一句上文，是更坏的结局。
            body = body[:cut].strip() or body
    return body


def _text_of(content: Any) -> str:
    """content 可能是字符串，也可能是 ``[{"type": "text", "text": ...}]``。"""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") in ("text", "output_text")
        ]
        return "".join(parts).strip()
    return ""


def _json_or_none(resp: httpx.Response) -> dict[str, Any] | None:
    try:
        body = resp.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def _preview(resp: httpx.Response, width: int = 200) -> str:
    return " ".join(resp.text.split())[:width]


def build_vision(settings: Settings, secrets: Secrets | None = None) -> VisionClient:
    """按 ``providers.vision`` 建一个 VLM 客户端。

    与 ``build_ocr`` 不同，这里**不做「没配就不启用」的降级**：调用方只有在
    ``ingest.images.describe`` 打开时才会建它，建了就说明要真的调 —— 缺 key 或
    缺地址应当当场报错，而不是默默少掉所有图片描述。
    """
    cfg = settings.providers.vision
    if not cfg.base_url.strip():
        raise ValueError(
            "providers.vision.base_url 为空：VLM 走 OpenAI 兼容通道，必须有地址。"
            "在 .env 里填 DASHSCOPE_VL_BASE_URL（或复用 DASHSCOPE_BASE_URL）。"
        )
    if not cfg.model.strip():
        raise ValueError("providers.vision.model 为空：在 .env 里填 DASHSCOPE_VL_MODEL_NAME。")
    return VisionClient(cfg, _require_key(cfg, secrets or _secrets()))


def _secrets() -> Secrets:
    # 延迟 import：providers 这一层在 import 时不该连带把 .env 读进来
    from rag.config import get_secrets

    return get_secrets()


def _require_key(cfg: VisionProviderConfig, secrets: Secrets) -> str:
    names = VISION_KEY_ENV.get(cfg.provider)
    if names is None:
        raise KeyError(
            f"未知的 vision provider: {cfg.provider!r}；已知: {sorted(VISION_KEY_ENV)}。"
            f"新增供应商时要在 providers/vision.py 的 VISION_KEY_ENV 里登记它的 key 变量名"
        )
    for name in names:
        key = getattr(secrets, name.lower(), None)
        if key:
            return key
    raise RuntimeError(
        f"缺少 {names[0]}：在项目根目录的 .env 里填上它"
        f"（也可以直接填 {names[-1]}，两者是同一个账号的同一把 key）。"
    )
