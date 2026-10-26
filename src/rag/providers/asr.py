"""语音转文字（ASR）：一段音频字节 → 一段文字。

**为什么单独一个客户端，不塞进 ``vision.py``**：它是另一族模型
（``qwen-audio-*-asr-*``，vision 是 ``qwen-vl-*``），**约定却完全一样** ——
OpenAI 兼容通道 + key / base_url / model 三件套。放一起会让「看图」和「听音」的开关、
超时、重试混在同一个类里，而这两件事的失败模式并不一样。

**调用形态：DashScope 原生端点，不是 OpenAI 兼容那条**。这一条是拿真调用换来的
（2026-10-08）—— 第一版照 ``vision.py`` 抄了兼容模式的 ``/chat/completions``，上游回
``{"message":"format is empty","type":"UNSUPPORTED_FORMAT"}``：兼容通道读的是
``input_audio.format``，而原生端点把 ``format`` / ``sample_rate`` 放在 ``parameters`` 里：

```json
{"model": "qwen-audio-3.0-asr-flash",
 "input": {"messages": [{"role": "user", "content": [
     {"type": "input_audio", "input_audio": {"data": "data:audio/wav;base64,…"}}]}]},
 "parameters": {"format": "wav", "sample_rate": "16000"}}
```

响应也是原生的那套：文字在 ``output.choices[0].message.content[0].text``
（兼容通道那条 ``choices[0].message.content`` 也认，见 ``_content``），错误则是
``{"code", "message", "request_id"}``。

``sample_rate`` **从 wav 头里读真实值**，不写死：前端固定录 16kHz，但「从别处拿一段 wav」
时写死会让整段音频变调 —— 那种错很难查。

**格式这件事的实际取舍**：浏览器 ``MediaRecorder`` 默认出的是 ``webm/opus``，而 ASR
那一侧对 webm 没有保证。所以**前端不录 webm**，改成自己采 PCM、编 16kHz 单声道 wav
（见 ``ChatView.vue`` 的录音那一段）—— 这种 wav 是所有 ASR 的一等公民，还能把体积压住
（≈32KB/s，10MB 够 5 分钟）。``format_of`` 仍然认一串后缀，是给「以后从别处拿音频」留的。

**按量计费**：每段音频一次调用，所以调用点只有一处（对话页那个麦克风），且**默认不重试**。
"""

from __future__ import annotations

import base64
from functools import cached_property
from pathlib import Path
from typing import Any

import httpx

from rag.config.schema import AsrProviderConfig, Settings
from rag.config.settings import Secrets

# 供应商名 → 取 key 的环境变量，按优先级排列（前面为空就退到后面）。与 vision 同一个约定：
# 回退到 DASHSCOPE_API_KEY 是因为 DashScope 的 key 是账号级的。
ASR_KEY_ENV: dict[str, tuple[str, ...]] = {
    "dashscope": ("DASHSCOPE_ASR_API_KEY", "DASHSCOPE_API_KEY"),
    "openai": ("OPENAI_ASR_API_KEY", "OPENAI_API_KEY"),
}

# DashScope **原生**端点（不是 OpenAI 兼容那条）。官方样例给的就是它。
_PATH = "/api/v1/services/aigc/multimodal-generation/generation"

# 认得的音频后缀。**前端录的是 wav**（16kHz 单声道），其余是给「从别处拿音频」留的口子。
FORMATS = ("wav", "mp3", "m4a", "aac", "ogg", "opus", "flac", "webm", "amr")

# Content-Type → 格式。**不能靠「后缀在 mime 里出现过」**：`audio/mpeg` 里没有 "mp3"、
# `audio/mp4` 里也没有 "m4a"（浏览器给 Safari 录的音就是这个），只有这张表认得出来。
BY_MIME: dict[str, str] = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/vnd.wave": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mp4": "m4a",
    "audio/x-m4a": "m4a",
    "audio/aac": "aac",
    "audio/ogg": "ogg",
    "audio/opus": "opus",
    "audio/flac": "flac",
    "audio/x-flac": "flac",
    "audio/webm": "webm",
    "audio/amr": "amr",
}


class AsrError(RuntimeError):
    """识别失败（格式认不出 / 服务连不上 / 返回不是 JSON / 返回里没有文字）。

    **要带上上游的原话**：这里最常见的失败是「模型名或请求体形状不对」，
    上游的 400 里通常直接说了缺什么 —— 吞掉它，人就得靠猜。
    """


class AsrNoSpeechError(AsrError):
    """**这段音频里没有说话**（上游回 `ASR_RESPONSE_HAVE_NO_WORDS`）。

    单独一类是为了状态码：这不是「上游挂了」（502），是「这次输入没内容」（422）。
    静音、麦克风没选中、离得太远都会走到这里 —— 实测拿 1 秒静音打过去就是这个码。
    """


# 上游那个「没识别到人声」的码。按它翻译成人话，原码照样附在后面（排查时要用）
_NO_WORDS = "ASR_RESPONSE_HAVE_NO_WORDS"


def format_of(name: str, content_type: str = "") -> str:
    """按文件名 / Content-Type 定格式。认不出来就报错 —— 猜错是一次 400 加一次计费。"""
    suffix = Path(name or "").suffix.lstrip(".").lower()
    if suffix in FORMATS:
        return suffix
    kind = (content_type or "").split(";")[0].strip().lower()
    if kind in BY_MIME:
        return BY_MIME[kind]
    for candidate in FORMATS:
        if kind.endswith(candidate):
            return candidate
    raise AsrError(
        f"认不出的音频格式：{name or '(没有文件名)'} / {content_type or '(没有类型)'}；"
        f"能收 {list(FORMATS)}"
    )


class AsrClient:
    """DashScope 原生 multimodal-generation 上的语音识别。同步接口（API 侧丢线程池调）。"""

    def __init__(self, cfg: AsrProviderConfig, api_key: str) -> None:
        self._cfg = cfg
        self._api_key = api_key

    @property
    def model(self) -> str:
        return self._cfg.model

    @property
    def base_url(self) -> str:
        return self._cfg.base_url.rstrip("/")

    @cached_property
    def _client(self) -> httpx.Client:
        return httpx.Client(
            timeout=self._cfg.timeout_s,
            transport=httpx.HTTPTransport(retries=max(0, self._cfg.max_retries)),
        )

    def transcribe(self, data: bytes, *, name: str = "", content_type: str = "") -> str:
        """一段音频 → 一段文字。空结果**当错误**：它多半意味着没录上，不该静默填个空串。"""
        fmt = format_of(name, content_type)
        parameters = {"format": fmt}
        if fmt == "wav" and (rate := wav_sample_rate(data)):
            # **从 wav 头里读真实值**，不写死 16000：写死时，一段 48kHz 的 wav 会被当成
            # 16kHz 解出来 —— 整段变调、还听不出是配置错，很难查
            parameters["sample_rate"] = str(rate)
        body = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {"data": _data_uri(data, fmt)},
                            }
                        ],
                    }
                ]
            },
            "parameters": parameters,
        }
        try:
            resp = self._client.post(
                f"{self.base_url}{_PATH}",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    # 官方样例带的头：一次性拿结果，不要 SSE
                    "X-DashScope-SSE": "disable",
                },
                json=body,
            )
        except httpx.HTTPError as exc:
            raise AsrError(f"ASR({self.model}) 请求失败：{type(exc).__name__}: {exc}") from exc
        if resp.status_code != 200:
            reason = _reason(resp)
            if _NO_WORDS in reason:
                raise AsrNoSpeechError(
                    f"没听清：这段音频里没识别到说话（上游 {_NO_WORDS}）—— 凑近点、再说一遍"
                )
            raise AsrError(f"ASR({self.model}) 返回 HTTP {resp.status_code}：{reason}")
        text = _content(resp, model=self.model).strip()
        if not text:
            raise AsrError(f"ASR({self.model}) 返回了空文字（录上了吗？）")
        return text


def _data_uri(data: bytes, fmt: str) -> str:
    """base64 的 data URI —— DashScope 收「公网 URL 或 base64」，base64 走的就是这个形态
    （与 vision 那边 `image_url.url` 同一个写法）。

    **音频格式不靠它**：`format` / `sample_rate` 走 `parameters`，官方样例就是这么分的。
    第一版把格式塞进 `input_audio` 是照兼容通道抄的，真调用回的是
    `{"message":"format is empty","type":"UNSUPPORTED_FORMAT"}`。
    """
    return f"data:audio/{fmt};base64,{base64.b64encode(data).decode('ascii')}"


def wav_sample_rate(data: bytes) -> int:
    """读 wav 头里的采样率（字节 24–27）；不是 wav / 头不完整就返回 0（调用方不传这个参数）。"""
    if len(data) < 28 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return 0
    return int.from_bytes(data[24:28], "little")


def _content(resp: httpx.Response, *, model: str) -> str:
    payload = _json_or_none(resp)
    if payload is None:
        raise AsrError(f"ASR({model}) 没返回 JSON：HTTP {resp.status_code} {_preview(resp)}")
    return _extract_text(payload)


def _extract_text(payload: dict[str, Any]) -> str:
    """从**三种**已知形状里取文字。多一种不亏，猜错一次就是一轮往返（实测两轮）。

    1. chat 那套：`output.choices[…].message.content`（多模态端点）/ 顶层 `choices`（兼容通道）；
    2. **ASR 的句子形状**（`qwen-audio-3.0-asr-flash` 实测）：
       `{"output": {"output": {"sentence": {"text": "Spring AI是什么？怎么用。"}}}}`
       —— 注意是**两层 `output`**，文字在 `sentence.text` 里，`words` 是逐字时间戳；
    3. 万一只剩一个裸 `text`。
    """
    nodes: list[Any] = [payload]
    output = payload.get("output")
    if isinstance(output, dict):
        nodes.append(output)
        if isinstance(output.get("output"), dict):
            nodes.append(output["output"])
    for node in nodes:
        if not isinstance(node, dict):
            continue
        choices = node.get("choices")
        if isinstance(choices, list) and choices:
            text = _text_of((choices[0].get("message") or {}).get("content"))
            if text.strip():
                return text
    # `sentence` 那条要放在 choices 后面试：多模态那条的 `content` 里也可能带 text 片段
    text = _sentences_text(payload)
    if text.strip():
        return text
    bare = payload.get("text")
    return str(bare) if isinstance(bare, str) else ""


def _sentences_text(node: Any, depth: int = 0) -> str:
    """把 `sentence` / `sentences` 里的文字按顺序拼起来。

    只认到带 `text` 的那一层就**返回**，不再往下走 —— 再走就进 `words` 了，
    那是逐字时间戳（`"text": "Spring"` 这种），拼出来的是碎片。
    """
    if depth > 4:
        return ""
    if isinstance(node, list):
        return "".join(_sentences_text(item, depth + 1) for item in node)
    if not isinstance(node, dict):
        return ""
    text = node.get("text")
    if isinstance(text, str) and text.strip():
        return text
    return "".join(
        _sentences_text(node[key], depth + 1)
        for key in ("sentences", "sentence", "output", "result")
        if key in node
    )


def _reason(resp: httpx.Response, width: int = 300) -> str:
    """错误原话。原生是 `{code, message, request_id}`，兼容通道是 `{error: {message}}`。"""
    payload = _json_or_none(resp)
    text = ""
    if payload is not None:
        error = payload.get("error")
        if isinstance(error, dict):
            text = str(error.get("message") or error)
        else:
            text = f"{payload.get('code') or ''} {payload.get('message') or ''}".strip()
    if not text:
        text = (resp.text or "").strip().replace("\n", " ")
    return text if len(text) <= width else text[:width] + "…"


def _text_of(content: Any) -> str:
    """`content` 可能是字符串，也可能是片段列表 —— 两种都收（不同的兼容实现给的不一样）。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
    return ""


def _json_or_none(resp: httpx.Response) -> dict[str, Any] | None:
    try:
        payload = resp.json()
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _preview(resp: httpx.Response, width: int = 300) -> str:
    text = (resp.text or "").strip().replace("\n", " ")
    return text if len(text) <= width else text[:width] + "…"


def build_asr(settings: Settings, secrets: Secrets | None = None) -> AsrClient:
    """按 ``providers.asr`` 建一个 ASR 客户端。

    **不做「没配就不启用」的降级**（与 `build_vision` 同一个立场）：只有真的要用它才会建，
    建了就说明要调 —— 缺 key 或地址当场报错，好过让人对着一个空输入框猜。
    """
    cfg = settings.providers.asr
    if not cfg.base_url.strip():
        raise ValueError(
            "providers.asr.base_url 为空：ASR 走 OpenAI 兼容通道，必须有地址。"
            "在 .env 里填 DASHSCOPE_ASR_BASE_URL（或复用 DASHSCOPE_BASE_URL）。"
        )
    if not cfg.model.strip():
        raise ValueError("providers.asr.model 为空：在 .env 里填 DASHSCOPE_ASR_MODEL_NAME。")
    return AsrClient(cfg, _require_key(cfg, secrets or _secrets()))


def _secrets() -> Secrets:
    """没显式传 Secrets 时现场读一次 `.env`（CLI / 脚本那条路）。"""
    return Secrets()


def _require_key(cfg: AsrProviderConfig, secrets: Secrets) -> str:
    names = ASR_KEY_ENV.get(cfg.provider, ("DASHSCOPE_ASR_API_KEY",))
    for name in names:
        value = getattr(secrets, name.lower(), None)
        if value:
            return str(value)
    raise ValueError(
        f"providers.asr.provider={cfg.provider!r} 缺 key：在 .env 里填 {names[0]}"
        f"（留空会自动回退到 {'、'.join(names[1:]) or '——'}）。"
    )
