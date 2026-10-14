"""语音转文字（ASR）：客户端 + `POST /chat/transcribe`。

**一个真调用都不打**：httpx 的 `MockTransport` 假装 DashScope，请求体形状（`input_audio`
里那个 base64 data URI）由用例断言钉住 —— 那是这条链路上最容易写错、又只有真调用才会
暴露的一处。上游的错误原话必须透出来，否则模型名/请求体写错时人只能靠猜。
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from rag.api.app import Components, create_app
from rag.config.schema import AsrProviderConfig, Settings
from rag.config.settings import Secrets
from rag.providers.asr import AsrClient, AsrError, AsrNoSpeechError, build_asr, format_of


def _client(handler) -> AsrClient:
    """一个把 `_client` 换成假通道的 AsrClient（`cached_property` 存在 `__dict__` 里）。"""
    client = AsrClient(AsrProviderConfig(), "fake-key")
    client.__dict__["_client"] = httpx.Client(transport=httpx.MockTransport(handler))
    return client


# ---- 格式识别 ----


def test_the_format_comes_from_the_name_or_the_content_type() -> None:
    assert format_of("a.wav") == "wav"
    assert format_of("录音.WAV") == "wav"
    assert format_of("", "audio/mpeg") == "mp3"
    assert format_of("blob", "audio/webm;codecs=opus") == "webm"


def test_an_unknown_format_is_rejected_before_any_call() -> None:
    """认不出就报错：猜错是一次 400 **加**一次计费。"""
    with pytest.raises(AsrError, match="认不出的音频格式"):
        format_of("a.txt", "text/plain")


# ---- 客户端 ----


def _wav(rate: int = 16000) -> bytes:
    """一个只有 44 字节头的空 wav —— 能读出采样率就够。"""
    head = bytearray(b"RIFF" + (36).to_bytes(4, "little") + b"WAVE")
    head += b"fmt " + (16).to_bytes(4, "little") + (1).to_bytes(2, "little")
    head += (1).to_bytes(2, "little") + rate.to_bytes(4, "little")
    head += (rate * 2).to_bytes(4, "little") + (2).to_bytes(2, "little")
    head += (16).to_bytes(2, "little") + b"data" + (0).to_bytes(4, "little")
    return bytes(head)


def test_the_call_follows_the_native_multimodal_example() -> None:
    """**原生端点 + `parameters` 里带 format/sample_rate**（官方那条 curl 的形状）。

    第一版照 vision 抄了 OpenAI 兼容的 `/chat/completions`，上游回
    `{"message":"format is empty","type":"UNSUPPORTED_FORMAT"}` —— 兼容通道读
    `input_audio.format`，原生端点把这两项放在 `parameters` 里。
    """
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["sse"] = request.headers.get("x-dashscope-sse")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"output": {"choices": [{"message": {"content": [{"text": "你好，世界。"}]}}]}},
        )

    text = _client(handler).transcribe(_wav(16000), name="a.wav")

    assert text == "你好，世界。"
    assert (
        seen["url"]
        == "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
    )
    assert seen["auth"] == "Bearer fake-key"
    assert seen["sse"] == "disable"
    body = seen["body"]
    assert body["model"] == "qwen-audio-3.0-asr-flash"
    assert "messages" not in body, "原生端点把消息放在 input 里"
    part = body["input"]["messages"][0]["content"][0]
    assert part["type"] == "input_audio"
    assert part["input_audio"]["data"].startswith("data:audio/wav;base64,")
    assert body["parameters"] == {"format": "wav", "sample_rate": "16000"}


def test_the_sample_rate_comes_from_the_wav_header() -> None:
    """写死 16000 的话，一段 48kHz 的 wav 会被当成 16kHz 解 —— 整段变调，很难查。"""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"output": {"choices": [{"message": {"content": [{"text": "x"}]}}]}}
        )

    _client(handler).transcribe(_wav(48000), name="a.wav")

    assert seen["body"]["parameters"]["sample_rate"] == "48000"


def test_a_non_wav_does_not_get_a_sample_rate() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"output": {"choices": [{"message": {"content": [{"text": "x"}]}}]}}
        )

    _client(handler).transcribe(b"ID3fake-mp3", name="a.mp3", content_type="audio/mpeg")

    assert seen["body"]["parameters"] == {"format": "mp3"}, "mp3 没头可读，别瞎填一个数"


def test_the_asr_sentence_shape_is_read() -> None:
    """**真调用回来的那个形状**（2026-10-08 那条录音）：两层 `output` + `sentence.text`。

    它跟 chat 那套完全不一样，而且 `words` 是逐字时间戳（每个 `text` 只是一个词）——
    认错了拼出来就是一堆碎片。
    """
    payload = {
        "output": {
            "output": {
                "request_id": "10565486-b8cc-92a7-aac9-7268443d9585",
                "sentence": {
                    "begin_time": 880,
                    "channel_id": 0,
                    "end_time": 3440,
                    "sentence_end": True,
                    "sentence_id": 1,
                    "text": "Spring AI是什么？怎么用。",
                    "words": [
                        {"begin_time": 880, "end_time": 1400, "fixed": True, "text": "Spring"}
                    ],
                },
            }
        }
    }
    client = _client(lambda request: httpx.Response(200, json=payload))

    assert client.transcribe(b"x", name="a.wav") == "Spring AI是什么？怎么用。"


def test_several_sentences_are_joined_in_order() -> None:
    """长一点的录音会分句。按顺序拼起来，别只留第一句。"""
    payload = {
        "output": {
            "output": {
                "sentences": [
                    {"sentence_id": 1, "text": "第一句。"},
                    {"sentence_id": 2, "text": "第二句。"},
                ]
            }
        }
    }
    client = _client(lambda request: httpx.Response(200, json=payload))

    assert client.transcribe(b"x", name="a.wav") == "第一句。第二句。"


def test_a_part_list_content_is_also_accepted() -> None:
    """不同的兼容实现给的内容形状不一样（字符串 / 片段列表），两种都收。"""
    client = _client(
        lambda request: httpx.Response(
            200, json={"choices": [{"message": {"content": [{"text": "转出来的字"}]}}]}
        )
    )

    assert client.transcribe(b"x", name="a.wav") == "转出来的字"


def test_a_silent_take_is_reported_as_no_speech() -> None:
    """静音那次**真调用**的原话就是它：`{"code":"CLIENT_ERROR",
    "message":"ASR_RESPONSE_HAVE_NO_WORDS"}`。

    归成单独一类错误（422 而不是 502）：不是上游挂了，是**这次输入**里没有内容 ——
    静音、麦克风没选中、离得太远都会走到这里。
    """
    client = _client(
        lambda request: httpx.Response(
            400,
            json={
                "request_id": "r-1",
                "code": "CLIENT_ERROR",
                "message": "ASR_RESPONSE_HAVE_NO_WORDS",
            },
        )
    )

    with pytest.raises(AsrNoSpeechError, match="没听清"):
        client.transcribe(b"x", name="a.wav")


def test_the_upstream_error_is_reported_verbatim() -> None:
    client = _client(
        lambda request: httpx.Response(400, json={"error": {"message": "model not found"}})
    )

    with pytest.raises(AsrError) as caught:
        client.transcribe(b"x", name="a.wav")

    assert "HTTP 400" in str(caught.value) and "model not found" in str(caught.value)


def test_an_empty_result_is_an_error() -> None:
    """返回空串多半是没录上 —— 静默填个空串会让人以为「识别成了空白」。"""
    client = _client(
        lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "  "}}]})
    )

    with pytest.raises(AsrError, match="空文字"):
        client.transcribe(b"x", name="a.wav")


def test_build_asr_says_where_to_put_the_key() -> None:
    # **`_env_file=None`**：不关掉它，Secrets() 会去读项目根的 `.env`，用例就跟着本机
    # 配置飘 —— 而且真配了 key 的话，这条「缺 key」的用例会一路调到线上
    with pytest.raises(ValueError, match="DASHSCOPE_ASR_API_KEY"):
        build_asr(Settings(), Secrets(_env_file=None))


def test_build_asr_falls_back_to_the_dashscope_key() -> None:
    client = build_asr(Settings(), Secrets(_env_file=None, dashscope_api_key="account-key"))

    assert client.model == "qwen-audio-3.0-asr-flash", "型号名是查过账号清单定下来的"


# ---- HTTP ----


class _FakeAsr:
    model = "fake-asr"

    def __init__(self, text: str = "转出来的字") -> None:
        self.text = text
        self.calls: list[tuple[bytes, str, str]] = []

    def transcribe(self, data: bytes, *, name: str = "", content_type: str = "") -> str:
        self.calls.append((data, name, content_type))
        return self.text


def _settings(tmp_path: Path) -> Settings:
    settings = Settings()
    settings.ingest.uploads.dir = str(tmp_path / "uploads")
    settings.ingest.stages.dir = str(tmp_path / "stages")
    settings.ingest.assets.dir = str(tmp_path / "assets")
    return settings


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    settings = _settings(tmp_path)
    # `_env_file=None`：这些用例**一个真调用都不该打**，读本机的 .env 就会真的打出去
    secrets = Secrets(_env_file=None, rag_api_key=None)
    app = create_app(settings, secrets)
    app.state.components = Components(settings, secrets)
    return TestClient(app)


def test_transcribe_returns_the_text_and_the_model(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeAsr()
    monkeypatch.setattr("rag.providers.build_asr", lambda settings, secrets: fake)

    res = client.post(
        "/chat/transcribe",
        files={"file": ("take.wav", b"RIFFfake", "audio/wav")},
    )

    assert res.status_code == 200
    assert res.json() == {"text": "转出来的字", "model": "fake-asr", "chars": 5}
    assert fake.calls == [(b"RIFFfake", "take.wav", "audio/wav")], "原样交给上游，不落盘"


def test_transcribe_rejects_an_empty_body(client: TestClient) -> None:
    res = client.post("/chat/transcribe", files={"file": ("take.wav", b"", "audio/wav")})

    assert res.status_code == 400 and "空的" in res.json()["error"]["message"]


def test_transcribe_stops_at_the_size_limit(client: TestClient) -> None:
    """超限**在上游之前**拦下：打上去就是一次按量计费，哪怕上游回 413。"""
    client.app.state.components.settings.api.media_max_audio_mb = 0.000001

    res = client.post("/chat/transcribe", files={"file": ("take.wav", b"RIFF" * 8, "audio/wav")})

    assert res.status_code == 413 and "上限" in res.json()["error"]["message"]


def test_transcribe_reports_a_missing_key_as_503(client: TestClient) -> None:
    """没配 key：503 且**说清去哪填**，而不是一个看不出所以然的 500。"""
    res = client.post("/chat/transcribe", files={"file": ("take.wav", b"RIFFfake", "audio/wav")})

    assert res.status_code == 503 and "DASHSCOPE_ASR_API_KEY" in res.json()["error"]["message"]


def test_transcribe_reports_no_speech_as_422(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """静音那条不该是 502：上游没挂，是这次输入里没有说话。"""

    class _Silent:
        model = "fake-asr"

        def transcribe(self, data: bytes, *, name: str = "", content_type: str = "") -> str:
            raise AsrNoSpeechError("没听清：这段音频里没识别到说话")

    monkeypatch.setattr("rag.providers.build_asr", lambda settings, secrets: _Silent())

    res = client.post("/chat/transcribe", files={"file": ("take.wav", b"RIFFfake", "audio/wav")})

    assert res.status_code == 422 and "没听清" in res.json()["error"]["message"]
