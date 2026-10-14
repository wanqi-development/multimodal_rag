"""检索页的「以图搜图」上传口：`POST /search/image`。

**图片只在检索这一条路上**（使用者 2026-10-08 定）：对话页的 agent 与快问答**都不收图**，
所以这里没有 `ChatRequest.images`、也没有「带图轮次切模型」那一套。

落点必须**不是** `data/uploads/` —— 那是文档语料根，`.png/.jpg` 在 loader 注册表里，
丢进去会被当成一份文档列出来、跟着走五步流程。下面有用例专门钉这一条。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag.api.app import Components, create_app
from rag.api.media import MediaError, save_image, suffix_for
from rag.config.schema import Settings
from rag.config.settings import Secrets

# 一个最小的 PNG 头（内容不重要，链路上只按字节落盘）
_PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 64


def _settings(tmp_path: Path) -> Settings:
    settings = Settings()
    settings.ingest.uploads.dir = str(tmp_path / "uploads")
    settings.ingest.stages.dir = str(tmp_path / "stages")
    settings.ingest.assets.dir = str(tmp_path / "assets")
    settings.api.media_dir = str(tmp_path / "chat" / "media")
    return settings


def test_the_suffix_comes_from_the_name_or_the_content_type() -> None:
    assert suffix_for("a.PNG") == ".png"
    assert suffix_for("", "image/jpeg") == ".jpg"
    assert suffix_for("blob", "image/webp;charset=binary") == ".webp"
    with pytest.raises(MediaError, match="认不出的图片类型"):
        suffix_for("a.txt", "text/plain")


def test_the_image_lands_outside_the_document_corpus(tmp_path: Path) -> None:
    """**这一条最要紧**：落在 `data/uploads/` 的图会被当成一份文档列出来、走完五步流程。"""
    settings = _settings(tmp_path)

    path = save_image(settings, _PNG, name="截图.png", content_type="image/png")

    assert path.is_file() and path.parent == tmp_path / "chat" / "media"
    assert path.parent != Path(settings.ingest.uploads.dir)
    assert "截图-" in path.name and path.suffix == ".png", "落盘名带内容哈希，真名留着看"


def test_same_name_is_idempotent_and_a_new_name_is_a_second_copy(tmp_path: Path) -> None:
    """落盘名 = `<真名>-<内容哈希><后缀>`，**与上传件同一个口径**：
    同名同内容重传是幂等的（覆盖自己）；换个名字就是另一份，两份并存、互不覆盖。

    一开始我把这条断言写成「内容一样就该是同一个文件」—— 那是错的：文件名里带真名，
    哈希只是防止**同名不同内容**互相覆盖。真按内容去重，两张不同用途的图会被并成一张。
    """
    settings = _settings(tmp_path)

    first = save_image(settings, _PNG, name="a.png")
    again = save_image(settings, _PNG, name="a.png")
    other = save_image(settings, _PNG, name="b.png")

    assert first == again, "同名同内容 → 还是那一份"
    assert other != first and other.is_file(), "换个名字 → 另一份"


def test_an_oversized_or_empty_image_is_rejected(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.api.media_max_image_mb = 0.000001

    with pytest.raises(MediaError, match="超过上限"):
        save_image(settings, _PNG, name="a.png")
    with pytest.raises(MediaError, match="空的"):
        save_image(_settings(tmp_path), b"", name="a.png")


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    settings = _settings(tmp_path)
    secrets = Secrets(_env_file=None, rag_api_key=None)
    app = create_app(settings, secrets)
    app.state.components = Components(settings, secrets)
    return TestClient(app)


def test_upload_returns_the_server_path(client: TestClient, tmp_path: Path) -> None:
    """**只回路径、不顺带检索**：`/search` 要过线上 embedding（按量计费），
    而「传了图还没点搜」是常态 —— 分开之后重搜、换词都不用重传图。"""
    res = client.post("/search/image", files={"file": ("截图.png", _PNG, "image/png")})

    assert res.status_code == 201
    body = res.json()
    assert body["bytes"] == len(_PNG)
    assert Path(body["path"]).is_file(), "给的路径要能直接交给 /search 的 image_path"
    assert Path(body["path"]).parent == tmp_path / "chat" / "media"
    assert client.get("/documents").json()["documents"] == [], "别混进文档列表"


def test_a_bad_type_is_a_400_and_an_oversized_one_is_a_413(client: TestClient) -> None:
    bad = client.post("/search/image", files={"file": ("a.txt", b"hi", "text/plain")})
    assert bad.status_code == 400 and "认不出" in bad.json()["error"]["message"]

    client.app.state.components.settings.api.media_max_image_mb = 0.000001
    big = client.post("/search/image", files={"file": ("a.png", _PNG, "image/png")})
    assert big.status_code == 413 and "上限" in big.json()["error"]["message"]
