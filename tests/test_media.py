"""图片落盘工具。

落点必须**不是** `data/uploads/` —— 那是文档语料根，`.png/.jpg` 在 loader 注册表里，
丢进去会被当成一份文档列出来、跟着走五步流程。下面有用例专门钉这一条。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.api.media import MediaError, save_image, suffix_for
from rag.config.schema import Settings
from rag.deployment import apply_deployment_profile

# 一个最小的 PNG 头（内容不重要，链路上只按字节落盘）
_PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 64


def _settings(tmp_path: Path) -> Settings:
    settings = Settings()
    settings.deployment.profile = "local-lite"
    settings.deployment.state_dir = str(tmp_path)
    apply_deployment_profile(settings)
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

    assert path.is_file() and path.parent == tmp_path / "media"
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
