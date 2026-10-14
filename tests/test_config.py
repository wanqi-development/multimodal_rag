"""配置加载：yaml 深合并 + ${} 插值。"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.config import load_settings
from rag.config import settings as settings_module
from rag.ingest import build_ocr

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """每个测试都在干净的 cwd 下跑，并清掉 .env 缓存。

    否则测试会读到开发者本机的 .env（里面有 MILVUS_URI 等部署参数），
    结果随本机环境飘——测试必须自足。
    """
    monkeypatch.chdir(tmp_path)
    settings_module._dotenv.cache_clear()


def test_base_config_loads() -> None:
    settings = load_settings(profile="__no_such_profile__", directory=CONFIG_DIR)
    assert settings.providers.embedding.backend == "sentence-transformers"
    assert settings.providers.embedding.model == ".models/Qwen3-Embedding-0.6B"
    assert settings.providers.embedding.dim == 1024
    # collection 名带指纹：换模型、或换块表结构，都得换名字重建（Milvus 不能改 schema）
    assert settings.indexing.vector.collection == "rag_qwen3_v2"


def test_dev_profile_overrides_base() -> None:
    settings = load_settings(profile="dev", directory=CONFIG_DIR)
    # dev 档走多模态那条（qwen3-vl-embedding，2560 维）：collection 必须跟着换名字
    assert settings.indexing.vector.collection == "rag_dev_qwen3vl_v1"
    assert settings.providers.embedding.backend == "qwen3-vl"
    # dev 档只开 Milvus 那一路（关键词就在它里面）；图谱要用得自己往 channels 里加一项
    # dev 档只开 Milvus 那一路；`kg`（图谱插件）**默认关**，只是让名字可被 `--channel` 选中
    assert [(item.name, item.type, item.enabled) for item in settings.retrieval.channels] == [
        ("milvus", "milvus", True),
        ("kg", "neo4j", False),
    ]


def test_the_model_slots_exist_and_default_to_follow_the_existing_config() -> None:
    """模型槽位（2026-10-09 第一步：只立结构）：八个槽位，默认**全部留空**。

    留空 = "沿用今天各处那份配置"，所以这一步**行为零变化** —— 真接存储与界面时
    才把空值解成具体模型名（见 `docs/backlog.md` 第 2 条）。
    """
    settings = load_settings(profile="dev", directory=CONFIG_DIR)

    assert settings.models.backend == "auto", "有 PG 用 PG，没有就用单机 SQLite"
    assert list(settings.models.slots) == [
        "chat.fast",
        "chat.agent",
        "chat.judge",
        "rerank",
        "embedding",
        "vision",
        "asr",
        "ocr",
    ]
    assert all(
        item.provider_key == "" and item.model == "" and item.params == {}
        for item in settings.models.slots.values()
    )
    # base 里未被 dev 覆盖的项要保留
    assert settings.providers.embedding.batch_size == 32
    assert settings.ingest.assets.dir == "./data/assets"


def test_dev_profile_turns_image_descriptions_on() -> None:
    """多模态那条**必须有描述**：融合向量的文字部分就是它（见 stages 的守卫）。"""
    settings = load_settings(profile="dev", directory=CONFIG_DIR)

    assert settings.ingest.images.describe is True


def test_dev_embedding_still_lets_env_win(monkeypatch: pytest.MonkeyPatch) -> None:
    """dev.yaml 里的默认值写成 ${VAR:-默认}：换机器、换回纯文本只改 .env，不动 yaml。"""
    monkeypatch.setenv("EMBEDDING_BACKEND", "sentence-transformers")
    monkeypatch.setenv("EMBEDDING_DIM", "1024")

    settings = load_settings(profile="dev", directory=CONFIG_DIR)

    assert settings.providers.embedding.backend == "sentence-transformers"
    assert settings.providers.embedding.dim == 1024


def test_hybrid_channel_defaults_to_dense_plus_sparse() -> None:
    settings = load_settings(profile="dev", directory=CONFIG_DIR)
    assert settings.retrieval.hybrid.dense.field == "dense"
    assert settings.retrieval.hybrid.sparse.field == "sparse"
    assert settings.retrieval.hybrid.fusion.strategy == "rrf"


def test_milvus_index_types_are_field_specific() -> None:
    """dense 与 sparse 的索引类型/度量必须不同，这是 Milvus 的硬要求。"""
    settings = load_settings(profile="dev", directory=CONFIG_DIR)
    indexes = settings.indexing.vector.indexes
    assert indexes["dense"].type == "IVF_FLAT"
    assert indexes["dense"].metric == "COSINE"
    assert indexes["sparse"].type == "SPARSE_INVERTED_INDEX"
    # 内建 BM25 要求 metric 是 BM25，不是 IP
    assert indexes["sparse"].metric == "BM25"


def test_bm25_uses_chinese_analyzer() -> None:
    """中文语料靠服务端分词；若退回默认英文分词器，中文 BM25 会几乎失效。"""
    settings = load_settings(profile="dev", directory=CONFIG_DIR)
    bm25 = settings.indexing.vector.bm25
    assert bm25.enabled is True
    assert bm25.analyzer == "chinese"
    assert bm25.output_field == "sparse"

    # sparse 不再由 embedding 模型产出，embedding 配置里不该再有 sparse 开关
    assert not hasattr(settings.providers.embedding, "sparse")


def test_vector_dim_comes_from_embedding_config_only() -> None:
    """维度只能有一个来源：两处各写一遍必然漂移，导致 collection 与模型不匹配。"""
    base = load_settings(profile="__no_such_profile__", directory=CONFIG_DIR)
    assert not hasattr(base.indexing.vector, "dim")
    assert base.providers.embedding.dim == 1024

    # dev 档换到 2560 维的多模态那条，维度跟着 embedding 走，collection 也跟着换名
    dev = load_settings(profile="dev", directory=CONFIG_DIR)
    assert dev.providers.embedding.dim == 2560
    assert dev.indexing.vector.collection != base.indexing.vector.collection


def test_hf_endpoint_defaults_to_mirror_and_can_be_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """huggingface.co 直连不通，默认必须指向镜像。"""
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    settings = load_settings(profile="dev", directory=CONFIG_DIR)
    assert settings.providers.embedding.hf_endpoint == "https://hf-mirror.com"

    monkeypatch.setenv("HF_ENDPOINT", "https://huggingface.co")
    settings = load_settings(profile="dev", directory=CONFIG_DIR)
    assert settings.providers.embedding.hf_endpoint == "https://huggingface.co"


def test_env_interpolation_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MILVUS_URI", "http://milvus.test:19530")
    settings = load_settings(profile="dev", directory=CONFIG_DIR)
    assert settings.indexing.vector.uri == "http://milvus.test:19530"


def test_env_interpolation_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MILVUS_URI", raising=False)
    settings = load_settings(profile="dev", directory=CONFIG_DIR)
    assert settings.indexing.vector.uri == "http://localhost:19530"


def test_missing_config_dir_uses_schema_defaults(tmp_path: Path) -> None:
    settings = load_settings(profile="dev", directory=tmp_path)
    assert settings.indexing.vector.collection == "rag_v1"


def test_dotenv_is_used_as_interpolation_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`.env` 只被 pydantic-settings 读进 Secrets，不进 os.environ，
    所以 yaml 插值必须自己去读它，否则 .env 里的部署参数形同虚设。"""
    monkeypatch.delenv("MILVUS_URI", raising=False)
    (tmp_path / ".env").write_text("MILVUS_URI=http://from-dotenv:19530\n", encoding="utf-8")

    settings = load_settings(profile="dev", directory=CONFIG_DIR)
    assert settings.indexing.vector.uri == "http://from-dotenv:19530"


def test_real_env_var_wins_over_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text("MILVUS_URI=http://from-dotenv:19530\n", encoding="utf-8")
    monkeypatch.setenv("MILVUS_URI", "http://from-env:19530")

    settings = load_settings(profile="dev", directory=CONFIG_DIR)
    assert settings.indexing.vector.uri == "http://from-env:19530"


def test_chat_base_url_defaults_and_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    """base_url 也走插值：换自建网关 / 代理只改 .env，不用动要提交的配置。"""
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    settings = load_settings(profile="dev", directory=CONFIG_DIR)
    assert settings.providers.chat["default"].base_url == "https://api.deepseek.com/v1"

    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://gw.internal/v1")
    settings = load_settings(profile="dev", directory=CONFIG_DIR)
    assert settings.providers.chat["default"].base_url == "https://gw.internal/v1"


def test_chat_base_url_can_come_from_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    (tmp_path / ".env").write_text("DEEPSEEK_BASE_URL=https://gw.internal/v1\n", encoding="utf-8")

    settings = load_settings(profile="dev", directory=CONFIG_DIR)
    assert settings.providers.chat["default"].base_url == "https://gw.internal/v1"


def test_native_provider_may_leave_base_url_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    """openai 有原生集成：默认留空表示用官方地址，设了才走网关。"""
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    settings = load_settings(profile="dev", directory=CONFIG_DIR)
    assert not settings.providers.chat["openai"].base_url

    monkeypatch.setenv("OPENAI_BASE_URL", "https://gw.internal/openai")
    settings = load_settings(profile="dev", directory=CONFIG_DIR)
    assert settings.providers.chat["openai"].base_url == "https://gw.internal/openai"


# ---- OCR 产线服务 ----


def test_ocr_is_off_until_an_address_is_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认留空 = 不启用 OCR。

    这条是「没部署 GPU 服务的机器也能跑通整条链路」的依据：客户端是 None，
    链路上不会发出任何请求。
    """
    monkeypatch.delenv("PADDLEOCR_BASE_URL", raising=False)
    settings = load_settings(profile="dev", directory=CONFIG_DIR)

    assert not settings.ingest.ocr.base_url
    assert build_ocr(settings.ingest.ocr) is None


def test_ocr_base_url_comes_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PADDLEOCR_BASE_URL", "http://gpu-host:8080")
    settings = load_settings(profile="dev", directory=CONFIG_DIR)

    # 这个用例只关心 base_url 的插值。它的 cwd 被隔离到 tmp（见 _empty_cwd），
    # 读不到项目根的 .env —— 而 cloud_fallback 是开着的，所以要显式给个 token，
    # 否则 build_ocr 会按「开了兜底却没有凭据」当场报错。
    client = build_ocr(settings.ingest.ocr, token="test-token")
    assert client is not None
    assert client.base_url == "http://gpu-host:8080"


def test_ocr_base_url_can_come_from_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PADDLEOCR_BASE_URL", raising=False)
    (tmp_path / ".env").write_text("PADDLEOCR_BASE_URL=http://from-dotenv:8080\n", encoding="utf-8")

    settings = load_settings(profile="dev", directory=CONFIG_DIR)
    client = build_ocr(settings.ingest.ocr, token="test-token")
    assert client is not None
    assert client.base_url == "http://from-dotenv:8080"


def test_ocr_cross_page_switches_are_on_by_default() -> None:
    """merge_tables / relevel_titles 不配 restructure_pages 就是不生效的，
    所以三个必须一起默认开 —— 它们正是「整份走 OCR」的主要理由。"""
    settings = load_settings(profile="__no_such_profile__", directory=CONFIG_DIR)
    # 所有 PDF 统一走 OCR —— 数字版走 pymupdf 只能拿纯文本，标题/表格/图片全丢。
    # 默认后端是 dots.ocr：只有它给的标题层级可用（见 ingest/dots.py）
    assert settings.ingest.pdf.parser == "dots"
    assert settings.ingest.dots.base_url.startswith("http")
    assert settings.ingest.dots.model
    assert settings.ingest.ocr.restructure_pages is True
    assert settings.ingest.ocr.merge_tables is True
    assert settings.ingest.ocr.relevel_titles is True
    # 整份 PDF 走一次 OCR 比单页慢得多
    assert settings.ingest.ocr.timeout_s >= 300
