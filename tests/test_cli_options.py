"""cli 的三个选项：--filter / --channel / --no-rerank。

只测解析与覆盖逻辑（真 CLI 要连库，这套项目里不跑）—— 逻辑抽成纯函数就是为了这个。
"""

from __future__ import annotations

import asyncio

import pytest
import typer

from rag.cli.main import _apply_channels, _parse_filters
from rag.config.schema import ChannelConfig, Settings


def test_filter_pairs_become_a_dict() -> None:
    assert _parse_filters(["modality=image", "doc_type=pdf"]) == {
        "modality": "image",
        "doc_type": "pdf",
    }


def test_filter_values_are_coerced_to_the_column_types() -> None:
    """命令行只有字符串，而字段有类型：`page` 是整数、bool 字段要真是 bool。"""
    assert _parse_filters(["page=3"]) == {"page": 3}
    assert _parse_filters(["page=3,5"]) == {"page": [3, 5]}
    assert _parse_filters(["x=true"]) == {"x": True}
    assert _parse_filters(["x=false"]) == {"x": False}
    assert _parse_filters(["modality=image,table"]) == {"modality": ["image", "table"]}


def test_a_malformed_filter_says_how_to_write_it() -> None:
    for bad in ("modality", "=image", "modality="):
        with pytest.raises(typer.BadParameter, match="key=value"):
            _parse_filters([bad])


def test_no_filter_means_no_filter() -> None:
    assert _parse_filters([]) == {}


def _two_channels() -> Settings:
    """装两路：Milvus（默认开）+ 图谱（默认关）。"""
    settings = Settings()
    settings.retrieval.channels = [
        ChannelConfig(name="milvus", type="milvus"),
        ChannelConfig(name="kg", type="neo4j", enabled=False),
    ]
    return settings


def test_the_channel_spec_turns_the_listed_ones_on_and_the_rest_off() -> None:
    """做对比时最想要的那一档：只要 Milvus（它内部已经含融合向量 + 关键词两条腿）。"""
    settings = _two_channels()

    _apply_channels(settings, "milvus")

    assert [item.name for item in settings.retrieval.channels if item.enabled] == ["milvus"]


def test_listing_a_disabled_channel_turns_it_on() -> None:
    settings = _two_channels()

    _apply_channels(settings, "milvus,kg")

    assert [item.name for item in settings.retrieval.channels if item.enabled] == ["milvus", "kg"]


def test_an_unknown_channel_is_rejected_with_the_options() -> None:
    """静默忽略 = 你以为开了某一路、其实没开。名字要按**配置里那几项**校验。"""
    settings = _two_channels()

    with pytest.raises(typer.BadParameter, match="未知的通道"):
        # web 是「文档里写了、但还没落地」的那一路：名字必须被拒，而不是静默当作没有
        _apply_channels(settings, "milvus,web")


# ---- 连不上库时说人话 ----


def test_a_milvus_outage_becomes_one_line(capsys) -> None:
    """Milvus 停掉时 CLI 原本吐 100 行 pymilvus 调用栈（实测），而 API 早就是一句话。
    同一件事两种体验没道理 —— 收成一句，并给出下一步。"""
    from pymilvus.exceptions import MilvusException

    from rag.cli.main import _run_async

    settings = Settings()
    settings.indexing.vector.uri = "http://vm:19530"

    async def boom():
        raise MilvusException(code=2, message="Fail connecting to server on vm:19530")

    with pytest.raises(typer.Exit) as exit_info:
        _run_async(boom(), settings)

    assert exit_info.value.exit_code == 1
    out = capsys.readouterr().out
    assert "Milvus 连不上：http://vm:19530" in out and "MILVUS_URI" in out
    assert "MilvusException" not in out


def test_other_failures_are_not_swallowed() -> None:
    """只翻译「连不上」那一类：别的异常要原样抛出去，不能把真 bug 也吞掉。"""
    from pymilvus.exceptions import MilvusException

    from rag.cli.main import _run_async

    async def broken_tool():
        raise MilvusException(code=1, message="collection not found")

    with pytest.raises(MilvusException, match="collection not found"):
        _run_async(broken_tool(), Settings())

    async def other():
        raise ValueError("别的错")

    with pytest.raises(ValueError, match="别的错"):
        _run_async(other(), Settings())


# ---- Milvus 探活：接不住就在几秒内说人话，别空转 ----


class _SlowIndex:
    async def ping(self, timeout_s: float) -> None:
        raise TimeoutError


class _DownIndex:
    async def ping(self, timeout_s: float) -> None:
        from pymilvus.exceptions import MilvusException

        raise MilvusException(code=2, message="Fail connecting to server on vm:19530")


class _BrokenIndex:
    async def ping(self, timeout_s: float) -> None:
        raise ValueError("别的错")


def test_a_milvus_probe_timeout_becomes_one_line(capsys) -> None:
    """实测：库停掉时 `rag ingest` 在写库那步空转三十几分钟（纯 CPU 烧、数据一行没进）。
    探活把这段等待压到几秒，而且是在**解析/OCR 之前**。"""
    from rag.cli.main import _probe_milvus

    settings = Settings()
    settings.indexing.vector.uri = "http://vm:19530"
    settings.indexing.vector.probe_timeout_s = 5

    with pytest.raises(typer.Exit) as exit_info:
        asyncio.run(_probe_milvus(_SlowIndex(), settings))

    assert exit_info.value.exit_code == 1
    out = capsys.readouterr().out
    assert "探活超时" in out and "http://vm:19530" in out and "MILVUS_URI" in out


def test_a_probe_failure_reuses_the_milvus_one_liner(capsys) -> None:
    """连不上与超时是两种表象，但该给人看的是同一句话。"""
    from rag.cli.main import _probe_milvus

    settings = Settings()
    settings.indexing.vector.uri = "http://vm:19530"

    with pytest.raises(typer.Exit):
        asyncio.run(_probe_milvus(_DownIndex(), settings))

    assert "Milvus 连不上：http://vm:19530" in capsys.readouterr().out


def test_a_probe_does_not_swallow_other_failures() -> None:
    """只翻译「连不上 / 超时」：别的异常原样抛出去，不能把真 bug 也吞掉。"""
    from rag.cli.main import _probe_milvus

    with pytest.raises(ValueError, match="别的错"):
        asyncio.run(_probe_milvus(_BrokenIndex(), Settings()))


# ---- serve：uvicorn 的 loop 必须显式指到我们的工厂 ----


def test_serve_tells_uvicorn_to_use_the_selector_loop_factory(monkeypatch) -> None:
    """uvicorn ≥0.36 **不看事件循环策略**：`Server.run()` 直接把 loop 工厂的返回值交给
    `asyncio_run`，而它在 Windows 上给的是 `ProactorEventLoop` —— 只要配了 PG，服务就会在
    startup 崩在「psycopg 的 async 不能跑在 ProactorEventLoop 上」（真机实测 2026-10-05）。

    所以这一条钉住的是：**`rag serve` 必须把这个参数传下去**，少了它服务起不来。
    """
    import uvicorn
    from typer.testing import CliRunner

    import rag.api
    import rag.cli.main as cli

    captured: dict = {}
    monkeypatch.setattr(
        uvicorn, "run", lambda app, **kwargs: captured.update({"app": app, **kwargs})
    )
    monkeypatch.setattr(rag.api, "create_app", lambda settings, secrets=None: "app")
    # 用一份**没配 PG** 的配置：免得这条用例真的去换本进程的事件循环策略
    monkeypatch.setattr(cli, "load_settings", lambda: Settings())

    result = CliRunner().invoke(cli.app, ["serve"])

    assert result.exit_code == 0
    assert captured["loop"] == "rag.persistence:selector_event_loop"
    assert captured["workers"] == 1


# ---- QueryGraph 的重对象只能建一次 ----


class _Embedder:
    name = "stub"
    dim = 2
    supports_images = False

    async def embed_query(self, text, image=None):
        from rag.contracts import Embeddings

        return Embeddings(dense=[[1.0, 0.0]])

    async def embed_documents(self, texts, images=None):
        from rag.contracts import Embeddings

        return Embeddings(dense=[[1.0, 0.0] for _ in texts])


class _Reranker:
    name = "rerank:stub"

    async def rerank(self, query, chunks, top_n):
        return chunks[:top_n]


class _Index:
    name = "milvus:test"

    async def search_hybrid(self, *args, **kwargs):
        return []

    async def search_sparse(self, *args, **kwargs):
        return []

    async def search_dense(self, *args, **kwargs):
        return []


def test_heavy_components_are_built_once(monkeypatch) -> None:
    """embedder 与 reranker 是 1GB 量级的重对象，而检索闭包**每个子查询、每一轮改写**都会被
    调一次。建在闭包里 = 每轮重新加载模型 —— 实测一次三轮的问答里出现三次「Loading weights」。

    现在「只建一次」是**结构上的**：重对象由 `build_heavy` 持有，闭包只从它取。
    """
    import rag.providers
    from rag.graphs import wiring

    built = {"embed": 0, "rerank": 0}

    def fake_embedder(settings):
        built["embed"] += 1
        return _Embedder()

    def fake_reranker(settings):
        built["rerank"] += 1
        return _Reranker()

    monkeypatch.setattr(rag.providers, "build_embedder", fake_embedder)
    monkeypatch.setattr(rag.providers, "build_reranker", fake_reranker)
    monkeypatch.setattr(rag.providers, "build_chat", lambda *a, **k: object())
    monkeypatch.setattr(wiring, "_secrets", lambda: object())
    settings = Settings()

    heavy = wiring.build_heavy(settings, _Index())
    assert built == {"embed": 1, "rerank": 1}, "build_heavy 建一次"

    retrieve = wiring.build_retrieve(heavy)
    asyncio.run(retrieve("第一轮", 5))
    asyncio.run(retrieve("第二轮", 5))

    assert built == {"embed": 1, "rerank": 1}, "两次检索不该再建一遍模型"

    retrieve_again, plan_llm, grade_llm, generate = wiring.build_query_nodes(
        settings, _Index(), heavy
    )
    asyncio.run(retrieve_again("再跑一次", 5))

    assert built == {"embed": 1, "rerank": 1}, "给了 heavy 就该复用，不能再建"
    assert plan_llm is grade_llm, "plan 与 grade 共用一个 chat 实例"
    assert callable(generate)


# ---- 启动失败必须留下栈（2026-10-10）----


def test_a_broken_config_leaves_the_stack_behind(tmp_path, monkeypatch) -> None:
    """配置坏掉时进程死在**装日志之前**：四个 .log 一个字没有，窗一关栈也没了。

    实测那次是 `config/dev.yaml` 的 `ingest.images` 成了空节点（底下只剩注释 = YAML 的
    `null`），pydantic 在 `Settings.model_validate` 抛 ValidationError；而 `rag serve` 是
    `Start-Process` 起在新窗口里的，秒退之后什么痕迹都没留下 —— 只能手工敲 `rag config`
    才看出是校验失败。所以这条钉住的是：**异常照抛（栈照打），但先落一份到盘上**。
    """
    from typer.testing import CliRunner

    import rag.cli.main as cli

    monkeypatch.chdir(tmp_path)  # STARTUP_ERROR_LOG 是相对路径，跟着 cwd 走

    def boom() -> Settings:
        raise RuntimeError("配置坏了（用例）")

    monkeypatch.setattr(cli, "load_settings", boom)

    result = CliRunner().invoke(cli.app, ["config"])

    assert result.exit_code != 0
    log = tmp_path / "data" / "logs" / "startup-error.log"
    assert log.is_file(), "启动失败的栈必须落到盘上（窗口关了也查得到）"
    text = log.read_text(encoding="utf-8")
    assert "启动失败：RuntimeError" in text
    assert "配置坏了（用例）" in text


def test_the_startup_error_log_appends_instead_of_overwriting(tmp_path, monkeypatch) -> None:
    """连坏两次要能都看见：失败现场按时间**追加**（覆盖的话第二次会把第一次擦掉）。"""
    from typer.testing import CliRunner

    import rag.cli.main as cli

    monkeypatch.chdir(tmp_path)
    seen = {"n": 0}

    def boom() -> Settings:
        seen["n"] += 1
        raise RuntimeError(f"第 {seen['n']} 次失败")

    monkeypatch.setattr(cli, "load_settings", boom)
    runner = CliRunner()
    runner.invoke(cli.app, ["config"])
    runner.invoke(cli.app, ["config"])

    text = (tmp_path / "data" / "logs" / "startup-error.log").read_text(encoding="utf-8")
    assert "第 1 次失败" in text
    assert "第 2 次失败" in text
