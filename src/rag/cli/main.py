"""CLI 入口：``config`` / ``ingest`` / ``search`` / ``ask`` / ``agent`` / ``sessions`` / ``serve``。

**重依赖一律在命令体内 import，模块级只留 typer 与配置。** 这不是洁癖：
实测模块级全量 import 要 **12 秒**（`-X importtime`，其中 `langchain_text_splitters` →
`sentence_transformers` → `torch` 一条链就 5.5 秒、`rag.indexing.commit` 那条 9.2 秒），
于是 `rag --help` / `rag --version` / `rag sessions list` 这些**根本用不到模型与切分器**的
命令也要先付这笔钱 —— 而它们的本分是秒回。新增命令请照这个来：需要什么就在函数里 import。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from rag import __version__
from rag.config import Settings, get_secrets, load_settings

if TYPE_CHECKING:  # 只为标注：这些都是重依赖，别在模块级真 import（见模块说明）
    from rag.contracts import ScoredChunk
    from rag.persistence import SessionStore

# rag ingest --stage 的可选值。四选一：三段分别跑，或 all 一口气跑完。
STAGES = ("parse", "chunk", "index", "all")

app = typer.Typer(
    name="rag",
    help="multimodal_rag：多模态 RAG 检索",
    no_args_is_help=True,
    add_completion=False,
)


def _make_output_encoding_safe() -> None:
    """别让一行打印把已经干完的活变成「失败」。

    实测：Windows 控制台是 GBK 时，成功提示里的 ``✅`` 会让 ``typer.secho`` 抛
    ``UnicodeEncodeError`` —— 而那行是**写完库之后**才打的，数据已经落了，
    退出码却是 1，看起来像整条命令失败。把编不出来的字符降级成 ``?`` 就够了，
    没必要为此让命令崩在最后一行。
    """
    for stream in (sys.stdout, sys.stderr):
        # 被重定向、被 pytest 捕获时不支持 reconfigure —— 忽略即可
        with contextlib.suppress(AttributeError, ValueError, OSError):
            stream.reconfigure(errors="replace")


_make_output_encoding_safe()


def _todo(milestone: str) -> None:
    typer.secho(f"尚未实现（计划在 {milestone}）", fg=typer.colors.YELLOW)
    raise typer.Exit(code=1)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"multimodal_rag {__version__}")
        raise typer.Exit


def _preview(text: str, width: int = 60) -> str:
    return " ".join(text.split())[:width]


def _print_hits(hits: list[ScoredChunk]) -> None:
    for i, hit in enumerate(hits, 1):
        source = hit.chunk.metadata.get("source_uri", "?")
        # 模态要显示出来：图片描述块与正文块长得一样，不打标记就看不出命中的是图
        modality = f"[{hit.chunk.modality}]" if hit.chunk.modality != "text" else ""
        # 来源通道：融合之后 channel 恒为 "rrf"，真实来源在 sources 里（单路时为空）
        channel = f"[{'+'.join(hit.sources) or hit.channel.split(':')[0]}] "
        # chunk_id 也打：`search` 是调试工具，要能拿它回 `data/stages/<doc_id>/chunks.json`
        # 或直接查库 —— 少了它，「命中的到底是哪一块」只能靠正文猜。
        typer.echo(
            f"  {i:>2}. {hit.score:.4f} {channel}{modality} {hit.chunk.chunk_id} "
            f"{source}  {_preview(hit.chunk.text)}"
        )


def _print_stats(stats: list) -> None:
    """每路召回多少条、进了最终榜几条、花多久、有没有报错。

    为什么值得单独打：融合之后只看得到最终榜，「这条到底是谁召回的」「某一路是不是挂了」
    在结果里看不出来 —— 而这两件事决定了该调哪一路。
    """
    if not stats:
        return
    typer.echo("")
    typer.echo("通道统计：")
    for item in stats:
        spent = f"{item.latency_ms:.0f}ms" if item.latency_ms is not None else "—"
        line = f"  {item.channel:<28} 召回 {item.candidates:>3}  入选 {item.kept:>3}  {spent}"
        if item.complement_dropped:
            # 「只补位」的通道：重复的候选被丢了 —— 说清楚，不然「召回 10 入选 0」像坏了
            line += f"  补位（丢重复 {item.complement_dropped}）"
        if item.error:
            typer.secho(f"{line}  失败：{item.error}", fg=typer.colors.YELLOW)
        else:
            typer.echo(line)


def _echo_stage(name: str, done: int, reused: int, todo: list[str], dry_run: bool) -> None:
    """打印一段的进度。``--dry-run`` 时 done 恒为 0，所以把「将会处理哪些」摊开。"""
    if dry_run:
        typer.echo(f"{name}：{len(todo)} 篇待处理（跳过 {reused} 篇，产物是现成的）")
        for uri in todo:
            typer.echo(f"  · {uri}")
        return
    typer.secho(f"{name} {done} 篇（跳过 {reused} 篇）", fg=typer.colors.GREEN)


def _echo_skipped(items: list) -> None:
    for item in items:
        typer.secho(f"  跳过 {item.path}：{item.reason}", fg=typer.colors.YELLOW)


def _echo_pending(items: list) -> None:
    """待人工核对的文档：**不是失败**，所以用自己的颜色与句式，别和「跳过」混在一起。"""
    if not items:
        return
    typer.secho(f"  待人工核对 {len(items)} 篇（核对完再跑一次即可）：", fg=typer.colors.CYAN)
    for item in items:
        typer.secho(f"    · {item.path}", fg=typer.colors.CYAN)
    typer.echo(f"    {items[0].reason}")


def _require(heavy, *names: str) -> None:
    """必需组件没建起来时说清**是哪一件、为什么**，别让它以 AttributeError 的形式炸在后面。

    组件在 `graphs/wiring.py::build_heavy` 里建，那边**单个建不起来不抛**（API 要靠它给出
    503 而不是起不来），所以「这次命令需不需要它」由这里判断。
    """
    missing = [f"{name}（{heavy.errors[name]}）" for name in names if name in heavy.errors]
    if missing:
        typer.secho("组件建不起来：" + "；".join(missing), fg=typer.colors.RED)
        raise typer.Exit(code=1)


async def _probe_milvus(index, settings) -> None:
    """开工前探一次活；接不住就一行话 + 退出 1。

    实测（§16）：虚拟机上的 Milvus 停掉后，`rag ingest` 在写库那一步空转**三十几分钟**
    （纯 CPU 烧、零 I/O，数据一行没进），而库还查不出为什么。探活把这段等待压到
    `indexing.vector.probe_timeout_s`（默认 5 秒），而且探在**解析/OCR 之前** ——
    否则那几十分钟的 OCR 也白花。
    """
    timeout = settings.indexing.vector.probe_timeout_s
    uri = settings.indexing.vector.uri
    try:
        await index.ping(timeout_s=timeout)
    except TimeoutError:
        typer.secho(f"Milvus 探活超时（{timeout:g} 秒没响应）：{uri}", fg=typer.colors.RED)
        typer.secho(
            "  先确认服务起来了（虚拟机上的 etcd + MinIO + milvus），或检查 .env 里的 MILVUS_URI；"
            "机器慢就把 indexing.vector.probe_timeout_s 调大。",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=1) from None
    except Exception as exc:  # noqa: BLE001 —— 连不上交给 _explain 翻成人话，别的原样抛
        _explain(exc, settings)


def _explain(exc: Exception, settings) -> None:
    """把「连不上 Milvus」说成人话，而不是甩一屏 pymilvus 的调用栈。

    实测：Milvus 停掉时 CLI 会吐 **100 行 traceback**（`Failed to connect …` 埋在中间），
    而 API 那边早就是「503 + 一句话」。同一件事两种体验没道理，所以这里也收成一句。
    只认「连不上」这一类；别的异常**原样抛出去** —— 不能把真 bug 也一起吞掉。
    """
    from pymilvus.exceptions import MilvusException

    if not isinstance(exc, MilvusException) or "Fail connecting" not in str(exc):
        raise exc
    uri = settings.indexing.vector.uri
    typer.secho(f"Milvus 连不上：{uri}", fg=typer.colors.RED)
    typer.secho(
        "  先确认服务起来了（虚拟机上那套 etcd + MinIO + milvus），或检查 .env 里的 MILVUS_URI。",
        fg=typer.colors.YELLOW,
    )
    raise typer.Exit(code=1) from None


def _run_async(coro, settings):
    """跑协程，并把「连不上 Milvus」翻译成人话（其它异常原样抛出）。

    所有会碰库的命令都走这里 —— 否则每个命令都要抄一遍 try/except。
    """
    try:
        return asyncio.run(coro)
    except Exception as exc:  # noqa: BLE001 —— 不匹配的会由 _explain 原样抛出
        _explain(exc, settings)


def _parse_filters(pairs: list[str]) -> dict[str, Any]:
    """`--filter key=value`（可重复）→ 过滤字典。

    值的写法有意做得宽松，因为命令行只能给字符串，而字段有类型（`page` 是整数、
    `modality` 是字符串）：

    * `a,b`   → 列表（对应 `filter_expr` 的 `in`）；
    * `3`     → 整数；
    * `true`  → 布尔；
    * 其余    → 字符串。

    写错（没有 `=`、键或值为空）**当场报错并给出例子** —— 过滤条件写错会静默少召回，
    那种「结果变少了但看不出为什么」最难查。
    """
    out: dict[str, Any] = {}
    for raw in pairs:
        key, sep, value = raw.partition("=")
        if not sep or not key.strip() or not value.strip():
            raise typer.BadParameter(
                f"--filter 要写成 key=value（多值用逗号），收到的是 {raw!r}。"
                f"例：--filter modality=image --filter page=3,5"
            )
        items = [item.strip() for item in value.split(",") if item.strip()]
        values = [_coerce(item) for item in items]
        # 单值给标量、多值给列表：`filter_expr` 对列表渲染成 `in [...]`，
        # 对单值渲染成 `==`，后者更贴近人写这条命令时的意思。
        out[key.strip()] = values[0] if len(values) == 1 else values
    return out


def _coerce(value: str) -> Any:
    """命令行给的字符串 → 能对上字段类型的值。"""
    if value.isdigit():
        return int(value)
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


def _apply_channels(settings, spec: str) -> None:
    """`--channel` 覆盖要跑哪些通道：**列出来的那几路开、其余关**（逗号分隔）。

    名字是**配置里那几项的 `name`**（`retrieval.channels[].name`），不是适配器类型 ——
    所以做对比最想要的那一档就是 `--channel milvus`（只要 Milvus 那一路，它内部已经含
    融合向量 + 内建 BM25 两条腿）。名字不认识就报错并列出配置里有什么：
    静默忽略 = 你以为关了某一路，其实没关。
    """
    names = [name.strip() for name in spec.split(",") if name.strip()]
    configured = [item.name for item in settings.retrieval.channels]
    unknown = [name for name in names if name not in configured]
    if unknown:
        raise typer.BadParameter(
            f"未知的通道 {unknown}；配置里有：{configured}"
            f"（逗号分隔，例：--channel {','.join(configured[:2])}）"
        )
    wanted = set(names)
    for item in settings.retrieval.channels:
        item.enabled = item.name in wanted


# **启动失败也要留下栈**（2026-10-10）：配置坏掉是唯一一类恰好发生在日志系统建立
# **之前**的失败 —— `configure_logging` 自己就得先读 settings，所以那一刻四个 .log
# 一个 handler 都没装上。路径写死成与 `observability.logging` 同一层的那份默认，
# 不依赖配置能不能读出来（读不出来正是要记的那件事）。
STARTUP_ERROR_LOG = Path("./data/logs/startup-error.log")


def _record_startup_error(exc: BaseException) -> None:
    """把启动期的异常栈追加进 `data/logs/startup-error.log`；**再失败也不许抛**。

    为什么自己写一份：`rag serve` 被 `scripts/windows/backend.ps1` 用 `Start-Process`
    起在新窗口里，进程秒退窗口就跟着关 —— 只打到 stderr 的栈会一起消失。实测那次
    （`config/dev.yaml` 的 `ingest.images` 成了空节点）四个日志文件一个字都没有，
    只剩一句「服务起不来」，还得靠手工跑 `rag config` 才看出是校验失败。
    """
    import traceback
    from datetime import datetime

    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"==== {stamp} 启动失败：{type(exc).__name__} ====\n"
    try:
        STARTUP_ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
        with STARTUP_ERROR_LOG.open("a", encoding="utf-8") as handle:
            handle.write(header + text + "\n")
    except OSError as write_failed:  # 只读盘 / 没权限：就只能靠 stderr 了
        typer.echo(f"（写 {STARTUP_ERROR_LOG} 也失败了：{write_failed}）", err=True)
        return
    typer.echo(f"（启动失败的完整栈已写入 {STARTUP_ERROR_LOG}）", err=True)


def _settings_for_startup() -> Settings:
    """读配置。**失败先留一份栈再往上抛**（见 `_record_startup_error`）。"""
    try:
        return load_settings()
    except Exception as exc:  # noqa: BLE001 —— 记一笔就原样抛，让 typer 照常打栈
        _record_startup_error(exc)
        raise


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="显示版本并退出",
    ),
) -> None:
    """multimodal_rag 命令行入口。"""
    # **每条命令都先把日志装好**：三类业务日志各落一个文件 + 控制台（见 `obs/setup.py`）。
    # 少了这一步，CLI 的日志行（INFO）会因为 root 上一个 handler 都没有被直接丢掉；
    # 装完之后第三方库（LangChain / pymilvus / httpx）的日志也一并进 system.log。
    # `--version` / `--help` 是 eager 的，走不到这里。
    from rag.net import configured_service_urls, ensure_local_bypasses_proxy
    from rag.obs import configure_logging, new_id, set_ids, set_source

    settings = _settings_for_startup()
    # **在建任何 HTTP 客户端之前**把回环 / 内网地址并进 NO_PROXY：开着系统代理的机器上，
    # 打本机 OCR 产线的请求会被送到代理并回 502（实测，见 rag/net.py）
    ensure_local_bypasses_proxy(configured_service_urls(settings))
    configure_logging(settings)
    # **CLI 也发一个入口 id**（与 HTTP 中间件同一个形状，都是 `req-`）：一次命令就是
    # 「一次请求」，查询时不用为它分第二套。谁是入口由 `source` 那一列区分（http / cli）。
    set_ids(request_id=new_id("req"))
    set_source("cli")


@app.command("config")
def show_config(
    profile: str | None = typer.Option(None, "--profile", "-p", help="配置 profile"),
    config_dir: Path | None = typer.Option(None, "--config-dir", help="配置目录"),
) -> None:
    """打印解析后的完整配置（含 profile 覆盖与 ${} 插值结果）。"""
    settings = load_settings(profile=profile, directory=config_dir)
    typer.echo(json.dumps(settings.model_dump(mode="json"), indent=2, ensure_ascii=False))


@app.command()
def ingest(
    paths: list[Path] = typer.Argument(..., help="待入库的文件或目录"),
    collection: str | None = typer.Option(None, help="覆盖 collection 名"),
    parser: str | None = typer.Option(
        None,
        "--parser",
        help="覆盖 ingest.pdf.parser（paddleocr|dots|mineru|pymupdf；老名字 ocr 也认）",
    ),
    describe: bool | None = typer.Option(
        None,
        "--describe/--no-describe",
        help="覆盖 ingest.images.describe。打开 = 每张图一次 VLM 调用，按量计费",
    ),
    stage: str = typer.Option(
        "all",
        "--stage",
        help="只跑某一段：parse（解析）/ chunk（切分+图片描述）/ index（写 Milvus）/ all",
    ),
    full: bool = typer.Option(False, "--full", help="忽略注册表，全部重灌"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只打印要做什么，不写库"),
    graph: bool | None = typer.Option(
        None,
        "--graph/--no-graph",
        help="--stage all 时按文件铺开跑（并发 + 单文件失败隔离）；默认听 graph.enabled",
    ),
) -> None:
    """灌库：解析 → 切分+图片描述 → 写 Milvus。

    默认**增量**：读文档注册表，只处理新增与变更的文件 —— 没变的**连解析都不做**
    （解析与 embedding 是全部代价所在，跳过才是省钱的地方）；语料里已删除的文件，
    它的块也会被清掉。`--full` 忽略注册表、全部重灌。

    先按 doc_id 清旧块再写新块，所以对同一份语料反复跑是幂等的，
    文档改短了也不会剩旧块；其他 doc_id 的数据不受影响。

    **三段可以分开跑**（`--stage`），中间产物落在 `ingest.stages.dir`
    （`data/stages/<doc_id>/{document,chunks}.json + .md`）：

    * `--stage parse` 只解析、`--stage chunk` 只切分并给图片写描述 —— **两段都不碰 Milvus**，
      所以库挂着也能先跑，或白天解析、夜里再灌；
    * `--stage index` 才写库，它只读切分产物，缺产物会明说先去跑哪一段；
    * 解析/切分对不对，直接看那两个 `.md`（页码行号、章节路径、块边界都摊在上面）。
    """
    from rag.graphs import run_index_graph
    from rag.indexing import build_index, build_registry
    from rag.indexing.commit import commit_chunks, drop_docs, index_fingerprint
    from rag.ingest import build_ocr_engine
    from rag.ingest.incremental import pipeline_version, plan_incremental
    from rag.ingest.loaders import discover
    from rag.ingest.stages import build_artifact_store, run_chunk, run_parse
    from rag.providers import build_embedder, build_vision

    settings = load_settings()
    _tracing(settings)
    if collection:
        settings.indexing.vector.collection = collection
    if parser:
        settings.ingest.pdf.parser = parser
    if describe is not None:
        settings.ingest.images.describe = describe
    if stage not in STAGES:
        raise typer.BadParameter(f"未知的 --stage {stage!r}；可选 {list(STAGES)}")

    # PDF 用哪家 OCR 由 `ingest.pdf.parser` 定，工厂按名字建 client（换家只改配置）
    parser_name = settings.ingest.pdf.parser
    # OCR 客户端由**工厂**按 ingest.pdf.parser 建（paddleocr / dots / mineru 三选一），
    # 建一次整批复用（连接池在它身上）。`pymupdf` 不是 OCR，工厂没有它 → None，
    # 那一路由 loader 自己抽文字层。
    ocr = (
        None
        if parser_name == "pymupdf"
        else build_ocr_engine(settings.ingest, parser_name, get_secrets())
    )
    # VLM 只在开关打开时建：关着的话这条链路上一次请求都没有（它按量计费）
    vision = build_vision(settings, get_secrets()) if settings.ingest.images.describe else None
    version = pipeline_version(settings)
    store = build_artifact_store(settings)

    # 每个 path 参数各自算**语料根**：doc_id 与 source_uri 都相对它算
    sources = []
    for item in paths:
        resolved = item.resolve()
        sources.append((resolved, resolved if resolved.is_dir() else resolved.parent))

    # 配置行先打：三段都该看得见自己会用到什么（尤其是描述开没开 —— 那要花钱）
    typer.echo(f"阶段={stage}  产物={store.root}")
    if stage in ("index", "all"):
        typer.echo(
            f"collection={settings.indexing.vector.collection}  uri={settings.indexing.vector.uri}"
        )
    # 从 client 自己报名字与地址 —— CLI 不用知道有哪几家
    where = f"{ocr.name} @ {ocr.base_url}" if ocr else "本机文字层（有损）"
    typer.echo(f"PDF 解析={parser_name}  {where}")
    if vision is None:
        typer.echo("图片描述=关闭（--describe 打开：每张图一次 VLM 调用，按量计费）")
    else:
        typer.echo(
            f"图片描述={vision.model} @ {vision.base_url}  缓存={settings.ingest.images.cache_path}"
        )

    # embedder 建一次就够（1.2GB 的模型）：阶段2 的语义兜底与阶段3 的向量化都用它。
    # 只有解析那一段不该为它付加载钱 —— 那一段压根不需要向量。
    embedder = build_embedder(settings) if stage in ("chunk", "index", "all") else None
    if graph is not None:
        settings.graph.enabled = graph
    if embedder is not None:
        # 两种向量化方式（纯文本 / 图+文同一空间）长得一模一样，只看库里的向量分不出来 ——
        # 把当前是哪种打在开头。dim 也一起打：它决定 collection 能不能对上。
        mode = "图+文同一空间" if embedder.supports_images else "纯文本"
        typer.echo(f"embedding={embedder.name}  {embedder.dim} 维  {mode}")

    # —— 阶段3 要碰 Milvus：**在解析/OCR 之前**先探活 ——
    # 实测那三十几分钟的空转就发生在写库那步（§16），而「先解析几十分钟、再发现库连不上」
    # 等于把那段 OCR 白送。`--stage parse/chunk` 按设计不碰库，所以不探（库挂着也能先跑）。
    index = None
    if stage in ("index", "all"):
        index = build_index(settings)
        _run_async(_probe_milvus(index, settings), settings)

    # —— 阶段1、2：不碰 Milvus。库挂了、或想等机器空闲再灌，都可以先跑这里 ——
    if stage in ("parse", "chunk"):
        for source, root in sources:
            files = discover(source)
            if stage == "parse":
                parsed = run_parse(files, root, settings, ocr, store, full=full, dry_run=dry_run)
                _echo_stage("解析", parsed.parsed, parsed.reused, parsed.todo, dry_run)
                _echo_skipped(parsed.skipped)
                if not dry_run:
                    typer.echo(
                        "  标题层级：**所有文档**都要人工核对一遍再切 —— 层级靠不住"
                        "（PDF 是 OCR 猜的、docx/html 是作者的排版习惯）"
                    )
                    typer.echo(
                        f"  清单在 {store.root}\\<doc_id>\\outline.tsv，"
                        f"核对三处：改不对的层级、选「最大层级」、把「已核对: 否」改成「是」"
                    )
            else:
                chunked = run_chunk(
                    files, root, settings, embedder, vision, ocr, store, full=full, dry_run=dry_run
                )
                _echo_stage("切分", chunked.chunked, chunked.reused, chunked.todo, dry_run)
                if chunked.described:
                    typer.echo(f"  图片描述 {chunked.described} 条")
                if chunked.shells:
                    typer.echo(
                        f"  丢弃空壳块 {chunked.shells} 个（剥掉标题与图片标记后无可检索文字）"
                    )
                _echo_pending(chunked.pending)
                _echo_skipped(chunked.skipped)
                if not dry_run:
                    typer.echo("  标题层级：以 outline.tsv（人工核对过）为准；改过它就会自动重切")
        if dry_run:
            typer.secho("--dry-run：什么都没写。", fg=typer.colors.YELLOW)
        else:
            typer.secho(f"产物目录：{store.root}（.md 是人读版）", fg=typer.colors.GREEN)
        return

    async def _plan() -> list:
        registry = build_registry(settings)
        if not dry_run:
            await registry.ensure_ready()
        records = {} if full else await registry.all()
        return [
            (
                root,
                plan_incremental(
                    discover(source),
                    root=root,
                    registry=records,
                    # 指纹按**文件**算：同一批语料里，人工对齐过清单的那几份才会重灌
                    pipeline=lambda path, root=root: index_fingerprint(version, store, path, root),
                    # 单文件源看不出「别的文件没了」，所以删除检测只在目录源上做
                    detect_deleted=source.is_dir() and not full,
                ),
            )
            for source, root in sources
        ]

    plans = _run_async(_plan(), settings)
    added = [p for _, plan in plans for p in plan.added]
    changed = [p for _, plan in plans for p in plan.changed]
    kept = [p for _, plan in plans for p in plan.skipped]
    deleted = [doc_id for _, plan in plans for doc_id in plan.deleted]

    typer.echo(
        f"扫描 {len(added) + len(changed) + len(kept)} 篇 -> 新增 {len(added)} / "
        f"变更 {len(changed)} / 跳过 {len(kept)} / 删除 {len(deleted)}"
    )
    for path in added:
        typer.secho(f"  + {path.name}", fg=typer.colors.GREEN)
    for path in changed:
        typer.secho(f"  ~ {path.name}", fg=typer.colors.CYAN)
    for doc_id in deleted:
        typer.secho(f"  - {doc_id}", fg=typer.colors.YELLOW)

    if dry_run:
        typer.secho("--dry-run：什么都没写。", fg=typer.colors.YELLOW)
        return
    if not added and not changed and not deleted:
        typer.secho("没有需要处理的文件，退出。", fg=typer.colors.GREEN)
        return

    # 语料里删掉的文件先清：**先删块、再删注册行、最后清产物**（顺序见 commit.py）。
    # 放在写之前做，且只做一次 —— 编排层按文件铺开写库时不该再顺手删一遍。
    if deleted:
        _run_async(drop_docs(settings, store, deleted, index=index), settings)

    # --stage all = 前三段一口气跑完：先把解析与切分的产物落上，再写库。
    # 复用同一组函数，所以 all 与分开跑的三段是同一套行为，不会各走各的。
    graph_mode = stage == "all" and settings.graph.enabled
    written = described = registered = 0
    if stage == "all":
        for (_source, root), (_, plan) in zip(sources, plans, strict=True):
            if not plan.todo:
                continue
            if graph_mode:
                # IndexGraph：按**文件**铺开（并发 + 单个文件失败不牵连其余）。
                # 闸门与产物都还是 run_parse / run_chunk 那套，只是编排换了。
                result = _run_async(
                    run_index_graph(
                        settings,
                        todo=plan.todo,
                        root=root,
                        ocr=ocr,
                        vision=vision,
                        store=store,
                        embedder=embedder,
                    ),
                    settings,
                )
                _echo_stage(
                    "解析",
                    len(result.parsed),
                    len(plan.todo) - len(result.parsed),
                    result.parsed,
                    dry_run,
                )
                _echo_stage("切分", len(result.chunked), 0, result.chunked, dry_run)
                if result.described:
                    typer.echo(f"  图片描述 {result.described} 条")
                if result.shells:
                    typer.echo(f"  丢弃空壳块 {result.shells} 个（剥掉标题与图片标记后无内容）")
                _echo_pending(result.pending)
                _echo_skipped(result.skipped)
                written += result.written
                described += result.described
                registered += result.registered
                continue
            parsed = run_parse(plan.todo, root, settings, ocr, store, full=full)
            _echo_stage("解析", parsed.parsed, parsed.reused, parsed.todo, dry_run)
            _echo_skipped(parsed.skipped)
            chunked = run_chunk(plan.todo, root, settings, embedder, vision, ocr, store, full=full)
            _echo_stage("切分", chunked.chunked, chunked.reused, chunked.todo, dry_run)
            if chunked.described:
                typer.echo(f"  图片描述 {chunked.described} 条")
                if chunked.shells:
                    typer.echo(f"  丢弃空壳块 {chunked.shells} 个（剥掉标题与图片标记后无内容）")
            _echo_pending(chunked.pending)
            _echo_skipped(chunked.skipped)

    if not graph_mode:
        # 写库那一段（含「先删后写」「缺产物整批不写」的规矩）抽到了 indexing/commit.py ——
        # 编排层要用同一份实现，规矩只该有一处。
        jobs = [
            (path, root)
            for (_source, root), (_, plan) in zip(sources, plans, strict=True)
            for path in plan.todo
        ]
        committed = _run_async(
            commit_chunks(settings, store, jobs, embedder=embedder, index=index), settings
        )
        if committed.missing:
            typer.secho(f"缺少切分产物：{', '.join(committed.missing)}", fg=typer.colors.YELLOW)
            typer.secho(
                "先跑 `rag ingest <同样的路径> --stage chunk`"
                "（或 --stage parse，两者都不碰 Milvus）。",
                fg=typer.colors.YELLOW,
            )
            raise typer.Exit(code=1)
        written, described = committed.written, committed.described
        registered = len(committed.registered)

    typer.secho(
        f"写入 {written} 个 chunk（其中图片描述 {described} 条），注册 {registered} 篇，完成。",
        fg=typer.colors.GREEN,
    )


@app.command()
def search(
    query: str = typer.Argument("", help="检索词；以图搜图时可以省略，只给 --image"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="返回条数"),
    collection: str | None = typer.Option(None, help="覆盖 collection 名"),
    stats: bool = typer.Option(False, "--stats", help="打印每一路召回的数量与耗时"),
    image: Path | None = typer.Option(
        None, "--image", help="查询图（以图搜图）：可只给图、也可图 + 词一起给"
    ),
    where: list[str] = typer.Option(
        [],
        "--filter",
        help="过滤条件 key=value（可重复；多值用逗号）。例：--filter modality=image",
    ),
    channels: str | None = typer.Option(
        None,
        "--channel",
        help="只跑这些通道（逗号分隔，默认用配置）。写 hybrid 表示只要主通道、不带 bm25",
    ),
    rerank: bool | None = typer.Option(
        None, "--rerank/--no-rerank", help="覆盖 retrieval.rerank.enabled"
    ),
) -> None:
    """纯检索，打印 chunk 与分数（调试用，不调用生成模型）。"""
    if not query.strip() and image is None:
        raise typer.BadParameter("至少给一个：检索词，或 --image（以图搜图）")
    if image is not None and not image.is_file():
        raise typer.BadParameter(f"--image 指向的文件不存在：{image}")
    settings = load_settings()
    _tracing(settings)
    _begin_run()
    if collection:
        settings.indexing.vector.collection = collection
    if channels:
        _apply_channels(settings, channels)
    if rerank is not None:
        settings.retrieval.rerank.enabled = rerank
    filters = _parse_filters(where)

    async def _run() -> tuple[list[ScoredChunk], list]:
        from rag.graphs.wiring import build_heavy
        from rag.indexing import build_index
        from rag.retrieval import build_retriever

        index = build_index(settings)
        await _probe_milvus(index, settings)
        await index.ensure_ready()
        heavy = build_heavy(settings, index)
        _require(heavy, "embedding")
        retriever = build_retriever(settings, index, heavy.embedder, heavy.reranker)
        hits = await retriever.retrieve(query, top_k, filters=filters or None, image=image)
        # 账本挂在多通道检索器上；单路没有这一项，取不到就是空
        return hits, getattr(retriever, "last_stats", [])

    hits, stats_rows = _run_async(_run(), settings)
    marks = [f"图 {image.name}"] if image is not None else []
    marks += [f"过滤 {filters}"] if filters else []
    typer.echo(f"命中 {len(hits)} 个 chunk{'（' + '，'.join(marks) + '）' if marks else ''}：")
    _print_hits(hits)
    if stats:
        _print_stats(stats_rows)


@app.command()
def ask(
    question: str = typer.Argument(..., help="问题"),
    top_k: int = typer.Option(8, "--top-k", "-k", help="召回条数"),
    show_sources: bool = typer.Option(True, "--show-sources/--no-sources", help="是否打印来源"),
    collection: str | None = typer.Option(None, help="覆盖 collection 名"),
    chat: str = typer.Option("default", help="用 providers.chat 里的哪个供应商"),
    stats: bool = typer.Option(False, "--stats", help="打印每一路召回的数量与耗时"),
    where: list[str] = typer.Option(
        [],
        "--filter",
        help="过滤条件 key=value（可重复；多值用逗号）。例：--filter modality=image",
    ),
    channels: str | None = typer.Option(
        None, "--channel", help="只跑这些通道（逗号分隔，默认用配置）"
    ),
    rerank: bool | None = typer.Option(
        None, "--rerank/--no-rerank", help="覆盖 retrieval.rerank.enabled"
    ),
    graph: bool | None = typer.Option(
        None,
        "--graph/--no-graph",
        help="走 QueryGraph（规划 + 检索 + 自查 + 生成）还是 M2 那条直线；默认听 graph.enabled",
    ),
) -> None:
    """检索 + 生成带引用的答案。"""
    settings = load_settings()
    _tracing(settings)
    _begin_run()
    if collection:
        settings.indexing.vector.collection = collection
    if channels:
        _apply_channels(settings, channels)
    if rerank is not None:
        settings.retrieval.rerank.enabled = rerank
    if graph is not None:
        settings.graph.enabled = graph
    filters = _parse_filters(where)

    async def _run() -> tuple[str, list[ScoredChunk], list, dict]:
        from rag.graphs import build_query_graph, run_query_graph
        from rag.graphs.wiring import build_heavy, build_query_nodes
        from rag.indexing import build_index
        from rag.retrieval import assemble, build_retriever

        index = build_index(settings)
        await _probe_milvus(index, settings)
        await index.ensure_ready()
        heavy = build_heavy(settings, index, chat_name=chat)
        _require(heavy, "embedding", "generator")
        if settings.graph.enabled:
            # QueryGraph：plan → retrieve → grade →（不够就改写重查）→ assemble → generate
            retrieve, plan_llm, grade_llm, generate = build_query_nodes(settings, index, heavy)
            # 计划里能选的通道名 = 这次启用的那几路（`retrieval.channels[].name`）
            known = [item.name for item in settings.retrieval.channels if item.enabled]
            graph_app = build_query_graph(
                settings,
                retrieve=retrieve,
                plan_llm=plan_llm,
                grade_llm=grade_llm,
                generate=generate,
                channels=known,
            )
            result = await run_query_graph(graph_app, question, top_k=top_k)
            return (
                result["answer"],
                result["hits"],
                result["stats"],
                {
                    "rounds": result["rounds"],
                    "grade": result["grade"],
                },
            )
        retriever = build_retriever(settings, index, heavy.embedder, heavy.reranker)
        hits = await retriever.retrieve(question, top_k, filters=filters or None)
        context = assemble(hits, settings.retrieval.assemble.max_tokens)
        answer = await heavy.generator.generate(question, context)
        return answer, hits, getattr(retriever, "last_stats", []), {}

    answer, hits, stats_rows, meta = _run_async(_run(), settings)
    typer.echo("")
    typer.echo(answer)
    if show_sources and hits:
        typer.echo("")
        typer.echo("来源：")
        _print_hits(hits)
    if stats:
        _print_stats(stats_rows)
    if meta:
        grade = meta.get("grade") or {}
        typer.echo(
            f"\n编排：{meta['rounds']} 轮，自评 {float(grade.get('score') or 0):.2f}"
            f"（阈值 {settings.graph.relevance_threshold}）"
            + (f"，缺：{grade['missing']}" if grade.get("missing") else "")
        )


def _begin_run() -> None:
    """给这一轮运行设关联 id（接口 / agent / 通道三类日志共用）。

    日志的落点由 CLI 根回调里的 `configure_logging` 装好，这里不再接落点。
    入口那个 `request_id` 由根回调发（一次命令一个），这里补的是**这一轮**的 `run_id`。
    """
    from rag.obs import new_id, set_ids

    set_ids(run_id=new_id("run"))


def _tracing(settings) -> None:
    """按配置开 LangSmith trace；**只在有话说的时候出声**（开了 / 配错了）。

    关着是默认档，每次刷一行「未开」只会让人习惯性忽略它 —— 那样配错时也就没人看见了。
    """
    from rag.observability import configure_tracing

    status = configure_tracing(settings)
    if not status.startswith("关闭"):
        typer.secho(f"LangSmith：{status}", fg=typer.colors.CYAN)


def _decisions(
    *, approve: bool, reject: bool, specs: list[str], count: int
) -> list[dict[str, Any]]:
    """把命令行给的「怎么处理」翻成 HITL 中间件要的形状。

    `--approve` / `--reject` 是**一键全批**（`count` 由中断状态数出来，不用人手数）；
    `--decision` 是逐个，**条数必须与待批操作对上** —— 对不上就在这儿报错，别等中间件
    在模型那头抛一句「decisions 数量不对」，那种错没人看得懂。
    """
    if count == 0:
        raise typer.BadParameter("这个会话没有等人批准的操作（可能已经跑完了）。")
    if approve or reject:
        if specs:
            raise typer.BadParameter("--approve / --reject 与 --decision 只用一种。")
        if approve and reject:
            raise typer.BadParameter("--approve 与 --reject 只能给一个。")
        return [{"type": "approve" if approve else "reject"} for _ in range(count)]
    if not specs:
        raise typer.BadParameter(
            "--resume 要说清怎么处理：--approve / --reject，或逐个 --decision。"
        )
    if len(specs) != count:
        raise typer.BadParameter(f"待批的操作有 {count} 条，--decision 给了 {len(specs)} 条。")
    out: list[dict[str, Any]] = []
    for spec in specs:
        kind, _, rest = spec.partition(":")
        kind = kind.strip().lower()
        if kind in ("approve", "reject"):
            out.append({"type": kind})
        elif kind == "respond":
            out.append({"type": "respond", "message": rest})
        elif kind == "edit":
            try:
                out.append({"type": "edit", "args": json.loads(rest)})
            except ValueError as exc:
                raise typer.BadParameter(f"--decision edit 后面要跟 JSON 参数：{exc}") from None
        else:
            raise typer.BadParameter(
                f"不认识的决定 {kind!r}；可选 approve / reject / respond / edit。"
            )
    return out


def _print_interrupt(result, pg_state: str) -> None:
    """把「它想干什么」摆在人眼前，并把**下一步该敲什么**写全。

    光打一句「需要批准」等于让人去翻文档；这里连 `--resume` 的命令都拼好，
    复制就能用。
    """
    actions = (result.interrupt or {}).get("action_requests") or []
    typer.secho(f"⏸ 它要动磁盘 / 执行命令，等你批准（{len(actions)} 条）：", fg=typer.colors.YELLOW)
    for i, action in enumerate(actions, 1):
        args = json.dumps(action.get("args") or {}, ensure_ascii=False)
        typer.secho(f"  {i}. {action.get('name', '?')} {_preview(args, 160)}", fg=typer.colors.CYAN)
        # description 里是中间件给的原文（工具名 + 参数），重复一遍没意义，只留第一行说明
        desc = (action.get("description") or "").strip().splitlines()
        if desc and desc[0] not in ("Tool execution requires approval", ""):
            typer.echo(f"     {desc[0]}")
    thread = result.thread_id or ""
    typer.echo("")
    typer.echo(f"  批准：rag agent --resume --thread-id {thread} --approve")
    typer.echo(f"  拒绝：rag agent --resume --thread-id {thread} --reject")
    typer.echo(
        f"  逐个：rag agent --resume --thread-id {thread} --decision approve --decision reject"
    )
    typer.echo(f"  （会话={thread}；PG={pg_state}；它一个字都还没写）")


@app.command()
def agent(
    question: str | None = typer.Argument(None, help="问题；--resume 时不用给"),
    thread_id: str | None = typer.Option(
        None, "--thread-id", help="会话 id：给了就是多轮（历史存在 PG 的 checkpointer 里）"
    ),
    resume: bool = typer.Option(
        False, "--resume", help="接着被中断的那次跑（要配 --thread-id 与一个决定）"
    ),
    approve: bool = typer.Option(False, "--approve", help="批准它想做的**全部**操作"),
    reject: bool = typer.Option(False, "--reject", help="拒绝全部操作"),
    decision: list[str] | None = typer.Option(
        None,
        "--decision",
        help="逐个决定，可重复：approve / reject / respond:文字 / edit:{JSON 参数}",
    ),
    writes: bool | None = typer.Option(
        None,
        "--writes/--no-writes",
        help="给不给 agent 落盘能力；默认按 agent.workspace_writes（真）",
    ),
    web: bool = typer.Option(
        False,
        "--web",
        help="**用户明确要求联网**：放行网络搜索（默认只在语料检索查空时才允许）",
    ),
    show_sources: bool = typer.Option(True, "--show-sources/--no-sources", help="是否打印引用"),
) -> None:
    """对话式 agent（M5，deepagents）：主 agent 自己决定查什么、派给哪个子 agent。

    与 `rag ask` 的区别：`ask` 是我们定好的直线/图（检索 → 装配 → 生成），
    agent 是**让模型自己决定**要不要查、查几轮、派给谁 —— 代价是 LLM 调用次数多得多。
    子 agent 默认 `mode="isolated"`：只把结论交回主 agent，中间的检索噪声不进主上下文。

    批次 B 起它**能往 `data/workspace` 写文件**（`--no-writes` 可关），但写盘 / 执行
    命令这四类工具**一律停下来等人批准**：那时命令打一遍「它想干什么」并以退出码
    **3** 结束，批准或拒绝用 `--resume`。
    """
    if resume and not thread_id:
        raise typer.BadParameter("--resume 要配 --thread-id：中断状态是按会话存进 PG 的。")
    if resume and question:
        raise typer.BadParameter("--resume 不用再给问题：接着的是上次那一轮。")
    if not resume and not question:
        raise typer.BadParameter(
            "给一个问题；或者用 --resume --thread-id <会话> --approve 继续被中断的那次。"
        )
    # PG 的 async saver 需要 Selector 循环（Windows），必须在建循环之前设 —— 越早越好
    from rag.persistence import ensure_compatible_event_loop

    settings = load_settings()
    _tracing(settings)
    if writes is not None:
        settings.agent.workspace_writes = writes
    if settings.persistence.postgres.url:
        ensure_compatible_event_loop()

    async def _run() -> tuple[object, str]:
        from rag.agent import (
            build_agent,
            default_workspace,
            load_registry,
            pending_actions,
            resume_agent,
            run_agent,
            save_registry,
        )
        from rag.graphs.wiring import build_heavy
        from rag.indexing import build_index
        from rag.persistence import build_persistence

        index = build_index(settings)
        await _probe_milvus(index, settings)
        await index.ensure_ready()
        heavy = build_heavy(settings, index)
        _require(heavy, "embedding", "generator")
        persistence = build_persistence(settings)
        await persistence.connect()  # 连不上就降级成无状态（下面会说）
        # **先记下状态再关**：`close()` 会把 checkpointer/store 置空，那时候再看
        # `available` 永远是 False，打印出来就是一句空话（第一版就是这么错的）。
        pg_state = "已连" if persistence.available else (persistence.reason or "不可用")
        from rag.obs import new_id, set_ids

        set_ids(run_id=new_id("run"))  # 本轮运行的关联 id（通道与 agent 日志共用）
        if resume and not persistence.available:
            # 中断状态在 PG 里，连不上就没得恢复 —— 明说，别让它以一句英文异常收场
            typer.secho(
                f"--resume 要读 PG 里的中断状态，但 PG 不可用（{pg_state}）", fg=typer.colors.RED
            )
            raise typer.Exit(code=1)
        if thread_id and not persistence.available:
            # 与 API 同一个判断（`Components._agent_persistence_check`）：多轮必须有 checkpointer，
            # 静默降级成单轮最坏 —— 调用方会以为「历史在，只是模型忘了」。
            typer.secho(
                f"--thread-id 要多轮历史，但 PG 不可用（{pg_state}）；不接 checkpointer 的"
                "多轮是无状态的，去掉 --thread-id 或先把 PG 连上。",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        graph, registry = build_agent(
            settings,
            heavy,
            checkpointer=persistence.checkpointer,
            store=persistence.store,
            workspace=default_workspace(settings),
            web_explicit=web,
        )
        if resume:
            # 待批条数得从**图的状态**里读：人是在另一个进程里敲这条命令的，
            # `--approve` 要按条数铺开，不能靠猜。
            actions = await pending_actions(graph, thread_id or "")
            decisions = _decisions(
                approve=approve, reject=reject, specs=decision or [], count=len(actions)
            )
            # 中断前的引用表按会话存过（save_registry），取回来答案里的 `[n]` 才对得上
            registry = await load_registry(persistence.store, thread_id or "")
            result = await resume_agent(
                graph, thread_id=thread_id or "", decisions=decisions, registry=registry
            )
        else:
            result = await run_agent(graph, question or "", registry=registry, thread_id=thread_id)
        if result.status == "interrupted" and result.thread_id:
            await save_registry(persistence.store, result.thread_id, registry)
        await persistence.close()
        return result, pg_state

    result, pg_state = _run_async(_run(), settings)
    typer.echo("")
    if result.status == "interrupted":
        _print_interrupt(result, pg_state)
        # 退出码 3 = 「没失败，在等人」：脚本里能一句话分辨「要人管」和「真出错」
        raise typer.Exit(code=3)
    typer.echo(result.answer)
    if show_sources and result.citations:
        typer.echo("")
        typer.echo("引用：")
        for item in result.citations:
            typer.secho(f"  [{item.index}] {item.source_uri}  {item.snippet}", fg=typer.colors.CYAN)
    typer.echo(
        f"\n（步骤 {result.steps}，本次召回 {result.retrieved} 条，"
        f"被引用 {len(result.citations)} 条；"
        # 打 result 里的 thread_id：单轮在接了 checkpointer 时会**自动生成**一个
        # （langgraph 强制要求），把它显示出来才能接着聊那一轮。
        f"会话={result.thread_id or '无（未接 checkpointer）'}；"
        f"PG={pg_state}）"
    )


# ---- 会话（PG 里的 checkpointer 残留）----

# LangGraph 的 checkpointer **没有自动清理**，所以给它一个显式的出口。
# 不做自动 TTL：中断（等人批准）也存这里，按时间自动删会把人还在等的那次一起删掉。
sessions_app = typer.Typer(
    help="PG 里的会话（LangGraph 的 checkpointer 残留）：看清楚有哪些、按需删掉",
    no_args_is_help=True,
)
app.add_typer(sessions_app, name="sessions")


def _sessions_store(settings) -> SessionStore:
    """建会话账本；没配 PG 就直接说清（会话落在进程内，没有表可清）。"""
    from rag.persistence import build_session_store

    if not settings.persistence.postgres.url:
        typer.secho(
            "没配 PG（persistence.postgres.url 为空）：会话只在进程内，没有残留可清。",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=1)
    return build_session_store(settings, get_secrets())


@sessions_app.command("list")
def sessions_list() -> None:
    """列出所有会话：版本数、待写的行、**有没有在等人批准**、最新一版 id。"""
    from psycopg.errors import UndefinedTable

    settings = load_settings()
    store = _sessions_store(settings)
    try:
        sessions = store.list()
    except UndefinedTable:
        typer.secho("PG 里还没有这些表：先跑 `rag sessions init` 建表。", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1) from None

    if not sessions:
        typer.secho(
            f"没有会话（schema {settings.persistence.postgres.schema_name}）。",
            fg=typer.colors.GREEN,
        )
        return
    typer.echo(f"{'会话':<28} {'版本':>6} {'待写':>6} {'最新一版':<38} 状态")
    for item in sessions:
        state = "⏸ 等人批准" if item.waiting_for_approval else ""
        typer.echo(
            f"{item.thread_id:<28} {item.versions:>6} {item.writes:>6} "
            f"{item.last_checkpoint_id:<38} {state}"
        )
    waiting = [item.thread_id for item in sessions if item.waiting_for_approval]
    typer.echo(f"\n共 {len(sessions)} 个会话。")
    if waiting:
        typer.secho(
            f"其中 {len(waiting)} 个停在等人批准：{', '.join(waiting)} —— 删掉它们等于把"
            "「待批准的那次操作」一起扔掉。",
            fg=typer.colors.YELLOW,
        )


@sessions_app.command("init")
def sessions_init() -> None:
    """建 PG 的表（幂等）。**这是一次写操作**，所以单独一条命令、由人显式触发。

    需要它是因为：库自己只 `CREATE TABLE IF NOT EXISTS`、**不建 schema**，
    而建表这一步以前没有入口（表是手工建的）。
    """
    from rag.persistence import build_persistence, ensure_compatible_event_loop

    settings = load_settings()
    store = _sessions_store(settings)

    async def _run():
        persistence = build_persistence(settings, get_secrets())
        if not await persistence.connect():
            return persistence.reason or "连不上"
        await persistence.setup()
        await persistence.close()
        return ""

    # PSycopg 的 async 需要 Selector 循环，必须在建循环之前设
    ensure_compatible_event_loop()
    # schema 得我们自己建：库里只有 CREATE TABLE IF NOT EXISTS
    store.ensure_schema(settings.persistence.postgres.schema_name)
    reason = _run_async(_run(), settings)
    if reason:
        typer.secho(f"建表失败：{reason}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.secho(
        f"表已就绪（schema {settings.persistence.postgres.schema_name}）。", fg=typer.colors.GREEN
    )


@sessions_app.command("prune")
def sessions_prune(
    thread_id: list[str] = typer.Option(
        [], "--thread-id", help="要删的会话 id，可重复；与 --keep 二选一"
    ),
    keep: int | None = typer.Option(
        None, "--keep", help="只保留最新的 N 个会话（uuid6 序），其余删掉"
    ),
    force: bool = typer.Option(
        False, "--force", help="连「停在等人批准」的会话也一起删（默认跳过它们）"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="只打印要删什么，不真删"),
) -> None:
    """删掉会话残留（`checkpoints` / `checkpoint_blobs` / `checkpoint_writes` + 引用表）。

    **删了就是删了**：那次会话的多轮历史、以及「停下等人批准」的中断都会没。
    默认跳过正停在等人批准那一步的会话，要一起删得加 `--force`。
    """
    from psycopg.errors import UndefinedTable

    from rag.persistence import select_prune_targets

    if bool(thread_id) == (keep is not None):
        raise typer.BadParameter("--thread-id 与 --keep 二选一：给具体会话，或说清保留几个。")
    if keep is not None and keep < 0:
        raise typer.BadParameter("--keep 不能是负数。")

    settings = load_settings()
    store = _sessions_store(settings)
    try:
        sessions = store.list()
    except UndefinedTable:
        typer.secho(
            "PG 里还没有这些表（没东西可清）：先跑 `rag sessions init`。", fg=typer.colors.YELLOW
        )
        raise typer.Exit(code=1) from None

    known = {item.thread_id for item in sessions}
    unknown = [item for item in thread_id if item not in known]
    if unknown:
        # 写错 id 时静默删掉 0 行最糟：人会以为已经清干净了
        typer.secho(
            f"没有这些会话：{', '.join(unknown)}（用 `rag sessions list` 看有哪些）",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    targets = (
        [item for item in sessions if item.thread_id in set(thread_id)]
        if thread_id
        else select_prune_targets(sessions, keep=keep)
    )
    waiting = [item.thread_id for item in targets if item.waiting_for_approval]
    if waiting and not force:
        skipped = ", ".join(waiting)
        typer.secho(
            f"跳过正在等人批准的会话：{skipped}（要一起删就加 --force）", fg=typer.colors.YELLOW
        )
        targets = [item for item in targets if not item.waiting_for_approval]
    if not targets:
        typer.secho("没有要删的会话。", fg=typer.colors.GREEN)
        return

    for item in targets:
        typer.echo(f"  - {item.thread_id}（{item.versions} 版 / {item.writes} 行待写）")
    if dry_run:
        typer.secho(f"--dry-run：以上 {len(targets)} 个会话没有删。", fg=typer.colors.YELLOW)
        return

    counts = store.prune([item.thread_id for item in targets])
    detail = "，".join(f"{table} {rows} 行" for table, rows in counts.items())
    typer.secho(f"删掉 {len(targets)} 个会话（{detail}）。", fg=typer.colors.GREEN)


# ---- 日志（三类，各落一个文件）----

logs_app = typer.Typer(
    help="日志：接口 / agent 运行 / 查询通道，三类各落一个文件（data/logs/ 下）",
    no_args_is_help=True,
)
app.add_typer(logs_app, name="logs")


chats_app = typer.Typer(help="UI 会话（对话记录与 token 账）。**与 `rag sessions` 不是一回事**")
app.add_typer(chats_app, name="chats")


@chats_app.command("init")
def chats_init() -> None:
    """建会话三张表（幂等）。**这是一次写操作**，所以要显式触发。"""
    from rag.persistence import build_chat_store, build_persistence, ensure_compatible_event_loop
    from rag.persistence.chat import SCHEMA

    settings = load_settings()
    if not settings.persistence.postgres.url:
        typer.secho(
            "没配 PG（persistence.postgres.url 为空），会话没有地方存。", fg=typer.colors.YELLOW
        )
        raise typer.Exit(code=1)
    ensure_compatible_event_loop()

    async def _run() -> str:
        persistence = build_persistence(settings, get_secrets())
        if not await persistence.connect():
            return persistence.reason or "连不上"
        await build_chat_store(settings, persistence).setup()
        await persistence.close()
        return ""

    reason = _run_async(_run(), settings)
    if reason:
        typer.secho(f"建表失败：{reason}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    tables = "、".join(statement.split()[5] for statement in SCHEMA if "create table" in statement)
    typer.secho(
        f"会话表已就绪：{tables}（schema {settings.persistence.postgres.schema_name}）",
        fg=typer.colors.GREEN,
    )


@chats_app.command("list")
def chats_list(limit: int = typer.Option(20, "--limit", "-n", help="看最近几段")) -> None:
    """列最近几段会话：标题 / 模式 / 轮数 / token。**只读。**"""
    from rag.persistence import build_chat_store, build_persistence, ensure_compatible_event_loop

    settings = load_settings()
    if not settings.persistence.postgres.url:
        typer.secho("没配 PG，没有会话可列。", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
    ensure_compatible_event_loop()

    async def _run() -> list[dict[str, Any]]:
        persistence = build_persistence(settings, get_secrets())
        if not await persistence.connect():
            raise RuntimeError(persistence.reason or "连不上")
        try:
            return await build_chat_store(settings, persistence).list(limit=limit)
        finally:
            await persistence.close()

    try:
        rows = asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"读失败：{type(exc).__name__}: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    if not rows:
        typer.echo("还没有会话。")
        return
    typer.echo(f"{'session_id':<14}{'模式':<8}{'轮数':>4}{'token':>10}  标题")
    for row in rows:
        typer.echo(
            f"{row['session_id']:<14}{row['mode']:<8}{row['turns']:>4}"
            f"{(row['total_tokens'] or 0):>10}  {(row['title'] or '')[:40]}"
        )


@logs_app.command("tail")
def logs_tail(
    limit: int = typer.Option(20, "--limit", "-n", help="看最近几行"),
    kind: str | None = typer.Option(
        None,
        "--kind",
        "-k",
        help="只看一类：interface / channel / system / agent_run / agent_llm / agent_tool",
    ),
) -> None:
    """看最近的日志行 —— **直接读文件尾巴，不连数据库**。

    一个文件里可能混着几类（`agent.log` 装着 agent_run / agent_llm / agent_tool），
    所以 `--kind` 是按**行**过滤的，不是按文件；默认把所有类别合成一条流水（按时间排）。
    文件路径来自 `observability.logging`（base.yaml）。内容字段（问题/查询原文）只在
    `logging.content=true` 时才有值。
    """
    from rag.obs.log import tail_items

    logging_cfg = load_settings().observability.logging
    files = {name: item.path for name, item in (logging_cfg.files or {}).items()}
    files["system"] = logging_cfg.system.path
    if kind is not None and kind not in files:
        raise typer.BadParameter(f"--kind 只认这几个：{'、'.join(files)}")

    items = tail_items(files, kind=kind or "all", limit=limit)
    if not items:
        typer.secho(
            "（没有日志行 —— 看看 observability.logging 里的路径配了没）", fg=typer.colors.YELLOW
        )
        return
    for item in items:
        # 类别与来源不同时才多打一个 `[类别]`（system 里的行来源是 ocr / uvicorn 这些）
        tag = f"[{item['kind']}] " if item["kind"] != item["source"] else ""
        when = item["ts"] or "----"
        level = f"{item['level'] or '-':<7}"
        typer.echo(f"{when} {level} {tag}{item['source']:<15} {item['text']}")


@logs_app.command("init")
def logs_init() -> None:
    """建日志索引表 `log_line`（幂等）。**这是一次写操作**，所以要显式触发。

    表在 PG 里，与 chat 那三张表同一个库、同一个 schema。**文件仍是唯一真相**：
    这张表只是为了让「按 request_id / 类别 / 时间查」不必去扫一堆文件（也没有索引）。
    没建它也能照常跑 —— 只是每一行进库那一步会失败，而那种失败只会在 stderr 上喊一句。
    """
    from rag.persistence import build_persistence, ensure_compatible_event_loop
    from rag.persistence.logs import SCHEMA, build_log_store

    settings = load_settings()
    if not settings.persistence.postgres.url:
        typer.secho(
            "没配 PG（persistence.postgres.url 为空），日志没有地方建索引。", fg=typer.colors.YELLOW
        )
        raise typer.Exit(code=1)
    ensure_compatible_event_loop()

    async def _run() -> str:
        persistence = build_persistence(settings, get_secrets())
        if not await persistence.connect():
            return persistence.reason or "连不上"
        await build_log_store(settings, persistence).setup()
        await persistence.close()
        return ""

    reason = _run_async(_run(), settings)
    if reason:
        typer.secho(f"建表失败：{reason}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    tables = "、".join(statement.split()[5] for statement in SCHEMA if "create table" in statement)
    typer.secho(
        f"日志索引表已就绪：{tables}（schema {settings.persistence.postgres.schema_name}）",
        fg=typer.colors.GREEN,
    )


@logs_app.command("search")
def logs_search(
    request_id: str = typer.Argument(..., help="要串起来的关联 id（如 req-ab12cd34）"),
    kind: str = typer.Option("", "--kind", "-k", help="只看一类；留空 = 全部"),
    since: str = typer.Option("", "--since", help="起始时间（ISO 8601，含）"),
    until: str = typer.Option("", "--until", help="结束时间（ISO 8601，含）"),
    limit: int = typer.Option(200, "--limit", "-n", help="最多几行"),
) -> None:
    """按 `request_id` 从 PG 里把一条链路查出来。**只读。**

    与 `rag logs tail` 的分工：那边读文件尾巴（永远最新、没有索引），这边查库
    （能按 id 串起 interface + channel + agent，也能按类别与时间收窄）。库里那份是
    **异步批量**写的，刚发生的一两行可能还没进去 —— 要最新的就用 tail。
    """
    from datetime import datetime

    from rag.persistence import build_persistence, ensure_compatible_event_loop
    from rag.persistence.logs import build_log_store

    def _when(text: str) -> Any:
        stem = text.strip()
        if not stem:
            return None
        try:
            parsed = datetime.fromisoformat(stem.replace("Z", "+00:00"))
        except ValueError:
            raise typer.BadParameter(f"时间格式看不懂：{text!r}（要 ISO 8601）") from None
        return parsed if parsed.tzinfo else parsed.astimezone()

    settings = load_settings()
    if not settings.persistence.postgres.url:
        typer.secho("没配 PG，日志索引查不了；用 `rag logs tail` 看文件。", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
    ensure_compatible_event_loop()

    async def _run() -> list[dict[str, Any]]:
        persistence = build_persistence(settings, get_secrets())
        if not await persistence.connect():
            raise RuntimeError(persistence.reason or "连不上")
        try:
            store = build_log_store(settings, persistence)
            return await store.search(
                request_id=request_id,
                category=kind,
                since=_when(since),
                until=_when(until),
                limit=limit,
            )
        finally:
            await persistence.close()

    try:
        rows = asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"查失败：{type(exc).__name__}: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    if not rows:
        typer.secho(
            "库里没有匹配的行（历史不回收；刚发生的行可能还在批量队列里）。",
            fg=typer.colors.YELLOW,
        )
        return
    for row in rows:
        when = row["ts"].astimezone() if row["ts"] else None
        stamp = f"{when:%m-%d %H:%M:%S}.{int(when.microsecond / 1000):03d}" if when else "----"
        origin = f" {row['source']}" if row.get("source") else ""
        run_id = f" run={row['run_id']}" if row.get("run_id") else ""
        typer.echo(
            f"{stamp} {row['level']:<7} [{row['category']}] {row.get('logger') or '-':<16}"
            f"{origin}{run_id} {row['text']}"
        )
    typer.echo(f"\n共 {len(rows)} 行（req={request_id}）")


@app.command("eval")
def eval_cmd(
    dataset: Path = typer.Option(
        Path("eval/golden.jsonl"), "--dataset", help="黄金集（JSONL，一条一个问题 + 判据）"
    ),
    top_k: int = typer.Option(8, "--top-k", "-k", help="检索几条"),
    judge: bool = typer.Option(
        False,
        "--judge",
        help="跑生成侧判定（引用准确率 / 要点覆盖 / 拒答）：**每次问题一次模型调用，按量计费**",
    ),
    ragas: bool = typer.Option(
        False,
        "--ragas",
        help="额外算 ragas 的 faithfulness（要 --judge 才有生成结果；每个问题再加一次调用）",
    ),
    label: str = typer.Option("baseline", "--label", help="报告文件名前缀，做 A/B 时改它"),
    channels: str | None = typer.Option(
        None, "--channel", help="覆盖可选通道（同 search：逗号分隔，如 hybrid,bm25）"
    ),
    rerank: bool | None = typer.Option(None, "--rerank/--no-rerank", help="覆盖重排开关"),
    graph: bool | None = typer.Option(
        None, "--graph/--no-graph", help="覆盖 graph.enabled（评测默认走直线，与线上 ask 一致）"
    ),
    out_dir: Path = typer.Option(Path("eval/reports"), "--out-dir", help="报告目录"),
) -> None:
    """跑评测：检索指标（免费）+ 可选的生成侧判定（`--judge`，花钱）。

    默认**只跑检索那层**：本地 embedding，不调模型、不花钱，随时能跑。指标是
    `recall@k` / `MRR` / `nDCG@k` / 命中率 —— 它回答的是「材料找不找得到」。

    `--judge` 再加生成侧：**引用准确率**（答案里的引用编号是否真支持那句话）、
    **要点覆盖率**、以及语料外问题**该不该拒答**。这部分每个问题一次模型调用。

    报告两份落 `eval/reports/`（JSON 给下次 diff、Markdown 给人看）。做 A/B 就改 `--label`
    再跑一次，比如 `--label no-rerank --no-rerank`，然后对比两份 JSON 的 summary。
    """
    from rag.eval import build_report, load_golden, run_eval, write_report
    from rag.eval.wiring import build_ask, build_generate, build_retrieve

    settings = load_settings()
    _tracing(settings)
    _begin_run()
    if channels:
        _apply_channels(settings, channels)
    if rerank is not None:
        settings.retrieval.rerank.enabled = rerank
    if graph is not None:
        settings.graph.enabled = graph

    cases = load_golden(dataset)

    async def _run():
        from rag.graphs.wiring import build_heavy
        from rag.indexing import build_index

        index = build_index(settings)
        await _probe_milvus(index, settings)
        await index.ensure_ready()
        heavy = build_heavy(settings, index)
        _require(heavy, "embedding")
        retrieve = build_retrieve(settings, heavy)
        generate = ask = None
        faithfulness = None
        if judge:
            _require(heavy, "chat", "generator")
            generate, ask = build_generate(settings, heavy), build_ask(heavy)
            if ragas:
                from rag.eval.ragas_backend import build_faithfulness

                faithfulness = build_faithfulness(settings, get_secrets())
        return await run_eval(
            cases,
            retrieve=retrieve,
            generate=generate,
            ask=ask,
            faithfulness=faithfulness,
            top_k=top_k,
        )

    rows = _run_async(_run(), settings)
    report = build_report(
        rows,
        top_k=top_k,
        meta={
            "label": label,
            "dataset": str(dataset),
            "config": {
                "collection": settings.indexing.vector.collection,
                "embedding": settings.providers.embedding.backend,
                "channels": [item.name for item in settings.retrieval.channels if item.enabled],
                "rerank": settings.retrieval.rerank.enabled,
                "graph": settings.graph.enabled,
                "top_k": top_k,
                "judge": judge,
                "ragas": ragas,
            },
        },
    )
    json_path, md_path = write_report(report, out_dir=out_dir, label=label)

    summary = report["summary"]
    typer.echo("")
    typer.secho(
        f"检索 {summary['retrieval_cases']} 条："
        + "  ".join(
            f"{name}={value * 100:.1f}%" for name, value in (summary["retrieval"] or {}).items()
        ),
        fg=typer.colors.CYAN,
    )
    if judge:
        typer.secho(
            f"生成 {summary['judged_cases']} 条：引用准确率"
            f"={_pct(summary['citation_accuracy'])}  要点覆盖={_pct(summary['point_coverage'])}  "
            f"拒答={summary['refused']}  判不了={summary['errors']}"
            + (f"  ragas 忠实度={_pct(summary['ragas_faithfulness'])}" if ragas else ""),
            fg=typer.colors.CYAN,
        )
    missed = [row for row in rows if row.retrieval_case and row.rank == 0]
    if missed:
        typer.secho(
            f"\n{len(missed)} 条没命中（报告里列了它们召回了什么）：", fg=typer.colors.YELLOW
        )
        for row in missed[:5]:
            top = row.top[0]
            typer.echo(f"  · {row.question[:40]}  → 召回 #{1} {top['heading_path'][:50]}")
    typer.secho(f"\n报告：{md_path}（机器读的在同一目录的 .json）", fg=typer.colors.GREEN)


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


@app.command()
def serve(
    host: str | None = typer.Option(None, help="监听地址；默认用 api.host"),
    port: int | None = typer.Option(None, help="监听端口；默认用 api.port"),
    reload: bool = typer.Option(False, "--reload", help="改代码自动重启（仅开发用）"),
) -> None:
    """启动 FastAPI 服务（M4）。

    端点：`/health`、`/search`、`/chat`（`stream=true` 走 SSE）、`/ingest`（202 + job_id）、
    `/jobs/{id}`。鉴权走 `X-API-Key`（`.env` 的 `RAG_API_KEY`；没配就只允许本机）。

    **单 worker**：任务状态与模型单例都在进程内，多 worker 会让内存翻倍且 `/jobs` 查不到
    别的 worker 提交的任务。要横向扩展得先把状态挪出去（Redis 已经接上了任务状态，
    但模型单例仍在进程内）。

    **loop 必须显式指到我们的工厂**：uvicorn ≥0.36 不再看事件循环策略，它在 Windows 上
    自己建 `ProactorEventLoop`，而 psycopg 的 async 在那上面直接拒绝启动（实测）。详见
    `rag.persistence.selector_event_loop`。
    """
    import uvicorn

    from rag.api import create_app

    settings = load_settings()
    if host or port:
        settings.api.host = host or settings.api.host
        settings.api.port = port or settings.api.port
    # **在 uvicorn 之前把 logging 装好**：装的是 root logger，第三方库（LangChain、
    # pymilvus、httpx…）的日志也会走同一套 handler —— 它们以前只能在 stderr 上裸奔
    from rag.obs import configure_logging

    configure_logging(settings)
    typer.echo(
        f"服务：http://{settings.api.host}:{settings.api.port}  "
        f"（鉴权=已关闭（2026-10-06 起不再用 RAG_API_KEY），"
        f"限流={settings.api.rate_limit_per_min}/分钟，"
        f"Redis={settings.api.redis.url or '未配（退回进程内）'}）"
    )
    uvicorn.run(
        create_app(settings),
        host=settings.api.host,
        port=settings.api.port,
        reload=reload,
        workers=1,
        # 接口日志我们自己记（带 request_id，与 agent/通道日志串得起来）；
        # 再叠一份 uvicorn 的 access log 只会重复且格式对不上
        access_log=False,
        # **别让 uvicorn 自己配一遍 handler**：它会盖掉我们那套格式（时间/关联 id），
        # 结果同一个进程里两套格式混着打
        log_config=None,
        # 别删：少了它，只要配了 PG，服务就会在 startup 崩在 ProactorEventLoop 上
        loop="rag.persistence:selector_event_loop",
    )


def _mount_plugin_commands() -> None:
    """把**已装插件**声明的命令挂到 `rag` 下（entry point 组名 `rag.cli`）。

    与通道那套是同一个设计：核心不认识 Neo4j，插件在自己的 pyproject 里声明
    `[project.entry-points."rag.cli"] graph = "pkg.cli:app"`，装了就多出 `rag graph …`。
    **插件坏了不该拖垮整个 CLI**：import 失败只打一行警告，别的命令照常可用。
    """
    from importlib import metadata

    for item in sorted(metadata.entry_points(group="rag.cli"), key=lambda ep: ep.name):
        try:
            sub = item.load()
        except Exception as exc:  # noqa: BLE001 —— 插件的事，不该让 CLI 起不来
            typer.secho(
                f"插件命令 {item.name!r} 加载失败（{type(exc).__name__}: {exc}）；"
                f"先 `pip install -e plugins/<插件目录>` 重装一次。",
                fg=typer.colors.YELLOW,
            )
            continue
        app.add_typer(sub, name=item.name)


# **模块级挂**：Typer/Click 先解析子命令再进回调，挂晚了 `rag graph …` 会认不出来。
_mount_plugin_commands()

models_app = typer.Typer(
    help="模型选型（槽位 → 谁）：存库 + 审计。**这一版写入后重启生效**（热切是遗留项）",
    no_args_is_help=True,
)
app.add_typer(models_app, name="models")


@models_app.command("init")
def models_init() -> None:
    """建两张表（幂等）：`model_setting`（当前选型）与 `model_setting_change`（审计）。

    **这是一次写操作**，所以要显式触发。库的选法见 `models.backend`：
    `auto`（默认）= 配了 PG 用 PG，否则单机 SQLite（`models.sqlite_path`）。
    """
    from rag.persistence.model_settings import build_model_setting_store

    settings = load_settings()
    store = build_model_setting_store(settings)
    store.setup()
    typer.secho(
        f"模型选型表已就绪：{store.name}（backend={settings.models.backend}）",
        fg=typer.colors.GREEN,
    )


@models_app.command("list")
def models_list() -> None:
    """八个槽位 + 当前选的谁 + **这份值是哪层来的**。只读。"""
    from rag.config.schema import MODEL_SLOTS
    from rag.persistence.model_settings import build_model_setting_store

    settings = load_settings()
    stored = build_model_setting_store(settings).all()
    typer.echo(f"{'槽位':<14}{'来源':<10}{'provider_key':<16}{'model':<34}{'版本':>4}  更新时间")
    for slot in MODEL_SLOTS:
        row = stored.get(slot)
        if row is None:
            typer.echo(f"{slot:<14}{'未设置':<10}{'-':<16}{'（沿用 .env / yaml）':<34}{'-':>4}  -")
            continue
        typer.echo(
            f"{slot:<14}{'库':<10}{row['provider_key'] or '-':<16}"
            f"{(row['model'] or '-'):<34}{row['version']:>4}  {row['updated_at']}"
        )
    typer.echo("\n优先级：**库 > .env > yaml 默认**（库里没有的槽位就走原处那份配置）")


@models_app.command("set")
def models_set(
    slot: str = typer.Argument(..., help="槽位名，如 chat.agent / rerank"),
    provider_key: str = typer.Option(
        "", "--provider-key", help="chat 指 providers.chat 的键名；rerank 指 backend"
    ),
    model: str = typer.Option("", "--model", help="模型名或本地路径"),
    actor: str = typer.Option("", "--actor", help="谁改的（写进审计）"),
) -> None:
    """写一个槽位（写库 + **写一行审计**）。**重启服务才生效** —— 这一版不做热切。"""
    from rag.persistence.model_settings import build_model_setting_store

    settings = load_settings()
    store = build_model_setting_store(settings)
    result = store.set(slot, provider_key=provider_key, model=model, actor=actor or "cli")
    typer.secho(
        f"{result['slot']} → provider_key={provider_key or '-'} model={model or '-'}"
        f"（版本 {result['version']}，已记审计）",
        fg=typer.colors.GREEN,
    )
    typer.secho("重启 `rag serve` 后生效（这一版不做热切）。", fg=typer.colors.YELLOW)


registry_app = typer.Typer(
    help="文档注册表（每篇一行：走到哪一步 + 入库信息）。**2026-10-09 起落数据库**（PG / SQLite）",
    no_args_is_help=True,
)
app.add_typer(registry_app, name="registry")


@registry_app.command("init")
def registry_init() -> None:
    """建 `document_registry` 表（幂等）。**写操作**，所以显式触发 —— 与
    `rag models init` / `rag logs init` / `rag chats init` 同一个规矩。"""
    from rag.persistence.documents import build_document_store

    settings = load_settings()
    store = build_document_store(settings)
    store.setup()
    typer.secho(
        f"文档注册表已就绪：{store.name}（backend={settings.documents.backend}）",
        fg=typer.colors.GREEN,
    )


@registry_app.command("migrate")
def registry_migrate(
    from_milvus: bool = typer.Option(
        False, "--from-milvus", help="再按旧 Milvus 注册表把「真的入库了」标出来（要 Milvus 活着）"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="只报数，一个字都不写"),
) -> None:
    """把**已经存在的**文档搬进新表（这一步不重灌、不调模型、不花钱）。

    为什么需要它：注册表从 Milvus 搬到数据库之后，老数据的"这份走到哪一步了"没有地方记 ——
    不搬的话页面会把明明做过解析/切分的文档显示成"什么都没做"，逼人重来一遍。

    两路来源，**都不依赖向量库**（除非显式加 `--from-milvus`）：

    * `data/uploads/` —— 每个上传件一行，状态至少是 `uploaded`；
    * `data/stages/<doc_id>/` —— 有 `document.md` → `parsed`、有 `outline.tsv` → `reviewed`、
      有 `chunks.json` → `chunked`（块数 / 文件哈希 / 切法指纹都从这份产物抄）。

    **`indexed` 只能问 Milvus**：产物齐不代表灌进去了。所以 `--from-stages` 走到
    `chunked` 为止（看起来像"没入库"，重跑 `/index` 是安全的幂等操作）；等向量库好了，
    再跑一次 `--from-milvus` 把真正入库的那些补标成 `indexed`。

    命令**幂等，且只前进不后退**：已经是 `indexed` 的行不会被 `chunked` 覆盖。
    """
    from rag.api.uploads import build_upload_store
    from rag.contracts import DocRecord, stage_rank
    from rag.ingest.stages import build_artifact_store
    from rag.persistence.documents import build_document_store

    settings = load_settings()
    store = build_document_store(settings)
    store.setup()
    current = store.sync_stages()

    # 1) 上传件：至少 `uploaded`（`sources` 存落盘名，注册表里 `source_uri` 记的就是它）
    planned: dict[str, str] = {}
    sources: dict[str, str] = {}
    for item in build_upload_store(settings).list():
        planned[item.doc_id] = "uploaded"
        sources[item.doc_id] = item.name

    # 2) 产物：按"最高走到哪一步"升级（**不读 Milvus**）
    artifacts = build_artifact_store(settings)
    payloads: dict[str, dict] = {}
    root = Path(settings.ingest.stages.dir)
    if root.is_dir():
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            doc_id = entry.name
            stage = planned.get(doc_id, "uploaded")
            if (entry / "document.md").is_file() or (entry / "document.json").is_file():
                stage = "parsed"
            if (entry / "outline.tsv").is_file():
                stage = "reviewed"
            payload = artifacts.read_payload(doc_id, "chunks.json")
            if payload is not None:
                stage = "chunked"
                payloads[doc_id] = payload
            planned[doc_id] = stage

    # 3) 可选：旧 Milvus 注册表 → 真正入库的那些（**只有向量库活着才拿得到**）
    records: dict[str, DocRecord] = {}
    if from_milvus:
        from rag.indexing.milvus.registry import DocRegistry

        try:
            records = asyncio.run(DocRegistry(settings.indexing.vector).all())
        except Exception as exc:  # noqa: BLE001 —— 库够不着就退回「只按产物搬」
            typer.secho(
                f"读不到旧 Milvus 注册表（{type(exc).__name__}: {str(exc)[:160]}）；"
                f"这次只按产物搬，入库状态等向量库好了再补。",
                fg=typer.colors.YELLOW,
            )
        for doc_id, record in records.items():
            planned[doc_id] = "indexed"
            sources.setdefault(doc_id, record.source_uri)

    counts: dict[str, int] = {}
    advanced = 0
    for doc_id, stage in sorted(planned.items()):
        # **只前进不后退**：已经是 indexed 的行不该被产物那一趟拽回 chunked
        if stage_rank(stage) <= stage_rank(current.get(doc_id, "")):
            continue
        counts[stage] = counts.get(stage, 0) + 1
        advanced += 1
        if dry_run:
            continue
        store.sync_set_stage(doc_id, stage, sources.get(doc_id, ""))
        record = records.get(doc_id)
        if record is None and doc_id in payloads:
            payload = payloads[doc_id]
            record = DocRecord(
                doc_id=doc_id,
                source_uri=str(payload.get("source_uri") or sources.get(doc_id, "")),
                content_hash=str(payload.get("file_hash") or ""),
                chunk_count=len(payload.get("chunks") or []),
                ingested_at=int(payload.get("saved_at") or 0),
                # 注册表那一列 = 切法指纹 + 这一份的清单指纹（见 commit.index_fingerprint）
                pipeline_version=f"{payload.get('pipeline_version') or ''}"
                f"+{payload.get('outline_hash') or ''}",
            )
        if record is not None:
            # `stage=""`：状态上面已经写好了，这里只补数字，别把状态再写一遍
            store.sync_upsert([DocRecord(**{**record.model_dump(), "stage": ""})])

    total = "、".join(f"{name} {count}" for name, count in sorted(counts.items()) if count)
    head = "--dry-run（什么都没写）：" if dry_run else "已写入："
    typer.secho(f"{head}{advanced} 份（{total or '无'}）", fg=typer.colors.GREEN)
    typer.echo(f"注册表：{store.name}（backend={settings.documents.backend}）")
    if not from_milvus:
        typer.secho(
            "入库状态没搬（那只能问 Milvus）。等向量库好了再跑 "
            "`rag registry migrate --from-milvus`，把真正入库的补标成 indexed。",
            fg=typer.colors.YELLOW,
        )


storage_app = typer.Typer(
    help="文件落点：本地盘 / MinIO 桶（多 worker 用桶，见 storage.py）", no_args_is_help=True
)
app.add_typer(storage_app, name="storage")


@storage_app.command("migrate")
def storage_migrate(
    dry_run: bool = typer.Option(False, "--dry-run", help="只报数，一个字都不写"),
    only: str = typer.Option(
        "uploads,stages,assets,media", "--only", help="只搬这几个域（逗号分隔）"
    ),
) -> None:
    """把**本地盘上已有的**文件推上 MinIO 桶（幂等）。改动的是桶，本地文件不删。

    什么时候用：把统一开关 `storage.backend`（或某一域的 `backend`）从 `local` 改成
    `minio` **之后**。不改的话新上传的会进桶，而老文件还在本地 —— 页面就少看见它们
    （台账已经独立在库里，不会丢）。只搬**解析后**为 minio 的那几个域；`--dry-run`
    先看会搬多少。

    键与两个后端保持一致（`<域前缀>/<相对路径>`），所以搬完读到的还是同一批 `doc_id`。
    """
    from rag.api.uploads import INDEX_NAME
    from rag.storage import LocalBlobStore, build_blob_store, resolve_storage_backend

    settings = load_settings()
    wanted = {item.strip() for item in only.split(",") if item.strip()}
    plan = (
        (
            "uploads",
            settings.ingest.uploads.backend,
            settings.ingest.uploads.dir,
            settings.ingest.uploads.prefix,
        ),
        (
            "stages",
            settings.ingest.stages.backend,
            settings.ingest.stages.dir,
            settings.ingest.stages.prefix,
        ),
        (
            "assets",
            settings.ingest.assets.backend,
            settings.ingest.assets.dir,
            settings.ingest.assets.prefix,
        ),
        ("media", settings.api.media_backend, settings.api.media_dir, settings.api.media_prefix),
    )
    total = 0
    for name, backend, directory, prefix in plan:
        if name not in wanted:
            continue
        # 四个域默认写 auto = 听统一开关 `storage.backend`，所以这里得先解析再判
        resolved = resolve_storage_backend(settings, backend)
        if resolved != "minio":
            typer.secho(
                f"{name}：backend={resolved}，跳过（只搬配成 minio 的域）", fg=typer.colors.YELLOW
            )
            continue
        local = LocalBlobStore(directory)
        # 那份老台账（`_index.json`）与半截文件（`.part`）不是内容，别搬进桶
        all_keys = local.keys()
        keys = [
            key for key in all_keys if not key.endswith(".part") and Path(key).name != INDEX_NAME
        ]
        total += len(keys)
        if dry_run:
            typer.echo(f"{name}：将搬 {len(keys)} 个文件 → {prefix}/")
            continue
        target = build_blob_store(backend="minio", settings=settings, dir=directory, prefix=prefix)
        for key in keys:
            target.put(key, local.get(key))
        typer.secho(f"{name}：已搬 {len(keys)} 个 → {target.name}", fg=typer.colors.GREEN)
    head = "--dry-run（什么都没写）" if dry_run else "搬完"
    typer.secho(f"{head}：共 {total} 个文件", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
