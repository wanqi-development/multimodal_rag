"""增量更新：指纹、三态 diff、以及第四态（文件被删除）。

全是纯逻辑 —— 不连 Milvus、不解析文件、不调模型。
"""

from __future__ import annotations

from pathlib import Path

from rag.config.schema import (
    ChunkingConfig,
    IndexingConfig,
    IngestConfig,
    IngestImagesConfig,
    Settings,
)
from rag.contracts import DocRecord
from rag.ingest.incremental import file_hash, pipeline_version, plan_incremental
from rag.ingest.loaders.base import doc_id_for


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _settings(**over: object) -> Settings:
    base = {"strategy": "markdown-semantic", "size": 1000, "overlap": 60}
    chunk = ChunkingConfig(**{**base, **over})  # type: ignore[arg-type]
    return Settings(indexing=IndexingConfig(chunk=chunk))


def _record(path: Path, root: Path, *, pipeline: str, chunk_count: int = 1) -> DocRecord:
    return DocRecord(
        doc_id=doc_id_for(path, root),
        source_uri=path.relative_to(root).as_posix(),
        content_hash=file_hash(path),
        chunk_count=chunk_count,
        ingested_at=1,
        pipeline_version=pipeline,
    )


# ---- 指纹 ----


def test_file_hash_follows_content_only(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.md", "内容")
    same = _write(tmp_path / "b.md", "内容")  # 内容相同、路径不同

    assert file_hash(a) == file_hash(same), "哈希的是内容，不是路径"
    _write(a, "改过了")
    assert file_hash(a) != file_hash(same)


def test_pipeline_version_is_stable_but_sensitive(tmp_path: Path) -> None:
    """同一套配置稳定；切法或模型一变就要变 —— 否则「内容没变但结果该变」会被跳过。"""
    assert pipeline_version(_settings()) == pipeline_version(_settings())

    for change in ({"size": 500}, {"strategy": "recursive"}, {"overlap": 0}):
        assert pipeline_version(_settings(**change)) != pipeline_version(_settings()), change

    other = _settings()
    other.providers.embedding.dim = 512
    assert pipeline_version(other) != pipeline_version(_settings())


def test_pipeline_version_follows_the_image_description_switch() -> None:
    """打开图片描述后文件一个字节都没变 —— 不把开关写进指纹，增量会把它们全跳过，
    结果一张图都没描述、库里却「看起来是新的」。"""
    off = Settings(ingest=IngestConfig(images=IngestImagesConfig(describe=False)))
    on = Settings(ingest=IngestConfig(images=IngestImagesConfig(describe=True)))

    assert pipeline_version(off) != pipeline_version(on)

    # 关着的时候不该受 VL 模型名影响：否则改一下模型名就白白重灌整个语料
    other_model = Settings(
        ingest=IngestConfig(images=IngestImagesConfig(describe=False)),
    )
    other_model.providers.vision.model = "qwen-vl-plus"
    assert pipeline_version(other_model) == pipeline_version(off)


def test_pipeline_version_ignores_separators_of_other_strategies() -> None:
    """切块指纹要**按策略**取分隔符：改了递归那组不该让 markdown-semantic 的语料重灌。

    重灌一次要重新解析、重新 OCR、重新问 VLM（按量计费），所以指纹宁可精确也不能宽。
    """
    from rag.ingest import incremental

    by_strategy = incremental.SEPARATORS_BY_STRATEGY
    assert "markdown-semantic" not in by_strategy, "它有自己的三级阶梯，不该挂在这张表上"

    original = by_strategy["recursive"]
    before = pipeline_version(_settings())  # 默认策略 markdown-semantic
    before_recursive = pipeline_version(_settings(strategy="recursive"))
    try:
        by_strategy["recursive"] = ("\n", "。", "")
        assert pipeline_version(_settings()) == before, "默认策略不碰递归那组分隔符"
        assert pipeline_version(_settings(strategy="recursive")) != before_recursive
    finally:
        by_strategy["recursive"] = original


# ---- 三态 diff ----


def test_fingerprint_can_be_computed_per_file(tmp_path: Path) -> None:
    """指纹可以按**文件**算：PDF 的标题清单是人工对齐的，改一份只该让那一份重灌。"""
    first = _write(tmp_path / "a.md", "同样的内容")
    second = _write(tmp_path / "b.md", "同样的内容")
    registry = {
        doc_id_for(first, tmp_path): _record(first, tmp_path, pipeline="v1"),
        doc_id_for(second, tmp_path): _record(second, tmp_path, pipeline="v1"),
    }
    # b 的人工清单改过了 → 它的有效指纹变成 v2；a 没变
    versions = {"a": "v1", "b": "v2"}

    plan = plan_incremental(
        [first, second],
        root=tmp_path,
        registry=registry,
        pipeline=lambda path: versions[path.stem],
    )

    assert plan.skipped == [first]
    assert plan.changed == [second]


def test_new_file_is_added(tmp_path: Path) -> None:
    target = _write(tmp_path / "a.md", "新文件")

    plan = plan_incremental([target], root=tmp_path, registry={}, pipeline="p1")

    assert plan.added == [target]
    assert plan.todo == [target]


def test_unchanged_file_is_skipped(tmp_path: Path) -> None:
    """**连解析都不做** —— 解析与 embedding 是全部代价所在，跳过才是省钱的点。"""
    target = _write(tmp_path / "a.md", "没变")
    registry = {doc_id_for(target, tmp_path): _record(target, tmp_path, pipeline="p1")}

    plan = plan_incremental([target], root=tmp_path, registry=registry, pipeline="p1")

    assert plan.skipped == [target]
    assert plan.todo == []


def test_edited_file_is_changed(tmp_path: Path) -> None:
    target = _write(tmp_path / "a.md", "改之前")
    registry = {doc_id_for(target, tmp_path): _record(target, tmp_path, pipeline="p1")}
    _write(target, "改之后")

    plan = plan_incremental([target], root=tmp_path, registry=registry, pipeline="p1")

    assert plan.changed == [target]


def test_a_new_pipeline_version_forces_a_rebuild(tmp_path: Path) -> None:
    """内容一个字没改，但切法变了 —— 库里那批块是用旧规则切的，必须重灌。"""
    target = _write(tmp_path / "a.md", "没变")
    registry = {doc_id_for(target, tmp_path): _record(target, tmp_path, pipeline="旧的")}

    plan = plan_incremental([target], root=tmp_path, registry=registry, pipeline="新的")

    assert plan.changed == [target]


# ---- 第四态：文件被删除 ----


def test_a_deleted_file_is_detected_for_a_directory_source(tmp_path: Path) -> None:
    """之前完全没处理：语料里删掉一份文件，库里的块会永远留着。"""
    gone = _write(tmp_path / "gone.md", "将被删掉")
    registry = {doc_id_for(gone, tmp_path): _record(gone, tmp_path, pipeline="p1")}
    gone.unlink()

    plan = plan_incremental(
        [], root=tmp_path, registry=registry, pipeline="p1", detect_deleted=True
    )

    assert plan.deleted == [doc_id_for(gone, tmp_path)]


def test_deletion_is_not_detected_for_a_single_file_source(tmp_path: Path) -> None:
    """单文件源看不出「别的文件没了」，所以它不该乱删。"""
    other = _write(tmp_path / "other.md", "别的文件")
    registry = {doc_id_for(other, tmp_path): _record(other, tmp_path, pipeline="p1")}
    other.unlink()

    plan = plan_incremental([], root=tmp_path, registry=registry, pipeline="p1")

    assert plan.deleted == []


def test_a_file_that_still_exists_is_not_deleted(tmp_path: Path) -> None:
    """注册表里有、但只是这次没被发现（比如后缀不支持）—— 文件还在就不该删。"""
    kept = _write(tmp_path / "kept.md", "还在")
    registry = {doc_id_for(kept, tmp_path): _record(kept, tmp_path, pipeline="p1")}

    plan = plan_incremental(
        [], root=tmp_path, registry=registry, pipeline="p1", detect_deleted=True
    )

    assert plan.deleted == []


def test_a_moved_file_reads_as_added_plus_deleted(tmp_path: Path) -> None:
    """改名 = 新路径（新增）+ 旧路径消失（删除）—— doc_id 是路径哈希，本来就该这样。"""
    old = _write(tmp_path / "old.md", "内容")
    registry = {doc_id_for(old, tmp_path): _record(old, tmp_path, pipeline="p1")}
    new = tmp_path / "new.md"
    old.rename(new)

    plan = plan_incremental(
        [new], root=tmp_path, registry=registry, pipeline="p1", detect_deleted=True
    )

    assert plan.added == [new]
    assert plan.deleted == [doc_id_for(old, tmp_path)]
