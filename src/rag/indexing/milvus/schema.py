"""Milvus collection schema：显式字段 + 幂等建索引。

字段全部显式声明（``enable_dynamic_field=False``）：字段类型与长度上限决定了
能写入什么、能怎么过滤。靠动态字段兜住一切，问题会推迟到灌数据那一刻才炸，
而且症状是「写入失败」而不是「schema 缺了字段」。
"""

from __future__ import annotations

from pymilvus import CollectionSchema, DataType, Function, FunctionType, MilvusClient

from rag.config.schema import VectorIndexConfig

# Milvus 的 VARCHAR max_length 按 **UTF-8 字节** 计，不是字符数（实测：9000 个汉字
# 报 "length: 27000, max length: 8192"）。一个汉字 3 字节，所以 8192 字节 ≈ 2700 汉字，
# 对默认 chunk.size=400 有 6 倍余量。真要调大 chunk.size，得一起抬这个上限。
MAX_TEXT_BYTES = 8192
MAX_URI_BYTES = 1024
MAX_ID_BYTES = 64
# 标题路径：`第一章、概念 -> 1、什么是Agent？`，按 UTF-8 字节算（一个汉字 3 字节）
MAX_HEADING_BYTES = 1024
# 短枚举：doc_type / modality / content_hash
MAX_TAG_BYTES = 16
# 一个块最多引用多少张图。块 ≤1000 字符、一个图片标记约 40 字符，理论上限约 25
MAX_ASSETS = 64

BM25_FUNCTION_NAME = "bm25_text"


def build_schema(cfg: VectorIndexConfig, dim: int) -> CollectionSchema:
    """按配置建表结构。

    dense 维度由调用方传入（来自 providers.embedding.dim，那是唯一来源）；
    sparse 字段不给维度也不由我们写入 —— 它由 BM25 function 在服务端生成。
    """
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)

    schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=MAX_ID_BYTES)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=MAX_ID_BYTES)
    schema.add_field("chunk_index", DataType.INT64)
    schema.add_field("source_uri", DataType.VARCHAR, max_length=MAX_URI_BYTES)
    # —— 下面这些是「要过滤的」，全部显式声明，不塞 JSON ——
    # 来源类型与模态：按格式筛（只看 pdf）、按模态筛（只要正文、不要图片描述）
    schema.add_field("doc_type", DataType.VARCHAR, max_length=MAX_TAG_BYTES)
    schema.add_field("modality", DataType.VARCHAR, max_length=MAX_TAG_BYTES)
    # 内容归属的标题：路径给人看/给大模型看，编号给过滤用（位置派生的 1-2-3）
    schema.add_field("heading_path", DataType.VARCHAR, max_length=MAX_HEADING_BYTES)
    schema.add_field("heading_numbers", DataType.VARCHAR, max_length=MAX_ID_BYTES)
    # 位置。page=0 表示该格式没有页的概念（docx/html/xlsx）
    schema.add_field("page", DataType.INT64)
    schema.add_field("line", DataType.INT64)
    # 本块引用到的资产 uri（图片路径）。不落这个字段的话，图片只能靠扫正文找
    schema.add_field(
        "assets",
        DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_capacity=MAX_ASSETS,
        max_length=MAX_URI_BYTES,
    )
    # sha1(text)[:16]，去重用
    schema.add_field("content_hash", DataType.VARCHAR, max_length=MAX_TAG_BYTES)
    # 剩下的块级附加信息（chunk_strategy、page_count 之类）
    schema.add_field("metadata", DataType.JSON)

    schema.add_field(
        cfg.bm25.input_field,
        DataType.VARCHAR,
        max_length=MAX_TEXT_BYTES,
        # 不开 analyzer 就会退回默认英文分词器，中文 BM25 基本失效（整段当一个 term）
        enable_analyzer=True,
        analyzer_params={"type": cfg.bm25.analyzer},
    )
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(cfg.bm25.output_field, DataType.SPARSE_FLOAT_VECTOR)

    if cfg.bm25.enabled:
        schema.add_function(
            Function(
                name=BM25_FUNCTION_NAME,
                function_type=FunctionType.BM25,
                input_field_names=[cfg.bm25.input_field],
                output_field_names=[cfg.bm25.output_field],
            )
        )
    return schema


def build_index_params(client: MilvusClient, cfg: VectorIndexConfig) -> object:
    """按字段分别给索引参数。

    不能一份参数套所有字段：dense 要 FLOAT_VECTOR 系的 IVF_FLAT/COSINE，
    sparse 要 SPARSE_INVERTED_INDEX，且内建 BM25 要求 metric 为 BM25。
    """
    params = client.prepare_index_params()
    for field_name, field_cfg in cfg.indexes.items():
        params.add_index(field_name, index_type=field_cfg.type, metric_type=field_cfg.metric)
    return params


def registry_collection(cfg: VectorIndexConfig) -> str:
    """注册表名字由块表名派生：两者必须成对，派生就不会走散。"""
    return f"{cfg.collection}_docs"


def build_registry_schema() -> CollectionSchema:
    """文档注册表：**每份文档一行**，不是每块一行。

    增量更新靠它：块表是 O(块数) 的，做 diff 时不该去扫它；注册表是 O(文档数) 的，
    读一遍就能算出「新增 / 变更 / 跳过 / 删除」。

    ⚠️ **必须带一个向量字段**：Milvus 拒绝只有标量字段的 collection
    （实测 `code=1100 schema does not contain vector field`）；而且维度**不能小于 2**
    （实测 `code=1100 invalid dimension: 1. should be in range 2 ~ 32768`）。
    所以放一个 2 维的 ``placeholder`` —— 它不参与任何检索，纯粹为了满足建表要求，
    代价是每行 8 字节。
    """
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("doc_id", DataType.VARCHAR, is_primary=True, max_length=MAX_ID_BYTES)
    schema.add_field("source_uri", DataType.VARCHAR, max_length=MAX_URI_BYTES)
    # 文件字节的 sha1[:16]
    schema.add_field("content_hash", DataType.VARCHAR, max_length=MAX_TAG_BYTES)
    schema.add_field("chunk_count", DataType.INT64)
    schema.add_field("ingested_at", DataType.INT64)
    schema.add_field("pipeline_version", DataType.VARCHAR, max_length=MAX_ID_BYTES)
    # 占位向量，见上面的说明。写入时固定 [0.0, 0.0]
    schema.add_field("placeholder", DataType.FLOAT_VECTOR, dim=2)
    return schema


def build_registry_index_params(client: MilvusClient) -> object:
    """占位向量的索引。注册表不被检索，用 FLAT（精确、无参）最省事。"""
    params = client.prepare_index_params()
    params.add_index("placeholder", index_type="FLAT", metric_type="COSINE")
    return params
