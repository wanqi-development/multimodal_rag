"""验证 Milvus 是否支持 dense + sparse 混合检索（项目的核心架构假设）。

建临时 collection __m0_verify -> 写入假向量 -> 分别测 dense/sparse/hybrid -> 删除。
v3：小词表让 sparse 真的能撞上 term；主键按字段名 chunk_id 取。
"""

import os
import random
import sys

from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker

COLLECTION = "__m0_verify"
DIM = 1024
N_DOCS = 20
VOCAB = 200  # 小词表：让 query 的 term 大概率出现在某些文档里
STRONG = {"consistency_level": "Strong"}

uri = os.environ.get("MILVUS_URI", "http://localhost:19530")
client = MilvusClient(uri=uri, timeout=30)
print(f"uri={uri}  server={client.get_server_version()}")

if client.has_collection(COLLECTION):
    client.drop_collection(COLLECTION)
    print(f"已删除残留的 {COLLECTION}")

# ---- 1. schema ----
schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
schema.add_field("dense", DataType.FLOAT_VECTOR, dim=DIM)
schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
schema.add_field("text", DataType.VARCHAR, max_length=2048)
print("1. schema 构建 OK（dense FLOAT_VECTOR + sparse SPARSE_FLOAT_VECTOR）")

# ---- 2. 索引 ----
index_params = client.prepare_index_params()
index_params.add_index("dense", index_type="IVF_FLAT", metric_type="COSINE")
index_params.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
print("2. index_params OK（dense=IVF_FLAT/COSINE, sparse=SPARSE_INVERTED_INDEX/IP）")

# ---- 3. 建 collection ----
client.create_collection(COLLECTION, schema=schema, index_params=index_params)
client.load_collection(COLLECTION)
print("3. create_collection + load_collection OK")

# ---- 4. 写入 + flush ----
rng = random.Random(42)
vocab = list(range(1, VOCAB + 1))
docs = [
    {"dense": [rng.random() for _ in range(DIM)], "terms": rng.sample(vocab, 5)}
    for _ in range(N_DOCS)
]
for d in docs:
    d["sparse"] = {t: round(rng.uniform(0.3, 1.0), 4) for t in d["terms"]}

rows = [
    {
        "chunk_id": f"c{i}",
        "text": f"document {i}",
        "dense": docs[i]["dense"],
        "sparse": docs[i]["sparse"],
    }
    for i in range(N_DOCS)
]
res = client.insert(COLLECTION, data=rows)
client.flush(COLLECTION)
cnt = client.query(COLLECTION, filter="chunk_id != ''", output_fields=["count(*)"], **STRONG)[0][
    "count(*)"
]
print(f"4. insert OK -> insert_count={res.get('insert_count')}  flush 后实际可查={cnt} 条")

# 查询：sparse 用 c7 的原始 term（保证能撞上），dense 用随机向量
TARGET = "c7"
q_dense = [rng.random() for _ in range(DIM)]
q_sparse = dict(docs[7]["sparse"])
print(f"   query sparse 复用 {TARGET} 的 {len(q_sparse)} 个 term：{sorted(q_sparse)}")

# ---- 5. 三种检索 ----
dense_hits = client.search(
    COLLECTION,
    data=[q_dense],
    anns_field="dense",
    search_params={"metric_type": "COSINE"},
    limit=5,
    output_fields=["text"],
    **STRONG,
)[0]
sparse_hits = client.search(
    COLLECTION,
    data=[q_sparse],
    anns_field="sparse",
    search_params={"metric_type": "IP"},
    limit=5,
    output_fields=["text"],
    **STRONG,
)[0]
hybrid_hits = client.hybrid_search(
    COLLECTION,
    [
        AnnSearchRequest(
            data=[q_dense], anns_field="dense", param={"metric_type": "COSINE"}, limit=5
        ),
        AnnSearchRequest(
            data=[q_sparse], anns_field="sparse", param={"metric_type": "IP"}, limit=5
        ),
    ],
    ranker=RRFRanker(60),
    limit=5,
    output_fields=["text"],
    **STRONG,
)[0]

print("5. 检索：")
for label, hits in (
    ("dense only", dense_hits),
    ("sparse only", sparse_hits),
    ("hybrid (RRFRanker)", hybrid_hits),
):
    ids = [h["chunk_id"] for h in hits]
    print(f"   {label:<22} -> {len(hits):>2} 条  ids={ids}")

# ---- 6. 确认 hybrid 真的融合了两路 ----
wide = {"output_fields": ["text"], "limit": 20, **STRONG}
dense20 = {
    h["chunk_id"]
    for h in client.search(
        COLLECTION,
        data=[q_dense],
        anns_field="dense",
        search_params={"metric_type": "COSINE"},
        **wide,
    )[0]
}
sparse20 = {
    h["chunk_id"]
    for h in client.search(
        COLLECTION,
        data=[q_sparse],
        anns_field="sparse",
        search_params={"metric_type": "IP"},
        **wide,
    )[0]
}
hybrid5 = {h["chunk_id"] for h in hybrid_hits}
from_dense = len(hybrid5 & dense20)
from_sparse = len(hybrid5 & sparse20)
top1 = sparse_hits[0]["chunk_id"]
print(f"6. dense top20={len(dense20)}  sparse top20={len(sparse20)}  hybrid top5={len(hybrid5)}")
print(f"   hybrid 命中：来自 dense 路 {from_dense} 条，来自 sparse 路 {from_sparse} 条")
verdict = "符合预期" if top1 == TARGET else "不符预期"
print(f"   sparse 路 rank#1 = {top1}（{verdict}，目标 {TARGET}）")

# ---- 7. 清理 ----
client.release_collection(COLLECTION)
client.drop_collection(COLLECTION)
left = list(client.list_collections())
print(f"7. 已删除 {COLLECTION}；剩余 collections: {left or '(空)'}")

ok = (
    len(dense_hits) > 0
    and len(sparse_hits) > 0
    and len(hybrid_hits) > 0
    and sparse_hits[0]["chunk_id"] == TARGET
    and len(hybrid5 & dense20) > 0
    and len(hybrid5 & sparse20) > 0
    and not left
)
print(
    "\n>>> 结论："
    + (
        "Milvus 支持 dense + sparse + hybrid_search + RRFRanker 服务端融合 ✅"
        if ok
        else "验证未通过 ❌ 见上方各步输出"
    )
)
sys.exit(0 if ok else 1)
