# 工程实测结果

**当前仓库不维护固定黄金集，也不发布基于合成问答集的检索准确率。**

历史 `eval/golden.jsonl` 的题目、证据和答案要点均由 LLM 自动生成，没有经过人工标注；
它无法代表真实用户问题分布，因此相关绝对指标已从 README 和架构文档撤回。

`src/rag/eval/` 中保留通用指标、引用判定和 ragas 适配器，供以后导入真实检索结果快照
或经过审核的数据集时使用。

---

## 1. 工程侧实测

这些数字用于解释当前实现、性能和排障取舍，不是检索质量基准。

| 事项 | 实测数字 | 出处 |
|---|---|---|
| PDF 解析选型 | 同一份数字版 PDF：pymupdf 带标题 0/20，OCR 产线 36/36 → PDF 默认走 OCR | ADR 23 |
| 精排耗时 × 候选池 | 池子 20 → 重排 26 条 6.2–7.7s；池子 10 → 重排 14 条 4.9s | §9.3 |
| flashrank vs bge-reranker | flashrank 在中文语料上分数饱和到 0.9995，排序退化；默认使用 bge-reranker | §9.4 |
| 本机 embedding 吞吐 | 满 1000 字的块约 0.32 块/秒；真实语料 243 块约 4 分钟 | §16 |
| OCR 产线转发质量 | 真实负载约 28% 失败 → 配置允许时切云端兜底 | §7.1 / §16 |
| Milvus 不可达 | 加抓取探活和超时后，连接被拒约 2.0 秒返回 | §16 |
| 关系库连接成本 | 本机 PostgreSQL 每次连接约 0.23s | backlog 3.1 |
| MinIO 迁移 | `rag storage migrate` 搬运 306 个对象，耗时 7.4s | backlog #12 |
| 语义切分自研 vs `SemanticChunker` | 自研 6.5s，官方 16.3s，且官方缺少尺寸护栏 | ADR 22 |
| 双版本索引探针 | legacy 245 块；临时版本化批次可完成活动 filter 与 hybrid 查询 | CHANGELOG 2026-10-23 |
| Milvus Lite 兼容 | 3.2.2 支持 jieba、BM25、dual-version collection 和 hybrid search | §14.0 |

## 2. agent 行为记录

真机问「Flink 和 Spark 在定位上有什么不同」：

```text
list_documents → search_keywords → search_knowledge → get_section
11 步 · 引用 12 条 · 未查询图谱通道
```

工具面收敛到少量高能力工具后，模型选择了直接查知识库和章节内容。

## 3. 边界

- 没有人工标注的独立检索测试集；
- 没有与公开 benchmark 做横向对比；
- 不应引用历史合成集数字宣称系统绝对准确率；
- 后续评测应优先导入真实用户问题或经过审核、与当前检索器隔离的数据。
