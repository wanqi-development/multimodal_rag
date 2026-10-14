# 黄金集与评测报告

```
eval/
  golden.jsonl   # 黄金集：一行一个问题 + 判据（人维护，进 git）
  reports/       # 每次跑的报告（.gitignore 已忽略，不进 git）
```

跑法：

```bash
rag eval                                # 只跑检索那层：本地 embedding，不花钱
rag eval --judge                        # 再加生成侧判定：每个问题一次模型调用（花钱）
rag eval --label no-rerank --no-rerank  # 做 A/B：报告名前缀改掉，对比两份 JSON
```

## 一条记录怎么写

```json
{"id": "flink-arch-sources",
 "question": "Flink 的架构图里画了哪些数据源？",
 "expected_chunk_ids": ["3481c48bcb813900:0032"],
 "answer_points": ["交易数据", "日志数据", "IOT 设备数据", "用户点击数据"],
 "tags": ["image", "flink"],
 "notes": "1.1.1 那张架构图的图片描述块"}
```

| 字段 | 作用 | 代价 |
|---|---|---|
| `id` | 唯一，报告里按它对齐（改问题别改 id，不然历史没法比） | — |
| `question` | 问题原话 | — |
| `expected_chunk_ids` | **最精确**的判据：命中这些块才算相关 | 免费 |
| `expected_doc_ids` + `expected_headings` | 宽一档：文档对上、章节路径含任一片段 | 免费 |
| `expected_doc_ids`（只给 doc） | 最宽：这份文档的任一块都算 | 免费 |
| `answer_points` | 答案该覆盖到的要点 | **要 `--judge`** |
| `must_refuse` | 语料外的问题：答案该明说「资料里没有」 | **要 `--judge`** |
| `tags` | 分组用（报告里按它聚合） | — |
| `notes` | 给自己看的备注 | — |

规矩：**至少要有一条判据**；`must_refuse` 的条目**不能再写** `expected_*`（语料外的
问题本来就该查不到，写期望命中是自相矛盾）。这两条 `dataset.py` 会当场拦下来。

`expected_headings` 用**子串**匹配章节路径（`1.1.1`、`Transaction` 这样写就行）——
章节路径是 `A -> B` 拼的，写全了又长，还会被 OCR 的空格坑（同一份文档里
`1.1.1 Flink的定义` 和 `1.1.1Flink的定义` 都出现过）。

## 指标怎么读

检索那层（每次都算）：

| 指标 | 回答的问题 |
|---|---|
| `recall@k` | 材料**找得到**吗（前 k 条里有没有） |
| `mrr` | 材料**好找**吗（第一条相关的排多前） |
| `ndcg@k` | 排序质量（相关材料排第 3 还是第 7） |
| `top1` | 第一条就命中的比例 —— 最直观的那个 |

生成那层（`--judge` 才有）：

| 指标 | 口径 |
|---|---|
| 引用准确率 | 答案里每个 `[n]`，它指的那段原文是否**真的**支持该编号所在的那句话（原文没说、只沾边、答案扩展过头都算不支持） |
| 要点覆盖率 | `answer_points` 里被讲到的比例 |
| 漏拒 / 误拒 | 该拒没拒 / 不该拒却拒了 —— 分开数，因为修法完全相反 |
| 判不了 | 模型输出解析不出来。**不算通过**，单列出来 |

## 换语料之后

`expected_chunk_ids` 是**本机核对过的落点**。换了 embedding、重灌、或者重新解析过
PDF，chunk_id 会变（doc_id 是内容指纹，也可能变），这时报告里会出现一片「没命中」——
先把黄金集里的 `expected_*` 重新核对一遍，再谈指标涨跌。

想确认某个问题现在落在哪一块，用：

```bash
rag search "问题原话" -k 5 --stats
```
