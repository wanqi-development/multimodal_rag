# rag-dots-parser

`multimodal_rag` 的可选 dots.ocr 解析包。它承载从 dots.ocr 官方仓库整包带入的
vendor 代码，避免第三方样例混进核心发行物。

安装：

```powershell
pip install -e plugins/rag-dots-parser
```

默认 PDF 链路使用 dots，因此通常需要安装。未安装时核心仍可启动；只有把
`ingest.pdf.parser` 设为 `dots` 并真正创建解析器时，才会报出上面的安装命令。

配置在核心的 `ingest.dots`：

```yaml
ingest:
  pdf:
    parser: dots
  dots:
    base_url: http://127.0.0.1:6006
    model: dots_ocr
    dpi: 200
    num_thread: 16
    max_retries: 2
    prompt_mode: prompt_layout_all_en
```

这个包通过 `rag_dots_parser.vendor.dots_ocr` 暴露官方样例代码；核心的
`src/rag/ingest/dots.py` 只保留客户端、重试、页序与 markdown 转换逻辑。
