// 答案的 markdown 渲染 —— **零依赖**，只认模型真会写出来的那几种（标题 / 列表 / 表格 /
// 粗体 / 行内代码 / 代码块 / 引用 / 分隔线）。
//
// 为什么不用 `marked`：那要联网装依赖（本项目前端刻意不引 UI 库），而收益只是多几种
// 边角语法。真要换成 marked + DOMPurify，只要替换这一个 `renderMarkdown` 就够了。
//
// **安全**：先 `escapeHtml` 再拼标签 —— 输入里的 `<script>` 之类只会变成文字，
// 进来的是模型输出（不可信），这一步不能省。
//
// 用法上有条纪律（见 ChatView）：**流式期间不要调它**，等这一轮跑完再渲染 ——
// 每个 token 都重解析一遍 markdown 会卡界面。

const ESCAPES: Record<string, string> = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
}

export function escapeHtml(text: string): string {
  return text.replace(/[&<>"]/g, (ch) => ESCAPES[ch])
}

/**
 * 当前这次渲染「哪些 `[n]` 是能点开的」（服务端只把引用编号与 chunk 的对应关系给当轮，
 * 从历史里恢复的那些没有这份映射 —— 那就别做成按钮，免得点了没反应）。
 */
let citeSet: Set<number> | null = null

/** 行内：粗体 / 斜体 / 行内代码 / 链接 / 引用编号。输入必须是**已转义**的文本。 */
function inline(text: string): string {
  return text
    .replace(/`([^`]+)`/g, (_m, code: string) => `<code>${code}</code>`)
    // 图片标记（`![image](asset:<doc_id>/img01.png)`）：解析稿里到处是它，页面上光看
    // 三十几个字符的资产路径没有意义，直接显示图。**鉴权那层 2026-10-06 拿掉了**，所以
    // 这里可以直连 `/assets/…`；哪天加回 token 校验，就要换成 api.ts 的 `assetBlob()`
    // （`<img>` 发不出请求头）。
    .replace(
      /!\[([^\]]*)\]\(asset:([^)]+)\)/g,
      (_m, alt: string, path: string) =>
        `<img class="md-img" src="/assets/${path}" alt="${alt}" loading="lazy">`,
    )
    // **远程图**（`![alt](https://…)`）：抓来的页面里会留这种 —— 典型是 SVG，
    // 它进不了库（`loaders/base.image_suffixes()` 只要栅格图，VLM / 多模态 embedding 都
    // 按位图设计），于是留在正文里当外链。这里直接让浏览器去取：**能看见**，
    // 而库里仍然没有它（检索不受影响）。`no-referrer`：别把我们的地址当 Referer 送出去。
    // **必须排在下面那条链接规则前面** —— 否则 `![alt](url)` 会被链接规则吃掉，
    // 前面留一个孤零零的 `!`（实测就是这个怪样子）。
    .replace(
      /!\[([^\]]*)\]\((https?:\/\/[^)\s]+)\)/g,
      (_m, alt: string, url: string) =>
        `<img class="md-img" src="${url}" alt="${alt}" loading="lazy" referrerpolicy="no-referrer">`,
    )
    // 剩下的图片引用：**本地相对路径**（抓页时那张图没下下来之类）—— 显示不了，
    // 只留 alt 文字，别把 `![图](path)` 原样露出来（页面看着就是 `!图` 这种残迹）。
    // 缺图的原因在解析面板的缺失清单里说得更清楚（`parse.missing_images`）。
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, (_m, alt: string) => alt)
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>')
    .replace(
      /\[([^\]]+)\]\(([^()\s]+)\)/g,
      (_m, label: string, url: string) =>
        /^https?:\/\//.test(url)
          ? `<a href="${url}" target="_blank" rel="noreferrer">${label}</a>`
          : label,
    )
    .replace(/\[(\d{1,3})\]/g, (match: string, index: string) =>
      citeSet?.has(Number(index))
        ? `<button type="button" class="cite" data-cite="${index}" title="看原文">[${index}]</button>`
        : match,
    )
}

/** 一个段落里单个换行按硬换行处理（模型写清单时经常只换行不空行）。 */
function paragraph(lines: string[]): string {
  const body = lines.map((line) => inline(escapeHtml(line.trim()))).join('<br>')
  return `<p>${body}</p>`
}

function splitRow(line: string): string[] {
  const trimmed = line.trim().replace(/^\|/, '').replace(/\|$/, '')
  return trimmed.split('|').map((cell) => cell.trim())
}

function isDivider(line: string): boolean {
  return /^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$/.test(line.trim())
}

/**
 * 产线（PDF 走的 OCR）给的表格是**内联 HTML**，不是 markdown 竖线 —— 这是 `cleaning.py`
 * 里有意保留的：产线会输出合并单元格，竖线表表达不了 rowspan/colspan，HTML 才是无损的。
 * 所以渲染这一端必须认得它，否则页面上看到的就是一堆 `<table>` 标签文本。
 *
 * **白名单重建，不是「放行一段 HTML」**：这里先让浏览器把片段解析成 DOM，再自己按白名单
 * 拼回去 —— 只留表格结构标签与 rowspan/colspan，属性值也转义。来源是「用户上传的文档
 * 经 OCR 出来的一段文本」（不可信），所以不能直接把原串塞进 v-html。
 */
const TABLE_TAGS = new Set([
  'TABLE',
  'THEAD',
  'TBODY',
  'TFOOT',
  'TR',
  'TH',
  'TD',
  'CAPTION',
  'SUB',
  'SUP',
  'BR',
])
const TABLE_ATTRS = new Set(['rowspan', 'colspan', 'align'])

function safeTable(html: string): string {
  // 没有 DOM（比如在 Node 里跑）就退回原样转义 —— 宁可显示成文字，也不要崩
  if (typeof DOMParser === 'undefined') return `<p>${escapeHtml(html)}</p>`
  const parsed = new DOMParser().parseFromString(`<body>${html}</body>`, 'text/html')
  const out: string[] = []
  const walk = (node: Node): void => {
    if (node.nodeType === Node.TEXT_NODE) {
      out.push(escapeHtml(node.textContent ?? ''))
      return
    }
    if (node.nodeType !== Node.ELEMENT_NODE) return // 注释、doctype 之类一律丢
    const element = node as Element
    const tag = element.tagName.toUpperCase()
    if (!TABLE_TAGS.has(tag)) {
      // 不认识的标签（script/style/div…）只保留里面的文字
      for (const child of Array.from(element.childNodes)) walk(child)
      return
    }
    const name = tag.toLowerCase()
    if (name === 'br') {
      out.push('<br>')
      return
    }
    const attrs = Array.from(element.attributes)
      .filter((attr) => TABLE_ATTRS.has(attr.name.toLowerCase()))
      .map((attr) => ` ${attr.name.toLowerCase()}="${escapeHtml(attr.value)}"`)
      .join('')
    out.push(`<${name}${attrs}>`)
    for (const child of Array.from(element.childNodes)) walk(child)
    out.push(`</${name}>`)
  }
  for (const child of Array.from(parsed.body.childNodes)) walk(child)
  return out.join('')
}

/** 代码围栏：行首最多 3 个空格 + 三个以上同种字符（` 或 ~）。规则与后端
 *  `rag/ingest/headings.py` 的 `code_fence_mask()` 一致 —— 两边都认同一件事：
 *  **代码块里的 `#` 不是标题**。 */
const FENCE = /^\s{0,3}(`{3,}|~{3,})/

/**
 * 把 markdown 按行切成「标题行 / 正文段」—— 解析页与核对页用它排版。
 *
 * **标题行是纯文本渲染的**（`{{ segment.text }}`，不走 markdown），所以代码块里的
 * `# Run the agent`（Python 注释）一旦被当成标题挑出来，就会以标题的样式显示在页面上 ——
 * 而且会让"标题行数"比清单行数多（后端已经跳过围栏，两边就对不齐了）。
 * 所以这里必须**跟踪围栏**：围栏内的行一律算正文，交给下面的 `renderMarkdown` 当代码块渲染。
 */
export function markdownSegments(
  markdown: string,
): { kind: 'heading' | 'body'; index: number; text: string }[] {
  const out: { kind: 'heading' | 'body'; index: number; text: string }[] = []
  let buffer: string[] = []
  let heading = 0
  let fence: string | null = null
  const flush = (): void => {
    const text = buffer.join('\n')
    if (text.trim()) out.push({ kind: 'body', index: -1, text })
    buffer = []
  }
  for (const line of (markdown ?? '').split('\n')) {
    const marker = FENCE.exec(line)
    if (marker) {
      // 开栏/闭栏本身都算正文；闭合要求同种字符且不短于开栏那串（CommonMark）
      if (fence === null) fence = marker[1]
      else if (marker[1][0] === fence[0] && marker[1].length >= fence.length) fence = null
      buffer.push(line)
      continue
    }
    const match = fence === null ? /^(#{1,6})\s+(.*)$/.exec(line) : null
    if (match && match[2].trim()) {
      flush()
      out.push({ kind: 'heading', index: heading++, text: match[2].trim() })
      continue
    }
    buffer.push(line)
  }
  flush()
  return out
}

/** 段落级解析。逐行走，遇到能识别的块就吃掉它，剩下的攒成段落。 */
function blocks(markdown: string): string[] {
  const lines = markdown.replace(/\r\n?/g, '\n').split('\n')
  const out: string[] = []
  let i = 0
  while (i < lines.length) {
    const line = lines[i]
    if (!line.trim()) {
      i += 1
      continue
    }
    // 代码块：``` 或 ~~~，整段原样（只转义）
    const fence = line.match(/^\s*(```|~~~)(.*)$/)
    if (fence) {
      const body: string[] = []
      i += 1
      while (i < lines.length && !lines[i].trim().startsWith(fence[1])) {
        body.push(lines[i])
        i += 1
      }
      i += 1 // 收尾那行
      const lang = fence[2].trim()
      const cls = lang ? ` class="language-${escapeHtml(lang)}"` : ''
      out.push(`<pre><code${cls}>${escapeHtml(body.join('\n'))}</code></pre>`)
      continue
    }
    // OCR 的内联 HTML 表格：整块交给白名单重建（判据与后端 `cleaning.py` 的产物一致）
    if (/^\s*<table\b/i.test(line)) {
      const chunk: string[] = []
      while (i < lines.length) {
        chunk.push(lines[i])
        const closed = /<\/table\s*>/i.test(lines[i])
        i += 1
        if (closed) break
      }
      out.push(safeTable(chunk.join('\n')))
      continue
    }
    const heading = line.match(/^(#{1,6})\s+(.*)$/)
    if (heading) {
      const level = Math.min(heading[1].length, 4)
      out.push(`<h${level}>${inline(escapeHtml(heading[2].trim()))}</h${level}>`)
      i += 1
      continue
    }
    if (/^\s*([-*_])\s*\1\s*\1[\s\-*_]*$/.test(line)) {
      out.push('<hr>')
      i += 1
      continue
    }
    // 表格：这一行是 | … |，下一行是分隔行
    if (line.trim().startsWith('|') && i + 1 < lines.length && isDivider(lines[i + 1])) {
      const head = splitRow(line)
      const rows: string[][] = []
      i += 2
      while (i < lines.length && lines[i].trim().startsWith('|')) {
        rows.push(splitRow(lines[i]))
        i += 1
      }
      const thead = head.map((cell) => `<th>${inline(escapeHtml(cell))}</th>`).join('')
      const tbody = rows
        .map(
          (row) =>
            `<tr>${row.map((cell) => `<td>${inline(escapeHtml(cell))}</td>`).join('')}</tr>`,
        )
        .join('')
      out.push(`<table><thead><tr>${thead}</tr></thead><tbody>${tbody}</tbody></table>`)
      continue
    }
    if (/^\s*>\s?/.test(line)) {
      const body: string[] = []
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
        body.push(lines[i].replace(/^\s*>\s?/, ''))
        i += 1
      }
      out.push(`<blockquote>${paragraph(body)}</blockquote>`)
      continue
    }
    const bullet = line.match(/^\s*[-*+]\s+(.*)$/)
    if (bullet) {
      const items: string[] = []
      while (i < lines.length) {
        const item = lines[i].match(/^\s*[-*+]\s+(.*)$/)
        if (!item) break
        items.push(`<li>${inline(escapeHtml(item[1].trim()))}</li>`)
        i += 1
      }
      out.push(`<ul>${items.join('')}</ul>`)
      continue
    }
    const ordered = line.match(/^\s*(\d+)[.)]\s+(.*)$/)
    if (ordered) {
      // **首项的编号要留着**：列表被正文段落打断时（`1. 甲` / 正文 / `2. 乙`）每一段都是
      // 独立的 `<ol>`，而浏览器会各自从 1 数起 —— 实测文档里 `1./2./3.` 显示成了 `1./1./1.`。
      // 带上 `start` 就按文档里写的编号显示；而连续写 `1. / 1. / 1.`（markdown 的惯用写法）
      // 首项编号是 1，不输出 `start`，浏览器照旧数成 1./2./3. —— 两种写法都保住。
      const start = Number(ordered[1])
      const items: string[] = []
      while (i < lines.length) {
        const item = lines[i].match(/^\s*\d+[.)]\s+(.*)$/)
        if (!item) break
        items.push(`<li>${inline(escapeHtml(item[1].trim()))}</li>`)
        i += 1
      }
      out.push(`<ol${start > 1 ? ` start="${start}"` : ''}>${items.join('')}</ol>`)
      continue
    }
    // 普通段落：攒到空行或下一个块级标记为止
    const body: string[] = []
    while (i < lines.length && lines[i].trim()) {
      const next = lines[i]
      if (
        body.length &&
        (/^\s*(```|~~~)/.test(next) ||
          /^#{1,6}\s/.test(next) ||
          /^\s*[-*+]\s+/.test(next) ||
          /^\s*\d+[.)]\s+/.test(next) ||
          /^\s*>\s?/.test(next) ||
          next.trim().startsWith('|'))
      ) {
        break
      }
      body.push(next)
      i += 1
    }
    out.push(paragraph(body))
  }
  return out
}

/** 模型写的 markdown → 安全的 HTML 字符串。 */
export function renderMarkdown(markdown: string, cites?: Iterable<number>): string {
  // **不要整体转义**：块级语法（#、-、|、```）要在原文上判断，
  // 每一处叶子（标题、单元格、列表项、段落、代码块）各自转义一次。
  return withCites(cites, () => blocks((markdown ?? '').trim()).join(''))
}

/**
 * 只有引用编号的一小块文本 → HTML（答案末尾那份「引用来源」清单用它）。
 *
 * 和 `renderMarkdown` 分开：清单是一行一条、靠 `pre` 保留换行，不该被当成 markdown 段落。
 */
export function citeText(text: string, cites?: Iterable<number>): string {
  return withCites(cites, () => inline(escapeHtml(text ?? '')))
}

/** 渲染期间临时记住「哪些引用能点」—— 同步渲染，用完还回去。 */
function withCites<T>(cites: Iterable<number> | undefined, run: () => T): T {
  const previous = citeSet
  citeSet = cites ? new Set(cites) : null
  try {
    return run()
  } finally {
    citeSet = previous
  }
}
