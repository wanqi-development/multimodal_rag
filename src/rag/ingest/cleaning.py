"""产线 markdown 的入库前清洗。

产线的表格是**内联 HTML**（不是 markdown 竖线），且每个格子上都挂着完整的内联样式：

    <td style='text-align: center; word-wrap: break-word;'>模型</td>

样式占了约 80% 的字符。这不只是浪费：递归切块器会把表格从**标签中间**切开，
实测出现过整块内容只剩 ``style='text-align: center; word-wrap: break-word;'``
而一个单元格值都没有的 chunk —— 那种块检索命中的全是版式，等于表格白抽了。

这里只做**归一化：保留结构，剥掉版式**。表格仍然是 HTML（行/列关系没丢），
只是变成 ``<td>模型</td>`` 这种紧凑形态，并在 ``<table>`` / ``<tr>`` 边界补上换行。

**补换行不是排版好看，是为了切块。** 产线给的是**一整行** HTML，中间没有任何换行；
递归切块器的分隔符优先级是 ``\n\n → \n → 。！？；， → 空格``，没有换行可断时只能退到
标点和空格，于是从**标签中间**切开。在行边界补上 ``\n`` 后，超长表格会在**行之间**断开 ——
每块都还是完整的 ``<tr>…</tr>``。

不做的事（有意为之）：

- **不转成 markdown 竖线表**。产线会输出合并单元格，竖线表表达不了 rowspan/colspan，
  转过去要么丢结构要么造出对不齐的表。HTML 本身就是无损的。
- **不动正文**。正文里的 ``#`` / ``**`` / ``|`` / 字符引用（``&amp;``）原样保留，
  markdown 切块器照常工作。唯一被改的空白是补在表格行边界的换行。
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

# 不可见字符，两类：
#   1. 零宽 —— 文档站的「复制链接」锚点会在标题里塞一个 U+200B
#   2. **私有使用区（PUA）** —— 图标字体的字形码点。Confluent 的每个标题尾部都挂着一个
#      U+F0C1，那是 Font Awesome 的 fa-link 图标；它不是文字，却会跟着进标题路径与正文，
#      让「标题精确匹配」失效。实测撞到过（U+200B 在 OCR 的标题里也撞到过一次）
INVISIBLE = re.compile(
    "[\u200b\u200c\u200d\ufeff"
    "\ue000-\uf8ff"  # BMP 私用区
    "\U000f0000-\U000ffffd\U00100000-\U0010fffd"  # 两个补充私用平面
    "]"
)


def strip_invisible(text: str) -> str:
    """去掉零宽字符与图标字体的私用区码点。"""
    return INVISIBLE.sub("", text)


# **label 为空的链接**：`[](url)`，label 里可能只有空白或零宽字符。
# 这类不是内容，是文档站给标题挂的**锚点**（mintlify 那类：`## [<ZWSP>](…#step-1) 标题`）。
# 留着会一路传到标题清单、heading_path 与库里的正文 —— 页面上看着就是标题前面挂了一串
# `[](https://… )`。**必须 `(?<!\!)`**：不然 markdown 的空 alt 图片 `![](x.png)` 会被
# 当成空链接删掉（那是图，不是锚点）。
_EMPTY_LINK = re.compile(r"(?<!!)\[[\s\u200b\u200c\u200d\ufeff]*\]\([^)\s]*\)")

# 标题行：`##    标题` → `## 标题`。删掉锚点之后会剩下一堆空格，收一收。
# 只认标题行 —— 正文里的缩进/空白是有意义的（尤其代码块），一个字都不动。
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*$", re.MULTILINE)


def _tidy_heading(match: re.Match[str]) -> str:
    body = match.group(2).strip()
    return f"{match.group(1)} {body}" if body else match.group(0)


# 文档**开头**的连续引用块（`> …`，中间可以夹空行）——站点提示常这么写。
_LEADING_QUOTE = re.compile(r"\A\s*((?:[ \t]*>[^\n]*\n?)+)")

# 文档**开头**的「跳到主内容」链接。mintlify 那类写成 `[Skip to main content](url)`。
_LEADING_SKIP = re.compile(
    r"\A\s*\[[^\]]*skip to (?:main )?content[^\]]*\]\([^)]*\)[ \t]*\n?", re.IGNORECASE
)

# 站点提示的判据：引用块里出现这些词才当它是 chrome —— 否则开头那个引用块是**内容**
_DOC_INDEX_HINTS = ("llms.txt", "documentation index")


def _drop_leading_chrome(text: str) -> str:
    """丢掉文档**开头**那两坨站点 chrome。

    实测来源（Firecrawl 抓 mintlify）：正文前 8 行根本不是内容 ——

        > ## Documentation Index
        > Fetch the complete documentation index at: [/llms.txt](…)
        > Use this file to discover all available pages before exploring further.

        [Skip to main content](…/overview#content)

    **只看开头**（`\\A`）：正文中段的引用块是内容，一个字不能动。两坨按顺序循环丢
    （谁先谁后不固定），丢完把开头的空行收掉。
    """
    while True:
        quote = _LEADING_QUOTE.match(text)
        if quote and any(hint in quote.group(1).lower() for hint in _DOC_INDEX_HINTS):
            text = text[quote.end() :]
            continue
        skip = _LEADING_SKIP.match(text)
        if skip:
            text = text[skip.end() :]
            continue
        return text.lstrip("\n")


def clean_markdown(text: str) -> str:
    """纯文本 / markdown 的入库前清洗。

    只做四件事，都跟「文档站自动加的东西」有关，**正文一个字不改**：

    1. 去不可见字符（零宽、图标字体的私用区）—— 与 OCR 那条路共用 `strip_invisible`；
    2. 去 **label 为空的链接**（标题锚点）；
    3. 收掉标题行里因此剩下的多余空格。
    4. 丢掉**开头**的站点 chrome：`> ## Documentation Index … llms.txt` 引用块与
       `[Skip to main content](…)` 跳转链接 —— 它们既不是正文，又会被切块、进检索。

    顺序要紧：先 `strip_invisible`，`[<ZWSP>](url)` 才会退化成标准的 `[](url)`，
    第 2 条才认得出它。**正常链接 `[文字](url)` 一律保留**（那是正文的一部分）。
    """
    cleaned = _EMPTY_LINK.sub("", strip_invisible(text))
    return _HEADING.sub(_tidy_heading, _drop_leading_chrome(cleaned))


# 唯一保留的属性：它们承载**结构**（跨行/跨列合并），不是版式。
# 其余（style / border / align / width / class…）纯属版式，全丢。
KEEP_ATTRS = frozenset({"rowspan", "colspan"})

# 在这几个标签的边界补换行，给切块器留断点。只放结构性标签，不放 <td> ——
# 同一行的单元格拆开等于把一条记录拦腰截断。
_BREAK_BEFORE = frozenset({"table", "tr"})
_BREAK_AFTER = frozenset({"table", "tr"})


class _AttrStripper(HTMLParser):
    """按原样重建标签，只丢掉不保留的属性。"""

    def __init__(self) -> None:
        # convert_charrefs=False：把字符引用原样传回来。
        # 开着的话 "&amp;" 会在正文里被还原成 "&"，等于我们顺手改了内容 —— 不做。
        super().__init__(convert_charrefs=False)
        self._parts: list[str] = []
        self._source = ""
        self._line_starts: list[int] = [0]

    def feed(self, data: str) -> None:
        # 记下原文与每行起点。handle_entityref 只给名字，**给不出「原来带不带分号」**：
        # "R&D" 是裸 `&`，"&amp;" 才带分号，两者必须区分 —— 统一补分号会把
        # "R&D" 改成 "R&D;"，那是往正文里塞垃圾。只能回原文按位置切。
        self._source = data
        self._line_starts = [0]
        self._line_starts.extend(i + 1 for i, ch in enumerate(data) if ch == "\n")
        super().feed(data)

    @property
    def text(self) -> str:
        return "".join(self._parts)

    def _tag(self, tag: str, attrs: list[tuple[str, str | None]], *, closed: bool) -> None:
        kept = "".join(
            f' {name}="{value}"'
            for name, value in attrs
            if name in KEEP_ATTRS and value is not None
        )
        # 上一段已经以换行结尾就不再补：否则 "# 标题\n\n" + 表格会被补成三个换行，
        # 凭空多出一个空段落。
        if tag in _BREAK_BEFORE and self._parts and not self._parts[-1].endswith("\n"):
            self._parts.append("\n")
        self._parts.append(f"<{tag}{kept}{' /' if closed else ''}>")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._tag(tag, attrs, closed=False)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._tag(tag, attrs, closed=True)

    def handle_endtag(self, tag: str) -> None:
        tail = "\n" if tag in _BREAK_AFTER else ""
        self._parts.append(f"</{tag}>{tail}")

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def _raw_ref(self) -> str:
        """把当前字符引用按**原文**取回来（含有没有分号）。"""
        line, col = self.getpos()
        start = self._line_starts[line - 1] + col
        end = start + 1
        if self._source[end : end + 1] == "#":
            end += 1
        while end < len(self._source) and self._source[end].isalnum():
            end += 1
        if self._source[end : end + 1] == ";":
            end += 1
        return self._source[start:end]

    def handle_entityref(self, name: str) -> None:
        self._parts.append(self._raw_ref())

    def handle_charref(self, name: str) -> None:
        self._parts.append(self._raw_ref())

    def handle_comment(self, data: str) -> None:
        self._parts.append(f"<!--{data}-->")

    def handle_decl(self, decl: str) -> None:
        self._parts.append(f"<!{decl}>")

    def handle_pi(self, data: str) -> None:
        self._parts.append(f"<?{data}>")

    def unknown_decl(self, data: str) -> None:
        self._parts.append(f"<![{data}]>")


def compact_html(text: str) -> str:
    """剥掉 HTML 标签上的版式属性，结构、正文、markdown 语法都不动。

    标签名会被 ``HTMLParser`` 规范成小写（``<TD>`` → ``<td>``）—— 这是它唯一的
    副作用，对表格无影响。
    """
    stripper = _AttrStripper()
    stripper.feed(text)
    stripper.close()
    return stripper.text


def clean_ocr_markdown(text: str) -> str:
    """产线 markdown → 可直接入库的正文。"""
    return strip_invisible(compact_html(text)).strip()
