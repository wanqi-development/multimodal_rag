"""标题层级的修正：按作者写的编号重建树，保证每条路径都从根开始。

**为什么需要这一步**（2026-09-30 用真实数据查出来的）：

层级以前完全按 markdown 标题级别推（`#` 是 1 级、`##` 是 2 级……），而 OCR 产线的
`relevelTitles` 重建出来的级别**本身不可靠**：同一份教材 PDF 里，章标题、
节标题、甚至列表项经常被标成同一级。三个真实例子：

| 文档 | 库里的路径（修之前） | 实际应该是 |
|---|---|---|
| 第一章 Flink 概述 | `APP移动端数据 -> 6. 实时报表分析`（编号 `4-17`） | `1 第一章… -> 1.5 …` |
| 第二章 Flink 入门 | `MyFlinkCode FlinkScalaCode`（只有一层，编号 `9`） | 在 `2.2.1` 之下 |
| Kafka 文档 | `Documentation -> For a rolling upgrade: -> 2. APIS` | `2. APIS`（它本身就是一章） |

后果有两个：**路径不从根开始**（中间层被同级标题挤掉），以及
**`heading_numbers` 失去层级含义**（`4-17` 这种编号，用户问「第 4 章」根本对不上）。

**修法**：标题里作者写的编号（`2`、`2.1`、`2.2.1`）是**可靠信号**，用它定深度：
编号有几段就是第几层，`2.1` 的父节点就是编号为 `2` 的那个。没有编号的标题
（`Flink版本`、`• Scala环境`）按位置挂到当前节点下。声明级别只在两个地方兜底：
文档里根本没有点分编号（说明作者没用多级编号），以及「回到更浅一级」的判断。

**一条重要取舍**：`heading_numbers` 在**有作者编号的层级上直接用作者的编号**
（`2.1` → `2-1`），因为用户和大模型问的就是「第 2.1 节」；没有编号的层级用位置序号补。
这改掉了以前「一律位置派生」的做法 —— 位置派生在层级可靠时才有意义，
而在这些 OCR 出来的教材上，位置派生的编号（`4-17`）没有任何人能对上。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 标题层级**规则**的版本。它进 `pipeline_version`：规则一变，切出来的 chunk 就变了
# （heading_path / heading_numbers 不同），已灌的语料必须重切 —— 不写进指纹的话
# 增量会跳过所有文件，库里留着用旧规则算出来的层级。
#   v1：按 markdown 声明级别 + 位置派生编号（2026-09-30 之前）
#   v2：按作者编号重建树、路径必从根开始（2026-09-30）
#   v3：切分只认到 MAX_LEVEL 级（见下），更深的留在正文里当内容（2026-10-01）
# 指纹里连 MAX_LEVEL 一起记（见 incremental.pipeline_version）：改那个数字也等于改规则。
RULE_VERSION = "3"

# **切分到第几级为止**。3 = 只到三级（用户 2026-10-01 定的）：比它深的标题**不是边界**，
# 不单独成节、不进标题路径，正文并进上一个小节 —— 切到四级会把一节拆成一堆小碎块，
# 一小节一小块反而不如「一节一块」看得全。
#
# 注意它管的是**边界**，不是「哪些行算标题」：清单（`outline.tsv`）仍然一行一个标题行，
# 只是第 4 级以上的行在切分时不切。两处要用同一个数：这里与 `chunkers.base.HEADERS`。
MAX_LEVEL = 3

# 标题开头的作者编号：`2`、`2.1`、`2.2.1`、`2．1`（全角点）。
# **只认开头**：`Step 1: Download the code`、`$ \underline{\text{Step 3…}} $` 不算编号，
# 否则正文里的步骤数字会被当成章节号。
_NUMBER = re.compile(r"^[\s·•]*(\d+(?:[.．]\d+)*)")

# markdown 标题行：`## 标题`（1~6 个 #）。行尾的 # 也算 markdown 语法，一并去掉。
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def parse_numbering(title: str) -> tuple[str, ...] | None:
    """标题开头的编号 → 段列表；没有编号返回 None。

    ``"2.1 Flink 开发环境准备"`` → ``("2", "1")``；``"2. APIS"`` → ``("2",)``；
    ``".GETTING STARTED"`` / ``"• Scala环境"`` → ``None``。
    """
    match = _NUMBER.match(title)
    if match is None:
        return None
    return tuple(match.group(1).replace("．", ".").split("."))


# markdown 的代码围栏：行首最多 3 个空格 + **三个以上**同种字符（` 或 ~）。
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def code_fence_mask(text: str) -> list[bool]:
    """逐行标记「这一行在代码围栏**外面**吗」—— 长度与 `text.split("\\n")` 一致。

    为什么需要它：**代码块里的 `#` 不是标题**。实测那份 Firecrawl 抓的文档，代码样例里的
    `# Run the agent`（Python 注释）被我们的逐行扫描当成了标题 —— 清单里多出 7 行假标题，
    人不手动标掉的话，切分就会在代码注释处断开、代码块被切碎，还带上假的章节路径。

    闭合规则按 CommonMark：同种字符、且**不短于**开栏那一串才算闭合（``` 开的不能用
    `` 或者 ~~~ 关）。围栏行本身算「外面」——它不可能是标题。
    """
    out: list[bool] = []
    fence: str | None = None
    for line in text.split("\n"):
        match = _FENCE.match(line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            out.append(True)
            continue
        out.append(fence is None)
    return out


def scan_headings(text: str) -> list[tuple[int, str]]:
    """正文里的标题行 → ``[(级别, 标题), …]``，按出现顺序。

    用来判断这份文档**有没有多级编号**（决定单段编号该算「章」还是「列表项」），
    所以扫的是原文而不是切块器给的 section。

    **六级都扫**：清单要一行一个标题行（`outline.heading_lines` 也是六级），少扫一级
    两边就对不上号；至于「切到第几级」是 ``MAX_LEVEL`` 管的事，跟「哪些行是标题」无关。
    """
    out: list[tuple[int, str]] = []
    for line, outside in zip(text.splitlines(), code_fence_mask(text), strict=False):
        if not outside:
            continue  # 代码块里的 `# 注释` 不是标题（见 code_fence_mask）
        match = _HEADING.match(line)
        if match is not None and match.group(2):
            out.append((len(match.group(1)), match.group(2).strip()))
    return out


def chapter_headings(headings: list[tuple[int, str]]) -> set[str]:
    """单段编号的标题里，哪些是**章**？

    判据是结构性的：**它后面出现了以它编号开头的多段编号**。

    * ``2. APIS`` 后面有 ``2.1 Producer API`` → 是章
    * ``3. 打开Structure, 创建项目新模块`` 后面没有 ``3.x`` → 是**列表项**，不是章

    「编号接在前一个章号后面就算章」这条看着更直觉，但实测会翻车：第二章这份教材里
    的列表步骤正好是 ``1.`` ``2.`` ``3.`` ``4.`` ``5.``，于是 ``3. 打开Structure``
    被判成第 3 章、``5. 分别在两个项目模块中导入…`` 被判成第 5 章，后面所有内容
    都挂到了它们下面。看「有没有下级」才稳。
    """
    found: set[str] = set()
    for index, (_, title) in enumerate(headings):
        number = parse_numbering(title)
        if number is None or len(number) != 1:
            continue
        prefix = f"{number[0]}."
        for _, later in headings[index + 1 :]:
            later_number = parse_numbering(later)
            if (
                later_number
                and len(later_number) >= 2
                and ".".join(later_number).startswith(prefix)
            ):
                found.add(title.strip())
                break
    return found


# 「编号连着写、前几项却没被标成标题」的兜底用的正则：
# `1. xxx` / `1、xxx`（编号后**要有空白**）。所以 `1.1.1 Flink的定义` 不算 ——
# 那是章节号，`第一步` 之类没有阿拉伯数字的也不算。
_NUMBERED_PLAIN = re.compile(r"^(\d+)[.、]\s+\S")


def promote_numbered_headings(text: str) -> str:
    """把**漏标的那串编号行**补成标题，返回新的 markdown。

    **要修的是什么**（2026-10-08，`第一章 Apache Flink 概述.pdf` 实测）：dots.ocr 在
    `1.3 Flink核心特性` 一节里把 `1./2./3.` 判成普通段落、`4.–9.` 判成 `##` ——
    前三项因此**根本不进清单**（清单一行一个 `#` 行），既不进标题路径、也不成节。
    同一份 PDF 里 1.4 的 `1.–6.` 又是 `###`，可见模型自己也不一致。

    **判据是结构性的，不猜版面**，两条**都要**成立才补：

    1. 一串从 `1.` 开始、编号连续的裸编号行，**紧后面跟着的下一个标题**编号正好接在它们
       后面（`1./2./3.` 后面是 `4.`）；
    2. 那个标题**自己也在一个同级、编号连续的标题串里**（它后面同级的标题编号接着走）。

    第 2 条是 2026-10-08 拿全库 19 份解析稿量出来的**必要**约束：只有第 1 条时，Kafka 那份
    文档里有 6 行被误伤 —— 比如 `1. sudo /usr/sbin/kadmin.local …` 后面碰巧挨着一个
    `## 3. Make sure all hosts …`，可它们其实是 `## 2.` 底下的**操作步骤**，接着的
    `## 4.` 也不存在（下一节直接跳回 `## 2. Configuring Kafka Brokers`）。
    加上第 2 条之后，全库只动 **6 行**（就是报告里那两章 Flink），Kafka 那 318 行命令
    一个不碰。

    **只往行首插 `#` 和空格，一个字都不改、一行都不增删**：这也正是它能放在
    ``loaders.base.pages_to_text`` 里、按页重新打点的前提（行数不变，行↔页的对应关系照旧）。
    """
    lines = text.split("\n")
    # 标题行的事前索引：判据 2 要问「这个标题后面同级的那个标题编号接得上吗」
    headings: list[tuple[int, int, str]] = []
    for line_index, line in enumerate(lines):
        match = _HEADING.match(line.strip())
        if match is not None and match.group(2).strip():
            headings.append((line_index, len(match.group(1)), match.group(2).strip()))
    promoted: dict[int, int] = {}
    index = 0
    while index < len(lines):
        first = _NUMBERED_PLAIN.match(lines[index].strip())
        # 只从 `1.` 开始收：一半的编号串（`3./4.`）是接着前面的，起点不该在这里
        if first is None or first.group(1) != "1":
            index += 1
            continue
        run = [index]
        expected = 2
        cursor = index + 1
        while cursor < len(lines):
            line = lines[cursor].strip()
            if _HEADING.match(line):
                break
            found = _NUMBERED_PLAIN.match(line)
            if found is not None and int(found.group(1)) == expected:
                run.append(cursor)
                expected += 1
            cursor += 1
        if cursor < len(lines):
            heading = _HEADING.match(lines[cursor].strip())
            level = len(heading.group(1))
            follows = [
                title
                for line_index, width, title in headings
                if line_index > cursor and width == level
            ]
            keeps_going = bool(follows) and parse_numbering(follows[0]) == (str(expected + 1),)
            if keeps_going and parse_numbering(heading.group(2)) == (str(expected),):
                for line_index in run:
                    promoted[line_index] = level
                index = cursor
                continue
        index += 1
    if not promoted:
        return text
    return "\n".join(
        f"{'#' * promoted[index]} {line}" if index in promoted else line
        for index, line in enumerate(lines)
    )


@dataclass
class _Node:
    """树里的一个节点。``number`` 是作者编号的最后一段（没有就是 None）。"""

    title: str
    level: int
    number: str | None = None
    depth: int = 1
    # 没有作者编号时，用「同一父下的第几个」当编号。**在落位时算好**，
    # 渲染时只读 —— 否则每渲染一次就把兄弟计数又加一（写成用例才发现的）。
    position: int = 0
    # 结构节点 = 带**可信编号**的标题（`2`、`2.1`、`2.2.1`）。骨架只由它们搭，
    # 其余标题（无编号的 `Flink版本`、列表项 `1. 打开IDEA`）都是挂在骨架上的内容。
    structural: bool = False


@dataclass
class HeadingTree:
    """把标题流修成「每条路径都从根开始」的树。

    用法：每个文档建一个，按出现顺序把标题喂进 ``push()``，拿回 ``(路径, 编号)``。
    ``chapters`` 来自 ``chapter_headings()``，它决定单段编号算章还是算列表项。
    """

    chapters: set[str] = field(default_factory=set)
    # **信任外部给的层级**（人工对齐过的清单 / 本来就可靠的 markdown 级别）：
    # 深度直接取那个层级，不再用编号去猜。见 ingest/outline.py。
    trust_levels: bool = False
    _slots: dict[int, _Node] = field(default_factory=dict)
    _siblings: dict[tuple, int] = field(default_factory=dict)
    _prev_level: int = 0
    _prev_depth: int = 0
    # 每个声明级别上次落在第几层。声明级别变浅时要靠它回到正确的层
    # （docx 的 h1「第二章」要回到第 1 层，而不是挂在第一章下面）。
    _level_depth: dict[int, int] = field(default_factory=dict)
    # 已经确认过的最大章号。列表步骤会从 1 重新数，所以「编号不比当前章号大」的单段
    # 编号一律不当章（`2. 在IntelliJ IDEA 中安装Scala插件` 与当前章号 2 相同）。
    _max_chapter: int = 0

    def push(self, level: int, title: str) -> tuple[str, str]:
        """一个新标题 → ``(从根开始的完整路径, 层级编号)``。"""
        number = parse_numbering(title)
        if self.trust_levels:
            # 层级是外部给的（人核对过），直接用；只压平跳级，保证路径逐级下探。
            depth = min(level, self._prev_depth + 1) if self._prev_depth else max(level, 1)
            structural = number  # 只用来取编号文本，不参与定深度
        else:
            structural = self._structural_number(number, title)
            depth = self._depth_for(level, structural)

        # 清掉这一级及更深的旧节点，然后落位。父节点没出现过时补一个「编号占位」，
        # 保证路径不断链（Kafka 那篇就出现过 `1.2` 而没有 `1.`）。
        self._slots = {d: node for d, node in self._slots.items() if d < depth}
        if not self.trust_levels:
            self._fill_gaps(structural, depth)
        parent_key = (depth, tuple(self._slots[d].title for d in range(1, depth)))
        if structural is None:
            self._siblings[parent_key] = self._siblings.get(parent_key, 0) + 1
        self._slots[depth] = _Node(
            title=title,
            level=level,
            number=structural[-1] if structural else None,
            depth=depth,
            position=self._siblings.get(parent_key, 0),
            structural=structural is not None,
        )
        self._prev_level = level
        self._prev_depth = depth
        self._level_depth[level] = depth
        return self._render()

    # ---- 深度决策 ----

    def _structural_number(
        self, number: tuple[str, ...] | None, title: str
    ) -> tuple[str, ...] | None:
        """这个编号可信到能当骨架吗？不可信就返回 None（当普通内容挂上去）。

        * `2.1` / `2.2.1` 这类**多段编号**：可信，段数就是层级。
        * `2. APIS` 这类**单段编号**：只有在 `chapter_headings()` 判它是章时才算
          —— 判据是「后面有没有它的小节」。`3. 打开Structure, 创建项目新模块` 后面
          没有 `3.x`，它只是**列表项**；当成章会把后面所有内容挂到它下面（实测撞到过）。
        * 没有编号：不可信。
        """
        if number is None:
            return None
        if len(number) >= 2:
            return number
        if title.strip() not in self.chapters:
            return None
        if number[0].isdigit() and int(number[0]) <= self._max_chapter:
            return None
        return number

    def _depth_for(self, level: int, structural: tuple[str, ...] | None) -> int:
        if structural is not None:
            # 结构节点：深度 = 编号段数（父节点按**位置**对齐到 depth-1）
            depth = max(len(structural), 1)
            if depth == 1 and structural[0].isdigit():
                self._max_chapter = max(self._max_chapter, int(structural[0]))
            return depth
        if not self._slots:
            return 1  # 第一个标题就是根
        anchors = [node.depth for node in self._slots.values() if node.structural]
        if not anchors:
            # 文档里还没有结构骨架（docx 那种「第一章」+「1、2、3」），按声明级别走
            return self._depth_by_level(level)
        # 内容节点：挂在**最近的结构节点**之下。这条是关键 —— 挂「栈顶」的话，
        # 列表项一旦成为栈顶，后面每个无编号标题都会再深一层、越漂越远
        # （实测漂到第 6 层，把章标题挂进了列表项里面）。
        base = max(anchors) + 1
        if level > self._prev_level:
            # 声明级别说它更深（docx 的 h1 → h2 → h3）→ 在锚点之下再下探一层
            return max(base, self._prev_depth + 1)
        if level == self._prev_level:
            # 同级：跟着上一个同级标题，别再往下探（`4. Flink Java Api…` 与
            # `3. Flink批和流…` 是两个并列的步骤，不该一个比一个深）
            return max(base, self._prev_depth)
        return base

    def _depth_by_level(self, level: int) -> int:
        """没有编号可用时，按声明级别的相对变化定深度。"""
        if level > self._prev_level:
            return self._prev_depth + 1
        if level == self._prev_level:
            return self._prev_depth
        return self._level_depth.get(level, 1)

    # ---- 渲染 ----

    def _fill_gaps(self, number: tuple[str, ...] | None, depth: int) -> None:
        """父节点没出现在标题流里时补占位，节点标题就用编号本身。"""
        if number is None or len(number) < 2:
            return
        for missing in range(1, min(len(number), depth)):
            if missing in self._slots:
                continue
            prefix = ".".join(number[:missing])
            self._slots[missing] = _Node(
                title=prefix, level=1, number=number[missing - 1], depth=missing
            )

    def _render(self) -> tuple[str, str]:
        depth = self._top_depth()
        titles: list[str] = []
        numbers: list[str] = []
        for d in range(1, depth + 1):
            node = self._slots[d]
            titles.append(node.title)
            numbers.append(node.number if node.number is not None else str(node.position))
        return " -> ".join(titles), "-".join(numbers)

    def _top_depth(self) -> int:
        return max(self._slots) if self._slots else 0
