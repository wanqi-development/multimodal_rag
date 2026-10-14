"""标题层级的人工核对：把「哪一行是几级标题、这份文档切到第几级」交给人定，之后才切分。

**为什么所有文档都要人工核对**（2026-10-02 定的，之前只对 PDF）：层级是切分的唯一依据，
而三个来源都靠不住：

* **版面**：拿真实响应量过 —— 标题块的 bbox 高度与真实层级**完全重叠**
  （已知第 3 层的比值 0.30–0.52、第 4 层 0.00–0.49、第 5 层 0.00–0.92），
  因为块高是多行块的整块高度，折一行就翻倍；标签也只有 `paragraph_title` 一档。
* **编号文本**：写法千变万化（`1` / `一、` / `第一章` / `(1)` / `（一）` / `①`），
  而且 Word 的编号是渲染时生成的、段落文本里根本没有。
* **声明级别**：docx/html/md 里的 `#` 数量只是作者的排版习惯，OCR 出来的 `#/##/###` 更
  是猜的 —— 实测同一份教材里章、节、列表项经常被标成同一级。

所以流程改成：**解析 → 出一份清单 → 人核对（改层级、**选最大层级**、喊「已核对」）
→ 才允许切分**。清单是 ``data/stages/<doc_id>/outline.tsv``，长这样：

    # 最大层级: 3
    # 已核对: 否
    层级	编号推断	标题
    1	2	2第二章 Apache Flink 快速入门
    2	3	2.1 Flink 开发环境准备
    3	2	Flink版本

三件事都在这个文件里：

1. 第一列是**唯一有效的那一列**（1..6），人改它就行；第三列是标题原文，第二列是按作者
   编号推断的结果，供对照。**写 `0` = 这一行不是标题、算正文**（使用者 2026-10-07 定）：
   OCR 常把一段普通文字标成标题，人看一眼就知道那不是标题 —— 但**正文一个字都不能改**
   （改了偏移就全错位），所以改的是它的「角色」。写 0 的行不进标题路径、不单独成节，
   那行原文（含 `#`）留在正文里；
2. ``# 最大层级: N`` —— 这份文档**切到第几级为止**：比 N 深的行不单独成节、不进标题路径，
   正文并进上一个小节（见 `sections`）。不同文档可以选不同的值；
3. ``# 已核对: 否 → 是`` —— 人工核对完喊的这一声。**没喊就不许切分**：
   `run_chunk` 会把这份文档列进「待人工核对」并跳过。

**两个实现上的关键点：**

1. **改层级不改变正文**。切 section 时标题行不进正文（与切分器的 `strip_headers=True` 一致），
   所以正文分片仍是原文的**逐字子串**，页码/行号映射（``positions``）依然精确。
2. **层级允许跳级但会被压平**：`1 → 3` 这种（中间缺一层）按 `1 → 2` 处理并提示，
   否则路径会缺一层、树会歪。**路径从根开始这条保证不因人写错而破**。

**为什么不改正文、而是另走一条切分路径**：正文一改，``Document.positions``（偏移 →
页码/行号）就全错位了。清单只改「层级」，而**哪些行是标题一行都没变**，所以按行号切
section 即可 —— 正文分片仍是原文的**逐字子串**，定位（``pipeline.locate``）照旧精确。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import NamedTuple

from rag.ingest.headings import (
    MAX_LEVEL,
    HeadingTree,
    chapter_headings,
    code_fence_mask,
    scan_headings,
)

# 与切分器同一套识别规则：`#` 开头、后面要么是行尾要么是空格。
# 多空格也认 —— 套用时正是靠补空格把行宽保持住的。
_HEADING = re.compile(r"^(#{1,6})(\s+)(.*)$")

HEADER = "层级\t编号推断\t标题"

# 清单头部的两条指令（都写成 `#` 注释行，人直接改数字/词）。
MAX_LEVEL_PREFIX = "# 最大层级:"
CONFIRM_PREFIX = "# 已核对:"
CONFIRMED_WORDS = ("是", "y", "yes", "true", "1")
UNCONFIRMED_WORDS = ("否", "n", "no", "false", "0")


@dataclass(frozen=True)
class Outline:
    """一份人工核对过的清单：层级 + 这份文档切到第几级 + 人有没有喊「已核对」。"""

    levels: list[int]
    max_level: int = MAX_LEVEL
    confirmed: bool = False
    # 清单头部没写指令、或写了个认不出来的词时记一句，给用户看的
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HeadingLine:
    """正文里的一行标题。``prefix`` 是 `#` 加空格，套用时按同样的宽度改写。"""

    line_index: int
    level: int
    prefix: str
    title: str


class Section(NamedTuple):
    """一个标题下的正文。``heading_path`` / ``heading_numbers`` 直接来自清单。"""

    heading_path: str
    heading_numbers: str
    body: str


def heading_lines(text: str) -> list[HeadingLine]:
    """按出现顺序列出正文里的标题行（六级都算）。

    **这里不按 ``MAX_LEVEL`` 过滤**：清单是一行一个标题行，切到第几级是
    ``sections`` 的事。少列一行就会与清单条目数对不上，那条校验整天响。

    **代码围栏里的行不算**（`code_fence_mask`）：代码样例里的 `# …` 是注释，不是标题 ——
    实测那份 Firecrawl 抓的文档，代码里的 `# Run the agent` 被当成标题，清单多出 7 行；
    不修的话切分会在代码注释处断开、代码块被切碎（2026-10-10）。
    """
    out: list[HeadingLine] = []
    for index, (line, outside) in enumerate(
        zip(text.split("\n"), code_fence_mask(text), strict=False)
    ):
        if not outside:
            continue
        match = _HEADING.match(line)
        if match is None or not match.group(3).strip():
            continue
        out.append(
            HeadingLine(
                line_index=index,
                level=len(match.group(1)),
                prefix=match.group(1) + match.group(2),
                title=match.group(3).strip(),
            )
        )
    return out


def first_title(text: str) -> str:
    """文档标题：**代码围栏之外**第一个一级标题的文字；没有就返回空串。

    给 loader 填 `metadata["title"]` 用（页面与将来的列表要显示文档名）。
    **不能拿正则直接搜 `^# `**：代码样例里的 `# 注释` 会被匹配到 —— 实测那份 Firecrawl
    抓的文档就是这么取到 "Run the agent" 的（2026-10-10 修）。
    """
    for line in heading_lines(text):
        if line.level == 1:
            return line.title
    return ""


def render_outline(
    text: str, levels: list[int] | None = None, *, max_level: int = MAX_LEVEL
) -> str:
    """生成人读的清单。

    **初值优先用解析器给的声明层级**（`#`/`##`/`###`）—— dots.ocr 是端到端的版面模型，
    它给的层级实测基本可用（章 1 / 节 2 / 小节 3），人只需修少数几行。第二列放
    「按作者编号推断」的结果供对照：两者差得远的行，往往就是要人工确认的那几行。

    ``levels`` 可以显式覆盖初值（测试与调试用）。

    生成的清单**永远是「未核对」**（`# 已核对: 否`）：切分只认人手改过的那一份，
    否则「自动生成的初值」会被当成核对结果直接切下去 —— 那正是这条流程要防的事。
    """
    lines = [
        "# 标题层级清单（人工核对用）",
        f"{MAX_LEVEL_PREFIX} {max_level}",
        f"{CONFIRM_PREFIX} 否",
        "#",
        "# 核对三件事：",
        "#   1) 第一列的层级对不对（1..6，从根开始逐级下探）；",
        "#      想表达「这是子项、别单独切一块」就写深一级（不切到那么深就行）；",
        "#      这一行压根不是标题（解析认错了）就写 0 —— 它算正文，不进标题路径；",
        "#   2) 上面那行「最大层级」选几 —— 比它深的行不单独成节、不进标题路径，正文并进上一节；",
        "#   3) 都对了，把上面那行「已核对」改成「是」；没改之前 --stage chunk 会跳过这份文档。",
        "#",
        "# 第一列的初值＝解析器给的层级；第二列是按作者编号推断的结果，供对照 ——",
        "# 它只是参考，一切以第一列为准。改完保存，再跑 --stage chunk。",
        HEADER,
    ]
    headings = heading_lines(text)
    guessed = initial_levels(text)
    for index, heading in enumerate(headings):
        level = levels[index] if levels else heading.level
        hint = guessed[index] if index < len(guessed) else heading.level
        lines.append(f"{level}\t{hint}\t{heading.title}")
    return "\n".join(lines) + "\n"


def render_reviewed_outline(
    text: str, levels: list[int], *, max_level: int = MAX_LEVEL, confirmed: bool = False
) -> str:
    """按人工给定的层级重排清单 —— 页面提交核对结果时写回用的就是这一份。

    **与 `render_outline` 只差「已核对」那一行**：自动生成的那条路永远写「否」，这里是
    人提交的，`confirmed=True` 就该写「是」。拆成两个函数是为了把那条不变量留在
    `render_outline` 里 —— 否则「自动生成的初值被当成核对结果」这道口子就开了
    （那正是这条流程要防的事）。

    正文一个字都不动：只重排「第几行是几级」这一列，标题原文照抄。
    """
    rendered = render_outline(text, levels, max_level=max_level)
    if not confirmed:
        return rendered
    return rendered.replace(f"{CONFIRM_PREFIX} 否", f"{CONFIRM_PREFIX} 是", 1)


def initial_levels(text: str) -> list[int]:
    """初值：按作者编号 + 声明级别推断出来的深度（``headings.HeadingTree``）。

    只是给人一个起点，省得从零填；改不改、怎么改都由人决定。
    """
    headings = scan_headings(text)
    tree = HeadingTree(chapters=chapter_headings(headings))
    return [len(tree.push(level, title)[0].split(" -> ")) for level, title in headings]


def parse_outline(text: str) -> list[int]:
    """读回清单里的层级。返回的列表长度 = 清单里的条目数。

    只认第一列；`#` 开头的注释行与空行跳过。**0 与 1..6** 之外的一律**直接报错** ——
    这是人工填的东西，写错了要立刻说清楚是第几行，别猜。

    `0` = 这一行不是标题、算正文（见模块说明）。它照样占一行：清单与正文的标题行必须
    一一对应，少一行整份就切不了。

    **条目数必须与 ``heading_lines`` 一致**（一行一个标题行），所以这里一行都不能少；
    「第 4 级以上的行不切分」是在 ``sections`` 里处理的，不是在这里丢行。
    """
    levels: list[int] = []
    for number, raw in enumerate(text.split("\n"), start=1):
        line = raw.strip()
        if not line or line.startswith("#") or line == HEADER:
            continue
        first = line.split("\t")[0].strip()
        if not first.isdigit() or not 0 <= int(first) <= 6:
            raise ValueError(f"outline.tsv 第 {number} 行的层级不是 0..6（0 = 算正文）：{raw!r}")
        levels.append(int(first))
    return levels


def read_rows(text: str) -> list[tuple[int, int, str]]:
    """清单的**表格部分** → `[(层级, 编号推断, 标题)]`（注释行与表头跳过）。

    页面要按「一行一行的层级」展示与编辑，所以要的是这三列；`read_outline` 只关心
    层级列表 + 最大层级 + 有没有核对过。两者读的是同一份 `outline.tsv`，不另立格式。
    """
    rows: list[tuple[int, int, str]] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line or line.startswith("#") or line == HEADER:
            continue
        parts = [item.strip() for item in line.split("\t")]
        if not parts[0].isdigit():
            continue
        level = int(parts[0])
        hint = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else level
        rows.append((level, hint, parts[2] if len(parts) > 2 else ""))
    return rows


def read_outline(text: str) -> Outline:
    """读清单：层级 + 最大层级 + 有没有人工核对过。

    头部那两行（``# 最大层级:`` / ``# 已核对:``）是**人写的**，所以：

    * 没写 / 认不出来的词 → 不报错，**当作未核对**并记一句 notes。报错会让人对着
      「本来是自动生成的旧清单」反复试；而「未核对」是安全的那一侧，切分自然会挡住。
    * 最大层级写了个不可能的数（0、7、abc）→ **报错**，指明是这一行：那是明确的笔误，
      猜一个数字继续跑会把切法悄悄改掉。
    """
    max_level = MAX_LEVEL
    confirmed = False
    notes: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if line.startswith(MAX_LEVEL_PREFIX):
            value = line[len(MAX_LEVEL_PREFIX) :].strip()
            if not value.isdigit() or not 1 <= int(value) <= 6:
                raise ValueError(f"{MAX_LEVEL_PREFIX} 后面要是 1..6，现在是 {raw!r}")
            max_level = int(value)
        elif line.startswith(CONFIRM_PREFIX):
            value = line[len(CONFIRM_PREFIX) :].strip().lower()
            if value in CONFIRMED_WORDS:
                confirmed = True
            elif value not in UNCONFIRMED_WORDS:
                notes.append(
                    f"{CONFIRM_PREFIX} 后面只认「是 / 否」，现在是 {raw!r}（按未核对处理）"
                )
    if not any(line.strip().startswith(CONFIRM_PREFIX) for line in text.split("\n")):
        notes.append(f"清单头部没有 {CONFIRM_PREFIX} 这一行（按未核对处理）")
    return Outline(
        levels=parse_outline(text), max_level=max_level, confirmed=confirmed, notes=notes
    )


def normalize_levels(levels: list[int]) -> tuple[list[int], list[str]]:
    """跳级压平：`1 → 3` 按 `1 → 2` 处理（路径必须逐级下探，否则会缺一层）。

    这是**人工**填的东西，写错了只提示、不报错 —— 报错会让人对着行号反复试；
    压平的结果看得见（切完的 `chunks.md` 里就是最终路径）。

    **`0`（算正文）原样穿过**：它不在层级阶梯上，既不参与压平、也不改变「上一条是几级」
    —— 否则 `1, 0, 2` 会被算成 `1, 1, 1`（0 被顶成 1 级、下一条又被压到 2 级以内）。
    """
    out: list[int] = []
    notes: list[str] = []
    previous = 0
    for raw in levels:
        if raw <= 0:
            out.append(0)
            continue
        level = min(raw, previous + 1) if previous else max(raw, 1)
        if level != raw:
            notes.append(
                f"第 {len(out) + 1} 条写成 {raw} 级，但上一条是 {previous} 级，已按 {level} 级处理"
            )
        out.append(level)
        previous = level
    return out, notes


def sections(text: str, levels: list[int], *, max_level: int = MAX_LEVEL) -> list[Section]:
    """按「清单给的层级」把正文切成若干 section。

    **哪些行是标题完全由 ``heading_lines`` 决定**（也就是正文里现有的 `#` 行），
    清单只决定它们的层级 —— 所以正文一个字都不动，分片仍是原文子串。
    标题行本身**不进正文**（与切分器 ``strip_headers=True`` 的表现一致），
    路径贴在块正文开头由上层负责（``chunkers.base.with_heading_path``）。

    **层级 0 = 这一行算正文**：不是切分边界、不进标题路径，那行原文（含 `#`）留在正文里。
    这与「比 max_level 深的行」是同一种处理 —— 区别只是人事先明说了它压根不是标题，
    而不是靠「切到第几级」把它们顺带并进去。

    **只到 ``max_level`` 级为止**（2026-10-01 定的规则，2026-10-02 改成按文档选）：
    层级比它深的那几行**不单独成节** ——
    它们连同自己的正文一起并进上一个小节。这正是「子项别单独切一块」的表达方式：
    清单里把子项写成更深一级即可，不必去删行（删行会让条目数与正文对不上）。
    所以合出来的正文里会**原样留着**那些更深标题的行（`## 无界流` 这种），
    和切分器对未知标题行的处理一致 —— 正文一个字都没动。
    """
    headings = heading_lines(text)
    flat, _ = normalize_levels(levels)
    if len(headings) != len(flat):
        raise ValueError(
            f"清单条目数（{len(flat)}）与正文里的标题数（{len(headings)}）对不上："
            f"清单只该改层级、不该增删行。要重新生成清单就删掉它，重跑 --stage parse。"
        )

    rows = text.split("\n")
    tree = HeadingTree(trust_levels=True)
    out: list[Section] = []
    # 只有这些行是边界；其余（层级更深、或标成 0 的正文行）并进上一节
    boundaries = [index for index, level in enumerate(flat) if 1 <= level <= max_level]
    if not boundaries:
        # 整份文档没有一个够浅的标题：就是一段，路径为空 —— 与切分器的行为一致
        body = text.strip()
        return [Section("", "", body)] if body else []
    # 第一个**边界**标题之前的内容（页眉、前言之类）单独成一段，路径为空
    if headings[boundaries[0]].line_index > 0:
        body = "\n".join(rows[: headings[boundaries[0]].line_index]).strip()
        if body:
            out.append(Section("", "", body))
    for position, index in enumerate(boundaries):
        heading = headings[index]
        # 正文一直取到**下一个边界标题**为止：中间那些更深的标题行与它们的正文都留在里面
        following = boundaries[position + 1] if position + 1 < len(boundaries) else None
        end = headings[following].line_index if following is not None else len(rows)
        body = "\n".join(rows[heading.line_index + 1 : end]).strip()
        path, numbers = tree.push(flat[index], heading.title)
        out.append(Section(path, numbers, body))
    return out
