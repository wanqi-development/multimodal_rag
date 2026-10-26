"""标题层级的人工对齐：清单读写、跳级压平、按清单切 section。

全程本地、无网络；标题序列取自真实文档（第二章 Flink 入门那份 OCR 输出）。
"""

from __future__ import annotations

import pytest

from rag.ingest.headings import HeadingTree
from rag.ingest.outline import (
    CONFIRM_PREFIX,
    HEADER,
    MAX_LEVEL_PREFIX,
    first_title,
    heading_lines,
    initial_levels,
    normalize_levels,
    parse_outline,
    read_outline,
    render_outline,
    sections,
)

# 真实文档的开头（OCR 给的层级本身就乱：章是 h2、节是 h3、列表项又是 h2/h3）
_DOC = (
    "## 2第二章 Apache Flink 快速入门\n"
    "\n"
    "学习一门新的编程语言时……\n"
    "\n"
    "### 2.1 Flink 开发环境准备\n"
    "\n"
    "本节讲环境准备。\n"
    "\n"
    "## Flink版本\n"
    "\n"
    "本套课程采用 1.16.0。\n"
    "\n"
    "### 1. 打开IDEA，创建空项目\n"
    "\n"
    "先建空项目。\n"
)


def test_heading_lines_列表按出现顺序带原前缀() -> None:
    lines = heading_lines(_DOC)

    assert [line.title for line in lines] == [
        "2第二章 Apache Flink 快速入门",
        "2.1 Flink 开发环境准备",
        "Flink版本",
        "1. 打开IDEA，创建空项目",
    ]
    assert [line.level for line in lines] == [2, 3, 2, 3]
    assert lines[0].prefix == "## "


# ---- 清单的读写 ----


def test_render_outline_两列分别是人工层级与编号推断() -> None:
    text = render_outline(_DOC, levels=[1, 2, 2, 3])

    assert text.startswith("# 标题层级清单")
    assert HEADER in text
    assert "1\t1\t2第二章 Apache Flink 快速入门" in text
    assert "2\t3\tFlink版本" in text, "第一列是人工层级，第二列是编号推断的参考值"


def test_初值默认用解析器给的声明层级() -> None:
    """dots.ocr 直接在标题里给 `#`/`##`/`###` —— 初值就该用它，人只改不对的几行。

    第二列放编号推断的结果供对照：两列差得远的行，往往就是要人工确认的那几行。
    """
    text = render_outline(_DOC)  # _DOC 里声明的层级是 2/3/2/3

    assert "2\t1\t2第二章 Apache Flink 快速入门" in text
    assert "3\t4\t1. 打开IDEA，创建空项目" in text


def test_parse_outline_跳过注释只读第一列() -> None:
    text = "# 注释\n" + HEADER + "\n1\t2\t标题甲\n3\t2\t标题乙\n\n"

    assert parse_outline(text) == [1, 3]


def test_parse_outline_非法层级直接报错并指出行号() -> None:
    with pytest.raises(ValueError, match="第 3 行"):
        parse_outline(f"{HEADER}\n1\t2\t甲\n七\t2\t乙\n")

    with pytest.raises(ValueError, match="第 2 行"):
        parse_outline(f"{HEADER}\n9\t2\t甲\n")


def test_render_then_parse_round_trips() -> None:
    text = render_outline(_DOC, levels=[1, 2, 3, 2])

    assert parse_outline(text) == [1, 2, 3, 2]


# ---- 只到三级为止（2026-10-01）----

_DEEP = "# 1 概述\n\n## 1.1 是什么\n\n### 1.1.1 定义\n\n#### 无界流\n\n更深的标题留在正文里。\n"


# ---- 人工核对：最大层级由这份清单自己定，核对完才允许切分（2026-10-02）----


def test_a_generated_list_is_never_confirmed() -> None:
    """自动生成的清单**永远**是「未核对」：初值再合理也只是初值，切分只认人点过头的。"""
    outline = read_outline(render_outline(_DOC))

    assert outline.confirmed is False
    assert outline.max_level == 3


def test_the_human_can_pick_the_max_level_per_document() -> None:
    """最大层级是**每份文档**自己选的：同一批语料里各份可以切到不同的深度。"""
    text = render_outline(_DOC).replace(f"{MAX_LEVEL_PREFIX} 3", f"{MAX_LEVEL_PREFIX} 2")
    text = text.replace(f"{CONFIRM_PREFIX} 否", f"{CONFIRM_PREFIX} 是")

    outline = read_outline(text)

    assert (outline.max_level, outline.confirmed) == (2, True)


def test_an_old_list_without_the_directives_counts_as_unconfirmed() -> None:
    """加这道门槛之前生成的清单没有这两行 —— 按「未核对」处理，并说清原因。"""
    outline = read_outline(f"{HEADER}\n1\t1\t甲\n")

    assert outline.confirmed is False
    assert outline.notes and "已核对" in outline.notes[0]


def test_an_unreadable_confirm_word_is_unconfirmed_not_an_error() -> None:
    outline = read_outline(f"{CONFIRM_PREFIX} 大概是吧\n{HEADER}\n1\t1\t甲\n")

    assert outline.confirmed is False
    assert outline.notes and "是 / 否" in outline.notes[0]


def test_a_broken_max_level_is_an_error() -> None:
    """这是明确的笔误，不能猜一个数字继续跑 —— 猜错等于悄悄换了切法。"""
    with pytest.raises(ValueError, match="最大层级"):
        read_outline(f"{MAX_LEVEL_PREFIX} 7\n{HEADER}\n1\t1\t甲\n")


def test_sections_can_stop_at_a_shallower_level_for_one_document() -> None:
    """同一份正文，选 3 切到三级、选 2 就把三级并进二级 —— 切多深由清单说了算。"""
    deep = sections(_DEEP, [1, 2, 3, 4], max_level=3)
    shallow = sections(_DEEP, [1, 2, 3, 4], max_level=2)

    assert len(deep) == 3
    assert len(shallow) == 2
    assert "### 1.1.1 定义" in shallow[-1].body


def test_heading_lines_列出每一行标题() -> None:
    """清单一行一个标题行 —— 这里不做级别过滤（过滤了条目数就对不上）。"""
    lines = heading_lines(_DEEP)

    assert [line.title for line in lines] == ["1 概述", "1.1 是什么", "1.1.1 定义", "无界流"]


def test_render_outline_每一行标题都给一行() -> None:
    text = render_outline(_DEEP)

    assert "3\t3\t1.1.1 定义" in text
    assert "4\t4\t无界流" in text, "四级照样出行：它由人决定要不要写浅一点"


def test_a_heading_deeper_than_max_level_is_not_its_own_section() -> None:
    """清单里写成 4 级的行**不单独成节**：它和它的正文并进上一个小节。

    「子项别单独切一块」就是这么表达的 —— 写深一级即可，不用删行（删行会让
    清单条目数与正文标题数对不上）。这也是人工对齐过的清单在用的手法。
    """
    result = sections(_DEEP, [1, 2, 3, 4])

    assert [s.heading_path for s in result] == [
        "1 概述",
        "1 概述 -> 1.1 是什么",
        "1 概述 -> 1.1 是什么 -> 1.1.1 定义",
    ]
    body = result[-1].body
    assert body.startswith("#### 无界流"), "更深那行原样留在正文里（正文一个字都没动）"
    assert "更深的标题留在正文里。" in body


def test_a_document_with_no_shallow_enough_heading_is_one_section() -> None:
    """一份文档的标题全深于 ``MAX_LEVEL`` 时不切碎它 —— 就是一段，路径为空。"""
    result = sections("#### 甲\n\n正文。\n", [4])

    assert len(result) == 1
    assert result[0].heading_path == ""
    assert "正文。" in result[0].body


# ---- 跳级压平 ----


def test_跳级被压平并给出提示() -> None:
    levels, notes = normalize_levels([1, 3, 2, 5])

    assert levels == [1, 2, 2, 3], "3 级接在 1 级后面 → 压成 2；5 级接在 2 级后面 → 压成 3"
    assert len(notes) == 2
    assert "跳级" in notes[0] or "已按" in notes[0]


def test_第一个标题不是一级时也会被抬到从根开始() -> None:
    assert normalize_levels([3, 4])[0] == [3, 4], "首个标题的层级就是根，不必压"


# ---- 层级 0：这一行不是标题、算正文（2026-10-07 定）----


def test_写零表示这一行算正文_不进路径也不单独成节() -> None:
    """解析常把一段普通文字认成标题；人把它标成 0 → 它留在正文里，不进标题路径。"""
    result = sections(_DOC, [1, 0, 3, 3])

    assert [s.heading_path for s in result] == [
        "2第二章 Apache Flink 快速入门",
        "2第二章 Apache Flink 快速入门 -> Flink版本",
        "2第二章 Apache Flink 快速入门 -> Flink版本 -> 1. 打开IDEA，创建空项目",
    ], "标 0 的那一行不出现"
    assert result[0].body == (
        "学习一门新的编程语言时……\n\n### 2.1 Flink 开发环境准备\n\n本节讲环境准备。"
    ), "那行的原文（含 ###）原样并进上一节 —— 正文一个字都没改"
    for section in result:
        assert section.body in _DOC, "仍是原文的逐字子串"


def test_标成零不会把后面的层级压平() -> None:
    """`1, 0, 2` 里的 0 不该被顶成 1 级、也不该把后面的 2 级压到 1 级。"""
    assert normalize_levels([1, 0, 2]) == ([1, 0, 2], [])
    assert normalize_levels([1, 0, 5]) == (
        [1, 0, 2],
        ["第 3 条写成 5 级，但上一条是 1 级，已按 2 级处理"],
    )


def test_整篇都被标成正文时就是一段_路径为空() -> None:
    result = sections(_DOC, [0, 0, 0, 0])

    assert len(result) == 1
    assert result[0].heading_path == ""
    assert "### 2.1 Flink 开发环境准备" in result[0].body


def test_清单里的零能读写往返() -> None:
    text = render_outline(_DOC, [1, 0, 3, 0])

    assert parse_outline(text) == [1, 0, 3, 0]
    assert "\n0\t" in text, "写回文件里就是第一列的 0"
    assert "算正文" in text, "头部注释里说清 0 是什么意思"


def test_零之外的越界值仍然报错() -> None:
    with pytest.raises(ValueError, match="0..6"):
        parse_outline(f"{HEADER}\n7\t2\t甲\n")


# ---- 按清单切 section ----


def test_sections_路径来自清单而正文一字不动() -> None:
    result = sections(_DOC, [1, 2, 3, 3])

    assert [s.heading_path for s in result] == [
        "2第二章 Apache Flink 快速入门",
        "2第二章 Apache Flink 快速入门 -> 2.1 Flink 开发环境准备",
        "2第二章 Apache Flink 快速入门 -> 2.1 Flink 开发环境准备 -> Flink版本",
        "2第二章 Apache Flink 快速入门 -> 2.1 Flink 开发环境准备 -> 1. 打开IDEA，创建空项目",
    ]
    assert result[0].body == "学习一门新的编程语言时……"
    assert result[1].body == "本节讲环境准备。"
    assert result[2].body == "本套课程采用 1.16.0。"
    # 正文分片必须是原文的**逐字子串**，否则切完的块定位不回页码/行号
    for section in result:
        assert section.body in _DOC


def test_sections_编号优先用作者写的() -> None:
    result = sections(_DOC, [1, 2, 3, 3])

    # 作者写的编号直接用（`2` / `2.1`）；`1. 打开IDEA…` 是**列表项不是章**，
    # 于是退回位置编号 —— 两条同级列表项拿到同一个编号，是这条规则的已知表现。
    assert [s.heading_numbers for s in result] == ["2", "2-1", "2-1-1", "2-1-1"]


def test_sections_第一个标题之前的内容单独成段() -> None:
    text = "（这是页眉）\n\n## 标题\n\n正文。\n"

    result = sections(text, [1])

    assert result[0].heading_path == ""
    assert result[0].body == "（这是页眉）"
    assert result[1].body == "正文。"


def test_sections_清单与正文条数对不上时报错() -> None:
    with pytest.raises(ValueError, match="对不上"):
        sections(_DOC, [1, 2])


def test_信任层级时按清单走不再用编号猜() -> None:
    """人工对齐过的层级就是权威：`1. 打开IDEA` 不会被当成一章。"""
    tree = HeadingTree(trust_levels=True)

    assert tree.push(1, "2第二章 Apache Flink 快速入门")[0] == "2第二章 Apache Flink 快速入门"
    assert tree.push(2, "2.1 Flink 开发环境准备")[0].endswith("-> 2.1 Flink 开发环境准备")
    # 清单说它是第 4 层 → 就挂到第 4 层，不管它的编号长得像不像章
    assert tree.push(4, "1. 打开IDEA，创建空项目")[0].endswith("-> 1. 打开IDEA，创建空项目")


# ---- 初值 ----


def test_initial_levels_用编号推断出一个可用的起点() -> None:
    """清单第一列的初值来自现有的编号推断 —— 只是起点，以人工修改为准。"""
    levels = initial_levels(_DOC)

    assert levels == [1, 2, 3, 4]
    rows = [
        line
        for line in render_outline(_DOC).splitlines()
        if line and not line.startswith("#") and line != HEADER
    ]
    assert len(rows) == len(levels), "清单里每个标题一行"


# ---- 清单与标题：代码围栏里的 `#` 不算（2026-10-10）----


def test_heading_lines_skips_the_ones_inside_a_code_fence() -> None:
    """实测来源：Firecrawl 抓的文档，代码样例里的 `# Run the agent` 进了清单（28 行里 8 行是假的），
    人不手动标掉的话切分就在代码注释处断开。"""
    text = "# 标题\n\n```python\n# Run the agent\nagent.invoke({})\n```\n\n## 小节\n"

    lines = heading_lines(text)

    assert [line.title for line in lines] == ["标题", "小节"]
    assert [line.level for line in lines] == [1, 2]


def test_first_title_ignores_a_hash_inside_a_code_fence() -> None:
    """文档标题只认**围栏之外**的一级标题 —— 不能拿 `^# ` 直接搜（那会取到代码注释）。"""
    fenced_first = "```\n# Run the agent\n```\n\n## 只有二级\n"
    real_first = "```\n# 注释\n```\n\n# 真标题\n"

    assert first_title(fenced_first) == "", "没真标题就返回空串，别猜"
    assert first_title(real_first) == "真标题"
