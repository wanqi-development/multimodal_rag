"""标题层级修正：按作者编号重建树，保证路径从根开始。

用例里的标题序列**取自真实文档**（第二章 Flink 入门的 OCR 输出、Kafka 2.5 文档的
一章、docx 的样式标题），不是编出来的 —— 这条链路上的坑全是真实数据里的形状。
"""

from __future__ import annotations

from rag.ingest.headings import (
    HeadingTree,
    chapter_headings,
    code_fence_mask,
    parse_numbering,
    promote_numbered_headings,
    scan_headings,
)

# ---- 代码围栏里的 `#` 不是标题（2026-10-10）----


def test_a_hash_inside_a_code_fence_is_not_a_heading() -> None:
    """实测来源：Firecrawl 抓的文档，代码样例里的 `# Run the agent`（Python 注释）
    被我们的逐行扫描当成了标题 —— 清单多出假标题、切分会在代码注释处断开。
    """
    text = "# 真标题\n\n```python\n# Run the agent\nagent.invoke({})\n```\n\n## 小节\n"

    assert scan_headings(text) == [(1, "真标题"), (2, "小节")]
    assert code_fence_mask(text) == [
        True,  # # 真标题
        True,
        True,  # 开栏那一行本身算"外面"
        False,  # # Run the agent
        False,
        True,  # 闭栏
        True,
        True,
        True,  # 结尾那个空行（文本以 \n 收尾）
    ]


def test_tilde_fences_and_indented_fences_count_too() -> None:
    """围栏不只有 ``` ——`~~~` 与缩进（最多 3 空格）同样算；闭合要求同种字符且不短于开栏。"""
    text = "~~~\n# 不算\n~~~\n\n   ```\n   # 也不算\n   ````\n\n# 算\n"

    assert scan_headings(text) == [(1, "算")]


def test_a_four_backtick_fence_is_not_closed_by_three() -> None:
    """CommonMark 的规则：开栏四个反引号，三个关不掉 —— 里面的 `#` 仍然不是标题。"""
    text = "````\n# 不算\n```\n# 还是不算\n````\n\n# 算\n"

    assert scan_headings(text) == [(1, "算")]


# ---- 编号识别 ----


def test_only_a_leading_number_counts() -> None:
    assert parse_numbering("2.1 Flink 开发环境准备") == ("2", "1")
    assert parse_numbering("2.2.1 IDEA Project 创建及配置") == ("2", "2", "1")
    assert parse_numbering("2. APIS") == ("2",)
    assert parse_numbering("2．1 全角点也一样") == ("2", "1")
    # 这些都不算编号，否则正文里的数字会被当成章节号
    assert parse_numbering("Step 1: Download the code") is None
    assert parse_numbering(".GETTING STARTED") is None
    assert parse_numbering("• Scala环境") is None
    assert parse_numbering(r"$ \underline{\text{Step 3: Create a topic}} $") is None


def test_a_single_number_counts_as_a_chapter_only_when_it_has_subsections() -> None:
    """`2. APIS` 后面有 `2.1` → 是章；`3. 打开Structure…` 后面没有 `3.x` → 是列表项。

    「编号接在前一个章号后面就算章」这条更直觉，但实测会翻车：那份教材的列表步骤
    正好是 1./2./3./4./5.，会把 `3. 打开Structure` 判成第 3 章。
    """
    headings = scan_headings(
        "# Documentation\n"
        "## 2. APIS\n"
        "### 2.1 Producer API\n"
        "## 3. 打开Structure, 创建项目新模块\n"
        "## 创建Java模块：\n"
    )

    chapters = chapter_headings(headings)

    assert "2. APIS" in chapters
    assert "3. 打开Structure, 创建项目新模块" not in chapters


def test_scan_headings_sees_every_level_and_strips_trailing_hashes() -> None:
    """六级都扫（清单要一行一个标题行），行尾的 `#` 是 markdown 语法要去掉。

    「切到第几级」不在这里管 —— 那是 `headings.MAX_LEVEL`，在 `sections` 里生效。
    """
    headings = scan_headings("# 一\n\n### 2.2 小节 ##\n\n#### 2.2.1 更深的一层\n")

    assert headings == [(1, "一"), (3, "2.2 小节"), (4, "2.2.1 更深的一层")]


# ---- 真实的第二章 Flink 入门（OCR 输出，章/节/列表项被标成同一级） ----

_FLINK_CH2 = [
    (2, "2 第二章 Apache Flink 快速入门"),
    (3, "2.1 Flink 开发环境准备"),
    (2, "Flink版本"),
    (2, "JDK环境"),
    (3, "2.2 Flink 入门案例"),
    (4, "2.2.1 IDEA Project 创建及配置"),
    (3, "1. 打开IDEA，创建空项目"),
    (3, "2. 在IntelliJ IDEA 中安装Scala插件"),
    (2, "创建Java模块："),
    (3, "2.3 Flink 案例实现"),
]


def _paths(headings: list[tuple[int, str]]) -> list[tuple[str, str]]:
    tree = HeadingTree(chapters=chapter_headings(headings))
    return [tree.push(level, title) for level, title in headings]


def test_flink_chapter_paths_all_start_from_the_root() -> None:
    """修之前 `Flink版本` 的路径就是它自己（父级被同级标题挤掉了）。"""
    paths = _paths(_FLINK_CH2)

    assert paths[0] == ("2 第二章 Apache Flink 快速入门", "2")
    assert paths[1] == ("2 第二章 Apache Flink 快速入门 -> 2.1 Flink 开发环境准备", "2-1")
    # 没有编号的标题挂在**当前节点之下**，而不是把父级顶掉
    assert paths[2] == (
        "2 第二章 Apache Flink 快速入门 -> 2.1 Flink 开发环境准备 -> Flink版本",
        "2-1-1",
    )
    assert paths[3][1] == "2-1-2", "JDK环境 是 Flink版本 的兄弟"
    # 编号 2.2 回到节一级，2.2.1 是它的小节
    assert paths[4] == ("2 第二章 Apache Flink 快速入门 -> 2.2 Flink 入门案例", "2-2")
    assert paths[5][0].endswith("2.2 Flink 入门案例 -> 2.2.1 IDEA Project 创建及配置")
    assert paths[5][1] == "2-2-1"
    # `1. 打开IDEA` 是列表项不是章：编号与当前编号体系不兼容，按位置挂到 2.2.1 之下
    assert paths[6][0].endswith("-> 2.2.1 IDEA Project 创建及配置 -> 1. 打开IDEA，创建空项目")
    assert paths[6][1] == "2-2-1-1"


def test_every_flink_path_starts_from_the_first_heading() -> None:
    """**这就是用户提的那个问题的判据**：每条路径都必须以根标题开头。"""
    root = _FLINK_CH2[0][1]

    for path, _ in _paths(_FLINK_CH2):
        assert path.split(" -> ")[0] == root


# ---- 真实的 Kafka 2.5 文档（章节标题被 OCR 挂到了别的节下面） ----


def test_kafka_chapter_titles_come_back_to_the_root() -> None:
    """库里的路径是 `Documentation -> For a rolling upgrade: -> 2. APIS`（明显错位）。"""
    headings = [
        (1, "Documentation"),
        (2, "Kafka 2.5 Documentation"),
        (2, "For a rolling upgrade:"),
        (3, "2. APIS"),
        (3, "2.1 Producer API"),
        (3, "3. CONFIGURATION"),
        (3, "3.1 Broker Configs"),
        (3, "zookeeper.connect"),
    ]

    paths = _paths(headings)

    assert paths[3] == ("2. APIS", "2"), "单段编号 + 点分体系 = 章，回到根"
    assert paths[4] == ("2. APIS -> 2.1 Producer API", "2-1")
    assert paths[5] == ("3. CONFIGURATION", "3")
    assert paths[6] == ("3. CONFIGURATION -> 3.1 Broker Configs", "3-1")
    assert paths[7][0] == "3. CONFIGURATION -> 3.1 Broker Configs -> zookeeper.connect"


def test_a_missing_parent_gets_a_placeholder_so_the_chain_never_breaks() -> None:
    """文档第一行就是个二级标题（`1.2` 而没有 `1.` 的标题）时，补一个编号占位。

    链不能从中间开始 —— 宁可出现一个只有编号的占位节点，也不要让路径少一层。
    """
    tree = HeadingTree()

    path, numbers = tree.push(2, "1.2 Use Cases")

    assert path == "1 -> 1.2 Use Cases"
    assert numbers == "1-2"


# ---- docx：中文章标题 + 每章重排的「1、2、3」 ----


def test_numbered_sections_under_a_chinese_chapter_stay_sections() -> None:
    """docx 里是「第一章」+「1、什么是Agent？」—— 单段编号不能当成章。

    这里没有点分编号（`uses_dotted_numbering` 为 False），所以单段编号按声明级别走，
    `1、` 是 h2 就老老实实待在第 2 层。
    """
    headings = [
        (1, "第一章、Agent和OpenClaw的相关概念"),
        (2, "1、什么是Agent？"),
        (3, "细节"),
        (2, "2、为什么需要Agent？"),
        (1, "第二章、DeepAgents框架初探"),
    ]

    paths = _paths(headings)

    assert paths[1] == ("第一章、Agent和OpenClaw的相关概念 -> 1、什么是Agent？", "1-1")
    assert paths[3] == ("第一章、Agent和OpenClaw的相关概念 -> 2、为什么需要Agent？", "1-2")
    # 换章回到第 1 层：路径只剩章标题，编号是「第 2 章」（位置数出来的，与作者写法一致）
    assert paths[4] == ("第二章、DeepAgents框架初探", "2")


def test_a_normal_markdown_document_is_untouched() -> None:
    """层级本来就可靠的普通 md：修法不该把它改坏。"""
    headings = [(1, "检索"), (2, "向量检索"), (3, "原理"), (2, "关键词检索")]

    paths = _paths(headings)

    assert paths[0] == ("检索", "1")
    assert paths[1] == ("检索 -> 向量检索", "1-1")
    assert paths[2] == ("检索 -> 向量检索 -> 原理", "1-1-1")
    assert paths[3] == ("检索 -> 关键词检索", "1-2")


# ---- 编号兜底：漏标的那串编号行（2026-10-08）----

# 形状取自 `第一章 Apache Flink 概述.pdf`：1./2./3. 是裸段落、`4.` 起才是标题，
# 而 4./5. 后面同级还接着走 —— 三条判据齐了才补。
_FLINK = """## 1.3 Flink核心特性

Flink具有先进的架构理念。

1. 批流一体化

Flink可以在底层用同样的数据抽象。

2. 同时支持高吞吐、低延迟、高性能

Flink是目前开源社区中唯一一套。

3. 支持事件时间(Event Time)概念

在流式计算领域中。

## 4. 支持有状态计算

Flink在1.4版本中实现了状态管理。

## 5. 支持高度灵活的窗口操作

在流处理应用中。

## 1.4 Flink应用场景

正文。
"""


def test_a_missed_number_run_gets_its_heading_marks_back() -> None:
    fixed = promote_numbered_headings(_FLINK)

    assert "## 1. 批流一体化" in fixed
    assert "## 2. 同时支持高吞吐、低延迟、高性能" in fixed
    assert "## 3. 支持事件时间(Event Time)概念" in fixed
    # 只往行首插 `#`：**一行都不增删**，正文一个字不改
    assert len(fixed.split("\n")) == len(_FLINK.split("\n"))
    assert (
        fixed.replace("## 1. 批流", "1. 批流")
        .replace("## 2. 同时", "2. 同时")
        .replace("## 3. 支持事件", "3. 支持事件")
        == _FLINK
    )
    # 幂等：补过一遍之后再跑一遍不动它
    assert promote_numbered_headings(fixed) == fixed


def test_an_already_headed_run_is_left_alone() -> None:
    """本来就标对了的（1.4 的 `### 1.` 那串）：不该被改成别的级别。"""
    text = (
        "## 1.4 Flink应用场景\n\n### 1. 实时智能推荐\n\n正文。\n\n### 2. 复杂事件处理\n\n正文。\n"
    )

    assert promote_numbered_headings(text) == text


def test_a_command_run_is_not_promoted() -> None:
    """Kafka 快速开始：步骤是 `### Step 1:` 底下的命令列表，一个都不该动。

    实测全库 318 行这种形状，靠「后面跟的标题不是编号标题」挡住。
    """
    text = (
        "### Step 1: Download the code\n\n"
        "1. `tar -xzf kafka_2.12-2.5.0.tgz`\n\n"
        "2. `cd kafka_2.12-2.5.0`\n\n"
        "### Step 2: Start the server\n\n"
        "1. `bin/zookeeper-server-start.sh config/zookeeper.properties`\n\n"
        "2. `bin/kafka-server-start.sh config/server.properties`\n"
    )

    assert promote_numbered_headings(text) == text


def test_a_number_run_whose_next_section_does_not_continue_is_left_alone() -> None:
    """编号碰巧接得上、**可那个标题自己不在连续编号的同级标题串里** —— 不补。

    这是拿 Kafka 那份文档量出来的反例（2026-10-08）：`1. sudo kadmin…` 后面紧挨着
    `## 3. Make sure all hosts…`，但它们其实是 `## 2.` 底下的操作步骤，
    再往后并没有 `## 4.`（下一节跳回 `## 2. Configuring Kafka Brokers`）。
    """
    text = (
        "## 2. Create Kerberos Principals\n\n"
        "If you have installed your own Kerberos.\n\n"
        "1. `sudo /usr/sbin/kadmin.local -q 'addprinc -randkey kafka/host@REALM'`\n\n"
        '2. `sudo /usr/sbin/kadmin.local -q "ktadd -k /etc/security/keytabs"`\n\n'
        "## 3. Make sure all hosts can be reachable\n\n"
        "正文。\n\n"
        "## 2. Configuring Kafka Brokers\n\n"
        "正文。\n"
    )

    assert promote_numbered_headings(text) == text
