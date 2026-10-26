"""CLI 启动：**重依赖不许在模块级 import**。

实测（`-X importtime`，2026-10-05）：模块级全量 import 要 12.0 秒，其中
`langchain_text_splitters` → `sentence_transformers` → `torch` 一条链 5.5 秒、
`rag.indexing.commit` 那条 9.2 秒。于是 `rag --version` / `rag --help` / `rag sessions list`
这些**根本用不到模型与切分器**的命令也要先付这笔钱。把重依赖下沉到命令体之后：
import 0.4 秒、`rag sessions list` 24.6 秒 → 1.1 秒。

这条用例在**子进程**里验 —— 本进程早就把那些库 import 过了，`sys.modules` 是脏的。
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# 只要 CLI 一被 import，这些就不该出现在 sys.modules 里
FORBIDDEN = (
    "torch",
    "transformers",
    "sentence_transformers",
    "langchain_text_splitters",
    "langgraph",
    "pymilvus",
    "pymupdf",
    "deepagents",
)


def test_importing_the_cli_does_not_pull_the_heavy_stack() -> None:
    script = (
        "import sys\n"
        "import rag.cli.main\n"
        f"bad = [name for name in {FORBIDDEN!r} if name in sys.modules]\n"
        "print(','.join(bad))\n"
        "raise SystemExit(1 if bad else 0)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )

    assert result.returncode == 0, (
        f"`rag --help` 不该把这几件拉起来，实测拉起来的是：{result.stdout.strip()}"
    )


@pytest.mark.parametrize("module", ["rag", "rag.cli.main"])
def test_running_the_cli_module_does_not_warn_about_runpy(module: str) -> None:
    """`python -m rag` 不许往 stderr 打 runpy 的 RuntimeWarning（2026-10-10）。

    启动脚本（`scripts/**/backend.*`）把子进程的 stderr 原样透出来，PowerShell 会把
    它渲染成一整片红字报错 —— 用户看到的就是那个。根因是 `rag/cli/__init__.py` 里
    import 了 `rag.cli.main`，而入口当时写的是 `-m rag.cli.main`：runpy 在执行前发现
    同名模块已在 `sys.modules` 里，于是抱怨一句。

    两条路都钉住：正式入口 `-m rag`（走 `src/rag/__main__.py`，天然不撞名），
    以及老的 `-m rag.cli.main`（靠 `rag/cli/__init__.py` 的惰性出口绕开）。
    """
    result = subprocess.run(
        [sys.executable, "-m", module, "--version"], capture_output=True, text=True, timeout=120
    )

    assert result.returncode == 0, result.stderr
    assert "RuntimeWarning" not in result.stderr, result.stderr
    assert "sys.modules" not in result.stderr, result.stderr
    assert result.stdout.strip(), "`--version` 应当有输出"
