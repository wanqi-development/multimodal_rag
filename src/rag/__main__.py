"""`python -m rag` 的入口（2026-10-10 加）。

为什么有这一个文件：启动脚本原先跑的是 `python -m rag.cli.main`，而
`rag/cli/__init__.py` 里恰好写着 `from rag.cli.main import app` —— 于是 runpy 在
**执行之前**就发现 `rag.cli.main` 已经躺在 `sys.modules` 里，往 stderr 打一句
`RuntimeWarning: 'rag.cli.main' found in sys.modules after import of package 'rag.cli',
but prior to execution of 'rag.cli.main'`。启动脚本把子进程的 stderr 原样透出来，
PowerShell 便把它当错误渲染成一片红（实测 2026-10-10，用户看到的就是那个窗口）。

`-m rag` 的执行目标是本文件（模块名 `rag.__main__`），与任何被 import 的模块都不撞名，
警告自然没有。命令本身一个字没变：`python -m rag serve --host … --port …`。

注意别把这里写成 `from rag.cli import app` 再 `app()` 之外的东西 —— 入口只做这一件事，
真正的命令都在 `rag/cli/main.py`。
"""

from rag.cli.main import app

if __name__ == "__main__":
    app()
