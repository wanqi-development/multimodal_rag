"""命令行入口。

**这里刻意不 `from rag.cli.main import app`**（2026-10-10 改）：那样会让
`python -m rag.cli.main` 在 runpy 执行之前就把 `rag.cli.main` 塞进 `sys.modules`，
于是 runpy 打一句
`RuntimeWarning: 'rag.cli.main' found in sys.modules after import of package 'rag.cli',
but prior to execution of 'rag.cli.main'`。启动脚本把子进程 stderr 原样透出去，
PowerShell 便渲染成一整片红字报错（用户实测看到的就是那个）。

绕开的办法有两条，两条都在：正式入口改成 `python -m rag`（见 `src/rag/__main__.py`），
这里再加一个惰性出口，让「谁还敲着老命令」也不会踩到。`from rag.cli import app`
照旧可用 —— 只是推迟到真正取 `app` 的那一刻才 import。
"""

__all__ = ["app"]


def __getattr__(name: str) -> object:
    """惰性 re-export：只有真的读 `rag.cli.app` 时才 import 那个模块。"""
    if name == "app":
        from rag.cli.main import app

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
