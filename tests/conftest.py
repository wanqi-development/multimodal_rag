"""全仓库共用的夹具。

**跑测试不许往仓库里的 `data/logs/*.log` 写**（2026-10-10）：CLI / serve 那几条用例会
真的把日志系统装起来，而 `logging` 的 handler 是**进程级**的 —— 装一次之后，后面所有
用例（TestClient 的请求行、假 OCR 的告警、第三方库的 WARNING）都会落进真实日志文件。
实测 `data/logs/system.log` 里就混着 `http://ocr.test:8080`（测试用的假地址）、
`interface.log` 里混着 `ip=testclient` —— 那几份文件是给人查现场的，被搅浑之后
「这次跑到底发生了什么」就得先做一遍排除法。

所以这里把落点改到 tmp，并关掉日志索引那份 PG 写入（用例不该往真库写东西）。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import rag.obs


@pytest.fixture(autouse=True)
def _logs_go_to_tmp_not_the_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = rag.obs.configure_logging
    log_dir = tmp_path / "logs"

    def redirected(settings: object) -> None:
        cfg = getattr(getattr(settings, "observability", None), "logging", None)
        if cfg is not None:
            # 只换目录，文件名照旧（interface.log / agent.log / channel.log / system.log）
            for file_cfg in (cfg.files or {}).values():
                name = Path(str(getattr(file_cfg, "path", "") or "")).name
                file_cfg.path = str(log_dir / name) if name else ""
            system = getattr(cfg, "system", None)
            if system is not None and str(getattr(system, "path", "") or ""):
                system.path = str(log_dir / Path(system.path).name)
            cfg.postgres = False  # `log_line` 那份索引：别让用例往真 PG 写
        real(settings)

    monkeypatch.setattr(rag.obs, "configure_logging", redirected)
    yield
    # 用完就摘干净（不摘的话句柄留到会话结束，Windows 上 tmp 目录都删不掉）
    rag.obs.reset_logging()
    for handler in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(handler)
