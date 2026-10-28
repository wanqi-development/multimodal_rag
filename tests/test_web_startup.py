"""web 开发端口只有一个配置来源，启动脚本不能再各说各话。"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_vite_reads_the_shared_web_binding() -> None:
    config = (ROOT / "web" / "vite.config.ts").read_text(encoding="utf-8")

    assert "loadEnv(mode, ROOT_ENV_DIR" in config
    assert "process.env.RAG_WEB_HOST" in config
    assert "process.env.RAG_WEB_PORT" in config
    assert "host: webHost" in config
    assert "port: webPort" in config
    assert "strictPort: true" in config


def test_windows_web_passes_the_resolved_port_to_vite() -> None:
    script = (ROOT / "scripts" / "windows" / "web.ps1").read_text(encoding="utf-8")
    backend = (ROOT / "scripts" / "windows" / "backend.ps1").read_text(encoding="utf-8")

    assert "RAG_WEB_HOST" in script
    assert "RAG_WEB_PORT" in script
    assert "$env:RAG_WEB_HOST = $WebHost" in script
    assert '$env:RAG_WEB_PORT = "$WebPort"' in script
    assert "[int]$WebPort = 5173" not in script
    assert "Get-ProfileRuntime" in script
    assert "[switch]$ProfilePaths" in backend
    assert "data\\logs" not in backend
    assert "data\\run" not in backend


def test_windows_orchestrator_does_not_override_the_shared_default() -> None:
    script = (ROOT / "scripts" / "start.ps1").read_text(encoding="utf-8")

    assert "[string]$WebHost" in script
    assert "[int]$WebPort = 0" in script
    assert "$webArgs['WebHost']" in script
    assert "$webArgs['WebPort']" in script
    assert "[int]$WebPort = 5173" not in script


def test_linux_scripts_use_the_same_web_binding() -> None:
    web = (ROOT / "scripts" / "linux" / "web.sh").read_text(encoding="utf-8")
    backend = (ROOT / "scripts" / "linux" / "backend.sh").read_text(encoding="utf-8")
    orchestrator = (ROOT / "scripts" / "start.sh").read_text(encoding="utf-8")

    assert "RAG_WEB_HOST" in web
    assert "RAG_WEB_PORT" in web
    assert 'pnpm dev --host "$WEB_HOST" --port "$WEB_PORT"' in web
    assert "--web-host" in orchestrator
    assert "--web-port" in orchestrator
    assert "--profile-paths" in web
    assert "--profile-paths" in backend
    assert "data/logs" not in backend
    assert "data/run" not in backend


def test_legacy_script_names_are_gone() -> None:
    assert not (ROOT / "scripts" / "windows" / "frontend.ps1").exists()
    assert not (ROOT / "scripts" / "windows" / "frontend.cmd").exists()
    assert not (ROOT / "scripts" / "linux" / "frontend.sh").exists()
