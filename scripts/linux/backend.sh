#!/usr/bin/env bash
# 只启动**后端**（`rag serve`）。Linux / macOS 版。
#
# 用法：
#   ./scripts/linux/backend.sh
#   ./scripts/linux/backend.sh --dry-run     # 只看它会做什么
#
# 顺序：从 .env 读地址/端口（RAG_API_HOST / RAG_API_PORT，缺省 127.0.0.1:8000）
#   → 端口被占时问一句 y/n → nohup 起 `rag serve`，日志落 data/logs/、PID 记 data/run/。
#
# 为什么是「杀掉再重启」而不是「跳过已经在跑的那个」：跳过的话你面对的可能是**旧代码**
# （这个项目上真踩过好几次），还以为自己重启过了。要么干净重来，要么干脆别动。
#
# 端口检测用 python 的 socket：这台机器本来就要跑 Python 后端，不额外依赖 ss/lsof/nc。
#
# **前端要等它起来**：那个等待在 `frontend.sh` 里（探 `/docs`，200 就算起来了）。

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="$ROOT/data/run"
LOG_DIR="$ROOT/data/logs"

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        *) echo "未知参数：$arg"; exit 2 ;;
    esac
done

env_value() {  # $1 = 键名
    [ -f "$ROOT/.env" ] || return 0
    sed -n "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*//p" "$ROOT/.env" | head -n 1 | tr -d '"'
}

BACKEND_HOST="${RAG_API_HOST_OVERRIDE:-$(env_value RAG_API_HOST)}"
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${RAG_API_PORT_OVERRIDE:-$(env_value RAG_API_PORT)}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
PYTHON="${RAG_PYTHON:-python}"

# 后端用哪个解释器/入口：**别假设 `python` 就是装了本项目那个环境**（实测会 import 不到 rag，
# 进程一闪就退）。顺序：$RAG_PYTHON → PATH 上的 `rag` → 能 import rag 的 python → conda 环境。
# 入口一律 `-m rag`（不是 `-m rag.cli.main`）：后者会因为 `rag/cli/__init__.py` 已经 import
# 过 `rag.cli.main` 而让 runpy 打一句 RuntimeWarning，启动脚本把 stderr 透出去就被当成报错。
EXPLICIT_PYTHON="${RAG_PYTHON:-$(env_value RAG_PYTHON)}"
BACKEND_FILE=""
BACKEND_BASE=()
BACKEND_HOW=""
if [ -n "$EXPLICIT_PYTHON" ]; then
    BACKEND_FILE="$EXPLICIT_PYTHON"; BACKEND_BASE=(-m rag); BACKEND_HOW='RAG_PYTHON'
elif command -v rag >/dev/null 2>&1; then
    BACKEND_FILE="$(command -v rag)"; BACKEND_BASE=(); BACKEND_HOW='PATH 上的 rag'
else
    for candidate in python3 python; do
        command -v "$candidate" >/dev/null 2>&1 || continue
        if "$candidate" -c 'import rag' >/dev/null 2>&1; then
            BACKEND_FILE="$candidate"; BACKEND_BASE=(-m rag); BACKEND_HOW="$candidate"; break
        fi
    done
fi
if [ -z "$BACKEND_FILE" ] && command -v conda >/dev/null 2>&1; then
    for env_path in $(conda env list --json 2>/dev/null | python -c 'import json,sys; print(" ".join(json.load(sys.stdin).get("envs", [])))' 2>/dev/null); do
        candidate="$env_path/bin/python"
        [ -x "$candidate" ] || continue
        if "$candidate" -c 'import rag' >/dev/null 2>&1; then
            BACKEND_FILE="$candidate"; BACKEND_BASE=(-m rag); BACKEND_HOW="conda 环境 $env_path"; break
        fi
    done
fi
if [ -z "$BACKEND_FILE" ]; then
    echo "✘ 找不到能 import 到 rag 的解释器。"
    echo "  先 conda activate <装了本项目的环境>，或者显式指定：RAG_PYTHON=/path/to/python $0"
    exit 1
fi

port_pid() {  # 端口被谁占着 -> 打印 pid；没占着 -> 空
    "$PYTHON" - "$1" <<'PY'
import socket, sys, subprocess
port = int(sys.argv[1])
s = socket.socket(); s.settimeout(1)
try:
    s.connect(("127.0.0.1", port))
except OSError:
    print(""); raise SystemExit
finally:
    s.close()
for cmd in (f"lsof -tiTCP:{port} -sTCP:LISTEN 2>/dev/null | head -n1",
            f"ss -ltnpH 'sport = :{port}' 2>/dev/null | sed -n 's/.*pid=\\([0-9]*\\).*/\\1/p' | head -n1"):
    try:
        out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True).stdout.strip()
    except Exception:
        out = ""
    if out:
        print(out); raise SystemExit
print("?")
PY
}

normalize_answer() {  # 去掉空白与 CR（管道喂 "y\r"、或用户粘 "y " 都能认）
    printf '%s' "$1" | tr -d '[:space:]' | tr 'A-Z' 'a-z'
}

BACKEND_PID="$(port_pid "$BACKEND_PORT")"

echo "仓库：$ROOT"
echo "后端：http://$BACKEND_HOST:$BACKEND_PORT"
echo "后端入口：$BACKEND_FILE ${BACKEND_BASE[*]}（$BACKEND_HOW）"

if [ -n "$BACKEND_PID" ]; then
    echo
    echo "⚠ 后端端口被占（:$BACKEND_PORT，PID $BACKEND_PID）"
    if [ "$DRY_RUN" = 1 ]; then
        echo "[dry-run] 会问一句「要不要杀掉上面的进程再启动」；答 n 就什么都不启动。"
    else
        read -r -p "杀掉上面的进程再启动？（y = 杀掉并重新启动；n = 放弃启动，什么都不动）[y/n] " answer
        case "$(normalize_answer "$answer")" in
            y|yes) ;;
            *) echo "已放弃启动。"; exit 3 ;;   # 3 = 用户主动放弃；start.sh 靠它决定前端也别起
        esac
        if [ "$BACKEND_PID" = "?" ]; then
            # 端口占着但认不出是谁（镜像里既没有 lsof 也没有 ss）—— 不能瞎杀
            echo "✘ 端口被占但拿不到 PID（缺 lsof/ss），请手动处理后重试。"
            exit 1
        fi
        kill "$BACKEND_PID" 2>/dev/null || true
        for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
            [ -z "$(port_pid "$BACKEND_PORT")" ] && break
            sleep 0.5
        done
        if [ -n "$(port_pid "$BACKEND_PORT")" ]; then
            echo "✘ 端口 $BACKEND_PORT 还占着，放弃启动。"
            exit 1
        fi
    fi
fi

if [ "$DRY_RUN" = 1 ]; then
    echo "[dry-run] 会启动后端：$BACKEND_FILE ${BACKEND_BASE[*]} serve --host $BACKEND_HOST --port $BACKEND_PORT"
    exit 0
fi

mkdir -p "$RUN_DIR" "$LOG_DIR"
( cd "$ROOT" && nohup "$BACKEND_FILE" "${BACKEND_BASE[@]}" serve --host "$BACKEND_HOST" --port "$BACKEND_PORT" \
    >"$LOG_DIR/backend.out.log" 2>&1 & echo $! >"$RUN_DIR/backend.pid" )
echo "✔ 后端已启动（PID $(cat "$RUN_DIR/backend.pid")，日志 $LOG_DIR/backend.out.log）"
echo "后端接口：   http://$BACKEND_HOST:$BACKEND_PORT/docs"
# 显式收尾：让调用方（`start.sh`）拿到确定的退出码，别靠"最后一条命令恰好是 echo"
exit 0
