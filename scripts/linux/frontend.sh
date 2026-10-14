#!/usr/bin/env bash
# 只启动**前端**（Vite 开发服务器）。Linux / macOS 版。
#
# 用法：
#   ./scripts/linux/frontend.sh
#   ./scripts/linux/frontend.sh --dry-run
#   ./scripts/linux/frontend.sh --wait-seconds 300   # 后端启动慢就多等一会儿
#
# **它会先等后端能应答，再起 Vite**（使用者 2026-10-10 定）。不等的话前端先起来、
# 接口全是 502，得手动刷 —— 而后端要加载 embedding 等重对象，几十秒是常态。
#
# 等待判据是 `GET /docs` 返回 200：**只看后端进程能不能服务**。
# 刻意不打 `/health` —— 那个会去探 Milvus / Redis / PG，任何一个挂着都会慢或假失败，
# 而"前端能不能开工"跟那些没关系。

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WEB="$ROOT/web"
RUN_DIR="$ROOT/data/run"
LOG_DIR="$ROOT/data/logs"

DRY_RUN=0
WAIT_SECONDS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --wait-seconds) WAIT_SECONDS="${2:-}"; shift ;;
        *) echo "未知参数：$1"; exit 2 ;;
    esac
    shift
done

env_value() {
    [ -f "$ROOT/.env" ] || return 0
    sed -n "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*//p" "$ROOT/.env" | head -n 1 | tr -d '"'
}

BACKEND_HOST="${RAG_API_HOST_OVERRIDE:-$(env_value RAG_API_HOST)}"
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${RAG_API_PORT_OVERRIDE:-$(env_value RAG_API_PORT)}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${RAG_WEB_PORT:-5173}"
if [ -z "$WAIT_SECONDS" ]; then
    WAIT_SECONDS="$(env_value RAG_BACKEND_WAIT_S)"
fi
WAIT_SECONDS="${WAIT_SECONDS:-120}"
DOCS_URL="http://$BACKEND_HOST:$BACKEND_PORT/docs"

normalize_answer() {
    printf '%s' "$1" | tr -d '[:space:]' | tr 'A-Z' 'a-z'
}

backend_ready() {  # `/docs` 200 就算能服务；连不上/超时一律当"还没好"
    python - "$DOCS_URL" <<'PY' 2>/dev/null
import sys, urllib.error, urllib.request
try:
    with urllib.request.urlopen(sys.argv[1], timeout=3) as resp:
        raise SystemExit(0 if resp.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
}

port_pid() {
    python - "$1" <<'PY'
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

echo "仓库：$ROOT"
echo "前端：http://127.0.0.1:$FRONTEND_PORT   后端：$DOCS_URL"

if [ ! -d "$WEB/node_modules" ]; then
    echo
    echo "⚠ 前端依赖还没装（web/node_modules 不存在）"
    if [ "$DRY_RUN" = 1 ]; then
        echo "[dry-run] 会问一句「现在装吗」；答 y 会执行：cd web && pnpm install"
    else
        read -r -p "现在装吗？（pnpm install）[y/n] " answer
        case "$(normalize_answer "$answer")" in
            y|yes) (cd "$WEB" && pnpm install) || { echo "依赖安装失败，退出。"; exit 1; } ;;
            *) echo "前端装了才能启动。"; exit 1 ;;
        esac
    fi
fi

if [ "$DRY_RUN" = 1 ]; then
    echo "[dry-run] 会先等后端就绪（$DOCS_URL，最多 $WAIT_SECONDS 秒）再起 Vite"
else
    if ! backend_ready; then
        echo
        echo "等待后端就绪（$DOCS_URL，最多 $WAIT_SECONDS 秒）…"
        ready=0
        elapsed=0
        while [ "$elapsed" -lt "$WAIT_SECONDS" ]; do
            sleep 1
            elapsed=$((elapsed + 1))
            if backend_ready; then ready=1; break; fi
            [ $((elapsed % 5)) -eq 0 ] && echo "  …还在起（已等 $elapsed 秒）"
        done
        if [ "$ready" = 0 ]; then
            echo "✘ 等了 $WAIT_SECONDS 秒后端还没应答。先看它的日志：" \
                 "tail -f $LOG_DIR/backend.out.log"
            echo "  或加大等待：$0 --wait-seconds 300"
            exit 1
        fi
    fi
    # 无论"本来就起着"还是"等到了"，都明确说一声
    echo "✔ 后端已就绪"
fi

FRONTEND_PID="$(port_pid "$FRONTEND_PORT")"
if [ -n "$FRONTEND_PID" ]; then
    echo "⚠ 前端端口被占（:$FRONTEND_PORT，PID $FRONTEND_PID）"
    if [ "$DRY_RUN" = 1 ]; then
        echo "[dry-run] 会问一句「要不要杀掉它再启动」；答 n 就什么都不启动。"
    else
        read -r -p "杀掉上面的进程再启动？（y = 杀掉并重新启动；n = 放弃启动）[y/n] " answer
        case "$(normalize_answer "$answer")" in
            y|yes) ;;
            *) echo "已放弃启动。"; exit 0 ;;
        esac
        if [ "$FRONTEND_PID" = "?" ]; then
            echo "✘ 端口被占但拿不到 PID（缺 lsof/ss），请手动处理后重试。"
            exit 1
        fi
        kill "$FRONTEND_PID" 2>/dev/null || true
        for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
            [ -z "$(port_pid "$FRONTEND_PORT")" ] && break
            sleep 0.5
        done
        if [ -n "$(port_pid "$FRONTEND_PORT")" ]; then
            echo "✘ 端口 $FRONTEND_PORT 还占着，放弃启动。"
            exit 1
        fi
    fi
fi

if [ "$DRY_RUN" = 1 ]; then
    echo "[dry-run] 会启动前端：cd web && pnpm dev --port $FRONTEND_PORT"
    exit 0
fi

mkdir -p "$RUN_DIR" "$LOG_DIR"
( cd "$WEB" && nohup pnpm dev --port "$FRONTEND_PORT" \
    >"$LOG_DIR/frontend.out.log" 2>&1 & echo $! >"$RUN_DIR/frontend.pid" )
echo "✔ 前端已启动（PID $(cat "$RUN_DIR/frontend.pid")，日志 $LOG_DIR/frontend.out.log）"
echo "前端（开发）：http://127.0.0.1:$FRONTEND_PORT"
exit 0
