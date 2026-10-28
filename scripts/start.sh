#!/usr/bin/env bash
# 一条命令起「后端 + web（Vite 开发服务器）」，Linux / macOS 版。
#
# 用法：
#   ./scripts/start.sh               # web + 后端
#   ./scripts/start.sh --backend     # 只起后端
#   ./scripts/start.sh --dry-run     # 只看它会做什么
#   ./scripts/start.sh --wait-seconds 300
#   ./scripts/start.sh --web-host 0.0.0.0 --web-port 5174
#
# **它现在只做编排**（2026-10-10 拆的，使用者要"web 等后端起好再启动"）：
#
#   1. scripts/linux/backend.sh    —— 起后端（解释器探测、端口 y/n、日志与 PID）
#   2. scripts/linux/web.sh        —— **等后端 `/docs` 能应答之后**再起 Vite
#
# 两份脚本各自都能单独用（开头都写了用法），所以这里不重复它们任何逻辑。

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DRY_RUN=0
BACKEND_ONLY=0
WEB_ARGS=()
BACKEND_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            BACKEND_ARGS+=(--dry-run)
            WEB_ARGS+=(--dry-run)
            ;;
        --backend) BACKEND_ONLY=1 ;;
        --wait-seconds)
            # 只管 web：后端那一步没有"等"这回事
            WEB_ARGS+=(--wait-seconds "${2:-}")
            shift
            ;;
        --web-host)
            WEB_ARGS+=(--host "${2:-}")
            shift
            ;;
        --web-port)
            WEB_ARGS+=(--port "${2:-}")
            shift
            ;;
        *) echo "未知参数：$1"; exit 2 ;;
    esac
    shift
done

"$ROOT/scripts/linux/backend.sh" "${BACKEND_ARGS[@]+"${BACKEND_ARGS[@]}"}"
status=$?
if [ "$status" = 3 ]; then
    # 用户对「端口被占」答了 n：后端都不起了，web 更不该起
    echo "（后端那一步被放弃，web 也不起了）"
    exit 0
fi
if [ "$status" != 0 ]; then
    exit "$status"
fi

if [ "$BACKEND_ONLY" = 1 ]; then
    echo "（--backend：只起后端，跳过 web）"
    exit 0
fi

"$ROOT/scripts/linux/web.sh" "${WEB_ARGS[@]+"${WEB_ARGS[@]}"}"
exit $?
