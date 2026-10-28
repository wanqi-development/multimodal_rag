#!/usr/bin/env bash
# 60 秒 demo：把 demo/corpus 灌进一个**独立** collection，再问几个问题。Linux / macOS 版。
#
# 用法（在仓库根目录）：
#   ./scripts/linux/demo.sh              # 灌 + 问
#   ./scripts/linux/demo.sh --dry-run    # 只打印要做什么，一个字都不写
#   ./scripts/linux/demo.sh --reset      # 先删掉 demo collection 再重灌
#
# **不启动 Docker、也不碰你现有的数据**：
#   * embedding 与重排都是本地模型 —— config/demo.yaml 里把值写死了，不吃 .env 的插值，
#     所以不会因为本机 .env 指向线上（dev 档就是 qwen3-vl，按量计费）而悄悄花钱；
#   * 检索用独立的 Milvus Lite 文件 + collection，与你库里的语料互不干扰；
#   * 语料产物、上传件、文档注册表全落在 data/demo/ 下，不与你自己的 data/ 重叠。
#
# 问答与 agent 仍会调用 chat 供应商，是否计费取决于该供应商。
#
# 前置：.models/ 下有两个本地模型（见 README）；demo 使用 local-lite，不需要 Docker。
# 问句那一半要 chat 供应商的 key。
#
# 档位必须走真实环境变量（`.env` 里写 RAG_PROFILE 不生效），所以这里自己 export。
#
# **解析的是「解释器」而不是 `rag` 命令**：除了跑 CLI，还要塞两小段 Python（写核对标记、
# 删 collection），而 `rag` 命令不是解释器、执行不了 `-c`。

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DRY_RUN=0
RESET=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --reset)   RESET=1 ;;
        *) echo "未知参数：$arg"; exit 2 ;;
    esac
done

env_value() {  # $1 = 键名
    [ -f "$ROOT/.env" ] || return 0
    sed -n "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*//p" "$ROOT/.env" | head -n 1 | tr -d '"'
}

# 顺序：$RAG_PYTHON → PATH 上能 import rag 的 python3/python
# （**别假设 `python` 就是装了本项目那个** —— backend.sh 的注释里记着这条实测）。
EXPLICIT_PYTHON="${RAG_PYTHON:-$(env_value RAG_PYTHON)}"
PY=""
PY_HOW=""
if [ -n "$EXPLICIT_PYTHON" ]; then
    if ! "$EXPLICIT_PYTHON" -c 'import rag' >/dev/null 2>&1; then
        echo "✘ RAG_PYTHON 指的 $EXPLICIT_PYTHON 里 import 不到 rag。"; exit 1
    fi
    PY="$EXPLICIT_PYTHON"; PY_HOW='RAG_PYTHON'
else
    for candidate in python3 python; do
        command -v "$candidate" >/dev/null 2>&1 || continue
        if "$candidate" -c 'import rag' >/dev/null 2>&1; then
            PY="$candidate"; PY_HOW="PATH 上的 $candidate"; break
        fi
    done
fi
if [ -z "$PY" ]; then
    echo "✘ 找不到能 import 到 rag 的解释器。"
    echo "  先 conda activate <装了本项目的环境>，或者显式指定：RAG_PYTHON=/path/to/python $0"
    exit 1
fi

# 档位走真实环境变量（也传给子进程，后面每条命令都吃得到）
export RAG_PROFILE=demo

echo "仓库：$ROOT"
echo "档位：demo（config/demo.yaml）"
echo "解释器：$PY（$PY_HOW）"
echo

run_step() {  # $1 = 标题，其余 = rag 的参数
    local title="$1"; shift
    echo "──────────────────────────────────────────────────────────────────────"
    echo "▶ $title"
    echo "  rag $*"
    "$PY" -m rag "$@"
    local code=$?
    if [ "$code" -ne 0 ]; then
        echo "✘ 这一步失败了（退出码 $code），后面的就不跑了。"
        exit "$code"
    fi
    echo
}

if [ "$DRY_RUN" = 1 ]; then
    if [ "$RESET" = 1 ]; then
        echo "[dry-run] 会先删掉 demo collection（带「名字必须以 _demo_qwen3_v1 结尾」的断言）"
    fi
    echo "[dry-run] 会依次跑："
    echo "  rag ingest demo/corpus --stage parse    # 第一阶段：按设计不碰 Milvus"
    echo "  + 替人把每份 outline.tsv 的「# 已核对: 否」改成「是」"
    echo "  rag ingest demo/corpus                  # 切分 + 入库"
    echo "  rag search 'Aurora 网关默认给每个租户多少 QPS？'"
    echo "  rag ask '2026-08 那次断电事故的根因是什么，哪个版本修的？'"
    echo "  rag agent '客户要临时提额，走什么流程？'"
    echo "  rag ask 'Aurora 网关支持哪些 Kubernetes 版本？'   # 应当拒答"
    echo "[dry-run] 一个字都没写。去掉 --dry-run 才真跑。"
    exit 0
fi

mkdir -p "$ROOT/data/demo"

if [ "$RESET" = 1 ]; then
    # `--reset` 要清的**两处**，少一处都不算清干净（实测踩到）：
    #   ① collection —— 向量与原文都在里面；
    #   ② data/demo/ —— 文档注册表（SQLite）在里面，它记着「这篇已入库」。只删 ① 不删 ②，
    #      下一次 ingest 的增量判断会「跳过 3 篇」，库还是空的。
    echo "──────────────────────────────────────────────────────────────────────"
    echo "▶ 清掉上一次的 demo 数据（collection + data/demo/）"
    # 断言是安全绳：档位万一没生效就当场拒绝，而不是把你真实语料的 collection 删掉。
    "$PY" -c '
from pymilvus import MilvusClient
from rag.config import load_settings

s = load_settings()
name = s.indexing.vector.collection
assert name.endswith("_demo_qwen3_v1"), f"拒绝：{name} 不像 demo collection"
MilvusClient(uri=s.indexing.vector.uri).drop_collection(name)
print("已删除 collection：", name)
' || exit 1
    # 递归删除前先验路径：必须是 <仓库>/data/demo。认不出就当场拒绝 ——
    # 这里删错一次就是把你真实语料的产物删了。
    STATE_DIR="$ROOT/data/demo"
    if [ -e "$STATE_DIR" ]; then
        RESOLVED="$(cd "$(dirname "$STATE_DIR")" && pwd)/$(basename "$STATE_DIR")"
        if [ "$(basename "$RESOLVED")" != "demo" ] || [ "$(basename "$(dirname "$RESOLVED")")" != "data" ]; then
            echo "✘ 拒绝删除：$RESOLVED 不是预期的 <仓库>/data/demo"; exit 1
        fi
        rm -rf -- "$RESOLVED"
        echo "已删除本地状态：$RESOLVED"
    fi
    echo
fi

run_step "解析 demo 语料（第一阶段：按设计不碰 Milvus）" ingest demo/corpus --stage parse

# —— demo 的捷径：替人把「已核对」勾上 ——
# 这个仓库有意要求「切分之前，标题层级必须有人核对过」（docs/architecture.md §7.6）：清单初值
# 是解析器/OCR 猜的，不核对就切等于把猜测当事实切进库。demo 语料是我们自己写的、标题本来就准，
# 所以这里替人勾一声；**真语料别这么干**。动的只是头部那一行，等价于
# rag.ingest.outline.render_reviewed_outline(confirmed=True) 内部那次替换。
echo "──────────────────────────────────────────────────────────────────────"
echo "▶ 替人核对标题层级（真语料这一步得人来，见 docs/architecture.md §7.6）"
"$PY" -c '
import pathlib, sys

count = 0
for path in sorted(pathlib.Path("data/demo/stages").glob("*/outline.tsv")):
    text = path.read_text(encoding="utf-8")
    if "# 已核对: 是" in text:
        continue
    updated = text.replace("# 已核对: 否", "# 已核对: 是", 1)
    if updated == text:
        sys.exit(f"{path} 里既没有「否」也没有「是」—— 清单格式变了？")
    path.write_text(updated, encoding="utf-8", newline="")
    count += 1
print(f"替人核对 {count} 份文档的标题层级（demo 的捷径：这份语料是我们自己写的）")
' || exit 1
echo

run_step "切分 + 入库（解析产物复用，只跑后两段；第一次要加载 embedding，慢一点）" ingest demo/corpus
run_step "检索：默认配额" search "Aurora 网关默认给每个租户多少 QPS？"
run_step "问答：事故根因与修复版本（跨文档）" ask "2026-08 那次断电事故的根因是什么，哪个版本修的？"
run_step "agent：临时提额的流程" agent "客户要临时提额，走什么流程？"
run_step "语料外的问题：应当拒答（而不是硬编）" ask "Aurora 网关支持哪些 Kubernetes 版本？"

echo "──────────────────────────────────────────────────────────────────────"
echo "✔ demo 跑完了。demo 的数据全在下面这两处，删掉不影响你别的语料："
echo "  * collection：rag_demo_qwen3_v1   → ./scripts/linux/demo.sh --reset"
echo "  * 本地文件：data/demo/            → 直接删目录"
echo "页面版（录屏用）：RAG_PROFILE=demo ./scripts/start.sh —— 否则页面是 dev 档，看不到这份语料。"
exit 0
