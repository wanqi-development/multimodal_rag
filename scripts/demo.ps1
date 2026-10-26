#requires -Version 5.1
<#
60 秒 demo：把 demo/corpus 灌进一个**独立** collection，再问几个问题。Windows 版。

用法（在仓库根目录）：
    scripts\demo.cmd                 # 灌 + 问三个问题 + 一个该被拒答的
    scripts\demo.cmd -DryRun         # 只打印要做什么，一个字都不写
    scripts\demo.cmd -Reset          # 先删掉 demo collection 再重灌（清干净重来）

**它不花钱、也不碰你现有的数据**：

  * embedding 与重排都是**本地**模型 —— config/demo.yaml 里把值写死了，不吃 .env 的插值，
    所以不会因为本机 .env 指向线上（dev 档就是 qwen3-vl，按量计费）而悄悄花钱；
  * 检索用的是单独的 collection（rag_demo_qwen3_v1），与你库里的语料互不干扰；
  * 语料产物、上传件、文档注册表全部落在 data/demo/ 下，不与你自己的 data/ 重叠。

前置：Milvus 起着（deploy/docker-compose.yml）；.models/ 下有两个本地模型（见 README）。
问句那一半要一个 chat 供应商的 key（.env 的三件套），与 `rag ask` 的要求一样。

**档位必须走真实环境变量**：`.env` 里写 RAG_PROFILE 不生效（config/demo.yaml 的注释里写了
原因），所以这个脚本自己设 `$env:RAG_PROFILE`。

**这个脚本解析的是「解释器」而不是 `rag` 命令**：它除了跑 CLI，还要塞两小段 Python
（写核对标记、删 collection），而 `rag` 命令不是解释器、执行不了 `-c`。
backend.ps1 那边可以退到 PATH 上的 `rag`，这里不行。
#>
[CmdletBinding()]
param(
    [switch]$DryRun,
    [switch]$Reset
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

function Test-Interpreter([string]$exe) {
    # 探测时**必须压住 stderr**：$ErrorActionPreference='Stop' 下，原生命令往 stderr 写会被
    # PowerShell 5.1 当成终止错误抛出来（backend.ps1 里踩过同一条）。
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $exe -c 'import rag' 2>&1 | Out-Null
        return $LASTEXITCODE -eq 0
    } finally {
        $ErrorActionPreference = $previous
    }
}

function Get-EnvValue([string]$name) {
    $file = Join-Path $root '.env'
    if (-not (Test-Path $file)) { return $null }
    foreach ($line in Get-Content -LiteralPath $file -Encoding UTF8) {
        if ($line -match "^\s*$name\s*=\s*(.*)$") { return $Matches[1].Trim().Trim('"') }
    }
    return $null
}

# 顺序：$env:RAG_PYTHON → PATH 上能 import rag 的 python（**别假设 `python` 就是装了本项目
# 那个** —— backend.ps1 的注释里记着实测：那样会 import 不到 rag、一闪就退）。
function Resolve-Python {
    $explicit = if ($env:RAG_PYTHON) { $env:RAG_PYTHON } else { Get-EnvValue 'RAG_PYTHON' }
    if ($explicit) {
        if (-not (Test-Path $explicit)) { throw "RAG_PYTHON 指的路径不存在：$explicit" }
        if (-not (Test-Interpreter $explicit)) { throw "$explicit 里 import 不到 rag。" }
        return @{ File = $explicit; How = 'RAG_PYTHON' }
    }
    foreach ($candidate in @('python', 'python3')) {
        if ((Get-Command $candidate -ErrorAction SilentlyContinue) -and (Test-Interpreter $candidate)) {
            return @{ File = $candidate; How = "PATH 上的 $candidate" }
        }
    }
    throw ("找不到能 import 到 rag 的解释器。先 conda activate <装了本项目的环境>，" +
           "或者显式指定：`$env:RAG_PYTHON = '<python 路径>'")
}

$py = Resolve-Python
function Invoke-Rag([string[]]$Arguments) { & $py.File -m rag @Arguments }

# 跑一小段 Python：**写成临时文件再执行，不走 `-c`**。
# 实测（2026-10-12）：PowerShell 5.1 把带换行与双引号的参数传给原生命令时，内层双引号会被
# 吃掉 —— `pathlib.Path("data/demo/stages")` 变成 `pathlib.Path(data/demo/stages)`，
# Python 直接 SyntaxError。写文件既没有引号问题，也没有长度限制。
# （Linux 那版用 `-c '...'` 就行：bash 的单引号不做任何解释，见 scripts/linux/demo.sh。）
function Invoke-Py([string]$Code) {
    $temp = Join-Path $env:TEMP 'rag-demo-snippet.py'
    [IO.File]::WriteAllText($temp, $Code, [Text.UTF8Encoding]::new($false))
    try {
        & $py.File $temp
    } finally {
        Remove-Item -LiteralPath $temp -Force -ErrorAction SilentlyContinue
    }
}

# 档位走**真实**环境变量（也传给子进程，所以后面每条命令都吃得到）
$env:RAG_PROFILE = 'demo'

Write-Host "仓库：$root"
Write-Host "档位：demo（config/demo.yaml）"
Write-Host "解释器：$($py.File)（$($py.How)）"
Write-Host

# 只删 demo 那个 collection。断言是安全绳：万一档位没生效，这里必须**当场拒绝**，
# 而不是把你真实语料的 collection 删掉。
$dropCode = @'
from pymilvus import MilvusClient
from rag.config import load_settings

s = load_settings()
name = s.indexing.vector.collection
assert name.endswith("_demo_qwen3_v1"), f"拒绝：{name} 不像 demo collection"
MilvusClient(uri=s.indexing.vector.uri).drop_collection(name)
print("已删除 collection：", name)
'@

# `-Reset` 要清的**两处**，少一处都不算清干净（实测踩到）：
#   ① collection —— 向量与原文都在里面；
#   ② data/demo/ —— 文档注册表（SQLite）在里面，它记着「这篇已入库」。只删 ① 不删 ②，
#      下一次 ingest 的增量判断会「跳过 3 篇」，库还是空的。
function Remove-DemoState {
    $stateDir = Join-Path $root 'data/demo'
    if (-not (Test-Path $stateDir)) { return }
    # 递归删除前先验路径：解析成绝对路径后，必须是 <仓库>\data\demo。认不出就当场拒绝 ——
    # 这里删错一次就是把你真实语料的产物删了。
    $resolved = (Resolve-Path $stateDir).Path
    $leaf = Split-Path $resolved -Leaf
    $parent = Split-Path (Split-Path $resolved -Parent) -Leaf
    if ($leaf -ne 'demo' -or $parent -ne 'data') {
        throw "拒绝删除：$resolved 不是预期的 <仓库>\data\demo"
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
    Write-Host "已删除本地状态：$resolved"
}

# 替人把「已核对」勾上 —— 见下面调用处的说明。
$confirmCode = @'
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
'@

$steps = @(
    @{ Title = '解析 demo 语料（第一阶段：按设计不碰 Milvus）'
       Args = @('ingest', 'demo/corpus', '--stage', 'parse') }
    @{ Title = '切分 + 入库（解析产物复用，只跑后两段；第一次要加载 embedding，慢一点）'
       Args = @('ingest', 'demo/corpus') }
    @{ Title = '检索：默认配额'
       Args = @('search', 'Aurora 网关默认给每个租户多少 QPS？') }
    @{ Title = '问答：事故根因与修复版本（跨文档）'
       Args = @('ask', '2026-08 那次断电事故的根因是什么，哪个版本修的？') }
    @{ Title = 'agent：临时提额的流程'
       Args = @('agent', '客户要临时提额，走什么流程？') }
    @{ Title = '语料外的问题：应当拒答（而不是硬编）'
       Args = @('ask', 'Aurora 网关支持哪些 Kubernetes 版本？') }
)

function Invoke-Step($Step) {
    Write-Host ("─" * 78)
    Write-Host "▶ $($Step.Title)"
    Write-Host "  rag $($Step.Args -join ' ')" -ForegroundColor DarkGray
    Invoke-Rag $Step.Args
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✘ 这一步失败了（退出码 $LASTEXITCODE），后面的就不跑了。" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host
}

if ($DryRun) {
    if ($Reset) { Write-Host "[dry-run] 会先删掉 demo collection（带「名字必须以 _demo_qwen3_v1 结尾」的断言）" }
    Write-Host "[dry-run] 会依次跑："
    foreach ($s in $steps) {
        Write-Host ("  rag {0}    # {1}" -f ($s.Args -join ' '), $s.Title)
    }
    Write-Host "  + 在第一步之后，替人把每份 outline.tsv 的「# 已核对: 否」改成「是」"
    Write-Host "[dry-run] 一个字都没写。去掉 -DryRun 才真跑。"
    exit 0
}

New-Item -ItemType Directory -Force -Path (Join-Path $root 'data/demo') | Out-Null

if ($Reset) {
    Write-Host ("─" * 78)
    Write-Host "▶ 清掉上一次的 demo 数据（collection + data/demo/）"
    Invoke-Py $dropCode
    if ($LASTEXITCODE -ne 0) { Write-Host "✘ 删 collection 失败。" -ForegroundColor Red; exit $LASTEXITCODE }
    Remove-DemoState
    Write-Host
}

Invoke-Step $steps[0]

# —— demo 的捷径：替人把「已核对」勾上 ——
# 这个仓库有意要求「切分之前，标题层级必须有人核对过」（docs/architecture.md §7.6）：清单初值
# 是解析器/OCR 猜的，不核对就切，等于把猜测当事实切进库。demo 语料是我们自己写的、标题本来
# 就是准的，所以这里替人勾这一声；**真语料别这么干**。
# 动的就是清单头部那一行，等价于 rag.ingest.outline.render_reviewed_outline(confirmed=True)
# 内部做的那次字符串替换（层级与正文一个字不动）。
Write-Host ("─" * 78)
Write-Host "▶ 替人核对标题层级（真语料这一步得人来，见 docs/architecture.md §7.6）"
Invoke-Py $confirmCode
if ($LASTEXITCODE -ne 0) { Write-Host "✘ 核对标记没写上。" -ForegroundColor Red; exit $LASTEXITCODE }
Write-Host

foreach ($s in $steps[1..($steps.Count - 1)]) { Invoke-Step $s }

Write-Host ("─" * 78)
Write-Host "✔ demo 跑完了。demo 的数据全在下面这两处，删掉不影响你别的语料："
Write-Host "  * collection：rag_demo_qwen3_v1   → scripts\demo.cmd -Reset"
Write-Host "  * 本地文件：data/demo/            → 直接删目录"
Write-Host "页面版（录屏用）：先 `$env:RAG_PROFILE = 'demo'，再 scripts\start.cmd —— 否则页面是 dev 档，看不到这份语料。"
exit 0
