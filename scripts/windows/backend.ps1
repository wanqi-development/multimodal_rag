#requires -Version 5.1
<#
只启动**后端**（`rag serve`）。Windows 版。

用法（仓库根目录，或任何地方）：
    powershell -ExecutionPolicy Bypass -File scripts\windows\backend.ps1
    scripts\windows\backend.cmd                 # 同一个东西，省得打那串参数
    scripts\windows\backend.cmd -DryRun         # 只看它会做什么

顺序：从 .env 读地址/端口（RAG_API_HOST / RAG_API_PORT，缺省 127.0.0.1:8000）
→ 端口被占时问一句 y/n → 起一个新窗口跑 `rag serve`。

为什么是「杀掉再重启」而不是「跳过已经在跑的那个」：跳过的话你面对的可能是**旧代码**
（这个项目上真踩过好几次），还以为自己重启过了。要么干净重来，要么干脆别动。

为什么用 `python -m rag` 而不是 `rag`：可编辑安装装的 `rag.exe` 可能不在 PATH 上
（本项目就遇到过），而 `python -m` 只要这个解释器能 import 到包就行。
**别写回 `-m rag.cli.main`**：`rag/cli/__init__.py` 里 import 了 `rag.cli.main`，runpy
会在执行前打一句 `RuntimeWarning: 'rag.cli.main' found in sys.modules …`，而这个脚本
把 stderr 原样透出去 → PowerShell 渲染成一片红（实测 2026-10-10）。`-m rag` 走的是
`src/rag/__main__.py`，没有这个撞名问题。要指定解释器：`$env:RAG_PYTHON = 'D:\...\python.exe'`。

**前端要等它起来**：那个等待在 `frontend.ps1` 里（探 `/docs`，200 就算起来了）。
分开成两个脚本就是为了这一点 —— 后端要加载 embedding 等重对象，几十秒是常态。
#>
[CmdletBinding()]
param(
    [switch]$DryRun,
    [string]$BackendHost,
    [int]$BackendPort
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path

function Get-EnvValue([string]$name) {
    $file = Join-Path $root '.env'
    if (-not (Test-Path $file)) { return $null }
    foreach ($line in Get-Content -LiteralPath $file -Encoding UTF8) {
        if ($line -match "^\s*$name\s*=\s*(.*)$") { return $Matches[1].Trim().Trim('"') }
    }
    return $null
}

function Get-ListenerPid([int]$port) {
    $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($conn) { return [int]$conn.OwningProcess }
    return 0
}

# 杀完之后等端口真的空出来：进程退出与端口释放之间有一小段，不等的话
# 后面 Start-Process 会直接因为「地址已被占用」失败（然后你就得到两个谜之报错）。
function Wait-PortFree([int]$port, [int]$seconds = 10) {
    for ($i = 0; $i -lt ($seconds * 2); $i++) {
        if ((Get-ListenerPid $port) -le 0) { return $true }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

function Test-Interpreter([string]$exe) {
    # 探测时**必须压住 stderr**：`$ErrorActionPreference='Stop'` 下，原生命令往 stderr 写
    # 会被 PowerShell 5.1 当成终止错误抛出来（实测：import 失败直接把脚本打断）。
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $exe -c 'import rag' 2>&1 | Out-Null
        return $LASTEXITCODE -eq 0
    } finally {
        $ErrorActionPreference = $previous
    }
}

# 后端用哪个解释器/入口：**别假设 shell 里的 `python` 就是装了本项目那个环境**
# （实测：直接 `python -m rag` 会因为 import 不到 rag 一闪就退，窗口都来不及看）。
# 顺序：$env:RAG_PYTHON → PATH 上的 `rag` → 能 import rag 的 `python` → conda 里的各个环境。
function Resolve-Backend {
    $explicit = if ($env:RAG_PYTHON) { $env:RAG_PYTHON } else { Get-EnvValue 'RAG_PYTHON' }
    if ($explicit) {
        return @{ File = $explicit; Base = @('-m', 'rag'); How = 'RAG_PYTHON' }
    }
    foreach ($exe in @('rag.exe', 'rag')) {
        $found = Get-Command $exe -ErrorAction SilentlyContinue
        if ($found) { return @{ File = $found.Source; Base = @(); How = "PATH 上的 $exe" } }
    }
    foreach ($candidate in @('python', 'python3')) {
        $found = Get-Command $candidate -ErrorAction SilentlyContinue
        if (-not $found) { continue }
        if (Test-Interpreter $found.Source) {
            return @{ File = $found.Source; Base = @('-m', 'rag'); How = $candidate }
        }
    }
    $conda = Get-Command conda -ErrorAction SilentlyContinue
    if ($conda) {
        $envs = (& $conda.Source env list --json 2>$null | ConvertFrom-Json).envs
        foreach ($envPath in $envs) {
            $candidate = Join-Path $envPath 'python.exe'
            if ((Test-Path $candidate) -and (Test-Interpreter $candidate)) {
                return @{ File = $candidate; Base = @('-m', 'rag'); How = "conda 环境 $envPath" }
            }
        }
    }
    return $null
}

# 不用 `??`：那是 PowerShell 7 才有的，5.1 会直接语法报错（这个脚本要两边都能跑）
if (-not $BackendPort) {
    $port = Get-EnvValue 'RAG_API_PORT'
    $BackendPort = if ($port) { [int]$port } else { 8000 }
}
if (-not $BackendHost) {
    $host_ = Get-EnvValue 'RAG_API_HOST'
    $BackendHost = if ($host_) { $host_ } else { '127.0.0.1' }
}

$backendEntry = Resolve-Backend
if (-not $backendEntry) {
    Write-Host '✘ 找不到能 import 到 rag 的解释器。' -ForegroundColor Red
    Write-Host '  先 conda activate <装了本项目的环境>，或者显式指定：' -ForegroundColor Yellow
    Write-Host '  $env:RAG_PYTHON = "D:\...\envs\multi_modal_rag\python.exe"' -ForegroundColor Yellow
    exit 1
}

Write-Host "仓库：$root"
Write-Host "后端：http://${BackendHost}:${BackendPort}"
Write-Host "后端入口：$($backendEntry.File) $($backendEntry.Base -join ' ')（$($backendEntry.How)）" -ForegroundColor DarkGray

$occupant = Get-ListenerPid $BackendPort
if ($occupant -gt 0) {
    Write-Host ''
    Write-Host "⚠ 后端端口被占（:$BackendPort，PID $occupant）" -ForegroundColor Yellow
    if ($DryRun) {
        Write-Host '[dry-run] 会问一句「要不要杀掉它再启动」；答 n 就什么都不启动。' -ForegroundColor DarkGray
    } else {
        $answer = Read-Host '杀掉上面的进程再启动？（y = 杀掉并重新启动；n = 放弃启动，什么都不动）[y/n]'
        if ($answer -notin @('y', 'Y', 'yes', 'YES')) {
            Write-Host '已放弃启动。' -ForegroundColor Yellow
            # 3 = 「用户主动放弃」。`start.ps1` 靠它决定**前端也别起了**（0 是"起好了"）
            exit 3
        }
        # 0 / 4 是系统进程，`$PID` 是本脚本自己 —— 端口判断万一出错，别把机器搞坏
        if ($occupant -le 4 -or $occupant -eq $PID) {
            Write-Host "✘ 拒绝结束 PID $occupant（系统进程或本脚本自己），放弃启动。" -ForegroundColor Red
            exit 1
        }
        Stop-Process -Id $occupant -Force -ErrorAction SilentlyContinue
        Write-Host "✔ 已结束 PID $occupant" -ForegroundColor Green
        if (-not (Wait-PortFree $BackendPort)) {
            Write-Host "✘ 端口 $BackendPort 还占着，放弃启动。" -ForegroundColor Red
            exit 1
        }
    }
}

$backendArgs = @($backendEntry.Base) + @('serve', '--host', $BackendHost, '--port', "$BackendPort")
$logDir = Join-Path $root 'data\logs'
$logFile = Join-Path $logDir 'backend-startup.log'
if ($DryRun) {
    Write-Host "[dry-run] 会启动后端：$($backendEntry.File) $($backendArgs -join ' ')  （工作目录 $root）"
    Write-Host "[dry-run] 它的输出会同时进控制台与 $logFile"
    exit 0
}
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# **输出同时进控制台与文件**（2026-10-10）：控制台那份给人看实时，文件那份给
# 「进程秒退、窗口跟着关掉」兜底 —— 实测那次是配置校验失败（`config/dev.yaml` 的
# `ingest.images` 成了空节点），窗口一闪栈就没了，而四个 .log 也一个字都没有：
# 配置坏掉恰好发生在**装日志之前**（`configure_logging` 自己也得先读 settings）。
# 外壳用一个新 powershell：`-NoExit` 让窗口留在原地（秒退也看得见），
# `Tee-Object` 两边都写；命令串走 `-EncodedCommand`（Base64/UTF-16）省掉一层引号转义。
# 参数用**数组字面量 + splatting**（`@argv`）传：PowerShell 5.1 会把一整条命令串里的
# 引号二次拆分（实测：`-c 'import sys; print("x")'` 到子进程变成 `print(x`，直接语法错），
# 而数组的每个元素就是一个参数 —— 路径里有空格也不会被拆开。
$arrayLiteral = '@(' + (($backendArgs | ForEach-Object { "'" + ($_ -replace "'", "''") + "'" }) -join ', ') + ')'
$stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
$inner = @"
`$argv = $arrayLiteral
'==== $stamp 启动：$($backendEntry.File) $($backendArgs -join ' ') ====' | Tee-Object -FilePath '$logFile' -Append
& '$($backendEntry.File)' @argv 2>&1 | Tee-Object -FilePath '$logFile' -Append
Write-Host ''
Write-Host '（后端已退出；这个窗口留着不关，看完关掉即可）'
"@
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($inner))

Start-Process -FilePath 'powershell.exe' `
    -ArgumentList '-NoProfile', '-ExecutionPolicy', 'Bypass', '-NoExit', '-EncodedCommand', $encoded `
    -WorkingDirectory $root | Out-Null
Write-Host '✔ 后端已启动（新窗口）' -ForegroundColor Green
Write-Host "后端接口：   http://${BackendHost}:${BackendPort}/docs"
Write-Host "启动输出：   $logFile（窗口关了也在这里）" -ForegroundColor DarkGray
# **必须显式 exit 0**：不写的话调用方（`start.ps1`）那边 `$LASTEXITCODE` 是 `$null`，
# 而 `$null -ne 0` 在 PowerShell 里为**真** —— 会被当成"后端失败"，前端就不起了（实测踩过）。
exit 0
