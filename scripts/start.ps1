#requires -Version 5.1
<#
一条命令起「后端 + web（Vite 开发服务器）」，Windows 版。

用法（在仓库根目录，或在任何地方）：
    scripts\start.cmd                 # web + 后端
    scripts\start.cmd -Backend        # 只起后端（调后端时不必拖一个 Vite 起来）
    scripts\start.cmd -DryRun         # 只看它会做什么
    scripts\start.cmd -WaitSeconds 300
    scripts\start.cmd -WebHost 0.0.0.0 -WebPort 5174

**它现在只做编排**（2026-10-10 拆的，使用者要"web 等后端起好再启动"）：

1. `windows\backend.ps1`  —— 起后端（解释器探测、端口占用 y/n、起 `rag serve` 都在那边）
2. `windows\web.ps1` —— **等后端 `/docs` 能应答之后**再起 Vite

两份脚本各自都能单独用（开头都写了用法），所以这里不再重复它们任何逻辑。

为什么是「杀掉再重启」而不是「跳过已经在跑的那个」：跳过的话你面对的可能是**旧代码**
（这个项目上真踩过好几次），还以为自己重启过了。要么干净重来，要么干脆别动。
#>
[CmdletBinding()]
param(
    [switch]$DryRun,
    [switch]$Backend,
    [string]$BackendHost,
    [int]$BackendPort,
    [string]$WebHost,
    [int]$WebPort = 0,
    [int]$WaitSeconds = 0
)

$ErrorActionPreference = 'Stop'
$windows = Join-Path $PSScriptRoot 'windows'

$backendArgs = @{}
if ($DryRun) { $backendArgs['DryRun'] = $true }
if ($BackendHost) { $backendArgs['BackendHost'] = $BackendHost }
if ($BackendPort) { $backendArgs['BackendPort'] = $BackendPort }

& (Join-Path $windows 'backend.ps1') @backendArgs
if ($LASTEXITCODE -eq 3) {
    # 用户对「端口被占」答了 n：后端都不起了，web 更不该起
    Write-Host '（后端那一步被放弃，web 也不起了）' -ForegroundColor Yellow
    exit 0
}
# **不能只写 `-ne 0`**：被调脚本没显式 exit 时 `$LASTEXITCODE` 是 `$null`，
# 而 `$null -ne 0` 为真 —— 那会把"正常起好了"误判成失败（实测踩过一次）。
if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Backend) {
    Write-Host '（-Backend：只起后端，跳过 web）' -ForegroundColor DarkGray
    exit 0
}

$webArgs = @{}
if ($DryRun) { $webArgs['DryRun'] = $true }
if ($BackendHost) { $webArgs['BackendHost'] = $BackendHost }
if ($BackendPort) { $webArgs['BackendPort'] = $BackendPort }
if ($WebHost) { $webArgs['WebHost'] = $WebHost }
if ($WebPort) { $webArgs['WebPort'] = $WebPort }
if ($WaitSeconds) { $webArgs['WaitSeconds'] = $WaitSeconds }

& (Join-Path $windows 'web.ps1') @webArgs
$code = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
exit $code
