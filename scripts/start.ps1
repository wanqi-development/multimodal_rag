#requires -Version 5.1
<#
一条命令起「后端 + 前端（Vite 开发服务器）」，Windows 版。

用法（在仓库根目录，或在任何地方）：
    scripts\start.cmd                 # 前端 + 后端
    scripts\start.cmd -Backend        # 只起后端（调后端时不必拖一个 Vite 起来）
    scripts\start.cmd -DryRun         # 只看它会做什么
    scripts\start.cmd -WaitSeconds 300

**它现在只做编排**（2026-10-10 拆的，使用者要"前端等后端起好再启动"）：

1. `windows\backend.ps1`  —— 起后端（解释器探测、端口占用 y/n、起 `rag serve` 都在那边）
2. `windows\frontend.ps1` —— **等后端 `/docs` 能应答之后**再起 Vite

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
    [int]$FrontendPort = 5173,
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
    # 用户对「端口被占」答了 n：后端都不起了，前端更不该起
    Write-Host '（后端那一步被放弃，前端也不起了）' -ForegroundColor Yellow
    exit 0
}
# **不能只写 `-ne 0`**：被调脚本没显式 exit 时 `$LASTEXITCODE` 是 `$null`，
# 而 `$null -ne 0` 为真 —— 那会把"正常起好了"误判成失败（实测踩过一次）。
if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Backend) {
    Write-Host '（-Backend：只起后端，跳过前端）' -ForegroundColor DarkGray
    exit 0
}

$frontendArgs = @{ FrontendPort = $FrontendPort }
if ($DryRun) { $frontendArgs['DryRun'] = $true }
if ($BackendHost) { $frontendArgs['BackendHost'] = $BackendHost }
if ($BackendPort) { $frontendArgs['BackendPort'] = $BackendPort }
if ($WaitSeconds) { $frontendArgs['WaitSeconds'] = $WaitSeconds }

& (Join-Path $windows 'frontend.ps1') @frontendArgs
$code = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
exit $code
