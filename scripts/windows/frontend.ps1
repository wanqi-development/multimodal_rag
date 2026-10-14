#requires -Version 5.1
<#
只启动**前端**（Vite 开发服务器）。Windows 版。

用法：
    scripts\windows\frontend.cmd
    scripts\windows\frontend.cmd -DryRun
    scripts\windows\frontend.cmd -WaitSeconds 300   # 后端启动慢就多等一会儿

**它会先等后端能应答，再起 Vite**（使用者 2026-10-10 定）。不等的话前端先起来、
接口全是 502，得手动刷 —— 而后端要加载 embedding 等重对象，几十秒是常态。

等待判据是 `GET /docs` 返回 200：**只看后端进程能不能服务**。
刻意不打 `/health` —— 那个会去探 Milvus / Redis / PG，任何一个挂着都会慢或假失败，
而"前端能不能开工"跟那些没关系。
#>
[CmdletBinding()]
param(
    [switch]$DryRun,
    [string]$BackendHost,
    [int]$BackendPort,
    [int]$FrontendPort = 5173,
    [int]$WaitSeconds = 0
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$web = Join-Path $root 'web'

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

function Wait-PortFree([int]$port, [int]$seconds = 10) {
    for ($i = 0; $i -lt ($seconds * 2); $i++) {
        if ((Get-ListenerPid $port) -le 0) { return $true }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

# 后端能应答了吗（`/docs` 200 就算）。失败一律当"还没好" —— 连不上、超时、502 都是。
function Test-BackendReady([string]$url) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $response = Invoke-WebRequest -Uri $url -TimeoutSec 3 -UseBasicParsing
        return $response.StatusCode -eq 200
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $previous
    }
}

if (-not $BackendPort) {
    $port = Get-EnvValue 'RAG_API_PORT'
    $BackendPort = if ($port) { [int]$port } else { 8000 }
}
if (-not $BackendHost) {
    $host_ = Get-EnvValue 'RAG_API_HOST'
    $BackendHost = if ($host_) { $host_ } else { '127.0.0.1' }
}
if (-not $WaitSeconds) {
    $configured = Get-EnvValue 'RAG_BACKEND_WAIT_S'
    $WaitSeconds = if ($configured) { [int]$configured } else { 120 }
}

$pnpm = if (Get-Command pnpm.cmd -ErrorAction SilentlyContinue) { 'pnpm.cmd' } else { 'npm.cmd' }
$docsUrl = "http://${BackendHost}:${BackendPort}/docs"

Write-Host "仓库：$root"
Write-Host "前端：http://127.0.0.1:${FrontendPort}   后端：$docsUrl"

if (-not (Test-Path (Join-Path $web 'node_modules'))) {
    Write-Host ''
    Write-Host "⚠ 前端依赖还没装（web\node_modules 不存在）" -ForegroundColor Yellow
    if ($DryRun) {
        Write-Host "[dry-run] 会问一句「现在装吗」；答 y 会执行：cd web; $pnpm install" -ForegroundColor DarkGray
    } else {
        $answer = Read-Host "现在装吗？（$pnpm install）[y/n]"
        if ($answer -in @('y', 'Y', 'yes', 'YES')) {
            Push-Location $web
            try { & cmd /c "$pnpm install" } finally { Pop-Location }
            if ($LASTEXITCODE -ne 0) { Write-Host '依赖安装失败，退出。' -ForegroundColor Red; exit 1 }
        } else {
            Write-Host '前端装了才能启动。' -ForegroundColor Yellow
            exit 1
        }
    }
}

if ($DryRun) {
    Write-Host "[dry-run] 会先等后端就绪（$docsUrl，最多 $WaitSeconds 秒）再起 Vite" -ForegroundColor DarkGray
} else {
    if (-not (Test-BackendReady $docsUrl)) {
        Write-Host ''
        Write-Host "等待后端就绪（$docsUrl，最多 $WaitSeconds 秒）…" -ForegroundColor DarkGray
        $deadline = (Get-Date).AddSeconds($WaitSeconds)
        $ready = $false
        while ((Get-Date) -lt $deadline) {
            Start-Sleep -Seconds 1
            if (Test-BackendReady $docsUrl) { $ready = $true; break }
            $left = [int]($deadline - (Get-Date)).TotalSeconds
            if ($left % 5 -eq 0) { Write-Host "  …还在起（剩 $left 秒）" -ForegroundColor DarkGray }
        }
        if (-not $ready) {
            Write-Host "✘ 等了 $WaitSeconds 秒后端还没应答。先看它的窗口有没有报错，" -ForegroundColor Red
            Write-Host '  或加大等待：scripts\windows\frontend.cmd -WaitSeconds 300' -ForegroundColor Yellow
            exit 1
        }
    }
    # 无论"本来就起着"还是"等到了"，都明确说一声 —— 否则用户不知道它在等什么
    Write-Host '✔ 后端已就绪' -ForegroundColor Green
}

$frontendPid = Get-ListenerPid $FrontendPort
if ($frontendPid -gt 0) {
    Write-Host "⚠ 前端端口被占（:$FrontendPort，PID $frontendPid）" -ForegroundColor Yellow
    if ($DryRun) {
        Write-Host '[dry-run] 会问一句「要不要杀掉它再启动」；答 n 就什么都不启动。' -ForegroundColor DarkGray
    } else {
        $answer = Read-Host '杀掉上面的进程再启动？（y = 杀掉并重新启动；n = 放弃启动）[y/n]'
        if ($answer -notin @('y', 'Y', 'yes', 'YES')) {
            Write-Host '已放弃启动。' -ForegroundColor Yellow
            exit 0
        }
        if ($frontendPid -le 4 -or $frontendPid -eq $PID) {
            Write-Host "✘ 拒绝结束 PID $frontendPid（系统进程或本脚本自己），放弃启动。" -ForegroundColor Red
            exit 1
        }
        Stop-Process -Id $frontendPid -Force -ErrorAction SilentlyContinue
        Write-Host "✔ 已结束 PID $frontendPid" -ForegroundColor Green
        if (-not (Wait-PortFree $FrontendPort)) {
            Write-Host "✘ 端口 $FrontendPort 还占着，放弃启动。" -ForegroundColor Red
            exit 1
        }
    }
}

if ($DryRun) {
    Write-Host "[dry-run] 会启动前端：cd web; $pnpm dev  （端口 $FrontendPort）"
    exit 0
}
Start-Process -FilePath 'cmd' -ArgumentList @('/c', "$pnpm dev") -WorkingDirectory $web | Out-Null
Write-Host '✔ 前端已启动（新窗口）' -ForegroundColor Green
Write-Host "前端（开发）：http://127.0.0.1:$FrontendPort"
# 与 backend.ps1 同一个理由：让调用方拿到确定的退出码，而不是 `$null`
exit 0
