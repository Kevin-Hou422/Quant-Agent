# start_forward.ps1 —— 起 OpenD → 上线预检 → 起后端（含每日调度器）
#
#   powershell -ExecutionPolicy Bypass -File scripts\start_forward.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\start_forward.ps1 -RegisterLogonTask   # 另外注册"登录即启动"
#
# 前提：backend\.env 已按 backend\.env.forward.example 配好；OpenD 已勾选"记住密码/自动登录"，
# 否则每次启动都要在 OpenD 窗口里手动登录一次。
# 预检任何一项 FAIL 都不会启动后端 —— 宁可不跑，不跑错。

param(
    [string]$OpenDExe = "$env:APPDATA\moomoo_OpenD\moomoo_OpenD.exe",
    [string]$DataDir = "C:\QuantAgentData",
    [int]$OpenDPort = 11111,
    [int]$ApiPort = 8000,
    [int]$LoginWaitSeconds = 300,
    [switch]$RegisterLogonTask
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Backend = Join-Path $Root "backend"
$Python = Join-Path $Root "venv\Scripts\python.exe"

if ($RegisterLogonTask) {
    $action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-ExecutionPolicy Bypass -WindowStyle Minimized -File `"$PSCommandPath`""
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
    Register-ScheduledTask -TaskName "QuantAgentForward" -Action $action -Trigger $trigger `
        -Description "Quant Agent 前向交易：OpenD + 后端" -Force | Out-Null
    Write-Host "已注册登录启动任务 QuantAgentForward（取消：Unregister-ScheduledTask QuantAgentForward）"
}

if (-not (Test-Path (Join-Path $Backend ".env"))) {
    throw "缺少 backend\.env —— 先复制 backend\.env.forward.example 并按注释修改"
}
foreach ($d in @($DataDir, (Join-Path $DataDir "logs"))) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null }
}

function Test-Port([int]$p) {
    [bool](Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue)
}

# 1) OpenD
if (-not (Test-Port $OpenDPort)) {
    if (-not (Get-Process -Name "moomoo_OpenD" -ErrorAction SilentlyContinue)) {
        if (-not (Test-Path $OpenDExe)) { throw "找不到 OpenD：$OpenDExe（用 -OpenDExe 指定）" }
        Write-Host "启动 OpenD：$OpenDExe"
        Start-Process $OpenDExe
    }
    Write-Host "等待 OpenD 监听 $OpenDPort（未自动登录的话，请在 OpenD 窗口里登录）..."
    $deadline = (Get-Date).AddSeconds($LoginWaitSeconds)
    while (-not (Test-Port $OpenDPort)) {
        if ((Get-Date) -gt $deadline) { throw "OpenD 在 $LoginWaitSeconds 秒内没有开始监听 —— 检查是否已登录" }
        Start-Sleep -Seconds 3
    }
}
Write-Host "OpenD 在监听 $OpenDPort"

# 2) 预检（会对账一次，不下单）
Push-Location $Backend
try {
    & $Python -m app.tasks.forward preflight
    if ($LASTEXITCODE -ne 0) { throw "上线预检未通过（见上面 FAIL 项），不启动后端" }

    # 3) 后端 + 调度器
    if (Test-Port $ApiPort) { throw "端口 $ApiPort 已被占用 —— 后端可能已在运行" }
    $log = Join-Path $DataDir ("logs\backend_{0:yyyyMMdd_HHmmss}.log" -f (Get-Date))
    Write-Host "启动后端（日志 $log）"
    Start-Process -FilePath $Python -WorkingDirectory $Backend -WindowStyle Minimized `
        -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$ApiPort" `
        -RedirectStandardError $log
    Start-Sleep -Seconds 8
    if (-not (Test-Port $ApiPort)) { throw "后端没有起来，看日志：$log" }
    Write-Host "后端已启动：http://127.0.0.1:$ApiPort/api/scheduler/status 可看下一次运行时间"
}
finally {
    Pop-Location
}
