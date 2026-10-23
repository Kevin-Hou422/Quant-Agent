"""
desktop_notifier.py — Windows 桌面通知（2026-10-22 用户决定 D15 的首版通道）

用系统自带的 Windows PowerShell 调 Windows.UI.Notifications 弹出一条 toast，不引入新依赖。
标题与正文经**环境变量**传给脚本，从不拼进命令文本（带引号 / 分号的告警文本不能变成命令）。

成功的含义只是"通知交给了操作系统"（handed_to_os）：专注模式、通知设置、锁屏都可能让人看不见。
"看得见"只能由人现场确认（见 dispatcher.confirm_visible）。
"""

from __future__ import annotations

import os
import subprocess

#: Windows PowerShell 自带的 AppUserModelID：用它创建通知者，不需要注册自己的应用
_POWERSHELL_APP_ID = r"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"

_SCRIPT = (
    "$ErrorActionPreference='Stop';"
    "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime] > $null;"
    "$t=[Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent("
    "[Windows.UI.Notifications.ToastTemplateType]::ToastText02);"
    "$x=$t.GetElementsByTagName('text');"
    "$x.Item(0).AppendChild($t.CreateTextNode($env:QA_TOAST_TITLE)) > $null;"
    "$x.Item(1).AppendChild($t.CreateTextNode($env:QA_TOAST_BODY)) > $null;"
    "$n=[Windows.UI.Notifications.ToastNotification]::new($t);"
    "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($env:QA_TOAST_APP).Show($n)"
)


def notify(title: str, body: str) -> None:
    """告警分发器用的通知入口（测试套件在 conftest 里整体替换它，测试绝不弹真实通知）。"""
    show_toast(title, body)


def show_toast(title: str, body: str, timeout_s: float = 20.0) -> None:
    """弹一条桌面通知。失败（非零退出、超时、找不到 PowerShell）→ OSError。"""
    env = dict(os.environ)
    env.update(QA_TOAST_TITLE=str(title)[:200], QA_TOAST_BODY=str(body)[:800], QA_TOAST_APP=_POWERSHELL_APP_ID)
    try:
        r = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", _SCRIPT],
                           env=env, capture_output=True, text=True, timeout=timeout_s)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise OSError(f"桌面通知没有发出：{exc}") from exc
    if r.returncode != 0:
        raise OSError(f"桌面通知没有发出（退出码 {r.returncode}）：{(r.stderr or r.stdout or '').strip()[:300]}")
