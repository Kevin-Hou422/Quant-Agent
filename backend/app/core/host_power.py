"""
host_power.py — 跑调度器的这台机器会不会睡着

来由（2026-11-14）：调度器跑在一台笔记本上。每日交易任务排在 21:30 UTC（北京时间 05:30），盘中巡检在美股时段
（北京时间夜里）。2026-10-30 系统睡眠记录显示这台机器在一次长任务中途待机了 3 个多小时；机器睡着时调度器、盘中巡检、
日亏熔断都不运行，醒来超过补跑宽限（1 小时）的每日任务被直接跳过 —— 当天不下单，前向记录缺一天。

这里做三件事（调度器启动时开、停止时关）：
    KeepAwake       请求系统在调度器运行期间不要因空闲待机（SetThreadExecutionState）。挡不住合盖、按电源键、
                    电量耗尽 —— 所以另外量。
    SuspendWatch    量出机器实际睡了多久：每拍记一次墙上时间，两拍之间隔得远超间隔 → 中间机器没在跑。
    power_problems  电源设置里会让机器睡着的项（上线预检用）：没接电源、接电源时空闲会睡、接电源时合盖会睡。

只用标准库与 Windows API；不是 Windows / 读不到 → 返回 None，不猜。
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

_ES_CONTINUOUS, _ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001

#: Windows 电源设置（powercfg 的 SUB_BUTTONS / LIDACTION、SUB_SLEEP / STANDBYIDLE、SUB_SLEEP / HIBERNATEIDLE）
_SUB_BUTTONS = "4f971e89-eebd-4455-a8de-9e59040e7347"
_LIDACTION = "5ca83367-6e45-459f-a27b-476b1d01c936"
_SUB_SLEEP = "238c9fa8-0aad-41ed-83f4-97be242c8f20"
_STANDBYIDLE = "29f6c1db-86da-48c5-9fdb-f2b67b1f44da"
_HIBERNATEIDLE = "9d7815a6-7ee4-497e-8888-515a05f02364"
LID_DO_NOTHING = 0


def power_status() -> dict:
    """{"ac": 接着电源?, "battery_pct": 电量, "saver": 节电模式?}；不是 Windows / 取不到 → 值为 None。"""
    out = {"ac": None, "battery_pct": None, "saver": None}
    if os.name != "nt":
        return out
    try:
        import ctypes

        class _Status(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                        ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                        ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
        st = _Status()
        if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(st)):
            out["ac"] = {0: False, 1: True}.get(int(st.ACLineStatus))
            out["battery_pct"] = int(st.BatteryLifePercent) if st.BatteryLifePercent <= 100 else None
            out["saver"] = bool(st.SystemStatusFlag & 1)
    except (OSError, AttributeError) as exc:
        logger.warning("[host] 读不到电源状态: %s", exc)
    return out


def power_settings() -> dict:
    """
    当前电源计划里接电源时的三项：{"lid_action": 合盖动作（0 = 不采取任何操作）, "standby_idle_s": 空闲多久睡眠（0 = 从不）,
    "hibernate_idle_s": 空闲多久休眠（0 = 从不）}。读不到 → None。
    """
    out = {"lid_action": None, "standby_idle_s": None, "hibernate_idle_s": None}
    if os.name != "nt":
        return out
    try:
        import ctypes
        from ctypes import wintypes

        class _GUID(ctypes.Structure):
            _fields_ = [("raw", ctypes.c_ubyte * 16)]

        def guid(s: str) -> _GUID:
            g = _GUID()
            ctypes.memmove(g.raw, uuid.UUID(s).bytes_le, 16)
            return g

        powrprof = ctypes.windll.powrprof
        scheme = ctypes.POINTER(_GUID)()
        if powrprof.PowerGetActiveScheme(None, ctypes.byref(scheme)) != 0:
            return out
        try:
            for key, sub, setting in (("lid_action", _SUB_BUTTONS, _LIDACTION),
                                      ("standby_idle_s", _SUB_SLEEP, _STANDBYIDLE),
                                      ("hibernate_idle_s", _SUB_SLEEP, _HIBERNATEIDLE)):
                val = wintypes.DWORD()
                sg, st = guid(sub), guid(setting)
                if powrprof.PowerReadACValueIndex(None, scheme, ctypes.byref(sg), ctypes.byref(st),
                                                  ctypes.byref(val)) == 0:
                    out[key] = int(val.value)
        finally:
            ctypes.windll.kernel32.LocalFree(scheme)
    except (OSError, AttributeError) as exc:
        logger.warning("[host] 读不到电源设置: %s", exc)
    return out


def power_problems(status: dict, settings: dict) -> List[str]:
    """会让这台机器在交易时段睡着的设置。读不到的项不算问题（另由调用方标为"无从判断"）。"""
    out = []
    if status.get("ac") is False:
        pct = status.get("battery_pct")
        out.append("没有接电源（在用电池" + (f"，剩 {pct}%" if pct is not None else "") + "）—— 电量耗尽会睡着")
    if settings.get("standby_idle_s"):
        out.append(f"接电源时空闲 {settings['standby_idle_s'] // 60} 分钟会睡眠 —— 改成「从不」")
    if settings.get("hibernate_idle_s"):
        out.append(f"接电源时空闲 {settings['hibernate_idle_s'] // 60} 分钟会休眠 —— 改成「从不」")
    if settings.get("lid_action") not in (None, LID_DO_NOTHING):
        out.append("接电源时合上盖子会睡眠 / 休眠 / 关机 —— 改成「不采取任何操作」，或交易期间别合盖")
    return out


def _set_execution_state(flags: int) -> bool:
    if os.name != "nt":
        return False
    try:
        import ctypes
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))
    except (OSError, AttributeError) as exc:
        logger.warning("[host] 设不了防待机: %s", exc)
        return False


class KeepAwake:
    """
    在一个专用线程里请求系统不要因空闲待机，直到 stop()。
    SetThreadExecutionState 的请求跟着**发出请求的线程**走：线程结束请求就失效 —— 所以要一个一直活着的线程。
    """

    def __init__(self, set_state: Optional[Callable[[int], bool]] = None):
        self._set = set_state or (lambda flags: _set_execution_state(flags))     # 调用时再取（测试可替换）
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.granted = False

    def start(self) -> "KeepAwake":
        def hold():
            self.granted = self._set(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
            self._ready.set()
            self._stop.wait()
            self._set(_ES_CONTINUOUS)
        self._thread = threading.Thread(target=hold, name="keep-awake", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


class SuspendWatch:
    """
    量出机器睡了多久：每 `interval` 秒记一次墙上时间，两次之间隔得比 interval + slack 还久，多出来的就是机器没在跑的时间。
    负载再重，一个只等待的线程也不会晚到几十秒。每发现一段就调 `on_gap(开始, 结束, 秒数)`。
    """

    def __init__(self, on_gap: Callable[[datetime, datetime, float], None], interval: float = 15.0,
                 slack: float = 60.0, clock: Callable[[], float] = time.time):
        self.on_gap, self.interval, self.slack, self.clock = on_gap, float(interval), float(slack), clock
        self._last = clock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def tick(self) -> None:
        now = self.clock()
        gap = now - self._last
        if gap > self.interval + self.slack:
            start = datetime.fromtimestamp(self._last, timezone.utc)
            end = datetime.fromtimestamp(now, timezone.utc)
            try:
                self.on_gap(start, end, gap - self.interval)
            except Exception as exc:                     # 报告出错不能让量的线程停下
                logger.error("[host] 待机报告失败: %s", exc)
        self._last = now

    def start(self) -> "SuspendWatch":
        def loop():
            while not self._stop.wait(self.interval):
                self.tick()
        self._thread = threading.Thread(target=loop, name="suspend-watch", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 1)
