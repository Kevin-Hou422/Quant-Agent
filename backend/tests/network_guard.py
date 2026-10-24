"""
network_guard.py — 测试不许连外网；连真实数据源的用例单独标出、单独跑（2026-10-22 用户决定 D4）

用户原话："仓库已公开，替换为等价合成行情夹具。保留测试合同；真实源测试单独管理。"

此前有用例悄悄去 yfinance 拉数据：网络一抖就红（与改动无关），网络断了又被"允许 502/503"一类的断言吞掉，
谁在连外网只能靠人记。现在：

    · 整个测试进程里，非回环的 DNS 解析与连接都被拒绝（抛 `NetworkBlocked`，是 OSError —— 走代码自己的离线分支）
    · 试图连外网的用例**判失败**（哪怕它把异常吞了），失败信息里写着它想连哪
    · 标了 `@pytest.mark.live_source` 的用例放行；它们缺省跳过，`QA_LIVE_SOURCES=1` 时才跑
    · 回环地址（本机 OpenD、TestClient）不受影响
    · 两层都要拦：Python socket（requests / httpx / moomoo SDK）和 curl_cffi —— yfinance 默认走 curl_cffi，
      那是 C 层的 libcurl，**不经过 Python socket**（只拦 socket 时守卫一条外网尝试都没记到，yfinance 照样联网）。
      curl_cffi 的每个请求都要先 `setopt(CurlOpt.URL, …)`，在那里按主机拦
"""

from __future__ import annotations

import ipaddress
import socket
from typing import List
from urllib.parse import urlsplit

_REAL_CONNECT = socket.socket.connect
_REAL_CONNECT_EX = socket.socket.connect_ex
_REAL_GETADDRINFO = socket.getaddrinfo


class NetworkBlocked(OSError):
    """测试进程里试图连外网。"""


class _State:
    allow = False
    attempts: List[str] = []


state = _State()


def is_local(host) -> bool:
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    host = str(host or "").strip("[]").split("%", 1)[0]
    if host in ("", "localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _check(host) -> None:
    if state.allow or is_local(host):
        return
    state.attempts.append(str(host))
    raise NetworkBlocked(f"测试不许连外网（{host}）—— 用替身 / 合成夹具；真实数据源用例标 @pytest.mark.live_source")


def _connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        _check(address[0])
    return _REAL_CONNECT(self, address)


def _connect_ex(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        _check(address[0])
    return _REAL_CONNECT_EX(self, address)


def _getaddrinfo(host, *args, **kwargs):
    if host is not None:
        _check(host)
    return _REAL_GETADDRINFO(host, *args, **kwargs)


def install() -> None:
    socket.socket.connect = _connect
    socket.socket.connect_ex = _connect_ex
    socket.getaddrinfo = _getaddrinfo
    _install_curl_cffi()


def _install_curl_cffi() -> None:
    try:
        from curl_cffi.curl import Curl, CurlOpt
    except ImportError:                      # 没装 curl_cffi：没有这条旁路
        return
    real_setopt = Curl.setopt
    if getattr(real_setopt, "_network_guard", False):
        return

    def setopt(self, option, value):
        if option == CurlOpt.URL:
            url = value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)
            _check(urlsplit(url).hostname or "")
        return real_setopt(self, option, value)
    setopt._network_guard = True
    Curl.setopt = setopt
