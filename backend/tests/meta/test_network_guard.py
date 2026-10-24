"""
测试进程不许连外网，真实数据源用例单独管理（2026-10-22 用户决定 D4，见 tests/network_guard.py）。
"""

from __future__ import annotations

import socket

import pytest

from tests import network_guard

pytest_plugins = ["pytester"]

#: 嵌套运行里只装守卫这三件（不带 tests/conftest.py 的其它 session 夹具）
_GUARDED_CONFTEST = ("from tests.conftest import _network_guard, pytest_collection_modifyitems, "
                     "pytest_configure, pytest_runtest_setup  # noqa: F401\n")


def test_an_outside_connection_is_refused_and_recorded():
    with pytest.raises(network_guard.NetworkBlocked):
        socket.create_connection(("query1.finance.yahoo.com", 443), timeout=1)
    assert network_guard.state.attempts == ["query1.finance.yahoo.com"]
    network_guard.state.attempts.clear()                 # 本条是故意的：别让守卫把它判失败


def test_a_raw_ip_connection_is_refused_too():
    s = socket.socket()
    try:
        with pytest.raises(network_guard.NetworkBlocked):
            s.connect(("93.184.216.34", 80))
    finally:
        s.close()
        network_guard.state.attempts.clear()


def test_curl_cffi_is_guarded_too():
    """yfinance 走 curl_cffi（C 层 libcurl，不经过 Python socket）—— 只拦 socket 时它照样联网。"""
    from curl_cffi import requests as curl_requests
    with pytest.raises(Exception) as e:
        curl_requests.get("https://query1.finance.yahoo.com/v8/finance/chart/AAPL", timeout=2)
    assert isinstance(e.value, network_guard.NetworkBlocked) or isinstance(e.value.__context__, network_guard.NetworkBlocked)
    assert network_guard.state.attempts == ["query1.finance.yahoo.com"]
    network_guard.state.attempts.clear()


def test_loopback_is_allowed():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    try:
        c = socket.create_connection(srv.getsockname(), timeout=2)
        c.close()
    finally:
        srv.close()
    assert network_guard.state.attempts == []


@pytest.mark.parametrize("host,local", [("localhost", True), ("127.0.0.1", True), ("::1", True), ("[::1]", True),
                                        ("10.0.0.7", False), ("sec.gov", False), (b"127.0.0.1", True)])
def test_what_counts_as_local(host, local):
    assert network_guard.is_local(host) is local


def test_a_test_that_swallows_the_error_still_fails(pytester):
    """把异常吞掉（"无网络时允许 502"）也藏不住：试图连外网本身就判失败。"""
    pytester.makeconftest(_GUARDED_CONFTEST)
    pytester.makepyfile(test_swallow="""
import socket
def test_swallows():
    try:
        socket.getaddrinfo("example.com", 443)
    except OSError:
        pass
""")
    res = pytester.runpytest_inprocess("-p", "no:cacheprovider")
    res.assert_outcomes(passed=1, errors=1)
    assert "试图连外网" in res.stdout.str()


def test_a_module_fixture_that_goes_online_is_charged_to_the_test(pytester):
    """module 级夹具先于函数级夹具建立 —— 守卫要在整个 setup 之前清零，否则夹具里联网会漏记。"""
    pytester.makeconftest(_GUARDED_CONFTEST)
    pytester.makepyfile(test_fixture_online="""
import pytest, socket
@pytest.fixture(scope="module")
def data():
    try:
        socket.getaddrinfo("query1.finance.yahoo.com", 443)
    except OSError:
        return None
def test_uses(data):
    assert data is None
""")
    res = pytester.runpytest_inprocess("-p", "no:cacheprovider")
    assert res.parseoutcomes() == {"passed": 1, "errors": 1}
    assert "query1.finance.yahoo.com" in res.stdout.str()


def test_live_source_tests_are_skipped_by_default_and_run_on_request(pytester, monkeypatch):
    pytester.makeconftest(_GUARDED_CONFTEST)
    pytester.makepyfile(test_live="""
import pytest, socket
from tests import network_guard
@pytest.mark.live_source
def test_live():
    assert network_guard.state.allow is True
""")
    monkeypatch.delenv("QA_LIVE_SOURCES", raising=False)
    assert pytester.runpytest_inprocess("-p", "no:cacheprovider").parseoutcomes() == {"skipped": 1}
    monkeypatch.setenv("QA_LIVE_SOURCES", "1")
    assert pytester.runpytest_inprocess("-p", "no:cacheprovider").parseoutcomes() == {"passed": 1}
