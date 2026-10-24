"""
conftest.py — 全局测试 fixtures

提供：
  - make_dataset(n_days, n_tickers, seed) → Dict[str, pd.DataFrame]
  - test_client → FastAPI TestClient
  - tmp_alpha_store → 内存 SQLite AlphaStore（每个测试独立）
  - tmp_chat_store  → 内存 SQLite ChatStore（每个测试独立）
  - basic_dsl       → 通用测试 DSL 字符串
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tests import network_guard


# ---------------------------------------------------------------------------
# 不许连外网；真实数据源用例单独标出、缺省不跑（D4，见 tests/network_guard.py）
# ---------------------------------------------------------------------------
def pytest_configure(config):
    config.addinivalue_line("markers", "live_source: 连真实外部数据源（yfinance / SEC / OpenD …）的用例；"
                                       "缺省跳过，QA_LIVE_SOURCES=1 才跑")
    network_guard.install()


def pytest_collection_modifyitems(config, items):
    if os.environ.get("QA_LIVE_SOURCES") == "1":
        return
    skip = pytest.mark.skip(reason="真实数据源用例：缺省不跑（QA_LIVE_SOURCES=1 才跑）")
    for item in items:
        if "live_source" in item.keywords:
            item.add_marker(skip)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    """在这条用例的**全部**夹具（含 module / session 级）建立之前清零 —— 夹具里联网同样算这条用例的。"""
    network_guard.state.allow = item.get_closest_marker("live_source") is not None
    network_guard.state.attempts = []
    yield


@pytest.fixture(autouse=True)
def _network_guard(request):
    live = request.node.get_closest_marker("live_source") is not None
    yield
    tried = sorted(set(network_guard.state.attempts))
    network_guard.state.allow, network_guard.state.attempts = False, []
    if tried and not live:
        pytest.fail(f"这条用例试图连外网：{tried[:5]} —— 用替身 / 合成夹具，或标 @pytest.mark.live_source 单独管理")


# ---------------------------------------------------------------------------
# 测试与部署 .env 解耦：强制关掉后台运行开关
# ---------------------------------------------------------------------------
# 部署 .env 会设 ENABLE_SCHEDULER/PAPER_TRADING/DISCOVERY=true 以真实运行；而测试从
# backend/ 跑会读到同一 .env → session 级 TestClient 的 lifespan 会启动真实调度器，
# 污染 test_status_reflects_running_state 等用例（并起后台线程）。此处强制置 False，
# 使测试结果与部署配置无关（见 DEV_LESSONS：production 配置不得泄漏进测试）。
@pytest.fixture(scope="session", autouse=True)
def _hermetic_run_flags(tmp_path_factory):
    from app.config import settings
    settings.enable_scheduler     = False
    settings.enable_paper_trading = False
    settings.enable_discovery     = False
    # Phase 12：部署 .env 可能设 EXECUTION_MODE=moomoo_paper（真实下纸交易单）。
    # 测试一律关掉；需要执行层的用例自己注入假券商并显式打开（§H）。
    settings.execution_mode       = "off"
    # 部署 .env 可能设 PRICE_SOURCE=moomoo（真实运行用）；测试强制回 yahoo，
    # 否则加载 US 数据集会去打 OpenD 网关（依赖运行时 + 烧历史K线额度）。见 DEV_LESSONS §H。
    settings.price_source = "yahoo"
    # 聊天路径生产默认走真实数据集（外部审计修复）；测试里显式用合成，避免打网络。
    # 注意：这是**显式 opt-in**，响应会标注 data_source=synthetic —— 与"静默回退"性质不同。
    settings.chat_allow_synthetic = True
    settings.chat_dataset = ""
    # DB 隔离：默认 store（含 TestClient 的 get_store）走 settings.database_url。
    # 若指向真实 alphas.db，测试里的 /alpha/save 等会污染生产账本（曾在 pending 队列
    # 里看到 rank(close) 测试垃圾）。session 级重定向到临时库。
    _tmp = tmp_path_factory.mktemp("hermetic_db") / "test_alphas.db"
    settings.database_url = f"sqlite:///{_tmp}"
    settings.pit_store_dir = str(tmp_path_factory.mktemp("hermetic_pit"))
    # 部署 .env 把调度库指到真实数据目录；测试建的调度器不得写进去（任务定义会被持久化）。
    settings.scheduler_db_url = f"sqlite:///{tmp_path_factory.mktemp('hermetic_sched') / 'jobs.db'}"
    # 账户锁是整台机器共用的（复审 2026-10-19 §5.3）：测试不得去碰真实部署的锁目录
    settings.account_lock_dir = str(tmp_path_factory.mktemp("hermetic_locks"))
    # 告警（D15）：事件文件写到临时目录；桌面通知整体替换成记录器 —— 测试绝不弹真实的系统通知
    settings.alert_event_file = str(tmp_path_factory.mktemp("hermetic_alerts") / "events.jsonl")
    from app.core.alerting import desktop_notifier
    def _recorded(title, body):
        _recorded.sent.append((title, body))
    _recorded.sent = []
    desktop_notifier.notify = _recorded
    yield


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """
    本条用例自己的库（含 API 依赖注入的 AlphaStore / StrategyStore 单例）。

    会**激活策略配置**或往组合账本写前向 IC 的用例必须用它：session 级共享库里留下一份
    active 配置，之后每一条跑 `run_portfolio` 的用例都会去交易它（审计 F08 之后，配置成分
    缺失还会让组合直接停摆）—— 一条用例的结论不能取决于前面跑过什么。
    """
    import app.dependencies as deps
    from app.config import settings
    from app.db.alpha_store import AlphaStore
    from app.db.strategy_store import StrategyStore
    url = f"sqlite:///{(tmp_path / 'fresh.db').as_posix()}"
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(settings, "database_url", url)
    a, s = AlphaStore(db_url=url), StrategyStore(db_url=url)
    monkeypatch.setattr(deps, "_get_store_singleton", lambda: a)
    monkeypatch.setattr(deps, "_get_strategy_store_singleton", lambda: s)
    return url


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def legacy_alpha(store, dsl: str, status: str) -> int:
    """
    构造一条 **D1 之前**就已处于 `status` 的存量因子记录（2026-10-22 D1 之后，正常路径 —— save / update_status ——
    不能再把用到基本面字段的因子放进 validated / paper / active）。只给测试"存量状态下其它功能照常"用：
    直接插行，不经过晋级校验。
    """
    from datetime import datetime
    from app.db.alpha_store import AlphaRecord
    with store._Session() as s:                              # noqa: SLF001
        rec = AlphaRecord(dsl=dsl, created_at=datetime.utcnow(), hypothesis="", status=status, reasoning="")
        s.add(rec)
        s.commit()
        return int(rec.id)


def approved_verdict(dsl_by_id: dict, combo_weights: dict, aum: float, method: str = "ic_weighted",
                     band: float = 0.0, settings=None, design_at: str = "2000-01-03T00:00:00") -> dict:
    """
    批准时的可执行规格（外部审查 2026-10-18 F07）。每日循环只按带规格的 active 配置交易；
    用例里激活配置时用它，并且要在 settings 改完之后调用（规格记的是批准那一刻的交易参数）。
    另带一份预登记的验证设计（复审 2026-10-19 N05-d：真实提案都会登记；→ACTIVE 只认登记之后的前向证据）。
    `design_at` 缺省早于用例里的任何证据日期；测"登记之前的证据不算"时显式给一个更晚的时刻。
    """
    from app.core.lifecycle.validation_design import design_hash, register_design
    from app.core.portfolio_manager.executable_spec import build_spec, spec_hash
    spec = build_spec(dsl_by_id, combo_weights, method, aum, band, settings=settings)
    design = register_design(now=pd.Timestamp(design_at))
    return {"executable_spec": spec, "spec_hash": spec_hash(spec),
            "validation_design": design, "validation_design_hash": design_hash(design)}


def _make_dataset(
    n_days:    int = 80,
    n_tickers: int = 10,
    seed:      int = 42,
) -> dict:
    """合成市场数据，用于单元 + 集成测试。"""
    rng     = np.random.default_rng(seed)
    dates   = pd.bdate_range("2022-01-03", periods=n_days)
    tickers = [f"T{i:03d}" for i in range(n_tickers)]

    close = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.01, (n_days, n_tickers)), axis=0),
        index=dates, columns=tickers,
    )
    volume = pd.DataFrame(
        rng.integers(500_000, 5_000_000, (n_days, n_tickers)).astype(float),
        index=dates, columns=tickers,
    )
    high  = close * (1 + rng.uniform(0, 0.01, (n_days, n_tickers)))
    low   = close * (1 - rng.uniform(0, 0.01, (n_days, n_tickers)))
    open_ = close * (1 + rng.normal(0, 0.005, (n_days, n_tickers)))
    vwap  = (high + low + close) / 3
    returns = close.pct_change().fillna(0.0)

    return {
        "close":   close,
        "open":    open_,
        "high":    high,
        "low":     low,
        "volume":  volume,
        "vwap":    vwap,
        "returns": returns,
    }


@pytest.fixture
def make_dataset():
    return _make_dataset


@pytest.fixture
def dataset():
    return _make_dataset()


@pytest.fixture
def basic_dsl() -> str:
    return "rank(ts_delta(log(close), 5))"


# ---------------------------------------------------------------------------
# FastAPI TestClient
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_client():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from app.main import app
    # D11：改状态的请求只接受本机回环客户端（app/api/loopback_guard.py）—— 测试客户端以 127.0.0.1 出现，
    # 生产代码里没有为测试开的口子（Starlette 缺省的客户端地址 "testclient" 不是回环）
    with TestClient(app, client=("127.0.0.1", 50000), raise_server_exceptions=False) as client:
        yield client


# ---------------------------------------------------------------------------
# Isolated in-memory stores
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_alpha_store(tmp_path):
    from app.db.alpha_store import AlphaStore
    db_path = tmp_path / "alpha_test.db"
    return AlphaStore(db_url=f"sqlite:///{db_path}")


@pytest.fixture
def tmp_chat_store(tmp_path):
    from app.db.chat_store import ChatStore
    db_path = tmp_path / "chat_test.db"
    return ChatStore(db_url=f"sqlite:///{db_path}")


# ---------------------------------------------------------------------------
# 进程外副作用的总闸：任何测试都不得弹出浏览器
# ---------------------------------------------------------------------------
# 教训（2026-09-14，实测事故）：`visualizer.plot()` 的签名是
# `show: bool = False`，为 False 时不调 `fig.show()`。变异测试把这个布尔
# 字面量翻成 True 之后，`test_backtest_plot_fidelity.py` 里三十多条
# 「画图然后检查 trace」的用例**每条都真的打开了一个浏览器标签**，
# 一次性在使用者屏幕上弹出几十个页面。
#
# 当时只在专门测 show 开关的那两条用例里 monkeypatch 了 `Figure.show`，
# 而变异测试**会把代码跑在你没预期的配置下** —— 所以凡是能对进程外
# 产生副作用的东西（浏览器、网络、进程外文件），必须在**全局**堵死，
# 不能只在"相关"用例里堵。
#
# 这里把 plotly 的 `Figure.show` 换成一个只记账的替身，并把记录挂在
# `pytest` 的全局位置上，供需要断言"show 有没有被调用"的用例读取。
_FIGURE_SHOW_CALLS: list = []


@pytest.fixture(scope="session", autouse=True)
def _never_open_a_browser():
    """把 plotly 的 Figure.show 全局换成记账替身（没装 plotly 则跳过）。"""
    try:
        import plotly.graph_objects as go
    except Exception:
        yield
        return

    original = go.Figure.show

    def _recording_show(self, *args, **kwargs):
        _FIGURE_SHOW_CALLS.append({"args": args, "kwargs": kwargs})

    go.Figure.show = _recording_show
    try:
        yield
    finally:
        go.Figure.show = original


@pytest.fixture
def figure_show_calls():
    """
    本条用例期间 `Figure.show` 的调用记录（进入时清空）。

    需要断言"show 到底有没有被调用"的用例用它，而不要自己去
    monkeypatch `Figure.show` —— 自己打补丁只保护自己那一条。
    """
    _FIGURE_SHOW_CALLS.clear()
    yield _FIGURE_SHOW_CALLS
    _FIGURE_SHOW_CALLS.clear()
