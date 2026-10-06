"""
test_phase12_execution_wiring.py — §K：执行层**接进了主线**，而且下的是**模拟账本当天交易的那一行**

只测 OrderManager 本身证明不了"每日组合账本真的会下单"。这里从 `run_portfolio` 入口走：
数据集以"最近一个已收盘的交易日"结尾（真实美股日历 + 真实时钟，新鲜度检查走真路径），
券商换成与已安装 SDK 契约对齐的替身。

判别性断言：纸交易订单的股数 = trunc(模拟账本当日目标权重 × 账户资产 / 当日收盘价)，
逐名比对 —— 执行层若取错行（例如倒数第二行、或未过风控的原始权重），股数就对不上。
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from app.core.data_engine.market_calendar import trading_days
from app.core.execution.broker_gateway import BrokerError
from app.core.execution.order_manager import last_closed_session
from tests.unit.execution.moomoo_fake import FakeTradeContext, make_gateway

AUM = 100_000.0


def _dataset(n=160, n_tickers=8, seed=3):
    last = last_closed_session(datetime.now(timezone.utc))
    idx = trading_days(pd.Timestamp(last) - pd.Timedelta(days=400), last)[-n:]
    rng = np.random.default_rng(seed)
    cols = [f"S{i}" for i in range(n_tickers)]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0003, 0.015, (n, n_tickers)), 0),
                         index=idx, columns=cols)
    amp = np.abs(rng.normal(0, 0.006, (n, n_tickers)))           # §O：真实量级的随机振幅
    return {"open": close.shift(1).fillna(close), "high": close * (1 + amp),
            "low": close * (1 - amp), "close": close, "vwap": close,
            "volume": pd.DataFrame(1e6, index=idx, columns=cols),
            "returns": close.pct_change().fillna(0.0)}


@pytest.fixture
def loop_env(tmp_path, monkeypatch):
    from app.config import settings
    from app.core.execution.paper_broker import PaperBroker
    from app.db.alpha_store import AlphaStore
    from app.db.execution_store import ExecutionStore
    from app.db.position_store import PositionStore

    monkeypatch.setattr(settings, "paper_aum", AUM)
    ctx = FakeTradeContext(cash=AUM)
    made = []

    def factory():
        made.append(1)
        return make_gateway(ctx)

    def build(**kw):
        from app.tasks.daily_trading_loop import DailyTradingLoop
        broker = PaperBroker(store=PositionStore(db_url=f"sqlite:///{tmp_path / 'p.db'}"),
                             initial_capital=AUM)
        return DailyTradingLoop(
            store=AlphaStore(db_url=f"sqlite:///{tmp_path / 'a.db'}"), broker=broker,
            execution_store=ExecutionStore(db_url=f"sqlite:///{tmp_path / 'e.db'}"),
            gateway_factory=kw.get("factory", factory))
    return ctx, made, build, settings


def test_paper_execution_trades_the_same_row_the_sim_book_traded(loop_env, monkeypatch):
    ctx, made, build, settings = loop_env
    monkeypatch.setattr(settings, "execution_mode", "moomoo_paper")
    snap = lambda tks: pd.DataFrame([{"code": "US.S0", "bid_price": 1.0, "ask_price": 1.0}])  # noqa: E731
    monkeypatch.setattr("app.core.trading_context.providers.moomoo_snapshot_fn",
                        lambda host, port: snap)
    loop = build()
    ds = _dataset()
    out = loop.run_portfolio(ds, aum=AUM)

    ex = out["execution"]
    assert made == [1] and ctx.closed, "网关没有被建立或没有被关闭"
    assert ex["blocked"] == "", ex
    assert ex["decision_date"] == str(ds["close"].index[-1].date())
    assert ex["reconcile"]["status"] == "baseline"

    last = ds["close"].index[-1]
    sim_fills = loop.broker.store.fills_on(0, last)
    assert sim_fills, "模拟账本最后一天没有成交记录 —— 无从比对"
    expected = {}
    for f in sim_fills:
        q = math.trunc(f.target_weight * AUM / f.fill_price)
        if q > 0:
            expected[f.ticker] = q
    assert expected, "模拟账本最后一天没有正目标 —— 用例失去判别力"
    placed = {kw["code"].split(".", 1)[1]: kw["qty"] for kw in ctx.calls_of("place_order")}
    rejected = {r["ticker"]: (r["qty"], r["reason"]) for r in ex["gate"]["rejected"]}
    # 下了的 + 被买入力拦下的 = 模拟账本当日目标，逐名、逐股对得上
    assert {**placed, **{t: q for t, (q, _) in rejected.items()}} == expected
    assert {r for _, r in rejected.values()} <= {"insufficient_buying_power"}
    assert ex["n_submitted"] == len(placed) > 0

    assert out["t3"]["mode"] == "live", "执行层在线时 T3 应来自券商实时账户"
    assert out["t3"]["buying_power"] == pytest.approx(AUM)

    from app.db.diagnostics_store import DiagnosticsStore
    assert DiagnosticsStore().recent(1)[0]["execution"]["n_submitted"] == ex["n_submitted"]


def test_default_mode_never_touches_the_broker(loop_env):
    ctx, made, build, settings = loop_env
    assert settings.execution_mode == "off"
    out = build().run_portfolio(_dataset(), aum=AUM)
    assert out["execution"] == {"mode": "off"}
    assert made == [] and ctx.calls == []
    assert out["t3"]["mode"] == "sim"


def test_broker_unavailable_blocks_execution_but_not_the_sim_book(loop_env, monkeypatch):
    ctx, made, build, settings = loop_env
    monkeypatch.setattr(settings, "execution_mode", "moomoo_paper")

    def down():
        raise BrokerError("OpenD not running")
    out = build(factory=down).run_portfolio(_dataset(), aum=AUM)
    assert out["execution"]["blocked"] == "gateway_unavailable: OpenD not running"
    assert out["days_processed"] > 0, "执行层失败拖垮了模拟账本"


def test_unknown_execution_mode_does_not_trade(loop_env, monkeypatch):
    ctx, made, build, settings = loop_env
    monkeypatch.setattr(settings, "execution_mode", "moomoo_live")
    out = build().run_portfolio(_dataset(), aum=AUM)
    assert out["execution"] == {"mode": "moomoo_live", "blocked": "unknown_execution_mode"}
    assert made == []


def test_days_without_tradable_signals_still_reconcile(loop_env, monkeypatch, tmp_path):
    """
    没有可交易信号的日子（这里：唯一的 PAPER 因子 DSL 跑不起来）不调仓，但必须照样对账 ——
    否则熔断期间这些天全平不会续做、账本也不会更新。
    """
    from app.db.alpha_store import AlphaResult
    ctx, made, build, settings = loop_env
    monkeypatch.setattr(settings, "execution_mode", "moomoo_paper")
    loop = build()
    aid = loop.store.save(AlphaResult(dsl="no_such_operator(close)", status="candidate"))
    loop.store.update_status(aid, "validated")
    loop.store.update_status(aid, "paper")
    out = loop.run_portfolio(_dataset(), aum=AUM)
    assert out["reason"] == "no valid signals"
    assert out["execution"]["blocked"] == "maintenance_only: no valid signals"
    assert out["execution"]["reconcile"]["status"] == "baseline"
    assert ctx.calls_of("position_list_query") and ctx.calls_of("place_order") == []
    assert made == [1] and ctx.closed


def test_execution_failure_inside_the_cycle_is_reported(loop_env, monkeypatch):
    ctx, made, build, settings = loop_env
    monkeypatch.setattr(settings, "execution_mode", "moomoo_paper")
    ctx.fail["accinfo_query"] = "account service busy"
    out = build().run_portfolio(_dataset(), aum=AUM)
    assert out["execution"]["blocked"].startswith("reconcile_failed")
    assert "account service busy" in out["execution"]["blocked"]
    assert ctx.calls_of("place_order") == []
    assert ctx.closed
