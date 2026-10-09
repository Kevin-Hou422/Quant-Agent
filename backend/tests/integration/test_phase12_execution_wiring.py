"""
test_phase12_execution_wiring.py — §K：执行层**接进了主线**，下的是**已激活策略**风控后的那一行

只测 OrderManager 本身证明不了"每日组合账本真的会下单"。这里从 `run_portfolio` 入口走：
数据集以"最近一个已收盘的交易日"结尾（真实美股日历 + 真实时钟，新鲜度检查走真路径），
券商换成与已安装 SDK 契约对齐的替身。

判别性断言：纸交易订单的股数 = trunc(决策日风控后目标权重 × 账户资产 / 当日收盘价)，
逐名比对；无交易带按**券商实际持仓**判断（审计 F07）—— 执行层若取错行、取了带后的模拟
账本权重、或未过风控的原始权重，股数就对不上。没有激活策略时券商侧只对账（审计 F08）。
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


def _activate_strategy(loop, settings, monkeypatch, tmp_path, dsls=("rank(ts_delta(close, 5))",
                                                                   "rank(-ts_delta(close, 20))")):
    """PAPER 因子 + 一份**已批准并激活**的策略配置（冻结等权组合权重）。返回因子 id 列表。"""
    from app.db.alpha_store import AlphaResult
    from app.db.strategy_store import StrategyConfig, StrategyStore
    monkeypatch.setattr(settings, "database_url", f"sqlite:///{tmp_path / 'a.db'}")
    ids = []
    for dsl in dsls:
        aid = loop.store.save(AlphaResult(dsl=dsl, status="candidate"))
        loop.store.update_status(aid, "validated")
        loop.store.update_status(aid, "paper")
        ids.append(str(aid))
    ss = StrategyStore()
    sid = ss.save(StrategyConfig(factors=ids, combo_weights={i: 1.0 / len(ids) for i in ids},
                                 aum=AUM, method="ic_weighted"))
    ss.update_status(sid, "approved")
    ss.update_status(sid, "active")
    return ids, sid


def test_paper_execution_trades_the_risk_gated_row_of_the_active_strategy(loop_env, monkeypatch,
                                                                        tmp_path):
    ctx, made, build, settings = loop_env
    monkeypatch.setattr(settings, "execution_mode", "moomoo_paper")
    snap = lambda tks: pd.DataFrame([{"code": "US.S0", "bid_price": 1.0, "ask_price": 1.0}])  # noqa: E731
    monkeypatch.setattr("app.core.trading_context.providers.moomoo_snapshot_fn",
                        lambda host, port: snap)
    loop = build()
    _, sid = _activate_strategy(loop, settings, monkeypatch, tmp_path)
    ds = _dataset()
    out = loop.run_portfolio(ds, aum=AUM)

    assert out["active_config"] == sid and out["combo_weights"] == pytest.approx(
        {k: 0.5 for k in out["combo_weights"]}), "没有用策略配置里冻结的组合权重"
    ex = out["execution"]
    assert made == [1] and ctx.closed, "网关没有被建立或没有被关闭"
    assert ex["blocked"] == "", ex
    assert ex["decision_date"] == str(ds["close"].index[-1].date())
    assert ex["reconcile"]["status"] == "baseline"
    assert out["post_band_violations"] == []

    # 券商收到的 = 风控后、无交易带前的决策日目标；空账户上带宽相对实际持仓（0）判断
    target, band = out["execution_target"], out["no_trade_band"]
    assert target and all(np.isfinite(w) and w != 0 for w in target.values()), "目标里混进了 0 / NaN 权重"
    px = ds["close"].iloc[-1]
    expected = {tk: math.trunc(w * AUM / px[tk]) for tk, w in target.items()
                if w >= band and math.trunc(w * AUM / px[tk]) > 0}
    assert expected, "目标里没有超过无交易带的正权重 —— 用例失去判别力"
    placed = {kw["code"].split(".", 1)[1]: kw["qty"] for kw in ctx.calls_of("place_order")}
    rejected = {r["ticker"]: (r["qty"], r["reason"]) for r in ex["gate"]["rejected"]}
    assert {**placed, **{t: q for t, (q, _) in rejected.items()}} == expected
    assert {r for _, r in rejected.values()} <= {"insufficient_buying_power"}
    assert ex["n_submitted"] == len(placed) > 0
    skipped_band = {s["ticker"] for s in ex["build"]["skipped"] if s["reason"] == "no_trade_band"}
    assert skipped_band == {tk for tk, w in target.items() if 0 < w < band}

    assert out["t3"]["mode"] == "live", "执行层在线时 T3 应来自券商实时账户"
    assert out["t3"]["buying_power"] == pytest.approx(AUM)

    from app.db.diagnostics_store import DiagnosticsStore
    assert DiagnosticsStore().recent(1)[0]["execution"]["n_submitted"] == ex["n_submitted"]


def test_without_an_active_strategy_the_broker_only_reconciles(loop_env, monkeypatch):
    """审计 F08：没有被批准并激活的策略版本 → 基准库只进模拟账本，券商侧只对账、不下单。"""
    ctx, made, build, settings = loop_env
    monkeypatch.setattr(settings, "execution_mode", "moomoo_paper")
    out = build().run_portfolio(_dataset(), aum=AUM)
    assert out["used_baseline"] is True and out["days_processed"] > 0
    assert out["strategy_verdict"] is None, "基准库不是被评审的策略，不该跑策略门"
    assert out["execution"]["blocked"] == "maintenance_only: no_active_strategy"
    assert out["execution"]["reconcile"]["status"] == "baseline"
    assert ctx.calls_of("place_order") == []


def test_unusable_active_strategy_halts_new_risk_instead_of_trading_other_factors(
        loop_env, monkeypatch, tmp_path):
    """审计 F08：active 配置引用的因子缺信号 → 不得改用其他 PAPER 因子，只维护。"""
    from app.db.alpha_store import AlphaResult
    ctx, made, build, settings = loop_env
    monkeypatch.setattr(settings, "execution_mode", "moomoo_paper")
    loop = build()
    _activate_strategy(loop, settings, monkeypatch, tmp_path,
                       dsls=("no_such_operator(close)", "rank(ts_delta(close, 5))"))
    other = loop.store.save(AlphaResult(dsl="rank(close)", status="candidate"))
    loop.store.update_status(other, "validated")
    loop.store.update_status(other, "paper")
    out = loop.run_portfolio(_dataset(), aum=AUM)
    assert out["reason"] == "active_config_unusable" and out["days_processed"] == 0
    assert out["execution"]["blocked"] == "maintenance_only: active_config_unusable"
    assert ctx.calls_of("place_order") == []


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
