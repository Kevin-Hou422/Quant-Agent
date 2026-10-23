"""
2026-10-22 用户决定 · 批 2：时钟（D14 (a) + C3）与 Validate 预热（D9）。

D14 原话："T 收盘信息完成并过健康门后冻结信号与目标，T+1 按明确交易规则执行，更新时钟版本。同步修改信号、
回测、内部模拟、IC、归档、基准及调度。'计划开盘执行'不等于限价单必然按开盘价成交；预测 IC 与实际持仓收益
分别记录。旧历史保留旧版本，先隔离重算。"
C3 原话："目前虽然要求开盘前存档，评分仍是前收盘到今收盘，包含已经发生的隔夜收益。必须与 D14 一起明确标签
起止、信息截止和归档截止，不能只提前 deadline。"
D9 原话："只用当时已知历史计算特征，只评分 Validate 日期；拟合与标准化不能读取未来评分信息。覆盖 252 日因子、
嵌套窗口、跨界标签、初始持仓及历史不足。"
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from app.core.portfolio_manager.executable_spec import signal_config

from app.core.data_engine.market_calendar import session_close_utc, session_open_utc


# ---------------------------------------------------------------------------
# 时钟定义（单一来源）
# ---------------------------------------------------------------------------

class TestClockDefinition:

    def test_the_clock_is_c2_and_states_every_boundary(self):
        from app.core import clock
        from app.core.data_engine.input_identity import CLOCK_VERSION, CLOCK_VERSIONS
        assert CLOCK_VERSION == "c2" and "c1" in CLOCK_VERSIONS and "c2" in CLOCK_VERSIONS
        d = clock.describe()
        for k in ("information_cutoff", "decision", "execution", "fill_rule", "label", "archive_deadline",
                  "valuation"):
            assert d.get(k), f"时钟没有写明 {k}"

    def test_the_label_starts_after_the_archive_deadline(self):
        """C3：信号 T 的标签 = 开盘(T+1) → 开盘(T+2)；归档截止 = 开盘(T+1) —— 标签里没有任何已发生的收益。"""
        from app.core import clock
        T = pd.Timestamp("2025-03-04")
        win = clock.label_window(T)
        assert win["start"] == session_open_utc(pd.Timestamp("2025-03-05"))
        assert win["end"] == session_open_utc(pd.Timestamp("2025-03-06"))
        assert clock.archive_deadline(T) == win["start"]
        assert clock.archive_deadline(T) > session_close_utc(T)

    def test_label_returns_are_open_to_open_aligned_to_the_signal_date(self):
        from app.core import clock
        idx = pd.bdate_range("2025-03-03", periods=5)
        o = pd.DataFrame({"A": [10., 11., 12.1, 13.31, 14.641]}, index=idx)
        c = pd.DataFrame({"A": [99., 99., 99., 99., 99.]}, index=idx)       # 收盘价不该出现在标签里
        lab = clock.label_returns({"open": o, "close": c})
        assert lab["A"].iloc[0] == pytest.approx(12.1 / 11 - 1)            # 信号 3-03 → 开 3-04 到开 3-05
        assert np.isnan(lab["A"].iloc[-1]) and np.isnan(lab["A"].iloc[-2])  # 还没发生的不给

    def test_label_returns_require_open_prices(self):
        from app.core import clock
        idx = pd.bdate_range("2025-03-03", periods=5)
        with pytest.raises(ValueError, match="open"):
            clock.label_returns({"close": pd.DataFrame({"A": 1.0}, index=idx)})


# ---------------------------------------------------------------------------
# 规格：时钟与信号参数进哈希
# ---------------------------------------------------------------------------

def test_the_spec_carries_clock_c2_and_no_extra_signal_delay():
    from app.core.portfolio_manager.executable_spec import build_spec
    s = build_spec({"1": "rank(close)"}, {"1": 1.0}, "equal_weight", 1e4, 0.01)
    assert s["clock"] == "c2" and s["signal"]["delay"] == 0


def test_a_c1_spec_no_longer_matches_the_runtime():
    """旧时钟的规格不能被新时钟悄悄执行（重新提案，不继承旧时钟证据）。"""
    from app.core.portfolio_manager.executable_spec import build_spec, verify_against_runtime
    s = build_spec({"1": "rank(close)"}, {"1": 1.0}, "equal_weight", 1e4, 0.01)
    old = dict(s, clock="c1", signal=dict(s["signal"], delay=1))
    diff = verify_against_runtime(old, {"1": "rank(close)"}, 1e4)
    assert any("clock" in d for d in diff) and any("signal" in d for d in diff)


# ---------------------------------------------------------------------------
# 每日循环：T 的决策用到 T 的数据，T+1 才成交；IC 用开盘到开盘的标签
# ---------------------------------------------------------------------------

def _evidence_world(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from app.config import settings
    from app.core.execution.paper_broker import PaperBroker
    from app.db.alpha_store import AlphaResult, AlphaStore
    from app.db.evidence_store import EvidenceStore
    from app.db.position_store import PositionStore
    from app.tasks.daily_trading_loop import DailyTradingLoop
    url = f"sqlite:///{(tmp_path / 'a.db').as_posix()}"
    monkeypatch.setattr(settings, "database_url", url)
    store, ev = AlphaStore(db_url=url), EvidenceStore(url)
    aid = store.save(AlphaResult(dsl="rank(ts_delta(close, 5))", status="candidate"))
    store.update_status(aid, "validated"); store.update_status(aid, "paper")

    def loop():
        return DailyTradingLoop(store=store, evidence_store=ev, broker=PaperBroker(
            store=PositionStore(f"sqlite:///{(tmp_path / 'p.db').as_posix()}"), initial_capital=10_000.0))
    return SimpleNamespace(store=store, ev=ev, aid=aid, loop=loop, url=url)


def test_forward_ic_is_scored_on_the_open_to_open_label(tmp_path, monkeypatch):
    """C3：前向 IC = 存档信号 T 对 开盘(T+1)→开盘(T+2)；改动 T 收盘→T+1 开盘的隔夜价格不影响它。"""
    from tests.integration.test_evidence_versions_1019 import _propose
    from tests.integration.test_forward_evidence_1018 import _after_close, _panel, _upto
    w = _evidence_world(tmp_path, monkeypatch)
    sid, _ = _propose(w)
    ds = _panel(T=70)
    days = ds["close"].index
    for d in days[-4:]:
        w.loop().run_portfolio(_upto(ds, d), aum=10_000.0, data_source="yahoo", now=_after_close(d))
    rows = w.ev.strategy_forward_ic(sid)
    assert rows, "没有前向证据"
    r = rows[-1]
    T = pd.Timestamp(r.signal_asof)
    i = list(days).index(T)
    assert pd.Timestamp(r.date) == days[i + 2], "标签结束日应为 T+2 的开盘"
    # 独立重算：存档的信号（库里取）× 开盘到开盘
    arch = w.ev.archived_before("strategy:%d" % sid, T, pd.Timestamp("2100-01-01").to_pydatetime())
    lab = ds["open"].iloc[i + 2] / ds["open"].iloc[i + 1] - 1
    from scipy.stats import spearmanr
    expect = spearmanr(arch["signal"].reindex(lab.index), lab).correlation
    assert r.ic == pytest.approx(expect, abs=1e-9)


# ---------------------------------------------------------------------------
# 旧时钟的历史隔离
# ---------------------------------------------------------------------------

def test_factor_monitoring_reads_only_current_clock_ic(tmp_path):
    from app.db.alpha_store import AlphaResult, AlphaStore
    import app.core.data_engine.input_identity as ii
    url = f"sqlite:///{(tmp_path / 'a.db').as_posix()}"
    s = AlphaStore(db_url=url)
    aid = s.save(AlphaResult(dsl="rank(close)", status="candidate"))
    orig = ii.CLOCK_VERSION
    try:
        ii.CLOCK_VERSION = "c1"
        s.record_ic(aid, "2025-01-02", 0.5, is_forward=True)
    finally:
        ii.CLOCK_VERSION = orig
    s.record_ic(aid, "2025-01-03", 0.1, is_forward=True)
    assert [float(r.realized_ic) for r in s.get_forward_ic(aid)] == [0.1], "旧时钟的前向 IC 混进了当前口径"
    assert len(s.get_ic_history(aid, all_clocks=True)) == 2, "旧版本要保留，只是隔离"
    # 面板接口：两个时钟的点都给（带版本），滚动统计只用当前时钟，并告诉前端当前是哪个
    from app.api.router import alpha_ic_history
    resp = alpha_ic_history(aid, limit=250, store=s)
    assert [(p.clock_version, p.realized_ic) for p in resp.points] == [("c1", 0.5), ("c2", 0.1)]
    assert resp.rolling_mean_ic == pytest.approx(0.1) and resp.current_clock == "c2"


# ---------------------------------------------------------------------------
# D9：Validate 段预热
# ---------------------------------------------------------------------------

def _long_panel(T=800, N=12, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=T)
    cols = [f"S{i}" for i in range(N)]
    c = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0003, 0.015, (T, N)), 0), idx, cols)
    o = c.shift(1).fillna(c) * (1 + rng.normal(0, 0.003, (T, N)))
    amp = np.abs(rng.normal(0, 0.006, (T, N)))
    return {"open": o, "high": np.maximum(o, c) * (1 + amp), "low": np.minimum(o, c) * (1 - amp),
            "close": c, "volume": pd.DataFrame(1e7, idx, cols), "vwap": c,
            "returns": c.pct_change().fillna(0.0)}


def _split(ds, is_end=560, oos_start=580):
    cut = lambda a, b: {k: v.iloc[a:b] for k, v in ds.items()}          # noqa: E731
    return cut(0, is_end), cut(oos_start, len(next(iter(ds.values()))))


class TestValidateWarmup:

    DSL_252 = "rank(ts_delta(log(close), 252))"
    DSL_NESTED = "rank(ts_mean(ts_delta(log(close), 20), 252))"

    @pytest.mark.parametrize("dsl", [DSL_252, DSL_NESTED])
    def test_a_long_window_factor_is_evaluable_on_validate(self, dsl):
        from app.core.backtest_engine.realistic_backtester import RealisticBacktester
        is_d, oos_d = _split(_long_panel())
        res = RealisticBacktester(config=signal_config(), min_obs=0).run(dsl, is_d, oos_dataset=oos_d)
        assert res.oos_result is not None
        pos = res.oos_result.positions
        assert pos.index[0] >= oos_d["close"].index[0], "Validate 段的结果里混进了更早的日期"
        assert float(pos.abs().sum(axis=1).gt(0).mean()) > 0.9, "252 日因子在 Validate 段仍然几乎没有持仓"

    def test_validate_scores_do_not_read_data_after_the_validate_end(self):
        from app.core.backtest_engine.realistic_backtester import RealisticBacktester
        ds = _long_panel()
        is_d, oos_d = _split(ds)
        a = RealisticBacktester(config=signal_config(), min_obs=0).run(self.DSL_252, is_d, oos_dataset=oos_d)
        # Validate 段最后 30 天的价格换掉 → 那之前的每一天结果一点都不变（没有读未来）
        cut = oos_d["close"].index[-30]
        bump = np.where(oos_d["close"].index < cut, 1.0, 1.5)
        late = {k: (v.mul(bump, axis=0) if k in ("open", "high", "low", "close", "vwap") else v)
                for k, v in oos_d.items()}
        b = RealisticBacktester(config=signal_config(), min_obs=0).run(self.DSL_252, is_d, oos_dataset=late)
        before = a.oos_result.net_returns.index < cut
        pd.testing.assert_series_equal(a.oos_result.net_returns[before], b.oos_result.net_returns[before])
        # IS 的结果不受 Validate 段价格影响（预热只单向：历史 → Validate）
        oos_bumped = {k: (v * 1.5 if k in ("open", "high", "low", "close", "vwap") else v) for k, v in oos_d.items()}
        c = RealisticBacktester(config=signal_config(), min_obs=0).run(self.DSL_252, is_d, oos_dataset=oos_bumped)
        pd.testing.assert_series_equal(a.is_result.net_returns, c.is_result.net_returns)

    def test_validate_starts_flat(self):
        """初始持仓：Validate 段从空仓开始，不继承 IS 段最后的仓位。"""
        from app.core.backtest_engine.realistic_backtester import RealisticBacktester
        is_d, oos_d = _split(_long_panel())
        res = RealisticBacktester(config=signal_config(), min_obs=0).run(self.DSL_252, is_d, oos_dataset=oos_d)
        assert float(res.oos_result.positions.iloc[0].abs().sum()) == 0.0

    def test_insufficient_history_is_reported_not_faked(self):
        """历史不足：预热不够长的那几天如实为空，并报告被评分的天数。"""
        from app.core.backtest_engine.realistic_backtester import RealisticBacktester
        ds = _long_panel(T=420)
        is_d, oos_d = _split(ds, is_end=150, oos_start=160)
        res = RealisticBacktester(config=signal_config(), min_obs=0).run(self.DSL_252, is_d, oos_dataset=oos_d)
        info = res.oos_warmup
        assert info["history_days"] == 150 + len(oos_d["close"])
        assert 0 < info["scored_days"] < len(oos_d["close"]), info
        assert pd.Timestamp(info["first_scored"]) > oos_d["close"].index[0]


# ---------------------------------------------------------------------------
# 合成权重的拟合（D9 "拟合不能读取未来"；D14 "同步 IC"）
# ---------------------------------------------------------------------------

def _two_factor_signals(ds):
    c = ds["close"]
    return {"mom": c.pct_change(20), "rev": -c.pct_change(5)}


def test_walk_forward_fit_uses_only_labels_realized_before_the_fold():
    """
    第 k 段的合成权重只用 start 收盘时已经实现的标签。配对收益第 start 行 = 开盘(start)→开盘(start+1)，
    start 收盘时还没实现 —— 从这一行起换掉，第 2 段的合成信号必须一点不变。
    两个因子都真有预测力（IC 为正），否则 ic_weighted 退回等权，看不出拟合读没读未来。
    """
    from app.core.portfolio_manager.manager import PortfolioManager
    rng = np.random.default_rng(5)
    T, N = 240, 20
    idx = pd.bdate_range("2022-01-03", periods=T)
    cols = [f"S{i}" for i in range(N)]
    a = pd.DataFrame(rng.normal(size=(T, N)), idx, cols)
    b = pd.DataFrame(rng.normal(size=(T, N)), idx, cols)
    ret = pd.DataFrame(rng.normal(0, 0.01, (T, N)), idx, cols)
    ret.iloc[1:] += 0.004 * a.iloc[:-1].to_numpy() + 0.002 * b.iloc[:-1].to_numpy()   # 第 t+1 行配信号 t
    sig = {"a": a, "b": b}
    pm = PortfolioManager(aum=1e6, method="ic_weighted")
    folds = np.array_split(np.arange(T), 3)
    start2, start3 = idx[folds[1][0]], idx[folds[2][0]]
    _, _, comp_a = pm.walk_forward_composite(sig, a, 3, returns=ret)
    swapped = ret.copy()
    swapped.loc[swapped.index >= start2] = rng.normal(0, 0.01, swapped.loc[swapped.index >= start2].shape)
    swapped.loc[swapped.index >= start2, "S0"] -= 0.05 * a.loc[a.index >= start2, "S0"].shift(1).fillna(0)
    swapped.loc[swapped.index >= start2] -= 0.02 * a.loc[a.index >= start2].shift(1).fillna(0).to_numpy()
    _, _, comp_b = pm.walk_forward_composite(sig, a, 3, returns=swapped)
    fold2 = (comp_a.index >= start2) & (comp_a.index < start3)
    pd.testing.assert_frame_equal(comp_a[fold2], comp_b[fold2])
    # 第 3 段确实用到了换掉的标签（否则上面的相等说明不了什么）
    assert not comp_a[comp_a.index >= start3].equals(comp_b[comp_b.index >= start3])


def test_fitting_combo_weights_without_paired_returns_is_refused():
    """需要拟合却没给 c2 配对收益 → 报错，不拿收盘到收盘悄悄凑。"""
    from app.core.portfolio_manager.manager import PortfolioManager
    ds = _long_panel(T=120)
    with pytest.raises(ValueError, match="配对收益"):
        PortfolioManager(aum=1e6, method="ic_weighted").combined_weights(_two_factor_signals(ds), ds["close"])


def test_paired_returns_put_the_open_to_open_label_at_row_t_plus_1():
    from app.core.clock import label_returns, paired_returns
    ds = _long_panel(T=40)
    pr, lab = paired_returns(ds), label_returns(ds)
    pd.testing.assert_frame_equal(pr.shift(-1).iloc[:-2], lab.iloc[:-2])


@pytest.mark.parametrize("day", ["2025-01-15", "2025-07-15", "2025-03-07", "2025-03-10", "2025-11-03"])
def test_the_scheduled_trading_runs_fall_between_the_close_and_the_archive_deadline(day):
    """调度（D14）：交易循环与重试都在 T 收盘之后、开盘(T+1)（归档截止）之前 —— 冬令时 / 夏令时 / 切换前后都成立。"""
    from datetime import datetime, timezone
    from app.core import clock
    from app.tasks.scheduler import DAILY_TRADING_AT, DAILY_TRADING_RETRY_AT
    T = pd.Timestamp(day)
    close, deadline = session_close_utc(T), clock.archive_deadline(T)
    for h, m in (DAILY_TRADING_AT, DAILY_TRADING_RETRY_AT):
        run = datetime(T.year, T.month, T.day, h, m, tzinfo=timezone.utc)
        assert close < run < deadline, (day, (h, m), close, deadline)


# ---------------------------------------------------------------------------
# D14："'计划开盘执行'不等于限价单必然按开盘价成交"
# ---------------------------------------------------------------------------

class TestPlannedOpenIsNotAssumedFilled:
    """参考价 = 昨收 100，限价带 50bp → 买单限价 100.5、卖单限价 99.5。"""

    @staticmethod
    def _broker(tmp_path):
        from app.core.backtest_engine.transaction_cost import CostParams
        from app.core.execution.paper_broker import PaperBroker
        from app.db.position_store import PositionStore
        free = CostParams(fixed_bps=0.0, min_ticket_fee=0.0, spread_bps=0.0, impact_coef=0.0,
                          short_borrow_annual_bps=0.0)
        return PaperBroker(store=PositionStore(db_url=f"sqlite:///{(tmp_path / 'pb.db').as_posix()}"),
                           initial_capital=1e6, cost_params=free, limit_band_bps=50.0)

    @staticmethod
    def _buy(b, o, h, lo, c=101.0):
        s = lambda v: pd.Series([v], index=["A"], dtype=float)          # noqa: E731
        b.step(1, "2025-03-05", target_w=s(0.5), prices_t=s(c), prices_prev=s(100.0),
               adv_usd=s(1e15), daily_vol=s(0.02), open_t=s(o), high_t=s(h), low_t=s(lo))
        return {f.ticker: f for f in b.store.fills_on(1, "2025-03-05")}

    def test_open_inside_the_limit_fills_at_the_open(self, tmp_path):
        f = self._buy(self._broker(tmp_path), o=100.3, h=101.5, lo=100.0)["A"]
        assert f.fill_price == pytest.approx(100.3) and f.reject_reason == ""

    def test_gap_above_the_limit_fills_at_the_limit_only_if_the_day_trades_back(self, tmp_path):
        f = self._buy(self._broker(tmp_path), o=102.0, h=103.0, lo=100.2)["A"]
        assert f.fill_price == pytest.approx(100.5), "开盘跳空到限价之外却按开盘价成交了"

    def test_gap_that_never_comes_back_does_not_fill_and_is_booked_as_unfilled(self, tmp_path):
        b = self._broker(tmp_path)
        f = self._buy(b, o=102.0, h=103.0, lo=101.0)["A"]
        assert f.reject_reason == "limit_not_reached"
        assert f.filled_weight == pytest.approx(0.0, abs=1e-12)
        assert f.unfilled_weight == pytest.approx(0.5, abs=1e-12)
        assert b.store.latest_positions(1).get("A", 0.0) == pytest.approx(0.0, abs=1e-12)

    def test_the_predictive_label_and_the_realized_book_are_separate_quantities(self, tmp_path):
        """同一天：限价没成交 → 账本收益 0；预测标签照样是开盘到开盘（与成交无关）。"""
        from app.core.clock import label_returns
        b = self._broker(tmp_path)
        self._buy(b, o=102.0, h=103.0, lo=101.0)
        assert b.store.latest_equity(1) == pytest.approx(1.0)
        idx = pd.bdate_range("2025-03-04", periods=3)
        lab = label_returns({"open": pd.DataFrame({"A": [99.0, 102.0, 104.0]}, index=idx)})
        assert lab["A"].iloc[0] == pytest.approx(104.0 / 102.0 - 1)
