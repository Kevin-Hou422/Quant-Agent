"""
test_phase7_paper.py — Phase 7 Paper Trading 验收（2026-08-02）

覆盖：
  7.2 PaperBroker 对账：replay 净值与 BacktestEngine 逐位一致（机器精度）
  7.2 幂等：同日重跑不重复记账、覆盖式写入
  7.2 崩溃恢复：从 PositionStore 续跑，持仓/净值正确接续
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _market(T=40, N=6, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=T)
    cols = [f"T{i}" for i in range(N)]
    prices = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0003, 0.012, (T, N)), axis=0),
                          index=idx, columns=cols)
    volume = pd.DataFrame(rng.integers(2e6, 8e6, (T, N)).astype(float), index=idx, columns=cols)
    signal = pd.DataFrame(rng.normal(0, 1, (T, N)), index=idx, columns=cols)
    return prices, volume, signal


class TestPaperBrokerReconciliation:

    def test_replay_matches_backtest_engine(self):
        from app.core.backtest_engine.backtest_engine import BacktestEngine
        from app.core.backtest_engine.portfolio_constructor import SignalWeightedPortfolio
        from app.core.execution.paper_broker import PaperBroker
        from app.db.position_store import PositionStore

        prices, volume, signal = _market()
        weights = SignalWeightedPortfolio(clip_z=3.0).construct(signal)

        eq_bt = BacktestEngine().run(weights, prices, volume, signal).equity_curve
        pb = PaperBroker(store=PositionStore(db_url="sqlite:///:memory:"))
        eq_pb = pb.replay(1, weights, prices, volume)

        # 逐位一致（机器精度）—— PaperBroker 复用同一成本/流动性模型
        assert np.max(np.abs(eq_bt.values - eq_pb.values)) < 1e-9
        # 也满足 roadmap 的 <1bp/日 验收
        daily_diff = np.abs(np.diff(eq_bt.values) - np.diff(eq_pb.values))
        assert np.max(daily_diff) < 1e-4


class TestDailyTradingLoop:

    @pytest.fixture
    def env(self, tmp_path):
        from app.db.alpha_store import AlphaStore, AlphaResult
        from app.db.position_store import PositionStore
        from app.core.execution.paper_broker import PaperBroker
        from app.core.monitor.alpha_monitor import AlphaMonitor
        from app.tasks.daily_trading_loop import DailyTradingLoop
        db = f"sqlite:///{tmp_path/'live.db'}"
        astore = AlphaStore(db_url=db)
        pstore = PositionStore(db_url=db)
        loop = DailyTradingLoop(store=astore,
                                broker=PaperBroker(store=pstore),
                                monitor=AlphaMonitor(astore, rolling_window=5))
        prices, volume, signal = _market(T=40, N=6, seed=3)
        ds = {"close": prices, "open": prices, "high": prices, "low": prices,
              "volume": volume, "vwap": prices,
              "returns": prices.pct_change().fillna(0.0)}
        return astore, pstore, loop, ds, AlphaResult

    def test_end_to_end_records(self, env):
        astore, pstore, loop, ds, AlphaResult = env
        aid = astore.save(AlphaResult(dsl="rank(ts_delta(close, 5))",
                                      hypothesis="mom", status="paper"))
        rep = loop.run(ds)
        assert rep.n_alphas == 1 and rep.results[0].error == ""
        assert rep.results[0].days_processed == 40
        assert len(pstore.pnl_history(aid)) == 40         # 每日 PnL
        assert len(astore.get_ic_history(aid)) > 20       # realized IC 记录

    def test_idempotent_rerun(self, env):
        astore, pstore, loop, ds, AlphaResult = env
        aid = astore.save(AlphaResult(dsl="rank(ts_delta(close, 5))", status="paper"))
        loop.run(ds)
        n1 = len(pstore.pnl_history(aid))
        rep2 = loop.run(ds)
        assert rep2.results[0].days_processed == 0        # 已全部处理
        assert len(pstore.pnl_history(aid)) == n1         # 无重复记账

    def test_per_alpha_isolation(self, env):
        """一个坏 DSL 的因子出错，不影响正常因子。"""
        astore, pstore, loop, ds, AlphaResult = env
        good = astore.save(AlphaResult(dsl="rank(ts_delta(close, 5))", status="paper"))
        bad  = astore.save(AlphaResult(dsl="this_is_not_valid_dsl(((", status="active"))
        rep = loop.run(ds)
        by_id = {r.alpha_id: r for r in rep.results}
        assert by_id[good].error == "" and by_id[good].days_processed == 40
        assert by_id[bad].error != ""                     # 坏因子被隔离报错
        assert rep.n_errors == 1
        assert len(pstore.pnl_history(good)) == 40         # 好因子正常记账

    def test_decay_triggers_transition(self, env):
        """ACTIVE 因子连续负 IC → 衰减告警 → 状态机 ACTIVE→DECAYING。"""
        astore, pstore, loop, ds, AlphaResult = env
        from app.core.monitor.alpha_monitor import AlphaMonitor
        # 用极敏感的监控器（连续 3 日负 IC 即告警）
        loop.monitor = AlphaMonitor(astore, rolling_window=5,
                                    consecutive_neg_limit=3, mean_ic_floor=999)
        aid = astore.save(AlphaResult(dsl="rank(ts_delta(close, 5))", status="active"))
        # 预置连续负 IC 历史（模拟已衰减）
        for i in range(6):
            astore.record_ic(aid, f"2022-12-{i+1:02d}", -0.05)
        rep = loop.run(ds)
        r = [x for x in rep.results if x.alpha_id == aid][0]
        # 衰减告警触发且状态已流转
        assert r.decay_alert is True
        assert astore.get_by_id(aid).status == "decaying"


class TestIdempotencyAndRecovery:

    @pytest.fixture
    def setup(self, tmp_path):
        from app.core.execution.paper_broker import PaperBroker
        from app.db.position_store import PositionStore
        from app.core.backtest_engine.portfolio_constructor import SignalWeightedPortfolio
        prices, volume, signal = _market()
        weights = SignalWeightedPortfolio(clip_z=3.0).construct(signal)
        store = PositionStore(db_url=f"sqlite:///{tmp_path/'p.db'}")
        return PaperBroker(store=store), store, weights, prices, volume

    def test_rerun_is_idempotent(self, setup):
        pb, store, weights, prices, volume = setup
        eq1 = pb.replay(1, weights, prices, volume)
        # 完整重跑一遍（模拟崩溃后从头补跑）
        eq2 = pb.replay(1, weights, prices, volume)
        assert np.allclose(eq1.values, eq2.values, atol=1e-12)
        # PnL 表每天只有一条（幂等，非追加）
        hist = store.pnl_history(1)
        assert len(hist) == len(weights)
        dates = [h.date for h in hist]
        assert len(dates) == len(set(dates))       # 无重复日期

    def test_crash_recovery_continues(self, setup):
        """崩溃续跑：分两段 step（0..mid，崩溃，mid..T）== 一次 step 到底。

        用**一致的全窗口 adv/vol**（每日循环的口径），验证 state_before 让续跑
        正确接续 t-1 状态，无回退无重复。
        """
        from app.db.position_store import PositionStore
        from app.core.execution.paper_broker import PaperBroker
        pb, store, weights, prices, volume = setup

        # 统一的每日市场上下文（全窗口，模拟循环每日可见的数据）
        prices_f = prices.ffill(limit=5).fillna(0.0)
        volume_f = volume.ffill(limit=5).fillna(0.0)
        adv_df = pb._liq.compute_adv(volume_f, prices_f)
        vol_df = prices_f.pct_change().rolling(20, min_periods=2).std().fillna(0.02)
        cal_years = max((weights.index[-1] - weights.index[0]).days / 365.25, 1/365.25)
        tdays = len(weights) / cal_years

        def _step_range(broker, lo, hi):
            for t in range(lo, hi):
                broker.step(1, weights.index[t], target_w=weights.iloc[t],
                            prices_t=prices_f.iloc[t],
                            prices_prev=prices_f.iloc[t-1] if t > 0 else prices_f.iloc[t],
                            adv_usd=adv_df.iloc[t], daily_vol=vol_df.iloc[t],
                            tdays_per_year=tdays)

        T = len(weights); mid = T // 2
        # 参照：一次 step 到底（独立 store）
        ref = PaperBroker(store=PositionStore(db_url="sqlite:///:memory:"))
        _step_range(ref, 0, T)

        # 崩溃续跑：前半 → （模拟崩溃）→ 后半
        _step_range(pb, 0, mid)
        assert store.latest_equity(1) != 1.0 and len(store.latest_positions(1)) > 0
        _step_range(pb, mid, T)

        assert abs(store.latest_equity(1) - ref.latest_equity(1)) < 1e-12

    def test_get_positions_and_equity(self, setup):
        pb, store, weights, prices, volume = setup
        pb.replay(1, weights, prices, volume)
        pos = pb.get_positions(1)
        assert isinstance(pos, pd.Series) and len(pos) > 0
        assert pb.latest_equity(1) > 0
