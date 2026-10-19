"""
复审 2026-10-19（`review-v6/audit-review/复审报告_2026-10-19.md`）N01–N09 的回归。

每条都从复审独立脚本（`independent_checks.py`）里对应的 residual_* 场景改写：原脚本断言"问题仍在"，
这里断言修复后应有的行为。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from tests.conftest import _make_dataset


# ---------------------------------------------------------------------------
# N01：一批订单发送途中开熔断，余下的调仓单不再发出
# ---------------------------------------------------------------------------

class TestKillSwitchMidBatch:

    def test_orders_after_the_switch_are_not_sent(self, tmp_path):
        from tests.unit.execution.test_order_manager import D1, _cycle, _env, _placed
        e = _env(tmp_path)
        orig = e.gw.place

        def place_then_switch(req):
            oid = orig(req)
            if len(_placed(e)) == 1:
                e.es.set_kill_switch(True, "review", "switch during order batch")
            return oid
        e.gw.place = place_then_switch
        rep = _cycle(e, D1)
        assert rep.n_submitted == 1, "熔断之后又发出了调仓单"
        assert rep.blocked == "kill_switch_engaged_mid_batch"
        assert [o["result"] for o in rep.orders].count("skipped_kill_switch") == 2
        assert rep.flatten is not None                                # 转去撤单 + 全平

    def test_a_switch_set_while_waiting_for_cancels_stops_the_build(self, tmp_path):
        """撤单等终态期间被开熔断：建单之前那一次读要拦住。"""
        from tests.unit.execution.test_order_manager import D1, D2, _cycle, _env, _placed
        e = _env(tmp_path)
        _cycle(e, D1)                                                 # D1 的单留在途 → D2 要先撤
        n_before = len(_placed(e))
        orig = e.om._await_terminal

        def await_then_switch(ids):
            e.es.set_kill_switch(True, "review", "switch while cancels settle")
            return orig(ids)
        e.om._await_terminal = await_then_switch
        rep = _cycle(e, D2)
        assert rep.blocked == "kill_switch_engaged"
        assert [p for p in _placed(e)[n_before:] if p[1] == "BUY"] == []


# ---------------------------------------------------------------------------
# N06：加风险订单的报价合同
# ---------------------------------------------------------------------------

def _buy(qty_held=0.0):
    from app.core.execution.order_builder import build_rebalance_orders
    held = {"AAA": qty_held} if qty_held else {}
    return build_rebalance_orders(pd.Series({"AAA": .05}), held, pd.Series({"AAA": 100.}), 10000,
                                  pd.Timestamp("2026-01-05").date(), book_id=-1, band_bps=50).orders


def _check(orders, quotes, held=0.0, now=datetime(2026, 1, 5, 22, tzinfo=timezone.utc)):
    from app.core.execution.pretrade_gate import GateContext, PreTradeLimits, check
    ctx = GateContext(10000, 10000, {"AAA": held} if held else {}, {"AAA": 100.} if held else {},
                      {"AAA": 1e9}, quotes=quotes, now=now)
    return check(orders, ctx, PreTradeLimits(.1, 1., .1, .05, .15))


class TestQuoteContract:

    def test_a_quote_without_a_time_does_not_open_a_position(self):
        from app.core.execution.broker_gateway import Quote
        dec = _check(_buy(), {"AAA": Quote(100., None)})
        assert [r for _, r in dec.rejected] == ["quote_without_time"]

    def test_adding_to_a_held_name_still_needs_a_quote(self):
        dec = _check(_buy(qty_held=1.0), {}, held=1.0)
        assert [r for _, r in dec.rejected] == ["no_independent_quote"], "已有持仓价不能替代独立报价去加仓"

    def test_a_quote_from_the_future_is_not_trusted(self):
        from app.core.execution.broker_gateway import Quote
        dec = _check(_buy(), {"AAA": Quote(100., datetime(2026, 1, 6, 9, 0))})   # 纽约次日上午
        assert [r for _, r in dec.rejected] == ["quote_in_future"]

    def test_a_quote_from_before_the_last_closed_session_is_stale(self):
        from app.core.execution.broker_gateway import Quote
        dec = _check(_buy(), {"AAA": Quote(100., datetime(2026, 1, 2, 16, 0))})   # 上周五，今天周一已收盘
        assert [r for _, r in dec.rejected] == ["stale_quote"]

    def test_a_good_quote_from_the_closed_session_opens_the_position(self):
        from app.core.execution.broker_gateway import Quote
        dec = _check(_buy(), {"AAA": Quote(100.2, datetime(2026, 1, 5, 16, 0))})
        assert len(dec.approved) == 1 and dec.rejected == []

    def test_a_reduction_needs_no_quote(self):
        from app.core.execution.order_builder import build_rebalance_orders
        orders = build_rebalance_orders(pd.Series({"AAA": .0}), {"AAA": 10.}, pd.Series({"AAA": 100.}), 10000,
                                        pd.Timestamp("2026-01-05").date(), book_id=-1, band_bps=50).orders
        assert [o.side for o in orders] == ["SELL"]
        dec = _check(orders, {}, held=10.0)
        assert len(dec.approved) == 1


# ---------------------------------------------------------------------------
# N04：健康门检查的是截断之后、真正被使用的面板；持仓标的单独检查
# ---------------------------------------------------------------------------

def _valid_ohlc(n_days=35, n=20):
    ds = _make_dataset(n_days, n)
    ds["high"] = pd.DataFrame(np.maximum(ds["open"], ds["close"]) * 1.01, index=ds["close"].index,
                              columns=ds["close"].columns)
    ds["low"] = pd.DataFrame(np.minimum(ds["open"], ds["close"]) * .99, index=ds["close"].index,
                             columns=ds["close"].columns)
    return ds


class TestHealthOnTheConsumedPanel:

    def test_an_unclosed_complete_bar_no_longer_masks_a_broken_closed_day(self):
        from app.tasks.daily_ingest import DailyIngest
        ds = _valid_ohlc()
        ds["close"].iloc[-2, :18] = np.nan                        # 最近一个已收盘日只剩 10% 有价
        target = ds["close"].index[-2]
        with patch("app.core.data_engine.dataset_registry.load_registry_dataset",
                   return_value=SimpleNamespace(name="audit", data=ds, universe=list(ds["close"].columns))):
            res = DailyIngest().ingest("audit", "2022-01-01", "2023-01-01", append_pit=False, upto=target)
        assert res.accepted is False and "10%" in res.reject_reason

    def test_a_held_name_without_a_close_rejects_the_day(self, fresh_db):
        from app.core.execution.order_manager import LIVE_BOOK_ID
        from app.db.position_store import DailyPnL, PositionStore
        from app.tasks.daily_ingest import DailyIngest
        ds = _valid_ohlc()
        held = ds["close"].columns[0]
        ds["close"].iloc[-1, 0] = np.nan                           # 95% 覆盖率：宇宙平均看着没问题
        PositionStore().record_day(LIVE_BOOK_ID, "2022-01-03", {held: 0.05}, [],
                                   DailyPnL(LIVE_BOOK_ID, "2022-01-03", 0., 0., 0., 1.))
        with patch("app.core.data_engine.dataset_registry.load_registry_dataset",
                   return_value=SimpleNamespace(name="audit", data=ds, universe=list(ds["close"].columns))):
            res = DailyIngest().ingest("audit", "2022-01-01", "2023-01-01", append_pit=False)
        assert res.accepted is False and f"持仓标的 ['{held}']" in res.reject_reason


# ---------------------------------------------------------------------------
# N02：证据路径也核对规格与真实数据源；无新 bar 时换源也重建
# ---------------------------------------------------------------------------

class TestEvidenceIdentity:

    def _setup(self, fresh_db):
        from app.core.execution.paper_broker import PaperBroker
        from app.core.portfolio_manager.executable_spec import build_spec, spec_hash
        from app.db.alpha_store import AlphaStore
        from app.db.evidence_store import EvidenceStore
        from app.db.position_store import PositionStore
        from app.db.strategy_store import StrategyConfig, StrategyStore
        from app.tasks.daily_trading_loop import DailyTradingLoop
        spec = build_spec({"1": "rank(close)"}, {"1": 1.}, "equal_weight", 10000, .01)
        sid = StrategyStore().save(StrategyConfig(["1"], {"1": 1.}, 10000, method="equal_weight",
                                                  verdict={"executable_spec": spec, "spec_hash": spec_hash(spec)}))
        ev = EvidenceStore()
        loop = DailyTradingLoop(store=AlphaStore(), evidence_store=ev,
                                broker=PaperBroker(PositionStore(fresh_db), initial_capital=10000))
        return spec, sid, ev, loop

    def test_a_source_mismatch_is_quarantined_not_counted(self, fresh_db, monkeypatch):
        from app.config import settings
        from app.core.portfolio_manager.executable_spec import spec_hash
        from tests.integration.test_forward_evidence_1018 import _after_close, _panel, _upto
        spec, sid, ev, loop = self._setup(fresh_db)
        ds = _panel()
        a, b = ds["close"].index[-2:]
        loop._record_strategy_evidence(_upto(ds, a), _after_close(a))
        monkeypatch.setattr(settings, "price_source", "moomoo")
        out = loop._record_strategy_evidence(ds, _after_close(b))
        assert ev.strategy_forward_ic(sid, spec_hash(spec)) == []
        assert out[f"strategy:{sid}"] == -1
        (q,) = ev.revisions("strategy_evidence_quarantine")
        assert "data.price_source" in q.new_value

    def test_the_actual_pit_source_wins_over_what_settings_claim(self, fresh_db):
        from app.core.portfolio_manager.executable_spec import spec_hash
        from tests.integration.test_forward_evidence_1018 import _after_close, _panel, _upto
        spec, sid, ev, loop = self._setup(fresh_db)
        ds = _panel()
        a, b = ds["close"].index[-2:]
        loop._record_strategy_evidence(_upto(ds, a), _after_close(a), data_source="yahoo")
        loop._record_strategy_evidence(ds, _after_close(b), data_source="moomoo")   # settings 仍是 yahoo
        assert ev.strategy_forward_ic(sid, spec_hash(spec)) == []

    def test_a_source_switch_without_a_new_bar_rebuilds_the_pit(self, tmp_path, monkeypatch):
        from tests.integration.test_adjusted_price_rebase import DAYS, Provider, _close_utc
        import app.core.data_engine.dataset_registry as reg
        from app.config import settings
        from app.core.data_engine.pit_store import PITStore
        from app.tasks.daily_ingest import DailyIngest
        monkeypatch.setattr(settings, "pit_store_dir", str(tmp_path / "pit"))
        monkeypatch.setattr(settings, "paper_start", str(DAYS[0].date()))
        monkeypatch.setattr(settings, "price_source", "moomoo")
        prov = Provider()
        monkeypatch.setattr(reg, "load_registry_dataset", prov.load)
        monkeypatch.setattr(reg, "check_dataset_health",
                            lambda ds, min_score=0.7, warn_only=True: SimpleNamespace(overall_score=1.0, hard_failures=[]))
        day = pd.Timestamp("2024-03-19")
        assert DailyIngest().ingest_incremental("px", now=_close_utc(day)).mode == "full"
        monkeypatch.setattr(settings, "price_source", "yahoo")
        res = DailyIngest().ingest_incremental("px", now=_close_utc(day) + timedelta(hours=1))
        assert res.accepted and res.mode == "rebased" and res.forward_from is None
        meta = PITStore(tmp_path / "pit").read_meta("px")
        assert meta["price_source"] == "yahoo" and "无新 bar" in meta["last_rebase_reason"]


# ---------------------------------------------------------------------------
# N03：信号必须在被评分那一天开盘前存档
# ---------------------------------------------------------------------------

class TestArchiveBeforeTheScoredWindow:

    def test_a_signal_archived_during_the_scored_session_is_not_forward(self, fresh_db):
        from app.core.data_engine.market_calendar import session_close_utc, session_open_utc
        from app.db.evidence_store import EvidenceStore
        ev = EvidenceStore()
        before, day = pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")
        ev.archive("strategy:1", before, pd.Series({"AAA": 1., "BBB": 0.}),
                   recorded_at=session_close_utc(day) - timedelta(minutes=1))       # 复审的复现
        assert ev.forward_signal("strategy:1", before, day) is None
        ev.archive("strategy:2", before, pd.Series({"AAA": 1., "BBB": 0.}),
                   recorded_at=session_open_utc(day) + timedelta(minutes=1))        # 开盘后 1 分钟也不行
        assert ev.forward_signal("strategy:2", before, day) is None

    def test_a_signal_archived_before_the_open_is_forward(self, fresh_db):
        from app.core.data_engine.market_calendar import session_close_utc, session_open_utc
        from app.db.evidence_store import EvidenceStore
        ev = EvidenceStore()
        before, day = pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")
        ev.archive("strategy:3", before, pd.Series({"AAA": 1., "BBB": 0.}),
                   recorded_at=session_close_utc(before) + timedelta(hours=2))
        assert ev.forward_signal("strategy:3", before, day) is not None
        assert session_open_utc(day) < session_close_utc(day)


# ---------------------------------------------------------------------------
# N05：提案的门评估的就是冻结规格；门与每日循环同一条风险路径
# ---------------------------------------------------------------------------

class TestProposalEvaluatesTheFrozenSpec:

    def test_the_gate_receives_the_frozen_weights_and_band(self):
        from app.core.portfolio_manager.strategy_builder import build_strategy_config
        from app.core.portfolio_manager.strategy_gate import StrategyGate
        ds = _make_dataset(80, 10)
        calls = []

        def ev(self, *a, **kw):
            calls.append(dict(kw))
            return SimpleNamespace(passed=True, sharpe=1., deflated_sharpe=.99, t_stat=3., reasons=[],
                                   to_dict=lambda: {"passed": True})
        with patch.object(StrategyGate, "evaluate", ev):
            cfg = build_strategy_config({"1": ds["close"].rank(axis=1)}, ds, 10000, method="equal_weight",
                                        factor_dsl={"1": "rank(close)"})
        frozen = calls[0]
        assert frozen["fixed_combo_weights"] == cfg.combo_weights
        assert frozen["no_trade_band"] == cfg.no_trade_band
        assert cfg.verdict["executable_spec"]["no_trade_band"] == pytest.approx(cfg.no_trade_band, abs=1e-10)
        assert "fixed_combo_weights" not in calls[1]                        # ② 流程的滚动前推
        assert cfg.verdict["evidence_scope"] == "frozen_spec_on_selection_in_sample"
        assert cfg.verdict["procedure_oos"]["scope"] == "walk_forward_refit_of_the_procedure"

    def test_failing_the_procedure_oos_fails_the_proposal(self):
        from app.core.portfolio_manager.strategy_builder import build_strategy_config
        from app.core.portfolio_manager.strategy_gate import StrategyGate
        ds = _make_dataset(80, 10)

        def ev(self, *a, **kw):
            ok = "fixed_combo_weights" in kw
            return SimpleNamespace(passed=ok, sharpe=1., deflated_sharpe=.99, t_stat=3.,
                                   reasons=[] if ok else ["DSR 不足"], to_dict=lambda: {"passed": ok})
        with patch.object(StrategyGate, "evaluate", ev):
            cfg = build_strategy_config({"1": ds["close"].rank(axis=1)}, ds, 10000,
                                        factor_dsl={"1": "rank(close)"})
        assert cfg.passed is False and any("流程的滚动前推 OOS 未过" in r for r in cfg.verdict["reasons"])

    def test_the_gate_applies_vol_targeting_and_the_post_band_projection(self, monkeypatch):
        from app.config import settings
        import app.core.portfolio_manager.executable_spec as es
        from app.core.portfolio_manager.risk_gate import PortfolioRiskGate
        from app.core.portfolio_manager.strategy_gate import strategy_net_returns
        monkeypatch.setattr(settings, "risk_target_vol_ann", 0.10)
        vols, projected = [], []
        real_apply, real_band = PortfolioRiskGate.apply, es.band_then_reproject

        def spy_apply(self, w, sectors=None, port_vol_ann=None):
            vols.append(port_vol_ann)
            return real_apply(self, w, sectors=sectors, port_vol_ann=port_vol_ann)

        def spy_band(*a, **k):
            projected.append(1)
            return real_band(*a, **k)
        monkeypatch.setattr(PortfolioRiskGate, "apply", spy_apply)
        monkeypatch.setattr(es, "band_then_reproject", spy_band)
        ds = _make_dataset(120, 8)
        strategy_net_returns({"a": ds["close"].rank(axis=1)}, ds, aum=1e4, no_trade_band=0.01)
        assert vols[0] is not None and vols[0] > 0, "门没有做和每日循环一样的目标波动缩放"
        assert projected == [1] and vols[1] is None                      # 带后硬约束投影、不再传波动


# ---------------------------------------------------------------------------
# N07：试验没记全，门不给正式结论
# ---------------------------------------------------------------------------

class TestIncompleteTrialHistory:

    def test_a_run_that_never_finished_blocks_the_strategy_gate(self, fresh_db):
        import app.core.portfolio_manager.strategy_gate as sg
        from app.db.trial_ledger import TrialLedger
        led = TrialLedger()
        led.add(1)
        run = led.begin_run("gp")                                       # 开始了、没结束（进程被杀）
        led.add_progress(run, 3)
        ds = _make_dataset(300, 10)
        rets = pd.Series(np.random.default_rng(2).normal(.01, .001, 300), index=ds["close"].index)
        later = datetime.utcnow() + timedelta(hours=7)
        with patch.object(sg, "strategy_net_returns", return_value=(rets, ds["close"])), \
             patch("app.db.trial_ledger.datetime") as dt:
            dt.utcnow.return_value = later
            verdict = sg.StrategyGate().evaluate({"f": ds["close"]}, ds)
        assert verdict.passed is False and "试验史不完整" in verdict.reasons[0]

    def test_resolving_the_run_adds_the_estimate_and_unblocks(self, fresh_db):
        from app.db.trial_ledger import TrialLedger, trial_history_problem
        led = TrialLedger()
        run = led.begin_run("gp")
        with patch("app.db.trial_ledger.datetime") as dt:
            dt.utcnow.return_value = datetime.utcnow() + timedelta(hours=7)
            assert trial_history_problem() != ""
        assert led.resolve_run(run, 50, "kevin", "进程被杀，按配置的种群 × 代数估计") == 50
        assert trial_history_problem() == ""
        (st,) = led.run_states()
        assert st.status == "resolved" and "kevin" in st.note

    def test_a_progress_write_failure_leaves_the_run_open(self, fresh_db, monkeypatch):
        from app.db.trial_ledger import TrialLedger, TrialRecorder
        rec = TrialRecorder("gp")
        monkeypatch.setattr(TrialLedger, "add_progress", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
        rec.add(7)
        rec.finish("done")
        (st,) = TrialLedger().run_states()
        assert rec.ok is False and st.status == "running"               # 故意不结束：过期后被门看见

    def test_a_crashing_search_keeps_what_it_already_recorded(self, fresh_db, monkeypatch):
        from app.core.gp_engine.population_evolver import PopulationEvolver
        from app.db.trial_ledger import TrialLedger
        d = _make_dataset(160, 8)
        ev = PopulationEvolver({k: v.iloc[:110] for k, v in d.items()}, {k: v.iloc[110:] for k, v in d.items()},
                               pop_size=4, n_generations=3, seed=1)
        real = ev._generate_next_population
        monkeypatch.setattr(ev, "_generate_next_population",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("killed")))
        with pytest.raises(RuntimeError, match="killed"):
            ev.run(seed_dsl="rank(close)", n_optuna_trials=0)
        (st,) = TrialLedger().run_states()
        assert st.status == "aborted" and st.n_recorded > 0 and TrialLedger().total() == st.n_recorded
        assert callable(real)


# ---------------------------------------------------------------------------
# N08：费用按调仓前市值算
# ---------------------------------------------------------------------------

def test_the_rebalance_fee_uses_the_pre_trade_value(tmp_path):
    from app.core.backtest_engine.transaction_cost import CostParams
    from app.core.execution.paper_broker import PaperBroker
    from app.db.position_store import DailyPnL, PositionStore
    ps = PositionStore(f"sqlite:///{(tmp_path / 'p.db').as_posix()}")
    ps.record_day(1, "2026-01-05", {"AAA": .5, "BBB": .5}, [], DailyPnL(1, "2026-01-05", 0., 0., 0., 1.))
    broker = PaperBroker(ps, CostParams(fixed_bps=100, min_ticket_fee=0, spread_bps=0, impact_coef=0,
                                        short_borrow_annual_bps=0), 10000)
    pnl = broker.step(1, "2026-01-06", pd.Series({"AAA": .5, "BBB": .5}), pd.Series({"AAA": 110., "BBB": 100.}),
                      pd.Series({"AAA": 100., "BBB": 100.}), pd.Series({"AAA": 1e9, "BBB": 1e9}),
                      pd.Series({"AAA": .02, "BBB": .02}))
    fee_usd = pnl.cost_bps / 1e4 * 10000                               # cost 按期初权益 10,000 归一
    assert fee_usd == pytest.approx(5.0)                                # 成交 500 美元 × 100 bps
    assert pnl.equity == pytest.approx((10500 - 5) / 10000)


def test_the_backtest_engine_charges_the_same_fee(tmp_path):
    from app.core.backtest_engine.backtest_engine import BacktestEngine
    from app.core.backtest_engine.transaction_cost import CostParams
    idx = pd.bdate_range("2026-01-05", periods=2)
    close = pd.DataFrame({"AAA": [100., 110.], "BBB": [100., 100.]}, index=idx)
    w = pd.DataFrame(.5, index=idx, columns=close.columns)
    res = BacktestEngine(cost_params=CostParams(fixed_bps=100, min_ticket_fee=0, spread_bps=0, impact_coef=0,
                                                short_borrow_annual_bps=0),
                         initial_capital=10000).run(w, close, pd.DataFrame(1e12, index=idx, columns=close.columns), w)
    # 第一天买入 10,000 × 100 bps = 100 → 期末权益 9,900（各 4,950）；第二天 A 涨到 5,445，调仓前市值 10,395，
    # 两腿各成交 247.5，共 495 × 100 bps = 4.95（按旧口径会少算成 495 / 1.05 × 1% ≈ 4.71）
    day2_cost = (res.gross_returns.iloc[1] - res.net_returns.iloc[1]) * res.equity_curve.iloc[0] * 10000
    assert day2_cost == pytest.approx(4.95)


# ---------------------------------------------------------------------------
# N09：CLI 也只看研究窗口
# ---------------------------------------------------------------------------

def test_cli_gp_never_hands_frozen_dates_to_gp(fresh_db):
    from app.core.data_engine.data_partitioner import split_from_settings
    from app.core.gp_engine.population_evolver import PopulationEvolver
    import app.main as main
    ds = _make_dataset(1600, 10)
    frozen = split_from_settings(ds, dataset_key="audit_us")
    seen = {}

    class Stop(Exception):
        pass

    def capture(self, *a, **kw):
        seen["overlap"] = int((kw["oos_data"]["close"].index >= frozen.test_start).sum())
        raise Stop()
    with patch.object(main, "_load_dataset", return_value=ds), patch.object(PopulationEvolver, "__init__", capture):
        with pytest.raises(Stop):
            main.run_gp(SimpleNamespace(pop_size=2, generations=1, optuna_trials=0, dataset="audit_us"))
    assert seen["overlap"] == 0


def test_cli_refuses_data_that_starts_inside_the_frozen_segment(fresh_db):
    from app.core.data_engine.data_partitioner import split_from_settings
    import app.main as main
    ds = _make_dataset(1600, 10)
    frozen = split_from_settings(ds, dataset_key="audit_us")
    late = {k: v.loc[frozen.test_start - pd.Timedelta(days=20):] for k, v in ds.items()}
    with patch.object(main, "_load_dataset", return_value=late):
        with pytest.raises(SystemExit):
            main.run_gp(SimpleNamespace(pop_size=2, generations=1, optuna_trials=0, dataset="audit_us"))


def test_workflow_pnl_split_stays_inside_the_research_window(fresh_db):
    import app.api.router as router
    from app.core.data_engine.data_partitioner import split_from_settings
    ds = _make_dataset(1600, 10)
    with patch("app.core.data_engine.dataset_registry.load_registry_dataset",
               return_value=SimpleNamespace(data=ds)):
        full, is_, oos = router._resolve_dataset("audit_us", "", "", 10, 100, 1, 0.3)
    frozen = split_from_settings(ds, dataset_key="audit_us")
    assert len(full["close"]) == 1600
    assert oos["close"].index[-1] < frozen.test_start


def test_the_earliest_exposed_test_window_can_be_looked_up_for_d12(fresh_db):
    from app.db.trial_ledger import HoldoutLedger
    led = HoldoutLedger()
    assert led.earliest_exposed_start("us_x") is None
    led.record_use("us_x", "2023-05-01..2025-05-01")
    led.record_use("us_x", "2022-11-03..2024-11-01")
    led.record_use("us_x", "empty")
    assert led.earliest_exposed_start("us_x") == pd.Timestamp("2022-11-03")
