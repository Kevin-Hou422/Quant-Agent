"""
test_audit_portfolio_regressions.py — 外部审计（2026-10-07）组合层 / 统计层缺陷的正确行为回归

F06 净敞口上限真的被施加；F07 无交易带之后再施加风控并独立复核；F09 硬门异常不放行；
F10 策略门的组合权重滚动前推、冻结权重满足时间前缀不变性；F12 PIT 写入中断不毁历史；
F16 因子级 IC 带前向标记。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.core.portfolio_manager.horizon import apply_no_trade_band
from app.core.portfolio_manager.risk_gate import PortfolioRiskGate, RiskLimits


# ---------------------------------------------------------------------------
# F06：净敞口
# ---------------------------------------------------------------------------

def test_net_limit_is_applied_long_only():
    gate = PortfolioRiskGate(RiskLimits(max_gross=1, max_net=.05, max_name_weight=.1,
                                        max_sector_weight=1, long_only=True))
    w = pd.DataFrame([[.1, .1]], columns=["AAA", "BBB"])
    out, _ = gate.apply(w, sectors=pd.Series({"AAA": 1, "BBB": 2}))
    assert out.sum(axis=1).iloc[0] == pytest.approx(.05)
    assert out.iloc[0]["AAA"] == pytest.approx(out.iloc[0]["BBB"])        # 按比例缩，不偏向谁
    assert gate.check(out, sectors=pd.Series({"AAA": 1, "BBB": 2})) == []


@pytest.mark.parametrize("w,net", [([.3, .2, -.1], .2), ([.1, -.3, -.2], -.2)])
def test_net_projection_only_shrinks_the_dominant_side(w, net):
    gate = PortfolioRiskGate(RiskLimits(max_gross=2, max_net=.2, max_name_weight=1,
                                        max_sector_weight=10, long_only=False))
    cols = ["A", "B", "C"]
    out, _ = gate.apply(pd.DataFrame([w], columns=cols), sectors=pd.Series(dict(zip(cols, [1, 2, 3]))))
    row = out.iloc[0].to_numpy()
    assert row.sum() == pytest.approx(net)
    minority = np.array(w) < 0 if net > 0 else np.array(w) > 0
    assert np.allclose(row[minority], np.array(w)[minority]), "少数一侧也被动了"
    assert (np.abs(row) <= np.abs(w) + 1e-12).all(), "净敞口投影放大了某一侧"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -0.1])
def test_risk_limits_must_be_positive_finite(bad):
    with pytest.raises(ValueError, match="正的有限数"):
        RiskLimits(max_net=bad)


# ---------------------------------------------------------------------------
# F07：无交易带之后的风控复核
# ---------------------------------------------------------------------------

def test_band_then_regate_restores_the_limits():
    """审计原复现：[.085,.085,.13] 在 band=.02 下变回 [.10,.10,.13]，gross 33% > 30%。"""
    gate = PortfolioRiskGate(RiskLimits(max_gross=.3, max_net=.3, max_name_weight=1,
                                        max_sector_weight=1, long_only=True))
    w = pd.DataFrame([[.1, .1, .1], [.085, .085, .13]], columns=["AAA", "BBB", "CCC"])
    sec = pd.Series({x: 1 for x in w.columns})
    safe, _ = gate.apply(w, sectors=sec)
    banded = apply_no_trade_band(safe, .02)
    assert gate.check(banded, sec), "前提：带后确实超限（否则这条用例没有判别力）"
    regated, _ = gate.apply(banded, sectors=sec)      # 与 run_portfolio 同一顺序
    assert gate.check(regated, sec) == []
    assert regated.iloc[-1].sum() == pytest.approx(.3)


def test_the_daily_loop_regates_after_the_band(tmp_path, monkeypatch):
    """
    行为而非源码文本（§Y）：让无交易带把权重放大 3 倍（gross ≫ 上限），
    模拟账本**实际交易**的每一天都必须仍在 gross 上限之内。
    """
    import app.core.portfolio_manager as pmod
    from app.config import settings
    from app.core.execution.paper_broker import PaperBroker
    monkeypatch.setattr(pmod, "apply_no_trade_band", lambda w, band: w * 3.0)
    traded = []
    real_step = PaperBroker.step

    def spy(self, book, d, target_w, *a, **k):
        traded.append(float(target_w.abs().sum()))
        return real_step(self, book, d, target_w, *a, **k)
    monkeypatch.setattr(PaperBroker, "step", spy)

    def ok_gate(self, *a, **k):
        return SimpleNamespace(passed=True, sharpe=1.0, deflated_sharpe=.99, t_stat=3.0,
                               reasons=[], to_dict=lambda: {"passed": True})
    loop, ds, _ = _loop(tmp_path, monkeypatch, gate_block=False, evaluate=ok_gate)
    out = loop.run_portfolio(ds)
    assert traded and max(traded) <= float(settings.risk_max_gross) + 1e-9, max(traded)
    assert out["post_band_violations"] == []


# ---------------------------------------------------------------------------
# F09：硬门
# ---------------------------------------------------------------------------

def _loop(tmp_path, monkeypatch, gate_eval=True, gate_block=True, experiment=True,
          evaluate=None, grade=None):
    from tests.conftest import _make_dataset
    from app.config import settings
    from app.core.execution.paper_broker import PaperBroker
    from app.core.portfolio_manager.strategy_gate import StrategyGate
    from app.db.position_store import PositionStore
    from app.tasks.daily_trading_loop import DailyTradingLoop
    monkeypatch.setattr(settings, "database_url", f"sqlite:///{tmp_path / 'a.db'}")
    monkeypatch.setattr(settings, "pm_strategy_gate_eval", gate_eval)
    monkeypatch.setattr(settings, "pm_strategy_gate_block", gate_block)
    monkeypatch.setattr(settings, "tr_experiment_mode", experiment)
    if evaluate is not None:
        monkeypatch.setattr(StrategyGate, "evaluate", evaluate)
    if grade is not None:
        monkeypatch.setattr("app.core.lifecycle.promotion_gate.grade_paper_entry", grade)
    store = SimpleNamespace(query=lambda **k: [SimpleNamespace(id=1, status="paper",
                                                               dsl="rank(close)")],
                            record_ic=lambda *a, **k: None)
    loop = DailyTradingLoop(store=store,
                            broker=PaperBroker(store=PositionStore(f"sqlite:///{tmp_path / 'p.db'}")),
                            monitor=SimpleNamespace(check_decay=lambda *a: None))
    calls = []
    monkeypatch.setattr(loop, "_execute_live", lambda *a, **k: (calls.append("exec") or {}, None))
    monkeypatch.setattr(loop, "_maintain_live", lambda r: calls.append(f"maintain:{r}") or {})
    return loop, _make_dataset(n_days=35, n_tickers=10), calls


def test_hard_gate_exception_does_not_reach_execution(tmp_path, monkeypatch):
    def fail(*a, **k):
        raise RuntimeError("gate unavailable")
    loop, ds, calls = _loop(tmp_path, monkeypatch, evaluate=fail)
    out = loop.run_portfolio(ds)
    assert out["reason"] == "strategy_gate_error" and calls == ["maintain:strategy_gate_error"]
    assert out["strategy_verdict"]["evaluated"] is False


def test_soft_gate_exception_is_recorded_but_not_blocking(tmp_path, monkeypatch):
    def fail(*a, **k):
        raise RuntimeError("gate unavailable")
    loop, ds, calls = _loop(tmp_path, monkeypatch, gate_block=False, evaluate=fail)
    out = loop.run_portfolio(ds)
    assert "exec" in calls or any(c.startswith("maintain:no_active") for c in calls)
    assert out["strategy_verdict"]["gate_error"] == "gate unavailable"


def test_block_without_eval_still_evaluates(tmp_path, monkeypatch):
    seen = []

    def ev(self, *a, **k):
        seen.append(1)
        raise RuntimeError("x")
    loop, ds, calls = _loop(tmp_path, monkeypatch, gate_eval=False, evaluate=ev)
    loop.run_portfolio(ds)
    assert seen == [1], "开了硬门却关了评估 —— 应按更严的一侧照样评估"


def test_non_experiment_mode_honours_the_grade(tmp_path, monkeypatch):
    def ok_gate(self, *a, **k):
        return SimpleNamespace(passed=True, sharpe=1.0, deflated_sharpe=.99, t_stat=3.0,
                               reasons=[], to_dict=lambda: {"passed": True})
    loop, ds, calls = _loop(tmp_path, monkeypatch, gate_block=False, experiment=False,
                            evaluate=ok_gate,
                            grade=lambda v: (False, {"grade": "C", "experiment_mode": False}))
    out = loop.run_portfolio(ds)
    assert out["reason"] == "paper_grade_not_allowed"
    assert calls == ["maintain:paper_grade_not_allowed"]


def test_grading_failure_blocks_only_outside_experiment_mode(tmp_path, monkeypatch):
    def ok_gate(self, *a, **k):
        return SimpleNamespace(passed=True, sharpe=1.0, deflated_sharpe=.99, t_stat=3.0,
                               reasons=[], to_dict=lambda: {"passed": True})

    def broken(v):
        raise RuntimeError("grader down")
    loop, ds, calls = _loop(tmp_path, monkeypatch, gate_block=False, experiment=False,
                            evaluate=ok_gate, grade=broken)
    assert loop.run_portfolio(ds)["reason"] == "paper_grading_error"
    loop, ds, calls = _loop(tmp_path, monkeypatch, gate_block=False, experiment=True,
                            evaluate=ok_gate, grade=broken)
    assert loop.run_portfolio(ds).get("reason") != "paper_grading_error"


# ---------------------------------------------------------------------------
# F10：组合权重的时间结构
# ---------------------------------------------------------------------------

def _signals(n=100, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-01", periods=n)
    cols = [f"T{i}" for i in range(10)]
    a = pd.DataFrame(rng.normal(size=(n, 10)), index=idx, columns=cols)
    b = pd.DataFrame(rng.normal(size=(n, 10)), index=idx, columns=cols)
    px = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, .01, (n, 10)), 0), index=idx, columns=cols)
    return {"a": a, "b": b}, px


def test_walk_forward_fold_weights_never_see_their_own_fold():
    from app.core.portfolio_manager.manager import PortfolioManager
    sig, px = _signals()
    pm = PortfolioManager(aum=1e6, method="ic_weighted", long_only=True)
    _, _, comp = pm.walk_forward_composite(sig, px, n_folds=5)
    # 改动第 3 段起点之后的价格（因而改动那之后的收益）：前 3 段的合成信号必须完全不变
    start3 = sig["a"].index[np.array_split(np.arange(100), 5)[2][0]]
    px2 = px.copy()
    late = px2.index >= start3
    # 逐股票随机扰动（同一乘数施加给所有股票不改横截面排序，IC 也就不变 —— 那样测不出东西）
    shock = np.cumprod(1 + np.random.default_rng(99).normal(0, .05, (late.sum(), 10)), 0)
    px2.loc[late] = px2.loc[late].to_numpy() * shock
    _, _, comp2 = pm.walk_forward_composite(sig, px2, n_folds=5)
    end3 = sig["a"].index[np.array_split(np.arange(100), 5)[2][-1]]
    pd.testing.assert_frame_equal(comp.loc[:end3], comp2.loc[:end3])
    assert not comp.loc[end3:].iloc[1:].equals(comp2.loc[end3:].iloc[1:]), "用例失去判别力"


def test_first_fold_is_equal_weight():
    from app.core.backtest_engine.alpha_combiner import AlphaCombiner
    from app.core.portfolio_manager.manager import PortfolioManager
    sig, px = _signals()
    _, _, comp = PortfolioManager(aum=1e6, long_only=True).walk_forward_composite(sig, px, 5)
    first = sig["a"].index[:20]
    eq = AlphaCombiner().combine({k: v.loc[first] for k, v in sig.items()},
                                 weights={"a": .5, "b": .5})
    pd.testing.assert_frame_equal(comp.loc[first], eq)


def test_frozen_weights_are_prefix_invariant():
    """冻结权重：追加新数据不改写历史合成（审计复现：全样本重拟合会把历史从 A 翻成 B）。"""
    from app.core.portfolio_manager.manager import PortfolioManager
    sig, px = _signals()
    pm = PortfolioManager(aum=1e6, long_only=True)
    fixed = {"a": .7, "b": .3}
    short = {k: v.iloc[:60] for k, v in sig.items()}
    _, cw1, c1 = pm.combined_weights(short, px.iloc[:60], fixed)
    _, cw2, c2 = pm.combined_weights(sig, px, fixed)
    assert cw1 == cw2 == fixed
    pd.testing.assert_frame_equal(c1, c2.iloc[:60])


def test_frozen_weights_must_cover_every_factor():
    from app.core.portfolio_manager.manager import PortfolioManager
    sig, px = _signals()
    with pytest.raises(ValueError, match="缺少因子"):
        PortfolioManager(aum=1e6, long_only=True).combined_weights(sig, px, {"a": 1.0})


def test_strategy_gate_asks_for_walk_forward_returns(monkeypatch):
    import app.core.portfolio_manager.strategy_gate as sg
    seen = {}

    def spy(*a, **k):
        seen.update(k)
        raise RuntimeError("stop")
    monkeypatch.setattr(sg, "strategy_net_returns", spy)
    sg.StrategyGate(use_global_trials=False, n_segments=4).evaluate({"a": pd.DataFrame()}, {})
    assert seen["walk_forward_folds"] == 4


# ---------------------------------------------------------------------------
# F12：PIT 写入中断
# ---------------------------------------------------------------------------

def test_interrupted_pit_write_keeps_the_existing_partition(monkeypatch, tmp_path):
    from app.core.data_engine.pit_store import PITStore
    s = PITStore(tmp_path)
    ds = {"close": pd.DataFrame([[100.]], index=pd.to_datetime(["2025-03-03"]), columns=["AAA"])}
    s.append(ds, as_of="2025-03-03T23:00:00Z", name="us")
    partition = tmp_path / "us" / "year=2025" / "data.parquet"
    before = partition.read_bytes()
    real = pd.DataFrame.to_parquet

    def partial_write(self, path, *a, **k):
        from pathlib import Path
        Path(path).write_bytes(b"PAR1-incomplete")
        raise OSError("simulated-interrupted-write")
    monkeypatch.setattr(pd.DataFrame, "to_parquet", partial_write)
    ds2 = {"close": pd.DataFrame([[101.]], index=pd.to_datetime(["2025-03-04"]), columns=["AAA"])}
    with pytest.raises(OSError):
        s.append(ds2, as_of="2025-03-04T23:00:00Z", name="us")
    monkeypatch.setattr(pd.DataFrame, "to_parquet", real)
    assert partition.read_bytes() == before, "写入中断毁掉了已有的历史分区"
    assert not list(partition.parent.glob("*.tmp")), "临时文件没有清理"
    assert s.latest_timestamp("us") == pd.Timestamp("2025-03-03")


def test_a_short_read_back_is_rejected(monkeypatch, tmp_path):
    from app.core.data_engine import pit_store
    real_read = pd.read_parquet
    monkeypatch.setattr(pit_store.pd, "read_parquet",
                        lambda p, *a, **k: real_read(p, *a, **k).iloc[:0]
                        if str(p).endswith(".tmp") else real_read(p, *a, **k))
    s = pit_store.PITStore(tmp_path)
    ds = {"close": pd.DataFrame([[100.]], index=pd.to_datetime(["2025-03-03"]), columns=["AAA"])}
    with pytest.raises(OSError, match="校验失败"):
        s.append(ds, as_of="2025-03-03T23:00:00Z", name="us")
    assert not (tmp_path / "us" / "year=2025" / "data.parquet").exists()


def test_backfill_pit_failure_rejects_the_ingest(monkeypatch, tmp_path):
    from app.tasks.daily_ingest import DailyIngest
    import app.core.data_engine.dataset_registry as dr
    from tests.conftest import _make_dataset
    data = _make_dataset(n_days=30, n_tickers=5)
    monkeypatch.setattr(dr, "load_registry_dataset",
                        lambda *a, **k: SimpleNamespace(data=data))
    monkeypatch.setattr(dr, "check_dataset_health", lambda *a, **k: None)

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(DailyIngest, "_append_pit", staticmethod(boom))
    r = DailyIngest().ingest("d", "2022-01-03", "2022-03-01", pit_required=True)
    assert not r.accepted and "pit_append_failed" in r.reject_reason
    r = DailyIngest().ingest("d", "2022-01-03", "2022-03-01")      # 非回填：只告警
    assert r.accepted


# ---------------------------------------------------------------------------
# F16：因子级 IC 的前向标记
# ---------------------------------------------------------------------------

def test_per_factor_ic_carries_the_forward_flag(tmp_path):
    from tests.conftest import _make_dataset
    from app.core.execution.paper_broker import PaperBroker
    from app.db.position_store import PositionStore
    from app.tasks.daily_trading_loop import DailyTradingLoop
    seen = []
    monitor = SimpleNamespace(
        update=lambda aid, d, ic, realized_return=0.0, is_forward=False: seen.append((d, is_forward)),
        check_decay=lambda *a: None)
    store = SimpleNamespace(query=lambda **k: [SimpleNamespace(id=7, status="paper",
                                                               dsl="rank(close)")])
    loop = DailyTradingLoop(store=store, monitor=monitor,
                            broker=PaperBroker(store=PositionStore(f"sqlite:///{tmp_path / 'p.db'}")))
    ds = _make_dataset(n_days=30, n_tickers=8)
    cut = ds["close"].index[25]
    loop.run(ds, forward_from=str(cut.date()))
    assert seen
    assert all(f is (d >= cut) for d, f in seen), "forward_from 之后的日子没标前向 / 之前的被标了"
    assert any(f for _, f in seen) and not all(f for _, f in seen)


# ---------------------------------------------------------------------------
# 逐点变异复核补的用例
# ---------------------------------------------------------------------------

def test_net_exactly_at_the_limit_is_left_alone():
    """净敞口恰好 = 上限（2 的幂，浮点精确）：不缩、也不记成"被缩放过"。"""
    gate = PortfolioRiskGate(RiskLimits(max_gross=1, max_net=.25, max_name_weight=1,
                                        max_sector_weight=10, long_only=True))
    w = pd.DataFrame([[.125, .125]], columns=["A", "B"])
    out, rep = gate.apply(w, sectors=pd.Series({"A": 1, "B": 2}))
    pd.testing.assert_frame_equal(out, w)
    assert rep.n_gross_scaled == 0


def test_net_projection_is_reported_as_scaling():
    gate = PortfolioRiskGate(RiskLimits(max_gross=1, max_net=.125, max_name_weight=1,
                                        max_sector_weight=10, long_only=True))
    out, rep = gate.apply(pd.DataFrame([[.125, .125]], columns=["A", "B"]),
                          sectors=pd.Series({"A": 1, "B": 2}))
    assert out.iloc[0].tolist() == [.0625, .0625] and rep.n_gross_scaled == 1


def test_net_projection_leaves_zero_weights_at_zero():
    gate = PortfolioRiskGate(RiskLimits(max_gross=2, max_net=.25, max_name_weight=1,
                                        max_sector_weight=10, long_only=False))
    out, _ = gate.apply(pd.DataFrame([[.5, 0.0, -.125]], columns=["A", "B", "C"]),
                        sectors=pd.Series({"A": 1, "B": 2, "C": 3}))
    assert out.iloc[0].tolist() == [.375, 0.0, -.125]


def test_walk_forward_fits_only_on_data_before_each_fold(monkeypatch):
    """直接看拟合函数收到了什么：每一段的拟合数据必须**严格早于**本段起点。"""
    from app.core.backtest_engine.alpha_combiner import AlphaCombiner
    from app.core.portfolio_manager.manager import PortfolioManager
    sig, px = _signals()
    seen = []
    real = AlphaCombiner.optimize_weights

    def spy(self, signals, returns=None, method="ic_weighted", **k):
        seen.append((max(s.index.max() for s in signals.values()), returns.index.max()))
        return real(self, signals, returns=returns, method=method, **k)
    monkeypatch.setattr(AlphaCombiner, "optimize_weights", spy)
    PortfolioManager(aum=1e6, long_only=True).walk_forward_composite(sig, px, n_folds=5)
    starts = [sig["a"].index[p[0]] for p in np.array_split(np.arange(100), 5)][1:]
    assert len(seen) == 4, "第 1 段不该拟合（等权），其余 4 段各拟合一次"
    for (sig_max, ret_max), start in zip(seen, starts):
        assert sig_max < start and ret_max < start, (sig_max, ret_max, start)
        assert sig_max == sig["a"].index[sig["a"].index.get_loc(start) - 1]


def test_a_single_factor_is_not_walk_forward_fitted(monkeypatch):
    from app.core.portfolio_manager.manager import PortfolioManager
    sig, px = _signals()
    called = []
    monkeypatch.setattr(PortfolioManager, "walk_forward_composite",
                        lambda self, *a, **k: called.append(1))
    vol = pd.DataFrame(1e6, index=px.index, columns=px.columns)
    PortfolioManager(aum=1e6, long_only=True).build_book({"a": sig["a"]}, px, vol,
                                                         walk_forward_folds=5)
    assert called == [], "单因子没有可合成的权重，不该走滚动拟合"


def test_monitor_update_defaults_to_replay(tmp_path):
    """不传 is_forward 时必须记成**回放**（保守方向）—— 前向样本只能由调用方明确声明。"""
    from app.core.monitor.alpha_monitor import AlphaMonitor
    from app.db.alpha_store import AlphaResult, AlphaStore
    store = AlphaStore(db_url=f"sqlite:///{tmp_path / 'm.db'}")
    aid = store.save(AlphaResult(dsl="rank(close)", status="candidate"))
    AlphaMonitor(store).update(aid, "2025-03-03", 0.1)
    assert store.get_forward_ic(aid) == []
    AlphaMonitor(store).update(aid, "2025-03-04", 0.1, is_forward=True)
    assert [str(h.date) for h in store.get_forward_ic(aid)] == ["2025-03-04"]


def test_pit_partitions_round_trip_without_an_index_column(tmp_path):
    import pyarrow.parquet as pq
    from app.core.data_engine.pit_store import PITStore
    s = PITStore(tmp_path)
    ds = {"close": pd.DataFrame([[100.]], index=pd.to_datetime(["2025-03-03"]), columns=["AAA"])}
    s.append(ds, as_of="2025-03-03T23:00:00Z", name="us")
    cols = pq.read_schema(tmp_path / "us" / "year=2025" / "data.parquet").names
    assert cols == ["timestamp", "ticker", "close", "as_of"], cols
