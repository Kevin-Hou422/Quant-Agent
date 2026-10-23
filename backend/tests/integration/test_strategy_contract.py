"""
复审 2026-10-19 N05-e（§3）：显式三步合同 —— 构建并冻结 spec → 验证该 spec → 以该 spec 生成目标 / 证据 / 订单；
回测与实时执行用不同的成交适配器，但共享同一信号、权重和风险决策过程。

此前四处各自装配（提案、策略门、每日循环、前向证据）。实测的分叉：门按 grounded 成本参数建合成账本
（影响容量），每日循环用默认 CostParams() —— 容量绑定时门验证的权重和每日循环交易的不是同一组。

断言：
  1. 结构：合成账本 / 风控 / 带后投影只在 strategy_contract.generate_targets 里装配；
  2. 提案：先冻结合同，门评估的就是它，存下的规格就是被评估的那份；风控快照走同一条路径；
  3. 每日循环：按 StrategyContract 生成目标；循环里的策略门与执行用的是**同一组**目标参数，
     券商目标 = 合同生成的风控后目标；
  4. 证据：按合同算合成信号；
  5. 规格覆盖波动估计窗口（它改变目标权重）。
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from tests.conftest import _make_dataset
from tests.integration.test_phase12_execution_wiring import AUM, _activate_strategy, _dataset, loop_env  # noqa: F401

APP = Path(__file__).resolve().parents[2] / "app"


def test_targets_are_assembled_in_one_place_only():
    allowed = {"strategy_contract.py", "executable_spec.py", "manager.py", "risk_gate.py"}
    pat = re.compile(r"\.build_book\(|PortfolioRiskGate\([^)]*\)\.apply\(|band_then_reproject\(")
    offenders = []
    for p in APP.rglob("*.py"):
        if p.name in allowed:
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            s = line.strip()
            if pat.search(s) and not s.startswith(("#", "def ")):
                offenders.append(f"{p.relative_to(APP)}:{i}: {s}")
    assert not offenders, "目标权重还有别的装配处（应统一经 generate_targets）：\n" + "\n".join(offenders)


# ---------------------------------------------------------------------------
# 提案：冻结 → 验证
# ---------------------------------------------------------------------------

def test_the_proposal_freezes_the_contract_before_the_gate_sees_it():
    from app.core.portfolio_manager import strategy_contract as sc
    from app.core.portfolio_manager.strategy_builder import build_strategy_config
    from app.core.portfolio_manager.strategy_gate import StrategyGate
    ds = _make_dataset(80, 10)
    order, frozen = [], []
    real_freeze = sc.StrategyContract.freeze.__func__

    def spy_freeze(cls, *a, **kw):
        c = real_freeze(cls, *a, **kw)
        order.append("freeze")
        frozen.append(c)
        return c

    def ev(self, *a, **kw):
        order.append("gate")
        return SimpleNamespace(passed=True, sharpe=1., deflated_sharpe=.99, t_stat=3., reasons=[],
                               to_dict=lambda: {"passed": True})
    with patch.object(sc.StrategyContract, "freeze", classmethod(spy_freeze)), \
            patch.object(StrategyGate, "evaluate", ev):
        cfg = build_strategy_config({"1": ds["close"].rank(axis=1)}, ds, 10000, method="equal_weight",
                                    factor_dsl={"1": "rank(close)"})
    assert order[0] == "freeze" and "gate" in order, order
    assert len(frozen) == 1
    assert cfg.verdict["spec_hash"] == frozen[0].hash, "存下的规格不是被门评估的那份"
    assert cfg.verdict["evaluated_spec_hash"] == frozen[0].hash


def test_the_proposals_risk_snapshot_uses_the_same_target_path():
    from app.core.portfolio_manager import strategy_contract as sc
    from app.core.portfolio_manager.strategy_builder import build_strategy_config
    from app.core.portfolio_manager.strategy_gate import StrategyGate
    ds = _make_dataset(80, 10)
    seen = []
    real = sc.generate_targets

    def spy(*a, **kw):
        seen.append(kw)
        return real(*a, **kw)

    def ev(self, *a, **kw):
        return SimpleNamespace(passed=True, sharpe=1., deflated_sharpe=.99, t_stat=3., reasons=[],
                               to_dict=lambda: {"passed": True})
    with patch.object(sc, "generate_targets", spy), patch.object(StrategyGate, "evaluate", ev):
        cfg = build_strategy_config({"1": ds["close"].rank(axis=1)}, ds, 10000, method="equal_weight",
                                    factor_dsl={"1": "rank(close)"})
    assert seen, "提案的风控快照没有走 generate_targets"
    assert seen[-1]["band"] == pytest.approx(cfg.no_trade_band)
    assert seen[-1]["fixed_combo_weights"] == pytest.approx(cfg.combo_weights)


# ---------------------------------------------------------------------------
# 每日循环：以合同生成目标；门与执行同一组参数
# ---------------------------------------------------------------------------

def test_the_daily_loop_trades_the_contracts_targets(loop_env, monkeypatch, tmp_path):
    from app.core.portfolio_manager import strategy_contract as sc
    from app.db.strategy_store import StrategyStore
    ctx, made, build, settings = loop_env
    monkeypatch.setattr(settings, "pm_strategy_gate_eval", True)
    loop = build()
    _, sid = _activate_strategy(loop, settings, monkeypatch, tmp_path)
    ds = _dataset()
    calls = []
    real = sc.generate_targets

    def spy(*a, **kw):
        out = real(*a, **kw)
        calls.append((kw, out, a))
        return out
    monkeypatch.setattr(sc, "generate_targets", spy)
    out = loop.run_portfolio(ds, aum=AUM)
    assert out["active_config"] == sid, out.get("reason")

    contract = sc.StrategyContract.from_record(StrategyStore().get(sid))
    assert out.get("spec_hash") == contract.hash
    # 门里的 PBO 会为**每个单因子**各建一个等权策略（反事实对照，不是这份策略的目标）—— 排除它们，
    # 但要证明被排除的确实只是单因子的那几次，没有把这份策略的某次调用藏起来
    others = [c for c in calls if c[0].get("fixed_combo_weights") is None]
    assert all(len(c[2][0]) == 1 and c[0]["method"] == "equal_weight" for c in others),         [(c[0]["method"], list(c[2][0])) for c in others]
    calls = [c for c in calls if c[0].get("fixed_combo_weights") is not None]
    assert len(calls) >= 2, "循环里的策略门和执行没有都走 generate_targets"
    keys = ("method", "band", "fixed_combo_weights", "limits", "vol_lookback")
    first = {k: calls[0][0][k] for k in keys}
    for kw, *_ in calls[1:]:
        assert {k: kw[k] for k in keys} == first, "门与执行用的目标参数不同"
    cps = [c[0].get("cost_params") for c in calls]
    assert all(c is not None for c in cps) and all(c == cps[0] for c in cps), \
        "门与执行的合成账本用了不同的成本 / 容量参数"
    # 所有调用给出同一组目标（同一份合同、同一份数据）
    w0 = calls[0][1].weights
    for c in calls[1:]:
        pd.testing.assert_frame_equal(c[1].weights, w0)
    # 券商目标 = 合同生成的风控后、带前的决策日目标
    gated_last = calls[-1][1].gated.iloc[-1]
    target = out.get("execution_target") or {}
    assert target, "执行目标为空 —— 下面的逐名比对会空转"
    for tk, w in target.items():
        assert w == pytest.approx(float(gated_last[tk]), abs=1e-12)


def test_the_strategy_gate_backtests_the_contracts_targets():
    """门回测的权重 = 合同 targets() 的权重（同一函数、同一参数），只是成交适配器是 BacktestEngine。"""
    from app.core.portfolio_manager import strategy_contract as sc
    from app.core.portfolio_manager.strategy_gate import resolve_cost_params, strategy_net_returns
    ds = _make_dataset(120, 8)
    contract = sc.StrategyContract.freeze({"a": "rank(close)"}, {"a": 1.0}, "equal_weight", 1e4, 0.01)
    sigs = contract.signals(ds)
    cp = resolve_cost_params(ds, 1e4, None)
    seen = []
    real = sc.generate_targets

    def spy(*a, **kw):
        out = real(*a, **kw)
        seen.append(out)
        return out
    with patch.object(sc, "generate_targets", spy):
        strategy_net_returns(sigs, ds, aum=1e4, method="equal_weight", cost_params=cp,
                             fixed_combo_weights=contract.combo_weights, no_trade_band=contract.band)
    direct = contract.targets(sigs, ds, cost_params=cp)
    assert len(seen) == 1
    pd.testing.assert_frame_equal(seen[0].weights, direct.weights)


# ---------------------------------------------------------------------------
# 证据：按合同算合成信号
# ---------------------------------------------------------------------------

def test_strategy_evidence_uses_the_contract(loop_env, monkeypatch, tmp_path):
    from app.core.portfolio_manager import strategy_contract as sc
    ctx, made, build, settings = loop_env
    loop = build()
    _, sid = _activate_strategy(loop, settings, monkeypatch, tmp_path)
    used = []
    real = sc.StrategyContract.composite

    def spy(self, *a, **kw):
        used.append(self.config_id)
        return real(self, *a, **kw)
    monkeypatch.setattr(sc.StrategyContract, "composite", spy)
    loop._record_strategy_evidence(_dataset(), None)
    assert sid in used, "前向证据没有按这份策略的合同算合成信号"


# ---------------------------------------------------------------------------
# 规格覆盖波动估计窗口
# ---------------------------------------------------------------------------

def test_the_vol_lookback_is_part_of_the_spec(monkeypatch):
    from app.config import settings
    from app.core.portfolio_manager.executable_spec import build_spec, spec_diff
    a = build_spec({"1": "rank(close)"}, {"1": 1.0}, "equal_weight", 1e4, 0.01)
    monkeypatch.setattr(settings, "risk_vol_lookback", int(getattr(settings, "risk_vol_lookback", 60)) + 5)
    b = build_spec({"1": "rank(close)"}, {"1": 1.0}, "equal_weight", 1e4, 0.01)
    assert any("risk_vol_lookback" in d for d in spec_diff(a, b)), "波动窗口改了，规格没变"
