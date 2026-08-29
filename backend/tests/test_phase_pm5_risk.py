"""
test_phase_pm5_risk.py — Phase PM.5 组合级风控门 验收

- 单票上限：超配名被削到 ≤ 上限
- 行业集中度：押注单一行业被缩到 ≤ 上限
- gross 上限：总敞口被缩到 ≤ 上限
- long-only：负权重被清零
- 目标波动：按估计波动缩放整体仓位
- 回撤熔断：净值回撤超阈值触发
- check()：只报违规不改权重
"""

from __future__ import annotations

import pandas as pd

from app.core.portfolio_manager import PortfolioRiskGate, RiskLimits


def _wdf(row: dict, n_days=3):
    cols = list(row)
    idx = pd.bdate_range("2024-01-02", periods=n_days)
    return pd.DataFrame([row] * n_days, index=idx)[cols].astype(float)


def test_name_cap_enforced():
    w = _wdf({"A": 0.5, "B": 0.3, "C": 0.2})           # A 超 10% 上限
    gate = PortfolioRiskGate(RiskLimits(max_name_weight=0.10, max_gross=1.0,
                                        max_sector_weight=1.0))
    adj, rep = gate.apply(w, sectors=pd.Series({"A": 1, "B": 2, "C": 3}))
    assert adj.abs().max().max() <= 0.10 + 1e-9
    assert rep.n_name_clipped > 0


def test_sector_concentration_enforced():
    # A,B 同行业(1) 合计 0.8 → 超 30% 行业上限
    w = _wdf({"A": 0.4, "B": 0.4, "C": 0.2})
    sectors = pd.Series({"A": 1, "B": 1, "C": 2})
    gate = PortfolioRiskGate(RiskLimits(max_name_weight=1.0, max_gross=1.0,
                                        max_sector_weight=0.30))
    adj, rep = gate.apply(w, sectors=sectors)
    last = adj.iloc[-1]
    # 行业为 NAV 比例绝对上限：sector1 绝对权重 ≤ 0.30×max_gross
    sec1_abs = last[["A", "B"]].abs().sum()
    assert sec1_abs <= 0.30 + 1e-6
    assert rep.n_sector_scaled > 0


def test_gross_cap_enforced():
    w = _wdf({"A": 0.6, "B": 0.6, "C": 0.6})           # gross=1.8 > 1.0
    gate = PortfolioRiskGate(RiskLimits(max_name_weight=1.0, max_gross=1.0,
                                        max_sector_weight=1.0))
    adj, rep = gate.apply(w, sectors=pd.Series({"A": 1, "B": 2, "C": 3}))
    assert (adj.abs().sum(axis=1) <= 1.0 + 1e-9).all()
    assert rep.n_gross_scaled > 0


def test_long_only_removes_negatives():
    w = _wdf({"A": 0.5, "B": -0.5})
    gate = PortfolioRiskGate(RiskLimits(long_only=True, max_name_weight=1.0,
                                        max_gross=1.0, max_sector_weight=1.0))
    adj, _ = gate.apply(w, sectors=pd.Series({"A": 1, "B": 2}))
    assert (adj >= -1e-12).all().all()


def test_target_vol_scales_down_when_over():
    w = _wdf({"A": 0.5, "B": 0.5})                      # gross=1.0
    gate = PortfolioRiskGate(RiskLimits(target_vol_ann=0.10, max_name_weight=1.0,
                                        max_gross=1.0, max_sector_weight=1.0))
    # 组合估计年化波动 20% > 目标 10% → 缩一半
    adj, rep = gate.apply(w, sectors=pd.Series({"A": 1, "B": 2}), port_vol_ann=0.20)
    assert abs(rep.vol_scalar - 0.5) < 1e-6
    assert adj.abs().sum(axis=1).iloc[-1] <= 0.5 + 1e-9


def test_drawdown_circuit_breaker():
    gate = PortfolioRiskGate(RiskLimits(max_drawdown=0.20))
    eq_ok = pd.Series([100, 102, 101, 103], dtype=float)          # 回撤 <20%
    eq_bad = pd.Series([100, 120, 110, 90], dtype=float)          # 峰值120→90 = 25%
    assert gate.should_halt(eq_ok)[0] is False
    halt, dd = gate.should_halt(eq_bad)
    assert halt is True and dd >= 0.20


def test_check_reports_violations_without_mutating():
    w = _wdf({"A": 0.5, "B": 0.5, "C": 0.5})           # gross=1.5, 单票 0.5
    gate = PortfolioRiskGate(RiskLimits(max_gross=1.0, max_name_weight=0.10,
                                        max_sector_weight=1.0))
    viol = gate.check(w, sectors=pd.Series({"A": 1, "B": 2, "C": 3}))
    assert any("gross" in v for v in viol)
    assert any("单票" in v for v in viol)
    # 原权重未被改动
    assert w.abs().max().max() == 0.5
