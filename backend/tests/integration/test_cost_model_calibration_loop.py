"""
test_phase8_cost_calib.py — Phase 8.2 成本模型校准回路验收

覆盖：
  - calibrate() 用 T+1 开盘价量化隔夜执行缺口，产出有界 impact_coef 建议
  - 拒绝单/零成交被剔除；无可匹配 T+1 价时安全返回
  - 建议缩放被 clip 到 [0.5, 2.0]，绝不自动改 CostParams
  - 调度器在 enable_paper_trading 时注册 monthly_cost_calibration 任务
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.tasks.cost_calibration import calibrate, CalibrationReport


def _prices(dates, tickers, open_vals):
    idx = pd.DatetimeIndex(dates)
    return {"open": pd.DataFrame(open_vals, index=idx, columns=tickers)}


def test_calibrate_basic_gap_and_recommendation():
    dates = ["2026-01-05", "2026-01-06"]
    tickers = ["AAA"]
    # 成交在 1/5 收盘价 100；1/6 开盘 101 → 隔夜缺口 +100bps
    fills = pd.DataFrame([{
        "date": "2026-01-05", "ticker": "AAA",
        "filled_weight": 0.5, "fill_price": 100.0, "reject_reason": "",
    }])
    prices = _prices(dates, tickers, [[100.0], [101.0]])
    rep = calibrate(fills, prices, spread_bps=2.0, impact_coef=0.1)
    assert isinstance(rep, CalibrationReport)
    assert rep.n_fills == 1
    assert rep.realized_gap_bps_abs_median == pytest.approx(100.0, abs=1e-6)
    # |gap|=100 >> spread 2 → 缩放触上界 2.0
    assert rep.recommended_scale == pytest.approx(2.0)
    assert rep.recommended_impact_coef == pytest.approx(0.2)


def test_scale_clipped_low():
    dates = ["2026-02-02", "2026-02-03"]
    fills = pd.DataFrame([{
        "date": "2026-02-02", "ticker": "AAA",
        "filled_weight": -0.3, "fill_price": 100.0, "reject_reason": "",
    }])
    # 隔夜缺口极小（0.1bps）→ 缩放触下界 0.5
    prices = _prices(dates, ["AAA"], [[100.0], [100.001]])
    rep = calibrate(fills, prices, spread_bps=50.0, impact_coef=0.1)
    assert rep.recommended_scale == pytest.approx(0.5)
    assert rep.recommended_impact_coef == pytest.approx(0.05)


def test_rejected_and_zero_fills_excluded():
    dates = ["2026-03-02", "2026-03-03"]
    fills = pd.DataFrame([
        {"date": "2026-03-02", "ticker": "AAA", "filled_weight": 0.0,
         "fill_price": 100.0, "reject_reason": ""},              # 零成交
        {"date": "2026-03-02", "ticker": "BBB", "filled_weight": 0.5,
         "fill_price": 100.0, "reject_reason": "adv_cap"},        # 被拒
    ])
    prices = _prices(dates, ["AAA", "BBB"], [[100.0, 100.0], [102.0, 102.0]])
    rep = calibrate(fills, prices, spread_bps=2.0, impact_coef=0.1)
    assert rep is None  # 无有效成交


def test_no_next_day_price_returns_none():
    # 成交日是价格序列最后一天，无 T+1 → 无法匹配
    fills = pd.DataFrame([{
        "date": "2026-04-03", "ticker": "AAA",
        "filled_weight": 0.5, "fill_price": 100.0, "reject_reason": "",
    }])
    prices = _prices(["2026-04-03"], ["AAA"], [[100.0]])
    assert calibrate(fills, prices, spread_bps=2.0, impact_coef=0.1) is None


def test_report_markdown_renders():
    fills = pd.DataFrame([{
        "date": "2026-05-04", "ticker": "AAA",
        "filled_weight": 0.5, "fill_price": 100.0, "reject_reason": "",
    }])
    prices = _prices(["2026-05-04", "2026-05-05"], ["AAA"], [[100.0], [100.5]])
    rep = calibrate(fills, prices, spread_bps=2.0, impact_coef=0.1)
    md = rep.to_markdown()
    assert "成本模型校准报告" in md
    assert "impact_coef" in md


def test_scheduler_registers_calibration_job(monkeypatch):
    monkeypatch.setattr("app.config.settings.enable_paper_trading", True)
    from app.tasks.scheduler import create_scheduler
    sched = create_scheduler(db_url="sqlite:///:memory:")
    ids = {j.id for j in sched.get_jobs()}
    assert "monthly_cost_calibration" in ids
    assert "daily_trading" in ids
