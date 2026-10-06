"""
test_monthly_fidelity.py — Phase 12.3 保真度报告进入月度调度主线（§K）

  · run_monthly_fidelity：区间内没有纸交易成交 → None；有 → 出报告并写文件
  · 开盘价多取 10 天（月末决策日的单在下月第一个交易日开盘成交）
  · 月度任务只在执行层开着时才出保真度报告
"""

from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from app.core.execution.broker_gateway import BrokerOrder
from app.core.execution.order_builder import PlannedOrder
from app.db.execution_store import ExecutionStore

D1, D2 = date(2025, 3, 31), date(2025, 4, 1)


@pytest.fixture
def db(tmp_path, monkeypatch):
    from app.config import settings
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(settings, "database_url", f"sqlite:///{tmp_path / 'm.db'}")
    return ExecutionStore()


def _seed_fill(es: ExecutionStore):
    po = PlannedOrder("qaR-1-20250331-AAA-B", "AAA", "BUY", 10, 50.0, 50.25, 0, 10, 0.05)
    es.add_intent(po, -1, D1)
    es.mark_submitted(po.client_id, "1")
    es.apply_broker_state(po.client_id, BrokerOrder(
        "1", po.client_id, "AAA", "BUY", 10, 50.25, "FILLED_ALL", 10, 50.6, "", "", ""), D2)


def test_no_live_fills_means_no_report(db, monkeypatch):
    from app.tasks import cost_calibration as cc
    import app.core.data_engine.dataset_registry as reg
    monkeypatch.setattr(reg, "load_registry_dataset",
                        lambda *a, **k: pytest.fail("没有成交还去加载数据集"))
    assert cc.run_monthly_fidelity("us_tech_large", "2025-03-01", "2025-03-31") is None


def test_report_is_built_and_written(db, monkeypatch, tmp_path):
    from app.tasks import cost_calibration as cc
    import app.core.data_engine.dataset_registry as reg
    _seed_fill(db)
    seen = {}
    opens = pd.DataFrame({"AAA": [50.5]}, index=pd.DatetimeIndex(["2025-04-01"]))

    def _load(name, start, end, health_check):
        seen.update(name=name, start=start, end=end, health_check=health_check)
        return SimpleNamespace(data={"open": opens})
    monkeypatch.setattr(reg, "load_registry_dataset", _load)
    out = tmp_path / "fid.md"
    rep = cc.run_monthly_fidelity("us_tech_large", "2025-03-01", "2025-03-31", write_path=str(out))
    assert seen == {"name": "us_tech_large", "start": "2025-03-01", "end": "2025-04-10",
                    "health_check": False}
    assert rep.n_live_fills == 1
    assert rep.total_slippage_bps["median"] == pytest.approx(120.0)     # (50.6−50)/50
    assert rep.overnight_gap_bps["median"] == pytest.approx(100.0)      # (50.5−50)/50
    assert "保真度报告" in out.read_text(encoding="utf-8")


def test_fills_outside_the_month_are_ignored(db, monkeypatch):
    from app.tasks import cost_calibration as cc
    import app.core.data_engine.dataset_registry as reg
    _seed_fill(db)                                                       # 决策日 3/31
    monkeypatch.setattr(reg, "load_registry_dataset",
                        lambda *a, **k: pytest.fail("区间外的成交触发了报告"))
    assert cc.run_monthly_fidelity("x", "2025-04-01", "2025-04-30") is None


@pytest.mark.parametrize("mode,expect_call", [("off", False), ("moomoo_paper", True)])
def test_monthly_job_adds_fidelity_only_when_execution_is_on(monkeypatch, mode, expect_call):
    from app.config import settings
    from app.tasks import cost_calibration as cc
    from app.tasks import scheduler
    monkeypatch.setattr(settings, "execution_mode", mode)
    monkeypatch.setattr(cc, "run_monthly_calibration", lambda *a, **k: None)
    calls = []

    def _fid(dataset, start, end, write_path=None):
        calls.append((dataset, start, end, write_path))
        return None
    monkeypatch.setattr(cc, "run_monthly_fidelity", _fid)
    scheduler.monthly_cost_calibration_job()
    if not expect_call:
        assert calls == []
        return
    first_this = date.today().replace(day=1)
    last_prev = first_this - timedelta(days=1)
    first_prev = last_prev.replace(day=1)
    assert calls == [(settings.paper_dataset, first_prev.isoformat(), last_prev.isoformat(),
                      f"exec_fidelity_{first_prev:%Y%m}.md")]
