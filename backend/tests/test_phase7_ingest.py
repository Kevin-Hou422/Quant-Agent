"""
test_phase7_ingest.py — Task 7.1 每日摄取健康门验收（2026-08-02）

核心验收 A2：数据坏（低健康分 / 加载失败）→ **拒绝当日摄取，绝不静默降级**。
用 monkeypatch 离线注入 load_registry_dataset / check_dataset_health。
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest


def _make_ds(n=40, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    cols = ["A", "B", "C"]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (n, 3)), axis=0),
                         index=idx, columns=cols)
    return {"close": close, "open": close, "high": close, "low": close,
            "volume": pd.DataFrame(1e6, index=idx, columns=cols),
            "vwap": close, "returns": close.pct_change().fillna(0.0)}


def _patch_loader(monkeypatch, data, health_score):
    import app.core.data_engine.dataset_registry as reg
    ds_obj = types.SimpleNamespace(data=data)
    monkeypatch.setattr(reg, "load_registry_dataset",
                        lambda name, start=None, end=None, health_check=False: ds_obj)
    report = types.SimpleNamespace(overall_score=health_score)
    monkeypatch.setattr(reg, "check_dataset_health",
                        lambda ds, min_score=0.7, warn_only=True: report)


class TestHealthGate:

    def test_good_data_accepted(self, monkeypatch):
        from app.tasks.daily_ingest import DailyIngest
        _patch_loader(monkeypatch, _make_ds(), health_score=0.95)
        res = DailyIngest(min_health=0.7).ingest("us_tech_large", "2023-01-01", "2023-03-01")
        assert res.accepted is True
        assert res.health_score == 0.95
        assert res.dataset is not None and "close" in res.dataset

    def test_bad_health_rejected_not_downgraded(self, monkeypatch):
        """A2：低健康分 → 拒绝，dataset 为 None（不返回坏数据继续）。"""
        from app.tasks.daily_ingest import DailyIngest
        _patch_loader(monkeypatch, _make_ds(), health_score=0.40)
        res = DailyIngest(min_health=0.7).ingest("us_tech_large", "2023-01-01", "2023-03-01")
        assert res.accepted is False
        assert res.dataset is None                      # 绝不把坏数据传下去
        assert "health_below_threshold" in res.reject_reason

    def test_load_failure_rejected(self, monkeypatch):
        """加载抛异常 → 拒绝（不静默降级到合成/过期数据）。"""
        import app.core.data_engine.dataset_registry as reg
        def _boom(*a, **k):
            raise RuntimeError("network down")
        monkeypatch.setattr(reg, "load_registry_dataset", _boom)
        from app.tasks.daily_ingest import DailyIngest
        res = DailyIngest().ingest("us_tech_large", "2023-01-01", "2023-03-01")
        assert res.accepted is False and "load_failed" in res.reject_reason

    def test_pipeline_skips_loop_on_reject(self, monkeypatch):
        """run_daily_pipeline：摄取被拒 → 不进交易循环。"""
        from app.tasks import daily_ingest as di
        _patch_loader(monkeypatch, _make_ds(), health_score=0.30)
        # 若进了循环会实例化 DailyTradingLoop（连真实库）——被拒则不应发生
        out = di.run_daily_pipeline("us_tech_large", "2023-01-01", "2023-03-01")
        assert out["ingest_accepted"] is False
        assert "health_below_threshold" in out["reject_reason"]
