"""
test_phase9_validation_gate.py — Phase 9.3 自动验证门验收

覆盖：
  - 强预测信号数据集 → 全折 OOS 为正 → 通过（隔离 DSR 阈值以确定性验证 WalkForward 逻辑）
  - 纯噪声数据集 → 不通过，且给出原因
  - DSR 阈值过高 → 不通过（原因含 DSR）
  - 坏 DSL → fail-closed（判不通过，不抛出、不静默放行）
  - 结果结构 to_dict 完整
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.lifecycle.validation_gate import ValidationGate, ValidationResult

_FIELDS = ["open", "high", "low", "close", "volume", "vwap", "returns"]


def _panel_from_close(close: pd.DataFrame) -> dict:
    ret = np.log(close / close.shift(1))
    vol = pd.DataFrame(1e6, index=close.index, columns=close.columns)
    return {
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": vol, "vwap": close, "returns": ret,
    }


def _signal_dataset(T=520, N=20, seed=0) -> dict:
    """每资产有固定 quality q_i，日收益 = 0.0015*q_i + 噪声 → 动量因子逐折 OOS 为正。"""
    rng = np.random.default_rng(seed)
    q = np.linspace(-1, 1, N)
    ret = 0.0015 * q[None, :] + rng.normal(0, 0.008, (T, N))
    close = pd.DataFrame(
        100 * np.cumprod(1 + ret, axis=0),
        index=pd.bdate_range("2021-01-01", periods=T),
        columns=[f"A{i:02d}" for i in range(N)],
    )
    return _panel_from_close(close)


def _noise_dataset(T=520, N=20, seed=1) -> dict:
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.01, (T, N))
    close = pd.DataFrame(
        100 * np.cumprod(1 + ret, axis=0),
        index=pd.bdate_range("2021-01-01", periods=T),
        columns=[f"A{i:02d}" for i in range(N)],
    )
    return _panel_from_close(close)


_MOMENTUM = "rank(ts_mean(returns,20))"


def test_strong_signal_passes_walkforward():
    # 隔离 DSR（阈值设为 0）→ 只考验 WalkForward 全折为正逻辑
    gate = ValidationGate(n_splits=3, embargo_days=5, dsr_threshold=0.0)
    res = gate.evaluate(_MOMENTUM, _signal_dataset(), n_trials=1)
    assert isinstance(res, ValidationResult)
    assert res.n_folds >= 3
    assert res.min_oos_sharpe > 0.0, res.reasons        # 逐折 OOS 为正
    assert res.pct_folds_positive == pytest.approx(1.0)
    assert res.passed is True, res.reasons


def test_noise_fails():
    gate = ValidationGate(n_splits=3, embargo_days=5, dsr_threshold=0.90)
    res = gate.evaluate(_MOMENTUM, _noise_dataset(), n_trials=50)
    assert res.passed is False
    assert len(res.reasons) >= 1


def test_high_dsr_threshold_blocks():
    # 即便信号强，DSR 阈值抬到 0.999 也应拦下（原因含 DSR）
    gate = ValidationGate(n_splits=3, embargo_days=5, dsr_threshold=0.999)
    res = gate.evaluate(_MOMENTUM, _signal_dataset(), n_trials=200)
    assert res.passed is False
    assert any("DSR" in r for r in res.reasons)


def test_bad_dsl_fails_closed():
    gate = ValidationGate(n_splits=3, embargo_days=5, dsr_threshold=0.90)
    res = gate.evaluate("rank(nonexistent_op(close))", _signal_dataset())
    assert res.passed is False            # 不抛出、不放行
    assert len(res.reasons) >= 1


def test_result_to_dict_complete():
    gate = ValidationGate(n_splits=3, embargo_days=5, dsr_threshold=0.0)
    d = gate.evaluate(_MOMENTUM, _signal_dataset()).to_dict()
    for k in ("passed", "dsl", "reasons", "n_folds", "min_oos_sharpe",
              "deflated_sharpe", "n_trials"):
        assert k in d


# --------------------------------------------------------------------------
# 验证门端点 POST /api/alphas/{id}/validate（wiring 测试，monkeypatch 掉重回测）
# --------------------------------------------------------------------------

def test_validate_endpoint_promotes_on_pass(test_client, tmp_path, monkeypatch):
    from app.api.router import get_store
    from app.main import app
    from app.db.alpha_store import AlphaStore, AlphaResult

    store = AlphaStore(db_url=f"sqlite:///{tmp_path/'v.db'}")
    aid = store.save(AlphaResult(dsl="rank(close)"))          # 默认 candidate
    assert store.get_by_id(aid).status == "candidate"

    class _DS:  data = {"close": None}
    monkeypatch.setattr(
        "app.core.data_engine.dataset_registry.load_registry_dataset",
        lambda *a, **k: _DS(),
    )
    monkeypatch.setattr(
        "app.core.lifecycle.validation_gate.ValidationGate.evaluate",
        lambda self, dsl, data, n_trials=1: ValidationResult(
            passed=True, dsl=dsl, n_folds=5, min_oos_sharpe=0.5, deflated_sharpe=0.95),
    )
    app.dependency_overrides[get_store] = lambda: store
    try:
        r = test_client.post(f"/api/alphas/{aid}/validate")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["passed"] is True and body["new_status"] == "validated"
        assert store.get_by_id(aid).status == "validated"      # 状态机确实升级
        # 已非 candidate → 再次调用应 409
        r2 = test_client.post(f"/api/alphas/{aid}/validate")
        assert r2.status_code == 409
    finally:
        app.dependency_overrides.pop(get_store, None)


def test_validate_endpoint_stays_candidate_on_fail(test_client, tmp_path, monkeypatch):
    from app.api.router import get_store
    from app.main import app
    from app.db.alpha_store import AlphaStore, AlphaResult

    store = AlphaStore(db_url=f"sqlite:///{tmp_path/'v2.db'}")
    aid = store.save(AlphaResult(dsl="rank(close)"))
    class _DS:  data = {"close": None}
    monkeypatch.setattr(
        "app.core.data_engine.dataset_registry.load_registry_dataset", lambda *a, **k: _DS())
    monkeypatch.setattr(
        "app.core.lifecycle.validation_gate.ValidationGate.evaluate",
        lambda self, dsl, data, n_trials=1: ValidationResult(
            passed=False, dsl=dsl, reasons=["DSR 0.4 ≤ 阈值 0.9"]),
    )
    app.dependency_overrides[get_store] = lambda: store
    try:
        r = test_client.post(f"/api/alphas/{aid}/validate")
        assert r.status_code == 200
        assert r.json()["passed"] is False
        assert store.get_by_id(aid).status == "candidate"      # 维持 candidate
    finally:
        app.dependency_overrides.pop(get_store, None)
