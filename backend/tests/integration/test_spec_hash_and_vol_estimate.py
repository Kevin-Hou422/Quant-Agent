"""
规格哈希、验证设计哈希、风控上限装配与组合波动估计（D5 关键路径变异复核：
app/core/portfolio_manager/executable_spec.py、app/core/lifecycle/validation_design.py 的哈希）。

复核时存活的：
    · 两个哈希的规范化（sort_keys / ensure_ascii）—— 字典键的顺序、中文字符改了哈希的写法都没人发现，
      而哈希一变，已批准的规格就全部"被改过"
    · 风控上限 / 规格里"能否做空"在配置缺这一项时的缺省（缺省必须是不能做空）
    · 波动估计的公式（权重 × 收益、年化 √252）、样本恰好 20 天、零波动返回 None
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.core.lifecycle.validation_design import design_hash
from app.core.portfolio_manager import executable_spec as es


@pytest.mark.parametrize("fn", [es.spec_hash, design_hash])
def test_the_hash_does_not_depend_on_key_order(fn):
    assert fn({"a": 1, "b": {"x": 1, "y": 2}}) == fn({"b": {"y": 2, "x": 1}, "a": 1})


@pytest.mark.parametrize("fn", [es.spec_hash, design_hash])
def test_the_hash_is_the_documented_canonical_json(fn):
    """紧凑分隔符、键排序、中文原样（不转 \\uXXXX）—— 写法一变，所有已批准规格的哈希都对不上。"""
    expected = hashlib.sha256('{"a":1,"名":"中文"}'.encode("utf-8")).hexdigest()
    assert fn({"名": "中文", "a": 1}) == expected


def test_a_settings_object_without_the_short_flag_means_long_only():
    limits = es.risk_limits_from_settings(SimpleNamespace())
    assert limits.long_only is True
    assert es.risk_limits_from_settings(SimpleNamespace(trading_allow_short=True)).long_only is False


def test_the_spec_records_no_shorting_when_the_flag_is_absent():
    from app.config import settings
    ns =SimpleNamespace(**{k: getattr(settings, k) for k in type(settings).model_fields if k != "trading_allow_short"})
    spec = es.build_spec({"a": "rank(close)"}, {"a": 1.0}, "equal_weight", 1e6, 0.0, settings=ns)
    assert spec["trading"]["allow_short"] is False and spec["risk_limits"]["long_only"] is True


def _close(n, daily):
    return pd.DataFrame({"A": 100.0 * np.cumprod(1 + np.asarray(daily, dtype=float))},
                        index=pd.bdate_range("2025-01-02", periods=n))


def test_the_vol_estimate_is_lagged_weight_times_return_annualised():
    rng = np.random.default_rng(7)
    daily = rng.normal(0, 0.01, 40)
    close = _close(40, daily)
    w = pd.DataFrame({"A": 0.5}, index=close.index)
    rets = close["A"].pct_change()
    expected = float((0.5 * rets).fillna(0.0).tail(60).std(ddof=1) * np.sqrt(252.0))
    assert es.estimate_port_vol(w, close, lookback=60) == pytest.approx(expected, rel=1e-12)


def test_exactly_the_minimum_sample_is_enough():
    rng = np.random.default_rng(3)
    close = _close(es.MIN_VOL_SAMPLE, rng.normal(0, 0.01, es.MIN_VOL_SAMPLE))
    w = pd.DataFrame({"A": 1.0}, index=close.index)
    assert es.estimate_port_vol(w, close, lookback=60) is not None
    assert es.estimate_port_vol(w.iloc[1:], close.iloc[1:], lookback=60) is None


def test_zero_volatility_is_no_estimate():
    close = pd.DataFrame({"A": 100.0}, index=pd.bdate_range("2025-01-02", periods=30))
    w = pd.DataFrame({"A": 1.0}, index=close.index)
    assert es.estimate_port_vol(w, close, lookback=60) is None


PROVEN_EQUIVALENT = {
    "app/core/portfolio_manager/executable_spec.py ×1 — L213 `vol > 1e-9` -> `>=`":
        "两者只在年化波动恰好等于 1e-9 时不同（日波动约 6.3e-11）。它由 sample.std(ddof=1) × √252 的浮点运算得出，"
        "真实价格序列（价格以分为单位、收益至少 1e-6 量级）构造不出恰好落在这个点上的样本；零波动由 "
        "test_zero_volatility_is_no_estimate 钉住，非零波动在 1e-9 两侧的行为由同一条 `isfinite and` 判定，结论一致。",
}


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 1
