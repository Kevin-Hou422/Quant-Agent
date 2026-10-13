"""
Phase 10 → R.2：风险模型的 size / value 风格。

R.2 交付时这两个风格缺席（没有市值、没有账面价值），归因结果里一直带着
`styles_missing: ["size", "value"]`。Phase 10 有了 `market_cap` / `book_to_price` 之后：
数据集带这两个字段且覆盖率够 → 进模型；否则仍然缺席，并且说明为什么。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.risk_engine import (
    FUNDAMENTAL_STYLES, STYLE_FACTORS, build_fundamental_exposures, fit_risk_model,
)
from app.core.risk_engine.factor_model import MIN_STYLE_COVERAGE, MIN_STYLE_DAYS

N_DAYS, N_TICK = 500, 20
IDX = pd.bdate_range("2021-01-04", periods=N_DAYS)
COLS = [f"S{i:02d}" for i in range(N_TICK)]


def _panel(seed: int = 0, *, size_premium: float = 0.0, fundamentals: bool = True) -> dict:
    """
    市值：S00 最小 … S19 最大，跨 3 个数量级，随股价漂移。
    size_premium：每天给"市值排名"加的收益斜率 —— 有它，size 才是一个真有收益波动的因子。
    """
    rng = np.random.default_rng(seed)
    rank = np.linspace(-1.0, 1.0, N_TICK)
    common = rng.normal(0.0, 0.01, N_DAYS)                       # size 因子自己的日收益
    ret = rng.normal(0.0003, 0.012, (N_DAYS, N_TICK)) + size_premium * np.outer(common, rank) / 0.01
    close = pd.DataFrame(100 * np.cumprod(1 + ret, axis=0), index=IDX, columns=COLS)
    ds = {"close": close,
          "volume": pd.DataFrame(rng.integers(1_000_000, 9_000_000, (N_DAYS, N_TICK)).astype(float),
                                 index=IDX, columns=COLS),
          "sector": pd.DataFrame(np.tile(np.arange(N_TICK) % 4, (N_DAYS, 1)), index=IDX, columns=COLS)}
    if fundamentals:
        shares = np.logspace(7, 10, N_TICK)
        ds["market_cap"] = close * shares
        book = rng.uniform(0.05, 0.9, N_TICK)
        ds["book_to_price"] = pd.DataFrame(np.tile(book, (N_DAYS, 1)), index=IDX, columns=COLS) * 100 / close
    return ds


def _attr(ds, weights=None, **kw):
    w = pd.Series(1.0 / N_TICK, index=COLS) if weights is None else weights
    return fit_risk_model(ds, lookback=252, **kw).attribute(w)


# ---------------------------------------------------------------------------
# 缺席：说得出原因
# ---------------------------------------------------------------------------

def test_price_only_dataset_reports_both_styles_missing_with_the_reason():
    d = _attr(_panel(fundamentals=False)).to_dict()
    assert d["styles_covered"] == list(STYLE_FACTORS)
    assert d["styles_missing"] == ["size", "value"]
    assert "market_cap" in d["styles_missing_reason"]["size"]
    assert "book_to_price" in d["styles_missing_reason"]["value"]
    assert "未开启基本面" in d["styles_missing_reason"]["size"]


def test_fundamental_styles_are_declared():
    assert FUNDAMENTAL_STYLES == ("size", "value")
    assert not set(FUNDAMENTAL_STYLES) & set(STYLE_FACTORS)


# ---------------------------------------------------------------------------
# 进模型
# ---------------------------------------------------------------------------

def test_dataset_with_fundamentals_gets_size_and_value():
    rm = fit_risk_model(_panel(), lookback=252)
    assert rm.styles == STYLE_FACTORS + ("size", "value")
    assert rm.styles_missing == {}
    assert list(rm.exposures.columns)[:7] == list(STYLE_FACTORS) + ["size", "value"]
    d = rm.attribute(pd.Series(1.0 / N_TICK, index=COLS)).to_dict()
    assert d["styles_missing"] == [] and d["styles_missing_reason"] == {}
    assert d["styles_covered"][-2:] == ["size", "value"]
    assert d["n_factors"] == 7 + 4                                  # 7 个风格 + 4 个行业


def test_variance_identity_still_holds_with_the_extra_styles():
    """加了两列之后分解仍必须精确：因子方差 + 特异方差 == 总方差，各因子贡献之和 == 因子方差。"""
    rng = np.random.default_rng(5)
    w = pd.Series(rng.dirichlet(np.ones(N_TICK)), index=COLS)
    rm = fit_risk_model(_panel(seed=2, size_premium=0.004), lookback=252)
    a = rm.attribute(w)
    assert a.factor_var + a.specific_var == pytest.approx(a.total_var, rel=1e-12)
    assert sum(a.by_factor.values()) == pytest.approx(a.factor_var, rel=1e-9)
    full = float(w.to_numpy() @ rm.covariance().to_numpy() @ w.to_numpy())
    assert full == pytest.approx(a.total_var, rel=1e-9)


def test_size_exposure_is_the_cross_sectional_z_score_of_log_market_cap():
    ds = _panel()
    exps, missing = build_fundamental_exposures(ds, list(IDX[-252:]))
    assert missing == {}
    last = exps["size"].iloc[-1]
    assert last[COLS[-1]] > 1.0 and last[COLS[0]] < -1.0             # S19 远大于 S00
    expected = np.log(ds["market_cap"].iloc[-1])
    expected = (expected - expected.mean()) / expected.std(ddof=1)
    np.testing.assert_allclose(last.to_numpy(), expected.to_numpy(), atol=1e-12)
    assert abs(last.mean()) < 1e-12


def test_size_exposure_does_not_depend_on_the_currency_unit():
    """市值整体乘一个常数（美元 → 百万美元），log 之后只差一个常数，z-score 不变。"""
    ds = _panel()
    a, _ = build_fundamental_exposures(ds, list(IDX[-252:]))
    ds["market_cap"] = ds["market_cap"] / 1e6
    b, _ = build_fundamental_exposures(ds, list(IDX[-252:]))
    np.testing.assert_allclose(a["size"].to_numpy(), b["size"].to_numpy(), atol=1e-9)


def test_portfolio_tilts_show_up_as_net_exposures():
    ds = _panel()
    big = pd.Series(0.0, index=COLS)
    big[COLS[-5:]] = 0.2                                             # 只买最大的 5 只
    small = pd.Series(0.0, index=COLS)
    small[COLS[:5]] = 0.2
    cheap = ds["book_to_price"].iloc[-1].nlargest(5).index
    value = pd.Series(0.0, index=COLS)
    value[cheap] = 0.2
    assert _attr(ds, big).exposures["size"] > 1.0
    assert _attr(ds, small).exposures["size"] < -1.0
    assert _attr(ds, value).exposures["value"] > 1.0
    assert abs(_attr(ds).exposures["size"]) < 1e-9                   # 等权：没有规模倾斜


def _size_world(seed: int, n: int = 120, premium: float = 0.006):
    """n 只票、唯一的共同因子是规模：r_i,t = premium · g_t · rank_i + 噪声，g ~ N(0,1)。"""
    rng = np.random.default_rng(seed)
    cols = [f"T{i:03d}" for i in range(n)]
    rank = np.linspace(-1.0, 1.0, n)
    g = rng.normal(0.0, 1.0, N_DAYS)
    ret = rng.normal(0.0, 0.012, (N_DAYS, n)) + premium * np.outer(g, rank)
    close = pd.DataFrame(100 * np.cumprod(1 + ret, axis=0), index=IDX, columns=cols)
    ds = {"close": close,
          "volume": pd.DataFrame(rng.integers(1_000_000, 9_000_000, (N_DAYS, n)).astype(float),
                                 index=IDX, columns=cols),
          "market_cap": close * np.logspace(7, 10, n)}
    return ds, cols, rank


def test_the_size_factor_recovers_the_planted_factor_variance():
    """
    暴露 z_i = rank_i / sd(rank)，所以因子收益 f_t = premium · g_t · sd(rank)，
    方差的真值是 premium² · sd(rank)²。截面取 120 只，回归的估计误差才小到可以对真值。
    """
    ds, cols, rank = _size_world(seed=3)
    rm = fit_risk_model(ds, lookback=252, with_sector=False)
    truth = 0.006 ** 2 * rank.std(ddof=1) ** 2
    assert rm.styles == STYLE_FACTORS + ("size",)
    assert rm.factor_cov.loc["size", "size"] == pytest.approx(truth, rel=0.25)


def test_a_size_bet_is_attributed_to_the_size_factor():
    """
    押"大减小"的组合：有 size 这一列时，风险的大头记在 size 名下；没有时这部分共同波动
    留在回归残差里，于是每只票的"特异"方差被抬高 —— 风险还在，只是没被点名。
    """
    ds, cols, rank = _size_world(seed=4)
    tilt = pd.Series(rank / np.abs(rank).sum(), index=cols)
    with_size = fit_risk_model(ds, lookback=252, with_sector=False)
    price_only = {k: v for k, v in ds.items() if k != "market_cap"}
    without = fit_risk_model(price_only, lookback=252, with_sector=False)
    a, b = with_size.attribute(tilt), without.attribute(tilt)
    assert a.by_factor["size"] / a.total_var > 0.8
    assert "size" not in b.by_factor
    assert (without.specific_var > with_size.specific_var).mean() > 0.9
    assert b.specific_var > a.specific_var


def test_missing_value_gives_zero_exposure_not_a_fabricated_one():
    """净资产为负的公司 B/P 为空 → value 暴露 0（不偏不倚），其余股票照常标准化。"""
    ds = _panel()
    ds["book_to_price"][COLS[3]] = np.nan
    exps, missing = build_fundamental_exposures(ds, list(IDX[-252:]))
    assert missing == {}
    assert (exps["value"][COLS[3]] == 0.0).all()
    assert exps["value"].drop(columns=COLS[3]).iloc[-1].abs().sum() > 1.0


def test_non_positive_market_cap_is_treated_as_missing():
    ds = _panel()
    ds["market_cap"][COLS[0]] = 0.0
    exps, _ = build_fundamental_exposures(ds, list(IDX[-252:]))
    assert (exps["size"][COLS[0]] == 0.0).all()
    assert np.isfinite(exps["size"].to_numpy()).all()


# ---------------------------------------------------------------------------
# 覆盖率门槛
# ---------------------------------------------------------------------------

def _with_coverage(n_names: int, bad_days: int = 0) -> dict:
    """只有前 n_names 只票有基本面；估计窗口最早的 bad_days 天整行为空。"""
    ds = _panel()
    for name in ("market_cap", "book_to_price"):
        df = ds[name].copy()
        df[COLS[n_names:]] = np.nan
        df.iloc[N_DAYS - 252:N_DAYS - 252 + bad_days] = np.nan
        ds[name] = df
    return ds


def test_cross_sectional_coverage_boundary():
    at = int(MIN_STYLE_COVERAGE * N_TICK)                            # 10/20 = 恰好 50%
    assert fit_risk_model(_with_coverage(at), lookback=252).styles_missing == {}
    below = fit_risk_model(_with_coverage(at - 1), lookback=252)
    assert set(below.styles_missing) == {"size", "value"}
    assert below.styles == STYLE_FACTORS
    assert "覆盖率不足" in below.styles_missing["size"]


def test_time_coverage_boundary():
    ok_days = int(np.ceil(MIN_STYLE_DAYS * 252))                     # 227 天达标 = 90.08%
    assert fit_risk_model(_with_coverage(N_TICK, bad_days=252 - ok_days), lookback=252).styles_missing == {}
    short = fit_risk_model(_with_coverage(N_TICK, bad_days=252 - ok_days + 1), lookback=252)
    assert set(short.styles_missing) == {"size", "value"}


def test_each_style_is_judged_on_its_own_field():
    ds = _panel()
    del ds["book_to_price"]
    rm = fit_risk_model(ds, lookback=252)
    assert rm.styles == STYLE_FACTORS + ("size",)
    assert list(rm.styles_missing) == ["value"]
    ds2 = _panel()
    ds2["market_cap"] = ds2["market_cap"] * np.nan
    rm2 = fit_risk_model(ds2, lookback=252)
    assert rm2.styles == STYLE_FACTORS + ("value",) and list(rm2.styles_missing) == ["size"]


def test_empty_fundamental_frame_counts_as_missing():
    ds = _panel()
    ds["market_cap"] = pd.DataFrame()
    assert "size" in fit_risk_model(ds, lookback=252).styles_missing


def test_coverage_is_measured_on_the_estimation_window_not_the_whole_history():
    """窗口之前的历史没有基本面不要紧 —— 协方差只用窗口里的截面回归。"""
    ds = _panel()
    for name in ("market_cap", "book_to_price"):
        df = ds[name].copy()
        df.iloc[:N_DAYS - 252] = np.nan
        ds[name] = df
    assert fit_risk_model(ds, lookback=252).styles_missing == {}
