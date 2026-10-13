"""
回测报告里的 IC 必须按**日期**把信号和前瞻收益配对。

2026-10-13 查出的缺陷：`rolling_rank_ic_from_prices` / `ic_decay_curve` / `decile_analysis`
按**行号**配对。信号有预热窗口时回测结果比价格面板短（开头被裁掉 window + delay 行），
于是信号第 t 行配到的是它自己**形成窗口之内**某一天的收益：

    纯随机游走（任何信号的真实 IC 都是 0）上，
      rank(-ts_delta(close, 5))        报 mean_ic = −0.417   （≈ −1/√5）
      rank(ts_delta(log(close), 252))  报 mean_ic = +0.055   （≈ +1/√252）

盈亏（Sharpe）不受影响 —— 错的只是 IC / ICIR / 分档收益这几个诊断量，但它们会落库
（因子记录的 ic_ir）并显示在面板上。发现的契机是 Phase 10 的真实数据预览：
5 日反转的 IC 在 IS 和 Validate 上都是 −0.35，这在真实市场里不可能。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from app.core.alpha_engine.signal_processor import SimulationConfig
from app.core.backtest_engine.performance_analyzer import PerformanceAnalyzer, _forward_returns
from app.core.backtest_engine.realistic_backtester import RealisticBacktester

T, N = 700, 60
IDX = pd.bdate_range("2021-01-04", periods=T)
COLS = [f"S{i:02d}" for i in range(N)]


def _dataset(ret: np.ndarray, volume: np.ndarray | None = None, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed + 1000)
    close = pd.DataFrame(100 * np.exp(np.cumsum(ret, axis=0)), index=IDX, columns=COLS)
    vol = rng.uniform(1e6, 2e6, (T, N)) if volume is None else volume
    return {"open": close.shift(1).fillna(100.0),
            "high": close * (1 + np.abs(rng.normal(0, 0.005, (T, N)))),
            "low": close * (1 - np.abs(rng.normal(0, 0.005, (T, N)))),
            "close": close, "volume": pd.DataFrame(vol, index=IDX, columns=COLS)}


@pytest.fixture(scope="module")
def random_walk():
    return _dataset(np.random.default_rng(0).normal(0.0003, 0.015, (T, N)))


@pytest.fixture(scope="module")
def planted():
    """
    成交量是随机的；10 天前的成交量排名决定"后天"的收益：
        r[t+2] = 0.01 · z(volume[t−10]) + 噪声
    引擎把信号延迟一天执行，所以 `ts_delay(volume, 10)` 在 t 日的信号（用 volume[t−10]）
    经延迟后对上的正是 t+1 → t+2 这段收益。
    """
    rng = np.random.default_rng(7)
    vol = rng.uniform(1e6, 2e6, (T, N))
    z = (vol - vol.mean(axis=1, keepdims=True)) / vol.std(axis=1, keepdims=True)
    ret = rng.normal(0.0, 0.01, (T, N))
    ret[12:] += 0.01 * z[:-12]
    return _dataset(ret, volume=vol)


def _run(dsl: str, ds: dict):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return RealisticBacktester(SimulationConfig()).run(dsl, ds)


# ---------------------------------------------------------------------------
# 随机游走：IC 必须是 0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dsl,old_bug_value", [
    ("rank(-ts_delta(close, 5))", -0.417),
    ("rank(ts_delta(log(close), 252))", 0.055),
    ("rank(ts_delta(close, 20))", 0.2),
])
def test_price_window_signals_have_zero_ic_on_a_random_walk(random_walk, dsl, old_bug_value):
    r = _run(dsl, random_walk)
    assert len(r.is_result.signal) < T                       # 预热窗口确实把结果裁短了
    assert abs(r.is_report.mean_ic) < 0.03, (
        f"{dsl} 在随机游走上报出 IC={r.is_report.mean_ic:+.3f}（按行号错位时是 {old_bug_value:+.3f}）")
    assert abs(r.is_report.ic_ir) < 0.25


def test_decile_returns_are_flat_on_a_random_walk(random_walk):
    """错位时 5 日反转的十分位收益是一条陡峭的单调线（最高档 − 最低档约 −1.5%/天）。"""
    r = _run("rank(-ts_delta(close, 5))", random_walk)
    dec = PerformanceAnalyzer(r.is_result).decile_analysis(prices=random_walk["close"])
    assert abs(dec.iloc[-1] - dec.iloc[0]) < 0.002


def test_ic_decay_curve_is_flat_on_a_random_walk(random_walk):
    r = _run("rank(-ts_delta(close, 5))", random_walk)
    curve = PerformanceAnalyzer(r.is_result).ic_decay_curve(r.is_result.signal, random_walk["close"],
                                                            horizons=[1, 5, 20])
    assert curve.abs().max() < 0.06


# ---------------------------------------------------------------------------
# 埋了信号：IC 必须认得出来
# ---------------------------------------------------------------------------

def test_a_planted_signal_behind_a_warmup_window_is_recovered(planted):
    """另一侧：错位时这条信号的 IC 接近 0（配到的是无关的日子），对齐后应当很高。"""
    r = _run("rank(ts_delay(volume, 10))", planted)
    assert len(r.is_result.signal) < T
    assert r.is_report.mean_ic > 0.5
    assert r.is_report.sharpe_ratio > 3.0                    # 盈亏本来就是对的，IC 现在与它一致


def test_planted_signal_decile_returns_are_monotone(planted):
    r = _run("rank(ts_delay(volume, 10))", planted)
    dec = PerformanceAnalyzer(r.is_result).decile_analysis(prices=planted["close"])
    assert dec.iloc[-1] - dec.iloc[0] > 0.02
    assert dec.is_monotonic_increasing


def test_planted_signal_decays_with_horizon(planted):
    """信号只预测一天；拉长到 20 天，累计收益里它只占 1/20，IC 应明显更低。"""
    r = _run("rank(ts_delay(volume, 10))", planted)
    curve = PerformanceAnalyzer(r.is_result).ic_decay_curve(r.is_result.signal, planted["close"],
                                                            horizons=[1, 20])
    assert curve[1] > 0.5 and curve[20] < 0.5 * curve[1]


# ---------------------------------------------------------------------------
# 对齐函数本身
# ---------------------------------------------------------------------------

def test_forward_returns_follow_the_signal_index_not_row_numbers():
    prices = pd.DataFrame({"A": [10.0, 11.0, 12.1, 12.1, 13.31], "B": [5.0, 5.0, 5.5, 6.05, 6.05]},
                          index=pd.bdate_range("2024-01-01", periods=5))
    late = pd.DataFrame(1.0, index=prices.index[2:], columns=["B", "A"])     # 裁掉前两行，列序也不同
    fwd = _forward_returns(prices, 1, late)
    assert list(fwd.index) == list(late.index) and list(fwd.columns) == ["B", "A"]
    assert fwd.loc[prices.index[2], "A"] == pytest.approx(0.0)               # 12.1 → 12.1
    assert fwd.loc[prices.index[2], "B"] == pytest.approx(0.10)              # 5.5 → 6.05
    assert fwd.loc[prices.index[3], "A"] == pytest.approx(0.10)
    assert fwd.iloc[-1].isna().all()                                         # 最后一天没有"明天"
    two = _forward_returns(prices, 2, late)
    assert two.loc[prices.index[2], "A"] == pytest.approx(13.31 / 12.1 - 1)


def test_signal_dates_missing_from_prices_get_no_forward_return():
    prices = pd.DataFrame({"A": [10.0, 11.0, 12.0]}, index=pd.bdate_range("2024-01-01", periods=3))
    sig = pd.DataFrame({"A": [1.0, 1.0]}, index=pd.to_datetime(["2024-01-02", "2024-02-15"]))
    fwd = _forward_returns(prices, 1, sig)
    assert fwd["A"].iloc[0] == pytest.approx(12.0 / 11.0 - 1) and np.isnan(fwd["A"].iloc[1])
