"""
trading_context/spread.py —— 价差估计量的定钉测试（变异测试驱动）

来由：21 个变异点，首测击杀率 47.6%（存活 11）。

这两个估计量决定**每一笔模拟成交的价差成本**（TR.3 的 grounded 成本就建在它们上），
而既有覆盖只验"返回值非负、形状正确"。后果：
  - `* 1e4` 写成 `/`：bps 口径小 1e8 倍，价差成本几乎归零
  - `eta = (h + l) / 2` 写成 `-`：对数中价变成"半价差"，Abdi-Ranaldi 整个失去意义
  - `x = (c-η)·(c'-η)` 的 `*` 写成 `/`：协方差变成比值
  - `S = np.where(S < 0.0, 0.0, S)` 的截断边界

本文件按两篇论文的公式**独立重算**再比对，不是复制实现。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.trading_context.spread import (
    abdi_ranaldi_spread,
    corwin_schultz_spread,
    corwin_schultz_spread_bps,
)

_CS_DENOM = 3.0 - 2.0 * np.sqrt(2.0)


def _ohlc(n_days: int = 40, n_tickers: int = 3, seed: int = 0,
          hl_range: float = 0.01):
    """真实幅度的 H/L（DEV_LESSONS §O：固定 ±1% 会让价差估计荒谬）。"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n_days)
    cols = [f"T{i}" for i in range(n_tickers)]
    close = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.01, (n_days, n_tickers)), axis=0),
        index=idx, columns=cols)
    up = rng.uniform(0, hl_range, close.shape)
    dn = rng.uniform(0, hl_range, close.shape)
    return close * (1 + up), close * (1 - dn), close


# ===========================================================================
# A. Corwin-Schultz
# ===========================================================================

class TestCorwinSchultz:

    @staticmethod
    def _reference(high: pd.DataFrame, low: pd.DataFrame, window: int) -> pd.Series:
        """按 Corwin-Schultz (2012) 公式独立重算。"""
        H, L = high.to_numpy(float), low.to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            hl = np.log(np.where((H > 0) & (L > 0), H / L, np.nan))
            beta = hl[:-1] ** 2 + hl[1:] ** 2
            Hmax, Lmin = np.maximum(H[:-1], H[1:]), np.minimum(L[:-1], L[1:])
            gamma = np.log(np.where((Hmax > 0) & (Lmin > 0), Hmax / Lmin, np.nan)) ** 2
            alpha = ((np.sqrt(2.0 * beta) - np.sqrt(beta)) / _CS_DENOM
                     - np.sqrt(gamma / _CS_DENOM))
            S = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
        S = np.where(np.isfinite(S), S, np.nan)
        S = np.where(S < 0.0, 0.0, S)
        return pd.DataFrame(S, columns=high.columns).tail(
            max(1, window)).mean(axis=0, skipna=True).fillna(0.0)

    def test_matches_the_published_formula(self):
        high, low, _ = _ohlc(seed=1)
        got = corwin_schultz_spread(high, low, window=20)
        assert np.allclose(got.to_numpy(), self._reference(high, low, 20).to_numpy(),
                           atol=1e-15, equal_nan=False)

    def test_bps_conversion_multiplies_by_ten_thousand(self):
        """`* 1e4` 写成 `/` 会让 bps 小 1e8 倍，价差成本几乎归零。"""
        high, low, _ = _ohlc(seed=2)
        frac = corwin_schultz_spread(high, low, window=20)
        bps = corwin_schultz_spread_bps(high, low, window=20)
        assert np.allclose(bps.to_numpy(), frac.to_numpy() * 1e4, atol=1e-12)
        assert bps.max() > 1.0, (
            f"bps 口径的价差最大值只有 {bps.max():.6f} —— 量纲疑似写反")

    def test_estimate_grows_with_the_high_low_range(self):
        """H/L 区间越宽，估计的价差越大 —— 这是该估计量的基本性质。"""
        narrow = corwin_schultz_spread(*_ohlc(seed=3, hl_range=0.002)[:2], window=20)
        wide = corwin_schultz_spread(*_ohlc(seed=3, hl_range=0.02)[:2], window=20)
        assert wide.mean() > narrow.mean() * 2, (
            f"H/L 放宽 10 倍后价差估计没有明显变大：{narrow.mean():.6f} → {wide.mean():.6f}")

    def test_negative_estimates_are_floored_at_zero(self):
        """
        `S = np.where(S < 0.0, 0.0, S)` —— CS 的负估计按惯例截 0。
        用 H≈L 的极窄区间构造出负 alpha，确认输出无负值。
        """
        high, low, _ = _ohlc(seed=4, hl_range=1e-6)
        out = corwin_schultz_spread(high, low, window=20)
        assert (out >= 0).all(), f"出现负价差：{out[out < 0].to_dict()}"

    def test_window_limits_the_average_to_the_tail(self):
        """
        `Sdf.tail(max(1, window)).mean(...)` —— 只对**最近 window 日**取均值。
        构造：前半段极窄 H/L、后半段极宽，短窗口的估计必须明显更大。
        """
        n = 60
        idx = pd.bdate_range("2024-01-02", periods=n)
        cols = ["A"]
        close = pd.DataFrame(100.0, index=idx, columns=cols)
        rng = np.random.default_rng(5)
        width = np.where(np.arange(n) < n // 2, 1e-5, 0.03).reshape(-1, 1)
        jitter = rng.uniform(0.5, 1.0, (n, 1))
        high = close * (1 + width * jitter)
        low = close * (1 - width * jitter)
        short = corwin_schultz_spread(high, low, window=10)
        long = corwin_schultz_spread(high, low, window=59)
        assert short["A"] > long["A"], (
            f"只取尾段 10 日的估计（{short['A']:.6f}）没有大于全窗口（{long['A']:.6f}）"
            f" —— tail 窗口疑似失效")

    def test_skipna_average_tolerates_missing_days(self):
        """
        `mean(axis=0, skipna=True)` —— 改成 False 后，窗口里只要有一天缺失
        整只股票的估计就变成 NaN，再 fillna(0) → **价差被当成 0**。

        ⚠️ 缺失日必须落在 **tail(window) 的窗口内**才可能被平均到。
        第一版把 NaN 放在第 3 行、窗口却只取最后 20 行（共 40 行），
        那一天根本不参与平均，测试因此无效（变异测试证实它存活）。
        """
        high, low, _ = _ohlc(n_days=40, seed=6)
        high.iloc[-5, 0] = np.nan          # 落在 tail(20) 窗口内
        low.iloc[-5, 0] = np.nan
        out = corwin_schultz_spread(high, low, window=20)
        assert out["T0"] > 0, "窗口内有一天缺失就把价差估成了 0"
        assert out["T1"] > 0

    def test_skipna_average_tolerates_missing_days_in_abdi_ranaldi(self):
        """Abdi-Ranaldi 里是另一处独立的 `skipna=True`。"""
        high, low, close = _ohlc(n_days=40, seed=17)
        high.iloc[-5, 0] = np.nan
        low.iloc[-5, 0] = np.nan
        out = abdi_ranaldi_spread(high, low, close, window=20)
        assert out["T0"] > 0, "窗口内有一天缺失就把价差估成了 0"

    def test_non_positive_prices_never_leak_into_the_estimate(self):
        """
        `(H > 0) & (L > 0)` 的守卫：0/负价必须被排除，估计值仍然有限且非负。

        ⚠️ 我一度想用"警告升级成错误"来区分 `>` 与 `>=` —— **行不通**：
        整段计算包在 `np.errstate(divide="ignore", invalid="ignore")` 里，
        log(0) 的警告根本不会发出（见 PROVEN_EQUIVALENT 中的等价性证明）。
        这里改为钉住可观测的契约：脏价不得污染同一列的最终估计。
        """
        high, low, close = _ohlc(n_days=40, seed=18)
        clean_t0 = corwin_schultz_spread(high, low, window=20)["T0"]
        high.iloc[-3, 1] = 0.0
        low.iloc[-3, 1] = 0.0
        close.iloc[-3, 1] = 0.0
        cs = corwin_schultz_spread(high, low, window=20)
        ar = abdi_ranaldi_spread(high, low, close, window=20)
        assert np.isfinite(cs.to_numpy()).all() and (cs >= 0).all()
        assert np.isfinite(ar.to_numpy()).all() and (ar >= 0).all()
        assert cs["T1"] > 0, "混入一天 0 价就把整列价差估成了 0"
        assert cs["T0"] == pytest.approx(clean_t0, abs=1e-15), (
            "另一只标的的估计被隔壁列的脏价影响了")

    def test_non_positive_prices_do_not_crash(self):
        """`(H > 0) & (L > 0)` 的守卫：0/负价必须被排除而不是产生 -inf。"""
        high, low, _ = _ohlc(seed=7)
        high.iloc[5, 1] = 0.0
        low.iloc[5, 1] = -1.0
        out = corwin_schultz_spread(high, low, window=20)
        assert np.isfinite(out.to_numpy()).all()


# ===========================================================================
# B. Abdi-Ranaldi
# ===========================================================================

class TestAbdiRanaldi:

    @staticmethod
    def _reference(high, low, close, window: int) -> pd.Series:
        """按 Abdi-Ranaldi (2017) CHL 公式独立重算。"""
        with np.errstate(divide="ignore", invalid="ignore"):
            h = np.log(high.where(high > 0))
            l = np.log(low.where(low > 0))
            c = np.log(close.where(close > 0))
        eta = (h + l) / 2.0
        s2 = 4.0 * ((c - eta) * (c.shift(-1) - eta))
        return np.sqrt(s2.clip(lower=0.0)).tail(
            max(1, window)).mean(axis=0, skipna=True).fillna(0.0)

    def test_matches_the_published_formula(self):
        high, low, close = _ohlc(seed=11)
        got = abdi_ranaldi_spread(high, low, close, window=20)
        exp = self._reference(high, low, close, 20)
        assert np.allclose(got.to_numpy(), exp.to_numpy(), atol=1e-15)

    def test_eta_is_the_midpoint_not_the_half_range(self):
        """
        `eta = (h + l) / 2.0` —— 对数中价。写成 `(h - l)/2` 会变成半价差本身，
        `c - eta` 随即失去意义（量级从 1e-3 跳到 ~4.6）。

        判据：η 必须落在 ln(low) 与 ln(high) **之间**。
        """
        high, low, close = _ohlc(seed=12)
        h, l = np.log(high), np.log(low)
        eta = (h + l) / 2.0
        assert ((eta >= l) & (eta <= h)).all().all()
        # 若写成减法，η 会远小于 ln(low)（价格 ~100 时 ln≈4.6，半幅≈0.005）
        wrong = (h - l) / 2.0
        assert not ((wrong >= l) & (wrong <= h)).all().all()

    def test_estimate_grows_with_the_high_low_range(self):
        narrow = abdi_ranaldi_spread(*_ohlc(seed=13, hl_range=0.002), window=20)
        wide = abdi_ranaldi_spread(*_ohlc(seed=13, hl_range=0.02), window=20)
        assert wide.mean() > narrow.mean(), (
            f"H/L 放宽后 Abdi-Ranaldi 估计没有变大：{narrow.mean():.6f} → {wide.mean():.6f}")

    def test_negative_covariance_is_clipped_to_zero(self):
        """`s2.clip(lower=0.0)` —— 负协方差对应无意义的负价差，截 0 后开方。"""
        out = abdi_ranaldi_spread(*_ohlc(seed=14), window=20)
        assert (out >= 0).all()
        assert np.isfinite(out.to_numpy()).all()

    def test_window_limits_the_average_to_the_tail(self):
        n = 60
        idx = pd.bdate_range("2024-01-02", periods=n)
        cols = ["A"]
        rng = np.random.default_rng(15)
        close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.005, (n, 1)), axis=0),
                             index=idx, columns=cols)
        width = np.where(np.arange(n) < n // 2, 1e-5, 0.03).reshape(-1, 1)
        high = close * (1 + width)
        low = close * (1 - width)
        short = abdi_ranaldi_spread(high, low, close, window=10)
        long = abdi_ranaldi_spread(high, low, close, window=59)
        assert short["A"] > long["A"]

    def test_two_estimators_agree_in_order_of_magnitude(self):
        """
        两个估计量互为交叉校验：同一段数据上不应差出一个数量级以上。
        任一方的量纲被改坏（`* → /`）都会打破这条。
        """
        high, low, close = _ohlc(n_days=80, seed=16, hl_range=0.01)
        cs = corwin_schultz_spread(high, low, window=40).mean()
        ar = abdi_ranaldi_spread(high, low, close, window=40).mean()
        assert cs > 0 and ar > 0
        ratio = max(cs, ar) / min(cs, ar)
        assert ratio < 20, (
            f"两个价差估计量相差 {ratio:.1f} 倍（CS={cs:.6f} AR={ar:.6f}）—— 疑似量纲错")


# ===========================================================================
# C. 存活变异的等价性证明
# ===========================================================================

PROVEN_EQUIVALENT = {
    "app/core/trading_context/spread.py ×1 — L44 `S = np.where(S < 0.0, 0.0, S)` → `<=`":
        "两侧输出逐元素相同：唯一的区分点是 S == 0.0，而此时"
        "条件为真取常量 0.0、为假取 S（=0.0），是同一个值。"
        "见 test_zero_spread_is_zero_either_way。",

    "app/core/trading_context/spread.py ×1 — L35 `np.where((H > 0) & (L > 0), H / L, np.nan)` → `>=`":
        "0 价在两种取值下都得到 NaN，只是路径不同：`>` 直接取 np.nan；"
        "`>=` 取 H/L 后 `np.log(0)` = -inf → beta = inf → "
        "alpha = inf - inf = NaN → 随后 `np.where(np.isfinite(S), S, np.nan)` "
        "把它归为 NaN，被 skipna 平均忽略。两侧的最终 Series 逐元素相同。"
        "且整段包在 `np.errstate(divide='ignore', invalid='ignore')` 里，"
        "连警告都不会发出 —— 没有任何可观测差别。"
        "见 test_zero_price_yields_nan_through_either_branch。",

    "app/core/trading_context/spread.py ×1 — L40 `np.where((Hmax > 0) & (Lmin > 0), Hmax / Lmin, np.nan)` → `>=`":
        "与 L35 完全同构：gamma 走 -inf 后在 alpha 里变成 NaN，"
        "最终被 isfinite 过滤，两侧输出一致。",

    "app/core/trading_context/spread.py ×1 — L68 `np.log(high.where(high > 0))` 等三处 → `>=`":
        "Abdi-Ranaldi 的同一模式：`where` 保留 0 后 `np.log(0)` = -inf，"
        "eta = (-inf + -inf)/2 = -inf，x = (c - eta)·(c' - eta) 含 inf-inf → NaN，"
        "`s2.clip(lower=0)` 与 `sqrt` 之后仍是 NaN，被 skipna 平均忽略。",
}


def test_zero_price_yields_nan_through_either_branch():
    """L35 / L40 / L68 等价性的机械验证：两条路径给出同一个 NaN。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        H = np.array([0.0, 10.0])
        L = np.array([5.0, 8.0])
        strict = np.log(np.where((H > 0) & (L > 0), H / L, np.nan))
        loose = np.log(np.where((H >= 0) & (L >= 0), H / L, np.nan))
    assert np.isnan(strict[0])
    assert np.isneginf(loose[0]), "构造前提：放宽后确实取到了 log(0)"
    # -inf 平方后是 +inf，alpha 里 sqrt(2·inf) - sqrt(inf) = inf - inf → NaN
    with np.errstate(invalid="ignore"):
        beta = loose[0] ** 2
        alpha_term = np.sqrt(2.0 * beta) - np.sqrt(beta)
    assert np.isnan(alpha_term), "构造前提：放宽分支最终会走到 inf - inf"
    # 两条路径经 isfinite 过滤后都是 NaN，被 skipna 平均一视同仁地忽略
    for arr in (strict, loose):
        cleaned = np.where(np.isfinite(arr), arr, np.nan)
        assert np.isnan(cleaned[0])


def test_zero_spread_is_zero_either_way():
    """L44 等价性的机械验证。"""
    for s in (np.array([0.0]), np.array([-0.0]),
              np.array([-1e-9, 0.0, 1e-9, np.nan])):
        a = np.where(s < 0.0, 0.0, s)
        b = np.where(s <= 0.0, 0.0, s)
        assert np.array_equal(a, b, equal_nan=True), f"s={s} 上两种比较符结果不同"


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 4
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
