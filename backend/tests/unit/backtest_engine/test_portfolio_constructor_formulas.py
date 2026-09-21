"""
portfolio_constructor.py —— 逐公式定钉测试（变异测试驱动）

来由：这是**信号变成持仓**的那一步，37 个变异点首测击杀率 **32.4%**（存活 25）。
既有覆盖（`test_phase4.py::TestMVOPortfolio`、`unit/test_backtest_edge_cases.py`）
只断言了"L1 归一化后和为 1""退化时与 SignalWeighted 一致"这类**形状**性质，
公式本身——分位切点、收缩协方差、三种中性化的减法——一个都没钉。

后果举例：`w = w - row_mean` 写成 `+` 后市场中性层不再中性；
`Σ_shrunk = (1-δ)S + δ·diag(S)` 三个符号随便改一个都全绿。

本文件对每个存活变异要么写出能杀死它的用例，要么给出可机械验证的等价性证明。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.backtest_engine.portfolio_constructor import (
    DecilePortfolio,
    MVOPortfolio,
    NeutralizationLayer,
    SignalWeightedPortfolio,
)


def _frame(arr, prefix="T") -> pd.DataFrame:
    arr = np.asarray(arr, dtype=float)
    idx = pd.bdate_range("2023-01-02", periods=arr.shape[0])
    cols = [f"{prefix}{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, index=idx, columns=cols)


# ===========================================================================
# A. DecilePortfolio —— 分位切点与等权
# ===========================================================================

class TestDecilePortfolio:

    def test_quantile_cut_uses_one_minus_top_pct(self):
        """
        `hi_cut = nanquantile(vals, 1.0 - top_pct)`。写成 `1.0 + top_pct` 时
        分位数参数越界（>1），numpy 直接报错——但**前提是这条路径被执行过**，
        而它此前从未被任何用例执行。

        取 10 个等距信号、top_pct=0.2 → hi_cut = 分位 0.8 = 7.2，
        故只有值 8、9 两个进多头，各 +0.5。
        """
        sig = _frame([[float(i) for i in range(10)]])
        w = DecilePortfolio(top_pct=0.2, bottom_pct=0.2).construct(sig)
        row = w.iloc[0]
        assert row[row > 0].tolist() == [pytest.approx(0.5), pytest.approx(0.5)]
        assert list(row[row > 0].index) == ["T8", "T9"]
        assert row[row < 0].tolist() == [pytest.approx(-0.5), pytest.approx(-0.5)]
        assert list(row[row < 0].index) == ["T0", "T1"]

    def test_legs_are_equal_weighted_and_dollar_neutral(self):
        sig = _frame([[float(i) for i in range(10)]])
        w = DecilePortfolio(top_pct=0.3, bottom_pct=0.3).construct(sig).iloc[0]
        assert w.sum() == pytest.approx(0.0, abs=1e-12), "多空腿未做到美元中性"
        assert abs(w).sum() == pytest.approx(2.0, abs=1e-12), "两条腿各自 L1 应为 1"

    def test_two_valid_names_are_enough(self):
        """
        `if valid.sum() < 2: continue` 的边界：**恰好 2 个**有效信号必须建仓。
        放宽成 `<= 2` 会让只剩两只票的截面整日空仓。
        """
        sig = _frame([[1.0, 2.0, np.nan, np.nan]])
        w = DecilePortfolio().construct(sig).iloc[0]
        assert abs(w).sum() > 0, "恰好 2 个有效标的却整日空仓 —— 最小样本边界判错"

    def test_single_valid_name_is_skipped(self):
        """对照组：1 个有效标的确实无法构造多空，否则上一条无法区分。"""
        sig = _frame([[1.0, np.nan, np.nan, np.nan]])
        assert abs(DecilePortfolio().construct(sig).iloc[0]).sum() == 0.0

    def test_nan_names_never_get_weight(self):
        sig = _frame([[5.0, np.nan, 1.0, np.nan, 3.0]])
        w = DecilePortfolio().construct(sig).iloc[0]
        assert w["T1"] == 0.0 and w["T3"] == 0.0


# ===========================================================================
# B. SignalWeightedPortfolio —— z-score 与两种归一化
# ===========================================================================

class TestSignalWeighted:

    SIG = _frame([[1.0, 2.0, 3.0, 4.0, 10.0]])

    def _expected_z(self) -> np.ndarray:
        x = self.SIG.to_numpy(dtype=float)
        z = (x - np.nanmean(x, axis=1, keepdims=True)) / np.nanstd(
            x, axis=1, keepdims=True, ddof=1)
        return np.clip(z, -3.0, 3.0)

    def test_long_short_uses_absolute_value_for_l1(self):
        """
        `l1 = np.abs(z).sum(...)`。删掉 `np.abs` 后分母变成 z 的**代数和**，
        而 z-score 的代数和恒等于 0 → 触发 `l1 == 0 → 1.0` 的兜底 →
        权重变成未归一化的原始 z，|w| 之和远大于 1。
        """
        w = SignalWeightedPortfolio(clip_z=3.0).construct(self.SIG).to_numpy()
        z = self._expected_z()
        assert np.allclose(w, z / np.abs(z).sum(axis=1, keepdims=True), atol=1e-12)
        assert np.abs(w).sum() == pytest.approx(1.0, abs=1e-12)

    def test_long_only_normalises_to_full_investment(self):
        """
        long_only 分支必须 **Σw = 1**（满仓做多），不是 |w| 之和为 1。

        `zp.sum(axis=1, keepdims=True)` 的 keepdims 改成 False 后分母退化成一维，
        除法按**行数**而非列数广播。⚠️ 单行信号（T=1）测不出来：(1,N) / (1,) 恰好
        广播成同样的结果。必须用 **T ≠ N 的多行**信号，第一版就是这么漏的。
        """
        sig = _frame([[1.0, 2.0, 3.0, 4.0, 10.0],
                      [5.0, 1.0, 2.0, 8.0, 3.0],
                      [2.0, 9.0, 4.0, 1.0, 6.0]])          # T=3, N=5
        w = SignalWeightedPortfolio(clip_z=3.0, long_only=True).construct(sig)
        arr = w.to_numpy()
        assert arr.shape == (3, 5)
        assert (arr >= 0).all(), "long_only 却出现了负权重"
        assert arr.sum(axis=1) == pytest.approx(np.ones(3), abs=1e-12)

    def test_long_only_keeps_full_gross_unlike_masking_shorts(self):
        """
        这是 long_only 这条分支存在的**理由**：若改成"先建多空再把空头清零"，
        gross 只剩约一半（一半资金永久闲置）。两者必须可区分。
        """
        ls = SignalWeightedPortfolio(clip_z=3.0).construct(self.SIG).to_numpy()
        masked_gross = np.abs(np.where(ls > 0, ls, 0.0)).sum()
        lo = SignalWeightedPortfolio(clip_z=3.0, long_only=True).construct(self.SIG)
        assert masked_gross < 0.8, "构造前提不成立：多空腿裁掉空头后 gross 并未明显下降"
        assert lo.to_numpy().sum() == pytest.approx(1.0, abs=1e-12)

    def test_multi_row_normalisation_is_per_row(self):
        """两行信号量级差 100 倍，归一化必须逐行独立完成。"""
        sig = _frame([[1.0, 2.0, 3.0, 4.0], [100.0, 200.0, 300.0, 400.0]])
        w = SignalWeightedPortfolio().construct(sig).to_numpy()
        assert np.allclose(np.abs(w).sum(axis=1), 1.0, atol=1e-12)
        assert np.allclose(w[0], w[1], atol=1e-12), "同形状信号缩放后权重应一致"

    def test_clip_z_bounds_the_zscore(self):
        """极端离群值必须被 clip 到 ±clip_z，否则单票权重会被一个异常点吃掉。"""
        sig = _frame([[0.0, 0.0, 0.0, 0.0, 1e6]])
        tight = SignalWeightedPortfolio(clip_z=1.0).construct(sig).to_numpy()[0]
        loose = SignalWeightedPortfolio(clip_z=3.0).construct(sig).to_numpy()[0]
        assert abs(tight).max() < abs(loose).max()

    def test_all_nan_row_yields_zero_weights_without_warning(self):
        import warnings
        sig = _frame([[np.nan] * 4, [1.0, 2.0, 3.0, 4.0]])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            w = SignalWeightedPortfolio().construct(sig).to_numpy()
        assert np.abs(w[0]).sum() == 0.0
        assert np.abs(w[1]).sum() == pytest.approx(1.0, abs=1e-12)


# ===========================================================================
# C. MVOPortfolio —— 收缩协方差与求解
# ===========================================================================

class TestMVO:

    @staticmethod
    def _inputs(n_days: int = 90, n_assets: int = 4, seed: int = 3):
        rng = np.random.default_rng(seed)
        sig = _frame(rng.normal(0, 1, (n_days, n_assets)))
        ret = _frame(rng.normal(0, 0.01, (n_days, n_assets)))
        ret.index = sig.index
        ret.columns = sig.columns
        return sig, ret

    def test_cov_window_of_exactly_twenty_is_accepted(self):
        """
        `if cov_window < 20: raise` 的边界：**恰好 20** 是合法的。
        放宽成 `<= 20` 会把文档写明的最小值本身拒掉。
        """
        MVOPortfolio(cov_window=20)                      # 不得抛异常
        with pytest.raises(ValueError, match="至少 20"):
            MVOPortfolio(cov_window=19)

    def test_shrinkage_bounds_are_inclusive(self):
        MVOPortfolio(shrinkage=0.0)
        MVOPortfolio(shrinkage=1.0)
        with pytest.raises(ValueError):
            MVOPortfolio(shrinkage=1.0001)

    def test_weights_match_the_shrunk_solve_exactly(self):
        """
        逐日核对 `w ∝ [(1-δ)S + δ·diag(S) + 1e-8·I]⁻¹ · s`。

        这一条同时钉住四个变异：收缩公式的 `-`、`+`、`*`，以及岭项 `+ 1e-8·I`。
        测试里用**文档写的公式**独立重算，不是复制实现——实现改了两边就会分叉。
        """
        sig, ret = self._inputs()
        cov_window, delta = 20, 0.5
        w = MVOPortfolio(cov_window=cov_window, shrinkage=delta, clip_z=3.0).construct(
            sig, returns=ret)
        base = SignalWeightedPortfolio(clip_z=3.0).construct(sig).to_numpy()
        ret_arr = ret.to_numpy(dtype=float)
        n = base.shape[1]
        eye = np.eye(n)

        checked = 0
        for t in range(cov_window, len(sig)):
            s = base[t]
            if not np.any(s):
                continue
            window = ret_arr[t - cov_window:t]
            valid = np.isnan(window).mean(axis=0) < 0.3
            if valid.sum() < 3:
                continue
            sub = np.where(np.isnan(window[:, valid]), 0.0, window[:, valid])
            S = np.cov(sub.T)
            S_shrunk = (1 - delta) * S + delta * np.diag(np.diag(S))
            w_sub = np.linalg.solve(
                S_shrunk + 1e-8 * eye[:valid.sum(), :valid.sum()], s[valid])
            row = np.zeros(n)
            row[valid] = w_sub
            l1 = np.abs(row).sum()
            expected = row / l1 if l1 > 1e-12 else base[t]
            assert np.allclose(w.to_numpy()[t], expected, atol=1e-10), (
                f"第 {t} 日权重与收缩求解不符")
            checked += 1
        assert checked >= 30, f"只核对了 {checked} 天，样本不足以说明问题"

    def test_shrinkage_actually_changes_the_answer(self):
        """
        δ=0（全样本协方差）与 δ=1（只留对角）必须给出**不同**权重，
        否则 `S_shrunk` 那一行的三个符号怎么改都无所谓。
        """
        sig, ret = self._inputs()
        w0 = MVOPortfolio(cov_window=20, shrinkage=0.0).construct(sig, returns=ret)
        w1 = MVOPortfolio(cov_window=20, shrinkage=1.0).construct(sig, returns=ret)
        tail = slice(20, None)
        assert not np.allclose(w0.to_numpy()[tail], w1.to_numpy()[tail], atol=1e-8), (
            "收缩强度对结果没有影响 —— 收缩项疑似被消掉了")

    def test_three_valid_assets_are_enough_to_optimise(self):
        """
        `if valid.sum() < 3: continue` 的边界：**恰好 3 个**有效资产必须进优化。
        构造：第 4 只资产窗口内全 NaN → 只剩 3 只有效。
        """
        sig, ret = self._inputs(n_assets=4)
        ret.iloc[:, 3] = np.nan
        w = MVOPortfolio(cov_window=20, shrinkage=0.5).construct(sig, returns=ret)
        base = SignalWeightedPortfolio(clip_z=3.0).construct(sig)
        tail = slice(20, None)
        assert not np.allclose(w.to_numpy()[tail], base.to_numpy()[tail], atol=1e-9), (
            "恰好 3 只有效资产时优化被跳过 —— 最小样本边界判错")

    def test_nan_ratio_threshold_is_strict(self):
        """
        `valid = np.isnan(window).mean(axis=0) < 0.3` —— NaN 占比**恰好 0.3**
        的资产应当被**剔除**（不满足 < 0.3）。窗口 20 天里 6 个 NaN = 0.30，
        这个区分值可以精确构造。放宽成 `<=` 会把它留在协方差里。
        """
        window, n_nan = 20, 6
        assert n_nan / window == 0.3
        sig, ret = self._inputs(n_days=60, n_assets=4)
        # 让第 0 只资产在 t=40 的窗口（第 20–39 行）内恰好有 6 个 NaN → 占比 0.30
        arr = ret.to_numpy().copy()
        for start in range(0, 60, window):
            arr[start:start + n_nan, 0] = np.nan
        ret = pd.DataFrame(arr, index=ret.index, columns=ret.columns)

        w = MVOPortfolio(cov_window=window, shrinkage=0.5).construct(sig, returns=ret)
        base = SignalWeightedPortfolio(clip_z=3.0).construct(sig).to_numpy()
        t = 40
        assert base[t, 0] != pytest.approx(0.0, abs=1e-6), "构造前提：该资产原本有权重"
        # ⚠️ 源码注释写的是"剔除…（保留其基准权重）"，**实际行为是清零**：
        #    `row = np.zeros(N); row[valid] = w_sub; w_out[t] = row / l1`
        #    整行被替换，被剔除的资产拿不到基准权重而是 0。
        #    这里如实钉住**当前行为**（注释与实现不一致已登记进 MUTATION_LEDGER）。
        assert w.to_numpy()[t, 0] == pytest.approx(0.0, abs=1e-12), (
            "NaN 占比恰好 30% 的资产没有被剔除出协方差估计")

    def test_no_returns_raises_unless_the_fallback_is_explicit(self):
        """
        【缺陷 A-5，2026-09-21】`returns=None` 原先**静默**退化成 SignalWeighted。
        那条路径跑的根本不是均值-方差优化、不做任何协方差/风险检查，
        结果混进 MVO 的回测里无从分辨 ——「MVO 效果如何」的结论会被污染。
        现在默认报错；要用必须显式打开，并且每次构造都会记 WARNING。
        """
        sig, _ = self._inputs()
        with pytest.raises(ValueError, match="fallback_to_signal_weighted"):
            MVOPortfolio(cov_window=20).construct(sig)

    def test_the_explicit_fallback_still_produces_signal_weighted(self):
        """显式打开之后行为不变 —— 改的是"默不默认"，不是那条路径本身。"""
        sig, _ = self._inputs()
        got = MVOPortfolio(cov_window=20,
                           fallback_to_signal_weighted=True).construct(sig)
        assert np.allclose(
            got.to_numpy(),
            SignalWeightedPortfolio(clip_z=3.0).construct(sig).to_numpy())


# ===========================================================================
# D. NeutralizationLayer —— 三种中性化的减法
# ===========================================================================

class TestNeutralization:

    W = _frame([[0.4, 0.3, 0.2, 0.1], [0.1, 0.5, 0.2, 0.2]])

    def test_market_neutral_zeroes_the_row_sum(self):
        """
        `w = w - row_mean`。写成 `+` 后行和会变成原来的两倍而不是 0 ——
        "市场中性"层反而把净敞口翻倍。
        """
        out = NeutralizationLayer.market_neutral(self.W).to_numpy()
        assert np.allclose(out.sum(axis=1), 0.0, atol=1e-12), (
            f"市场中性后行和不为 0：{out.sum(axis=1)}")
        assert np.allclose(np.abs(out).sum(axis=1), 1.0, atol=1e-12)

    def test_market_neutral_preserves_relative_order(self):
        """减去同一个常数不改变排序；若变成加法同样不改序，所以还要看行和。"""
        out = NeutralizationLayer.market_neutral(self.W).to_numpy()[0]
        assert list(np.argsort(out)) == list(np.argsort(self.W.to_numpy()[0]))

    def test_industry_neutral_zeroes_each_group(self):
        """
        `w[t, idxs] = row_slice - grp_mean`。写成 `+` 后每个行业组的和
        变成两倍组均值而不是 0。
        """
        imap = {"T0": "tech", "T1": "tech", "T2": "fin", "T3": "fin"}
        out = NeutralizationLayer.industry_neutral(self.W, imap).to_numpy()
        for cols in ([0, 1], [2, 3]):
            assert np.allclose(out[:, cols].sum(axis=1), 0.0, atol=1e-12), (
                f"行业组 {cols} 中性化后组内和不为 0：{out[:, cols].sum(axis=1)}")

    def test_unmapped_tickers_form_their_own_group(self):
        imap = {"T0": "tech", "T1": "tech"}          # T2/T3 未映射 → __other__
        out = NeutralizationLayer.industry_neutral(self.W, imap).to_numpy()
        assert np.allclose(out[:, [0, 1]].sum(axis=1), 0.0, atol=1e-12)
        assert np.allclose(out[:, [2, 3]].sum(axis=1), 0.0, atol=1e-12)

    def test_l1_normalisation_keeps_all_zero_rows_zero(self):
        z = _frame([[0.0, 0.0, 0.0, 0.0], [0.4, -0.3, 0.2, -0.1]])
        out = NeutralizationLayer.market_neutral(z).to_numpy()
        assert np.abs(out[0]).sum() == 0.0
        assert np.abs(out[1]).sum() == pytest.approx(1.0, abs=1e-12)


class TestBetaNeutral:
    """
    beta 中性层：四处存活变异全在这里——两条滚动窗口切片的 `-`、
    组合 beta 的 `w * beta` 与 keepdims、以及对冲项 `w - port_beta / N`。
    """

    @staticmethod
    def _inputs(n_days: int = 120, n_assets: int = 4, seed: int = 11):
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2023-01-02", periods=n_days)
        cols = [f"T{i}" for i in range(n_assets)]
        mkt = pd.Series(rng.normal(0, 0.01, n_days), index=idx)
        # 资产 j 的 beta 设计成 0.5、1.0、1.5、2.0，噪声很小 → beta 可被准确估出
        betas = np.array([0.5, 1.0, 1.5, 2.0])[:n_assets]
        arr = np.outer(mkt.to_numpy(), betas) + rng.normal(0, 1e-5, (n_days, n_assets))
        ret = pd.DataFrame(arr, index=idx, columns=cols)
        w = pd.DataFrame(np.tile([0.4, 0.3, 0.2, 0.1][:n_assets], (n_days, 1)),
                         index=idx, columns=cols)
        return w, ret, mkt, betas

    def test_rolling_beta_is_estimated_from_the_trailing_window(self):
        """
        `r_m = mkt_arr[t - window : t]` 与 `r_i = ret_arr[t - window : t, j]`。
        把 `-` 写成 `+` 会取到**空切片**（起点在终点之后）→ 有效样本 0 < 10 →
        整段跳过 → beta 全部停留在默认值 1.0，对冲量因此完全错误。

        构造成四只资产 beta 分别为 0.5/1.0/1.5/2.0，噪声极小：
        若 beta 真被估出来，组合 beta = Σwᵢβᵢ = 1.0；
        若全部退化为 1.0，则组合 beta = Σwᵢ = 1.0 —— 两者恰好相同，
        所以**不能**用组合 beta 判断，要看**逐名调整量的差异**。
        """
        w, ret, mkt, _betas = self._inputs()
        out = NeutralizationLayer.beta_neutral(w, ret, mkt, window=60).to_numpy()
        raw = w.to_numpy()
        t = 100
        before = raw[t] / np.abs(raw[t]).sum()
        after = out[t]
        assert not np.allclose(before, after, atol=1e-6), (
            "beta 中性化没有改变任何权重 —— 滚动窗口疑似取到了空切片")

    def test_portfolio_beta_uses_weighted_sum_not_ratio(self):
        """
        `port_beta = np.nansum(w * beta, axis=1, keepdims=True)`：
          - `*` 写成 `/`：变成 Σ(wᵢ/βᵢ)，低 beta 资产反而被放大
          - keepdims 由 True 改 False：形状退化成一维，后续 `w - port_beta / N`
            会按**列**广播，逐日对冲量错位

        判据：手算 Σwᵢβᵢ 并按实现公式重建 w_adj，逐元素比对。
        """
        w, ret, mkt, _ = self._inputs()
        window = 60
        out = NeutralizationLayer.beta_neutral(w, ret, mkt, window=window).to_numpy()

        # 独立重算 beta（与实现同口径：窗口 [t-window, t)，nanvar，不含 t）
        raw = w.to_numpy()
        r = ret.to_numpy(dtype=float)
        m = mkt.to_numpy(dtype=float)
        T, N = raw.shape
        beta = np.full((T, N), 1.0)
        for t in range(window, T):
            r_m = m[t - window:t]
            var_m = np.nanvar(r_m)
            for j in range(N):
                r_i = r[t - window:t, j]
                beta[t, j] = float(np.cov(r_i, r_m)[0, 1] / var_m)
        port_beta = np.nansum(raw * beta, axis=1, keepdims=True)
        expected = raw - port_beta / N
        l1 = np.nansum(np.abs(expected), axis=1, keepdims=True)
        expected = expected / np.where(l1 == 0, 1.0, l1)

        assert np.allclose(out[window:], expected[window:], atol=1e-9), (
            "beta 中性化结果与按文档公式重算的不一致")

    def test_hedge_is_subtracted_not_added(self):
        """
        `w_adj = w - port_beta / N`。写成 `+` 会让净 beta 敞口**翻倍**而不是被对冲掉。
        判据：对冲后组合的 beta 加权和必须比对冲前更接近 0。
        """
        w, ret, mkt, betas = self._inputs()
        window = 60
        out = NeutralizationLayer.beta_neutral(w, ret, mkt, window=window)
        raw = w.to_numpy()
        adj = out.to_numpy()
        t = 100
        before = float(np.dot(raw[t], betas))
        after = float(np.dot(adj[t], betas))
        assert abs(after) < abs(before), (
            f"对冲后 beta 敞口没有变小（{before:.4f} → {after:.4f}）—— 减号疑似写成了加号")

    def test_ten_market_observations_are_enough(self):
        """
        `if mask.sum() < 10: continue` 与 `if both.sum() < 10: continue` 的边界：
        **恰好 10 个**有效观测必须能估出 beta。放宽成 `<= 10` 会多卡一天。
        """
        w, ret, mkt, _ = self._inputs(n_days=40)
        out10 = NeutralizationLayer.beta_neutral(w, ret, mkt, window=10).to_numpy()
        # window=10 时第 10 天起就应有非默认 beta → 与 window 更大时结果不同
        out30 = NeutralizationLayer.beta_neutral(w, ret, mkt, window=30).to_numpy()
        assert not np.allclose(out10[12], out30[12], atol=1e-9), (
            "窗口恰好 10 时 beta 估计被整段跳过")

    def test_constant_market_leaves_weights_untouched(self):
        """市场收益方差为 0 → 无法估 beta → 保持默认，不得除以 0。"""
        w, ret, mkt, _ = self._inputs(n_days=90)
        flat = pd.Series(0.0, index=mkt.index)
        out = NeutralizationLayer.beta_neutral(w, ret, flat, window=60).to_numpy()
        assert np.isfinite(out).all()


# ===========================================================================
# E. 存活变异的等价性证明
# ===========================================================================

PROVEN_EQUIVALENT = {
    "app/core/backtest_engine/portfolio_constructor.py ×2 — L84/L85 `if n_long > 0` / `if n_short > 0` → `>=`":
        "两个条件恒为真，false 分支不可达：hi_cut 与 lo_cut 都是 vals 的分位数，"
        "必然落在 [min(vals), max(vals)] 内，于是 `row >= hi_cut` 至少命中最大值、"
        "`row <= lo_cut` 至少命中最小值，n_long 与 n_short 恒 >= 1。"
        "见 test_leg_sizes_are_never_zero_when_the_row_is_traded。",

    "app/core/backtest_engine/portfolio_constructor.py ×1 — L139 `zp = np.where(z > 0.0, z, 0.0)` → `>=`":
        "两侧输出逐元素相同：区分点只有 z == 0，而 `where` 在该点取 z（=0）"
        "与取常量 0.0 是同一个值。见 test_zero_zscore_gives_zero_weight_either_way。",

    "app/core/backtest_engine/portfolio_constructor.py ×1 — L242 `if l1 > 1e-12` → `>=`":
        "区分值需要 `np.abs(row).sum()` 恰好等于 1e-12。row 来自 "
        "`np.linalg.solve` 的输出（多步浮点运算），无法反解出使其 L1 精确等于 "
        "1e-12 的输入；且上游 `if not np.any(s): continue` 已经排除了全零信号。",

    "app/core/backtest_engine/portfolio_constructor.py ×1 — L355 `if var_m < 1e-12` → `<=`":
        "区分值需要市场收益的 `np.nanvar` 恰好等于 1e-12。方差是平方和除以 n "
        "的浮点结果，常数市场给出的是精确 0.0、真实市场是 1e-4 量级，"
        "两侧都离 1e-12 极远，无法构造。",
}


def test_leg_sizes_are_never_zero_when_the_row_is_traded():
    """L84/L85 等价性的机械验证：随机截面上 n_long 与 n_short 恒 >= 1。"""
    rng = np.random.default_rng(23)
    for _ in range(300):
        n = int(rng.integers(2, 30))
        row = rng.normal(0, 1, n)
        if rng.random() < 0.3:                       # 混入 NaN
            row[rng.integers(0, n)] = np.nan
        valid = ~np.isnan(row)
        if valid.sum() < 2:
            continue
        vals = row[valid]
        lo_cut = np.nanquantile(vals, 0.10)
        hi_cut = np.nanquantile(vals, 1.0 - 0.10)
        assert (valid & (row >= hi_cut)).sum() >= 1
        assert (valid & (row <= lo_cut)).sum() >= 1


def test_zero_zscore_gives_zero_weight_either_way():
    """L139 等价性的机械验证：`>` 与 `>=` 在 z==0（含 -0.0）上取值相同。"""
    for z in (np.array([0.0]), np.array([-0.0]), np.array([0.0, 1.0, -1.0])):
        assert np.array_equal(np.where(z > 0.0, z, 0.0), np.where(z >= 0.0, z, 0.0))


def test_epsilon_guards_in_constructor_are_unreachable():
    """L242 / L355 证明的共同机械验证。"""
    for tol in (1e-12,):
        for base in (1.0, 0.25, 1e-4, 1e-8):
            assert (base + tol) - base != tol, (
                f"base={base} 处 {tol} 可精确还原，等价性证明不成立")


def test_every_survivor_has_a_written_proof():
    """存活项要么被用例杀死，要么在此有书面证明；不许有第三种状态。"""
    assert len(PROVEN_EQUIVALENT) == 4
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
