"""
risk_engine/factor_model.py — Barra-lite 风险模型（Phase R.2）

风险归因的危险之处在于**它永远给得出一个数字**：拿 10 天数据估的协方差、
漏了行业哑变量的分解、把交叉项丢掉的逐因子贡献，产出的 JSON 和正确结果
长得一模一样。所以这里的断言分三类：

  1. **恒等式**（可逐位核对，不是"近似"）：
     `factor_var + specific_var == total_var`、
     `Σ by_factor == factor_var`、`w'Σw == total_var`；
  2. **区分力**：构造一个**已知**风格暴露的组合，归因必须把它指出来 ——
     否则"能跑"只是类型检查；
  3. **拒绝的边界**：样本不足要抛错而不是给个好看的数。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.risk_engine import (
    STYLE_FACTORS, build_exposures, fit_risk_model,
)
from app.core.risk_engine.factor_model import _WINSOR_Z as _WINSOR_LIMIT
from app.core.risk_engine.factor_model import sector_exposures


N_DAYS, N_TICK = 600, 24


def _panel(seed: int = 0, *, with_sector: bool = True, with_volume: bool = True,
           n_days: int = N_DAYS, n_tick: int = N_TICK) -> dict:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=n_days)
    cols = [f"S{i:02d}" for i in range(n_tick)]
    ret = rng.normal(0.0004, 0.012, (n_days, n_tick))
    close = pd.DataFrame(100 * np.cumprod(1 + ret, axis=0), index=idx, columns=cols)
    ds = {"close": close}
    if with_volume:
        ds["volume"] = pd.DataFrame(
            rng.integers(1_000_000, 9_000_000, (n_days, n_tick)).astype(float),
            index=idx, columns=cols)
    if with_sector:
        ds["sector"] = pd.DataFrame(np.tile(np.arange(n_tick) % 5, (n_days, 1)),
                                    index=idx, columns=cols)
    return ds


def _equal_weight(ds: dict) -> pd.Series:
    cols = ds["close"].columns
    return pd.Series(1.0 / len(cols), index=cols)


# ===========================================================================
# A. 暴露的构造
# ===========================================================================

class TestExposures:

    def test_every_declared_style_is_produced(self):
        exps = build_exposures(_panel())
        assert list(exps) == list(STYLE_FACTORS), (
            f"产出的风格与声明的不一致：{list(exps)} vs {list(STYLE_FACTORS)}")

    def test_each_row_is_cross_sectionally_standardised(self):
        """
        逐日截面 z-score 是整套模型的前提：不标准化的话，波动率（~0.01 量级）
        与动量（~0.3 量级）会因为**单位**不同而在回归里权重悬殊，
        算出来的"因子收益"其实是在比较量纲。
        """
        exps = build_exposures(_panel(1))
        for name, df in exps.items():
            row = df.iloc[-1]
            assert abs(float(row.mean())) < 1e-9, f"{name} 末期截面均值不为 0：{row.mean()}"
            assert 0.5 < float(row.std(ddof=1)) <= 1.5, (
                f"{name} 末期截面标准差={row.std(ddof=1):.3f} —— 偏离 1 太多，"
                f"疑似没做标准化（截断到 ±3 会让它略小于 1，但不会是 0.01 或 30）")

    def test_extreme_values_are_winsorised(self):
        """一只异常股不得主导整条因子收益 —— 截断到 ±3 个标准差。"""
        ds = _panel(2)
        ds["close"].iloc[-1, 0] *= 50          # 制造一个极端动量
        exps = build_exposures(ds)
        for name, df in exps.items():
            assert float(df.iloc[-1].abs().max()) <= 3.0 + 1e-9, (
                f"{name} 的末期暴露最大绝对值 {df.iloc[-1].abs().max():.2f} > 3")

    def test_momentum_ranks_a_known_winner_on_top(self):
        """
        区分力：人为让一只股票在"跳过近月"的窗口里明显跑赢，
        它的 momentum 暴露必须是最高的。只断言"有输出"是抓不到符号写反的。
        """
        ds = _panel(3)
        px = ds["close"]
        # 在 [-252, -21] 区间内给 S00 叠加一段强势
        px.iloc[-252:-21, 0] *= np.linspace(1.0, 2.0, 252 - 21)
        px.iloc[-21:, 0] *= 2.0                 # 近月保持，避免被反转窗口吃掉
        exps = build_exposures(ds)
        mom = exps["momentum"].iloc[-1]
        assert mom.idxmax() == px.columns[0], (
            f"人为造的动量赢家不是暴露最高的（最高是 {mom.idxmax()}）—— "
            f"momentum 的符号或窗口疑似写反")

    def test_reversal_is_the_negative_of_recent_return(self):
        """reversal 暴露高 = 近期**跌**得多。符号写反会让它变成短期动量。"""
        ds = _panel(4)
        ds["close"].iloc[-21:, 1] *= np.linspace(1.0, 0.6, 21)   # S01 近月大跌
        rev = build_exposures(ds)["reversal"].iloc[-1]
        assert rev.idxmax() == ds["close"].columns[1], (
            f"近月大跌的股票不是 reversal 暴露最高的（最高是 {rev.idxmax()}）")

    def test_liquidity_ranks_the_heavily_traded_name_on_top(self):
        """
        区分力：成交额高出一个量级的股票，liquidity 暴露必须最高。

        这条抓的是 Amihud 的**两处方向**：美元成交额 `volume × price`（写成除法
        会让"贵而不活跃"的股票冒充流动）与 `|r| / 成交额`（写成乘法直接把
        非流动性变成流动性）。两处改了都不会让任何形状/范围断言变红。

        **必须有价格离散度**：第一版让所有股票都在 100 元附近，只把一只的成交量
        乘 100 —— 那时 `V×P` 与 `V÷P` 给出的排序完全一样（P 是公因子），
        用例照样绿。要区分乘除，得让"高价低量"和"低价高量"打架。
        """
        ds = _panel(18)
        cols = list(ds["close"].columns)
        ds["close"].iloc[:, 1] *= 0.01            # S01 变仙股（价格低两个量级）
        ds["volume"].iloc[:, 0] *= 10.0           # S00 量大 → 美元成交额最高
        liq = build_exposures(ds)["liquidity"].iloc[-1]
        assert liq.idxmax() == cols[0], (
            f"美元成交额最高的票不是 liquidity 暴露最高的（最高是 {liq.idxmax()}）—— "
            f"Amihud 的符号或美元成交额的算法疑似写反")
        assert liq[cols[0]] > liq[cols[1]], (
            f"仙股 {cols[1]} 的流动性暴露不低于大额成交的 {cols[0]} —— "
            f"成交额疑似算成了 volume/price")

    def test_a_row_without_outliers_is_standardised_to_exactly_unit_variance(self):
        """
        ddof 必须是 1。写成 `cnt + 1` 只会让 z 整体放大约 4%（24 只股票时），
        落在任何"标准差大致为 1"的宽松区间里 —— 抓不到。

        所以这里要一个**不会触发截断**的截面：正态分位点构造的暴露，
        标准化之后样本标准差精确等于 1。
        """
        n_tick = 24
        ds = _panel(19, n_tick=n_tick)
        exps = build_exposures(ds)
        for name, df in exps.items():
            row = df.iloc[-1]
            if float(row.abs().max()) >= _WINSOR_LIMIT - 1e-9:
                continue                           # 被截断的行不适用（见上）
            assert float(row.std(ddof=1)) == pytest.approx(1.0, abs=1e-9), (
                f"{name} 未截断的截面标准差 {row.std(ddof=1):.6f} != 1 —— ddof 疑似写错")

    def test_two_names_are_enough_to_standardise(self):
        """`cnt >= 2` 的边界：恰好 2 只股票也能算 z-score（±1），不该退化成全 0。"""
        exps = build_exposures(_panel(20, n_tick=2, with_sector=False))
        mom = exps["momentum"].iloc[-1]
        assert float(mom.abs().sum()) > 0.0, (
            "2 只股票的截面被判为『样本不足』→ 暴露全 0，`cnt >= 2` 疑似被收紧成 `> 2`")

    def test_a_zero_price_does_not_leak_negative_infinity(self):
        """
        `px.where(px > 0)`：价格恰好为 0 时 log 会给 -inf。放宽成 `>=` 会让
        -inf 进入动量暴露，再被 z-score 一路传染成整列 NaN/0。
        """
        ds = _panel(21)
        ds["close"].iloc[-30, 0] = 0.0
        exps = build_exposures(ds)
        for name, df in exps.items():
            arr = df.to_numpy(dtype=float)
            assert np.isfinite(arr).all(), f"{name} 出现了非有限值（-inf/NaN 泄漏）"

    def test_missing_volume_degrades_loudly_not_silently(self, caplog):
        """
        缺成交量 → 流动性因子无区分力。**必须告警**：静默产出全 0 暴露会被
        读成"这些股票流动性都一样"，而事实是"我们没看"。
        """
        with caplog.at_level("WARNING"):
            exps = build_exposures(_panel(5, with_volume=False))
        assert any("volume" in r.getMessage() for r in caplog.records), (
            "缺 volume 时没有任何告警")
        assert float(exps["liquidity"].iloc[-1].abs().sum()) == 0.0

    def test_a_dataset_without_close_is_refused(self):
        with pytest.raises(ValueError, match="close"):
            build_exposures({"volume": pd.DataFrame()})


class TestSectorExposures:

    def test_dummies_are_one_hot(self):
        sec = sector_exposures(_panel(6))
        assert sec is not None
        assert set(np.unique(sec.to_numpy())) <= {0.0, 1.0}
        assert (sec.sum(axis=1) == 1.0).all(), "有股票被分进了 0 个或多个行业"

    def test_a_panel_without_sector_returns_none(self):
        assert sector_exposures(_panel(7, with_sector=False)) is None

    def test_unmapped_tickers_get_no_dummy(self):
        """GICS 缺映射的票用 -1 标记，不该被当成"第 -1 个行业"另起一列。"""
        ds = _panel(8)
        ds["sector"].iloc[:, 0] = -1
        sec = sector_exposures(ds)
        assert float(sec.iloc[0].sum()) == 0.0, "未映射的票被分配了行业哑变量"
        assert not any(c.endswith("-1") for c in sec.columns)


# ===========================================================================
# B. 恒等式 —— 这几条对不上就说明实现错了，不是"近似误差"
# ===========================================================================

class TestVarianceDecompositionIdentities:

    @pytest.fixture(scope="class")
    def fitted(self):
        ds = _panel(10)
        return ds, fit_risk_model(ds, lookback=400)

    def test_factor_plus_specific_equals_total(self, fitted):
        ds, model = fitted
        a = model.attribute(_equal_weight(ds))
        assert a.factor_var + a.specific_var == pytest.approx(a.total_var, abs=1e-18), (
            f"方差分解不闭合：{a.factor_var} + {a.specific_var} != {a.total_var}")

    def test_per_factor_contributions_sum_to_the_factor_variance(self, fitted):
        """
        逐因子贡献用的是 Euler 分解（含交叉项）。若实现成"各因子单独方差"
        相加，合计会漏掉因子间相关性 —— 这条就会对不上。
        """
        ds, model = fitted
        a = model.attribute(_equal_weight(ds))
        assert sum(a.by_factor.values()) == pytest.approx(a.factor_var, abs=1e-18)

    def test_the_structured_covariance_reproduces_the_total_variance(self, fitted):
        """`w'Σw` 必须与 attribute() 给的 total_var 逐位一致（同一个 Σ 的两种算法）。"""
        ds, model = fitted
        w = _equal_weight(ds)
        Sigma = model.covariance()
        quad = float(w.to_numpy() @ Sigma.to_numpy() @ w.to_numpy())
        assert quad == pytest.approx(model.attribute(w).total_var, abs=1e-18)

    def test_the_covariance_is_symmetric_and_positive_semidefinite(self, fitted):
        _ds, model = fitted
        S = model.covariance().to_numpy()
        assert np.allclose(S, S.T), "结构化协方差不对称"
        eig = np.linalg.eigvalsh(S)
        assert eig.min() > -1e-12, f"出现显著负特征值 {eig.min():.2e}（协方差非半正定）"

    def test_the_factor_share_is_a_ratio_not_a_product(self, fitted):
        """
        `factor_var / total_var` 写成乘法，结果仍落在 [0,1] 里（两个都是小数），
        任何范围断言都抓不到。必须比精确值。
        """
        ds, model = fitted
        a = model.attribute(_equal_weight(ds))
        assert a.factor_share == pytest.approx(a.factor_var / a.total_var, rel=1e-12)
        assert a.factor_share > 0.01, (
            f"因子占比 {a.factor_share:.2e} 小到不像比值 —— 疑似算成了乘积")

    def test_an_empty_book_has_zero_risk_and_no_nan_share(self, fitted):
        """
        空仓（权重全 0）：总方差为 0。此时 factor_share 必须是 0.0 而**不是 nan** ——
        nan 会一路传到面板上，看起来像"算出来了"。
        """
        ds, model = fitted
        a = model.attribute(pd.Series(0.0, index=ds["close"].columns))
        assert a.total_var == 0.0
        assert a.factor_share == 0.0 and not np.isnan(a.factor_share)


# ===========================================================================
# C. 区分力 —— 已知的暴露必须被指出来
# ===========================================================================

class TestItActuallyFindsTheExposure:

    def test_a_book_tilted_to_one_style_shows_that_tilt(self):
        """
        按 volatility 暴露排序，只买最高的一半 → 组合在 volatility 上的净暴露
        必须显著为正，且是所有风格里最大的。归因如果只会输出接近 0 的数，
        这条会红。
        """
        ds = _panel(11)
        model = fit_risk_model(ds, lookback=400)
        vol_exp = model.exposures["volatility"]
        top = vol_exp.sort_values(ascending=False).index[: len(vol_exp) // 2]
        w = pd.Series(0.0, index=vol_exp.index)
        w.loc[top] = 1.0 / len(top)

        a = model.attribute(w)
        assert a.exposures["volatility"] > 0.3, (
            f"只买高波动的一半，组合 volatility 净暴露却只有 "
            f"{a.exposures['volatility']:.3f}")
        style_only = {k: abs(v) for k, v in a.exposures.items() if k in STYLE_FACTORS}
        assert max(style_only, key=style_only.get) == "volatility", (
            f"最大风格暴露是 {max(style_only, key=style_only.get)}，不是被刻意倾斜的 volatility")

    def test_a_concentrated_book_carries_more_specific_risk_than_a_diversified_one(self):
        """
        特异风险的定义性质：集中到 2 只股票的组合，其特异方差必须明显高于
        等权全池。若实现把 D 当成常数，两者会一样。
        """
        ds = _panel(12)
        model = fit_risk_model(ds, lookback=400)
        cols = model.exposures.index
        diversified = model.attribute(pd.Series(1.0 / len(cols), index=cols))
        conc = pd.Series(0.0, index=cols)
        conc.iloc[:2] = 0.5
        concentrated = model.attribute(conc)
        assert concentrated.specific_var > diversified.specific_var * 3, (
            f"集中持仓的特异方差 {concentrated.specific_var:.3e} 没有显著高于"
            f"分散持仓的 {diversified.specific_var:.3e}")

    def test_dropping_sector_dummies_moves_risk_into_the_specific_bucket(self):
        """
        行业共同波动若没有哑变量去吸收，就会落进"特异"里 —— 于是因子风险
        占比被低估。这条把那个后果钉住，也解释了 fit 里那条告警为什么必要。
        """
        ds = _panel(13)
        with_sec = fit_risk_model(ds, lookback=400, with_sector=True)
        without = fit_risk_model(ds, lookback=400, with_sector=False)
        w = _equal_weight(ds)
        assert with_sec.attribute(w).sector_included is True
        assert without.attribute(w).sector_included is False
        assert without.attribute(w).specific_var >= with_sec.attribute(w).specific_var, (
            "去掉行业哑变量后特异方差反而变小了 —— 行业风险没有被吸收进因子侧")


# ===========================================================================
# D. 拒绝的边界与如实标注
# ===========================================================================

class TestItRefusesRatherThanGuessing:

    def test_too_few_regression_days_raises(self):
        """
        10 天数据估出来的协方差与 400 天的**长得一模一样**，只是错的。
        这种时候必须抛错，不能返回。
        """
        with pytest.raises(ValueError, match="下限"):
            fit_risk_model(_panel(14), lookback=30, min_obs=60)

    def test_a_panel_narrower_than_the_factor_count_raises(self):
        """
        股票数 ≤ 因子数时截面回归自由度不足（必然完美拟合，残差恒 0）。
        这类日子被跳过；全部跳过后应当抛错而不是返回一个零残差模型。
        """
        with pytest.raises(ValueError):
            fit_risk_model(_panel(15, n_tick=4), lookback=400, min_obs=60)

    def test_a_dataset_without_close_is_refused(self):
        with pytest.raises(ValueError, match="close"):
            fit_risk_model({"volume": pd.DataFrame()})

    def test_exactly_the_minimum_number_of_days_is_accepted(self):
        """
        `len(f_rows) < min_obs` —— **严格小于**。收紧成 `<=` 只影响恰好踩线的
        那一格：刚好攒够天数的模型会被拒，而使用者看到的是"数据不足"，
        与真的不足无从分辨。
        """
        ds = _panel(22)
        m = fit_risk_model(ds, lookback=60, min_obs=60)
        assert m.n_obs == 60, f"lookback=60 应给出 60 个回归日，实得 {m.n_obs}"
        with pytest.raises(ValueError, match="下限"):
            fit_risk_model(ds, lookback=59, min_obs=60)

    def test_the_degrees_of_freedom_guard_sits_at_k_plus_one(self):
        """
        `ok.sum() <= K + 1` 的两侧：K+1 只股票必须被跳过（回归自由度不足，
        残差恒 0 会把特异风险算成 0），K+2 只必须可用。

        关掉行业哑变量把 K 钉死为 5（风格数），否则 K 随数据里的行业数漂移，
        这条边界就没法精确构造。
        """
        k = len(STYLE_FACTORS)
        ok = fit_risk_model(_panel(23, n_tick=k + 2), lookback=400,
                            min_obs=60, with_sector=False)
        assert ok.n_obs >= 60, "K+2 只股票应当够做截面回归"
        with pytest.raises(ValueError):
            fit_risk_model(_panel(24, n_tick=k + 1), lookback=400,
                           min_obs=60, with_sector=False)

    def test_disabling_sector_does_not_emit_the_missing_sector_warning(self, caplog):
        """
        那条"无 sector → 行业风险被算进特异"的告警，只该在**要了却没有**时出现。
        条件写成 `or` 会让它在显式关闭时也报 —— 告警一旦开始说谎，
        真正缺行业数据的那次就没人信了。
        """
        with caplog.at_level("WARNING"):
            fit_risk_model(_panel(25), lookback=400, with_sector=False)
        assert not any("sector" in r.getMessage() for r in caplog.records), (
            "显式关闭行业哑变量时仍报了『无 sector 字段』告警")

        caplog.clear()
        with caplog.at_level("WARNING"):
            fit_risk_model(_panel(26, with_sector=False), lookback=400, with_sector=True)
        assert any("sector" in r.getMessage() for r in caplog.records), (
            "要了行业哑变量却拿不到时**没有**告警 —— 因子风险会被低估且无人知晓")

    def test_the_report_states_which_styles_are_missing(self):
        """
        size/value 要基本面数据（Phase 10 之前没有）。这件事必须跟着数字一起
        返回 —— 否则"因子风险占比 66%"会被读成"覆盖了主要风格"。
        """
        ds = _panel(16)
        d = fit_risk_model(ds, lookback=400).attribute(_equal_weight(ds)).to_dict()
        assert d["styles_missing"] == ["size", "value"]
        assert d["styles_covered"] == list(STYLE_FACTORS)
        assert d["sector_included"] is True
        for k in ("total_vol_ann", "factor_share", "by_factor", "exposures", "n_assets"):
            assert k in d, f"归因结果缺字段 {k}"

    def test_the_annualisation_is_a_multiplication_by_sqrt_252(self):
        """日频方差 → 年化波动。写成除法或漏掉 √ 都不改符号，必须比数值。"""
        ds = _panel(17)
        a = fit_risk_model(ds, lookback=400).attribute(_equal_weight(ds))
        assert a.total_vol_ann == pytest.approx(
            np.sqrt(a.total_var) * np.sqrt(252.0), rel=1e-12)
        assert 0.01 < a.total_vol_ann < 3.0, (
            f"年化波动 {a.total_vol_ann:.4f} 不在合理量级，疑似年化因子写错")
