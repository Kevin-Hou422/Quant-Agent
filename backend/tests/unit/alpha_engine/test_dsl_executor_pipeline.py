"""
alpha_engine/dsl_executor.py —— 执行管线的定钉测试（变异测试驱动）

来由：25 个变异点，首测击杀率 **36.0%**（存活 16）。

Executor 是"DSL 字符串 → (T×N) 信号面板"的唯一入口。它做四件会改变结果的事：
对齐、派生字段、求值、后处理。存活项正好铺满这四处：

  - `_align_dataset` 里 `not (f in _AUX 且 isinstance ndarray)` 的 `and`
    —— 放宽成 `or` 会把 groups/sector 的 (N,) 数组塞进 DataFrame 对齐流程
  - `_add_derived` 里三个 `X not in aligned and Y in aligned`
    —— 删掉 `not` / 放宽 `and` 会让**已有的 returns 被重新覆盖**，
    调用方精心准备的收益率（比如已做过除权调整的）被悄悄换成 log 差分
  - `vwap = (high + low + close) / 3` 的 `+` —— 典型价格改成 (high-low+close)/3
  - `winsorize_k`/`neutralize` 的 `mu ± k*sigma` 与 `x - mu`
    —— 缩尾上下界颠倒会把整截面压成一个数，去均值写成加均值会把
    多空信号整体平移，组合从中性变成单边

既有覆盖（test_dsl_operators / test_dsl_engine / test_leak_filter）只验证
"跑得通、形状对"，后处理那一段几乎没有断言。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.alpha_engine.dsl_executor import (
    Executor,
    _add_derived,
    _align_dataset,
    _to_arrays,
)


T, N = 10, 4
IDX = pd.bdate_range("2024-01-02", periods=T)
COLS = [f"T{i}" for i in range(N)]


def _frame(seed: int = 0, base: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(base + rng.normal(0, 1, (T, N)).cumsum(axis=0),
                        index=IDX, columns=COLS)


def _dataset(**extra) -> dict:
    close = _frame(0)
    ds = {
        "close": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "open": _frame(1),
        "volume": _frame(2, base=1e6),
    }
    ds.update(extra)
    return ds


# ===========================================================================
# A. 派生字段：只在缺失时补，绝不覆盖
# ===========================================================================

class TestDerivedFields:

    def test_existing_returns_are_not_overwritten(self):
        """
        `if "returns" not in aligned and "close" in aligned` —— 删掉 `not`
        会让**已有的 returns 每次都被重算覆盖**。调用方传进来的收益率
        往往已经做过除权/停牌处理，被 log(close/close.shift(1)) 顶掉之后
        所有基于 returns 的因子都换了口径，而且没有任何提示。
        """
        mine = pd.DataFrame(0.123, index=IDX, columns=COLS)
        out = _add_derived(dict(_dataset(returns=mine)))
        pd.testing.assert_frame_equal(out["returns"], mine,
                                      check_names=False,
                                      obj="调用方提供的 returns 被覆盖了")

    def test_returns_are_derived_when_absent(self):
        ds = _dataset()
        out = _add_derived(dict(ds))
        assert "returns" in out, "缺 returns 时没有补上"
        expected = np.log(ds["close"] / ds["close"].shift(1))
        np.testing.assert_allclose(out["returns"].to_numpy(), expected.to_numpy(),
                                   rtol=1e-12, equal_nan=True,
                                   err_msg="派生的 returns 不是对数收益")

    def test_returns_need_close_not_just_any_field(self):
        """`and` 放宽成 `or` 会让没有 close 时也去取 aligned["close"] → KeyError。"""
        out = _add_derived({"volume": _frame(3)})
        assert "returns" not in out, "没有 close 却造出了 returns"

    def test_vwap_is_the_typical_price(self):
        """
        `(high + low + close) / 3` —— 第一个 `+` 改成 `-` 会得到
        (high - low + close)/3，那是"真实波幅 + 收盘价"的混合物，
        与成交量加权均价毫无关系，但量纲接近、图形正常，看不出来。
        """
        ds = _dataset()
        out = _add_derived(dict(ds))
        expected = (ds["high"] + ds["low"] + ds["close"]) / 3.0
        np.testing.assert_allclose(out["vwap"].to_numpy(), expected.to_numpy(),
                                   rtol=1e-12, err_msg="vwap 不是典型价格 (H+L+C)/3")
        wrong = (ds["high"] - ds["low"] + ds["close"]) / 3.0
        assert not np.allclose(expected.to_numpy(), wrong.to_numpy()), (
            "构造的数据分不出 (H+L+C) 与 (H-L+C) —— 这条断言没有区分力")

    def test_existing_vwap_is_not_overwritten(self):
        mine = pd.DataFrame(7.0, index=IDX, columns=COLS)
        out = _add_derived(dict(_dataset(vwap=mine)))
        pd.testing.assert_frame_equal(out["vwap"], mine, check_names=False,
                                      obj="调用方提供的 vwap 被覆盖了")

    def test_vwap_requires_all_three_price_fields(self):
        out = _add_derived({"close": _frame(4), "high": _frame(5)})
        assert "vwap" not in out, "缺 low 却造出了 vwap"

    def test_sector_is_copied_into_groups_only_when_groups_is_absent(self):
        sector = np.array([0, 0, 1, 1])
        out = _add_derived(dict(_dataset(sector=sector)))
        np.testing.assert_array_equal(out["groups"], sector,
                                      err_msg="sector 没有被映射到 groups")

        existing = np.array([2, 2, 3, 3])
        out2 = _add_derived(dict(_dataset(sector=sector, groups=existing)))
        np.testing.assert_array_equal(out2["groups"], existing,
                                      err_msg="已有的 groups 被 sector 覆盖了")


# ===========================================================================
# B. 对齐：辅助字段透传
# ===========================================================================

class TestAlignment:

    def test_one_dimensional_group_arrays_pass_through_untouched(self):
        """
        `not (f in _AUX_PASSTHROUGH and isinstance(df, np.ndarray))` ——
        `and` 放宽成 `or` 会把 groups 这个 (N,) ndarray 当成面板去 reindex，
        ndarray 没有 .index/.columns，直接抛属性错误；
        更隐蔽的是它同时会把**名字叫 groups 的真 DataFrame** 排除出对齐。
        """
        groups = np.array([0, 0, 1, 1])
        _, _, aligned = _align_dataset(_dataset(groups=groups))
        assert isinstance(aligned["groups"], np.ndarray), "分组数组被当成面板处理了"
        np.testing.assert_array_equal(aligned["groups"], groups)

    def test_a_dataframe_named_groups_still_gets_aligned(self):
        """名字在 _AUX_PASSTHROUGH 里但类型是 DataFrame → 仍要参与对齐。"""
        g = pd.DataFrame(1.0, index=IDX[:5], columns=COLS[:2])
        idx, cols, aligned = _align_dataset({"close": _frame(0), "groups": g})
        assert isinstance(aligned["groups"], pd.DataFrame)
        assert len(aligned["groups"]) == len(idx), "DataFrame 形式的 groups 没被对齐"

    def test_union_of_indices_and_columns(self):
        a = pd.DataFrame(1.0, index=IDX[:5], columns=COLS[:2])
        b = pd.DataFrame(2.0, index=IDX[3:], columns=COLS[1:])
        idx, cols, aligned = _align_dataset({"close": a, "open": b})
        assert len(idx) == T and list(cols) == COLS, "对齐取的不是索引/列的并集"
        assert aligned["close"].shape == (T, N)

    def test_empty_dataset_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            _align_dataset({})

    def test_dataset_with_only_passthrough_fields_is_rejected(self):
        with pytest.raises(ValueError, match="no panel"):
            _align_dataset({"groups": np.array([0, 1])})

    def test_to_arrays_keeps_one_dimensional_aux_fields(self):
        arrays = _to_arrays({"close": _frame(0), "groups": np.array([0, 0, 1, 1])})
        assert arrays["groups"].shape == (N,), "透传字段被改了形状"
        assert arrays["close"].shape == (T, N)


# ===========================================================================
# C. 后处理：缩尾与中性化
# ===========================================================================

class TestPostprocess:

    @staticmethod
    def _signal_with_an_outlier() -> np.ndarray:
        x = np.tile(np.array([1.0, 2.0, 3.0, 50.0]), (3, 1))
        return x

    def test_winsorize_clips_the_outlier_but_keeps_dispersion(self):
        """
        `lo = mu - k*sigma` / `hi = mu + k*sigma`。两个符号互换会让 lo > hi，
        `np.clip` 在 lower > upper 时**一律返回 upper**，整截面塌成一个数 ——
        之后 cs_rank 全并列、权重全相等，因子彻底失效但回测照跑。
        """
        ex = Executor(validate=False, winsorize_k=1.0)
        x = self._signal_with_an_outlier()
        got = ex._postprocess(x.copy())
        mu, sd = np.nanmean(x, axis=1), np.nanstd(x, axis=1, ddof=1)
        assert got[0].max() <= mu[0] + sd[0] + 1e-9, "上界没有生效"
        assert got[0].min() >= mu[0] - sd[0] - 1e-9, "下界没有生效"
        assert got[0, 3] < 50.0, "极端值没有被缩尾"
        assert len(np.unique(np.round(got[0], 9))) > 1, (
            "整截面被压成了同一个数 —— 上下界疑似颠倒")

    def test_winsorize_bounds_are_exactly_mu_plus_minus_k_sigma(self):
        """
        `lo = mu - k*sigma` / `hi = mu + k*sigma` 里的 `*` 改成 `/`。

        只比"k 越大越宽松"是抓不住的：`k/sigma` 同样随 k 单调增。
        差别在于**与离散度的关系**：`k*sigma` 随离散度放大，`k/sigma` 随之收缩。
        这里直接钉死上界的**数值**，一处符号被改立刻对不上。
        """
        # 需要离群值**真的越界**才测得到上界：4 个数里放一个 1000，
        # sd 会被它自己撑大到越不过界（第一版就是这么失手的）。
        # 10 个近零值 + 1 个 1000 → mu+1.5σ ≈ 0.55×1000，离群值确实被削。
        x = np.array([np.concatenate([np.arange(10, dtype=float), [1000.0]])])
        k = 1.5
        mu = float(np.nanmean(x))
        sd = float(np.nanstd(x, ddof=1))
        assert x.max() > mu + k * sd, "构造的离群值没有越过上界，测不到 clip"
        got = Executor(validate=False, winsorize_k=k)._postprocess(x.copy())
        np.testing.assert_allclose(got.max(), mu + k * sd, rtol=1e-12,
                                   err_msg="缩尾上界不等于 mu + k*sigma")
        assert not np.isclose(mu + k * sd, mu + k / sd), (
            "构造的数据里 k*sigma 与 k/sigma 恰好相等 —— 这条断言没有区分力")

    def test_winsorize_lower_bound_is_exactly_mu_minus_k_sigma(self):
        """
        `lo` 与 `hi` 是**两行独立的代码**，只钉上界抓不到下界那一行
        （上一轮复测里 L218 就是这么活下来的）。这里放一个**低端**离群值。
        """
        x = np.array([np.concatenate([[-1000.0], np.arange(10, dtype=float)])])
        k = 1.5
        mu = float(np.nanmean(x))
        sd = float(np.nanstd(x, ddof=1))
        assert x.min() < mu - k * sd, "构造的低端离群值没有越界，测不到 clip"
        got = Executor(validate=False, winsorize_k=k)._postprocess(x.copy())
        np.testing.assert_allclose(got.min(), mu - k * sd, rtol=1e-12,
                                   err_msg="缩尾下界不等于 mu - k*sigma")

    def test_winsorize_bounds_widen_with_dispersion(self):
        """
        同一个 k、两行离散度差 100 倍：`*` 让界随之放大，`/` 让界随之收缩。
        这条把方向性也钉住，而不只是数值。
        """
        ex = Executor(validate=False, winsorize_k=1.0)
        narrow = np.array([[0.0, 1.0, 2.0, 3.0]])
        wide = narrow * 100.0
        wn = ex._postprocess(narrow.copy())
        ww = ex._postprocess(wide.copy())
        width_n = float(wn.max() - wn.min())
        width_w = float(ww.max() - ww.min())
        assert width_w > width_n * 50, (
            f"离散度放大 100 倍，缩尾区间宽度只从 {width_n} 变到 {width_w} —— "
            f"上下界与 sigma 的关系疑似被改成了除法")

    def test_winsorize_is_off_by_default(self):
        x = self._signal_with_an_outlier()
        np.testing.assert_array_equal(
            Executor(validate=False)._postprocess(x.copy()), x,
            err_msg="没有设 winsorize_k 却做了缩尾")

    def test_neutralize_removes_the_cross_sectional_mean(self):
        """
        `x = x - mu` 写成 `x + mu` 会把信号整体**平移到两倍均值**上，
        组合从多空中性变成单边净多/净空 —— 敞口直接翻倍。
        """
        ex = Executor(validate=False, neutralize=True)
        x = np.array([[1.0, 2.0, 3.0, 6.0], [0.0, 0.0, 0.0, 4.0]])
        got = ex._postprocess(x.copy())
        np.testing.assert_allclose(np.nanmean(got, axis=1), [0.0, 0.0], atol=1e-12,
                                   err_msg="中性化之后每行均值不为 0")
        np.testing.assert_allclose(got[0], [-2.0, -1.0, 0.0, 3.0], rtol=1e-12)

    def test_neutralize_keeps_rows_independent(self):
        """`keepdims=True` 改成 False 会让各行的均值错位相减。"""
        ex = Executor(validate=False, neutralize=True)
        x = np.array([[5.0, 5.0, 5.0, 5.0], [0.0, 10.0, 20.0, 30.0]])
        got = ex._postprocess(x.copy())
        np.testing.assert_allclose(got[0], 0.0, atol=1e-12,
                                   err_msg="全同值行被另一行的均值污染了")

    def test_winsorize_uses_sample_std_per_row(self):
        """`ddof=1` 与 `keepdims=True` 同时被钉：两行离散度不同，界必须不同。"""
        ex = Executor(validate=False, winsorize_k=1.0)
        x = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 100.0, 200.0, 300.0]])
        got = ex._postprocess(x.copy())
        assert got[1].max() > got[0].max() * 10, (
            "两行用了同一套上下界 —— keepdims 或逐行统计被改坏了")


# ===========================================================================
# D. 端到端：validate 开关与标量广播
# ===========================================================================

class TestRunEndToEnd:

    def test_validate_defaults_to_on(self):
        """
        `validate: bool = True` 改成 False 会让**校验默认关闭**：
        窗口越界、深度超限、前视算子全部放行，validator 那一整套门形同虚设。
        """
        ex = Executor()
        assert ex.validate is True, "Executor 的校验默认不再是开启"
        assert ex._validator is not None, "默认构造没有装上 validator"
        with pytest.raises(Exception):
            ex.run_expr("ts_mean(close, 99999)", _dataset())

    def test_validate_can_be_turned_off_explicitly(self):
        ex = Executor(validate=False)
        assert ex._validator is None
        out = ex.run_expr("ts_mean(close, 3)", _dataset())
        assert out.shape == (T, N)

    def test_scalar_expression_broadcasts_to_the_panel_shape(self):
        out = Executor(validate=False).run_expr("1", _dataset())
        assert out.shape == (T, N), f"标量没有广播成面板，得到 {out.shape}"
        assert (out.to_numpy() == 1.0).all()

    def test_result_keeps_the_dataset_index_and_columns(self):
        out = Executor(validate=False).run_expr("rank(close)", _dataset())
        pd.testing.assert_index_equal(out.index, IDX)
        assert list(out.columns) == COLS

    def test_cache_is_per_run_not_shared_across_datasets(self):
        """缓存必须每次 run 重建，否则换了数据集还会读到上一次的结果。"""
        ex = Executor(validate=False)
        a = ex.run_expr("close", _dataset())
        other = _dataset()
        other["close"] = other["close"] + 1000.0
        b = ex.run_expr("close", other)
        assert not np.allclose(a.to_numpy(), b.to_numpy()), (
            "换了数据集结果不变 —— 缓存跨 run 泄漏了")

    def test_get_cache_keys_lists_memoized_subexpressions(self):
        keys = Executor(validate=False).get_cache_keys(
            Executor(validate=False)._parser.parse("ts_mean(close,3)+ts_mean(close,3)"),
            _dataset())
        assert any("ts_mean" in k for k in keys), f"子表达式没有进缓存：{keys}"
        assert len(keys) == len(set(keys)), "缓存键有重复"
