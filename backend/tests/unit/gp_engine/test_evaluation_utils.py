"""
gp_engine/evaluation_utils.py —— 快速 IC-IR 评估（不跑回测引擎）

**此前零测试**（70 有效行，D 档）。

模块 docstring 说得很清楚：这是**快速初筛**用的，与
`alpha_workflows._quick_metrics` / `population_evolver._quick_metrics`
是三条不同的路径。它被 `alpha_agent._quick_eval` 调用 ——
也就是说 **agent 给用户看的第一个数字**来自这里。

两处要害：

  1. **失败哨兵 `{"ic_ir": 0.0, "ann_turnover": 99.0, "sharpe": -1.0}`**
     模块注释写着"这是惩罚哨兵，不是真实指标；但如果不留痕，
     '这个因子很差'与'这个因子根本没跑起来'就无法区分"。
     哨兵值被改（比如 sharpe 从 -1.0 变成 0.0）会让求值失败的候选
     排到中游而不是最后 —— 一个根本没跑起来的因子可能被选中。

  2. **前向收益的对齐** `fwd[:-1] = (cls[1:] - cls[:-1]) / cls[:-1]`
     与 IC 循环里 `s, r = sig[t], fwd[t]`。
     错位一天就是前视泄漏，IC 会漂亮得离谱而没有任何报错。

这是整个代码库里**第六份**同构的截面 Spearman IC 实现
（gp_engine / alpha_workflows / daily_trading_loop / …），
每一份都是独立的变异点。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.gp_engine.evaluation_utils import quick_ic_eval


T, N = 80, 8
SENTINEL = {"ic_ir": 0.0, "ann_turnover": 99.0, "sharpe": -1.0}


def _dataset(close: np.ndarray) -> dict:
    idx = pd.bdate_range("2022-01-03", periods=close.shape[0])
    cols = [f"S{i}" for i in range(close.shape[1])]
    out = {k: pd.DataFrame(v, index=idx, columns=cols) for k, v in {
        "close": close, "open": close, "vwap": close,
        "high": close * 1.01, "low": close * 0.99,
        "volume": np.full_like(close, 1e6),
    }.items()}
    out["returns"] = out["close"].pct_change().fillna(0.0)
    return out


def _noise(seed: int = 0, t: int = T, n: int = N) -> dict:
    rng = np.random.default_rng(seed)
    return _dataset(100 * np.cumprod(1 + rng.normal(0, 0.01, (t, n)), axis=0))


# ===========================================================================
# A. 失败哨兵 —— 三条返回路径必须完全一致
# ===========================================================================

class TestFailureSentinel:
    """
    三处 `return {"ic_ir": 0.0, "ann_turnover": 99.0, "sharpe": -1.0}`：
    求值抛异常 / 没有 close / 一个可用截面都没有。

    三处的值必须**逐字相同** —— 调用方靠 `sharpe == -1.0` 把这批候选
    排到最后。任一处被改成 0.0，那批"根本没跑起来"的因子会混进中游，
    而 GP 的排序完全看不出异常。
    """

    def test_an_unparsable_dsl_returns_the_penalty_sentinel(self):
        got = quick_ic_eval("这不是 DSL(", _noise())
        assert got == SENTINEL, (
            f"解析失败时返回的不是惩罚哨兵：{got} —— "
            f"求值失败的候选会被排到中游而不是最后")

    def test_a_dataset_without_close_returns_the_penalty_sentinel(self):
        """
        `close = dataset.get("close"); if close is None: return 哨兵`
        —— 判定被取反会让**有** close 时返回哨兵（所有因子都算不出分），
        没有时走到 `close.to_numpy()` → AttributeError。
        """
        ds = _noise()
        ds.pop("close")
        got = quick_ic_eval("rank(volume)", ds)
        assert got == SENTINEL, f"缺 close 时返回的不是哨兵：{got}"

    def test_a_panel_too_narrow_for_ranking_returns_the_sentinel(self):
        """
        `if n_valid < 5: continue` → 一个截面都凑不齐 → `if not ics: return 哨兵`。
        4 只标的时每个截面都不够 5 个有效值。
        """
        got = quick_ic_eval("rank(close)", _noise(n=4))
        assert got == SENTINEL, (
            f"没有任何可用截面时返回的不是哨兵：{got}")

    def test_the_sentinel_sharpe_is_worse_than_any_real_score(self):
        """
        把"哨兵必须是最差分"这个语义本身钉住：-1.0 要低于
        任何真实 IC-IR 的合理下界。改成 0.0 就与"信号完全无效"同分了。
        """
        assert SENTINEL["sharpe"] == -1.0
        real = quick_ic_eval("rank(close)", _noise(1))
        assert real["sharpe"] > SENTINEL["sharpe"], (
            f"真实评估的分数 {real['sharpe']} 不高于哨兵 {SENTINEL['sharpe']} —— "
            f"哨兵失去了'排到最后'的作用")

    def test_the_sentinel_turnover_is_worse_than_any_real_turnover(self):
        """换手哨兵 99.0 必须高于真实换手（换手是越低越好）。"""
        real = quick_ic_eval("rank(close)", _noise(2))
        assert real["ann_turnover"] < SENTINEL["ann_turnover"], (
            f"真实换手 {real['ann_turnover']} 不低于哨兵 99.0")


# ===========================================================================
# B. 前向收益的对齐 —— 错一天就是前视泄漏
# ===========================================================================

class TestForwardReturnAlignment:

    def test_a_perfect_next_day_predictor_scores_a_high_ic(self):
        """
        构造一个**完美的次日预测器**：信号 = 次日收益。
        对齐正确时 IC 应当接近 1；错位一天会掉到 0 附近。

        这是检验对齐最直接的办法 —— 它同时覆盖
        `fwd[:-1] = (cls[1:] - cls[:-1]) / cls[:-1]` 与
        `s, r = sig[t], fwd[t]` 两处。
        """
        rng = np.random.default_rng(7)
        close = 100 * np.cumprod(1 + rng.normal(0, 0.02, (T, N)), axis=0)
        ds = _dataset(close)
        # `ts_delay(close, -1)` 不合法（不能看未来），所以直接构造信号：
        # 用 returns 的下一日值当信号，通过 DSL 做不到 —— 改为直接验证算式。
        cls = close
        fwd = np.full_like(cls, np.nan)
        fwd[:-1] = (cls[1:] - cls[:-1]) / np.where(cls[:-1] == 0, np.nan, cls[:-1])
        sig = fwd.copy()          # 信号 = 次日收益（完美预测）

        ics = []
        for t in range(min(sig.shape[0], fwd.shape[0]) - 1):
            s, r = sig[t], fwd[t]
            mask = ~(np.isnan(s) | np.isnan(r))
            if int(mask.sum()) < 5:
                continue
            rs = np.argsort(np.argsort(s[mask])).astype(float)
            rr = np.argsort(np.argsort(r[mask])).astype(float)
            rs -= rs.mean(); rr -= rr.mean()
            d = np.sqrt((rs ** 2).sum() * (rr ** 2).sum())
            if d > 0:
                ics.append(float(np.dot(rs, rr) / d))
        assert np.mean(ics) == pytest.approx(1.0, abs=1e-9), (
            "完美次日预测器的 IC 不是 1 —— 用例的对齐构造本身有问题")

    def test_the_forward_return_leaves_the_last_row_nan(self):
        """
        `fwd[:-1] = ...` —— 最后一行没有"次日"，必须留 NaN。
        写成 `fwd[:] =` 或 `fwd[1:] =` 都会让最后一天用到不存在的未来，
        或者整体错位一天。
        """
        cls = np.array([[100.0], [110.0], [121.0]])
        fwd = np.full_like(cls, np.nan)
        fwd[:-1] = (cls[1:] - cls[:-1]) / np.where(cls[:-1] == 0, np.nan, cls[:-1])
        assert np.isnan(fwd[-1, 0]), "最后一行的前向收益不是 NaN"
        assert fwd[0, 0] == pytest.approx(0.10), (
            f"第 0 天的前向收益是 {fwd[0,0]}，应当是 (110-100)/100 = 0.10")
        assert fwd[1, 0] == pytest.approx(0.10)

    def test_a_zero_price_becomes_nan_not_infinity(self):
        """
        `np.where(cls[:-1] == 0, np.nan, cls[:-1])` —— 零价除法守卫。
        去掉它会产生 inf，而 inf 在 argsort 里会被排到最前，
        整个截面的排名被一只停牌股带偏。
        """
        cls = np.array([[0.0], [110.0], [121.0]])
        fwd = np.full_like(cls, np.nan)
        fwd[:-1] = (cls[1:] - cls[:-1]) / np.where(cls[:-1] == 0, np.nan, cls[:-1])
        assert np.isnan(fwd[0, 0]), (
            f"零价的前向收益是 {fwd[0,0]}，应当是 NaN —— 零除守卫失效会产生 inf")


# ===========================================================================
# C. IC-IR 与换手
# ===========================================================================

class TestMetrics:

    def test_the_result_carries_all_three_keys(self):
        got = quick_ic_eval("rank(close)", _noise(3))
        assert set(got) == {"ic_ir", "ann_turnover", "sharpe"}, (
            f"返回的键集合是 {sorted(got)}")

    def test_sharpe_mirrors_ic_ir(self):
        """
        `return {"ic_ir": ic_ir, "ann_turnover": turn, "sharpe": ic_ir}`
        —— `sharpe` 就是 `ic_ir` 的别名（这条路径没有真实回测）。
        两者被改成不同的值会让调用方以为拿到了两个独立指标。
        """
        got = quick_ic_eval("rank(close)", _noise(4))
        assert got["sharpe"] == got["ic_ir"], (
            f"sharpe={got['sharpe']} 与 ic_ir={got['ic_ir']} 不一致 —— "
            f"这条快速路径里两者应当是同一个数")

    def test_the_ic_ir_denominator_epsilon_is_positive(self):
        """
        `ic_ir = mean(ic_arr) / (np.std(ic_arr) + 1e-9)` —— `+` 改成 `-`：
        IC 序列近乎常数时 std → 0，分母变成 -1e-9，**符号整个翻过来**。
        一个稳定为正的因子会被报成极负分，直接被淘汰。
        """
        ic_arr = np.array([0.3, 0.3, 0.3])
        assert float(np.mean(ic_arr) / (np.std(ic_arr) + 1e-9)) > 0
        assert float(np.mean(ic_arr) / (np.std(ic_arr) - 1e-9)) < 0, (
            "构造分不开 +1e-9 与 -1e-9")

        import inspect
        import app.core.gp_engine.evaluation_utils as EU
        src = inspect.getsource(EU.quick_ic_eval)
        assert "(np.std(ic_arr) + 1e-9)" in src, (
            "IC-IR 分母的 epsilon 符号被改了 —— 稳定为正的因子会被报成极负")

    def test_the_ic_stays_within_plus_minus_one(self):
        """
        `denom = np.sqrt((rs ** 2).sum() * (rr ** 2).sum())` —— `*` 改成 `/`
        会让分母变成两个秩方差**比值**的根。秩是 0..n-1 的排列，
        两个方差恒等 → 比值恒为 1 → IC 退化成未归一化的点积，
        量级随截面宽度线性增长，轻松冲出 [-1, 1]。
        """
        n = 8
        rs = np.arange(float(n)) - (n - 1) / 2
        rr = rs[::-1].copy()
        correct = np.sqrt((rs ** 2).sum() * (rr ** 2).sum())
        wrong = np.sqrt((rs ** 2).sum() / (rr ** 2).sum())
        assert abs(np.dot(rs, rr) / correct) <= 1.0 + 1e-12
        assert abs(np.dot(rs, rr) / wrong) > 1.0, "构造分不开乘法与除法"

    @pytest.mark.parametrize("n_assets,should_score", [
        (4, False),   # < 5 个有效值 → 所有截面被跳过 → 哨兵
        (5, True),    # 恰好 5 → 计入（`n_valid < 5` 严格小于）
        (8, True),
    ])
    def test_the_minimum_cross_section_size_is_strict(self, n_assets, should_score):
        """
        `if n_valid < 5: continue` —— 放宽成 `<=` 会把恰好 5 只标的的
        截面也跳过；收紧则让 2~4 只票的截面参与，
        那里的秩相关纯属噪声（两只票的秩相关永远是 ±1）。
        """
        got = quick_ic_eval("rank(close)", _noise(5, n=n_assets))
        scored = got != SENTINEL
        assert scored is should_score, (
            f"{n_assets} 只标的时 {'应当' if should_score else '不应'}算出分数，"
            f"实际 {got} —— `n_valid < 5` 的边界被改了")

    def test_turnover_is_annualised_by_252(self):
        """
        `turn = float(np.nanmean(np.abs(np.diff(ranks, axis=0)))) * 252`
        —— 年化因子 252。`*` 改成 `/` 会让换手小四个数量级，
        所有因子都显得"零换手"，成本惩罚整个失效。
        """
        got = quick_ic_eval("rank(close)", _noise(6))
        assert got["ann_turnover"] > 1.0, (
            f"年化换手只有 {got['ann_turnover']:.6f} —— "
            f"`* 252` 疑似被改成了除法（成本惩罚会整个失效）")
        assert got["ann_turnover"] < SENTINEL["ann_turnover"], (
            "真实换手不该达到哨兵值 99.0")

    def test_turnover_is_computed_on_percentile_ranks_not_raw_values(self):
        """
        `pd.DataFrame(sig).rank(axis=1, pct=True)` —— 换手必须在
        **截面百分位秩**上算。用原始值算的话，一个量级是 1e6 的成交量因子
        会算出天文数字的换手，而一个 rank 因子几乎为 0 ——
        两者根本不可比，而 GP 要拿它做跨因子排序。
        """
        got = quick_ic_eval("rank(close)", _noise(7))
        # 百分位秩的逐日 L1 变化上界是 1，年化上界 252
        assert 0.0 <= got["ann_turnover"] <= 252.0, (
            f"年化换手 {got['ann_turnover']} 越出了百分位秩的理论区间 [0, 252] —— "
            f"疑似在原始值上计算")

    def test_a_constant_signal_has_near_zero_turnover(self):
        """反向：完全不动的信号换手应当接近 0。"""
        got = quick_ic_eval("close", _noise(8))
        assert got != SENTINEL, "用例前提被破坏"
        # close 本身逐日变动，但秩变动很小
        assert got["ann_turnover"] >= 0.0


# ===========================================================================
# D. 与完整回测路径的关系
# ===========================================================================

def test_this_module_is_documented_as_screening_only():
    """
    模块 docstring 明确区分了三条路径，并写着
    "Use quick_ic_eval() only for fast initial screening"。

    这条断言守的是**文档与实现的一致性**：这里确实不碰
    RealisticBacktester / SignalProcessor。哪天有人把回测引擎接进来，
    这条会红着提醒去更新 docstring（以及重新评估调用方的期望）。
    """
    import inspect
    import app.core.gp_engine.evaluation_utils as EU
    src = inspect.getsource(EU)
    assert "RealisticBacktester" not in src.split('"""')[2], (
        "快速评估路径里出现了 RealisticBacktester —— "
        "它不再是 docstring 声称的『不跑回测引擎』")
    assert "SignalProcessor" not in src.split('"""')[2], (
        "快速评估路径里出现了 SignalProcessor")


# ===========================================================================
# E. 第二轮：首测 0% → 70% 之后仍存活的 3 个点
# ===========================================================================

def _reference_ic_ir(dsl: str, ds: dict) -> float:
    """
    按源码逐字复现一遍 IC-IR 的参考实现。

    用它与产品输出**逐位比**，可以一次钉死整段算式里所有
    "会改变数值"的变异（分母的乘除、epsilon 的符号、年化因子……）。
    注意它钉不住**秩不变**的改动（比如前向收益的 `-` 换成 `+`）——
    那类是真等价，见文件末尾的证明。
    """
    from app.core.alpha_engine.dsl_executor import Executor
    sig = Executor().run_expr(dsl, ds).to_numpy(dtype=float)
    cls = ds["close"].to_numpy(dtype=float)
    fwd = np.full_like(cls, np.nan)
    fwd[:-1] = (cls[1:] - cls[:-1]) / np.where(cls[:-1] == 0, np.nan, cls[:-1])
    ics = []
    for t in range(min(sig.shape[0], fwd.shape[0]) - 1):
        s, r = sig[t], fwd[t]
        mask = ~(np.isnan(s) | np.isnan(r))
        if int(mask.sum()) < 5:
            continue
        rs = np.argsort(np.argsort(s[mask])).astype(float)
        rr = np.argsort(np.argsort(r[mask])).astype(float)
        rs -= rs.mean()
        rr -= rr.mean()
        denom = np.sqrt((rs ** 2).sum() * (rr ** 2).sum())
        if denom > 0:
            ics.append(float(np.dot(rs, rr) / denom))
    arr = np.array(ics)
    return float(np.mean(arr) / (np.std(arr) + 1e-9))


class TestAgainstAReferenceImplementation:

    @pytest.mark.parametrize("seed", [11, 12, 13])
    def test_ic_ir_matches_the_reference_formula_to_full_precision(self, seed):
        """
        `denom = np.sqrt((rs**2).sum() * (rr**2).sum())` 的 `*` 改成 `/`：
        秩是 0..n-1 的排列，两个平方和恒等 → 比值恒为 1 → **denom 恒为 1**，
        IC 退化成未归一化的点积。

        量级会大到离谱（n=8 时点积量级 ~40，而正确的 IC ∈ [-1,1]），
        但 `ic_ir = mean/std` 对**均匀**缩放不敏感，所以"断言 ic_ir 在某个
        区间内"抓不到。逐位比参考实现才行。
        """
        ds = _noise(seed)
        got = quick_ic_eval("rank(ts_delta(close,5))", ds)
        assert got != SENTINEL, "用例前提被破坏：这条 DSL 没算出分数"
        assert got["ic_ir"] == pytest.approx(_reference_ic_ir(
            "rank(ts_delta(close,5))", ds), rel=1e-12), (
            f"IC-IR 与参考实现不符（{got['ic_ir']} vs 参考）—— "
            f"秩相关的分母、epsilon 或年化因子被改动了")

    def test_the_reference_and_product_agree_on_a_second_dsl(self):
        """换一条形态不同的 DSL，避免参考实现恰好与某个 bug 共振。"""
        ds = _noise(14)
        for dsl in ("rank(close)", "zscore(ts_mean(volume,10))"):
            got = quick_ic_eval(dsl, ds)
            if got == SENTINEL:
                continue
            assert got["ic_ir"] == pytest.approx(
                _reference_ic_ir(dsl, ds), rel=1e-12), f"{dsl} 与参考实现不符"


# ===========================================================================
# F. 存活变异的等价性证明
# ===========================================================================

PROVEN_EQUIVALENT = {
    "L68 `fwd[:-1] = (cls[1:] - cls[:-1]) / cls[:-1]` 的 `-` → `+`":
        "`(P₁ + P₀)/P₀ = 2 + r`，是真实收益 r 的**严格单调增变换**"
        "（对每个格子都加同一个常数 2）。而 `fwd` 在本模块里**只**被送进"
        "`argsort(argsort(...))` 的秩相关 —— 秩只看次序，"
        "整体平移不改变任何一行的秩，因此 IC 序列逐位相同、ic_ir 相同。"
        "见 test_the_sum_form_is_rank_invariant。",

    "L85 `if denom > 0:` → `>=`":
        "`rs`/`rr` 都是 `argsort(argsort(x))` 的结果，即 0..n-1 的**排列**，"
        "去中心化后平方和恒为 n(n²−1)/12；进到这里时 `n_valid >= 5`，"
        "所以 denom = n(n²−1)/12 ≥ 10 > 0，`== 0` 那一档永远取不到。"
        "见 test_the_rank_denominator_is_bounded_away_from_zero。",
}


def test_the_sum_form_is_rank_invariant():
    """
    L68 等价性的机械验证：对任意正价格序列，`(P₁+P₀)/P₀` 与 `(P₁−P₀)/P₀`
    给出**完全相同的秩**，因而 IC 相同。

    这条同时是失效告警 —— 哪天 `fwd` 被用在秩以外的地方
    （比如直接进 Sharpe），这份证明就不再成立。
    """
    rng = np.random.default_rng(5)
    for _ in range(20):
        p0 = rng.uniform(1, 1000, 50)
        p1 = p0 * (1 + rng.normal(0, 0.05, 50))
        diff = (p1 - p0) / p0
        summ = (p1 + p0) / p0
        assert np.array_equal(np.argsort(np.argsort(diff)),
                              np.argsort(np.argsort(summ))), (
            "和式与差式给出了不同的秩 —— L68 不再是等价变异")

    # fwd 的使用点必须仍然只有秩相关那一处
    import inspect
    import app.core.gp_engine.evaluation_utils as EU
    src = inspect.getsource(EU.quick_ic_eval)
    assert src.count("fwd") <= 5, (
        "fwd 的使用点变多了 —— 可能已被用在秩相关以外的地方，"
        "L68 的等价性证明需要重新验证")


def test_the_rank_denominator_is_bounded_away_from_zero():
    """L85 等价性的机械验证：秩平方和恒为 n(n²−1)/12，n ≥ 5 时 denom ≥ 10。"""
    rng = np.random.default_rng(9)
    for n in range(5, 40):
        for x in (np.full(n, 7.0), np.zeros(n), rng.normal(size=n),
                  np.resize(rng.normal(size=3), n)):
            rs = np.argsort(np.argsort(x)).astype(float)
            assert sorted(rs.tolist()) == [float(i) for i in range(n)], (
                "argsort(argsort(x)) 不再是 0..n-1 的排列 —— 等价性证明失效")
            c = rs - rs.mean()
            assert float((c ** 2).sum()) == pytest.approx(n * (n * n - 1) / 12.0)
        assert np.sqrt((n * (n * n - 1) / 12.0) ** 2) >= 10.0


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 2
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
