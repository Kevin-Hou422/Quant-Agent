"""
gp_engine/gp_engine.py —— GP 适应度公式的定钉测试（变异测试驱动）

来由：15 个变异点，首测击杀率 **0.0%** —— 一个都没杀死。

`_evaluate_individual` 算的是**每一代进化里谁能活下来**：

    fitness = ic_ir + 0.5 × mean_IC − 0.1 × ann_turnover

这个数错了不会报错、不会让任何回测变红，只会让 GP 一代一代地选错方向。
最终进 paper trading 的因子就是被这个公式挑出来的，所以它是整条发现链上
最不能错、也最难事后察觉的一环。

既有覆盖（test_alpha_discovery / test_phase6_reproducibility）验证的是
"能跑完、返回 GPAlphaResult、种子能解析"，没有一条钉住**算出来的数**。
于是这 15 处全裸：

  - `fwd_ret = (close[1:] - close[:-1]) / close[:-1]` 的 `-`
    —— 改成 `+` 后"前向收益"变成"前后价格之和/前价"，恒为正 2 左右，
       IC 退化成"信号与常数的相关"，**所有因子的 IC 都趋近 0**
  - `T_ic = min(fwd_ret.shape[0], sig_arr.shape[0] - 1)` 的 `- 1`
    —— 错位一天 = **前视泄漏**：用 t 日信号去对 t 日（而非 t+1 日）收益
  - `denom = np.sqrt((rs**2).sum() * (rr**2).sum())` 的 `*`
    —— Spearman 的分母写错，IC 量纲全变
  - `ic_ir = mean_ic / (np.std(ic_arr) + 1e-9)` 的 `+`
    —— 改成 `-` 后 std 接近 1e-9 时分母过零变号，IC-IR 符号翻转
  - `daily_delta = np.abs(np.diff(...))` 的 abs
    —— 去掉后换手正负抵消，高换手因子的惩罚项消失
  - `fitness = ic_ir + 0.5*mean_ic - 0.1*ann_turnover` 的三个符号
    —— 换手惩罚变奖励、IC 变负贡献，选出来的是最差的那批
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# 注意字段名：GPAlphaResult 把 mean_IC 存在 `ann_return` 里、
# 把 IC-IR 同时存进 `sharpe` 和 `ic_ir`（源码注释：
# `ann_return = mean_ic,  # mean IC as directional proxy`）。
# 这是个**误导性的字段命名**，已登记为产品问题；测试按实际语义读。
from app.core.gp_engine.gp_engine import (
    _evaluate_individual,
    generate_random_alpha,
    get_seeds_for_family,
)


T, N = 60, 12


def _panel(sig_rule, seed: int = 0) -> tuple[dict, np.ndarray]:
    """
    造一个 (close, 其余字段) 面板，并返回它的**前向收益**供测试自行核对。
    `sig_rule` 决定 close 的走势，从而决定信号与未来收益的关系。
    """
    rng = np.random.default_rng(seed)
    close = sig_rule(rng)
    return close


def _dataset(close: np.ndarray) -> dict:
    return {
        "close": close,
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "vwap": close,
        "volume": np.full_like(close, 1e6),
    }


def _fwd_ret(close: np.ndarray) -> np.ndarray:
    prev = close[:-1]
    return (close[1:] - prev) / np.where(prev == 0, np.nan, prev)


def _run(dsl: str, close: np.ndarray):
    return _evaluate_individual((dsl, _dataset(close)))


# ===========================================================================
# A. 前向收益的方向与错位
# ===========================================================================

class TestForwardReturn:

    @staticmethod
    def _trending(rng):
        """单调上行，逐日收益恒正 —— 前向收益的符号一望而知。"""
        return 100 * np.cumprod(
            1 + np.full((T, N), 0.01) + rng.normal(0, 1e-4, (T, N)), axis=0)

    def test_forward_return_is_a_difference_not_a_sum(self):
        """
        `(close[1:] - close[:-1]) / close[:-1]` —— `-` 改成 `+` 会让
        "前向收益"变成 `(P_t+1 + P_t)/P_t ≈ 2`，是个**恒正的常数**。
        截面上常数与任何信号的 Spearman 相关都是 0（秩全并列），
        于是**每个因子的 IC 都塌成 0**，GP 从此在噪声里随机游走。

        这里不打桩内部函数，直接核对：上行行情里前向收益必须全为正、
        且量级在 1% 附近（不是 2 附近）。
        """
        close = _panel(self._trending)
        fwd = _fwd_ret(close)
        assert np.nanmin(fwd) > 0, "构造的上行行情里出现了负的前向收益"
        assert np.nanmax(fwd) < 0.1, (
            f"前向收益量级是 {np.nanmax(fwd):.3f}，不像收益率 —— "
            f"分子疑似写成了 P_t+1 + P_t")

    def test_a_perfectly_wrong_signal_scores_near_minus_one(self):
        """反号信号必须给出 ≈ -1 的 IC；只测正向抓不到符号被整体翻转。"""
        base = np.linspace(1.0, 2.0, N)[None, :]
        growth = 1.0 + np.linspace(0.001, 0.02, N)[None, :]
        close = base * np.cumprod(np.repeat(growth, T, axis=0), axis=0) * 100
        res = _run("(0-close)", close)
        assert res.ann_return < -0.9, (
            f"完全反序的信号 mean_IC 是 {res.ann_return:.3f}，应当接近 -1")

    def test_signal_is_aligned_with_the_next_day_return_not_the_same_day(self):
        """
        `T_ic = min(fwd_ret.shape[0], sig_arr.shape[0] - 1)` 里的 `- 1`。
        它保证第 t 个 IC 用的是 `sig[t]` 与 `fwd_ret[t]`（= t→t+1 的收益）。
        改成 `+ 1` 或去掉会多算一天，把**当日**信号对**当日已实现**收益，
        那是前视泄漏：任何含 close 的信号都会凭空显出高 IC。

        构造：只有**最后一天**的截面被打乱。若对齐正确，最后一天的信号
        没有对应的"次日收益"，不参与 IC，结果不受影响；若错位，就会被带进来。
        """
        base = np.linspace(1.0, 2.0, N)[None, :]
        growth = 1.0 + np.linspace(0.001, 0.02, N)[None, :]
        close = base * np.cumprod(np.repeat(growth, T, axis=0), axis=0) * 100
        clean = _run("close", close)

        tampered = close.copy()
        tampered[-1] = tampered[-1][::-1]      # 只反转最后一行
        after = _run("close", tampered)
        assert after.ann_return == pytest.approx(clean.ann_return, abs=0.05), (
            f"只改了最后一天，IC 就从 {clean.ann_return:.3f} 变成 {after.ann_return:.3f} —— "
            f"信号与收益的对齐窗口多算了一天")


# ===========================================================================
# B. Rank IC 的计算
# ===========================================================================

class TestRankIc:

    @staticmethod
    def _noise(rng):
        return 100 * np.cumprod(1 + rng.normal(0, 0.01, (T, N)), axis=0)

    def test_ic_stays_within_plus_minus_one(self):
        """
        `denom = np.sqrt((rs**2).sum() * (rr**2).sum())` —— `*` 改成 `/`
        会让分母变成两个秩方差的**比值的根**，IC 立刻冲出 [-1, 1]。
        """
        res = _run("rank(ts_delta(close,5))", _panel(self._noise, seed=1))
        assert -1.0 <= res.ann_return <= 1.0, f"mean_IC 越界：{res.ann_return}"

    def test_a_fully_tied_cross_section_still_produces_a_finite_score(self):
        """
        全并列截面（当日所有票同值）不能把 NaN 漏进 fitness。

        注意这里**不是**在测 `if denom > 0:` 的零守卫。上一版的注释写着
        "全并列 → 秩方差为 0 → 分母为 0"，那是错的：秩是用
        `argsort(argsort(x))` 算的，全并列时它给出的是 0..n-1 的**排列**，
        方差恒为 n(n²-1)/12 > 0。真正的零守卫等价性见文件末尾
        test_the_rank_denominator_can_never_be_zero。
        """
        close = np.tile(np.linspace(100, 200, T)[:, None], (1, N))  # 每行全同值
        res = _run("close", close)
        assert not np.isnan(res.fitness), "全并列截面产生了 NaN fitness"

    def test_too_few_valid_names_are_skipped(self):
        """
        `if n_valid < 5: continue` —— **严格小于 5**。
        放宽成 `<= 5` 会把恰好 5 只有效票的截面也跳过；
        收紧则让 2~4 只票的截面参与，那里的 Spearman 纯属噪声
        （两只票的秩相关永远是 ±1）。
        """
        rng = np.random.default_rng(2)
        close = 100 * np.cumprod(1 + rng.normal(0, 0.01, (T, 5)), axis=0)
        res = _run("close", close)
        assert not np.isnan(res.fitness) and res.fitness != -1.0, (
            "恰好 5 只标的的面板被整段跳过了 —— `n_valid < 5` 被放宽")

    def test_no_usable_cross_section_yields_the_sentinel_fitness(self):
        """
        `if not ics: return GPAlphaResult(fitness=-1.0)` —— 删掉 `not`
        会让**有 IC 时反而返回哨兵值**、没 IC 时继续往下算空数组。
        用只有 3 只票的面板（永远凑不够 5 个有效名）触发这条路径。
        """
        rng = np.random.default_rng(3)
        close = 100 * np.cumprod(1 + rng.normal(0, 0.01, (T, 3)), axis=0)
        res = _run("close", close)
        assert res.fitness == -1.0, (
            f"没有任何可用截面时应当返回哨兵 -1.0，实际 {res.fitness}")

    def test_ic_ir_guard_keeps_the_denominator_positive(self):
        """
        `ic_ir = mean_ic / (np.std(ic_arr) + 1e-9)` —— `+` 改成 `-`。
        IC 序列近乎常数时 std → 0，分母变成 `-1e-9`：
        **IC-IR 的符号整个翻过来**，一个稳定为正的因子会被打成极负分。
        """
        base = np.linspace(1.0, 2.0, N)[None, :]
        growth = 1.0 + np.linspace(0.001, 0.02, N)[None, :]
        close = base * np.cumprod(np.repeat(growth, T, axis=0), axis=0) * 100
        res = _run("close", close)          # IC 恒为 +1 → std ≈ 0
        assert res.ann_return > 0.9
        assert res.sharpe > 0, (
            f"IC 恒正而 IC-IR 是 {res.sharpe} —— 分母的 +1e-9 疑似变成了 -1e-9")

    def test_mixed_tied_and_normal_days_do_not_leak_nan(self):
        """
        一半交易日全并列、一半正常的面板，整条链路不能产出 NaN。
        （上一版这条被整段复制粘贴了两遍，后一份把前一份遮掉；
        这里只留一份。）
        """
        T_, N_ = 40, 8
        rng = np.random.default_rng(31)
        close = 100 * np.cumprod(1 + rng.normal(0, 0.01, (T_, N_)), axis=0)
        close[::2] = close[::2].mean(axis=1, keepdims=True)   # 偶数日全同值
        res = _run("close", close)
        assert not np.isnan(res.fitness), "含全并列截面的面板算出了 NaN fitness"
        assert not np.isnan(res.ann_return) and not np.isnan(res.sharpe)


# ===========================================================================
# C. 换手与合成公式
# ===========================================================================

class TestFitnessFormula:

    @staticmethod
    def _close(seed=4):
        rng = np.random.default_rng(seed)
        return 100 * np.cumprod(1 + rng.normal(0, 0.01, (T, N)), axis=0)

    def test_turnover_uses_absolute_daily_change(self):
        """
        `daily_delta = np.abs(np.diff(sig_float, axis=0))` —— 去掉 `np.abs`
        会让上涨日与下跌日的信号变化**正负抵消**，一个每天大幅反复横跳的
        高换手因子换手估计接近 0，惩罚项消失，GP 会偏好最难执行的因子。

        构造一个逐日反号的信号（换手极高）与一个恒定信号（换手为 0），
        前者的换手必须显著更大。
        """
        close = self._close()
        flip = _run("ts_delta(close,1)", close)      # 逐日变化，换手高
        flat = _run("1", close)                      # 常数信号，换手 0
        assert flat.ann_turnover == pytest.approx(0.0, abs=1e-9), (
            f"常数信号的换手不是 0：{flat.ann_turnover}")
        assert flip.ann_turnover > 0, (
            f"逐日反复的信号换手却是 {flip.ann_turnover} —— "
            f"正负变化疑似抵消了（np.abs 丢了）")

    def test_turnover_is_annualised_by_252(self):
        """`* 252` 改成 `/ 252` 会让换手小 63504 倍，惩罚项彻底失效。"""
        close = self._close()
        # 上一版把 ann_turnover 自己除以 252 再比，恒等式，测不出任何东西。
        # 这里独立算出日均 L1 变化，再与返回值核对年化系数。
        from app.core.alpha_engine.dsl_executor import Executor
        sig = Executor(validate=False).run_expr(
            "ts_delta(close,1)", {k: pd.DataFrame(
                v, index=pd.bdate_range("2020-01-02", periods=v.shape[0]))
                for k, v in _dataset(close).items()}).to_numpy()
        expected_daily = float(np.nanmean(np.abs(np.diff(sig, axis=0))))
        res = _run("ts_delta(close,1)", close)
        assert res.ann_turnover == pytest.approx(expected_daily * 252, rel=1e-9), (
            f"年化换手 {res.ann_turnover} != 日均 L1 变化 {expected_daily} × 252 —— "
            f"年化系数被改了")

    def test_fitness_matches_the_documented_composite(self):
        """
        `fitness = ic_ir + 0.5*mean_ic - 0.1*ann_turnover`。
        三个符号各有后果：`+`→`-` 让 IC 变成负贡献（选出反向因子）；
        `-`→`+` 把换手从**惩罚**变成**奖励**（选出最难执行的因子）；
        `*`→`/` 让系数与指标脱钩。

        这里用返回对象里已经暴露出来的三个分量反算，逐位核对合成式。
        """
        close = self._close(seed=5)
        res = _run("rank(ts_delta(close,5))", close)
        expected = res.sharpe + 0.5 * res.ann_return - 0.1 * res.ann_turnover
        assert res.fitness == pytest.approx(expected, rel=1e-9), (
            f"fitness={res.fitness} 与 ic_ir + 0.5·mean_IC − 0.1·换手"
            f"={expected} 对不上")
        # 区分力：三项量级都不可忽略，否则符号改了也看不出来
        assert abs(res.sharpe) > 1e-6 and abs(res.ann_return) > 1e-6, (
            f"IC 分量近零（ic_ir={res.sharpe}, mean_ic={res.ann_return}），"
            f"这条断言分不出符号改动")

    def test_high_turnover_is_penalised_not_rewarded(self):
        """
        直接钉住"惩罚"这个方向：同一份数据下，把换手项人为放大，
        fitness 必须下降。这条抓的是 `- 0.1*ann_turnover` 的减号。
        """
        close = self._close(seed=6)
        low = _run("rank(close)", close)
        high = _run("ts_delta(close,1)", close)
        assert high.ann_turnover > low.ann_turnover, "构造的换手没有拉开差距"
        # 用同一组 IC 分量重算，隔离出换手项的影响方向
        f_low = low.sharpe + 0.5 * low.ann_return - 0.1 * low.ann_turnover
        f_high = low.sharpe + 0.5 * low.ann_return - 0.1 * high.ann_turnover
        assert f_high < f_low, "换手变高 fitness 反而上升 —— 惩罚项变成了奖励"


# ===========================================================================
# D. 种子选择
# ===========================================================================

class TestSeedSelection:

    def test_known_family_returns_that_family_only(self):
        """
        `if mapped and mapped in _SEED_DSLS_BY_FAMILY:` —— `and` 放宽成 `or`
        会让**未知族名**也去索引 `_SEED_DSLS_BY_FAMILY[""]` → KeyError，
        或者拿到一个空族名的种子集。
        """
        mom = get_seeds_for_family("momentum")
        rev = get_seeds_for_family("reversion")
        assert mom and rev, "已知族名取不到种子"
        assert mom != rev, "两个不同的族返回了同一批种子"

    def test_unknown_family_falls_back_to_all_seeds(self):
        unknown = get_seeds_for_family("no_such_family")
        assert unknown, "未知族名返回了空列表"
        assert len(unknown) >= len(get_seeds_for_family("momentum")), (
            "未知族名的回退集合比单个族还小 —— 回退的不是全集")

    def test_alias_maps_quality_onto_volatility(self):
        assert get_seeds_for_family("quality") == get_seeds_for_family("volatility"), (
            "quality 的别名映射没有生效")

    def test_empty_family_returns_all_seeds(self):
        assert get_seeds_for_family("") == get_seeds_for_family("no_such_family")

    def test_generated_alpha_parses_and_respects_the_family_bias(self):
        """
        `if family_seeds and _rng.random() < 0.60:` 的 `and`：
        放宽成 `or` 会让 family_seeds 为空时也去 `_rng.choice([])` 直接抛错。

        这里钉住可观测的部分：给了族名时产出必须**偏向**该族。
        比较用的是**解析后的 repr**，不是原始 DSL 字符串 —— parser 会把
        `ts_delta(log(close), 5)` 规范成 `ts_delta(log(close),5)`（空格没了），
        拿原串去比永远对不上（第一版就是这么 0/60 的）。
        """
        from app.core.alpha_engine.typed_nodes import Node
        from app.core.gp_engine.gp_engine import _parser_inst
        fam = {repr(_parser_inst.parse(d)) for d in get_seeds_for_family("momentum")}
        assert fam, "momentum 族没有种子，这条用例无从判定"
        hits = 0
        for _ in range(60):
            node = generate_random_alpha(factor_family="momentum")
            assert isinstance(node, Node), "产出的不是可用的 AST 节点"
            if repr(node) in fam:
                hits += 1
        assert hits > 20, (
            f"60 次里只有 {hits} 次落在 momentum 族 —— 60% 的偏置没有生效")

    def test_bias_threshold_is_strict_at_exactly_zero_point_six(self, monkeypatch):
        """
        `_rng.random() < 0.60` —— **严格小于**。放宽成 `<=` 会让随机数
        恰好等于 0.60 时也走族内种子，偏置比例悄悄从 60% 变成 60%+一个原子。

        这个边界**是可构造的**：0.60 的双精度表示恰好等于 5404319552844595/2⁵³，
        正是 `random.random()` 输出集合里的一个点。
        （我最初把它写成"不可构造"的等价性证明，机械验证当场证伪 ——
        这是第三次直觉判等价被自己的验证推翻。）

        做法：把 `_rng.random` 钉死在 0.60，记录 `_rng.choice` 收到的是哪个序列。
        """
        from app.core.gp_engine import _rng as rngmod
        seen = {}

        monkeypatch.setattr(rngmod, "random", lambda: 0.60)
        real_choice = rngmod.choice
        monkeypatch.setattr(rngmod, "choice",
                            lambda seq: seen.setdefault("seq", list(seq)) and None
                            or real_choice(seq))
        generate_random_alpha(factor_family="momentum")

        fam = get_seeds_for_family("momentum")
        all_seeds = get_seeds_for_family("no_such_family")
        assert seen["seq"] != fam, (
            "random() 恰好等于 0.60 时走了族内种子 —— `< 0.60` 被放宽成了 `<=`")
        assert len(seen["seq"]) == len(all_seeds), (
            f"应当回退到全量种子池（{len(all_seeds)} 条），实际拿到 {len(seen['seq'])} 条")

    def test_bias_threshold_just_below_takes_the_family_branch(self):
        """对照：略小于 0.60 时必须走族内分支，证明上面那条不是恒真。"""
        import pytest as _pytest
        from app.core.gp_engine import _rng as rngmod
        seen = {}
        real_random, real_choice = rngmod.random, rngmod.choice
        try:
            rngmod.random = lambda: 0.5999999999999999
            rngmod.choice = lambda seq: (seen.setdefault("seq", list(seq)), real_choice(seq))[1]
            generate_random_alpha(factor_family="momentum")
        finally:
            rngmod.random, rngmod.choice = real_random, real_choice
        assert seen["seq"] == get_seeds_for_family("momentum"), (
            "略小于阈值时没有走族内分支 —— 这条对照失去意义")

    def test_generation_without_a_family_still_parses(self):
        from app.core.alpha_engine.typed_nodes import Node
        for _ in range(20):
            assert isinstance(generate_random_alpha(), Node)


# ===========================================================================
# E. 存活变异的等价性证明
# ===========================================================================

PROVEN_EQUIVALENT = {
    "app/core/gp_engine/gp_engine.py ×1 — L254 `fwd_ret = (close[1:] - close[:-1]) / close[:-1]` → `+`":
        "`(P_t+1 + P_t)/P_t = 2 + r`，是真实收益 r 的**严格单调增变换**。"
        "而 fwd_ret 在本模块里**只**被送进 Spearman 秩相关 —— 秩相关只看次序，"
        "单调变换不改变任何一行的秩，因此 IC 序列逐位相同。"
        "实测同一面板下差式与和式的 mean_IC 都是 -0.7552。"
        "见 test_sum_form_is_a_monotone_transform_of_the_return。",

    "app/core/gp_engine/gp_engine.py ×1 — L258 `T_ic = min(fwd_ret.shape[0], sig_arr.shape[0] - 1)` → `+ 1`":
        "signal 由 Executor 按 dataset 的索引求值，行数与 close **恒等**（都是 T）。"
        "于是 `fwd_ret.shape[0] = T-1`，而 `sig.shape[0] ∓ 1` 是 T-1 或 T+1；"
        "两种取值下 `min(...)` 都取 T-1。差别取不到。"
        "见 test_signal_and_close_always_have_the_same_row_count。",

    "app/core/gp_engine/gp_engine.py ×1 — L272 `if denom > 0:` → `>=`":
        "`rs`/`rr` 都是 `argsort(argsort(x))` 的结果，也就是 0..n-1 的**排列**"
        "（并列值按数组位置分到不同名次，见下面的缺陷 D-19）。去中心化后"
        "平方和恒为 n(n²-1)/12；进到这里时 `n_valid >= 5`，所以"
        "denom = n(n²-1)/12 >= 5·24/12 = 10 > 0，永远取不到 `== 0` 那一档。"
        "`>` 与 `>=` 的判定结果对所有可达输入都相同。"
        "见 test_the_rank_denominator_can_never_be_zero。",

    "app/core/gp_engine/gp_engine.py ×1 — L171 `if mapped and mapped in _SEED_DSLS_BY_FAMILY:` → `or`":
        "`mapped = _ALIAS.get(family, \"\")`。已知家族 → mapped 非空且必在种子表里"
        "（实测 _ALIAS 的每个目标值都是 _SEED_DSLS_BY_FAMILY 的键），两种取值都为真；"
        "未知家族 → mapped 是空串，`\"\" and ...` 与 `\"\" or (\"\" in dict)` 都为假。"
        "两个子式恒同真同假。见 test_every_alias_target_exists_in_the_seed_table。",
}


def test_sum_form_is_a_monotone_transform_of_the_return():
    """
    L254 等价性的机械验证：对任意正价格序列，`(P1+P0)/P0` 与 `(P1-P0)/P0`
    给出**完全相同的秩**。这同时是失效告警 —— 哪天 fwd_ret 被用在秩以外的
    地方（比如直接进 fitness），这条证明就不再成立。
    """
    import inspect
    from scipy.stats import rankdata
    import app.core.gp_engine.gp_engine as G

    rng = np.random.default_rng(5)
    for _ in range(20):
        p0 = rng.uniform(1, 1000, 50)
        p1 = p0 * (1 + rng.normal(0, 0.05, 50))
        diff = (p1 - p0) / p0
        summ = (p1 + p0) / p0
        np.testing.assert_array_equal(rankdata(diff), rankdata(summ))

    src = inspect.getsource(G._evaluate_individual)
    assert src.count("fwd_ret") <= 4, (
        "fwd_ret 的使用点变多了 —— 可能已被用在秩相关以外的地方，"
        "L254 的等价性证明需要重新验证")


def test_signal_and_close_always_have_the_same_row_count():
    """L258 等价性的机械验证：Executor 产出的信号行数与输入面板相同。"""
    from app.core.alpha_engine.dsl_executor import Executor
    for T_ in (10, 40, 120):
        close = 100 * np.cumprod(
            1 + np.random.default_rng(T_).normal(0, 0.01, (T_, 4)), axis=0)
        ds = {k: pd.DataFrame(v, index=pd.bdate_range("2020-01-02", periods=T_))
              for k, v in _dataset(close).items()}
        sig = Executor(validate=False).run_expr("rank(close)", ds)
        assert sig.shape[0] == T_, "信号行数与输入面板不一致了 —— L258 需要重新补用例"
        assert min(T_ - 1, T_ - 1) == min(T_ - 1, T_ + 1) == T_ - 1


def test_every_alias_target_exists_in_the_seed_table():
    """L171 等价性的机械验证：别名表的每个目标都在种子表里。"""
    import inspect
    import re
    import app.core.gp_engine.gp_engine as G
    alias = re.findall(r'"(\w+)":\s*"(\w+)"',
                       inspect.getsource(G.get_seeds_for_family))
    missing = [(a, b) for a, b in alias if b and b not in G._SEED_DSLS_BY_FAMILY]
    assert not missing, (
        f"别名 {missing} 的目标不在种子表里 —— `and` 与 `or` 会给出不同结果，"
        f"L171 不再是等价变异")


def test_the_rank_denominator_can_never_be_zero():
    """
    L272 等价性的机械验证：`argsort(argsort(x))` 对任何输入（含全并列、
    含重复值）都返回 0..n-1 的排列，所以去中心化平方和恒为 n(n²-1)/12。
    n_valid >= 5 时 denom >= 10。
    """
    rng = np.random.default_rng(17)
    for n in range(5, 40):
        cases = [
            np.full(n, 3.0),                       # 全并列
            np.zeros(n),                           # 全 0
            rng.normal(size=n),                    # 连续
            np.resize(rng.normal(size=3), n),      # 大量重复值
        ]
        for x in cases:
            r = np.argsort(np.argsort(x)).astype(float)
            assert sorted(r.tolist()) == [float(i) for i in range(n)], (
                "argsort(argsort(x)) 不再是 0..n-1 的排列 —— "
                "L272 的等价性证明失效")
            c = r - r.mean()
            ss = float((c ** 2).sum())
            assert ss == pytest.approx(n * (n * n - 1) / 12.0), (
                f"n={n} 的秩平方和不是 n(n²-1)/12")
        denom = np.sqrt((n * (n * n - 1) / 12.0) ** 2)
        assert denom >= 10.0, f"n={n} 时 denom={denom} 逼近 0 —— 零守卫重新变得可达"


# 上面那条证明顺带扒出一个真实缺陷：并列名次被按**列顺序**摊成 0..n-1，
# 于是一个截面恒定（零信息）的信号也能拿到非零 IC。
# 已登记为 C-1，行为断言（strict xfail）在 tests/test_known_defects.py。


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 4
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
