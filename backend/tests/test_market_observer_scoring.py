"""
discovery/market_observer.py —— 自主发现打分的定钉测试（变异测试驱动）

来由：22 个变异点，首测击杀率 **22.7%**（存活 17）。

MarketObserver 是 Phase 9「自主发现」的起点：它把市场状态压成六个因子族的
分数，`top_families()` 决定**今晚 GP 往哪个方向挖**。它算错不会报错、
不会让任何测试变红，只会让系统长期往错的方向挖因子 ——
这是最难事后发现的一类问题：产出一直有，只是一直挖错方向。

存活项几乎铺满了整个打分公式：

  - `trend_strength = abs(breadth - 0.5) * 2.0` 的 `-` 与 `*`
    —— 改成 `+` 会让广度 0.5（完全无方向）算出最强趋势；改成 `/` 让值域塌到 [0,0.25]
  - 六个族分的 `0.20 + 0.50 * trend_strength` 里的 `+` 与 `*`
    —— `*`→`/` 时 trend_strength≈0 会让分母趋零，分数炸到 inf，
       那个族**永远排第一**
  - `(mom > 0).mean()` 的 `>`、`len(s) > 5 and s.std() > 0` 的 `and`
    —— 放宽后零波动序列进入自相关计算，autocorr 返回 NaN 污染均值

既有覆盖（test_phase9_market_observer）只验证"返回了六个族、分数在 [0,1]、
regime 是已知字符串"，没有一条把**具体市场形态 → 具体族排序**钉死。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.discovery.market_observer import MarketObserver


T, N = 260, 12
IDX = pd.bdate_range("2023-01-02", periods=T)
COLS = [f"T{i}" for i in range(N)]


def _panel(returns: np.ndarray) -> dict:
    r = pd.DataFrame(returns, index=IDX, columns=COLS)
    close = 100 * np.exp(r.fillna(0.0).cumsum())
    return {"returns": r, "close": close}


def _trending_up(strength: float = 0.004, seed: int = 0) -> dict:
    """所有资产同向上涨 → 广度 ≈ 1，趋势一致性 ≈ 1。"""
    rng = np.random.default_rng(seed)
    return _panel(strength + rng.normal(0, 0.002, (T, N)))


def _no_direction(seed: int = 1) -> dict:
    """一半涨一半跌 → 广度 ≈ 0.5，趋势一致性 ≈ 0。"""
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.002, (T, N))
    r[:, : N // 2] += 0.004
    r[:, N // 2:] -= 0.004
    return _panel(r)


def _mean_reverting(seed: int = 2, balanced: bool = False) -> dict:
    """
    强负一阶自相关 → reversal_strength 打满。

    `balanced=True` 再叠一层 ±0.002 的对称漂移，把趋势广度压成**恰好 0.5**
    （一半涨一半跌），于是 trend_strength = 0、momentum 的原始分锁定为 0.20。
    这给了一个可以反推 reversion 原始分的支点（见
    test_reversal_strength_saturates_at_one）。
    """
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, 0.01, (T, N))
    r = np.zeros((T, N))
    r[0] = eps[0]
    for t in range(1, T):
        r[t] = eps[t] - 0.9 * r[t - 1]
    if balanced:
        r = r + np.where(np.arange(N) < N // 2, 0.002, -0.002)[None, :]
    return _panel(r)


def _score(obs, family: str) -> float:
    return next(h.score for h in obs.hypotheses if h.family == family)


# ===========================================================================
# A. 趋势一致性
# ===========================================================================

class TestTrendStrength:

    def test_uniform_uptrend_gives_full_breadth(self):
        """
        `breadth = (mom > 0).mean()` —— **严格大于 0**。放宽成 `>=` 会把
        累计收益恰好为 0 的资产也算成"上涨"，广度虚高。
        """
        obs = MarketObserver().observe(_trending_up())
        assert obs.momentum_breadth > 0.95, (
            f"全体同向上涨时广度只有 {obs.momentum_breadth}")

    def test_flat_market_breadth_counts_zero_as_not_up(self):
        flat = _panel(np.zeros((T, N)))
        obs = MarketObserver().observe(flat)
        assert obs.momentum_breadth == 0.0, (
            f"累计收益恰好为 0 被算成了上涨（广度={obs.momentum_breadth}）—— "
            f"`mom > 0` 被放宽成了 `>=`")

    def test_momentum_scores_high_when_the_market_is_one_directional(self):
        """
        `trend_strength = abs(breadth - 0.5) * 2.0`：
        `-`→`+` 会让 breadth=0.5（毫无方向）变成 abs(1.0)*2=2 的满分趋势；
        `*`→`/` 会把值域从 [0,1] 压到 [0,0.25]，趋势市与震荡市几乎无差别。
        这里用"全体同向"与"一半对一半"两个极端把它钉住。
        """
        one_way = MarketObserver().observe(_trending_up())
        split = MarketObserver().observe(_no_direction())
        assert _score(one_way, "momentum") > _score(split, "momentum"), (
            f"方向一致的市场动量分 {_score(one_way,'momentum'):.3f} "
            f"没有高于无方向市场 {_score(split,'momentum'):.3f}")

    def test_no_direction_market_does_not_rank_momentum_first(self):
        obs = MarketObserver().observe(_no_direction())
        assert obs.top_families(1)[0] != "momentum", (
            f"一半涨一半跌的市场把 momentum 排在了第一：{obs.top_families(3)}")

    def test_trend_following_tracks_momentum(self):
        """两个族共用 trend_strength，一个被改坏时排序关系会破裂。"""
        obs = MarketObserver().observe(_trending_up())
        assert _score(obs, "momentum") >= _score(obs, "trend_following"), (
            "momentum 的系数比 trend_following 大，分数却更低")

    def test_trend_strength_reaches_exactly_one_at_full_breadth(self):
        """
        `trend_strength = abs(breadth - 0.5) * 2.0` —— `*` 改成 `/` 会把值域
        从 [0,1] 压到 [0,0.25]，趋势市与震荡市的分差被抹掉四分之三。
        只比"谁大谁小"抓不住（两种写法都单调），必须钉住**数值**。

        归一化把绝对分抹掉了，但**两个族的比值**不受影响：
          momentum        = 0.20 + 0.50·ts + 0.30·trending
          trend_following = 0.15 + 0.45·ts + 0.25·trending
        全体同向上涨 → breadth=1 → ts 应当恰好 = 1；该面板 regime=high_vol
        （trending=0），于是比值 = 0.60 / 0.70 = 6/7。
        若 ts 被压成 0.25，比值变成 0.2625/0.325 = 0.8077，立刻对不上。
        """
        obs = MarketObserver().observe(_trending_up())
        assert obs.momentum_breadth == 1.0, (
            f"构造的面板广度不是 1.0（{obs.momentum_breadth}）—— 前提不成立")
        assert obs.regime == "high_vol", (
            f"该面板的 regime 变成了 {obs.regime} —— 下面的期望值按 trending=0 推导，"
            f"regime 判定改了就要重新推导")
        ratio = _score(obs, "trend_following") / _score(obs, "momentum")
        assert ratio == pytest.approx(0.60 / 0.70, rel=1e-9), (
            f"trend_following / momentum = {ratio:.6f}，应为 6/7 —— "
            f"trend_strength 没有取到 1.0")


# ===========================================================================
# B. 反转强度
# ===========================================================================

class TestReversal:

    def test_mean_reverting_market_scores_reversion_above_a_trending_one(self):
        """`np.clip(-reversal * 3.0, 0, 1)` 的 `*`→`/` 会让强反转市场的分数
        反而更低（除以 3 之后再 clip），reversion 族永远挖不出来。"""
        mr = MarketObserver().observe(_mean_reverting())
        tr = MarketObserver().observe(_trending_up())
        assert mr.short_term_reversal < -0.2, (
            f"构造的均值回复序列自相关只有 {mr.short_term_reversal} —— "
            f"数据没有区分力")
        assert _score(mr, "reversion") > _score(tr, "reversion"), (
            f"均值回复市场的 reversion 分 {_score(mr,'reversion'):.3f} "
            f"没有高于趋势市场 {_score(tr,'reversion'):.3f}")

    def test_reversal_strength_saturates_at_one(self):
        """
        `reversal_strength = clip(-reversal * 3.0, 0, 1)` —— `*` 改成 `/`
        会让最强的均值回复（autocorr ≈ -0.9）只得到 0.30 而不是打满的 1.0，
        reversion 族从此永远拿不到高分，震荡市里挖不出反转因子。

        用 momentum 当支点把归一化抵消掉。面板取 balanced=True，
        广度恰好 0.5 → trend_strength = 0；regime=high_vol → trending=0，
        于是 momentum 的**原始分锁定为 0.20**，与 reversal_strength 完全无关。
          reversion = 0.20 + 0.50·rs + 0.30·choppy(=1) → rs=1 时 = 1.00（同时也是它的上界）
        所以 momentum 的归一化分 = 0.20 / reversion_raw，应当恰好是 0.20。
        若 rs 被压成 0.296（=0.887/3），reversion 只有 0.648，
        momentum 的归一化分升到 0.309 —— 一眼看得出。
        """
        obs = MarketObserver().observe(_mean_reverting(balanced=True))
        assert obs.regime == "high_vol", (
            f"该面板 regime 变成了 {obs.regime} —— 期望值按 choppy=1/trending=0 推导")
        assert obs.momentum_breadth == 0.5, (
            f"广度不是 0.5（{obs.momentum_breadth}）—— momentum 原始分不再锁定在 0.20")
        assert obs.short_term_reversal < -0.8, (
            f"构造的自相关只有 {obs.short_term_reversal} —— 达不到饱和区，前提不成立")
        assert _score(obs, "reversion") == pytest.approx(1.0), (
            "强均值回复市场里 reversion 不是最高分")
        assert _score(obs, "momentum") == pytest.approx(0.20, rel=1e-9), (
            f"momentum 的归一化分是 {_score(obs,'momentum'):.6f}，应为 0.20 —— "
            f"reversal_strength 没有饱和到 1.0")

    def test_relative_volatility_is_measured_against_a_half_baseline(self):
        """
        `vol_norm = clip(vol_level - 0.5, 0, 1)` —— `-` 改成 `+` 会让
        **任何**近期波动（vol_level ≈ 1，即与历史持平）都算成满格高波动，
        volatility 族从此长期霸榜。

        该面板 vol_level ≈ 1.03 → vol_norm 应为 0.53 左右，
        volatility 原始分 0.15+0.55·0.53+0.20 ≈ 0.64 < momentum 的 0.70。
        若改成 `+`，vol_norm 被 clip 到 1.0 → volatility 原始分 0.90 > 0.70，
        它会顶掉 momentum 成为第一。
        """
        obs = MarketObserver().observe(_trending_up())
        assert 0.9 < obs.vol_level < 1.2, (
            f"该面板的相对波动是 {obs.vol_level}，不在预期区间 —— 前提不成立")
        assert _score(obs, "volatility") < _score(obs, "momentum"), (
            f"相对波动只有 {obs.vol_level:.3f}（与历史基本持平），"
            f"volatility 却压过了 momentum —— `vol_level - 0.5` 疑似变成了 `+`")
        assert obs.top_families(1)[0] == "momentum"

    def test_autocorr_skips_zero_variance_columns(self):
        """
        `if len(s) > 5 and s.std() > 0` —— `and` 放宽成 `or` 会让**零波动**列
        进入 `s.autocorr()`，返回 NaN，被 `np.isfinite` 挡下之前就已经污染循环；
        更糟的是 `or` 让长度 <= 5 的列也进来，autocorr 在样本过少时同样是 NaN。
        这里放一整列常数进去，要求自相关仍是有限值。
        """
        rng = np.random.default_rng(3)
        r = rng.normal(0, 0.01, (T, N))
        r[:, 0] = 0.0                       # 零波动列
        obs = MarketObserver().observe(_panel(r))
        assert np.isfinite(obs.short_term_reversal), (
            f"零波动列把自相关污染成了 {obs.short_term_reversal}")

    def test_only_qualifying_columns_contribute_to_the_autocorr_mean(self):
        """
        `if len(s) > 5 and s.std() > 0:` 与 `if a is not None and np.isfinite(a):`
        两道筛子，三处变异各自放进不同的垃圾：

          `> 5` → `>= 5`      : 恰好 5 个有效点的列被放进来（样本太少，自相关是噪声）
          `and` → `or`（L174）: 只要"点数够"**或**"有波动"就放进来 ——
                                3 个点的列也会进，autocorr 在那里是纯噪声
          `and` → `or`（L176）: NaN 的自相关被 append，`np.mean` 直接变 NaN，
                                整个 short_term_reversal 报废

        构造一张表，只有**一列**该被采纳，其余四列各踩一种情形；
        断言最终值恰好等于那一列自己的 lag-1 自相关。
        """
        rng = np.random.default_rng(11)
        r = np.full((T, 5), np.nan)
        r[:, 0] = 0.01                                  # 零波动 → std == 0
        r[:3, 1] = rng.normal(0, 0.01, 3)               # 只有 3 个有效点
        r[:5, 2] = rng.normal(0, 0.01, 5)               # 恰好 5 个有效点
        r[:, 3] = 0.01
        r[-1, 3] = 0.05                                 # std>0 但滞后序列恒定 → autocorr = NaN
        good = rng.normal(0, 0.01, T)
        r[:, 4] = good                                  # 唯一合格列

        recent = pd.DataFrame(r, index=IDX, columns=[f"C{i}" for i in range(5)])
        obs = MarketObserver(lookback=T).observe(
            {"returns": recent, "close": 100 * np.exp(recent.fillna(0).cumsum())})

        expected = pd.Series(good).autocorr(lag=1)
        assert np.isfinite(obs.short_term_reversal), (
            f"自相关变成了 {obs.short_term_reversal} —— NaN 被放进了均值")
        assert obs.short_term_reversal == pytest.approx(expected, rel=1e-9), (
            f"自相关均值 {obs.short_term_reversal:.6f} != 唯一合格列的 "
            f"{expected:.6f} —— 有不该参与的列混了进来")

    def test_all_columns_degenerate_falls_back_to_zero(self):
        obs = MarketObserver().observe(_panel(np.zeros((T, N))))
        assert obs.short_term_reversal == 0.0, (
            "全常数面板的自相关不是 0 —— 空列表回退失效")


# ===========================================================================
# C. 六个族分的结构
# ===========================================================================

class TestFamilyScores:
    """
    每个族分都是 `基线 + 系数 × 特征`。`+`→`-` 会让基线变成扣分，
    `*`→`/` 在特征趋零时把分数炸成 inf —— 归一化之后那个族恒为 1.0，
    **永远排第一**，自主发现从此只挖一个方向。
    """

    @pytest.fixture(scope="class")
    def obs(self):
        return MarketObserver().observe(_trending_up())

    def test_all_six_families_are_present(self, obs):
        fams = {h.family for h in obs.hypotheses}
        assert fams == {"momentum", "trend_following", "reversion",
                        "volatility", "liquidity", "price_volume_corr"}, fams

    def test_every_score_is_finite_and_within_zero_one(self, obs):
        for h in obs.hypotheses:
            assert np.isfinite(h.score), f"{h.family} 的分数是 {h.score}"
            assert 0.0 <= h.score <= 1.0, f"{h.family} 的分数越界：{h.score}"

    def test_scores_are_normalised_so_the_best_is_exactly_one(self, obs):
        assert max(h.score for h in obs.hypotheses) == pytest.approx(1.0), (
            "归一化之后最高分不是 1.0")

    def test_hypotheses_are_sorted_by_score_descending(self, obs):
        scores = [h.score for h in obs.hypotheses]
        assert scores == sorted(scores, reverse=True), f"没有按分数降序：{scores}"

    def test_scores_are_not_all_equal(self, obs):
        """`*`→`/` 造成的 inf 会让归一化后多个族并列 1.0。"""
        scores = [round(h.score, 6) for h in obs.hypotheses]
        assert len(set(scores)) > 1, f"六个族给出了同一个分数：{scores}"
        assert sum(1 for s in scores if s == 1.0) == 1, (
            f"有多个族并列满分，疑似分数被除出了 inf：{scores}")

    def test_liquidity_and_price_volume_share_the_same_formula(self, obs):
        """两行公式完全相同；任一行被改坏，这条相等断言立刻破裂。"""
        assert _score(obs, "liquidity") == pytest.approx(
            _score(obs, "price_volume_corr")), (
            "两个共用同一公式的族给出了不同分数")

    def test_top_families_returns_the_requested_count_in_order(self, obs):
        top3 = obs.top_families(3)
        assert top3 == [h.family for h in obs.hypotheses[:3]]
        assert len(obs.top_families(1)) == 1

    def test_recent_dispersion_spike_lifts_the_liquidity_family(self):
        """
        dispersion_norm = clip(近窗截面 std / 全样本截面 std / 2, 0, 1)，
        只喂 liquidity / price_volume_corr 两族。

        两个面板的**每日市场收益（截面均值）完全相同** —— 只把近 90 天的
        截面离差放大 5 倍。这样 regime、vol_level、trend_strength 都不变，
        变的只有离散度，`0.20 + 0.40 * dispersion_norm` 那一项被单独隔离出来。
        （第一版对比两个独立面板，结果 regime 也跟着变了，动量分把归一化
        基准抬高，liquidity 的**归一化**分反而更低 —— 断言错在构造，不在实现。）
        """
        rng = np.random.default_rng(4)
        base = rng.normal(0, 0.002, (T, N))
        mkt = base.mean(axis=1, keepdims=True)
        dev = base - mkt
        flat = mkt + dev
        spike = mkt + dev * np.where(np.arange(T)[:, None] >= T - 90, 5.0, 1.0)

        lo = MarketObserver().observe(_panel(flat))
        hi = MarketObserver().observe(_panel(spike))
        assert hi.dispersion > lo.dispersion * 2, (
            f"构造的离散度尖峰没有生效：{hi.dispersion} vs {lo.dispersion}")
        assert hi.regime == lo.regime, (
            f"两个面板的 regime 不同（{hi.regime} vs {lo.regime}）—— 隔离失败")
        assert _score(hi, "liquidity") > _score(lo, "liquidity"), (
            f"近期离散度飙升时 liquidity 分没有提高："
            f"{_score(hi,'liquidity'):.3f} vs {_score(lo,'liquidity'):.3f}")


# ===========================================================================
# D. 输入与序列化
# ===========================================================================

class TestInputHandling:

    def test_returns_are_derived_from_close_when_absent(self):
        ds = _trending_up()
        only_close = {"close": ds["close"]}
        obs = MarketObserver().observe(only_close)
        assert np.isfinite(obs.momentum_breadth)
        assert obs.momentum_breadth > 0.9, "从 close 派生的收益率算出了不同的广度"

    def test_infinities_in_returns_do_not_leak_into_the_scores(self):
        ds = _trending_up()
        r = ds["returns"].copy()
        r.iloc[5, 0] = np.inf
        r.iloc[6, 1] = -np.inf
        obs = MarketObserver().observe({"returns": r, "close": ds["close"]})
        assert np.isfinite(obs.dispersion) and np.isfinite(obs.vol_level), (
            "inf 没有被清成 NaN，污染了离散度/波动水平")

    def test_regime_falls_back_to_sideways_on_insufficient_data(self):
        short = pd.DataFrame(np.zeros((3, 2)),
                             index=pd.bdate_range("2024-01-02", periods=3),
                             columns=["A", "B"])
        obs = MarketObserver().observe({"returns": short,
                                        "close": (1 + short).cumprod() * 100})
        assert obs.regime == "sideways", f"数据不足时 regime 不是 sideways：{obs.regime}"

    def test_to_dict_round_trips_every_field(self):
        obs = MarketObserver().observe(_trending_up())
        d = obs.to_dict()
        assert set(d) == {"regime", "dispersion", "momentum_breadth",
                          "short_term_reversal", "vol_level", "hypotheses"}
        assert len(d["hypotheses"]) == 6
        assert d["momentum_breadth"] == round(obs.momentum_breadth, 4)

    def test_every_hypothesis_carries_a_rationale(self):
        obs = MarketObserver().observe(_trending_up())
        for h in obs.hypotheses:
            assert h.rationale and len(h.rationale) > 10, (
                f"{h.family} 没有可读的理由：{h.rationale!r}")
