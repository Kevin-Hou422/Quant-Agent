"""
portfolio_manager/horizon.py —— 无交易带 + 因子快慢分类（Phase PM.6）

**此前零专属测试**（7 个变异点，D 档）。

模块只有两件事，但第一件**直接改真实下单权重**：

  `apply_no_trade_band` —— 目标权重相对已持仓漂移不足 `band` 就不动。
  这是"省成本"的减法。它坏掉的两种方式都不会报错：
    - 比较符翻面（`>=` → `>`）：恰好等于 band 的漂移不再触发调仓，
      权重永远停在旧仓位上 → 信号失效但账面一切正常；
    - `np.where(move, target, held)` 两个分支互换：变成"漂移小才调、
      漂移大反而不调"，等于**把无交易带彻底反过来用**。

  `annualized_turnover` —— 换手口径。`/2.0`（单边）翻成 `*2.0` 会让
  换手翻四倍，进而让所有因子被划成 fast，快慢分类整体失真；
  `* tdays_per_year` 翻成 `/` 则让年化换手掉到 1e-5 量级，全变 slow。

所以这里的手法是：**手算参考值逐位比对** + **边界精确构造**
（漂移精确等于 band、换手精确等于阈值）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.portfolio_manager.horizon import (
    FactorHorizon,
    annualized_turnover,
    apply_no_trade_band,
    classify_horizon,
    horizon_profile,
)


def _w(rows, cols=("A", "B")) -> pd.DataFrame:
    idx = pd.bdate_range("2022-01-03", periods=len(rows))
    return pd.DataFrame(np.array(rows, dtype=float), index=idx, columns=list(cols))


# ===========================================================================
# A. 年化换手
# ===========================================================================

class TestAnnualisedTurnover:

    def test_it_matches_the_documented_formula_bit_for_bit(self):
        """
        `mean_t(0.5 · Σ|w_t − w_{t-1}|) × 交易日/年`

        这是唯一能同时钉住 `/2.0`、`.iloc[1:]`、`* tdays_per_year`
        三处的办法：手算出参考值，逐位比。
        """
        w = _w([[0.5, 0.5],
                [0.2, 0.8],
                [0.9, 0.1],
                [0.9, 0.1]])
        # 逐日单边换手：|Δ| 之和的一半
        daily = [(abs(0.2 - 0.5) + abs(0.8 - 0.5)) / 2.0,
                 (abs(0.9 - 0.2) + abs(0.1 - 0.8)) / 2.0,
                 0.0]
        ref = float(np.mean(daily)) * 252.0
        got = annualized_turnover(w)
        assert got == pytest.approx(ref, rel=1e-12), (
            f"年化换手是 {got}，手算参考值 {ref}")

    def test_the_first_row_is_excluded_from_the_average(self):
        """
        `.iloc[1:]` —— 去掉之后第一行的 NaN（diff 的产物）会被
        `mean()` 跳过，**分母**却从 n-1 变成 n-1……看上去一样，
        但一旦 `diff()` 的第一行被 fillna(0) 之类改成 0，
        分母就多了一格，换手被系统性低估。

        构造：恒定权重后接一次大调仓 —— 若把第 0 行算进去，
        均值会被那一格稀释。
        """
        w = _w([[1.0, 0.0], [0.0, 1.0]])
        # 只有一次调仓，单边换手 = (1+1)/2 = 1.0，均值就是它本身
        assert annualized_turnover(w, tdays_per_year=1.0) == pytest.approx(1.0), (
            "两行一次全换的单边换手不是 1.0 —— 首行被算进了均值")

    def test_a_single_row_panel_has_zero_turnover(self):
        """
        `if len(weights) < 2: return 0.0`
        —— 守卫被删会让 `diff()` 全是 NaN，`mean()` 返回 NaN，
        然后 `float(nan) * 252` = nan 一路流进分类函数，
        `nan > 4.0` 恒假 → 所有因子被划成 slow。
        """
        assert annualized_turnover(_w([[1.0, 0.0]])) == 0.0

    def test_an_empty_panel_has_zero_turnover(self):
        assert annualized_turnover(pd.DataFrame()) == 0.0

    def test_the_guard_fires_below_two_rows_not_at_two(self):
        """`len < 2` 翻成 `<=` 会让两行的面板也返回 0，换手永远是 0。"""
        two = _w([[1.0, 0.0], [0.0, 1.0]])
        assert annualized_turnover(two, tdays_per_year=1.0) > 0.0, (
            "两行的面板换手被判成 0 —— `len(weights) < 2` 被翻成了 `<=`")

    def test_turnover_is_one_sided_not_double_counted(self):
        """
        `/ 2.0` —— 一次"卖光 A 买满 B"的调仓，双边变动是 2.0，
        单边口径应当是 1.0。这个 2 倍差别会直接翻倍所有成本估计。
        """
        w = _w([[1.0, 0.0], [0.0, 1.0]])
        assert annualized_turnover(w, tdays_per_year=1.0) == pytest.approx(1.0), (
            "全部换仓一次的单边换手不是 1.0 —— `/ 2.0` 被改了")

    def test_the_annualisation_multiplies_rather_than_divides(self):
        """`* tdays_per_year` 翻成 `/` 会让年化换手从 252 掉到 1/252。"""
        w = _w([[1.0, 0.0], [0.0, 1.0]])
        assert annualized_turnover(w, tdays_per_year=252.0) == pytest.approx(252.0)
        assert annualized_turnover(w, tdays_per_year=2.0) == pytest.approx(2.0)

    def test_a_never_changing_panel_has_zero_turnover(self):
        assert annualized_turnover(_w([[0.4, 0.6]] * 10)) == pytest.approx(0.0)

    def test_the_default_trading_year_is_two_hundred_and_fifty_two(self):
        w = _w([[1.0, 0.0], [0.0, 1.0]])
        assert annualized_turnover(w) == annualized_turnover(w, tdays_per_year=252.0)


# ===========================================================================
# B. 快慢分类
# ===========================================================================

class TestHorizonClassification:

    def test_turnover_exactly_on_the_threshold_is_slow(self):
        """
        `"fast" if turnover_ann > fast_threshold else "slow"`

        `>` 翻成 `>=` 只改变**恰好等于阈值**的那一格。
        4.0 是二进制可精确表示的，构造没有误差。
        """
        assert classify_horizon(4.0) == "slow", (
            "换手恰好等于阈值被划成了 fast —— `>` 被翻成了 `>=`")
        assert classify_horizon(np.nextafter(4.0, np.inf)) == "fast"
        assert classify_horizon(np.nextafter(4.0, 0.0)) == "slow"

    def test_the_two_branches_are_not_swapped(self):
        """三元表达式两个分支互换会让高换手因子被当成慢因子放大配额。"""
        assert classify_horizon(50.0) == "fast"
        assert classify_horizon(0.1) == "slow"

    def test_the_threshold_is_configurable(self):
        assert classify_horizon(3.0, fast_threshold=2.0) == "fast"
        assert classify_horizon(3.0, fast_threshold=10.0) == "slow"

    def test_the_default_threshold_is_four_per_year(self):
        assert classify_horizon(4.5) == "fast"
        assert classify_horizon(3.5) == "slow"
        assert classify_horizon(4.5, fast_threshold=4.0) == "fast"


# ===========================================================================
# C. 无交易带 —— 直接改下单权重的那一段
# ===========================================================================

class TestNoTradeBand:

    def test_a_drift_exactly_on_the_band_does_trade(self):
        """
        `move = np.abs(target - held) >= band`

        `>=` 翻成 `>` 只改变**漂移精确等于 band** 的一格。
        构造 0.0 → 0.1，band = 0.1 —— 这三个数在二进制下
        `abs(0.1 - 0.0) == 0.1` 精确成立（下面第一条断言守住这个前提）。

        契约是"**≥ band 才调**"，所以等于时必须调到目标。
        """
        band = 0.1
        assert abs(0.1 - 0.0) == band, "前提失效：漂移不再精确等于 band"
        out = apply_no_trade_band(_w([[0.0, 0.0], [0.1, 0.0]]), band=band)
        assert out.iloc[1, 0] == pytest.approx(0.1), (
            "漂移恰好等于 band 却没有调仓 —— `>= band` 被翻成了 `> band`")

    def test_a_downward_drift_triggers_a_trade_just_like_an_upward_one(self):
        """
        **首测存活项（L52）**：`move = np.abs(target - held) >= band`

        `np.abs` 被删之后条件变成 `(target - held) >= band` ——
        **只有加仓会被执行，减仓永远不执行**。
        结果是权重单调向上棘轮：信号让你减到 0，账户却一直满仓，
        而权重面板本身完全合法，换手甚至更低（看起来"更省成本"）。

        之前的用例全是从 0 往上加，所以一条都杀不掉它。
        这里对称地测一次减仓。
        """
        # 先建仓到 0.5，再让目标降到 0.0（漂移 -0.5，绝对值远超 band）
        out = apply_no_trade_band(_w([[0.5, 0.0], [0.0, 0.0]]), band=0.1)
        assert out.iloc[1, 0] == pytest.approx(0.0), (
            f"目标从 0.5 降到 0，无交易带却把仓位留在 {out.iloc[1, 0]} —— "
            f"`np.abs(target - held)` 的绝对值没了，减仓永远不触发")

    def test_the_band_is_symmetric_around_the_holding(self):
        """
        同一行里放一个上漂、一个下漂，幅度相同。
        绝对值被删时下漂那一列会被卡住，上漂那一列照常 —— 必被发现。
        """
        out = apply_no_trade_band(_w([[0.5, 0.5], [0.9, 0.1]]), band=0.2)
        assert out.iloc[1].tolist() == pytest.approx([0.9, 0.1]), (
            f"等幅的上漂与下漂被区别对待：{out.iloc[1].tolist()} —— "
            f"无交易带不再关于持仓对称")

    def test_a_small_downward_drift_is_still_suppressed(self):
        """绝对值的另一侧：小幅减仓同样应当被无交易带吃掉。"""
        out = apply_no_trade_band(_w([[0.5, 0.0], [0.45, 0.0]]), band=0.1)
        assert out.iloc[1, 0] == pytest.approx(0.5), "小幅减仓不该触发调仓"

    def test_a_drift_below_the_band_keeps_the_old_position(self):
        out = apply_no_trade_band(_w([[0.0, 0.0], [0.09, 0.0]]), band=0.1)
        assert out.iloc[1, 0] == pytest.approx(0.0), (
            "漂移不足 band 却调仓了 —— 无交易带没生效")

    def test_the_two_where_branches_are_not_swapped(self):
        """
        `np.where(move, target, held)` —— 两个分支互换会让逻辑完全反过来：
        小漂移去调仓、大漂移反而按兵不动。
        一行里同时放一个大漂移和一个小漂移，互换必被发现。
        """
        out = apply_no_trade_band(_w([[0.0, 0.0], [0.50, 0.01]]), band=0.1)
        assert out.iloc[1, 0] == pytest.approx(0.50), "大漂移没有调到目标"
        assert out.iloc[1, 1] == pytest.approx(0.00), "小漂移却被调走了"

    def test_the_holding_is_carried_forward_across_multiple_days(self):
        """
        `held = np.where(...)` 每天更新。写成对 `arr[t-1]` 取差
        会让"相对已持仓"退化成"相对昨日目标"——
        于是连续多日的小幅漂移会被逐日忽略，而它们累计起来早已超过 band。
        """
        # 每天漂 0.04（< 0.1），五天累计 0.20（> 0.1）→ 第三天起应当触发
        rows = [[0.00, 0.0], [0.04, 0.0], [0.08, 0.0], [0.12, 0.0], [0.16, 0.0]]
        out = apply_no_trade_band(_w(rows), band=0.1)
        col = out.iloc[:, 0].tolist()
        assert col[1] == pytest.approx(0.0), "第 1 天漂 0.04 不应触发"
        assert col[2] == pytest.approx(0.0), "第 2 天累计 0.08 不应触发"
        assert col[3] == pytest.approx(0.12), (
            f"第 3 天累计漂移 0.12 已超过 band 却没调仓：{col} —— "
            f"漂移是相对**已持仓**算的，不是相对昨日目标")
        assert col[4] == pytest.approx(0.12), "第 4 天相对新持仓只漂 0.04，不应触发"

    def test_the_first_row_is_always_taken_as_is(self):
        """`held = arr[0].copy()` —— 建仓日没有"已持仓"，必须原样接受。"""
        out = apply_no_trade_band(_w([[0.3, 0.7], [0.31, 0.69]]), band=0.1)
        assert out.iloc[0].tolist() == pytest.approx([0.3, 0.7])

    def test_a_non_positive_band_returns_the_panel_untouched(self):
        """
        `if band <= 0 ... return weights`
        —— 守卫被删会让 band=0 时 `>= 0` 恒真，虽然结果相同，
        但 band 为负（配置错误）时 `np.abs(...) >= -1` 也恒真，
        同样"全调"，掩盖了配置错误。这里钉住"原样返回"这个契约。
        """
        w = _w([[0.1, 0.9], [0.5, 0.5]])
        assert apply_no_trade_band(w, band=0.0) is w, "band=0 没有原样返回同一对象"
        assert apply_no_trade_band(w, band=-1.0) is w, "band<0 没有原样返回同一对象"

    def test_the_guard_lets_a_positive_band_through(self):
        """`band <= 0` 翻成 `< 0` 或 `<= 1` 都会改变哪些 band 生效。"""
        w = _w([[0.0, 0.0], [0.01, 0.0]])
        out = apply_no_trade_band(w, band=0.5)
        assert out is not w, "正的 band 被当成关闭 —— 守卫的比较符被改了"
        assert out.iloc[1, 0] == pytest.approx(0.0)

    def test_a_short_or_empty_panel_is_returned_untouched(self):
        one = _w([[1.0, 0.0]])
        assert apply_no_trade_band(one, band=0.1) is one
        empty = pd.DataFrame()
        assert apply_no_trade_band(empty, band=0.1) is empty

    def test_the_input_panel_is_not_mutated_in_place(self):
        """
        `np.array(..., copy=True)` —— `copy=False` 会让这个"只做减法"
        的函数把调用方手里的目标权重面板改掉，
        后面任何重算都会基于被污染的数据。
        """
        w = _w([[0.0, 0.0], [0.01, 0.0], [0.9, 0.0]])
        before = w.copy(deep=True)
        apply_no_trade_band(w, band=0.1)
        pd.testing.assert_frame_equal(w, before)

    def test_the_index_and_columns_survive(self):
        w = _w([[0.0, 0.0], [0.5, 0.5]], cols=("XOM", "AAPL"))
        out = apply_no_trade_band(w, band=0.1)
        assert list(out.columns) == ["XOM", "AAPL"]
        assert list(out.index) == list(w.index)

    def test_the_band_only_ever_reduces_turnover(self):
        """
        模块纪律写明"只做减法"。无交易带之后的换手**不可能**
        高于原始换手 —— 这条不变量对任何随机面板都成立，
        任何把逻辑写反的变异都会撞上它。
        """
        rng = np.random.default_rng(20260914)
        raw = pd.DataFrame(rng.uniform(-1, 1, (120, 8)),
                           index=pd.bdate_range("2022-01-03", periods=120),
                           columns=[f"S{i}" for i in range(8)])
        for band in (0.05, 0.1, 0.3):
            banded = apply_no_trade_band(raw, band=band)
            assert annualized_turnover(banded) <= annualized_turnover(raw) + 1e-12, (
                f"band={band} 之后换手反而上升了 —— 无交易带做成了加法")

    def test_a_larger_band_never_trades_more(self):
        rng = np.random.default_rng(7)
        raw = pd.DataFrame(rng.uniform(-1, 1, (150, 5)),
                           index=pd.bdate_range("2022-01-03", periods=150),
                           columns=list("ABCDE"))
        tos = [annualized_turnover(apply_no_trade_band(raw, band=b))
               for b in (0.02, 0.10, 0.40)]
        assert tos[0] >= tos[1] >= tos[2], f"band 越大换手没有单调下降：{tos}"


# ===========================================================================
# D. 因子画像
# ===========================================================================

class _StubPortfolio:
    """把 SignalWeightedPortfolio 换成"信号即权重"，让换手可手算。"""

    def __init__(self, clip_z=3.0):
        _StubPortfolio.last_clip_z = clip_z

    def construct(self, signal):
        return signal


class TestHorizonProfile:

    @staticmethod
    def _patch(monkeypatch, cls=_StubPortfolio):
        import app.core.backtest_engine.portfolio_constructor as pc
        monkeypatch.setattr(pc, "SignalWeightedPortfolio", cls)

    def test_each_factor_gets_one_entry_with_its_own_turnover(self, monkeypatch):
        self._patch(monkeypatch)
        signals = {
            "fast_one": _w([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
            "slow_one": _w([[0.5, 0.5]] * 4),
        }
        out = horizon_profile(signals, tdays_per_year=252.0)
        assert [f.factor for f in out] == ["fast_one", "slow_one"], (
            "因子顺序或数量不对 —— 画像与输入对不上")
        assert out[0].horizon == "fast" and out[1].horizon == "slow"
        assert out[0].turnover_ann == pytest.approx(252.0)
        assert out[1].turnover_ann == pytest.approx(0.0)

    def test_a_failing_factor_is_skipped_rather_than_killing_the_profile(self,
                                                                         monkeypatch):
        """
        `except Exception: continue` —— 源码注释自己写明
        "静默跳过 = 该因子从画像里消失"。这里钉住：
        一个坏因子不能带走其他因子的结论。
        """
        class _Flaky(_StubPortfolio):
            def construct(self, signal):
                if signal.attrs.get("boom"):
                    raise RuntimeError("constructor blew up")
                return signal

        self._patch(monkeypatch, _Flaky)
        bad = _w([[1.0, 0.0], [0.0, 1.0]])
        bad.attrs["boom"] = True
        out = horizon_profile({"bad": bad,
                               "good": _w([[1.0, 0.0], [0.0, 1.0]])})
        assert [f.factor for f in out] == ["good"], (
            f"坏因子没有被跳过，或把好因子也带走了：{[f.factor for f in out]}")

    def test_the_thresholds_are_forwarded(self, monkeypatch):
        self._patch(monkeypatch)
        sig = {"f": _w([[1.0, 0.0], [0.0, 1.0]])}
        # 该信号每日全换，年化换手 252 —— 阈值抬到 1000 之上就该判 slow
        slow = horizon_profile(sig, fast_threshold=1000.0, tdays_per_year=252.0)
        assert slow[0].turnover_ann == pytest.approx(252.0)
        assert slow[0].horizon == "slow", "fast_threshold 没有透传"
        assert horizon_profile(sig, fast_threshold=1.0,
                               tdays_per_year=252.0)[0].horizon == "fast"

    def test_clip_z_is_forwarded_to_the_portfolio_constructor(self, monkeypatch):
        self._patch(monkeypatch)
        horizon_profile({"f": _w([[1.0, 0.0], [0.0, 1.0]])}, clip_z=2.5)
        assert _StubPortfolio.last_clip_z == 2.5, "clip_z 没有透传给组合构造器"

    def test_the_trading_year_is_forwarded(self, monkeypatch):
        self._patch(monkeypatch)
        out = horizon_profile({"f": _w([[1.0, 0.0], [0.0, 1.0]])},
                              tdays_per_year=10.0)
        assert out[0].turnover_ann == pytest.approx(10.0), "tdays_per_year 没有透传"

    def test_an_empty_input_gives_an_empty_profile(self, monkeypatch):
        self._patch(monkeypatch)
        assert horizon_profile({}) == []

    def test_the_defaults_are_the_documented_ones(self, monkeypatch):
        self._patch(monkeypatch)
        sig = {"f": _w([[1.0, 0.0], [0.0, 1.0]])}
        assert horizon_profile(sig) == horizon_profile(
            sig, clip_z=3.0, fast_threshold=4.0, tdays_per_year=252.0)


class TestFactorHorizonDto:

    def test_to_dict_rounds_turnover_to_three_decimals(self):
        """
        `round(self.turnover_ann, 3)` —— 位数被改会让前端展示的
        换手要么精度不足、要么刷一长串小数。
        """
        d = FactorHorizon("mom", 12.3456789, "fast").to_dict()
        assert d == {"factor": "mom", "turnover_ann": 12.346, "horizon": "fast"}

    def test_to_dict_carries_all_three_fields(self):
        d = FactorHorizon("rev", 1.0, "slow").to_dict()
        assert set(d) == {"factor", "turnover_ann", "horizon"}

    def test_the_dataclass_compares_by_value(self):
        assert FactorHorizon("a", 1.0, "slow") == FactorHorizon("a", 1.0, "slow")
        assert FactorHorizon("a", 1.0, "slow") != FactorHorizon("a", 1.0, "fast")
