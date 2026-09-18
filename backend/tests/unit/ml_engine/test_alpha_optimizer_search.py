"""
ml_engine/alpha_optimizer.py —— 超参数搜索的定钉测试（变异测试驱动）

来由：24 个变异点，首测击杀率 **4.2%**（存活 23）—— B 档倒数第二。

AlphaOptimizer 用 Optuna 在 IS 数据上调 SimulationConfig。它的核心承诺只有一条：
**OOS 数据在整个优化期间不得出现**，靠的是 IS 内部再切一刀 walk-forward
（IS_early 训练 / IS_late 伪 OOS）并用 `compute_fitness(is, oos)` 惩罚过拟合。
这一刀切错了，惩罚项就失效，优化器会挑出在 IS 上最过拟合的那组参数。

存活项正好铺在这条链上：
  - `late_start = int(total_days * (1.0 - is_late_ratio))` 的 `*` 与 `-`
    —— 切点算错：`*`→`/` 会让切点几乎为 0（整段都是"伪 OOS"），
       `-`→`+` 会让切点越过末尾（伪 OOS 为空，惩罚项直接消失）
  - `self._use_walkforward = True/False` 两处赋值 —— 开关反了就是"永远不惩罚"
  - `if self._use_walkforward and self._is_late` 的 `and`
  - 四处 `if not _isnan(x) else 兜底值` 的 `not`
    —— 取反之后**正常值被丢弃、一律用兜底常数**，所有 trial 打分相同
  - `if allow_neutralize and groups is not None` 的 `and` 与 `not`
  - `_isnan` 的两个 `return True`
  - `show_progress_bar=False`、`trial_values` 的 NaN 过滤

既有覆盖（unit/test_ml_optimizer）只跑了一次小规模 optimize 并检查返回类型。
本文件全部走**直接构造 + 打桩**，不跑真 Optuna（那太慢，且掩盖细节）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import app.core.ml_engine.alpha_optimizer as AO
from app.core.ml_engine.alpha_optimizer import (
    AlphaOptimizer,
    SearchSpace,
    StudySummary,
    _isnan,
)


def _ds(days: int, n: int = 4) -> dict:
    idx = pd.bdate_range("2022-01-03", periods=days)
    cols = [f"T{i}" for i in range(n)]
    rng = np.random.default_rng(0)
    close = pd.DataFrame(100 + rng.normal(0, 1, (days, n)).cumsum(axis=0),
                         index=idx, columns=cols)
    return {"close": close, "open": close, "high": close * 1.01,
            "low": close * 0.99, "vwap": close,
            "volume": pd.DataFrame(1e6, index=idx, columns=cols),
            "returns": np.log(close / close.shift(1))}


def _opt(days: int, **kw) -> AlphaOptimizer:
    return AlphaOptimizer(dsl="rank(close)", is_dataset=_ds(days), **kw)


# ===========================================================================
# A. walk-forward 切点
# ===========================================================================

class TestWalkForwardSplit:
    """
    `late_start = int(total_days * (1.0 - is_late_ratio))`，
    `late_days = total_days - late_start`，
    `if late_days >= _MIN_LATE_DAYS: 启用 walk-forward`。

    三行决定了过拟合惩罚项跑不跑、以及在多少数据上跑。
    """

    def test_split_point_follows_the_ratio(self):
        o = _opt(400, is_late_ratio=0.25)
        assert o._use_walkforward is True
        assert len(next(iter(o._is_late.values()))) == 400 - int(400 * 0.75), (
            "IS_late 的长度与 is_late_ratio 对不上 —— 切点算错了")

    def test_late_slice_is_the_tail_not_the_head(self):
        """
        `v.iloc[late_start:]` —— 伪 OOS 必须是**后**一段。
        切成前一段会让"训练在后、验证在前"，惩罚项彻底反向。
        """
        o = _opt(400, is_late_ratio=0.25)
        full = o._is_dataset["close"]
        late = o._is_late["close"]
        pd.testing.assert_frame_equal(late, full.iloc[-len(late):],
                                      obj="IS_late 取的不是尾段")

    def test_a_larger_ratio_gives_a_longer_holdout(self):
        """`*`→`/` 会让切点几乎恒为 0，两种 ratio 给出同样长的 hold-out。"""
        short = _opt(400, is_late_ratio=0.20)
        long_ = _opt(400, is_late_ratio=0.40)
        n_s = len(next(iter(short._is_late.values())))
        n_l = len(next(iter(long_._is_late.values())))
        assert n_l > n_s, f"ratio 变大 hold-out 没有变长（{n_s} → {n_l}）"
        assert n_s == 400 - int(400 * 0.80) and n_l == 400 - int(400 * 0.60)

    def test_holdout_never_swallows_the_whole_sample(self):
        """`*`→`/` 的直接后果：late_start≈0 → 伪 OOS 就是全样本，惩罚项失去意义。"""
        o = _opt(400, is_late_ratio=0.25)
        assert len(next(iter(o._is_late.values()))) < 400, (
            "IS_late 覆盖了整个 IS —— 切点塌到了 0")

    def test_walkforward_is_disabled_when_the_holdout_is_too_short(self):
        """
        `late_days >= _MIN_LATE_DAYS`：hold-out 不足 60 天就退化成纯 IS 目标。
        `self._use_walkforward = False` 改成 True 会让优化器去回测一个
        **空的/极短的** IS_late，得到毫无意义的 sharpe_late。
        """
        o = _opt(100, is_late_ratio=0.10)      # hold-out 只有 10 天
        assert o._use_walkforward is False, "hold-out 不足却仍启用了 walk-forward"
        assert o._is_late == {}, "禁用 walk-forward 时 IS_late 应当为空"

    def test_walkforward_is_enabled_at_exactly_the_minimum(self):
        """
        `>= _MIN_LATE_DAYS` 的边界：恰好 60 天必须**启用**。
        收紧成 `>` 会白白丢掉一档合法配置。
        """
        days, ratio = 300, 0.20
        o = _opt(days, is_late_ratio=ratio)
        assert days - int(days * 0.80) == AlphaOptimizer._MIN_LATE_DAYS, (
            "构造的 hold-out 长度不是恰好 60 —— 边界没测到")
        assert o._use_walkforward is True

    def test_the_input_dataset_is_deep_copied(self):
        """外部后续改动不得影响优化器内部（防止 OOS 数据被偷渡进来）。"""
        ds = _ds(300)
        o = AlphaOptimizer(dsl="rank(close)", is_dataset=ds)
        ds["close"].iloc[0, 0] = -999.0
        assert o._is_dataset["close"].iloc[0, 0] != -999.0, (
            "优化器持有的是外部对象的引用 —— 外部改动会渗进优化过程")


# ===========================================================================
# B. NaN 兜底值
# ===========================================================================

class TestNanFallbacks:
    """
    `sharpe = float(x) if not _isnan(x) else -5.0` —— 四处同形。
    删掉 `not` 之后逻辑彻底反过来：**有效值被丢弃、一律用兜底常数**。
    后果是所有 trial 拿到同一个 fitness，优化器退化成随机挑选，
    而 Optuna 照常跑完、照常返回一个 best_params，没有任何异常。
    """

    @staticmethod
    def _fake_report(sharpe, turnover, dd):
        return type("R", (), {"sharpe_ratio": sharpe, "ann_turnover": turnover,
                              "max_drawdown": dd})()

    @pytest.mark.parametrize("value", [np.nan, None, np.float64("nan")])
    def test_isnan_recognises_missing_values(self, value):
        # 注意用 bool()：数值分支返回的是 np.bool_，`is True` 会假失败。
        assert bool(_isnan(value)) is True

    def test_isnan_accepts_real_numbers(self):
        for v in (0.0, -3.5, 1e9, np.float64(2.0)):
            assert bool(_isnan(v)) is False, f"{v!r} 被判成了缺失值"

    def test_isnan_treats_unparseable_values_as_missing(self):
        """`except (TypeError, ValueError): return True` —— 无法转 float 的一律算缺失。"""
        for v in ("abc", object(), [1, 2]):
            assert bool(_isnan(v)) is True, f"{v!r} 没有被判成缺失值"

    def test_isnan_distinguishes_the_two_cases(self):
        """两个 `return True` 都改成 False 时，这条相等断言会成立 —— 必须不等。"""
        assert bool(_isnan(None)) != bool(_isnan(1.0)), (
            "_isnan 对缺失与有效给出了同一个答案")

    def _run_objective(self, monkeypatch, report, late_report=None):
        """
        把 RealisticBacktester 换成返回指定报告的替身，跑一次 `_objective`。

        这是唯一能真正压住那四行 `if not _isnan(...) else 常数` 的办法 ——
        在测试里把表达式照抄一遍只是在测我自己写的副本，被测代码改成什么样都不会红。
        """
        class _Res:
            def __init__(self, rep):
                self.is_report = rep

        class _BT:
            def __init__(self, config=None):
                self.cfg = config

            def run(self, dsl, dataset):
                n = len(next(iter(dataset.values())))
                use = late_report if (late_report is not None and n < 400) else report
                return _Res(use)

        # `_objective` 在函数体内 `from ...realistic_backtester import RealisticBacktester`，
        # 所以要替换的是**源模块**的属性，不是 alpha_optimizer 的（后者没有这个名字）。
        import app.core.backtest_engine.realistic_backtester as rb
        monkeypatch.setattr(rb, "RealisticBacktester", _BT)

        class _Trial:
            number = 0

            def suggest_int(self, name, lo, hi):
                return lo

            def suggest_float(self, name, lo, hi):
                return lo

            def suggest_categorical(self, name, choices):
                return choices[0]

        return _opt(400, is_late_ratio=0.25)._objective(_Trial())

    def test_a_valid_sharpe_scores_differently_from_a_missing_one(self, monkeypatch):
        """
        `sharpe_f = float(x) if not _isnan(x) else -5.0` —— 删掉 `not` 会让
        **有效 Sharpe 被丢弃、一律用 -5.0**：所有 trial 得到同一个 fitness，
        优化器退化成随机挑选，而 Optuna 照常跑完、照常返回一个 best_params。
        """
        good = self._run_objective(monkeypatch, self._fake_report(2.0, 0.5, -0.1))
        missing = self._run_objective(monkeypatch,
                                      self._fake_report(np.nan, np.nan, np.nan))
        assert good != missing, (
            f"有效 Sharpe 与缺失 Sharpe 打出了同一个分数（{good}）—— "
            f"`not _isnan(...)` 疑似被取反，指标根本没进目标函数")
        assert good > missing, (
            f"Sharpe=2.0 的得分 {good} 不高于全缺失的 {missing}")

    def test_the_objective_responds_monotonically_to_sharpe(self, monkeypatch):
        lo = self._run_objective(monkeypatch, self._fake_report(0.2, 0.5, -0.1))
        hi = self._run_objective(monkeypatch, self._fake_report(3.0, 0.5, -0.1))
        assert hi > lo, f"Sharpe 从 0.2 升到 3.0，目标值反而没变高（{lo} → {hi}）"

    def test_walk_forward_penalty_actually_bites(self, monkeypatch):
        """
        IS 高、伪 OOS 低 → 必须比两段都高的情形得分更低。
        这是过拟合惩罚项的存在意义；`_use_walkforward` 或切点被改坏之后，
        这两种情形会打出同一个分。
        """
        consistent = self._run_objective(
            monkeypatch, self._fake_report(2.0, 0.5, -0.1),
            late_report=self._fake_report(2.0, 0.5, -0.1))
        overfit = self._run_objective(
            monkeypatch, self._fake_report(2.0, 0.5, -0.1),
            late_report=self._fake_report(-1.0, 0.5, -0.1))
        assert overfit < consistent, (
            f"伪 OOS 崩掉的配置得分 {overfit} 不低于稳定配置 {consistent} —— "
            f"walk-forward 惩罚项没有生效")

    def test_inverted_truncation_bounds_are_rejected_outright(self, monkeypatch):
        """`if trunc_min >= trunc_max: return -999.0` —— 非法配置直接判死。"""
        class _Trial:
            number = 0

            def suggest_int(self, name, lo, hi):
                return lo

            def suggest_float(self, name, lo, hi):
                return 0.9        # min 与 max 都给 0.9 → min >= max

            def suggest_categorical(self, name, choices):
                return choices[0]

        space = SearchSpace(trunc_min_range=(0.9, 0.9), trunc_max_range=(0.9, 0.9))
        val = _opt(400, search_space=space)._objective(_Trial())
        assert val == -999.0, f"上下分位数颠倒的配置没有被判死，得分 {val}"


# ===========================================================================
# C. 行业中性化的开关
# ===========================================================================

class TestNeutralizeGating:
    """
    `if self._space.allow_neutralize and self._space.neutralize_groups is not None`
    与 `_params_to_config` 里的 `if params.get("neutralize") and groups is not None`。

    `and` 放宽成 `or`：只要开了 allow_neutralize 就去用 `groups`，而它是 None ——
    `ind_neutralize(x, None)` 会**静默退化成 cs_zscore**，
    参数表上写着"做了行业中性"，实际一次都没做。
    删掉 `not`：反过来，给了 groups 反而不用。
    """

    GROUPS = np.array([0, 0, 1, 1])

    @staticmethod
    def _objective_config(monkeypatch, space, pick=0):
        """
        跑一次 `_objective`，把它内部构造的 SimulationConfig 捞出来。

        `pick` 决定打桩的 `suggest_categorical` 取候选列表的第几个 ——
        `[True, False]` 这个字面量被改成 `[True, True]` / `[False, False]` 时，
        只取第 0 个是看不出来的，必须两端都取一次。
        """
        seen: dict = {}

        class _Res:
            is_report = type("R", (), {"sharpe_ratio": 1.0, "ann_turnover": 0.5,
                                       "max_drawdown": -0.1})()

        class _BT:
            def __init__(self, config=None):
                seen["cfg"] = config

            def run(self, dsl, dataset):
                return _Res()

        import app.core.backtest_engine.realistic_backtester as rb
        monkeypatch.setattr(rb, "RealisticBacktester", _BT)

        class _Trial:
            number = 0

            def suggest_int(self, name, lo, hi):
                return lo

            def suggest_float(self, name, lo, hi):
                return lo

            def suggest_categorical(self, name, choices):
                return choices[pick % len(choices)]

        _opt(400, search_space=space)._objective(_Trial())
        return seen["cfg"]

    def test_objective_applies_groups_when_the_trial_says_yes(self, monkeypatch):
        """
        `if allow_neutralize and groups is not None:` 后面紧跟
        `use_neutral = trial.suggest_categorical("neutralize", [True, False])`。

        `and`→`or` / 删 `not`：开关与分组不再需要同时具备 ——
        没有分组也会去 suggest，最终把 None 当成分组传下去，
        `ind_neutralize(x, None)` **静默退化成 cs_zscore**：
        配置上写着"做了行业中性"，实际一次都没做。
        """
        space = SearchSpace(allow_neutralize=True, neutralize_groups=self.GROUPS)
        cfg = self._objective_config(monkeypatch, space, pick=0)   # choices[0] = True
        assert cfg.neutralize_groups is not None, (
            "开关与分组都具备、trial 也选了 True，配置里却没有分组")
        np.testing.assert_array_equal(cfg.neutralize_groups, self.GROUPS)

    def test_objective_skips_groups_when_the_trial_says_no(self, monkeypatch):
        """
        `[True, False]` 这个候选列表必须**两个值都在**。
        改成 `[True, True]` 会让"不做中性化"这条路径在搜索空间里消失，
        优化器再也比较不出"中性化到底有没有用"。
        """
        space = SearchSpace(allow_neutralize=True, neutralize_groups=self.GROUPS)
        cfg = self._objective_config(monkeypatch, space, pick=1)   # choices[1] = False
        assert cfg.neutralize_groups is None, (
            "trial 选了 False，配置里却带上了分组 —— "
            "候选列表疑似被改成了 [True, True]")

    def test_objective_never_neutralises_without_groups(self, monkeypatch):
        space = SearchSpace(allow_neutralize=True, neutralize_groups=None)
        for pick in (0, 1):
            cfg = self._objective_config(monkeypatch, space, pick=pick)
            assert cfg.neutralize_groups is None, (
                "没有分组数组却在配置里标了中性化")

    def test_objective_never_neutralises_when_the_switch_is_off(self, monkeypatch):
        space = SearchSpace(allow_neutralize=False, neutralize_groups=self.GROUPS)
        for pick in (0, 1):
            cfg = self._objective_config(monkeypatch, space, pick=pick)
            assert cfg.neutralize_groups is None, (
                "allow_neutralize=False 却仍然做了中性化")

    def test_search_space_groups_stay_out_of_the_repr(self):
        """
        `neutralize_groups: ... = field(default=None, repr=False)` ——
        改成 True 会把整个行业码数组塞进 `repr(SearchSpace)`，
        而搜索空间的 repr 会进日志与 run_manifest。
        """
        s = SearchSpace(allow_neutralize=True, neutralize_groups=np.arange(500))
        r = repr(s)
        assert "neutralize_groups" not in r, f"分组数组进了 repr：{r[:200]}…"
        assert len(r) < 400, f"SearchSpace 的 repr 膨胀到了 {len(r)} 字符"

    def test_config_carries_groups_only_when_both_conditions_hold(self):
        o = _opt(300, search_space=SearchSpace(allow_neutralize=True,
                                               neutralize_groups=self.GROUPS))
        cfg = o._params_to_config({"neutralize": True})
        assert cfg.neutralize_groups is not None, (
            "开关与分组都具备，配置里却没有分组")
        np.testing.assert_array_equal(cfg.neutralize_groups, self.GROUPS)

    def test_no_groups_means_no_neutralisation_even_if_requested(self):
        o = _opt(300, search_space=SearchSpace(allow_neutralize=True,
                                               neutralize_groups=None))
        cfg = o._params_to_config({"neutralize": True})
        assert cfg.neutralize_groups is None, (
            "没有分组数组却在配置里标了中性化 —— 会静默退化成 cs_zscore")

    def test_not_requested_means_no_neutralisation(self):
        o = _opt(300, search_space=SearchSpace(allow_neutralize=True,
                                               neutralize_groups=self.GROUPS))
        assert o._params_to_config({"neutralize": False}).neutralize_groups is None
        assert o._params_to_config({}).neutralize_groups is None

    def test_search_space_defaults_keep_neutralisation_off(self):
        """
        `allow_neutralize: bool = False` 改成 True 会让**所有**优化默认
        把行业中性纳入搜索空间，而绝大多数调用方根本没提供 groups ——
        于是一半的 trial 跑的是"以为中性化了、实际没有"的配置。
        """
        s = SearchSpace()
        assert s.allow_neutralize is False, "搜索空间默认打开了行业中性"
        assert s.neutralize_groups is None, "搜索空间默认带上了分组数组"

    def test_other_search_space_defaults_are_pinned(self):
        s = SearchSpace()
        assert s.delay_range == (0, 5) and s.decay_range == (0, 10)
        assert s.trunc_min_range == (0.01, 0.10)
        assert s.trunc_max_range == (0.90, 0.99)
        assert s.portfolio_modes == ("long_short", "decile")

    def test_params_to_config_defaults_match_simulation_defaults(self):
        o = _opt(300)
        cfg = o._params_to_config({})
        assert (cfg.delay, cfg.decay_window) == (1, 0)
        assert (cfg.truncation_min_q, cfg.truncation_max_q) == (0.05, 0.95)
        assert cfg.portfolio_mode == "long_short"


# ===========================================================================
# D. StudySummary
# ===========================================================================

class TestStudySummary:

    def test_nan_trial_values_are_dropped_from_the_dict(self):
        """
        `[v for v in self.trial_values if not np.isnan(v)]` —— 删掉 `not`
        会让**只有 NaN 被保留**：摘要里的 trial_values 全是 NaN，
        任何据此画收敛曲线/算分位的分析都变成空的。
        """
        s = StudySummary(n_trials=4, best_value=1.0, best_params={"delay": 1},
                         trial_values=[1.0, np.nan, 0.5, np.nan])
        assert s.to_dict()["trial_values"] == [1.0, 0.5], (
            f"NaN 过滤方向反了：{s.to_dict()['trial_values']}")

    def test_all_nan_values_yield_an_empty_list(self):
        s = StudySummary(n_trials=2, best_value=0.0, best_params={},
                         trial_values=[np.nan, np.nan])
        assert s.to_dict()["trial_values"] == []

    def test_summary_dict_carries_every_field(self):
        s = StudySummary(n_trials=3, best_value=2.5,
                         best_params={"delay": 2}, trial_values=[1.0, 2.5])
        d = s.to_dict()
        assert d == {"n_trials": 3, "best_value": 2.5,
                     "best_params": {"delay": 2}, "trial_values": [1.0, 2.5]}

    def test_trial_values_stay_out_of_the_repr(self):
        """
        `trial_values: ... = field(repr=False)` —— 改成 True 会把几百个 trial 的
        取值全塞进 repr，而这个 repr 会进日志与 run_manifest。
        """
        s = StudySummary(n_trials=500, best_value=1.0, best_params={},
                         trial_values=[float(i) for i in range(500)])
        r = repr(s)
        assert "trial_values" not in r, f"trial_values 进了 repr：{r[:200]}…"
        assert len(r) < 300, f"摘要 repr 膨胀到了 {len(r)} 字符"


# ===========================================================================
# E. optimize() 的调用契约
# ===========================================================================

class TestOptimizeContract:

    def test_optuna_is_called_without_a_progress_bar(self, monkeypatch):
        """
        `show_progress_bar=False` —— 改成 True 会让 Optuna 往 stderr 打进度条。
        这个优化器跑在 nightly job 与 API 请求里，进度条会污染日志、
        在非 TTY 环境下还会刷出成千上万行控制字符。
        """
        pytest.importorskip("optuna")
        seen: dict = {}

        class _Study:
            trials = []
            best_params: dict = {"delay": 1}
            best_value = 1.0

            def optimize(self, fn, **kw):
                seen.update(kw)

        import optuna
        monkeypatch.setattr(optuna, "create_study", lambda **kw: _Study())
        o = _opt(300)
        o.optimize()
        assert seen.get("show_progress_bar") is False, (
            f"show_progress_bar={seen.get('show_progress_bar')} —— 进度条会污染日志")
        assert seen.get("n_jobs") == 1, (
            "n_jobs 不是 1 —— IS 数据会在多线程中被并发访问")

    def test_summary_counts_every_trial_but_only_scores_the_valid_ones(self, monkeypatch):
        """
        `trial_vals = [t.value for t in study.trials if t.value is not None]`
        —— 删掉 `not` 会让**只有失败的 trial**（value 为 None）进列表，
        随后 np.isnan(None) 抛 TypeError，整个 optimize 在最后一步炸掉。
        """
        pytest.importorskip("optuna")
        trials = [type("T", (), {"value": v})() for v in (1.0, None, 2.0)]

        class _Study:
            best_params: dict = {"delay": 1}
            best_value = 2.0

            def __init__(self):
                self.trials = trials

            def optimize(self, fn, **kw):
                pass

        import optuna
        monkeypatch.setattr(optuna, "create_study", lambda **kw: _Study())
        _cfg, summary = _opt(300).optimize()
        assert summary.n_trials == 3, "n_trials 应当统计全部 trial（含失败的）"
        assert summary.trial_values == [1.0, 2.0], (
            f"失败 trial 的 None 没有被过滤掉：{summary.trial_values}")


# ===========================================================================
# F. 存活变异的等价性证明
# ===========================================================================

PROVEN_EQUIVALENT = {
    "app/core/ml_engine/alpha_optimizer.py ×1 — L223 `if self._use_walkforward and self._is_late:` → `or`":
        "两个子式在 `__init__` 里是**同一个 if/else 的两条分支**同时赋的："
        "`late_days >= _MIN_LATE_DAYS` 成立 → `_is_late = {非空}` 且 "
        "`_use_walkforward = True`；否则 `_is_late = {}` 且 `_use_walkforward = False`。"
        "两者恒同真同假，`and` 与 `or` 给出相同结果。"
        "见 test_walkforward_flag_and_late_slice_are_always_consistent —— "
        "它同时是失效告警：哪天这两处被拆开赋值，那条会红。",
}


def test_walkforward_flag_and_late_slice_are_always_consistent():
    """L223 等价性的机械验证：两个子式在任何输入下真值相同。"""
    for days, ratio in ((400, 0.25), (300, 0.20), (100, 0.10), (80, 0.50),
                        (200, 0.05), (1000, 0.30)):
        o = _opt(days, is_late_ratio=ratio)
        assert bool(o._use_walkforward) == bool(o._is_late), (
            f"days={days} ratio={ratio}：_use_walkforward={o._use_walkforward} 与 "
            f"_is_late（{'非空' if o._is_late else '空'}）不一致 —— "
            f"L223 不再是等价变异，必须补用例")


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 1
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
