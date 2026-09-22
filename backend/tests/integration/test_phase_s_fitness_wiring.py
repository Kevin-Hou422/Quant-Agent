"""
test_phase_s_fitness_wiring.py — Phase S.1：purged CV 口径**真的接进了** GP

为什么在 integration 而不是 unit：这些用例会构造 `PopulationEvolver` 并跑
真实的 `RealisticBacktester`（一次完整的信号→权重→回测管线）。按 DEV_LESSONS §Z，
跑重引擎的用例属于集成层；unit 层只留 `purged_cv_sharpe` 的纯函数契约
（见 tests/unit/gp_engine/test_purged_cv_fitness.py）。

守住的两件事：
  1. 口径真的换了（不是写了个函数没人调 —— §K）；
  2. 换了之后**说得出来**：`oos_metric` 跟着数字走，否则跨轮对比会把口径变化
     读成因子变好/变坏。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.gp_engine.population_evolver import PopulationEvolver


def _panel(T=900, N=6, seed=0, drift=0.0) -> dict:
    rng = np.random.default_rng(seed)
    q = np.linspace(-1, 1, N)
    ret = drift * q[None, :] + rng.normal(0, 0.01, (T, N))
    idx = pd.bdate_range("2019-01-02", periods=T)
    cols = [f"S{i}" for i in range(N)]
    close = pd.DataFrame(100 * np.cumprod(1 + ret, axis=0), index=idx, columns=cols)
    # 随机日内幅度，不用固定 ±1%（DEV_LESSONS §O）
    out = {"close": close, "open": close, "vwap": close,
           "high": close * (1 + rng.uniform(0.001, 0.008, close.shape)),
           "low":  close * (1 - rng.uniform(0.001, 0.008, close.shape)),
           "volume": pd.DataFrame(1e6, index=idx, columns=cols)}
    out["returns"] = close.pct_change().fillna(0.0)
    return out


_DSL = "rank(ts_mean(returns,20))"


class TestTheEvolverActuallyUsesPurgedCv:

    def test_the_release_default_is_purged_cv(self):
        """
        不传 fitness_mode 时读的是发布配置。这条钉住"默认口径已经换了" ——
        否则新代码可以一直躺在一个没人打开的开关后面。
        """
        ev = PopulationEvolver(is_data=_panel(600), oos_data=_panel(300, seed=9))
        assert ev._fitness_mode == "purged_cv", (
            f"发布默认仍是 {ev._fitness_mode!r} —— S.1 的口径没有接进主线")

    def test_the_metric_label_follows_the_number(self):
        ev = PopulationEvolver(is_data=_panel(900, drift=0.002),
                               oos_data=_panel(300, seed=9), pop_size=4, n_generations=1)
        r = ev._evaluate_one_single(_DSL)
        assert r is not None, "评估失败，本用例无从断言"
        assert r.oos_metric == "purged_cv", (
            f"用了 K 折口径却仍自称 {r.oos_metric!r} —— 跨轮对比会把口径变化"
            f"读成因子变好/变坏")
        assert r.to_dict()["oos_metric"] == "purged_cv", "to_dict 没把口径带出去"

    def test_holdout_mode_still_labels_itself_holdout(self):
        ev = PopulationEvolver(is_data=_panel(900, drift=0.002),
                               oos_data=_panel(300, seed=9),
                               pop_size=4, n_generations=1, fitness_mode="holdout")
        r = ev._evaluate_one_single(_DSL)
        assert r is not None and r.oos_metric == "holdout"

    def test_the_two_modes_really_produce_different_selection_scores(self):
        """
        两种口径必须给出**不同**的 sharpe_oos。若相同，说明 purged_cv 分支
        其实没走到（或者悄悄退回了单段），那这个开关就是个装饰。
        """
        is_data, oos_data = _panel(900, drift=0.002), _panel(300, seed=9, drift=0.002)
        a = PopulationEvolver(is_data=is_data, oos_data=oos_data, pop_size=4,
                              n_generations=1, fitness_mode="holdout")._evaluate_one_single(_DSL)
        b = PopulationEvolver(is_data=is_data, oos_data=oos_data, pop_size=4,
                              n_generations=1, fitness_mode="purged_cv")._evaluate_one_single(_DSL)
        assert a is not None and b is not None
        assert a.sharpe_oos != b.sharpe_oos, (
            f"两种口径给出完全相同的 sharpe_oos={a.sharpe_oos} —— purged_cv 分支没生效")

    def test_exactly_two_usable_folds_still_count_as_purged_cv(self):
        """
        `if cv["n_folds"] >= 2` —— **恰好 2 折**就够用了。收紧成 `> 2`
        会让这种情形静默退回单段口径，而 `oos_metric` 仍标着…… 不，正是
        因为标签跟着走，这条才测得出来：标签会变回 holdout。
        """
        ev = PopulationEvolver(is_data=_panel(900, drift=0.002),
                               oos_data=_panel(300, seed=9), pop_size=4,
                               n_generations=1, cv_folds=2, embargo_days=20)
        r = ev._evaluate_one_single(_DSL)
        assert r is not None, "评估失败，本用例无从断言"
        assert r.oos_metric == "purged_cv", (
            "只有 2 折时被退回单段口径 —— 2 折已经能去掉单段的段位偏倚了")

    def test_it_still_uses_purged_cv_when_the_config_cannot_be_read(self, monkeypatch):
        """
        §U：读不到配置要朝**更严**的一侧倒。单段 holdout 方差更大、且分数取决于
        最后那一段恰好是什么行情 —— "读不到就退回单段"是更松的那一侧。

        （同组的其他兜底断言在 tests/unit/entrypoint/test_phase_s_config_fallbacks.py；
        这一条因为要构造 PopulationEvolver 而留在集成层，见 §Z。）
        """
        import sys
        monkeypatch.setitem(sys.modules, "app.config", None)
        ev = PopulationEvolver(is_data=_panel(300), oos_data=_panel(200))
        assert ev._fitness_mode == "purged_cv", (
            f"读不到 s_fitness_mode 兜底成了 {ev._fitness_mode!r}")
        assert ev._cv_folds == 5 and ev._embargo_days == 20, (
            f"折数/隔离兜底成了 {ev._cv_folds}/{ev._embargo_days}")

    def test_an_unknown_fitness_mode_is_rejected_at_construction(self):
        """
        拼错的口径名如果被默默当成"非 purged_cv 即 holdout"，使用者会以为
        自己开了 CV 而实际没开。当场拒绝。
        """
        with pytest.raises(ValueError, match="fitness_mode"):
            PopulationEvolver(is_data=_panel(300), oos_data=_panel(200),
                              fitness_mode="purged-cv")
