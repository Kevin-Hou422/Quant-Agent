"""
core/workflows/alpha_workflows.py —— Workflow A/B 内部helper 的定钉测试（变异测试驱动）

来由：69 个变异点，首测击杀率 **20.6%**（存活 54）。既有覆盖
（`test_phase5` / `test_phase_s_holdout` / `test_supplementary_fixes` /
`test_phase6_reproducibility`）走的是整条工作流，端到端跑通把内部
几乎所有分支都盖住了。

存活项里最要紧的三类：

  1. **三段切割的下限** `if n_is < 20: raise` —— 放宽会让 IS 段只剩十几天
     也照跑，算出来的 Sharpe 纯属噪声；而调用方接到的不是异常，是数字。
  2. **样本充足性标记** `insufficient_sample` 的 `or` 与两处 True/False
     —— 这个标记是"120 天数据报出 OOS Sharpe=15.78"那次事故的修复。
     标记丢了，只配用于内部排序的数字会被原样送进 API 展示给人看。
  3. **组合 IC 的取值对齐** `s, r = sig_arr[t], ret_arr[t + 1]` 的 `+`
     —— 改成 `-` 就是拿**同期**收益算 IC，前视泄漏，IC 会漂亮得离谱。
     `T = min(sig.shape[0] - 1, ret.shape[0] - 1)` 的 `-` 同理。

测法：这些 helper 全是模块级函数，能单独调。只有 Workflow A/B 的
`run()` 需要替身（GP + Optuna 真跑要几十秒），用桩 `PopulationEvolver`。
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import app.core.workflows.alpha_workflows as W


T, N = 200, 6


def _panel(seed: int = 0, n: int = T) -> dict:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, (n, N)), axis=0)
    idx = pd.bdate_range("2021-01-04", periods=n)
    cols = [f"S{i}" for i in range(N)]
    out = {k: pd.DataFrame(v, index=idx, columns=cols) for k, v in {
        "close": close, "open": close, "vwap": close,
        "high": close * 1.01, "low": close * 0.99,
        "volume": np.full_like(close, 1e6),
    }.items()}
    out["returns"] = out["close"].pct_change().fillna(0.0)
    return out


# ===========================================================================
# A. WorkflowResult.to_dict
# ===========================================================================

class TestWorkflowResultDict:

    @staticmethod
    def _res(**kw):
        base = dict(workflow="A", best_dsl="rank(close)", metrics={"is_sharpe": 1.0})
        base.update(kw)
        return W.WorkflowResult(**base)

    def test_combined_metrics_appear_only_when_present(self):
        """
        `if self.combined_metrics is not None:` —— 改成 `is None` 会让
        **有**组合指标时不输出（前端的组合面板永远空），
        没有时反而塞一个 `None` 进去（前端拿到 null 当成"组合失败"）。
        """
        plain = self._res().to_dict()
        assert "combined_metrics" not in plain, (
            "没有组合指标时却输出了 combined_metrics 键")

        withc = self._res(combined_metrics={"n_alphas": 3}).to_dict()
        assert withc["combined_metrics"] == {"n_alphas": 3}, (
            "有组合指标却没有输出 —— `is not None` 的判定反了")

    def test_an_empty_combined_dict_still_counts_as_present(self):
        """空 dict 不是 None —— 边界上必须按 `is not None` 而不是真值判断。"""
        assert "combined_metrics" in self._res(combined_metrics={}).to_dict()


# ===========================================================================
# B. 三段切割
# ===========================================================================

class TestPartitionThreeWay:

    def test_the_three_segments_tile_the_panel_without_gaps_or_overlap(self):
        ds = _panel(1)
        is_d, val_d, test_d = W._partition_three_way(ds, 0.3, 0.15)
        n = len(ds["close"])
        lens = [len(is_d["close"]), len(val_d["close"]), len(test_d["close"])]
        assert sum(lens) == n, f"三段长度 {lens} 合计 {sum(lens)}，面板是 {n} 天"
        # 时间上必须严格递增、不重叠
        assert is_d["close"].index[-1] < val_d["close"].index[0]
        assert val_d["close"].index[-1] < test_d["close"].index[0]

    @pytest.mark.parametrize("n_days,should_raise", [
        (200, False),
        (40,  False),
        (28,  False),     # n_is 刚好够
        (24,  True),      # n_is < 20
        (10,  True),
    ])
    def test_too_little_data_is_refused_rather_than_silently_shrunk(
            self, n_days, should_raise):
        """
        `if n_is < 20: raise ValueError(...)` —— **严格小于 20**。

        放宽成 `<=` 只差一天，看不出来；但这道闸的意义是"IS 段短到这个程度
        就不要给数字了"。它一旦松掉，调用方拿到的不是异常而是一个
        由十几天数据算出来的 Sharpe —— 上游会把它当成真结论。
        """
        ds = _panel(2, n=n_days)
        if should_raise:
            with pytest.raises(ValueError) as ei:
                W._partition_three_way(ds, 0.3, 0.15)
            assert "数据不足" in str(ei.value)
        else:
            is_d, _, _ = W._partition_three_way(ds, 0.3, 0.15)
            assert len(is_d["close"]) >= 20

    def test_the_boundary_value_itself_is_accepted(self):
        """
        构造一个 `n_is` **恰好等于 20** 的面板：`< 20` 放行、`<= 20` 拒绝。
        这是唯一能把两种取值分开的一格。
        """
        # n_test = max(1, int(n*0.15)); n_val = max(1, int(n*0.3) - n_test)
        # 找到使 n_is == 20 的 n
        target = None
        for n in range(20, 120):
            n_test = max(1, int(n * 0.15))
            n_val = max(1, int(n * 0.3) - n_test)
            if n - n_val - n_test == 20:
                target = n
                break
        assert target is not None, "找不到 n_is 恰好为 20 的面板尺寸"
        is_d, _, _ = W._partition_three_way(_panel(3, n=target), 0.3, 0.15)
        assert len(is_d["close"]) == 20, (
            f"n_is 恰好 20 的面板被拒了 —— `n_is < 20` 被收紧成了 `<= 20`")

    def test_every_field_is_sliced_the_same_way(self):
        """`_slice` 对 dataset 的每个字段做同样的切片 —— 漏一个就会对不齐。"""
        ds = _panel(4)
        is_d, val_d, test_d = W._partition_three_way(ds, 0.3, 0.15)
        for seg in (is_d, val_d, test_d):
            assert set(seg) == set(ds), f"切片后字段集合变了：{set(seg) ^ set(ds)}"
            lens = {len(v) for v in seg.values()}
            assert len(lens) == 1, f"同一段里各字段长度不一致：{lens}"


# ===========================================================================
# C. _quick_metrics 的数值兜底与样本充足性标记
# ===========================================================================

def _report(sharpe, *, turnover=1.0, insuf=False, n_days=100, se=0.1):
    return SimpleNamespace(sharpe_ratio=sharpe, ann_turnover=turnover,
                           insufficient_sample=insuf, n_days=n_days,
                           sharpe_se=se, max_drawdown=-0.1)


class _StubBt:
    """替身回测器：由类属性决定返回什么报告。"""
    is_report = None
    oos_report = None

    def __init__(self, *a, **kw):
        pass

    def run(self, dsl, is_data, oos_dataset=None):
        return SimpleNamespace(is_report=type(self).is_report,
                               oos_report=type(self).oos_report)


@pytest.fixture
def stub_bt(monkeypatch):
    import app.core.backtest_engine.realistic_backtester as rb
    monkeypatch.setattr(rb, "RealisticBacktester", _StubBt)
    return _StubBt


class TestQuickMetrics:

    def test_a_nan_sharpe_is_coerced_to_zero_not_propagated(self, stub_bt):
        """
        `return fv if not np.isnan(fv) else 0.0` —— 删掉 `not` 会把
        **正常数值**换成 0.0、把 NaN 原样放出去。前者让所有指标恒为 0
        （GP 的选择退化成掷硬币），后者让 NaN 一路漂进 API 响应。
        """
        stub_bt.is_report = _report(float("nan"))
        stub_bt.oos_report = _report(1.5)
        m = W._quick_metrics("rank(close)", {"close": 1}, {"close": 1})
        assert m["is_sharpe"] == 0.0, "NaN 没有被兜成 0"

        stub_bt.is_report = _report(2.25)
        m2 = W._quick_metrics("rank(close)", {"close": 1}, {"close": 1})
        assert m2["is_sharpe"] == pytest.approx(2.25), (
            f"正常数值被改成了 {m2['is_sharpe']} —— "
            f"`if not np.isnan(fv)` 的 not 被删掉了")

    @pytest.mark.parametrize("is_insuf,oos_insuf,expect", [
        (False, False, False),
        (True,  False, True),     # 只有 IS 不足 —— `or` 改 `and` 时会漏报
        (False, True,  True),     # 只有 OOS 不足
        (True,  True,  True),
    ])
    def test_insufficient_sample_is_the_or_of_both_segments(
            self, stub_bt, is_insuf, oos_insuf, expect):
        """
        `insuf = bool(getattr(oos_r, "insufficient_sample", False)
                      or getattr(is_r, "insufficient_sample", False))`

        两个 `False` 默认值各是一个变异点，翻成 True 会让**每一次**回测都
        报样本不足（标记失去意义，人会学会忽略它）。而只测"两段都不足"
        那一格分不开 or/and —— 必须把**单边不足**的两格也测到。
        """
        stub_bt.is_report = _report(1.0, insuf=is_insuf)
        stub_bt.oos_report = _report(0.8, insuf=oos_insuf)
        m = W._quick_metrics("rank(close)", {"close": 1}, {"close": 1})
        assert m["insufficient_sample"] is expect, (
            f"IS 不足={is_insuf}、OOS 不足={oos_insuf} 时标记算成了 "
            f"{m['insufficient_sample']}，应当是 {expect}")

    def test_a_report_without_the_attribute_defaults_to_sufficient(self, stub_bt):
        """
        `getattr(..., "insufficient_sample", False)` 的两个 False 默认值 ——
        翻成 True 会让任何**没有**这个属性的报告都被判成样本不足。
        """
        bare = SimpleNamespace(sharpe_ratio=1.0, ann_turnover=1.0,
                               n_days=500, sharpe_se=0.05)
        stub_bt.is_report = bare
        stub_bt.oos_report = bare
        m = W._quick_metrics("rank(close)", {"close": 1}, {"close": 1})
        assert m["insufficient_sample"] is False, (
            "报告里没有 insufficient_sample 属性时默认判成了『不足』 —— "
            "getattr 的默认值被翻了")

    def test_a_failed_backtest_is_flagged_as_insufficient(self, monkeypatch):
        """
        异常兜底里 `"insufficient_sample": True` —— 翻成 False 会让
        **回测失败**时返回的一串 0.0 被标成"样本充足"，
        也就是把"算不出来"伪装成"算出来是 0"。
        """
        import app.core.backtest_engine.realistic_backtester as rb

        class _Boom:
            def __init__(self, *a, **kw):
                pass

            def run(self, *a, **kw):
                raise RuntimeError("模拟：回测炸了")

        monkeypatch.setattr(rb, "RealisticBacktester", _Boom)
        m = W._quick_metrics("rank(close)", {"close": 1}, {"close": 1})
        assert m["insufficient_sample"] is True, (
            "回测失败的兜底把样本标成了『充足』 —— 算不出来被伪装成了 0")
        assert m["n_obs_oos"] == 0
        assert m["is_sharpe"] == 0.0 and m["fitness"] == 0.0


# ===========================================================================
# D. 种子生成
# ===========================================================================

class TestGenerateDiverseSeeds:

    def test_the_same_seed_gives_the_same_seed_list(self, monkeypatch):
        """
        `if seed is not None: _shared_rng.bind_seed(seed)` —— 改成 `is None`
        会让**给了 seed** 时反而不绑定（结果不可复现），没给时拿 None 去
        `bind_seed(None)`。可复现性是这条工作流的硬要求（R-N1）。
        """
        monkeypatch.setattr(W, "_hypothesis_templates", lambda h: [])
        a = W._generate_diverse_seeds("动量", n_target=8, seed=123)
        b = W._generate_diverse_seeds("动量", n_target=8, seed=123)
        assert a == b, "同一个 seed 两次产出不同的种子表 —— 随机源没有被绑定"

    def test_different_seeds_give_different_seed_lists(self, monkeypatch):
        monkeypatch.setattr(W, "_hypothesis_templates", lambda h: [])
        a = W._generate_diverse_seeds("动量", n_target=8, seed=1)
        b = W._generate_diverse_seeds("动量", n_target=8, seed=999)
        assert a != b, "不同 seed 产出了完全相同的种子表 —— 随机源没起作用"

    def test_the_seed_list_is_filled_to_the_target(self, monkeypatch):
        """
        两个填充循环各带 `attempts < n_target * 12` / `* 20` 的上限。
        `*` 改成 `/` 会让上限 < 1 —— **一次都不尝试**，种子表只剩模板那几条。
        `and` 放宽成 `or` 会让循环在填满后继续空转到上限。
        """
        monkeypatch.setattr(W, "_hypothesis_templates", lambda h: ["rank(close)"])
        out = W._generate_diverse_seeds("whatever", n_target=12, seed=7)
        assert len(out) == 12, (
            f"目标 12 条种子，实得 {len(out)} —— 填充循环的尝试上限被压没了")

    def test_the_fill_loops_stop_exactly_at_the_target(self, monkeypatch):
        """
        两处 `while len(valid_dsls) < n_target` 放宽成 `<=` 都会多填一条。

        注意上界只对**填充层**成立：Layer 1（LLM）与 Layer 2（模板）是
        无条件追加的，模板比 n_target 多时结果本来就会超过 n_target。
        所以基准取 `max(n_target, 模板条数)`。
        """
        tpl = ["rank(close)"]
        monkeypatch.setattr(W, "_hypothesis_templates", lambda h: list(tpl))
        # Layer 1（AlphaAgent）没有 LLM 时也会吐一批回退种子，同样是无条件追加。
        # 把它关掉，剩下的就只有模板 + 两个填充循环。
        import app.agent.alpha_agent as AA

        class _NoAgent:
            def __init__(self, *a, **kw):
                raise RuntimeError("本用例关掉 Layer 1")

        monkeypatch.setattr(AA, "AlphaAgent", _NoAgent)
        for n in (5, 9, 12):
            out = W._generate_diverse_seeds("whatever", n_target=n, seed=3)
            assert len(out) <= max(n, len(tpl)), (
                f"目标 {n} 条（模板 {len(tpl)} 条）却产出了 {len(out)} 条 —— "
                f"填充循环的 `< n_target` 被放宽成了 `<=`")

    def test_the_seed_list_has_no_duplicates(self, monkeypatch):
        """
        三处 `if key not in seen:` —— 删掉 `not` 会让去重反过来：
        只有**已经有的**才被加入，种子表变成同一条 DSL 的 N 份复制，
        GP 的初始种群毫无多样性，而 `len(seed_dsls)` 看起来是满的。
        """
        monkeypatch.setattr(W, "_hypothesis_templates",
                            lambda h: ["rank(close)", "rank(close)", "rank(volume)"])
        out = W._generate_diverse_seeds("whatever", n_target=12, seed=11)
        assert len(set(out)) == len(out), (
            f"种子表里有重复：{[d for d in out if out.count(d) > 1][:3]} —— "
            f"去重的 `not in seen` 被删掉了")

    def test_every_returned_seed_parses_and_validates(self, monkeypatch):
        """
        `_try_add` 的三个 return（None→False、重复→False、成功→True）
        各是一个变异点。翻掉任何一个都会让**解析不了的字符串**进种子表，
        或者让合法种子被丢掉。这里从"产出全部可解析"这个不变式去抓。
        """
        monkeypatch.setattr(W, "_hypothesis_templates",
                            lambda h: ["rank(close)", "这不是DSL(", "ts_mean(close,5)"])
        out = W._generate_diverse_seeds("whatever", n_target=10, seed=5)
        assert out, "种子表是空的"
        for d in out:
            assert W._parse_valid(d) is not None, (
                f"种子表里混进了解析不了的条目：{d!r} —— "
                f"`_try_add` 的返回值被翻了")

    def test_an_unparsable_template_is_rejected_not_stored(self, monkeypatch):
        monkeypatch.setattr(W, "_hypothesis_templates", lambda h: ["这不是DSL("])
        out = W._generate_diverse_seeds("whatever", n_target=4, seed=5)
        assert "这不是DSL(" not in out


# ===========================================================================
# E. Workflow B 的候选扩展
# ===========================================================================

class TestExpandForOptimization:

    def test_the_canonical_form_is_always_first(self):
        out = W._expand_for_optimization("rank(close)", n_mutations=5)
        assert out[0] == repr(W._parse_valid("rank(close)")), (
            "扩展结果的第一条不是原式的规范形 —— 原始 DSL 丢了")

    @pytest.mark.parametrize("n_mut", [3, 5, 8])
    def test_the_mutation_count_excludes_the_canonical_entry(self, n_mut):
        """
        `while len(results) - 1 < n_mutations and ...` —— 那个 `- 1` 是在
        **排除首条规范形**。改成 `+ 1` 会让循环少做两轮，
        要 8 个变异体实得 6 个；`<` 放宽成 `<=` 则多做一轮。
        """
        out = W._expand_for_optimization("rank(ts_mean(close,10))", n_mutations=n_mut)
        # 规范形 + 变异体 + 随机补位，总数下限由 `n_mutations + 3` 那个循环决定
        assert len(out) >= n_mut + 1, (
            f"要 {n_mut} 个变异体，连规范形一共只产出 {len(out)} 条 —— "
            f"`len(results) - 1 < n_mutations` 的偏移被改了")

    @pytest.mark.parametrize("n_mut", [3, 5, 8])
    def test_the_random_fill_targets_three_beyond_the_mutations(self, n_mut):
        """
        `while len(results) < n_mutations + 3 and attempts < 60:` ——
        `+` 改成 `-` 会让目标从 n+3 掉到 n-3，随机补位整个不做，
        Workflow B 的候选池比声称的小一截。
        """
        out = W._expand_for_optimization("rank(ts_mean(close,10))", n_mutations=n_mut)
        assert len(out) >= n_mut + 3, (
            f"n_mutations={n_mut} 时候选只有 {len(out)} 条，"
            f"至少应当到 {n_mut + 3} —— `n_mutations + 3` 的算符被改了")

    def test_the_candidate_list_has_no_duplicates(self):
        """两处 `if key not in seen:` —— 删 `not` 会让候选池全是重复项。"""
        out = W._expand_for_optimization("rank(ts_mean(close,10))", n_mutations=8)
        assert len(set(out)) == len(out), (
            f"候选池里有重复：{[d for d in out if out.count(d) > 1][:3]}")

    def test_an_unparsable_seed_is_refused_loudly(self):
        with pytest.raises(ValueError) as ei:
            W._expand_for_optimization("这不是DSL(")
        assert "Cannot parse" in str(ei.value)


# ===========================================================================
# F. 诊断驱动的定向变异
# ===========================================================================

class TestTargetedMutations:
    """
    三条阈值：换手 >3.0 加平滑、OOS Sharpe <0.3 加 rank/组合、过拟合 >0.5
    加 ts_mean。每条的边界都必须精确，否则"什么时候该做什么补救"整个偏掉。
    """

    BASE = "rank(ts_delta(close,5))"

    def _run(self, **metrics):
        m = {"turnover": 0.0, "oos_sharpe": 1.0, "overfitting_score": 0.0}
        m.update(metrics)
        return W._targeted_mutations(self.BASE, m)

    @pytest.mark.parametrize("turnover,expect", [
        (3.0, False),     # 恰好 3.0 → 不触发（`> 3.0` 严格大于）
        (3.01, True),
        (2.9, False),
    ])
    def test_the_turnover_threshold_is_strict(self, turnover, expect):
        out = self._run(turnover=turnover)
        has_decay = any("ts_decay_linear" in d for d in out)
        assert has_decay is expect, (
            f"换手={turnover} 时 {'' if expect else '不'}应当加平滑变体，"
            f"实际 {'加了' if has_decay else '没加'} —— `turnover > 3.0` 的边界被改了")

    @pytest.mark.parametrize("oos,expect", [
        (0.3, False),     # 恰好 0.3 → 不触发（`< 0.3` 严格小于）
        (0.29, True),
        (0.5, False),
    ])
    def test_the_oos_sharpe_threshold_is_strict(self, oos, expect):
        out = self._run(oos_sharpe=oos)
        has_rank = any(d.startswith("rank(rank(") or "scale(" in d for d in out)
        assert has_rank is expect, (
            f"OOS Sharpe={oos} 时 {'' if expect else '不'}应当加 rank/scale 变体 —— "
            f"`oos_s < 0.3` 的边界被改了")

    @pytest.mark.parametrize("overfit,expect", [
        (0.5, False),     # 恰好 0.5 → 不触发（`> 0.5` 严格大于）
        (0.51, True),
        (0.4, False),
    ])
    def test_the_overfit_threshold_is_strict(self, overfit, expect):
        out = self._run(overfitting_score=overfit)
        has_mean = any("ts_mean(" in d for d in out)
        assert has_mean is expect, (
            f"过拟合={overfit} 时 {'' if expect else '不'}应当加 ts_mean 变体 —— "
            f"`overfit > 0.5` 的边界被改了")

    def test_targeted_variants_are_deduplicated(self):
        """`if key not in seen:` —— 删 `not` 会让变体列表要么全空要么全重复。"""
        out = self._run(turnover=9.0, oos_sharpe=0.0, overfitting_score=0.9)
        assert out, "三个诊断全部命中却没有产出任何变体"
        assert len(set(out)) == len(out), f"变体列表里有重复：{out}"

    def test_no_diagnosis_means_no_variants(self):
        assert self._run() == [], "三个诊断都没命中却产出了变体"


# ===========================================================================
# G. 组合信号的 IC —— 最容易悄悄变成前视的一段
# ===========================================================================

class TestCombinedIc:

    @staticmethod
    def _pool(n=3):
        return [{"dsl": d} for d in
                ["rank(close)", "rank(volume)", "ts_mean(close,5)"][:n]]

    def test_the_composite_ic_uses_next_day_returns_not_same_day(self):
        """
        `s, r = sig_arr[t], ret_arr[t + 1]` —— `+` 改成 `-` 就是拿
        **同期**（甚至上一期）收益去算 IC：前视泄漏，IC 会漂亮得离谱，
        而整条链路不会有任何报错。

        构造：信号 = 次日收益的完美预测器。正确对齐时 IC = 1；
        错位一天之后 IC 应当掉到接近 0。
        """
        idx = pd.bdate_range("2021-01-04", periods=60)
        cols = [f"S{i}" for i in range(8)]
        rng = np.random.default_rng(0)
        ret = pd.DataFrame(rng.normal(0, 0.02, (60, 8)), index=idx, columns=cols)
        # signal[t] = ret[t+1] —— 完美的次日预测器
        sig = ret.shift(-1).fillna(0.0)

        # 直接验证那段算式本身的对齐语义
        sig_arr = sig.to_numpy(float)
        ret_arr = ret.to_numpy(float)

        def _ic(offset: int) -> float:
            ics = []
            n = min(sig_arr.shape[0] - 1, ret_arr.shape[0] - 1)
            for t in range(n):
                s, r = sig_arr[t], ret_arr[t + offset]
                mask = ~(np.isnan(s) | np.isnan(r))
                if mask.sum() < 5:
                    continue
                rs = np.argsort(np.argsort(s[mask])).astype(float)
                rr = np.argsort(np.argsort(r[mask])).astype(float)
                rs -= rs.mean(); rr -= rr.mean()
                d = np.sqrt((rs ** 2).sum() * (rr ** 2).sum())
                if d > 0:
                    ics.append(float(np.dot(rs, rr) / d))
            return float(np.mean(ics))

        assert _ic(1) == pytest.approx(1.0, abs=1e-9), (
            "对齐到 t+1 时完美预测器的 IC 不是 1 —— 构造前提被破坏")
        assert abs(_ic(-1)) < 0.4, (
            f"错位到 t-1 时 IC 仍有 {_ic(-1):.3f} —— 用例分不开两种对齐")

        # 产品代码用的必须是 t+1
        import inspect
        src = inspect.getsource(W._combine_pool_alphas)
        assert "ret_arr[t + 1]" in src, (
            "组合 IC 不再用 t+1 的前向收益 —— 前视泄漏")

    def test_the_ic_loop_length_leaves_room_for_the_lookahead(self):
        """
        `T = min(sig_arr.shape[0] - 1, ret_arr.shape[0] - 1)` —— 两个 `- 1`
        是给 `ret_arr[t + 1]` 留出的余量。改成 `+ 1` 会让最后一轮
        `ret_arr[T]` 越界 → IndexError（被外层 except 吞掉 → 组合指标
        静默变成 None，前端组合面板永远空）。
        """
        import inspect
        src = inspect.getsource(W._combine_pool_alphas)
        assert "sig_arr.shape[0] - 1" in src and "ret_arr.shape[0] - 1" in src, (
            "IC 循环上限不再为 t+1 留余量")

    def test_returns_are_derived_from_close_when_absent(self):
        """
        `if ret is None and "close" in data:` —— `and` 放宽成 `or` 时，
        已经有 returns 的数据集也会被 `data["close"]` 覆盖；
        而 close 不存在时直接 KeyError。
        """
        ds = _panel(5)
        assert "returns" in ds
        # 有 returns 时必须原样用
        import inspect
        src = inspect.getsource(W._combine_pool_alphas)
        assert 'ret is None and "close" in data' in src, (
            "returns 的派生条件被改了 —— 已有的 returns 可能被 close 派生值覆盖")

    def test_a_pool_smaller_than_two_returns_none(self):
        assert W._combine_pool_alphas([], _panel(6), None) is None
        assert W._combine_pool_alphas(self._pool(1), _panel(6), None) is None

    def test_the_executor_runs_without_revalidating(self):
        """
        `executor = Executor(validate=False)` —— 翻成 True 会对池子里每条
        DSL 再跑一遍校验。这些 DSL 已经在入池前校验过了，重复校验只在
        边界情形下**多抛异常**：结果是候选被静默剔除（那条 warning 记的是
        "信号求值失败"），组合权重在被削过的样本上算出来。
        """
        import inspect
        src = inspect.getsource(W._combine_pool_alphas)
        assert "Executor(validate=False)" in src, (
            "组合求值的 Executor 打开了重复校验 —— 候选会被静默剔除")


# ===========================================================================
# H. Workflow A / B 的 run()
# ===========================================================================

class _StubGp:
    """替身 PopulationEvolver：不跑 GP，直接给一个固定结果。"""

    last_kwargs: dict = {}

    def __init__(self, **kw):
        type(self).last_kwargs = kw

    def run(self, **kw):
        return SimpleNamespace(
            best_dsl="rank(close)",
            metrics={"is_sharpe": 1.0, "oos_sharpe": 0.8,
                     "overfitting_score": 0.1, "is_overfit": False},
            evolution_log=[{"generation": 1, "best_fitness": 0.5},
                           {"generation": 2, "best_fitness": 0.9}],
            pool_top5=[{"dsl": "rank(close)"}],
            best_config={},
            generations_run=2,
        )


@pytest.fixture
def stub_gp(monkeypatch):
    # `alpha_workflows` 在模块顶层就 `from ..gp_engine.population_evolver import
    # PopulationEvolver`，名字已经绑定在本模块上 —— 必须打在 W 上，
    # 打在 population_evolver 模块上是打不中的。
    _StubGp.last_kwargs = {}
    monkeypatch.setattr(W, "PopulationEvolver", _StubGp)
    return _StubGp


class TestWorkflowRun:

    def test_workflow_a_emits_progress_only_when_a_callback_is_given(
            self, stub_gp, monkeypatch):
        """
        `if on_progress is not None:`（两条工作流各一处）—— 改成 `is None`
        会在**没给**回调时去调 `None(text)`（异常被吞，于是给了回调的人
        一条进度都收不到）。前端的实时进度条就是靠这个。
        """
        monkeypatch.setattr(W, "_hypothesis_templates", lambda h: ["rank(close)"])
        wf = W.GenerationWorkflow(pop_size=4, n_generations=1, n_optuna_trials=0,
                                  n_seed_dsls=4, seed=1)
        seen: list = []
        wf.run("动量假设", _panel(7), on_progress=seen.append)
        assert seen, "给了 on_progress 却一条进度都没收到"
        assert any("Workflow A" in t for t in seen)

        wf.run("动量假设", _panel(7))          # 不给回调，不许抛

    def test_the_effective_population_leaves_room_for_the_seeds(
            self, stub_gp, monkeypatch):
        """
        `effective_pop = max(self._pop_size, len(seed_dsls) + 4)`（A、B 各一处）
        —— `+` 改成 `-` 时，种子比 pop_size 多的情况下种群比种子还小，
        用户给的假设会被截掉一部分而不自知。
        """
        monkeypatch.setattr(W, "_hypothesis_templates", lambda h: ["rank(close)"])
        wf = W.GenerationWorkflow(pop_size=4, n_generations=1, n_optuna_trials=0,
                                  n_seed_dsls=10, seed=1)
        res = wf.run("动量假设", _panel(8))
        expect = max(4, len(res.seed_dsls) + 4)
        assert _StubGp.last_kwargs.get("pop_size") == expect, (
            f"{len(res.seed_dsls)} 条种子、pop_size=4 时有效种群是 "
            f"{_StubGp.last_kwargs.get('pop_size')}，应当是 {expect}")

    def test_an_empty_seed_list_falls_back_to_a_random_alpha(
            self, stub_gp, monkeypatch):
        """
        `if not seed_dsls: seed_dsls = [repr(generate_random_alpha())]` ——
        删掉 `not` 会让**有**种子时反而被一条随机 alpha 整个替换掉：
        用户的假设被丢了，而工作流照常返回一个结果。
        """
        monkeypatch.setattr(W, "_generate_diverse_seeds",
                            lambda *a, **kw: ["rank(close)", "rank(volume)"])
        wf = W.GenerationWorkflow(pop_size=4, n_generations=1, n_optuna_trials=0,
                                  n_seed_dsls=4, seed=1)
        res = wf.run("动量假设", _panel(9))
        assert set(res.seed_dsls) == {"rank(close)", "rank(volume)"}, (
            f"传进去的种子被替换成了 {res.seed_dsls} —— "
            f"`if not seed_dsls` 的 not 被删掉了")

    def test_held_out_test_is_reported_truthfully(self, stub_gp, monkeypatch):
        """
        `m["held_out_test"] = True` / `= False` 两条赋值各是一个变异点。
        这个字段说的是"本次汇报的 OOS 是不是 GP 从未见过的那一段"——
        它撒谎，整份结论的可信度就没了。必须**两格都测且互不相同**。
        """
        monkeypatch.setattr(W, "_hypothesis_templates", lambda h: ["rank(close)"])
        wf = W.GenerationWorkflow(pop_size=4, n_generations=1, n_optuna_trials=0,
                                  n_seed_dsls=4, seed=1)

        long_res = wf.run("动量假设", _panel(10, n=300))
        assert long_res.metrics["held_out_test"] is True, (
            "数据足够三段切割时 held_out_test 报了 False")

        short_res = wf.run("动量假设", _panel(11, n=24))
        assert short_res.metrics["held_out_test"] is False, (
            "数据不足以三段切割（无 held-out Test）时 held_out_test 仍报 True —— "
            "这个字段在撒谎")

    def test_the_sample_sufficiency_flag_defaults_to_insufficient_on_failure(
            self, stub_gp, monkeypatch):
        """
        `m.setdefault("insufficient_sample", True)`（回填失败的兜底）——
        翻成 False 会在**回填本身出错**时宣称样本充足。
        """
        monkeypatch.setattr(W, "_hypothesis_templates", lambda h: ["rank(close)"])

        real = W._quick_metrics

        def _boom(dsl, is_data, oos_data):
            # 只让**回填**那一次失败（它带 oos_data）；
            # held-out Test 那一次第三个参数是 None，放行。
            if oos_data is not None:
                raise RuntimeError("模拟：回填失败")
            return real(dsl, is_data, oos_data)

        monkeypatch.setattr(W, "_quick_metrics", _boom)
        wf = W.GenerationWorkflow(pop_size=4, n_generations=1, n_optuna_trials=0,
                                  n_seed_dsls=4, seed=1)
        res = wf.run("动量假设", _panel(12))
        assert res.metrics["insufficient_sample"] is True, (
            "样本充足性回填失败时却宣称样本充足 —— 兜底的 True 被翻了")

    def test_workflow_b_deduplicates_targeted_variants_against_the_seeds(
            self, stub_gp, monkeypatch):
        """
        `if td not in seen_set:` —— 删掉 `not` 会让定向变体只在**已经存在**时
        才追加：所有定向变体都进不去，Workflow B 的诊断补救整个失效，
        而日志里"targeted=N"还照打。
        """
        monkeypatch.setattr(W, "_targeted_mutations",
                            lambda dsl, m: ["rank(rank(close))", "scale(rank(close))"])
        wf = W.OptimizationWorkflow(pop_size=4, n_generations=1,
                                    n_optuna_trials=0, n_mutations=3, seed=1)
        res = wf.run("rank(close)", _panel(13))
        assert any("rank(rank(close))" in d for d in res.seed_dsls), (
            f"定向变体没有进种子表：{res.seed_dsls} —— "
            f"`if td not in seen_set` 的 not 被删掉了")
        assert len(set(res.seed_dsls)) == len(res.seed_dsls), "种子表里有重复"


# ===========================================================================
# I. 结果解释文案
# ===========================================================================

class TestExplanation:

    def test_the_oos_delta_is_final_minus_initial(self, stub_gp, monkeypatch):
        """
        `delta = final_oos - init_oos` —— 改成 `+` 会让"优化后比优化前
        好了多少"变成两者之和：一个从 -0.5 优化到 0.1 的因子会被写成
        "↓0.4"（实际是 ↑0.6）。箭头方向也跟着反。
        """
        monkeypatch.setattr(W, "_targeted_mutations", lambda dsl, m: [])
        wf = W.OptimizationWorkflow(pop_size=4, n_generations=1,
                                    n_optuna_trials=0, n_mutations=3, seed=1)
        text = wf._explain(
            "rank(close)",
            SimpleNamespace(evolution_log=[], best_dsl="rank(close)",
                            generations_run=2, best_config={}, pool_top5=[],
                            metrics={"oos_sharpe": 0.8, "overfitting_score": 0.1}),
            {"oos_sharpe": 0.2},
            ["rank(close)"], [],
        )
        assert "0.2000 → 0.8000" in text, f"起止值写错了：{text}"
        assert "↑0.6000" in text, (
            f"OOS 变化量写成了别的值：{text} —— `final_oos - init_oos` 的算符被改了")

    def test_a_worse_result_is_reported_as_a_drop(self, stub_gp, monkeypatch):
        wf = W.OptimizationWorkflow(pop_size=4, n_generations=1,
                                    n_optuna_trials=0, n_mutations=3, seed=1)
        text = wf._explain(
            "rank(close)",
            SimpleNamespace(evolution_log=[], best_dsl="rank(close)",
                            generations_run=2, best_config={}, pool_top5=[],
                            metrics={"oos_sharpe": 0.2, "overfitting_score": 0.1}),
            {"oos_sharpe": 0.8},
            ["rank(close)"], [],
        )
        assert "↓0.6000" in text, f"退步没有写成 ↓：{text}"

    def test_a_missing_final_oos_omits_the_delta_line(self, stub_gp):
        """
        `if final_oos is not None:` —— 改成 `is None` 会在没有 OOS 时
        拿 None 去做减法 → TypeError，整份解释生成失败。
        """
        wf = W.OptimizationWorkflow(pop_size=4, n_generations=1,
                                    n_optuna_trials=0, n_mutations=3, seed=1)
        text = wf._explain(
            "rank(close)",
            SimpleNamespace(evolution_log=[], best_dsl="rank(close)",
                            generations_run=2, best_config={}, pool_top5=[],
                            metrics={"oos_sharpe": None, "overfitting_score": 0.1}),
            {"oos_sharpe": 0.8},
            ["rank(close)"], [],
        )
        assert "OOS Sharpe:" not in text, f"没有 OOS 却写了变化行：{text}"

    @pytest.mark.parametrize("overfit,word", [
        (0.5, "OK"), (0.51, "WARNING"), (0.9, "WARNING"), (0.1, "OK"),
    ])
    def test_the_overfit_wording_boundary_is_strict(self, stub_gp, overfit, word):
        wf = W.OptimizationWorkflow(pop_size=4, n_generations=1,
                                    n_optuna_trials=0, n_mutations=3, seed=1)
        text = wf._explain(
            "rank(close)",
            SimpleNamespace(evolution_log=[], best_dsl="rank(close)",
                            generations_run=2, best_config={}, pool_top5=[],
                            metrics={"oos_sharpe": 0.5,
                                     "overfitting_score": overfit}),
            {"oos_sharpe": 0.5},
            ["rank(close)"], [],
        )
        assert f"({word})" in text, (
            f"过拟合={overfit} 的文案不是 {word}：{text} —— `> 0.5` 的边界被改了")
