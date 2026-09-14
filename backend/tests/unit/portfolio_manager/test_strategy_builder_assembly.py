"""
portfolio_manager/strategy_builder.py —— 把六道工序收敛成一份可审批的策略配置

**此前零专属测试**（8 个变异点，D 档）。

这个模块不下单，但它产出的 `StrategyConfig` **就是人要审批的那份东西**。
它的全部风险集中在一点：**五个 `try/except` 把每一道工序都圈了起来**。
圈起来本身是对的（一道工序崩了不该让整份配置产不出来），
危险的是"崩了之后还长得像正常结果"：

  - 策略门崩了 → `passed=False`。而"没有证据"与"证据不足"在
    审批人眼里都是"没过门"，**必须靠 `degraded` 留痕区分**。
  - 风控快照崩了 → `risk_report` 是 `{"evaluated": False}`，
    不是空字典 —— 空字典会被前端当成"无风险项"。
  - 无交易带推导崩了 → band 退回 0，换手被**高估**（偏保守，可接受），
    但同样必须留痕。

所以本文件的重心不是"正常路径能跑通"，而是：
**每一条降级路径都留下了可识别的痕迹，且互不覆盖**。
六道工序全部用桩替换，让每一条 except 都能被单独点着。
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

import app.core.portfolio_manager as PM
from app.core.portfolio_manager.strategy_builder import (
    build_strategy_config,
    propose_from_paper_factors,
)

IDX = pd.bdate_range("2022-01-03", periods=60)
COLS = ["AAA", "BBB", "CCC"]


def _panel(with_volume=True, with_sector=False) -> dict:
    rng = np.random.default_rng(4)
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (60, 3)), axis=0),
                         index=IDX, columns=COLS)
    ds = {"close": close}
    if with_volume:
        ds["volume"] = pd.DataFrame(1e6, index=IDX, columns=COLS)
    if with_sector:
        ds["sector"] = pd.DataFrame("tech", index=IDX, columns=COLS)
    return ds


def _sig(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0, 1, (60, 3)), index=IDX, columns=COLS)


# ---------------------------------------------------------------------------
# 六道工序的桩
# ---------------------------------------------------------------------------

class _Book:
    def __init__(self, weights, combo):
        self.weights = weights
        self.combo_weights = combo


class _Verdict:
    def __init__(self, passed=True, payload=None):
        self.passed = passed
        self._payload = payload or {"passed": passed, "sharpe": 1.2}

    def to_dict(self):
        return dict(self._payload)


class _RiskRep:
    def __init__(self, payload=None):
        self._payload = payload or {"gross": 1.0, "violations": []}

    def to_dict(self):
        return dict(self._payload)


@pytest.fixture
def wiring(monkeypatch):
    """
    把 `build_strategy_config` 依赖的六件东西全部换成可控桩，
    并把每一次调用的入参记下来（用于断言"参数确实透传了"）。
    """
    seen: dict = {"marginal": [], "pm": [], "gate": [], "risk": [], "ctx": [],
                  "band": [], "grade": []}

    weights = pd.DataFrame(0.0, index=IDX, columns=COLS)
    weights.iloc[30:, 0] = 0.5          # 制造一次真实换手

    def marginal(signals, dataset, aum, min_improve):
        seen["marginal"].append({"signals": sorted(signals), "aum": aum,
                                 "min_improve": min_improve})
        return types.SimpleNamespace(selected=list(signals))

    class _PMgr:
        def __init__(self, aum, method, cost_params):
            seen["pm"].append({"aum": aum, "method": method,
                               "cost_params": cost_params})

        def build_book(self, signals, prices, volume):
            seen["pm"][-1]["n_signals"] = len(signals)
            seen["pm"][-1]["volume_mean"] = float(np.asarray(volume).mean())
            return _Book(weights, {k: 1.0 / len(signals) for k in signals})

    class _Gate:
        def __init__(self, aum, method):
            seen["gate"].append({"aum": aum, "method": method})

        def evaluate(self, signals, dataset, cost_params=None):
            seen["gate"][-1]["cost_params"] = cost_params
            return _Verdict()

    class _RiskGate:
        def __init__(self, limits):
            seen["risk"].append({"limits": limits})

        def apply(self, w, sectors=None):
            seen["risk"][-1]["sectors_is_none"] = sectors is None
            return w, _RiskRep()

    class _Limits:
        pass

    monkeypatch.setattr(PM, "marginal_factor_selection", marginal, raising=False)
    monkeypatch.setattr(PM, "PortfolioManager", _PMgr, raising=False)
    monkeypatch.setattr(PM, "StrategyGate", _Gate, raising=False)
    monkeypatch.setattr(PM, "PortfolioRiskGate", _RiskGate, raising=False)
    monkeypatch.setattr(PM, "RiskLimits", _Limits, raising=False)

    import app.core.portfolio_manager.strategy_gate as SG
    monkeypatch.setattr(SG, "resolve_cost_params",
                        lambda dataset, aum, cp: cp or {"resolved": True})

    import app.core.lifecycle.promotion_gate as PG
    def grade(v, th=None):
        seen["grade"].append(dict(v))
        return True, {"grade": "A"}
    monkeypatch.setattr(PG, "grade_paper_entry", grade)

    import app.core.trading_context.context as TC

    class _Ctx:
        def __init__(self, aum):
            seen["ctx"].append({"aum": aum})

        def analyze(self, dataset):
            return types.SimpleNamespace(rebalance_band=0.02)

    monkeypatch.setattr(TC, "TradingContext", _Ctx)

    return types.SimpleNamespace(seen=seen, weights=weights, monkeypatch=monkeypatch)


# ===========================================================================
# A. 正常路径 —— 每个字段都来自它该来的地方
# ===========================================================================

class TestHappyPath:

    def test_the_config_is_proposed_and_carries_every_upstream_result(self, wiring):
        cfg = build_strategy_config({"f1": _sig(1), "f2": _sig(2)}, _panel(),
                                    aum=100_000.0, name="策略甲")
        assert cfg.status == "proposed", (
            f"新建配置的状态是 {cfg.status!r} —— 必须是 proposed（等人审批），"
            f"绝不能直接是 active")
        assert sorted(cfg.factors) == ["f1", "f2"]
        assert cfg.name == "策略甲"
        assert cfg.aum == 100_000.0
        assert cfg.method == "ic_weighted"
        assert cfg.passed is True
        assert cfg.verdict["passed"] is True
        assert cfg.risk_report == {"gross": 1.0, "violations": []}
        assert cfg.no_trade_band == pytest.approx(0.02)
        assert "degraded" not in cfg.verdict, (
            f"一切正常却留下了降级痕迹：{cfg.verdict.get('degraded')}")

    def test_the_aum_and_method_reach_every_stage(self, wiring):
        build_strategy_config({"f1": _sig(1), "f2": _sig(2)}, _panel(),
                              aum=250_000.0, method="equal_weight")
        s = wiring.seen
        assert s["marginal"][0]["aum"] == 250_000.0
        assert s["pm"][0] == dict(s["pm"][0], aum=250_000.0, method="equal_weight")
        assert s["gate"][0]["aum"] == 250_000.0 and s["gate"][0]["method"] == "equal_weight"
        assert s["ctx"][0]["aum"] == 250_000.0

    def test_resolved_cost_params_are_shared_by_the_book_and_the_gate(self, wiring):
        """
        `cost_params = resolve_cost_params(...)` 的结果必须**同时**
        喂给合成账本与策略门。只喂一边会让"门评过的成本"与
        "账本用的成本"不是同一套 —— 审批人看到的证据与实际要交易的东西对不上。
        """
        build_strategy_config({"f1": _sig(1)}, _panel(), aum=100_000.0)
        s = wiring.seen
        assert s["pm"][0]["cost_params"] == {"resolved": True}
        assert s["gate"][0]["cost_params"] == {"resolved": True}
        assert s["pm"][0]["cost_params"] == s["gate"][0]["cost_params"]

    def test_an_explicit_cost_params_is_passed_through_the_resolver(self, wiring):
        build_strategy_config({"f1": _sig(1)}, _panel(), aum=1e5,
                              cost_params={"mine": 1})
        assert wiring.seen["pm"][0]["cost_params"] == {"mine": 1}

    def test_turnover_is_measured_after_the_no_trade_band(self, wiring):
        """
        `turnover = annualized_turnover(apply_no_trade_band(weights, band))`

        两个调用被拆开（先算换手再套 band）会让配置里记录的换手
        **高于**实际要发生的换手 —— 成本预算因此偏保守。
        这里用一个大到吃掉全部调仓的 band 来区分：
        套带之后换手必须是 0。
        """
        import app.core.trading_context.context as TC

        class _WideBand(TC.TradingContext):
            pass

        wiring.monkeypatch.setattr(
            TC, "TradingContext",
            lambda aum: types.SimpleNamespace(
                analyze=lambda ds: types.SimpleNamespace(rebalance_band=10.0)))

        cfg = build_strategy_config({"f1": _sig(1)}, _panel(), aum=1e5)
        assert cfg.no_trade_band == 10.0
        assert cfg.turnover_ann == pytest.approx(0.0), (
            f"band=10 已经大到不可能有任何调仓，换手却是 {cfg.turnover_ann} —— "
            f"换手是在套无交易带**之前**算的")

    def test_combo_weights_are_floats_keyed_by_factor(self, wiring):
        cfg = build_strategy_config({"f1": _sig(1), "f2": _sig(2)}, _panel(), aum=1e5)
        assert set(cfg.combo_weights) == {"f1", "f2"}
        assert all(isinstance(v, float) for v in cfg.combo_weights.values())

    def test_a_missing_combo_weights_degrades_to_an_empty_dict(self, wiring):
        """`(book.combo_weights or {})` —— None 时不能让 `.items()` 抛。"""
        class _PMgr:
            def __init__(self, **k):
                pass

            def build_book(self, signals, prices, volume):
                return _Book(wiring.weights, None)

        wiring.monkeypatch.setattr(PM, "PortfolioManager", _PMgr, raising=False)
        cfg = build_strategy_config({"f1": _sig(1)}, _panel(), aum=1e5)
        assert cfg.combo_weights == {}


# ===========================================================================
# B. 缺省数据补齐
# ===========================================================================

class TestDatasetDefaults:

    def test_a_missing_volume_field_is_filled_with_a_flat_one_million(self, wiring):
        """
        `volume = pd.DataFrame(1e6, ...)` —— 缺成交量时用一个**正的**
        常数兜底。填 0 会让下游所有 ADV 相关的成本/容量计算归零，
        于是"任意大的单子都没有冲击成本"。
        """
        build_strategy_config({"f1": _sig(1)}, _panel(with_volume=False), aum=1e5)
        assert wiring.seen["pm"][0]["volume_mean"] == pytest.approx(1e6), (
            "缺 volume 时的兜底值不是 1e6")

    def test_a_present_volume_field_is_used_as_is(self, wiring):
        ds = _panel()
        ds["volume"] = pd.DataFrame(7e5, index=IDX, columns=COLS)
        build_strategy_config({"f1": _sig(1)}, ds, aum=1e5)
        assert wiring.seen["pm"][0]["volume_mean"] == pytest.approx(7e5), (
            "已有的 volume 被兜底值覆盖了")

    def test_sectors_are_taken_from_the_last_row_when_present(self, wiring):
        build_strategy_config({"f1": _sig(1)}, _panel(with_sector=True), aum=1e5)
        assert wiring.seen["risk"][0]["sectors_is_none"] is False, (
            "数据里有 sector 却没有传给风控门 —— 行业集中度限额失效")

    def test_sectors_are_none_when_the_field_is_absent(self, wiring):
        build_strategy_config({"f1": _sig(1)}, _panel(), aum=1e5)
        assert wiring.seen["risk"][0]["sectors_is_none"] is True

    def test_supplied_risk_limits_win_over_the_default(self, wiring):
        """`limits = risk_limits or RiskLimits()` —— 传了就不能被默认值顶掉。"""
        mine = object()
        build_strategy_config({"f1": _sig(1)}, _panel(), aum=1e5, risk_limits=mine)
        assert wiring.seen["risk"][0]["limits"] is mine

    def test_the_default_risk_limits_are_constructed_when_none_supplied(self, wiring):
        build_strategy_config({"f1": _sig(1)}, _panel(), aum=1e5)
        assert wiring.seen["risk"][0]["limits"] is not None


# ===========================================================================
# C. 边际准入的触发条件
# ===========================================================================

class TestMarginalSelection:

    def test_a_single_factor_skips_marginal_selection(self, wiring):
        """
        `if len(signals) > 1:` —— `>` 翻成 `>=` 会让单因子也去跑
        边际准入。单因子的"边际改进"没有比较基准，
        要么抛、要么返回空 selected，白白多一次昂贵的回测。
        """
        build_strategy_config({"only": _sig(1)}, _panel(), aum=1e5)
        assert wiring.seen["marginal"] == [], (
            "只有一个因子却跑了边际准入 —— `len(signals) > 1` 被翻成了 `>=`")

    def test_two_factors_do_run_marginal_selection(self, wiring):
        build_strategy_config({"f1": _sig(1), "f2": _sig(2)}, _panel(), aum=1e5)
        assert len(wiring.seen["marginal"]) == 1

    def test_the_min_improve_threshold_is_forwarded(self, wiring):
        build_strategy_config({"f1": _sig(1), "f2": _sig(2)}, _panel(), aum=1e5,
                              marginal_min_improve=0.25)
        assert wiring.seen["marginal"][0]["min_improve"] == 0.25

    def test_the_default_min_improve_is_five_percent(self, wiring):
        build_strategy_config({"f1": _sig(1), "f2": _sig(2)}, _panel(), aum=1e5)
        assert wiring.seen["marginal"][0]["min_improve"] == 0.05

    def test_the_selection_actually_narrows_the_factor_set(self, wiring):
        def only_f1(signals, dataset, aum, min_improve):
            return types.SimpleNamespace(selected=["f1"])

        wiring.monkeypatch.setattr(PM, "marginal_factor_selection", only_f1,
                                   raising=False)
        cfg = build_strategy_config({"f1": _sig(1), "f2": _sig(2)}, _panel(), aum=1e5)
        assert cfg.factors == ["f1"], (
            f"边际准入选了 f1，配置里却是 {cfg.factors} —— 筛选结果没有生效")

    def test_an_empty_selection_keeps_all_factors_rather_than_none(self, wiring):
        """
        `if sel.selected:` —— 守卫被删会让"一个都没选中"变成
        **组合里一个因子都没有**，账本直接空掉。
        保守做法是沿用全部，这条把它钉下来。
        """
        wiring.monkeypatch.setattr(
            PM, "marginal_factor_selection",
            lambda *a, **k: types.SimpleNamespace(selected=[]), raising=False)
        cfg = build_strategy_config({"f1": _sig(1), "f2": _sig(2)}, _panel(), aum=1e5)
        assert sorted(cfg.factors) == ["f1", "f2"], (
            "边际准入返回空集之后因子被清空了 —— `if sel.selected:` 守卫没了")

    def test_the_original_signal_dict_is_not_mutated(self, wiring):
        """`signals = dict(factor_signals)` —— 少了这层拷贝会改掉调用方的字典。"""
        original = {"f1": _sig(1), "f2": _sig(2)}
        wiring.monkeypatch.setattr(
            PM, "marginal_factor_selection",
            lambda *a, **k: types.SimpleNamespace(selected=["f1"]), raising=False)
        build_strategy_config(original, _panel(), aum=1e5)
        assert sorted(original) == ["f1", "f2"], "调用方传入的信号字典被改掉了"


# ===========================================================================
# D. 五条降级路径 —— 本文件的重心
# ===========================================================================

def _degraded(cfg) -> list:
    return cfg.verdict.get("degraded", [])


class TestDegradationTrail:

    def test_a_failing_marginal_selection_is_recorded_and_all_factors_kept(self,
                                                                           wiring):
        def boom(*a, **k):
            raise RuntimeError("边际回测炸了")

        wiring.monkeypatch.setattr(PM, "marginal_factor_selection", boom,
                                   raising=False)
        cfg = build_strategy_config({"f1": _sig(1), "f2": _sig(2)}, _panel(), aum=1e5)
        assert sorted(cfg.factors) == ["f1", "f2"], "筛选失败后因子集合被改动了"
        trail = _degraded(cfg)
        assert any("marginal_selection_failed" in t for t in trail), (
            f"边际筛选失败没有留痕：{trail} —— "
            f"'筛选失败' 与 '筛完就是全选' 将无法区分")
        assert any("边际回测炸了" in t for t in trail), "留痕里丢了原始异常信息"

    def test_a_failing_strategy_gate_is_marked_as_unevaluated_not_as_failed(self,
                                                                            wiring):
        """
        这是本模块最要紧的一条区分：
        **"没有证据"(evaluated=False) ≠ "证据不足"(passed=False)**。
        `verdict = {"gate_error": ..., "evaluated": False}` 被简化成
        `verdict = {}` 会让审批人以为门评过了只是没通过。
        """
        class _BoomGate:
            def __init__(self, **k):
                pass

            def evaluate(self, *a, **k):
                raise RuntimeError("门评算不出来")

        wiring.monkeypatch.setattr(PM, "StrategyGate", _BoomGate, raising=False)
        cfg = build_strategy_config({"f1": _sig(1)}, _panel(), aum=1e5)

        assert cfg.passed is False
        assert cfg.verdict.get("evaluated") is False, (
            "策略门崩溃后 verdict 里没有 evaluated=False —— "
            "无法与'评过但没通过'区分")
        assert "门评算不出来" in cfg.verdict.get("gate_error", ""), "原始异常没有留下"
        assert any("strategy_gate_failed" in t for t in _degraded(cfg))

    def test_a_failing_risk_snapshot_yields_an_unevaluated_marker_not_an_empty_report(
            self, wiring):
        """
        风控报告是 `{}` 时前端会显示"无违规"。
        必须是 `{"evaluated": False}` 才能显示"没有风险证据"。
        """
        class _BoomRisk:
            def __init__(self, limits):
                pass

            def apply(self, *a, **k):
                raise RuntimeError("风控快照炸了")

        wiring.monkeypatch.setattr(PM, "PortfolioRiskGate", _BoomRisk, raising=False)
        cfg = build_strategy_config({"f1": _sig(1)}, _panel(), aum=1e5)
        assert cfg.risk_report != {}, "风控失败后报告是空字典 —— 会被读成'无风险项'"
        assert cfg.risk_report.get("evaluated") is False
        assert "风控快照炸了" in cfg.risk_report.get("risk_error", "")
        assert any("risk_snapshot_failed" in t for t in _degraded(cfg))

    def test_a_failing_band_derivation_falls_back_to_zero_and_is_recorded(self,
                                                                          wiring):
        """
        `band = 0.0` 的兜底方向是**保守**的（换手被高估）。
        兜底值被改成一个大数会让换手被低估 —— 成本预算偏乐观，
        这是唯一一个"往危险方向"错的兜底，所以值钉死。
        """
        import app.core.trading_context.context as TC

        def boom(aum):
            raise RuntimeError("上下文推导炸了")

        wiring.monkeypatch.setattr(TC, "TradingContext", boom)
        cfg = build_strategy_config({"f1": _sig(1)}, _panel(), aum=1e5)
        assert cfg.no_trade_band == 0.0, (
            f"无交易带推导失败后没有退回 0，而是 {cfg.no_trade_band} —— "
            f"非零兜底会让换手被低估")
        assert cfg.turnover_ann > 0.0, "band=0 时换手应当是原始换手"
        assert any("band_derivation_failed" in t for t in _degraded(cfg))

    def test_a_failing_paper_grade_becomes_unknown_and_is_recorded(self, wiring):
        import app.core.lifecycle.promotion_gate as PG

        def boom(v, th=None):
            raise RuntimeError("分级炸了")

        wiring.monkeypatch.setattr(PG, "grade_paper_entry", boom)
        cfg = build_strategy_config({"f1": _sig(1)}, _panel(), aum=1e5)
        assert cfg.verdict["paper_grade"] == "unknown", (
            "分级失败后没有标 unknown —— 会被当成某个真实等级")
        assert any("paper_grading_failed" in t for t in _degraded(cfg))

    def test_multiple_failures_all_appear_in_the_trail(self, wiring):
        """
        `degraded.append(...)` 五处共用一个列表。
        任一处写成赋值（`degraded = [...]`）会**覆盖**掉前面的记录，
        审批人只会看到最后一个坏掉的工序。
        """
        import app.core.trading_context.context as TC
        import app.core.lifecycle.promotion_gate as PG

        class _BoomGate:
            def __init__(self, **k):
                pass

            def evaluate(self, *a, **k):
                raise RuntimeError("g")

        class _BoomRisk:
            def __init__(self, limits):
                pass

            def apply(self, *a, **k):
                raise RuntimeError("r")

        wiring.monkeypatch.setattr(PM, "marginal_factor_selection",
                                   lambda *a, **k: (_ for _ in ()).throw(
                                       RuntimeError("m")), raising=False)
        wiring.monkeypatch.setattr(PM, "StrategyGate", _BoomGate, raising=False)
        wiring.monkeypatch.setattr(PM, "PortfolioRiskGate", _BoomRisk, raising=False)
        wiring.monkeypatch.setattr(PG, "grade_paper_entry",
                                   lambda v, th=None: (_ for _ in ()).throw(
                                       RuntimeError("p")))
        wiring.monkeypatch.setattr(TC, "TradingContext",
                                   lambda aum: (_ for _ in ()).throw(
                                       RuntimeError("b")))

        cfg = build_strategy_config({"f1": _sig(1), "f2": _sig(2)}, _panel(), aum=1e5)
        trail = _degraded(cfg)
        kinds = {t.split(":")[0] for t in trail}
        assert kinds == {"marginal_selection_failed", "strategy_gate_failed",
                         "paper_grading_failed", "risk_snapshot_failed",
                         "band_derivation_failed"}, (
            f"五道工序全崩，降级痕迹只剩 {sorted(kinds)} —— "
            f"append 被写成了赋值，前面的记录被覆盖")
        assert len(trail) == 5

    def test_a_clean_run_leaves_no_degraded_key_at_all(self, wiring):
        """
        `if degraded:` —— 守卫被删会在一切正常时也写进一个空的
        `degraded: []`，前端一看有这个键就会打降级标记。
        """
        cfg = build_strategy_config({"f1": _sig(1), "f2": _sig(2)}, _panel(), aum=1e5)
        assert "degraded" not in cfg.verdict


# ===========================================================================
# E. 进 PAPER 分级
# ===========================================================================

class TestPaperGrading:

    def test_the_grade_and_detail_are_both_stored(self, wiring):
        import app.core.lifecycle.promotion_gate as PG
        wiring.monkeypatch.setattr(
            PG, "grade_paper_entry",
            lambda v, th=None: (True, {"grade": "B", "why": "观察期"}))
        cfg = build_strategy_config({"f1": _sig(1)}, _panel(), aum=1e5)
        assert cfg.verdict["paper_grade"] == "B"
        assert cfg.verdict["paper_entry"] == {"grade": "B", "why": "观察期"}

    def test_a_blocked_entry_is_flagged(self, wiring):
        """
        `if not allowed: verdict["paper_entry_blocked"] = True`
        —— `not` 被删会把**允许进 PAPER**的配置标成被拦截，
        反过来则让被拦截的配置看起来可以进。
        """
        import app.core.lifecycle.promotion_gate as PG
        wiring.monkeypatch.setattr(
            PG, "grade_paper_entry", lambda v, th=None: (False, {"grade": "C"}))
        cfg = build_strategy_config({"f1": _sig(1)}, _panel(), aum=1e5)
        assert cfg.verdict.get("paper_entry_blocked") is True

    def test_an_allowed_entry_carries_no_blocked_flag(self, wiring):
        import app.core.lifecycle.promotion_gate as PG
        wiring.monkeypatch.setattr(
            PG, "grade_paper_entry", lambda v, th=None: (True, {"grade": "A"}))
        cfg = build_strategy_config({"f1": _sig(1)}, _panel(), aum=1e5)
        assert "paper_entry_blocked" not in cfg.verdict, (
            "允许进 PAPER 的配置被标了 blocked —— `if not allowed` 的 not 被删了")

    def test_the_grader_sees_the_strategy_gate_verdict(self, wiring):
        """分级的输入必须是策略门的结论，拿错对象会让等级与证据脱节。"""
        build_strategy_config({"f1": _sig(1)}, _panel(), aum=1e5)
        assert wiring.seen["grade"][0]["passed"] is True
        assert wiring.seen["grade"][0]["sharpe"] == 1.2


# ===========================================================================
# F. 从 AlphaStore 取因子
# ===========================================================================

class _Rec:
    def __init__(self, rid, status, dsl="rank(close)"):
        self.id = rid
        self.status = status
        self.dsl = dsl


class _Store:
    def __init__(self, recs):
        self._recs = recs
        self.last_limit = None

    def query(self, limit=None):
        self.last_limit = limit
        return list(self._recs)


@pytest.fixture
def alpha_wiring(wiring, monkeypatch):
    """信号生成链路（Executor / SignalProcessor）换成恒等桩。"""
    import app.core.alpha_engine.dsl_executor as DE
    import app.core.alpha_engine.signal_processor as SP

    seen = {"cfg": [], "dsl": []}

    class _Exec:
        def __init__(self, validate=True):
            seen.setdefault("validate", []).append(validate)

        def run_expr(self, dsl, dataset):
            seen["dsl"].append(dsl)
            if dsl == "BOOM":
                raise ValueError("DSL 跑不通")
            return _sig(len(seen["dsl"]))

    class _SP:
        def __init__(self, cfg):
            seen["cfg"].append(cfg)

        def process(self, raw):
            return raw

    monkeypatch.setattr(DE, "Executor", _Exec)
    monkeypatch.setattr(SP, "SignalProcessor", _SP)
    return types.SimpleNamespace(seen=seen, wiring=wiring)


class TestProposeFromPaperFactors:

    def test_only_paper_active_and_decaying_factors_are_included(self, alpha_wiring):
        """
        `coerce_status(rec.status) in (PAPER, ACTIVE, DECAYING)`

        名单里多一个状态（比如 CANDIDATE）会让**未经验证的因子
        直接进入待审批的策略**；少一个（比如 DECAYING）会让正在
        衰减但仍持仓的因子从组合里凭空消失。
        """
        store = _Store([
            _Rec(1, "paper"), _Rec(2, "active"), _Rec(3, "decaying"),
            _Rec(4, "candidate"), _Rec(5, "validated"), _Rec(6, "retired"),
        ])
        cfg = propose_from_paper_factors(store, _panel(), aum=1e5)
        assert sorted(cfg.factors) == ["1", "2", "3"], (
            f"入选因子是 {sorted(cfg.factors)}，应当只有 paper/active/decaying —— "
            f"状态名单被改了")

    def test_a_factor_whose_dsl_fails_is_skipped_not_fatal(self, alpha_wiring):
        """
        `except ... continue` —— 源码注释写明"静默 continue 会让因子
        从组合里消失而无人知晓"。这里钉住：坏因子不能带走好因子。
        """
        store = _Store([_Rec(1, "paper", dsl="BOOM"), _Rec(2, "paper")])
        cfg = propose_from_paper_factors(store, _panel(), aum=1e5)
        assert cfg.factors == ["2"], f"坏因子没被跳过或把好因子带走了：{cfg.factors}"

    def test_no_usable_factor_returns_none_rather_than_an_empty_strategy(self,
                                                                         alpha_wiring):
        """
        `if not signals: return None` —— `not` 被删会在一个因子都没有时
        继续往下构建，产出一份**空组合**的待审批策略。
        """
        assert propose_from_paper_factors(_Store([]), _panel(), aum=1e5) is None
        assert propose_from_paper_factors(
            _Store([_Rec(1, "candidate")]), _panel(), aum=1e5) is None
        assert propose_from_paper_factors(
            _Store([_Rec(1, "paper", dsl="BOOM")]), _panel(), aum=1e5) is None

    def test_the_signal_config_matches_the_run_portfolio_caliber(self, alpha_wiring):
        """
        `SimulationConfig(delay=1, decay_window=0, truncation_min_q=0.05,
                          truncation_max_q=0.95)`

        `delay=1` 是**防前视**的那一格 —— 改成 0 意味着用当天信号
        交易当天收盘，整份策略的回测结论全部作废。
        另外三项必须与 `run_portfolio` 同口径，否则审批时看到的
        信号与实际交易的信号不是同一个。
        """
        store = _Store([_Rec(1, "paper")])
        propose_from_paper_factors(store, _panel(), aum=1e5)
        cfg = alpha_wiring.seen["cfg"][0]
        assert cfg.delay == 1, f"delay={cfg.delay} —— 不是 1 就有前视风险"
        assert cfg.decay_window == 0
        assert cfg.truncation_min_q == 0.05
        assert cfg.truncation_max_q == 0.95

    def test_the_executor_skips_validation_for_stored_dsls(self, alpha_wiring):
        """`Executor(validate=False)` —— 库里的 DSL 已经验过，重复验证只是浪费。"""
        propose_from_paper_factors(_Store([_Rec(1, "paper")]), _panel(), aum=1e5)
        assert alpha_wiring.seen["validate"] == [False]

    def test_the_store_query_is_bounded(self, alpha_wiring):
        """`query(limit=500)` —— 去掉上限会在因子库变大后拖垮构建。"""
        store = _Store([_Rec(1, "paper")])
        propose_from_paper_factors(store, _panel(), aum=1e5)
        assert store.last_limit == 500

    def test_factor_keys_are_the_record_ids_as_strings(self, alpha_wiring):
        """
        `signals[str(rec.id)]` —— 不转字符串会让 combo_weights 的键
        是 int，序列化进 JSON 再读回来就变成了字符串，前后对不上。
        """
        cfg = propose_from_paper_factors(_Store([_Rec(7, "paper")]), _panel(),
                                         aum=1e5)
        assert cfg.factors == ["7"]
        assert all(isinstance(k, str) for k in cfg.combo_weights)

    def test_the_method_and_name_reach_the_built_config(self, alpha_wiring):
        cfg = propose_from_paper_factors(_Store([_Rec(1, "paper")]), _panel(),
                                         aum=1e5, method="equal_weight",
                                         name="纸面组合")
        assert cfg.method == "equal_weight"
        assert cfg.name == "纸面组合"
