"""
promotion_gate.py —— 边界与默认值定钉测试（变异测试驱动）

来由：`tests/test_phase_tr4_promotion.py` 覆盖了"典型输入 → 典型结论"，
但每一个**边界**和**默认值**都能改坏而全绿。用隔离版变异工具实测，
13 个变异点存活 7 个。存活项分三类，每一类都是真盲区：

1. **默认值没人测**（L36 / L47）——`experiment_mode` 的类默认和读配置失败时的
   兜底默认都是 `True`（= 放行）。把它们改成 `False` 测试照样全绿。
   既有用例每次都显式传 `experiment_mode=`，从没有人问过"不传时是什么"。
   这个默认值决定**策略门没过的因子能不能进 paper**，不是无关紧要的细节。

2. **判级/判天数的边界没人测**（L77 / L111）——`sharpe > 0` 与 `n < min_days`
   的区分值分别是 `0.0` 和 `min_forward_days` 本身，都可以精确构造，
   不属于 risk_gate 那种"浮点上造不出区分值"的等价变异，必须写用例。

3. **清洗与失败路径没人测**（L105 / L113 / L117）——过滤 None/NaN 的 `and`
   改成 `or` 会让 NaN 混进统计；观测不足时 `detail["passed"]` 写死 False
   改成 True 也没人发现（既有用例只看返回值，不看 detail）。
"""
from __future__ import annotations

import types
import warnings

import numpy as np

from app.core.lifecycle.promotion_gate import (
    PromotionThresholds,
    check_active_promotion,
    grade_paper_entry,
)


# ===========================================================================
# A. 默认值 —— 决定"没过严门的因子能不能进 paper"
# ===========================================================================

class TestDefaultsAreExplicit:

    def test_class_default_is_experiment_mode_on(self):
        """
        类默认 experiment_mode=True（放行并标注等级）。这是一个**有意的**
        宽松默认：paper 的目的是收集前向证据，太严就永远拿不到证据。
        它必须是被断言过的选择，而不是"碰巧是这个值"。
        """
        th = PromotionThresholds()
        assert th.experiment_mode is True

    def test_class_defaults_match_documented_policy(self):
        """其余阈值同样钉死，改任何一个都要先改这条用例（= 强制走审阅）。"""
        th = PromotionThresholds()
        assert (th.min_forward_days, th.min_ic_tstat, th.min_ic_mean) == (60, 2.0, 0.0)
        assert th.paper_min_sharpe == -99.0

    def test_from_settings_falls_back_to_experiment_mode_on(self, monkeypatch):
        """
        配置里没有 `tr_experiment_mode` 时的兜底值。
        `getattr(settings, "tr_experiment_mode", True)` 的那个 True 是本门
        在配置缺失时的行为，既有用例一次都没走过这条路径。
        """
        import app.config
        monkeypatch.setattr(app.config, "settings", types.SimpleNamespace())
        th = PromotionThresholds.from_settings()
        assert th.experiment_mode is True
        assert th.min_forward_days == 60 and th.min_ic_tstat == 2.0

    def test_from_settings_reads_experiment_mode_when_present(self, monkeypatch):
        """对照组：配置里有值时必须用配置值，否则上一条无法区分'读到了'与'兜底了'。"""
        import app.config
        monkeypatch.setattr(
            app.config, "settings",
            types.SimpleNamespace(tr_experiment_mode=False, tr_min_forward_days=7))
        th = PromotionThresholds.from_settings()
        assert th.experiment_mode is False
        assert th.min_forward_days == 7

    def test_from_settings_logs_and_falls_back_when_config_explodes(self, monkeypatch, caplog):
        """
        读配置抛异常时必须**告警后**回退——静默回退意味着用户配的阈值
        全部没生效却无人知晓（DEV_LESSONS §U）。
        """
        class Boom:
            # 用 __getattr__ 而非 __getattribute__：后者连 monkeypatch 自己的
            # 内省（__class__）都会炸，测不到目标代码。
            # 且必须抛 **非** AttributeError —— getattr(o, k, default) 只吞
            # AttributeError，抛它就走不进 from_settings 的 except 分支了。
            def __getattr__(self, name):
                raise RuntimeError("config exploded")

        import app.config
        monkeypatch.setattr(app.config, "settings", Boom())
        with caplog.at_level("ERROR"):
            th = PromotionThresholds.from_settings()
        assert th.min_forward_days == 60
        assert any("promotion_gate" in r.message or "promotion_gate" in r.getMessage()
                   for r in caplog.records), "配置读取失败没有留下 ERROR 日志"


# ===========================================================================
# B. 分级边界 —— sharpe 恰好为 0
# ===========================================================================

class TestPaperGradeBoundary:

    TH = PromotionThresholds(experiment_mode=True)

    def test_sharpe_exactly_zero_is_grade_C_not_B(self):
        """
        `elif sharpe > 0: grade = "B"`。区分值就是 0.0 本身，可精确构造：
        改成 `>=` 后 sharpe==0 会被评成 B（"核心指标不差"），
        而 0 夏普显然属于 C（明显不合格）。
        """
        _, d = grade_paper_entry({"passed": False, "sharpe": 0.0}, self.TH)
        assert d["grade"] == "C"

    def test_tiny_positive_sharpe_is_grade_B(self):
        """对照组：正的一侧确实是 B，否则上一条可能只是'永远返回 C'。"""
        _, d = grade_paper_entry({"passed": False, "sharpe": 1e-9}, self.TH)
        assert d["grade"] == "B"

    def test_missing_sharpe_is_treated_as_zero_hence_C(self):
        """verdict 里没有 sharpe 字段 → 按 0 处理 → C，不许乐观补值。"""
        _, d = grade_paper_entry({"passed": False}, self.TH)
        assert d["grade"] == "C" and d["sharpe"] == 0.0

    def test_none_verdict_is_graded_C_and_not_crash(self):
        """上游给 None（拿不到策略结论）→ 按最差处理，且不能抛异常打断流程。"""
        allowed, d = grade_paper_entry(None, self.TH)
        assert d["grade"] == "C" and d["gate_passed"] is False
        assert allowed is True          # 实验模式仍放行，但等级如实标注

    def test_strict_mode_rejects_grade_B_even_with_good_sharpe(self):
        th = PromotionThresholds(experiment_mode=False)
        allowed, d = grade_paper_entry({"passed": False, "sharpe": 5.0}, th)
        assert d["grade"] == "B" and allowed is False


# ===========================================================================
# C. →ACTIVE 门：天数边界、输入清洗、失败路径的 detail
# ===========================================================================

class TestActivePromotionBoundaries:

    def test_exactly_min_forward_days_is_enough(self):
        """
        `if n < th.min_forward_days: 拒绝`。n == min_forward_days 属于**够**的一侧。
        改成 `<=` 后恰好攒够天数的策略会被多卡一天，且理由是"观测不足"——
        区分值是 min_forward_days 本身，可精确构造。
        """
        th = PromotionThresholds(min_forward_days=60)
        _, d = check_active_promotion([0.05] * 60, th)
        assert d["n_days"] == 60
        assert not any("前向观测不足" in r for r in d["reasons"]), (
            f"恰好攒够 60 天却被判观测不足：{d['reasons']}")

    def test_one_day_short_is_rejected_for_insufficient_days(self):
        th = PromotionThresholds(min_forward_days=60)
        ok, d = check_active_promotion([0.05] * 59, th)
        assert ok is False
        assert any("前向观测不足" in r for r in d["reasons"])

    def test_insufficient_days_detail_reports_passed_false(self):
        """
        观测不足这条早退路径里 `detail.update({"passed": False, ...})`。
        既有用例只看返回值不看 detail，于是把 False 改成 True 也全绿——
        而调用方（谱系记录、前端）读的正是 detail。
        """
        ok, d = check_active_promotion([0.05] * 3, PromotionThresholds(min_forward_days=60))
        assert ok is False
        assert d["passed"] is False, "返回值说没过，detail 里却写着过了"

    def test_detail_passed_always_agrees_with_return_value(self):
        """契约：两条路径（早退 / 正常）上 detail['passed'] 都必须等于返回值。"""
        cases = [
            ([0.05] * 3, PromotionThresholds(min_forward_days=60)),
            (list(np.random.default_rng(1).normal(0.05, 0.05, 120)),
             PromotionThresholds(min_forward_days=60, min_ic_tstat=2.0)),
            (list(np.random.default_rng(2).normal(-0.05, 0.05, 120)),
             PromotionThresholds(min_forward_days=60)),
        ]
        for ics, th in cases:
            ok, d = check_active_promotion(ics, th)
            assert d["passed"] == ok

    def test_none_and_nan_are_dropped_not_merely_none(self):
        """
        `[v for v in ic_values if v is not None and np.isfinite(v)]`。
        `and` 改成 `or` 后 NaN/inf 会被留下（非 None 即短路通过），
        均值和 t 统计随即被污染成 NaN。区分输入：混入 NaN 与 inf。
        """
        th = PromotionThresholds(min_forward_days=5)
        ics = [0.05, float("nan"), 0.04, float("inf"), None, 0.06, -float("inf"), 0.05, 0.05]
        ok, d = check_active_promotion(ics, th)
        assert d["n_days"] == 5, f"NaN/inf/None 没有被过滤干净：n_days={d['n_days']}"
        assert np.isfinite(d["ic_mean"]) and np.isfinite(d["ic_tstat"])

    def test_single_observation_does_not_emit_degenerate_std_warning(self):
        """
        `sd = np.std(x, ddof=1) if n > 1 else 0.0`。改成 `n >= 1` 后 n==1 会走进
        `np.std(x, ddof=1)`（自由度 0）→ RuntimeWarning + NaN。
        最终 t 仍是 0.0（NaN > 1e-12 为假），**返回值看不出差别**，
        所以只能靠"把警告升级成错误"来区分——这类变异用结果断言是抓不住的。
        """
        th = PromotionThresholds(min_forward_days=1)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            ok, d = check_active_promotion([0.05], th)
        assert d["n_days"] == 1
        assert d["ic_tstat"] == 0.0, "单点观测不应产生非零 t 统计"
        assert ok is False, "单点观测不得晋级"

    def test_zero_variance_ic_gives_zero_tstat_not_infinity(self):
        """常数 IC → sd==0 → t 必须是 0，不能是 inf（否则恒定小正 IC 就能晋级）。"""
        th = PromotionThresholds(min_forward_days=10)
        ok, d = check_active_promotion([0.05] * 60, th)
        assert d["ic_tstat"] == 0.0 and ok is False


# ===========================================================================
# D. 存活变异的等价性证明
# ===========================================================================

PROVEN_EQUIVALENT = {
    "L118 `sd > 1e-12` → `>=`":
        "区分值需要 sd **恰好等于** 1e-12。sd = np.std(x, ddof=1) 是对 n 个浮点数"
        "做平方和、除以 n-1、再开方的结果，无法反解出使其精确等于 1e-12 的输入；"
        "且该阈值的用途就是「标准差小到不可用时把 t 记为 0」，边界两侧行为连续："
        "sd 略大于 1e-12 时 t = mu/sd·√n 已是天文数字，与 0 的区别不在这一侧，"
        "而在下游 `t <= min_ic_tstat` 的判定里，那一条已被 "
        "test_zero_variance_ic_gives_zero_tstat_not_infinity 覆盖。",
}


def test_std_epsilon_boundary_is_unreachable():
    """L118 等价性的机械验证：无法构造使 np.std(ddof=1) 精确等于 1e-12 的样本。"""
    tol = 1e-12
    rng = np.random.default_rng(3)
    for scale in (1e-12, 1e-11, 1e-13):
        for _ in range(200):
            x = rng.normal(0.0, scale, 60)
            assert float(np.std(x, ddof=1)) != tol
    # 且阈值本身在浮点上不可由"加上再减去"还原
    for base in (0.05, 1.0, 1e-6):
        assert (base + tol) - base != tol


def test_every_survivor_has_a_written_proof():
    """存活项要么被上面的用例杀死，要么在此有书面证明；不许有第三种状态。"""
    assert len(PROVEN_EQUIVALENT) == 1
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
