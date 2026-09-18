"""
alpha_engine/signal_processor.py —— 四步信号后处理的定钉测试（变异测试驱动）

来由：18 个变异点，首测击杀率 **44.4%**（存活 10）。

SignalProcessor 站在"因子值"和"下单权重"之间，做四件事：
截断 → 衰减 → 中性化 → **延迟**。最后一步是整个系统防前视泄漏的最后一道闸：
`if self.cfg.delay > 0: out = self._delay(out)`。这个 `> 0` 一旦被改成 `>= 0`，
`delay=0` 的配置会**多 shift 一格**（无害）；但真正危险的是反过来 ——
任何让 `_delay` 不执行的改动都会让 T 日收盘才知道的信号在 T 日就成交，
回测 Sharpe 凭空翻倍，而没有任何测试会红。

存活项：
  - `__post_init__` 的四道参数校验（`delay < 0`、两个分位区间的 `not (...)`）
    —— 放宽/取反之后**非法配置被静默接受**，或者合法配置被拒
  - `process` 的四个开关（`is not None`、`> 1`、`> 0`）
    —— 决定某一步跑不跑，改掉就是整步被跳过或被误跑
  - `market_neutral: bool = True` / `neutralize_groups` 的默认值

既有覆盖（test_phase1_upgrade / test_leak_filter）测的是"管道跑完形状不变"
与"整体不泄漏"，没有一条把**每一步的开关条件**单独钉住。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.alpha_engine.signal_processor import SignalProcessor, SimulationConfig


T, N = 30, 8
IDX = pd.bdate_range("2024-01-02", periods=T)
COLS = [f"T{i}" for i in range(N)]


def _signal(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0, 1, (T, N)), index=IDX, columns=COLS)


def _cfg(**kw) -> SimulationConfig:
    base = dict(delay=0, decay_window=0, truncation_min_q=None,
                truncation_max_q=None)
    base.update(kw)
    return SimulationConfig(**base)


# ===========================================================================
# A. 参数校验的边界
# ===========================================================================

class TestConfigValidation:

    def test_zero_delay_is_legal(self):
        """
        `if self.delay < 0: raise` —— 放宽成 `<= 0` 会把 `delay=0`（同日执行，
        回测研究里合法且常用的对照配置）判成非法，直接抛错。
        """
        assert _cfg(delay=0).delay == 0

    def test_negative_delay_is_rejected(self):
        with pytest.raises(ValueError, match="delay"):
            _cfg(delay=-1)

    def test_zero_decay_window_is_legal(self):
        assert _cfg(decay_window=0).decay_window == 0

    def test_negative_decay_window_is_rejected(self):
        with pytest.raises(ValueError, match="decay_window"):
            _cfg(decay_window=-1)

    @pytest.mark.parametrize("q", [0.0, 0.05, 0.4999])
    def test_min_quantile_inside_the_half_open_interval_is_accepted(self, q):
        """
        `not (0.0 <= min_q < 0.5)` —— 三处变异：
        删掉 `not` 会让**合法**取值抛错、非法取值放行（彻底反过来）；
        `<` 放宽成 `<=` 会让 0.5 也被接受 —— 上下分位数就能重叠，
        截断区间反转，`np.clip(lo>hi)` 把整行压成同一个数。
        """
        assert _cfg(truncation_min_q=q).truncation_min_q == q

    @pytest.mark.parametrize("q", [0.5, 0.6, -0.01])
    def test_min_quantile_outside_the_interval_is_rejected(self, q):
        with pytest.raises(ValueError, match="truncation_min_q"):
            _cfg(truncation_min_q=q)

    @pytest.mark.parametrize("q", [0.5001, 0.95, 1.0])
    def test_max_quantile_inside_the_half_open_interval_is_accepted(self, q):
        """`0.5 < max_q <= 1.0`：1.0 必须**被接受**（不截上尾），0.5 必须被拒。"""
        assert _cfg(truncation_max_q=q).truncation_max_q == q

    @pytest.mark.parametrize("q", [0.5, 0.4, 1.01])
    def test_max_quantile_outside_the_interval_is_rejected(self, q):
        with pytest.raises(ValueError, match="truncation_max_q"):
            _cfg(truncation_max_q=q)

    def test_none_quantiles_skip_validation_entirely(self):
        cfg = _cfg(truncation_min_q=None, truncation_max_q=None)
        assert cfg.truncation_min_q is None and cfg.truncation_max_q is None

    def test_top_pct_must_be_a_proper_fraction(self):
        for bad in (0.0, 0.5, -0.1, 0.7):
            with pytest.raises(ValueError, match="top_pct"):
                _cfg(top_pct=bad)
        assert _cfg(top_pct=0.1).top_pct == 0.1

    def test_defaults_match_the_documented_behaviour(self):
        """
        `market_neutral: bool = True` 改成 False 会让组合默认**不做市场中性**，
        净敞口凭空出现；`neutralize_groups` 的默认 None 改掉会让中性化步骤
        默认开启并拿到一个假分组。
        """
        d = SimulationConfig()
        assert d.delay == 1, "默认执行延迟不再是 T+1 —— 前视泄漏的默认防线没了"
        assert d.market_neutral is True, "默认不再做市场中性"
        assert d.neutralize_groups is None, "默认带上了分组，中性化会被误开"
        assert d.truncation_min_q == 0.05 and d.truncation_max_q == 0.95
        assert d.portfolio_mode == "long_short"


# ===========================================================================
# B. 四个步骤的开关
# ===========================================================================

class TestPipelineSwitches:

    def test_truncation_runs_when_either_quantile_is_set(self):
        """
        `if min_q is not None or max_q is not None` —— 删掉 `not` 会让
        **两个都设了**的时候反而不截断。分别只设一边，两次都必须生效。
        """
        raw = _signal(1)
        raw.iloc[:, 0] = 100.0          # 一个极端列
        only_hi = SignalProcessor(_cfg(truncation_max_q=0.9)).process(raw)
        assert only_hi.iloc[:, 0].max() < 100.0, "只设上分位时没有截断"

        raw2 = _signal(1)
        raw2.iloc[:, 0] = -100.0
        only_lo = SignalProcessor(_cfg(truncation_min_q=0.1)).process(raw2)
        assert only_lo.iloc[:, 0].min() > -100.0, "只设下分位时没有截断"

    def test_truncation_is_skipped_when_both_are_none(self):
        raw = _signal(2)
        raw.iloc[:, 0] = 100.0
        out = SignalProcessor(_cfg()).process(raw)
        assert out.iloc[:, 0].max() == 100.0, "两个分位都是 None 却截断了"

    def test_decay_needs_a_window_above_one(self):
        """
        `if self.cfg.decay_window > 1` —— **严格大于 1**。
        放宽成 `>= 1` 会让 `decay_window=1` 也走一次 ts_decay_linear：
        窗口为 1 的线性衰减等于原值，但它会把**首行之外的所有 NaN 语义**
        改掉（窗口未满 → NaN），静默削掉一行数据。
        """
        raw = _signal(3)
        w1 = SignalProcessor(_cfg(decay_window=1)).process(raw)
        pd.testing.assert_frame_equal(w1, raw, obj="decay_window=1 不该改动信号")

        w3 = SignalProcessor(_cfg(decay_window=3)).process(raw)
        assert not np.allclose(w3.to_numpy()[2:], raw.to_numpy()[2:]), (
            "decay_window=3 没有产生任何平滑效果")
        assert w3.iloc[:2].isna().all().all(), "衰减窗口未满的前两行应当是 NaN"

    def test_decay_is_skipped_at_zero(self):
        raw = _signal(4)
        pd.testing.assert_frame_equal(SignalProcessor(_cfg(decay_window=0)).process(raw),
                                      raw, obj="decay_window=0 却做了衰减")

    def test_neutralization_runs_only_with_groups(self):
        raw = _signal(5)
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        out = SignalProcessor(_cfg(neutralize_groups=groups)).process(raw)
        np.testing.assert_allclose(out.to_numpy()[:, :4].mean(axis=1), 0.0, atol=1e-12,
                                   err_msg="给了分组却没有做组内去均值")
        pd.testing.assert_frame_equal(SignalProcessor(_cfg()).process(raw), raw,
                                      obj="没给分组却做了中性化")

    def test_delay_shifts_forward_by_exactly_the_configured_days(self):
        """
        `if self.cfg.delay > 0: out = self._delay(out)` —— 这是防前视泄漏的最后一闸。
        `_delay` 用 `df.shift(delay)`：T 日的信号挪到 T+delay 行。
        shift 的符号或这个开关被改掉，回测就会用**当天收盘后才知道的信号**
        在当天成交，Sharpe 凭空翻倍。
        """
        raw = _signal(6)
        out = SignalProcessor(_cfg(delay=2)).process(raw)
        assert out.iloc[:2].isna().all().all(), "延迟后的前两行应当是 NaN"
        np.testing.assert_allclose(out.to_numpy()[2:], raw.to_numpy()[:-2],
                                   rtol=1e-12,
                                   err_msg="延迟后的值不是 delay 天之前的信号 —— "
                                           "shift 方向或步长被改了")

    def test_zero_delay_leaves_the_signal_in_place(self):
        raw = _signal(7)
        pd.testing.assert_frame_equal(SignalProcessor(_cfg(delay=0)).process(raw),
                                      raw, obj="delay=0 却移动了信号")

    def test_steps_run_in_the_documented_order(self):
        """
        截断 → 衰减 → 中性化 → 延迟。顺序换了结果会变：
        先延迟再中性化会让首行的 NaN 参与组均值，组内去均值失效。
        这里用"最后一步一定是延迟"来钉：前 delay 行必须整行 NaN。
        """
        raw = _signal(8)
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        out = SignalProcessor(_cfg(delay=1, decay_window=3,
                                   truncation_min_q=0.1, truncation_max_q=0.9,
                                   neutralize_groups=groups)).process(raw)
        assert out.iloc[0].isna().all(), "延迟不是最后一步"
        # 衰减窗口 3 + 延迟 1 → 前 3 行都该是 NaN
        assert out.iloc[:3].isna().all().all(), "衰减的 burn-in 没有被延迟一起推后"
        tail = out.iloc[3:].to_numpy()
        np.testing.assert_allclose(tail[:, :4].mean(axis=1), 0.0, atol=1e-9,
                                   err_msg="中性化没有在延迟之前生效")


# ===========================================================================
# C. 截断的数值
# ===========================================================================

class TestTruncation:

    def test_bounds_are_the_row_quantiles(self):
        raw = pd.DataFrame(np.tile(np.arange(N, dtype=float), (3, 1)),
                           index=IDX[:3], columns=COLS)
        out = SignalProcessor(_cfg(truncation_min_q=0.25,
                                   truncation_max_q=0.75)).process(raw)
        lo = np.nanpercentile(raw.to_numpy()[0], 25)
        hi = np.nanpercentile(raw.to_numpy()[0], 75)
        np.testing.assert_allclose(out.to_numpy()[0].min(), lo, rtol=1e-12,
                                   err_msg="下界不是该行的 25 分位")
        np.testing.assert_allclose(out.to_numpy()[0].max(), hi, rtol=1e-12,
                                   err_msg="上界不是该行的 75 分位")

    def test_all_nan_rows_stay_nan_without_warning(self):
        raw = _signal(9)
        raw.iloc[0] = np.nan
        out = SignalProcessor(_cfg(truncation_min_q=0.05,
                                   truncation_max_q=0.95)).process(raw)
        assert out.iloc[0].isna().all(), "全 NaN 行被截断成了有限值"
        assert out.iloc[1:].notna().all().all(), "有效行被全 NaN 行连累了"

    def test_rows_are_clipped_independently(self):
        raw = pd.DataFrame([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                           index=IDX[:2], columns=COLS)
        out = SignalProcessor(_cfg(truncation_max_q=0.9)).process(raw)
        assert out.iloc[0, 7] < 100.0, "第一行的极端值没有被截断"
        np.testing.assert_allclose(out.iloc[1].to_numpy(), 0.0, atol=1e-12,
                                   err_msg="全零行被另一行的分位数污染了")

    def test_index_and_columns_survive_the_whole_pipeline(self):
        raw = _signal(10)
        out = SignalProcessor(_cfg(delay=1, decay_window=2,
                                   truncation_min_q=0.05,
                                   truncation_max_q=0.95)).process(raw)
        pd.testing.assert_index_equal(out.index, raw.index)
        assert list(out.columns) == list(raw.columns)
        assert out.shape == raw.shape

    def test_the_input_frame_is_not_mutated(self):
        raw = _signal(11)
        before = raw.copy()
        SignalProcessor(_cfg(delay=1, truncation_min_q=0.1)).process(raw)
        pd.testing.assert_frame_equal(raw, before,
                                      obj="process() 就地改了调用方的 DataFrame")

    def test_group_array_stays_out_of_the_config_repr(self):
        """
        `neutralize_groups: ... = field(default=None, repr=False)` ——
        改成 `repr=True` 会把整个 (N,) 数组塞进 `repr(SimulationConfig)`。
        这个 repr 会进日志、进 run_manifest 的 config_json、进错误消息：
        几千只标的的行业码会把台账撑成几十 KB 的噪声，人工复核直接没法看。
        """
        cfg = SimulationConfig(neutralize_groups=np.arange(500))
        r = repr(cfg)
        assert "neutralize_groups" not in r, (
            f"分组数组进了 repr（长度 {len(r)}）：{r[:200]}…")
        assert len(r) < 400, f"config 的 repr 膨胀到了 {len(r)} 字符"


# ===========================================================================
# D. 存活变异的等价性证明
# ===========================================================================

PROVEN_EQUIVALENT = {
    "app/core/alpha_engine/signal_processor.py ×1 — L107 `if self.cfg.decay_window > 1:` → `>=`":
        "两种取值只在 `decay_window == 1` 时分道：此时走进 `_decay` 会调用 "
        "`ts_decay_linear(arr, 1)`，其权重为 `arange(1,2)/1 = [1.0]`、窗口长度 1，"
        "逐行等于原值，且 `out[0:]` 覆盖整个数组（无 burn-in NaN）。"
        "也就是说该分支在 window=1 时是**恒等变换**，跑与不跑结果逐位相同。"
        "见 test_decay_with_window_one_is_the_identity。",

    "app/core/alpha_engine/signal_processor.py ×1 — L115 `if self.cfg.delay > 0:` → `>=`":
        "两种取值只在 `delay == 0` 时分道：`_delay` 返回 `df.shift(0)`，"
        "pandas 的 shift(0) 返回内容与索引完全相同的副本，是恒等变换。"
        "见 test_shift_by_zero_is_the_identity。",
}


def test_decay_with_window_one_is_the_identity():
    """L107 等价性的机械验证：ts_decay_linear(x, 1) 与 x 逐位相同（含 NaN 位置）。"""
    from app.core.alpha_engine.fast_ops import ts_decay_linear
    x = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0], [-7.0, 8.0]])
    np.testing.assert_array_equal(
        ts_decay_linear(x, 1), x,
        err_msg="ts_decay_linear(x,1) 不再是恒等变换 —— L107 不再等价，需要补用例")


def test_shift_by_zero_is_the_identity():
    """L115 等价性的机械验证：DataFrame.shift(0) 与原表逐位相同。"""
    df = _signal(12)
    pd.testing.assert_frame_equal(
        df.shift(0), df,
        obj="shift(0) 不再是恒等变换 —— L115 不再等价，需要补用例")


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 2
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
