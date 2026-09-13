"""
monitor/alpha_monitor.py —— 衰减检测边界的定钉测试（变异测试驱动）

来由：12 个变异点，首测击杀率 **50.0%**（存活 6）—— 存活的**全部是边界**。

AlphaMonitor 决定一个在跑的因子什么时候被判为"衰减"。判错的两个方向都要命：
判早了 → 还在赚钱的因子被降级下线；判晚了 → 已经失效的因子继续占着资金。
而这六个存活点全是"差一格"：

  - `len(ics) < self.rolling_window` —— 记录数**恰好等于**窗口时该不该出结论
  - `consec >= consecutive_neg_limit` 的伴生边界 `consecutive_neg_limit < 2`
  - `roll_mean < self.mean_ic_floor` —— 恰好等于下限时该不该告警
  - `_consecutive_negative` 里的 `if v < 0` —— IC **恰好为 0** 算不算负
  - `_ic_ir` 的 `len(ics) < 2` 与 `sd > 1e-12`

既有覆盖（test_phase5 / test_phase_pm / test_daily_loop_*）验证的是
"有告警/无告警"的明显情形，没有一条站在边界上。
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.monitor.alpha_monitor import AlphaMonitor


class _NullStore:
    """AlphaMonitor 只在 _ic_values / get_dashboard 用到 store；本文件全部走 _ics 注入。"""

    def get_ic_history(self, alpha_id, limit=None):
        return []

    def query(self, limit=500):
        return []


@pytest.fixture
def monitor():
    return AlphaMonitor(_NullStore(), rolling_window=5,
                        consecutive_neg_limit=3, mean_ic_floor=-0.01)


# ===========================================================================
# A. 构造参数的下限
# ===========================================================================

class TestConstructorBounds:

    def test_minimum_rolling_window_is_accepted(self):
        """`rolling_window < 5` 放宽成 `<= 5` 会把最小合法窗口 5 也拒掉。"""
        assert AlphaMonitor(_NullStore(), rolling_window=5).rolling_window == 5

    def test_rolling_window_below_five_is_rejected(self):
        with pytest.raises(ValueError, match="rolling_window"):
            AlphaMonitor(_NullStore(), rolling_window=4)

    def test_minimum_consecutive_limit_is_accepted(self):
        """
        `consecutive_neg_limit < 2` 放宽成 `<= 2` 会把 2 判成非法 ——
        "连续两天负 IC 就告警"是最激进也最常用的一档，直接用不了。
        """
        m = AlphaMonitor(_NullStore(), consecutive_neg_limit=2)
        assert m.consecutive_neg_limit == 2

    def test_consecutive_limit_below_two_is_rejected(self):
        """1 意味着"任何一天负 IC 都告警"，等于没有阈值。"""
        with pytest.raises(ValueError, match="consecutive_neg_limit"):
            AlphaMonitor(_NullStore(), consecutive_neg_limit=1)


# ===========================================================================
# B. 样本量门槛
# ===========================================================================

class TestSampleSizeGate:

    def test_exactly_window_many_records_do_produce_a_verdict(self, monitor):
        """
        `if len(ics) < self.rolling_window: return None` —— **严格小于**。
        放宽成 `<=` 会让记录数**恰好等于窗口**时仍然不下结论，
        告警整整晚一天；对一个已经连续 5 天负 IC 的因子，那一天是真金白银。
        """
        ics = [-0.05] * 5
        alert = monitor.check_decay(1, _ics=ics)
        assert alert is not None, (
            "记录数恰好等于 rolling_window 时没有出结论 —— 门槛被放宽了一格")
        assert alert.reason == "consecutive_negative"

    def test_one_record_short_of_the_window_stays_silent(self, monitor):
        assert monitor.check_decay(1, _ics=[-0.05] * 4) is None, (
            "记录不足窗口就下了结论 —— 数据不足却告警")


# ===========================================================================
# C. 连续负 IC
# ===========================================================================

class TestConsecutiveNegative:

    def test_zero_ic_breaks_the_negative_streak(self, monitor):
        """
        `if v < 0: n += 1 else: break` —— **严格小于 0**。
        放宽成 `<= 0` 会把 IC 恰好为 0 的日子也算成"负"，连续计数虚高，
        因子被提前判死。IC 为 0 在停牌/无交易日并不罕见。
        """
        ics = [0.1] * 2 + [-0.05, -0.05, 0.0, -0.05, -0.05, -0.05]
        # 末尾连续负只有 3 天（被中间那个 0.0 截断）
        assert monitor._consecutive_negative(ics) == 3, (
            f"连续负计数 {monitor._consecutive_negative(ics)}，应为 3 —— "
            f"IC 恰好为 0 被算成了负")

    def test_streak_counts_from_the_end_only(self, monitor):
        ics = [-0.9] * 10 + [0.2]
        assert monitor._consecutive_negative(ics) == 0, (
            "最后一天为正却仍在计数 —— 不是从末尾数起")

    def test_exactly_the_limit_triggers_an_alert(self, monitor):
        """`consec >= limit`：恰好达到阈值就要告警，不能要求超过。"""
        ics = [0.1, 0.1] + [-0.05] * 3
        alert = monitor.check_decay(1, _ics=ics)
        assert alert is not None and alert.reason == "consecutive_negative", (
            f"连续 3 天负（阈值 3）没有告警：{alert}")
        assert alert.consecutive_neg == 3

    def test_one_short_of_the_limit_does_not_trigger(self, monitor):
        ics = [0.1, 0.1, 0.1] + [-0.05] * 2
        alert = monitor.check_decay(1, _ics=ics)
        assert alert is None or alert.reason != "consecutive_negative", (
            f"只连续 2 天负（阈值 3）却报了连续负告警：{alert}")


# ===========================================================================
# D. 滚动均值下限
# ===========================================================================

class TestRollingMeanFloor:

    def test_mean_exactly_at_the_floor_does_not_alert(self, monitor):
        """
        `if roll_mean < self.mean_ic_floor` —— **严格小于**。
        放宽成 `<=` 会让恰好踩在下限上的因子被判衰减。
        下限是"可以接受的最差表现"，踩线应当放行。

        序列刻意让**末位为正**，避免连续负那条规则先触发把这条遮住
        （下限是 -0.01，全负序列会先撞上 consecutive_negative）。
        `[-0.02]*4 + [0.03]` 的均值在浮点上**恰好**是 -0.01。
        """
        ics = [-0.02] * 4 + [0.03]
        assert float(np.mean(ics)) == monitor.mean_ic_floor, (
            f"构造的均值 {np.mean(ics)!r} 没有精确落在下限上 —— 边界没测到")
        alert = monitor.check_decay(1, _ics=ics)
        assert alert is None, (
            f"滚动均值恰好等于下限却告警了：{alert} —— `<` 被放宽成了 `<=`")

    def test_mean_just_below_the_floor_alerts(self, monitor):
        ics = [-0.02] * 4 + [0.0299]
        assert float(np.mean(ics)) < monitor.mean_ic_floor
        alert = monitor.check_decay(1, _ics=ics)
        assert alert is not None, "滚动均值低于下限却没有告警"
        assert alert.reason == "rolling_mean_below_floor", (
            f"触发的是 {alert.reason}，不是下限规则")

    def test_only_the_last_window_is_averaged(self, monitor):
        """
        `np.mean(ics[-self.rolling_window:])` —— 窗口切片写错会把很久以前的
        表现算进来，一个早已失效的因子靠历史高 IC 续命。
        这里前段极好、后段极差，均值必须由**后段**决定。
        """
        ics = [0.5] * 20 + [-0.5] * 5
        alert = monitor.check_decay(1, _ics=ics)
        assert alert is not None, "只看最近 5 天应当告警，却被历史高 IC 稀释了"
        assert alert.rolling_mean_ic == pytest.approx(-0.5), (
            f"滚动均值是 {alert.rolling_mean_ic}，应为 -0.5")

    def test_consecutive_negative_takes_priority_over_the_floor(self, monitor):
        """两个条件同时成立时，reason 必须是先判的那个。"""
        ics = [-0.5] * 5
        alert = monitor.check_decay(1, _ics=ics)
        assert alert.reason == "consecutive_negative", (
            f"两个条件都成立时 reason 是 {alert.reason}，应当是先判的连续负")

    def test_healthy_alpha_produces_no_alert(self, monitor):
        assert monitor.check_decay(1, _ics=[0.03] * 10) is None


# ===========================================================================
# E. IC 信息比率
# ===========================================================================

class TestIcIr:

    def test_two_points_are_enough(self, monitor):
        """
        `if len(ics) < 2: return nan` —— **严格小于 2**。
        放宽成 `<= 2` 会让恰好两条记录也返回 NaN，仪表板上那一列长期空着。
        """
        val = monitor._ic_ir([0.02, 0.04])
        assert np.isfinite(val), "两条记录就该能算 IR，却返回了 NaN"

    def test_a_single_point_is_not_enough(self, monitor):
        assert np.isnan(monitor._ic_ir([0.02])), "一条记录也算出了 IR"

    def test_zero_dispersion_returns_nan_not_infinity(self, monitor):
        """
        `sd > 1e-12` —— 常数 IC 序列的样本标准差是 0（或 1e-17 量级的浮点残渣），
        守卫失效就会算出 1e16 量级的"完美信息比率"，仪表板上排第一。
        """
        assert np.isnan(monitor._ic_ir([0.02] * 10)), (
            "零离散度的 IC 序列算出了有限 IR")

    def test_ir_is_mean_over_sample_std(self, monitor):
        ics = [0.01, 0.03, -0.01, 0.05, 0.02]
        arr = np.asarray(ics, dtype=float)
        expected = arr.mean() / arr.std(ddof=1)
        assert monitor._ic_ir(ics) == pytest.approx(expected), (
            "IC IR 不是 均值 / 样本标准差(ddof=1)")

    def test_dispersion_exactly_at_the_epsilon_is_still_rejected(self, monitor):
        """
        `sd > 1e-12` —— **严格大于**。边界值是可构造的：
        `[-c, 0, c]` 的 ddof=1 样本标准差在浮点上**恰好等于 |c|**，取 c=1e-12。
        放宽成 `>=` 会让这种纯数值噪声级别的序列也算出一个 IR
        （这里会是 0.0），仪表板上看起来"有数据"，实际毫无意义。
        """
        ics = [-1e-12, 0.0, 1e-12]
        assert np.asarray(ics).std(ddof=1) == 1e-12, (
            f"构造的标准差 {np.asarray(ics).std(ddof=1)!r} 没有精确命中边界")
        assert np.isnan(monitor._ic_ir(ics)), (
            "标准差恰好等于 1e-12 时算出了 IR —— `sd > 1e-12` 被放宽成了 `>=`")

    def test_tiny_but_real_dispersion_still_yields_a_number(self, monitor):
        """
        `> 1e-12` 是**严格大于**：刚好越过守卫的离散度必须算得出来。
        收紧成更大的阈值会让低波动但真实的 IC 序列被判成"无离散度"。
        """
        ics = [0.02, 0.02 + 1e-6] * 5
        assert np.isfinite(monitor._ic_ir(ics)), (
            "1e-6 量级的真实离散度被当成了零")
