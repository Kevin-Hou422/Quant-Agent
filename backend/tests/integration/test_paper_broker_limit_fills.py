"""
模拟账本的 T+1 限价成交：开盘在限价内按开盘价、跳空后当日触及按限价、不触及不成交并如实挂账；预测标签与账本收益分开。

来源（原测试文件的说明，保留出处）：

2026-10-22 用户决定 · 时钟（D14 (a) + C3）与 Validate 预热（D9）。

D14 原话："T 收盘信息完成并过健康门后冻结信号与目标，T+1 按明确交易规则执行，更新时钟版本。同步修改信号、
回测、内部模拟、IC、归档、基准及调度。'计划开盘执行'不等于限价单必然按开盘价成交；预测 IC 与实际持仓收益
分别记录。旧历史保留旧版本，先隔离重算。"
C3 原话："目前虽然要求开盘前存档，评分仍是前收盘到今收盘，包含已经发生的隔夜收益。必须与 D14 一起明确标签
起止、信息截止和归档截止，不能只提前 deadline。"
D9 原话："只用当时已知历史计算特征，只评分 Validate 日期；拟合与标准化不能读取未来评分信息。覆盖 252 日因子、
嵌套窗口、跨界标签、初始持仓及历史不足。"
"""

from __future__ import annotations

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# D14："'计划开盘执行'不等于限价单必然按开盘价成交"
# ---------------------------------------------------------------------------

class TestPlannedOpenIsNotAssumedFilled:
    """参考价 = 昨收 100，限价带 50bp → 买单限价 100.5、卖单限价 99.5。"""

    @staticmethod
    def _broker(tmp_path):
        from app.core.backtest_engine.transaction_cost import CostParams
        from app.core.execution.paper_broker import PaperBroker
        from app.db.position_store import PositionStore
        free = CostParams(fixed_bps=0.0, min_ticket_fee=0.0, spread_bps=0.0, impact_coef=0.0,
                          short_borrow_annual_bps=0.0)
        return PaperBroker(store=PositionStore(db_url=f"sqlite:///{(tmp_path / 'pb.db').as_posix()}"),
                           initial_capital=1e6, cost_params=free, limit_band_bps=50.0)

    @staticmethod
    def _buy(b, o, h, lo, c=101.0):
        s = lambda v: pd.Series([v], index=["A"], dtype=float)          # noqa: E731
        b.step(1, "2025-03-05", target_w=s(0.5), prices_t=s(c), prices_prev=s(100.0),
               adv_usd=s(1e15), daily_vol=s(0.02), open_t=s(o), high_t=s(h), low_t=s(lo))
        return {f.ticker: f for f in b.store.fills_on(1, "2025-03-05")}

    def test_open_inside_the_limit_fills_at_the_open(self, tmp_path):
        f = self._buy(self._broker(tmp_path), o=100.3, h=101.5, lo=100.0)["A"]
        assert f.fill_price == pytest.approx(100.3) and f.reject_reason == ""

    def test_gap_above_the_limit_fills_at_the_limit_only_if_the_day_trades_back(self, tmp_path):
        f = self._buy(self._broker(tmp_path), o=102.0, h=103.0, lo=100.2)["A"]
        assert f.fill_price == pytest.approx(100.5), "开盘跳空到限价之外却按开盘价成交了"

    def test_gap_that_never_comes_back_does_not_fill_and_is_booked_as_unfilled(self, tmp_path):
        b = self._broker(tmp_path)
        f = self._buy(b, o=102.0, h=103.0, lo=101.0)["A"]
        assert f.reject_reason == "limit_not_reached"
        assert f.filled_weight == pytest.approx(0.0, abs=1e-12)
        assert f.unfilled_weight == pytest.approx(0.5, abs=1e-12)
        assert b.store.latest_positions(1).get("A", 0.0) == pytest.approx(0.0, abs=1e-12)

    def test_the_predictive_label_and_the_realized_book_are_separate_quantities(self, tmp_path):
        """同一天：限价没成交 → 账本收益 0；预测标签照样是开盘到开盘（与成交无关）。"""
        from app.core.clock import label_returns
        b = self._broker(tmp_path)
        self._buy(b, o=102.0, h=103.0, lo=101.0)
        assert b.store.latest_equity(1) == pytest.approx(1.0)
        idx = pd.bdate_range("2025-03-04", periods=3)
        lab = label_returns({"open": pd.DataFrame({"A": [99.0, 102.0, 104.0]}, index=idx)})
        assert lab["A"].iloc[0] == pytest.approx(104.0 / 102.0 - 1)
