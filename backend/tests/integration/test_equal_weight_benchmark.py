"""
等权宇宙基准：成分在正式前向启动时冻结、缺价与隔离成分按 0 留在分母、规则带版本、相对满仓基准的差额不叫 alpha。

来源（原测试文件的说明，保留出处）：

2026-10-22 用户决定 · 证据 / 研究纪律。

D12 原话："同意先只读查询，不授权写边界。使用明确的活库路径和真正只读连接，不调用会建表或迁移的 Store 构造函数。
列出已有边界、最早曝光日期、次数、股票池和实验身份，给出日期建议后由我确认。正式入口不要继续依赖首次调用偶然冻结；
改实验名不能刷新重叠数据的曝光额度。"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# D16：首轮等权宇宙基准 —— 预先固定股票池时点、再平衡、收益窗口、缺价和公司行动处理
# 原话："不能事后删除缺价股票再平均。SPY 第二基准后排；低仓位策略对满仓基准的差额不要直接称为 alpha。"
# ---------------------------------------------------------------------------

_IDX = pd.bdate_range("2025-03-03", periods=2)


_D = "2025-03-04"


def _rec(close: pd.DataFrame, *, forward_run=None, result=None, now="2025-03-04 22:30"):
    from types import SimpleNamespace

    from app.db.forward_record import build_daily_record
    r = {"execution": {"mode": "moomoo_paper", "reconcile": {"status": "clean"}}}
    r.update(result or {})
    return build_daily_record(r, strategy_id=1, spec_hash="h", decision_date=_D,
                              now=pd.Timestamp(now, tz="UTC"), dataset={"close": close},
                              paper_history=[SimpleNamespace(date=_D, net_ret=0.01, cost_bps=1.0, equity=1.01)],
                              forward_run=forward_run, strategy_status="paper_running")


def _run(universe):
    import json
    from types import SimpleNamespace

    from app.core.benchmark import freeze_universe
    return SimpleNamespace(started_at=pd.Timestamp("2025-01-01"), benchmark=json.dumps(freeze_universe(universe)))


class TestEqualWeightUniverseBenchmark:

    def test_a_missing_price_stays_in_the_denominator(self):
        close = pd.DataFrame({"A": [100.0, 110.0], "B": [50.0, 50.0], "C": [20.0, np.nan], "D": [10.0, 9.8]},
                             index=_IDX)
        b = _rec(close, forward_run=_run(["A", "B", "C", "D"]))["benchmark"]
        assert b["return"] == pytest.approx((0.10 + 0.0 + 0.0 - 0.02) / 4), "缺价的票被删掉后再平均了"
        assert b["missing"] == ["C"] and b["missing_share"] == pytest.approx(0.25)

    def test_the_universe_is_the_one_frozen_at_the_formal_start(self):
        """启动之后新出现的票（E）不进基准；启动时有、现在没有列的票（D）留在分母、记缺价。"""
        close = pd.DataFrame({"A": [100.0, 110.0], "B": [50.0, 50.0], "E": [5.0, 10.0]}, index=_IDX)
        b = _rec(close, forward_run=_run(["A", "B", "D"]))["benchmark"]
        assert b["universe_frozen"] is True and b["n"] == 3
        assert b["return"] == pytest.approx(0.10 / 3)
        assert b["missing"] == ["D"]

    def test_before_a_formal_start_the_universe_is_marked_as_not_frozen(self):
        close = pd.DataFrame({"A": [100.0, 110.0], "B": [50.0, 50.0]}, index=_IDX)
        b = _rec(close, forward_run=None)["benchmark"]
        assert b["universe_frozen"] is False and b["return"] == pytest.approx(0.05)

    def test_the_rules_are_fixed_in_advance_and_recorded(self):
        from app.core.benchmark import BENCHMARK_RULES
        for k in ("universe", "rebalance", "return_window", "price", "missing_price", "corporate_actions"):
            assert BENCHMARK_RULES.get(k), f"基准规则没有预先写明 {k}"
        close = pd.DataFrame({"A": [100.0, 110.0]}, index=_IDX)
        b = _rec(close, forward_run=_run(["A"]))["benchmark"]
        assert b["name"] == "ew_universe" and b["rules_version"] == BENCHMARK_RULES["version"]
        assert b["secondary"] is None and "SPY" in b["secondary_note"]

    def test_corporate_action_quarantined_members_are_held_at_zero_not_dropped(self):
        close = pd.DataFrame({"A": [100.0, 110.0], "B": [50.0, 25.0]}, index=_IDX)     # B 疑似未知拆股
        rec = _rec(close, forward_run=_run(["A", "B"]), result={"ca_quarantined": ["B"]})
        b = rec["benchmark"]
        assert b["return"] == pytest.approx(0.10 / 2) and b["quarantined"] == ["B"]

    def test_a_low_exposure_difference_is_not_called_alpha(self):
        close = pd.DataFrame({"A": [100.0, 102.0], "B": [50.0, 51.0]}, index=_IDX)     # 基准 +2%
        rec = _rec(close, forward_run=_run(["A", "B"]),
                   result={"execution_target": {"A": 0.15, "B": 0.15}})               # 只有 30% 净敞口
        ex = rec["excess_return"]
        assert ex["paper_vs_full_benchmark"] == pytest.approx(0.01 - 0.02)
        assert ex["paper_exposure_matched"] == pytest.approx(0.01 - 0.30 * 0.02)
        assert "alpha" not in " ".join(ex.keys()).lower()
        assert "不是 alpha" in ex["label"]

    def test_a_formal_start_must_freeze_the_benchmark_universe(self, fresh_db):
        from app.db.forward_record import ForwardRecordStore
        from app.db.strategy_store import StrategyConfig, StrategyStore
        ss = StrategyStore()
        sid = ss.save(StrategyConfig(factors=["1"], combo_weights={"1": 1.0}, aum=1e4, status="proposed",
                                     verdict={"spec_hash": "h"}))
        ss.update_status(sid, "approved"); ss.update_status(sid, "paper_running")
        with pytest.raises(ValueError, match="基准"):
            ForwardRecordStore().start(sid, "h", "kevin", "go")
        run = ForwardRecordStore().start(sid, "h", "kevin", "go", benchmark_universe=["B", "A", "A"])
        import json
        assert json.loads(run.benchmark)["tickers"] == ["A", "B"]
