"""
基本面来源 / 期间质量门：逐项写明问题、找出表达式用到的字段、报告存下供门读取、没有合格报告的基本面策略不交易。

来源（原测试文件的说明，保留出处）：

复审 2026-10-19 F13 余项 / §5.4 / D8（基本面）：

  "`_shares` 的 diluted weighted-average fallback 不等于期末股本；shares 的事实期间没有与其他指标同样的
   逐字段过期检查。小于 1.5 倍的疑似送股部分只告警，仍会影响市值/估值；SUE 事件仅按日期关系屏蔽，
   未明确匹配盈利期间。启用基本面交易前应加来源/期间质量门。"
  §3 模块表："保留来源、有效期间、fallback 和缺失标记；小比例公司行动必须进入异常隔离。"
  D8："先把 filing-based 因子名称、期间和事件配对写清。"
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# §5.4 来源 / 期间质量门
# ---------------------------------------------------------------------------

def test_the_quality_gate_names_each_problem():
    from app.core.data_engine.fundamentals.panels import FundamentalsReport
    from app.core.data_engine.fundamentals.quality import quality_problems
    rep = FundamentalsReport(coverage={"earnings_yield": 0.5, "roe": 0.95}, estimated_shares=["A", "B"],
                             n_tickers=10)
    probs = quality_problems(rep, ["earnings_yield"])
    assert any("覆盖" in p for p in probs) and any("股本" in p for p in probs)
    assert quality_problems(rep, ["roe"]) == []
    assert quality_problems(rep, []) == []


def test_fields_used_by_an_expression_are_found():
    from app.core.data_engine.fundamentals.quality import fundamental_fields_in
    assert fundamental_fields_in(["rank(earnings_yield) - rank(ts_delta(close, 5))"]) == ["earnings_yield"]
    assert fundamental_fields_in(["rank(close)"]) == []


def test_attach_saves_the_report_for_the_gate(tmp_path):
    from app.core.data_engine.fundamentals.panels import FundamentalsReport
    from app.core.data_engine.fundamentals.store import FundamentalsStore
    st = FundamentalsStore(tmp_path)
    st.save_report(FundamentalsReport(coverage={"roe": 0.9}, n_tickers=3), pd.Timestamp("2024-11-27"))
    rep, asof = st.load_report()
    assert asof == pd.Timestamp("2024-11-27") and rep.coverage == {"roe": 0.9} and rep.n_tickers == 3


def test_a_strategy_using_fundamentals_does_not_trade_without_a_passing_quality_report(tmp_path, monkeypatch):
    from app.config import settings
    from app.core.data_engine.fundamentals.quality import strategy_quality_problem
    monkeypatch.setattr(settings, "fundamentals_dir", str(tmp_path / "f"))
    last = pd.Timestamp("2024-11-27")
    assert "质量报告" in strategy_quality_problem({"1": "rank(earnings_yield)"}, last)
    assert strategy_quality_problem({"1": "rank(close)"}, last) == ""
