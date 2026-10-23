"""
拆股识别：真拆股让原始价反向跳变、没有原始价跳变的复权因子跳变不是拆股、小比例跳变只报告并隔离估值字段直到新股本到来。

来源（原测试文件的说明，保留出处）：

外部审查 2026-10-18 F13：基本面字段的时间正确不等于经济口径正确。

  · SUE 等 10-Q，days_since_earnings 按 8-K 2.02 归零 —— 新 8-K 已到、10-Q 未到时，
    `days_since_earnings < 60` 会选中**上一季**的 SUE。
  · 过期按"最近一次披露"统一算：只更新股本的披露会让一年前的利润数字继续算新鲜。
  · 稀释加权平均股数是兜底，不是期末股本 —— 来源要看得见。
  · 拆股识别是启发式：复权因子跳了、不复权价没动（数据拼接问题）不能当拆股。

---

复审 2026-10-19 F13 余项 / §5.4 / D8（基本面）：

  "`_shares` 的 diluted weighted-average fallback 不等于期末股本；shares 的事实期间没有与其他指标同样的
   逐字段过期检查。小于 1.5 倍的疑似送股部分只告警，仍会影响市值/估值；SUE 事件仅按日期关系屏蔽，
   未明确匹配盈利期间。启用基本面交易前应加来源/期间质量门。"
  §3 模块表："保留来源、有效期间、fallback 和缺失标记；小比例公司行动必须进入异常隔离。"
  D8："先把 filing-based 因子名称、期间和事件配对写清。"
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.core.data_engine.fundamentals.panels import (
    build_panels,
    metric_events,
)
from app.core.data_engine.fundamentals.prices import detect_splits
from tests.unit.data_engine import fundamentals_fixtures as fx


def _series(vals):
    return pd.Series(vals, index=pd.bdate_range("2024-06-05", periods=len(vals)), name="X", dtype=float)


IDX = pd.bdate_range("2023-07-03", "2024-11-27")


def _aapl():
    return fx.facts("AAPL"), fx.events("AAPL")


def _raw(idx=IDX, px=180.0):
    return pd.DataFrame({"AAPL": px}, index=idx)


def test_a_real_split_moves_the_raw_price_the_other_way():
    raw, adj = _series([1200, 1210, 121, 122]), _series([120, 121, 121, 122])
    s = detect_splits(raw, adj)
    assert list(s.round(2)) == [10.0] and s.index[0] == raw.index[2]


def test_a_factor_jump_without_a_raw_price_jump_is_not_a_split(caplog):
    raw, adj = _series([100, 101, 100, 101]), _series([10, 10.1, 100, 101])     # 两个 vintage 拼在一起
    with caplog.at_level("WARNING"):
        assert detect_splits(raw, adj).empty
    assert any("没有伴随不复权价的反向跳变" in r.getMessage() for r in caplog.records)


def test_an_ambiguous_small_jump_is_reported_not_applied(caplog):
    raw, adj = _series([100, 100, 77, 77]), _series([100, 100, 100, 100])       # 1.3 倍：送股？特别分红？
    with caplog.at_level("WARNING"):
        assert detect_splits(raw, adj).empty
    assert any("小比例送股或特别分红" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# 小比例公司行动：异常隔离
# ---------------------------------------------------------------------------

def test_an_ambiguous_small_jump_quarantines_valuation_until_new_shares_arrive():
    facts, events = _aapl()
    jump = pd.Timestamp("2024-03-04")
    amb = {"AAPL": pd.Series({jump: 1.25})}
    p, rep = build_panels(IDX, ["AAPL"], {"AAPL": facts}, {"AAPL": events}, raw_close=_raw(), ambiguous=amb)
    mev = metric_events(facts)
    basis_after = mev.loc[mev["shares_basis"] > jump, "filed"].min()
    q = IDX[(IDX >= jump) & (IDX <= basis_after)]
    assert len(q) > 5
    for f in ("market_cap", "earnings_yield", "book_to_price"):
        assert p[f]["AAPL"][q].isna().all(), f"{f} 在疑似送股之后、新股本到来之前仍然有值"
    assert p["market_cap"]["AAPL"][IDX[IDX > basis_after + pd.Timedelta(days=5)]].notna().any()
    assert p["roe"]["AAPL"][q].notna().any(), "不依赖股本的字段不该被隔离"
    assert rep.quarantined["AAPL"]["events"] == [str(jump.date())]


def test_the_moomoo_path_reports_ambiguous_jumps_and_the_store_keeps_them(tmp_path):
    from app.core.data_engine.fundamentals import prices
    from app.core.data_engine.fundamentals.store import FundamentalsStore
    idx = pd.bdate_range("2024-06-03", periods=6)
    raw = pd.Series([100, 100, 100, 80, 80, 80], index=idx, dtype=float, name="X")
    adj = pd.Series([100, 100, 100, 100, 100, 100], index=idx, dtype=float, name="X")
    amb = prices.detect_ambiguous(raw, adj)
    assert list(amb.index) == [idx[3]] and amb.iloc[0] == pytest.approx(1.25)
    st = FundamentalsStore(tmp_path)
    st.save_raw_prices("moomoo", raw.to_frame(), {}, {"X": ["2024-06-03", "2024-06-10"]}, ambiguous={"X": amb})
    got = st.load_ambiguous("moomoo")
    assert list(got["X"].index) == [idx[3]]
