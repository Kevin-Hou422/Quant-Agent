"""
SUE 的可见性：新财报发布到申报之间为空、按盈利期间配对（更正公告不清空已申报季）、每次公告都映射到财季。

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

import numpy as np
import pandas as pd

from app.core.data_engine.fundamentals.panels import (
    MAX_PERIOD_AGE_DAYS,
    build_panels,
    metric_events,
)
from tests.unit.data_engine import fundamentals_fixtures as fx

IDX = pd.bdate_range("2023-07-03", "2024-11-27")


def _aapl():
    return fx.facts("AAPL"), fx.events("AAPL")


def test_sue_is_blank_between_a_new_earnings_release_and_its_filing():
    facts, events = _aapl()
    p, rep = build_panels(IDX, ["AAPL"], {"AAPL": facts}, {"AAPL": events})
    sue, dse = p["sue"]["AAPL"], p["days_since_earnings"]["AAPL"]
    released = [d for d in IDX if dse.get(d) == 0]                     # 新公告生效的那天
    assert released
    filed = sorted(pd.to_datetime(facts["filed"]).unique())
    checked = 0
    for d in released:
        prev_10q = max(f for f in filed if f < d)
        nxt = [f for f in filed if f >= d - pd.Timedelta(days=1)]
        if nxt and nxt[0] > d - pd.Timedelta(days=1):                  # 申报晚于公告：中间这几天 SUE 为空
            assert np.isnan(sue[d]), f"{d.date()} 公告已出、申报未到，却给出了上一季（{prev_10q.date()}）的 SUE"
            checked += 1
    assert checked >= 3
    assert rep.sue_pending["AAPL"] > 0
    assert sue.notna().mean() > 0.95                                   # 只空公告到申报之间那一两天


def test_old_earnings_do_not_stay_fresh_because_a_later_filing_updated_shares():
    facts, events = _aapl()
    cut = pd.Timestamp("2023-12-31")
    keep = (pd.to_datetime(facts["filed"]) <= cut) | facts["concept"].astype(str).str.startswith("shares")
    p, _ = build_panels(IDX, ["AAPL"], {"AAPL": facts[keep]}, {"AAPL": events})
    ends = metric_events(facts[keep])["ni_end"].dropna()
    last_ni_end = ends[ends <= cut].max()
    roe = p["roe"]["AAPL"]
    expired = IDX[IDX > last_ni_end + pd.Timedelta(days=MAX_PERIOD_AGE_DAYS)]
    assert len(expired) > 0 and roe[expired].isna().all(), "期末日早已过期的利润仍在给 ROE"
    assert roe[IDX[IDX < cut + pd.Timedelta(days=30)]].notna().any()


# ---------------------------------------------------------------------------
# SUE：按盈利期间配对，不靠时间先后
# ---------------------------------------------------------------------------

def test_a_correction_8k_for_the_quarter_already_filed_does_not_blank_sue():
    """10-Q 已到之后、同一季的一份更正 8-K 2.02：按时间先后会把 SUE 置空到下一份申报（两个多月）。"""
    facts, events = _aapl()
    filed = sorted(pd.to_datetime(facts["filed"]).unique())
    q_filing = [f for f in filed if pd.Timestamp("2024-05-01") < f < pd.Timestamp("2024-06-30")][0]
    extra = events.iloc[[-1]].copy()
    extra["filing_date"] = q_filing + pd.Timedelta(days=10)
    extra["accn"] = "0000000000-24-999999"
    ev2 = pd.concat([events, extra]).sort_values("filing_date")
    p, rep = build_panels(IDX, ["AAPL"], {"AAPL": facts}, {"AAPL": ev2})
    window = IDX[(IDX > q_filing + pd.Timedelta(days=12)) & (IDX < pd.Timestamp("2024-07-25"))]
    assert len(window) > 20
    assert p["sue"]["AAPL"][window].notna().all(), "同一季的更正公告把已申报那季的 SUE 置空了"
    # 面板期内的公告都要配上财季（最早几条在任何财季披露之前，按时点推不出 —— 那是对的）
    inside = [e for e in rep.sue_pairing["AAPL"]["events"] if e["announced"] >= IDX[0]]
    assert inside and all(e["quarter_end"] is not None for e in inside)
    extra_q = [e for e in inside if e["announced"] == q_filing + pd.Timedelta(days=10)]
    assert extra_q and extra_q[0]["quarter_end"] < q_filing, "更正公告没有配到已申报的那一季"


def test_a_new_quarters_release_before_its_filing_still_blanks_sue():
    facts, events = _aapl()
    p, rep = build_panels(IDX, ["AAPL"], {"AAPL": facts}, {"AAPL": events})
    assert rep.sue_pending["AAPL"] > 0
    assert p["sue"]["AAPL"].notna().mean() > 0.95


def test_every_announcement_is_mapped_to_a_fiscal_quarter_in_the_report():
    facts, events = _aapl()
    _, rep = build_panels(IDX, ["AAPL"], {"AAPL": facts}, {"AAPL": events})
    m = [e for e in rep.sue_pairing["AAPL"]["events"] if e["announced"] >= IDX[0]]
    assert m and all(e["quarter_end"] is not None and e["quarter_end"] < e["announced"] for e in m)
    assert all(e["announced"] - e["quarter_end"] <= pd.Timedelta(days=100) for e in m)
