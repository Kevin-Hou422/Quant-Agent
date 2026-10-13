"""
Phase 10.2 / 10.3 — 时点计算（TTM / 单季 / SUE / 时点值）。

数值全部对着**真实财报**核过：AAPL 财年净利润 2022 = 99.803B、2023 = 96.995B、2024 = 93.736B；
NVDA 财年 2024（截至 2024-01-28）= 29.760B。这些是公司 10-K 里的数，不是本模块算出来再抄回来的。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.data_engine.fundamentals import edgar, pit
from tests.unit.data_engine import fundamentals_fixtures as fx

B = 1e9


def _visible(ticker: str, as_of: str) -> pd.DataFrame:
    f = fx.facts(ticker)
    return f[f["filed"] <= pd.Timestamp(as_of)]


def _fact(concept, start, end, filed, val, accn="a"):
    return {"ticker": "X", "concept": concept, "tag": "t", "start": start, "end": end,
            "filed": filed, "val": float(val), "form": "10-Q", "accn": accn}


def _frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=edgar.FACT_COLUMNS)
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c])
    return df


# ---------------------------------------------------------------------------
# TTM
# ---------------------------------------------------------------------------

def test_ttm_right_after_the_annual_report_is_the_fiscal_year():
    val, end = pit.ttm(_visible("AAPL", "2023-11-03"), "net_income")
    assert val == pytest.approx(96.995 * B)
    assert end == pd.Timestamp("2023-09-30")


def test_ttm_mid_year_rolls_one_quarter_forward():
    """FY2023 全年 96.995 + FY24Q1 33.916 − FY23Q1 29.998 = 100.913。"""
    val, end = pit.ttm(_visible("AAPL", "2024-02-02"), "net_income")
    assert val == pytest.approx((96.995 + 33.916 - 29.998) * B)
    assert end == pd.Timestamp("2023-12-30")


def test_ttm_is_what_was_known_then_not_what_is_known_now():
    """2024-02-01（季报前一天）看到的还是全年数；季报当天才变。"""
    before, _ = pit.ttm(_visible("AAPL", "2024-02-01"), "net_income")
    after, _ = pit.ttm(_visible("AAPL", "2024-02-02"), "net_income")
    assert before == pytest.approx(96.995 * B)
    assert after == pytest.approx(100.913 * B)


def test_ttm_handles_a_january_fiscal_year_end():
    val, end = pit.ttm(_visible("NVDA", "2024-02-21"), "net_income")
    assert val == pytest.approx(29.760 * B) and end == pd.Timestamp("2024-01-28")
    val, _ = pit.ttm(_visible("NVDA", "2024-05-29"), "net_income")
    assert val == pytest.approx(42.598 * B)          # 29.760 + 14.881 − 2.043


def test_ttm_nine_month_ytd_uses_the_longest_window_at_period_end():
    """三季报同时披露 3 个月和 9 个月两列 —— 必须取 9 个月那列。"""
    rows = [
        _fact("net_income", "2022-01-01", "2022-12-31", "2023-02-01", 400),      # 上一财年
        _fact("net_income", "2022-01-01", "2022-09-30", "2022-11-01", 290),      # 上年同期 9M
        _fact("net_income", "2022-07-01", "2022-09-30", "2022-11-01", 100),      # 上年同期单季
        _fact("net_income", "2023-01-01", "2023-09-30", "2023-11-01", 350),      # 本年 9M
        _fact("net_income", "2023-07-01", "2023-09-30", "2023-11-01", 130),      # 本年单季
    ]
    val, end = pit.ttm(_frame(rows), "net_income")
    assert val == pytest.approx(400 + 350 - 290) and end == pd.Timestamp("2023-09-30")


@pytest.mark.parametrize("drop", ["prev_fy", "prev_ytd"])
def test_ttm_is_none_when_a_leg_is_missing(drop):
    rows = {
        "prev_fy":  _fact("net_income", "2022-01-01", "2022-12-31", "2023-02-01", 400),
        "prev_ytd": _fact("net_income", "2022-01-01", "2022-03-31", "2022-05-01", 90),
        "cur_ytd":  _fact("net_income", "2023-01-01", "2023-03-31", "2023-05-01", 120),
    }
    val, end = pit.ttm(_frame(rows.values()), "net_income")
    assert val == pytest.approx(430.0) and end == pd.Timestamp("2023-03-31")
    del rows[drop]
    assert pit.ttm(_frame(rows.values()), "net_income") is None


def test_ttm_of_an_unknown_concept_is_none():
    assert pit.ttm(_visible("AAPL", "2024-02-02"), "nope") is None
    assert pit.ttm(_frame([]), "net_income") is None


def test_ttm_tolerates_52_53_week_fiscal_calendars():
    """AAPL 的季末日每年挪几天（52/53 周制），上年同期按 ±10 天匹配而不是整 365 天。"""
    vis = _visible("AAPL", "2024-02-02")
    f = pit.flows(vis, "net_income")
    q_now = f[(f["end"] == "2023-12-30") & (f["days"] < 100)].iloc[0]
    q_then = f[(f["end"] == "2022-12-31") & (f["days"] < 100)].iloc[0]
    assert (q_now["end"] - q_then["end"]).days == 364          # 不是 365
    assert pit.ttm(vis, "net_income") is not None


# ---------------------------------------------------------------------------
# 重述：只从重述的披露日起生效
# ---------------------------------------------------------------------------

def test_restatement_changes_the_value_only_from_its_own_filing_date():
    rows = [
        _fact("net_income", "2022-01-01", "2022-12-31", "2023-02-01", 400, "orig"),
        _fact("net_income", "2022-01-01", "2022-12-31", "2023-08-15", 360, "restated"),
    ]
    f = _frame(rows)
    assert pit.ttm(f[f["filed"] <= "2023-08-14"], "net_income")[0] == 400
    assert pit.ttm(f[f["filed"] <= "2023-08-15"], "net_income")[0] == 360


def test_latest_version_wins_regardless_of_row_order():
    rows = [
        _fact("net_income", "2022-01-01", "2022-12-31", "2023-08-15", 360, "restated"),
        _fact("net_income", "2022-01-01", "2022-12-31", "2023-02-01", 400, "orig"),
    ]
    assert pit.ttm(_frame(rows), "net_income")[0] == 360
    assert pit.ttm(_frame(rows[::-1]), "net_income")[0] == 360


def test_real_restated_share_count_is_invisible_before_it_was_filed():
    """NVDA 2024-01-28 的股本：当时披露 24.64 亿，一年后（拆股后）重述为 246.43 亿。"""
    then = pit.latest_instant(_visible("NVDA", "2024-03-01"), "shares_bs")
    assert then["val"] == 2_464_000_000.0 and then["end"] == pd.Timestamp("2024-01-28")
    f = fx.facts("NVDA")
    later = f[(f["filed"] <= "2025-03-01") & (f["end"] <= "2024-01-28")]
    assert pit.latest_instant(later, "shares_bs")["val"] == 24_643_000_000.0


# ---------------------------------------------------------------------------
# 时点值
# ---------------------------------------------------------------------------

def test_latest_instant_is_the_most_recent_period_end():
    row = pit.latest_instant(_visible("AAPL", "2024-02-02"), "equity")
    assert row["end"] == pd.Timestamp("2023-12-30") and row["val"] == pytest.approx(74.100 * B)


def test_latest_instant_none_when_absent():
    assert pit.latest_instant(_frame([]), "equity") is None
    assert pit.latest_instant(_visible("AAPL", "2024-02-02"), "nope") is None


def test_instant_near_matches_within_tolerance_only():
    vis = _visible("AAPL", "2024-02-02")
    assert pit.instant_near(vis, "assets", pd.Timestamp("2022-12-31")) == pytest.approx(346.747 * B)
    # 容差边界：期末日 2022-12-31，±10 天内匹配，11 天不匹配
    assert pit.instant_near(vis, "assets", pd.Timestamp("2023-01-10")) == pytest.approx(346.747 * B)
    assert pit.instant_near(vis, "assets", pd.Timestamp("2023-01-11")) is None
    assert pit.instant_near(vis, "assets", pd.Timestamp("2022-12-21")) == pytest.approx(346.747 * B)
    assert pit.instant_near(vis, "assets", pd.Timestamp("2022-12-20")) is None


# ---------------------------------------------------------------------------
# 单季
# ---------------------------------------------------------------------------

def test_quarterly_uses_reported_three_month_windows():
    q = pit.quarterly(_visible("AAPL", "2024-08-28"), "net_income")
    assert q[pd.Timestamp("2023-12-30")] == pytest.approx(33.916 * B)
    assert q[pd.Timestamp("2024-03-30")] == pytest.approx(23.636 * B)
    assert q[pd.Timestamp("2024-06-29")] == pytest.approx(21.448 * B)


def test_fourth_quarter_is_full_year_minus_nine_months():
    """10-K 不单列第四季度。FY2023：96.995 − 74.039（前三季）= 22.956。"""
    q = pit.quarterly(_visible("AAPL", "2023-11-03"), "net_income")
    assert q[pd.Timestamp("2023-09-30")] == pytest.approx(22.956 * B)


def test_four_quarters_sum_to_the_fiscal_year():
    q = pit.quarterly(_visible("NVDA", "2024-02-21"), "net_income")
    ends = [e for e in sorted(q) if pd.Timestamp("2023-02-01") <= e <= pd.Timestamp("2024-01-28")]
    assert len(ends) == 4
    # 2.043 + 6.188 + 9.243 + 12.285 = 29.759：财报各季按百万美元四舍五入，与全年数差 1 个单位
    assert sum(q[e] for e in ends) == pytest.approx(29.760 * B, abs=2e6)
    assert q[ends[-1]] == pytest.approx(12.285 * B)          # Q4 = 全年 29.760 − 前三季 17.475


def test_fourth_quarter_absent_when_nine_month_figure_is_missing():
    rows = [_fact("net_income", "2022-01-01", "2022-12-31", "2023-02-01", 400)]
    assert pit.quarterly(_frame(rows), "net_income") == {}


def test_directly_reported_quarter_is_not_overwritten_by_the_derived_one():
    rows = [
        _fact("net_income", "2022-01-01", "2022-12-31", "2023-02-01", 400),
        _fact("net_income", "2022-01-01", "2022-09-30", "2022-11-01", 290),
        _fact("net_income", "2022-10-01", "2022-12-31", "2023-02-01", 105),   # 直接披露的 Q4
    ]
    assert pit.quarterly(_frame(rows), "net_income")[pd.Timestamp("2022-12-31")] == 105


# ---------------------------------------------------------------------------
# SUE
# ---------------------------------------------------------------------------

def test_sue_matches_an_independent_recomputation():
    vis = _visible("AAPL", "2024-08-02")
    got, end = pit.sue(vis)
    q = pit.quarterly(vis, "net_income")
    ends = sorted(q)

    def year_ago(e):
        return next(p for p in ends if abs((e - p).days - 364) <= 10)

    diffs = [q[e] - q[year_ago(e)] for e in ends[-9:]]
    expected = diffs[-1] / np.std(diffs[:-1], ddof=1)
    assert end == ends[-1] == pd.Timestamp("2024-06-29")
    assert got == pytest.approx(expected)
    assert got == pytest.approx(0.598280, abs=1e-5)


def test_sue_sign_follows_the_year_over_year_change():
    """AAPL FY2024Q4 有一笔一次性税费，净利润同比大降 → SUE 显著为负。"""
    got, _ = pit.sue(_visible("AAPL", "2024-11-01"))
    assert got < -3.0


def test_sue_needs_enough_history():
    """夹具收 2019 年以后披露的文件（带出 2017 年起的比较期）：前 6 份文件凑不齐 6 个同比差。"""
    assert pit.sue(_visible("AAPL", "2020-05-01")) is None
    assert pit.sue(_visible("AAPL", "2020-07-31")) is not None


def _quarters(values, start="2018-01-01"):
    rows, s = [], pd.Timestamp(start)
    for i, v in enumerate(values):
        a = s + pd.DateOffset(months=3 * i)
        b = a + pd.DateOffset(months=3) - pd.Timedelta(days=1)
        rows.append(_fact("net_income", a, b, b + pd.Timedelta(days=30), v, f"q{i}"))
    return _frame(rows)


def test_sue_minimum_history_boundary():
    """要 SUE_MIN=6 个历史同比差：4（基期）+ 6（历史）+ 1（本季）= 11 个季度刚好够。"""
    vals = [10, 11, 12, 13, 12, 14, 13, 16, 15, 17, 30]
    assert pit.sue(_quarters(vals[:10])) is None
    got, _ = pit.sue(_quarters(vals))
    diffs = [vals[i] - vals[i - 4] for i in range(4, 11)]
    assert got == pytest.approx(diffs[-1] / np.std(diffs[:-1], ddof=1))


def test_sue_window_uses_only_the_last_eight_differences():
    vals = [10, 10, 10, 10] + [500, -400, 300, -200] + [11, 13, 12, 15, 14, 16, 13, 17, 16, 19, 15, 21, 40]
    got, _ = pit.sue(_quarters(vals))
    diffs = [vals[i] - vals[i - 4] for i in range(4, len(vals))]
    assert got == pytest.approx(diffs[-1] / np.std(diffs[-9:-1], ddof=1))
    assert got != pytest.approx(diffs[-1] / np.std(diffs[:-1], ddof=1))


def test_sue_none_when_history_has_no_variation():
    assert pit.sue(_quarters([10] * 12)) is None            # 同比差全为 0 → 标准差 0
    assert pit.sue(_frame([])) is None


def test_filing_dates_are_sorted_and_unique():
    d = pit.filing_dates(fx.facts("AAPL"))
    assert d == sorted(set(d)) and len(d) == 31
    assert d[0] == pd.Timestamp("2019-01-30") and d[-1] == pd.Timestamp("2026-07-31")
