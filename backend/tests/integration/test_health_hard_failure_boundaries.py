"""
健康门硬性条件的边界（D5 关键路径变异复核：app/core/data_engine/health_report.py::hard_failures）。

复核时存活 3 个：缺失交易日恰好 5 天时的省略号、最低价恰好落在容差边上、最低价恰好为 0。
"""

from __future__ import annotations

import pandas as pd

from app.core.data_engine.health_report import hard_failures

U = ["A", "B"]


def _wide(n=10, low_last=None, open_last=100.0, close_last=100.0):
    idx = pd.bdate_range("2025-03-03", periods=n)
    close = pd.DataFrame(100.0, index=idx, columns=U)
    close.iloc[-1] = close_last
    o, h, lo = close.copy(), close * 1.01, close * 0.99
    o.iloc[-1] = open_last
    if low_last is not None:
        lo.iloc[-1, 0] = low_last
    return {"close": close, "open": o, "high": h, "low": lo,
            "volume": pd.DataFrame(1e6, index=idx, columns=U)}


def test_exactly_five_missing_sessions_are_all_listed_without_an_ellipsis():
    w = _wide(n=10)
    sessions = pd.bdate_range("2025-02-24", periods=15)              # 前 5 个交易日数据里没有
    msg = next(m for m in hard_failures(w, U, expected_sessions=sessions) if "整天没有任何报价" in m)
    assert msg.startswith("5 个交易日") and "…" not in msg
    sessions6 = pd.bdate_range("2025-02-21", periods=16)
    msg6 = next(m for m in hard_failures(w, U, expected_sessions=sessions6) if "整天没有任何报价" in m)
    assert msg6.startswith("6 个交易日") and msg6.endswith("…）")


def test_a_low_exactly_at_the_tolerance_edge_is_accepted():
    edge = pd.concat([pd.Series([100.0]), pd.Series([100.0])], axis=1).min(axis=1).iloc[0] * (1 + 1e-9)
    assert hard_failures(_wide(low_last=edge), U) == []


def test_a_zero_low_is_invalid():
    out = hard_failures(_wide(low_last=0.0), U)
    assert any("OHLC 不合法" in m and "'A'" in m for m in out), out


def test_a_zero_close_is_flagged_whatever_the_other_prices_are():
    """L435 `c <= 0` 等价的机械验证：收盘恰为 0 时，不论开 / 高 / 低取什么，前面几项已经判它不合法。"""
    for o in (-1.0, 0.0, 50.0):
        for lo in (-1.0, 0.0, 1e-12, 50.0):
            for h in (0.0, 50.0, 200.0):
                w = _wide(low_last=lo, open_last=o, close_last=0.0)
                w["high"].iloc[-1, 0] = h
                assert any("OHLC 不合法" in m and "'A'" in m for m in hard_failures(w, U)), (o, lo, h)


PROVEN_EQUIVALENT = {
    "app/core/data_engine/health_report.py ×1 — L435 `(c <= 0)` -> `(c < 0)`":
        "两者只在收盘恰为 0 时不同。此时最低价要么 ≤ 0（被 `low <= 0` 判坏），要么 > 0 ≥ min(开, 0)·(1+tol)"
        "（被 `low > min(开, 收)·(1+tol)` 判坏）；收盘缺失的行最后被 `bad &= c.notna()` 排除，两种写法都一样。"
        "见 test_a_zero_close_is_flagged_whatever_the_other_prices_are。",
}


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 1
