"""
test_phase11_forward.py — Phase 11：交易日历 + 回放/前向分离

日历(11.2)
  - 节假日/周末识别、DST 感知的收盘 UTC、半日市、数据↔日历交叉校验
回放/前向分离
  - ic_history 新增 is_forward；旧库能安全迁移（ALTER TABLE ADD COLUMN）
  - 前向标记**只升不降**（回放重跑不会抹掉已积累的前向证据）
  - run_portfolio 按 forward_from 标记；TR.4 →ACTIVE 门只吃前向样本
"""

from __future__ import annotations
import pytest

import numpy as np
import pandas as pd

from app.core.data_engine.market_calendar import (
    is_trading_day, last_trading_day, next_trading_day,
    session_close_utc, minutes_after_close_utc, cross_check, calendar_available,
)


# ── 11.2 交易日历 ────────────────────────────────────────────────────────

def test_calendar_holidays_and_weekends():
    assert calendar_available()
    assert is_trading_day("2026-09-04") is True        # 周五
    assert is_trading_day("2026-09-05") is False       # 周六
    assert is_trading_day("2026-09-07") is False       # 劳动节
    assert last_trading_day("2026-09-07").date().isoformat() == "2026-09-04"
    assert next_trading_day("2026-09-04").date().isoformat() == "2026-09-08"


def test_session_close_is_dst_aware():
    edt = session_close_utc("2026-09-04")              # 夏令时 → 20:00 UTC
    est = session_close_utc("2026-01-15")              # 冬令时 → 21:00 UTC
    assert edt.hour == 20 and est.hour == 21
    assert session_close_utc("2026-09-07") is None     # 非交易日
    # 摄取时点 = 收盘后 N 分钟
    assert minutes_after_close_utc(30, "2026-09-04").hour == 20
    assert minutes_after_close_utc(30, "2026-09-04").minute == 30


def test_half_day_close_detected():
    # 感恩节次日提前收市（13:00 ET = 18:00 UTC）——固定时刻调度必然搞错
    assert session_close_utc("2026-11-27").hour == 18


def test_cross_check_flags_both_directions():
    assert cross_check("2026-09-04", has_bar=True)[0] is True
    assert cross_check("2026-09-04", has_bar=False)[0] is False   # 交易日却没数据
    assert cross_check("2026-09-07", has_bar=True)[0] is False    # 非交易日却有数据


# ── 回放 / 前向分离 ──────────────────────────────────────────────────────

def test_is_forward_column_and_forward_only_query(tmp_path):
    from app.db.alpha_store import AlphaStore
    s = AlphaStore(db_url=f"sqlite:///{tmp_path/'f.db'}")
    s.record_ic(1, "2024-01-02", 0.01, is_forward=False)   # 回放
    s.record_ic(1, "2024-01-03", 0.02, is_forward=True)    # 前向
    s.record_ic(1, "2024-01-04", 0.03, is_forward=True)
    fwd = s.get_forward_ic(1)
    assert [f.date.isoformat() for f in fwd] == ["2024-01-03", "2024-01-04"]
    assert len(s.get_ic_history(1, limit=100)) == 3        # 全量仍是 3 条


def test_forward_flag_is_monotonic(tmp_path):
    """前向标记只升不降：回放重跑不能把已确认的前向证据抹掉。"""
    from app.db.alpha_store import AlphaStore
    s = AlphaStore(db_url=f"sqlite:///{tmp_path/'m.db'}")
    s.record_ic(1, "2024-02-01", 0.05, is_forward=True)
    s.record_ic(1, "2024-02-01", 0.05, is_forward=False)   # 回放重跑
    assert len(s.get_forward_ic(1)) == 1                    # 仍是前向


def test_migration_adds_column_to_legacy_db(tmp_path):
    """旧库（无 is_forward 列）能被安全迁移，且迁移后可用。"""
    import sqlite3
    db = tmp_path / "legacy.db"
    con = sqlite3.connect(db)
    con.execute("""CREATE TABLE alpha_ic_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT, alpha_id INTEGER NOT NULL,
        date DATE NOT NULL, realized_ic FLOAT, realized_return FLOAT, recorded_at DATETIME)""")
    con.execute("INSERT INTO alpha_ic_history (alpha_id,date,realized_ic,realized_return) "
                "VALUES (1,'2024-01-02',0.01,0.0)")
    con.commit(); con.close()

    from app.db.alpha_store import AlphaStore
    s = AlphaStore(db_url=f"sqlite:///{db}")               # __init__ 触发迁移
    con = sqlite3.connect(db)
    cols = {r[1] for r in con.execute("PRAGMA table_info(alpha_ic_history)").fetchall()}
    con.close()
    assert "is_forward" in cols
    s.record_ic(1, "2024-01-03", 0.02, is_forward=True)
    assert len(s.get_forward_ic(1)) == 1                    # 老行默认非前向，新行前向


def test_run_portfolio_marks_forward_from(tmp_path):
    from app.db.alpha_store import AlphaStore, AlphaResult
    from app.db.position_store import PositionStore
    from app.core.execution.paper_broker import PaperBroker
    from app.tasks.daily_trading_loop import DailyTradingLoop, PORTFOLIO_BOOK_ID

    store = AlphaStore(db_url=f"sqlite:///{tmp_path/'a.db'}")
    aid = store.save(AlphaResult(dsl="rank(ts_delta(close,5))", status="candidate"))
    store.update_status(aid, "validated"); store.update_status(aid, "paper")
    broker = PaperBroker(store=PositionStore(db_url=f"sqlite:///{tmp_path/'p.db'}"),
                         initial_capital=10_000.0)

    rng = np.random.default_rng(0)
    T, N = 120, 8
    idx = pd.bdate_range("2024-01-02", periods=T); cols = [f"S{i}" for i in range(N)]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0003, 0.015, (T, N)), 0), idx, cols)
    amp = np.abs(rng.normal(0, 0.006, (T, N)))
    ds = {"open": close, "high": close * (1 + amp), "low": close * (1 - amp), "close": close,
          "vwap": close, "volume": pd.DataFrame(1e6, idx, cols),
          "returns": close.pct_change().fillna(0.0)}

    cutoff = idx[-20].date()                     # 最后 20 天算前向
    DailyTradingLoop(store=store, broker=broker).run_portfolio(
        ds, aum=10_000.0, forward_from=cutoff)

    all_ic = store.get_ic_history(PORTFOLIO_BOOK_ID, limit=5000)
    fwd = store.get_forward_ic(PORTFOLIO_BOOK_ID)
    assert len(all_ic) > len(fwd) > 0                       # 有回放也有前向
    assert all(f.date >= cutoff for f in fwd)               # 前向都在 cutoff 之后


# ---------------------------------------------------------------------------
# CI 事故回归（2026-09-06）：pandas_market_calendars 未在 requirements 声明，
# CI 干净环境缺库 → market_calendar 静默退回 pd.bdate_range → 本文件 4 个用例失败。
# 依赖已声明；此外把"静默降级"本身改为 fail-closed，并在此锁定该行为。
# ---------------------------------------------------------------------------

def test_calendar_missing_library_fails_closed(monkeypatch):
    """
    缺库时必须**抛错**，不得静默退回工作日启发式 ——
    退回后 7/4、12/25 都会被判为交易日，调度会在休市日下单，且没有任何报错。
    """
    import sys, importlib
    import app.core.data_engine.market_calendar as mc
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", None)
    with pytest.raises(mc.CalendarUnavailable):
        mc.is_trading_day("2024-07-04")


def test_calendar_heuristic_requires_explicit_optin(monkeypatch):
    """显式设 calendar_allow_heuristic=True 才允许降级（并会 ERROR 告警）。"""
    import sys
    import app.core.data_engine.market_calendar as mc
    from app.config import settings
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", None)
    monkeypatch.setattr(settings, "calendar_allow_heuristic", True, raising=False)
    # 降级后 7/4 会被误判为交易日 —— 这正是我们默认拒绝它的原因
    assert mc.is_trading_day("2024-07-04") is True


def test_scheduler_does_not_trade_when_calendar_unavailable(monkeypatch, caplog):
    """判不出交易日 → **不交易**（fail-closed），而不是冒出不透明的 job error。"""
    import logging
    from app.tasks import scheduler as sched
    import app.core.data_engine.market_calendar as mc

    def _boom(*a, **k):
        raise mc.CalendarUnavailable("simulated")
    monkeypatch.setattr(mc, "is_trading_day", _boom)

    called = {"n": 0}
    monkeypatch.setattr("app.tasks.daily_ingest.run_daily_pipeline",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    caplog.set_level(logging.ERROR)
    sched.daily_trading_job()
    assert called["n"] == 0, "日历不可用时仍然跑了交易管线"
    assert any("日历" in r.message or "calendar" in r.message.lower()
               for r in caplog.records), "未记录日历不可用的错误"
