"""
test_incremental_ingest_forward.py — Phase 11.1/11.3：真前向增量摄取

- PIT.latest_timestamp 取最新 bar 日期
- 空库 → 历史回填(mode=full, forward_from=None → 全部算回放)
- 已有数据 → 只拉增量(mode=incremental, forward_from=首根新 bar)
- 无新 bar → mode=no_new_bar，不写库不交易
- **只追加增量**：PIT 行数按增量增长，而非每天重写整段（旧实现 1000× 膨胀）
- 调度器非交易日跳过
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _panel(start, periods, N=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=periods)
    cols = [f"S{i}" for i in range(N)]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0003, 0.012, (periods, N)), 0), idx, cols)
    amp = np.abs(rng.normal(0, 0.006, (periods, N)))
    return {"open": close, "high": close * (1 + amp), "low": close * (1 - amp),
            "close": close, "vwap": close, "volume": pd.DataFrame(1e6, idx, cols),
            "returns": close.pct_change().fillna(0.0)}


def _rows(store, name):
    """PIT 中该数据集的总行数（验证是否只追加增量）。"""
    import glob
    n = 0
    for p in glob.glob(str(store.store_dir / name / "year=*/data.parquet")):
        n += len(pd.read_parquet(p, columns=["timestamp"]))
    return n


def test_pit_latest_timestamp(tmp_path):
    from app.core.data_engine.pit_store import PITStore
    s = PITStore(str(tmp_path / "pit"))
    assert s.latest_timestamp("d") is None                  # 空库
    ds = _panel("2024-01-02", 10)
    s.append({"close": ds["close"]}, as_of="2024-01-16T00:00:00", name="d")
    assert s.latest_timestamp("d") == ds["close"].index[-1]


def test_incremental_only_appends_new_rows(tmp_path, monkeypatch):
    """核心：第二次只写增量，PIT 不按整段翻倍（旧实现会 1000× 膨胀）。"""
    from app.core.data_engine.pit_store import PITStore
    s = PITStore(str(tmp_path / "pit"))
    base = _panel("2024-01-02", 20)
    s.append({"close": base["close"]}, as_of="2024-01-30T00:00:00", name="d")
    n1 = _rows(s, "d")

    # 只追加 3 根新 bar
    ext = _panel("2024-01-02", 23)
    inc = {"close": ext["close"].iloc[20:]}
    s.append(inc, as_of="2024-02-02T00:00:00", name="d")
    n2 = _rows(s, "d")

    N = base["close"].shape[1]
    assert n2 - n1 == 3 * N          # 只多了 3 日 × N 只，而不是整段重写
    assert s.latest_timestamp("d") == ext["close"].index[-1]


def test_ingest_incremental_backfill_then_increment(tmp_path, monkeypatch):
    from app.config import settings
    from app.tasks.daily_ingest import DailyIngest
    monkeypatch.setattr(settings, "pit_store_dir", str(tmp_path / "pit"))
    monkeypatch.setattr(settings, "paper_start", "2024-01-02")

    full = _panel("2024-01-02", 30)

    # 用可控的"当前面板"替代 registry 加载：ingest() 返回给定**闭区间**窗口的切片。
    # 这里测的是 ingest_incremental 的游标逻辑；provider 的真实区间语义（yfinance 的 end
    # 是排他的）由 test_forward_daily_e2e.py 用契约替身覆盖 —— 本替身不冒充 Yahoo。
    def fake_ingest(self, name, start, end, append_pit=True, as_of=None, upto=None,
                    pit_required=False):
        from app.tasks.daily_ingest import IngestResult
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        sl = {k: v.loc[(v.index >= s) & (v.index <= e)] for k, v in full.items()}
        if sl["close"].empty:
            return IngestResult(False, name, "as_of", reject_reason="empty_close")
        if append_pit:
            DailyIngest._append_pit(name, sl, f"{start}T00:00:00")
        return IngestResult(True, name, "as_of", health_score=1.0,
                            n_dates=len(sl["close"]), n_tickers=sl["close"].shape[1], dataset=sl)
    monkeypatch.setattr(DailyIngest, "ingest", fake_ingest)
    from app.core.data_engine.market_calendar import session_close_utc
    now = session_close_utc(full["close"].index[-1]) + pd.Timedelta(hours=1)

    ing = DailyIngest()
    r1 = ing.ingest_incremental("d", now=now)             # 空库 → 回填到最近已收盘交易日
    assert r1.accepted and r1.mode == "full"
    assert r1.forward_from is None                        # 回填不算前向证据
    from app.core.data_engine.pit_store import PITStore
    s = PITStore(str(tmp_path / "pit"))
    assert s.latest_timestamp("d") == full["close"].index[-1]

    r2 = ing.ingest_incremental("d", now=now)             # 再跑一次 → 无新 bar
    assert (not r2.accepted) and r2.mode == "no_new_bar"


def test_scheduler_skips_non_trading_day(monkeypatch):
    """非交易日不跑摄取（不空转）。"""
    import app.tasks.scheduler as sch
    called = {"n": 0}
    monkeypatch.setattr("app.tasks.daily_ingest.run_daily_pipeline",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1) or {})
    monkeypatch.setattr("app.core.data_engine.market_calendar.is_trading_day",
                        lambda *a, **k: False)
    sch.daily_trading_job()
    assert called["n"] == 0                                # 非交易日 → 未调用管线
