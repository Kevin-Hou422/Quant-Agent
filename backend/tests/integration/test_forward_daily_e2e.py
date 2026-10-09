"""
test_forward_daily_e2e.py — 前向交易**多日端到端**：从每日管线入口一路走到券商对账

外部审计（2026-10-07）击穿的根因之一：没有任何测试从 `run_daily_pipeline` 出发连跑几个交易日。
单元测试各自绿着，接口之间的语义（数据源的区间、执行层的新鲜度、重跑的幂等）从未在一起被验过。

两个替身都按**真实契约**、默认取**悲观**行为：
  · YahooContract：`yfinance.download` 的 end **排他**（官方文档）；只返回已"发布"的 bar
    —— 收盘后数据源更新有延迟时，当天的 bar 根本拿不到。
    原先 test_incremental_ingest_forward 的替身是"包含结束日"的，和代码犯的是同一个错（审计 F01）。
  · 券商：撤单异步生效（受理后订单仍在途，下一次查询才进入终态）；新单下单后暂不可见。

日程（真实美股日历、注入时钟）：
  D1 首跑 → 回填 → 决策日 D1 → 下单
  D2 开盘成交 → 收盘后增量 → 决策日 D2，新 bar 的 returns 有值（F11）→ 对账成交 → 下单
  D2 同日重跑 → no_new_bar → **仍然**跑组合 + 执行（F02）→ 全部去重，不重复下单
  D3 数据源延迟 → no_new_bar → 执行层判决策日陈旧、不下单、但照样对账 → run-now 退出码非 0（F14）
  D3 补跑（数据到了）→ 决策日 D3 → 下单
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from app.core.data_engine.market_calendar import session_close_utc, trading_days
from tests.unit.execution.moomoo_fake import FakeTradeContext, make_gateway

DATASET = "us_tech_large"
D1, D2, D3 = pd.Timestamp("2025-03-03"), pd.Timestamp("2025-03-04"), pd.Timestamp("2025-03-05")
START = "2024-09-03"
AUM = 100_000.0


class YahooContract:
    """yfinance.download 的契约替身：end **排他**；只返回 ≤ published_through 的 bar。"""

    def __init__(self, tickers, seed=7):
        self.days = trading_days(pd.Timestamp(START), D3 + timedelta(days=10))
        rng = np.random.default_rng(seed)
        n = len(self.days)
        self.close = pd.DataFrame(
            100 * np.cumprod(1 + rng.normal(0.0004, 0.012, (n, len(tickers))), 0),
            index=self.days, columns=tickers)
        self.amp = pd.DataFrame(np.abs(rng.normal(0, 0.006, (n, len(tickers)))),
                                index=self.days, columns=tickers)
        self.published_through = D1
        self.calls = []

    def __call__(self, tickers, start=None, end=None, **kw):
        self.calls.append({"start": start, "end": end})
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        idx = [d for d in self.days if s <= d < e and d <= self.published_through]
        c = self.close.loc[idx, tickers]
        a = self.amp.loc[idx, tickers]
        frames = {"Open": c.shift(1).fillna(c), "High": c * (1 + a), "Low": c * (1 - a),
                  "Close": c, "Volume": pd.DataFrame(2e6, index=c.index, columns=tickers)}
        out = pd.concat(frames, axis=1)
        return out


@pytest.fixture
def world(tmp_path, monkeypatch):
    import yfinance as yf
    from app.config import settings
    from app.core.data_engine.dataset_registry import clear_registry_cache, registry_spec
    import app.core.execution.broker_gateway as bg
    import app.core.execution.order_manager as om

    monkeypatch.delenv("DATABASE_URL", raising=False)
    for k, v in {"database_url": f"sqlite:///{(tmp_path / 'live.db').as_posix()}",
                 "pit_store_dir": str(tmp_path / "pit"), "execution_mode": "moomoo_paper",
                 "paper_aum": AUM, "price_source": "yahoo", "paper_dataset": DATASET,
                 "pm_strategy_gate_block": False}.items():
        monkeypatch.setattr(settings, k, v)
    clear_registry_cache()

    yahoo = YahooContract(list(registry_spec(DATASET).universe))
    monkeypatch.setattr(yf, "download", yahoo)

    ctx = FakeTradeContext(cash=AUM)
    ctx.async_cancel = True
    ctx.hide_new_orders = True

    def poll(c):                       # 撤单在**下一次**查询时才生效
        c.settle_cancels()
    ctx.on_order_read = poll
    monkeypatch.setattr(bg, "make_gateway_from_settings", lambda: make_gateway(ctx))
    snap = lambda tks: pd.DataFrame([{"code": "US.AAPL", "bid_price": 1.0, "ask_price": 1.0}])  # noqa: E731
    monkeypatch.setattr("app.core.trading_context.providers.moomoo_snapshot_fn",
                        lambda host, port: snap)

    clock = {"now": None}
    monkeypatch.setattr(om, "_utcnow", lambda: clock["now"])

    # PAPER 因子 + 一份已批准并激活的策略（冻结等权组合权重）
    from app.db.alpha_store import AlphaResult, AlphaStore
    from app.db.strategy_store import StrategyConfig, StrategyStore
    store = AlphaStore()
    ids = []
    for dsl in ("rank(ts_delta(close, 5))", "rank(-ts_delta(close, 20))"):
        aid = store.save(AlphaResult(dsl=dsl, status="candidate"))
        store.update_status(aid, "validated")
        store.update_status(aid, "paper")
        ids.append(str(aid))
    ss = StrategyStore()
    sid = ss.save(StrategyConfig(factors=ids, combo_weights={i: 0.5 for i in ids}, aum=AUM))
    ss.update_status(sid, "approved")
    ss.update_status(sid, "active")

    def run_day(day, minutes_after_close=90):
        from app.tasks.daily_ingest import run_daily_pipeline
        clock["now"] = session_close_utc(day) + timedelta(minutes=minutes_after_close)
        ctx.now = f"{day:%Y-%m-%d} 16:{min(minutes_after_close, 59):02d}:00.000"
        return run_daily_pipeline(DATASET, START, now=clock["now"])

    def open_session(day):
        """day 开盘：可见所有单，按前一日收盘价开盘撮合（买单限价 = 收盘 × 1.005，必然成交）。"""
        ctx.reveal_orders()
        prev = yahoo.days[yahoo.days.get_loc(day) - 1]
        ctx.now = f"{day:%Y-%m-%d} 09:30:01.000"
        ctx.open_session({tk: float(yahoo.close.loc[prev, tk]) for tk in yahoo.close.columns})
        ctx.mark({tk: float(yahoo.close.loc[day, tk]) for tk in yahoo.close.columns})

    clear = clear_registry_cache
    yield {"yahoo": yahoo, "ctx": ctx, "run_day": run_day, "open_session": open_session,
           "clear": clear, "tmp": tmp_path}
    clear_registry_cache()


def _ex(out):
    return (out.get("portfolio") or {}).get("execution") or {}


def test_a_week_of_forward_trading_end_to_end(world):
    from app.core.data_engine.pit_store import PITStore
    from app.config import settings
    from app.tasks.forward import EXIT_OK, EXIT_RUN_FAILED, classify_run
    yahoo, ctx, run_day = world["yahoo"], world["ctx"], world["run_day"]

    # ---- D1：首跑回填，决策日必须是 D1（审计 F01：以前 end 排他 → 只拿到 D1 的前一天）----
    out = run_day(D1)
    assert out["ingest_accepted"] and out["mode"] == "full"
    assert yahoo.calls[-1]["end"] == "2025-03-04", "没有按 yfinance 的排他语义把 end 推后一天"
    ex = _ex(out)
    assert ex["decision_date"] == "2025-03-03" and ex["blocked"] == "", ex.get("blocked")
    assert ex["n_submitted"] > 0
    assert classify_run(out)[0] == EXIT_OK
    n_d1 = len(ctx.calls_of("place_order"))

    # ---- D2：开盘成交 → 收盘后增量 ----
    world["open_session"](D2)
    yahoo.published_through = D2
    world["clear"]()
    out = run_day(D2)
    assert out["ingest_accepted"] and out["mode"] == "incremental"
    assert out["forward_from"] == "2025-03-04" and out["n_new_bars"] == 1
    assert yahoo.calls[-1]["start"] == "2025-03-03", "增量窗口没有重叠上一根 bar（returns 派生需要）"
    pit = PITStore(settings.pit_store_dir).load_pit(name=DATASET)
    r = pit["returns"].loc[D2].dropna()
    expect = np.log(pit["close"].loc[D2] / pit["close"].loc[D1]).reindex(r.index)
    assert len(r) == pit["close"].shape[1] and np.allclose(r, expect), "新 bar 的 returns 是 NaN（F11）"
    ex = _ex(out)
    assert ex["decision_date"] == "2025-03-04" and ex["blocked"] == "", ex.get("blocked")
    assert ex["reconcile"]["status"] == "clean" and ex["reconcile"]["fills"], "D1 的成交没有对账回来"
    n_d2 = len(ctx.calls_of("place_order"))
    assert n_d2 > n_d1

    # ---- D2 同日重跑：无新 bar 也照样跑执行（F02），全部去重 ----
    ctx.reveal_orders()
    world["clear"]()
    out = run_day(D2, minutes_after_close=150)
    assert out["reason"] == "no_new_bar"
    ex = _ex(out)
    # 已发出的单全部去重；首跑被风控拒掉的（可重试）重跑时照样再过一遍风控、照样被拒
    assert ex["blocked"] == "" and ex["n_submitted"] == 0 and ex["n_duplicate"] > 0
    assert ex["n_duplicate"] + ex["n_gate_rejected"] == ex["n_planned"]
    assert len(ctx.calls_of("place_order")) == n_d2, "同日重跑重复下单"
    assert classify_run(out)[0] == EXIT_OK

    # ---- D3：数据源延迟 → 不下单，但照样对账；run-now 判失败（F14）----
    world["open_session"](D3)
    world["clear"]()
    out = run_day(D3)
    assert out["reason"] == "no_new_bar"
    ex = _ex(out)
    assert ex["blocked"] == "stale_decision_date", ex.get("blocked")
    assert ex["reconcile"]["status"] in ("clean", "baseline") and ex["reconcile"]["fills"]
    assert len(ctx.calls_of("place_order")) == n_d2
    code, verdict = classify_run(out)
    assert code == EXIT_RUN_FAILED and "stale_decision_date" in verdict

    # ---- D3 补跑：数据到了 ----
    yahoo.published_through = D3
    world["clear"]()
    out = run_day(D3, minutes_after_close=165)
    assert out["ingest_accepted"] and out["forward_from"] == "2025-03-05"
    ex = _ex(out)
    assert ex["decision_date"] == "2025-03-05" and ex["blocked"] == "", ex.get("blocked")
    assert classify_run(out)[0] == EXIT_OK


def test_an_unclosed_session_bar_is_never_ingested(world):
    """盘中跑（收盘前）：最近已收盘交易日是前一天，当天的半根 bar 即使数据源给了也必须丢弃。"""
    from app.core.data_engine.pit_store import PITStore
    from app.config import settings
    yahoo = world["yahoo"]
    yahoo.published_through = D2                       # 数据源"已经"有 D2（盘中半根）
    out = world["run_day"](D2, minutes_after_close=-120)
    assert out["ingest_accepted"]
    last = PITStore(settings.pit_store_dir).latest_timestamp(DATASET)
    assert last == D1, f"盘中的 D2 半根 bar 进了 PIT：{last}"
    assert _ex(out)["decision_date"] == "2025-03-03"


def test_ingest_rejection_still_reconciles_and_continues_a_flatten(world, monkeypatch):
    """审计 F02：摄取被拒（数据源挂了）以前直接 return —— 熔断后的续做全平也跟着停了。"""
    import yfinance as yf
    from app.db.execution_store import ExecutionStore
    out = world["run_day"](D1)
    assert _ex(out)["n_submitted"] > 0
    world["open_session"](D2)
    ExecutionStore().set_kill_switch(True, "kevin", "drill")

    def down(*a, **k):
        raise ConnectionError("yahoo down")
    monkeypatch.setattr(yf, "download", down)
    world["clear"]()
    ctx = world["ctx"]
    n = len(ctx.calls_of("place_order"))
    out = world["run_day"](D2)
    assert out["ingest_accepted"] is False and "load_failed" in out["reject_reason"]
    ex = _ex(out)
    assert ex["blocked"].startswith("maintenance_only: ingest_rejected")
    assert ex["reconcile"]["fills"], "摄取失败的日子没有对账"
    sells = [p for p in ctx.calls_of("place_order")[n:] if p["trd_side"] == "SELL"]
    assert sells, "熔断开启、摄取失败时没有续做全平"


def test_without_execution_no_broker_work_happens_on_rejections(tmp_path, monkeypatch):
    from app.config import settings
    from app.tasks import daily_ingest as di
    from app.tasks.daily_trading_loop import DailyTradingLoop
    monkeypatch.setattr(settings, "execution_mode", "off")
    calls = []
    monkeypatch.setattr(DailyTradingLoop, "_maintain_live", lambda self, r: calls.append(r))
    monkeypatch.setattr(di.DailyIngest, "ingest_incremental", lambda *a, **k: di.IngestResult(
        False, "d", "t", reject_reason="no_new_bar", mode="no_new_bar"))
    assert di.run_daily_pipeline("d")["reason"] == "no_new_bar" and calls == []
    monkeypatch.setattr(di.DailyIngest, "ingest_incremental", lambda *a, **k: di.IngestResult(
        False, "d", "t", reject_reason="load_failed: x"))
    out = di.run_daily_pipeline("d")
    assert "portfolio" not in out and calls == []


def test_a_portfolio_crash_still_maintains_the_broker(monkeypatch):
    from app.config import settings
    from app.tasks import daily_ingest as di
    from app.tasks.daily_trading_loop import DailyTradingLoop
    monkeypatch.setattr(settings, "execution_mode", "moomoo_paper")
    calls = []

    class Loop:
        def run_portfolio(self, ds, forward_from=None):
            raise RuntimeError("pm exploded")

        def _maintain_live(self, reason):
            calls.append(reason)
            return {"blocked": f"maintenance_only: {reason}"}

    pf = di._run_portfolio_safely(Loop(), {}, None)
    assert pf["error"] == "pm exploded" and calls == ["portfolio_error: pm exploded"]
    assert pf["execution"]["blocked"].startswith("maintenance_only")
    assert DailyTradingLoop is not None


# ---------------------------------------------------------------------------
# 逐点变异复核补的用例
# ---------------------------------------------------------------------------

def test_first_backfill_is_rejected_when_pit_cannot_be_written(world, monkeypatch):
    """首次回填写 PIT 失败 → 拒绝：没落盘的数据不能成为前向记录的起点（审计 F12 末段）。"""
    from app.tasks.daily_ingest import DailyIngest

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(DailyIngest, "_append_pit", staticmethod(boom))
    out = world["run_day"](D1)
    assert out["ingest_accepted"] is False and "pit_append_failed" in out["reject_reason"]


def test_the_overlapping_bar_is_not_rewritten_as_a_new_vintage(world):
    """增量窗口重叠的那一根旧 bar 只用来派生 returns，不得以新的 as_of 再写进 PIT（缺陷 A-1）。"""
    from app.core.data_engine.pit_store import PITStore
    from app.config import settings
    world["run_day"](D1)
    world["yahoo"].published_through = D2
    world["clear"]()
    world["run_day"](D2)
    long = PITStore(settings.pit_store_dir)._load_long(DATASET)
    vintages = long.groupby("timestamp")["as_of"].nunique()
    assert vintages.loc[D1] == 1 and vintages.loc[D2] == 1, vintages.tail(3).to_dict()


def test_no_new_bar_with_an_unreadable_panel_only_maintains(monkeypatch):
    from app.config import settings
    from app.core.data_engine.pit_store import PITStore
    from app.tasks import daily_ingest as di
    from app.tasks.daily_trading_loop import DailyTradingLoop
    monkeypatch.setattr(settings, "execution_mode", "moomoo_paper")
    monkeypatch.setattr(di.DailyIngest, "ingest_incremental", lambda *a, **k: di.IngestResult(
        False, "d", "t", reject_reason="no_new_bar", mode="no_new_bar"))
    monkeypatch.setattr(PITStore, "load_pit", lambda self, **k: {})
    calls = []
    monkeypatch.setattr(DailyTradingLoop, "__init__", lambda self, *a, **k: None)
    monkeypatch.setattr(DailyTradingLoop, "_maintain_live", lambda self, r: calls.append(r) or {})
    out = di.run_daily_pipeline("d")
    assert calls == ["no_data"] and out["portfolio"] == {"execution": {}}


def test_bars_after_the_last_closed_session_are_dropped_even_if_the_source_returns_them(
        tmp_path, monkeypatch, caplog):
    """
    数据源不守区间（把盘中的半根 bar 也返回了）：摄取必须按"最近已收盘交易日"截断，
    并在日志里列出丢掉的日期。契约替身守规矩，所以这道防线要用一个**不守规矩**的替身来测。
    """
    import logging
    from types import SimpleNamespace
    import app.core.data_engine.dataset_registry as dr
    from app.tasks.daily_ingest import DailyIngest
    from tests.conftest import _make_dataset
    data = _make_dataset(n_days=12, n_tickers=4)
    idx = data["close"].index
    monkeypatch.setattr(dr, "load_registry_dataset", lambda *a, **k: SimpleNamespace(data=data))
    monkeypatch.setattr(dr, "check_dataset_health", lambda *a, **k: None)
    cut = idx[-3]
    with caplog.at_level(logging.WARNING, logger="app.tasks.daily_ingest"):
        r = DailyIngest().ingest("d", str(idx[0].date()), str(cut.date()), append_pit=False,
                                 upto=cut)
    assert r.accepted and r.dataset["close"].index.max() == cut
    assert all(df.index.max() == cut for df in r.dataset.values())
    msg = [m.getMessage() for m in caplog.records if "丢弃" in m.getMessage()]
    # 恰好 2 根（截止日本身不算"晚于"）；列出的正是那两天
    assert len(msg) == 1 and msg[0].startswith("[daily_ingest] 丢弃 2 根")
    assert str(idx[-1].date()) in msg[0] and str(idx[-2].date()) in msg[0]
