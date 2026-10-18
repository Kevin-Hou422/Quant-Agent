"""
外部审查 2026-10-18 F06：正式前向证据可以被改写，也没有策略版本归属。

R12  先写前向 IC 0.1，再以回放写 −0.9 → 读出 −0.9 且仍标前向。
另外：停机多日补拉的 bar 全按 first_new 标为前向；落盘后计算失败、重跑时 forward_from=None，证据丢标；
组合 IC 全在 book 0，→ACTIVE 门按"创建日期之后"筛，数可能来自别的策略。

修复后的契约：前向证据只能是"用结果发生前就存档的信号结算"出来的；写入后不可改（改写留痕）；
按策略 id + 规格哈希记账，→ACTIVE 门只读这份策略自己的记录。
"""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.core.data_engine.market_calendar import session_close_utc
from app.db.alpha_store import AlphaResult, AlphaStore
from app.db.evidence_store import EvidenceStore


def _panel(T=60, N=8, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-03-01", periods=T)
    idx = pd.DatetimeIndex([d for d in idx if session_close_utc(d) is not None])
    cols = [f"S{i}" for i in range(N)]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0003, 0.015, (len(idx), N)), 0), idx, cols)
    amp = np.abs(rng.normal(0, 0.006, close.shape))
    return {"open": close, "high": close * (1 + amp), "low": close * (1 - amp), "close": close,
            "vwap": close, "volume": pd.DataFrame(1e6, idx, cols),
            "returns": close.pct_change().fillna(0.0)}


def _after_close(d, hours=2):
    return session_close_utc(d) + timedelta(hours=hours)


def _upto(ds, d):
    return {k: v.loc[:d] for k, v in ds.items()}


@pytest.fixture
def world(tmp_path):
    from app.core.execution.paper_broker import PaperBroker
    from app.db.position_store import PositionStore
    from app.tasks.daily_trading_loop import DailyTradingLoop
    url = f"sqlite:///{(tmp_path / 'a.db').as_posix()}"
    store, ev = AlphaStore(db_url=url), EvidenceStore(url)
    aid = store.save(AlphaResult(dsl="rank(ts_delta(close, 5))", status="candidate"))
    store.update_status(aid, "validated"); store.update_status(aid, "paper")

    def loop():
        return DailyTradingLoop(store=store, evidence_store=ev,
                                broker=PaperBroker(store=PositionStore(f"sqlite:///{(tmp_path / 'p.db').as_posix()}"),
                                                   initial_capital=10_000.0))
    return SimpleNamespace(store=store, ev=ev, aid=aid, loop=loop, url=url)


def _fwd_dates(store, aid):
    return [pd.Timestamp(h.date) for h in store.get_forward_ic(aid)]


# ---------------------------------------------------------------------------
# R12：前向证据不可改写
# ---------------------------------------------------------------------------

class TestImmutable:

    def test_the_review_r12_replay_no_longer_rewrites_forward_evidence(self, world):
        world.store.record_ic(0, "2026-01-05", .1, is_forward=True)
        world.store.record_ic(0, "2026-01-05", -.9, is_forward=False)
        (e,) = world.store.get_forward_ic(0)
        assert (e.realized_ic, e.is_forward) == (.1, True)
        (rev,) = world.ev.revisions("alpha_ic")
        assert rev.key == "alpha:0" and "ic=0.1" in rev.old_value and "ic=-0.9" in rev.new_value

    def test_an_identical_rewrite_is_silent(self, world):
        world.store.record_ic(0, "2026-01-05", .1, is_forward=True)
        world.store.record_ic(0, "2026-01-05", .1, is_forward=True)
        assert world.ev.revisions() == []

    def test_a_replay_row_can_still_be_upgraded_to_forward(self, world):
        world.store.record_ic(0, "2026-01-05", .3, is_forward=False)
        world.store.record_ic(0, "2026-01-05", .2, is_forward=True)
        (e,) = world.store.get_forward_ic(0)
        assert e.realized_ic == .2

    def test_strategy_evidence_and_archives_are_write_once(self, world):
        ev, d = world.ev, pd.Timestamp("2026-01-06")
        assert ev.record_strategy_ic(5, "h" * 64, d, .2, d - pd.Timedelta(days=1), _after_close(d - pd.Timedelta(days=1)))
        assert not ev.record_strategy_ic(5, "h" * 64, d, -.5, d, _after_close(d))
        assert [e.ic for e in ev.strategy_forward_ic(5)] == [.2]
        assert ev.archive("k", d, pd.Series({"A": 1.0}), recorded_at=_after_close(d))
        assert not ev.archive("k", d, pd.Series({"A": 2.0}), recorded_at=_after_close(d))
        assert [r.kind for r in ev.revisions()] == ["strategy_ic", "signal_archive"]


# ---------------------------------------------------------------------------
# 前向 = 结果发生前就存档的信号
# ---------------------------------------------------------------------------

class TestSignalBeforeOutcome:

    def test_day_by_day_running_produces_forward_evidence(self, world):
        ds = _panel()
        days = ds["close"].index
        for d in days[-4:]:
            world.loop().run(_upto(ds, d), now=_after_close(d))
        assert _fwd_dates(world.store, world.aid) == list(days[-3:])     # 第一天只存档，后三天结算

    def test_the_forward_ic_is_computed_from_the_archived_signal(self, world):
        from app.tasks.daily_trading_loop import _cs_spearman
        ds = _panel()
        days = ds["close"].index
        a, b = days[-2], days[-1]
        world.loop().run(_upto(ds, a), now=_after_close(a))
        arch = world.ev.archived_before(f"alpha:{world.aid}", a, session_close_utc(b).replace(tzinfo=None))
        world.loop().run(ds, now=_after_close(b))
        (e,) = world.store.get_forward_ic(world.aid)
        ret = ds["close"].pct_change().loc[b]
        assert e.realized_ic == pytest.approx(_cs_spearman(arch["signal"].reindex(ret.index).to_numpy(float),
                                                           ret.to_numpy(float)))

    def test_one_run_over_history_marks_nothing_forward_whatever_forward_from_says(self, world):
        """此前：一次补拉的 bar 全按 forward_from 标前向。那些天的信号当时从没产生过。"""
        ds = _panel()
        days = ds["close"].index
        world.loop().run(ds, forward_from=str(days[-20].date()), now=_after_close(days[-1]))
        assert _fwd_dates(world.store, world.aid) == []

    def test_after_downtime_only_the_first_missed_day_is_forward(self, world):
        ds = _panel()
        days = ds["close"].index
        world.loop().run(_upto(ds, days[-5]), now=_after_close(days[-5]))
        world.loop().run(ds, now=_after_close(days[-1]))                  # 停了三天才跑
        assert _fwd_dates(world.store, world.aid) == [days[-4]]

    def test_a_rerun_without_forward_from_keeps_the_evidence(self, world):
        """此前：落盘后计算失败、重跑时 forward_from=None → 证据丢标。"""
        ds = _panel()
        days = ds["close"].index
        world.loop().run(_upto(ds, days[-2]), now=_after_close(days[-2]))
        world.loop().run(ds, forward_from=None, now=_after_close(days[-1]))
        assert _fwd_dates(world.store, world.aid) == [days[-1]]

    def test_a_signal_archived_after_the_outcome_closed_is_not_forward(self, world):
        ds = _panel()
        days = ds["close"].index
        world.loop().run(_upto(ds, days[-2]), now=_after_close(days[-1]))  # 存档晚于下一天收盘
        world.loop().run(ds, now=_after_close(days[-1], hours=3))
        assert _fwd_dates(world.store, world.aid) == []

    def test_the_portfolio_book_follows_the_same_rule(self, world):
        from app.tasks.daily_trading_loop import PORTFOLIO_BOOK_ID
        ds = _panel()
        days = ds["close"].index
        world.loop().run_portfolio(ds, aum=10_000.0, forward_from=str(days[-20].date()),
                                   now=_after_close(days[-1]))
        assert world.store.get_forward_ic(PORTFOLIO_BOOK_ID) == []
        nxt = _panel(T=61)                                              # 同一种子，多一天
        world.loop().run_portfolio(nxt, aum=10_000.0, now=_after_close(nxt["close"].index[-1]))
        assert _fwd_dates(world.store, PORTFOLIO_BOOK_ID) == [nxt["close"].index[-1]]


# ---------------------------------------------------------------------------
# 按策略版本记账
# ---------------------------------------------------------------------------

class TestStrategyVersionEvidence:

    def _propose(self, world, dsl="rank(ts_delta(close, 5))"):
        from app.config import settings
        from app.db.strategy_store import StrategyConfig, StrategyStore
        from tests.conftest import approved_verdict
        f = str(world.aid)
        v = approved_verdict({f: dsl}, {f: 1.0}, 10_000.0)
        return StrategyStore(db_url=settings.database_url).save(
            StrategyConfig(factors=[f], combo_weights={f: 1.0}, aum=10_000.0, status="proposed",
                           verdict=v)), v["spec_hash"]

    def test_a_proposed_strategy_accumulates_its_own_forward_record(self, world, monkeypatch):
        from app.config import settings
        monkeypatch.setattr(settings, "database_url", world.url)
        sid, h = self._propose(world)
        ds = _panel()
        days = ds["close"].index
        for d in days[-3:]:
            world.loop().run_portfolio(_upto(ds, d), aum=10_000.0, now=_after_close(d))
        rows = world.ev.strategy_forward_ic(sid)
        assert [pd.Timestamp(r.date) for r in rows] == list(days[-2:])
        assert {r.spec_hash for r in rows} == {h}
        assert [pd.Timestamp(r.signal_asof) for r in rows] == list(days[-3:-1])

    def test_two_versions_keep_separate_records(self, world, monkeypatch):
        from app.config import settings
        monkeypatch.setattr(settings, "database_url", world.url)
        a, _ = self._propose(world)
        b, _ = self._propose(world, dsl="rank(-ts_delta(close, 20))")
        ds = _panel()
        days = ds["close"].index
        for d in days[-2:]:
            world.loop().run_portfolio(_upto(ds, d), aum=10_000.0, now=_after_close(d))
        ra, rb = world.ev.strategy_forward_ic(a), world.ev.strategy_forward_ic(b)
        assert len(ra) == len(rb) == 1 and ra[0].ic != rb[0].ic

    def test_the_active_gate_reads_only_this_strategys_record(self, world, monkeypatch, test_client):
        from app.config import settings
        import app.dependencies as deps
        from app.db.strategy_store import StrategyStore
        monkeypatch.setattr(settings, "database_url", world.url)
        ss = StrategyStore(db_url=world.url)
        monkeypatch.setattr(deps, "_get_strategy_store_singleton", lambda: ss)
        monkeypatch.setattr(deps, "_get_store_singleton", lambda: world.store)
        from app.tasks.daily_trading_loop import PORTFOLIO_BOOK_ID
        for i, d in enumerate(pd.bdate_range("2024-01-02", periods=80)):       # book 0 有一大把别人的前向 IC
            world.store.record_ic(PORTFOLIO_BOOK_ID, d, 0.2 + 0.01 * (i % 3), is_forward=True)
        sid, h = self._propose(world)
        world.ev.record_strategy_ic(sid, h, pd.Timestamp("2024-06-03"), .05,
                                    pd.Timestamp("2024-05-31"), _after_close(pd.Timestamp("2024-05-31")))
        r = test_client.post(f"/api/strategies/{sid}/approve", json={"activate": True})
        assert r.status_code == 200, r.text
        act = [d for d in ss.get_decisions(sid) if d.decision == "activate"][0]
        assert "n=1 " in act.reason                                  # 此前这里会是 n=80
