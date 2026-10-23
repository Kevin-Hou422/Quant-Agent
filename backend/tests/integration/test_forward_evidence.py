"""
前向证据：只有在标签起点之前存档的信号才算前向；写入后不可改；按策略版本记账；回放保留但不进正式前向。

来源（原测试文件的说明，保留出处）：

外部审查 2026-10-18 F06：正式前向证据可以被改写，也没有策略版本归属。

R12  先写前向 IC 0.1，再以回放写 −0.9 → 读出 −0.9 且仍标前向。
另外：停机多日补拉的 bar 全按 first_new 标为前向；落盘后计算失败、重跑时 forward_from=None，证据丢标；
组合 IC 全在 book 0，→ACTIVE 门按"创建日期之后"筛，数可能来自别的策略。

修复后的契约：前向证据只能是"用结果发生前就存档的信号结算"出来的；写入后不可改（改写留痕）；
按策略 id + 规格哈希记账，→ACTIVE 门只读这份策略自己的记录。

---

复审 2026-10-19 余项（不依赖待决事项的部分）：

§5.1  "尚未统一时钟前，历史结果应分版本标识。"
      → 每一类存下来的结果（研究指标、因子 IC、信号存档、策略证据、规格、提案 verdict、验证门结论、
        正式前向记录）都带时钟版本；时钟一变，规格哈希跟着变（旧证据不会被新时钟的策略继承）。
N03-b "补拉、回放、晚到归档应保留研究价值，但不能升级成正式前向样本。"
      §7.1 工作包 4："将影子信号证据与有实际成交的策略收益分开记录。"
      → 策略证据分 forward / replay 两类：没有及时存档的那几天也记下来（replay，可供研究），
        →ACTIVE 门只读 forward。
N05-d "新规格的资格应来自预登记的验证设计和此后真正未见的前向数据。"
      → 提案时登记验证设计（门槛、t 统计口径、证据口径、登记时刻，带哈希）；→ACTIVE 按**登记的**
        设计判定（之后改 settings 不影响），只算登记之后的前向证据；设计缺失 / 被改 → 不通过。

---

复审 2026-10-19（`review-v6/audit-review/复审报告_2026-10-19.md`）N01–N09 的回归。

每条都从复审独立脚本（`independent_checks.py`）里对应的 residual_* 场景改写：原脚本断言"问题仍在"，
这里断言修复后应有的行为。
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


def _propose(world, dsl="rank(ts_delta(close, 5))"):
    from app.config import settings
    from app.db.strategy_store import StrategyConfig, StrategyStore
    from tests.conftest import approved_verdict
    f = str(world.aid)
    v = approved_verdict({f: dsl}, {f: 1.0}, 10_000.0)
    return StrategyStore(db_url=settings.database_url).save(
        StrategyConfig(factors=[f], combo_weights={f: 1.0}, aum=10_000.0, status="proposed",
                       verdict=v)), v["spec_hash"]


# ---------------------------------------------------------------------------
# R12：前向证据不可改写
# ---------------------------------------------------------------------------

class TestImmutable:

    def test_a_replay_no_longer_rewrites_forward_evidence(self, world):
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
    """
    时钟 c2（2026-10-22 D14 + C3）：信号日 T 的标签 = 开盘(T+1) → 开盘(T+2)，在 T+2 收盘后结算；
    前向要求 T 的信号在开盘(T+1)（标签起点）之前存档 —— 即 T 收盘后那一轮就存下。
    """

    def test_day_by_day_running_produces_forward_evidence(self, world):
        ds = _panel()
        days = ds["close"].index
        for d in days[-4:]:
            world.loop().run(_upto(ds, d), data_source="yahoo", now=_after_close(d))
        # 前两天只存档；days[-2] 结算 days[-4] 的信号、days[-1] 结算 days[-3] 的
        assert _fwd_dates(world.store, world.aid) == list(days[-2:])

    def test_the_forward_ic_is_computed_from_the_archived_signal(self, world):
        from app.tasks.daily_trading_loop import _cs_spearman
        ds = _panel()
        days = ds["close"].index
        from app.core.clock import archive_deadline
        a, b, c = days[-3], days[-2], days[-1]
        world.loop().run(_upto(ds, a), data_source="yahoo", now=_after_close(a))
        arch = world.ev.archived_before(f"alpha:{world.aid}", a, archive_deadline(a).replace(tzinfo=None))
        assert arch is not None, "a 收盘后那一轮没有在标签起点之前存下信号"
        world.loop().run(ds, data_source="yahoo", now=_after_close(c))
        (e,) = world.store.get_forward_ic(world.aid)
        assert pd.Timestamp(e.date) == c
        lab = ds["open"].loc[c] / ds["open"].loc[b] - 1.0                  # 信号 a 的标签：开盘(b) → 开盘(c)
        assert e.realized_ic == pytest.approx(_cs_spearman(arch["signal"].reindex(lab.index).to_numpy(float),
                                                           lab.to_numpy(float)))

    def test_one_run_over_history_marks_nothing_forward_whatever_forward_from_says(self, world):
        """此前：一次补拉的 bar 全按 forward_from 标前向。那些天的信号当时从没产生过。"""
        ds = _panel()
        days = ds["close"].index
        world.loop().run(ds, forward_from=str(days[-20].date()), data_source="yahoo", now=_after_close(days[-1]))
        assert _fwd_dates(world.store, world.aid) == []

    def test_after_downtime_only_the_first_missed_day_is_forward(self, world):
        ds = _panel()
        days = ds["close"].index
        world.loop().run(_upto(ds, days[-5]), data_source="yahoo", now=_after_close(days[-5]))
        world.loop().run(ds, data_source="yahoo", now=_after_close(days[-1]))                  # 停了三天才跑
        # 只有停机前最后存下的那份信号（days[-5]）及时存档；它在 days[-3] 结算
        assert _fwd_dates(world.store, world.aid) == [days[-3]]

    def test_a_rerun_without_forward_from_keeps_the_evidence(self, world):
        """此前：落盘后计算失败、重跑时 forward_from=None → 证据丢标。"""
        ds = _panel()
        days = ds["close"].index
        world.loop().run(_upto(ds, days[-3]), data_source="yahoo", now=_after_close(days[-3]))
        world.loop().run(ds, forward_from=None, data_source="yahoo", now=_after_close(days[-1]))
        assert _fwd_dates(world.store, world.aid) == [days[-1]]

    def test_a_signal_archived_after_the_outcome_closed_is_not_forward(self, world):
        ds = _panel()
        days = ds["close"].index
        # 时钟 c2：days[-3] 的信号在 days[-1] 结算，要求存档早于开盘(days[-2])；这里晚到 days[-2] 收盘之后
        world.loop().run(_upto(ds, days[-3]), data_source="yahoo", now=_after_close(days[-2]))
        world.loop().run(ds, data_source="yahoo", now=_after_close(days[-1], hours=3))
        assert _fwd_dates(world.store, world.aid) == []

    def test_the_portfolio_book_follows_the_same_rule(self, world):
        from app.tasks.daily_trading_loop import PORTFOLIO_BOOK_ID
        ds = _panel()
        days = ds["close"].index
        world.loop().run_portfolio(ds, aum=10_000.0, forward_from=str(days[-20].date()),
                                   data_source="yahoo", now=_after_close(days[-1]))
        assert world.store.get_forward_ic(PORTFOLIO_BOOK_ID) == []
        # 同一种子，多两个交易日（c2：T 在 T+2 结算；按交易日数，不按工作日数 —— 中间可能有假日）
        nxt = next(p for p in (_panel(T=t) for t in range(61, 70)) if len(p["close"]) == len(days) + 2)
        assert nxt["close"].index[-3] == days[-1]
        world.loop().run_portfolio(nxt, aum=10_000.0, data_source="yahoo", now=_after_close(nxt["close"].index[-1]))
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
        for d in days[-4:]:
            world.loop().run_portfolio(_upto(ds, d), aum=10_000.0, data_source="yahoo", now=_after_close(d))
        rows = world.ev.strategy_forward_ic(sid)
        # 时钟 c2：信号日 T 在 T+2 结算
        assert [pd.Timestamp(r.date) for r in rows] == list(days[-2:])
        assert {r.spec_hash for r in rows} == {h}
        assert [pd.Timestamp(r.signal_asof) for r in rows] == list(days[-4:-2])

    def test_two_versions_keep_separate_records(self, world, monkeypatch):
        from app.config import settings
        monkeypatch.setattr(settings, "database_url", world.url)
        a, _ = self._propose(world)
        b, _ = self._propose(world, dsl="rank(-ts_delta(close, 20))")
        ds = _panel()
        days = ds["close"].index
        for d in days[-3:]:                                             # 时钟 c2：T 在 T+2 结算
            world.loop().run_portfolio(_upto(ds, d), aum=10_000.0, data_source="yahoo", now=_after_close(d))
        ra, rb = world.ev.strategy_forward_ic(a), world.ev.strategy_forward_ic(b)
        assert len(ra) == len(rb) == 1 and ra[0].ic != rb[0].ic

    def test_the_active_gate_reads_only_this_strategys_record(self, world, monkeypatch, test_client):
        import app.dependencies as deps
        from app.config import settings
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
        # 2026-10-22 C6：门在 /verify；它数到的只能是这份策略自己的 1 条（此前这里会是 book 0 的 80 条）
        r = test_client.post(f"/api/strategies/{sid}/verify", json={})
        assert r.status_code == 409 and "n=1）" in r.json()["detail"], r.text


# ---------------------------------------------------------------------------
# N03-b：回放保留研究价值，但不进正式前向
# ---------------------------------------------------------------------------

class TestReplayIsKeptButNotForward:

    def test_one_run_over_history_keeps_replay_rows_and_no_forward_rows(self, world, monkeypatch):
        from app.config import settings
        monkeypatch.setattr(settings, "database_url", world.url)
        sid, _ = _propose(world)
        ds = _panel()
        world.loop().run_portfolio(ds, aum=10_000.0, data_source="yahoo", now=_after_close(ds["close"].index[-1]))
        assert world.ev.strategy_forward_ic(sid) == [], "没有及时存档的那几天被算成了前向"
        rep = world.ev.strategy_replay_ic(sid)
        assert rep, "回放那几天的 IC 没有留下（研究价值丢了）"
        assert {r.evidence_class for r in rep} == {"replay"}

    def test_day_by_day_rows_are_forward_and_the_gate_ignores_replay(self, world, monkeypatch):
        from app.config import settings
        monkeypatch.setattr(settings, "database_url", world.url)
        sid, h = _propose(world)
        ds = _panel()
        days = ds["close"].index
        world.loop().run_portfolio(_upto(ds, days[-4]), aum=10_000.0, data_source="yahoo", now=_after_close(days[-4]))
        for d in days[-3:]:
            world.loop().run_portfolio(_upto(ds, d), aum=10_000.0, data_source="yahoo", now=_after_close(d))
        fwd = world.ev.strategy_forward_ic(sid, h)
        # 时钟 c2：days[-2] 结算 days[-4] 的信号，days[-1] 结算 days[-3] 的（都在标签起点之前存档）
        assert [pd.Timestamp(r.date) for r in fwd] == list(days[-2:])
        assert {r.evidence_class for r in fwd} == {"forward"}
        assert world.ev.strategy_replay_ic(sid), "首跑那天之前的回放行没有留下"


# ---------------------------------------------------------------------------
# N03：信号必须在被评分那一天开盘前存档
# ---------------------------------------------------------------------------

class TestArchiveBeforeTheScoredWindow:

    def test_a_signal_archived_during_the_scored_session_is_not_forward(self, fresh_db):
        from app.core.data_engine.market_calendar import (
            session_close_utc,
            session_open_utc,
        )
        from app.db.evidence_store import EvidenceStore
        ev = EvidenceStore()
        before, day = pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")
        ev.archive("strategy:1", before, pd.Series({"AAA": 1., "BBB": 0.}),
                   recorded_at=session_close_utc(day) - timedelta(minutes=1))       # 复审的复现
        assert ev.forward_signal("strategy:1", before, day) is None
        ev.archive("strategy:2", before, pd.Series({"AAA": 1., "BBB": 0.}),
                   recorded_at=session_open_utc(day) + timedelta(minutes=1))        # 开盘后 1 分钟也不行
        assert ev.forward_signal("strategy:2", before, day) is None

    def test_a_signal_archived_before_the_open_is_forward(self, fresh_db):
        from app.core.data_engine.market_calendar import (
            session_close_utc,
            session_open_utc,
        )
        from app.db.evidence_store import EvidenceStore
        ev = EvidenceStore()
        before, day = pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")
        ev.archive("strategy:3", before, pd.Series({"AAA": 1., "BBB": 0.}),
                   recorded_at=session_close_utc(before) + timedelta(hours=2))
        assert ev.forward_signal("strategy:3", before, day) is not None
        assert session_open_utc(day) < session_close_utc(day)
