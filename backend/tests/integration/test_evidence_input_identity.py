"""
前向证据的输入身份：实际数据源可验证（未知 / 不一致 → 隔离，不退回配置）、规格与运行时核对、数据摘要 / 股票池 / vintage / 代码版本。

来源（原测试文件的说明，保留出处）：

2026-10-22 用户决定 · 批 1：正确性与状态（`docs/DECISION_IMPLEMENTATION_TRACKER.md` C1 / C2 / C4 / C5 / C6）。

每条断言都对着用户的原话写；先跑、先看它们在旧代码上失败，再改。

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

from tests.conftest import _make_dataset

# ---------------------------------------------------------------------------
# C2：实际数据源未知 → 不退回 settings；正式证据隔离
# ---------------------------------------------------------------------------

class TestUnknownPriceSource:

    def test_the_loop_identity_does_not_claim_the_settings_source(self):
        from app.tasks.daily_trading_loop import DailyTradingLoop
        loop = DailyTradingLoop.__new__(DailyTradingLoop)
        ident = loop._set_identity(_make_dataset(30, 5), None, None)
        assert ident["price_source"] == "" and ident["source_verified"] is False

    def test_the_pipeline_identity_does_not_claim_the_settings_source(self):
        from app.tasks.daily_ingest import _pipeline_identity
        ing = SimpleNamespace(identity={"price_source": "yahoo", "last_day": "2025-03-04"})
        ident = _pipeline_identity(ing, None)
        assert ident["price_source"] == "" and ident["source_verified"] is False
        assert _pipeline_identity(ing, "moomoo")["source_verified"] is True

    def test_unverified_source_days_are_replay_not_forward(self, tmp_path, monkeypatch):
        from app.config import settings
        from app.core.execution.paper_broker import PaperBroker
        from app.db.alpha_store import AlphaResult, AlphaStore
        from app.db.evidence_store import EvidenceStore
        from app.db.position_store import PositionStore
        from app.tasks.daily_trading_loop import DailyTradingLoop
        from tests.integration.test_forward_evidence import _after_close, _panel, _upto
        from tests.integration.test_forward_evidence import _propose as propose_strategy
        url = f"sqlite:///{(tmp_path / 'a.db').as_posix()}"
        monkeypatch.setattr(settings, "database_url", url)
        store, ev = AlphaStore(db_url=url), EvidenceStore(url)
        aid = store.save(AlphaResult(dsl="rank(ts_delta(close, 5))", status="candidate"))
        store.update_status(aid, "validated"); store.update_status(aid, "paper")
        sid, _ = propose_strategy(SimpleNamespace(aid=aid))

        def loop():
            return DailyTradingLoop(store=store, evidence_store=ev, broker=PaperBroker(
                store=PositionStore(f"sqlite:///{(tmp_path / 'p.db').as_posix()}"), initial_capital=10_000.0))
        ds = _panel()
        days = ds["close"].index
        for d in days[-3:]:
            loop().run_portfolio(_upto(ds, d), aum=10_000.0, now=_after_close(d))     # 来源未知
        assert ev.strategy_forward_ic(sid) == [], "来源未知的日子被记成了合格前向"
        assert ev.strategy_replay_ic(sid), "探索仍应可继续（记 replay）"


# ---------------------------------------------------------------------------
# N02：证据路径也核对规格与真实数据源；无新 bar 时换源也重建
# ---------------------------------------------------------------------------

class TestEvidenceIdentity:

    def _setup(self, fresh_db):
        from app.core.execution.paper_broker import PaperBroker
        from app.core.portfolio_manager.executable_spec import build_spec, spec_hash
        from app.db.alpha_store import AlphaStore
        from app.db.evidence_store import EvidenceStore
        from app.db.position_store import PositionStore
        from app.db.strategy_store import StrategyConfig, StrategyStore
        from app.tasks.daily_trading_loop import DailyTradingLoop
        spec = build_spec({"1": "rank(close)"}, {"1": 1.}, "equal_weight", 10000, .01)
        sid = StrategyStore().save(StrategyConfig(["1"], {"1": 1.}, 10000, method="equal_weight",
                                                  verdict={"executable_spec": spec, "spec_hash": spec_hash(spec)}))
        ev = EvidenceStore()
        loop = DailyTradingLoop(store=AlphaStore(), evidence_store=ev,
                                broker=PaperBroker(PositionStore(fresh_db), initial_capital=10000))
        return spec, sid, ev, loop

    def test_a_source_mismatch_is_quarantined_not_counted(self, fresh_db, monkeypatch):
        from app.config import settings
        from app.core.portfolio_manager.executable_spec import spec_hash
        from tests.integration.test_forward_evidence import _after_close, _panel, _upto
        spec, sid, ev, loop = self._setup(fresh_db)
        ds = _panel()
        a, b = ds["close"].index[-3], ds["close"].index[-1]          # 时钟 c2：信号日 a 在 a+2 结算
        loop._record_strategy_evidence(_upto(ds, a), _after_close(a))
        monkeypatch.setattr(settings, "price_source", "moomoo")
        out = loop._record_strategy_evidence(ds, _after_close(b))
        assert ev.strategy_forward_ic(sid, spec_hash(spec)) == []
        assert out[f"strategy:{sid}"] == -1
        (q,) = ev.revisions("strategy_evidence_quarantine")
        assert "data.price_source" in q.new_value

    def test_the_actual_pit_source_wins_over_what_settings_claim(self, fresh_db):
        from app.core.portfolio_manager.executable_spec import spec_hash
        from tests.integration.test_forward_evidence import _after_close, _panel, _upto
        spec, sid, ev, loop = self._setup(fresh_db)
        ds = _panel()
        a, b = ds["close"].index[-3], ds["close"].index[-1]          # 时钟 c2：信号日 a 在 a+2 结算
        loop._record_strategy_evidence(_upto(ds, a), _after_close(a), data_source="yahoo")
        loop._record_strategy_evidence(ds, _after_close(b), data_source="moomoo")   # settings 仍是 yahoo
        assert ev.strategy_forward_ic(sid, spec_hash(spec)) == []

    def test_a_source_switch_without_a_new_bar_rebuilds_the_pit(self, tmp_path, monkeypatch):
        import app.core.data_engine.dataset_registry as reg
        from app.config import settings
        from app.core.data_engine.pit_store import PITStore
        from app.tasks.daily_ingest import DailyIngest
        from tests.integration.test_adjusted_price_rebase import (
            DAYS,
            Provider,
            _close_utc,
        )
        monkeypatch.setattr(settings, "pit_store_dir", str(tmp_path / "pit"))
        monkeypatch.setattr(settings, "paper_start", str(DAYS[0].date()))
        monkeypatch.setattr(settings, "price_source", "moomoo")
        prov = Provider()
        monkeypatch.setattr(reg, "load_registry_dataset", prov.load)
        monkeypatch.setattr(reg, "check_dataset_health",
                            lambda ds, min_score=0.7, warn_only=True: SimpleNamespace(overall_score=1.0, hard_failures=[]))
        day = pd.Timestamp("2024-03-19")
        assert DailyIngest().ingest_incremental("px", now=_close_utc(day)).mode == "full"
        monkeypatch.setattr(settings, "price_source", "yahoo")
        res = DailyIngest().ingest_incremental("px", now=_close_utc(day) + timedelta(hours=1))
        assert res.accepted and res.mode == "rebased" and res.forward_from is None
        meta = PITStore(tmp_path / "pit").read_meta("px")
        assert meta["price_source"] == "yahoo" and "无新 bar" in meta["last_rebase_reason"]


# ---------------------------------------------------------------------------
# N02-c/d + N04-b：输入身份（数据摘要、有效股票池、价格源、数据 vintage、代码版本）
# ---------------------------------------------------------------------------

class TestInputIdentity:

    def test_identity_changes_with_the_data_and_the_universe(self):
        from app.core.data_engine.input_identity import identify
        ds = _make_dataset(60, 8)
        a = identify(ds, price_source="moomoo", vintage="2026-01-05T22:00:00Z")
        changed = {k: v.copy() for k, v in ds.items()}
        changed["close"].iloc[10, 2] *= 1.01                              # 历史中段改一个数
        fewer = {k: v.copy() for k, v in ds.items()}
        fewer["close"].iloc[-1, 0] = np.nan                               # 末日少一只票
        assert identify(changed, price_source="moomoo").data_digest != a.data_digest
        assert identify(fewer, price_source="moomoo").universe_digest != a.universe_digest
        assert identify(fewer, price_source="moomoo").universe_n == a.universe_n - 1
        assert a.price_source == "moomoo" and a.vintage == "2026-01-05T22:00:00Z" and a.last_day == str(ds["close"].index[-1].date())
        assert a.code_version != ""

    def test_strategy_evidence_rows_carry_the_input_identity(self, fresh_db):
        from app.core.portfolio_manager.executable_spec import spec_hash
        from tests.integration.test_forward_evidence import _after_close, _panel, _upto
        spec, sid, ev, loop = TestEvidenceIdentity()._setup(fresh_db)
        ds = _panel()
        a, b = ds["close"].index[-3], ds["close"].index[-1]          # 时钟 c2：信号日 a 在 a+2 结算
        loop._record_strategy_evidence(_upto(ds, a), _after_close(a), data_source="yahoo")
        loop._record_strategy_evidence(ds, _after_close(b), data_source="yahoo")
        (row,) = ev.strategy_forward_ic(sid, spec_hash(spec))
        assert row.price_source == "yahoo" and len(row.data_digest) == 64
        assert row.universe_n == ds["close"].shape[1] and row.code_version != ""
        assert row.signal_digest and row.signal_digest != row.data_digest   # 决策时的输入 ≠ 结算时的输入

    def test_a_signal_archived_under_another_price_source_does_not_settle_as_forward(self, tmp_path):
        """N02-d：不同数据源的历史与前向证据不能混用 —— 不靠规格，账本层面就拦住。"""
        from tests.integration.test_forward_evidence import (
            _after_close,
            _fwd_dates,
            _panel,
            _upto,
            world,
        )
        w = world.__wrapped__(tmp_path)
        ds = _panel()
        # 时钟 c2：信号日 a 在 a+2 结算 —— 只隔一天的话无论来源如何都不会有前向，本用例就空转了
        a, b = ds["close"].index[-3], ds["close"].index[-1]
        w.loop().run(_upto(ds, a), now=_after_close(a), data_source="yahoo")
        w.loop().run(ds, now=_after_close(b), data_source="moomoo")
        assert _fwd_dates(w.store, w.aid) == []

    def test_the_ingest_reports_the_identity_of_the_panel_it_hands_to_trading(self, tmp_path, monkeypatch):
        import app.core.data_engine.dataset_registry as reg
        from app.config import settings
        from app.tasks.daily_ingest import DailyIngest
        from tests.integration.test_adjusted_price_rebase import (
            DAYS,
            Provider,
            _close_utc,
        )
        monkeypatch.setattr(settings, "pit_store_dir", str(tmp_path / "pit"))
        monkeypatch.setattr(settings, "paper_start", str(DAYS[0].date()))
        monkeypatch.setattr(settings, "price_source", "moomoo")
        monkeypatch.setattr(reg, "load_registry_dataset", Provider().load)
        monkeypatch.setattr(reg, "check_dataset_health",
                            lambda ds, min_score=0.7, warn_only=True: SimpleNamespace(overall_score=1.0, hard_failures=[]))
        DailyIngest().ingest_incremental("px", now=_close_utc(pd.Timestamp("2024-03-19")))
        res = DailyIngest().ingest_incremental("px", now=_close_utc(pd.Timestamp("2024-03-20")))
        ident = res.identity
        assert ident["last_day"] == "2024-03-20" and ident["price_source"] == "moomoo"
        assert ident["universe_n"] == 2 and ident["vintage"] == res.as_of
        assert ident["checked_last_day"] == ident["last_day"]                # 检查的末日 = 交出去的末日

    def test_a_consumed_panel_that_does_not_end_where_the_health_check_ended_is_refused(self, tmp_path, monkeypatch):
        import app.core.data_engine.dataset_registry as reg
        from app.config import settings
        from app.core.data_engine.pit_store import PITStore
        from app.tasks.daily_ingest import DailyIngest
        from tests.integration.test_adjusted_price_rebase import (
            DAYS,
            Provider,
            _close_utc,
        )
        monkeypatch.setattr(settings, "pit_store_dir", str(tmp_path / "pit"))
        monkeypatch.setattr(settings, "paper_start", str(DAYS[0].date()))
        monkeypatch.setattr(settings, "price_source", "moomoo")
        monkeypatch.setattr(reg, "load_registry_dataset", Provider().load)
        monkeypatch.setattr(reg, "check_dataset_health",
                            lambda ds, min_score=0.7, warn_only=True: SimpleNamespace(overall_score=1.0, hard_failures=[]))
        DailyIngest().ingest_incremental("px", now=_close_utc(pd.Timestamp("2024-03-19")))
        real = PITStore.load_pit
        monkeypatch.setattr(PITStore, "load_pit",                           # PIT 读回来少了最后一天
                            lambda self, *a, **k: {f: df.iloc[:-1] for f, df in real(self, *a, **k).items()})
        res = DailyIngest().ingest_incremental("px", now=_close_utc(pd.Timestamp("2024-03-20")))
        assert res.accepted is False and res.reject_reason.startswith("consumed_panel_mismatch")

    def test_old_evidence_tables_get_the_new_columns(self, tmp_path):
        import sqlite3
        db = tmp_path / "old.db"
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE signal_archive (id INTEGER PRIMARY KEY, book_key VARCHAR(64) NOT NULL, asof DATE NOT NULL,"
                    " spec_hash VARCHAR(64) NOT NULL, payload TEXT NOT NULL, recorded_at DATETIME NOT NULL)")
        con.execute("CREATE TABLE strategy_evidence (id INTEGER PRIMARY KEY, strategy_id INTEGER NOT NULL, spec_hash VARCHAR(64) NOT NULL,"
                    " date DATE NOT NULL, ic FLOAT NOT NULL, signal_asof DATE NOT NULL, decision_ts DATETIME NOT NULL,"
                    " recorded_at DATETIME NOT NULL)")
        con.commit(); con.close()
        from app.db.evidence_store import EvidenceStore
        EvidenceStore(f"sqlite:///{db.as_posix()}")
        con = sqlite3.connect(db)
        cols = {r[1] for r in con.execute("PRAGMA table_info(strategy_evidence)")}
        arch = {r[1] for r in con.execute("PRAGMA table_info(signal_archive)")}
        assert {"price_source", "data_digest", "universe_digest", "universe_n", "vintage", "code_version",
                "signal_digest"} <= cols
        assert {"price_source", "data_digest", "universe_digest", "universe_n", "vintage", "code_version"} <= arch
