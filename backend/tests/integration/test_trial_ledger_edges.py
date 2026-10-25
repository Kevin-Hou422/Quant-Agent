"""
试验账本 / holdout 计次 / 冻结边界的边界（D5 关键路径变异复核：app/db/trial_ledger.py 的逻辑部分；
表结构与引擎配置见 test_audit_table_schema.py）。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from app.db import trial_ledger as tl

T0 = datetime(2026, 10, 24, 12, 0)


@pytest.fixture
def url(tmp_path, monkeypatch):
    from app.config import settings
    u = f"sqlite:///{(tmp_path / 'tl.db').as_posix()}"
    monkeypatch.setattr(settings, "database_url", u)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    return u


class TestRuns:

    def test_progress_returns_the_running_total(self, url):
        led = tl.TrialLedger(db_url=url)
        led.add(5)
        rid = led.begin_run("gp")
        assert led.add_progress(rid, 3) == 8 == led.total()

    def test_a_stale_run_is_older_than_six_hours_not_exactly_six(self, url):
        from sqlalchemy import update
        led = tl.TrialLedger(db_url=url)
        a, b, c, d = (led.begin_run("gp") for _ in range(4))
        with led._Session() as s:                                  # noqa: SLF001 —— 定死启动时刻
            for rid, started in ((a, T0 - tl.STALE_AFTER), (b, T0 - tl.STALE_AFTER - timedelta(seconds=1)),
                                 (c, T0), (d, T0)):
                s.execute(update(tl.TrialRunState).where(tl.TrialRunState.run_id == rid).values(started_at=started))
            s.commit()
        led.finish_run(c, "incomplete")
        led.finish_run(d, "done")
        assert {r.run_id for r in led.incomplete_runs(now=T0)} == {b, c}
        assert {r.run_id for r in led.open_runs()} == {a, b, c}

    def test_resolving_adds_the_estimate_to_the_run_and_the_total(self, url):
        led = tl.TrialLedger(db_url=url)
        rid = led.begin_run("gp")
        led.add_progress(rid, 4)
        assert led.resolve_run(rid, 6, actor="Kevin-Hou422", reason="进程被杀，按最大代数估") == 10
        row = next(r for r in led.run_states() if r.run_id == rid)
        assert row.n_recorded == 10 and row.status == "resolved"

    def test_recording_nothing_does_not_touch_the_ledger(self, url):
        rec = tl.TrialRecorder("gp")

        class _Broken:
            def add_progress(self, run_id, n):
                raise RuntimeError("写不进去")
        rec.ledger = _Broken()
        rec.add(0)
        assert rec.ok is True and rec.lost == 0
        rec.add(2)
        assert rec.ok is False and rec.lost == 2

    def test_the_problem_text_says_which_runs_are_open_and_why(self, url):
        led = tl.TrialLedger(db_url=url)
        running, leaky = led.begin_run("gp"), led.begin_run("optuna")
        led.finish_run(leaky, "incomplete")
        why = tl.trial_history_problem(now=datetime.utcnow() + timedelta(days=1))
        assert f"{leaky}（有漏记）" in why
        assert f"{running}（疑似僵尸）" in why
        assert f"{running}（进行中）" in tl.trial_history_problem(now=datetime.utcnow())


class TestHoldoutOverlap:

    @pytest.mark.parametrize("other,overlaps", [
        ("2024-01-01..2024-06-30", True),
        ("2024-06-30..2024-12-31", True),           # 首尾在同一天：重叠
        ("2023-07-01..2024-01-01", True),
        ("2024-07-01..2024-12-31", False),
        ("2023-01-01..2023-12-31", False),
        ("empty", False),
    ])
    def test_windows_touching_on_a_day_overlap(self, other, overlaps):
        assert tl._count_overlapping([other], "2024-01-01..2024-06-30") == int(overlaps)


class TestBoundaryRegistration:

    def test_an_actor_without_a_reason_is_not_a_registration(self, url):
        st = tl.BoundaryStore(db_url=url)
        with st._Session() as s:                                   # noqa: SLF001 —— 造一条只有操作人的旧行
            s.add(tl.FrozenBoundary(dataset_key="d", test_start="2024-10-24", frozen_by="legacy", actor="x", reason=""))
            s.commit()
        assert st.status("d")[1] == tl.BOUNDARY_UNCONFIRMED

    def test_a_legacy_boundary_is_confirmed_only_at_its_own_date(self, url):
        st = tl.BoundaryStore(db_url=url)
        st.pin("d", pd.Timestamp("2024-10-24"), frozen_by="legacy")
        with pytest.raises(tl.BoundaryAlreadyRegistered):
            st.register("d", pd.Timestamp("2024-11-01"), actor="Kevin-Hou422", reason="改日期")
        st.register("d", pd.Timestamp("2024-10-24"), actor="Kevin-Hou422", reason="确认旧边界")
        assert st.status("d") == (pd.Timestamp("2024-10-24"), tl.BOUNDARY_REGISTERED)
