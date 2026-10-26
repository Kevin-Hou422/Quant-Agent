"""
唯一运行槽（2026-10-25 审计 R7）。

审计原话："`_start_steps` 在提交迁移事务之前查询待退役策略。两个激活请求都先读到'没有运行策略'，再分别执行各自事务，
就会同时留下两份 paper_running。……latest_running 随后只选一份，掩盖了状态不变量破坏。"
验收："两个请求的交错、独立连接/进程并发、崩溃回滚；任何时刻同一账户/book 的实际运行所有权唯一，冲突明确返回，
不得静默选最新记录。"integration、paper_running、verified_active 及旧 active 都覆盖。
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from app.db.strategy_store import (
    SLOT_STATUSES,
    RunningSlotConflict,
    StrategyConfig,
    StrategyStore,
)

BACKEND = Path(__file__).resolve().parents[2]


def _store(tmp_path, n=3, statuses=None):
    url = f"sqlite:///{(tmp_path / 'alphas.db').as_posix()}"
    st = StrategyStore(db_url=url)
    ids = [st.save(StrategyConfig(factors=["1"], combo_weights={"1": 1.0}, aum=1e4, name=f"s{i}",
                                  status="approved")) for i in range(n)]
    return st, ids, url


def _running(db: Path):
    con = sqlite3.connect(db)
    try:
        return [r[0] for r in con.execute(
            "SELECT id FROM strategy_configs WHERE status IN (%s) ORDER BY id" % ",".join("?" * len(SLOT_STATUSES)),
            SLOT_STATUSES)]
    finally:
        con.close()


def test_the_slot_statuses_are_the_running_states_plus_legacy_active():
    from app.core.lifecycle.strategy_states import LEGACY_ACTIVE, RUNNING_STATES
    assert set(SLOT_STATUSES) == set(RUNNING_STATES) | {LEGACY_ACTIVE}


def test_the_audit_interleaving_now_fails_loudly_instead_of_leaving_two_runners(tmp_path):
    """审计的交错：两个请求都在事务外读到"没有运行策略"，各自只迁移自己 —— 第二个被数据库约束拒绝。"""
    st, (a, b, _), _ = _store(tmp_path)
    stale_a = [(a, "paper_running", "start_paper_running", "r", "x")]      # 两边都没有要退役的
    stale_b = [(b, "paper_running", "start_paper_running", "r", "x")]
    st.apply_transitions(stale_a)
    with pytest.raises(RunningSlotConflict):
        st.apply_transitions(stale_b)
    assert _running(tmp_path / "alphas.db") == [a]


def test_a_second_running_row_written_by_raw_sql_is_rejected_by_the_database(tmp_path):
    st, (a, b, _), _ = _store(tmp_path)
    st.start_running(a, "integration", "r", "x")
    con = sqlite3.connect(tmp_path / "alphas.db")
    try:
        with pytest.raises(sqlite3.IntegrityError):
            con.execute("UPDATE strategy_configs SET status = 'verified_active' WHERE id = ?", (b,))
    finally:
        con.close()


@pytest.mark.parametrize("first,second", [("integration", "paper_running"), ("paper_running", "integration"),
                                          ("paper_running", "paper_running")])
def test_starting_retires_whoever_holds_the_slot_in_the_same_transaction(tmp_path, first, second):
    st, (a, b, _), _ = _store(tmp_path)
    assert st.start_running(a, first, "r", "x") == []
    assert st.start_running(b, second, "r", "x") == [a]
    assert _running(tmp_path / "alphas.db") == [b]
    assert [d.decision for d in st.get_decisions(a)][-1] == "retire"


def test_a_verified_strategy_is_retired_when_another_starts(tmp_path):
    st, (a, b, _), _ = _store(tmp_path)
    st.start_running(a, "paper_running", "r", "x")
    st.apply_transitions([(a, "verified_active", "verify", "r", "x")])          # 同一行在槽内换状态：允许
    assert st.start_running(b, "paper_running", "r", "x") == [a]
    assert str(st.get(a).status) == "retired"


def test_concurrent_starts_from_threads_leave_exactly_one_runner(tmp_path):
    _st, ids, url = _store(tmp_path, n=6)
    barrier = threading.Barrier(len(ids))
    errors = []

    def go(sid):
        s = StrategyStore(db_url=url)                                       # 各自的连接池
        barrier.wait()
        try:
            s.start_running(sid, "paper_running", "r", f"t{sid}")
        except RunningSlotConflict as exc:                                  # 明确返回的冲突是允许的结果
            errors.append(str(exc))
        finally:
            s._engine.dispose()                                             # noqa: SLF001
    ts = [threading.Thread(target=go, args=(i,)) for i in ids]
    for t in ts:
        t.start()
    for t in ts:
        t.join(60)
    assert len(_running(tmp_path / "alphas.db")) == 1


def test_concurrent_starts_from_separate_processes_leave_exactly_one_runner(tmp_path):
    _st, ids, url = _store(tmp_path, n=4)
    code = ("import sys; from app.db.strategy_store import StrategyStore, RunningSlotConflict\n"
            "s = StrategyStore(db_url=sys.argv[1])\n"
            "try:\n    s.start_running(int(sys.argv[2]), 'paper_running', 'r', 'proc')\n"
            "except RunningSlotConflict:\n    sys.exit(3)\n")
    procs = [subprocess.Popen([sys.executable, "-c", code, url, str(i)], cwd=BACKEND) for i in ids]
    codes = [p.wait(120) for p in procs]
    assert set(codes) <= {0, 3} and 0 in codes
    assert len(_running(tmp_path / "alphas.db")) == 1


def test_a_crash_inside_the_transaction_rolls_everything_back(tmp_path, monkeypatch):
    st, (a, b, _), _ = _store(tmp_path)
    st.start_running(a, "paper_running", "r", "x")
    real = StrategyStore._apply
    calls = {"n": 0}

    def boom(s, steps):
        calls["n"] += 1
        real(s, steps)
        if calls["n"] == 2:                                                 # 退役 a、启动 b 都已 flush，提交之前崩
            raise RuntimeError("进程在提交前崩溃")
    monkeypatch.setattr(StrategyStore, "_apply", staticmethod(boom))
    with pytest.raises(RuntimeError):
        st.start_running(b, "paper_running", "r", "x")
    assert _running(tmp_path / "alphas.db") == [a]
    assert [d.decision for d in st.get_decisions(b)] == []


def test_an_illegal_step_leaves_the_slot_untouched(tmp_path):
    from app.db.strategy_store import IllegalStrategyTransition
    st, (a, b, _), _ = _store(tmp_path)
    st.start_running(a, "paper_running", "r", "x")
    with pytest.raises(IllegalStrategyTransition):
        st.start_running(b, "paper_running", "r", "x", before=[(b, "approved", "approve", "r", "x")])   # 已是 approved
    assert _running(tmp_path / "alphas.db") == [a]


def test_a_legacy_database_with_two_runners_is_reported_and_never_resolved_by_picking_the_latest(tmp_path):
    db = tmp_path / "alphas.db"
    st, (a, b, c), url = _store(tmp_path)
    con = sqlite3.connect(db)
    con.execute("DROP INDEX uq_strategy_running_slot")
    con.execute("UPDATE strategy_configs SET status = 'active' WHERE id = ?", (a,))
    con.execute("UPDATE strategy_configs SET status = 'paper_running' WHERE id = ?", (b,))
    con.commit()
    con.close()
    old = StrategyStore(db_url=url)
    assert old.running_slot_problem and str(a) in old.running_slot_problem
    with pytest.raises(RunningSlotConflict):
        old.latest_running()
    assert sorted(old.start_running(c, "paper_running", "人工处理", "x")) == [a, b]
    assert StrategyStore(db_url=url).running_slot_problem is None, "槽恢复唯一后，约束补建上"
    assert _running(db) == [c]


def test_the_api_activation_goes_through_the_atomic_start(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from app.api import router as r
    st, (a, b, _), _ = _store(tmp_path)
    seen = []
    real = StrategyStore.start_running

    def spy(self, sid, target, reason, actor, before=None):
        seen.append((sid, target))
        return real(self, sid, target, reason, actor, before=before)
    monkeypatch.setattr(StrategyStore, "start_running", spy)
    req = SimpleNamespace(reason="r", actor="x")
    r.strategy_integration(a, req, sstore=st)
    r.strategy_activate(b, req, sstore=st)
    assert seen == [(a, "integration"), (b, "paper_running")]
    assert _running(tmp_path / "alphas.db") == [b]
