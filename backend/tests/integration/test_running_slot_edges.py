"""
唯一运行槽的边界（复审 21cd90b 之后的关键路径变异复核：app/db/strategy_store.py）。

首轮 24 个点存活 3 个（导入选路 25 个文件的基线超过 300 秒单点超时，改用直接用到 StrategyStore 的 8 个文件重跑）：
    · L249 SQLite 上要先 `BEGIN IMMEDIATE` 拿写锁、再读当前运行者 —— 原有用例的并发场景里，不拿锁也恰好由唯一索引
      兜住，看不出差别；这里直接核对执行顺序
    · L142 工厂的 expire_on_commit：证明在 tests/unit/db/test_strategy_store_schema.py（已按运行槽之后的代码重核）
    · L251 start_running 自己的 Session 的 expire_on_commit：等价，见 PROVEN_EQUIVALENT
"""

from __future__ import annotations

import inspect
import textwrap

from sqlalchemy import event

import app.db.strategy_store as ss
from tests.integration.test_running_slot import _running, _store


def test_the_write_lock_is_taken_before_the_current_runner_is_read(tmp_path):
    st, (a, b, _), _url = _store(tmp_path)
    st.start_running(a, "paper_running", reason="r", actor="x")
    seen = []
    event.listen(st._engine, "before_cursor_execute", lambda conn, cur, stmt, *args: seen.append(" ".join(stmt.split())))
    st.start_running(b, "paper_running", reason="r", actor="x")
    first_read = next(i for i, s in enumerate(seen) if s.upper().startswith("SELECT") and "strategy_configs" in s)
    assert seen[0] == "BEGIN IMMEDIATE" and 0 < first_read, seen[:3]


PROVEN_EQUIVALENT = {
    "app/db/strategy_store.py ×1 — L251 start_running 里 `Session(bind=conn, expire_on_commit=False)` -> True":
        "提交之后这个 Session 里的 ORM 对象一个也没再读：要返回的退役 id 在提交前就取成了元组里的整数，"
        "提交后只有 `conn.commit()` 与 `return [r[0] for r in retire]`。过期与否观察不到。"
        "见 test_a_mutated_start_running_gives_the_same_outcomes（同一组场景比较返回值与库里的状态）。",
}


def test_a_mutated_start_running_gives_the_same_outcomes(tmp_path):
    code = textwrap.dedent(inspect.getsource(ss.StrategyStore.start_running))
    old, new = "Session(bind=conn, expire_on_commit=False)", "Session(bind=conn, expire_on_commit=True)"
    a_, b_ = (old, new) if old in code else (new, old)                # 双向：变异工具运行时源码已是变异式
    assert code.count(a_) == 1
    ns = dict(vars(ss))
    exec(compile(code.replace(a_, b_), "<mutant>", "exec"), ns)          # noqa: S102
    Mutant = type("Mutant", (ss.StrategyStore,), {"start_running": ns["start_running"]})
    results = []
    for k, cls in enumerate((ss.StrategyStore, Mutant)):
        d = tmp_path / str(k)
        d.mkdir()
        st, (a, b, c), _url = _store(d)
        st.__class__ = cls
        out = [st.start_running(a, "integration", reason="r", actor="x"),
               st.start_running(b, "paper_running", reason="r", actor="x"),
               st.start_running(c, "paper_running", reason="r", actor="x")]
        results.append((out, _running(d / "alphas.db"), [(r.id, r.status) for r in st.query()]))
    assert results[0] == results[1]


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 1
