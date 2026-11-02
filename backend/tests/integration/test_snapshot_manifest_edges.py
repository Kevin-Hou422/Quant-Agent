"""
停写快照清单的边界（复审 21cd90b 之后的关键路径变异复核：app/tasks/snapshot_manifest.py）。

首轮 38 个点存活 11 个。补的用例：
    · 主库旁边有回滚日志（-journal）→ 拒绝复制，报的是 -journal（-wal 为空时不能报成 -wal）
    · 副本目录多层不存在 / 已存在都能复制
    · 只有一个运行中的策略不算违反；成交合计与订单成交量相同不列出，差 10 股列出，恰好 1e-6 的差是浮点噪声
    · 内容不同而业务不变量相同的表 → 恢复核验仍然不通过
    · 清单文件是给人看的：中文原样写出，不转义
剩下的等价变异写在 PROVEN_EQUIVALENT（各配一条机械验证的用例）。
"""

from __future__ import annotations

import inspect
import sqlite3
import textwrap

import pytest

from app.tasks import snapshot_manifest as sm
from tests.integration.test_snapshot_consistency import _business_db, _commit_during_copy, _wal_db


@pytest.fixture
def writer(tmp_path):
    db, w = _wal_db(tmp_path)
    yield db, w
    w.close()


def test_a_rollback_journal_next_to_the_db_is_refused_and_named(writer, tmp_path):
    db, _w = writer
    (db.parent / (db.name + "-journal")).write_bytes(b"\x00" * 512)
    with pytest.raises(sm.SourceBusy, match=r"alphas\.db-journal（512 字节）"):
        sm.copy_in_quiet_window(db, tmp_path / "frozen" / db.name)
    assert not (tmp_path / "frozen" / db.name).exists()


@pytest.mark.parametrize("sub", ["a/b/c", "exists"])
def test_the_copy_directory_may_be_nested_or_already_there(writer, tmp_path, sub):
    db, _w = writer
    (tmp_path / "exists").mkdir()
    rep = sm.copy_in_quiet_window(db, tmp_path / sub / db.name)
    assert (tmp_path / sub / db.name).is_file() and rep["copy_sha256"] == rep["before"]["db"]["sha256"]


def _orders_db(tmp_path, dealt):
    p = _business_db(tmp_path / "o.db")
    con = sqlite3.connect(p)
    con.execute("UPDATE exec_orders SET dealt_qty = ? WHERE client_id = 'c-1'", (dealt,))
    con.execute("INSERT INTO exec_orders VALUES (3, 'c-3', 'SUBMITTED', 1e-6, 10)")      # 没有成交行，差恰好 1e-6
    con.commit()
    con.close()
    return sm.describe_db(p)["invariants"]


def test_one_running_strategy_is_not_a_violation_and_matching_fills_are_not_listed(tmp_path):
    inv = _orders_db(tmp_path, dealt=50)
    assert inv["violations"] == [] and inv["facts"]["running_strategies"] == [[1, "paper_running"]]
    assert inv["facts"]["fill_totals_differ"] == []


def test_fills_that_do_not_add_up_to_the_order_are_listed(tmp_path):
    assert _orders_db(tmp_path, dealt=40)["facts"]["fill_totals_differ"] == [["c-1", 40.0, 50.0]]


def test_a_content_difference_outside_the_business_invariants_still_fails(tmp_path):
    from app.tasks.restore import verify_restore
    a, b = _business_db(tmp_path / "a.db"), _business_db(tmp_path / "b.db")
    for p, v in ((a, "原文"), (b, "改过")):
        con = sqlite3.connect(p)
        con.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, v TEXT)")
        con.execute("INSERT INTO notes VALUES (1, ?)", (v,))
        con.commit()
        con.close()
    v = verify_restore(a, b)
    assert v["invariant_mismatches"] == [] and v["content_mismatches"] == ["notes"] and v["ok"] is False


def test_the_manifest_file_is_readable_text(tmp_path):
    p = sm.write_manifest(tmp_path, {"violations": ["运行槽不唯一：[1, 2] 同时处于运行状态"]})
    assert "运行槽不唯一" in p.read_text(encoding="utf-8")
    assert sm.read_manifest(tmp_path) == {"violations": ["运行槽不唯一：[1, 2] 同时处于运行状态"]}


# ── 等价变异 ─────────────────────────────────────────────

PROVEN_EQUIVALENT = {
    "app/tasks/snapshot_manifest.py ×1 — L99 `dest.unlink(missing_ok=True)` -> False":
        "走到这一行时副本一定存在：上面 `_copy_bytes(db, dest)` 刚写出它，`_sha(dest)` 刚读过它（不存在会先抛错）。"
        "missing_ok 只在文件不存在时起作用。见 test_mutated_copies_behave_identically（窗口内有提交 → 两式都删掉副本、抛 SourceChanged）。",
    "app/tasks/snapshot_manifest.py ×1 — L121 `_row_key` 的 `ensure_ascii=False` -> True":
        "行的键只用来算摘要，摘要只与**同一函数**算出的摘要比较（冻结副本 vs 恢复库、清单 vs 恢复库）。两种写法都是单射，"
        "两组行的摘要相等当且仅当行内容相等，与是否转义无关。见 test_mutated_copies_behave_identically（含中文与字节的随机表，"
        "比较「两表摘要是否相等」的结论）。",
}


def _mutated(fn, old, new):
    """同一变异施加到函数源码的副本上（双向：变异工具运行时模块已是变异式，这时副本换回原式）。"""
    code = textwrap.dedent(inspect.getsource(fn))
    a, b = (old, new) if old in code else (new, old)
    assert code.count(a) == 1 and b not in code, (old, new)
    ns = dict(vars(sm))
    exec(compile(code.replace(a, b), "<mutant>", "exec"), ns)             # noqa: S102
    return ns, ns[fn.__name__]


def test_mutated_copies_behave_identically(writer, tmp_path, monkeypatch):
    import random
    db, w = writer
    _commit_during_copy(monkeypatch, w, checkpoint=True)               # 提交后立即合并：WAL 回到空，第二次还能进窗口
    _ns, copy = _mutated(sm.copy_in_quiet_window, "dest.unlink(missing_ok=True)", "dest.unlink(missing_ok=False)")
    for f, sub in ((sm.copy_in_quiet_window, "x"), (copy, "y")):
        with pytest.raises(sm.SourceChanged):
            f(db, tmp_path / sub / db.name)
        assert not (tmp_path / sub / db.name).exists()

    ns, _ = _mutated(sm._row_key, "ensure_ascii=False", "ensure_ascii=True")
    rnd = random.Random(3)
    vals = ["a", "运行", "é", b"\x00\xff", 1, 1.5, None]
    for _ in range(300):
        t1 = [tuple(rnd.choice(vals) for _ in range(3)) for _ in range(rnd.randint(0, 4))]
        t2 = list(t1) if rnd.random() < 0.5 else [tuple(rnd.choice(vals) for _ in range(3)) for _ in range(len(t1))]

        def digest(rows, key):
            return sorted(key(r) for r in rows)
        assert (digest(t1, sm._row_key) == digest(t2, sm._row_key)) == \
            (digest(t1, ns["_row_key"]) == digest(t2, ns["_row_key"]))


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 2
