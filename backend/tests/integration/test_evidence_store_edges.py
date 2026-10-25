"""
证据存档的边界（D5 关键路径变异复核：app/db/evidence_store.py 的逻辑部分；表结构见 test_audit_table_schema.py）。

复核时没人钉的：同一份信号换个顺序再存一次不是改写、完全相同的内容再存不记修订、
恰好在开盘截止时刻存的不算前向（存档查询与 ForwardBook 两处）、没给身份时按当前时钟打标、给了就照给的记。
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from app.db.evidence_store import EvidenceStore, ForwardBook, session_deadline

D, NEXT = pd.Timestamp("2025-03-04"), pd.Timestamp("2025-03-05")
VEC = pd.Series({"AAA": 0.5, "BBB": -0.25, "CCC": 0.125})


@pytest.fixture
def es(tmp_path):
    s = EvidenceStore(db_url=f"sqlite:///{(tmp_path / 'ev.db').as_posix()}")
    yield s
    s._engine.dispose()                                           # noqa: SLF001


def _rec_at(minutes=-60):
    return session_deadline(NEXT) + timedelta(minutes=minutes)


class TestArchive:

    def test_the_same_signal_in_another_order_is_not_a_rewrite(self, es):
        assert es.archive("alpha:1", D, VEC, spec_hash="h", recorded_at=_rec_at()) is True
        assert es.archive("alpha:1", D, VEC.iloc[::-1], spec_hash="h", recorded_at=_rec_at(-30)) is False
        assert es.revisions() == []

    def test_a_different_signal_or_spec_is_a_refused_rewrite(self, es):
        es.archive("alpha:1", D, VEC, spec_hash="h", recorded_at=_rec_at())
        es.archive("alpha:1", D, VEC * 2, spec_hash="h", recorded_at=_rec_at(-30))
        es.archive("alpha:1", D, VEC, spec_hash="other", recorded_at=_rec_at(-20))
        assert [r.kind for r in es.revisions()] == ["signal_archive", "signal_archive"]

    def test_archived_exactly_at_the_deadline_is_too_late(self, es):
        es.archive("alpha:1", D, VEC, recorded_at=session_deadline(NEXT))
        es.archive("alpha:2", D, VEC, recorded_at=session_deadline(NEXT) - timedelta(microseconds=1))
        assert es.forward_signal("alpha:1", D, NEXT) is None
        assert es.forward_signal("alpha:2", D, NEXT) is not None
        assert ForwardBook(es, "alpha:1", {"price_source": "yahoo"}).signal_for(D, NEXT) is None


def test_the_forward_book_and_the_archive_apply_the_same_deadline(es):
    """L329 等价的机械验证：ForwardBook 先按存档时刻筛一次，之后的 archived_before 用同一个截止时刻再判一次 `>=`；
    前一道放宽成 `>` 时，恰好在截止时刻的那份照样被后一道拦下。"""
    es.archive("alpha:1", D, VEC, recorded_at=session_deadline(NEXT))
    book = ForwardBook(es, "alpha:1", {"price_source": "yahoo"})
    book._idx = {k: v + timedelta(microseconds=1) for k, v in book._idx.items()}    # 绕过第一道（模拟 `>`）
    assert es.archived_before("alpha:1", D, session_deadline(NEXT)) is None
    assert book.signal_for(D, NEXT) is None


PROVEN_EQUIVALENT = {
    "app/db/evidence_store.py ×1 — L329 ForwardBook.signal_for 的 `rec_at >= deadline` -> `>`":
        "两者只在存档时刻恰好等于截止时刻时不同；那时 signal_for 接着调 archived_before(…, deadline)，"
        "后者用同一个截止时刻判 `row.recorded_at >= deadline` 返回 None —— 结论仍是\"不算前向\"。"
        "见 test_the_forward_book_and_the_archive_apply_the_same_deadline。",
}


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 1

    def test_without_an_identity_the_current_clock_is_recorded(self, es):
        from app.core.data_engine.input_identity import clock_version
        from app.db.evidence_store import SignalArchive
        es.archive("alpha:1", D, VEC, recorded_at=_rec_at())
        es.archive("alpha:2", D, VEC, recorded_at=_rec_at(), identity={"clock_version": "c1", "price_source": "yahoo"})
        with es._Session() as s:                                    # noqa: SLF001
            got = {r.book_key: r.clock_version for r in s.query(SignalArchive)}
        assert got == {"alpha:1": clock_version(), "alpha:2": "c1"}


class TestStrategyEvidence:

    def test_the_same_ic_recorded_again_is_not_a_rewrite(self, es):
        ts = _rec_at()
        assert es.record_strategy_ic(7, "h", NEXT, 0.03, D, ts) is True
        assert es.record_strategy_ic(7, "h", NEXT, 0.03, D, ts) is False
        assert es.revisions() == []

    def test_a_different_ic_or_spec_is_a_refused_rewrite(self, es):
        ts = _rec_at()
        es.record_strategy_ic(7, "h", NEXT, 0.03, D, ts)
        es.record_strategy_ic(7, "h", NEXT, 0.04, D, ts)
        es.record_strategy_ic(7, "other", NEXT, 0.03, D, ts)
        assert [r.kind for r in es.revisions()] == ["strategy_ic", "strategy_ic"]
        assert es.strategy_forward_ic(7)[0].ic == 0.03


def test_a_difference_of_exactly_the_tolerance_is_noise_not_a_rewrite(es):
    """0 与 1e-12 恰好差 1e-12（容差本身）：同一个数的浮点噪声，不记修订。"""
    assert abs(0.0 - 1e-12) == 1e-12
    ts = _rec_at()
    es.record_strategy_ic(7, "h", NEXT, 0.0, D, ts)
    es.record_strategy_ic(7, "h", NEXT, 1e-12, D, ts)
    assert es.revisions() == []
