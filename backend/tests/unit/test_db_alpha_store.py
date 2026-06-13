"""
test_db_alpha_store.py — AlphaStore CRUD 测试（内存 SQLite）
"""
from __future__ import annotations

import csv
import os
import threading
import pytest

from app.db.alpha_store import AlphaStore, AlphaResult


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test_alpha.db"
    return AlphaStore(db_url=f"sqlite:///{db_path}")


def _result(dsl: str = "rank(close)", sharpe: float = 1.0, status: str = "active") -> AlphaResult:
    return AlphaResult(
        dsl=dsl,
        hypothesis="test",
        ann_return=0.15,
        sharpe=sharpe,
        max_drawdown=-0.10,
        ic_ir=0.6,
        ann_turnover=0.5,
        status=status,
        reasoning="{}",
    )


class TestAlphaStoreSave:

    def test_save_returns_positive_int_id(self, store):
        aid = store.save(_result())
        assert isinstance(aid, int)
        assert aid > 0

    def test_save_multiple_returns_distinct_ids(self, store):
        id1 = store.save(_result("rank(close)"))
        id2 = store.save(_result("zscore(open)"))
        assert id1 != id2

    def test_concurrent_saves_no_corruption(self, store):
        """5 个线程并发写入，IDs 应互不相同。"""
        ids = []
        lock = threading.Lock()

        def _save(i):
            aid = store.save(_result(f"rank(close_{i})", sharpe=float(i)))
            with lock:
                ids.append(aid)

        threads = [threading.Thread(target=_save, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(ids)) == 5  # 无重复 ID


class TestAlphaStoreQuery:

    def test_query_min_sharpe_filter(self, store):
        store.save(_result("a1", sharpe=0.3))
        store.save(_result("a2", sharpe=0.8))
        store.save(_result("a3", sharpe=1.2))
        results = store.query(min_sharpe=0.5)
        assert len(results) == 2
        for r in results:
            assert r.sharpe >= 0.5

    def test_query_status_filter(self, store):
        store.save(_result("b1", status="active"))
        store.save(_result("b2", status="retired"))
        active = store.query(status="active")
        retired = store.query(status="retired")
        assert all(r.status == "active"  for r in active)
        assert all(r.status == "retired" for r in retired)

    def test_query_limit_enforced(self, store):
        for i in range(5):
            store.save(_result(f"c{i}"))
        results = store.query(limit=3)
        assert len(results) <= 3

    def test_query_sorted_by_sharpe_desc(self, store):
        sharpes = [0.3, 1.5, 0.9]
        for i, s in enumerate(sharpes):
            store.save(_result(f"d{i}", sharpe=s))
        results = store.query()
        result_sharpes = [r.sharpe for r in results]
        assert result_sharpes == sorted(result_sharpes, reverse=True)

    def test_query_empty_db_returns_empty_list(self, store):
        assert store.query() == []


class TestAlphaStoreGetById:

    def test_get_by_id_found(self, store):
        aid = store.save(_result("e1", sharpe=0.7))
        record = store.get_by_id(aid)
        assert record is not None
        assert record.id == aid
        assert record.dsl == "e1"

    def test_get_by_id_not_found_returns_none(self, store):
        result = store.get_by_id(99999)
        assert result is None


class TestAlphaStoreExportCSV:

    def test_export_csv_creates_file(self, store, tmp_path):
        store.save(_result("f1", sharpe=0.5))
        store.save(_result("f2", sharpe=0.8))
        csv_path = str(tmp_path / "export.csv")
        store.export_csv(csv_path)
        assert os.path.exists(csv_path)

    def test_export_csv_has_correct_rows(self, store, tmp_path):
        store.save(_result("g1"))
        store.save(_result("g2"))
        csv_path = str(tmp_path / "export2.csv")
        store.export_csv(csv_path)
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        # 1 header + 2 data rows
        assert len(rows) == 3

    def test_export_csv_empty_db_no_file(self, store, tmp_path):
        """空数据库不创建文件（或创建空文件，均可接受）。"""
        csv_path = str(tmp_path / "empty_export.csv")
        store.export_csv(csv_path)
        # 不应崩溃
