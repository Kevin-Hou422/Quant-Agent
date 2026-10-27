"""
IC 均值与 IC-IR 分列、样本与评估状态（2026-10-25 审计的决定 2）。

原话："IC 均值与真正 IC-IR 分列，不接受直接给混合语义的一列统一换名后宣称修好。……没有足够样本时为 NULL，并写原因。
同步记录 n_obs、样本区间、评估状态、数据源/摘要、clock、计算版本和研究切分。……'从未评估'与'评估后恰好为 0'必须分开，
不再靠所有指标是否为 0 推断 evaluated。……其他未知历史行保留原始值和来源状态。"
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from app.core.alpha_engine import ic_metrics as icm


def _meta(fields):
    return json.loads(fields["ic_metrics"])


class TestTheRule:

    def test_mean_and_ir_go_to_separate_fields(self):
        f = icm.ic_fields(0.031, 0.42, 250, source="t", segment="selection", window=["2021-01-01", "2023-12-29"],
                          clock="c2")
        assert (f["ic_mean"], f["ic_ir"]) == (0.031, 0.42)
        m = _meta(f)
        assert m["status"] == icm.EVALUATED and m["n_obs"] == 250 and m["segment"] == "selection"
        assert m["window"] == ["2021-01-01", "2023-12-29"] and m["clock"] == "c2" and m["version"] == icm.ICM_VERSION
        assert "不年化" in m["ic_ir_formula"]

    def test_too_few_observations_leave_ir_empty_with_a_reason(self):
        f = icm.ic_fields(0.05, 0.9, icm.MIN_OBS_FOR_IR - 1, source="t", segment="selection", clock="c2")
        assert f["ic_mean"] == 0.05 and f["ic_ir"] is None
        assert str(icm.MIN_OBS_FOR_IR) in _meta(f)["ic_ir_null_reason"]

    def test_zero_variance_leaves_ir_empty(self):
        f = icm.ic_fields(0.02, float("nan"), 100, source="t", segment="selection", clock="c2")
        assert f["ic_ir"] is None and "标准差为 0" in _meta(f)["ic_ir_null_reason"]

    def test_an_evaluated_zero_is_not_never_evaluated(self):
        zero = icm.ic_fields(0.0, 0.0, 100, source="t", segment="selection", clock="c2")
        assert zero["ic_mean"] == 0.0 and _meta(zero)["status"] == icm.EVALUATED
        assert _meta(icm.never_evaluated("t"))["status"] == icm.NEVER_EVALUATED

    def test_no_observations_is_not_computed(self):
        f = icm.ic_fields(0.03, None, 0, source="t", segment="selection", clock="c2")
        assert f["ic_mean"] is None and _meta(f)["status"] == icm.NOT_COMPUTED


class TestTheStore:

    @pytest.fixture
    def store(self, tmp_path):
        from app.db.alpha_store import AlphaStore
        return AlphaStore(db_url=f"sqlite:///{(tmp_path / 'a.db').as_posix()}")

    def test_a_record_saved_without_any_ic_is_never_evaluated_and_empty(self, store):
        from app.db.alpha_store import AlphaResult
        rec = store.get_by_id(store.save(AlphaResult(dsl="rank(close)")))
        assert rec.ic_ir is None and rec.ic_mean is None
        assert icm.status_of(rec)["status"] == icm.NEVER_EVALUATED

    def test_values_without_provenance_are_kept_but_flagged(self, store):
        from app.db.alpha_store import AlphaResult
        rec = store.get_by_id(store.save(AlphaResult(dsl="rank(close)", ic_ir=0.3)))
        assert rec.ic_ir == 0.3 and icm.status_of(rec)["status"] == icm.NO_PROVENANCE

    def test_workflow_metrics_put_the_in_sample_mean_into_ic_mean(self, store):
        from app.db.alpha_store import AlphaResult
        m = {"is_ic": 0.021, "is_ic_ir": 0.18, "is_ic_n": 300}
        rec = store.get_by_id(store.save(AlphaResult(dsl="rank(close)", **icm.from_workflow_metrics(m, source="t"))))
        assert (rec.ic_mean, rec.ic_ir) == (0.021, 0.18)
        assert icm.status_of(rec)["n_obs"] == 300

    def test_an_old_database_gets_the_columns_and_its_rows_stay_as_they_were(self, tmp_path):
        """旧库：没有 ic_mean / ic_metrics 列；补列后原值不动，状态读成 legacy_unknown（不按数值推断）。"""
        db = tmp_path / "old.db"
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE alpha_records (id INTEGER PRIMARY KEY, dsl TEXT NOT NULL, created_at DATETIME, "
                    "hypothesis VARCHAR(512), ann_return FLOAT, sharpe FLOAT, max_drawdown FLOAT, ic_ir FLOAT, "
                    "ann_turnover FLOAT, status VARCHAR(32), reasoning TEXT)")
        con.execute("INSERT INTO alpha_records (id, dsl, sharpe, ic_ir, status) VALUES (1, 'rank(close)', 0.2, 0.031, 'paper')")
        con.execute("INSERT INTO alpha_records (id, dsl, sharpe, ic_ir, status) VALUES (2, 'rank(open)', 0.0, 0.0, 'paper')")
        con.commit()
        con.close()
        from app.db.alpha_store import AlphaStore
        st = AlphaStore(db_url=f"sqlite:///{db.as_posix()}")
        a, b = st.get_by_id(1), st.get_by_id(2)
        assert a.ic_ir == 0.031 and a.ic_mean is None and b.ic_ir == 0.0
        assert {icm.status_of(a)["status"], icm.status_of(b)["status"]} == {icm.LEGACY_UNKNOWN}


class TestTheProducers:

    def test_the_backtest_report_carries_the_number_of_ic_observations(self):
        from app.agent._data_utils import _run_backtest_core
        from app.core.alpha_engine.signal_processor import SimulationConfig
        from tests.conftest import _make_dataset
        data = _make_dataset(n_days=260, n_tickers=12, seed=5)
        m = _run_backtest_core("rank(ts_delta(close, 5))", SimulationConfig(), data, None)
        assert m["is_ic_n"] > icm.MIN_OBS_FOR_IR and m["is_ic"] is not None
        assert m["is_ic_ir"] is not None and m["is_ic_ir"] != m["is_ic"], "IC-IR 不能是 IC 均值换个名字"
        f = icm.from_workflow_metrics(m, source="t")
        assert f["ic_mean"] == m["is_ic"] and f["ic_ir"] == m["is_ic_ir"]

    def test_the_screening_eval_reports_mean_n_and_a_ddof1_ir(self):
        import numpy as np

        from app.core.gp_engine.evaluation_utils import quick_ic_eval
        from tests.conftest import _make_dataset
        out = quick_ic_eval("rank(ts_delta(close, 5))", _make_dataset(n_days=200, n_tickers=12, seed=6))
        assert out["ic_n"] > 0 and np.isfinite(out["ic_mean"])
        assert out["ic_ir_ddof1"] is None or abs(out["ic_ir_ddof1"]) < 50

    def test_the_report_api_returns_mean_ir_and_status(self, fresh_db):
        from fastapi.testclient import TestClient

        from app.db.alpha_store import AlphaResult, AlphaStore
        from app.main import app
        AlphaStore().save(AlphaResult(dsl="rank(close)", **icm.ic_fields(0.02, None, 5, source="t", segment="s",
                                                                          clock="c2")))
        with TestClient(app) as c:
            rows = c.get("/api/report/query").json()["records"]
        r = next(x for x in rows if x["dsl"] == "rank(close)")
        assert r["ic_mean"] == 0.02 and r["ic_ir"] is None and r["ic_status"] == "evaluated"
        assert "5" in r["ic_metrics"]["ic_ir_null_reason"]
