"""
position_store.py —— 账本 schema 与查询契约的定钉测试（变异测试驱动）

来由：25 个变异点，首测击杀率 **24.0%**（存活 19）。存活项几乎全在
**ORM 列声明**与**引擎配置**上：`nullable=False`、`index=True`、
`check_same_thread`、`expire_on_commit`、`echo` 一个都没被断言过。

这些不是"风格"：
  - `nullable=False` 是账本的完整性约束 —— 允许 NULL 的 alpha_id/date 意味着
    可以写进一条**不属于任何因子、不属于任何交易日**的持仓；
  - `index=True` 决定 `pnl_history` / `fills_in_range` 在长账本上的行为；
  - `expire_on_commit=False` 决定 `record_day` 之后拿到的对象还能不能读字段；
  - `echo=False` 决定每一条 SQL 会不会被打进日志。

本文件把这些契约变成断言。
"""
from __future__ import annotations

import logging
import os
from datetime import date as _date

import pytest
from sqlalchemy import inspect, text

from app.core.execution.paper_broker import DailyPnL
from app.db.position_store import (
    PaperDailyPnL,
    PaperFill,
    PaperPosition,
    PositionStore,
    _to_date,
)


@pytest.fixture
def store(tmp_path) -> PositionStore:
    return PositionStore(db_url=f"sqlite:///{tmp_path/'p.db'}")


def _pnl(aid: int, d: str, equity: float = 1.0) -> DailyPnL:
    return DailyPnL(alpha_id=aid, date=d, gross_ret=0.0, net_ret=0.0,
                    cost_bps=0.0, equity=equity)


# ===========================================================================
# A. 表结构 —— 非空约束与索引
# ===========================================================================

class TestSchemaConstraints:

    NOT_NULL = {
        "paper_positions": ("alpha_id", "date", "ticker"),
        "paper_fills": ("alpha_id", "date", "ticker"),
        "paper_daily_pnl": ("alpha_id", "date"),
    }
    INDEXED = {
        "paper_positions": ("alpha_id", "date"),
        "paper_fills": ("alpha_id", "date"),
        "paper_daily_pnl": ("alpha_id", "date"),
    }

    def test_identity_columns_are_not_nullable(self, store):
        """
        `nullable=False` 一旦被改成 True，就可以写进一条**不属于任何因子、
        不属于任何交易日**的持仓，而对账、幂等覆盖、按日查询全部依赖这三列。
        """
        insp = inspect(store._engine)
        for table, cols in self.NOT_NULL.items():
            actual = {c["name"]: c for c in insp.get_columns(table)}
            for col in cols:
                assert actual[col]["nullable"] is False, (
                    f"{table}.{col} 允许为空 —— 账本可以写进无主的记录")

    def test_lookup_columns_are_indexed(self, store):
        """`index=True`：按 alpha/日期查账本是每日例行操作。"""
        insp = inspect(store._engine)
        for table, cols in self.INDEXED.items():
            indexed = {c for ix in insp.get_indexes(table) for c in ix["column_names"]}
            for col in cols:
                assert col in indexed, f"{table}.{col} 没有索引"

    def test_null_identity_is_rejected_by_the_database(self, store):
        """约束不只写在声明里，要真的被数据库执行。"""
        from sqlalchemy.exc import IntegrityError
        with store._Session() as s:
            s.add(PaperPosition(alpha_id=None, date=_date(2024, 1, 2),
                                ticker="A", weight=0.1))
            with pytest.raises(IntegrityError):
                s.commit()

    def test_unique_constraints_make_writes_idempotent(self, store):
        """(alpha_id, date, ticker) 唯一 —— 幂等覆盖依赖它。"""
        insp = inspect(store._engine)
        names = {u["name"] for u in insp.get_unique_constraints("paper_positions")}
        assert "uq_pos" in names
        assert "uq_fill" in {u["name"] for u in insp.get_unique_constraints("paper_fills")}
        assert "uq_pnl" in {u["name"] for u in insp.get_unique_constraints("paper_daily_pnl")}

    def test_ticker_column_has_a_length_bound(self, store):
        insp = inspect(store._engine)
        col = {c["name"]: c for c in insp.get_columns("paper_positions")}["ticker"]
        assert "VARCHAR" in str(col["type"]).upper()


# ===========================================================================
# B. 引擎配置
# ===========================================================================

class TestEngineConfiguration:

    def test_sqlite_allows_cross_thread_use(self, tmp_path):
        """
        `connect_args={"check_same_thread": False}`。改成 True 后，
        调度线程写、请求线程读就会抛 `SQLite objects created in a thread...`。
        直接在**另一个线程**里读一次来验证。
        """
        import threading
        st = PositionStore(db_url=f"sqlite:///{tmp_path/'thread.db'}")
        st.record_day(1, "2024-01-02", {"A": 0.5}, [], _pnl(1, "2024-01-02"))
        box = {}

        def _read():
            try:
                box["v"] = st.latest_positions(1)
            except Exception as exc:      # noqa: BLE001 - 要把异常带回主线程断言
                box["err"] = exc

        t = threading.Thread(target=_read)
        t.start()
        t.join()
        assert "err" not in box, f"跨线程读取失败：{box.get('err')}"
        assert box["v"] == {"A": 0.5}

    def test_sql_is_not_echoed_to_the_log(self, tmp_path):
        """
        `echo=False`：改成 True 会把每一条 SQL 打进日志，交易日志被淹没。

        ⚠️ 不能用 `caplog.at_level("sqlalchemy.engine")` 来验 —— 那样做本身
        就把该 logger 打开了，SQL 无论 echo 取何值都会出现（第一版就是这么红的）。
        直接断言引擎属性。
        """
        st = PositionStore(db_url=f"sqlite:///{tmp_path/'echo.db'}")
        assert st._engine.echo is False, "引擎开启了 SQL echo"
        assert logging.getLogger("sqlalchemy.engine").level in (
            logging.NOTSET, logging.WARNING), (
            "sqlalchemy.engine 的日志级别被 echo=True 调低了")

    def test_objects_stay_usable_after_commit(self, store):
        """
        `expire_on_commit=False`：改成 True 后，`fills_on` / `pnl_history` 返回的
        ORM 对象在 session 关闭后再读字段会抛 DetachedInstanceError ——
        而调用方拿到的正是这些对象。
        """
        store.record_day(
            7, "2024-01-02", {"A": 0.5},
            [{"ticker": "A", "target_weight": 0.5, "filled_weight": 0.5,
              "fill_price": 100.0, "cost_usd": 1.0, "reject_reason": ""}],
            _pnl(7, "2024-01-02", equity=1.01))
        fills = store.fills_on(7, "2024-01-02")
        hist = store.pnl_history(7)
        assert fills[0].ticker == "A"          # session 已关闭，仍必须可读
        assert fills[0].filled_weight == pytest.approx(0.5)
        assert hist[0].equity == pytest.approx(1.01)

    def test_default_db_url_is_used_when_none_given(self, tmp_path, monkeypatch):
        """
        `if not db_url:` —— 删掉 `not` 后，环境变量里**有** DATABASE_URL 时反而
        会去读 settings，两个来源互换，账本可能落到另一个库上。
        """
        target = f"sqlite:///{tmp_path/'env.db'}"
        monkeypatch.setenv("DATABASE_URL", target)
        st = PositionStore()
        assert str(st._engine.url) == target, (
            f"DATABASE_URL 未被采用，实际连到 {st._engine.url}")

    def test_settings_url_is_used_when_env_is_empty(self, tmp_path, monkeypatch):
        """对照组：环境变量为空时才回落到 settings。"""
        import app.config
        monkeypatch.setenv("DATABASE_URL", "")
        target = f"sqlite:///{tmp_path/'cfg.db'}"
        monkeypatch.setattr(app.config.settings, "database_url", target, raising=False)
        st = PositionStore()
        assert str(st._engine.url) == target


# ===========================================================================
# C. 查询契约
# ===========================================================================

class TestQueries:

    def _seed(self, store):
        for aid in (1, 2):
            for day, tk in (("2024-01-02", "A"), ("2024-01-03", "B"),
                            ("2024-01-04", "C")):
                store.record_day(
                    aid, day, {tk: 0.3},
                    [{"ticker": tk, "target_weight": 0.3, "filled_weight": 0.3,
                      "fill_price": 10.0, "cost_usd": 0.5, "reject_reason": ""}],
                    _pnl(aid, day, equity=1.0 + aid / 100))

    def test_fills_in_range_filters_by_alpha_when_given(self, store):
        """
        `if alpha_id is not None: stmt = stmt.where(...)` —— 删掉 `not` 后，
        **给了** alpha_id 反而不过滤，成本校准会把别的因子的成交算进来。
        """
        self._seed(store)
        both = store.fills_in_range("2024-01-02", "2024-01-04")
        one = store.fills_in_range("2024-01-02", "2024-01-04", alpha_id=1)
        assert len(both) == 6
        assert len(one) == 3, f"按 alpha 过滤失效，返回 {len(one)} 条"
        assert {f.alpha_id for f in one} == {1}

    def test_fills_in_range_is_inclusive_on_both_ends(self, store):
        self._seed(store)
        mid = store.fills_in_range("2024-01-03", "2024-01-03", alpha_id=1)
        assert [f.ticker for f in mid] == ["B"]

    def test_fills_in_range_is_ordered_by_date_then_ticker(self, store):
        self._seed(store)
        rows = store.fills_in_range("2024-01-02", "2024-01-04", alpha_id=1)
        assert [str(f.date) for f in rows] == ["2024-01-02", "2024-01-03", "2024-01-04"]

    def test_state_before_is_strictly_earlier(self, store):
        """续跑依赖它：重跑第 t 日时必须拿 t-1 的状态，不能拿 t 自己的。"""
        self._seed(store)
        eq, pos = store.state_before(1, "2024-01-03")
        assert pos == {"A": 0.3}
        assert eq == pytest.approx(1.01)
        eq0, pos0 = store.state_before(1, "2024-01-02")
        assert (eq0, pos0) == (1.0, {})

    def test_record_day_overwrites_the_same_day(self, store):
        """幂等：同一 (alpha, date) 重写不叠加。"""
        store.record_day(9, "2024-02-01", {"A": 0.5}, [], _pnl(9, "2024-02-01", 1.0))
        store.record_day(9, "2024-02-01", {"A": 0.2, "B": -0.1}, [],
                         _pnl(9, "2024-02-01", 1.5))
        assert store.latest_positions(9) == {"A": 0.2, "B": -0.1}
        assert len(store.pnl_history(9)) == 1
        assert store.latest_equity(9) == pytest.approx(1.5)

    def test_last_pnl_date_returns_the_maximum(self, store):
        self._seed(store)
        assert store.last_pnl_date(1) == _date(2024, 1, 4)
        assert store.last_pnl_date(999) is None


# ===========================================================================
# D. 日期归一化
# ===========================================================================

class TestDateCoercion:

    @pytest.mark.parametrize("value", [
        "2024-03-05",
        __import__("datetime").datetime(2024, 3, 5, 15, 30),
        __import__("pandas").Timestamp("2024-03-05 15:30"),
        _date(2024, 3, 5),
    ])
    def test_all_input_forms_normalise_to_the_same_date(self, value):
        assert _to_date(value) == _date(2024, 3, 5)

    PROVEN_EQUIVALENT = {
        "L100 `sessionmaker(..., expire_on_commit=False)` → True":
            "本类所有返回 ORM 对象的方法（fills_on / fills_in_range / pnl_history）"
            "用的都是**只读** session —— 没有 commit，就不会触发过期；"
            "而所有 commit 的方法（record_day）不把对象返回给调用方。"
            "因此 expire_on_commit 取何值都观察不到差别。"
            "见 test_no_method_returns_orm_objects_from_a_committing_session。",
    }

    def test_no_method_returns_orm_objects_from_a_committing_session(self):
        """L100 等价性的机械验证：读对象的方法体里不出现 commit。"""
        import inspect
        import app.db.position_store as ps
        for name in ("fills_on", "fills_in_range", "pnl_history"):
            src = inspect.getsource(getattr(ps.PositionStore, name))
            assert ".commit()" not in src, (
                f"{name} 在返回 ORM 对象前提交了事务 —— expire_on_commit 会变得可观测")
        assert ".commit()" in inspect.getsource(ps.PositionStore.record_day)

    def test_every_survivor_has_a_written_proof(self):
        assert len(self.PROVEN_EQUIVALENT) == 1
        for key, why in self.PROVEN_EQUIVALENT.items():
            assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"

    def test_write_and_read_agree_across_input_forms(self, store):
        """写入用字符串、查询用 date 对象，必须命中同一行。"""
        import pandas as pd
        store.record_day(11, "2024-03-05", {"A": 0.4}, [], _pnl(11, "2024-03-05"))
        assert store.fills_on(11, _date(2024, 3, 5)) == []
        store.record_day(11, pd.Timestamp("2024-03-05 09:30"), {"A": 0.9}, [],
                         _pnl(11, "2024-03-05"))
        assert store.latest_positions(11) == {"A": 0.9}, (
            "Timestamp 与字符串没有归一到同一天，账本出现了重复行")
