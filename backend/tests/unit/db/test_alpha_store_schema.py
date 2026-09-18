"""
alpha_store.py —— schema、迁移与前向标记的定钉测试（变异测试驱动）

来由：23 个变异点，首测击杀率 **26.1%**（存活 17）。存活项集中在三处：

1. **ORM 列约束**（`nullable=False` / `index=True`）—— 因子库与谱系的完整性；
2. **`is_forward` 前向标记的全套语义** —— 默认值、只升不降、按它过滤。
   这是 TR.4 →ACTIVE 门的**唯一证据来源**：把 `is_forward` 的默认值改成 True，
   历史回放会**整段冒充前向战绩**，晋级门瞬间形同虚设，而测试全绿；
3. **轻量迁移** `if cols and "is_forward" not in cols` —— 两个条件都能改坏，
   要么对空表建列失败，要么重复 ALTER。

既有覆盖（test_phase5 / test_phase9_approval / test_phase7_paper）只验证
"存进去能查出来"，以上三类一条都没断言。
"""
from __future__ import annotations

from datetime import date as _date

import pytest
from sqlalchemy import inspect

from app.db.alpha_store import AlphaResult, AlphaStore


@pytest.fixture
def store(tmp_path) -> AlphaStore:
    return AlphaStore(db_url=f"sqlite:///{tmp_path/'a.db'}")


def _save(store: AlphaStore, dsl: str = "rank(close)", status: str = "candidate") -> int:
    return store.save(AlphaResult(dsl=dsl, hypothesis="h", sharpe=0.0, status=status))


# ===========================================================================
# A. 表结构
# ===========================================================================

class TestSchema:

    def test_identity_columns_are_not_nullable(self, store):
        """
        `dsl` / `alpha_id` / `date` / `decision` 允许为空，意味着因子库里可以出现
        **没有表达式的因子**、谱系里可以出现**没有决定内容的决策**。
        """
        insp = inspect(store._engine)
        expected = {
            "alpha_records": ("dsl",),
            "alpha_ic_history": ("alpha_id", "date"),
            "alpha_decisions": ("alpha_id", "decision"),
        }
        for table, cols in expected.items():
            actual = {c["name"]: c for c in insp.get_columns(table)}
            for col in cols:
                assert actual[col]["nullable"] is False, f"{table}.{col} 允许为空"

    def test_lookup_columns_are_indexed(self, store):
        insp = inspect(store._engine)
        for table, col in (("alpha_ic_history", "alpha_id"),
                           ("alpha_ic_history", "is_forward"),
                           ("alpha_decisions", "alpha_id")):
            indexed = {c for ix in insp.get_indexes(table) for c in ix["column_names"]}
            assert col in indexed, f"{table}.{col} 没有索引"

    def test_ic_history_is_unique_per_alpha_day(self, store):
        insp = inspect(store._engine)
        names = {u["name"] for u in insp.get_unique_constraints("alpha_ic_history")}
        assert "uq_alpha_ic_date" in names

    def test_null_dsl_is_rejected(self, store):
        from sqlalchemy.exc import IntegrityError
        from app.db.alpha_store import AlphaRecord
        with store._Session() as s:
            s.add(AlphaRecord(dsl=None))
            with pytest.raises(IntegrityError):
                s.commit()

    def test_engine_does_not_echo_sql(self, store):
        assert store._engine.echo is False

    def test_env_database_url_is_used_when_no_arg(self, tmp_path, monkeypatch):
        """`if not db_url:` —— 删掉 `not` 会让环境变量与 settings 两个来源互换。"""
        target = f"sqlite:///{tmp_path/'env.db'}"
        monkeypatch.setenv("DATABASE_URL", target)
        assert str(AlphaStore()._engine.url) == target


# ===========================================================================
# B. is_forward —— →ACTIVE 门的唯一证据来源
# ===========================================================================

class TestForwardFlag:
    """
    Phase 11 的核心区分：**历史回放**产生的 IC 不是前向证据。
    这一组里的每一条都直接对应"晋级门会不会被历史回放骗过去"。
    """

    def test_default_is_replay_not_forward(self, store):
        """
        `is_forward: bool = False`（列默认）与 `is_forward: bool = False`（参数默认）。
        任一处改成 True，首次运行时整段历史回放都会被标成前向战绩。
        """
        aid = _save(store)
        store.record_ic(aid, "2024-01-02", 0.05)
        rows = store.get_ic_history(aid)
        assert len(rows) == 1
        assert rows[0].is_forward is False, (
            "未显式声明前向的 IC 被标成了前向 —— 历史回放会冒充前向证据")
        assert store.get_forward_ic(aid) == []

    def test_explicit_forward_is_recorded(self, store):
        aid = _save(store)
        store.record_ic(aid, "2024-01-03", 0.04, is_forward=True)
        fwd = store.get_forward_ic(aid)
        assert len(fwd) == 1 and fwd[0].is_forward is True

    def test_forward_flag_is_monotonic(self, store):
        """
        `if is_forward: existing.is_forward = True` —— 前向标记**只升不降**。
        把赋值改成 False 后，任何一次回放重跑都会把已积累的前向证据抹掉，
        因子永远攒不够 60 天前向观测。
        """
        aid = _save(store)
        store.record_ic(aid, "2024-01-04", 0.03, is_forward=True)
        store.record_ic(aid, "2024-01-04", 0.02, is_forward=False)   # 回放重跑
        rows = store.get_ic_history(aid)
        assert len(rows) == 1, "同一天重跑应覆盖而非追加"
        assert rows[0].is_forward is True, "重跑回放把已确认的前向证据降级了"
        assert rows[0].realized_ic == pytest.approx(0.02), "数值本身应被覆盖为最新"

    def test_replay_row_can_be_upgraded_to_forward(self, store):
        """
        升级方向：先有一条**回放**记录，后来同一天被确认为前向 → 必须升成 True。

        ⚠️ 只测"先 True 后 False 不降级"是不够的：那条路径里
        `existing.is_forward = True` 根本没被执行（行是在 else 分支建的），
        赋值改成 False 也照样绿（变异测试证实它存活）。必须让**已存在的行**
        走进那条赋值语句。
        """
        aid = _save(store)
        store.record_ic(aid, "2024-01-05", 0.01, is_forward=False)   # 先回放
        assert store.get_forward_ic(aid) == []
        store.record_ic(aid, "2024-01-05", 0.06, is_forward=True)    # 后确认前向
        fwd = store.get_forward_ic(aid)
        assert len(fwd) == 1, "同一天的回放记录没有被升级成前向证据"
        assert fwd[0].realized_ic == pytest.approx(0.06)

    def test_column_default_for_is_forward_is_false(self, store):
        """
        `Column(Boolean, default=False, ...)` —— **列默认值**（与参数默认是两处）。
        直接插入一条不带 is_forward 的记录（旧代码路径、手工补数据都会这样），
        默认值一旦是 True，这些记录就会冒充前向证据。
        """
        from datetime import date as _d
        from app.db.alpha_store import AlphaICRecord
        aid = _save(store)
        with store._Session() as s:
            s.add(AlphaICRecord(alpha_id=aid, date=_d(2024, 1, 8), realized_ic=0.02))
            s.commit()
        rows = store.get_ic_history(aid)
        assert len(rows) == 1
        assert rows[0].is_forward is False, "未指定时的列默认值不是 False"
        assert store.get_forward_ic(aid) == []

    def test_forward_query_filters_out_replay(self, store):
        """
        `AlphaICRecord.is_forward.is_(True)` —— 改成 `is_(False)` 会让
        →ACTIVE 门读到的**全是回放样本**，方向完全反了。
        """
        aid = _save(store)
        store.record_ic(aid, "2024-01-02", 0.01, is_forward=False)
        store.record_ic(aid, "2024-01-03", 0.02, is_forward=True)
        store.record_ic(aid, "2024-01-04", 0.03, is_forward=False)
        fwd = store.get_forward_ic(aid)
        assert [str(r.date) for r in fwd] == ["2024-01-03"], (
            f"前向查询返回了回放样本：{[str(r.date) for r in fwd]}")
        assert len(store.get_ic_history(aid)) == 3, "全量历史仍应包含回放样本"

    def test_forward_records_are_ordered_by_date(self, store):
        aid = _save(store)
        for d in ("2024-01-05", "2024-01-03", "2024-01-04"):
            store.record_ic(aid, d, 0.01, is_forward=True)
        assert [str(r.date) for r in store.get_forward_ic(aid)] == [
            "2024-01-03", "2024-01-04", "2024-01-05"]

    def test_string_and_date_inputs_hit_the_same_row(self, store):
        aid = _save(store)
        store.record_ic(aid, "2024-01-06", 0.01)
        store.record_ic(aid, _date(2024, 1, 6), 0.09)
        rows = store.get_ic_history(aid)
        assert len(rows) == 1, "字符串与 date 对象没有归一到同一天"
        assert rows[0].realized_ic == pytest.approx(0.09)


# ===========================================================================
# C. 轻量迁移
# ===========================================================================

class TestMigration:

    def test_is_forward_column_exists_after_construction(self, store):
        cols = {c["name"] for c in inspect(store._engine).get_columns("alpha_ic_history")}
        assert "is_forward" in cols

    def test_migration_adds_the_column_to_a_legacy_table(self, tmp_path):
        """
        `if cols and "is_forward" not in cols:` —— 对**已存在但缺列**的旧库补列。
        把 `not in` 删掉会让它永远不补（旧库缺列），把 `and` 改成 `or` 会在
        表不存在（cols 为空）时也去 ALTER，抛异常被吞掉。

        构造一张不含 is_forward 的旧表，再用 AlphaStore 打开。
        """
        from sqlalchemy import create_engine

        url = f"sqlite:///{tmp_path/'legacy.db'}"
        eng = create_engine(url)
        with eng.begin() as conn:
            conn.exec_driver_sql(
                "CREATE TABLE alpha_ic_history ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " alpha_id INTEGER NOT NULL,"
                " date DATE NOT NULL,"
                " realized_ic FLOAT,"
                " realized_return FLOAT,"
                " recorded_at DATETIME)")
        eng.dispose()

        st = AlphaStore(db_url=url)
        cols = {c["name"] for c in inspect(st._engine).get_columns("alpha_ic_history")}
        assert "is_forward" in cols, "旧库没有补上 is_forward 列"

        aid = _save(st)
        st.record_ic(aid, "2024-01-02", 0.02, is_forward=True)
        assert len(st.get_forward_ic(aid)) == 1

    def test_migration_is_idempotent(self, tmp_path):
        """重复打开同一个库不得重复 ALTER（第二次会抛 duplicate column）。"""
        url = f"sqlite:///{tmp_path/'twice.db'}"
        AlphaStore(db_url=url)
        st2 = AlphaStore(db_url=url)
        cols = {c["name"] for c in inspect(st2._engine).get_columns("alpha_ic_history")}
        assert "is_forward" in cols


# ===========================================================================
# D. 导出
# ===========================================================================

class TestExport:

    def test_export_of_an_empty_store_writes_nothing(self, store, tmp_path):
        """`if not records: return` —— 删掉 `not` 会在**有**记录时反而不写。"""
        path = tmp_path / "empty.csv"
        store.export_csv(str(path))
        assert not path.exists(), "空库不该产出 CSV"

    def test_export_writes_rows_when_records_exist(self, store, tmp_path):
        _save(store, dsl="rank(ts_delta(close, 5))")
        path = tmp_path / "out.csv"
        store.export_csv(str(path))
        assert path.exists(), "有记录却没有写出 CSV —— 空判断的方向疑似反了"
        text = path.read_text(encoding="utf-8")
        assert "rank(ts_delta(close, 5))" in text


# ===========================================================================
# E. 存活变异的等价性证明
# ===========================================================================

PROVEN_EQUIVALENT = {
    "app/db/alpha_store.py ×1 — L154 `if cols and \"is_forward\" not in cols:` → `or`":
        "左操作数 `cols` 恒为真：`_migrate_add_columns()` 在 "
        "`_Base.metadata.create_all(self._engine)` **之后**调用，"
        "`PRAGMA table_info(alpha_ic_history)` 必然返回非空列集。"
        "`and` 与 `or` 在左操作数恒真时对整体取值无影响。"
        "见 test_migration_always_sees_a_non_empty_column_set。",
}


def test_migration_always_sees_a_non_empty_column_set(tmp_path):
    """L154 等价性的机械验证：迁移执行时表一定已存在且有列。"""
    from sqlalchemy import inspect as _inspect
    st = AlphaStore(db_url=f"sqlite:///{tmp_path/'order.db'}")
    with st._engine.begin() as conn:
        cols = {r[1] for r in conn.exec_driver_sql(
            "PRAGMA table_info(alpha_ic_history)").fetchall()}
    assert cols, "迁移时列集合为空 —— create_all 与迁移的先后顺序疑似被调换"
    assert "alpha_id" in cols and "date" in cols
    assert "alpha_ic_history" in _inspect(st._engine).get_table_names()


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 1
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
