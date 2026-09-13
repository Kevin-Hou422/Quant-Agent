"""
db/strategy_store.py —— 策略配置台账的 schema 定钉测试（变异测试驱动）

来由：13 个变异点，首测击杀率 38.5%（存活 8）。存活项全在声明层：
`status` 的默认值与索引、`passed` 的默认值（两处：列默认与 dataclass 默认）、
`strategy_id` / `decision` 的非空约束与索引、引擎的 `echo` 与 `expire_on_commit`。

这些决定：
  - 新建的策略配置默认是 `proposed` 还是别的状态（状态机的起点）
  - `passed`（策略门是否通过）默认是 **False** 还是 True ——
    默认 True 意味着**没跑过门的配置一进库就显示"已通过"**
  - 决策谱系能不能写进没有主体、没有决定内容的记录
"""
from __future__ import annotations

import json

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from app.db.strategy_store import (
    IllegalStrategyTransition,
    StrategyConfig,
    StrategyConfigRecord,
    StrategyDecision,
    StrategyStore,
)


@pytest.fixture
def store(tmp_path) -> StrategyStore:
    return StrategyStore(db_url=f"sqlite:///{tmp_path/'s.db'}")


def _cfg(**kw) -> StrategyConfig:
    base = dict(factors=["f1"], combo_weights={"f1": 1.0}, aum=10_000.0)
    base.update(kw)
    return StrategyConfig(**base)


# ===========================================================================
# A. 默认值 —— 状态机起点与"是否通过"
# ===========================================================================

class TestDefaults:

    def test_passed_defaults_to_false_in_the_dataclass(self):
        """
        `StrategyConfig.passed: bool = False` —— **没跑过策略门**的配置
        默认必须是"未通过"。改成 True 后，任何一个只填了因子的配置
        一进库就显示"策略门已通过"。
        """
        assert _cfg().passed is False

    def test_passed_defaults_to_false_in_the_column(self, store):
        """
        `Column(Boolean, default=False)` —— 这是**另一处独立的默认值**
        （直接 ORM 插入、旧数据补录走的是它）。
        """
        with store._Session() as s:
            rec = StrategyConfigRecord(name="raw")
            s.add(rec)
            s.commit()
            rid = rec.id
        got = store.get(rid)
        assert got.passed is False, "列默认值不是 False —— 未验证的配置会显示为已通过"

    def test_status_defaults_to_proposed(self, store):
        """
        `Column(String(16), default="proposed", index=True)` ——
        状态机的起点。默认值被改掉会让新配置跳过审批直接落在别的状态上。
        """
        assert _cfg().status == "proposed"
        with store._Session() as s:
            rec = StrategyConfigRecord(name="raw")
            s.add(rec)
            s.commit()
            rid = rec.id
        assert store.get(rid).status == "proposed"

    def test_saved_config_keeps_its_explicit_status(self, store):
        sid = store.save(_cfg(status="proposed", passed=True))
        got = store.get(sid)
        assert got.status == "proposed" and got.passed is True

    def test_version_defaults_to_one(self, store):
        assert store.get(store.save(_cfg())).version == 1


# ===========================================================================
# B. 表结构
# ===========================================================================

class TestSchema:

    def test_decision_columns_are_not_nullable(self, store):
        """
        `strategy_id` / `decision` 允许为空后，谱系里会出现
        **不属于任何策略、没有决定内容**的记录 —— 而这张表是审批的唯一凭据。
        """
        insp = inspect(store._engine)
        cols = {c["name"]: c for c in insp.get_columns("strategy_decisions")}
        for col in ("strategy_id", "decision"):
            assert cols[col]["nullable"] is False, f"strategy_decisions.{col} 允许为空"

    def test_lookup_columns_are_indexed(self, store):
        insp = inspect(store._engine)
        cfg_ix = {c for ix in insp.get_indexes("strategy_configs")
                  for c in ix["column_names"]}
        dec_ix = {c for ix in insp.get_indexes("strategy_decisions")
                  for c in ix["column_names"]}
        assert "status" in cfg_ix, "按状态找 active 配置是每日循环的主查询，却没有索引"
        assert "strategy_id" in dec_ix

    def test_null_decision_is_rejected_by_the_database(self, store):
        sid = store.save(_cfg())
        with store._Session() as s:
            s.add(StrategyDecision(strategy_id=sid, decision=None))
            with pytest.raises(IntegrityError):
                s.commit()

    def test_null_strategy_id_is_rejected_by_the_database(self, store):
        with store._Session() as s:
            s.add(StrategyDecision(strategy_id=None, decision="approve"))
            with pytest.raises(IntegrityError):
                s.commit()

    def test_engine_does_not_echo_sql(self, store):
        assert store._engine.echo is False

    def test_objects_stay_usable_after_commit(self, store):
        """`expire_on_commit=False`：save 之后拿到的对象仍可读字段。"""
        sid = store.save(_cfg(name="alpha-book"))
        got = store.get(sid)
        assert got.name == "alpha-book"
        assert json.loads(got.factors) == ["f1"]

    def test_sqlite_allows_cross_thread_use(self, tmp_path):
        import threading
        st = StrategyStore(db_url=f"sqlite:///{tmp_path/'t.db'}")
        sid = st.save(_cfg())
        box = {}

        def _read():
            try:
                box["v"] = st.get(sid)
            except Exception as exc:      # noqa: BLE001
                box["err"] = exc

        th = threading.Thread(target=_read)
        th.start()
        th.join()
        assert "err" not in box, f"跨线程读取失败：{box.get('err')}"


# ===========================================================================
# C. 状态流转与 active 查询
# ===========================================================================

class TestTransitions:

    def test_latest_active_returns_only_active_configs(self, store):
        a = store.save(_cfg(name="a"))
        b = store.save(_cfg(name="b"))
        assert store.latest_active() is None, "没有 active 配置时不得返回任何东西"
        store.update_status(a, "approved")
        store.update_status(a, "active")
        got = store.latest_active()
        assert got is not None and got.id == a
        assert got.id != b

    def test_illegal_transition_is_rejected(self, store):
        sid = store.save(_cfg())
        with pytest.raises(IllegalStrategyTransition):
            store.update_status(sid, "active")      # proposed 不能直接到 active

    def test_decision_is_recorded_with_its_transition(self, store):
        sid = store.save(_cfg())
        store.update_status(sid, "approved")
        store.record_decision(sid, "approve", from_status="proposed",
                              to_status="approved", reason="ok")
        rows = store.get_decisions(sid)
        assert [r.decision for r in rows] == ["approve"]
        assert rows[0].from_status == "proposed" and rows[0].to_status == "approved"
        assert rows[0].actor == "human", "决策主体默认应为 human（人批准）"

    PROVEN_EQUIVALENT = {
        "L108 `sessionmaker(..., expire_on_commit=False)` → True":
            "所有 commit 的方法（save / update_status / record_decision）都在"
            "**同一个 session 内**读取需要的字段（过期后自动 refresh 仍拿得到）；"
            "get / latest_active / get_decisions 走只读 session，不触发过期。"
            "两种取值都观察不到差别。",
    }

    def test_every_survivor_has_a_written_proof(self):
        assert len(self.PROVEN_EQUIVALENT) == 1
        for key, why in self.PROVEN_EQUIVALENT.items():
            assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"

    def test_read_only_methods_do_not_commit(self):
        """L108 等价性的机械验证：读方法体里不出现 commit。"""
        import inspect
        import app.db.strategy_store as ss
        for name in ("get", "latest_active", "get_decisions"):
            src = inspect.getsource(getattr(ss.StrategyStore, name))
            assert ".commit()" not in src, (
                f"{name} 在返回 ORM 对象前提交了事务 —— expire_on_commit 会变得可观测")

    def test_json_payloads_round_trip(self, store):
        sid = store.save(_cfg(verdict={"passed": True, "sharpe": 1.23},
                              risk_report={"max_drawdown": -0.1}))
        got = store.get(sid)
        assert json.loads(got.verdict)["sharpe"] == pytest.approx(1.23)
        assert json.loads(got.risk_report)["max_drawdown"] == pytest.approx(-0.1)
