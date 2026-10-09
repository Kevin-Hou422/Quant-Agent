"""
Phase 12 执行层新增点里**存活 / 未被常规用例判定的变异**的处置

能杀的都已经用用例杀掉了（见 test_order_builder / test_pretrade_gate / test_fidelity /
test_order_manager / test_broker_gateway / tests/unit/db/test_execution_store_schema）。
这里只剩两类，每条都配一个**会随前提变化而变红**的可执行验证：

1. NEW_POINT_EQUIVALENCE —— 变异与原式在所有可达输入上输出相同。证明写清"差异只可能
   出现在哪个输入、为什么那个输入到不了这里"，验证用 AST 钉住"到不了"的那个结构。
2. NEW_POINT_LOUD_AT_IMPORT —— 变异让模块在**导入 / 构造时**就崩（pytest 退出码 2，
   变异工具按规矩不记为"杀死"）。这里用程序对源码施加同一个变异再执行，断言它确实崩、
   且崩在声称的那个原因上 —— 而不是靠"应该会崩"的口头判断。

命名与 R.2 同：故意**不叫** PROVEN_EQUIVALENT（那个名字由 test_invariants 的逐模块对账
消费，对的是全量测量的存活数；这些点从未进过全量测量，分母不同）。
"""
from __future__ import annotations

import ast
import inspect
import math
import re
import types
from pathlib import Path

import pytest

from app.core.execution import fidelity as fid
from app.core.execution import order_builder as ob
from app.core.execution import pretrade_gate as pg
from app.db import execution_store as es


NEW_POINT_EQUIVALENCE = {
    "app/core/execution/order_builder.py ×1 — round_to_tick `price >= 1.0` -> `>`":
        "两式只在 price == 1.0 时选不同的价位（0.01 vs 0.0001）。1.0 在两种价位网格上都"
        "恰好落在格点上：floor/ceil(1.0/tick ± 1e-9)·tick 两边都回到 1.0。"
        "来源反驳：若取整公式不再带 ±1e-9 的护栏，1.0/0.0001 的浮点误差可能让一边落到"
        "0.9999 —— 由 test_tick_boundary_is_on_both_grids 对两种价位逐一实算。",

    "app/core/execution/order_builder.py ×1 — build_flatten_orders `cur > 0` -> `>=`":
        "差异只在 cur == 0。该比较之前有 `if abs(cur) <= DUST_QTY: continue`，"
        "cur == 0 到不了这里。由 test_flatten_side_is_guarded_by_the_dust_check 用 AST "
        "确认守卫在前，并实测零持仓不产生订单。",

    "app/db/execution_store.py ×1 — apply_broker_state `dq > 0` -> `>=`":
        "差异只在 dq == 0。该比较嵌在 `if abs(dq) > DUST_QTY:` 之内，dq == 0 进不来。"
        "由 test_increment_sign_is_inside_the_dust_guard 用 AST 确认嵌套关系。",

    "app/core/execution/fidelity.py ×1 — 与模拟账本匹配 `intended > 0` -> `>=`":
        "差异只在 intended == 0。条件写成 `abs(intended) <= WEIGHT_DUST or (intended > 0) != …`，"
        "or 短路：intended == 0 时第一项已为真，第二项不求值。由 "
        "test_match_direction_is_short_circuited_by_the_dust_check 用 AST 确认左右顺序。",

    "app/core/execution/order_builder.py ×1 — build_rebalance_orders `side = \"BUY\" if delta > 0` -> `>=`":
        "差异只在 delta == 0。之前有 `if delta == 0: continue`（审计修复批把带宽判断挪到它之后），"
        "delta == 0 到不了这里。由 test_side_is_decided_after_the_zero_delta_skip 用 AST 确认顺序，"
        "并实测目标 = 持仓时不产生订单。",

    "app/core/execution/order_manager.py ×1 — 撤旧单的筛选 `status == SUBMITTED and broker_order_id` -> `or`":
        "两式只在 (SUBMITTED 且无订单号) 或 (非 SUBMITTED 且有订单号) 时不同。执行账本里每一条"
        "把状态设为 SUBMITTED 的路径（mark_submitted / adopt_broker_order / apply_broker_state）"
        "都在同一函数里写订单号；写意图（PENDING）时订单号一律置 None —— 两种组合都不可达。"
        "由 test_submitted_rows_always_carry_a_broker_order_id 用 AST 逐函数核对。",

    "app/core/portfolio_manager/manager.py ×1 — 首段等权 `1.0 / len(factor_signals)` -> `*`":
        "两式给出的权重只差一个正的公共倍数（1/n 与 n）。AlphaCombiner.combine 对权重做归一化，"
        "合成信号与权重的整体尺度无关。由 test_combine_is_invariant_to_weight_scale 实算验证前提。",

    "app/core/execution/fidelity.py ×1 — 成交方向 `q > 0` -> `>=`":
        "差异只在 q == 0。该行之前 `if ref <= 0 or px <= 0 or abs(q) <= DUST_QTY: continue` "
        "已排除 q == 0。由 test_fill_sign_is_guarded_by_the_dust_check 用 AST 确认守卫在前。",
}

NEW_POINT_LOUD_AT_IMPORT = {
    "app/db/execution_store.py ×5 — 五张表的 `primary_key=True` -> False":
        "SQLAlchemy 声明式映射要求主键，去掉后类定义时就抛 ArgumentError，"
        "任何导入 execution_store 的测试文件收集即失败。"
        "由 test_removing_any_primary_key_breaks_the_import 逐个施加变异并执行源码验证。",

    "app/core/execution/pretrade_gate.py ×1 — PreTradeLimits 校验 `if not (...)` 删掉 not":
        "变异后**合法**的限额（正有限数）会被拒，构造即抛 ValueError；"
        "生产里 from_settings、测试里模块级的限额常量都在导入 / 收集时崩。"
        "由 test_inverted_limit_validation_rejects_valid_limits 施加变异并执行源码验证。",

    "app/tasks/forward.py ×1 — `if __name__ == \"__main__\"` -> `!=`":
        "变异后模块被**导入**时就执行 main()：argparse 解析的是宿主进程的 argv（pytest 的参数），"
        "不认识 → SystemExit(2)，导入 app.tasks.forward 的测试文件收集即失败（变异工具记为"
        "internal_error）。由 test_inverted_main_guard_exits_on_import 施加变异并执行源码验证。",
}


# ===========================================================================
# 机械验证
# ===========================================================================

def _tree(mod) -> ast.Module:
    return ast.parse(Path(inspect.getfile(mod)).read_text(encoding="utf-8"))


def _func(tree: ast.Module, name: str) -> ast.FunctionDef:
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise AssertionError(f"找不到函数 {name} —— 证明需要重写")


def _is_dust_guard(node: ast.AST, var: str) -> bool:
    """`abs(var) <= DUST_QTY` 或 `abs(var) > DUST_QTY` 形式的比较。"""
    return (isinstance(node, ast.Compare) and isinstance(node.left, ast.Call)
            and getattr(node.left.func, "id", None) == "abs"
            and getattr(node.left.args[0], "id", None) == var
            and getattr(node.comparators[0], "id", None) == "DUST_QTY")


def test_tick_boundary_is_on_both_grids():
    for tick in (0.01, 0.0001):
        for side in ("BUY", "SELL"):
            n = 1.0 / tick
            n = math.floor(n + 1e-9) if side == "BUY" else math.ceil(n - 1e-9)
            assert round(n * tick, 4) == 1.0, f"1.0 在价位 {tick} 下取整成了 {n * tick}"
    assert ob.round_to_tick(1.0, "BUY") == ob.round_to_tick(1.0, "SELL") == 1.0


def test_flatten_side_is_guarded_by_the_dust_check():
    fn = _func(_tree(ob), "build_flatten_orders")
    guard_line = side_line = None
    for n in ast.walk(fn):
        if isinstance(n, ast.If) and _is_dust_guard(n.test, "cur"):
            guard_line = n.lineno
        if (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "side"):
            side_line = n.lineno
    assert guard_line and side_line and guard_line < side_line, "零股守卫不在方向判断之前"
    from app.core.execution.broker_gateway import BrokerPosition
    res = ob.build_flatten_orders({"Z": BrokerPosition("Z", 0.0, 0.0, 10.0, 0.0)},
                                  "20250303000000", book_id=-1, band_bps=500.0)
    assert res.orders == [] and res.skipped == []


def test_increment_sign_is_inside_the_dust_guard():
    fn = _func(_tree(es), "apply_broker_state")
    outer = [n for n in ast.walk(fn) if isinstance(n, ast.If) and _is_dust_guard(n.test, "dq")]
    assert len(outer) == 1, "找不到 `if abs(dq) > DUST_QTY` 守卫"
    # 只钉前提（"dq 与 0 的比较位于守卫之内"），不钉运算符本身 —— 钉运算符等于源码字面断言，
    # 会把这个等价点虚记成"被杀死"
    inner = [n for n in ast.walk(outer[0]) if isinstance(n, ast.Compare)
             and getattr(n.left, "id", None) == "dq"
             and getattr(n.comparators[0], "value", None) == 0]
    assert inner, "`dq` 与 0 的比较不在零股守卫之内 —— 等价性前提失效"


def test_match_direction_is_short_circuited_by_the_dust_check():
    fn = _func(_tree(fid), "build_fidelity_report")
    ors = [n for n in ast.walk(fn) if isinstance(n, ast.BoolOp) and isinstance(n.op, ast.Or)
           and any(isinstance(v, ast.Compare) and isinstance(v.left, ast.Call)
                   and getattr(v.left.func, "id", None) == "abs"
                   and getattr(v.left.args[0], "id", None) == "intended" for v in n.values)]
    assert len(ors) == 1, "匹配条件的结构变了 —— 证明需要重写"
    first = ors[0].values[0]
    assert (isinstance(first, ast.Compare) and isinstance(first.ops[0], ast.LtE)
            and getattr(first.comparators[0], "id", None) == "WEIGHT_DUST"), \
        "`abs(intended) <= WEIGHT_DUST` 不再是 or 的第一项，短路不成立"


def test_fill_sign_is_guarded_by_the_dust_check():
    fn = _func(_tree(fid), "build_fidelity_report")
    guard_line = sign_line = None
    for n in ast.walk(fn):
        if (isinstance(n, ast.If) and isinstance(n.test, ast.BoolOp)
                and any(_is_dust_guard(v, "q") for v in n.test.values)
                and isinstance(n.body[0], ast.Continue)):
            guard_line = n.lineno
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "s":
            sign_line = n.lineno
    assert guard_line and sign_line and guard_line < sign_line, "零股守卫不在方向判断之前"


def test_side_is_decided_after_the_zero_delta_skip():
    fn = _func(_tree(ob), "build_rebalance_orders")
    skip = side = None
    for n in ast.walk(fn):
        if (isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                and getattr(n.test.left, "id", None) == "delta"
                and isinstance(n.test.ops[0], ast.Eq)
                and getattr(n.test.comparators[0], "value", None) == 0
                and isinstance(n.body[0], ast.Continue)):
            skip = n.lineno
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "side":
            side = n.lineno
    assert skip and side and skip < side, "`if delta == 0: continue` 不在方向判断之前"
    import pandas as pd
    from datetime import date
    res = ob.build_rebalance_orders(pd.Series({"A": .25}), {"A": 256.0}, pd.Series({"A": 1.0}),
                                    1024.0, date(2025, 3, 3), book_id=-1, band_bps=0.)
    assert res.orders == [] and res.skipped == []


def test_submitted_rows_always_carry_a_broker_order_id():
    tree = _tree(es)
    dumps = {fn.name: ast.dump(fn) for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef)}
    sets_submitted = [name for name, d in dumps.items()
                      if ("ST_SUBMITTED" in d or "local_status_from_broker" in d)
                      and name not in ("local_status_from_broker", "open_orders")]
    assert {"mark_submitted", "adopt_broker_order", "apply_broker_state"} <= set(sets_submitted)
    missing = [name for name in sets_submitted if "broker_order_id" not in dumps[name]]
    assert missing == [], f"这些函数设置了 SUBMITTED 却没写订单号：{missing}"
    planned = _func(tree, "_upsert_planned")
    assert any(isinstance(n, ast.Assign) and getattr(n.targets[0], "attr", None) == "broker_order_id"
               and getattr(n.value, "value", "x") is None for n in ast.walk(planned)), \
        "写意图时没有把订单号置 None"


def test_combine_is_invariant_to_weight_scale():
    import numpy as np
    import pandas as pd
    from app.core.backtest_engine.alpha_combiner import AlphaCombiner
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2025-01-01", periods=30)
    sig = {k: pd.DataFrame(rng.normal(size=(30, 6)), index=idx) for k in ("a", "b", "c")}
    c = AlphaCombiner()
    base = c.combine(sig, weights={k: 1 / 3 for k in sig})
    pd.testing.assert_frame_equal(base, c.combine(sig, weights={k: 3.0 for k in sig}))


def _exec_source(name: str, source: str) -> types.ModuleType:
    """在一个全新、临时登记进 sys.modules 的模块里执行源码（dataclass 需要按名字找到模块）。"""
    import sys
    ns = types.ModuleType(name)
    sys.modules[name] = ns
    try:
        exec(compile(source, name, "exec"), ns.__dict__)
    finally:
        sys.modules.pop(name, None)
    return ns


@pytest.mark.parametrize("table", ["exec_orders", "exec_fills", "exec_snapshots",
                                   "exec_events", "exec_state"])
def test_removing_any_primary_key_breaks_the_import(table):
    import sqlalchemy.exc
    src = Path(inspect.getfile(es)).read_text(encoding="utf-8").splitlines(keepends=True)
    # 找到该表的 class 段，在段内唯一一行 primary_key=True 上施加变异
    start = next(i for i, ln in enumerate(src) if f'__tablename__ = "{table}"' in ln)
    end = next((i for i in range(start + 1, len(src)) if src[i].startswith("class ")), len(src))
    pk = [i for i in range(start, end) if "primary_key=True" in src[i]]
    assert len(pk) == 1, f"{table} 的主键声明不是恰好一行"
    mutated = list(src)
    mutated[pk[0]] = mutated[pk[0]].replace("primary_key=True", "primary_key=False", 1)
    with pytest.raises(sqlalchemy.exc.ArgumentError, match="primary key"):
        _exec_source(f"_mut_execution_store_{table}", "".join(mutated))
    _exec_source(f"_orig_execution_store_{table}", "".join(src))     # 对照：原文能执行


def test_inverted_limit_validation_rejects_valid_limits():
    src = Path(inspect.getfile(pg)).read_text(encoding="utf-8")
    pattern = "if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):"
    assert src.count(pattern) == 1, "限额校验那一行的形态变了 —— 证明需要重写"
    valid = dict(max_name_weight=0.1, max_gross=1.0, max_participation_pct=0.1,
                 max_daily_loss=0.05, max_price_deviation=0.15)
    mut = _exec_source("_mut_pretrade_gate", src.replace(pattern, pattern.replace("if not ", "if ", 1)))
    with pytest.raises(ValueError, match="正的有限数"):
        mut.PreTradeLimits(**valid)
    orig = _exec_source("_orig_pretrade_gate", src)                  # 对照：原文接受合法限额
    assert orig.PreTradeLimits(**valid).max_gross == 1.0


def test_inverted_main_guard_exits_on_import(monkeypatch):
    import sys
    from app.tasks import forward
    src = Path(inspect.getfile(forward)).read_text(encoding="utf-8")
    guard = 'if __name__ == "__main__":'
    assert src.count(guard) == 1, "入口守卫的形态变了 —— 证明需要重写"
    monkeypatch.setattr(sys, "argv", ["pytest", "-q", "tests/"])
    with pytest.raises(SystemExit) as ei:
        _exec_source("_mut_forward", src.replace(guard, guard.replace("==", "!="), 1))
    assert ei.value.code == 2                                        # argparse 拒绝未知参数
    assert hasattr(_exec_source("_orig_forward", src), "main")       # 对照：原文导入不执行


def test_every_disposition_is_written_out():
    for table in (NEW_POINT_EQUIVALENCE, NEW_POINT_LOUD_AT_IMPORT):
        for key, why in table.items():
            assert re.match(r"^app/.+\.py ×\d+ — ", key), f"证明键格式不对：{key}"
            assert len(why) >= 60, f"{key} 的说明过于敷衍"
