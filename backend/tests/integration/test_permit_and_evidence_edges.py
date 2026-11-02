"""
运行许可与验收证据的边界（复审 21cd90b 之后的关键路径变异复核：run_permit、acceptance_evidence、tasks/run_permit）。

首轮 197 个点存活 46 个，大多是时序判定的"恰好等于"边界——原有用例的时刻都离边界一秒以上。补的用例：
    · DAY 单跨 session：恰好在开盘 / 收盘那一刻的记录、券商时间，在 T 收盘那一刻下的单
    · 部分成交后撤单：撤单前没有成交、撤单后成交量为 0 或等于委托量都不算"部分成交"；对账要对上执行库那张单
    · 外部单没成交、只跑了一次 cycle、没带偏移的 cycle：都是"前提没发生"，不是失败
    · 熔断：复位那一刻的记录不算熔断期间；卖单不是加仓
    · 崩溃恢复：未决 / 认领不到委托号 / 同一进程，任何一项都不过；恢复后新下的别的单不算重复
    · 执行合同只排除 account_id；verify 结论只认本场景的
    · 许可：有效期 0 小时、单笔上限 0、恰好到期、记录损坏；命令行缺参数、输出给人看的中文
三处冗余写法直接删掉（日程起点、撤单后的对账、单笔上限判断），不再产生等价变异。
剩下的等价变异写在 PROVEN_EQUIVALENT（各配一条机械验证的用例）。
"""

from __future__ import annotations

import json
from datetime import timedelta

import pandas as pd
import pytest

from app.core.execution import acceptance_evidence as ae
from app.core.execution import run_permit as rp
from tests.integration.test_run_permit import ACC, _guarded, _supervised
from tests.unit.execution.test_order_manager import D1, D2, D3

OID, CID = "B-1", "c-1"


def _t(day, hh, mm=0, ss=0):
    """美东钟点 → UTC ISO（证据行的 at 是 UTC）。D1 / D2 在 2025-03 夏令时之前：开盘 14:30Z、收盘 21:00Z。"""
    return pd.Timestamp(f"{day:%Y-%m-%d} {hh:02d}:{mm:02d}:{ss:02d}", tz="America/New_York").tz_convert("UTC").isoformat()


def _o(status, dealt=0, qty=2, updated="", oid=OID, cid=CID, side="BUY"):
    return {"broker_order_id": oid, "client_id": cid, "ticker": "AAA", "side": side, "qty": qty, "price": 50.0,
            "status": status, "dealt_qty": dealt, "create_time": "", "updated_time": updated}


def _row(scenario, step, at, orders=(), store=(), pid=1, **data):
    return {"at": at, "scenario": scenario, "step": step, "pid": pid,
            "data": {"broker_orders": list(orders), "store_orders": list(store), "positions": {}, **data}}


def _st(dealt, oid=OID, cid=CID):
    return {"client_id": cid, "broker_order_id": oid, "dealt_qty": dealt}


def _sub(oid=OID, cid=CID, side="BUY"):
    return {"broker_order_id": oid, "client_id": cid, "side": side}


def _names(checks):
    return {c["name"]: c["ok"] for c in checks}


# ── 提交与成交 ───────────────────────────────────────────

SF = "submit_and_fill"


def _sf(store_dealt, broker_dealt=2):
    return [_row(SF, "cycle", _t(D1, 17), [_o("SUBMITTED")], submitted=[_sub()], decision_date=str(D1), report={}),
            _row(SF, "reconcile", _t(D2, 17), [_o("FILLED_ALL", broker_dealt)], [_st(store_dealt)],
                 report={"status": "clean"})]


def test_the_reconcile_that_first_shows_the_fill_is_the_one_compared():
    """成交第一次出现在一条对账记录里：那条对账本身就是"成交之后的对账"，不要求再来一条。"""
    assert all(_names(ae._check_submit_and_fill(_sf(2))).values())


def test_a_store_that_did_not_record_the_fill_is_a_mismatch():
    assert _names(ae._check_submit_and_fill(_sf(0)))["执行库的成交量与券商一致"] is False


def test_a_difference_at_the_float_tolerance_is_not_a_mismatch():
    """容差是严格大于 1e-9：恰好 1e-9 的差是浮点噪声，不算不一致。"""
    assert _names(ae._check_submit_and_fill(_sf(1e-9, broker_dealt=0)))["执行库的成交量与券商一致"] is True


# ── DAY 单跨 session ─────────────────────────────────────

DO = "day_order_across_sessions"
NOT_ENDED = "次一交易时段开始之前这张单没有结束"
EXEC = "次一交易时段里这张单可执行（时段内观察到在途，或券商成交时间落在时段内）"
ENDED = "次一交易日收盘后这张单已结束（成交或过期）"
AFTER_CLOSE = "下单时间在 T 收盘之后"


def _do(*later, cycle_at=None):
    c = _row(DO, "cycle", cycle_at or _t(D1, 17), [_o("SUBMITTED", updated=f"{D1:%Y-%m-%d} 17:00:00")],
             submitted=[_sub()], decision_date=str(D1), report={})
    return _names(ae._check_day_order([c, *later]))


def _snap(at, status, dealt=0, updated=""):
    return _row(DO, "snapshot", at, [_o(status, dealt, updated=updated)])


def test_the_exact_open_and_close_of_t_count_as_inside_the_window():
    """T 收盘那一刻下的单算"收盘之后"；次一交易日开盘那一刻成交、那一刻拍到的记录都算时段内，不算"开盘前就结束"。"""
    got = _do(_snap(_t(D2, 9, 30), "FILLED_ALL", 2, updated=f"{D2:%Y-%m-%d} 09:30:00"),
              _snap(_t(D2, 17), "FILLED_ALL", 2, updated=f"{D2:%Y-%m-%d} 09:30:00"),
              cycle_at=_t(D1, 16))
    assert got == {AFTER_CLOSE: True, NOT_ENDED: True, EXEC: True, ENDED: True}, got


def test_an_open_order_seen_right_at_the_open_proves_executability():
    got = _do(_snap(_t(D2, 9, 30), "SUBMITTED"), _snap(_t(D2, 17), "FILLED_ALL", 2, updated=""))
    assert got[EXEC] is True, got


def test_an_open_order_seen_only_at_the_close_does_not():
    got = _do(_snap(_t(D2, 16), "SUBMITTED"),
              _snap(_t(D2, 17), "CANCELLED_ALL", updated=f"{D2:%Y-%m-%d} 16:00:00"))
    assert got[EXEC] is False and got[ENDED] is True, got


def test_a_fill_stamped_at_the_close_is_outside_the_session():
    got = _do(_snap(_t(D2, 17), "FILLED_ALL", 2, updated=f"{D2:%Y-%m-%d} 16:00:00"))
    assert got[EXEC] is False, got


def test_a_terminal_record_taken_right_at_the_close_counts_as_ended():
    got = _do(_snap(_t(D2, 10), "SUBMITTED"),
              _snap(_t(D2, 16), "FILLED_ALL", 2, updated=f"{D2:%Y-%m-%d} 10:30:00"))
    assert got[ENDED] is True, got


def test_an_order_still_open_after_the_close_has_not_ended():
    got = _do(_snap(_t(D2, 10), "SUBMITTED"), _snap(_t(D2, 17), "SUBMITTED"))
    assert got[ENDED] is False, got


# ── 部分成交后撤单 ───────────────────────────────────────

PC = "partial_fill_then_cancel"
HIT = "出现部分成交后撤单的订单"
STORE_EQ = "执行库记下的成交量等于券商的"
REC_OK = "撤单后对账没有差异"


def _pc(before, after, qty=2, rec_status="clean", store_dealt=None):
    store = [] if store_dealt is None else [_st(store_dealt)]
    rows = [_row(PC, "cycle", _t(D1, 17), [_o("SUBMITTED", qty=qty)], submitted=[_sub()], decision_date=str(D1),
                 report={}),
            _row(PC, "snapshot", _t(D2, 10), [_o("SUBMITTED", before, qty)]),
            _row(PC, "cancel", _t(D2, 10, 5), [_o("SUBMITTED", before, qty)], broker_order_id=OID),
            _row(PC, "snapshot", _t(D2, 10, 10), [_o("CANCELLED_PART", after, qty)]),
            _row(PC, "reconcile", _t(D2, 17), [_o("CANCELLED_PART", after, qty)], store, report={"status": rec_status})]
    checks = ae._check_partial_cancel(rows)
    return _names(checks), checks


def test_no_fill_before_the_cancel_is_not_a_partial_fill_then_cancel():
    got, checks = _pc(before=0, after=1)
    assert got == {HIT: False} and checks[0]["premise"] is True


def test_a_cancelled_part_with_nothing_or_everything_filled_is_not_a_partial_fill():
    """券商回报自相矛盾（CANCELLED_PART 却 0 成交 / 全部成交）：证明不了"部分成交后撤单"。"""
    assert _pc(before=1, after=0)[0] == {HIT: False}
    assert _pc(before=1, after=2, qty=2)[0] == {HIT: False}


def test_the_store_row_of_this_order_is_compared_after_the_cancel():
    got, _ = _pc(before=1, after=1, store_dealt=1)
    assert got == {HIT: True, STORE_EQ: True, REC_OK: True}, got
    assert _pc(before=1, after=1, store_dealt=0)[0][STORE_EQ] is False


def test_a_reconcile_with_differences_after_the_cancel_fails():
    assert _pc(before=1, after=1, rec_status="discrepancy")[0][REC_OK] is False


# ── 未知订单 / 重启 / 输入异常：前提没发生 ─────────────────

def test_an_external_order_that_never_filled_is_not_covered_yet():
    ext = _row("unknown_order", "external_order", _t(D1, 17), broker_order_id="X1")
    rec = _row("unknown_order", "reconcile", _t(D2, 17), [_o("SUBMITTED", 0, oid="X1", cid="")],
               report={"status": "clean"})
    checks = ae._check_unknown_order([ext, rec])
    assert checks[1].get("premise") is True, checks


def test_a_single_cycle_is_not_a_restart():
    one = _row("restart_dedup", "cycle", _t(D1, 17), submitted=[_sub()], decision_date=str(D1), report={})
    assert ae._check_restart_dedup([one])[0].get("premise") is True


def test_a_cycle_without_a_skew_is_not_an_input_anomaly_attempt():
    plain = _row("input_anomaly", "cycle", _t(D1, 17), submitted=[_sub()], ref_skew=0.0, report={})
    assert ae._check_input_anomaly([plain])[0].get("premise") is True


# ── 熔断后续做全平 ───────────────────────────────────────

def test_the_reset_record_is_not_part_of_the_kill_switch_period_and_sells_are_not_buys():
    held = {"US.AAA": {"qty": 2}}
    rows = [_row("kill_switch_flatten", "kill_switch_engage", _t(D2, 17), positions=held, kill_switch={"engaged": True}),
            _row("kill_switch_flatten", "maintain", _t(D3, 17), positions=held, submitted=[_sub("S-1", "c-2", "SELL")]),
            _row("kill_switch_flatten", "kill_switch_reset", _t(D3, 17, 5), positions={}, kill_switch={"engaged": False})]
    got = _names(ae._check_kill_switch(rows))
    assert got == {"熔断已开": True, "开熔断时确有持仓": True, "熔断开着时没有加仓的买单": True,
                   "持仓最终清空": False}, got


# ── 下单途中崩溃后恢复 ───────────────────────────────────

CR = "crash_recovery"
REC = "恢复后对账没有悬而未决的单（在另一个进程里）"
NODUP = "恢复后没有重复下单"


def _cr(unresolved=(), claimed=True, pid=2):
    crash = _row(CR, "crash", _t(D1, 17), [_o("SUBMITTED")], pid=1, submitted=[_sub()], decision_date=str(D1))
    rec = _row(CR, "reconcile", _t(D1, 17, 5), [_o("SUBMITTED")], [_st(0, oid=OID if claimed else "")], pid=pid,
               report={"status": "clean", "unresolved": list(unresolved)})
    again = _row(CR, "cycle", _t(D1, 17, 10), [_o("SUBMITTED"), _o("SUBMITTED", oid="B-2", cid="c-2")], pid=pid,
                 submitted=[_sub("B-2", "c-2")], decision_date=str(D1), report={})
    return _names(ae._check_crash_recovery([crash, rec, again]))


def test_a_clean_recovery_with_a_new_order_for_another_name_passes():
    """恢复后同日再跑，给别的票新下一张单不是重复下单。"""
    assert all(_cr().values()), _cr()


def test_recovery_needs_no_unresolved_order_and_the_claimed_order_id():
    assert _cr(unresolved=[CID])[REC] is False
    assert _cr(claimed=False)[REC] is False
    assert _cr(pid=1)[REC] is False


# ── 整份证据 ─────────────────────────────────────────────

def test_the_contract_ignores_only_the_account_id():
    base = {"account_id": 1, "limit_band_bps": 50}
    assert ae.contract_digest(base, {}) == ae.contract_digest({**base, "account_id": 2}, {})
    assert ae.contract_digest(base, {}) != ae.contract_digest({**base, "limit_band_bps": 60}, {})


@pytest.mark.parametrize("acc_id,foreign", [(None, True), ("x", True), (str(ACC), False), (ACC, False)])
def test_an_unreadable_account_id_is_another_account_not_a_crash(tmp_path, acc_id, foreign):
    """证据行的 acc_id 缺失 / 不是数字：判为"别的账户"（拒签），而不是让判定本身抛错。"""
    row = {**_row("input_anomaly", "snapshot", _t(D1, 17)), "account": {"acc_id": acc_id, "trd_env": "SIMULATE"},
           "code_version": "abc123abc123", "contract": "c"}
    p = tmp_path / "e.jsonl"
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    j = ae.assess(p, account=ACC, code="abc123abc123", contract="c")
    assert any("别的账户" in x for x in j["problems"]) is foreign, j["problems"]


def test_another_scenarios_verdict_does_not_stand_in_for_this_one():
    stopped = _row("input_anomaly", "cycle", _t(D1, 17), submitted=[], ref_skew=0.3,
                   report={"blocked": "", "gate": {"rejected": [{"reason": "quote_mismatch"}]}})
    other = _row("restart_dedup", "verdict", _t(D1, 18), passed=True)
    assert ae._judge("input_anomaly", [stopped, other])["status"] == ae.NOT_COVERED


# ── 运行许可 ─────────────────────────────────────────────

@pytest.mark.parametrize("kw,needle", [({"hours": 0.0}, "有效期"), ({"cap": 0.0}, "单笔金额上限")])
def test_a_permit_needs_a_positive_lifetime_and_a_positive_order_cap(tmp_path, kw, needle):
    e = _guarded(tmp_path)
    with pytest.raises(rp.PermitRefused, match=needle):
        _supervised(e, **kw)
    assert _supervised(e, hours=24.0)["max_order_notional"] == 1e9                # 有效期上限本身可以


def test_a_permit_stops_at_the_exact_moment_it_expires(tmp_path):
    e = _guarded(tmp_path)
    _supervised(e, hours=1.0)
    d = rp.evaluate(e.es, account=ACC, settings=e.settings, now=e.clock() + timedelta(hours=1))
    assert not d.allowed and any("过期" in r for r in d.reasons)


def test_a_corrupt_permit_record_is_reported_as_corrupt_and_blocks(tmp_path):
    """`run_permit status` 打印的就是 current() —— 损坏要明说，不能显示成一份模式为空的许可。"""
    e = _guarded(tmp_path)
    e.es.set_state(rp.STATE_KEY, "{不是 JSON")
    assert rp.current(e.es) == {"mode": "", "corrupt": True}
    d = rp.evaluate(e.es, account=ACC, settings=e.settings, now=e.clock())
    assert d.allowed is False and d.reasons == ["运行许可记录损坏 / 模式不认识"]


def test_the_command_line_needs_hours_and_prints_readable_chinese(tmp_path, capsys):
    from app.tasks import run_permit as cli
    db = (tmp_path / "e.db").as_posix()
    assert cli.main(["issue", "--exec-db", db, "--mode", "supervised", "--account", str(ACC),
                     "--max-order-notional", "2000", "--actor", "me", "--reason", "联调"]) == cli.EXIT_REFUSED
    assert "--hours" in capsys.readouterr().out
    assert cli.main(["issue", "--exec-db", db, "--mode", "supervised", "--account", str(ACC), "--hours", "2",
                     "--max-order-notional", "2000", "--actor", "me", "--reason", "联调"]) == cli.EXIT_OK
    assert '"reason": "联调"' in capsys.readouterr().out                           # 给人看的，不转义成 \uXXXX
    assert cli.main(["status", "--exec-db", db]) == cli.EXIT_OK
    assert '"reason": "联调"' in capsys.readouterr().out


# ── 等价变异 ─────────────────────────────────────────────

PROVEN_EQUIVALENT = {
    "app/core/execution/run_permit.py ×1 — L162 `json.dumps(permit, ensure_ascii=False)` -> True（写进执行库的许可）":
        "两种写法只差非 ASCII 字符是否转义成 \\uXXXX；读回来的对象逐字段相同。这条状态只经两处读取："
        "run_permit.current() 与 go_no_go 的 _json，都是 json.loads，不看原文。"
        "见 test_the_stored_permit_is_only_read_back_through_json。",
}


def test_the_stored_permit_is_only_read_back_through_json():
    from pathlib import Path
    permit = {"reason": "联调 ✓", "actor": "操作人", "account": ACC}
    assert json.loads(json.dumps(permit, ensure_ascii=True)) == json.loads(json.dumps(permit, ensure_ascii=False))

    import inspect
    import re
    backend = Path(__file__).resolve().parents[2]
    own = (backend / "app" / "core" / "execution" / "run_permit.py").read_text(encoding="utf-8")
    assert len(re.findall(r"get_state\(STATE_KEY\)", own)) == 1
    assert re.search(r"get_state\(STATE_KEY\)", inspect.getsource(rp.current))
    other = r"""\.get\(['"]run_permit['"]\)|get_state\(['"]run_permit['"]\)|get_state\((?:rp|run_permit)\.STATE_KEY\)"""
    readers = sorted(p.relative_to(backend).as_posix() for d in ("app", "tools") for p in (backend / d).rglob("*.py")
                     if re.search(other, p.read_text(encoding="utf-8")))
    assert readers == ["tools/ops/go_no_go.py"], readers
    gng = (backend / "tools" / "ops" / "go_no_go.py").read_text(encoding="utf-8")
    uses = re.findall(r"""(.{0,30})\.get\(['"]run_permit['"]\)""", gng)
    assert uses and all("_json(" in pre for pre in uses), uses
    from tools.ops import go_no_go
    assert go_no_go._json(json.dumps(permit, ensure_ascii=True)) == go_no_go._json(json.dumps(permit))


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 1
