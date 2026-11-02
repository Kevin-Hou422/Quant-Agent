"""
无人值守许可与现场验收证据的绑定（复审固定提交 21cd90b，"放行边界 1"）。

复审原话："当前代码看起来只要求验收文件存在并记录哈希；Go/No-Go 分别读取许可和所提供的验收文件，还需要证明两者是同一份
有效证据。"要求："失败、未覆盖、版本或账户不一致、文件变更时，不能自动认定无人值守就绪……补充错误文件、另一账户、另一代码
版本、缺场景、失败 verdict、文件修改、许可与报告引用不同文件等用例。"

修复前（HEAD 21cd90b，反例 C:\\QuantAgentEnv\\review-repro\\repro_at_21cd90b.json）：八个场景全是"不通过"的结论、
一行随便写的字、另一账户、另一代码版本 —— 四种文件都签出了无人值守许可；Go/No-Go 对手写的八行"通过"判全部通过。

证据用 `record_all_scenarios`：在替身券商上把八个场景真的走一遍（与 test_simulate_acceptance 同一流程）。
"""

from __future__ import annotations

import json
import shutil

import pytest

from app.core.execution import acceptance_evidence as ae
from app.core.execution import run_permit as rp
from tests.integration.test_simulate_acceptance import record_all_scenarios

ACC = 2001
CODE = "abc123abc123"
SPEC_H = "h" * 64


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    return record_all_scenarios(tmp_path_factory.mktemp("acceptance"), code=CODE)


@pytest.fixture
def evidence(recorded, tmp_path):
    p = tmp_path / "simulate_acceptance.jsonl"
    shutil.copy(recorded, p)
    return p


def _settings():
    from app.config import settings
    return settings.model_copy(update={"price_source": "moomoo"})


def _store(tmp_path):
    from app.db.execution_store import ExecutionStore
    return ExecutionStore(db_url=f"sqlite:///{(tmp_path / 'exec.db').as_posix()}")


def _issue(tmp_path, evidence, **kw):
    kw.setdefault("code", CODE)
    kw.setdefault("contract", ae.contract_of_settings())
    st = kw.pop("store", None) or _store(tmp_path)
    return st, rp.issue(st, mode="unattended", actor="tester", reason="无人值守", account=kw.pop("account", ACC),
                        hours=48, acceptance_evidence=str(evidence), settings=_settings(), spec_hash=SPEC_H,
                        health=lambda s, now: [], **kw)


def _evaluate(st, **kw):
    kw.setdefault("code", CODE)
    kw.setdefault("contract", ae.contract_of_settings())
    return rp.evaluate(st, account=ACC, settings=_settings(), spec_hash=SPEC_H, health=lambda s, now: [], **kw)


def _rows(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _write(p, rows):
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _refused(tmp_path, evidence, needle, **kw):
    with pytest.raises(rp.PermitRefused, match=needle):
        _issue(tmp_path, evidence, **kw)


# ── 签发 ─────────────────────────────────────────────────

def test_a_complete_real_evidence_file_is_bound_with_its_judgement(tmp_path, evidence):
    st, p = _issue(tmp_path, evidence)
    b = p["acceptance_evidence"]
    assert b["path"] == str(evidence.resolve()) and b["sha256"] == ae.file_sha256(evidence)
    assert b["scenarios"] == {s: ae.PASSED for s in ae.SCENARIOS} and b["contract"] == ae.contract_of_settings()
    assert _evaluate(st).allowed


@pytest.mark.parametrize("body,needle", [
    ("随便写的\n", "不是完整的 JSON"), ("", "空的"), ('{"scenario": "submit_and_fill", "step": "verdict"}\n', "缺字段")])
def test_a_wrong_file_is_refused(tmp_path, body, needle):
    p = tmp_path / "wrong.jsonl"
    p.write_text(body, encoding="utf-8")
    _refused(tmp_path, p, needle)


def test_a_truncated_line_in_an_otherwise_complete_file_is_refused(tmp_path, evidence):
    with evidence.open("a", encoding="utf-8") as f:
        f.write('{"at": "2026-10-30T21:00:00+00:00", "scenario": "submit_')
    _refused(tmp_path, evidence, "不是完整的 JSON")


def test_handwritten_passing_verdicts_without_records_are_not_coverage(tmp_path):
    """修复前 Go/No-Go 只看结论行：八行手写的"通过"就算过。"""
    p = tmp_path / "fake.jsonl"
    _write(p, [{"at": "2026-10-30T21:00:00+00:00", "scenario": s, "step": "verdict", "account":
                {"trd_env": "SIMULATE", "acc_id": ACC}, "code_version": CODE, "contract": ae.contract_of_settings(),
                "data": {"passed": True, "checks": []}} for s in ae.SCENARIOS])
    _refused(tmp_path, p, "not_covered")


def test_evidence_from_another_account_is_refused(tmp_path, evidence):
    _refused(tmp_path, evidence, "别的账户", account=3003)
    rows = _rows(evidence)
    rows[3]["account"]["acc_id"] = 3003                    # 中途换了账户的一行
    _write(evidence, rows)
    _refused(tmp_path, evidence, "别的账户")


def test_evidence_from_a_real_money_environment_is_refused(tmp_path, evidence):
    rows = _rows(evidence)
    rows[0]["account"]["trd_env"] = "REAL"
    _write(evidence, rows)
    _refused(tmp_path, evidence, "SIMULATE")


def test_evidence_from_another_code_version_is_refused(tmp_path, evidence):
    _refused(tmp_path, evidence, "别的代码版本", code="fedcba987654")


def test_evidence_under_another_execution_contract_is_refused(tmp_path, evidence):
    """验收时的执行合同（规格的 execution / submission）与运行中策略的规格不一致 → 证据不支持这份规格。"""
    _refused(tmp_path, evidence, "执行合同", contract="0" * 16)
    _refused(tmp_path, evidence, "执行合同", contract=None)       # 不给 → 读运行中策略的规格；没有 → 无从比较


def test_a_missing_scenario_is_not_covered_and_refused(tmp_path, evidence):
    _write(evidence, [r for r in _rows(evidence) if r["scenario"] != "partial_fill_then_cancel"])
    _refused(tmp_path, evidence, "partial_fill_then_cancel：not_covered")


def test_a_failed_last_verdict_is_refused_even_if_the_records_would_pass(tmp_path, evidence):
    rows = _rows(evidence)
    last = [r for r in rows if r["scenario"] == "restart_dedup" and r["step"] == "verdict"][-1]
    rows.append({**last, "data": {**last["data"], "passed": False}})
    _write(evidence, rows)
    _refused(tmp_path, evidence, "restart_dedup：failed")


def test_a_passing_verdict_over_records_that_do_not_pass_is_refused(tmp_path, evidence):
    """结论行写着通过，但重算不过（对账记录被删了）→ 失败：只信重算。"""
    _write(evidence, [r for r in _rows(evidence) if not (r["scenario"] == "submit_and_fill" and r["step"] == "reconcile")])
    _refused(tmp_path, evidence, "submit_and_fill：failed")


def test_an_event_that_did_not_happen_is_not_covered_rather_than_failed():
    rows = [{"at": "2026-10-30T21:00:00+00:00", "scenario": "partial_fill_then_cancel", "step": "snapshot",
             "account": {"trd_env": "SIMULATE", "acc_id": ACC}, "code_version": CODE,
             "data": {"broker_orders": [], "store_orders": []}}]
    assert ae._judge("partial_fill_then_cancel", rows)["status"] == ae.NOT_COVERED    # noqa: SLF001


def test_only_an_uncovered_scenario_can_be_waived_and_only_with_a_reason(tmp_path, evidence):
    _write(evidence, [r for r in _rows(evidence) if r["scenario"] != "partial_fill_then_cancel"])
    _refused(tmp_path, evidence, "要写明理由", waive_uncovered={"partial_fill_then_cancel": "  "})
    st, p = _issue(tmp_path, evidence, waive_uncovered={"partial_fill_then_cancel": "当天没出现部分成交"})
    assert p["acceptance_evidence"]["scenarios"]["partial_fill_then_cancel"] == ae.WAIVED
    assert p["acceptance_evidence"]["waived"] == {"partial_fill_then_cancel": "当天没出现部分成交"}
    assert _evaluate(st).allowed


def test_a_failed_scenario_cannot_be_waived(tmp_path, evidence):
    _write(evidence, [r for r in _rows(evidence) if not (r["scenario"] == "submit_and_fill" and r["step"] == "reconcile")])
    _refused(tmp_path, evidence, "不能豁免", waive_uncovered={"submit_and_fill": "不想做"})


# ── 每轮核对 ─────────────────────────────────────────────

def test_any_change_to_the_bound_file_stops_new_risk(tmp_path, evidence):
    st, _p = _issue(tmp_path, evidence)
    with evidence.open("a", encoding="utf-8") as f:                 # 追加一条无害的记录：内容仍"通过"，但不是那一份了
        f.write(evidence.read_text(encoding="utf-8").splitlines()[0] + "\n")
    d = _evaluate(st)
    assert not d.allowed and any("被改过" in r for r in d.reasons)


def test_a_new_code_version_after_issuing_stops_new_risk(tmp_path, evidence):
    st, _p = _issue(tmp_path, evidence)
    d = _evaluate(st, code="fedcba987654")
    assert not d.allowed and any("代码版本变了" in r for r in d.reasons)
    assert any("别的代码版本" in r for r in d.reasons)              # 证据按此刻的代码重新判定也不通过


def test_a_new_spec_contract_after_issuing_stops_new_risk(tmp_path, evidence):
    st, _p = _issue(tmp_path, evidence)
    d = _evaluate(st, contract="0" * 16)
    assert not d.allowed and any("执行合同" in r for r in d.reasons)


def test_the_supervised_permit_stays_separate_and_capped(tmp_path):
    st = _store(tmp_path)
    p = rp.issue(st, mode="supervised", actor="t", reason="联调", account=ACC, hours=8, max_order_notional=2000,
                 settings=_settings(), code=CODE)
    assert p["acceptance_evidence"] is None and p["max_order_notional"] == 2000.0
    with pytest.raises(rp.PermitRefused, match="有效期"):
        rp.issue(st, mode="supervised", actor="t", reason="r", account=ACC, hours=48, max_order_notional=2000,
                 settings=_settings(), code=CODE)


def test_the_command_line_takes_waivers(tmp_path, evidence, monkeypatch):
    from app.tasks import run_permit as cli
    seen = {}
    monkeypatch.setattr(rp, "issue", lambda store, **kw: seen.update(kw) or {"ok": True})
    db = (tmp_path / "e.db").as_posix()
    assert cli.main(["issue", "--exec-db", db, "--mode", "unattended", "--account", str(ACC), "--hours", "48",
                     "--acceptance-evidence", str(evidence), "--actor", "me", "--reason", "r",
                     "--waive-uncovered", "partial_fill_then_cancel=当天没出现部分成交"]) == cli.EXIT_OK
    assert seen["waive_uncovered"] == {"partial_fill_then_cancel": "当天没出现部分成交"}
    assert cli.main(["issue", "--exec-db", db, "--mode", "unattended", "--account", str(ACC), "--hours", "48",
                     "--acceptance-evidence", str(evidence), "--actor", "me", "--reason", "r",
                     "--waive-uncovered", "没有等号"]) == cli.EXIT_REFUSED


def test_a_watchdog_that_lost_sight_of_a_holding_blocks_issuing_and_running(tmp_path, evidence):
    """复审"放行边界 2"对许可的影响：最近一拍有持仓的行情缺失 → 不签、已签的不新增风险；恢复的一拍之后放行。"""
    from app.core.execution.intraday_watchdog import STATE_KEY
    st = _store(tmp_path)
    bad = json.dumps({"at": "2026-10-30T15:00:00+00:00", "state": "partial", "names": {"AAA": "fresh", "BBB": "missing"}})
    st.set_state(STATE_KEY, bad)
    _refused(tmp_path, evidence, "盘中巡检", store=st)
    st.set_state(STATE_KEY, json.dumps({"at": "2026-10-30T15:01:00+00:00", "state": "fresh", "names": {"AAA": "fresh"}}))
    st, _p = _issue(tmp_path, evidence, store=st)
    assert _evaluate(st).allowed
    st.set_state(STATE_KEY, bad)
    d = _evaluate(st)
    assert not d.allowed and any("BBB" in r for r in d.reasons)
