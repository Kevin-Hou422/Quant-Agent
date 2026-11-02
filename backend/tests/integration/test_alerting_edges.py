"""
告警与盘中巡检的边界（复审 21cd90b 之后的关键路径变异复核：dispatcher、intraday_watchdog、tasks/alerts）。

首轮 129 个点存活 23 个。补的用例：
    · 事件文件：多层目录不存在也能写；中文原样写出（人要读它）；损坏行报的是文件里的真实行号
    · 去重恰好满窗口就不再压；重试只在两次之间等待、等待 1 秒、2 秒……
    · 就绪：事件文件读不了 → 不就绪；确认恰好 14 天仍有效；与确认同一时刻的失败 / 结果未知 / 写不进 → 都算在确认之后
    · 告警通道故障：落库成功 / 失败的返回值；执行库里的记录给人看（Go/No-Go 原样展示）
    · 巡检：恰好 120 秒后的行情时间不算未来、恰好 600 秒前的不算过期；只把过期的列为"偏旧"；
      "无法证明新鲜"满 600 秒告警一次、之后不重复
    · 命令行：不给子命令、confirm 不给 --actor → 参数错误
剩下的等价变异写在 PROVEN_EQUIVALENT（各配一条机械验证的用例）。
"""

from __future__ import annotations

import inspect
import re
import textwrap
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from app.core.alerting import dispatcher as disp
from app.core.execution import intraday_watchdog as iw
from tests.integration.test_alerting import Clock, Notifier, _dispatcher, _ev


@pytest.fixture(autouse=True)
def _own_fault_state(fresh_db, monkeypatch):
    monkeypatch.setattr(disp, "_PROCESS_FAULTS", [])
    return fresh_db


def _confirmed(tmp_path, clock):
    d = _dispatcher(tmp_path, clock=clock, confirm_valid_days=14)
    d.confirm_visible(d.send_test_notification(), actor="kevin")
    return d


# ── 事件文件 ─────────────────────────────────────────────

def test_the_event_file_may_sit_several_directories_deep_and_stays_readable(tmp_path):
    from app.core.alerting.dispatcher import AlertDispatcher
    d = AlertDispatcher(event_file=tmp_path / "a" / "b" / "events.jsonl", notifier=Notifier(), now=Clock(),
                        sleep=lambda s: None)
    d.emit(_ev())
    assert "摄取被拒" in d.path.read_text(encoding="utf-8")                  # 给人看的，不转义成 \uXXXX


def test_a_corrupt_line_is_reported_with_its_real_line_number(tmp_path):
    d = _dispatcher(tmp_path)
    d.emit(_ev())
    with open(d.path, "a", encoding="utf-8") as fh:
        fh.write("{截断\n")
    n = len(d.path.read_text(encoding="utf-8").splitlines())
    d.history()
    assert d.corrupt_lines == [n]


def test_a_repeat_exactly_one_dedup_window_later_is_delivered_again(tmp_path):
    clock = Clock()
    d = _dispatcher(tmp_path, clock=clock, dedup_minutes=60)
    d.emit(_ev())
    clock.t += timedelta(minutes=60)
    assert d.emit(_ev()).status == "handed_to_os"


def test_retries_wait_one_then_two_seconds_and_not_after_the_last_attempt(tmp_path):
    from app.core.alerting.dispatcher import AlertDispatcher
    slept = []
    d = AlertDispatcher(event_file=tmp_path / "e.jsonl", notifier=Notifier(always_fail=True), now=Clock(),
                        sleep=slept.append, retry_attempts=3)
    assert d.emit(_ev()).status == "failed"
    assert slept == [1.0, 2.0]


# ── 就绪 ─────────────────────────────────────────────────

def test_an_unreadable_event_file_is_not_ready(tmp_path):
    d = _dispatcher(tmp_path)
    d.path.mkdir(parents=True)                                     # 事件文件的位置是个目录：打不开
    ok, why = d.unattended_readiness(channel_fault=None)
    assert ok is False and "读不了" in why


def test_a_confirmation_exactly_at_its_validity_limit_still_counts(tmp_path):
    clock = Clock()
    d = _confirmed(tmp_path, clock)
    clock.t += timedelta(days=14)
    assert d.unattended_readiness(channel_fault=None)[0] is True
    clock.t += timedelta(seconds=1)
    assert d.unattended_readiness(channel_fault=None)[0] is False


def test_a_failure_at_the_very_moment_of_confirmation_counts_as_after_it(tmp_path):
    clock = Clock()
    d = _confirmed(tmp_path, clock)
    at = clock.t.isoformat()
    d._append({"type": "event", "event_id": "f1", "kind": "x", "severity": "critical", "title": "同一时刻失败",
               "detail": "", "dedup_key": "k1", "extra": {}, "at": at})
    d._append({"type": "status", "event_id": "f1", "status": "failed", "at": at})
    ok, why = d.unattended_readiness(channel_fault=None)
    assert ok is False and "通知失败" in why


def test_an_unknown_outcome_at_the_moment_of_confirmation_counts_as_after_it(tmp_path):
    clock = Clock()
    d = _confirmed(tmp_path, clock)
    d._append({"type": "event", "event_id": "u1", "kind": "x", "severity": "critical", "title": "没有结论",
               "detail": "", "dedup_key": "k2", "extra": {}, "at": clock.t.isoformat()})
    ok, why = d.unattended_readiness(channel_fault=None)
    assert ok is False and "没有送达结论" in why


def test_a_write_fault_at_the_moment_of_confirmation_counts_as_after_it(tmp_path):
    clock = Clock()
    d = _confirmed(tmp_path, clock)
    fault = {"last_at": clock.t.isoformat(), "count": 1, "title": "摄取被拒", "error": "PermissionError"}
    ok, why = d.unattended_readiness(channel_fault=fault)
    assert ok is False and "没能写进事件文件" in why


class _Store:
    def __init__(self, fail=False):
        self.fail, self.state, self.events = fail, {}, []

    def get_state(self, k):
        return self.state.get(k)

    def set_state(self, k, v):
        if self.fail:
            raise OSError("执行库只读")
        self.state[k] = v

    def add_event(self, kind, **kw):
        self.events.append(kind)


def test_recording_a_channel_fault_reports_whether_it_reached_the_store():
    ok_store, bad_store = _Store(), _Store(fail=True)
    assert disp.record_channel_fault(_ev(), OSError("磁盘满"), store=ok_store) is True
    assert "摄取被拒" in ok_store.state[disp.FAULT_KEY]          # Go/No-Go 把这条记录原样给人看
    assert disp.record_channel_fault(_ev(), OSError("磁盘满"), store=bad_store) is False
    assert len(disp._PROCESS_FAULTS) == 2                          # 落不了库的那条留在进程里


# ── 盘中巡检 ─────────────────────────────────────────────

NOW = pd.Timestamp("2025-03-05 14:35", tz="UTC")                   # 美东 09:35


def _wd(alerts=None):
    return iw.IntradayWatchdog(make_order_manager=lambda: None, now=lambda: NOW.to_pydatetime(),
                               alert=(alerts.append if alerts is not None else (lambda ev: None)),
                               stale_after_s=600, health_store=_Store())


def _q(asof):
    return SimpleNamespace(price=1.0, asof=asof)


def test_quote_times_exactly_at_the_tolerances_are_fresh():
    wd = _wd()
    assert wd._name_state(_q("2025-03-05 09:37:00"), NOW)["state"] == "fresh"      # 恰好晚 120 秒
    assert wd._name_state(_q("2025-03-05 09:37:01"), NOW)["state"] == "future"
    assert wd._name_state(_q("2025-03-05 09:25:00"), NOW)["state"] == "fresh"      # 恰好 600 秒前
    assert wd._name_state(_q("2025-03-05 09:24:59"), NOW)["state"] == "stale"


def test_only_stale_holdings_are_listed_as_old():
    om = SimpleNamespace(gw=SimpleNamespace(quotes=lambda tks: {"AAA": _q("2025-03-05 09:34:00"),
                                                                "BBB": _q("2025-03-05 09:00:00")}))
    f = _wd()._freshness(om, {"AAA": 1, "BBB": 1, "CCC": 1}, NOW.to_pydatetime())
    assert f["old_names"] == ["BBB"] and f["names"]["CCC"]["state"] == "missing"


def test_unproven_freshness_alerts_once_when_the_window_is_reached():
    alerts = []
    wd = _wd(alerts)
    f, fp, t0 = {"state": "unproven", "names": {}, "detail": "持仓行情都没有时间戳"}, (1.0, ()), NOW.to_pydatetime()
    seen = []
    for dt in (0, 599, 600, 700, 1300):
        wd._on_freshness(f, fp, t0 + timedelta(seconds=dt))
        seen.append(len(alerts))
    assert seen == [0, 0, 1, 1, 1] and alerts[0].title == "盘中巡检：无法证明快照新鲜"


# ── 命令行 ───────────────────────────────────────────────

@pytest.mark.parametrize("argv", [[], ["confirm", "abc"]])
def test_the_command_line_rejects_missing_arguments(tmp_path, argv):
    from app.tasks.alerts import main
    with pytest.raises(SystemExit) as e:
        main(argv, dispatcher=_dispatcher(tmp_path))
    assert e.value.code == 2


# ── 等价变异 ─────────────────────────────────────────────

PROVEN_EQUIVALENT = {
    "app/core/execution/intraday_watchdog.py ×1 — L98 `self._unproven_alerted = False`（构造时）-> True":
        "这个初值从来不会被读到：唯一的读取点在 _on_freshness，而它第一句就在指纹变化时把它重置为 False；"
        "构造时指纹是 None，券商给出的指纹是元组，第一拍必然『变化』。"
        "见 test_mutated_watchdog_copies_behave_identically（同一串拍子，比较发出的告警）。",
    "app/core/execution/intraday_watchdog.py ×1 — L110 巡检结论落库的 `ensure_ascii=False` -> True":
        "两种写法只差非 ASCII 字符是否转义；读回来的对象相同。这条状态只经 blocking_reasons（json.loads）与 "
        "go_no_go 的 _json 读取，不看原文。见 test_mutated_watchdog_copies_behave_identically。",
}


def _mutated_method(cls, name, old, new):
    """同一变异施加到方法源码的副本上，返回带这个方法的子类（双向：变异工具运行时源码已是变异式，这时换回原式）。"""
    code = textwrap.dedent(inspect.getsource(getattr(cls, name)))
    a, b = (old, new) if old in code else (new, old)
    assert code.count(a) == 1 and b not in code, (old, new)
    ns = dict(vars(iw))
    exec(compile(code.replace(a, b), "<mutant>", "exec"), ns)              # noqa: S102
    return type("Mutant", (cls,), {name: ns[name]})


def test_mutated_watchdog_copies_behave_identically():
    import random
    rnd = random.Random(5)
    init = _mutated_method(iw.IntradayWatchdog, "__init__", "self._unproven_alerted = False",
                           "self._unproven_alerted = True")
    states = ["unproven", "fresh", "partial", "stale", "no_positions"]
    for _ in range(100):
        seq = [(rnd.choice(states), rnd.choice([(1.0, ()), (2.0, ())]), rnd.randint(0, 900)) for _ in range(8)]
        out = []
        for cls in (iw.IntradayWatchdog, init):
            alerts = []
            wd = cls(make_order_manager=lambda: None, now=lambda: NOW.to_pydatetime(), alert=alerts.append,
                     stale_after_s=600, health_store=_Store())
            t = NOW.to_pydatetime()
            for st, fp, dt in seq:
                t += timedelta(seconds=dt)
                wd._on_freshness({"state": st, "names": {"AAA": {"state": "stale", "detail": "旧"}}, "detail": ""}, fp, t)
            out.append([(a.severity, a.title) for a in alerts])
        assert out[0] == out[1]

    rec = _mutated_method(iw.IntradayWatchdog, "_record", "ensure_ascii=False", "ensure_ascii=True")
    for cls in (iw.IntradayWatchdog, rec):
        st = _Store()
        cls(make_order_manager=lambda: None, alert=lambda ev: None, health_store=st)._record(
            None, NOW.to_pydatetime(), "partial", {"AAA": "missing"}, "缺行情")
        assert iw.blocking_reasons(st) == [
            f"盘中巡检最近一拍（{NOW.to_pydatetime().isoformat()}）有持仓的行情缺失 / 时间在未来 / 时区不可判定："
            "{'AAA': 'missing'}"]
    backend = Path(__file__).resolve().parents[2]
    own = (backend / "app" / "core" / "execution" / "intraday_watchdog.py").read_text(encoding="utf-8")
    assert own.count("get_state(STATE_KEY)") == 1 and "get_state(STATE_KEY)" in inspect.getsource(iw.blocking_reasons)
    other = r"""\.get\(['"]intraday_watchdog['"]\)|get_state\(['"]intraday_watchdog['"]\)|get_state\((?:iw|intraday_watchdog)\.STATE_KEY\)"""
    readers = sorted(p.relative_to(backend).as_posix() for d in ("app", "tools") for p in (backend / d).rglob("*.py")
                     if re.search(other, p.read_text(encoding="utf-8")))
    assert readers == ["tools/ops/go_no_go.py"], readers
    gng = (backend / "tools" / "ops" / "go_no_go.py").read_text(encoding="utf-8")
    uses = re.findall(r"""(.{0,6})state\.get\(['"]intraday_watchdog['"]\)(.{0,6})""", gng)
    assert uses and all(pre.endswith("_json(") or post.startswith(" else") for pre, post in uses), uses


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 2
