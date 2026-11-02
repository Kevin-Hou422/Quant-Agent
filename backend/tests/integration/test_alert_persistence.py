"""
告警事件写不进事件文件时的健康状态（复审固定提交 21cd90b，"放行边界 3"）。

复审原话："当前 emit_safely 在事件文件写入异常时可能只写日志并返回 None，而 readiness 依据事件文件中的确认和失败历史；如果
失败本身没有写进去，许可可能看不到它。"要求："用不可写目录、文件损坏或可模拟的存储故障构造反例；事件无法持久化、通知结果
未知或通道失败时，应有可靠的健康状态使无人值守新增风险受阻；不以一条日志代替可供许可判断的状态；修复后测试故障发生、重启、
恢复和人工重新确认的流程。"

修复前（HEAD 21cd90b，反例 C:\\QuantAgentEnv\\review-repro\\repro_at_21cd90b.json）：确认可见之后把事件文件改成只读，
一条 critical 告警写不进去，emit_safely 返回 None；readiness 仍是 (True, "此后没有通知失败")。
"""

from __future__ import annotations

import json
import os
import stat

import pytest

from app.core.alerting import dispatcher as disp
from tests.integration.test_alerting import Clock, Notifier, _dispatcher, _ev


@pytest.fixture(autouse=True)
def _own_fault_state(fresh_db, monkeypatch):
    """故障记在本用例自己的库（fresh_db）；进程内的最后一层也每条用例重新开始。"""
    monkeypatch.setattr(disp, "_PROCESS_FAULTS", [])
    return fresh_db


def _ready_world(tmp_path, clock=None):
    clock = clock or Clock()
    d = _dispatcher(tmp_path, clock=clock)
    d.confirm_visible(d.send_test_notification(), actor="kevin")
    assert d.unattended_readiness()[0]
    return d, clock


@pytest.fixture
def read_only():
    made = []

    def make(path):
        os.chmod(path, stat.S_IREAD)
        made.append(path)
    yield make
    for p in made:
        os.chmod(p, stat.S_IREAD | stat.S_IWRITE)


def _later(clock, minutes=5):
    from datetime import timedelta
    clock.t += timedelta(minutes=minutes)


class TestFaultIsAStateNotALogLine:

    def test_an_unwritable_event_file_blocks_unattended(self, tmp_path, read_only):
        d, clock = _ready_world(tmp_path)
        _later(clock)
        read_only(d.path)
        assert disp.emit_safely(_ev(), d) is None                       # 调用方照常继续
        ok, why = d.unattended_readiness()
        assert not ok and "没能写进事件文件" in why and "PermissionError" in why
        fault = disp.read_channel_fault()
        assert fault["title"] == "摄取被拒" and fault["count"] == 1

    def test_an_event_directory_that_cannot_be_created_blocks_unattended(self, tmp_path):
        d, clock = _ready_world(tmp_path)
        _later(clock)
        blocker = tmp_path / "not_a_dir"
        blocker.write_text("这是文件，不是目录", encoding="utf-8")
        bad = disp.AlertDispatcher(event_file=blocker / "events.jsonl", notifier=Notifier(), now=clock)
        assert disp.emit_safely(_ev(), bad) is None
        assert not d.unattended_readiness()[0]                          # 同一个执行库里的故障，任何读者都看得到

    def test_repeated_faults_are_counted(self, tmp_path, read_only):
        d, clock = _ready_world(tmp_path)
        _later(clock)
        read_only(d.path)
        for k in range(3):
            disp.emit_safely(_ev(key=f"k{k}"), d)
        assert disp.read_channel_fault()["count"] == 3
        from app.db.execution_store import ExecutionStore
        assert [e.kind for e in ExecutionStore().events(10)].count("alert_channel_fault") == 3

    def test_when_the_execution_db_also_fails_the_process_still_remembers(self, tmp_path, read_only, monkeypatch):
        d, clock = _ready_world(tmp_path)
        _later(clock)
        read_only(d.path)

        def broken(*a, **k):
            raise OSError("磁盘满")
        monkeypatch.setattr(disp, "read_channel_fault", lambda store=None: None)
        from app.db import execution_store
        monkeypatch.setattr(execution_store.ExecutionStore, "set_state", broken)
        assert disp.emit_safely(_ev(), d) is None
        assert disp._PROCESS_FAULTS and not d.unattended_readiness(channel_fault=None)[0]   # noqa: SLF001


class TestUnknownOrUnreadableHistory:

    def test_a_corrupt_line_in_the_event_file_blocks(self, tmp_path):
        d, clock = _ready_world(tmp_path)
        with open(d.path, "a", encoding="utf-8") as f:
            f.write('{"type": "status", "event_id": "x", "status": "fai\n')
        ok, why = d.unattended_readiness()
        assert not ok and "读不出来" in why

    def test_an_event_without_a_delivery_outcome_is_unknown_not_delivered(self, tmp_path):
        """进程在通知途中退出：事件行在，送达结论行没有 → 通知结果未知。"""
        d, clock = _ready_world(tmp_path)
        _later(clock)
        d._append({"type": "event", "event_id": "e1", "kind": "ingest_rejected", "severity": "critical",   # noqa: SLF001
                   "title": "摄取被拒", "detail": "", "dedup_key": "k", "extra": {}, "at": clock().isoformat()})
        ok, why = d.unattended_readiness()
        assert not ok and "没有送达结论" in why

    def test_an_info_event_without_an_outcome_does_not_count(self, tmp_path):
        d, clock = _ready_world(tmp_path)
        _later(clock)
        d._append({"type": "event", "event_id": "e2", "kind": "ingest_rejected", "severity": "info",       # noqa: SLF001
                   "title": "只进文件", "detail": "", "dedup_key": "k", "extra": {}, "at": clock().isoformat()})
        assert d.unattended_readiness()[0]

    def test_an_unreadable_fault_record_blocks(self, tmp_path, monkeypatch):
        d, _clock = _ready_world(tmp_path)

        def broken(store=None):
            raise OSError("库打不开")
        monkeypatch.setattr(disp, "read_channel_fault", broken)
        ok, why = d.unattended_readiness()
        assert not ok and "读不到告警通道故障记录" in why


class TestRestartRecoveryReconfirmation:

    def test_the_fault_survives_a_restart_and_only_a_new_confirmation_clears_it(self, tmp_path, read_only):
        d, clock = _ready_world(tmp_path)
        _later(clock)
        read_only(d.path)
        disp.emit_safely(_ev(), d)
        disp._PROCESS_FAULTS.clear()                                      # noqa: SLF001  "重启"：进程内的记忆没了
        restarted = _dispatcher(tmp_path, clock=clock)
        assert not restarted.unattended_readiness()[0]                    # 执行库里的记录还在
        os.chmod(d.path, stat.S_IREAD | stat.S_IWRITE)                    # 修好通道
        _later(clock)
        assert restarted.emit(_ev(key="after-fix")).status == "handed_to_os"
        assert not restarted.unattended_readiness()[0], "通道恢复不会自动清除：要人重新确认"
        _later(clock)
        restarted.confirm_visible(restarted.send_test_notification(), actor="kevin")
        ok, why = restarted.unattended_readiness()
        assert ok, why

    def test_the_operator_status_shows_the_fault(self, tmp_path, read_only, capsys):
        from app.tasks import alerts as cli
        d, clock = _ready_world(tmp_path)
        _later(clock)
        read_only(d.path)
        disp.emit_safely(_ev(), d)
        assert cli.main(["status"], dispatcher=d) == 1
        out = capsys.readouterr().out
        assert "未就绪" in out and "告警通道故障记录" in out


def test_the_unattended_permit_health_sees_the_fault(tmp_path, read_only, monkeypatch):
    """无人值守许可的真实健康检查（不注入）读到故障 → 不新增风险。"""
    from app.config import settings
    from app.core.execution import run_permit as rp
    d, clock = _ready_world(tmp_path)
    _later(clock)
    read_only(d.path)
    disp.emit_safely(_ev(), d)
    s = settings.model_copy(update={"alert_event_file": str(d.path), "backup_dir": str(tmp_path / "bk")})
    (tmp_path / "bk").mkdir()
    (tmp_path / "bk" / "last_backup.json").write_text(json.dumps({"ok": True, "at": clock().isoformat()}))
    problems = rp._unattended_health(s, clock())                                    # noqa: SLF001
    assert problems and all("通知未就绪" in p for p in problems) and "没能写进事件文件" in problems[0]
