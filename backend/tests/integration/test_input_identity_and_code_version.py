"""
输入身份与代码版本（D5 关键路径变异复核：app/core/data_engine/input_identity.py）。

复核时 5 个变异点全部存活：
    · 数据摘要不含日期索引（`hash_pandas_object(index=False)`）也没人发现 —— 同样的数挪到别的日期，身份不变
    · 代码版本的 git 调用改成拿字节 / 不捕获输出、拼接 +dirty 的加号改成减号，都没人发现
      （测试环境往往不是 git 仓库，代码版本恒为 "unknown"，这几行从来没被真正执行过）
"""

from __future__ import annotations

import subprocess

import numpy as np
import pandas as pd
import pytest

from app.core.data_engine import input_identity as ii


def _panel(start="2025-01-02"):
    idx = pd.bdate_range(start, periods=5)
    close = pd.DataFrame(np.arange(10.0).reshape(5, 2) + 10, index=idx, columns=["A", "B"])
    return {"close": close}


def test_the_same_numbers_on_other_dates_are_a_different_input():
    a, b = ii.identify(_panel("2025-01-02")), ii.identify(_panel("2025-02-03"))
    assert a.data_digest != b.data_digest


def test_the_same_panel_has_the_same_identity():
    assert ii.identify(_panel()).data_digest == ii.identify(_panel()).data_digest


class _Git:
    """按真实 subprocess.run 的契约应答：text=True 才给 str，capture_output=False 时 stdout 是 None。"""

    def __init__(self, head="0123456789ab", dirty=" M app/x.py", ignored=""):
        self.head, self.dirty, self.ignored = head, dirty, ignored

    def __call__(self, cmd, cwd=None, capture_output=False, text=False, timeout=None, check=False):
        out = self.head if "rev-parse" in cmd else (self.ignored if "ls-files" in cmd else self.dirty)
        if not capture_output:
            out = None
        elif not text:
            out = out.encode()
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="" if text else b"")


def test_code_version_is_the_short_head_with_a_dirty_suffix(monkeypatch):
    monkeypatch.setattr(ii.subprocess, "run", _Git())
    v = ii.code_version()
    assert v == "0123456789ab+dirty" and isinstance(v, str)


def test_a_clean_tree_has_no_suffix(monkeypatch):
    monkeypatch.setattr(ii.subprocess, "run", _Git(dirty=""))
    assert ii.code_version() == "0123456789ab"


class _GitBytesStatus(_Git):
    """只有 `git status` 那次按字节返回（L74 第二个 text=True 的变异）。"""

    def __call__(self, cmd, cwd=None, capture_output=False, text=False, timeout=None, check=False):
        if "status" in cmd:
            text = False
        return super().__call__(cmd, cwd=cwd, capture_output=capture_output, text=text, timeout=timeout, check=check)


class _GitBytesIgnored(_Git):
    def __call__(self, cmd, cwd=None, capture_output=False, text=False, timeout=None, check=False):
        if "ls-files" in cmd:
            text = False
        return super().__call__(cmd, cwd=cwd, capture_output=capture_output, text=text, timeout=timeout, check=check)


def test_ignored_output_only_controls_the_dirty_flag(monkeypatch):
    for ignored in ("", "app/ignored.py\0"):
        for runner in (_Git(dirty="", ignored=ignored), _GitBytesIgnored(dirty="", ignored=ignored)):
            monkeypatch.setattr(ii.subprocess, "run", runner)
            assert ii.code_version() == ("0123456789ab+dirty" if ignored else "0123456789ab")


def test_the_dirty_flag_only_depends_on_whether_there_is_output(monkeypatch):
    """L74 等价的机械验证：`dirty` 只拿来判真假，str 与 bytes 去掉空白后真假相同 —— 结果逐字相同。"""
    for dirty in ("", " M app/x.py", "\n"):
        for runner in (_Git(dirty=dirty), _GitBytesStatus(dirty=dirty)):
            monkeypatch.setattr(ii.subprocess, "run", runner)
            got = ii.code_version()
            assert got == ("0123456789ab+dirty" if dirty.strip() else "0123456789ab"), (dirty, runner)


def test_version_rechecks_real_git_changes_without_restarting(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    app = root / "backend" / "app"
    app.mkdir(parents=True)
    tracked = app / "strategy.py"
    tracked.write_text("value = 1\n", encoding="utf-8")
    (root / ".gitignore").write_text("backend/app/ignored*\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "initial"], cwd=root, check=True, capture_output=True)
    monkeypatch.setattr(ii, "__file__", str(app / "core" / "data_engine" / "input_identity.py"))

    clean = ii.code_version()
    assert clean not in ("", "unknown") and not clean.endswith("+dirty")
    tracked.write_text("value = 2\n", encoding="utf-8")
    assert ii.code_version() == clean + "+dirty"
    tracked.write_text("value = 1\n", encoding="utf-8")
    assert ii.code_version() == clean
    new_file = app / "untracked_strategy.py"
    new_file.write_text("value = 3\n", encoding="utf-8")
    assert ii.code_version() == clean + "+dirty"
    new_file.unlink()
    assert ii.code_version() == clean
    ignored_file = app / "ignored_strategy.py"
    ignored_file.write_text("value = 4\n", encoding="utf-8")
    assert ii.code_version() == clean + "+dirty"
    ignored_file.unlink()
    assert ii.code_version() == clean


@pytest.mark.parametrize("failed_command", ["rev-parse", "status", "ls-files"])
def test_failed_git_command_is_unknown_not_clean(monkeypatch, failed_command):
    def git(cmd, **kwargs):
        if failed_command in cmd:
            if kwargs.get("check"):
                raise subprocess.CalledProcessError(1, cmd)
            output = "0123456789ab" if failed_command == "rev-parse" else ""
            return subprocess.CompletedProcess(cmd, 1, stdout=output, stderr="failed")
        return subprocess.CompletedProcess(cmd, 0, stdout="0123456789ab", stderr="")

    monkeypatch.setattr(ii.subprocess, "run", git)
    assert ii.code_version() == "unknown"


def test_a_checkout_during_identity_check_is_unknown(monkeypatch):
    class SwitchingGit(_Git):
        def __init__(self):
            super().__init__(dirty="")
            self.head_reads = 0

        def __call__(self, cmd, **kwargs):
            if "rev-parse" in cmd:
                self.head_reads += 1
                if self.head_reads == 2:
                    self.head = "fedcba987654"
            return super().__call__(cmd, **kwargs)

    monkeypatch.setattr(ii.subprocess, "run", SwitchingGit())
    assert ii.code_version() == "unknown"


def test_failure_to_recheck_head_is_unknown(monkeypatch):
    class FailingSecondHead(_Git):
        def __init__(self):
            super().__init__(dirty="")
            self.head_reads = 0

        def __call__(self, cmd, **kwargs):
            if "rev-parse" in cmd:
                self.head_reads += 1
                if self.head_reads == 2:
                    if kwargs.get("check"):
                        raise subprocess.CalledProcessError(1, cmd)
                    return subprocess.CompletedProcess(cmd, 1, stdout=self.head, stderr="failed")
            return super().__call__(cmd, **kwargs)

    monkeypatch.setattr(ii.subprocess, "run", FailingSecondHead())
    assert ii.code_version() == "unknown"


def test_a_live_permit_detects_changes_without_a_process_restart(tmp_path, monkeypatch):
    from app.core.execution import run_permit
    from tests.integration.test_run_permit import ACC, _guarded, _supervised

    git = _Git(dirty="")
    monkeypatch.setattr(ii.subprocess, "run", git)
    env = _guarded(tmp_path)
    _supervised(env)
    assert run_permit.evaluate(env.es, account=ACC, settings=env.settings, now=env.clock()).allowed
    git.dirty = "?? backend/app/new_strategy.py"
    decision = run_permit.evaluate(env.es, account=ACC, settings=env.settings, now=env.clock())
    assert not decision.allowed
    assert any("代码版本变了" in reason for reason in decision.reasons)


PROVEN_EQUIVALENT = {
    "app/core/data_engine/input_identity.py ×1 — L74 `git status` 的 `text=True` -> `text=False`":
        "`dirty` 只在 `\"+dirty\" if dirty else \"\"` 里判真假；str 与 bytes 经 .strip() 之后是否为空完全一致，"
        "拼进结果的是常量 \"+dirty\" 而不是 dirty 本身。见 test_the_dirty_flag_only_depends_on_whether_there_is_output。",
    "app/core/data_engine/input_identity.py ×1 — 忽略源码列表的 text=True -> False":
        "忽略源码列表只用于判断有无输出，文本和字节输出的空与非空状态相同；见 test_ignored_output_only_controls_the_dirty_flag。",
}


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 2
