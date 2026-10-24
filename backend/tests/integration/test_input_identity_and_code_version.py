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

    def __init__(self, head="0123456789ab", dirty=" M app/x.py"):
        self.head, self.dirty = head, dirty

    def __call__(self, cmd, cwd=None, capture_output=False, text=False, timeout=None):
        out = self.head if "rev-parse" in cmd else self.dirty
        if not capture_output:
            out = None
        elif not text:
            out = out.encode()
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="" if text else b"")


def test_code_version_is_the_short_head_with_a_dirty_suffix(monkeypatch):
    monkeypatch.setattr(ii.subprocess, "run", _Git())
    ii.code_version.cache_clear()
    try:
        v = ii.code_version()
    finally:
        ii.code_version.cache_clear()
    assert v == "0123456789ab+dirty" and isinstance(v, str)


def test_a_clean_tree_has_no_suffix(monkeypatch):
    monkeypatch.setattr(ii.subprocess, "run", _Git(dirty=""))
    ii.code_version.cache_clear()
    try:
        assert ii.code_version() == "0123456789ab"
    finally:
        ii.code_version.cache_clear()


class _GitBytesStatus(_Git):
    """只有 `git status` 那次按字节返回（L74 第二个 text=True 的变异）。"""

    def __call__(self, cmd, cwd=None, capture_output=False, text=False, timeout=None):
        if "status" in cmd:
            text = False
        return super().__call__(cmd, cwd=cwd, capture_output=capture_output, text=text, timeout=timeout)


def test_the_dirty_flag_only_depends_on_whether_there_is_output(monkeypatch):
    """L74 等价的机械验证：`dirty` 只拿来判真假，str 与 bytes 去掉空白后真假相同 —— 结果逐字相同。"""
    for dirty in ("", " M app/x.py", "\n"):
        for runner in (_Git(dirty=dirty), _GitBytesStatus(dirty=dirty)):
            monkeypatch.setattr(ii.subprocess, "run", runner)
            ii.code_version.cache_clear()
            try:
                got = ii.code_version()
            finally:
                ii.code_version.cache_clear()
            assert got == ("0123456789ab+dirty" if dirty.strip() else "0123456789ab"), (dirty, runner)


PROVEN_EQUIVALENT = {
    "app/core/data_engine/input_identity.py ×1 — L74 `git status` 的 `text=True` -> `text=False`":
        "`dirty` 只在 `\"+dirty\" if dirty else \"\"` 里判真假；str 与 bytes 经 .strip() 之后是否为空完全一致，"
        "拼进结果的是常量 \"+dirty\" 而不是 dirty 本身。见 test_the_dirty_flag_only_depends_on_whether_there_is_output。",
}


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 1
