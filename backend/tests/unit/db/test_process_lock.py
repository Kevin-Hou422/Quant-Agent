"""
跨进程互斥锁（外部审查 2026-10-18 F08 / F14 的基础件）。

进程内 RLock 管不到另一个进程：调度器、run-now 命令行、API 手动触发是不同进程。
这里用真的子进程验证：一个进程持锁时另一个进程拿不到；持锁进程退出（包括被杀）后锁自动释放；
两个进程同时往同一个 PIT 分区追加，一行都不丢。
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from app.db.process_lock import ProcessLockTimeout, describe, lock_path_for, process_lock

BACKEND = Path(__file__).resolve().parents[3]


def _hold_in_subprocess(lock: Path, ready: Path, release: Path) -> subprocess.Popen:
    code = textwrap.dedent(f"""
        import sys, time
        from pathlib import Path
        sys.path.insert(0, {str(BACKEND)!r})
        from app.db.process_lock import process_lock
        with process_lock({str(lock)!r}, timeout_s=10):
            Path({str(ready)!r}).write_text("x")
            for _ in range(400):
                if Path({str(release)!r}).exists():
                    break
                time.sleep(0.05)
    """)
    return subprocess.Popen([sys.executable, "-c", code])


def _wait_for(path: Path, timeout=20.0):
    t0 = time.monotonic()
    while not path.exists():
        assert time.monotonic() - t0 < timeout, f"等不到 {path}"
        time.sleep(0.05)


def test_another_process_cannot_take_a_held_lock(tmp_path):
    lock, ready, release = tmp_path / "a.lock", tmp_path / "ready", tmp_path / "release"
    proc = _hold_in_subprocess(lock, ready, release)
    try:
        _wait_for(ready)
        with pytest.raises(ProcessLockTimeout, match="另一个进程"):
            with process_lock(lock, timeout_s=0.3):
                pass
        assert describe(lock) == (True, "另一个进程正在持有")
    finally:
        release.write_text("x")
        proc.wait(timeout=20)
    with process_lock(lock, timeout_s=5):                     # 对方退出后拿得到
        pass
    assert describe(lock) == (False, "空闲")


def test_a_killed_holder_does_not_leave_the_lock_stuck(tmp_path):
    lock, ready, release = tmp_path / "b.lock", tmp_path / "ready", tmp_path / "release"
    proc = _hold_in_subprocess(lock, ready, release)
    _wait_for(ready)
    proc.kill()
    proc.wait(timeout=20)
    with process_lock(lock, timeout_s=5):
        pass


def test_reentrant_in_the_same_thread_exclusive_across_threads(tmp_path):
    lock = tmp_path / "c.lock"
    got = []
    with process_lock(lock, timeout_s=1):
        with process_lock(lock, timeout_s=1):                 # 同线程重入直接通过
            got.append("inner")

        def other():
            try:
                with process_lock(lock, timeout_s=0.2):
                    got.append("other")
            except ProcessLockTimeout:
                got.append("blocked")

        t = threading.Thread(target=other)
        t.start()
        t.join()
    assert got == ["inner", "blocked"]
    with process_lock(lock, timeout_s=1):                     # 全部退出后又可拿
        got.append("again")
    assert got[-1] == "again"


def test_lock_paths_are_namespaced_and_safe(tmp_path):
    p = lock_path_for("account", "moomoo/123:SIM", tmp_path)
    assert p.parent == tmp_path / ".locks" and p.name == "account-moomoo_123_SIM.lock"


def test_concurrent_pit_appends_from_two_processes_lose_nothing(tmp_path):
    """
    PIT 的追加是"读年度分区 → 合并 → 原子替换"。没有跨进程锁时，两个进程交错执行，
    后写的会把先写的那批覆盖掉（原子替换只保证文件不残缺）。
    """
    code = textwrap.dedent("""
        import sys
        sys.path.insert(0, %r)
        import pandas as pd
        from app.core.data_engine.pit_store import PITStore
        tk, n = sys.argv[1], int(sys.argv[2])
        st = PITStore(%r)
        for i in range(n):
            day = pd.Timestamp("2025-01-02") + pd.Timedelta(days=i)
            st.append({"close": pd.DataFrame({tk: [float(i)]}, index=[day])},
                      as_of=f"2025-02-01T00:00:{i %% 60:02d}Z", name="race")
    """) % (str(BACKEND), str(tmp_path / "pit"))
    n = 25
    procs = [subprocess.Popen([sys.executable, "-c", code, tk, str(n)]) for tk in ("AAA", "BBB")]
    for p in procs:
        assert p.wait(timeout=180) == 0
    from app.core.data_engine.pit_store import PITStore
    long = PITStore(tmp_path / "pit")._load_long("race")
    assert len(long) == 2 * n
    assert sorted(long["ticker"].unique()) == ["AAA", "BBB"]
