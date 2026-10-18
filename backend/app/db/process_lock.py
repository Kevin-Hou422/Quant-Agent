"""
process_lock.py — 跨进程互斥锁（操作系统级文件锁）

为什么需要
----------
进程内的 `threading.RLock` 只管同一个 Python 进程。调度器、`python -m app.tasks.forward run-now`、
API 里手动触发的执行、恢复任务是**不同进程**，它们同时动同一个券商账户 / 同一个 PIT 分区时，
进程锁什么都挡不住（外部审查 2026-10-18 F08：两个独立数据库会话同时认领了同一条订单意图）。

做法
----
锁文件 + 操作系统的排他锁（Windows `msvcrt.locking`，POSIX `fcntl.flock`）。持锁进程被杀时
操作系统自动释放，不会留下"死锁文件"。锁文件本身不删（删了会让两个进程各锁一个新文件）。

    with process_lock(path, timeout_s=30):
        ...                                  # 拿不到锁 → 抛 ProcessLockTimeout，调用方据此不做事

同一进程内重入：同一线程对同一路径重复加锁会直接通过（引用计数），其它线程照常等待。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Tuple, Union  # noqa: F401  (Tuple 用于 describe 的返回注解)

logger = logging.getLogger(__name__)

_LOCAL = threading.local()
_GUARD = threading.Lock()
#: 路径 → 进程内的线程锁（挡住同一进程的其它线程；跨进程靠操作系统文件锁）
_THREAD_LOCKS: Dict[str, threading.Lock] = {}


class ProcessLockTimeout(TimeoutError):
    """在给定时间内没拿到锁 —— 另一个进程正在做同一件事。"""


def _thread_lock(key: str) -> threading.Lock:
    with _GUARD:
        lk = _THREAD_LOCKS.get(key)
        if lk is None:
            lk = _THREAD_LOCKS[key] = threading.Lock()
        return lk


def _held() -> Dict[str, int]:
    if not hasattr(_LOCAL, "held"):
        _LOCAL.held = {}
    return _LOCAL.held


def _try_os_lock(fd: int) -> bool:
    if os.name == "nt":
        import msvcrt
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError as exc:            # 被别的进程占着（这是"没拿到"，不是故障）
            logger.debug("[process_lock] 锁被占用: %s", exc)
            return False
    import fcntl
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError as exc:                # 被别的进程占着（这是"没拿到"，不是故障）
        logger.debug("[process_lock] 锁被占用: %s", exc)
        return False


def _os_unlock(fd: int) -> None:
    if os.name == "nt":
        import msvcrt
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def process_lock(path: Union[str, Path], timeout_s: float = 30.0,
                 poll_s: float = 0.05) -> Iterator[Path]:
    """排他持有 `path` 对应的锁；`timeout_s` 内拿不到 → ProcessLockTimeout。"""
    p = Path(path).resolve()
    key = str(p).lower() if os.name == "nt" else str(p)
    held = _held()
    if held.get(key):
        held[key] += 1
        try:
            yield p
        finally:
            held[key] -= 1
        return

    deadline = time.monotonic() + max(float(timeout_s), 0.0)
    tlock = _thread_lock(key)
    if not tlock.acquire(timeout=max(float(timeout_s), 0.0)):
        raise ProcessLockTimeout(f"{timeout_s:.0f}s 内没拿到锁 {p}（本进程的另一个线程正在持有）")
    fd = -1
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(p), os.O_RDWR | os.O_CREAT, 0o644)
        if os.fstat(fd).st_size == 0:
            os.write(fd, b"\0")
        while not _try_os_lock(fd):
            if time.monotonic() >= deadline:
                raise ProcessLockTimeout(f"{timeout_s:.0f}s 内没拿到锁 {p}（另一个进程正在持有）")
            time.sleep(poll_s)
        held[key] = 1
        try:
            yield p
        finally:
            held.pop(key, None)
            _os_unlock(fd)
    finally:
        if fd >= 0:
            os.close(fd)
        tlock.release()


def lock_path_for(kind: str, name: str, base: Union[str, Path]) -> Path:
    """统一的锁文件位置：<base>/.locks/<kind>-<name>.lock（name 里的路径分隔符换成 _）。"""
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))
    return Path(base) / ".locks" / f"{kind}-{safe}.lock"


def describe(path: Union[str, Path]) -> Tuple[bool, str]:
    """(当前是否有人持锁, 说明)。只用于状态展示 —— 结果在返回那一刻就可能过期。"""
    try:
        with process_lock(path, timeout_s=0.0):
            return False, "空闲"
    except ProcessLockTimeout as exc:
        logger.debug("[process_lock] %s", exc)
        return True, "另一个进程正在持有"
