"""
账户锁：跨进程互斥、按券商账户建锁域、执行库绑定账户身份。

来源（原测试文件的说明，保留出处）：

外部审查 2026-10-18 F08 / F09。

F08（R11）两个独立数据库会话同时认领同一条被拒的意图，两边都返回 True —— 随后两个进程都可能调券商。
     进程内 RLock 挡不住另一个进程（调度器 / run-now / API / 恢复任务）。
F09（R5）新开仓的票不在持仓里、没有券商价，下单门跳过价格偏差检查，拿自己的参考价给自己估值 ——
     空账户第一笔 BUY 没有任何独立报价就放行了。

---

复审 2026-10-19（`review-v6/audit-review/复审报告_2026-10-19.md`）N01–N09 的回归。

每条都从复审独立脚本（`independent_checks.py`）里对应的 residual_* 场景改写：原脚本断言"问题仍在"，
这里断言修复后应有的行为。
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from app.core.execution.broker_gateway import BrokerError
from tests.unit.execution.test_order_manager import (
    D1,
    _cycle,
    _env,
    _placed,
)

BACKEND = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# F08：账户级跨进程互斥
# ---------------------------------------------------------------------------

def _hold(lock: Path, ready: Path, release: Path):
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


class TestAccountLock:

    @pytest.fixture
    def busy(self, tmp_path):
        e = _env(tmp_path)
        e.om.ACCOUNT_LOCK_TIMEOUT_S = 0.3
        ready, release = tmp_path / "ready", tmp_path / "release"
        proc = _hold(e.om._account_lock_path(), ready, release)
        t0 = time.monotonic()
        while not ready.exists():
            assert time.monotonic() - t0 < 20
            time.sleep(0.05)
        yield e
        release.write_text("x")
        proc.wait(timeout=20)

    def test_the_lock_is_named_after_the_broker_account(self, tmp_path):
        """复审 2026-10-19 §5.3：此前锁在执行库同目录 —— 不同目录的两套部署不互斥。"""
        from pathlib import Path

        from app.config import settings
        e = _env(tmp_path)
        assert e.om._account_lock_path() == (
            Path(settings.account_lock_dir) / ".locks" / "account-moomoo-FUTUINC-SIMULATE-2001.lock")

    def test_another_process_holding_the_account_blocks_the_cycle(self, busy):
        with pytest.raises(BrokerError, match="另一个进程正在操作这个账户"):
            _cycle(busy, D1)
        assert _placed(busy) == []

    def test_reconcile_and_maintenance_are_serialised_too(self, busy):
        with pytest.raises(BrokerError):
            busy.om.reconcile()
        with pytest.raises(BrokerError):
            busy.om.maintain("x")

    def test_the_kill_switch_is_persisted_even_while_the_account_is_busy(self, busy):
        out = busy.om.engage_kill_switch("kevin", "test")
        assert out["state"]["engaged"] is True and "另一个进程" in out["error"]
        assert busy.es.kill_switch()["engaged"] is True              # 下一个周期会续做全平

    def test_once_released_the_cycle_runs(self, tmp_path):
        e = _env(tmp_path)
        rep = _cycle(e, D1)
        assert rep.blocked == "" and rep.n_submitted == 3


# ---------------------------------------------------------------------------
# §5.3：账户锁按券商账户建锁域；校验账户身份
# ---------------------------------------------------------------------------

class TestAccountLockDomain:

    def _om(self, tmp_path, sub, acc_id=2001):
        from app.core.execution.order_manager import OrderManager
        from app.db.execution_store import ExecutionStore
        from app.db.position_store import PositionStore
        from tests.unit.execution.moomoo_fake import (
            FakeTradeContext,
            default_accounts,
            make_gateway,
        )
        from tests.unit.execution.test_order_manager import LIM, Clock
        accts = [dict(a, acc_id=acc_id) if a["trd_env"] == "SIMULATE" else a for a in default_accounts()]
        ctx = FakeTradeContext(accounts=accts)
        d = tmp_path / sub
        d.mkdir(parents=True, exist_ok=True)
        return OrderManager(make_gateway(ctx), ExecutionStore(f"sqlite:///{(d / 'e.db').as_posix()}"),
                            PositionStore(f"sqlite:///{(d / 'p.db').as_posix()}"), limits=LIM,
                            paper_aum=100_000.0, limit_band_bps=50.0, flatten_band_bps=500.0, clock=Clock())

    def test_two_deployments_in_different_directories_share_one_lock_for_one_account(self, tmp_path):
        a, b = self._om(tmp_path, "deploy_a"), self._om(tmp_path, "deploy_b")
        assert a._account_lock_path() == b._account_lock_path()
        assert "2001" in a._account_lock_path().name

    def test_different_accounts_have_different_locks(self, tmp_path):
        assert self._om(tmp_path, "a", 2001)._account_lock_path() != self._om(tmp_path, "b", 2002)._account_lock_path()

    def test_the_lock_dir_is_machine_wide_not_next_to_the_db(self, tmp_path):
        from app.config import settings
        p = self._om(tmp_path, "x")._account_lock_path()
        assert p.parent == __import__("pathlib").Path(settings.account_lock_dir) / ".locks"

    def test_a_different_account_behind_the_same_store_is_refused(self, tmp_path):
        """moomoo_trd_acc_id=0 会自动取第一个模拟账户 —— 换了账户不能悄悄接着交易。"""
        from app.core.execution.broker_gateway import BrokerError
        first = self._om(tmp_path, "same", 2001)
        first.reconcile()                                               # 绑定 2001
        second = self._om(tmp_path, "same", 2002)                        # 同一个执行库，另一个账户
        with pytest.raises(BrokerError, match="账户身份变了"):
            second.reconcile()
        assert second.store.get_state("bound_account").endswith("2001")
