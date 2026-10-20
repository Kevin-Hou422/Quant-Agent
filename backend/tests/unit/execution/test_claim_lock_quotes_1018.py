"""
外部审查 2026-10-18 F08 / F09。

F08（R11）两个独立数据库会话同时认领同一条被拒的意图，两边都返回 True —— 随后两个进程都可能调券商。
     进程内 RLock 挡不住另一个进程（调度器 / run-now / API / 恢复任务）。
F09（R5）新开仓的票不在持仓里、没有券商价，下单门跳过价格偏差检查，拿自己的参考价给自己估值 ——
     空账户第一笔 BUY 没有任何独立报价就放行了。
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pandas as pd
import pytest

from app.core.execution.broker_gateway import BrokerError, Quote
from app.db.execution_store import ST_GATE, ExecutionStore
from tests.unit.execution.test_order_manager import D1, LIVE_BOOK_ID, PX1, _cycle, _env, _placed

BACKEND = Path(__file__).resolve().parents[3]


def _order(es_tmp):
    from app.core.execution.order_builder import PlannedOrder
    return PlannedOrder(client_id="qaR-1-20250303-AAA-B", ticker="AAA", side="BUY", qty=10,
                        ref_price=50.0, limit_price=50.25, current_qty=0.0, target_qty=10.0,
                        target_weight=0.1)


# ---------------------------------------------------------------------------
# F08：原子认领
# ---------------------------------------------------------------------------

class TestAtomicClaim:

    def test_the_review_r11_race_has_exactly_one_winner(self, tmp_path):
        from sqlalchemy import event
        es = ExecutionStore(f"sqlite:///{(tmp_path / 'race.db').as_posix()}")
        o = _order(tmp_path)
        es.record_gate_rejection(o, LIVE_BOOK_ID, D1, "retryable")
        barrier = Barrier(2)

        def both_start_together(conn, cursor, statement, *a):
            if statement.lstrip().upper().startswith("UPDATE") and "exec_orders" in statement:
                barrier.wait(timeout=5)     # 两个认领在同一时刻发出（等不齐 → 线程抛错 → 用例失败）
        event.listen(es._engine, "before_cursor_execute", both_start_together)
        with ThreadPoolExecutor(max_workers=2) as pool:
            claims = [f.result() for f in [pool.submit(es.add_intent, o, LIVE_BOOK_ID, D1) for _ in range(2)]]
        event.remove(es._engine, "before_cursor_execute", both_start_together)
        assert sorted(claims) == [False, True]

    def test_a_sent_order_cannot_be_claimed_again(self, tmp_path):
        es = ExecutionStore(f"sqlite:///{(tmp_path / 'e.db').as_posix()}")
        o = _order(tmp_path)
        assert es.add_intent(o, LIVE_BOOK_ID, D1) is True
        es.mark_submitted(o.client_id, "90001")
        assert es.add_intent(o, LIVE_BOOK_ID, D1) is False
        assert es.get(o.client_id).broker_order_id == "90001"

    def test_a_gate_rejected_intent_is_retryable_once(self, tmp_path):
        es = ExecutionStore(f"sqlite:///{(tmp_path / 'e.db').as_posix()}")
        o = _order(tmp_path)
        es.record_gate_rejection(o, LIVE_BOOK_ID, D1, "no_independent_quote")
        assert es.add_intent(o, LIVE_BOOK_ID, D1) is True
        assert es.add_intent(o, LIVE_BOOK_ID, D1) is False          # 已是 PENDING，不可再认领
        assert es.get(o.client_id).reject_reason == ""


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
        from app.config import settings
        from pathlib import Path
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
# F09：独立报价
# ---------------------------------------------------------------------------

class TestIndependentQuotes:

    def test_the_review_r5_first_buy_without_a_quote_is_not_placed(self, tmp_path):
        e = _env(tmp_path)
        rep = _cycle(e, D1, quote=False)
        assert _placed(e) == [] and rep.n_submitted == 0
        assert {r["reason"] for r in rep.gate["rejected"]} == {"no_independent_quote"}
        assert {r.status for r in e.es.orders()} == {ST_GATE}       # 可重试：报价回来后照常下

    def test_a_quote_far_from_the_reference_price_blocks_the_order(self, tmp_path):
        e = _env(tmp_path)
        e.ctx.quotes.update({"AAA": 50.0 * 1.3})
        e.ctx.quotes.update({"BBB": 20.0, "CCC": 10.0})
        e.clock.after_close(D1)
        rep = e.om.run_cycle(pd.Series({"AAA": .1, "BBB": .08, "CCC": .05}), pd.Series(PX1),
                             pd.Series({k: 1e9 for k in PX1}), D1)
        rej = {r["ticker"]: r["reason"] for r in rep.gate["rejected"]}
        assert rej == {"AAA": "quote_mismatch"}
        assert sorted(p[0] for p in _placed(e)) == ["US.BBB", "US.CCC"]

    def test_a_stale_quote_counts_as_none(self, tmp_path):
        e = _env(tmp_path)
        e.ctx.quotes.update(PX1)
        e.ctx.quote_time = "2025-02-28 16:00:00.000"                  # 上周五；决策日 2025-03-03 周一已收盘
        rep = _cycle(e, D1, quote=False)
        assert {r["reason"] for r in rep.gate["rejected"]} == {"stale_quote"}

    def test_a_quote_outage_blocks_new_positions_but_reports_why(self, tmp_path):
        e = _env(tmp_path)
        e.ctx.quote_fail = "OpenD quote down"
        rep = _cycle(e, D1)
        assert "OpenD quote down" in rep.quote_error and _placed(e) == []
        assert {r["reason"] for r in rep.gate["rejected"]} == {"no_independent_quote"}

    def test_reducing_an_existing_position_needs_no_quote(self, tmp_path):
        e = _env(tmp_path)
        e.ctx.set_position("AAA", 400, 50.0)
        e.ctx.quotes.clear()
        e.ctx.quote_fail = "down"
        rep = _cycle(e, D1, w={"AAA": 0.10}, px={"AAA": 50.0}, quote=False)   # 400 股 → 200 股
        assert [p[:3] for p in _placed(e)] == [("US.AAA", "SELL", 200)], rep.gate

    def test_flatten_is_never_blocked_by_quotes(self, tmp_path):
        e = _env(tmp_path)
        e.ctx.set_position("AAA", 100, 50.0)
        e.ctx.quote_fail = "down"
        e.clock.after_close(D1)
        e.om.reconcile()
        out = e.om.engage_kill_switch("kevin", "test")
        assert out["flatten"] and [p[:3] for p in _placed(e)] == [("US.AAA", "SELL", 100)]


def test_gateway_quotes_parse_the_snapshot_and_skip_unpriced_rows():
    from tests.unit.execution.moomoo_fake import FakeTradeContext, make_gateway
    rows = pd.DataFrame([
        {"code": "US.AAA", "update_time": "2025-03-03 16:00:00", "last_price": 50.5, "bid_price": 1, "ask_price": 1},
        {"code": "US.BBB", "update_time": "2025-03-03 16:00:00", "last_price": 0.0, "bid_price": 19.9, "ask_price": 20.1},
        {"code": "US.CCC", "update_time": "N/A", "last_price": "N/A", "bid_price": "N/A", "ask_price": "N/A"},
    ])
    gw = make_gateway(FakeTradeContext(), quote_fn=lambda tks: rows)
    q = gw.quotes(["AAA", "BBB", "CCC"])
    assert q["AAA"] == Quote(50.5, pd.Timestamp("2025-03-03 16:00:00").to_pydatetime())
    assert q["BBB"].price == pytest.approx(20.0)
    assert "CCC" not in q
    assert gw.quotes([]) == {}


def test_a_gateway_without_a_quote_source_refuses():
    from app.core.execution.broker_gateway import BrokerGateway

    class Bare(BrokerGateway):
        account = positions = place = orders = cancel = describe = lambda *a, **k: None
    with pytest.raises(BrokerError, match="不提供独立报价"):
        Bare().quotes(["AAA"])
