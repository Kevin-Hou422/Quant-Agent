"""
订单的原子认领：同一笔订单不会被两个进程各发一次。

来源（原测试文件的说明，保留出处）：

外部审查 2026-10-18 F08 / F09。

F08（R11）两个独立数据库会话同时认领同一条被拒的意图，两边都返回 True —— 随后两个进程都可能调券商。
     进程内 RLock 挡不住另一个进程（调度器 / run-now / API / 恢复任务）。
F09（R5）新开仓的票不在持仓里、没有券商价，下单门跳过价格偏差检查，拿自己的参考价给自己估值 ——
     空账户第一笔 BUY 没有任何独立报价就放行了。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from app.db.execution_store import ExecutionStore
from tests.unit.execution.test_order_manager import (
    D1,
    LIVE_BOOK_ID,
)


def _order(es_tmp):
    from app.core.execution.order_builder import PlannedOrder
    return PlannedOrder(client_id="qaR-1-20250303-AAA-B", ticker="AAA", side="BUY", qty=10,
                        ref_price=50.0, limit_price=50.25, current_qty=0.0, target_qty=10.0,
                        target_weight=0.1)


# ---------------------------------------------------------------------------
# F08：原子认领
# ---------------------------------------------------------------------------

class TestAtomicClaim:

    def test_a_claim_race_has_exactly_one_winner(self, tmp_path):
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
