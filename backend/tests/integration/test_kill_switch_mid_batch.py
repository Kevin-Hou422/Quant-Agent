"""
一批订单发送途中开熔断：余下的调仓单不再发出。

来源（原测试文件的说明，保留出处）：

复审 2026-10-19（`review-v6/audit-review/复审报告_2026-10-19.md`）N01–N09 的回归。

每条都从复审独立脚本（`independent_checks.py`）里对应的 residual_* 场景改写：原脚本断言"问题仍在"，
这里断言修复后应有的行为。
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# N01：一批订单发送途中开熔断，余下的调仓单不再发出
# ---------------------------------------------------------------------------

class TestKillSwitchMidBatch:

    def test_orders_after_the_switch_are_not_sent(self, tmp_path):
        from tests.unit.execution.test_order_manager import D1, _cycle, _env, _placed
        e = _env(tmp_path)
        orig = e.gw.place

        def place_then_switch(req):
            oid = orig(req)
            if len(_placed(e)) == 1:
                e.es.set_kill_switch(True, "review", "switch during order batch")
            return oid
        e.gw.place = place_then_switch
        rep = _cycle(e, D1)
        assert rep.n_submitted == 1, "熔断之后又发出了调仓单"
        assert rep.blocked == "kill_switch_engaged_mid_batch"
        assert [o["result"] for o in rep.orders].count("skipped_kill_switch") == 2
        assert rep.flatten is not None                                # 转去撤单 + 全平

    def test_a_switch_set_while_waiting_for_cancels_stops_the_build(self, tmp_path):
        """撤单等终态期间被开熔断：建单之前那一次读要拦住。"""
        from tests.unit.execution.test_order_manager import (
            D1,
            D2,
            _cycle,
            _env,
            _placed,
        )
        e = _env(tmp_path)
        _cycle(e, D1)                                                 # D1 的单留在途 → D2 要先撤
        n_before = len(_placed(e))
        orig = e.om._await_terminal

        def await_then_switch(ids):
            e.es.set_kill_switch(True, "review", "switch while cancels settle")
            return orig(ids)
        e.om._await_terminal = await_then_switch
        rep = _cycle(e, D2)
        assert rep.blocked == "kill_switch_engaged"
        assert [p for p in _placed(e)[n_before:] if p[1] == "BUY"] == []
