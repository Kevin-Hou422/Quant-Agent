"""
未知公司行动：识别、隔离标记（人结案）、模拟账本与券商都不加风险、对账标注拆股比例、结案命令。

来源（原测试文件的说明，保留出处）：

2026-10-22 用户决定 · 证据 / 研究纪律。

D12 原话："同意先只读查询，不授权写边界。使用明确的活库路径和真正只读连接，不调用会建表或迁移的 Store 构造函数。
列出已有边界、最早曝光日期、次数、股票池和实验身份，给出日期建议后由我确认。正式入口不要继续依赖首次调用偶然冻结；
改实验名不能刷新重叠数据的曝光额度。"
"""

from __future__ import annotations

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# 用户决定第 18 条："价量策略同样依赖拆股、分红、复权及券商股数 / 现金正确。
# 未知公司行动必须隔离、停止新增风险并对账，不能以'没基本面因子'豁免。"
# ---------------------------------------------------------------------------

def _ca_panel():
    idx = pd.bdate_range("2025-03-03", periods=6)
    close = pd.DataFrame({"A": [100.0, 101, 102, 103, 104, 52.5],      # 末日约 1/2：疑似未复权 2:1 拆股
                          "B": [50.0, 50.5, 51, 51.5, 52, 53],         # 正常
                          "C": [20.0, 20, 20, 20, 20, 8.0],            # −60%：极端波动，不像任何拆股比例
                          "D": [10.0, 10, 10, 10, 10, 8.0]},           # −20%：不算
                         index=idx)
    vol = pd.DataFrame(1e6, index=idx, columns=close.columns)
    vol.iloc[-1, 0] = 2e6
    return close, vol


class TestUnknownCorporateActions:

    def test_split_like_and_extreme_moves_are_flagged_on_recent_bars(self):
        from app.core.data_engine.corporate_actions import detect_suspects
        close, vol = _ca_panel()
        got = {s["ticker"]: s for s in detect_suspects(close, vol)}
        assert set(got) == {"A", "C"}
        assert got["A"]["nearest_split"] == "2:1" and got["A"]["bar_date"] == str(close.index[-1].date())
        assert got["C"]["nearest_split"] is None

    def test_flags_persist_until_a_person_resolves_them(self, fresh_db):
        from app.db.corporate_actions import CorporateActionStore
        st = CorporateActionStore()
        assert st.flag("A", "2025-03-10", kind="suspected", ratio=0.505, nearest_split="1:2") is True
        assert st.flag("A", "2025-03-10", kind="suspected", ratio=0.505, nearest_split="1:2") is False   # 去重
        assert st.open_tickers() == {"A": "2025-03-10"}
        with pytest.raises(ValueError):
            st.resolve("A", actor="", resolution="x")
        st.resolve("A", actor="Kevin-Hou422", resolution="券商股数 ×2、现金未变；数据源已复权")
        assert st.open_tickers() == {}
        assert st.history("A")[0]["resolved_by"] == "Kevin-Hou422"

    def test_quarantined_names_cannot_gain_risk_in_the_paper_targets(self):
        from app.core.data_engine.corporate_actions import no_new_risk
        idx = pd.bdate_range("2025-03-03", periods=4)
        w = pd.DataFrame({"A": [0.10, 0.10, 0.20, -0.05], "B": [0.1, 0.2, 0.3, 0.4]}, index=idx)
        out = no_new_risk(w, {"A": str(idx[2].date())})
        assert out["A"].tolist() == [0.10, 0.10, 0.10, 0.0], "隔离之后加了仓 / 反手了"
        pd.testing.assert_series_equal(out["B"], w["B"])

    def test_the_broker_gate_rejects_risk_increasing_orders_on_quarantined_names(self):
        from app.core.execution.order_builder import PlannedOrder
        from app.core.execution.pretrade_gate import GateContext, PreTradeLimits, check
        mk = lambda tk, side, q: PlannedOrder(client_id=f"c-{tk}-{side}", ticker=tk, side=side, qty=q,   # noqa: E731
                                              ref_price=100.0, limit_price=100.5, current_qty=10.0,
                                              target_qty=10.0 + (q if side == "BUY" else -q), target_weight=0.01)
        ctx = GateContext(equity=1e5, buying_power=1e5, current_qty={"A": 10.0}, broker_prices={"A": 100.0},
                          adv_usd={"A": 1e9}, quotes={}, quarantined=frozenset({"A"}))
        dec = check([mk("A", "BUY", 5), mk("A", "SELL", 4)], ctx, PreTradeLimits(.1, 1., .1, .05, .15))
        rej = {o.side: r for o, r in dec.rejected}
        assert rej.get("BUY") == "corporate_action_quarantine"
        assert [o.side for o in dec.approved] == ["SELL"], "隔离不应挡住减仓"

    def test_a_broker_share_jump_is_annotated_for_reconciliation(self):
        from app.core.data_engine.corporate_actions import annotate_discrepancies
        out = annotate_discrepancies([{"ticker": "A", "expected": 10.0, "broker": 20.0},
                                      {"ticker": "B", "expected": 10.0, "broker": 9.0}])
        assert out[0]["suspected_corporate_action"] == "2:1"
        assert "suspected_corporate_action" not in out[1]


def test_the_daily_loop_quarantines_a_suspected_split_and_adds_no_risk(tmp_path, monkeypatch):
    """端到端：末日某只票价格翻倍（疑似合股）→ 记隔离标记、结果里列出、券商目标不超过前一行、隔离名单送到下单层。"""
    from types import SimpleNamespace

    from app.db.corporate_actions import CorporateActionStore
    from tests.unit.portfolio_manager.test_audit_portfolio_regressions import _loop

    def ok_gate(self, *a, **k):
        return SimpleNamespace(passed=True, sharpe=1.0, deflated_sharpe=.99, t_stat=3.0,
                               reasons=[], to_dict=lambda: {"passed": True})
    loop, ds, _ = _loop(tmp_path, monkeypatch, gate_block=False, evaluate=ok_gate)
    tk = ds["close"].columns[0]
    for f in ("open", "high", "low", "close", "vwap"):
        if f in ds:
            ds[f] = ds[f].copy()
            ds[f].iloc[-1, 0] = ds[f].iloc[-2, 0] * 2.0          # 疑似 1:2 合股（价格翻倍）
    clipped = []
    import app.core.data_engine.corporate_actions as ca
    real = ca.no_new_risk

    def spy(w, q):
        # 收缩结果的最后一行强制为 0（合规：平仓不加风险）—— 下面看券商目标是否真的用了收缩后的这份
        res = real(w, q)
        if tk in q:
            res.loc[res.index[-1], tk] = 0.0
        clipped.append((w, res))
        return res
    monkeypatch.setattr(ca, "no_new_risk", spy)

    out = loop.run_portfolio(ds)
    assert tk in out["ca_quarantined"] and out["ca_flags"][tk] == str(ds["close"].index[-1].date())
    assert tk in CorporateActionStore().open_tickers()
    # 券商目标（execution_target）= 收缩后的最后一行；它不比标记前一行更大、也不反手
    w_in, w_out = clipped[-1]
    assert abs(float(w_in[tk].iloc[-1])) > 1e-9, "前提：不收缩时这只票有目标"
    assert float(out["execution_target"].get(tk, 0.0)) == pytest.approx(float(w_out[tk].iloc[-1])) == 0.0, (
        "券商目标没有用收缩后的权重")


def test_the_quarantine_reaches_the_order_gate(monkeypatch):
    """_execute_live 把隔离名单交给 OrderManager.run_cycle（下单层按实际持仓拒绝加风险单）。"""
    from types import SimpleNamespace

    import app.core.execution.order_manager as om
    from app.tasks.daily_trading_loop import DailyTradingLoop
    seen = {}

    class FakeOM:
        def run_cycle(self, **k):
            seen.update(k)
            return SimpleNamespace(to_dict=lambda: {"ok": True})
    monkeypatch.setattr(om.OrderManager, "from_settings", classmethod(lambda cls, *a, **k: FakeOM()))
    loop = DailyTradingLoop(store=SimpleNamespace(), broker=SimpleNamespace(store=None))
    monkeypatch.setattr(loop, "_open_live_gateway", lambda: (SimpleNamespace(close=lambda: None), None))
    idx = pd.bdate_range("2025-03-03", periods=2)
    px = pd.DataFrame({"A": [1.0, 1.0]}, index=idx)
    loop._execute_live(pd.Series({"A": 0.1}), idx[-1], px, px, quarantined={"A"})
    assert seen["quarantined"] == {"A"}


def test_the_quarantine_cli_lists_read_only_and_resolves_with_a_reason(tmp_path):
    from app.db.corporate_actions import CorporateActionStore
    from tools.ops.ca_flags import list_flags, main
    db = tmp_path / "x.db"
    st = CorporateActionStore(db_url=f"sqlite:///{db.as_posix()}")
    st.flag("A", "2025-03-10", kind="suspected_price_jump", ratio=0.5, nearest_split="2:1")
    st._engine.dispose()
    assert [r["status"] for r in list_flags(db)] == ["open"]
    with pytest.raises(SystemExit):
        main(["--db", str(db), "resolve", "A", "--actor", "Kevin-Hou422"])            # 缺结论
    assert main(["--db", str(db), "resolve", "A", "--actor", "Kevin-Hou422", "--resolution", "股数 ×2，已复权"]) == 0
    assert [r["status"] for r in list_flags(db)] == ["resolved"]
