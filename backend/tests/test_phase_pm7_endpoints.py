"""
test_phase_pm7_endpoints.py — Phase PM.7 策略配置端点 + active 配置交易接线

- /strategies/pending、/approve(activate)、/reject、/{id}(+谱系) 端点闭环
- run_portfolio 在有 active 配置时**只交易该配置的成分**
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _save_cfg(status="proposed", factors=("1", "2")):
    from app.dependencies import get_strategy_store
    from app.db.strategy_store import StrategyConfig
    s = get_strategy_store()
    return s, s.save(StrategyConfig(factors=list(factors), combo_weights={f: 0.5 for f in factors},
                                    aum=10_000, passed=True, status=status, name="ep-test"))


def test_strategy_endpoints_lifecycle(test_client):
    s, sid = _save_cfg()
    # pending 含它
    r = test_client.get("/api/strategies/pending")
    assert r.status_code == 200 and any(x["id"] == sid for x in r.json())
    # approve + activate
    r = test_client.post(f"/api/strategies/{sid}/approve", json={"activate": True, "reason": "ok"})
    assert r.status_code == 200 and r.json()["status"] == "active"
    # 详情含谱系
    r = test_client.get(f"/api/strategies/{sid}")
    assert r.status_code == 200 and len(r.json()["decisions"]) >= 2   # approve + activate
    # 已 active 不能再 approve
    assert test_client.post(f"/api/strategies/{sid}/approve").status_code == 409


def test_strategy_reject_endpoint(test_client):
    s, sid = _save_cfg()
    r = test_client.post(f"/api/strategies/{sid}/reject", json={"reason": "no"})
    assert r.status_code == 200 and r.json()["status"] == "rejected"


def test_run_portfolio_trades_active_config(tmp_path):
    from app.db.alpha_store import AlphaStore, AlphaResult
    from app.db.position_store import PositionStore
    from app.db.strategy_store import StrategyStore, StrategyConfig
    from app.core.execution.paper_broker import PaperBroker
    from app.tasks.daily_trading_loop import DailyTradingLoop

    db = f"sqlite:///{tmp_path/'a.db'}"
    store = AlphaStore(db_url=db)
    ids = []
    for dsl in ["rank(ts_delta(close,5))", "rank((-ts_std(returns,20)))", "rank(ts_delta(log(close),60))"]:
        aid = store.save(AlphaResult(dsl=dsl, status="candidate"))
        store.update_status(aid, "validated"); store.update_status(aid, "paper")
        ids.append(str(aid))
    # active 配置只含前 1 个因子
    sstore = StrategyStore(db_url=db)
    sid = sstore.save(StrategyConfig(factors=[ids[0]], combo_weights={ids[0]: 1.0},
                                     aum=10_000, passed=True, status="proposed"))
    sstore.update_status(sid, "approved"); sstore.update_status(sid, "active")

    broker = PaperBroker(store=PositionStore(db_url=f"sqlite:///{tmp_path/'p.db'}"), initial_capital=10_000.0)
    loop = DailyTradingLoop(store=store, broker=broker)
    # 让 run_portfolio 的 StrategyStore() 指向同一库
    import app.config as cfgmod
    old = cfgmod.settings.database_url
    cfgmod.settings.database_url = db
    try:
        rng = np.random.default_rng(0); idx = pd.bdate_range("2022-01-03", periods=160); cols = [f"S{i}" for i in range(10)]
        close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0003, 0.015, (160, 10)), 0), idx, cols)
        ds = {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
              "vwap": close, "volume": pd.DataFrame(1e6, idx, cols), "returns": close.pct_change().fillna(0.0)}
        out = loop.run_portfolio(ds, aum=10_000.0)
    finally:
        cfgmod.settings.database_url = old
    assert out["active_config"] == sid        # 按 active 配置交易
    assert out["n_factors"] == 1              # 只交易配置里的 1 个因子（不是全部 3 个）


# ---------------------------------------------------------------------------
# B-12：/api/strategies/propose 此前**零测试覆盖** —— 它是进审批队列的源头。
# 该端点会加载真实数据集（需网络），故此处 monkeypatch 数据加载为合成面板。
# ---------------------------------------------------------------------------

def _panel(n_days=180, n_tickers=8, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n_days)
    cols = [f"T{i:02d}" for i in range(n_tickers)]
    close = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.012, (n_days, n_tickers)), axis=0),
        index=idx, columns=cols)
    # DEV_LESSONS §O：H/L 用随机幅度，固定 ±1% 会让价差估计荒谬
    high = close * (1 + rng.uniform(0, 0.006, close.shape))
    low  = close * (1 - rng.uniform(0, 0.006, close.shape))
    vol  = pd.DataFrame(rng.integers(8e5, 5e6, close.shape).astype(float), index=idx, columns=cols)
    return {"close": close, "open": close, "high": high, "low": low, "volume": vol,
            "vwap": (high + low + close) / 3.0,
            "returns": close.pct_change().fillna(0.0)}


class _FakeDS:
    def __init__(self, data): self.data = data


def _patch_dataset(monkeypatch):
    import app.core.data_engine.dataset_registry as reg
    monkeypatch.setattr(reg, "load_registry_dataset",
                        lambda *a, **k: _FakeDS(_panel()))


def test_propose_without_paper_factors_returns_400(test_client, monkeypatch):
    """无 PAPER/ACTIVE 因子时必须 400，而不是产出一份空策略配置。"""
    _patch_dataset(monkeypatch)
    from app.dependencies import get_store
    store = get_store()
    for rec in store.query(limit=500):
        if str(rec.status) in ("paper", "active", "decaying"):
            store.update_status(rec.id, "retired")
    r = test_client.post("/api/strategies/propose")
    assert r.status_code == 400, f"实际 {r.status_code}：{r.text[:300]}"
    assert "因子" in r.json()["detail"]


def test_propose_creates_proposed_config_with_evidence(test_client, monkeypatch):
    """
    有 PAPER 因子时：产出 proposed 配置，且必须带**证据字段** ——
    verdict / risk_report / turnover_ann。审批者靠这些判断该不该批。
    """
    _patch_dataset(monkeypatch)
    from app.dependencies import get_store
    from app.db.alpha_store import AlphaResult
    store = get_store()
    aid = store.save(AlphaResult(dsl="rank(ts_delta(log(close), 5))",
                                 hypothesis="propose-test", sharpe=0.0, status="candidate"))
    for nxt in ("validated", "paper"):
        store.update_status(aid, nxt)

    r = test_client.post("/api/strategies/propose")
    assert r.status_code == 200, f"实际 {r.status_code}：{r.text[:400]}"
    cfg = r.json()
    assert cfg["status"] == "proposed", cfg["status"]
    assert str(aid) in cfg["factors"], f"提案未包含该 PAPER 因子：{cfg['factors']}"
    for k in ("verdict", "risk_report", "turnover_ann", "no_trade_band", "aum"):
        assert k in cfg, f"提案缺少证据字段 {k}：{sorted(cfg)}"
    # 审计 #9 整改：门评估若中途降级，必须在 verdict 里留痕而不是静默产出
    v = cfg["verdict"]
    assert isinstance(v, dict) and v, "verdict 为空 —— 策略门没有产生任何证据"
    # 条件断言：degraded 缺席时什么都不检查 → 该字段整个消失也能通过。
    # 改为无条件契约：要么没有降级（键缺席/为空），要么是**非空字符串列表**。
    deg = v.get("degraded")
    assert deg is None or (isinstance(deg, list) and all(isinstance(x, str) for x in deg)), (
        f"degraded 字段格式不合契约：{deg!r}"
    )

    # 提案必须真的进了审批队列
    pend = test_client.get("/api/strategies/pending").json()
    assert any(x["id"] == cfg["id"] for x in pend), "提案未出现在 pending 队列"
