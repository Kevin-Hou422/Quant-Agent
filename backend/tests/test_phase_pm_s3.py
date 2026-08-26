"""
test_phase_pm_s3.py — Phase PM.S3 经典基准策略库验收

- 库内每个经典策略的 DSL 都能解析 + 执行，产出非空横截面信号
- baseline_signals 可直接喂 PortfolioManager 组一个真实 AUM 账本（"没因子也能交易"）
- seed_baselines 幂等：重复种入不产生重复记录
- 库元数据自洽（名唯一、类别合法、有文献）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.strategies import (
    BASELINE_LIBRARY,
    list_baselines,
    get_baseline,
    baseline_signals,
    seed_baselines,
)


def _dataset(T=300, N=6, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=T)
    cols = [f"T{i}" for i in range(N)]
    close = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0.0003, 0.015, (T, N)), axis=0),
        index=idx, columns=cols,
    )
    vol = pd.DataFrame(rng.uniform(1e5, 5e6, (T, N)), index=idx, columns=cols)
    return {
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "vwap": close, "volume": vol,
        "returns": close.pct_change().fillna(0.0),
    }


# --------------------------------------------------------------------------
# 库内每个策略都可执行
# --------------------------------------------------------------------------

def test_every_baseline_dsl_executes_to_nonempty_signal():
    ds = _dataset()
    sigs = baseline_signals(ds)
    # 全部 8 个都应成功（没有静默丢失）
    assert set(sigs) == set(BASELINE_LIBRARY), f"缺失：{set(BASELINE_LIBRARY) - set(sigs)}"
    for name, df in sigs.items():
        assert isinstance(df, pd.DataFrame)
        assert df.shape[1] == ds["close"].shape[1]
        assert df.notna().any().any(), f"{name} 全 NaN"


def test_subset_selection_by_name():
    ds = _dataset()
    sigs = baseline_signals(ds, names=["xs_momentum_12_1", "low_volatility"])
    assert set(sigs) == {"xs_momentum_12_1", "low_volatility"}


# --------------------------------------------------------------------------
# 直接组一个真实账本：没有自研因子时也能交易
# --------------------------------------------------------------------------

def test_baselines_build_a_real_portfolio_book():
    from app.core.portfolio_manager import PortfolioManager

    ds = _dataset()
    sigs = baseline_signals(ds)
    pm = PortfolioManager(aum=10_000.0, method="equal_weight")
    book = pm.build_book(sigs, prices=ds["close"], volume=ds["volume"])
    # 末日应有一个非空、L1 合理（≤1）的净持仓账本
    last = book.book_on(book.weights.index[-1])
    assert len(last) > 0
    gross = float(book.weights.abs().sum(axis=1).iloc[-1])
    assert 0.0 < gross <= 1.0 + 1e-6
    # 具体美元账本可读
    dollars = sum(abs(v["dollars"]) for v in last.values())
    assert dollars <= 10_000.0 + 1.0


# --------------------------------------------------------------------------
# 种入生命周期 + 幂等
# --------------------------------------------------------------------------

def test_seed_baselines_persists_as_candidate_and_is_idempotent(tmp_path):
    from app.db.alpha_store import AlphaStore

    store = AlphaStore(db_url=f"sqlite:///{tmp_path/'seed.db'}")
    first = seed_baselines(store)
    assert set(first) == set(BASELINE_LIBRARY)          # 全部种入
    # 都是 candidate 起步
    for aid in first.values():
        assert store.get_by_id(aid).status == "candidate"
    # 再种一次：DSL 去重 → 0 条新增（幂等）
    second = seed_baselines(store)
    assert second == {}
    cands = store.query(status="candidate", limit=1000)
    assert len(cands) == len(BASELINE_LIBRARY)


# --------------------------------------------------------------------------
# 元数据自洽
# --------------------------------------------------------------------------

def test_library_metadata_is_sane():
    strategies = list_baselines()
    names = [s.name for s in strategies]
    assert len(names) == len(set(names))                # 名唯一
    valid_cats = {"momentum", "reversal", "volatility", "liquidity", "skew", "trend"}
    for s in strategies:
        assert s.category in valid_cats
        assert s.dsl and s.reference and s.description
    assert get_baseline("xs_momentum_12_1").category == "momentum"


def test_category_filter():
    mom = list_baselines(category="momentum")
    assert {s.name for s in mom} == {"xs_momentum_12_1", "xs_momentum_12m", "high_52w"}


# --------------------------------------------------------------------------
# 交易环节兜底：门控下无自研因子时，run_portfolio 回退经典基准照常交易
# --------------------------------------------------------------------------

def test_run_portfolio_falls_back_to_baselines_when_no_active_factors(tmp_path):
    from app.db.alpha_store import AlphaStore
    from app.db.position_store import PositionStore
    from app.core.execution.paper_broker import PaperBroker
    from app.tasks.daily_trading_loop import DailyTradingLoop

    store = AlphaStore(db_url=f"sqlite:///{tmp_path/'a.db'}")           # 空库：无任何 PAPER/ACTIVE 因子
    broker = PaperBroker(store=PositionStore(db_url=f"sqlite:///{tmp_path/'p.db'}"),
                         initial_capital=10_000.0)
    loop = DailyTradingLoop(store=store, broker=broker)

    out = loop.run_portfolio(_dataset(), aum=10_000.0)
    assert out["used_baseline"] is True             # 回退到基准
    assert out["n_factors"] > 0                      # 有成分在交易
    assert out["days_processed"] > 0                 # 真的走了交易日
    # 一旦有自研 PAPER 因子，就不再回退（用自研）
    from app.db.alpha_store import AlphaResult
    aid = store.save(AlphaResult(dsl="rank(ts_delta(close,5))", status="candidate"))
    store.update_status(aid, "validated"); store.update_status(aid, "paper")
    out2 = loop.run_portfolio(_dataset(), aum=10_000.0)
    assert out2["used_baseline"] is False
