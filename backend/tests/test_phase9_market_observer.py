"""
test_phase9_market_observer.py — Phase 9.1 市场观察引擎验收

覆盖：
  - 强趋势市 → 动量/趋势家族排前
  - 均值回复市（负滞后自相关）→ 反转家族排前
  - 输出结构完整、家族均为有效 GP 家族、确定性可复现
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.discovery.market_observer import MarketObserver, MarketObservation

_VALID_FAMILIES = {"momentum", "trend_following", "reversion",
                    "volatility", "liquidity", "price_volume_corr"}


def _dataset_from_returns(ret: np.ndarray, T: int, N: int) -> dict:
    idx = pd.bdate_range("2021-01-01", periods=T)
    cols = [f"A{i:02d}" for i in range(N)]
    r = pd.DataFrame(ret, index=idx, columns=cols)
    close = 100 * np.exp(np.cumsum(r.values, axis=0))
    close = pd.DataFrame(close, index=idx, columns=cols)
    vol = pd.DataFrame(1e6, index=idx, columns=cols)
    return {"open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": vol, "vwap": close, "returns": r}


def _trending(T=320, N=15, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    ret = 0.002 + rng.normal(0, 0.004, (T, N))     # 全资产正漂移 → 上升趋势
    return _dataset_from_returns(ret, T, N)


def _mean_reverting(T=320, N=15, seed=1) -> dict:
    rng = np.random.default_rng(seed)
    ret = np.zeros((T, N))
    eps = rng.normal(0, 0.01, (T, N))
    for t in range(1, T):
        ret[t] = -0.5 * ret[t - 1] + eps[t]         # 负滞后-1 自相关
    return _dataset_from_returns(ret, T, N)


def test_trending_market_ranks_momentum_high():
    obs = MarketObserver().observe(_trending())
    assert isinstance(obs, MarketObservation)
    assert obs.regime in ("bull", "bear", "high_vol", "sideways")
    top2 = obs.top_families(2)
    assert ("momentum" in top2) or ("trend_following" in top2), obs.to_dict()


def test_mean_reverting_market_ranks_reversion_high():
    obs = MarketObserver().observe(_mean_reverting())
    assert obs.short_term_reversal < 0        # 负自相关被检出
    assert "reversion" in obs.top_families(2), obs.to_dict()


def test_output_structure_and_valid_families():
    obs = MarketObserver().observe(_trending())
    assert len(obs.hypotheses) == 6
    # 分数降序
    scores = [h.score for h in obs.hypotheses]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == 1.0                    # 归一化后最高为 1
    for h in obs.hypotheses:
        assert h.family in _VALID_FAMILIES
        assert 0.0 <= h.score <= 1.0
        assert h.rationale
    d = obs.to_dict()
    assert set(d) >= {"regime", "dispersion", "momentum_breadth",
                      "short_term_reversal", "vol_level", "hypotheses"}


def test_deterministic():
    a = MarketObserver().observe(_trending())
    b = MarketObserver().observe(_trending())
    assert a.top_families(6) == b.top_families(6)
    assert a.to_dict() == b.to_dict()
