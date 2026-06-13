"""
conftest.py — 全局测试 fixtures

提供：
  - make_dataset(n_days, n_tickers, seed) → Dict[str, pd.DataFrame]
  - test_client → FastAPI TestClient
  - tmp_alpha_store → 内存 SQLite AlphaStore（每个测试独立）
  - tmp_chat_store  → 内存 SQLite ChatStore（每个测试独立）
  - basic_dsl       → 通用测试 DSL 字符串
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def _make_dataset(
    n_days:    int = 80,
    n_tickers: int = 10,
    seed:      int = 42,
) -> dict:
    """合成市场数据，用于单元 + 集成测试。"""
    rng     = np.random.default_rng(seed)
    dates   = pd.bdate_range("2022-01-03", periods=n_days)
    tickers = [f"T{i:03d}" for i in range(n_tickers)]

    close = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.01, (n_days, n_tickers)), axis=0),
        index=dates, columns=tickers,
    )
    volume = pd.DataFrame(
        rng.integers(500_000, 5_000_000, (n_days, n_tickers)).astype(float),
        index=dates, columns=tickers,
    )
    high  = close * (1 + rng.uniform(0, 0.01, (n_days, n_tickers)))
    low   = close * (1 - rng.uniform(0, 0.01, (n_days, n_tickers)))
    open_ = close * (1 + rng.normal(0, 0.005, (n_days, n_tickers)))
    vwap  = (high + low + close) / 3
    returns = close.pct_change().fillna(0.0)

    return {
        "close":   close,
        "open":    open_,
        "high":    high,
        "low":     low,
        "volume":  volume,
        "vwap":    vwap,
        "returns": returns,
    }


@pytest.fixture
def make_dataset():
    return _make_dataset


@pytest.fixture
def dataset():
    return _make_dataset()


@pytest.fixture
def basic_dsl() -> str:
    return "rank(ts_delta(log(close), 5))"


# ---------------------------------------------------------------------------
# FastAPI TestClient
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_client():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from app.main import app
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


# ---------------------------------------------------------------------------
# Isolated in-memory stores
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_alpha_store(tmp_path):
    from app.db.alpha_store import AlphaStore
    db_path = tmp_path / "alpha_test.db"
    return AlphaStore(db_url=f"sqlite:///{db_path}")


@pytest.fixture
def tmp_chat_store(tmp_path):
    from app.db.chat_store import ChatStore
    db_path = tmp_path / "chat_test.db"
    return ChatStore(db_url=f"sqlite:///{db_path}")
