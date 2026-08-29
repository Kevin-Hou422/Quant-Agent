"""
test_phase_tr2_moomoo.py — Phase TR.2 MoomooProvider 验收

离线（不需网关）
  - assemble_panel 把单标的 K 线拼成与 Yahoo 一致结构的 RawDataset（字段/索引/列/vwap/returns 推导）
  - ticker 映射 AAPL↔US.AAPL；缺失标的 → 全 NaN 列
实盘冒烟（OpenD 未起则 skip）
  - 真连 OpenD 拉一根 AAPL 日K，结构/数值自洽
"""

from __future__ import annotations

import socket

import numpy as np
import pandas as pd
import pytest

from app.core.data_engine.providers.moomoo_provider import (
    MoomooProvider,
    assemble_panel,
    _to_moomoo_code,
    _from_moomoo_code,
)


def _fake_kline(ticker, dates, base=100.0):
    n = len(dates)
    close = base + np.arange(n, dtype=float)
    return pd.DataFrame({
        "code": [f"US.{ticker}"] * n,
        "time_key": [d.strftime("%Y-%m-%d 00:00:00") for d in dates],
        "open": close - 0.5, "high": close + 1.0, "low": close - 1.0,
        "close": close, "volume": np.full(n, 1_000_000.0),
    })


# --------------------------------------------------------------------------
# 映射
# --------------------------------------------------------------------------

def test_ticker_mapping():
    assert _to_moomoo_code("AAPL") == "US.AAPL"
    assert _to_moomoo_code("aapl") == "US.AAPL"
    assert _to_moomoo_code("HK.700") == "HK.700"          # 已有前缀不动
    assert _from_moomoo_code("US.AAPL") == "AAPL"


# --------------------------------------------------------------------------
# 离线：assemble_panel 结构与 Yahoo 对齐
# --------------------------------------------------------------------------

def test_assemble_panel_structure_and_derived_fields():
    dates = pd.bdate_range("2024-01-02", periods=10)
    kline = {"AAPL": _fake_kline("AAPL", dates, 100.0),
             "MSFT": _fake_kline("MSFT", dates, 300.0)}
    fields = ["open", "high", "low", "close", "volume", "vwap", "returns"]
    ds = assemble_panel(kline, ["AAPL", "MSFT"], fields)

    assert set(ds) == set(fields)
    for f in fields:
        assert isinstance(ds[f].index, pd.DatetimeIndex)
        assert list(ds[f].columns) == ["AAPL", "MSFT"]
        assert len(ds[f]) == len(dates)
    # vwap = (h+l+c)/3
    exp_vwap = (ds["high"] + ds["low"] + ds["close"]) / 3.0
    pd.testing.assert_frame_equal(ds["vwap"], exp_vwap)
    # returns = log(c/c.shift)
    exp_ret = np.log(ds["close"] / ds["close"].shift(1))
    pd.testing.assert_frame_equal(ds["returns"], exp_ret)
    # 首行 return 为 NaN
    assert ds["returns"].iloc[0].isna().all()


def test_assemble_panel_missing_ticker_is_all_nan():
    dates = pd.bdate_range("2024-01-02", periods=5)
    kline = {"AAPL": _fake_kline("AAPL", dates)}
    ds = assemble_panel(kline, ["AAPL", "GHOST"], ["close"])
    assert list(ds["close"].columns) == ["AAPL", "GHOST"]
    assert ds["close"]["GHOST"].isna().all()
    assert ds["close"]["AAPL"].notna().all()


def test_available_fields_matches_yahoo():
    from app.core.data_engine.yahoo_provider import YahooFinanceProvider
    assert MoomooProvider().available_fields() == YahooFinanceProvider().available_fields()


def test_missing_sdk_raises_clear_error(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def _blocked(name, *a, **k):
        if name == "moomoo":
            raise ImportError("no moomoo")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    with pytest.raises(RuntimeError, match="moomoo-api"):
        MoomooProvider()._sdk()


# --------------------------------------------------------------------------
# 实盘冒烟：OpenD 未监听则 skip
# --------------------------------------------------------------------------

def _opend_up(host="127.0.0.1", port=11111) -> bool:
    s = socket.socket()
    s.settimeout(1.5)
    try:
        s.connect((host, port))
        return True
    except Exception:
        return False
    finally:
        s.close()


@pytest.mark.skipif(not _opend_up(), reason="OpenD 网关未运行（127.0.0.1:11111）")
def test_live_fetch_real_aapl():
    prov = MoomooProvider()
    ds = prov.fetch(["AAPL"], start="2024-06-03", end="2024-06-10")
    assert "close" in ds and "AAPL" in ds["close"].columns
    closes = ds["close"]["AAPL"].dropna()
    assert len(closes) >= 3
    assert (closes > 0).all()                      # 真实价格为正
    assert ds["returns"]["AAPL"].dropna().abs().max() < 0.5   # 日收益合理
