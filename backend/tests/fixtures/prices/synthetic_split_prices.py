"""
synthetic_split_prices.py — 拆股 + 除息的**合成**行情夹具（2026-10-22 用户决定 D4）

用户原话："仓库已公开，替换为等价合成行情夹具。保留测试合同；真实源测试单独管理。不擅自重写 Git 历史。"

此前这里是 2026-10-12 从 moomoo OpenD / yfinance 取回的真实价格。现在的数字是**手写的**，
只保留测试要钉的那几件事（测试合同）：

    · 22 个交易日 2024-05-20 … 2024-06-20（05-27 阵亡将士日、06-19 六月节休市）
    · NVDA 2024-06-10 一拆十（不复权价当天掉到约十分之一）；拆后次日 06-11 除息 0.01 ——
      不复权 / 前复权的比值在除息日只动约 0.008%，不能被当成拆股
    · AAPL 同期没有拆股；窗口之后的分红让前复权价整体低一点（比值恒定）
    · moomoo：不复权（raw）与前复权（qfq）两套收盘
    · yfinance（`yf.download(auto_adjust=False, actions=True, group_by='column')` 的形状）：
      Close 已按拆股调整（拆前 = 不复权 / 10）、Adj Close 再按分红调整、Stock Splits / Dividends 列；
      价格按 float32 存（yfinance 的真实精度），所以从它还原出的不复权价与 moomoo 只在 1e-3 内一致，不是恒等

股票代码沿用 NVDA / AAPL，是为了和 `tests/fixtures/edgar/` 里的公司事实（SEC 公开申报）对上；**价格不是真实行情**。

重新生成（结果必须与仓库里的 CSV 逐位一致，见 test_fundamentals_prices_store.py）：
    python tests/fixtures/prices/synthetic_split_prices.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MOOMOO_CSV = HERE / "split_dividend_moomoo.csv"
YAHOO_CSV = HERE / "split_dividend_yahoo.csv"

DATES = pd.DatetimeIndex([
    "2024-05-20", "2024-05-21", "2024-05-22", "2024-05-23", "2024-05-24",
    "2024-05-28", "2024-05-29", "2024-05-30", "2024-05-31",
    "2024-06-03", "2024-06-04", "2024-06-05", "2024-06-06", "2024-06-07",
    "2024-06-10", "2024-06-11", "2024-06-12", "2024-06-13", "2024-06-14",
    "2024-06-17", "2024-06-18", "2024-06-20",
])
#: 手写的不复权收盘（NVDA 06-10 起是拆后价）
RAW = {
    "NVDA": [950.25, 955.50, 948.75, 1035.40, 1060.10, 1130.60, 1150.20, 1105.80, 1100.00,
             1150.30, 1165.70, 1225.40, 1210.90, 1200.00,
             121.00, 122.45, 125.10, 129.35, 131.05, 130.20, 135.55, 130.45],
    "AAPL": [190.10, 191.55, 190.45, 187.20, 189.05, 189.60, 190.25, 191.10, 192.05,
             194.15, 194.60, 195.40, 194.05, 196.55, 193.10, 207.25, 213.05, 214.30,
             212.45, 216.65, 214.25, 209.70],
}
SPLIT_DATE, SPLIT_RATIO = pd.Timestamp("2024-06-10"), 10.0
DIV_DATE, DIV_AMOUNT = pd.Timestamp("2024-06-11"), 0.01
#: 窗口之后的分红对整段前复权价的统一折算（比值恒定，不影响识别）
LATER_DIVIDENDS = {"NVDA": 0.9990, "AAPL": 1 / 1.0095}


def raw_close() -> pd.DataFrame:
    return pd.DataFrame(RAW, index=DATES)


def _dividend_factor(raw: pd.Series) -> pd.Series:
    """除息日之前的每一天乘 (1 − 分红 / 除息前一日收盘)。"""
    f = 1.0 - DIV_AMOUNT / raw[raw.index < DIV_DATE].iloc[-1]
    return pd.Series(np.where(raw.index < DIV_DATE, f, 1.0), index=raw.index)


def _split_factor(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(np.where(index < SPLIT_DATE, SPLIT_RATIO, 1.0), index=index)


def moomoo_frame() -> pd.DataFrame:
    raw = raw_close()
    qfq = pd.DataFrame({
        "NVDA": raw["NVDA"] * _dividend_factor(raw["NVDA"]) * LATER_DIVIDENDS["NVDA"] / _split_factor(DATES),
        "AAPL": raw["AAPL"] * LATER_DIVIDENDS["AAPL"],
    }).round(9)
    return pd.concat({"raw": raw, "qfq": qfq}, axis=1)


def yahoo_frame() -> pd.DataFrame:
    raw = raw_close()["NVDA"]
    close = raw / _split_factor(DATES)
    adj = close * _dividend_factor(raw) * LATER_DIVIDENDS["NVDA"]
    rng = np.random.default_rng(20240610)
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.008, len(close)))
    top = np.maximum(open_, close) * (1 + rng.uniform(0.002, 0.025, len(close)))
    bottom = np.minimum(open_, close) * (1 - rng.uniform(0.002, 0.025, len(close)))
    f32 = lambda s: pd.Series(np.asarray(s, dtype=np.float32).astype(float), index=DATES)   # noqa: E731
    cols = {
        "Adj Close": f32(adj), "Close": f32(close),
        "Dividends": pd.Series(np.where(DATES == DIV_DATE, DIV_AMOUNT, 0.0), index=DATES),
        "High": f32(top), "Low": f32(bottom), "Open": f32(open_),
        "Stock Splits": pd.Series(np.where(DATES == SPLIT_DATE, SPLIT_RATIO, 0.0), index=DATES),
        "Volume": pd.Series(rng.integers(200_000_000, 520_000_000, len(close)), index=DATES),
    }
    df = pd.concat({k: v.rename("NVDA").to_frame() for k, v in cols.items()}, axis=1)
    df.columns.names = ["Price", "Ticker"]
    df.index.name = "Date"
    return df


def write() -> None:
    moomoo_frame().to_csv(MOOMOO_CSV, lineterminator="\n")
    yahoo_frame().to_csv(YAHOO_CSV, lineterminator="\n")


if __name__ == "__main__":
    write()
