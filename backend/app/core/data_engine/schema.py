"""
标准化 Schema 定义与强制层。

所有数据源的输出在进入面板工厂前，都必须经过 SchemaEnforcer
转换为统一的 long-format DataFrame，列顺序与类型严格一致。

标准格式（每行代表一个 ticker 在一个时间点的数据）：
    timestamp  : datetime64[ns]   交易日期（去除时区，UTC 00:00）
    ticker     : str / category   资产代码
    open       : float64
    high       : float64
    low        : float64
    close      : float64
    volume     : float64
    vwap       : float64          可为 NaN，由预处理层合成
    adj_factor : float64          复权因子，默认 1.0
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 列定义
# ---------------------------------------------------------------------------

STANDARD_COLUMNS: List[str] = [
    "timestamp", "ticker",
    "open", "high", "low", "close", "volume",
    "vwap", "adj_factor",
]

PRICE_FIELDS: List[str]   = ["open", "high", "low", "close", "vwap"]
NUMERIC_FIELDS: List[str] = PRICE_FIELDS + ["volume", "adj_factor"]

# 列 → 默认填充值（NaN 表示不自动填充，保持 NaN）
_COLUMN_DEFAULTS = {
    "adj_factor": 1.0,
    "vwap":       np.nan,   # 由 SyntheticFieldBuilder 后续计算
}

# 列 → dtype
_COLUMN_DTYPES = {
    "timestamp":  "datetime64[ns]",
    "ticker":     "object",          # str
    "open":       "float64",
    "high":       "float64",
    "low":        "float64",
    "close":      "float64",
    "volume":     "float64",
    "vwap":       "float64",
    "adj_factor": "float64",
}


# ---------------------------------------------------------------------------
# 自定义异常
# ---------------------------------------------------------------------------

class SchemaError(Exception):
    """当数据不符合标准 Schema 时抛出。"""
    pass


# ---------------------------------------------------------------------------
# SchemaEnforcer
# ---------------------------------------------------------------------------

class SchemaEnforcer:
    """
    将任意来源的 long-format DataFrame 强制转换为标准 Schema。

    行为规则
    --------
    - 缺失的数值列 → 用 _COLUMN_DEFAULTS 填充（adj_factor=1.0，其余 NaN）
    - 缺失的 timestamp / ticker 列 → 抛出 SchemaError
    - 多余列 → 默认保留（allow_extra=True），或丢弃（allow_extra=False）
    - timestamp 强制转为 UTC-aware → naive datetime64[ns]
    - ticker 强制转为 str
    """

    def __init__(self, allow_extra: bool = True) -> None:
        self.allow_extra = allow_extra

    def enforce(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        强制转换 DataFrame 到标准 Schema。

        Parameters
        ----------
        df : 输入 DataFrame（long-format 或 wide-format 均可，
             但必须包含 timestamp 和 ticker 列）

        Returns
        -------
        pd.DataFrame  列顺序为 STANDARD_COLUMNS（+ 额外列）
        """
        if df is None or df.empty:
            return self._empty_frame()

        df = df.copy()

        # 1. 规范化列名（小写、去空格）
        df.columns = [str(c).strip().lower() for c in df.columns]

        # 2. 必填列检查
        self._check_required(df)

        # 3. 处理 timestamp
        df["timestamp"] = self._coerce_timestamp(df["timestamp"])

        # 4. 处理 ticker
        df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

        # 5. 补充缺失的标准列
        for col in STANDARD_COLUMNS:
            if col not in df.columns:
                default = _COLUMN_DEFAULTS.get(col, np.nan)
                df[col] = default

        # 6. 数值列类型转换
        for col in NUMERIC_FIELDS:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        # 7. 处理多余列
        extra_cols = [c for c in df.columns if c not in STANDARD_COLUMNS]
        if extra_cols and not self.allow_extra:
            warnings.warn(
                f"SchemaEnforcer: 丢弃多余列 {extra_cols}",
                stacklevel=2,
            )
            df = df.drop(columns=extra_cols)

        # 8. 按标准列顺序重排（多余列追加到末尾）
        ordered = STANDARD_COLUMNS + [c for c in df.columns if c not in STANDARD_COLUMNS]
        df = df[ordered]

        # 9. 去重（同一 timestamp+ticker 保留最后一行）
        before = len(df)
        df = df.drop_duplicates(subset=["timestamp", "ticker"], keep="last")
        if len(df) < before:
            warnings.warn(
                f"SchemaEnforcer: 去除重复行 {before - len(df)} 条",
                stacklevel=2,
            )

        # 10. 排序
        df = df.sort_values(["timestamp", "ticker"]).reset_index(drop=True)

        return df

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _check_required(df: pd.DataFrame) -> None:
        required = {"timestamp", "ticker"}
        missing = required - set(df.columns)
        if missing:
            raise SchemaError(
                f"缺少必填列：{missing}。"
                f"当前列：{list(df.columns)}"
            )

    @staticmethod
    def _coerce_timestamp(series: pd.Series) -> pd.Series:
        """将任意时间格式统一转为 naive datetime64[ns]（UTC 日期）。"""
        ts = pd.to_datetime(series, utc=False, errors="coerce")
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
        # 截断到日（去除时间部分）
        ts = ts.dt.normalize()
        return ts

    @staticmethod
    def _empty_frame() -> pd.DataFrame:
        return pd.DataFrame(columns=STANDARD_COLUMNS).astype(
            {col: _COLUMN_DTYPES[col] for col in STANDARD_COLUMNS}
        )


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def wide_to_long(
    wide_dict: dict[str, pd.DataFrame],
    ticker_axis: int = 1,
) -> pd.DataFrame:
    """
    将 RawDataset（field → wide DataFrame）转换为 long-format DataFrame。

    Parameters
    ----------
    wide_dict  : dict[field_name, DataFrame(time × asset)]
    ticker_axis: 1 表示列是 ticker（默认）

    Returns
    -------
    pd.DataFrame  long-format，含 timestamp、ticker、各字段列
    """
    frames: list[pd.DataFrame] = []
    for field, df in wide_dict.items():
        melted = df.reset_index().melt(
            id_vars=df.index.name or "index",
            var_name="ticker",
            value_name=field,
        )
        melted = melted.rename(columns={df.index.name or "index": "timestamp"})
        frames.append(melted.set_index(["timestamp", "ticker"]))

    if not frames:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    combined = pd.concat(frames, axis=1).reset_index()
    return combined
