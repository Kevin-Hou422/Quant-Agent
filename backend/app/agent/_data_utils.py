"""
_data_utils.py — data utilities shared by QuantTools.

  _make_synthetic_dataset  — deterministic synthetic OHLCV panel
  _partition               — IS / OOS split via DataPartitioner
  load_real_dataset        — load real market data via DatasetRegistry
  _run_backtest_core       — RealisticBacktester wrapper → metrics dict
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from app.agent._constants import _OVERFIT_THRESHOLD


def _make_synthetic_dataset(
    n_tickers: int = 20,
    n_days:    int = 252,
    seed:      int = 42,
) -> Dict[str, pd.DataFrame]:
    """Generate a fully deterministic synthetic OHLCV + vwap + returns panel."""
    rng     = np.random.default_rng(seed)
    dates   = pd.bdate_range("2021-01-04", periods=n_days)
    tickers = [f"T{i:03d}" for i in range(n_tickers)]

    close   = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.012, (n_days, n_tickers)), axis=0),
        index=dates, columns=tickers,
    )
    volume  = pd.DataFrame(
        rng.integers(500_000, 5_000_000, (n_days, n_tickers)).astype(float),
        index=dates, columns=tickers,
    )
    high    = close * (1 + rng.uniform(0, 0.02, close.shape))
    low     = close * (1 - rng.uniform(0, 0.02, close.shape))
    open_   = close.shift(1).fillna(close)
    vwap    = (high + low + close) / 3
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


def _partition(
    dataset:   Dict[str, pd.DataFrame],
    oos_ratio: float,
) -> Tuple[Dict, Dict]:
    """Split *dataset* into IS (train) and OOS (test) portions."""
    from app.core.data_engine.data_partitioner import DataPartitioner

    dates = next(iter(dataset.values())).index
    dp    = DataPartitioner(
        start     = str(dates[0].date()),
        end       = str(dates[-1].date()),
        oos_ratio = oos_ratio,
    )
    part = dp.partition(dataset)
    return part.train(), part.test()


def _partition_three_way(
    dataset:    Dict[str, pd.DataFrame],
    oos_ratio:  float = 0.30,
    test_ratio: float = 0.10,
) -> Tuple[Dict, Dict, Dict]:
    """
    F6 修复：三段式时间切割 IS / Validate / Test。

    Temporal ordering (no shuffling, no data leakage):
        IS:       [0,   n_is)           — GP 结构搜索 + Optuna 参数调优
        Validate: [n_is, n_is+n_val)    — GP 个体适应度评估（内部 "OOS"）
        Test:     [n_is+n_val, n)       — 真实样本外，GP 全程不可见

    Parameters
    ----------
    oos_ratio  : 非 IS 总比例（Validate + Test），默认 0.30
    test_ratio : Test 占全数据比例，默认 0.10；Validate = oos_ratio - test_ratio

    Returns
    -------
    (is_data, val_data, test_data)

    Raises
    ------
    ValueError : 数据量不足以支持三段切割（n_is < 20）
    """
    ref  = next(iter(dataset.values()))
    n    = len(ref)

    n_test = max(1, int(n * test_ratio))
    n_val  = max(1, int(n * oos_ratio) - n_test)
    n_is   = n - n_val - n_test

    if n_is < 20:
        raise ValueError(
            f"数据量不足以支持三段切割: n={n}, n_is={n_is}, n_val={n_val}, "
            f"n_test={n_test}。至少需要 n_is >= 20 个交易日。"
        )

    def _slice(start: int, end: int) -> Dict[str, pd.DataFrame]:
        return {field: df.iloc[start:end] for field, df in dataset.items()}

    return _slice(0, n_is), _slice(n_is, n_is + n_val), _slice(n_is + n_val, n)


def load_real_dataset(
    name:      str,
    start:     str   = "2021-01-01",
    end:       str   = "2024-01-01",
    oos_ratio: float = 0.30,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Load a real market dataset from DatasetRegistry and split IS / OOS.

    Parameters
    ----------
    name      : Registry dataset name (e.g. "us_tech_large", "crypto_major")
    start/end : Date range for the data fetch
    oos_ratio : Fraction of data reserved for out-of-sample evaluation

    Returns
    -------
    (is_data, oos_data) — both in dict[field → DataFrame(T × N)] format.

    Raises
    ------
    RuntimeError if the registry cannot load the dataset.
    """
    _, is_data, oos_data = load_real_research(name, start, end, oos_ratio, purpose="load_real_dataset")
    return is_data, oos_data


def load_real_research(
    name:      str,
    start:     str   = "2021-01-01",
    end:       str   = "2024-01-01",
    oos_ratio: float = 0.30,
    purpose:   str   = "chat",
):
    """
    加载真实数据集 → 研究上下文 → 在**选择段内**切 IS / OOS。返回 (ctx, is_data, oos_data)。

    复审 2026-10-19 N09-c：此前对全量面板按 70/30 切，OOS 的后段就是冻结 Test 段
    （chat 里指定数据集跑 GP 时 OOS 含 474 个冻结日期）。有冻结边界却切不出选择段 → 抛错，不退回全量。
    """
    from app.core.data_engine.data_partitioner import FrozenBoundaryError
    from app.core.data_engine.dataset_registry import load_registry_dataset
    from app.core.data_engine.research_context import open_research
    try:
        ds = load_registry_dataset(name, start=start, end=end, use_cache=True)
    except Exception as exc:
        raise RuntimeError(
            f"load_real_dataset: failed to load '{name}' [{start} → {end}]: {exc}"
        ) from exc
    try:
        ctx = open_research(ds.data, name, purpose)
    except FrozenBoundaryError as exc:
        raise RuntimeError(f"load_real_dataset: '{name}' 切不出研究窗口：{exc}") from exc
    is_data, oos_data = _partition(ctx.selection, oos_ratio)
    return ctx, is_data, oos_data


def _run_backtest_core(
    dsl:      str,
    cfg:      Any,
    is_data:  Dict,
    oos_data: Optional[Dict],
) -> Dict[str, Any]:
    """
    Run IS+OOS backtest and compute overfitting score.

    Returns
    -------
    dict with keys: is_sharpe, oos_sharpe, is_return, is_turnover, is_ic（IC 均值）,
                    is_ic_ir（IC-IR，mean/std 不年化）, is_ic_n（有效观测数）,
                    overfitting_score, is_overfit, summary
    """
    from app.core.backtest_engine.realistic_backtester import RealisticBacktester

    bt     = RealisticBacktester(config=cfg, min_obs=0)  # min_obs=0：内部搜索排序用，只标注不置空（见 RiskReport._apply_sample_sufficiency）
    result = bt.run(dsl, is_data, oos_dataset=oos_data)
    is_r   = result.is_report
    oos_r  = result.oos_report

    def _f(v: Any) -> Optional[float]:
        return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None

    is_sharpe  = _f(is_r.sharpe_ratio) or 0.0
    oos_sharpe = _f(oos_r.sharpe_ratio) if oos_r else None

    if oos_sharpe is not None and abs(is_sharpe) > 1e-9:
        degradation   = (is_sharpe - oos_sharpe) / abs(is_sharpe)
        overfit_score = float(np.clip(degradation, 0.0, 1.0))
    else:
        overfit_score = 0.0

    return {
        "is_sharpe":         is_sharpe,
        "oos_sharpe":        oos_sharpe,
        "is_return":         _f(is_r.annualized_return),
        "is_turnover":       _f(is_r.ann_turnover),
        "is_ic":             _f(is_r.mean_ic),
        "is_ic_ir":          _f(getattr(is_r, "ic_ir", None)),   # 真正的 IC-IR（mean/std，不年化；审计决定 2）
        "is_ic_n":           int(getattr(is_r, "ic_n_obs", 0) or 0),
        "overfitting_score": overfit_score,
        "is_overfit":        overfit_score > _OVERFIT_THRESHOLD,
        "summary":           result.summary(),
    }
