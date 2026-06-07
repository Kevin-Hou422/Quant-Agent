"""
gp_engine.py — 种子库与随机 Alpha 生成工具。

提供供 PopulationEvolver 使用的：
  - _SEED_DSLS / _SEED_DSLS_BY_FAMILY  — 按因子家族分类的种子 DSL 库
  - generate_random_alpha()             — 随机 DSL 节点树生成
  - get_seeds_for_family()              — 按家族筛选种子
  - GPAlphaResult                       — 评估结果数据类

已退役（2026-06-07）：
  AlphaEvolver — 旧版 DEAP 风格 GP 引擎，由 PopulationEvolver 完全替代。
  请使用: from app.core.gp_engine.population_evolver import PopulationEvolver
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..alpha_engine.parser import Parser as _Parser
from ..alpha_engine.typed_nodes import Node
from ..alpha_engine.dsl_executor import Executor as DSLExecutor

logger = logging.getLogger(__name__)

_executor    = DSLExecutor()
_parser_inst = _Parser()

# ---------------------------------------------------------------------------
# Seed DSL library — organised by factor family.
# Each family has ≥5 distinct seeds covering different window sizes,
# data fields, and signal constructions.
# ---------------------------------------------------------------------------

_SEED_DSLS_BY_FAMILY: dict[str, list[str]] = {

    # ── Momentum ─────────────────────────────────────────────────────────
    # Log-return price momentum — canonical, multiple horizons
    "momentum": [
        "rank(ts_delta(log(close), 5))",
        "rank(ts_delta(log(close), 10))",
        "rank(ts_delta(log(close), 20))",
        "rank(ts_delta(log(close), 40))",
        "zscore(ts_delta(log(close), 5))",
        "rank(ts_delta(vwap, 10))",
        "rank(ts_rank(close, 20))",
        "rank(ts_rank(returns, 10))",
    ],

    # ── Mean Reversion ────────────────────────────────────────────────────
    # Short-term price reversal — 1-5 day horizons
    "reversion": [
        "rank(-ts_delta(close, 1))",
        "rank(-ts_delta(close, 3))",
        "rank(-ts_delta(close, 5))",
        "rank(-ts_rank(close, 5))",
        "rank(ts_zscore(returns, 10))",
        "zscore(-ts_delta(close, 3))",
    ],

    # ── Volatility / Low-Vol ──────────────────────────────────────────────
    # Low-volatility anomaly — Ang et al. (2006)
    "volatility": [
        "rank(-ts_std(returns, 20))",
        "rank(-ts_std(returns, 60))",
        "rank(-ts_var(returns, 20))",
        "zscore(-ts_std(returns, 10))",
        "rank(-ts_std(close, 20))",
    ],

    # ── Risk-Adjusted Momentum ────────────────────────────────────────────
    # Momentum normalised by realised volatility (Sharpe-like)
    "risk_adjusted": [
        "rank(ts_delta(log(close), 20) / ts_std(returns, 20))",
        "rank(ts_delta(log(close), 10) / ts_std(returns, 10))",
        "rank(ts_mean(returns, 20) / ts_std(returns, 20))",
        "rank(ts_delta(log(close), 5) / ts_std(returns, 5))",
        "rank(ts_delta(log(vwap), 10) / ts_std(returns, 20))",
    ],

    # ── Liquidity / Volume ────────────────────────────────────────────────
    # Illiquidity premium + volume-based signals
    "liquidity": [
        "rank(-ts_mean(volume, 20))",
        "rank(-ts_mean(volume, 60))",
        "rank(ts_delta(log(volume), 5))",
        "rank(ts_delta(log(volume), 10))",
        "zscore(ts_mean(volume, 10))",
        "rank(ts_std(volume, 20))",
    ],

    # ── Price-Volume Correlation ──────────────────────────────────────────
    # Informed trading leaves traces in price-volume co-movement
    "price_volume_corr": [
        "rank(-ts_corr(close, volume, 20))",
        "rank(-ts_corr(close, volume, 10))",
        "rank(ts_corr(returns, volume, 20))",
        "rank(-ts_corr(vwap, volume, 20))",
        "zscore(-ts_corr(close, volume, 10))",
    ],

    # ── Trend Following ───────────────────────────────────────────────────
    # Medium/long-window moving average alignment
    "trend_following": [
        "rank(ts_mean(close, 60))",
        "rank(ts_mean(close, 120))",
        "rank(ts_mean(returns, 20))",
        "rank(ts_decay_linear(close, 20))",
        "rank(ts_decay_linear(returns, 10))",
    ],

    # ── Composite / Multi-Signal ──────────────────────────────────────────
    # Combinations of two orthogonal signals
    "composite": [
        "rank(ts_delta(log(close), 10) * ts_delta(log(volume), 5))",
        "rank(ts_rank(close, 20) + ts_rank(volume, 10))",
        "rank(ts_delta(log(close), 10) - ts_std(returns, 20))",
        "rank(ts_delta(log(close), 5) - ts_delta(log(close), 20))",
        "rank(ts_mean(returns, 5) / ts_std(returns, 20))",
        "rank(-ts_delta(log(close), 3) + ts_delta(log(close), 20))",
        "rank(ts_delta(vwap, 5) + ts_delta(log(volume), 5))",
    ],

    # ── Regime-Conditional ────────────────────────────────────────────────
    # Signal with explicit market-state gating
    "conditional": [
        "rank(trade_when(close > ts_mean(close, 60), ts_delta(log(close), 10)))",
        "rank(trade_when(volume > ts_mean(volume, 20), -ts_delta(close, 3)))",
        "rank(trade_when(close > ts_mean(close, 20), ts_rank(close, 10)))",
    ],
}

# Flat list — backward-compatible, used by generate_random_alpha()
_SEED_DSLS: list[str] = [
    dsl
    for seeds in _SEED_DSLS_BY_FAMILY.values()
    for dsl in seeds
]


def get_seeds_for_family(family: str) -> list[str]:
    """
    Return seed DSLs that are appropriate for ``family``.

    Falls back to all seeds if the family is unknown or empty.
    Maps the 8-family taxonomy from FinancialInterpreter to the seed groups.
    """
    # Direct match
    if family in _SEED_DSLS_BY_FAMILY:
        return list(_SEED_DSLS_BY_FAMILY[family])

    # Alias mapping
    _ALIAS = {
        "momentum":          "momentum",
        "reversion":         "reversion",
        "volatility":        "volatility",
        "liquidity":         "liquidity",
        "price_volume_corr": "price_volume_corr",
        "trend_following":   "trend_following",
        "composite":         "composite",
        "quality":           "volatility",    # quality ≈ low-vol for OHLCV
        "risk_adjusted":     "risk_adjusted",
    }
    mapped = _ALIAS.get(family, "")
    if mapped and mapped in _SEED_DSLS_BY_FAMILY:
        return list(_SEED_DSLS_BY_FAMILY[mapped])

    return list(_SEED_DSLS)  # fallback: all seeds


def generate_random_alpha(depth: int = 4, factor_family: str = "") -> Node:
    """
    Generate a random valid typed_nodes.Node by parsing a seed DSL.

    When ``factor_family`` is provided, biases selection 60 % toward
    seeds from that family (remaining 40 % from the full pool).
    Retries up to 20 times to handle any rare parse failures.
    """
    family_seeds = get_seeds_for_family(factor_family) if factor_family else []

    for _ in range(20):
        try:
            if family_seeds and random.random() < 0.60:
                dsl = random.choice(family_seeds)
            else:
                dsl = random.choice(_SEED_DSLS)
            return _parser_inst.parse(dsl)
        except Exception:
            continue

    # Ultimate fallback — guaranteed to parse
    return _parser_inst.parse("rank(ts_delta(log(close), 10))")


# ---------------------------------------------------------------------------
# AlphaResult (GP 版，轻量，含 fitness)
# ---------------------------------------------------------------------------

@dataclass
class GPAlphaResult:
    dsl:        str
    fitness:    float   = 0.0
    sharpe:     float   = 0.0
    ic_ir:      float   = 0.0
    ann_return: float   = 0.0
    ann_turnover: float = 0.0


# ---------------------------------------------------------------------------
# Fitness 评估函数（顶层，可被 multiprocessing.Pool 序列化）
# ---------------------------------------------------------------------------

def _evaluate_individual(
    args: Tuple[str, Dict[str, np.ndarray]],
) -> GPAlphaResult:
    """
    对一条 DSL 字符串计算 fitness。

    完整公式（对标基线规格）:
        fitness = ic_ir + 0.5 × mean_IC - 0.1 × ann_turnover

    其中：
      - ic_ir        = mean(IC) / std(IC)    （截面 Spearman Rank IC 的 IC-IR）
      - mean_IC      = mean(IC series)        （平均 IC，体现方向性）
      - ann_turnover = mean(daily L1 signal Δ) × 252  （年化换手估算）

    不依赖 BacktestEngine（纯信号质量快速评估，支持多进程序列化）。
    """
    dsl, dataset = args
    try:
        # dataset values 是 numpy arrays（为可序列化）
        # 构建带 DatetimeIndex 的 DataFrame 供 Executor 使用
        T_raw = next(iter(dataset.values())).shape[0]
        dates = pd.bdate_range("2020-01-02", periods=T_raw)
        df_dataset = {
            k: pd.DataFrame(v, index=dates)
            for k, v in dataset.items()
        }
        signal = _executor.run_expr(dsl, df_dataset)
        close_arr = dataset.get("close")
        if close_arr is None or signal is None:
            return GPAlphaResult(dsl=dsl, fitness=-1.0)

        close = close_arr

        # ── 前向收益 ──────────────────────────────────────────────────────
        fwd_ret = (close[1:] - close[:-1]) / np.where(close[:-1] == 0, np.nan, close[:-1])
        sig_arr = signal.to_numpy() if hasattr(signal, "to_numpy") else np.array(signal)

        # ── 截面 Rank IC（向量化 Spearman via double-argsort）────────────
        T_ic = min(fwd_ret.shape[0], sig_arr.shape[0] - 1)
        ics: list[float] = []
        for t in range(T_ic):
            s = sig_arr[t]
            r = fwd_ret[t]
            mask = ~(np.isnan(s) | np.isnan(r))
            n_valid = mask.sum()
            if n_valid < 5:
                continue
            # Vectorised rank correlation via argsort (no scipy import per call)
            rs = np.argsort(np.argsort(s[mask])).astype(float)
            rr = np.argsort(np.argsort(r[mask])).astype(float)
            rs -= rs.mean(); rr -= rr.mean()
            denom = np.sqrt((rs ** 2).sum() * (rr ** 2).sum())
            if denom > 0:
                ics.append(float(np.dot(rs, rr) / denom))

        if not ics:
            return GPAlphaResult(dsl=dsl, fitness=-1.0)

        ic_arr  = np.array(ics)
        mean_ic = float(np.mean(ic_arr))
        ic_ir   = float(mean_ic / (np.std(ic_arr) + 1e-9))

        # ── 年化换手（信号的截面 L1 日变化量）────────────────────────────
        # pct-rank the signal cross-sectionally then measure daily L1 change
        sig_float = sig_arr.astype(float)
        # simple abs-diff of raw signal normalised by cross-sectional std
        daily_delta = np.abs(np.diff(sig_float, axis=0))
        ann_turnover = float(np.nanmean(daily_delta) * 252)

        # ── Composite fitness (baseline formula) ─────────────────────────
        fitness = ic_ir + 0.5 * mean_ic - 0.1 * ann_turnover

        return GPAlphaResult(
            dsl          = dsl,
            fitness      = fitness,
            sharpe       = ic_ir,      # IC-IR used as Sharpe proxy
            ic_ir        = ic_ir,
            ann_return   = mean_ic,    # mean IC as directional proxy
            ann_turnover = ann_turnover,
        )
    except Exception as e:
        logger.debug("Eval failed for '%s': %s", dsl[:80], e)
        return GPAlphaResult(dsl=dsl, fitness=-1.0)


# ---------------------------------------------------------------------------
# 辅助：DSL 字符串 → Node
# ---------------------------------------------------------------------------

def _parse_dsl(dsl: str) -> Node:
    from ..alpha_engine.parser import Parser
    return Parser().parse(dsl)
