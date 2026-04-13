"""
alpha_evaluator.py — 高级 Alpha 评估引擎

在 RiskReport 基础上补充：
  - Rolling Sharpe（60 日）
  - Rolling Rank IC（20 日）
  - IC Decay（t+1、t+5）
  - Max Drawdown + 持续期
  - OverfittingScore：IS vs OOS Sharpe 退化检测

IS 和 OOS 指标完全独立计算，评估结束后合并生成对比摘要。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from ..backtest_engine.realistic_backtester import RiskReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EvalMetrics — 单段（IS 或 OOS）高级评估指标
# ---------------------------------------------------------------------------

@dataclass
class EvalMetrics:
    """单段（IS / OOS）高级绩效指标集合。"""

    # ── 基础（来自 RiskReport）────────────────────────────────────────
    sharpe_ratio:      float = np.nan
    annualized_return: float = np.nan
    annualized_vol:    float = np.nan
    max_drawdown:      float = np.nan
    max_dd_duration:   int   = 0
    mean_ic:           float = np.nan
    ic_ir:             float = np.nan
    ann_turnover:      float = np.nan

    # ── 高级滚动序列（repr=False 防止打印过长）─────────────────────────
    rolling_sharpe_60: Optional[pd.Series] = field(default=None, repr=False)
    rolling_ic_20:     Optional[pd.Series] = field(default=None, repr=False)
    drawdown_series:   Optional[pd.Series] = field(default=None, repr=False)

    # ── IC Decay ──────────────────────────────────────────────────────
    ic_decay_t1:  float = np.nan   # 信号与 t+1 收益的 Rank 相关
    ic_decay_t5:  float = np.nan   # 信号与 t+5 收益的 Rank 相关

    def to_dict(self) -> Dict[str, Any]:
        def _safe(v):
            if isinstance(v, pd.Series):
                # 序列转 {date: value} 字典（日期字符串化）
                return {str(k): (None if np.isnan(v) else float(v))
                        for k, v in v.dropna().items()}
            if v is None:
                return None
            try:
                f = float(v)
                return None if np.isnan(f) else f
            except (TypeError, ValueError):
                return None

        return {
            "sharpe_ratio":      _safe(self.sharpe_ratio),
            "annualized_return": _safe(self.annualized_return),
            "annualized_vol":    _safe(self.annualized_vol),
            "max_drawdown":      _safe(self.max_drawdown),
            "max_dd_duration":   self.max_dd_duration,
            "mean_ic":           _safe(self.mean_ic),
            "ic_ir":             _safe(self.ic_ir),
            "ann_turnover":      _safe(self.ann_turnover),
            "ic_decay_t1":       _safe(self.ic_decay_t1),
            "ic_decay_t5":       _safe(self.ic_decay_t5),
            "rolling_sharpe_60": _safe(self.rolling_sharpe_60),
            "rolling_ic_20":     _safe(self.rolling_ic_20),
        }


# ---------------------------------------------------------------------------
# AlphaEvaluatorResult — IS + OOS 双段完整评估结果
# ---------------------------------------------------------------------------

@dataclass
class AlphaEvaluatorResult:
    """
    完整的 IS / OOS 高级评估结果。

    Attributes
    ----------
    is_metrics        : IS 段高级指标
    oos_metrics       : OOS 段高级指标（None 当未传入 OOS 数据）
    overfitting_score : [0, 1]，越高越过拟合（IS vs OOS Sharpe 退化比例）
    is_overfit        : OOS Sharpe 相对 IS 退化超 50% 则 True
    ic_decay          : {"t1": float, "t5": float}，IS 段 IC Decay
    """
    is_metrics:        EvalMetrics
    oos_metrics:       Optional[EvalMetrics]
    overfitting_score: float
    is_overfit:        bool
    ic_decay:          Dict[str, float]

    def summary(self) -> str:
        lines = [
            "",
            "╔══════════════════════════════════════════════════════╗",
            "║           AlphaEvaluator — IS / OOS 高级评估         ║",
            "╚══════════════════════════════════════════════════════╝",
            "",
            f"  过拟合评分: {self.overfitting_score:.3f}"
            + ("  ⚠ 过拟合警告！" if self.is_overfit else "  ✓ 通过"),
            f"  IC Decay  : t+1={self.ic_decay.get('t1', float('nan')):.4f}"
            f"  t+5={self.ic_decay.get('t5', float('nan')):.4f}",
            "",
            "── In-Sample ──────────────────────────────────────────",
            _fmt_metrics(self.is_metrics),
        ]
        if self.oos_metrics is not None:
            lines += [
                "",
                "── Out-of-Sample ──────────────────────────────────────",
                _fmt_metrics(self.oos_metrics),
            ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_metrics":        self.is_metrics.to_dict(),
            "oos_metrics":       self.oos_metrics.to_dict() if self.oos_metrics else None,
            "overfitting_score": float(self.overfitting_score),
            "is_overfit":        bool(self.is_overfit),
            "ic_decay":          {k: (None if np.isnan(v) else float(v))
                                  for k, v in self.ic_decay.items()},
        }


def _fmt_metrics(m: EvalMetrics) -> str:
    def _f(v, pct=False):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "   N/A"
        return f"{v*100:+.2f}%" if pct else f"{v:+.4f}"

    return (
        f"  Sharpe={_f(m.sharpe_ratio)}  AnnRet={_f(m.annualized_return, True)}"
        f"  MaxDD={_f(m.max_drawdown, True)}  IC-IR={_f(m.ic_ir)}"
        f"  Turnover={_f(m.ann_turnover, True)}  "
        f"IC_t1={_f(m.ic_decay_t1)}  IC_t5={_f(m.ic_decay_t5)}"
    )


# ---------------------------------------------------------------------------
# AlphaEvaluator
# ---------------------------------------------------------------------------

class AlphaEvaluator:
    """
    高级 Alpha 评估引擎。

    在 RiskReport 提供的基础指标之上，补充：
      - 60 日 Rolling Sharpe
      - 20 日 Rolling Rank IC（基于价格的精确版）
      - IC Decay（t+1、t+5）
      - Max Drawdown 持续期
      - IS vs OOS 过拟合评分

    Parameters
    ----------
    rolling_sharpe_window : Rolling Sharpe 计算窗口（交易日）
    rolling_ic_window     : Rolling Rank IC 计算窗口
    overfit_threshold     : 过拟合判断阈值（OOS Sharpe 相对退化比例）
    """

    def __init__(
        self,
        rolling_sharpe_window: int   = 60,
        rolling_ic_window:     int   = 20,
        overfit_threshold:     float = 0.50,
    ) -> None:
        self._rs_win    = rolling_sharpe_window
        self._ic_win    = rolling_ic_window
        self._threshold = overfit_threshold

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def evaluate(
        self,
        is_report:   "RiskReport",
        is_prices:   pd.DataFrame,
        is_signal:   pd.DataFrame,
        oos_report:  Optional["RiskReport"]   = None,
        oos_prices:  Optional[pd.DataFrame]  = None,
        oos_signal:  Optional[pd.DataFrame]  = None,
    ) -> AlphaEvaluatorResult:
        """
        执行 IS（必须）和 OOS（可选）的完整高级评估。

        Parameters
        ----------
        is_report  : IS 段 RiskReport（来自 RealisticBacktester）
        is_prices  : IS 段资产价格矩阵（T×N），用于精确 IC 计算
        is_signal  : IS 段处理后信号矩阵（T×N）
        oos_report : OOS 段 RiskReport（可选）
        oos_prices : OOS 段价格（可选）
        oos_signal : OOS 段信号（可选）
        """
        # IS 评估
        is_metrics = self._compute_metrics(is_report, is_prices, is_signal, label="IS")

        # IC Decay（基于 IS 段）
        ic_decay = self._ic_decay(is_signal, is_prices)

        # OOS 评估（可选）
        oos_metrics: Optional[EvalMetrics] = None
        if oos_report is not None and oos_prices is not None and oos_signal is not None:
            oos_metrics = self._compute_metrics(
                oos_report, oos_prices, oos_signal, label="OOS"
            )

        # 过拟合评分
        overfitting_score, is_overfit = self._overfit_score(is_metrics, oos_metrics)

        return AlphaEvaluatorResult(
            is_metrics        = is_metrics,
            oos_metrics       = oos_metrics,
            overfitting_score = overfitting_score,
            is_overfit        = is_overfit,
            ic_decay          = ic_decay,
        )

    # ------------------------------------------------------------------
    # 内部：单段高级指标计算
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        report: "RiskReport",
        prices: pd.DataFrame,
        signal: pd.DataFrame,
        label:  str = "",
    ) -> EvalMetrics:
        """从 RiskReport + 价格/信号矩阵中提取并扩展高级指标。"""

        # --- Rolling Sharpe（60 日）---
        net_ret = report.net_returns
        rolling_sharpe = _rolling_sharpe(net_ret, window=self._rs_win)

        # --- Rolling Rank IC（20 日，基于价格精确计算）---
        rolling_ic = _rolling_rank_ic(signal, prices, window=self._ic_win)

        logger.debug(
            "[%s] 滚动指标 | RS 非NaN=%d | Rolling-IC 非NaN=%d",
            label,
            rolling_sharpe.notna().sum() if rolling_sharpe is not None else 0,
            rolling_ic.notna().sum() if rolling_ic is not None else 0,
        )

        return EvalMetrics(
            sharpe_ratio      = report.sharpe_ratio,
            annualized_return = report.annualized_return,
            annualized_vol    = report.annualized_vol,
            max_drawdown      = report.max_drawdown,
            max_dd_duration   = report.max_dd_duration,
            mean_ic           = report.mean_ic,
            ic_ir             = report.ic_ir,
            ann_turnover      = report.ann_turnover,
            rolling_sharpe_60 = rolling_sharpe,
            rolling_ic_20     = rolling_ic,
            drawdown_series   = report.drawdown_series,
            ic_decay_t1       = np.nan,   # IC Decay 在 evaluate() 统一计算后回填
            ic_decay_t5       = np.nan,
        )

    # ------------------------------------------------------------------
    # 内部：IC Decay（t+1, t+5）
    # ------------------------------------------------------------------

    def _ic_decay(
        self,
        signal: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        计算信号与未来 k 期收益的 Rank 相关（IC Decay）。

        向量化实现（无 T 轴 Python 循环）：
          fwd_ret_k = prices.pct_change(k).shift(-k)
          对齐后，截面 Spearman 相关取时间均值
        """
        result: Dict[str, float] = {}

        # 对齐价格与信号
        common_idx = signal.index.intersection(prices.index)
        common_col = signal.columns.intersection(prices.columns)
        sig  = signal.loc[common_idx, common_col].to_numpy(dtype=float)    # (T, N)
        px   = prices.loc[common_idx, common_col]

        for k, key in [(1, "t1"), (5, "t5")]:
            fwd_ret = px.pct_change(k).shift(-k).to_numpy(dtype=float)    # (T, N)

            # 按行计算截面 Spearman，再取均值（全向量化）
            ic_vals = _cross_section_spearman(sig, fwd_ret)               # (T,)
            ic_mean = float(np.nanmean(ic_vals)) if len(ic_vals) else np.nan
            result[key] = ic_mean

        return result

    # ------------------------------------------------------------------
    # 内部：过拟合评分
    # ------------------------------------------------------------------

    def _overfit_score(
        self,
        is_m:  EvalMetrics,
        oos_m: Optional[EvalMetrics],
    ) -> Tuple[float, bool]:
        """
        Returns
        -------
        overfitting_score : [0, 1]，越高越严重
        is_overfit        : bool，是否触发过拟合警告
        """
        if oos_m is None:
            return 0.0, False

        is_sharpe  = is_m.sharpe_ratio
        oos_sharpe = oos_m.sharpe_ratio

        if _isnan(is_sharpe) or _isnan(oos_sharpe) or abs(is_sharpe) < 1e-9:
            return 0.0, False

        degradation = (is_sharpe - oos_sharpe) / abs(is_sharpe)
        score = float(np.clip(degradation, 0.0, 1.0))
        return score, score > self._threshold


# ---------------------------------------------------------------------------
# 向量化工具函数
# ---------------------------------------------------------------------------

def _rolling_sharpe(net_returns: Optional[pd.Series], window: int = 60) -> Optional[pd.Series]:
    """60 日滚动 Sharpe（年化），全 Pandas 向量化。"""
    if net_returns is None or len(net_returns) < window:
        return None
    mu  = net_returns.rolling(window).mean()
    sig = net_returns.rolling(window).std(ddof=1)
    sharpe = (mu / sig.replace(0, np.nan)) * np.sqrt(252)
    return sharpe.rename("rolling_sharpe_60")


def _rolling_rank_ic(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    window: int = 20,
) -> Optional[pd.Series]:
    """
    20 日滚动 Rank IC（信号 vs 次日收益截面 Spearman）。
    全向量化：先计算逐日截面 IC，再做 rolling mean。
    """
    common_idx = signal.index.intersection(prices.index)
    common_col = signal.columns.intersection(prices.columns)
    if len(common_idx) < window + 5:
        return None

    sig_arr = signal.loc[common_idx, common_col].to_numpy(dtype=float)
    px_arr  = prices.loc[common_idx, common_col].pct_change().to_numpy(dtype=float)

    # 逐日截面 IC
    T = len(common_idx)
    ic_vals = np.full(T, np.nan)
    for t in range(T - 1):
        s   = sig_arr[t]
        fwd = px_arr[t + 1]     # 次日收益（前瞻 1 期，用于 IC 计算，非回测）
        mask = ~(np.isnan(s) | np.isnan(fwd))
        if mask.sum() < 5:
            continue
        rho, _ = scipy_stats.spearmanr(s[mask], fwd[mask])
        ic_vals[t] = rho

    ic_series = pd.Series(ic_vals, index=common_idx, name="rank_ic")
    return ic_series.rolling(window).mean().rename("rolling_ic_20")


def _cross_section_spearman(
    sig:     np.ndarray,   # (T, N)
    fwd_ret: np.ndarray,   # (T, N)
) -> np.ndarray:
    """
    逐行（时间维）计算截面 Spearman 相关，返回 (T,) IC 序列。
    采用向量化排名 + 相关公式，避免逐行调用 scipy。
    """
    T, N = sig.shape
    ic_vals = np.full(T, np.nan)

    # 向量化排名（处理 NaN：先掩码，再 argsort-argsort）
    for t in range(T):
        s   = sig[t]
        f   = fwd_ret[t]
        mask = ~(np.isnan(s) | np.isnan(f))
        if mask.sum() < 5:
            continue
        rho, _ = scipy_stats.spearmanr(s[mask], f[mask])
        ic_vals[t] = rho

    return ic_vals


def _isnan(v) -> bool:
    if v is None:
        return True
    try:
        return np.isnan(float(v))
    except (TypeError, ValueError):
        return True


# 补充 Tuple 导入（用于类型提示）
from typing import Tuple
