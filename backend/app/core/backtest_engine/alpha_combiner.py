"""
alpha_combiner.py — Multi-Alpha joint portfolio construction.

Combines multiple low-correlated Alpha signals from AlphaPool into a single
composite signal using several weighting strategies.

Motivation:
  AlphaPool accumulates diverse candidates but they are never merged into a
  joint portfolio — each Alpha is evaluated independently.  AlphaCombiner
  bridges this gap: it computes the optimal combination weights on IS data
  and returns a composite signal ready for portfolio construction.

Supported methods
-----------------
  ic_weighted  : weight ∝ IS IC-IR per alpha (default)
  equal_weight : uniform 1/N weights (baseline)
  min_variance : minimize composite signal variance on IS data

Usage
-----
    combiner = AlphaCombiner()
    weights  = combiner.optimize_weights(signals, returns, method="ic_weighted")
    joint    = combiner.combine(signals, weights=weights)
    # joint is a (T × N) signal DataFrame ready for PortfolioConstructor
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
from ..alpha_engine.fast_ops import average_ranks_1d
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IC-IR helper
# ---------------------------------------------------------------------------

def _ic_ir(signal: pd.DataFrame, returns: pd.DataFrame) -> float:
    """
    Compute the IC-IR (Spearman rank IC mean / std) between a signal matrix
    and 1-day forward returns on the same universe.

    Returns 0.0 on failure or insufficient data.
    """
    sig = signal.to_numpy(dtype=float)
    ret = returns.to_numpy(dtype=float)
    T   = min(sig.shape[0] - 1, ret.shape[0] - 1)
    ics: list[float] = []
    for t in range(T):
        # ret[t+1] is the t→t+1 realized return: signal at t must predict
        # the FORWARD return, not the same-day (t-1→t) return
        s, r = sig[t], ret[t + 1]
        mask = ~(np.isnan(s) | np.isnan(r))
        if mask.sum() < 5:
            continue
        # 缺陷 C-1：必须用**平均秩**（并列取均值），不是 argsort 两次的序数名次 ——
        # 后者把截面恒定的零信息信号排成 [0,1,2,...]，算出按列顺序的伪 IC。
        rs = average_ranks_1d(s[mask])
        rr = average_ranks_1d(r[mask])
        rs -= rs.mean()
        rr -= rr.mean()
        denom = np.sqrt((rs ** 2).sum() * (rr ** 2).sum())
        if denom > 0:
            ics.append(float(np.dot(rs, rr) / denom))
    if len(ics) < 5:
        return 0.0
    arr = np.array(ics)
    return float(np.mean(arr) / (np.std(arr) + 1e-9))


# ---------------------------------------------------------------------------
# AlphaCombiner
# ---------------------------------------------------------------------------

class AlphaCombiner:
    """
    Combine multiple Alpha signal matrices into one composite signal.

    Parameters
    ----------
    clip_extreme : If True, winsorize each input signal at ±3σ before combining.
    """

    def __init__(self, clip_extreme: bool = True) -> None:
        self.clip_extreme = clip_extreme

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize_weights(
        self,
        signals:  Dict[str, pd.DataFrame],
        returns:  Optional[pd.DataFrame] = None,
        method:   str = "ic_weighted",
    ) -> Dict[str, float]:
        """
        Estimate combination weights from IS data.

        Parameters
        ----------
        signals : dict[dsl → (T × N) signal DataFrame]
        returns : (T × N) 1-day forward returns (required for ic_weighted)
        method  : "ic_weighted" | "equal_weight" | "min_variance"

        Returns
        -------
        dict[dsl → float weight], normalised to sum to 1.
        """
        if not signals:
            return {}

        if method == "equal_weight":
            n = len(signals)
            return {dsl: 1.0 / n for dsl in signals}

        if method == "ic_weighted":
            if returns is None:
                logger.warning("ic_weighted requires returns; falling back to equal_weight")
                return self.optimize_weights(signals, method="equal_weight")
            return self._ic_weights(signals, returns)

        if method == "min_variance":
            if returns is None:
                logger.warning("min_variance 需要收益序列（按因子组合收益估协方差）；退回**等权**")
                return self.optimize_weights(signals, method="equal_weight")
            return self._min_variance_weights(signals, returns)

        raise ValueError(
            f"Unknown method '{method}'. Choose: ic_weighted, equal_weight, min_variance"
        )

    def combine(
        self,
        signals: Dict[str, pd.DataFrame],
        weights: Optional[Dict[str, float]] = None,
        method:  str = "ic_weighted",
        returns: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Produce the composite signal matrix.

        Parameters
        ----------
        signals : dict[dsl → (T × N) signal DataFrame]
        weights : pre-computed weights (None = compute from signals using method)
        method  : weighting method when weights is None
        returns : forward returns (needed when method="ic_weighted")

        Returns
        -------
        (T × N) pd.DataFrame composite signal
        """
        if not signals:
            raise ValueError("signals dict is empty")

        # Align all signals to the same (index × columns)
        aligned = self._align(list(signals.values()))

        # Optionally winsorize each signal
        if self.clip_extreme:
            aligned = [self._winsorize(s) for s in aligned]

        if weights is None:
            weights = self.optimize_weights(
                {dsl: s for dsl, s in zip(signals.keys(), aligned)},
                returns=returns,
                method=method,
            )

        dsls = list(signals.keys())
        w    = np.array([weights.get(dsl, 0.0) for dsl in dsls], dtype=float)

        # Normalise weights (guard against all-zero)
        w_sum = np.abs(w).sum()
        if w_sum < 1e-12:
            w = np.ones(len(w)) / len(w)
        else:
            w = w / w_sum

        # Weighted average of signals row-by-row
        ref  = aligned[0]
        mat  = np.stack([s.to_numpy(dtype=float) for s in aligned], axis=2)  # (T, N, K)
        comp = np.nansum(mat * w[np.newaxis, np.newaxis, :], axis=2)         # (T, N)

        result = pd.DataFrame(comp, index=ref.index, columns=ref.columns)
        logger.info(
            "AlphaCombiner: combined %d alphas → shape=%s | method inferred from weights",
            len(signals), result.shape,
        )
        return result

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _ic_weights(
        self,
        signals: Dict[str, pd.DataFrame],
        returns: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Weight each alpha proportionally to its IS IC-IR (clipped to ≥ 0).
        Alphas with negative IC-IR get zero weight.
        """
        raw: Dict[str, float] = {}
        for dsl, sig in signals.items():
            # Align returns to signal's index/columns
            ret_aligned = returns.reindex(index=sig.index, columns=sig.columns)
            ic = _ic_ir(sig, ret_aligned)
            raw[dsl] = max(0.0, ic)

        total = sum(raw.values())
        if total < 1e-12:
            # All alphas have zero IC → equal weight
            n = len(signals)
            return {dsl: 1.0 / n for dsl in signals}

        return {dsl: v / total for dsl, v in raw.items()}

    #: 协方差向对角收缩的比例（样本协方差噪声大；固定值，不做数据驱动调参 —— 那又是一个被拟合的参数）
    MIN_VAR_SHRINK = 0.2
    #: 估协方差至少要这么多个可用日：max(每因子 × 因子数, 下限)。复审 2026-10-19 F12 的蒙特卡洛
    #: （tools/research/min_variance_study.py，docs/MIN_VARIANCE_STUDY.md）：等波动时 8 因子 20 天，
    #: 估出来的权重比等权差 10% 以上的概率是 86%；15 天 / 因子才降到 5% 以下。此前门槛是 max(2n, 20)。
    MIN_VAR_DAYS_PER_FACTOR = 15
    MIN_VAR_MIN_DAYS = 60

    def _min_variance_weights(
        self,
        signals: Dict[str, pd.DataFrame],
        returns: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        最小方差权重：**因子组合收益**的协方差 → 闭式最小方差解（只做多因子权重，向对角收缩）。

        外部审查 2026-10-18 F12（R9）：此前用每个信号**每日截面均值**的时间序列估协方差。
        对 zscore / demean 过的因子，截面均值恒为 0 —— 两个相同或相反的因子与第三个毫无区别，
        a、−a、a 拿到各 1/3。现在每个因子先变成一个美元中性的因子组合（截面去均值、按 Σ|w| 归一），
        收益 = Σ_i w_i,t · r_i,t，用这些收益序列的协方差。数据不够 → 显式等权（并告警）。
        """
        dsls = list(signals.keys())
        n    = len(dsls)
        if n < 2:
            return {dsls[0]: 1.0}

        vecs: list[np.ndarray] = []
        for sig in signals.values():
            arr = sig.to_numpy(dtype=float)
            ret = returns.reindex(index=sig.index, columns=sig.columns).to_numpy(dtype=float)
            w = arr - np.nanmean(arr, axis=1, keepdims=True)
            gross = np.nansum(np.abs(w), axis=1, keepdims=True)
            w = np.where(gross > 1e-12, w / np.where(gross > 1e-12, gross, 1.0), np.nan)
            both = np.isfinite(w) & np.isfinite(ret)
            pr = np.where(both, w * ret, 0.0).sum(axis=1)
            pr[~both.any(axis=1)] = np.nan
            vecs.append(pr)

        mat  = np.stack(vecs, axis=1)  # (T, n)：各因子组合的日收益
        mask = ~np.isnan(mat).any(axis=1)
        need = max(self.MIN_VAR_DAYS_PER_FACTOR * n, self.MIN_VAR_MIN_DAYS)
        if mask.sum() < need:
            logger.warning("[AlphaCombiner] min_variance 可用的因子组合收益只有 %d 天（%d 个因子至少要 %d 天），"
                           "权重退回**等权**", int(mask.sum()), n, need)
            return {dsl: 1.0 / n for dsl in dsls}

        sample = np.cov(mat[mask].T)                        # (n, n)
        cov = (1.0 - self.MIN_VAR_SHRINK) * sample + self.MIN_VAR_SHRINK * np.diag(np.diag(sample))
        try:
            ones      = np.ones(n)
            inv_cov   = np.linalg.inv(cov + 1e-6 * np.eye(n))
            raw_w     = inv_cov @ ones
            w         = np.maximum(raw_w, 0.0)              # long-only constraint
            w_sum     = w.sum()
            if w_sum < 1e-12:
                w = ones / n
            else:
                w /= w_sum
        except np.linalg.LinAlgError as exc:
            # 协方差奇异 → 退回等权。等权是合理兜底，但调用方以为拿到的是
            # 最小方差/IC 加权 —— 组合结论口径已经变了，必须说出来。
            logger.warning("[AlphaCombiner] 协方差奇异，权重退回**等权**（非最小方差）: %s", exc)
            w = np.ones(n) / n

        return {dsl: float(w[i]) for i, dsl in enumerate(dsls)}

    @staticmethod
    def _align(signals: list[pd.DataFrame]) -> list[pd.DataFrame]:
        """Reindex all signals to their common (index ∩ columns)."""
        if len(signals) == 1:
            return list(signals)
        idx  = signals[0].index
        cols = signals[0].columns
        for s in signals[1:]:
            idx  = idx.intersection(s.index)
            cols = cols.intersection(s.columns)
        return [s.reindex(index=idx, columns=cols) for s in signals]

    @staticmethod
    def _winsorize(signal: pd.DataFrame, k: float = 3.0) -> pd.DataFrame:
        """Clip cross-sectional outliers at ±k standard deviations per row."""
        arr = signal.to_numpy(dtype=float)
        mu  = np.nanmean(arr, axis=1, keepdims=True)
        sd  = np.nanstd(arr, axis=1, keepdims=True)
        arr = np.clip(arr, mu - k * sd, mu + k * sd)
        return pd.DataFrame(arr, index=signal.index, columns=signal.columns)
