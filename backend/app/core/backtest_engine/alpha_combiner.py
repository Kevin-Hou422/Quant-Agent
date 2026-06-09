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
        s, r = sig[t], ret[t]
        mask = ~(np.isnan(s) | np.isnan(r))
        if mask.sum() < 5:
            continue
        rs = np.argsort(np.argsort(s[mask])).astype(float)
        rr = np.argsort(np.argsort(r[mask])).astype(float)
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
            return self._min_variance_weights(signals)

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

    def _min_variance_weights(
        self,
        signals: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """
        Minimum-variance weights: minimise the variance of the composite signal.

        Uses closed-form minimum-variance portfolio on the signal correlation matrix.
        Falls back to equal-weight if fewer than 2 signals or matrix is singular.
        """
        dsls = list(signals.keys())
        n    = len(dsls)
        if n < 2:
            return {dsls[0]: 1.0}

        # Stack cross-sectional means of each signal as proxies for their time series
        vecs: list[np.ndarray] = []
        for sig in signals.values():
            arr = sig.to_numpy(dtype=float)
            vec = np.nanmean(arr, axis=1)   # (T,) cross-sectional mean
            vecs.append(vec)

        mat  = np.stack(vecs, axis=1)  # (T, n)
        # Drop rows with any NaN
        mask = ~np.isnan(mat).any(axis=1)
        if mask.sum() < n + 1:
            # Insufficient data → equal weight
            return {dsl: 1.0 / n for dsl in dsls}

        cov  = np.cov(mat[mask].T)                          # (n, n)
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
        except np.linalg.LinAlgError:
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
