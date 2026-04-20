"""
alpha_pool.py — AlphaPool: persistent in-process population memory.

Maintains a ranked collection of evaluated alpha candidates with:
  - Deduplication:        rejects DSL strings already seen
  - Diversity filter:     rejects alphas whose time-series signal is too
                          correlated (|ρ| >= corr_threshold) with any existing entry
  - Capacity management:  prunes lowest-fitness entries when over max_size

The signal_vec is the cross-sectional mean of the rank signal across IS dates,
giving a 1-D fingerprint for fast correlation comparison.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PoolEntry
# ---------------------------------------------------------------------------

@dataclass
class PoolEntry:
    """A single evaluated alpha in the pool."""

    dsl:               str
    fitness:           float
    sharpe_is:         float
    sharpe_oos:        float
    turnover:          float
    overfitting_score: float
    generation:        int
    signal_vec:        Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "dsl":               self.dsl,
            "fitness":           round(self.fitness,          4),
            "sharpe_is":         round(self.sharpe_is,        4),
            "sharpe_oos":        round(self.sharpe_oos,       4),
            "turnover":          round(self.turnover,         4),
            "overfitting_score": round(self.overfitting_score, 4),
            "generation":        self.generation,
        }


# ---------------------------------------------------------------------------
# AlphaPool
# ---------------------------------------------------------------------------

class AlphaPool:
    """
    Ranked, diversity-controlled alpha candidate pool.

    Parameters
    ----------
    max_size       : maximum entries retained (worst pruned on overflow)
    corr_threshold : |ρ| >= this → new entry rejected as too similar
    """

    def __init__(
        self,
        max_size:       int   = 200,
        corr_threshold: float = 0.90,
    ) -> None:
        self._max_size       = max_size
        self._corr_threshold = corr_threshold
        self._entries:  List[PoolEntry] = []
        self._seen_dsls: set            = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, entry: PoolEntry) -> bool:
        """
        Attempt to add entry. Returns True if accepted, False if rejected.

        Rejection criteria:
          1. Identical DSL already in pool
          2. Signal correlation with any existing entry >= corr_threshold
        """
        if entry.dsl in self._seen_dsls:
            return False

        if entry.signal_vec is not None and self._is_too_correlated(entry.signal_vec):
            logger.debug(
                "AlphaPool: rejected '%s' (signal corr >= %.2f)",
                entry.dsl[:60], self._corr_threshold,
            )
            return False

        self._entries.append(entry)
        self._seen_dsls.add(entry.dsl)
        self._prune()
        return True

    def top_k(self, k: int) -> List[PoolEntry]:
        """Return top-k entries sorted by fitness (descending)."""
        return sorted(self._entries, key=lambda e: e.fitness, reverse=True)[:k]

    def best(self) -> Optional[PoolEntry]:
        """Return the highest-fitness entry, or None if empty."""
        if not self._entries:
            return None
        return max(self._entries, key=lambda e: e.fitness)

    def all_entries(self) -> List[PoolEntry]:
        return list(self._entries)

    def population_diagnostics(self) -> Dict[str, float]:
        """Aggregate metrics for adaptive mutation weighting (Phase 8)."""
        if not self._entries:
            return {"mean_sharpe_oos": 0.0, "mean_turnover": 1.0, "mean_overfit": 0.0}
        return {
            "mean_sharpe_oos": float(np.mean([e.sharpe_oos        for e in self._entries])),
            "mean_turnover":   float(np.mean([e.turnover          for e in self._entries])),
            "mean_overfit":    float(np.mean([e.overfitting_score for e in self._entries])),
        }

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_too_correlated(self, new_vec: np.ndarray) -> bool:
        for entry in self._entries:
            if entry.signal_vec is None:
                continue
            n, m = len(entry.signal_vec), len(new_vec)
            if n != m:
                continue
            mask = ~(np.isnan(entry.signal_vec) | np.isnan(new_vec))
            if mask.sum() < 10:
                continue
            corr = float(np.corrcoef(entry.signal_vec[mask], new_vec[mask])[0, 1])
            if not np.isnan(corr) and abs(corr) >= self._corr_threshold:
                return True
        return False

    def _prune(self) -> None:
        """Discard lowest-fitness entries when over capacity."""
        if len(self._entries) > self._max_size:
            self._entries.sort(key=lambda e: e.fitness, reverse=True)
            removed = self._entries[self._max_size:]
            self._entries = self._entries[: self._max_size]
            for r in removed:
                self._seen_dsls.discard(r.dsl)
