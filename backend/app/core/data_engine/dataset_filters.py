"""
dataset_filters.py — Dynamic filter system applied after dataset selection.

Filter categories (8 total):
  market_cap       : mega_cap / large_cap / mid_cap / small_cap
  liquidity        : ultra_high / high / medium / low  (based on adv20)
  volatility       : high_vol / medium_vol / low_vol   (based on ts_std(ret,20))
  regime           : bull / bear / sideways             (based on SPY 200d MA)
  beta             : high_beta / low_beta               (60d rolling beta to SPY)
  correlation      : high_corr / low_corr               (60d rolling corr to SPY)
  momentum_regime  : strong_uptrend / strong_downtrend  (ts_delta(close,60))
  earnings_window  : pre_earnings / post_earnings       (yfinance calendar, US only)

Filters that need SPY (regime, beta, correlation) auto-fetch SPY data once.
Filters that need market_cap auto-fetch it per ticker (slow — use sparingly).
Earnings window only works for US stocks with yfinance support.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filter spec constants
# ---------------------------------------------------------------------------

# market_cap thresholds (USD / CNY depending on dataset)
_MARKET_CAP_THRESHOLDS = {
    "mega_cap":  200_000_000_000,
    "large_cap":  10_000_000_000,
    "mid_cap_lo":  2_000_000_000,
    "mid_cap_hi": 10_000_000_000,
    "small_cap":   2_000_000_000,
}

# adv20 = 20-day average dollar volume
_LIQUIDITY_THRESHOLDS = {
    "ultra_high": 100_000_000,
    "high":        50_000_000,
    "medium_lo":   10_000_000,
    "medium_hi":   50_000_000,
    "low":         10_000_000,
}

# 20-day return std-dev
_VOLATILITY_THRESHOLDS = {
    "high_vol":   0.04,
    "medium_lo":  0.02,
    "medium_hi":  0.04,
    "low_vol":    0.02,
}

# 60-day cumulative return threshold for momentum regime
_MOMENTUM_THRESHOLD = 0.15

# SPY regime
_SPY_MA_LONG  = 200
_SPY_MA_SHORT = 50
_SIDEWAYS_PCT = 0.03   # ±3% around 50d MA = sideways

# Beta
_BETA_HIGH = 1.2
_BETA_LOW  = 0.8

# Correlation
_CORR_HIGH = 0.7
_CORR_LOW  = 0.3

# Rolling windows
_BETA_WINDOW = 60
_CORR_WINDOW = 60


# ---------------------------------------------------------------------------
# FilterConfig  (dataclass, not Pydantic — kept in data layer)
# ---------------------------------------------------------------------------

@dataclass
class FilterConfig:
    """
    Specifies which filters to apply after dataset selection.
    Set a field to None to skip that filter category.
    """
    market_cap:      Optional[str] = None   # mega_cap|large_cap|mid_cap|small_cap
    liquidity:       Optional[str] = None   # ultra_high|high|medium|low
    volatility:      Optional[str] = None   # high_vol|medium_vol|low_vol
    regime:          Optional[str] = None   # bull|bear|sideways
    beta:            Optional[str] = None   # high_beta|low_beta
    correlation:     Optional[str] = None   # high_corr|low_corr
    momentum_regime: Optional[str] = None   # strong_uptrend|strong_downtrend
    earnings_window: Optional[str] = None   # pre_earnings|post_earnings

    # Internal: how many trailing days to use for metric computation
    lookback_days: int = 60

    def is_empty(self) -> bool:
        return all(
            getattr(self, f) is None
            for f in ("market_cap", "liquidity", "volatility", "regime",
                      "beta", "correlation", "momentum_regime", "earnings_window")
        )

    def needs_spy(self) -> bool:
        return any(getattr(self, f) is not None
                   for f in ("regime", "beta", "correlation"))

    def needs_market_cap(self) -> bool:
        return self.market_cap is not None

    def needs_earnings(self) -> bool:
        return self.earnings_window is not None


# ---------------------------------------------------------------------------
# FilterResult
# ---------------------------------------------------------------------------

@dataclass
class FilterResult:
    """Outcome of applying a FilterConfig to a dataset."""
    passed_tickers:  List[str]
    rejected_tickers: List[str]
    filter_metrics:  Dict[str, pd.Series]   # computed metrics at evaluation date
    regime_label:    Optional[str] = None   # detected regime (bull/bear/sideways)
    notes:           List[str] = field(default_factory=list)

    @property
    def n_passed(self) -> int:
        return len(self.passed_tickers)

    @property
    def n_total(self) -> int:
        return self.n_passed + len(self.rejected_tickers)

    def to_dict(self) -> dict:
        return {
            "passed":   self.passed_tickers,
            "rejected": self.rejected_tickers,
            "n_passed": self.n_passed,
            "n_total":  self.n_total,
            "regime":   self.regime_label,
            "notes":    self.notes,
        }


# ---------------------------------------------------------------------------
# Main filter engine
# ---------------------------------------------------------------------------

class DatasetFilterEngine:
    """
    Apply a FilterConfig to a loaded dataset (dict[field → T×N DataFrame]).

    Usage::

        engine = DatasetFilterEngine()
        result = engine.apply(data, FilterConfig(liquidity="high", volatility="high_vol"))
        filtered_data = engine.slice_data(data, result.passed_tickers)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(
        self,
        data:      Dict[str, pd.DataFrame],
        config:    FilterConfig,
        region:    str = "US",
        market_cap_data: Optional[Dict[str, float]] = None,
    ) -> FilterResult:
        """
        Compute filter metrics and return which tickers pass ALL active filters.

        Parameters
        ----------
        data          : dict[field → DataFrame(T × N)]
        config        : which filters to apply
        region        : "US" | "China" | "HongKong" | "Global" (affects market cap, earnings)
        market_cap_data: pre-fetched market_cap dict (optional; auto-fetched if None and needed)

        Returns
        -------
        FilterResult
        """
        if config.is_empty():
            all_tickers = list(next(iter(data.values())).columns)
            return FilterResult(
                passed_tickers  = all_tickers,
                rejected_tickers = [],
                filter_metrics  = {},
            )

        all_tickers: List[str] = list(next(iter(data.values())).columns)
        passing: Set[str]      = set(all_tickers)
        metrics: Dict[str, pd.Series] = {}
        notes:   List[str] = []
        regime_label: Optional[str] = None

        close   = data["close"]
        volume  = data.get("volume", pd.DataFrame(np.nan, index=close.index, columns=close.columns))
        returns = data.get("returns", np.log(close / close.shift(1)))

        # ── 1. Liquidity ─────────────────────────────────────────────────
        if config.liquidity is not None:
            adv20 = (close * volume).rolling(20, min_periods=10).mean()
            adv20_last = adv20.iloc[-1]
            metrics["adv20"] = adv20_last
            passed = self._apply_liquidity(adv20_last, config.liquidity)
            passing &= passed
            notes.append(f"liquidity={config.liquidity}: {len(passed)}/{len(all_tickers)} pass")

        # ── 2. Volatility ────────────────────────────────────────────────
        if config.volatility is not None:
            vol20 = returns.rolling(20, min_periods=10).std()
            vol20_last = vol20.iloc[-1]
            metrics["vol20"] = vol20_last
            passed = self._apply_volatility(vol20_last, config.volatility)
            passing &= passed
            notes.append(f"volatility={config.volatility}: {len(passed)}/{len(all_tickers)} pass")

        # ── 3. Momentum regime ───────────────────────────────────────────
        if config.momentum_regime is not None:
            mom60 = close.pct_change(60)
            mom60_last = mom60.iloc[-1]
            metrics["mom60"] = mom60_last
            passed = self._apply_momentum(mom60_last, config.momentum_regime)
            passing &= passed
            notes.append(f"momentum_regime={config.momentum_regime}: {len(passed)}/{len(all_tickers)} pass")

        # ── 4. Regime / Beta / Correlation (need SPY) ────────────────────
        if config.needs_spy():
            spy_returns = self._fetch_spy_returns(close.index)

            if config.regime is not None:
                spy_close = (1 + spy_returns.fillna(0)).cumprod() * 100
                regime_label = self._detect_regime(spy_close)
                target = config.regime
                if regime_label != target:
                    # Regime filter is dataset-level: if regime doesn't match, no tickers pass
                    passing.clear()
                    notes.append(
                        f"regime={target}: BLOCKED (detected '{regime_label}')"
                    )
                else:
                    notes.append(f"regime={target}: OK (current regime = '{regime_label}')")

            if config.beta is not None and spy_returns is not None:
                beta_series = self._compute_beta(returns, spy_returns)
                metrics["beta"] = beta_series
                passed = self._apply_beta(beta_series, config.beta)
                passing &= passed
                notes.append(f"beta={config.beta}: {len(passed)}/{len(all_tickers)} pass")

            if config.correlation is not None and spy_returns is not None:
                corr_series = self._compute_corr(returns, spy_returns)
                metrics["corr_spy"] = corr_series
                passed = self._apply_corr(corr_series, config.correlation)
                passing &= passed
                notes.append(f"correlation={config.correlation}: {len(passed)}/{len(all_tickers)} pass")

        # ── 5. Market cap ────────────────────────────────────────────────
        if config.market_cap is not None:
            if market_cap_data is None:
                market_cap_data = self._fetch_market_cap(all_tickers, region)
            mc = pd.Series(market_cap_data).reindex(all_tickers)
            metrics["market_cap"] = mc
            passed = self._apply_market_cap(mc, config.market_cap)
            passing &= passed
            notes.append(f"market_cap={config.market_cap}: {len(passed)}/{len(all_tickers)} pass")

        # ── 6. Earnings window ───────────────────────────────────────────
        if config.earnings_window is not None:
            if region in ("US", "HongKong"):
                passed = self._apply_earnings(list(passing), config.earnings_window)
                before = len(passing)
                passing &= passed
                notes.append(
                    f"earnings_window={config.earnings_window}: {len(passing)}/{before} pass"
                )
            else:
                notes.append(
                    f"earnings_window filter skipped for region='{region}' (US/HK only)"
                )

        passed_list   = [t for t in all_tickers if t in passing]
        rejected_list = [t for t in all_tickers if t not in passing]

        return FilterResult(
            passed_tickers   = passed_list,
            rejected_tickers = rejected_list,
            filter_metrics   = metrics,
            regime_label     = regime_label,
            notes            = notes,
        )

    @staticmethod
    def slice_data(
        data:    Dict[str, pd.DataFrame],
        tickers: List[str],
    ) -> Dict[str, pd.DataFrame]:
        """Return a view of data restricted to ``tickers`` columns."""
        result = {}
        for field, df in data.items():
            available = [t for t in tickers if t in df.columns]
            result[field] = df[available].copy()
        return result

    # ------------------------------------------------------------------
    # Per-filter implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_liquidity(adv20: pd.Series, level: str) -> Set[str]:
        if level == "ultra_high":
            mask = adv20 > _LIQUIDITY_THRESHOLDS["ultra_high"]
        elif level == "high":
            mask = adv20 > _LIQUIDITY_THRESHOLDS["high"]
        elif level == "medium":
            mask = (adv20 > _LIQUIDITY_THRESHOLDS["medium_lo"]) & \
                   (adv20 <= _LIQUIDITY_THRESHOLDS["medium_hi"])
        elif level == "low":
            mask = adv20 <= _LIQUIDITY_THRESHOLDS["low"]
        else:
            raise ValueError(f"Unknown liquidity level: '{level}'")
        return set(adv20[mask.fillna(False)].index)

    @staticmethod
    def _apply_volatility(vol20: pd.Series, level: str) -> Set[str]:
        if level == "high_vol":
            mask = vol20 > _VOLATILITY_THRESHOLDS["high_vol"]
        elif level == "medium_vol":
            mask = (vol20 > _VOLATILITY_THRESHOLDS["medium_lo"]) & \
                   (vol20 <= _VOLATILITY_THRESHOLDS["medium_hi"])
        elif level == "low_vol":
            mask = vol20 <= _VOLATILITY_THRESHOLDS["low_vol"]
        else:
            raise ValueError(f"Unknown volatility level: '{level}'")
        return set(vol20[mask.fillna(False)].index)

    @staticmethod
    def _apply_momentum(mom60: pd.Series, level: str) -> Set[str]:
        if level == "strong_uptrend":
            mask = mom60 > _MOMENTUM_THRESHOLD
        elif level == "strong_downtrend":
            mask = mom60 < -_MOMENTUM_THRESHOLD
        else:
            raise ValueError(f"Unknown momentum_regime: '{level}'")
        return set(mom60[mask.fillna(False)].index)

    @staticmethod
    def _apply_market_cap(mc: pd.Series, level: str) -> Set[str]:
        if level == "mega_cap":
            mask = mc > _MARKET_CAP_THRESHOLDS["mega_cap"]
        elif level == "large_cap":
            mask = mc > _MARKET_CAP_THRESHOLDS["large_cap"]
        elif level == "mid_cap":
            mask = (mc > _MARKET_CAP_THRESHOLDS["mid_cap_lo"]) & \
                   (mc <= _MARKET_CAP_THRESHOLDS["mid_cap_hi"])
        elif level == "small_cap":
            mask = mc <= _MARKET_CAP_THRESHOLDS["small_cap"]
        else:
            raise ValueError(f"Unknown market_cap level: '{level}'")
        return set(mc[mask.fillna(False)].index)

    @staticmethod
    def _apply_beta(beta: pd.Series, level: str) -> Set[str]:
        if level == "high_beta":
            mask = beta > _BETA_HIGH
        elif level == "low_beta":
            mask = beta < _BETA_LOW
        else:
            raise ValueError(f"Unknown beta level: '{level}'")
        return set(beta[mask.fillna(False)].index)

    @staticmethod
    def _apply_corr(corr: pd.Series, level: str) -> Set[str]:
        if level == "high_corr":
            mask = corr > _CORR_HIGH
        elif level == "low_corr":
            mask = corr < _CORR_LOW
        else:
            raise ValueError(f"Unknown correlation level: '{level}'")
        return set(corr[mask.fillna(False)].index)

    # ------------------------------------------------------------------
    # Regime detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_regime(spy_close: pd.Series) -> str:
        """Return 'bull', 'bear', or 'sideways' based on SPY moving averages."""
        if len(spy_close) < _SPY_MA_LONG:
            return "sideways"
        ma200 = spy_close.rolling(_SPY_MA_LONG).mean().iloc[-1]
        ma50  = spy_close.rolling(_SPY_MA_SHORT).mean().iloc[-1]
        last  = spy_close.iloc[-1]
        if last > ma200:
            pct_from_50 = abs(last - ma50) / ma50
            return "sideways" if pct_from_50 < _SIDEWAYS_PCT else "bull"
        return "bear"

    # ------------------------------------------------------------------
    # Rolling beta / correlation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_beta(
        returns:     pd.DataFrame,
        spy_returns: pd.Series,
    ) -> pd.Series:
        """
        Compute 60-day rolling beta of each asset to SPY.
        Returns a pd.Series (asset → beta at last date).
        """
        spy = spy_returns.reindex(returns.index).fillna(0)
        spy_var = spy.rolling(_BETA_WINDOW, min_periods=20).var().iloc[-1]
        if spy_var < 1e-12:
            return pd.Series(np.nan, index=returns.columns)
        result = {}
        for col in returns.columns:
            asset = returns[col].fillna(0)
            cov = asset.rolling(_BETA_WINDOW, min_periods=20).cov(spy).iloc[-1]
            result[col] = float(cov) / float(spy_var)
        return pd.Series(result)

    @staticmethod
    def _compute_corr(
        returns:     pd.DataFrame,
        spy_returns: pd.Series,
    ) -> pd.Series:
        """60-day rolling correlation of each asset to SPY (last date)."""
        spy = spy_returns.reindex(returns.index).fillna(0)
        result = {}
        for col in returns.columns:
            asset = returns[col].fillna(0)
            corr_series = asset.rolling(_CORR_WINDOW, min_periods=20).corr(spy)
            result[col] = corr_series.iloc[-1] if len(corr_series) > 0 else np.nan
        return pd.Series(result)

    # ------------------------------------------------------------------
    # External data fetchers
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_spy_returns(index: pd.DatetimeIndex) -> pd.Series:
        """Download SPY daily returns aligned to ``index``."""
        try:
            import yfinance as yf
            start = str(index[0].date())
            end   = str(index[-1].date())
            spy   = yf.download("SPY", start=start, end=end,
                                 auto_adjust=True, progress=False)
            if spy.empty:
                return pd.Series(dtype=float)
            close = spy["Close"] if "Close" in spy.columns else spy.iloc[:, 0]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            rets = np.log(close / close.shift(1))
            return rets.reindex(index)
        except Exception as exc:
            logger.warning("Could not fetch SPY for regime/beta filter: %s", exc)
            return pd.Series(dtype=float)

    @staticmethod
    def _fetch_market_cap(tickers: List[str], region: str) -> Dict[str, float]:
        """
        Fetch market cap for all tickers.
          US / HK  → yfinance fast_info
          China    → akshare (slow, rate-limited)
          Crypto   → not supported (returns NaN)
        """
        result: Dict[str, float] = {}

        if region == "Global":
            # Crypto: skip market_cap
            logger.info("Market cap filter not supported for Crypto datasets; all pass.")
            return {t: np.nan for t in tickers}

        if region in ("US", "HongKong"):
            try:
                import yfinance as yf
                for t in tickers:
                    try:
                        fi = yf.Ticker(t).fast_info
                        result[t] = float(getattr(fi, "market_cap", np.nan) or np.nan)
                    except Exception:
                        result[t] = np.nan
            except ImportError:
                result = {t: np.nan for t in tickers}

        elif region == "China":
            try:
                from .providers.akshare_provider import AkshareProvider
                result = AkshareProvider.fetch_market_cap(tickers)
            except Exception as exc:
                logger.warning("akshare market cap fetch failed: %s", exc)
                result = {t: np.nan for t in tickers}

        return result

    @staticmethod
    def _apply_earnings(tickers: List[str], level: str) -> Set[str]:
        """
        Filter tickers based on proximity to earnings date (US/HK only).
        Returns set of tickers that satisfy the earnings window condition.
        """
        try:
            import yfinance as yf
            today  = pd.Timestamp.today()
            passed = set()
            for t in tickers:
                try:
                    cal = yf.Ticker(t).calendar
                    if cal is None or cal.empty:
                        continue
                    # calendar columns: Earnings Date, ...
                    earn_col = [c for c in cal.columns if "Earnings" in str(c)]
                    if not earn_col:
                        continue
                    earn_date = pd.Timestamp(cal[earn_col[0]].iloc[0])
                    delta = (earn_date - today).days
                    if level == "pre_earnings"  and 0 <= delta <= 5:
                        passed.add(t)
                    elif level == "post_earnings" and -5 <= delta < 0:
                        passed.add(t)
                except Exception:
                    pass
            return passed
        except ImportError:
            return set(tickers)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def apply_filters(
    data:   Dict[str, pd.DataFrame],
    config: FilterConfig,
    region: str = "US",
    market_cap_data: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, pd.DataFrame], FilterResult]:
    """
    Apply ``config`` filters to ``data`` and return (filtered_data, result).

    filtered_data contains only tickers that passed ALL active filters.
    """
    engine = DatasetFilterEngine()
    result = engine.apply(data, config, region=region, market_cap_data=market_cap_data)
    filtered = engine.slice_data(data, result.passed_tickers)
    return filtered, result


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

VALID_FILTER_VALUES: Dict[str, List[str]] = {
    "market_cap":      ["mega_cap", "large_cap", "mid_cap", "small_cap"],
    "liquidity":       ["ultra_high", "high", "medium", "low"],
    "volatility":      ["high_vol", "medium_vol", "low_vol"],
    "regime":          ["bull", "bear", "sideways"],
    "beta":            ["high_beta", "low_beta"],
    "correlation":     ["high_corr", "low_corr"],
    "momentum_regime": ["strong_uptrend", "strong_downtrend"],
    "earnings_window": ["pre_earnings", "post_earnings"],
}


def validate_filter_config(filters: Dict[str, str]) -> List[str]:
    """Return list of validation error strings (empty = valid)."""
    errors = []
    for key, val in filters.items():
        if key not in VALID_FILTER_VALUES:
            errors.append(f"Unknown filter category '{key}'. Valid: {list(VALID_FILTER_VALUES)}")
        elif val not in VALID_FILTER_VALUES[key]:
            errors.append(
                f"Invalid value '{val}' for filter '{key}'. "
                f"Valid: {VALID_FILTER_VALUES[key]}"
            )
    return errors
