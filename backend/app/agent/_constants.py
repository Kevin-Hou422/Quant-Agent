"""
_constants.py — shared constants for the QuantAgent module.

Centralised here so every sub-module imports from one place; avoids
scattered magic-number duplication.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Thresholds & defaults
# ---------------------------------------------------------------------------

_MAX_CORRECTION_ROUNDS = 2
_OVERFIT_THRESHOLD     = 0.50
_MIN_OOS_SHARPE        = 0.20
_DEFAULT_N_TICKERS     = 20
_DEFAULT_N_DAYS        = 252
_DEFAULT_OOS_RATIO     = 0.30
_DEFAULT_N_TRIALS      = 15

# ---------------------------------------------------------------------------
# Valid field / operator lists (used in LLM prompts to prevent hallucination)
# ---------------------------------------------------------------------------

_VALID_FIELDS = "close, open, high, low, volume, vwap, returns"

_VALID_OPS = (
    "rank, zscore, scale, "
    "ts_mean, ts_std, ts_delta, ts_delay, ts_max, ts_min, ts_rank, "
    "ts_decay_linear, ts_corr, "
    "log, abs, sqrt, sign, signed_power, if_else"
)

# ---------------------------------------------------------------------------
# 基本面字段（Phase 10）—— 只在数据集真的带这些字段时才告诉 LLM
# ---------------------------------------------------------------------------

#: 字段 → 一句话说明（写进提示词）。顺序即提示词里的顺序。
_FUNDAMENTAL_FIELD_NOTES: Dict[str, str] = {
    "market_cap":          "market capitalisation in USD (changes daily)",
    "earnings_yield":      "trailing-12-month net income / market cap (E/P, changes daily)",
    "book_to_price":       "book equity / market cap (B/P, changes daily)",
    "roe":                 "trailing net income / book equity (quarterly step)",
    "gross_margin":        "trailing gross profit / revenue (quarterly step)",
    "asset_growth":        "total assets vs. one year earlier (quarterly step)",
    "sue":                 "standardized earnings surprise (quarterly step)",
    "days_since_earnings": "trading days since the last earnings release (counter)",
}

_FUNDAMENTAL_GUIDANCE = (
    "Fundamental fields are point-in-time (visible only after the filing date). Rules:\n"
    "- rank them with fund_rank(x), not rank(x): it refuses to rank a day on which fewer than "
    "half the stocks have a value.\n"
    "- roe, gross_margin, asset_growth, sue and days_since_earnings are constant between filings: "
    "never apply ts_std / ts_var / ts_zscore / ts_rank / ts_corr / ts_cov to them with a window "
    "below 126 (the expression is rejected). ts_mean and ts_delta are fine."
)

_FUNDAMENTAL_FEW_SHOT = (
    "- hypothesis='cheap stocks outperform' "
    '→ {"dsl": "fund_rank(earnings_yield)", "explanation": "Earnings-yield rank."}\n'
    "- hypothesis='post earnings drift' "
    '→ {"dsl": "fund_rank(trade_when(days_since_earnings < 60, sue))", '
    '"explanation": "Earnings surprise held for 60 sessions."}'
)

#: 无 LLM 路径：关键词 → (表达式, 需要的字段)。只有数据集带齐所需字段时才会命中。
_FALLBACK_FUNDAMENTAL_DSL_MAP: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "earnings yield": ("fund_rank(earnings_yield)", ("earnings_yield",)),
    "book":           ("fund_rank(book_to_price)", ("book_to_price",)),
    "value":          ("fund_rank(earnings_yield) + fund_rank(book_to_price)",
                       ("earnings_yield", "book_to_price")),
    "cheap":          ("fund_rank(earnings_yield)", ("earnings_yield",)),
    "估值":            ("fund_rank(earnings_yield) + fund_rank(book_to_price)",
                       ("earnings_yield", "book_to_price")),
    "价值":            ("fund_rank(earnings_yield) + fund_rank(book_to_price)",
                       ("earnings_yield", "book_to_price")),
    "earnings":       ("fund_rank(sue)", ("sue",)),
    "surprise":       ("fund_rank(sue)", ("sue",)),
    "财报":            ("fund_rank(sue)", ("sue",)),
    "roe":            ("fund_rank(roe)", ("roe",)),
    "profitab":       ("fund_rank(roe) + fund_rank(gross_margin)", ("roe", "gross_margin")),
    "quality":        ("fund_rank(roe) + fund_rank(gross_margin)", ("roe", "gross_margin")),
    "盈利能力":         ("fund_rank(roe) + fund_rank(gross_margin)", ("roe", "gross_margin")),
    "small cap":      ("fund_rank(-log(market_cap))", ("market_cap",)),
    "market cap":     ("fund_rank(-log(market_cap))", ("market_cap",)),
    "市值":            ("fund_rank(-log(market_cap))", ("market_cap",)),
}


def prompt_field_lines(dataset_fields) -> Tuple[str, str, str, str]:
    """
    按数据集**实际有的**字段生成提示词片段：(字段行, 算子行, 额外规则, 额外示例)。
    没有基本面字段时与原来逐字相同 —— 不向 LLM 宣传一个算不出来的字段。
    """
    have = [f for f in _FUNDAMENTAL_FIELD_NOTES if f in set(dataset_fields)]
    if not have:
        return _VALID_FIELDS, _VALID_OPS, "", ""
    listed = "; ".join(f"{f} = {_FUNDAMENTAL_FIELD_NOTES[f]}" for f in have)
    return (f"{_VALID_FIELDS}\nFundamental fields: {listed}",
            f"{_VALID_OPS}, fund_rank, trade_when",
            _FUNDAMENTAL_GUIDANCE, _FUNDAMENTAL_FEW_SHOT)


# ---------------------------------------------------------------------------
# Fallback DSL keyword→expression mapping (no-LLM path)
# ---------------------------------------------------------------------------

_FALLBACK_DSL_MAP: Dict[str, str] = {
    "volume":     "rank(ts_delta(log(volume), 5))",
    "vwap":       "rank(ts_delta(vwap, 5))",
    "momentum":   "rank(ts_mean(returns, 10))",
    "reversal":   "rank(-ts_delta(close, 1))",
    "volatility": "rank(-ts_std(returns, 20))",
    "price":      "rank(ts_delta(log(close), 5))",
    "default":    "rank(ts_delta(log(close), 5))",
}

# ---------------------------------------------------------------------------
# Structural mutation templates (no-LLM path; {dsl} is the current formula)
# ---------------------------------------------------------------------------

_FALLBACK_MUTATION_TEMPLATES: List[Tuple[str, str]] = [
    ("rank_wrap",    "rank({dsl})"),
    ("vol_filter",   "rank({dsl}) * rank(-ts_std(returns, 20))"),
    ("decay_smooth", "rank(ts_decay_linear(ts_mean(returns, 10), 5))"),
    ("zscore_wrap",  "zscore(ts_delta(log(close), 5))"),
    ("reversal_mix", "rank(-ts_delta(close, 3))"),
    ("momentum_mix", "rank(ts_mean(returns, 5))"),
]
