"""
Phase 10.1 / 10.4 — DSL 字段类别、阶梯字段校验器、fund_rank。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.alpha_engine import fields as F
from app.core.alpha_engine.dsl_executor import Executor
from app.core.alpha_engine.fast_ops import FUND_RANK_MIN_COVERAGE, cs_fund_rank, cs_rank
from app.core.alpha_engine.financial_interpreter import FACTOR_FAMILIES, FinancialInterpreter
from app.core.alpha_engine.parser import Parser
from app.core.alpha_engine.validator import AlphaValidator, SparseFieldValidator, ValidationError
from app.core.data_engine.fundamentals import panels


def _errors(dsl: str):
    return SparseFieldValidator().collect(Parser().parse(dsl))


# ---------------------------------------------------------------------------
# 字段类别
# ---------------------------------------------------------------------------

def test_dsl_field_names_match_what_the_data_layer_produces():
    """两边各有一份字段名单 —— 这里钉住它们是同一份。"""
    assert F.FUNDAMENTAL_FIELDS == set(panels.ALL_FIELDS)
    assert F.VALUATION_FIELDS == set(panels.VALUATION_FIELDS)
    assert F.EVENT_FIELDS == set(panels.EVENT_FIELDS)
    assert F.QUARTERLY_FIELDS == set(panels.FUNDAMENTAL_FIELDS) - set(panels.VALUATION_FIELDS)


def test_field_classes_are_disjoint_and_classified():
    groups = [F.PRICE_FIELDS, F.VALUATION_FIELDS, F.QUARTERLY_FIELDS, F.EVENT_FIELDS]
    assert sum(len(g) for g in groups) == len(set().union(*groups))
    assert F.field_class("close") == "price"
    assert F.field_class("earnings_yield") == "valuation"
    assert F.field_class("roe") == "quarterly"
    assert F.field_class("days_since_earnings") == "event"
    assert F.field_class("sector") == "other"
    assert F.STEP_FIELDS == F.QUARTERLY_FIELDS | F.EVENT_FIELDS


# ---------------------------------------------------------------------------
# SparseFieldValidator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dsl", [
    "ts_std(roe, 20)",
    "rank(close / ts_std(gross_margin, 60))",
    "ts_zscore(sue, 20)",
    "ts_var(asset_growth, 10)",
    "ts_rank(roe, 60)",
    "ts_skew(roe, 40)", "ts_kurt(roe, 40)", "ts_entropy(roe, 40)",
    "ts_std(days_since_earnings, 20)",
    "ts_std(roe * gross_margin, 20)",          # 两个阶梯相乘还是阶梯
    "ts_std(rank(roe), 20)",                   # 套一层截面算子，叶子仍然全是阶梯字段
    "ts_corr(roe, gross_margin, 60)",
    "ts_corr(close, roe, 60)",                 # 任一侧是纯阶梯都不行
    "ts_cov(sue, volume, 20)",
])
def test_short_window_dispersion_on_step_fields_is_rejected(dsl):
    errs = _errors(dsl)
    assert len(errs) == 1 and "only" in errs[0] and "126" in errs[0]
    with pytest.raises(ValidationError):
        AlphaValidator().validate(Parser().parse(dsl))


@pytest.mark.parametrize("dsl", [
    "ts_std(roe, 126)", "ts_std(roe, 252)",            # 窗口够长：里面至少有两次财报
    "ts_mean(roe, 20)", "ts_delta(roe, 63)", "ts_delay(sue, 5)",   # 水平类运算
    "ts_max(gross_margin, 20)", "ts_sum(sue, 10)", "ts_decay_linear(roe, 10)",
    "ts_std(earnings_yield, 20)",                      # 估值字段：分母是市值，每天变
    "ts_zscore(book_to_price, 60)",
    "ts_std(roe * returns, 20)",                       # 混了日频字段 → 每天都在变
    "ts_corr(close, volume, 20)", "ts_std(returns, 5)",
    "fund_rank(roe) + rank(ts_std(returns, 20))",
    "rank(roe)", "roe",
    "ts_std(5, 20)",                                   # 没有字段叶子
])
def test_meaningful_expressions_pass(dsl):
    assert _errors(dsl) == []
    AlphaValidator().validate(Parser().parse(dsl))


def test_window_boundary_is_exactly_min_window():
    w = SparseFieldValidator.MIN_WINDOW
    assert _errors(f"ts_std(roe, {w - 1})") != []
    assert _errors(f"ts_std(roe, {w})") == []


def test_error_names_the_operator_window_and_fields():
    (err,) = _errors("ts_corr(sue, roe, 30)")
    assert "ts_corr" in err and "window=30" in err and "roe" in err and "sue" in err


def test_every_offending_node_is_reported():
    errs = _errors("ts_std(roe, 20) + ts_var(sue, 10)")
    assert len(errs) == 2


def test_price_only_expressions_are_untouched_by_the_new_validator():
    """现有的全部量价种子都必须照常通过 —— 新校验器只管阶梯字段。"""
    from app.core.gp_engine.gp_engine import _SEED_DSLS
    v, sparse = AlphaValidator(), SparseFieldValidator()
    assert len(_SEED_DSLS) > 40
    rejected = [dsl for dsl in _SEED_DSLS if not v.is_valid(Parser().parse(dsl))]
    assert rejected == []
    assert [dsl for dsl in _SEED_DSLS if sparse.collect(Parser().parse(dsl))] == []


# ---------------------------------------------------------------------------
# fund_rank
# ---------------------------------------------------------------------------

def test_fund_rank_equals_rank_when_coverage_is_full():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 9))
    np.testing.assert_array_equal(cs_fund_rank(x), cs_rank(x))


def test_fund_rank_blanks_rows_below_the_coverage_threshold():
    x = np.array([
        [1.0, 2.0, 3.0, 4.0],            # 4/4
        [1.0, 2.0, np.nan, np.nan],      # 2/4 = 0.5 → 恰好达标
        [1.0, np.nan, np.nan, np.nan],   # 1/4 → 整行 NaN
        [np.nan] * 4,
    ])
    out = cs_fund_rank(x)
    assert out[0].tolist() == [0.0, pytest.approx(1 / 3), pytest.approx(2 / 3), 1.0]
    assert out[1, :2].tolist() == [0.0, 1.0] and np.isnan(out[1, 2:]).all()
    assert np.isnan(out[2]).all() and np.isnan(out[3]).all()
    assert FUND_RANK_MIN_COVERAGE == 0.5


def test_fund_rank_threshold_is_configurable_and_does_not_mutate_input():
    x = np.array([[1.0, 2.0, np.nan, np.nan, np.nan]])     # 2/5 = 0.4
    before = x.copy()
    assert np.isnan(cs_fund_rank(x)).all()
    assert cs_fund_rank(x, min_coverage=0.4)[0, :2].tolist() == [0.0, 1.0]
    np.testing.assert_array_equal(x, before)


def test_plain_rank_would_have_ranked_three_names_as_if_it_were_a_full_cross_section():
    """fund_rank 存在的理由：rank 在只有 3/10 有值的那天照样给出 0 / 0.5 / 1。"""
    x = np.full((1, 10), np.nan)
    x[0, :3] = [5.0, 1.0, 3.0]
    assert cs_rank(x)[0, :3].tolist() == [1.0, 0.0, 0.5]
    assert np.isnan(cs_fund_rank(x)).all()


def test_fund_rank_parses_roundtrips_and_executes():
    idx = pd.bdate_range("2024-01-02", periods=3)
    cols = list("ABCD")
    ey = pd.DataFrame([[0.01, 0.05, 0.03, 0.02], [0.01, np.nan, np.nan, np.nan], [0.04, 0.03, 0.02, 0.01]],
                      index=idx, columns=cols)
    node = Parser().parse("fund_rank(earnings_yield)")
    assert repr(node) == "fund_rank(earnings_yield)"
    assert repr(Parser().parse(repr(node))) == repr(node)
    out = Executor().run(node, {"earnings_yield": ey, "close": ey * 0 + 1.0})
    assert out.iloc[0].tolist() == [0.0, 1.0, pytest.approx(2 / 3), pytest.approx(1 / 3)]
    assert out.iloc[1].isna().all()
    assert out.iloc[2].tolist() == [1.0, pytest.approx(2 / 3), pytest.approx(1 / 3), 0.0]


def test_fund_rank_takes_exactly_one_argument():
    from app.core.alpha_engine.parser import ParseError
    with pytest.raises(ParseError, match="exactly 1"):
        Parser().parse("fund_rank(roe, 5)")


def test_fund_rank_counts_as_a_scale_stable_root():
    from app.core.gp_engine.fitness import _is_scale_stable
    assert _is_scale_stable(Parser().parse("fund_rank(roe)"))
    assert not _is_scale_stable(Parser().parse("roe"))
    assert FinancialInterpreter().interpret("fund_rank(roe)").is_normalized


def test_fund_rank_is_not_in_the_random_cs_replacement_pool():
    """在量价子树上它与 rank 等价 —— 进了随机池只会稀释搜索，还会改掉既有 seed 的随机序列。"""
    from app.core.gp_engine import mutations
    assert "fund_rank" not in mutations._CS_LIST
    assert "fund_rank" not in mutations._WRAP_CS_OPS


def test_missing_fundamental_field_is_a_loud_error():
    """数据集没开基本面 → 引用基本面字段的表达式必须报错，而不是算出一列 NaN。"""
    idx = pd.bdate_range("2024-01-02", periods=3)
    close = pd.DataFrame(1.0, index=idx, columns=list("AB"))
    with pytest.raises(KeyError, match="earnings_yield"):
        Executor().run_expr("fund_rank(earnings_yield)", {"close": close})


# ---------------------------------------------------------------------------
# 家族归类
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dsl,family", [
    ("fund_rank(earnings_yield)", "value"),
    ("fund_rank(earnings_yield) + fund_rank(book_to_price)", "value"),
    ("fund_rank(roe)", "profitability"),
    ("fund_rank(roe) - fund_rank(asset_growth)", "profitability"),
    ("fund_rank(sue)", "earnings"),
    ("fund_rank(trade_when(days_since_earnings < 60, sue))", "earnings"),
    ("fund_rank(sue) + fund_rank(roe)", "earnings"),
    ("fund_rank(earnings_yield) + fund_rank(roe)", "composite"),
    ("fund_rank(earnings_yield) + fund_rank(sue)", "composite"),
    ("fund_rank(sue) + rank(ts_delta(log(close), 20))", "composite"),
    ("rank(ts_delta(log(close), 20))", "momentum"),            # 量价表达式的归类不变
    ("rank(-ts_std(returns, 20))", "volatility"),
])
def test_family_follows_the_fields_used(dsl, family):
    assert FinancialInterpreter().interpret(dsl).factor_family == family


def test_new_families_have_descriptions():
    assert {"value", "profitability", "earnings"} <= set(FACTOR_FAMILIES)
