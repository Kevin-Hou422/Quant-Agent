"""
Phase 10 — agent 只向 LLM 宣传数据集**真的有**的字段，并且不放行引用了不存在字段的表达式。

此前提示词写死 `close, open, high, low, volume, vwap, returns`：开了基本面 LLM 也不知道有这些字段；
反过来 LLM 编出一个不存在的字段时，表达式会原样进 GP，每一代都求值失败。
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest

from app.agent._constants import _FUNDAMENTAL_FIELD_NOTES, _VALID_FIELDS, _VALID_OPS, prompt_field_lines
from app.agent._tools import QuantTools
from app.core.alpha_engine.fields import FUNDAMENTAL_FIELDS
from app.core.alpha_engine.parser import Parser
from app.core.alpha_engine.validator import AlphaValidator

FUND = ["market_cap", "earnings_yield", "book_to_price", "roe", "gross_margin", "sue", "days_since_earnings"]


class _LLM:
    """记下提示词、按脚本应答的 LLM 替身。"""

    def __init__(self, dsl: str) -> None:
        self.prompts: list = []
        self._dsl = dsl

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return json.dumps({"dsl": self._dsl, "explanation": "scripted"})


def _tools(llm=None, fields=()) -> QuantTools:
    t = QuantTools(llm=llm, allow_synthetic=True, n_days=320)
    close = t._full_dataset["close"]
    for name in fields:
        t._full_dataset[name] = pd.DataFrame(np.random.default_rng(1).uniform(0.01, 1.0, close.shape),
                                             index=close.index, columns=close.columns)
    return t


def _answer(tools: QuantTools, hypothesis: str) -> dict:
    return json.loads(tools.tool_generate_alpha_dsl(hypothesis))


# ---------------------------------------------------------------------------
# 提示词
# ---------------------------------------------------------------------------

def test_price_only_dataset_gets_the_original_prompt_lines_verbatim():
    assert prompt_field_lines({"close", "volume", "returns"}) == (_VALID_FIELDS, _VALID_OPS, "", "")
    llm = _LLM("rank(close)")
    _answer(_tools(llm), "anything")
    (prompt,) = llm.prompts
    assert f"Available data fields: {_VALID_FIELDS}\n" in prompt
    assert not any(f in prompt for f in FUNDAMENTAL_FIELDS)
    assert "fund_rank" not in prompt


def test_fundamental_fields_are_advertised_only_when_present():
    llm = _LLM("fund_rank(roe)")
    _answer(_tools(llm, fields=["roe", "sue"]), "quality")
    (prompt,) = llm.prompts
    assert "roe = " in prompt and "sue = " in prompt
    assert "earnings_yield" not in prompt.split("Examples:")[0]          # 没有的字段不列
    assert "fund_rank" in prompt and "126" in prompt                     # 算子与阶梯字段的规则
    assert "fund_rank(earnings_yield)" in prompt.split("Examples:")[1]   # 带了基本面示例


def test_every_fundamental_field_has_a_prompt_note():
    assert set(_FUNDAMENTAL_FIELD_NOTES) == set(FUNDAMENTAL_FIELDS)
    fields, ops, rules, shots = prompt_field_lines(FUNDAMENTAL_FIELDS | {"close"})
    assert all(f"{f} = " in fields for f in FUNDAMENTAL_FIELDS)
    assert ops.endswith("fund_rank, trade_when") and rules and shots


def test_the_examples_shown_to_the_llm_are_valid_expressions():
    _, _, _, shots = prompt_field_lines(FUNDAMENTAL_FIELDS)
    found = [seg.split('"')[0] for seg in shots.split('"dsl": "')[1:]]
    assert len(found) == 2 and len(shots.splitlines()) == 2
    for dsl in found:
        AlphaValidator().validate(Parser().parse(dsl))


# ---------------------------------------------------------------------------
# LLM 的回答要过字段检查
# ---------------------------------------------------------------------------

def test_llm_answer_using_available_fundamental_fields_is_accepted():
    out = _answer(_tools(_LLM("fund_rank(earnings_yield)"), fields=FUND), "cheap stocks")
    assert out == {"dsl": "fund_rank(earnings_yield)", "explanation": "scripted"}


@pytest.mark.parametrize("dsl", ["fund_rank(roe)", "rank(made_up_field)", "rank(ts_delta(pe_ratio, 5))"])
def test_llm_answer_referencing_a_field_the_dataset_lacks_falls_back(dsl, caplog):
    with caplog.at_level(logging.WARNING):
        out = _answer(_tools(_LLM(dsl)), "momentum")
    assert out["dsl"] != dsl and "[Fallback]" in out["explanation"]
    assert any("没有的字段" in m for m in caplog.messages)
    AlphaValidator().validate(Parser().parse(out["dsl"]))


def test_unparsable_llm_answer_falls_back():
    out = _answer(_tools(_LLM("rank(ts_delta(close")), "momentum")
    assert "[Fallback]" in out["explanation"]


def test_derived_fields_count_as_available():
    """returns / vwap 由执行器派生，数据集里没有这一列也算可用。"""
    t = _tools(_LLM("rank(ts_mean(returns, 10)) + rank(vwap)"))
    t._full_dataset.pop("returns", None)
    t._full_dataset.pop("vwap", None)
    assert _answer(t, "x")["explanation"] == "scripted"


# ---------------------------------------------------------------------------
# 无 LLM 的降级路径
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hypothesis,expected", [
    ("cheap value stocks", "fund_rank(earnings_yield) + fund_rank(book_to_price)"),
    ("低估值", "fund_rank(earnings_yield) + fund_rank(book_to_price)"),
    ("post earnings drift", "fund_rank(sue)"),
    ("财报超预期", "fund_rank(sue)"),
    ("high quality companies", "fund_rank(roe) + fund_rank(gross_margin)"),
    ("small cap premium", "fund_rank(-log(market_cap))"),
    ("小市值", "fund_rank(-log(market_cap))"),
])
def test_fallback_maps_fundamental_hypotheses_when_the_fields_exist(hypothesis, expected):
    out = _answer(_tools(fields=FUND), hypothesis)
    assert out["dsl"] == expected
    AlphaValidator().validate(Parser().parse(out["dsl"]))


@pytest.mark.parametrize("hypothesis", ["cheap value stocks", "post earnings drift", "small cap premium"])
def test_fallback_never_emits_fundamental_expressions_on_a_price_only_dataset(hypothesis):
    dsl = _answer(_tools(), hypothesis)["dsl"]
    assert not any(f in dsl for f in FUNDAMENTAL_FIELDS)


def test_fallback_needs_all_fields_of_the_mapped_expression():
    """'quality' 的表达式要 roe 和 gross_margin 两个字段；只有一个时不用它。"""
    dsl = _answer(_tools(fields=["roe"]), "high quality companies")["dsl"]
    assert "gross_margin" not in dsl
    assert _answer(_tools(fields=["roe"]), "roe leaders")["dsl"] == "fund_rank(roe)"


def test_price_volume_hypotheses_are_unaffected_by_fundamentals_being_present():
    with_fund = _answer(_tools(fields=FUND), "volume spike")
    without = _answer(_tools(), "volume spike")
    assert with_fund == without
