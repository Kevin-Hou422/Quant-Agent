"""
变异体必须能写回 DSL 再读出来。

2026-10-12 查出的缺陷：`point_mutation` 会把 `ts_corr(close, volume, 10)` 的算子换成 `ts_min`，
而第二输入还留在节点上 —— 求值时第二输入被悄悄忽略，`repr` 出来的
`ts_min(close,volume,10)` 解析不了。种子生成的第 3 层把这种字符串存进种子表
（40 个随机种子 × 2 个假设里出现了 63 种），落了库的因子也就再也读不回来。
"""

from __future__ import annotations

import pytest

from app.core.alpha_engine.parser import Parser
from app.core.alpha_engine.typed_nodes import TimeSeriesNode
from app.core.gp_engine import _rng, mutations
from app.core.gp_engine.gp_engine import _SEED_DSLS

TWO_INPUT_SEEDS = [d for d in _SEED_DSLS if "ts_corr" in d or "ts_cov" in d]
UNARY_OPERATORS = ["point_mutation", "hoist_mutation", "param_mutation", "wrap_rank",
                   "add_ts_smoothing", "add_condition", "add_volume_filter", "replace_subtree",
                   "add_operator"]


def _walk(node):
    yield node
    for child in node.children():
        yield from _walk(child)


def _mutants(op_name: str, dsl: str, n: int = 60):
    op = getattr(mutations, op_name)
    saved = _rng.current()
    try:
        for seed in range(n):
            _rng.bind_seed(seed)
            yield op(Parser().parse(dsl))
    finally:
        _rng.bind(saved)


def test_there_are_two_input_seeds_to_mutate():
    assert len(TWO_INPUT_SEEDS) >= 4


@pytest.mark.parametrize("dsl", TWO_INPUT_SEEDS)
def test_point_mutation_of_a_two_input_operator_stays_parsable(dsl):
    seen = set()
    for mutant in _mutants("point_mutation", dsl, n=80):
        text = repr(mutant)
        seen.add(text)
        assert repr(Parser().parse(text)) == text
    assert len(seen) > 1                      # 确实发生了变异，不是每次都原样返回


def test_two_input_node_is_only_ever_swapped_for_another_two_input_operator():
    ops = set()
    for mutant in _mutants("point_mutation", "ts_corr(close, volume, 10)", n=40):
        (node,) = [n for n in _walk(mutant) if isinstance(n, TimeSeriesNode)]
        assert node.second_child is not None
        ops.add(node.op)
    assert ops == {"ts_cov"}                  # 唯一的 TS 节点带第二输入 → 只能换成 ts_cov


def test_single_input_node_never_becomes_a_two_input_operator():
    ops = set()
    for mutant in _mutants("point_mutation", "ts_mean(close, 10)", n=80):
        (node,) = [n for n in _walk(mutant) if isinstance(n, TimeSeriesNode)]
        assert node.second_child is None
        ops.add(node.op)
    assert not ops & {"ts_corr", "ts_cov"} and len(ops) > 5


@pytest.mark.parametrize("op_name", UNARY_OPERATORS)
def test_every_mutation_operator_produces_expressions_that_round_trip(op_name):
    """对整张种子表施加每一种变异：产物都必须能 repr → parse → repr 不变。"""
    checked = 0
    for dsl in _SEED_DSLS:
        for mutant in _mutants(op_name, dsl, n=3):
            text = repr(mutant)
            assert repr(Parser().parse(text)) == text, f"{op_name}({dsl!r}) → {text!r}"
            checked += 1
    assert checked == 3 * len(_SEED_DSLS)


def test_seed_generation_emits_only_parsable_expressions(monkeypatch):
    from app.core.workflows import alpha_workflows as W
    monkeypatch.setattr("app.agent.alpha_agent.AlphaAgent._generate_dsls", lambda self, h: [])
    seeds = [d for seed in range(12) for hyp in ("volume liquidity", "anything")
             for d in W._generate_diverse_seeds(hyp, n_target=20, seed=seed)]
    assert len(seeds) == 12 * 2 * 20
    assert [d for d in seeds if W._parse_valid(d) is None] == []
