"""
agent/_prompts.py —— 系统提示词与代码的一致性

**此前零测试**（35 个"变异点"，D 档 —— 但见下面的口径说明）。

口径：为什么这个模块不按击杀率验收
----------------------------------
`_prompts.py` 整个文件就是**一个字符串常量**。变异工具的 `_in_string`
守卫对"整模块级三引号字符串"失效，于是它把提示词正文里的
`not` / `and` / `>` / `*` 全当成代码算符改了一遍，报出 35 个"变异点"。

这些"变异"里没有一个是代码行为的改变 —— 它们改的是**给 LLM 读的散文**。
要"杀死"其中的 `删掉 not`、`and -> or` 之类，只能去断言
"提示词里第 10 行含有 not"，那正是自伤教训 #6 明令禁止的源码文本断言：
同一个词在文件里出现几十次，断言杀不掉任何东西，却把提示词钉死得无法迭代
（而模块 docstring 第一句就写着"分离出来是为了让提示词迭代不碰 Python 逻辑"）。

所以本文件**不追求击杀率**，改为验一件真正会坏、也真正有人吃亏的事：

    **提示词与代码脱节。**

提示词里写的每一个算子、字段、工具名、阈值、公式，都是 LLM 会照抄的。
代码改了而提示词没跟着改，后果不是报错，而是 LLM 一直按过时的规则
生成 DSL、传参、判断指标 —— 全链路静默走偏。本轮就是靠这组断言
抓出了 D-4（`neg` 模板解析不了）与 D-5（相关度阈值 0.9 vs 0.70）。
"""
from __future__ import annotations

import inspect
import re

import pytest

from app.agent._prompts import _SYSTEM_PROMPT as PROMPT
from app.core.alpha_engine.parser import ParseError, Parser
from app.core.alpha_engine.validator import AlphaValidator

_parser = Parser()
_validator = AlphaValidator()


def _declared(header: str) -> list[str]:
    """
    从 `AVAILABLE XXX: a, b, c` 段落里抽出逗号分隔的清单。

    清单可能换行续写（续行有缩进），但**下一个顶格的标题行或空行**
    就是边界 —— 不能一路吃到下一段去（第一版就是这么把
    OPERATORS 那行吞进 FIELDS 清单的）。
    """
    i = PROMPT.index(header)
    lines = PROMPT[i + len(header):].splitlines()
    taken = [lines[0]]
    for ln in lines[1:]:
        if not ln.strip() or not ln.startswith((" ", "\t")):
            break
        taken.append(ln)
    block = " ".join(taken)
    return [t.strip() for t in block.split(",") if t.strip()]


# 每个算子一组能过校验的实参
_ARITY = {
    "rank": "close", "zscore": "close", "scale": "close",
    "ts_mean": "close,20", "ts_std": "close,20", "ts_delta": "close,20",
    "ts_delay": "close,20", "ts_max": "close,20", "ts_min": "close,20",
    "ts_rank": "close,20", "ts_decay_linear": "close,20",
    "ts_corr": "close,volume,20", "ts_zscore": "close,20", "ts_skew": "close,20",
    "log": "close", "abs": "close", "sqrt": "close", "sign": "close",
    "signed_power": "close,2", "if_else": "close,close,close",
    "trade_when": "close,close", "ind_neutralize": "close",
    "neg": "close",
}


# ===========================================================================
# A. 字段与算子清单
# ===========================================================================

class TestDeclaredVocabulary:

    def test_every_declared_data_field_is_parseable(self):
        """
        `AVAILABLE DATA FIELDS: ...`

        清单里多一个引擎不认的字段（比如将来加了 `market_cap` 却没接进
        DSL），LLM 会照着用，产出的公式全部解析失败 ——
        而提示词里还有一条"用户点名的字段必须出现在公式里"的硬规则，
        于是它会**反复**尝试同一个不存在的字段。
        """
        fields = _declared("AVAILABLE DATA FIELDS:")
        assert fields, "提示词里找不到字段清单"
        for f in fields:
            try:
                _parser.parse(f"rank({f})")
            except ParseError as exc:
                pytest.fail(f"提示词声明的字段 `{f}` 解析器不认：{exc}")

    def test_the_declared_fields_are_the_seven_standard_ones(self):
        assert set(_declared("AVAILABLE DATA FIELDS:")) == {
            "close", "open", "high", "low", "volume", "vwap", "returns"}

    def test_every_declared_operator_is_parseable(self):
        """
        `AVAILABLE OPERATORS: ...`

        这是 LLM 唯一的"可用算子"依据。清单与解析器脱节的两个方向
        都很贵：多列 → 生成非法公式；少列 → 引擎支持的能力用不上。
        """
        ops = _declared("AVAILABLE OPERATORS:")
        assert ops, "提示词里找不到算子清单"
        unknown = [o for o in ops if o not in _ARITY]
        assert not unknown, (
            f"提示词声明了本用例不认识的算子 {unknown} —— "
            f"请在 _ARITY 里补上它的实参，或确认它是否真的存在")
        for o in ops:
            try:
                node = _parser.parse(f"{o}({_ARITY[o]})")
                _validator.validate(node)
            except Exception as exc:
                pytest.fail(f"提示词声明的算子 `{o}` 走不通：{type(exc).__name__}: {exc}")

    def test_the_operator_list_has_no_duplicates(self):
        ops = _declared("AVAILABLE OPERATORS:")
        assert len(ops) == len(set(ops)), (
            f"算子清单里有重复项：{[o for o in set(ops) if ops.count(o) > 1]}")


# ===========================================================================
# B. 提示词里的 DSL 范例 —— LLM 会直接照抄
# ===========================================================================

def _dsl_snippets() -> list[str]:
    """
    抽出提示词里所有形如 `op(...)` 的片段，并截到括号平衡处。

    只保留**不含占位符**的片段：`ts_delta(close, N)` 里的 `N`、
    `ts_mean(signal, 3-5)` 里的 `signal` / `3-5` 是刻意写给人看的模板，
    不该当成字面 DSL 校验。
    """
    pat = (r"\b(?:rank|zscore|scale|ts_\w+|log|abs|sqrt|sign|signed_power|"
           r"if_else|trade_when|ind_neutralize|neg)\s*\(")
    out = []
    for m in re.finditer(pat, PROMPT):
        start = m.start()
        depth = 0
        end = None
        for i in range(start, len(PROMPT)):
            ch = PROMPT[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
            elif ch == "\n":
                break
        if end is None:
            continue
        snip = PROMPT[start:end]
        # 过滤掉带占位符的模板
        if re.search(r"\b[A-Z]\b", snip) or "signal" in snip or "-" in snip:
            continue
        # 过滤掉 rank() / zscore() 这类"只是提到函数名"的裸写法
        if re.match(r"^\w+\(\s*\)$", snip):
            continue
        out.append(snip)
    return sorted(set(out))


class TestEmbeddedDslExamples:

    def test_there_are_some_concrete_examples_to_check(self):
        """守住本组用例的前提：提示词里确实有可校验的具体范例。"""
        snips = _dsl_snippets()
        assert len(snips) >= 5, (
            f"只抽出了 {len(snips)} 条具体 DSL 范例：{snips} —— "
            f"抽取规则可能与提示词的写法脱节了")

    def test_every_concrete_dsl_example_except_the_known_broken_ones_parses(self):
        """
        **本轮靠这条抓出了缺陷 D-4**。

        提示词里的 `DSL pattern:` 是 LLM 的主要模仿对象。
        其中任何一条解析不了，LLM 就会按它写出非法公式，
        然后在 `_validate_and_fix` 里白烧两次修复调用后被丢弃 ——
        对外只表现为"agent 老是生成非法公式"。

        曾经有一批坏的：含 `neg(` 的四条（缺陷 D-4）。它们于 2026-09-20 改成
        一元负号 `-x`，**例外已取消** —— 现在提示词里的每一条范例都必须解析得了。
        """
        bad = []
        for snip in _dsl_snippets():
            try:
                _parser.parse(snip)
            except Exception as exc:
                bad.append((snip, f"{type(exc).__name__}: {str(exc)[:80]}"))
        assert not bad, (
            "提示词里的 DSL 范例解析失败：\n  " +
            "\n  ".join(f"{x}  ->  {e}" for x, e in bad))

    def test_every_parseable_example_also_validates(self):
        """解析得了还不够 —— 还要过 AlphaValidator（嵌套深度、链式归一化等）。"""
        bad = []
        for snip in _dsl_snippets():
            try:
                _validator.validate(_parser.parse(snip))
            except Exception as exc:
                bad.append((snip, f"{type(exc).__name__}: {str(exc)[:80]}"))
        assert not bad, (
            "提示词里的 DSL 范例过不了校验：\n  " +
            "\n  ".join(f"{x}  ->  {e}" for x, e in bad))

    def test_no_example_calls_a_negation_function(self):
        """
        **缺陷 D-4，2026-09-20 已修**，这条由"钉住现状"改成反向护栏。

        提示词曾把 `rank(neg(...))` 当作四个因子家族的标准模板，而解析器只认
        一元负号 `-x` —— `neg` 不是函数。这里同时钉两件事：
        ① 提示词里不再出现 `neg(`；② 解析器确实不认它
        （不然"提示词不写"就成了没有理由的忌讳）。
        """
        assert not [x for x in _dsl_snippets() if "neg(" in x], (
            "提示词的范例里又出现了 `neg(` —— 解析器不认这个函数，D-4 复发")
        with pytest.raises(ParseError):
            _parser.parse("neg(close)")

    def test_the_unary_minus_form_is_what_the_parser_accepts(self):
        """
        D-4 的修法参照：把模板里的 neg(x) 换成 -x 就能解析。
        这条同时证明"问题在写法而不在取负这个能力本身"。
        """
        forms = ("rank(-ts_delta(close, 5))",
                 "rank(-ts_std(returns, 20))",
                 "rank(-ts_mean(volume, 20))",
                 "rank(-ts_corr(close, volume, 20))")
        for good in forms:
            node = _parser.parse(good)
            assert node is not None, f"{good} 解析出空节点"
            assert _validator.validate(node) is None, f"{good} 没有通过校验"
        assert len(forms) == 4, "四个家族的模板各要有一条对照写法"

    def test_every_function_used_in_an_example_is_declared(self):
        """
        范例里用到的算子必须出现在 `AVAILABLE OPERATORS` 清单上 ——
        否则 LLM 会得到互相矛盾的两份信息。

        `neg` 曾是唯一的例外（缺陷 D-4）：出现在四条范例里，却不在清单上，
        而且解析器根本没有这个函数。**例外已于 2026-09-20 取消** ——
        修法不是把 `neg` 加进清单（那会让提示词与解析器一起错），
        而是把模板改成解析器接受、且系统自己序列化时也输出的 `-x`。
        """
        declared = set(_declared("AVAILABLE OPERATORS:"))
        used = {m.group(1) for m in re.finditer(r"\b(\w+)\s*\(", PROMPT)
                if m.group(1) in _ARITY}
        missing = sorted(used - declared)
        assert not missing, (
            f"这些算子在提示词正文里被当成范例用了，却不在 "
            f"AVAILABLE OPERATORS 清单上：{missing} —— "
            f"LLM 拿到的是自相矛盾的两份说明")


# ===========================================================================
# C. 工具名
# ===========================================================================

class TestToolNames:

    def test_every_tool_named_in_the_prompt_exists(self):
        """
        提示词里 Workflow A/B 逐步点名了要调哪个工具。
        名字与 `QuantTools` 脱节 → LLM 发出的工具调用找不到实现，
        AgentExecutor 走进 `handle_parsing_errors` 兜底，
        表现为"模型答非所问"。
        """
        from app.agent._tools import QuantTools
        named = set(re.findall(r"\btool_[a-z_]+", PROMPT))
        assert named, "提示词里一个工具名都没提到"
        missing = sorted(n for n in named if not hasattr(QuantTools, n))
        assert not missing, (
            f"提示词点名了不存在的工具：{missing}")

    def test_the_set_of_tools_the_prompt_never_mentions_does_not_grow(self):
        """
        反方向：接线层注册了但提示词只字未提的工具，LLM 基本不会主动用。

        当前有两个：`tool_run_backtest` 与 `tool_run_optuna`。
        这是**刻意的** —— 提示词里写明"GP 已经内部调用 Optuna"，
        回测也包在 GP 流程里，所以 Workflow A/B 不点它们的名；
        它们仍然注册着，作为"GP 被跳过时"的手动入口。

        钉住这个集合的意义：**再多一个未提及的工具**（比如新加了一个
        本该由 LLM 主动调用的工具却忘了写进提示词）就会红。
        """
        import app.agent._lc_agent as LC
        # 工具定义在 `_build_tools`（2026-09-20 从 `_build_langchain_agent` 抽出，
        # 为的是让接线能用真的 @lc_tool 验、不必 mock 掉整个 langchain）。
        registered = set(re.findall(r"def (tool_[a-z_]+)\(",
                                    inspect.getsource(LC._build_tools)))
        assert registered, "接线层里抽不出工具名 —— 本用例需要重写"
        named = set(re.findall(r"\btool_[a-z_]+", PROMPT))
        unmentioned = sorted(registered - named)
        assert unmentioned == ["tool_run_backtest", "tool_run_optuna"], (
            f"提示词未提及的工具集合变成了 {unmentioned} —— "
            f"新增的那个若需要 LLM 主动调用，必须写进提示词")


# ===========================================================================
# D. 数字与公式 —— 提示词里写的必须就是代码里算的
# ===========================================================================

class TestNumbersMatchTheCode:

    def test_the_fitness_formula_coefficients_match_compute_fitness(self):
        """
        提示词：`fitness = sharpe_oos - 0.2×turnover - 0.3×|max_drawdown|
                  - 0.5×overfit_penalty`

        这三个系数直接决定 LLM 如何解读 GP 的排序结果。
        代码改了系数而提示词没改，LLM 会按旧权重去推荐"该往哪个方向改"，
        建议方向与实际优化目标背道而驰。
        从提示词里抽出系数，与 `compute_fitness` 的源码逐个比对。
        """
        from app.core.gp_engine import fitness as FIT

        line = next(l for l in PROMPT.splitlines() if "fitness =" in l)
        in_prompt = re.findall(r"(\d*\.?\d+)\s*×", line)
        assert in_prompt, f"提示词里的适应度公式抽不出系数：{line}"

        src = inspect.getsource(FIT.compute_fitness)
        in_code = re.findall(r"-\s*(\d*\.?\d+)\s*\*", src)
        assert in_code, "compute_fitness 源码里抽不出系数 —— 本用例需要重写"

        assert in_prompt == in_code, (
            f"适应度公式的系数对不上：提示词 {in_prompt} vs 代码 {in_code}")

    def test_the_fitness_formula_signs_are_all_penalties(self):
        """
        三项都必须是**减号**。任一被写成加号，LLM 会把"高换手"
        理解成加分项，于是一路建议往高换手方向改。
        """
        line = next(l for l in PROMPT.splitlines() if "fitness =" in l)
        body = line.split("=", 1)[1]
        terms = re.findall(r"([+-])\s*\d*\.?\d+\s*×", body)
        assert terms, f"抽不出符号：{line}"
        assert set(terms) == {"-"}, (
            f"适应度公式里出现了加号项：{line} —— "
            f"换手/回撤/过拟合三项都必须是惩罚")

    def test_the_overfitting_threshold_matches_the_constant(self):
        """
        提示词：`If overfitting (overfitting_score > 0.5)`
        代码：`_OVERFIT_THRESHOLD = 0.50`
        """
        from app.agent._constants import _OVERFIT_THRESHOLD

        m = re.search(r"overfitting_score\s*>\s*([\d.]+)", PROMPT)
        assert m, "提示词里找不到过拟合阈值"
        assert float(m.group(1)) == pytest.approx(_OVERFIT_THRESHOLD), (
            f"过拟合阈值对不上：提示词 {m.group(1)} vs 代码 {_OVERFIT_THRESHOLD}")

    def test_the_gp_defaults_named_in_the_prompt_match_the_tool_signature(self):
        """
        提示词的 Workflow A/B 写死了 `n_generations=4, pop_size=12`。
        接线层的默认值若不同，LLM 显式传参与不传参会得到两种规模的搜索，
        而报告里只会写"跑了一轮 GP"。
        """
        import app.agent._lc_agent as LC

        src = inspect.getsource(LC._build_tools)      # 工具签名在这里
        for key in ("n_generations", "pop_size"):
            in_prompt = set(re.findall(rf"{key}=(\d+)", PROMPT))
            in_code = set(re.findall(rf"{key}:\s*int\s*=\s*(\d+)", src))
            assert in_prompt and in_code, (
                f"{key} 在提示词({in_prompt})或代码({in_code})里抽不出来")
            assert in_prompt == in_code, (
                f"{key} 的默认值对不上：提示词 {in_prompt} vs 接线层 {in_code}")

    def test_the_mutation_targets_named_in_the_prompt_exist(self):
        """
        提示词把 `wrap_rank, add_ts_smoothing, add_condition, ...`
        列为 "GP mutations"，而 `tool_mutate_ast` 会按名字直接分派。
        名字对不上 → 直接分派失败，退回随机变异，而返回值看不出区别。
        """
        import app.core.gp_engine.mutations as M

        i = PROMPT.index("GP mutations include:")
        block = PROMPT[i:i + 400]
        names = set(re.findall(r"\b(wrap_rank|add_ts_smoothing|add_condition|"
                               r"add_volume_filter|combine_signals|"
                               r"replace_subtree|add_operator)\b", block))
        assert len(names) >= 5, f"提示词里的变异算子清单抽取异常：{names}"
        missing = sorted(n for n in names if not hasattr(M, n))
        assert not missing, (
            f"提示词点名了 mutations.py 里不存在的算子：{missing}")

    def test_the_alpha_pool_correlation_threshold_matches_the_default(self):
        """
        **本轮靠这条抓出了缺陷 D-5**（当前钉住的是**现状**，见下）。

        提示词：`AlphaPool rejects signal-correlated alphas (corr > 0.9)`
        代码：  `corr_threshold: float = 0.70`，判定 `abs(corr) >= threshold`

        两处都对不上。本阶段只登记不修，所以这里钉住现状；
        与 `tests/meta/test_known_defects.py` 里的 D-5 xfail 用例成对，
        修好时两条一起改。
        """
        import inspect as _i

        from app.core.gp_engine.alpha_pool import AlphaPool

        m = re.search(r"corr\s*>\s*([\d.]+)", PROMPT)
        assert m, "提示词里找不到相关度阈值"
        prompt_value = float(m.group(1))

        default = _i.signature(AlphaPool.__init__).parameters["corr_threshold"].default

        assert prompt_value == 0.9, (
            f"提示词里的相关度阈值变成了 {prompt_value} —— "
            f"如果已改成与代码一致，请同步删除缺陷 D-5")
        assert default == pytest.approx(0.70), (
            f"AlphaPool 的默认阈值变成了 {default} —— 请同步更新缺陷 D-5")
        assert prompt_value != default, (
            "提示词与代码的相关度阈值已经一致了 —— 缺陷 D-5 已修复，"
            "请删掉 test_known_defects 里的 D-5 与本断言")


# ===========================================================================
# E. 提示词本身的结构
# ===========================================================================

class TestPromptStructure:

    def test_the_placeholder_markers_are_well_formed(self):
        """
        提示词用 `<extracted above>` / `<from step 3>` 这类尖括号占位符
        指示"把上一步的结果填进来"。任何一处写成 `<...>=` 之类的畸形，
        LLM 会把它当成字面量抄进参数里。

        （这也顺带杀掉变异工具在 `<placeholder>` 上误判出的那一批
        `> → >=`：畸形的 `>=` 会让占位符失配。）
        """
        broken = re.findall(r"<[^<>\n]*>=", PROMPT)
        assert not broken, f"畸形的占位符：{broken}"
        opens = PROMPT.count("<")
        closes = PROMPT.count(">")
        # `>` 还用在 `→` 之外的比较里，所以只要求每个 `<` 都能配到一个 `>`
        assert opens <= closes, f"占位符尖括号不配对：{opens} 个 < vs {closes} 个 >"

    def test_the_two_workflows_are_both_present(self):
        for frag in ("WORKFLOW A", "WORKFLOW B"):
            assert frag in PROMPT, f"提示词里缺少 {frag}"

    def test_the_six_factor_families_are_all_documented(self):
        for fam in ("MOMENTUM", "MEAN REVERSION", "VOLATILITY", "LIQUIDITY",
                    "PRICE-VOLUME CORRELATION", "COMPOSITE"):
            assert fam in PROMPT, f"因子分类学里缺少 {fam}"

    def test_the_factor_families_match_the_interpreter_vocabulary(self):
        """
        提示词让 LLM 把 `tool_interpret_factor` 返回的 `factor_family`
        原样传给 `tool_run_gp_optimization`。两边的取值域必须重合，
        否则 GP 拿到一个它不认识的家族名 → 权重偏置静默失效。
        """
        import app.core.alpha_engine.financial_interpreter as FI

        src = inspect.getsource(FI)
        for fam in ("momentum", "reversion", "volatility", "liquidity",
                    "composite"):
            assert fam in src, (
                f"提示词里出现的家族 `{fam}` 在 financial_interpreter 里找不到")

    def test_the_prompt_forbids_substituting_a_user_named_field(self):
        """
        "用户点名了哪个字段就必须用哪个" 是这份提示词里唯一一条
        大写加粗的硬规则 —— 它防的是 LLM 把 `vwap` 悄悄换成 `close`。
        删掉它不会有任何报错，只会让生成的因子与用户的意图渐行渐远。
        """
        assert "NEVER substitute a different data field" in PROMPT, (
            "提示词里删掉了『不得替换用户指定字段』的硬规则")

    def test_the_prompt_is_not_accidentally_truncated(self):
        """
        提示词是一整个三引号字符串，很容易在编辑时被截断。
        结尾段（MEMORY）在就说明整份完整。
        """
        assert PROMPT.strip().endswith("Full conversation history is automatically provided."), (
            f"提示词末尾不是预期的 MEMORY 段，疑似被截断：\n"
            f"...{PROMPT[-120:]!r}")
        assert len(PROMPT) > 4000, f"提示词只剩 {len(PROMPT)} 字符，疑似被截断"
