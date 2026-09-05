"""
test_dsl_edge_cases.py — DSL 解析器边界值与异常输入测试

覆盖：空字符串、未闭合括号、错误参数个数、未知函数、
深度限制、CS 嵌套约束、窗口/延迟参数校验、超长输入、
SQL 注入无害化、Unicode 输入。
"""
from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# 帮助函数
# ---------------------------------------------------------------------------

def _parse(dsl: str):
    from app.core.alpha_engine.parser import Parser
    return Parser().parse(dsl)


def _validate(dsl: str):
    from app.core.alpha_engine.parser import Parser
    from app.core.alpha_engine.validator import AlphaValidator
    node = Parser().parse(dsl)
    AlphaValidator().validate(node)
    return node


# ---------------------------------------------------------------------------
# 解析失败 — 语法错误
# ---------------------------------------------------------------------------

class TestParseErrors:

    def test_empty_string(self):
        from app.core.alpha_engine.parser import ParseError
        with pytest.raises((ParseError, Exception)):
            _parse("")

    def test_whitespace_only(self):
        from app.core.alpha_engine.parser import ParseError
        with pytest.raises((ParseError, Exception)):
            _parse("   ")

    def test_unclosed_paren(self):
        from app.core.alpha_engine.parser import ParseError
        with pytest.raises((ParseError, Exception)):
            _parse("rank(close")

    def test_double_operator(self):
        from app.core.alpha_engine.parser import ParseError
        with pytest.raises((ParseError, Exception)):
            _parse("close ++ open")

    def test_unknown_function(self):
        from app.core.alpha_engine.parser import ParseError
        with pytest.raises((ParseError, Exception)):
            _parse("foo(close, 5)")

    def test_wrong_arity_ts_mean_missing_window(self):
        """ts_mean 需要两个参数：field + window。"""
        from app.core.alpha_engine.parser import ParseError
        with pytest.raises((ParseError, Exception)):
            _parse("ts_mean(close)")

    def test_wrong_arity_too_many_args(self):
        from app.core.alpha_engine.parser import ParseError
        with pytest.raises((ParseError, Exception)):
            _parse("rank(close, 5, 10)")

    def test_very_long_dsl_does_not_crash(self):
        """超长 DSL 应抛出异常，不应让进程崩溃或无限阻塞。"""
        import time
        long_dsl = "rank(" * 200 + "close" + ")" * 200
        t0 = time.perf_counter()
        try:
            node = _parse(long_dsl)
        except Exception as e:
            # 允许拒绝，但必须是**可理解的**解析错误，不能是 RecursionError
            assert not isinstance(e, RecursionError), (
                "200 层嵌套触发 RecursionError —— 解析器未做深度保护，"
                "恶意/误输入可直接打崩进程"
            )
            assert str(e).strip(), "抛出空异常信息"
        else:
            assert node is not None
        assert time.perf_counter() - t0 < 5.0, "超长 DSL 解析超过 5 秒（可被用于拖垮服务）"

    def test_sql_injection_does_not_execute(self):
        """SQL 注入字符串应被解析器拒绝（不执行 SQL）。"""
        from app.core.alpha_engine.parser import ParseError
        # 契约：必须**拒绝**。原用例 try/except pass 让"解析成功"也算通过 ——
        # 那才是真正危险的情况，却测不出来。
        with pytest.raises(Exception):
            _parse("rank('; DROP TABLE alpha_records; --)")

    def test_unicode_field_name_rejected(self):
        """Unicode 字段名不是有效 DSL 标识符，应被拒绝。"""
        from app.core.alpha_engine.parser import ParseError
        with pytest.raises((ParseError, Exception)):
            _parse("rank(收盘)")


# ---------------------------------------------------------------------------
# 验证失败 — 语义约束
# ---------------------------------------------------------------------------

class TestValidationErrors:

    def test_depth_limit_enforced(self):
        """树深度超过验证阈值应被验证器拒绝（11层 ts_mean 嵌套 → depth≈12）。"""
        dsl = "ts_mean(" * 11 + "close" + ", 5)" * 11
        with pytest.raises(Exception):
            _validate(dsl)

    def test_depth_10_accepted(self):
        """合理深度的 DSL 应通过验证。"""
        dsl = "rank(ts_delta(log(ts_mean(close, 5)), 5))"
        node = _validate(dsl)
        assert node is not None

    def test_cs_nested_in_cs_behavior(self):
        """rank(rank(x)) — 当前验证器允许通过（记录实际行为）。

        注：CS 嵌套约束在现有验证器中未强制执行。
        此测试记录当前行为（通过），若将来添加约束则需更新。
        """
        # 记录当前行为需要**确定的**断言：当前验证器允许 CS 嵌套。
        # 原写法（try/except pass）对"通过"和"抛异常"都算成功 ——
        # 行为一旦反转也不会有人知道，等于没有记录任何行为。
        node = _validate("rank(rank(close))")
        assert node is not None, "rank(rank(x)) 现行为应通过验证；若已改为拒绝，请更新本用例"

    def test_ts_window_zero_rejected(self):
        """ts_mean(close, 0) 窗口必须 ≥ 1。"""
        with pytest.raises(Exception):
            _validate("ts_mean(close, 0)")

    def test_ts_window_negative_rejected(self):
        with pytest.raises(Exception):
            _validate("ts_mean(close, -5)")

    def test_ts_delay_zero_rejected(self):
        """ts_delay(close, 0) 延迟必须 ≥ 1（防止前视偏差）。"""
        with pytest.raises(Exception):
            _validate("ts_delay(close, 0)")


# ---------------------------------------------------------------------------
# 合法 DSL — 不应抛出异常
# ---------------------------------------------------------------------------

class TestValidDSL:

    def test_simple_rank(self):
        node = _parse("rank(close)")
        assert node is not None

    def test_ts_delta_log(self):
        node = _parse("rank(ts_delta(log(close), 5))")
        assert node is not None

    def test_arithmetic(self):
        node = _parse("(close - open) / close")
        assert node is not None

    def test_if_else(self):
        node = _parse("if_else(close > open, close, open)")
        assert node is not None

    def test_signed_power(self):
        node = _parse("signed_power(rank(close), 2)")
        assert node is not None

    def test_group_rank(self):
        node = _parse("group_rank(close, 'sector')")
        assert node is not None

    def test_combined_operators(self):
        dsl = "rank(ts_delta(log(close), 5)) / ts_std(close, 20)"
        node = _validate(dsl)
        assert node is not None
