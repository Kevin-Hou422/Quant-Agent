"""
main.py —— 启动自检、CORS、合成数据与 CLI 参数的定钉测试（变异测试驱动）

来由：21 个变异点，首测击杀率 **9.5%**（存活 19）。

`main.py` 里最要紧的是 `_assert_safe_bind()`：**本服务零认证**，
审批/拒绝/删除/跑 GP 的端点任何能访问到它的人都能调用。
绑到非回环地址等于把这些能力开给整个网络，所以那句守卫是唯一的闸。
它被改掉不会有任何测试变红 —— 服务照常启动，只是闸没了。

存活项：
  - `if not loopback and not getattr(settings, "allow_insecure_bind", False)`
    里的 `False` 默认值 —— 改成 True 等于**默认解除保险**
  - `if "*" in cors_origins and ... is False` 的 `and` 与 `False`
    —— 通配 CORS 的告警静默消失
  - `allow_credentials=True` —— 改成 False 会让带 cookie 的前端请求全挂
  - 合成数据生成里的 `cumprod`、`1 + rng.normal(...)`、`close*(1±u)`
    —— 合成价格变成负数或不再是随机游走，"仅供单元测试"的数据失去意义
  - `use_synthetic = getattr(args, "use_synthetic", False)` 的默认 False
    —— **默认用合成数据**跑回测，结论全部无效而没有任何提示
  - `required=True` / 两处 argparse `default=False`
  - `if args.oos_ratio > 0:` 的边界

既有覆盖（test_production_defaults / integration/test_api_health 等 18 个文件）
测的是"服务能起、端点能通"，没有一条去改配置看那句守卫会不会拦。
"""
from __future__ import annotations

import argparse
import importlib
import logging

import numpy as np
import pytest


@pytest.fixture
def main_mod():
    import app.main as m
    return m


# ===========================================================================
# A. 绑定地址的安全闸
# ===========================================================================

class TestSafeBindGuard:

    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", ""])
    def test_loopback_hosts_are_allowed(self, main_mod, monkeypatch, host):
        """
        回环地址必须放行。上一版写的是"调用一下，不抛即通过"——
        **一条断言都没有**，把 `_assert_safe_bind` 整个清空也照样绿
        （test_lessons_enforced 的零断言检查抓到了这一点）。
        这里把"没抛"显式记下来再断言。
        """
        from app.config import settings
        monkeypatch.setattr(settings, "api_bind_host", host, raising=False)
        monkeypatch.setattr(settings, "allow_insecure_bind", False, raising=False)

        raised = None
        try:
            main_mod._assert_safe_bind()
        except RuntimeError as exc:
            raised = exc
        assert raised is None, (
            f"回环地址 {host!r} 被守卫拒绝了：{raised}")

    @pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "example.com"])
    def test_non_loopback_is_refused_by_default(self, main_mod, monkeypatch, host):
        """
        `if not loopback and not getattr(settings, "allow_insecure_bind", False)`
        —— 那个 `False` 是**默认不解除保险**。改成 True 之后，
        任何没有显式写 `ALLOW_INSECURE_BIND` 的部署都会直接对外暴露一个
        零认证的服务：审批、拒绝、删会话、跑 GP，全网可调。
        """
        from app.config import settings
        monkeypatch.setattr(settings, "api_bind_host", host, raising=False)
        monkeypatch.delattr(settings, "allow_insecure_bind", raising=False)
        with pytest.raises(RuntimeError, match="拒绝启动"):
            main_mod._assert_safe_bind()

    def test_explicit_optin_allows_a_public_bind(self, main_mod, monkeypatch):
        """显式解除保险之后才允许对外绑定 —— 同样要把"没抛"断言出来。"""
        from app.config import settings
        monkeypatch.setattr(settings, "api_bind_host", "0.0.0.0", raising=False)
        monkeypatch.setattr(settings, "allow_insecure_bind", True, raising=False)

        raised = None
        try:
            main_mod._assert_safe_bind()
        except RuntimeError as exc:
            raised = exc
        assert raised is None, (
            f"显式 allow_insecure_bind=True 之后仍被拒绝：{raised}")

    def test_the_guard_distinguishes_the_two_cases(self, main_mod, monkeypatch):
        """守卫恒抛或恒不抛时，这条相等断言会成立 —— 必须不相等。"""
        from app.config import settings

        errors: list = []

        def _raises(host, optin):
            monkeypatch.setattr(settings, "api_bind_host", host, raising=False)
            monkeypatch.setattr(settings, "allow_insecure_bind", optin, raising=False)
            try:
                main_mod._assert_safe_bind()
                return False
            except RuntimeError as exc:
                errors.append((host, exc))
                return True

        public = _raises("0.0.0.0", False)
        loop = _raises("127.0.0.1", False)
        assert public is True, "0.0.0.0 没有被拒"
        assert loop is False, "127.0.0.1 被误拒"
        assert public != loop, "守卫对回环与非回环给出了同一个结论"
        # 吞掉的异常要落地检查，否则"抛了什么"完全不可见
        assert [h for h, _ in errors] == ["0.0.0.0"], (
            f"被拒的地址清单是 {[h for h, _ in errors]}，应当只有 0.0.0.0")
        assert "拒绝启动" in str(errors[0][1]), (
            f"拒绝的理由不对：{errors[0][1]}")

    def test_the_error_names_the_offending_host_and_the_optin_flag(self, main_mod,
                                                                   monkeypatch):
        from app.config import settings
        monkeypatch.setattr(settings, "api_bind_host", "10.0.0.7", raising=False)
        monkeypatch.setattr(settings, "allow_insecure_bind", False, raising=False)
        with pytest.raises(RuntimeError) as exc:
            main_mod._assert_safe_bind()
        msg = str(exc.value)
        assert "10.0.0.7" in msg, f"报错没有指出是哪个地址：{msg}"
        assert "ALLOW_INSECURE_BIND" in msg, f"报错没有给出解除方式：{msg}"


class TestCorsWarning:

    def test_wildcard_cors_without_optin_warns(self, main_mod, monkeypatch, caplog):
        """
        `if "*" in settings.cors_origins and getattr(..., False) is False:`
        —— `and` 放宽成 `or` 会让**没有**通配来源时也告警（噪声），
        `False` 改成 True 会让真正危险的组合**不再告警**。
        """
        from app.config import settings
        monkeypatch.setattr(settings, "api_bind_host", "127.0.0.1", raising=False)
        monkeypatch.setattr(settings, "cors_origins", ["*"], raising=False)
        # 用 delattr 而不是设成 False：`getattr(settings, "allow_insecure_bind", False)`
        # 里的那个**默认值**只有在属性缺失时才可达。显式设成 False 时走的是属性，
        # 默认值被改成 True 也看不出来（上一版就是这么让它活下来的）。
        monkeypatch.delattr(settings, "allow_insecure_bind", raising=False)
        with caplog.at_level(logging.WARNING, logger="main"):
            main_mod._assert_safe_bind()
        assert any("CORS" in r.getMessage() for r in caplog.records), (
            "通配 CORS + 零认证的组合没有告警")

    def test_specific_origins_do_not_warn(self, main_mod, monkeypatch, caplog):
        from app.config import settings
        monkeypatch.setattr(settings, "api_bind_host", "127.0.0.1", raising=False)
        monkeypatch.setattr(settings, "cors_origins",
                            ["http://localhost:5173"], raising=False)
        monkeypatch.setattr(settings, "allow_insecure_bind", False, raising=False)
        with caplog.at_level(logging.WARNING, logger="main"):
            main_mod._assert_safe_bind()
        assert not any("CORS" in r.getMessage() for r in caplog.records), (
            "指定了具体来源却仍然告警 —— 条件被放宽成了 or")

    def test_cors_middleware_allows_credentials(self, main_mod):
        """
        `allow_credentials=True` —— 改成 False 会让前端所有带 cookie/凭据的
        请求被浏览器拦掉，症状是"接口通、页面空"，极难定位。
        """
        mws = [m for m in main_mod.app.user_middleware
               if "CORS" in str(m.cls)]
        assert mws, "没有装上 CORS 中间件"
        opts = mws[0].kwargs if hasattr(mws[0], "kwargs") else {}
        assert opts.get("allow_credentials") is True, (
            f"allow_credentials={opts.get('allow_credentials')}")


# ===========================================================================
# B. 合成数据必须显式 opt-in，且确实是随机游走
# ===========================================================================

class TestSyntheticData:

    def test_synthetic_is_off_unless_the_flag_is_present(self, main_mod):
        """
        `use_synthetic = getattr(args, "use_synthetic", False)` —— 这个默认
        `False` 改成 True 会让**没有传 --use-synthetic 的每一次运行**都用
        合成随机游走数据，而所有结论在真实市场上无效。
        这正是项目里"合成数据冒充真实数据"那条铁律要防的事。
        """
        # 上一版写的是 `assert getattr(args, "use_synthetic", False) is False` ——
        # 那是在测 Python 的 getattr，不是测产品，恒真。
        # 正确做法：给一个**没有该属性**的 Namespace，看它会不会去走合成分支。
        seen = {"synthetic": False}
        real_rng = np.random.default_rng

        def _spy(*a, **k):
            seen["synthetic"] = True
            return real_rng(*a, **k)

        args = argparse.Namespace(dataset="us_tech_large")   # 无 use_synthetic 属性
        orig = np.random.default_rng
        np.random.default_rng = _spy
        errors: list = []
        try:
            try:
                main_mod._load_dataset(args)
            except Exception as exc:
                # 走真实数据源失败无所谓（本用例只看有没有造合成数据），
                # 但**不能静默吞掉** —— 落地记下来，下面断言它确实是
                # "加载真实数据失败"而不是别的错（比如 spy 自己炸了）。
                errors.append(exc)
        finally:
            np.random.default_rng = orig

        for exc in errors:
            assert not isinstance(exc, (TypeError, AttributeError)), (
                f"_load_dataset 抛的是 {type(exc).__name__}: {exc} —— "
                f"这不像'真实数据源不可用'，用例的前提可能已经不成立")

        assert seen["synthetic"] is False, (
            "参数里没有 --use-synthetic，却走了合成数据分支 —— "
            "`getattr(args, \"use_synthetic\", False)` 的默认值疑似变成了 True")

    def test_synthetic_prices_are_a_positive_random_walk(self, main_mod):
        """
        `100 * np.cumprod(1 + rng.normal(0, 0.01, (T, N)), axis=0)` ——
        三处变异各有后果：`cumprod`→`cumsum` 让价格变成收益的累加（可为负）；
        `1 +`→`1 -` 让走势整体翻转；`*`→`/` 让价格量级崩到 1e-2。
        价格出现非正数时，所有 `log(close)` / 收益率计算立刻产生 NaN。
        """
        args = argparse.Namespace(use_synthetic=True, n_days=120, n_tickers=8)
        ds = main_mod._load_dataset(args)
        close = ds["close"].to_numpy() if hasattr(ds["close"], "to_numpy") else ds["close"]
        assert np.all(close > 0), (
            "合成价格里出现了非正数 —— cumprod 疑似被改成了 cumsum")
        assert 1.0 < float(np.nanmedian(close)) < 1e4, (
            f"合成价格中位数 {np.nanmedian(close)} 不在合理量级 —— 基准 100 的缩放被改了")
        # `cumprod` → `cumsum` 之后价格仍然为正（每步加 ≈1），量级也还在区间内 ——
        # 只断言"为正、量级对"抓不到它。随机游走的**日收益率**应当是
        # 均值≈0、标准差≈1% 的噪声；累加式会得到逐日递减的 1/t 型收益。
        ret = close[1:] / close[:-1] - 1.0
        assert abs(float(np.nanmean(ret))) < 0.005, (
            f"合成日收益均值 {np.nanmean(ret):.4f} 明显偏离 0 —— "
            f"cumprod 疑似变成了 cumsum（那会得到 1/t 型的确定性漂移）")
        assert 0.003 < float(np.nanstd(ret)) < 0.03, (
            f"合成日收益波动 {np.nanstd(ret):.4f} 不在 1% 量级 —— 不是随机游走了")

    def test_synthetic_prices_are_reproducible_to_the_exact_value(self, main_mod):
        """
        合成数据的 rng 种子写死为 42，所以整张面板是**逐位可复现**的。
        这让 `100 * np.cumprod(1 + rng.normal(...))` 这一行的三处算术
        （`*`、`1 +`、`cumprod`）都能用一个精确值钉住 ——
        分布层面的断言（均值/方差/正负）对 `*`→`/`、`1+`→`1-` 全都无能为力：
        倒数仍是随机游走、正态是对称的，两者的收益分布一模一样
        （上一版就是这么让这两个变异活下来的）。

        期望值来自独立复算：`np.random.default_rng(42).normal(0, 0.01, (T,N))`
        的第一个抽样是 0.0030471707975444，于是 close[0,0] = 100×1.0030471…
        """
        args = argparse.Namespace(use_synthetic=True, n_days=60, n_tickers=4)
        ds = main_mod._load_dataset(args)
        close = ds["close"]
        got = float(close.iloc[0, 0] if hasattr(close, "iloc") else close[0, 0])
        rng = np.random.default_rng(42)
        expected = 100.0 * (1.0 + rng.normal(0, 0.01, (60, 4))[0, 0])
        assert got == pytest.approx(expected, rel=1e-12), (
            f"合成面板首值 {got} != 独立复算的 {expected} —— "
            f"`100 * np.cumprod(1 + ...)` 这一行的算术被改过")

    def test_synthetic_high_low_bracket_the_close(self, main_mod):
        """
        `high = close * (1 + u)` / `low = close * (1 - u)` ——
        符号互换会让 high < low，所有价差/波幅估计变成负数。
        """
        args = argparse.Namespace(use_synthetic=True, n_days=60, n_tickers=6)
        ds = main_mod._load_dataset(args)
        to = lambda k: ds[k].to_numpy() if hasattr(ds[k], "to_numpy") else ds[k]
        high, low, close = to("high"), to("low"), to("close")
        assert np.all(high >= close), "high 低于 close —— (1+u) 疑似变成了 (1-u)"
        assert np.all(low <= close), "low 高于 close —— (1-u) 疑似变成了 (1+u)"
        assert np.all(high >= low)

    def test_synthetic_vwap_is_the_typical_price(self, main_mod):
        """`vwap = (high + low + close) / 3` —— 任一符号被改都会偏离。"""
        args = argparse.Namespace(use_synthetic=True, n_days=40, n_tickers=5)
        ds = main_mod._load_dataset(args)
        to = lambda k: ds[k].to_numpy() if hasattr(ds[k], "to_numpy") else ds[k]
        np.testing.assert_allclose(
            to("vwap"), (to("high") + to("low") + to("close")) / 3, rtol=1e-9,
            err_msg="合成 vwap 不是典型价格 (H+L+C)/3")


# ===========================================================================
# C. CLI 参数
# ===========================================================================

class TestCliArguments:
    """
    parser 内联在 `_cli_main()` 里，没有独立的工厂函数可调
    （上一版的 `_parser()` 找不到入口，四条用例全 skip，等于没测）。
    这里改成对 `add_argument` 调用做 AST 断言 —— 变异会改源码，
    所以这种断言同样杀得死那些 `default=False` / `required=True`。
    """

    @staticmethod
    def _add_argument_calls(main_mod):
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(main_mod._cli_main))
        out = []
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", "") == "add_argument"):
                flags = [a.value for a in node.args
                         if isinstance(a, ast.Constant) and isinstance(a.value, str)]
                kw = {k.arg: k.value for k in node.keywords}
                out.append((flags, kw))
        assert out, "没有解析到任何 add_argument 调用"
        return out

    def _kw_of(self, main_mod, flag):
        for flags, kw in self._add_argument_calls(main_mod):
            if flag in flags:
                return kw
        pytest.fail(f"CLI 里找不到参数 {flag}")

    @pytest.mark.parametrize("flag", ["--use-synthetic", "--walk-forward"])
    def test_dangerous_flags_default_to_false(self, main_mod, flag):
        """
        `--use-synthetic` 默认 True = 每次跑都用合成数据；
        `--walk-forward` 默认 True = 每次回测代价翻几倍。
        两个都必须显式 opt-in。
        """
        import ast
        kw = self._kw_of(main_mod, flag)
        assert kw["action"].value == "store_true", f"{flag} 不再是 store_true"
        default = kw.get("default")
        assert isinstance(default, ast.Constant) and default.value is False, (
            f"{flag} 的 default 不是 False")

    def test_the_mode_argument_is_required(self, main_mod):
        """
        `required=True` 改成 False 会让缺 `--mode` 时静默用 None 继续跑，
        随后在某个分支里以难懂的方式崩掉。
        """
        import ast
        required = [flags for flags, kw in self._add_argument_calls(main_mod)
                    if isinstance(kw.get("required"), ast.Constant)
                    and kw["required"].value is True]
        assert required, "CLI 里没有任何必填参数 —— required=True 疑似被改掉了"

    def test_oos_ratio_zero_skips_the_split_branch(self, main_mod):
        """
        `if args.oos_ratio > 0:` —— **严格大于 0**，0 表示"不切分"。
        上一版写的是 `assert (0.0 > 0) is False`，那是在测 Python 不是测产品。
        这里断言源码里的比较仍然是严格大于。
        """
        import inspect
        # 这个判定在 realistic 模式的执行函数里，不在 _cli_main 中 ——
        # 取整个模块的源码来找。
        src = inspect.getsource(main_mod)
        assert "if args.oos_ratio > 0:" in src, (
            "oos_ratio 的切分判定不再是 `> 0` —— 0（表示不切分）可能被当成要切分")


# ===========================================================================
# C2. realistic 模式的分支选择（行为断言，不是源码断言）
# ===========================================================================

class TestRealisticWalkForwardDefault:
    """
    `walk_forward = getattr(args, "walk_forward", False)` —— 默认值一旦翻成
    True，任何**没有**显式带 `--walk-forward` 的调用都会静默走多折验证：
    代价翻几倍，而且打印的 summary 是另一种报表，用户拿到的数字对不上。

    这里不用源码文本断言（`inspect.getsource` 走 linecache，测的是缓存不是
    真正执行的字节码），而是真的把 `run_realistic()` 跑一遍，看它选了哪个
    回测器。
    """

    @staticmethod
    def _args(**over):
        base = dict(
            use_synthetic    = True,
            n_days           = 40,
            n_tickers        = 3,
            delay            = 1,
            decay_window     = 0,
            truncation_min_q = 0.0,
            truncation_max_q = 1.0,
            portfolio_mode   = "long_short",
            top_pct          = 0.2,
            oos_ratio        = 0.0,
            dsl              = "cs_rank(close)",
        )
        base.update(over)
        return argparse.Namespace(**base)

    @staticmethod
    def _spy_backtesters(monkeypatch):
        """把两个回测器换成只记账的替身，返回被调用的类名列表。"""
        import app.core.backtest_engine.realistic_backtester as rb

        used: list[str] = []

        def _make(name):
            class _Spy:
                def __init__(self, *a, **kw):
                    used.append(name)

                def run(self, *a, **kw):
                    class _R:
                        @staticmethod
                        def summary():
                            return f"<{name} summary>"
                    return _R()
            return _Spy

        monkeypatch.setattr(rb, "RealisticBacktester",  _make("realistic"))
        monkeypatch.setattr(rb, "WalkForwardBacktester", _make("walkforward"))
        return used

    def test_a_namespace_without_the_flag_runs_the_single_pass_backtest(
            self, main_mod, monkeypatch, capsys):
        """
        Namespace 里**根本没有** walk_forward 这个属性 —— 只有 getattr 的
        默认值说了算。默认 False ⇒ 必须走单次 IS+OOS 回测。
        """
        used = self._spy_backtesters(monkeypatch)
        args = self._args()
        assert not hasattr(args, "walk_forward"), "用例前提被破坏"

        main_mod.run_realistic(args)

        assert used == ["realistic"], (
            f"没带 --walk-forward 却走了 {used} —— "
            f"getattr 的默认值不再是 False")
        assert "<realistic summary>" in capsys.readouterr().out

    def test_the_flag_when_present_still_selects_walk_forward(
            self, main_mod, monkeypatch, capsys):
        """反向用例：显式 True 时必须真的走多折，否则 `--walk-forward` 是摆设。"""
        used = self._spy_backtesters(monkeypatch)
        main_mod.run_realistic(self._args(walk_forward=True))
        assert used == ["walkforward"], f"显式打开却走了 {used}"
        assert "<walkforward summary>" in capsys.readouterr().out

    @pytest.mark.parametrize("attr,default", [("wf_splits", 5),
                                              ("embargo_days", 20)])
    def test_walk_forward_sub_defaults_reach_the_backtester(
            self, main_mod, monkeypatch, attr, default):
        """
        `wf_splits` / `embargo_days` 的 getattr 默认值也会被变异。它们只在
        walk_forward 分支里透传给 WalkForwardBacktester —— 抓构造参数。
        """
        import app.core.backtest_engine.realistic_backtester as rb

        seen: dict = {}

        class _Spy:
            def __init__(self, *a, **kw):
                seen.update(kw)

            def run(self, *a, **kw):
                class _R:
                    @staticmethod
                    def summary():
                        return ""
                return _R()

        monkeypatch.setattr(rb, "WalkForwardBacktester", _Spy)
        main_mod.run_realistic(self._args(walk_forward=True))

        key = {"wf_splits": "n_splits", "embargo_days": "embargo_days"}[attr]
        assert seen[key] == default, (
            f"{attr} 的 getattr 默认值变了：期望 {default}，实际 {seen[key]}")


# ===========================================================================
# D. sys.path 注入
# ===========================================================================

def test_backend_root_is_on_sys_path_exactly_once():
    """
    `if _ROOT not in sys.path:` —— 删掉 `not` 会让**已在路径里**时再插一次，
    重复导入时 sys.path 无限膨胀；反过来则永远不插，作为脚本运行时导不到包。
    """
    import sys
    import app.main as m
    root = m._ROOT
    assert root in sys.path, "backend 根目录没有进 sys.path"
    assert sys.path.count(root) == 1, (
        f"backend 根目录在 sys.path 里出现了 {sys.path.count(root)} 次 —— "
        f"`not in` 的守卫失效")
    importlib.reload(m)
    assert sys.path.count(m._ROOT) == 1, "重新导入后 sys.path 里出现了重复项"
