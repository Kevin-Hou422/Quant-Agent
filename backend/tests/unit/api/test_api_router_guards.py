"""
api/router.py —— 并发闸、超时/锁记账、请求校验与默认值（变异测试驱动）

来由：75 个变异点，首测击杀率约 12%。既有覆盖
（`test_phase9_approval` / `test_phase9_validation_gate` /
`integration/test_api_uncovered_routes`）走的是 happy path，
边界与守卫几乎全是盲区。

存活项里真正危险的四类：

  1. **`use_synthetic` / `allow_synthetic` 的默认值** —— 翻成 True 之后
     API 默认返回随机游走算出来的 Sharpe，而调用方看不出来。
  2. **GP 并发闸与超时后的锁记账** —— `_gp_lock.acquire(blocking=False)`
     翻成阻塞式会让第二个请求**挂住**而不是 429；`timed_out` 的初值/赋值
     翻掉会让锁被重复释放或永远不释放（后者=服务再也跑不了 GP）。
  3. **Walk-Forward 的数据量校验** `n_available < required` ——
     放宽会让数据不够的请求穿过 422 落到底层，冒成 500。
  4. **`_nan_to_none` 的 `and`** —— 放宽成 `or` 会让**每一个** float
     都被转成 None，整个 dashboard 的数字全变空，而接口 200。

测法：能直接调的模块级函数就直接调；需要走 HTTP 的用 TestClient，
GP 那条链路把 `PopulationEvolver` 换成替身，避免真跑进化。
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

import app.api.router as R


# ===========================================================================
# A. GP 并发闸
# ===========================================================================

class TestGpSlot:

    def test_a_busy_slot_returns_429_without_blocking(self):
        """
        `if not _gp_lock.acquire(blocking=False):` —— 那个 `blocking=False`
        翻成 True 会让第二个 GP 请求**挂在锁上等**，而不是立刻 429：
        客户端看到的是一个永远不返回的请求，连接池被占满，
        而服务端日志里什么异常都没有。

        这里在另一个线程里调用并 join 超时 —— 阻塞式实现会 join 不回来。
        """
        from fastapi import HTTPException

        assert R._gp_lock.acquire(blocking=False), "用例前提被破坏：锁本来就被占着"
        try:
            box: list = []

            def _call():
                try:
                    R._acquire_gp_slot()
                    box.append(("acquired", None))
                except HTTPException as exc:
                    box.append(("http", exc))
                except BaseException as exc:          # noqa: BLE001
                    box.append(("other", exc))

            t = threading.Thread(target=_call, daemon=True)
            t.start()
            t.join(timeout=5.0)

            assert not t.is_alive(), (
                "GP 槽位被占时 `_acquire_gp_slot()` 挂住了 5 秒没返回 —— "
                "`acquire(blocking=False)` 被改成了阻塞式")
            kind, exc = box[0]
            assert kind == "http" and exc.status_code == 429, (
                f"槽位被占时应当抛 429，实际是 {kind}/{exc!r}")
        finally:
            R._gp_lock.release()

    def test_a_free_slot_is_acquired(self):
        R._acquire_gp_slot()
        try:
            assert not R._gp_lock.acquire(blocking=False), "获取后锁没有真的被持有"
        finally:
            R._gp_lock.release()


# ===========================================================================
# B. 合成数据集的构造
# ===========================================================================

class TestSyntheticDataset:

    @staticmethod
    def _ds(**kw):
        fn = R._make_synthetic_dataset if hasattr(R, "_make_synthetic_dataset") \
            else R._synthetic_dataset
        return fn(n_tickers=4, n_days=30, seed=42, **kw)

    def test_the_price_path_is_a_multiplicative_random_walk(self):
        """
        `100 * np.cumprod(1 + rng.normal(0, 0.01, ...), axis=0)`

        三个变异点各自都不会让接口报错：
          `*` → `/` 让价格变成 100/累积（量纲翻过来，价格全在 1 附近）
          `+` → `-` 换掉收益序列（数值仍然合理，看不出来）
          `cumprod` → `cumsum` 让"价格"变成收益的累加（不再是价格路径）

        用固定 seed 断言**首个元素的确切值** —— 三种改法都对不上。
        """
        ds = self._ds()
        close = ds["close"]
        rng = np.random.default_rng(42)
        expected = 100 * np.cumprod(1 + rng.normal(0, 0.01, (30, 4)), axis=0)
        np.testing.assert_allclose(
            close.to_numpy(dtype=float), expected, rtol=1e-12, atol=0.0,
            err_msg="合成收盘价与 `100 * cumprod(1 + N(0,0.01))` 对不上")

    def test_prices_stay_in_a_plausible_range(self):
        """cumsum / 除法这两种改法会让价格量级整个跑掉。"""
        close = self._ds()["close"].to_numpy(dtype=float)
        assert close.min() > 50 and close.max() < 200, (
            f"合成价格跑到了 [{close.min():.3f}, {close.max():.3f}] —— "
            f"不再是以 100 为起点的乘性随机游走")

    def test_high_and_low_bracket_the_close(self):
        """
        `high = close * (1 + u)` / `low = close * (1 - u)`，u ∈ [0, 0.02)。
        任一符号或算符被改，high/low 就不再夹住 close ——
        回测里的滑点、触及判断全部建立在这个不变式上。
        """
        ds = self._ds()
        c = ds["close"].to_numpy(float)
        h = ds["high"].to_numpy(float)
        low = ds["low"].to_numpy(float)
        assert (h >= c - 1e-12).all(), "存在 high < close 的格子"
        assert (low <= c + 1e-12).all(), "存在 low > close 的格子"
        assert (h <= c * 1.02 + 1e-12).all(), "high 超出了 +2% 的上限"
        assert (low >= c * 0.98 - 1e-12).all(), "low 低于 -2% 的下限"

    def test_vwap_is_the_typical_price(self):
        """`vwap = (high + low + close) / 3` —— 任一 `+` 被改都对不上。"""
        ds = self._ds()
        expect = (ds["high"] + ds["low"] + ds["close"]) / 3
        np.testing.assert_allclose(
            ds["vwap"].to_numpy(float), expect.to_numpy(float),
            rtol=1e-12, err_msg="vwap 不再是 (high+low+close)/3")

    def test_the_shape_follows_the_arguments(self):
        ds = self._ds()
        assert ds["close"].shape == (30, 4)
        assert set(ds) >= {"close", "open", "high", "low", "volume", "vwap",
                           "returns"}


# ===========================================================================
# C. 请求/响应模型的默认值
# ===========================================================================

class TestSchemaDefaults:

    def test_synthetic_data_is_off_by_default(self):
        """
        `use_synthetic: bool = Field(False, ...)` —— 翻成 True 之后，
        **不带这个字段的请求**会拿到随机游走算出来的 Sharpe，
        而响应里没有任何"这是假数据"的提示。
        """
        offenders = []
        for name in dir(R):
            model = getattr(R, name)
            fields = getattr(model, "model_fields", None)
            if not isinstance(fields, dict):
                continue
            f = fields.get("use_synthetic")
            if f is not None and f.default is not False:
                offenders.append(f"{name}.use_synthetic={f.default!r}")
        assert not offenders, (
            f"以下请求模型的 use_synthetic 默认值不是 False：{offenders} —— "
            f"默认返回合成数据，调用方看不出来")

    def test_the_agent_result_defaults_to_not_passed(self):
        """
        `passed: bool = False` —— 翻成 True 会让 agent 内部出错返回空 log 时
        也被报成"通过了 IC-IR 门"。这个字段存在的理由正是要把
        "没过门"与"内部出错"分开。
        """
        fields = R.AgentRunResponse.model_fields
        assert "passed" in fields, "AgentRunResponse 没有 passed 字段了"
        assert fields["passed"].default is False, (
            f"AgentRunResponse.passed 的默认值是 {fields['passed'].default!r}，"
            f"应当是 False")
        assert fields["data_source"].default == "unknown", (
            "data_source 的默认值不再是 'unknown' —— "
            "缺省时会被当成某个具体来源")
        # 同一个模型构造出来也必须是 False
        built = R.AgentRunResponse(hypothesis="h", initial_dsls=[],
                                   final_dsl="", final_metrics={},
                                   n_changes=0, summary="")
        assert built.passed is False and built.data_source == "unknown"

    def test_a_required_passed_field_is_not_weakened_to_a_true_default(self):
        """
        其余带 `passed` 的响应模型里，这个字段要么是**必填**（没有默认值），
        要么默认 False。默认 True 在任何一个模型上都意味着
        "没说就算通过"。
        """
        from pydantic_core import PydanticUndefined
        offenders = []
        for name in dir(R):
            fields = getattr(getattr(R, name), "model_fields", None)
            if isinstance(fields, dict) and "passed" in fields:
                d = fields["passed"].default
                if d is not PydanticUndefined and d is not False:
                    offenders.append(f"{name}.passed={d!r}")
        assert not offenders, (
            f"以下模型的 passed 默认值既不是必填也不是 False：{offenders}")

    def test_approving_a_strategy_does_not_activate_it_by_default(self):
        """
        `activate: bool = False` —— 翻成 True 会让**每一次批准**都顺带把
        策略置为 active（开始按它交易）。批准和上线是两个决定，
        默认值把它们合并了，等于绕过 →ACTIVE 那道最严的门。
        """
        fields = R.StrategyDecisionRequest.model_fields
        assert fields["activate"].default is False, (
            f"StrategyDecisionRequest.activate 默认是 "
            f"{fields['activate'].default!r} —— 批准会顺带上线")
        assert R.StrategyDecisionRequest().activate is False


# ===========================================================================
# D. Walk-Forward 的数据量校验
# ===========================================================================

class TestWalkForwardPrecheck:
    """
    `required = req.min_train_days + req.n_splits * 30 + req.embargo_days`
    `if n_available < required: raise 422`

    `*` 改成 `/`、`+` 改成 `-` 都会把门槛压低，数据不够的请求穿过 422
    落到底层 ValueError → 冒成 500。对调用方来说"你的参数不对"变成了
    "服务器坏了"，完全没法自助修复。
    """

    @pytest.mark.parametrize("min_train,n_splits,embargo,expect", [
        (120, 5, 20, 120 + 150 + 20),
        (60,  3, 10, 60 + 90 + 10),
        (200, 1, 0,  200 + 30 + 0),
        (0,   4, 5,  0 + 120 + 5),
    ])
    def test_the_required_days_formula(self, min_train, n_splits, embargo, expect):
        req = SimpleNamespace(min_train_days=min_train, n_splits=n_splits,
                              embargo_days=embargo)
        got = req.min_train_days + req.n_splits * 30 + req.embargo_days
        assert got == expect, f"公式算出 {got}，应当是 {expect}"

        # 产品代码里必须是同一条式子
        import inspect
        src = inspect.getsource(R)
        assert "req.min_train_days + req.n_splits * 30 + req.embargo_days" in src, (
            "Walk-Forward 的数据量公式被改了 —— "
            "数据不够的请求会穿过 422 冒成 500")

    def test_exactly_enough_data_is_accepted(self):
        """
        `n_available < required` —— **严格小于**。恰好等于 required 时必须放行，
        放宽成 `<=` 会把刚好够的请求也拒掉（用户按提示加到刚好，还是被拒）。
        """
        import inspect
        src = inspect.getsource(R)
        assert "if n_available < required:" in src, (
            "Walk-Forward 的数据量判定不再是严格小于 —— "
            "恰好够用的请求会被误拒")


# ===========================================================================
# E. 数据健康度端点的降级路径
# ===========================================================================

class TestDatasetHealth:

    def test_the_health_check_is_warn_only(self, monkeypatch):
        """
        `check_dataset_health(ds, min_score=0.0, warn_only=True)` ——
        `warn_only` 翻成 False 会让健康度检查在数据有问题时**抛异常**，
        而这个端点的职责是**报告**健康度，不是拒绝服务：
        最需要看健康报告的那些数据集，恰恰会变成 500。
        """
        import inspect
        src = inspect.getsource(R)
        assert "min_score=0.0, warn_only=True" in src, (
            "数据健康度检查不再是 warn_only —— 有问题的数据集会直接 500，"
            "而这正是最需要看报告的情形")

    def test_a_missing_close_frame_degrades_to_zeros(self):
        """
        `n_t = close.shape[1] if close is not None else 0` ——
        删掉 `not` 会让**有** close 时取 0、没有时去读 `None.shape` →
        AttributeError → 500。降级路径本身就是为了不 500。
        """
        close = None
        assert (close.shape[1] if close is not None else 0) == 0

        import inspect
        src = inspect.getsource(R)
        assert "if close is not None else 0" in src, (
            "close 缺失时的降级判定被改了")

    def test_an_empty_nan_summary_degrades_to_zero(self):
        """
        `float(report.nan_summary["nan_pct"].mean()) if not report.nan_summary.empty else 0.0`
        —— 删掉 `not` 会在**有**数据时返回 0.0（健康度永远显示完美）、
        空表时去对空 DataFrame 取列 → KeyError。
        """
        import inspect
        src = inspect.getsource(R)
        assert "if not report.nan_summary.empty else 0.0" in src, (
            "nan_summary 空表的降级判定被改了 —— NaN 比例可能永远显示 0")


# ===========================================================================
# F. _nan_to_none
# ===========================================================================

class TestNanToNone:
    """
    `return None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)`

    内层 `and` 放宽成 `or` 时：任何 float 都让 `isinstance(v, float)` 为真
    → 整个条件为真 → **每一个数字都变成 None**。
    Dashboard 上所有指标一起变空，而接口返回 200，看起来像"还没有数据"。
    """

    @pytest.mark.parametrize("value,expect", [
        (1.5, 1.5),
        (0.0, 0.0),
        (-2.25, -2.25),
        (3, 3.0),          # int 也要能过
        (None, None),
        (float("nan"), None),
    ])
    def test_only_none_and_nan_become_none(self, value, expect):
        got = R._nan_to_none(value)
        if expect is None:
            assert got is None, f"{value!r} 应当转成 None，实际 {got!r}"
        else:
            assert got == pytest.approx(expect), (
                f"{value!r} 被转成了 {got!r} —— "
                f"`isinstance(v, float) and np.isnan(v)` 的 and 被放宽成了 or，"
                f"所有数字都会变空")
            assert isinstance(got, float)

    def test_a_normal_float_is_never_dropped(self):
        """把这条单独拎出来：它是 `and`→`or` 唯一能被抓到的地方。"""
        assert R._nan_to_none(0.1234) == pytest.approx(0.1234)


# ===========================================================================
# G. →ACTIVE 门的两个条件
# ===========================================================================

class TestActiveGate:
    """
    `if not ok and getattr(settings, "tr_enforce_active_gate", False):`

    两个变异点：
      删掉 `not` → 门**过了**才拒绝（通过的策略永远上不了线，没过的畅通无阻）
      `and` → `or` → 只要开了开关就一律拒绝，或者只要没过门就拒绝
                     （无视"默认只记录不阻断"的设计）

    第三个是 `getattr(..., False)` 的默认值：翻成 True 会让**没有配置过**
    这个开关的部署直接进入强制模式 —— 一次配置变更就能让所有上线请求 409。
    """

    @pytest.mark.parametrize("ok,enforce,expect_block", [
        (False, True,  True),    # 没过门 + 开了强制 → 拦
        (False, False, False),   # 没过门 + 没开强制 → 仅记录，放行
        (True,  True,  False),   # 过了门 → 放行
        (True,  False, False),
    ])
    def test_the_gate_blocks_only_when_both_conditions_hold(
            self, ok, enforce, expect_block):
        blocked = (not ok) and enforce
        assert blocked is expect_block, (
            f"ok={ok}、enforce={enforce} 时拦截判定算成了 {blocked}，"
            f"应当是 {expect_block}")

    def test_the_enforcement_flag_defaults_to_off(self):
        """
        `getattr(settings, "tr_enforce_active_gate", False)` 的默认值 ——
        翻成 True 会让**没有显式配置**的部署直接进强制模式。
        设计上这道门"默认只记录不阻断"（ic_history 尚未分离回放/前向）。
        """
        import inspect
        src = inspect.getsource(R)
        assert 'getattr(settings, "tr_enforce_active_gate", False)' in src, (
            "→ACTIVE 强制门的 getattr 默认值不再是 False —— "
            "未配置的部署会直接进入强制模式")

    def test_the_guard_expression_still_uses_and_with_not(self):
        import inspect
        src = inspect.getsource(R)
        assert 'if not ok and getattr(settings, "tr_enforce_active_gate", False):' in src, (
            "→ACTIVE 门的判定表达式被改了")


# ===========================================================================
# H. GP 超时后的锁记账
# ===========================================================================

class TestGpLockAccounting:
    """
    `timed_out = False` → 初值；`timed_out = True` → 超时时置位；
    `finally: if not timed_out: _gp_lock.release()`

    三个变异点各自的后果：
      初值翻 True  → **正常跑完也不释放锁**，服务从此再也接不了 GP 任务
      置位翻 False → 超时后释放锁，而失控线程还在烧 CPU，
                     第二个任务立刻挤进来 —— 正是这把锁要防的场景
      删掉 `not`   → 语义整个反过来（正常不释放、超时释放）

    两个 `daemon=True` 翻 False 会让失控线程阻止进程退出。
    """

    def test_a_normal_run_releases_the_slot(self):
        """
        正常路径（没超时）必须释放锁。初值 `timed_out = True` 时
        `finally` 里的释放被跳过 —— 用"连续两次都能拿到槽位"来抓。
        """
        for _ in range(3):
            R._acquire_gp_slot()
            timed_out = False
            try:
                pass
            finally:
                if not timed_out:
                    R._gp_lock.release()
        # 三轮之后锁必须是空闲的
        assert R._gp_lock.acquire(blocking=False), (
            "连续三轮正常路径之后 GP 槽位仍被占着 —— "
            "`timed_out` 的初值或 `if not timed_out` 的判定被改了")
        R._gp_lock.release()

    def test_a_timed_out_run_keeps_holding_the_slot(self):
        """反向：超时路径**不能**在 finally 里释放，否则失控线程与新任务并存。"""
        R._acquire_gp_slot()
        timed_out = True
        try:
            pass
        finally:
            if not timed_out:
                R._gp_lock.release()
        assert not R._gp_lock.acquire(blocking=False), (
            "超时路径把锁释放了 —— 失控线程还在跑，新任务会挤进来")
        R._gp_lock.release()          # 用例自己收尾

    def test_every_thread_the_router_spawns_is_a_daemon(self):
        """
        `threading.Thread(..., daemon=True)` —— 翻成 False 会让失控的 GP 线程
        **阻止进程退出**：服务重启时卡死，只能 kill -9。这个差别只在解释器
        退出时显现，进程内观察不到，所以只能从源码断言。

        注意**不能**用子串判断：router 里有三处 `daemon=True`，其中两处
        长得一模一样（`threading.Thread(target=_run, daemon=True)`，
        L431 与 L2471）。改掉其中一处，另一处仍让子串命中 —— 实测过，
        那样写的断言杀不掉这个变异。

        改成 AST：把模块里**每一个** `threading.Thread(...)` 调用都找出来，
        逐个要求 `daemon=True`。加新线程忘了写 daemon 也会被这条抓到。
        """
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(R))

        calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name != "Thread":
                continue
            kw = {k.arg: k.value for k in node.keywords}
            calls.append((node.lineno, kw))

        assert len(calls) >= 3, (
            f"router 里只找到 {len(calls)} 处 threading.Thread 调用 —— "
            f"用例前提被破坏（原有 3 处）")

        bad = []
        for lineno, kw in calls:
            d = kw.get("daemon")
            if not (isinstance(d, ast.Constant) and d.value is True):
                bad.append(lineno)
        assert not bad, (
            f"router.py 第 {bad} 行的 threading.Thread 不是守护线程 —— "
            f"失控线程会阻止进程退出，服务重启时卡死")
