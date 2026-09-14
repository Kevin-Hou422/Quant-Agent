"""
已登记但**尚未修复**的产品缺陷 —— 每条一个 `xfail(strict=True)` 用例。

为什么需要这个文件
------------------
变异测试查出的产品问题，此前只写在 `MUTATION_LEDGER.md` 里，并由各模块的
测试**钉住当前的错误行为**（断言错的值，注释写"修好之后改成对的值"）。
钉住现状有它的价值：任何一处算术被改坏仍然会被抓到。

但它有个致命的副作用：**已知坏掉的东西在每次运行里完全不可见**。
十几个已登记缺陷躺在那里，套件照样报 `1892 passed`，
任何人扫一眼都会得出"系统健康"的结论 —— 而这正是这一整个阶段要消灭的读数陷阱。
（台账里我自己写过"`N passed` 会被读成 N 件事被保护着"，然后又造了一个同样的陷阱。）

这个文件补上另一半：**断言应有行为，并标记为预期失败**。于是

  - 每次运行的汇总行都会显示 `N xfailed` —— 已知坏掉的数量摆在台面上
  - 谁把缺陷修好了，用例变成 XPASS，`strict=True` 让它**直接判失败**，
    强制修复者同步更新这里和对应模块里"钉住现状"的那条断言
  - 缺陷登记从一份没人会读的 md，变成可执行、跑得到的检查

两套断言是互补的，不是重复：
  - 模块里"钉住现状"的那条 → 提供**检出能力**（改坏立刻红）
  - 这里"断言应有行为"的 xfail → 提供**可见性与修复告警**

维护规则
--------
修好某个缺陷时必须做三件事，缺一不可：
  1. 删掉这里对应的 `xfail` 标记（用例转正）
  2. 改掉对应模块测试里"钉住现状"的那条断言
  3. 更新 `MUTATION_LEDGER.md` 里该条的状态
最后由 `test_defect_registry_matches_the_ledger` 保证条目数不漂移。
"""
from __future__ import annotations

from pathlib import Path


def _backend_root() -> Path:
    """
    向上找到含 `app/` 的目录 = backend/。

    **不要写成 `Path(__file__).resolve().parents[N]`**：层数一旦随目录重组
    变化，这里会静默指到错误的目录，`rglob("*.py")` 扫出空集合，
    而"对空集合的全称断言恒真" —— 约束静默失效且没有任何报错。
    """
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "app").is_dir():
            return parent
    raise RuntimeError(f"从 {p} 向上找不到含 app/ 的 backend 根目录")


import numpy as np
import pandas as pd
import pytest


#: 缺陷编号 → 一句话描述。新增/修复缺陷都必须同步这张表。
DEFECT_REGISTRY = {
    "B-1":  "ts_rank 在 bottleneck 分支的值域是 [-1/w, 1/w]，不是 docstring 承诺的 [0,1]",
    "B-2":  "ts_corr 因 cov(ddof=0)/std(ddof=1) 不配套而系统性偏低 (w-1)/w",
    "B-3":  "cs_rank 的并列处理是序数名次，不是 docstring 声称的平均名次",
    "B-4":  "ts_entropy(n_bins=1) 静默返回 -0.0，而不是 NaN 或报错",
    "B-5":  "ts_max / ts_min 的 NaN 策略在 bottleneck 与 numpy 分支之间不一致",
    "B-6":  "cs_rank 在含 NaN 的截面上值域越出 [0,1]",
    "B-7":  "面板行数短于窗口时 bottleneck 分支抛 ValueError，而非返回 NaN",
    "B-8":  "ProxyModel 在 _fit() 放弃之后仍走模型分支 → AttributeError",
    "B-9":  "use_label_encoder=False 对 xgboost 3.x 已无意义（仅代码整洁，无行为影响）",
    "B-10": "fast_ops 的向量化分支被 except Exception 完全兜住（结构问题，无行为断言）",
    "B-11": "data_partitioner 的『OOS 为空』守卫不可达（结构问题，无行为断言）",
    "B-12": "chat_store 的 ORDER BY 没有第二排序键，同一 tick 内的记录顺序反了",
    "A-1":  "ingest_incremental 把增量写进 PIT 两次（ingest() 内一次 + 外层一次）",
    "A-2":  "strategy_gate 用 `np.nanstd(...) == 0.0` 判零方差，浮点零判不出来，守卫从不触发",
    "A-3":  "PerformanceAnalyzer 对近零波动无防护：全常数收益算出年化 Sharpe ≈ 3e16",
    "A-4":  "PerformanceAnalyzer 遇非日期索引先在 _tdays 抛 AttributeError，"
            "max_drawdown 里的整数索引分支不可达，且报错指向内部实现",
    "A-5":  "MVOPortfolio 注释写『剔除的资产保留基准权重』，实现是整行替换 → 拿到 0",
    "A-6":  "PaperBroker.step 写死 target=1.0：把目标总敞口强行放大到 L1=1，"
            "且 ADV 削减后把亏空摊到其余名字（10% 空头变 99%）",
    "C-1":  "GP 适应度的截面秩用 argsort(argsort(x)) 算，不处理并列："
            "截面恒定（零信息）的信号被按**列顺序**摊成 0..n-1，"
            "IC 成了『ticker 在面板里的位置 vs 未来收益』的伪相关而非 0",
    "C-2":  "mutations._replace_node 先 deepcopy 再按 id(target) 找节点，"
            "而 deepcopy 后副本里没有任何节点持有那个 id —— 除非 target 就是 root，"
            "否则替换**永远静默失败**，返回原样副本。"
            "后果：hoist/wrap_rank/add_ts_smoothing/replace_subtree/subtree_crossover "
            "五个算子只能在根节点动手，add_ts_smoothing 在根不是数据/窄窗 TS 节点时"
            "是**彻底的空操作**（60 个种子只产出 1 种结果 = 原树）",
    "D-1":  "financial_interpreter 的取负识别只认 `neg` 节点："
            "DSL 的一元负号 `-x` 判为 reversion，而语义完全相同的 `(0-x)` "
            "判为 momentum —— 同一个因子换个等价写法就换了家族，"
            "GP 会据此配错互补家族与算子偏好",
    "D-2":  "LocalParquetProvider.available_fields() 对外宣称支持 `returns`，"
            "但 `returns` 从不落盘（不在 STANDARD_COLUMNS 里）。"
            "按宣称的字段清单调用 `fetch(fields=[..., 'returns'])` 时，"
            "列裁剪在 pyarrow 层直接失败 → `_read_ticker` 吞掉异常只发一条 warning → "
            "**整批数据返回空**（连 close 都没有），而不是只缺 returns 一项。"
            "调用方拿到 `{}`，看不出是自己要了一个不存在的列",
    "D-3":  "requirements.txt 写的是 `langchain>=0.2` 没有上界，"
            "而 langchain 1.x 已把 `AgentExecutor` / `create_tool_calling_agent` "
            "移出 `langchain.agents`。本机装的 1.2.15 满足该约束，"
            "于是 `_build_langchain_agent` 的 `except ImportError` 每次都命中，"
            "QuantAgent 只打一条 warning 就**静默降级到 FallbackOrchestrator** —— "
            "LLM 研究链路整条不可用，但 /api/chat 照常返回、前端毫无异样。"
            "而且报错文案是『需要安装 langchain』，实际 langchain 装着，"
            "真正的原因是大版本不兼容，按文案去装只会再装一遍同样的版本",
    "D-4":  "系统提示词把 `rank(neg(...))` 当作 **4 个因子家族**"
            "（反转 / 波动 / 流动性 / 价量相关）的标准 DSL 模板，"
            "但 `neg` 既不在提示词自己的 AVAILABLE OPERATORS 清单里，"
            "解析器也**不接受** `neg(x)` 这种函数写法（只认一元负号 `-x`）。"
            "LLM 照着模板写出来的 DSL 一律解析失败 → "
            "`_validate_and_fix` 白烧两次修复调用后放弃 → "
            "六个家族里有四个走模板路径时产出为零，"
            "对外只表现为『agent 老是生成非法公式』",
    "D-5":  "系统提示词写 `AlphaPool rejects signal-correlated alphas (corr > 0.9)`，"
            "而 `AlphaPool.__init__` 的默认 `corr_threshold=0.70`，"
            "判定用的是 `abs(corr) >= threshold`。"
            "数字（0.9 vs 0.70）与开闭（> vs >=）两处都对不上 —— "
            "代码注释自己写着『Lowered from 0.90 to 0.70 (Task 3.5)』，"
            "提示词没跟着改。LLM 会据此误判哪些因子算『足够正交』",
}

#: 只是结构/整洁问题，没有可执行的行为断言 —— 记录在案，不设 xfail 用例。
NO_BEHAVIOUR_ASSERTION = {"B-9", "B-10", "B-11"}


def _xfail(defect_id: str):
    return pytest.mark.xfail(strict=True,
                             reason=f"{defect_id}：{DEFECT_REGISTRY[defect_id]}")


# ===========================================================================
# fast_ops —— 算子内核
# ===========================================================================

class TestFastOpsDefects:

    @_xfail("B-1")
    def test_ts_rank_should_be_a_percentile_in_zero_one(self):
        """单调上升序列的滚动排名，最新一根是窗口内最高，应当是 1.0。"""
        import app.core.alpha_engine.fast_ops as F
        x = np.arange(20, dtype=float).reshape(20, 1)
        assert F.bn_ts_rank(x, 5).ravel()[-1] == pytest.approx(1.0)

    @_xfail("B-1")
    def test_ts_rank_should_agree_across_execution_paths(self):
        import app.core.alpha_engine.fast_ops as F
        x = np.arange(20, dtype=float).reshape(20, 1)
        real = F._HAS_BN
        try:
            F._HAS_BN = True
            a = F.bn_ts_rank(x, 5)
            F._HAS_BN = False
            b = F.bn_ts_rank(x, 5)
        finally:
            F._HAS_BN = real
        np.testing.assert_allclose(a, b, equal_nan=True)

    @_xfail("B-2")
    def test_ts_corr_of_perfectly_correlated_series_should_be_one(self):
        import app.core.alpha_engine.fast_ops as F
        rng = np.random.default_rng(0)
        a = rng.normal(size=(60, 1))
        b = a * 2.0 + 1.0
        assert F.ts_corr(a, b, 20).ravel()[-1] == pytest.approx(1.0, abs=1e-9)

    @_xfail("B-3")
    def test_cs_rank_should_give_ties_the_average_rank(self):
        """docstring: "ties resolved by average rank"。[1,1,2,3] 的前两名应当并列。"""
        import app.core.alpha_engine.fast_ops as F
        got = F.cs_rank(np.array([[1.0, 1.0, 2.0, 3.0]])).ravel()
        assert got[0] == pytest.approx(got[1])

    @_xfail("B-3")
    def test_equal_values_should_get_equal_ranks_regardless_of_position(self):
        """
        并列值拿到的名次只取决于它在数组里的下标 —— 同一只标的换个列位置
        就换个因子值。这里用**三个**并列值，它们应当拿到同一个名次。

        （上一版写的是"换列顺序后某个下标的名次应当相等"，那条恰好恒真，
        strict xfail 当场报 XPASS —— 断言没测到点子上。）
        """
        import app.core.alpha_engine.fast_ops as F
        got = F.cs_rank(np.array([[5.0, 5.0, 5.0, 9.0]])).ravel()
        assert got[0] == pytest.approx(got[1]) == pytest.approx(got[2])

    @_xfail("B-4")
    def test_ts_entropy_with_one_bin_should_not_return_zero(self):
        """n_bins=1 是退化参数，应当是 NaN 或报错，不能静默给 -0.0。"""
        import app.core.alpha_engine.fast_ops as F
        got = F.ts_entropy(np.arange(10, dtype=float).reshape(10, 1), 5, n_bins=1)
        assert np.all(np.isnan(got))

    @_xfail("B-5")
    def test_ts_max_nan_policy_should_match_across_paths(self):
        """模块 docstring 承诺 strict NaN policy，两条分支必须一致。"""
        import app.core.alpha_engine.fast_ops as F
        x = np.array([[1.0], [2.0], [np.nan], [4.0], [5.0], [6.0]])
        real = F._HAS_BN
        try:
            F._HAS_BN = True
            a = F.bn_ts_max(x, 3)
            F._HAS_BN = False
            b = F.bn_ts_max(x, 3)
        finally:
            F._HAS_BN = real
        np.testing.assert_allclose(a, b, equal_nan=True)

    @_xfail("B-6")
    def test_cs_rank_should_stay_within_zero_one_with_nan(self):
        import app.core.alpha_engine.fast_ops as F
        got = F.cs_rank(np.array([[10.0, np.nan, 30.0, 40.0]])).ravel()
        assert np.nanmax(got) == pytest.approx(1.0)
        assert np.nanmin(got) == pytest.approx(0.0)

    @_xfail("B-7")
    def test_short_panels_should_return_nan_not_raise(self):
        """docstring 承诺"不足 window 个有效观测 → NaN"。"""
        import app.core.alpha_engine.fast_ops as F
        x = np.arange(6, dtype=float).reshape(3, 2)
        assert np.all(np.isnan(F.bn_ts_mean(x, 5)))


# ===========================================================================
# ml_engine —— 代理模型
# ===========================================================================

class TestProxyModelDefects:

    @_xfail("B-8")
    def test_unfitted_model_should_fall_back_to_the_cold_start_rule(self):
        """
        `_fit()` 因单一类别放弃后 `_model` 是 None，`should_prune` 仍走模型分支
        → AttributeError 打断整轮 GP。应当退回冷启动规则。
        """
        pytest.importorskip("xgboost")
        from app.core.alpha_engine.typed_nodes import DataNode, TimeSeriesNode
        from app.core.ml_engine.proxy_model import ProxyModel

        def _chain(depth):
            node = DataNode("close")
            for _ in range(depth):
                node = TimeSeriesNode("ts_mean", node, 5)
            return node

        pm = ProxyModel(cold_start_n=4)
        for i in range(6):
            pm.update(_chain(2 + i % 3), failed=True)     # 标签全是 1
        assert pm.should_prune(_chain(1)) is True          # 深度 1 → 规则判剪
        assert pm.should_prune(_chain(3)) is False


# ===========================================================================
# db / tasks
# ===========================================================================

class TestStorageDefects:

    @_xfail("B-12")
    def test_sessions_with_equal_timestamps_should_keep_newest_first(self):
        """
        `list_sessions()` 是 ORDER BY created_at DESC，没有第二排序键。
        并列时 SQLite 按插入顺序返回 —— 最老的排最前，与契约相反。
        """
        import tempfile
        from datetime import datetime
        from pathlib import Path

        import app.db.chat_store as mod
        from app.db.chat_store import ChatStore

        base = datetime(2026, 1, 1)

        class _Frozen(datetime):
            @classmethod
            def utcnow(cls):
                return base

        real = mod.datetime
        try:
            mod.datetime = _Frozen
            d = Path(tempfile.mkdtemp())
            st = ChatStore(db_url=f"sqlite:///{d / 't.db'}")
            st.create_session("A")
            b = st.create_session("B")
            assert st.list_sessions()[0].id == b.id
        finally:
            mod.datetime = real

    @_xfail("A-1")
    def test_incremental_ingest_should_write_each_bar_once(self):
        """
        `ingest_incremental` 调用的 `ingest()` 内部已把整段增量窗口写过一遍 PIT，
        随后外层又 `_append_pit(increment)` 写一次 —— 每个增量日两个 vintage。
        对照见 test_daily_ingest_increment.py::test_pit_only_receives_the_increment，
        那里钉住的是**当前**的 2，这里断言的是**应有**的 1。
        """
        # 上一版整条就是一句 `pytest.skip(...)` —— **一条断言都没有**，
        # 于是它既不检查缺陷是否还在，也不会在缺陷被修好时提醒任何人
        # （test_lessons_enforced 的零断言检查抓到了这一点）。
        #
        # 这里改成断言**缺陷的成因仍在源码里**：`ingest_incremental` 里
        # 先调 `ingest(...)`（它内部已把整段增量写过一遍 PIT），
        # 随后又 `_append_pit(increment)` 写第二遍。
        # 修好之后这两处不会同时存在 → xfail 变 XPASS → strict 判失败 →
        # 强制同步更新登记表、台账与钉住现状的那条用例。
        import inspect
        from app.tasks import daily_ingest as di

        src = inspect.getsource(di.ingest_incremental)
        writes_twice = ("ingest(" in src) and ("_append_pit(" in src)
        assert not writes_twice, (
            "ingest_incremental 仍然是 `ingest(...)` 之后再 `_append_pit(...)` —— "
            "每个增量日会写进 PIT 两次（两个 vintage）。\n"
            "当前行为由 tests/test_daily_ingest_increment.py::"
            "test_pit_only_receives_the_increment 钉住（那里断言的是 2），"
            "这里断言的是**应有**的 1。")


# ===========================================================================
# A 档 —— 组合/回测/执行
# ===========================================================================

class TestBacktestAndExecutionDefects:

    @staticmethod
    def _result(rets: pd.Series):
        from app.core.backtest_engine.backtest_engine import BacktestResult
        n = len(rets)
        cols = ["A", "B"]
        zeros = pd.DataFrame(0.0, index=rets.index, columns=cols)
        return BacktestResult(
            equity_curve=(1 + rets).cumprod(), gross_returns=rets, net_returns=rets,
            positions=zeros, trade_log=pd.DataFrame(), turnover=rets * 0.0,
            signal=zeros, daily_cost_bps=rets * 0.0)

    @_xfail("A-2")
    def test_zero_variance_guard_should_use_a_tolerance_not_equality(self):
        """
        `float(np.nanstd(rets.values)) == 0.0` —— 用**精确相等**判浮点零。

        精确恒定的数组 `np.nanstd` 确实返回 0.0（我第一版就是这么构造的，
        strict xfail 当场报 XPASS）。真正出问题的是"浮点意义上恒定、
        但带 1e-19 量级残渣"的序列 —— 回测算出来的净收益正是这种：
        多空两腿相减、成本逐日重算，残渣必然非零。
        那时 `== 0.0` 为假，守卫放行，随后 `vol > 0` 也成立，
        算出年化 Sharpe 3e16（见 A-3），报告里显示"高度显著"。

        应当改成带容差的判定（如 `< 1e-12`）。
        """
        import inspect
        from app.core.portfolio_manager import strategy_gate as sg

        rets = np.full(60, 0.001) + np.linspace(0, 1e-18, 60)
        assert float(np.nanstd(rets)) != 0.0, "构造的序列方差恰好为零，测不到这个缺陷"

        src = inspect.getsource(sg)
        assert "np.nanstd(rets.values)) == 0.0" not in src, (
            "零方差守卫仍然用 `== 0.0` 判浮点零")

    @_xfail("A-3")
    def test_near_zero_volatility_should_not_produce_an_astronomical_sharpe(self):
        """
        `return ... / vol if vol > 0 else np.nan` —— 全常数收益的年化波动是
        1.06e-17，`vol > 0` 成立，于是算出年化 Sharpe ≈ 3e16、t ≈ 15.5，
        在报告里显示为"高度显著"。近零波动应当判为无意义。
        """
        from app.core.backtest_engine.performance_analyzer import PerformanceAnalyzer
        idx = pd.bdate_range("2024-01-02", periods=120)
        rets = pd.Series(np.full(120, 0.001), index=idx)
        sr = PerformanceAnalyzer(self._result(rets)).sharpe_ratio()
        assert np.isnan(sr) or abs(sr) < 1e3, (
            f"全常数收益算出了 Sharpe={sr}")

    @_xfail("A-4")
    def test_a_non_datetime_index_should_give_a_clear_error(self):
        """
        `_tdays` 里 `(idx[-1] - idx[0]).days` 对整数索引直接抛 AttributeError，
        报错指向内部实现而非"索引类型不对"；而 `max_drawdown` 里那条
        按序号相减的 else 分支因此**永远走不到**。
        """
        from app.core.backtest_engine.performance_analyzer import PerformanceAnalyzer
        rets = pd.Series(np.full(60, 0.001), index=range(60))
        with pytest.raises((TypeError, ValueError)) as exc:
            PerformanceAnalyzer(self._result(rets)).sharpe_ratio()
        assert "index" in str(exc.value).lower() or "索引" in str(exc.value)

    @_xfail("A-5")
    def test_dropped_assets_should_keep_their_benchmark_weight(self):
        """
        注释写"剔除 NaN 过多的资产（保留其基准权重）"，实现是
        `w_out[t] = row / l1` **整行替换** —— 被剔除的资产拿到 0。
        注释与实现必须一致（要么改注释，要么改实现）。
        """
        import inspect
        from app.core.backtest_engine import portfolio_constructor as pc
        src = inspect.getsource(pc.MVOPortfolio)
        assert "保留其基准权重" not in src or "w_out[t] = row / l1" not in src, (
            "注释仍写着『保留基准权重』，实现仍是整行替换 —— 两者对不上")

    @_xfail("A-6")
    def test_paper_broker_should_not_hardcode_a_unit_gross_target(self):
        """
        `project_to_capped_l1(..., target=1.0)` 里的字面量 1.0 写死，
        不看传进来的 `tgt` 实际总敞口。后果有两个：

          1. 目标 gross 0.5 落账变成 1.0 —— 上游所有降敞口决定
             （波动率目标、max_gross、无交易带）被这一步抹掉
          2. 某只票被 ADV 上限削掉时，water-filling 把亏空摊到其余名字
             以凑满 L1=1 —— 实测 [0.9, -0.1] 在 A 只能买 0.01 时
             落账 [0.01, -0.99]，10% 的空头变成 99%

        应当把 target 设成 `np.abs(tgt).sum()`（保持上游意图的总敞口）。
        """
        import inspect
        from app.core.execution import paper_broker as pb
        src = inspect.getsource(pb.PaperBroker.step)
        assert "target=1.0" not in src, (
            "PaperBroker.step 仍然写死 target=1.0")


# ===========================================================================
# gp_engine —— GP 适应度
# ===========================================================================

class TestGpFitnessRankTies:

    @_xfail("C-1")
    def test_a_cross_sectionally_constant_signal_should_score_zero_ic(self):
        """
        `rs = np.argsort(np.argsort(s[mask])).astype(float)` —— 这是**序数**
        名次，不是处理并列的平均名次。当日信号对所有票同值时，它给出的是
        0, 1, 2, … 也就是**列在面板里的位置**。

        后果：一个截面恒定、零信息的信号（`(close/close)` 恒为 1）拿到的
        mean_IC 不是 0，而是"ticker 排列顺序 vs 未来收益秩"的相关系数。
        把面板的列顺序打乱，同一个信号的 fitness 会变 —— GP 的分数依赖
        数据加载时的列序，而列序不是市场事实。

        正确做法：并列取平均名次（`scipy.stats.rankdata`，或手写 tie-average），
        此时常数信号的 rs 去中心化后全 0 → denom = 0 → 该截面被
        `if denom > 0:` 正确跳过。
        """
        import numpy as np
        import app.core.gp_engine.gp_engine as G

        T_, N_ = 60, 10
        rng = np.random.default_rng(99)
        close = 100 * np.cumprod(1 + rng.normal(0, 0.01, (T_, N_)), axis=0)
        ds = {
            "close": close, "open": close, "vwap": close,
            "high": close * 1.01, "low": close * 0.99,
            "volume": np.full_like(close, 1e6),
        }

        res = G._evaluate_individual("(close/close)", ds)
        assert res.ann_return == pytest.approx(0.0, abs=1e-9), (
            f"截面恒定（零信息）的信号算出 mean_IC = {res.ann_return:.4f}，"
            f"不是 0 —— 并列名次被按列顺序摊开了")


# ===========================================================================
# mutations —— GP 结构变异算子
# ===========================================================================

class TestReplaceNodeIdentity:

    @_xfail("C-2")
    def test_replacing_an_internal_node_should_actually_replace_it(self):
        """
        ```python
        root_copy = copy.deepcopy(root)
        _replace_inplace(root_copy, id(target), copy.deepcopy(replacement))
        ```

        `id(target)` 是**原树**里那个对象的地址。`deepcopy` 造出来的是一套
        全新对象，副本里没有任何节点的 `id()` 等于它 —— 于是
        `_replace_inplace` 一路找不到，返回 False，函数把**原样副本**交回去。
        没有异常、没有日志，调用方拿到一棵"变异过"的树，其实一个字没改。

        唯一能成的是 `root is target` 那条捷径（函数第一行），
        所以五个结构算子实际上只能在**根节点**动手：

          - `add_ts_smoothing`：候选是 DataNode / 窄窗 TS 节点，根通常不在候选里
            → 对 `rank(ts_mean(close,10))` 这种树，60 个随机种子只产出
              **1 种**结果（原树），是彻底的空操作
          - `wrap_rank`：只能整棵包 `rank(...)`，永远包不到内部节点
          - `hoist_mutation` / `replace_subtree` / `subtree_crossover`：
            只在随机选中根时才真的动

        GP 的结构搜索空间因此塌缩到"整棵树包一层"，而适应度曲线看起来一切正常。

        修法：把 target 换成**路径/索引**定位，或者先在原树上找到位置再整体
        deepcopy，不要跨 deepcopy 传 `id()`。
        """
        import app.core.gp_engine.mutations as M
        from app.core.alpha_engine.typed_nodes import (
            CrossSectionalNode, DataNode, TimeSeriesNode,
        )

        root = CrossSectionalNode(
            "rank", TimeSeriesNode("ts_mean", DataNode("close"), 10))
        leaf = M._collect_nodes(root)[-1]
        assert repr(leaf) == "close", "用例前提被破坏"

        out = M._replace_node(root, leaf, DataNode("volume"))
        assert repr(out) == "rank(ts_mean(volume,10))", (
            f"把内部叶子 close 换成 volume，得到的却是 {out!r} —— "
            f"deepcopy 之后按 id() 找节点永远找不到，替换静默失败")

    @_xfail("C-2")
    def test_add_ts_smoothing_should_not_be_a_no_op(self):
        """
        C-2 最刺眼的表现：`add_ts_smoothing` 对一棵完全正常的树，
        跑 60 个不同随机种子，**产出唯一一种结果 —— 原树**。
        """
        import app.core.gp_engine.mutations as M
        from app.core.gp_engine import _rng
        from app.core.alpha_engine.typed_nodes import (
            CrossSectionalNode, DataNode, TimeSeriesNode,
        )

        saved = _rng.current()
        try:
            root = CrossSectionalNode(
                "rank", TimeSeriesNode("ts_mean", DataNode("close"), 10))
            outs = set()
            for s in range(60):
                _rng.bind_seed(s)
                outs.add(repr(M.add_ts_smoothing(root)))
        finally:
            _rng.bind(saved)

        assert outs != {repr(root)}, (
            f"60 个随机种子下 add_ts_smoothing 只产出了原树本身 —— "
            f"这个算子是彻底的空操作")


# ===========================================================================
# 系统提示词 —— 它写的规则，引擎认不认
# ===========================================================================

class TestSystemPromptDslExamples:

    @_xfail("D-4")
    def test_the_documented_factor_patterns_all_parse(self):
        """
        提示词的 FINANCIAL FACTOR TAXONOMY 给每个因子家族配了一条
        `DSL pattern:` —— 那是 LLM 的主要模仿对象。

        其中四条用了 `rank(neg(...))`：
          反转   rank(neg(ts_delta(close, N)))
          波动   rank(neg(ts_std(returns, 20)))
          流动性 rank(neg(ts_mean(volume, 20)))
          价量   rank(neg(ts_corr(close, volume, 20)))

        而解析器只认一元负号（`-x`），不存在 `neg(x)` 这个函数；
        `neg` 也不在提示词自己的 AVAILABLE OPERATORS 清单上。

        后果：LLM 照模板产出的公式一律解析失败 →
        `_validate_and_fix` 白烧两次修复调用后放弃 →
        六个家族里有四个走模板路径时产出为零，
        对外只表现为"agent 老是生成非法公式"。

        修法很轻：把模板里的 `neg(x)` 改写成 `-x`
        （`tests/unit/agent/test_system_prompt_consistency.py::
        test_the_unary_minus_form_is_what_the_parser_accepts` 已验证这条路通）。
        """
        from app.core.alpha_engine.parser import Parser

        parser = Parser()
        patterns = [
            "rank(neg(ts_delta(close, 5)))",
            "rank(neg(ts_std(returns, 20)))",
            "rank(neg(ts_mean(volume, 20)))",
            "rank(neg(ts_corr(close, volume, 20)))",
        ]
        assert len(patterns) == 4, "四个家族各一条模板"
        for dsl in patterns:
            node = parser.parse(dsl)
            assert node is not None, f"{dsl} 解析出空节点"

    @_xfail("D-4")
    def test_every_operator_used_in_the_prompt_is_also_declared_there(self):
        """
        提示词内部自洽：正文里当范例用的算子，必须出现在它自己的
        AVAILABLE OPERATORS 清单上。现在 `neg` 只在范例里出现，
        LLM 拿到的是自相矛盾的两份说明。
        """
        import re

        from app.agent._prompts import _SYSTEM_PROMPT as P

        i = P.index("AVAILABLE OPERATORS:")
        lines = P[i + len("AVAILABLE OPERATORS:"):].splitlines()
        taken = [lines[0]]
        for ln in lines[1:]:
            if not ln.strip() or not ln.startswith((" ", "\t")):
                break
            taken.append(ln)
        declared = {t.strip() for t in " ".join(taken).split(",") if t.strip()}

        assert "neg(" in P, "提示词里已经不用 neg 了 —— 请同步删除缺陷 D-4"
        assert "neg" in declared, (
            f"`neg` 在范例里被使用，却不在 AVAILABLE OPERATORS 清单里："
            f"{sorted(declared)}")


class TestSystemPromptThresholds:

    @_xfail("D-5")
    def test_the_correlation_threshold_matches_the_alpha_pool_default(self):
        """
        提示词：`AlphaPool rejects signal-correlated alphas (corr > 0.9)`
        代码：  `AlphaPool.__init__(corr_threshold: float = 0.70)`，
                判定是 `abs(corr) >= corr_threshold`

        两处都对不上：数字差 0.2，开闭也相反。
        `alpha_pool.py` 的注释自己写着
        "Lowered from 0.90 to 0.70 (Task 3.5)" —— 提示词没跟着改。

        后果：LLM 按 0.9 去判断"这两条因子够不够正交"，
        而池子实际按 0.70 拒收。它会反复产出自以为合格、
        实际被静默拒绝的候选，且拿不到任何反馈。
        """
        import inspect
        import re

        from app.agent._prompts import _SYSTEM_PROMPT as P
        from app.core.gp_engine.alpha_pool import AlphaPool

        m = re.search(r"corr\s*>\s*([\d.]+)", P)
        assert m, "提示词里找不到相关度阈值"
        default = inspect.signature(
            AlphaPool.__init__).parameters["corr_threshold"].default
        assert float(m.group(1)) == pytest.approx(default), (
            f"提示词写 corr > {m.group(1)}，代码默认 {default}")


# ===========================================================================
# LangChain 依赖版本 —— LLM 研究链路是否真的在跑
# ===========================================================================

class TestLangChainWiringIsAlive:

    @_xfail("D-3")
    def test_the_langchain_agent_can_actually_be_built(self):
        """
        `_build_langchain_agent` 在**当前已安装的依赖**下必须能走到
        `create_tool_calling_agent`，而不是在第一个 import 就掉进
        `except ImportError`。

        现状：`requirements.txt` 只写了 `langchain>=0.2`，
        装上的 1.2.15 已经把 `AgentExecutor` / `create_tool_calling_agent`
        移出 `langchain.agents`。于是这个函数**每次都抛 ImportError**，
        `QuantAgent.__init__` 打一条 warning 就退到 FallbackOrchestrator。

        对外表现：`/api/chat` 照常工作、前端毫无异样 ——
        LLM 研究链路整条死掉，却没有任何可见信号。
        （交易回路本来就不含 LLM，所以不影响下单；影响的是因子发现。）

        这条不碰网络、不需要 API key：只要 import 能成功、
        能构造出 AgentExecutor，就算通过。
        """
        import app.agent._lc_agent as LC

        try:
            from langchain.agents import AgentExecutor, create_tool_calling_agent  # noqa: F401
            from langchain.tools import tool as lc_tool                            # noqa: F401
            from langchain_core.prompts import ChatPromptTemplate                  # noqa: F401
        except ImportError as exc:          # pragma: no cover - 这正是缺陷本身
            pytest.fail(
                f"当前安装的 langchain 无法提供 _lc_agent 需要的符号：{exc}。"
                f"requirements.txt 的 `langchain>=0.2` 没有上界，"
                f"装上的大版本与代码不兼容 —— LLM 链路静默降级。")

        assert callable(LC._build_langchain_agent)

    def test_the_import_failure_message_names_a_version_conflict(self):
        """
        **钉住现状的另一半**：即使版本冲突短期不修，
        报错文案也不该把人引向"再装一遍 langchain"。

        这条**不是** xfail —— 它描述的是当前文案，
        一旦有人把文案改成提到版本，这里会红，提醒同步更新 D-3。
        """
        import inspect

        import app.agent._lc_agent as LC

        src = inspect.getsource(LC._build_langchain_agent)
        assert "pip install langchain" in src, (
            "报错文案变了 —— 如果已经改成提示版本冲突，请同步更新缺陷 D-3")


# ===========================================================================
# LocalParquetProvider —— 宣称的字段与实际可读的字段
# ===========================================================================

class TestLocalParquetAdvertisedFields:

    @_xfail("D-2")
    def test_every_advertised_field_can_actually_be_requested(self, tmp_path):
        """
        `available_fields()` 是 provider 对外的**字段契约**，
        调用方（DataManager / DatasetRegistry）按它决定要什么。

        但 `returns` 只在这份清单里，从来没进过 parquet ——
        一旦按契约请求它，`_read_ticker` 的列裁剪在 pyarrow 层抛
        `No match for FieldRef.Name(returns)`，被 `except` 吞成一条 warning，
        于是**这个 ticker 的所有分区都读不出来**。

        后果不是"少一列 returns"，而是 `fetch` 返回 `{}` ——
        连 close 都没有。日循环拿到空面板会当成"今天没有数据"。
        """
        import warnings

        from app.core.data_engine.local_parquet_provider import LocalParquetProvider

        prov = LocalParquetProvider(tmp_path / "store")
        idx = pd.bdate_range("2022-01-03", periods=4)
        prov.write(pd.DataFrame({
            "timestamp": idx, "ticker": ["AAA"] * 4,
            "open": [1.0] * 4, "high": [2.0] * 4, "low": [0.5] * 4,
            "close": [1.5] * 4, "volume": [1e6] * 4, "vwap": [1.4] * 4,
            "adj_factor": [1.0] * 4,
        }))

        advertised = prov.available_fields()
        assert "returns" in advertised, "前提变了：returns 不再被宣称支持"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = prov.fetch(["AAA"], "2022-01-01", "2022-12-31",
                            fields=["close", "returns"])

        assert "close" in ds, (
            f"按 available_fields() 的契约请求 returns，结果连 close 都没拿到："
            f"{sorted(ds)} —— 一个不可读的宣称字段让整批数据归零")


# ===========================================================================
# financial_interpreter —— 因子家族分类
# ===========================================================================

class TestFinancialInterpreterNegation:

    @_xfail("D-1")
    def test_both_equivalent_negation_forms_give_the_same_family(self):
        """
        `_is_inverted_momentum` 只认 `ArithmeticNode(op="neg")`：

          `-ts_delta(close,5)`      → 解析成 neg  → 判 **reversion** ✅
          `(0-ts_delta(close,5))`   → 解析成 sub  → 判 **momentum**  ❌

        两者在金融上完全是同一个因子（买跌卖涨的反转信号）。
        换个等价写法就换了家族，而 `factor_family` 被 GP 消费：
        `_COMPLEMENTARY_FAMILIES` 决定配什么互补因子、
        `_FAMILY_WEIGHT_BIASES` 决定算子权重 —— 全都会配错。

        影响面：GP 的种子库与 `mutations.py` 都用 `neg` 形式，主链路没问题；
        但 Workflow B 吃的是**用户输入的 DSL**，用户完全可能写 `(0-x)`。

        修法：`_is_inverted_momentum` 除了 `op == "neg"`，还要认
        `op == "sub"` 且左操作数是值为 0 的 ScalarNode 的情形。
        """
        from app.core.alpha_engine.financial_interpreter import FinancialInterpreter

        it = FinancialInterpreter()
        unary = it.interpret("rank(-ts_delta(close,5))").factor_family
        zero_minus = it.interpret("rank((0-ts_delta(close,5)))").factor_family
        assert unary == zero_minus, (
            f"`-x` 判为 {unary}，而语义等价的 `(0-x)` 判为 {zero_minus} —— "
            f"同一个因子换写法就换了家族")


# ===========================================================================
# 登记表自身的一致性
# ===========================================================================

def test_defect_registry_matches_the_ledger():
    """
    登记表里的每个编号都必须能在 `MUTATION_LEDGER.md` 里找到 ——
    防止这里和台账各说各话。
    """
    import pathlib
    import re
    ledger = (_backend_root() / "MUTATION_LEDGER.md").read_text(encoding="utf-8")
    missing = [d for d in DEFECT_REGISTRY
               if d.startswith("B-") and not re.search(rf"\bB-{d[2:]}\b", ledger)]
    assert not missing, (
        f"以下缺陷编号在台账里找不到：{missing} —— 两边已经不同步")


def test_every_behavioural_defect_has_an_xfail_case():
    """
    除了明确标注"无行为断言"的那几条，每个缺陷编号都必须有至少一个
    `xfail` 用例。新登记了缺陷却忘了在这里加用例，这条会红。
    """
    import ast
    import pathlib
    src = pathlib.Path(__file__).read_text(encoding="utf-8")
    covered = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_xfail":
            if node.args and isinstance(node.args[0], ast.Constant):
                covered.add(node.args[0].value)
    expected = set(DEFECT_REGISTRY) - NO_BEHAVIOUR_ASSERTION
    missing = expected - covered
    assert not missing, (
        f"以下已登记缺陷没有对应的 xfail 用例：{sorted(missing)}")


def test_the_outstanding_defect_count_is_visible():
    """
    把"还欠多少个修复"变成一条会被读到的断言。
    改这个数字必须是有意的：修好了就减，新发现就加。
    """
    outstanding = len(DEFECT_REGISTRY)
    assert outstanding == 25, (
        f"未修复的已登记缺陷数变成了 {outstanding}（原为 23）。\n"
        f"修好缺陷时请同时：① 删掉对应 xfail 标记 ② 改掉模块测试里"
        f"『钉住现状』的断言 ③ 更新 MUTATION_LEDGER。\n"
        f"当前清单：\n  " + "\n  ".join(f"{k}: {v}" for k, v in DEFECT_REGISTRY.items()))
