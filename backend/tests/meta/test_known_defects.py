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
    assert outstanding == 20, (
        f"未修复的已登记缺陷数变成了 {outstanding}（原为 20）。\n"
        f"修好缺陷时请同时：① 删掉对应 xfail 标记 ② 改掉模块测试里"
        f"『钉住现状』的断言 ③ 更新 MUTATION_LEDGER。\n"
        f"当前清单：\n  " + "\n  ".join(f"{k}: {v}" for k, v in DEFECT_REGISTRY.items()))
