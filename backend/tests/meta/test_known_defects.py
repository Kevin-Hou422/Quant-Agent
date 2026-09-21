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

import tempfile
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
from _pytest.outcomes import Failed

from app.core.alpha_engine.parser import ParseError


#: 缺陷编号 → 一句话描述。新增/修复缺陷都必须同步这张表。
DEFECT_REGISTRY = {
    # B-1 / B-2 / B-3 / B-5 / B-6 / B-7 —— fast_ops 算子族，2026-09-20 全部修复并移出登记表。
    # 正确性断言见 tests/unit/alpha_engine/test_fast_ops_kernel.py 的 H 节
    # （TestFormerlyBrokenOperators，参照 scipy/pandas/np.corrcoef）。
    # B-4 —— ts_entropy 单箱契约，2026-09-21 修复并移出登记表。

    "B-9":  "use_label_encoder=False 对 xgboost 3.x 已无意义（仅代码整洁，无行为影响）",
    "B-10": "fast_ops 的向量化分支被 except Exception 完全兜住（结构问题，无行为断言）",
    "B-11": "data_partitioner 的『OOS 为空』守卫不可达（结构问题，无行为断言）",
    # A-2 —— strategy_gate 零方差守卫，2026-09-20 修复并移出登记表。

    # A-5 —— MVO 无有效目标 vs 主动清仓，2026-09-21 修复并移出登记表。

    "A-6":  "【执行层已于 2026-09-18/19/20 修复，回测引擎侧未动】"
            "PaperBroker 曾同时犯两个错：① `target=1.0` 写死，把目标总敞口强行"
            "放大到 L1=1（实测日循环**目标持仓**总敞口 0.27–0.30 → 0.90–1.00，"
            "3.33×；**成交名义额** 1.54×）；② 用 water-filling 裁剪**目标持仓**"
            "冒充成交，A 被削掉后亏空摊给 B（[0.9,-0.1] → [0.01,-0.99]）。"
            "执行层已改为 `simulate_partial_fills`：按**交易差额**逐名部分成交、"
            "不再分配、组合级净敞口回查、未成交量如实记账。"
            "**仍未完成的是引擎统一**：`BacktestEngine` 走的还是 "
            "`LiquidityConstraint` 的 water-filling **持仓**裁剪，于是两个引擎在"
            "限流场景下语义已经分家 —— `test_replay_matches_backtest_engine` 的 "
            "1e-9 对账目前只在『上限不绑定』的数据上成立，绑定时会分叉。"
            "统一之后历史回测收益会变，需要前后对比与差异解释",
    # A-7 —— 构建层三处写死的 target=1.0，2026-09-21 修复并移出登记表。

    # D-3 —— langchain 大版本不兼容，2026-09-20 迁移到 create_agent 并移出登记表。

    # D-4 —— 提示词的 neg() 模板，2026-09-20 修复并移出登记表。

    # D-5 —— 提示词相关度文案，2026-09-21 修复并移出登记表。

    "N-6":  "（前端，本轮不做）useQuantWorkspace.switchSession 在 await 之后无条件 "
            "`setMessages` —— 不检查响应回来时当前会话是否还是发起时那个。"
            "先切 A 再切 B、B 先返回 A 后返回时，store 里 sessionId=B 而展示内容"
            "是 A 的消息。已在当前代码上复现",
}

#: 只是结构/整洁问题，没有可执行的行为断言 —— 记录在案，不设 xfail 用例。
NO_BEHAVIOUR_ASSERTION = {"B-9", "B-10", "B-11"}

#: **技术债，不是已证明的产品行为故障。**
#:
#: 外部审计 2026-09-15 的判定：B-9（`use_label_encoder=False` 对 xgboost 3.x
#: 已无意义）、B-10（`except Exception` 兜得太宽）、B-11（不可达的冗余守卫）
#: 三条只有"代码结构不好"的依据，没有任何一条给出了**产品结果是错的**的复现。
#: 把它们和 B-1（值域错）、A-6（敞口被放大）算在同一个"缺陷数"里，
#: 会让那个数字读起来比实际严重，也让真正该优先修的被稀释。
#:
#: 于是拆开计数：`behavioural_defect_count()` 只数有行为依据的，
#: 技术债单列。两边都不许悄悄消失 —— 见 test_the_outstanding_defect_count_is_visible。
TECHNICAL_DEBT = {"B-9", "B-10", "B-11"}

#: 前端缺陷 —— 后端套件里没有可执行断言。**不是豁免，是分工**：
#: 登记在这里保证它出现在缺陷总数里、不被遗忘；用例欠在前端。
FRONTEND_ONLY = {"N-6"}


def _xfail(defect_id: str, raises=AssertionError):
    """
    标记"断言应有行为、当前必然失败"的用例。

    `raises` 不是装饰：**没有它，`strict=True` 只保证"意外通过要报错"，
    完全不保证"失败是因为登记的那个原因"。** 外部审计 2026-09-15 抓到 A-1
    正是如此 —— `inspect.getsource(di.ingest_incremental)` 因为那个符号
    早已不存在而先抛 AttributeError，用例在碰到目标行为之前就"失败"了，
    于是汇总行里的 `xfailed` 计数看着正常，缺陷却从未被验证过。
    同一轮自查又发现 C-1 是同样的情形（`_evaluate_individual()` 签名变了 →
    TypeError）。

    限定异常类型后，这两种情形会变成**用例失败**（pytest 对 `raises` 不匹配
    的 xfail 判 failed），而不是继续伪装成"预期失败"。

    默认 `AssertionError` = "断言了应有行为、断言没过"。缺陷本身就是抛异常的
    （B-7 抛 ValueError、B-8/A-4 抛 AttributeError、D-6 抛 ImportError），
    在调用处显式传入对应类型。
    """
    return pytest.mark.xfail(strict=True, raises=raises,
                             reason=f"{defect_id}：{DEFECT_REGISTRY[defect_id]}")


# ===========================================================================
# fast_ops —— 算子内核
# ===========================================================================

class TestFastOpsDefects:
    """
    B-1/B-2/B-3/B-5/B-6/B-7 于 2026-09-20 修复，本类下这些用例**已转正**
    （去掉 xfail，从此作为回归护栏）。详尽的正确性断言在
    tests/unit/alpha_engine/test_fast_ops_kernel.py 的 H 节，那里用
    scipy/pandas/np.corrcoef 作独立参照；这里只留最小的复现输入。
    仍挂着 xfail 的只剩 B-4（契约未定）。
    """

    def test_ts_rank_should_be_a_percentile_in_zero_one(self):
        """单调上升序列的滚动排名，最新一根是窗口内最高，应当是 1.0。"""
        import app.core.alpha_engine.fast_ops as F
        x = np.arange(20, dtype=float).reshape(20, 1)
        assert F.bn_ts_rank(x, 5).ravel()[-1] == pytest.approx(1.0)

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

    def test_ts_corr_of_perfectly_correlated_series_should_be_one(self):
        import app.core.alpha_engine.fast_ops as F
        rng = np.random.default_rng(0)
        a = rng.normal(size=(60, 1))
        b = a * 2.0 + 1.0
        assert F.ts_corr(a, b, 20).ravel()[-1] == pytest.approx(1.0, abs=1e-9)

    def test_cs_rank_should_give_ties_the_average_rank(self):
        """docstring: "ties resolved by average rank"。[1,1,2,3] 的前两名应当并列。"""
        import app.core.alpha_engine.fast_ops as F
        got = F.cs_rank(np.array([[1.0, 1.0, 2.0, 3.0]])).ravel()
        assert got[0] == pytest.approx(got[1])

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

    def test_ts_entropy_rejects_a_degenerate_bin_count(self):
        """
        **缺陷 B-4，2026-09-21 定契约并修**（本用例已转正）。

        `n_bins=1` 原先静默返回 **-0.0**：单箱 probs=[1.0] → h=-0.0，
        再除以顶替的分母 1.0（`log(1)=0` 不能当分母，代码拿 1.0 换了量纲）。

        契约定为**报错**，依据是 `n_bins` 走不到 GP/DSL —— `FAST_TS_OPS` 按
        `fn(x, window)` 派发、n_bins 恒为默认 10，fast_ops.py 之外全库零引用。
        所以它只可能是开发者写错参数。

        登记表列的第二点（负零）与 n_bins 无关、合法参数照样触发，
        已一并修掉；详尽断言见
        test_fast_ops_kernel.py::TestEntropyBinContract。
        """
        import app.core.alpha_engine.fast_ops as F
        x = np.arange(10, dtype=float).reshape(10, 1)
        with pytest.raises(ValueError, match="n_bins"):
            F.ts_entropy(x, 5, n_bins=1)
        # 反向对照：合法参数不受影响，且常数窗口给的是**正**零
        flat = F.ts_entropy(np.full((10, 1), 2.0), 5, n_bins=4)[4:]
        np.testing.assert_allclose(flat, 0.0, atol=1e-12)
        assert not np.signbit(flat).any(), "常数窗口仍返回负零"

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

    def test_cs_rank_should_stay_within_zero_one_with_nan(self):
        import app.core.alpha_engine.fast_ops as F
        got = F.cs_rank(np.array([[10.0, np.nan, 30.0, 40.0]])).ravel()
        assert np.nanmax(got) == pytest.approx(1.0)
        assert np.nanmin(got) == pytest.approx(0.0)

    def test_short_panels_should_return_nan_not_raise(self):
        """docstring 承诺"不足 window 个有效观测 → NaN"。"""
        import app.core.alpha_engine.fast_ops as F
        x = np.arange(6, dtype=float).reshape(3, 2)
        assert np.all(np.isnan(F.bn_ts_mean(x, 5)))


# ===========================================================================
# ml_engine —— 代理模型
# ===========================================================================

class TestProxyModelDefects:

    def test_an_unfitted_model_falls_back_to_the_cold_start_rule(self):
        """
        **缺陷 B-8，2026-09-20 已修**（本用例已转正）。

        `_fit()` 放弃（标签单一类 / 缺 xgboost-sklearn）时 `self._model` 仍是 None，
        而 `should_prune` 只看样本数就走模型分支 → `None.predict_proba` →
        AttributeError 打断整轮 GP。**样本够 ≠ 模型就绪。**
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
            pm.update(_chain(2 + i % 3), failed=True)      # 标签全 1 → 放弃拟合
        assert pm._model is None, "构造前提变了：这里本应没训出模型"
        assert pm.should_prune(_chain(1)) is True
        assert pm.should_prune(_chain(3)) is False

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

    def test_sessions_with_equal_timestamps_keep_newest_first(self):
        """
        **缺陷 B-12，2026-09-20 已修**（本用例已转正）。

        `list_sessions()` 原来是 `ORDER BY created_at DESC` 且**没有第二排序键**。
        `created_at` 并列时 SQLite 按插入顺序返回 —— 最老的排最前，与契约相反。

        第二排序键不能用主键：`ChatSession.id` 是 **UUID 字符串**，
        按它排是随机序、不是插入序。所以另开了一列单调递增的 `seq`。
        （`ChatMessage.id` 本来就是自增整数，消息侧直接用它。）
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
            bsess = st.create_session("B")
            c = st.create_session("C")
            got = [x.id for x in st.list_sessions()]
            assert got[0] == c.id and got[1] == bsess.id, (
                f"同一时间戳下应按插入序倒排（C,B,A），实际 {got}")
        finally:
            mod.datetime = real

    def test_messages_with_equal_timestamps_keep_insertion_order(self):
        """
        消息侧的另一半：一问一答落在同一秒时，**必须先问后答**。
        排反了在前端就是"AI 先答、用户后问"。
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
            st = ChatStore(db_url=f"sqlite:///{d / 'm.db'}")
            sid = st.create_session("S").id
            for role, text in (("user", "问"), ("assistant", "答"), ("user", "再问")):
                st.save_message(sid, role, text)
            got = [m.content for m in st.get_history(sid)]
            assert got == ["问", "答", "再问"], f"同一时间戳下消息顺序错了：{got}"
        finally:
            mod.datetime = real

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

    def test_incremental_ingest_writes_each_bar_once(self):
        """
        `ingest_incremental` 调用的 `ingest()` 内部已把整段增量窗口写过一遍 PIT，
        随后外层又 `_append_pit(increment)` 写一次 —— 每个增量日两个 vintage。
        对照见 test_daily_ingest_increment.py::test_pit_only_receives_the_increment，
        那里钉住的是**当前**的 2，这里断言的是**应有**的 1。
        """
        # 这条用例前后错了两次，两次都是"看起来在查，其实没查"：
        #
        #   第一版：整条就是一句 `pytest.skip(...)` —— 零断言。
        #   第二版：改成 `inspect.getsource(di.ingest_incremental)` 查源码文本。
        #           但 `ingest_incremental` 是 `DailyIngest` 的**方法**，
        #           模块上根本没有这个属性 → AttributeError 在碰到目标行为
        #           之前就抛出来了。`strict=True` 只管 XPASS，对"因为别的原因
        #           失败"一无所知，于是汇总行里的 xfailed 计数一直很好看，
        #           而 A-1 从未被验证过（外部审计 2026-09-15 抓到）。
        #
        # 现在改成**真的跑一遍摄取、数 PIT 里的 vintage**，
        # 并由 `_xfail(..., raises=AssertionError)` 保证它只能因断言失败。
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_a1_increment_helpers",
            _backend_root() / "tests" / "integration" / "test_daily_ingest_increment.py")
        M = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(M)
        # 复用集成用例的装配，而不是抄一份：抄的那份会随产品演进而烂掉，
        # 且没人会同时改两处。装配缺任何一件都在这里立刻报出来。
        for name in ("_panel", "_stub_loader", "_freeze_today", "TestIncrementWindow"):
            assert hasattr(M, name), (
                f"增量集成用例的装配 `{name}` 不在了 —— A-1 的复现脚手架已失效，"
                f"请同步修本用例，不要让它退回『源码文本断言』")

        W = M.TestIncrementWindow
        with pytest.MonkeyPatch.context() as mp:
            tmp = Path(tempfile.mkdtemp(prefix="a1_pit_"))
            from app.config import settings
            mp.setattr(settings, "pit_store_dir", str(tmp / "pit"), raising=False)
            mp.setattr(settings, "paper_start", "2024-01-02", raising=False)

            from app.tasks.daily_ingest import DailyIngest
            ing = DailyIngest()

            W._inject_clock(mp)
            panel = W._seed(None, ing, mp, 5)
            seeded = {d.normalize() for d in panel["close"].index}

            full = M._panel(8)
            M._stub_loader(mp, full)
            M._freeze_today(mp, str(full["close"].index[-1].date()))
            ing.ingest_incremental("px")

            from app.core.data_engine.pit_store import PITStore
            store = PITStore(settings.pit_store_dir)
            part = next((store.store_dir / "px").glob("year=*")) / "data.parquet"
            df = pd.read_parquet(part)

        per_day = df.groupby("timestamp")["as_of"].nunique()
        increment_days = {pd.Timestamp(ts).normalize(): int(n)
                          for ts, n in per_day.items()
                          if pd.Timestamp(ts).normalize() not in seeded}
        assert increment_days, "没有任何增量日 —— 本用例没测到东西"
        offenders = {str(d.date()): n for d, n in increment_days.items() if n != 1}
        assert not offenders, (
            f"增量日被写进 PIT 多次（日期 → vintage 数）：{offenders}。\n"
            "成因：`ingest_incremental` 先调 `ingest(...)`（内部已把整段增量窗口"
            "写过一遍 PIT），随后又 `_append_pit(increment)` 写第二遍。\n"
            "当前行为由 tests/integration/test_daily_ingest_increment.py::"
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

    @staticmethod
    def _float_noise_only(n: int, base: float = 0.001) -> np.ndarray:
        """
        "浮点意义上恒定、但 nanstd 非零"的序列 —— A-2 真正漏掉的那一类。

        **不能**用 `np.full(n, base)`：那种数组的 `np.nanstd` 给 0 还是 2e-19
        取决于 n 的浮点运气（实测 n=30/80/100 给 2.168e-19，n=40/50/60/70 给
        精确 0.0）。旧版本用的就是 `np.full(60, 0.001)`，于是它在**前置守卫**
        那一行就 AssertionError 了 —— 而那条 xfail 声明的 `raises` 也是
        AssertionError，两者分不开，**这条用例从未真正验证过 A-2**。
        （与 A-1 的 `inspect.getsource` 失败、C-1 的签名 TypeError 同一形态，
        第三次了；`raises=` 挡得住异常类型不同的情形，挡不住同类型的。）

        改用 1 ulp 扰动：sd 恒为 1.5e-19 量级，与 n 无关，确定性成立。
        """
        v = np.full(n, base)
        v[::2] = np.nextafter(base, 1.0)
        return v

    def test_zero_variance_guard_should_use_a_tolerance_not_equality(self):
        """
        **缺陷 A-2，2026-09-20 已修**（本用例已转正）。

        原判据 `float(np.nanstd(rets.values)) == 0.0` 用**精确相等**判浮点零。
        严格全零序列 `np.nanstd` 确实返回 0.0、守卫会触发；漏掉的是
        **非零、但只在浮点噪声级别变动**的序列 —— 回测净收益正是这种：
        多空两腿相减、成本逐日重算，残渣必然非零。
        那时 `== 0.0` 为假，守卫放行，随后 `vol > 0` 也成立，
        算出年化 Sharpe 3e16（见 A-3），报告里显示"高度显著"。

        **登记文字曾被收窄**（外部审计 2026-09-15）：原先写"守卫从不触发"是错的。
        而且后续 `_sharpe` 与 t 统计量另有容差保护，最终 `passed` 多数情况仍是
        False —— 缺陷在于**诊断说错了原因**，不在于"巨大 Sharpe 被批准"。
        因此本用例断言的是 `reasons`，不是 `passed`。

        更早一版是 `assert "...== 0.0" not in src` 的**源码字符串断言**
        （自伤教训 #6 的形态），改掉实现里任何一处等价写法它都察觉不到。
        """
        from app.core.portfolio_manager import strategy_gate as sg

        idx = pd.bdate_range("2024-01-02", periods=60)
        rets = pd.Series(self._float_noise_only(60), index=idx)
        assert float(np.nanstd(rets.values)) != 0.0, (
            "构造的序列方差恰好为零 —— 那走的是守卫**会**触发的分支，测不到本缺陷")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sg, "strategy_net_returns",
                       lambda *a, **kw: (rets, pd.DataFrame()))
            res = sg.StrategyGate(use_global_trials=False).evaluate(
                {"f": pd.DataFrame(1.0, index=idx, columns=["A", "B"])},
                {"close": pd.DataFrame(100.0, index=idx, columns=["A", "B"])})

        assert any("方差为 0" in r for r in res.reasons), (
            f"浮点意义上恒定的净收益（nanstd={float(np.nanstd(rets.values)):.3e}）"
            f"没有被零方差守卫认出来，给出的理由是：{res.reasons}")

    def test_a_genuinely_low_volatility_strategy_is_not_killed_by_the_guard(self):
        """
        反向对照：修 A-2 时容易矫枉过正，把阈值写成绝对值（如 `sd < 1e-12`），
        于是**真实**的低波动策略被当成零方差毙掉。判据是**相对**的
        （`sd > 1e-12 × mean|r|`），所以日波动小到 1e-9 也必须照常评估。

        没有这一条，"守卫更严格了"和"守卫开始误杀"分不开。
        """
        from app.core.portfolio_manager import strategy_gate as sg

        idx = pd.bdate_range("2024-01-02", periods=60)
        rets = pd.Series(np.random.default_rng(0).normal(0.001, 1e-9, 60), index=idx)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sg, "strategy_net_returns",
                       lambda *a, **kw: (rets, pd.DataFrame()))
            res = sg.StrategyGate(use_global_trials=False).evaluate(
                {"f": pd.DataFrame(1.0, index=idx, columns=["A", "B"])},
                {"close": pd.DataFrame(100.0, index=idx, columns=["A", "B"])})

        assert not any("方差为 0" in r for r in res.reasons), (
            f"日波动 {float(rets.std(ddof=1)):.3e} 的**真实**策略被零方差守卫毙了 —— "
            f"判据被写成了绝对阈值。理由：{res.reasons}")

    def test_near_zero_volatility_is_reported_as_meaningless(self):
        """
        **缺陷 A-3，2026-09-20 已修**（本用例已转正）。

        `return ... / vol if vol > 0 else np.nan` —— 全常数收益（每日恰好 +0.1%）
        的 `std(ddof=1)` 不是 0 而是 **6.5e-19** 的浮点求和残渣，`vol > 0` 成立，
        于是算出年化 Sharpe ≈ **3e16**、t ≈ 15.5，在报告里显示为"高度显著"。

        现在用**相对**判据（`sd > 1e-12 × 收益自身量级`）—— 相对而非绝对，
        是为了不把低波动但真实的策略误杀。
        """
        from app.core.backtest_engine.performance_analyzer import PerformanceAnalyzer
        idx = pd.bdate_range("2024-01-02", periods=120)

        flat = PerformanceAnalyzer(self._result(pd.Series(np.full(120, 0.001), index=idx)))
        assert np.isnan(flat.sharpe_ratio()), (
            f"全常数收益算出了 Sharpe={flat.sharpe_ratio()}")
        assert np.isnan(flat.sharpe_tstat()), "同一序列的 t 统计量也必须是 NaN"

        # 对照：**真实**低波动序列不得被误杀 —— 否则这个修复就是把功能砍掉
        rng = np.random.default_rng(3)
        tiny = PerformanceAnalyzer(self._result(
            pd.Series(0.001 + rng.normal(0, 1e-6, 120), index=idx)))
        assert np.isfinite(tiny.sharpe_ratio()), (
            "日波动 1e-6（真实但很小）被误判成了『没有波动』")

    def test_a_non_datetime_index_gives_a_clear_error(self):
        """
        **缺陷 A-4，2026-09-20 已修**（本用例已转正）。

        `_tdays` 无条件做 `(idx[-1] - idx[0]).days` —— 整数索引抛
        `AttributeError: 'int' object has no attribute 'days'`，
        报错指向**内部实现**，调用方看不出真正的问题是"索引类型不对"。
        现在提前判类型并给出可操作的说明。
        """
        from app.core.backtest_engine.performance_analyzer import PerformanceAnalyzer
        rets = pd.Series(np.full(60, 0.001), index=range(60))
        with pytest.raises(TypeError) as exc:
            PerformanceAnalyzer(self._result(rets)).sharpe_ratio()
        msg = str(exc.value)
        assert "DatetimeIndex" in msg, f"报错没说清楚要什么索引：{msg}"
        assert "net_returns" in msg, f"报错没指出是哪个字段的索引：{msg}"

    def test_data_ineligible_assets_get_no_allocation(self):
        """
        **缺陷 A-5，2026-09-21 按用户裁定的「数据资格先于优化」修复**（已转正）。

        旧注释写"剔除 NaN 过多的资产（保留其基准权重）"，实现给的是 0。
        **两者都不是答案** —— 真正的问题是返回值把三件事混成了一种表示：

          · 有限数值 = 本日的有效目标
          · 全零     = **主动清仓**
          · 全 NaN   = **本日没有有效目标**（数据不合格 / 求解失败 / 预热期）

        旧实现有五条静默回退路径，每条都产出一行看起来完全正常的权重，
        而且自相矛盾：部分资产数据不合格 → 那些资产清零；不合格到
        `valid < 3` → **全部保留**基准权重。缺得越多，持仓反而越满。

        本条钉住裁定后的规则：合格资产足够时，不合格的那些**明确不配置**（0），
        其余照常优化且整行不是 NaN。

        （更早一版是 `assert "保留其基准权重" not in src` 的源码字符串断言 ——
        改个措辞就静默转绿，而权重仍然是 0。外部审计 2026-09-15 点名。）
        """
        from app.core.backtest_engine.portfolio_constructor import MVOPortfolio

        T, N, W = 30, 5, 20
        idx = pd.bdate_range("2024-01-02", periods=T)
        cols = [f"T{i}" for i in range(N)]
        rng = np.random.default_rng(7)
        signal = pd.DataFrame(rng.normal(size=(T, N)), index=idx, columns=cols)
        returns = pd.DataFrame(rng.normal(0, 0.01, (T, N)), index=idx, columns=cols)
        # 第 0 列在协方差窗口里 50% 缺失 → 不合格（阈值 30%）；
        # 其余 4 列干净，合格数 4 >= min_valid_assets，优化分支照常走。
        returns.iloc[::2, 0] = np.nan

        out = MVOPortfolio(cov_window=W, clip_z=3.0).construct(signal, returns)
        live = out.iloc[W:]

        assert not live.isna().all(axis=1).any(), (
            "有 4 个合格资产，却整行判成了「无有效目标」")
        assert np.allclose(live.iloc[:, 0].to_numpy(), 0.0, atol=1e-12), (
            f"数据不合格的 {cols[0]} 仍拿到了配置："
            f"{live.iloc[:, 0].abs().max():.6g} —— 数据资格没有先于优化")
        assert np.allclose(live.abs().sum(axis=1).to_numpy(), 1.0, atol=1e-6), (
            "剔除之后没有在合格集合上重新归一")

    def test_no_valid_target_is_not_the_same_as_a_flat_book(self):
        """
        「本日无有效目标」(NaN) 与「主动清仓」(全零) 必须分得开。

        把 NaN 当 0 会在数据缺失的日子直接把仓位卖光 —— 这正是用户裁定里
        点名禁止的"返回全零权重冒充成功"。
        """
        from app.core.backtest_engine.portfolio_constructor import MVOPortfolio

        T, N, W = 40, 4, 20
        idx = pd.bdate_range("2024-01-02", periods=T)
        cols = [f"T{i}" for i in range(N)]
        rng = np.random.default_rng(11)
        signal = pd.DataFrame(rng.normal(size=(T, N)), index=idx, columns=cols)
        returns = pd.DataFrame(np.nan, index=idx, columns=cols)   # 全部不合格

        out = MVOPortfolio(cov_window=W, clip_z=3.0).construct(signal, returns)
        tail = out.iloc[W:]
        assert tail.isna().all(axis=1).all(), (
            "合格资产为零时没有给「无有效目标」")
        # 关键区分：这些行必须是 NaN，**不能**是 0 —— 0 是"卖光"的指令
        assert not (tail == 0.0).any().any(), (
            "「无有效目标」被写成了全零，下游会照此清仓")
        # 预热期同样是 NaN —— 那几天根本没有协方差估计
        assert out.iloc[:W].isna().all(axis=1).all(), "预热期仍在产出权重"

    def test_construction_layer_projections_keep_the_upstream_gross(self):
        """
        **缺陷 A-7，2026-09-21 已修**（本用例已转正）。

        构建层的三处 `project_to_capped_l1(..., target=1.0)`：`row_target` 是
        `min(target, budget)`，而 budget 在 cap **有限**时是 Σcap
        （ADV / 单票上限通常远大于 1），于是 row_target 恒为 1.0 ——
        上游 gross≠1 的输入被整体**放大**回 L1=1，
        每一个降敞口的决定（波动率目标、风控缩减、部分空仓）都被抹掉。
        这是 A-6 在构建层的同型错误。

        三处现在都传逐行真实 gross：容量足→原样保敞口，容量不足→降到 budget。

        本条挑 `manager.apply_capacity` 作代表 —— 它**自相矛盾**得最明显：
        docstring 写着"容量不足时 gross<1"，传的却是 `target=1.0`。
        """
        from app.core.portfolio_manager.manager import PortfolioManager

        idx = pd.bdate_range("2024-01-02", periods=30)
        cols = ["A", "B"]
        w = pd.DataFrame([[0.2, -0.1]] * 30, index=idx, columns=cols)   # gross 0.3
        px = pd.DataFrame(100.0, index=idx, columns=cols)
        vol = pd.DataFrame(1e9, index=idx, columns=cols)                # 容量远超需求

        pm = PortfolioManager(aum=1_000_000.0)
        out = pm.apply_capacity(w, px, vol)
        got = float(out.abs().sum(axis=1).iloc[-1])
        assert got == pytest.approx(0.3, abs=1e-9), (
            f"容量不绑定时 gross 应原样保持 0.3，实际 {got:.4f} —— "
            f"上游的降敞口决定被抹掉了")

    def test_capacity_shortfall_still_lowers_the_gross(self):
        """
        反向对照：修 A-7 时容易矫枉过正 —— 把 target 换成逐行 gross 之后，
        **容量不足**那条路径必须仍然生效（`row_target = min(gross, budget)`）。

        没有这一条，"不再放大"和"再也不削减"就分不开，而
        `apply_capacity` 的 docstring 承诺的正是"容量不足时 gross<1"。
        """
        from app.core.portfolio_manager.manager import PortfolioManager

        idx = pd.bdate_range("2024-01-02", periods=5)
        cols = ["A", "B"]
        w = pd.DataFrame([[0.5, -0.5]] * 5, index=idx, columns=cols)    # gross 1.0
        px = pd.DataFrame(100.0, index=idx, columns=cols)
        vol = pd.DataFrame(100.0, index=idx, columns=cols)              # ADV 极小

        pm = PortfolioManager(aum=1_000_000.0)
        out = pm.apply_capacity(w, px, vol)
        got = float(out.abs().sum(axis=1).iloc[-1])
        assert got < 1.0 - 1e-9, (
            f"容量严重不足时 gross 仍是 {got:.4f} —— 容量约束失效了")
        assert got >= 0.0

    def test_the_single_name_cap_does_not_change_total_exposure(self):
        """
        `realistic_backtester` 那一处：单票上限的职责只是**压住集中度**，
        不该改变组合总敞口。

        喂一份 gross=0.4 且**没有任何一只票触顶**的权重，
        输出 gross 必须还是 0.4 —— 上限没绑定就什么都不该发生。
        """
        from app.core.backtest_engine.transaction_cost import project_to_capped_l1

        w = np.array([[0.25, -0.15]])          # gross 0.4，单票上限 0.3 不绑定
        cap = np.full_like(w, 0.3)
        out = project_to_capped_l1(w, cap, target=np.abs(w).sum(axis=1))
        assert float(np.abs(out).sum()) == pytest.approx(0.4, abs=1e-12), (
            "上限不绑定时组合总敞口被改动了")
        np.testing.assert_allclose(out, w, atol=1e-12,
                                   err_msg="上限不绑定时权重就不该变")

    @_xfail("A-6")
    def test_the_two_engines_agree_when_liquidity_binds(self):
        """
        A-6 的**剩余部分**：执行层已改成"按交易差额部分成交、不再分配"，
        而 `BacktestEngine` 走的还是 `LiquidityConstraint` 的 water-filling
        **持仓**裁剪 —— 两个引擎在限流场景下语义已经分家。

        `test_replay_matches_backtest_engine` 的 1e-9 对账用的是成交量充足的数据
        （上限不绑定），所以它**看不见**这个分叉。这里把成交量压到上限真的绑定，
        再对同一份权重跑两个引擎。

        ⚠️ 这条**不是**"两边一样就算对"。正确性由各自的独立断言守
        （执行侧见 `test_paper_broker_accounting.py::TestGrossExposureFollowsTheTarget`
        与 `Test*PartialFill*`）；这条只守"统一"这一半 ——
        两者都对且彼此一致，才算完成。
        """
        import tempfile as _tf

        from app.core.backtest_engine.backtest_engine import BacktestEngine
        from app.core.execution.paper_broker import PaperBroker
        from app.db.position_store import PositionStore

        T, N = 30, 3
        idx = pd.bdate_range("2024-01-02", periods=T)
        cols = ["A", "B", "C"]
        rng = np.random.default_rng(5)
        prices = pd.DataFrame(
            100 * np.cumprod(1 + rng.normal(0, 0.01, (T, N)), axis=0),
            index=idx, columns=cols)
        # 成交量压得很低 → ADV 上限**真的绑定**
        volume = pd.DataFrame(2_000.0, index=idx, columns=cols)
        weights = pd.DataFrame([[0.5, -0.3, 0.2]] * T, index=idx, columns=cols)
        signal = pd.DataFrame(0.0, index=idx, columns=cols)

        eq_bt = BacktestEngine().run(weights, prices, volume, signal).equity_curve
        tmp = Path(_tf.mkdtemp(prefix="a6eng_"))
        pb = PaperBroker(store=PositionStore(db_url=f"sqlite:///{tmp/'e.db'}"))
        eq_pb = pb.replay(1, weights, prices, volume)

        gap = float(np.max(np.abs(eq_bt.to_numpy() - eq_pb.to_numpy())))
        assert gap < 1e-9, (
            f"限流绑定时两个引擎的净值最大相差 {gap:.3e} —— "
            f"回测仍用 water-filling 裁剪目标持仓（会把 A 的未成交额度摊给 B），"
            f"执行层已改成按成交量部分成交。语义未统一")

# ===========================================================================
# gp_engine —— GP 适应度
# ===========================================================================

class TestGpFitnessRankTies:

    def test_a_cross_sectionally_constant_signal_scores_zero_ic(self):
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

        # `_evaluate_individual` 收的是**一个元组**，不是两个位置参数。
        # 旧版写成 `G._evaluate_individual("(close/close)", ds)` → TypeError，
        # 在算到 IC 之前就抛了 —— 和 A-1 同型：xfail 一直"预期失败"，
        # 但失败原因根本不是 C-1（本轮自查发现，审计未列出这一条）。
        res = G._evaluate_individual(("(close/close)", ds))
        assert res.ann_return == pytest.approx(0.0, abs=1e-9), (
            f"截面恒定（零信息）的信号算出 mean_IC = {res.ann_return:.4f}，"
            f"不是 0 —— 并列名次被按列顺序摊开了")


# ===========================================================================
# mutations —— GP 结构变异算子
# ===========================================================================

class TestReplaceNodeIdentity:
    """**缺陷 C-2，2026-09-20 已修**（两条用例均已转正）。

    `_replace_node` 原来先 `deepcopy` 再按 `id(target)` 找 —— 深拷贝之后副本里
    没有任何节点持有那个 id，除非 target 就是 root，替换**永远静默失败**。
    现在改成：先在**原树**里定位 target 的路径，再沿同一条路径在副本上替换。
    """

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

def _prompt_dsl_patterns() -> list:
    """
    从 `_SYSTEM_PROMPT` 里抽出所有 `DSL pattern:` 模板，整理成可解析的 DSL。

    **必须从提示词读，不能抄成字面量**：抄下来的字面量在提示词被改对之后
    仍然是那串旧文本，用例照旧失败，修复识别不到（外部审计 2026-09-15）。

    整理规则（都是提示词自己的书写约定）：
      - `... where N = 5 to 60` 之类的说明后缀切掉，`N` 用区间下界代入
      - 一行里用 `or` 并列的多个模板拆开
    """
    import re

    from app.agent._prompts import _SYSTEM_PROMPT as P

    out = []
    for m in re.finditer(r"DSL pattern:\s*(.+)", P):
        body = m.group(1).strip()
        low = 5
        where = re.search(r"\bwhere\s+N\s*=\s*(\d+)", body)
        if where:
            low = int(where.group(1))
        body = re.split(r"\s+where\s+", body)[0]
        for part in re.split(r"\s+or\s+", body):
            part = part.strip()
            if not part:
                continue
            out.append(re.sub(r"\bN\b", str(low), part))
    return out


class TestSystemPromptDslExamples:

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

        **被测输入从提示词本身读取，不再抄成字面量。** 外部审计
        2026-09-15 指出：旧版把 `"rank(neg(ts_delta(close, 5)))"` 写死在用例里，
        于是把提示词改对之后这条**仍然失败**（它测的是那串已经不存在的文本），
        修复无法被识别。独立进程实验的结论是 `2 xfailed, 退出码 0`。
        """
        from app.core.alpha_engine.parser import Parser

        patterns = _prompt_dsl_patterns()
        assert len(patterns) >= 4, (
            f"只从提示词里抽到 {len(patterns)} 条 DSL 模板 —— "
            f"抽取规则和提示词格式已经对不上，本用例测不到东西：{patterns}")

        parser = Parser()
        for dsl in patterns:
            node = parser.parse(dsl)
            assert node is not None, f"{dsl} 解析出空节点"

    @staticmethod
    def _declared_operators() -> set:
        from app.agent._prompts import _SYSTEM_PROMPT as P

        i = P.index("AVAILABLE OPERATORS:")
        lines = P[i + len("AVAILABLE OPERATORS:"):].splitlines()
        taken = [lines[0]]
        for ln in lines[1:]:
            if not ln.strip() or not ln.startswith((" ", "\t")):
                break
            taken.append(ln)
        return {t.strip() for t in " ".join(taken).split(",") if t.strip()}

    def test_every_function_used_in_a_documented_pattern_is_declared(self):
        """
        **缺陷 D-4，2026-09-20 已修**（本用例已转正，并改成全称断言）。

        提示词内部自洽：`DSL pattern:` 里当范例用的每个函数，都必须出现在它
        自己的 AVAILABLE OPERATORS 清单上。`neg` 曾是反例 —— 四个家族的模板
        都写 `rank(neg(...))`，而 `neg` 既不在清单里、解析器也不认它
        （只认一元负号 `-x`）。LLM 照模板产出的公式一律解析失败，
        `_validate_and_fix` 白烧两次修复调用后放弃，六个家族里四个产出为零，
        对外只表现为"agent 老是生成非法公式"。

        旧版断言的是 `"neg" in declared`（把 `neg` 加进清单）。那是**错的修法**：
        解析器里根本没有 `neg` 函数，加进清单只会让提示词与解析器一起错。
        正确的修法是把模板改成解析器接受的 `-x` —— 那也正是系统自己序列化
        时输出的形式（`str(parse("rank(-ts_delta(close,5))"))` 往返稳定）。

        现在断言的是**全称命题**：从提示词抽出的每条模板里用到的每个函数名
        都要被声明。这样新加一条用了未声明算子的模板也会被抓到，
        而不只是防住 `neg` 这一个。
        """
        import re

        declared = self._declared_operators()
        patterns = _prompt_dsl_patterns()
        assert patterns, "没从提示词里抽到 DSL 模板 —— 抽取规则与提示词格式对不上"

        used = set()
        for dsl in patterns:
            used |= set(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", dsl))

        missing = sorted(used - declared)
        assert not missing, (
            f"提示词的 DSL 模板用了未在 AVAILABLE OPERATORS 里声明的函数：{missing}\n"
            f"模板：{patterns}\n已声明：{sorted(declared)}")

    def test_the_prompt_does_not_advertise_a_negation_function(self):
        """
        D-4 的反向护栏：不许有人"顺手"把 `neg` 写回提示词。

        解析器没有 `neg` 函数（实测 `neg(close)` 抛 ParseError），
        取负只能写一元负号。把 `neg(` 写回范例 = D-4 复发。
        """
        from app.agent._prompts import _SYSTEM_PROMPT as P

        assert "neg(" not in P, (
            "提示词里又出现了 `neg(` —— 解析器不认这个函数，D-4 复发。"
            "取负请写一元负号 `-x`。")

        from app.core.alpha_engine.parser import ParseError as PE, Parser
        with pytest.raises(PE):
            Parser().parse("neg(close)")


class TestSystemPromptThresholds:

    @staticmethod
    def _evolver():
        """构造一个最小 PopulationEvolver —— 只为读它实际建出来的池子。"""
        from app.core.gp_engine.population_evolver import PopulationEvolver
        idx = pd.bdate_range("2024-01-02", periods=30)
        df = pd.DataFrame(100.0, index=idx, columns=["A", "B"])
        data = {"close": df, "open": df, "high": df * 1.01,
                "low": df * 0.99, "volume": df * 1e4}
        return PopulationEvolver(is_data=data, oos_data=data)

    def test_the_production_pool_threshold_is_not_the_class_default(self):
        """
        **先把事实钉住，再谈不一致。** 不带 xfail。

        旧版 D-5 用 `AlphaPool.__init__` 的**类默认值** 0.70 去断言"生产链路按
        0.70 拒收"，外部审计 2026-09-15 指出这一步是错的：真正建池子的是
        `PopulationEvolver`，它的默认是 0.90 并显式传进去。
        拿类默认值推断链路行为 = 判据与被测对象不是同一个东西。

        这条断言实际对象上的阈值，所以谁改了任何一侧都会在这里看见。
        """
        import inspect

        from app.core.gp_engine.population_evolver import PopulationEvolver

        evolver_default = inspect.signature(
            PopulationEvolver.__init__).parameters["corr_threshold"].default
        assert evolver_default == pytest.approx(0.90), (
            f"PopulationEvolver 的 corr_threshold 默认变成了 {evolver_default} —— "
            f"D-5 的收窄结论基于它是 0.90，请重新核对")

        ev = self._evolver()
        assert ev._pool._corr_threshold == pytest.approx(evolver_default), (
            "PopulationEvolver 没有把自己的 corr_threshold 传给 AlphaPool —— "
            "那样类默认值才会真的生效，D-5 的原始指控就重新成立了")

    def test_the_prompt_threshold_matches_the_actual_rejection_rule(self):
        """
        收窄后仍然成立的那一半：提示词写 `corr > 0.9`，
        代码判的是 `abs(corr) >= 0.9`。

          - **绝对值**：ρ = -0.95 的因子会被拒，而提示词的写法读不出这一点
          - **边界开闭**：ρ 恰好 0.9 时提示词说收、代码拒

        后果：LLM 按提示词判断"这两条够不够正交"，与池子实际的拒收规则不同，
        它会反复产出自以为合格、实际被静默拒绝的候选，且拿不到反馈。
        """
        import re

        from app.agent._prompts import _SYSTEM_PROMPT as P

        # 提示词里带一份**规范形式** `abs(corr) >= 0.9`，测试锚定它而不是扫散文。
        # 上一版正则是 `corr\s*([<>=]+)\s*([\d.]+)` —— 我把措辞改通顺之后它就
        # 抓不到了，当场判红。规则的三要素（绝对值 / 比较符 / 阈值）必须都能
        # 机械读出来，否则"提示词和代码一致"这件事就只能靠人眼。
        m = re.search(r"abs\(corr\)\s*(>=|>|<=|<)\s*([\d.]+)", P)
        assert m, (
            "提示词里找不到规范形式 `abs(corr) <op> <阈值>` —— "
            "措辞可以改，但这份机器可读的规则必须留着")
        op, num = m.group(1), float(m.group(2))
        thr = self._evolver()._pool._corr_threshold

        assert op == ">=", (
            f"提示词写的比较符是 `{op}`，代码判的是 `>=`（边界闭）")
        assert num == pytest.approx(thr), (
            f"提示词写的阈值是 {num}，生产链路实际是 {thr}")


# ===========================================================================
# LangChain 依赖版本 —— LLM 研究链路是否真的在跑
# ===========================================================================

class TestLangChainWiringIsAlive:
    """
    **缺陷 D-3，2026-09-20 已修**（两条用例均已转正）。

    详尽的迁移验收在 tests/unit/agent/test_lc_agent_migration.py（22 条，
    用**实际安装的** LangChain 真跑工具调用、两轮会话、会话隔离、
    工具异常、调用次数上限）。这里只留最小的"链路没死"断言。
    """

    def test_the_langchain_agent_can_actually_be_built(self):
        """
        `_build_langchain_agent` 在**当前已安装的依赖**下必须真的建得出来。

        原缺陷：`requirements.txt` 写 `langchain>=0.2` 无上界、lock 锁 1.4.0，
        而代码要的 `AgentExecutor` / `create_tool_calling_agent` 在 1.x 已搬进
        未安装的 `langchain-classic` —— 于是这个函数**每次都抛 ImportError**，
        `QuantAgent.__init__` 打一条 warning 就退到 FallbackOrchestrator。
        对外表现：`/api/chat` 照常工作、前端毫无异样，LLM 研究链路整条死掉。
        （交易回路本来就不含 LLM，所以不影响下单；影响的是因子发现。）

        这条不碰网络、不需要 API key —— 建得出来就算通过。
        """
        import app.agent._lc_agent as LC

        try:
            from langchain.agents import create_agent                      # noqa: F401
            from langchain.agents.middleware import (                      # noqa: F401
                ModelCallLimitMiddleware, wrap_tool_call)
            from langchain.tools import tool as lc_tool                    # noqa: F401
        except ImportError as exc:          # pragma: no cover
            pytest.fail(
                f"当前安装的 langchain 无法提供 _lc_agent 需要的符号：{exc}。"
                f"声明的依赖集合建不出 agent —— LLM 链路会静默降级。")

        assert callable(LC._build_langchain_agent)
        assert callable(LC._build_tools)

    def test_the_import_failure_message_names_a_version_conflict(self):
        """
        报错文案必须指向**版本冲突**，不能把人引向"再装一遍 langchain"。

        这条原本钉的是**误导文案本身**（`assert "pip install langchain" in src`），
        并写明"一旦有人把文案改成提到版本，这里会红，提醒同步更新 D-3"。
        2026-09-20 文案确实改了，这条如期变红 —— 现在翻成正向断言。

        `LangChainIncompatibleError` 是独立异常类型，让调用方按**类型**而不是
        文案匹配来分类；文案怎么改都不会让分类失效。
        """
        import inspect

        import app.agent._lc_agent as LC

        src = inspect.getsource(LC._build_langchain_agent)
        assert "pip install langchain" not in src, (
            "报错文案又变回『去装 langchain』了 —— 而 langchain 是装着的，"
            "照着这句做只会再装一遍同样的版本")
        assert "LangChainIncompatibleError" in src
        assert issubclass(LC.LangChainIncompatibleError, ImportError)


# ===========================================================================
# LocalParquetProvider —— 宣称的字段与实际可读的字段
# ===========================================================================

class TestLocalParquetAdvertisedFields:

    def test_every_advertised_field_can_actually_be_requested(self):
        """
        **缺陷 D-2，2026-09-20 已修**（本用例已转正）。

        `available_fields()` 曾宣称支持 `returns`，而它不在 `STANDARD_COLUMNS` 里、
        从不落盘。按契约请求 `fields=["close", "returns"]` 时，pyarrow 的列裁剪
        在读取层失败 → `_read_ticker` 的 except 吞掉异常只发一条 warning →
        **整批数据返回 `{}`**（连 close 都没有）。调用方拿到空 dict，
        看不出是自己要了一个不存在的列。

        ⚠️ 旧版把前提 `assert "returns" in advertised` 写在**用例体内** ——
        `returns` 一旦不再被宣称，这条前提先失败、用例仍是"预期失败"，
        **修复无法被识别**（与 D-4 同型，见 `_xfail` 的 docstring）。
        现在断言的是全称命题：**宣称的每一个字段都必须真的能请求到**。
        """
        import tempfile

        from app.core.data_engine.local_parquet_provider import LocalParquetProvider

        root = Path(tempfile.mkdtemp(prefix="d2_"))
        pv = LocalParquetProvider(root_dir=str(root))
        idx = pd.bdate_range("2022-01-03", periods=6)
        pv.write(pd.DataFrame({
            "timestamp": idx, "ticker": "AAA",
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
            "volume": 1e6, "vwap": 100.2, "adj_factor": 1.0,
        }))

        advertised = pv.available_fields()
        assert advertised, "没有宣称任何字段，本用例测不到东西"
        for f in advertised:
            ds = pv.fetch(["AAA"], "2022-01-01", "2022-12-31", fields=["close", f])
            assert ds, f"宣称支持 {f!r}，但按契约请求后整批数据为空"
            assert "close" in ds, (
                f"请求 [close, {f!r}] 之后连 close 都没了 —— "
                f"一个不可读的宣称字段让整批数据归零")

class TestFinancialInterpreterNegation:

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
        # 反向对照：`(1 - x)` **不是**取负，不得被误判成 reversion ——
        # 否则这个修复就是把判据放宽到把别的东西也吞进来。
        assert it.interpret("rank((1-ts_delta(close,5)))").factor_family == "momentum", (
            "`(1-x)` 被当成了取负 —— 判据放得太宽")
        assert unary == zero_minus, (
            f"`-x` 判为 {unary}，而语义等价的 `(0-x)` 判为 {zero_minus} —— "
            f"同一个因子换写法就换了家族")


# ===========================================================================
# proxy_model —— 可选依赖的降级路径
# ===========================================================================

class TestProxyModelOptionalDependency:

    def test_missing_sklearn_degrades_instead_of_crashing(self):
        """
        **缺陷 D-6，2026-09-20 已修**（本用例已转正）。

        `XGBClassifier` 是 xgboost 的 sklearn API：import 成功、**构造时**才抛
        `ImportError: sklearn needs to be installed`。原来的 try 只包住 import，
        那个异常越过守卫直接冒泡 —— 缺依赖时 GP 进化**直接崩**，
        而不是像 warning 承诺的那样退回 rule-based。

        用替身精确复刻那条路径：能 import、一构造就抛 ImportError。
        """
        import xgboost

        from app.core.alpha_engine.typed_nodes import DataNode, TimeSeriesNode
        from app.core.ml_engine.proxy_model import ProxyModel

        def _chain(depth):
            node = DataNode("close")
            for _ in range(depth):
                node = TimeSeriesNode("ts_mean", node, 5)
            return node

        class _NeedsSklearn:
            def __init__(self, *a, **kw):
                raise ImportError(
                    "sklearn needs to be installed in order to use this module")

        original = xgboost.XGBClassifier
        xgboost.XGBClassifier = _NeedsSklearn
        try:
            pm = ProxyModel(cold_start_n=2)
            for i in range(3):
                pm.update(_chain(2 + i % 3), failed=bool(i % 2))
        finally:
            xgboost.XGBClassifier = original

        assert pm._fitted is False, "构造抛了 ImportError 却自称已拟合"
        assert pm.should_prune(_chain(3)) is False, "未拟合时应退回冷启动规则"

    def test_missing_sklearn_degrades_instead_of_crashing(self):
        """
        `_fit()` 的 except 只包住 `from xgboost import XGBClassifier`，
        而 `ImportError: sklearn needs to be installed` 是 **XGBClassifier(...)
        构造时**抛的 —— 在 except 的作用域之外，直接向上冒泡。

        本用例不依赖真实环境缺不缺 scikit-learn：直接把 XGBClassifier 换成
        一个"能 import、一构造就抛 ImportError"的替身，精确复刻那条路径。
        期望行为是 `update()` 正常返回且模型停在未拟合状态（代码注释承诺的
        『stays in rule-based mode』）；实际会把 ImportError 抛给调用方。
        """
        import xgboost

        from app.core.alpha_engine.typed_nodes import DataNode, TimeSeriesNode
        from app.core.ml_engine.proxy_model import ProxyModel

        def _chain(depth: int):
            node = DataNode("close")
            for _ in range(depth):
                node = TimeSeriesNode("ts_mean", node, 5)
            return node

        class _NeedsSklearn:
            def __init__(self, *a, **kw):
                raise ImportError(
                    "sklearn needs to be installed in order to use this module")

        original = xgboost.XGBClassifier
        xgboost.XGBClassifier = _NeedsSklearn
        try:
            pm = ProxyModel(cold_start_n=2)
            for i in range(3):
                pm.update(_chain(2 + i % 3), failed=bool(i % 2))
        finally:
            xgboost.XGBClassifier = original

        assert pm._fitted is False, "构造抛了 ImportError 却自称已拟合"


# ===========================================================================
# 外部审计 2026-09-15 的独立发现
# ===========================================================================

def _panel_for_gate(days: int = 60, n: int = 4, seed: int = 3) -> dict:
    """给 strategy_gate 用的最小 WidePanel（close/high/low/volume）。"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=days)
    cols = [f"T{i}" for i in range(n)]
    close = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.01, (days, n)), axis=0),
        index=idx, columns=cols)
    return {
        "close":  close,
        "high":   close * 1.05,
        "low":    close * 0.98,
        "volume": pd.DataFrame(1e6, index=idx, columns=cols),
    }


class TestStrategyGateCacheAndFailurePaths:

    def test_risk_gate_failure_refuses_to_produce_returns(self):
        """
        **缺陷 N-3，2026-09-20 已修**（本用例已转正，不再是 xfail）。

        `apply_risk=True` 是调用方**明确要求**过风控。原先风控 `apply()` 抛错后
        只打一条 warning 就**用未经风控的原始权重继续回测** —— 于是这段"策略
        净收益"对应的组合，和会去交易的组合不是同一个，而该函数 docstring
        承诺的恰恰是「门评估的账本 == 实际交易的账本」。

        现在做不到就抛，让调用方知道。
        """
        from app.core.portfolio_manager import strategy_gate as sg

        class _Boom:
            def __init__(self, *a, **kw):
                pass

            def apply(self, *a, **kw):
                raise RuntimeError("风控注入故障")

        with pytest.MonkeyPatch.context() as mp:
            import app.core.portfolio_manager.risk_gate as rg
            mp.setattr(rg, "PortfolioRiskGate", _Boom)
            ds = _panel_for_gate(days=90, n=4)
            sig = {"f": pd.DataFrame(
                np.linspace(-1, 1, 4 * 90).reshape(90, 4),
                index=ds["close"].index, columns=ds["close"].columns)}
            with pytest.raises(RuntimeError, match="拒绝用未经风控的权重回测"):
                sg.strategy_net_returns(sig, ds, apply_risk=True)

    def test_the_gate_reports_a_failure_instead_of_a_verdict(self):
        """
        端到端另一半：抛出去之后**调用方要接住并判不通过**，
        而不是让异常冒到更上层、或者被谁吞掉又变成一个"通过"。
        """
        from app.core.portfolio_manager import strategy_gate as sg

        class _Boom:
            def __init__(self, *a, **kw):
                pass

            def apply(self, *a, **kw):
                raise RuntimeError("风控注入故障")

        idx = pd.bdate_range("2024-01-02", periods=120)
        with pytest.MonkeyPatch.context() as mp:
            import app.core.portfolio_manager.risk_gate as rg
            mp.setattr(rg, "PortfolioRiskGate", _Boom)
            res = sg.StrategyGate(use_global_trials=False).evaluate(
                {"f": pd.DataFrame(1.0, index=idx, columns=["A", "B"])},
                _panel_for_gate(days=120, n=2))

        assert res.passed is False, "风控失败却给出了通过结论"
        assert any("回测失败" in r or "风控" in r for r in res.reasons), (
            f"拒绝的理由里看不出是风控失败：{res.reasons}")

def test_defect_registry_matches_the_ledger():
    """
    登记表里的每个编号都必须能在 `MUTATION_LEDGER.md` 里找到 ——
    防止这里和台账各说各话。
    """
    import pathlib
    import re
    ledger = (_backend_root() / "MUTATION_LEDGER.md").read_text(encoding="utf-8")
    # 早期这里只核对 `B-` 前缀，于是台账的缺陷索引停在 A-1 就再没更新过，
    # 而登记表已经涨到 25 条 —— 两边各说各话了很久都没人发现。
    # 现在**所有**编号都核，漏一个就红。
    missing = [d for d in DEFECT_REGISTRY
               if not re.search(rf"\b{re.escape(d)}\b", ledger)]
    assert not missing, (
        f"以下缺陷编号在台账里找不到：{missing} —— 两边已经不同步。\n"
        f"修法：在 MUTATION_LEDGER.md 的『已登记产品缺陷』索引表里补上一行。")


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
    expected = set(DEFECT_REGISTRY) - NO_BEHAVIOUR_ASSERTION - FRONTEND_ONLY
    missing = expected - covered
    assert not missing, (
        f"以下已登记缺陷没有对应的 xfail 用例：{sorted(missing)}")


def test_the_outstanding_defect_count_is_visible():
    """
    把"还欠多少个修复"变成一条会被读到的断言。
    改这个数字必须是有意的：修好了就减，新发现就加。

    **分三类数，不混成一个数**（外部审计 2026-09-15 的判定）：

      · 有行为依据的产品缺陷 —— 有复现、有"应有行为"的断言
      · 技术债 —— 只有结构证据，没有"产品结果是错的"的复现
      · 前端 —— 后端套件里没有可执行断言，欠一条前端用例

    混成一个数会让它读起来比实际严重，也会稀释真正该优先修的那几条。
    """
    behavioural = set(DEFECT_REGISTRY) - TECHNICAL_DEBT - FRONTEND_ONLY
    assert (len(behavioural), len(TECHNICAL_DEBT), len(FRONTEND_ONLY)) == (1, 3, 1), (
        f"缺陷分类计数变了：行为缺陷 {len(behavioural)} / 技术债 "
        f"{len(TECHNICAL_DEBT)} / 前端 {len(FRONTEND_ONLY)}"
        f"（登记总数 {len(DEFECT_REGISTRY)}，此前 2/3/1；"
        f"B-1/B-2/B-3/B-5/B-6/B-7 这一族 fast_ops 算子缺陷已于 2026-09-20 一并修复）。\n"
        f"修好缺陷时请同时：① 删掉对应 xfail 标记 ② 改掉模块测试里"
        f"『钉住现状』的断言 ③ 更新 MUTATION_LEDGER。\n"
        f"当前清单：\n  " + "\n  ".join(f"{k}: {v}" for k, v in DEFECT_REGISTRY.items()))
    assert TECHNICAL_DEBT <= set(DEFECT_REGISTRY) and FRONTEND_ONLY <= set(DEFECT_REGISTRY), (
        "技术债/前端分类里有不在登记表中的编号")
    assert TECHNICAL_DEBT == NO_BEHAVIOUR_ASSERTION, (
        "技术债与『无行为断言』两个集合分叉了 —— 它们指的是同一批条目，"
        "分开维护迟早对不上")
