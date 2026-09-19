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
    "B-1":  "ts_rank 在 bottleneck 分支的值域是 [-1/w, 1/w]，不是 docstring 承诺的 [0,1]",
    "B-2":  "ts_corr 因 cov(ddof=0)/std(ddof=1) 不配套而系统性偏低 (w-1)/w",
    "B-3":  "cs_rank 的并列处理是序数名次，不是 docstring 声称的平均名次",
    "B-4":  "ts_entropy(n_bins=1) 返回 -0.0 而非 NaN/报错 —— **契约未定**："
            "普通 Shannon 熵单箱本来就是 0，代码也显式选了分母 1"
            "（`log_nbins = np.log(n_bins) if n_bins > 1 else 1.0`），"
            "没有任何外部接口契约要求 NaN。真正的问题只有两点："
            "① 归一化熵在退化输入上的约定没写下来；② 返回的是 **负零**。"
            "第 3 阶段要先定约定再谈修不修（外部审计 2026-09-15 要求收窄）",
    "B-5":  "ts_max / ts_min 的 NaN 策略在 bottleneck 与 numpy 分支之间不一致",
    "B-6":  "cs_rank 在含 NaN 的截面上值域越出 [0,1]",
    "B-7":  "面板行数短于窗口时 bottleneck 分支抛 ValueError，而非返回 NaN",
    "B-8":  "ProxyModel 在 _fit() 放弃之后仍走模型分支 → AttributeError",
    "B-9":  "use_label_encoder=False 对 xgboost 3.x 已无意义（仅代码整洁，无行为影响）",
    "B-10": "fast_ops 的向量化分支被 except Exception 完全兜住（结构问题，无行为断言）",
    "B-11": "data_partitioner 的『OOS 为空』守卫不可达（结构问题，无行为断言）",
    "B-12": "chat_store 的 ORDER BY 没有第二排序键，同一 tick 内的记录顺序反了",
    "A-1":  "ingest_incremental 把增量写进 PIT 两次（ingest() 内一次 + 外层一次）",
    "A-2":  "strategy_gate 用 `np.nanstd(...) == 0.0` 判零方差。**指控已收窄**"
            "（外部审计 2026-09-15）：原写『守卫从不触发』是错的 —— 严格全零序列"
            "`np.nanstd` 返回精确 0.0，守卫会触发；不触发的是**非零常数**序列"
            "（`nanstd([0.001]*100) = 2.17e-19`）。而且下游 `_sharpe` 与 t 统计量"
            "另有容差保护，最终多半仍判 passed=false。因此这是**诊断说错了原因**，"
            "不是『巨大 Sharpe 被批准』",
    "A-3":  "PerformanceAnalyzer 对近零波动无防护：全常数收益算出年化 Sharpe ≈ 3e16",
    "A-4":  "PerformanceAnalyzer 遇非日期索引先在 _tdays 抛 AttributeError，"
            "max_drawdown 里的整数索引分支不可达，且报错指向内部实现",
    "A-5":  "MVOPortfolio 注释写『剔除的资产保留基准权重』，实现是整行替换 → 拿到 0",
    "A-6":  "【前半已于 2026-09-18 修复】PaperBroker.step 曾写死 target=1.0，"
            "把目标总敞口强行放大到 L1=1 —— 已改为 `|tgt|` 之和，由 "
            "TestGrossExposureFollowsTheTarget 钉住。\n"
            "**剩下的后半仍未修**：ADV 上限削掉某只票之后，water-filling 把亏空"
            "**摊到其余名字**以凑满目标 gross。实测 [0.9, -0.1] 在 A 只能买 0.01 时"
            "落账 [0.01, -0.99] —— 10% 的空头变成 99%，这已经不是同一个组合。"
            "改法牵涉设计取舍：`project_to_capped_l1` 的 water-filling 是 Task 6.6 "
            "**有意**的（用来修旧的『clip→整体归一化』缺陷），且回测引擎与 "
            "PortfolioManager 走同一条路（`test_replay_matches_backtest_engine` "
            "以 1e-9 对账两者）。只改执行侧会让两个引擎在限流场景下分家",
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
    "D-5":  "系统提示词写 `AlphaPool rejects signal-correlated alphas (corr > 0.9)`。"
            "**数字这一半的指控不成立**（外部审计 2026-09-15）：`AlphaPool` 的"
            "**类默认**确实是 0.70，但生产链路 `PopulationEvolver` 的默认是 0.90 "
            "并显式传进池子（`AlphaPool(max_size=200, corr_threshold=corr_threshold)`），"
            "实际对象上就是 0.90 —— 不能拿类默认值去断言链路行为。"
            "剩下的真实不一致只有一处：提示词说 `corr > 0.9`，代码判的是 "
            "`abs(corr) >= 0.9` —— **绝对值**（负相关同样被拒）与**边界开闭**两处差异",
    "D-6":  "ProxyModel._fit 的 `try: from xgboost import XGBClassifier "
            "except ImportError` 只包住了 **import**，而 XGBClassifier 是在"
            "**构造时**才检查 scikit-learn（`ImportError: sklearn needs to be "
            "installed`）。import 成功、构造抛错，异常越过那个 except 直接向上"
            "冒泡 —— 于是缺 scikit-learn 时 GP 进化**直接崩**，而不是像代码"
            "注释承诺的那样 warning 一句后退回 rule-based 模式。"
            "requirements.txt 已补上 scikit-learn（CI 事故 2026-09-10 的根因），"
            "但这个 except 的覆盖范围本身仍然是错的",

    # ---- 外部审计 2026-09-15 的独立发现（沿用它的编号，便于交叉引用）----
    "N-1":  "strategy_gate 的成本推导缓存键（`_cache_key`）只含 close 的形状、"
            "索引端点、首末行 nansum、前 4 个列名与 aum —— **不含 high/low、"
            "volume、券商配置**。而 Corwin-Schultz 价差正是从 high/low 算的。"
            "于是 close 相同、high/low 不同的两个数据集命中同一条缓存，"
            "第二个拿到第一个的成本参数。审计实测：真实值 1919.83 bps "
            "被缓存里的 37.47 bps 顶替",
    "N-2":  "StrategyGate 读全局试验台账失败时 `n_trials` 退回 1 —— 而 n_trials 是 "
            "Deflated Sharpe 的多重检验校正项，退回 1 等于宣称『只试过一个策略』，"
            "DSR 被高估、门变**松**。代码注释自己写着『门在读不到试验台账时应当"
            "更保守，而不是更宽松』，实现却相反。审计实测：注入台账不可读后仍得 "
            "passed=true、reasons=[]、DSR≈0.99997",
    "N-3":  "strategy_net_returns 里 `PortfolioRiskGate.apply` 与无交易带对齐被 "
            "`except Exception` 整个兜住，失败后只打一条 warning 就**用未经风控的"
            "原始权重继续回测**。同一函数的 docstring 明写『门评估的账本 == 实际"
            "交易的账本』—— 异常路径打破的正是这条保证：验证用的组合和实际会交易的"
            "组合不再是同一个",
    "N-4":  "PerformanceAnalyzer.sharpe_tstat 把**年化** Sharpe 与**日频**样本数"
            "混用：t = SR_年化 × √T_日 / √(1+0.5·SR²)。频率不一致使 t 被放大约 "
            "√(TDAYS) 倍。同一组 120 日收益：产品 8.3236，同频日口径 0.5660，"
            "单样本 t 参考 0.5664。risk_report 按 1.96 判显著，于是不显著的策略"
            "被显示成『✓显著』。与 A-3 的近零波动是两回事",
    "N-5":  "PaperBroker.step 用 `tickers = list(target_w.index)` 截断旧持仓："
            "目标资产集合缩小时，不在新目标里的旧持仓既不参与估值（当日收益丢失），"
            "也不产生平仓成交（仓位凭空消失）。审计实测：A/B 各半仓、次日目标只留 "
            "B 且 A 涨 10%，gross_ret=0（应为 +5%），成交记录里没有 A 的平仓。"
            "应以旧持仓与新目标的**并集**估值与交易",
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

    @_xfail("B-7", raises=ValueError)
    def test_short_panels_should_return_nan_not_raise(self):
        """docstring 承诺"不足 window 个有效观测 → NaN"。"""
        import app.core.alpha_engine.fast_ops as F
        x = np.arange(6, dtype=float).reshape(3, 2)
        assert np.all(np.isnan(F.bn_ts_mean(x, 5)))


# ===========================================================================
# ml_engine —— 代理模型
# ===========================================================================

class TestProxyModelDefects:

    @_xfail("B-8", raises=AttributeError)
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

        **登记文字已收窄**（外部审计 2026-09-15）：原先写"守卫从不触发"是错的 ——
        严格全零序列 `np.nanstd` 返回精确 0.0，守卫会触发。真正不触发的是
        **非零常数**序列（`np.nanstd([0.001]*100) = 2.17e-19`）。
        而且后续 `_sharpe` 与 t 统计量另有容差保护，所以最终 `passed` 多数情况
        仍是 False —— 缺陷在于**诊断说错了原因**，不在于"巨大 Sharpe 被批准"。
        因此本用例断言的是 `reasons`，不是 `passed`。

        旧版这条是 `assert "...== 0.0" not in src` 的**源码字符串断言**
        （自伤教训 #6 的形态），改掉实现里任何一处等价写法它都察觉不到。
        """
        from app.core.portfolio_manager import strategy_gate as sg

        idx = pd.bdate_range("2024-01-02", periods=60)
        rets = pd.Series(np.full(60, 0.001), index=idx)
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

    @_xfail("A-4", raises=AttributeError)
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

        旧版这条是 `assert "保留其基准权重" not in src or ...` 的**源码字符串
        断言**（外部审计 2026-09-15 点名）：它只能发现"这两句话同时出现在源码
        里"，改个措辞就静默转绿，而权重仍然是 0。这里改成真的跑一遍构造器、
        比对被剔除资产的权重。

        **契约尚未裁定**（审计的意见，我同意）：对坏数据资产保留非零仓位
        是否正确，不能由一句注释决定。本用例钉住的是"注释承诺的行为"，
        第 3 阶段要先定政策 —— 若结论是"归零才对"，那要改的是注释，
        同时删掉本用例与 A-5 登记。
        """
        from app.core.backtest_engine.portfolio_constructor import (
            MVOPortfolio, SignalWeightedPortfolio)

        T, N, W = 30, 5, 20
        idx = pd.bdate_range("2024-01-02", periods=T)
        cols = [f"T{i}" for i in range(N)]
        rng = np.random.default_rng(7)
        signal = pd.DataFrame(rng.normal(size=(T, N)), index=idx, columns=cols)
        returns = pd.DataFrame(rng.normal(0, 0.01, (T, N)), index=idx, columns=cols)
        # 第 0 列在协方差窗口里 50% 缺失 → `valid` 判 False（阈值 30%），
        # 其余 4 列干净（`valid.sum() == 4 >= 3`，优化分支照常走）。
        returns.iloc[::2, 0] = np.nan

        base = SignalWeightedPortfolio(clip_z=3.0).construct(signal)
        out = MVOPortfolio(cov_window=W, clip_z=3.0).construct(signal, returns)

        optimised = [t for t in range(W, T) if not np.allclose(
            out.to_numpy()[t], base.to_numpy()[t], atol=1e-12)]
        assert optimised, (
            "没有任何一天真的走进优化分支 —— 本用例测不到剔除逻辑")

        t0 = optimised[0]
        assert out.iloc[t0, 0] == pytest.approx(base.iloc[t0, 0], abs=1e-12), (
            f"第 {t0} 天：被剔除资产 {cols[0]} 的权重是 {out.iloc[t0, 0]:.6g}，"
            f"而注释承诺『保留其基准权重』= {base.iloc[t0, 0]:.6g}")

    @_xfail("A-6")
    def test_an_adv_capped_name_does_not_inflate_the_others(self):
        """
        A-6 的**后半**（前半已修，见 `TestGrossExposureFollowsTheTarget`）。

        A 想要 90% 但 ADV 只允许 1%，B 想要 -10%。执行侧该做的是"能成交多少
        成交多少" —— 落账 `[0.01, -0.10]`，总敞口不足是**事实**，不该被掩盖。
        产品当前用 water-filling 把 89% 的亏空摊给 B，落账 `[0.01, -0.99]`：
        一个 10% 的对冲腿变成 99% 的方向性空头，**这已经不是同一个组合**。

        这里断言"其余名字不被放大"，而不是断言最终 gross —— 后者取决于
        补不补的设计取舍，前者是无论怎么取舍都不该发生的。

        旧版这条是 `assert "target=1.0" not in src` 的**源码字符串断言**
        （自伤教训 #6）：把字面量换成同值变量它就静默转绿，而行为分毫未变。
        """
        import tempfile as _tf

        from app.core.execution.paper_broker import PaperBroker
        from app.db.position_store import PositionStore

        tmp = Path(_tf.mkdtemp(prefix="a6_"))
        cap_pct = PaperBroker(store=PositionStore(db_url="sqlite:///:memory:")
                              ).params.adv_cap_pct
        capital = 1_000_000.0
        adv_a = 0.01 * capital / cap_pct        # 让 A 的上限恰好是 1% 权重

        b = PaperBroker(store=PositionStore(db_url=f"sqlite:///{tmp/'a6.db'}"),
                        initial_capital=capital)
        tk = ["A", "B"]
        b.step(alpha_id=1, date="2024-01-02",
               target_w=pd.Series([0.9, -0.1], index=tk),
               prices_t=pd.Series([100.0, 100.0], index=tk),
               prices_prev=pd.Series([100.0, 100.0], index=tk),
               adv_usd=pd.Series([adv_a, 1e15], index=tk),
               daily_vol=pd.Series([0.02, 0.02], index=tk))
        pos = b.store.latest_positions(1)

        assert pos.get("A", 0.0) == pytest.approx(0.01, abs=1e-9), (
            f"A 应被 ADV 上限削到 1%，实际 {pos.get('A', 0.0)} —— 构造前提变了")
        assert abs(pos.get("B", 0.0)) == pytest.approx(0.10, abs=1e-9), (
            f"B 的目标是 -10%，落账 {pos.get('B', 0.0):.4f} —— "
            f"A 被限流后的亏空被摊到了 B 头上，对冲腿变成了方向性头寸")


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

    @_xfail("D-4", raises=ParseError)
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

    def test_the_prompt_still_uses_neg_so_d4_is_still_open(self):
        """
        **前置条件，不带 xfail。** 旧版把 `assert "neg(" in P` 写在下面那条
        xfail 用例的**体内**，于是提示词一旦改对，这条前置先失败 → 用例仍是
        "预期失败" → 修复被吃掉（外部审计 2026-09-15）。

        前置条件必须单独成立，并且在缺陷被修好时**变红**，
        逼人来删掉 D-4 与这条检查本身。
        """
        from app.agent._prompts import _SYSTEM_PROMPT as P

        assert "neg(" in P, (
            "提示词里已经不用 neg 了 —— D-4 已修复。"
            "请删除 DEFECT_REGISTRY['D-4']、下面两条 xfail 用例和本条前置检查。")

    @_xfail("D-4")
    def test_every_operator_used_in_the_prompt_is_also_declared_there(self):
        """
        提示词内部自洽：正文里当范例用的算子，必须出现在它自己的
        AVAILABLE OPERATORS 清单上。现在 `neg` 只在范例里出现，
        LLM 拿到的是自相矛盾的两份说明。

        前置条件（提示词里确实还在用 `neg`）由
        `test_the_prompt_still_uses_neg_so_d4_is_still_open` 单独把关，
        不放在本用例体内。
        """
        declared = self._declared_operators()
        assert "neg" in declared, (
            f"`neg` 在范例里被使用，却不在 AVAILABLE OPERATORS 清单里："
            f"{sorted(declared)}")


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

    @_xfail("D-5")
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

        m = re.search(r"corr\s*([<>=]+)\s*([\d.]+)", P)
        assert m, "提示词里找不到相关度阈值"
        op, num = m.group(1), float(m.group(2))
        thr = self._evolver()._pool._corr_threshold

        assert (op, num) == (">=", thr) and "abs" in P.lower(), (
            f"提示词写的是 `corr {op} {num}`，实际拒收规则是 "
            f"`abs(corr) >= {thr}` —— 绝对值与边界开闭两处都没写对"
            f"（数字本身 {num} vs {thr} 是对的，那一半指控已撤销）")


# ===========================================================================
# LangChain 依赖版本 —— LLM 研究链路是否真的在跑
# ===========================================================================

class TestLangChainWiringIsAlive:

    @_xfail("D-3", raises=Failed)
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
# proxy_model —— 可选依赖的降级路径
# ===========================================================================

class TestProxyModelOptionalDependency:

    @_xfail("D-6", raises=ImportError)
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

    @_xfail("N-1")
    def test_cost_cache_key_covers_every_input_the_cost_actually_depends_on(self):
        """
        `_cache_key` 只指纹了 close。价差用的是 high/low —— 换掉 high/low、
        close 不变，缓存照样命中，第二个数据集拿到第一个的成本参数。

        本用例不去比 bps 数字（那会把单元测试绑死在成本模型的具体数值上），
        而是直接问：**清不清缓存，结果是否一样**。一样就说明缓存键漏了输入。
        """
        from app.core.portfolio_manager import strategy_gate as sg

        a = _panel_for_gate()
        b = {**a, "high": a["close"] * 1.60, "low": a["close"] * 0.40}   # 价差大得多

        sg._DERIVE_CACHE.clear()
        sg.resolve_cost_params(a, 1_000_000.0)
        cached = sg.resolve_cost_params(b, 1_000_000.0)      # 命中 a 的条目？

        sg._DERIVE_CACHE.clear()
        fresh = sg.resolve_cost_params(b, 1_000_000.0)       # b 的真实结果

        assert cached == fresh, (
            f"同一个数据集 b，走缓存与不走缓存得到不同的成本参数 —— "
            f"缓存键漏掉了它实际依赖的输入（high/low）。\n"
            f"  命中缓存: {cached}\n  清缓存后: {fresh}")

    @_xfail("N-2")
    def test_unreadable_trial_ledger_must_not_produce_a_pass(self):
        """
        台账读不到 → n_trials 退回 1 → DSR 少做多重检验校正 → 门变松。
        代码注释自己写的是"应当更保守"。这里断言：读不到必要数据时，
        结论**不能**是 passed（要么拒绝，要么明确标注验证不完整）。
        """
        from app.core.portfolio_manager import strategy_gate as sg

        idx = pd.bdate_range("2024-01-02", periods=120)
        rng = np.random.default_rng(11)
        rets = pd.Series(rng.normal(0.004, 0.004, 120), index=idx)   # 稳定盈利

        class _Broken:
            def __init__(self):
                raise RuntimeError("试验台账不可读（注入）")

        with pytest.MonkeyPatch.context() as mp:
            import app.db.trial_ledger as tl
            mp.setattr(tl, "TrialLedger", _Broken)
            mp.setattr(sg, "strategy_net_returns",
                       lambda *a, **kw: (rets, pd.DataFrame()))
            res = sg.StrategyGate(use_global_trials=True).evaluate(
                {"f": pd.DataFrame(1.0, index=idx, columns=["A", "B"])},
                _panel_for_gate(days=120, n=2))

        assert not (res.passed and not res.reasons), (
            f"全局试验台账读不到，门仍然给出无保留的通过："
            f"passed={res.passed} n_trials={res.n_trials} reasons={res.reasons}")

    @_xfail("N-3")
    def test_risk_gate_failure_must_not_silently_fall_back_to_raw_weights(self):
        """
        风控/无交易带对齐抛异常后，产品用**原始权重**继续回测，
        于是"验证的组合"与"会去交易的组合"不再是同一个 ——
        而这正是该函数 docstring 承诺的东西。

        断言：必要风控失败时不得继续产出净收益（应当抛出或明确标记未验证）。
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
            produced = None
            try:
                produced, _ = sg.strategy_net_returns(sig, ds, apply_risk=True)
            except Exception:
                produced = None

        assert produced is None, (
            f"风控 apply() 抛错之后仍然产出了 {len(produced)} 天净收益 —— "
            f"这段收益对应的是**未经风控**的权重，与实际会交易的组合不是同一个")


class TestSharpeTStatFrequency:

    @_xfail("N-4")
    def test_sharpe_tstat_uses_a_consistent_frequency(self):
        """
        t = SR × √T / √(1 + 0.5·SR²) 里 SR 与 T 必须同频。
        产品用年化 SR 配日频 T，t 被放大约 √TDAYS 倍。

        参考值用**同频日 Sharpe** 代入同一个 Lo(2002) 分母得到；
        再与 scipy 的单样本 t 交叉验证（两者对这组样本应当很接近），
        避免"参考值也是照着实现算的"这种同源判据。
        """
        from scipy import stats

        from app.core.backtest_engine.performance_analyzer import PerformanceAnalyzer

        idx = pd.bdate_range("2023-01-02", periods=120)
        ret = pd.Series(([0.012, -0.008, 0.008, -0.010] * 30), index=idx)

        sr_d = float(ret.mean() / ret.std(ddof=1))
        expected = sr_d * np.sqrt(len(ret)) / np.sqrt(1.0 + 0.5 * sr_d ** 2)
        crosscheck = float(stats.ttest_1samp(ret, 0.0).statistic)
        assert abs(expected - crosscheck) < 0.01, (
            f"同频参考值 {expected:.6f} 与单样本 t {crosscheck:.6f} 相差太大 —— "
            f"参考口径本身有问题，先修参考值再谈产品")

        got = PerformanceAnalyzer(self._result(ret)).sharpe_tstat()
        assert got == pytest.approx(expected, rel=1e-6), (
            f"sharpe_tstat() = {got:.6f}，同频口径应为 {expected:.6f}（单样本 t "
            f"{crosscheck:.6f}）。产品把**年化** Sharpe 与**日频**样本数混用，"
            f"t 被放大约 √TDAYS 倍 → 按 1.96 判定时不显著的结果显示为『✓显著』")

    _result = staticmethod(TestBacktestAndExecutionDefects._result)


class TestPaperBrokerShrinkingUniverse:

    @_xfail("N-5")
    def test_positions_outside_the_new_target_are_still_valued_and_closed(self):
        """
        第 1 天 A/B 各半仓；第 2 天目标只留 B，而 A 从 100 涨到 110。
        A 的那半仓当天应当贡献 +5% 毛收益，并产生一笔平仓成交。

        产品先用 `target_w.index` 截断旧持仓，于是 A 既不估值也不平仓 ——
        仓位和收益一起消失。
        """
        import tempfile as _tf

        from app.core.execution.paper_broker import PaperBroker
        from app.db.position_store import PositionStore

        tmp = Path(_tf.mkdtemp(prefix="n5_"))
        b = PaperBroker(store=PositionStore(db_url=f"sqlite:///{tmp/'n5.db'}"),
                        initial_capital=1_000_000.0)
        both = ["A", "B"]
        b.step(alpha_id=1, date="2024-01-02",
               target_w=pd.Series([0.5, 0.5], index=both),
               prices_t=pd.Series([100.0, 100.0], index=both),
               prices_prev=pd.Series([100.0, 100.0], index=both),
               adv_usd=pd.Series([1e12, 1e12], index=both),
               daily_vol=pd.Series([0.02, 0.02], index=both))

        pnl = b.step(alpha_id=1, date="2024-01-03",
                     target_w=pd.Series([1.0], index=["B"]),
                     prices_t=pd.Series([110.0, 100.0], index=both),
                     prices_prev=pd.Series([100.0, 100.0], index=both),
                     adv_usd=pd.Series([1e12, 1e12], index=both),
                     daily_vol=pd.Series([0.02, 0.02], index=both))

        assert pnl.gross_ret == pytest.approx(0.05, abs=1e-9), (
            f"昨仓 A 占 50% 且当日 +10%，毛收益应为 +5%，实际 {pnl.gross_ret:.6f} —— "
            f"不在新目标里的旧持仓被 `target_w.index` 截掉了，既没估值也没平仓")


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
    assert (len(behavioural), len(TECHNICAL_DEBT), len(FRONTEND_ONLY)) == (28, 3, 1), (
        f"缺陷分类计数变了：行为缺陷 {len(behavioural)} / 技术债 "
        f"{len(TECHNICAL_DEBT)} / 前端 {len(FRONTEND_ONLY)}"
        f"（登记总数 {len(DEFECT_REGISTRY)}，此前 28/3/1）。\n"
        f"修好缺陷时请同时：① 删掉对应 xfail 标记 ② 改掉模块测试里"
        f"『钉住现状』的断言 ③ 更新 MUTATION_LEDGER。\n"
        f"当前清单：\n  " + "\n  ".join(f"{k}: {v}" for k, v in DEFECT_REGISTRY.items()))
    assert TECHNICAL_DEBT <= set(DEFECT_REGISTRY) and FRONTEND_ONLY <= set(DEFECT_REGISTRY), (
        "技术债/前端分类里有不在登记表中的编号")
    assert TECHNICAL_DEBT == NO_BEHAVIOUR_ASSERTION, (
        "技术债与『无行为断言』两个集合分叉了 —— 它们指的是同一批条目，"
        "分开维护迟早对不上")
