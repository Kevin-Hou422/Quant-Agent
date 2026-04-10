"""
main.py — Quant Agent 全流程入口

用法:
  python main.py --mode agent   --hypothesis "Post-earnings drift"
  python main.py --mode gp      --generations 5 --pop-size 10
  python main.py --mode backtest --dsl "rank(ts_delta(log(close),5))"
  python main.py --mode report  --alpha-id 1
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

# ── 把 backend/ 加入 sys.path ────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# 合成数据集（用于演示与测试）
# ---------------------------------------------------------------------------

def _make_synthetic_dataset(
    n_tickers: int = 20,
    n_days:    int = 120,
    seed:      int = 42,
) -> dict[str, pd.DataFrame]:
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    tickers = [f"T{i:03d}" for i in range(n_tickers)]

    close  = pd.DataFrame(
        (100 * np.cumprod(1 + rng.normal(0, 0.01, (n_days, n_tickers)), axis=0)),
        index=dates, columns=tickers,
    )
    volume = pd.DataFrame(
        rng.integers(500_000, 5_000_000, (n_days, n_tickers)).astype(float),
        index=dates, columns=tickers,
    )
    high   = close * (1 + rng.uniform(0, 0.02, close.shape))
    low    = close * (1 - rng.uniform(0, 0.02, close.shape))
    open_  = close.shift(1).fillna(close)
    vwap   = (high + low + close) / 3
    returns = close.pct_change().fillna(0.0)

    return {
        "close":   close,
        "open":    open_,
        "high":    high,
        "low":     low,
        "volume":  volume,
        "vwap":    vwap,
        "returns": returns,
    }


# ---------------------------------------------------------------------------
# 模式实现
# ---------------------------------------------------------------------------

def run_agent(args: argparse.Namespace) -> None:
    """agent 模式：假设 → DSL → 评估 → 修正 → 持久化。"""
    from app.core.alpha_discovery.alpha_agent import AlphaAgent
    from app.core.alpha_discovery.alpha_store import AlphaStore

    dataset = _make_synthetic_dataset()
    store   = AlphaStore()
    agent   = AlphaAgent(store=store)

    logger.info("AlphaAgent 启动 | hypothesis='%s'", args.hypothesis)
    log = agent.run(hypothesis=args.hypothesis, dataset=dataset)

    print("\n" + "=" * 60)
    print("ReasoningLog:")
    print(log.summary())
    if log.final_dsl:
        print(f"\n最终 Alpha DSL: {log.final_dsl}")
    else:
        print("\n本轮未发现合格 Alpha（可检查 IC-IR / 换手阈值）")


def run_gp(args: argparse.Namespace) -> None:
    """gp 模式：遗传规划种群进化。"""
    from app.core.alpha_discovery.gp_engine import AlphaEvolver
    from app.core.alpha_discovery.alpha_store import AlphaStore, AlphaResult

    dataset = _make_synthetic_dataset(n_tickers=args.n_tickers, n_days=args.n_days)
    store   = AlphaStore()
    evolver = AlphaEvolver(
        pop_size  = args.pop_size,
        n_gen     = args.generations,
        n_workers = args.workers,
    )

    logger.info("AlphaEvolver 启动 | pop=%d gen=%d workers=%d",
                args.pop_size, args.generations, args.workers)
    hof = evolver.evolve(dataset)

    print("\n" + "=" * 60)
    print(f"Hall of Fame ({len(hof)} Alpha):")
    for i, r in enumerate(hof[:10], 1):
        print(f"  {i:2d}. IC-IR={r.ic_ir:.4f}  DSL={r.dsl[:80]}")
        ar = AlphaResult(
            dsl     = r.dsl,
            sharpe  = r.sharpe,
            ic_ir   = r.ic_ir,
            status  = "active",
        )
        store.save(ar)
    print(f"\nHoF 已保存到 AlphaStore。")


def run_backtest(args: argparse.Namespace) -> None:
    """backtest 模式：对单条 DSL 执行完整回测并打印 RiskReport。"""
    from app.core.alpha_engine.dsl_executor import DSLExecutor
    from app.core.alpha_engine.parser import Parser
    from app.core.alpha_engine.validator import AlphaValidator
    from app.core.backtest_engine.backtest_engine import BacktestEngine
    from app.core.backtest_engine.portfolio_constructor import SignalWeightedPortfolio, NeutralizationLayer
    from app.core.backtest_engine.performance_analyzer import PerformanceAnalyzer
    from app.core.backtest_engine.risk_report import RiskReport

    dsl = args.dsl
    logger.info("单条 DSL 回测 | dsl='%s'", dsl)

    dataset = _make_synthetic_dataset(n_tickers=20, n_days=120)

    # 验证
    node = Parser().parse(dsl)
    AlphaValidator().validate(node)

    # 生成信号
    executor = DSLExecutor()
    signal   = executor.run(dsl, dataset)

    # 构建权重
    constructor = SignalWeightedPortfolio()
    weights     = constructor.construct(signal)
    NeutralizationLayer.market_neutral(weights)

    # 回测
    engine = BacktestEngine()
    result = engine.run(
        weights = weights,
        prices  = dataset["close"],
        volume  = dataset["volume"],
        signal  = signal,
    )

    # 指标
    analyzer = PerformanceAnalyzer()
    report   = RiskReport.from_result(result, analyzer)

    print("\n" + "=" * 60)
    print(report.summary())


def run_report(args: argparse.Namespace) -> None:
    """report 模式：从 SQLite 读取 Alpha 记录并打印。"""
    from app.core.alpha_discovery.alpha_store import AlphaStore

    store = AlphaStore()
    if args.alpha_id:
        record = store.get_by_id(args.alpha_id)
        if record is None:
            print(f"Alpha id={args.alpha_id} 不存在。")
            return
        print(f"\nAlpha #{record.id}")
        print(f"  DSL        : {record.dsl}")
        print(f"  Hypothesis : {record.hypothesis}")
        print(f"  Sharpe     : {record.sharpe:.4f}")
        print(f"  IC-IR      : {record.ic_ir:.4f}")
        print(f"  MaxDD      : {record.max_drawdown:.4f}")
        print(f"  AnnTurnover: {record.ann_turnover:.2f}")
        print(f"  Status     : {record.status}")
    else:
        records = store.query(min_sharpe=-999, limit=50)
        print(f"\n共 {len(records)} 条 Alpha 记录（按 Sharpe 降序）:")
        for r in records[:20]:
            print(f"  id={r.id:4d}  sharpe={r.sharpe:6.3f}  ic_ir={r.ic_ir:6.3f}  dsl={r.dsl[:60]}")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quant Agent — 自主 Alpha 发现流水线",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["agent", "gp", "backtest", "report"],
                        required=True, help="运行模式")

    # agent / backtest
    parser.add_argument("--hypothesis", type=str, default="momentum",
                        help="市场假设（agent 模式）")
    parser.add_argument("--dsl", type=str,
                        default="rank(ts_delta(log(close),5))",
                        help="DSL 表达式（backtest 模式）")

    # gp
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--pop-size",    type=int, default=20)
    parser.add_argument("--workers",     type=int, default=1)
    parser.add_argument("--n-tickers",   type=int, default=20)
    parser.add_argument("--n-days",      type=int, default=120)

    # report
    parser.add_argument("--alpha-id", type=int, default=None,
                        help="查询指定 id 的 Alpha（report 模式）")

    args = parser.parse_args()

    dispatch = {
        "agent":    run_agent,
        "gp":       run_gp,
        "backtest": run_backtest,
        "report":   run_report,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
