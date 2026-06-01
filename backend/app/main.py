"""
main.py — Quant Agent 双模式入口

FastAPI 应用:
  uvicorn app.main:app --reload --app-dir backend

CLI 模式:
  # 使用真实市场数据（默认 us_tech_large, 2020-2024）
  python app/main.py --mode agent    --hypothesis "Post-earnings drift"
  python app/main.py --mode gp       --generations 5 --pop-size 20
  python app/main.py --mode backtest --dsl "rank(ts_delta(log(close),5))"
  python app/main.py --mode report   --alpha-id 1
  python app/main.py --mode realistic --dsl "rank(ts_delta(log(close),5))"

  # 切换数据集
  python app/main.py --mode gp --dataset china_tech --start 2021-01-01 --end 2024-01-01
  python app/main.py --mode gp --dataset crypto_major

  # 可用数据集（dataset_registry.py 中注册）：
  #   us_tech_large / us_financials / us_healthcare / us_energy
  #   china_tech / china_consumer / china_state_owned
  #   hk_china_tech / crypto_major / crypto_alt

  # 回退到合成数据（仅用于极快的单元测试）：
  python app/main.py --mode backtest --use-synthetic
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd

# ── 把 backend/ 加入 sys.path（此文件位于 backend/app/，需上溯一级到 backend/）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── FastAPI 应用 ───────────────────────────────────────────────────────────
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.router import router
from app.api.chat_router import chat_router

app = FastAPI(
    title   = settings.app_title,
    version = settings.app_version,
    debug   = settings.debug,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = settings.cors_origins,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.include_router(router)
app.include_router(chat_router)


@app.get("/health", tags=["System"])
def health() -> dict:
    return {"status": "ok", "version": settings.app_version}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# 数据加载：真实市场数据（首选）或合成数据（回退/测试）
# ---------------------------------------------------------------------------

def _load_dataset(args: argparse.Namespace) -> Dict[str, pd.DataFrame]:
    """
    统一数据加载入口。

    优先级：
      1. --use-synthetic 标志 → 合成随机游走数据（仅供快速测试）
      2. 其他情况 → 通过 dataset_registry 加载真实市场数据

    返回 dict[field → pd.DataFrame(T × N)]，字段为
    open / high / low / close / volume / vwap / returns / sector。
    """
    use_synthetic = getattr(args, "use_synthetic", False)

    if use_synthetic:
        logger.warning(
            "⚠ 使用合成随机游走数据（--use-synthetic）。"
            "所有回测结论在真实市场上无效，仅供单元测试使用。"
        )
        return _make_synthetic_dataset(
            n_tickers = getattr(args, "n_tickers", settings.default_n_tickers),
            n_days    = getattr(args, "n_days",    settings.default_n_days),
        )

    dataset_name = getattr(args, "dataset", settings.default_dataset)
    start        = getattr(args, "start",   settings.default_start)
    end          = getattr(args, "end",     settings.default_end)

    logger.info(
        "加载真实市场数据集: %s  [%s → %s]",
        dataset_name, start, end,
    )

    try:
        from app.core.data_engine.dataset_registry import load_registry_dataset
        ds = load_registry_dataset(dataset_name, start=start, end=end)
        logger.info(
            "数据集就绪: %s | 资产=%d | 日期=%d | 字段=%s",
            ds.name, ds.n_assets, ds.n_dates, list(ds.data.keys()),
        )
        return ds.data
    except Exception as exc:
        logger.error(
            "加载数据集 '%s' 失败: %s\n"
            "请检查网络连接或改用 --use-synthetic 进行离线测试。",
            dataset_name, exc,
        )
        raise


def _make_synthetic_dataset(
    n_tickers: int = 20,
    n_days:    int = 120,
    seed:      int = 42,
) -> Dict[str, pd.DataFrame]:
    """合成随机游走数据集（仅供 --use-synthetic 回退使用）。"""
    rng     = np.random.default_rng(seed)
    dates   = pd.bdate_range("2022-01-03", periods=n_days)
    tickers = [f"T{i:03d}" for i in range(n_tickers)]

    close  = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.01, (n_days, n_tickers)), axis=0),
        index=dates, columns=tickers,
    )
    volume = pd.DataFrame(
        rng.integers(500_000, 5_000_000, (n_days, n_tickers)).astype(float),
        index=dates, columns=tickers,
    )
    high    = close * (1 + rng.uniform(0, 0.02, close.shape))
    low     = close * (1 - rng.uniform(0, 0.02, close.shape))
    open_   = close.shift(1).fillna(close)
    vwap    = (high + low + close) / 3
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
# CLI 模式函数
# ---------------------------------------------------------------------------

def run_agent(args: argparse.Namespace) -> None:
    """agent 模式：假设 → DSL → 评估 → 修正 → 持久化。"""
    from app.agent.alpha_agent import AlphaAgent
    from app.db.alpha_store import AlphaStore

    dataset = _load_dataset(args)
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
    """
    gp 模式：GP 种群进化（使用 PopulationEvolver + 真实 IS/OOS 分区）。

    主流程：
      1. 加载真实市场数据
      2. DataPartitioner 做 IS/OOS 物理分区（默认 OOS 30%）
      3. PopulationEvolver 执行 GP 进化（含 Optuna 参数调优）
      4. 输出 Hall of Fame 并写入 AlphaStore
    """
    from app.core.data_engine.data_partitioner import DataPartitioner
    from app.core.gp_engine.population_evolver import PopulationEvolver
    from app.db.alpha_store import AlphaStore, AlphaResult

    dataset = _load_dataset(args)
    store   = AlphaStore()

    # IS / OOS 物理分区
    dates = next(iter(dataset.values())).index
    oos_ratio = getattr(args, "oos_ratio", 0.30)
    partitioner = DataPartitioner(
        start     = str(dates[0].date()),
        end       = str(dates[-1].date()),
        oos_ratio = oos_ratio,
    )
    partitioned = partitioner.partition(dataset)
    is_data     = partitioned.train()
    oos_data    = partitioned.test()
    print(partitioner.summary())

    logger.info(
        "PopulationEvolver 启动 | pop=%d gen=%d | IS=%d天 OOS=%d天",
        args.pop_size, args.generations,
        partitioned.is_days, partitioned.oos_days,
    )

    evolver = PopulationEvolver(
        is_data       = is_data,
        oos_data      = oos_data,
        pop_size      = args.pop_size,
        n_generations = args.generations,
    )

    def _on_gen(log: dict) -> None:
        print(
            f"  Gen {log['generation']:2d}/{args.generations}"
            f" | pop={log['population_size']}"
            f" | best_fitness={log['best_fitness']:+.4f}"
            f" | oos_sharpe={log['best_oos_sharpe']:+.4f}"
            f"\n      → {log['best_dsl'][:80]}"
        )

    gp_result = evolver.run(
        n_optuna_trials   = getattr(args, "optuna_trials", 5),
        on_generation_end = _on_gen,
    )

    print("\n" + "=" * 60)
    print(f"GP 完成 | {gp_result.generations_run} 代 | 最优 DSL:")
    print(f"  {gp_result.best_dsl}")
    m = gp_result.metrics
    print(f"  IS Sharpe={m.get('is_sharpe', 'N/A')}  "
          f"OOS Sharpe={m.get('oos_sharpe', 'N/A')}  "
          f"Turnover={m.get('is_turnover', 'N/A')}")

    print(f"\nPool Top-5:")
    for i, entry in enumerate(gp_result.pool_top5, 1):
        print(
            f"  {i}. fitness={entry['fitness']:+.4f}"
            f"  oos={entry['sharpe_oos']:+.4f}"
            f"  DSL={entry['dsl'][:70]}"
        )
        ar = AlphaResult(
            dsl          = entry["dsl"],
            sharpe       = entry["sharpe_oos"],
            ic_ir        = entry.get("fitness", 0.0),
            ann_turnover = entry["turnover"],
            status       = "candidate",
        )
        store.save(ar)

    print("\nPool Top-5 已保存到 AlphaStore（status=candidate）。")


def run_backtest(args: argparse.Namespace) -> None:
    """backtest 模式：对单条 DSL 执行完整回测并打印 RiskReport。"""
    from app.core.alpha_engine.dsl_executor import Executor
    from app.core.alpha_engine.parser import Parser
    from app.core.alpha_engine.validator import AlphaValidator
    from app.core.backtest_engine.backtest_engine import BacktestEngine
    from app.core.backtest_engine.portfolio_constructor import (
        SignalWeightedPortfolio, NeutralizationLayer,
    )
    from app.core.backtest_engine.risk_report import RiskReport

    dsl = args.dsl
    logger.info("单条 DSL 回测 | dsl='%s'", dsl)

    dataset = _load_dataset(args)

    # 验证
    node = Parser().parse(dsl)
    AlphaValidator().validate(node)

    # 生成信号
    executor = Executor()
    signal   = executor.run_expr(dsl, dataset)

    # 构建权重
    weights = SignalWeightedPortfolio().construct(signal)
    NeutralizationLayer.market_neutral(weights)

    # 回测
    engine = BacktestEngine()
    result = engine.run(
        weights = weights,
        prices  = dataset["close"],
        volume  = dataset["volume"],
        signal  = signal,
    )

    report = RiskReport.from_result(result, prices=dataset["close"])

    print("\n" + "=" * 60)
    print(report.summary())


def run_report(args: argparse.Namespace) -> None:
    """report 模式：从 SQLite 读取 Alpha 记录并打印。"""
    from app.db.alpha_store import AlphaStore

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
            print(f"  id={r.id:4d}  sharpe={r.sharpe:6.3f}  "
                  f"ic_ir={r.ic_ir:6.3f}  status={r.status:<12}  "
                  f"dsl={r.dsl[:60]}")


def run_realistic(args: argparse.Namespace) -> None:
    """realistic 模式：带信号处理管道的 IS+OOS 双段回测。"""
    from app.core.alpha_engine.signal_processor import SimulationConfig
    from app.core.backtest_engine.realistic_backtester import RealisticBacktester
    from app.core.data_engine.data_partitioner import DataPartitioner

    dataset = _load_dataset(args)

    cfg = SimulationConfig(
        delay            = args.delay,
        decay_window     = args.decay_window,
        truncation_min_q = args.truncation_min_q,
        truncation_max_q = args.truncation_max_q,
        portfolio_mode   = args.portfolio_mode,
        top_pct          = args.top_pct,
    )

    oos_dataset = None
    if args.oos_ratio > 0:
        dates = next(iter(dataset.values())).index
        partitioner = DataPartitioner(
            start     = str(dates[0].date()),
            end       = str(dates[-1].date()),
            oos_ratio = args.oos_ratio,
        )
        partitioned = partitioner.partition(dataset)
        is_data     = partitioned.train()
        oos_dataset = partitioned.test()
        print(partitioner.summary())
    else:
        is_data = dataset

    backtester = RealisticBacktester(config=cfg)
    result = backtester.run(args.dsl, is_data, oos_dataset=oos_dataset)
    print(result.summary())


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    parser = argparse.ArgumentParser(
        description="Quant Agent — 自主 Alpha 发现流水线",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["agent", "gp", "backtest", "report", "realistic"],
        required=True,
        help="运行模式",
    )

    # ── 数据集参数（Phase 0: 真实数据接入）────────────────────────────────
    parser.add_argument(
        "--dataset",
        type=str,
        default=settings.default_dataset,
        help=(
            "数据集名称。可选: us_tech_large / us_financials / us_healthcare / "
            "us_energy / china_tech / china_consumer / china_state_owned / "
            "hk_china_tech / crypto_major / crypto_alt"
        ),
    )
    parser.add_argument(
        "--start",
        type=str,
        default=settings.default_start,
        help="数据起始日期（YYYY-MM-DD）",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=settings.default_end,
        help="数据截止日期（YYYY-MM-DD）",
    )
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        default=False,
        dest="use_synthetic",
        help="使用合成随机游走数据（仅供离线单元测试，结论无金融意义）",
    )

    # ── Agent 参数 ────────────────────────────────────────────────────────
    parser.add_argument(
        "--hypothesis",
        type=str,
        default="momentum",
        help="市场假设（agent 模式）",
    )

    # ── Backtest / Realistic 参数 ─────────────────────────────────────────
    parser.add_argument(
        "--dsl",
        type=str,
        default="rank(ts_delta(log(close),5))",
        help="DSL 表达式（backtest / realistic 模式）",
    )

    # ── GP 参数 ───────────────────────────────────────────────────────────
    parser.add_argument("--generations",    type=int,   default=5)
    parser.add_argument("--pop-size",       type=int,   default=20,   dest="pop_size")
    parser.add_argument("--optuna-trials",  type=int,   default=5,    dest="optuna_trials")
    parser.add_argument("--oos-ratio",      type=float, default=0.30, dest="oos_ratio")

    # ── 合成数据回退参数（--use-synthetic 时生效）────────────────────────
    parser.add_argument("--n-tickers",  type=int, default=settings.default_n_tickers, dest="n_tickers")
    parser.add_argument("--n-days",     type=int, default=settings.default_n_days,    dest="n_days")

    # ── Report 参数 ───────────────────────────────────────────────────────
    parser.add_argument(
        "--alpha-id",
        type=int,
        default=None,
        dest="alpha_id",
        help="查询指定 id 的 Alpha（report 模式）",
    )

    # ── Realistic 模式专属参数 ─────────────────────────────────────────────
    parser.add_argument("--delay",            type=int,   default=1)
    parser.add_argument("--decay-window",     type=int,   default=0,    dest="decay_window")
    parser.add_argument("--truncation-min-q", type=float, default=0.05, dest="truncation_min_q")
    parser.add_argument("--truncation-max-q", type=float, default=0.95, dest="truncation_max_q")
    parser.add_argument("--portfolio-mode",   type=str,   default="long_short", dest="portfolio_mode")
    parser.add_argument("--top-pct",          type=float, default=0.10, dest="top_pct")

    args = parser.parse_args()

    dispatch = {
        "agent":     run_agent,
        "gp":        run_gp,
        "backtest":  run_backtest,
        "report":    run_report,
        "realistic": run_realistic,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    _cli_main()
