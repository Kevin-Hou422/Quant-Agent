"""
router.py — Quant Agent FastAPI 路由层

端点：
  POST /api/agent/run       → AlphaAgent 假设驱动 DSL 生成 + 评估
  POST /api/gp/evolve       → AlphaEvolver GP 进化
  POST /api/backtest/run    → 单条 DSL 完整回测 + RiskReport
  GET  /api/report/query    → AlphaStore 历史 Alpha 查询
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.dependencies import get_store
from app.db.alpha_store import AlphaStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# 合成数据集（演示/测试用，与 CLI 保持一致）
# ---------------------------------------------------------------------------

def _make_synthetic_dataset(
    n_tickers: int = 20,
    n_days: int = 120,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    rng     = np.random.default_rng(seed)
    dates   = pd.bdate_range("2022-01-03", periods=n_days)
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    close   = pd.DataFrame(
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
        "close": close, "open": open_, "high": high,
        "low": low, "volume": volume, "vwap": vwap, "returns": returns,
    }


# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------

class AgentRunRequest(BaseModel):
    hypothesis: str = Field("momentum", description="市场假设（自然语言）")
    n_tickers: int  = Field(20, ge=5, le=200)
    n_days: int     = Field(120, ge=60, le=1000)
    seed: int       = Field(42)


class AgentRunResponse(BaseModel):
    hypothesis: str
    initial_dsls: List[str]
    final_dsl: Optional[str]
    final_metrics: Dict[str, Any]
    n_changes: int
    summary: str


class GPEvolveRequest(BaseModel):
    pop_size: int   = Field(20, ge=5, le=200)
    n_gen: int      = Field(5, ge=1, le=50)
    n_workers: int  = Field(1, ge=1, le=8)
    n_tickers: int  = Field(20, ge=5, le=200)
    n_days: int     = Field(120, ge=60, le=1000)
    seed: int       = Field(42)


class GPAlphaItem(BaseModel):
    dsl: str
    fitness: float
    sharpe: float
    ic_ir: float
    ann_return: float
    ann_turnover: float


class GPEvolveResponse(BaseModel):
    n_hof: int
    hof: List[GPAlphaItem]
    saved_ids: List[int]


class BacktestRequest(BaseModel):
    dsl: str = Field("rank(ts_delta(log(close),5))", description="Alpha DSL 表达式")
    n_tickers: int = Field(20, ge=5, le=200)
    n_days: int    = Field(120, ge=60, le=1000)
    seed: int      = Field(42)


class BacktestResponse(BaseModel):
    dsl: str
    report: Dict[str, Any]


class SimulationConfigSchema(BaseModel):
    delay:            int   = Field(1,    ge=0, le=20)
    decay_window:     int   = Field(0,    ge=0, le=60)
    truncation_min_q: float = Field(0.05, ge=0.0, le=0.5)
    truncation_max_q: float = Field(0.95, ge=0.5, le=1.0)
    portfolio_mode:   str   = Field("long_short")
    top_pct:          float = Field(0.10, gt=0.0, lt=0.5)


class RealisticBacktestRequest(BaseModel):
    dsl:       str                  = Field("rank(ts_delta(log(close),5))")
    config:    SimulationConfigSchema = Field(default_factory=SimulationConfigSchema)
    n_tickers: int                  = Field(20, ge=5, le=200)
    n_days:    int                  = Field(120, ge=60, le=1000)
    oos_ratio: float                = Field(0.30, ge=0.0, lt=1.0)
    seed:      int                  = Field(42)


class RealisticBacktestResponse(BaseModel):
    dsl:    str
    is_report:  Dict[str, Any]
    oos_report: Optional[Dict[str, Any]]
    config: Dict[str, Any]


class AlphaRecord(BaseModel):
    id: int
    dsl: str
    hypothesis: Optional[str]
    sharpe: Optional[float]
    ic_ir: Optional[float]
    ann_turnover: Optional[float]
    status: Optional[str]
    created_at: Optional[str]


class ReportQueryResponse(BaseModel):
    total: int
    records: List[AlphaRecord]


# ---------------------------------------------------------------------------
# POST /api/agent/run
# ---------------------------------------------------------------------------

@router.post("/agent/run", response_model=AgentRunResponse, tags=["Agent"])
def agent_run(
    req: AgentRunRequest,
    store: AlphaStore = Depends(get_store),
) -> AgentRunResponse:
    """
    使用 AlphaAgent（LangChain + GPT-4o）执行一轮假设→DSL→评估→修正循环。
    无 OPENAI_API_KEY 时自动使用内置默认 DSL 列表（降级模式）。
    """
    from app.agent.alpha_agent import AlphaAgent

    dataset = _make_synthetic_dataset(req.n_tickers, req.n_days, req.seed)
    agent   = AlphaAgent(store=store)

    try:
        log = agent.run(hypothesis=req.hypothesis, dataset=dataset)
    except Exception as exc:
        logger.exception("AlphaAgent.run failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return AgentRunResponse(
        hypothesis   = log.hypothesis,
        initial_dsls = log.initial_dsls or [],
        final_dsl    = log.final_dsl,
        final_metrics = log.final_metrics or {},
        n_changes    = len(log.changes),
        summary      = log.summary(),
    )


# ---------------------------------------------------------------------------
# POST /api/gp/evolve
# ---------------------------------------------------------------------------

@router.post("/gp/evolve", response_model=GPEvolveResponse, tags=["GP"])
def gp_evolve(
    req: GPEvolveRequest,
    store: AlphaStore = Depends(get_store),
) -> GPEvolveResponse:
    """
    启动 GP 遗传规划进化，返回 Hall of Fame，并将结果持久化到 AlphaStore。
    """
    from app.core.gp_engine.gp_engine import AlphaEvolver
    from app.db.alpha_store import AlphaResult

    dataset = _make_synthetic_dataset(req.n_tickers, req.n_days, req.seed)
    evolver = AlphaEvolver(
        pop_size  = req.pop_size,
        n_gen     = req.n_gen,
        n_workers = req.n_workers,
    )

    try:
        hof = evolver.evolve(dataset)
    except Exception as exc:
        logger.exception("AlphaEvolver.evolve failed")
        raise HTTPException(status_code=500, detail=str(exc))

    saved_ids: List[int] = []
    for r in hof:
        try:
            ar = AlphaResult(
                dsl          = r.dsl,
                sharpe       = r.sharpe,
                ic_ir        = r.ic_ir,
                ann_turnover = r.ann_turnover,
                status       = "active",
            )
            saved_ids.append(store.save(ar))
        except Exception:
            pass

    return GPEvolveResponse(
        n_hof     = len(hof),
        hof       = [GPAlphaItem(**r.__dict__) for r in hof],
        saved_ids = saved_ids,
    )


# ---------------------------------------------------------------------------
# POST /api/backtest/run
# ---------------------------------------------------------------------------

@router.post("/backtest/run", response_model=BacktestResponse, tags=["Backtest"])
def backtest_run(req: BacktestRequest) -> BacktestResponse:
    """
    对单条 Alpha DSL 执行完整回测（解析 → 验证 → 信号生成 → 组合构建 → 回测 → RiskReport）。
    """
    from app.core.alpha_engine.parser import Parser
    from app.core.alpha_engine.validator import AlphaValidator
    from app.core.alpha_engine.dsl_executor import Executor
    from app.core.backtest_engine.backtest_engine import BacktestEngine
    from app.core.backtest_engine.portfolio_constructor import (
        SignalWeightedPortfolio, NeutralizationLayer,
    )
    from app.core.backtest_engine.risk_report import RiskReport

    dataset = _make_synthetic_dataset(req.n_tickers, req.n_days, req.seed)

    # 1. 解析 + 验证
    try:
        node = Parser().parse(req.dsl)
        AlphaValidator().validate(node)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"DSL 解析/验证失败: {exc}")

    # 2. 信号生成
    try:
        signal = Executor().run_expr(req.dsl, dataset)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"DSL 执行失败: {exc}")

    # 3. 组合权重
    weights = SignalWeightedPortfolio().construct(signal)
    NeutralizationLayer.market_neutral(weights)

    # 4. 回测
    try:
        result = BacktestEngine().run(
            weights = weights,
            prices  = dataset["close"],
            volume  = dataset["volume"],
            signal  = signal,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"回测失败: {exc}")

    # 5. RiskReport
    report = RiskReport.from_result(result, prices=dataset["close"])
    report_dict = report.to_dict()

    return BacktestResponse(dsl=req.dsl, report=report_dict)


# ---------------------------------------------------------------------------
# GET /api/report/query
# ---------------------------------------------------------------------------

@router.get("/report/query", response_model=ReportQueryResponse, tags=["Report"])
def report_query(
    min_sharpe: float = Query(-999.0, description="最低 Sharpe 过滤"),
    status: Optional[str] = Query(None, description="状态过滤 (active/retired)"),
    limit: int = Query(50, ge=1, le=500),
    alpha_id: Optional[int] = Query(None, description="查询指定 id"),
    store: AlphaStore = Depends(get_store),
) -> ReportQueryResponse:
    """
    查询 AlphaStore 中已保存的 Alpha 记录。
    可按 Sharpe、状态过滤，或通过 alpha_id 精确查询单条记录。
    """
    if alpha_id is not None:
        rec = store.get_by_id(alpha_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"Alpha id={alpha_id} 不存在")
        records = [rec]
    else:
        records = store.query(min_sharpe=min_sharpe, status=status, limit=limit)

    items = [
        AlphaRecord(
            id           = r.id,
            dsl          = r.dsl,
            hypothesis   = r.hypothesis,
            sharpe       = r.sharpe,
            ic_ir        = r.ic_ir,
            ann_turnover = r.ann_turnover,
            status       = r.status,
            created_at   = str(r.created_at) if r.created_at else None,
        )
        for r in records
    ]
    return ReportQueryResponse(total=len(items), records=items)


# ---------------------------------------------------------------------------
# POST /api/backtest/realistic
# ---------------------------------------------------------------------------

@router.post("/backtest/realistic", response_model=RealisticBacktestResponse, tags=["Backtest"])
def backtest_realistic(req: RealisticBacktestRequest) -> RealisticBacktestResponse:
    """
    执行带信号处理管道（截断→衰减→中性化→延迟）的增强型 IS+OOS 双段回测。
    """
    from app.core.alpha_engine.signal_processor import SimulationConfig
    from app.core.backtest_engine.realistic_backtester import RealisticBacktester
    from app.core.data_engine.data_partitioner import DataPartitioner

    dataset = _make_synthetic_dataset(req.n_tickers, req.n_days, req.seed)

    # 构造 SimulationConfig
    cfg = SimulationConfig(
        delay            = req.config.delay,
        decay_window     = req.config.decay_window,
        truncation_min_q = req.config.truncation_min_q,
        truncation_max_q = req.config.truncation_max_q,
        portfolio_mode   = req.config.portfolio_mode,
        top_pct          = req.config.top_pct,
    )

    # IS/OOS 分割
    oos_dataset = None
    if req.oos_ratio > 0:
        try:
            dates = next(iter(dataset.values())).index
            partitioner = DataPartitioner(
                start     = str(dates[0].date()),
                end       = str(dates[-1].date()),
                oos_ratio = req.oos_ratio,
            )
            partitioned = partitioner.partition(dataset)
            is_data  = partitioned.train()
            oos_dataset = partitioned.test()
        except Exception as exc:
            logger.warning("DataPartitioner 分割失败，使用全量数据作 IS: %s", exc)
            is_data = dataset
    else:
        is_data = dataset

    backtester = RealisticBacktester(config=cfg)
    try:
        result = backtester.run(req.dsl, is_data, oos_dataset=oos_dataset)
    except Exception as exc:
        logger.exception("RealisticBacktester.run failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return RealisticBacktestResponse(
        dsl        = req.dsl,
        is_report  = result.is_report.to_dict(),
        oos_report = result.oos_report.to_dict() if result.oos_report else None,
        config     = result.to_dict()["config"],
    )


# ===========================================================================
# Phase 2 — /alpha/simulate  &  /alpha/optimize
# ===========================================================================

# ---------------------------------------------------------------------------
# Phase 2 Pydantic Schemas
# ---------------------------------------------------------------------------

class SearchSpaceSchema(BaseModel):
    delay_range:      List[int]   = Field([0, 5],   min_length=2, max_length=2)
    decay_range:      List[int]   = Field([0, 10],  min_length=2, max_length=2)
    trunc_min_range:  List[float] = Field([0.01, 0.10], min_length=2, max_length=2)
    trunc_max_range:  List[float] = Field([0.90, 0.99], min_length=2, max_length=2)
    portfolio_modes:  List[str]   = Field(["long_short", "decile"])
    allow_neutralize: bool        = Field(False)


class SimulateRequest(BaseModel):
    dsl:       str                   = Field("rank(ts_delta(log(close),5))")
    config:    SimulationConfigSchema = Field(default_factory=SimulationConfigSchema)
    n_tickers: int                   = Field(20, ge=5, le=200)
    n_days:    int                   = Field(252, ge=60, le=2000)
    oos_ratio: float                 = Field(0.30, ge=0.0, lt=1.0)
    seed:      int                   = Field(42)


class OptimizeRequest(BaseModel):
    dsl:          str               = Field("rank(ts_delta(log(close),5))")
    search_space: SearchSpaceSchema = Field(default_factory=SearchSpaceSchema)
    n_trials:     int               = Field(30, ge=1, le=200)
    n_tickers:    int               = Field(20, ge=5, le=200)
    n_days:       int               = Field(252, ge=60, le=2000)
    oos_ratio:    float             = Field(0.30, ge=0.0, lt=1.0)
    seed:         int               = Field(42)


class EvalResponse(BaseModel):
    dsl:               str
    best_config:       Optional[Dict[str, Any]]
    is_metrics:        Dict[str, Any]
    oos_metrics:       Optional[Dict[str, Any]]
    overfitting_score: float
    is_overfit:        bool
    ic_decay:          Dict[str, Any]
    n_trials_run:      Optional[int]
    # PnL series for visualization (daily net returns, IS then OOS)
    pnl_is:            List[float] = Field(default_factory=list)
    pnl_oos:           List[float] = Field(default_factory=list)
    split_date:        Optional[str] = None


# ---------------------------------------------------------------------------
# 共享辅助：分区 + 评估
# ---------------------------------------------------------------------------

def _partition_dataset(dataset, oos_ratio: float, seed: int = 42):
    """返回 (is_data, oos_data)，oos_ratio=0 时 oos_data=None。"""
    from app.core.data_engine.data_partitioner import DataPartitioner

    if oos_ratio <= 0:
        return dataset, None

    dates = next(iter(dataset.values())).index
    dp = DataPartitioner(
        start     = str(dates[0].date()),
        end       = str(dates[-1].date()),
        oos_ratio = oos_ratio,
    )
    part = dp.partition(dataset)
    return part.train(), part.test()


def _run_evaluate(
    dsl:         str,
    config,                 # SimulationConfig
    is_data:     dict,
    oos_data:    Optional[dict],
    best_config: Optional[dict] = None,
    n_trials:    Optional[int]  = None,
) -> EvalResponse:
    """
    共享逻辑：用给定 config 执行 IS+OOS 回测 + AlphaEvaluator 高级评估。
    """
    from app.core.backtest_engine.realistic_backtester import RealisticBacktester
    from app.core.ml_engine.alpha_evaluator import AlphaEvaluator

    bt     = RealisticBacktester(config=config)
    result = bt.run(dsl, is_data, oos_dataset=oos_data)

    evaluator = AlphaEvaluator()

    is_prices = is_data.get("close")
    is_signal = result.processed_signal

    oos_prices = oos_data.get("close") if oos_data else None
    oos_signal: Optional[Any] = None
    if result.oos_result is not None and oos_prices is not None:
        # OOS 段信号：通过重新运行 SignalProcessor 得到
        # （result.processed_signal 是 IS 段的）
        from app.core.alpha_engine.signal_processor import SignalProcessor
        from app.core.alpha_engine.parser import Parser
        from app.core.alpha_engine.dsl_executor import Executor
        node       = Parser().parse(dsl)
        raw_oos    = Executor().run(node, oos_data)
        oos_signal = SignalProcessor(config).process(raw_oos)

    eval_result = evaluator.evaluate(
        is_report  = result.is_report,
        is_prices  = is_prices,
        is_signal  = is_signal,
        oos_report = result.oos_report,
        oos_prices = oos_prices,
        oos_signal = oos_signal,
    )

    # IC Decay 回填到 is_metrics
    eval_dict = eval_result.to_dict()
    eval_dict["is_metrics"]["ic_decay_t1"] = eval_result.ic_decay.get("t1")
    eval_dict["is_metrics"]["ic_decay_t5"] = eval_result.ic_decay.get("t5")

    # ── Extract PnL series for frontend visualization ───────────────────────
    import pandas as pd, numpy as np

    def _series_to_list(s) -> list:
        """Convert pd.Series / np.ndarray / list to plain Python float list."""
        if s is None:
            return []
        if isinstance(s, pd.Series):
            return [float(v) for v in s.dropna().values]
        if isinstance(s, np.ndarray):
            return [float(v) for v in s[~np.isnan(s)]]
        return list(s)

    pnl_is:   list = []
    pnl_oos:  list = []
    split_dt: str | None = None

    # Try net_returns → equity_curve in that priority
    if result.is_report is not None:
        nr = getattr(result.is_report, "net_returns", None)
        if nr is None:
            nr = getattr(result.is_report, "equity_curve", None)
        pnl_is = _series_to_list(nr)
        if isinstance(nr, pd.Series) and len(nr) > 0:
            split_dt = str(nr.index[-1].date())

    if result.oos_report is not None:
        nr_oos = getattr(result.oos_report, "net_returns", None)
        if nr_oos is None:
            nr_oos = getattr(result.oos_report, "equity_curve", None)
        pnl_oos = _series_to_list(nr_oos)

    return EvalResponse(
        dsl               = dsl,
        best_config       = best_config,
        is_metrics        = eval_dict["is_metrics"],
        oos_metrics       = eval_dict["oos_metrics"],
        overfitting_score = eval_result.overfitting_score,
        is_overfit        = eval_result.is_overfit,
        ic_decay          = eval_result.ic_decay,
        n_trials_run      = n_trials,
        pnl_is            = pnl_is,
        pnl_oos           = pnl_oos,
        split_date        = split_dt,
    )


# ---------------------------------------------------------------------------
# POST /alpha/simulate
# ---------------------------------------------------------------------------

@router.post("/alpha/simulate", response_model=EvalResponse, tags=["Phase2"])
def alpha_simulate(req: SimulateRequest) -> EvalResponse:
    """
    手动模式：用用户提供的 SimulationConfig 执行 IS+OOS 回测，
    返回 IS 和 OOS 完整高级指标（含过拟合评分、IC Decay）。
    """
    from app.core.alpha_engine.signal_processor import SimulationConfig

    dataset = _make_synthetic_dataset(req.n_tickers, req.n_days, req.seed)
    is_data, oos_data = _partition_dataset(dataset, req.oos_ratio)

    cfg = SimulationConfig(
        delay            = req.config.delay,
        decay_window     = req.config.decay_window,
        truncation_min_q = req.config.truncation_min_q,
        truncation_max_q = req.config.truncation_max_q,
        portfolio_mode   = req.config.portfolio_mode,
        top_pct          = req.config.top_pct,
    )

    try:
        return _run_evaluate(req.dsl, cfg, is_data, oos_data)
    except Exception as exc:
        logger.exception("alpha_simulate failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# POST /alpha/optimize
# ---------------------------------------------------------------------------

@router.post("/alpha/optimize", response_model=EvalResponse, tags=["Phase2"])
def alpha_optimize(req: OptimizeRequest) -> EvalResponse:
    """
    Auto-ML 模式：
      1. Optuna 在 IS 数据集上搜索最优 SimulationConfig（OOS 全程锁定）
      2. 用最优参数对 IS+OOS 进行最终评估
      3. 返回最优参数 + IS/OOS 高级指标 + 过拟合评分

    严格遵循 Train → Optimize → Lock → Test 工作流。
    """
    from app.core.ml_engine.alpha_optimizer import AlphaOptimizer, SearchSpace

    dataset = _make_synthetic_dataset(req.n_tickers, req.n_days, req.seed)

    # Step 1: 分区 — OOS 此后物理隔离
    is_data, oos_data = _partition_dataset(dataset, req.oos_ratio)

    # Step 2: 构造搜索空间
    ss = req.search_space
    search_space = SearchSpace(
        delay_range      = tuple(ss.delay_range),
        decay_range      = tuple(ss.decay_range),
        trunc_min_range  = tuple(ss.trunc_min_range),
        trunc_max_range  = tuple(ss.trunc_max_range),
        portfolio_modes  = tuple(ss.portfolio_modes),
        allow_neutralize = ss.allow_neutralize,
    )

    # Step 3: Optuna 优化（仅传入 IS）
    try:
        optimizer = AlphaOptimizer(
            dsl          = req.dsl,
            is_dataset   = is_data,   # OOS 完全不传入
            search_space = search_space,
            n_trials     = req.n_trials,
            seed         = req.seed,
        )
        best_config, study_summary = optimizer.optimize()
    except Exception as exc:
        logger.exception("AlphaOptimizer.optimize failed")
        raise HTTPException(status_code=500, detail=f"优化失败: {exc}")

    # Step 4: 最优参数锁定后，执行 IS+OOS 最终评估（OOS 解锁）
    try:
        return _run_evaluate(
            dsl         = req.dsl,
            config      = best_config,
            is_data     = is_data,
            oos_data    = oos_data,
            best_config = study_summary.best_params,
            n_trials    = study_summary.n_trials,
        )
    except Exception as exc:
        logger.exception("alpha_optimize final evaluation failed")
        raise HTTPException(status_code=500, detail=f"最终评估失败: {exc}")
