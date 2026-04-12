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
from app.core.ml_engine.alpha_store import AlphaStore

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
    from app.core.ml_engine.alpha_agent import AlphaAgent

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
    from app.core.ml_engine.alpha_store import AlphaResult

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
