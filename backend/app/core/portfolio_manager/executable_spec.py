"""
executable_spec.py — 一份**可执行策略规格**（外部审查 2026-10-18 F07）

为什么需要
----------
此前审批、验证、内部模拟、券商执行各自装配策略：配置冻结了成分和合成权重，但 AUM、无交易带、
风控上限、信号参数都在运行时从 `settings` 现读；每日策略门还按默认 `ic_weighted` 重新拟合，
评估的不是被批准的那组冻结权重。审查的 R8：批准的是 equal_weight / AUM 1234 / band .123，
每日门是 ic_weighted，账本用 AUM 1,000,000 / band .00134。改一个全局设置也不会触发重新审批。

现在
----
提案时把**影响交易的全部参数**收进一份规格，算哈希，随配置存下（`verdict["executable_spec"]` /
`verdict["spec_hash"]`，免表结构迁移）。每日循环：

  · **照规格执行**：成分、冻结合成权重、合成方法、无交易带从规格里取；
  · **核对规格**：AUM、风控上限、信号参数、券商与做空口径、数据集与价格源、各成分的表达式
    与运行时比对 —— 任何一项变了 → 本轮不调仓（只对账 / 熔断维护），要求重新提案、审批。
  · 规格缺失（2026-10-18 之前批准的配置）→ 同样不调仓：没有规格就无法证明交易的是被批准的那份。

哈希只覆盖规格内容；`created_at`、门的证据等不进哈希（证据可以补充，交易参数不可以悄悄改）。
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

#: 2：加入 universe / execution / submission（复审 2026-10-19 §5.2）
SPEC_VERSION = 3          # v3（2026-10-25 审计）：加 `validation`（表达式校验政策的版本、摘要与结果）

#: 每日循环 / 提案 / 策略门共用的信号处理参数（此前三处各写一份字面量）。
#: delay 取自时钟（c2 = 0：T 收盘的信息就用于 T 的决策，T+1 执行由引擎负责；见 app/core/clock.py）
from app.core.clock import SIGNAL_DELAY as _SIGNAL_DELAY  # noqa: E402

SIGNAL_CONFIG = {"delay": _SIGNAL_DELAY, "decay_window": 0, "truncation_min_q": 0.05, "truncation_max_q": 0.95}


def signal_config():
    from app.core.alpha_engine.signal_processor import SimulationConfig
    return SimulationConfig(**SIGNAL_CONFIG)


def risk_limits_from_settings(settings=None):
    """生产风控上限的**唯一**装配处（每日循环、提案、策略门都用它）。"""
    from app.core.portfolio_manager.risk_gate import RiskLimits
    if settings is None:
        from app.config import settings
    return RiskLimits(
        max_gross=float(getattr(settings, "risk_max_gross", 1.0)),
        max_net=float(getattr(settings, "risk_max_net", 1.0)),
        max_name_weight=float(getattr(settings, "risk_max_name_weight", 0.10)),
        max_sector_weight=float(getattr(settings, "risk_max_sector_weight", 0.30)),
        target_vol_ann=(float(getattr(settings, "risk_target_vol_ann", 0.0)) or None),
        max_drawdown=float(getattr(settings, "risk_max_drawdown", 0.20)),
        long_only=(not getattr(settings, "trading_allow_short", False)),
    )


def _r(x: float) -> float:
    return round(float(x), 10)


def build_spec(factor_dsl: Dict[str, Optional[str]], combo_weights: Dict[str, float], method: str,
               aum: float, no_trade_band: float, limits=None, settings=None) -> dict:
    """规格字典。`limits` / 券商 / 数据口径缺省读 settings（与每日循环同一来源）。"""
    if settings is None:
        from app.config import settings
    limits = limits if limits is not None else risk_limits_from_settings(settings)
    return {
        "version": SPEC_VERSION,
        "factors": {str(k): (None if v is None else str(v)) for k, v in sorted(factor_dsl.items())},
        "combo_weights": {str(k): _r(v) for k, v in sorted((combo_weights or {}).items())},
        "method": str(method),
        "aum": _r(aum),
        "no_trade_band": _r(no_trade_band),
        "risk_limits": {k: (None if v is None else (_r(v) if isinstance(v, float) else v))
                        for k, v in asdict(limits).items()},
        # 目标波动缩放用的波动估计窗口（复审 2026-10-19 N05-e：它改变目标权重，此前不在规格里）
        "risk_vol_lookback": int(getattr(settings, "risk_vol_lookback", 60)),
        "signal": dict(SIGNAL_CONFIG),
        # 信号 / 成交 / IC 的时钟口径（复审 2026-10-19 §5.1）：D14 改口径 = 新的策略版本
        "clock": _clock(),
        "trading": {"broker": str(getattr(settings, "trading_broker", "moomoo_us")),
                    "allow_short": bool(getattr(settings, "trading_allow_short", False))},
        "data": {"dataset": str(getattr(settings, "paper_dataset", "")),
                 "price_source": str(getattr(settings, "price_source", ""))},
        "universe": _universe_section(str(getattr(settings, "paper_dataset", ""))),
        "execution": _execution_section(settings),
        "submission": _submission_section(settings),
        # 表达式校验凭据（2026-10-25 审计）：按 CURRENT 政策校验的版本 / 摘要 / 结果；在规格里 → 绑定在 spec_hash 上
        "validation": _validation_section(factor_dsl),
    }


def _validation_section(factor_dsl: Dict[str, Optional[str]]) -> dict:
    from app.core.alpha_engine.validation_policy import validation_record
    return validation_record({str(k): v for k, v in factor_dsl.items()})


def _clock() -> str:
    from app.core.data_engine.input_identity import clock_version
    return clock_version()


def _universe_section(dataset: str) -> dict:
    """有效股票池：数据集登记的标的列表（排序后的哈希）。登记里换了票就是另一个策略。"""
    try:
        from app.core.data_engine.dataset_registry import registry_spec
        tickers = sorted(str(t) for t in registry_spec(dataset).universe)
    except Exception as exc:
        logger.warning("[spec] 数据集 %s 的股票池读不到，规格里记为空: %s", dataset, exc)
        tickers = []
    return {"tickers_digest": hashlib.sha256("\x1f".join(tickers).encode("utf-8")).hexdigest()[:16],
            "n": len(tickers)}


def _execution_section(settings) -> dict:
    """执行账户与下单 / 报价规则（下单门与订单管理实际读取的那几项）。"""
    from app.core.execution.pretrade_gate import REGULAR_CLOSE_TOL
    return {
        "account_firm": str(getattr(settings, "moomoo_security_firm", "")),
        "account_id": int(getattr(settings, "moomoo_trd_acc_id", 0) or 0),
        "limit_band_bps": _r(getattr(settings, "exec_limit_band_bps", 0.0)),
        "flatten_band_bps": _r(getattr(settings, "exec_flatten_band_bps", 0.0)),
        "max_daily_loss": _r(getattr(settings, "exec_max_daily_loss", 0.0)),
        "max_price_deviation": _r(getattr(settings, "exec_max_price_deviation", 0.0)),
        "max_quote_age_days": int(getattr(settings, "exec_max_quote_age_days", 0)),
        "quote_close_tol_min": int(REGULAR_CLOSE_TOL.total_seconds() // 60),
    }


def _submission_section(settings) -> dict:
    """提交时段：调度时刻、时区、订单有效期、是否允许常规盘外成交。"""
    from app.core.execution.broker_gateway import ORDER_OUTSIDE_RTH, ORDER_TIME_IN_FORCE
    from app.tasks.scheduler import DAILY_TRADING_AT, DAILY_TRADING_RETRY_AT
    return {
        "daily_trading": "%02d:%02d" % DAILY_TRADING_AT,
        "retry": "%02d:%02d" % DAILY_TRADING_RETRY_AT,
        "scheduler_timezone": str(getattr(settings, "scheduler_timezone", "")),
        "time_in_force": ORDER_TIME_IN_FORCE,
        "outside_rth": bool(ORDER_OUTSIDE_RTH),
    }


def spec_hash(spec: dict) -> str:
    blob = json.dumps(spec, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def spec_diff(approved: dict, current: dict) -> List[str]:
    """两份规格的差异（人能读的一行一条）。空 = 一致。"""
    out: List[str] = []
    for key in sorted(set(approved) | set(current)):
        a, b = approved.get(key), current.get(key)
        if a == b:
            continue
        if isinstance(a, dict) and isinstance(b, dict):
            for sub in sorted(set(a) | set(b)):
                if a.get(sub) != b.get(sub):
                    out.append(f"{key}.{sub}: 批准时 {a.get(sub)!r} → 现在 {b.get(sub)!r}")
        else:
            out.append(f"{key}: 批准时 {a!r} → 现在 {b!r}")
    return out


def approved_spec(config_record) -> Optional[dict]:
    """从策略配置记录里取出批准时的规格（没有 → None）。哈希对不上同样当作没有。"""
    try:
        verdict = json.loads(getattr(config_record, "verdict", None) or "{}")
    except (TypeError, ValueError) as exc:
        logger.error("[spec] 策略 #%s 的 verdict 解析失败: %s", getattr(config_record, "id", "?"), exc)
        return None
    spec = verdict.get("executable_spec")
    if not isinstance(spec, dict):
        return None
    if spec_hash(spec) != verdict.get("spec_hash"):
        logger.error("[spec] 策略 #%s 的规格与存下的哈希不一致（被改过？）→ 按没有规格处理",
                     getattr(config_record, "id", "?"))
        return None
    return spec


def verify_against_runtime(spec: dict, factor_dsl: Dict[str, Optional[str]], aum: float,
                           settings=None) -> List[str]:
    """
    规格里"运行时也能算出来"的部分与现在比对：AUM、风控、信号、券商、数据、各成分表达式。
    合成权重 / 方法 / 无交易带是**照规格执行**的，不在这里比。
    """
    current = build_spec(factor_dsl, spec.get("combo_weights") or {}, spec.get("method", ""),
                         aum, float(spec.get("no_trade_band", 0.0)), settings=settings)
    return spec_diff(spec, current)


# ---------------------------------------------------------------------------
# 共用的风险路径（复审 2026-10-19 N05：策略门与每日循环此前各拼一套）
# ---------------------------------------------------------------------------

#: 波动估计至少要这么多个已实现日收益
MIN_VOL_SAMPLE = 20


def estimate_port_vol(weights, close, lookback: int = 60) -> Optional[float]:
    """
    组合年化波动的估计：**已实现**日收益 = 权重(t−1) · 收益(t)（不用当日权重看当日收益）。
    样本不足 MIN_VOL_SAMPLE / 非有限 / 非正 → None（调用方据此本轮不做目标波动缩放）。
    """
    import numpy as np
    rets = close.pct_change().reindex(columns=weights.columns)
    port_ret = (weights.shift(1) * rets).sum(axis=1).dropna()
    sample = port_ret.tail(int(lookback))
    if len(sample) < MIN_VOL_SAMPLE:
        return None
    vol = float(sample.std(ddof=1) * np.sqrt(252.0))
    return vol if np.isfinite(vol) and vol > 1e-9 else None


def band_then_reproject(weights, band: float, limits, sectors=None):
    """
    无交易带 → **硬约束投影**（不再传波动估计：波动缩放是一次性的，第二次传会把同一个系数乘两遍）。
    无交易带会取消小幅减仓、保留大幅加仓，重新引入上游已削掉的超限，所以带后必须再投影一次。
    """
    from app.core.portfolio_manager.horizon import apply_no_trade_band
    from app.core.portfolio_manager.risk_gate import PortfolioRiskGate
    banded = apply_no_trade_band(weights, float(band))
    out, _ = PortfolioRiskGate(limits).apply(banded, sectors=sectors, port_vol_ann=None)
    return out

