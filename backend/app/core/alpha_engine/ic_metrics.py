"""
ic_metrics.py — 因子记录里的 IC 指标：均值与 IC-IR 分列、样本与评估状态（2026-10-25 审计的决定 2）

决定原话："IC 均值与真正 IC-IR 分列，不接受直接给混合语义的一列统一换名后宣称修好。新增明确的 `ic_mean`；`ic_ir` 只存
已定义公式的 IC-IR。统一 IC 序列、有效样本、标准差自由度、零方差/样本不足处理、是否年化。没有足够样本时为 NULL，并写原因。
同步记录 n_obs、样本区间、评估状态、数据源/摘要、clock、计算版本和研究切分。……'从未评估'与'评估后恰好为 0'必须分开，
不再靠所有指标是否为 0 推断 evaluated。"

口径（版本 ICM_VERSION，改口径 = 换版本）：

    IC 序列   逐日截面 Spearman rank IC（`PerformanceAnalyzer.rolling_rank_ic*`，与回测报告同一序列）
    ic_mean   序列的算术均值；没有有效观测 → NULL
    ic_ir     mean / std（ddof = 1），**不年化**；有效观测 < MIN_OBS_FOR_IR 或标准差为 0 → NULL，原因写进 ic_metrics
    状态      evaluated（算过，ic_mean 有值，哪怕恰好是 0）/ not_computed（这条路径没有算 IC，写明原因）/
              never_evaluated（入库时就没有任何评估）/ values_without_provenance（给了数值、没给样本与来源）。旧行没有 ic_metrics → 读出来是 legacy_unknown：原值保留、来源靠
              写入路径推断（tools/research/factor_metrics_dry_run.py），不当作已评估
"""

from __future__ import annotations

import json
import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

ICM_VERSION = "icm1"
MIN_OBS_FOR_IR = 20

EVALUATED = "evaluated"
NOT_COMPUTED = "not_computed"
NEVER_EVALUATED = "never_evaluated"
LEGACY_UNKNOWN = "legacy_unknown"
#: 调用方给了数值、却没给样本 / 切分 / 来源 —— 数值保留，但不能当作按统一口径评估过
NO_PROVENANCE = "values_without_provenance"


def _finite(v) -> Optional[float]:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def ic_fields(mean_ic, ic_ir, n_obs, *, source: str, segment: str, window=None, clock: Optional[str] = None,
              data_digest: str = "") -> dict:
    """
    一次评估的产物 → 写进因子记录的三个字段 {ic_mean, ic_ir, ic_metrics}。
    `ic_ir` 由调用方按本模块口径（mean/std, ddof=1, 不年化）给出；样本不足 / 零方差 / 非有限 → NULL 并写原因。
    """
    if clock is None:
        from app.core.data_engine.input_identity import clock_version
        clock = clock_version()
    n = int(n_obs) if _finite(n_obs) is not None else 0
    mean = _finite(mean_ic) if n > 0 else None
    ir = _finite(ic_ir)
    why = ""
    if mean is None:
        status, ir, why = NOT_COMPUTED, None, "没有有效的 IC 观测"
    else:
        status = EVALUATED
        if n < MIN_OBS_FOR_IR:
            ir, why = None, f"有效观测 {n} < {MIN_OBS_FOR_IR}，IC-IR 不可靠，记为空"
        elif ir is None:
            why = "IC 序列标准差为 0 或 IC-IR 算不出，记为空"
    meta = {"status": status, "n_obs": n, "window": list(window) if window else None, "segment": segment,
            "source": source, "clock": clock, "version": ICM_VERSION, "ic_ir_formula": "mean/std(ddof=1)，不年化",
            "ic_ir_null_reason": why or None, "data_digest": data_digest or None}
    return {"ic_mean": mean, "ic_ir": ir, "ic_metrics": json.dumps(meta, ensure_ascii=False)}


def not_computed(source: str, reason: str) -> dict:
    """这条写入路径没有算 IC（例如 GP 池只有适应度）。"""
    meta = {"status": NOT_COMPUTED, "source": source, "reason": reason, "version": ICM_VERSION}
    return {"ic_mean": None, "ic_ir": None, "ic_metrics": json.dumps(meta, ensure_ascii=False)}


def no_provenance(source: str) -> dict:
    meta = {"status": NO_PROVENANCE, "source": source, "version": ICM_VERSION,
            "reason": "写入时给了 IC 数值，但没有样本数、切分与来源"}
    return {"ic_metrics": json.dumps(meta, ensure_ascii=False)}


def never_evaluated(source: str) -> dict:
    meta = {"status": NEVER_EVALUATED, "source": source, "version": ICM_VERSION}
    return {"ic_mean": None, "ic_ir": None, "ic_metrics": json.dumps(meta, ensure_ascii=False)}


def from_workflow_metrics(m: dict, *, source: str, segment: str = "in_sample") -> dict:
    """工作流 / GP 的指标字典（is_ic、is_ic_ir、is_ic_n）→ 三个字段。"""
    return ic_fields(m.get("is_ic"), m.get("is_ic_ir"), m.get("is_ic_n"), source=source, segment=segment)


def status_of(record) -> dict:
    """读一条因子记录的 IC 状态。没有 ic_metrics 的旧行 → legacy_unknown（不按数值是否为 0 推断）。"""
    raw = getattr(record, "ic_metrics", None)
    if not raw:
        return {"status": LEGACY_UNKNOWN, "note": "本项之前写入的旧行：ic_ir 列里可能是样本内 IC 均值，原值保留"}
    try:
        return json.loads(raw)
    except (TypeError, ValueError) as exc:
        logger.warning("[ic_metrics] 因子 #%s 的 ic_metrics 解析失败（按口径未知处理）: %s", getattr(record, "id", "?"), exc)
        return {"status": LEGACY_UNKNOWN, "note": "ic_metrics 解析失败"}
