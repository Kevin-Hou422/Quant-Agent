"""
validation_design.py — 预登记的验证设计（复审 2026-10-19 N05-d）

复审原话："若冻结权重是在整个选择段拟合，不能再把其在同段收益当作未见样本外；
新规格的资格应来自预登记的验证设计和此后真正未见的前向数据。"

此前 →ACTIVE 门判定时现读 settings 的门槛：提案之后把 `TR_MIN_FORWARD_DAYS` 调低、或在看到前向数字之后
改 t 阈值，门就按新门槛判 —— 那就不是"预先登记"的检验了；证据也是这份规格的全部前向行，没有"登记之后"这条线。

现在提案时登记一份设计（随 verdict 存下，带哈希）：
  · 门槛：最少前向天数、IC 均值下限、t 阈值（取登记那一刻的配置，之后改配置不影响这份策略）
  · t 的口径：普通 t 与 Newey-West t 取较小者
  · 证据口径：这份规格（同一 spec_hash）的 **forward** 类证据，且结果日在登记日**之后**
  · 时钟口径（§5.1）
→ACTIVE 门按登记的设计判；设计缺失（2026-10-20 之前提出的策略）或被改过（哈希不符）→ 不通过。
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

DESIGN_VERSION = 1
EVIDENCE_RULE = "strategy_evidence.forward, same spec_hash, date after registration"
TSTAT_RULE = "min(iid, newey_west)"


def register_design(th=None, now: Optional[datetime] = None) -> Dict[str, Any]:
    """提案时调用：把**此刻**的门槛与口径冻结成一份设计。"""
    from app.core.data_engine.input_identity import clock_version
    from app.core.lifecycle.promotion_gate import PromotionThresholds
    th = th or PromotionThresholds.from_settings()
    t = pd.Timestamp(now or datetime.now(timezone.utc))
    t = t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t
    return {
        "version": DESIGN_VERSION,
        "registered_at": t.isoformat(timespec="seconds"),
        "min_forward_days": int(th.min_forward_days),
        "min_ic_mean": float(th.min_ic_mean),
        "min_ic_tstat": float(th.min_ic_tstat),
        "tstat": TSTAT_RULE,
        "evidence": EVIDENCE_RULE,
        "clock_version": clock_version(),
    }


def design_hash(design: Dict[str, Any]) -> str:
    blob = json.dumps(design, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def check_registered_activation(config_record, evidence_rows: Iterable[Any]) -> Tuple[bool, dict]:
    """
    →ACTIVE 判定：按这份策略**预登记**的设计，只用登记之后的 forward 证据。
    evidence_rows：`EvidenceStore.strategy_forward_ic(sid, spec_hash)` 的结果（带 .date / .ic）。
    """
    from app.core.data_engine.input_identity import clock_version
    from app.core.lifecycle.promotion_gate import PromotionThresholds, check_active_promotion
    rows = list(evidence_rows)
    try:
        verdict = json.loads(getattr(config_record, "verdict", None) or "{}")
    except (TypeError, ValueError) as exc:
        logger.error("[validation_design] 策略 #%s 的 verdict 解析失败 → →ACTIVE 不通过: %s",
                     getattr(config_record, "id", "?"), exc)
        return False, _refuse(f"verdict 解析失败：{exc}", len(rows))
    design = verdict.get("validation_design")
    if not isinstance(design, dict):
        return False, _refuse("没有预登记的验证设计（2026-10-20 之前提出的策略须重新提案）", len(rows))
    if design_hash(design) != verdict.get("validation_design_hash"):
        return False, _refuse("验证设计与登记时的哈希不一致（被改过？）", len(rows))
    if design.get("clock_version") != clock_version():
        return False, _refuse(f"验证设计登记时的时钟口径是 {design.get('clock_version')}，现在是 {clock_version()}",
                              len(rows))
    reg = pd.Timestamp(design["registered_at"]).date()
    after = [r for r in rows if getattr(r, "evidence_class", "forward") in (None, "forward")
             and pd.Timestamp(r.date).date() > reg]
    th = PromotionThresholds(min_forward_days=int(design["min_forward_days"]),
                             min_ic_mean=float(design["min_ic_mean"]),
                             min_ic_tstat=float(design["min_ic_tstat"]))
    ok, detail = check_active_promotion([float(r.ic) for r in after], th)
    detail.update({"design": design, "registered_at": design["registered_at"],
                   "n_before_registration": len(rows) - len(after)})
    return ok, detail


def _refuse(reason: str, n: int) -> dict:
    return {"step": "active_promotion", "passed": False, "reasons": [reason], "n_days": n}
