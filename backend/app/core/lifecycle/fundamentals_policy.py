"""
fundamentals_policy.py — 基本面因子本轮不正式晋级（2026-10-22 用户决定 D1）

用户原话："基本面暂不正式晋级。允许选择段内预登记诊断和试验记账，不自动消耗最终 Test。"

    不晋级   用到基本面字段的表达式不能流转到 validated / paper / active（`AlphaStore.update_status` 统一拦，
             API、验证门、发现、审批走的都是它）；验证门对它们只给诊断（`diagnostic_only`），不给通过；
             策略提案里多一道必须的门 `fundamentals_policy`
    诊断     `run_fundamental_diagnostics`：先**预登记**（谁、为什么、哪几条表达式的摘要、条数）并把条数记进
             全局试验史，再在选择段上逐条跑验证门 —— 全程不碰冻结 Test 段（report_test=False，不记 holdout 使用）
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Dict, Iterable, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

#: 正式晋级的状态（进了它们就意味着"通过了验证 / 可以用来交易"）
PROMOTED_STATUSES = frozenset({"validated", "paper", "active"})


def fundamental_fields(expressions: Iterable[Optional[str]]) -> List[str]:
    from app.core.data_engine.fundamentals.quality import fundamental_fields_in
    return fundamental_fields_in(expressions)


def promotion_problem(dsl: Optional[str]) -> str:
    """这条表达式能不能正式晋级：能 → ""；不能 → 原因（D1）。"""
    used = fundamental_fields([dsl])
    if not used:
        return ""
    return (f"D1：基本面因子本轮不正式晋级（用到 {used}）—— 只允许选择段内预登记诊断与试验记账")


def run_fundamental_diagnostics(dsls: List[str], dataset: Dict[str, pd.DataFrame], dataset_key: str, *,
                                registered_by: str, note: str, gate_kwargs: Optional[dict] = None) -> dict:
    """
    预登记的选择段诊断。顺序固定：校验登记信息 → 研究上下文（只给选择段）→ 登记运行并把**全部**条数记进
    试验史（之后哪条报错也照样算尝试过）→ 逐条跑验证门（不动用 Test 段）→ 结案。
    返回 {"registration": {...}, "results": [每条的诊断]}。
    """
    registered_by, note = (registered_by or "").strip(), (note or "").strip()
    if not registered_by or not note:
        raise ValueError("预登记诊断必须写明登记人和目的")
    dsls = [str(d) for d in dsls if str(d).strip()]
    if not dsls:
        raise ValueError("没有要诊断的表达式")
    from app.core.data_engine.research_context import open_research
    from app.core.lifecycle.validation_gate import ValidationGate
    from app.db.trial_ledger import TrialRecorder

    digest = hashlib.sha256(json.dumps(sorted(dsls), ensure_ascii=False).encode("utf-8")).hexdigest()
    registration = {"registered_by": registered_by, "note": note, "digest": digest, "n": len(dsls),
                    "dataset_key": dataset_key, "fields": fundamental_fields(dsls)}
    ctx = open_research(dataset, dataset_key, "fund_diag")
    results: List[dict] = []
    with ctx.active():
        rec = TrialRecorder("fund_diag", panels=[ctx.selection])
        rec.add(len(dsls))                                       # 预登记的条数先记账
        logger.warning("[D1] 预登记基本面诊断 %d 条（%s：%s，摘要 %s）—— 只在选择段上跑，不晋级、不动 Test",
                       len(dsls), registered_by, note, digest[:12])
        status = "done"
        try:
            gate = ValidationGate(**(gate_kwargs or {}))
            for dsl in dsls:
                try:
                    res = gate.evaluate(dsl, dataset, n_trials=None, dataset_key=dataset_key, report_test=False)
                    d = res.to_dict()
                except Exception as exc:                        # 单条失败照样算尝试过（已记账）
                    logger.error("[D1] 诊断 %s 失败: %s", dsl[:60], exc)
                    d = {"dsl": dsl, "error": str(exc)}
                d.update(diagnostic_only=True, promotable=False)
                results.append(d)
        except BaseException:
            status = "aborted"
            raise
        finally:
            rec.finish(status)
    return {"registration": registration, "context": ctx.describe(), "results": results}
