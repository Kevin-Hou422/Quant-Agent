"""
validation_policy.py — 表达式校验政策：版本化、批准时冻结进规格、运行时按冻结的那一版解释

2026-10-25 审计（对"批准规格运行时是否重新校验表达式"的决定）："按批准时冻结的校验政策运行；不要每天强行改用最新研究
规则，但也不同意永久、无条件跳过所有校验。……记录 DSL 语义/操作符版本、validator_policy_version 或摘要、资源边界、
校验结果及其绑定的 spec_hash。加载已批准规格时核对完整性和兼容性，按该已批准政策解释。普通政策收紧通过重新提案迁移；
重大安全问题、未来信息读取、不支持的操作符或超出安全资源边界仍须阻止新增风险……旧规格没有凭据的，做一次明确迁移核验
或重新提案，不追认不存在的审批证据。不要偷偷截断 300 为 252。"

两层规则：
    政策（POLICIES，按版本）  研究口径的规则：窗口上限、嵌套深度、阶梯字段的最短离散窗口。收紧 = 新版本；
                             已批准的规格继续按它批准时的那一版解释，新提案按 CURRENT
    安全底线（SAFETY，不分版本） 读未来信息（future_* 字段、延迟 < 1）、不支持的算子（解析不了）、超出资源边界
                             （窗口 / 深度 / 表达式长度的硬上限）—— 任何版本都必须通过，否则不新增风险

规格里的 `validation`（`validation_record`）：政策版本、政策内容摘要、DSL 算子集摘要、安全底线、逐成分的结果。
它在规格里，所以绑定在 spec_hash 上；`check_frozen` 在运行时核对：有凭据、版本认识、政策内容与摘要一致、算子集没变、
按那一版重算结果仍通过且与记录一致、安全底线通过。任何一项不满足 → 列出原因（调用方据此不生成目标、不交易）。
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Dict, List, Optional

from .validator import (
    AlphaValidator,
    DepthValidator,
    LookAheadValidator,
    SparseFieldValidator,
    WindowValidator,
)

#: 版本 → 规则参数。**只许新增版本，不许改已有版本的内容**（改了摘要就对不上，已批准的规格会被拒绝）
POLICIES: Dict[str, dict] = {
    "v1": {"max_window": 252, "max_depth": 10, "sparse_min_window": 126},
}
CURRENT = "v1"

#: 安全底线：与政策版本无关
SAFETY = {"max_window": 1260, "max_depth": 20, "max_expr_chars": 2000}

logger = logging.getLogger(__name__)


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def policy_digest(version: str) -> str:
    return _digest({"version": version, **POLICIES[version]})


def operators_digest() -> str:
    """DSL 支持的算子集（时序 / 截面 / 分组 / 算术 / 一元 / 高级）的摘要：算子集变了，语义兼容性要重新确认。"""
    from . import typed_nodes as tn
    ops = {name: sorted(getattr(tn, name)) for name in
           ("_TS_OPS", "_CS_OPS", "_GROUP_OPS", "_BINOP_OPS", "_UNARY_OPS", "_ADVANCED_OPS")}
    return _digest(ops)


def validator_for(version: str) -> AlphaValidator:
    """按某一版政策配置的校验器。"""
    p = POLICIES[version]
    v = AlphaValidator()
    win, dep, sparse = WindowValidator(), DepthValidator(), SparseFieldValidator()
    win.MAX_WINDOW, dep.MAX_DEPTH, sparse.MIN_WINDOW = int(p["max_window"]), int(p["max_depth"]), int(p["sparse_min_window"])
    v._validators = [dep, win, LookAheadValidator(), sparse]                     # noqa: SLF001
    return v


def _parse(dsl: str):
    from .parser import Parser
    return Parser().parse(dsl)


def _max_window(node) -> int:
    from .typed_nodes import TimeSeriesNode
    w = node.window if isinstance(node, TimeSeriesNode) else 0
    return max([w] + [_max_window(c) for c in node.children()])


def safety_problems(dsl: Optional[str]) -> List[str]:
    """安全底线（不分版本）。空 = 通过。"""
    if dsl is None:
        return ["没有表达式"]
    if len(dsl) > SAFETY["max_expr_chars"]:
        return [f"表达式长度 {len(dsl)} 超过安全上限 {SAFETY['max_expr_chars']}"]
    try:
        node = _parse(dsl)
    except Exception as exc:                                 # 解析不了 = 不支持的算子 / 语法
        logger.info("[validation] 表达式解析失败（记为不通过）: %s", exc)
        return [f"解析失败（不支持的算子或语法）：{exc}"]
    out = list(LookAheadValidator().collect(node))
    if node.depth() > SAFETY["max_depth"]:
        out.append(f"嵌套深度 {node.depth()} 超过安全上限 {SAFETY['max_depth']}")
    w = _max_window(node)
    if w > SAFETY["max_window"]:
        out.append(f"窗口 {w} 超过安全上限 {SAFETY['max_window']}")
    return out


def _check_one(dsl: Optional[str], version: str) -> dict:
    from .validator import ValidationError
    errors = safety_problems(dsl)
    if not errors:
        try:
            validator_for(version).validate(_parse(dsl))
        except ValidationError as exc:
            logger.info("[validation] 表达式按政策 %s 不通过: %s", version, exc.errors)
            errors = list(exc.errors)
    return {"ok": not errors, "errors": errors}


def validation_record(factor_dsl: Dict[str, Optional[str]], version: Optional[str] = None) -> dict:
    """提案 / 批准时冻结进规格的校验凭据（缺省按调用那一刻的 CURRENT）。"""
    version = version or CURRENT
    results = {str(k): _check_one(v, version) for k, v in sorted(factor_dsl.items())}
    return {"policy_version": version, "policy_digest": policy_digest(version),
            "operators_digest": operators_digest(), "safety": dict(SAFETY), "resource_bounds": dict(POLICIES[version]),
            "results": results, "passed": bool(results) and all(r["ok"] for r in results.values())}


def check_frozen(spec: dict) -> List[str]:
    """已批准规格的运行时核对。空 = 可以按它生成目标。"""
    rec = spec.get("validation")
    if not isinstance(rec, dict):
        return ["规格里没有表达式校验凭据（本项之前批准的旧规格）—— 不追认，重新提案（或做一次明确的迁移核验）"]
    ver = rec.get("policy_version")
    if ver not in POLICIES:
        return [f"规格的校验政策版本 {ver!r} 不认识"]
    out = []
    if rec.get("policy_digest") != policy_digest(ver):
        out.append(f"校验政策 {ver} 的内容与批准时的摘要不一致（已登记的政策被改过）")
    if rec.get("operators_digest") != operators_digest():
        out.append("DSL 算子集与批准时不同 —— 语义兼容性要重新确认（重新提案）")
    if not rec.get("passed"):
        out.append("批准时的校验结论就是不通过")
    factors = spec.get("factors") or {}
    if set(rec.get("results") or {}) != {str(k) for k in factors}:
        out.append("校验结果的成分与规格成分不一致")
    for f, dsl in factors.items():
        now = _check_one(dsl, ver)
        if not now["ok"]:
            out.append(f"成分 {f} 按政策 {ver} 不通过：{'; '.join(now['errors'])}")
    return out
