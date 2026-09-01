"""
promotion_gate.py — 分级晋级门 + 阈值配置化 + 实验模式（Phase TR.4）

实现「统一门控政策」的第 4/5 步（严门主体在策略级验证=第 2 步，见 PM.S1）：
    **第 4 步 进 PAPER**：**较松/分级** —— paper 的目的是**收集前向证据**，太严就永远拿不到证据。
                          实验模式（默认开）允许策略门未过也进 paper 观察，但会**如实标注等级**。
    **第 5 步 → ACTIVE（逼近真钱）**：**最严** —— 要求 ≥N 交易日**前向** realized IC 均值>0 且 **t>阈值**。

阈值全部来自配置（`tr_*`），**不写死**。

⚠️ 诚实的限制（已知缺口，待 Phase 11 修）
    目前 `alpha_ic_history` **没有区分"历史回放"与"真前向"**（首次运行会把整段历史逐日回放记进去）。
    因此本门的"前向天数"暂时是**全部已记录 IC 的天数**，**不是纯前向**。在 Phase 11 做完
    回放/前向分离前，`→ACTIVE` 门只应作**参考**（默认 `tr_enforce_active_gate=False` 不阻断），
    否则会把回放段当成前向证据——那正是这套门要防的事。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class PromotionThresholds:
    min_forward_days: int   = 60      # →ACTIVE 需要的前向观测天数
    min_ic_tstat:     float = 2.0     # →ACTIVE 的 realized IC t 门槛
    min_ic_mean:      float = 0.0     # →ACTIVE 的 realized IC 均值下限
    paper_min_sharpe: float = -99.0   # 进 PAPER 的宽松下限（实验模式下基本不卡）
    experiment_mode:  bool  = True    # 实验模式：策略门未过也允许进 paper（标注等级）

    @classmethod
    def from_settings(cls) -> "PromotionThresholds":
        try:
            from app.config import settings
            return cls(
                min_forward_days=int(getattr(settings, "tr_min_forward_days", 60)),
                min_ic_tstat=float(getattr(settings, "tr_min_ic_tstat", 2.0)),
                min_ic_mean=float(getattr(settings, "tr_min_ic_mean", 0.0)),
                paper_min_sharpe=float(getattr(settings, "tr_paper_min_sharpe", -99.0)),
                experiment_mode=bool(getattr(settings, "tr_experiment_mode", True)),
            )
        except Exception:
            return cls()


# ---------------------------------------------------------------------------
# 第 4 步：进 PAPER（较松 / 分级）
# ---------------------------------------------------------------------------

def grade_paper_entry(strategy_verdict: dict,
                      th: PromotionThresholds | None = None) -> Tuple[bool, dict]:
    """
    进 PAPER 的**分级**判定。返回 (allowed, detail)，detail.grade ∈ {A, B, C}：
        A = 策略门全过（最强证据）
        B = 策略门未过但核心指标不差（实验观察）
        C = 明显不合格（仅实验模式放行，并标注）
    实验模式下 B/C 也放行——目的是**收集前向证据**，但等级如实标注，不粉饰。
    """
    th = th or PromotionThresholds.from_settings()
    v = strategy_verdict or {}
    passed = bool(v.get("passed"))
    sharpe = float(v.get("sharpe") or 0.0)

    if passed:
        grade = "A"
    elif sharpe > 0:
        grade = "B"
    else:
        grade = "C"

    allowed = True if th.experiment_mode else (grade == "A" and sharpe >= th.paper_min_sharpe)
    return allowed, {
        "step": "paper_entry", "grade": grade, "allowed": allowed,
        "gate_passed": passed, "sharpe": round(sharpe, 4),
        "experiment_mode": th.experiment_mode,
        "note": "实验模式：未过严门也放行以收集前向证据（等级已标注）" if th.experiment_mode
                else "非实验模式：仅 A 级可进 paper",
    }


# ---------------------------------------------------------------------------
# 第 5 步：→ ACTIVE（逼近真钱，最严）
# ---------------------------------------------------------------------------

def check_active_promotion(ic_values: Sequence[float],
                           th: PromotionThresholds | None = None) -> Tuple[bool, dict]:
    """
    →ACTIVE 的**最严**判定：需 ≥`min_forward_days` 个观测，realized IC 均值 > `min_ic_mean`
    且 t = mean/std·√n > `min_ic_tstat`。返回 (passed, detail)。

    注：`ic_values` 当前包含回放段（见模块顶部限制说明），Phase 11 分离后才是纯前向。
    """
    th = th or PromotionThresholds.from_settings()
    x = np.asarray([v for v in ic_values if v is not None and np.isfinite(v)], dtype=float)
    n = int(x.size)
    reasons: List[str] = []
    detail = {"step": "active_promotion", "n_days": n,
              "min_forward_days": th.min_forward_days}

    if n < th.min_forward_days:
        reasons.append(f"前向观测不足（{n} < {th.min_forward_days} 交易日）")
        detail.update({"passed": False, "reasons": reasons})
        return False, detail

    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    t = (mu / sd) * np.sqrt(n) if sd > 1e-12 else 0.0
    detail.update({"ic_mean": round(mu, 6), "ic_tstat": round(float(t), 3),
                   "min_ic_tstat": th.min_ic_tstat})

    if mu <= th.min_ic_mean:
        reasons.append(f"realized IC 均值 {mu:.5f} ≤ {th.min_ic_mean}")
    if t <= th.min_ic_tstat:
        reasons.append(f"realized IC t={t:.2f} ≤ {th.min_ic_tstat}")

    passed = len(reasons) == 0
    detail.update({"passed": passed, "reasons": reasons})
    return passed, detail
