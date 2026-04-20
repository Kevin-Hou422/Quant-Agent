"""
_critic.py — OverfitCritic: anti-overfitting gate (pure Python, 0 LLM tokens).

Called after every backtest.  Returns (passed: bool, reason: str) so the
orchestrator can decide whether to mutate the DSL or save it.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from app.agent._constants import _MIN_OOS_SHARPE, _OVERFIT_THRESHOLD


class OverfitCritic:
    """
    Stateless validator that checks two overfitting signals:

    1. is_overfit flag / overfitting_score  > _OVERFIT_THRESHOLD (0.50)
       → IS→OOS Sharpe degradation exceeded 50 %
    2. oos_sharpe < _MIN_OOS_SHARPE (0.20)
       → absolute out-of-sample performance too low
    """

    @staticmethod
    def check(metrics: Dict[str, Any]) -> Tuple[bool, str]:
        oos_sharpe    = metrics.get("oos_sharpe")        or 0.0
        overfit_score = metrics.get("overfitting_score") or 0.0
        is_overfit    = metrics.get("is_overfit")        or False

        if is_overfit or overfit_score > _OVERFIT_THRESHOLD:
            return False, (
                f"过拟合！IS vs OOS Sharpe 退化 {overfit_score * 100:.1f}%。"
                "建议进行结构性变异（tool_mutate_ast）而非仅调整参数。"
            )
        if oos_sharpe is not None and oos_sharpe < _MIN_OOS_SHARPE:
            return False, (
                f"OOS Sharpe={oos_sharpe:.3f} < {_MIN_OOS_SHARPE}，OOS 表现不足。"
            )
        return True, "通过反过拟合检验。"
