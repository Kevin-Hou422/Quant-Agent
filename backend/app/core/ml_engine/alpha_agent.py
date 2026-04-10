"""
AlphaAgent — LangChain + GPT-4o 驱动的假设→DSL→回测→修正循环。

流程：
  1. 接收市场假设（自然语言）
  2. LLM 生成 5 条 DSL 候选
  3. 对每条 DSL：
     a. Validator 静态检查（失败→LLM 自动修正，最多 2 次）
     b. ProxyModel 剪枝（概率 > 0.7 → 跳过）
     c. 快速信号评估（IC_IR + 年化换手）
     d. 过拟合检测（IC_IR < 0.3 或年化换手 > 5）→ LLM Self-Correction → 重测（最多 3 次）
  4. 合格 Alpha 保存到 AlphaStore
  5. 输出 ReasoningLog JSON

依赖：
  OPENAI_API_KEY 环境变量
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..alpha_engine.parser import Parser, ParseError
from ..alpha_engine.validator import AlphaValidator, ValidationError
from .alpha_store import AlphaResult, AlphaStore
from .proxy_model import ProxyModel
from .reasoning_log import ReasoningLog

logger = logging.getLogger(__name__)

_parser    = Parser()
_validator = AlphaValidator()

# DSL 算子白名单（注入 System Prompt）
_OP_WHITELIST = (
    "close, open, high, low, volume, vwap, returns | "
    "ts_mean(x,w), ts_std(x,w), ts_delta(x,w), ts_delay(x,w), "
    "ts_max(x,w), ts_min(x,w), ts_rank(x,w), ts_decay_linear(x,w) | "
    "rank(x), zscore(x), scale(x) | "
    "log(x), abs(x), sqrt(x), sign(x) | "
    "+, -, *, / | "
    "signed_power(x,p), if_else(cond,x,y)"
)

_SYSTEM_PROMPT = f"""You are a quantitative researcher. Generate Alpha DSL formulas.
Allowed operators and fields: {_OP_WHITELIST}

Rules:
- Windows must be positive integers (2-60).
- No look-ahead bias (no future data).
- Max nesting depth: 8.
- rank/zscore/scale cannot be chained (e.g., rank(rank(x)) is invalid).

Output ONLY a JSON object with key "dsls" containing a list of exactly 5 DSL strings.
Example: {{"dsls": ["rank(ts_mean(close,20))", "zscore(ts_delta(close,5))"]}}"""


# ---------------------------------------------------------------------------
# LLM 接口（可选导入，无 API Key 时降级）
# ---------------------------------------------------------------------------

def _build_llm():
    """构建 LangChain ChatOpenAI 实例。OPENAI_API_KEY 未设置时返回 None。"""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=api_key)
    except ImportError:
        warnings.warn("langchain_openai not installed; AlphaAgent uses fallback DSLs.")
        return None


def _call_llm(llm, prompt: str) -> str:
    """调用 LLM，返回响应文本。"""
    try:
        from langchain_core.messages import HumanMessage
        resp = llm.invoke([HumanMessage(content=prompt)])
        return resp.content
    except Exception as e:
        logger.warning("LLM 调用失败: %s", e)
        return ""


def _parse_dsls_from_response(text: str) -> List[str]:
    """从 LLM 响应中提取 DSL 列表。"""
    try:
        data = json.loads(text)
        return data.get("dsls", [])
    except json.JSONDecodeError:
        pass
    import re
    matches = re.findall(r'"([^"]+\([^"]+\))"', text)
    return matches[:5]


# ---------------------------------------------------------------------------
# 快速信号评估（无需完整 BacktestEngine）
# ---------------------------------------------------------------------------

def _quick_eval(
    dsl: str,
    dataset: Dict[str, pd.DataFrame],
) -> Dict[str, float]:
    """
    计算 IC_IR 和年化换手的快速代理（不运行完整回测）。
    返回 {ic_ir, ann_turnover, sharpe}。
    """
    from ..alpha_engine.dsl_executor import Executor as DSLExecutor
    executor = DSLExecutor()
    try:
        signal_df = executor.run_expr(dsl, dataset)
    except Exception:
        return {"ic_ir": 0.0, "ann_turnover": 99.0, "sharpe": -1.0}

    close = dataset.get("close")
    if close is None:
        return {"ic_ir": 0.0, "ann_turnover": 99.0, "sharpe": -1.0}

    sig = signal_df.to_numpy(dtype=float)
    cls = close.to_numpy(dtype=float)
    fwd = np.full_like(cls, np.nan)
    fwd[:-1] = (cls[1:] - cls[:-1]) / np.where(cls[:-1] == 0, np.nan, cls[:-1])

    T = min(sig.shape[0], fwd.shape[0])
    ics = []
    for t in range(T - 1):
        s, r = sig[t], fwd[t]
        mask = ~(np.isnan(s) | np.isnan(r))
        if mask.sum() < 5:
            continue
        from scipy.stats import spearmanr
        rho, _ = spearmanr(s[mask], r[mask])
        if not np.isnan(rho):
            ics.append(rho)

    if not ics:
        return {"ic_ir": 0.0, "ann_turnover": 99.0, "sharpe": -1.0}

    ic_arr = np.array(ics)
    ic_ir  = float(np.mean(ic_arr) / (np.std(ic_arr) + 1e-9))

    ranks = pd.DataFrame(sig).rank(axis=1, pct=True).to_numpy()
    turn  = float(np.nanmean(np.abs(np.diff(ranks, axis=0)))) * 252

    return {"ic_ir": ic_ir, "ann_turnover": turn, "sharpe": ic_ir}


# ---------------------------------------------------------------------------
# AlphaAgent
# ---------------------------------------------------------------------------

class AlphaAgent:
    """
    假设驱动的 Alpha 发现 Agent。

    Parameters
    ----------
    store     : AlphaStore 实例（None 则不持久化）
    proxy     : ProxyModel 实例（None 则不剪枝）
    max_refine: 过拟合检测后最大修正次数
    """

    def __init__(
        self,
        store:      Optional[AlphaStore]  = None,
        proxy:      Optional[ProxyModel]  = None,
        max_refine: int = 3,
    ) -> None:
        self._llm        = _build_llm()
        self._store      = store or AlphaStore()
        self._proxy      = proxy or ProxyModel()
        self._max_refine = max_refine

    def run(
        self,
        hypothesis: str,
        dataset:    Dict[str, pd.DataFrame],
    ) -> ReasoningLog:
        """执行一轮假设→DSL→评估→修正的完整循环。"""
        log = ReasoningLog(hypothesis=hypothesis, initial_dsls=[])

        dsls = self._generate_dsls(hypothesis)
        log.initial_dsls = dsls
        logger.info("Agent 生成 %d 条 DSL 候选", len(dsls))

        passed_any = False
        for dsl in dsls:
            result_dsl, metrics = self._evaluate_and_refine(
                dsl, hypothesis, dataset, log
            )
            if result_dsl and metrics.get("ic_ir", 0) >= 0.3:
                ar = AlphaResult(
                    dsl          = result_dsl,
                    hypothesis   = hypothesis,
                    sharpe       = metrics.get("sharpe", 0.0),
                    ic_ir        = metrics.get("ic_ir", 0.0),
                    ann_turnover = metrics.get("ann_turnover", 0.0),
                    reasoning    = log.to_json(),
                    status       = "active",
                )
                self._store.save(ar)
                log.final_dsl     = result_dsl
                log.final_metrics = metrics
                passed_any = True
                logger.info("Alpha 通过: %s | IC-IR=%.4f", result_dsl, metrics["ic_ir"])
                break

        if not passed_any:
            logger.info("本轮无 Alpha 通过筛选，hypothesis='%s'", hypothesis)

        return log

    def _generate_dsls(self, hypothesis: str) -> List[str]:
        """调用 LLM 生成 DSL 候选；若 LLM 不可用则返回默认列表。"""
        if self._llm is None:
            return [
                "rank(ts_mean(close,20))",
                "zscore(ts_delta(close,5))",
                "rank(ts_std(close,10))",
                "zscore(ts_mean(volume,20))",
                "rank(ts_delta(log(close),5))",
            ]
        prompt = (
            f"Market hypothesis: {hypothesis}\n\n"
            "Generate 5 alpha DSL formulas that capture this hypothesis.\n"
            "Output JSON only."
        )
        full_prompt = _SYSTEM_PROMPT + "\n\n" + prompt
        resp  = _call_llm(self._llm, full_prompt)
        dsls  = _parse_dsls_from_response(resp)
        return dsls if dsls else [
            "rank(ts_delta(close,5))",
            "zscore(ts_mean(returns,10))",
        ]

    def _validate_and_fix(self, dsl: str) -> Optional[str]:
        """静态验证 DSL；若失败最多调用 LLM 修正 2 次。返回合法 DSL 或 None。"""
        for attempt in range(3):
            try:
                node = _parser.parse(dsl)
                _validator.validate(node)
                return dsl
            except (ParseError, ValidationError) as e:
                if attempt >= 2 or self._llm is None:
                    return None
                prompt = (
                    f"Fix this DSL formula: `{dsl}`\n"
                    f"Error: {e}\n"
                    "Output a single corrected DSL string in JSON: {{\"dsl\": \"...\"}}"
                )
                resp = _call_llm(self._llm, prompt)
                try:
                    data = json.loads(resp)
                    dsl  = data.get("dsl", dsl)
                except Exception:
                    return None
        return None

    def _evaluate_and_refine(
        self,
        dsl:        str,
        hypothesis: str,
        dataset:    Dict[str, pd.DataFrame],
        log:        ReasoningLog,
    ) -> Tuple[Optional[str], Dict[str, float]]:
        """验证 + 剪枝 + 评估 + 最多 max_refine 次修正。"""
        valid_dsl = self._validate_and_fix(dsl)
        if valid_dsl is None:
            return None, {}

        try:
            node = _parser.parse(valid_dsl)
            if self._proxy.should_prune(node):
                logger.debug("ProxyModel 剪枝: %s", valid_dsl)
                return None, {}
        except Exception:
            return None, {}

        metrics = _quick_eval(valid_dsl, dataset)

        for refine in range(self._max_refine):
            ic_ir    = metrics.get("ic_ir", 0.0)
            turnover = metrics.get("ann_turnover", 0.0)

            if ic_ir >= 0.3 and turnover <= 5.0:
                try:
                    self._proxy.update(_parser.parse(valid_dsl), failed=False)
                except Exception:
                    pass
                return valid_dsl, metrics

            reason = (
                f"IC_IR={ic_ir:.4f} (<0.3)" if ic_ir < 0.3
                else f"AnnTurnover={turnover:.2f} (>5)"
            )
            if self._llm is None:
                break

            prompt = (
                f"This alpha DSL underperforms: `{valid_dsl}`\n"
                f"Problem: {reason}\n"
                f"Hypothesis: {hypothesis}\n"
                "Suggest an improved DSL. Output JSON: {\"dsl\": \"...\", \"reason\": \"...\"}"
            )
            resp = _call_llm(self._llm, _SYSTEM_PROMPT + "\n\n" + prompt)
            try:
                data     = json.loads(resp)
                new_dsl  = data.get("dsl", valid_dsl)
                change_r = data.get("reason", reason)
            except Exception:
                break

            fixed = self._validate_and_fix(new_dsl)
            if fixed is None:
                break

            new_metrics = _quick_eval(fixed, dataset)
            log.add_change(
                old_dsl = valid_dsl,
                new_dsl = fixed,
                reason  = change_r,
                metrics = new_metrics,
            )

            valid_dsl = fixed
            metrics   = new_metrics

        try:
            self._proxy.update(_parser.parse(valid_dsl), failed=(metrics.get("ic_ir", 0) < 0.3))
        except Exception:
            pass

        return valid_dsl, metrics
