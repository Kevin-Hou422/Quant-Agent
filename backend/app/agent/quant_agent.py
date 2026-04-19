"""
quant_agent.py — 对话式量化研究 Agent（Phase 3）

架构设计原则（最小化 LLM Token）：
  - LLM 仅执行 2 类任务：
      1. 自然语言假设 → Alpha DSL 翻译（~200 tokens/次）
      2. 过拟合检测失败时的 DSL 简化建议（~150 tokens/次，仅当触发）
  - 所有数值计算（Optuna / 回测 / 过拟合评分）由 Python 引擎执行（0 tokens）
  - 过拟合判断为纯 Python 布尔逻辑，不消耗 LLM

工作流：
  Workflow A（从零生成）:
    hypothesis → DSL → Optuna(IS) → Backtest(IS+OOS) →
    [过拟合?] → 最多 2 次 self-correction → Save

  Workflow B（优化现有）:
    user_dsl → Optuna(IS) → Backtest(IS+OOS) →
    [过拟合?] → 简化 DSL → 重新验证 → Save

LangChain 降级策略：
  - 有 OPENAI_API_KEY：使用 LangChain AgentExecutor + GPT-4o
  - 无 API Key：使用 FallbackOrchestrator（纯 Python，预设规则映射）
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量 & 配置
# ---------------------------------------------------------------------------

_MAX_CORRECTION_ROUNDS = 2          # 过拟合时最多自我修正次数
_OVERFIT_THRESHOLD     = 0.50       # OOS Sharpe 退化 > 50% 触发修正
_MIN_OOS_SHARPE        = 0.20       # 通过阈值：OOS Sharpe > 0.2
_DEFAULT_N_TICKERS     = 20
_DEFAULT_N_DAYS        = 252
_DEFAULT_OOS_RATIO     = 0.30
_DEFAULT_N_TRIALS      = 15         # Optuna trials（Agent 内部默认，比 API 少）

# 预设 DSL 模板（降级模式 / 无 LLM 时使用）
_FALLBACK_DSL_MAP: Dict[str, str] = {
    "volume":    "rank(ts_delta(log(volume), 5))",
    "momentum":  "rank(ts_mean(returns, 10))",
    "reversal":  "rank(-ts_delta(close, 1))",
    "volatility": "rank(-ts_std(returns, 20))",
    "price":     "rank(ts_delta(log(close), 5))",
    "default":   "rank(ts_delta(log(close), 5))",
}


# ---------------------------------------------------------------------------
# 模块级辅助：从文本中提取带平衡括号的 DSL 子串
# ---------------------------------------------------------------------------

def _extract_balanced(text: str, start: int) -> Optional[str]:
    """
    从 ``text[start:]`` 提取第一个带平衡括号的 DSL 表达式。

    处理可选前导 ``-``、嵌套括号（如 ``rank(ts_delta(log(close),5))``），
    以及操作符前后的空白字符。

    Returns
    -------
    完整 DSL 字符串（已 strip），若未找到左括号则返回 None。
    """
    # 找到第一个 "(" 的位置
    paren_idx = text.find("(", start)
    if paren_idx == -1:
        return None

    depth = 0
    for i in range(paren_idx, len(text)):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[start : i + 1].strip()

    return None   # 括号未闭合


# ---------------------------------------------------------------------------
# ConversationMemory — 轻量对话记忆（无 LangChain 依赖的独立版本）
# ---------------------------------------------------------------------------

@dataclass
class TurnRecord:
    """单轮对话记录。"""
    role:    str   # "user" | "assistant"
    content: str
    dsl:     Optional[str]              = None
    metrics: Optional[Dict[str, Any]]  = None


class ConversationMemory:
    """
    轻量对话缓冲记忆。

    - 最多保留 `max_turns` 轮（user + assistant 各算 1 轮）
    - 暴露 `last_dsl` / `last_metrics` 便于跨轮引用
    - 可导出为 LangChain ChatMessageHistory 格式（需要时）
    """

    def __init__(self, max_turns: int = 10) -> None:
        self._buffer:      Deque[TurnRecord] = deque(maxlen=max_turns * 2)
        self._last_dsl:    Optional[str]             = None
        self._last_metrics: Optional[Dict[str, Any]] = None

    def add_user(self, content: str) -> None:
        self._buffer.append(TurnRecord(role="user", content=content))

    def add_assistant(self, content: str,
                      dsl: Optional[str] = None,
                      metrics: Optional[Dict[str, Any]] = None) -> None:
        rec = TurnRecord(role="assistant", content=content,
                         dsl=dsl, metrics=metrics)
        self._buffer.append(rec)
        if dsl:
            self._last_dsl     = dsl
        if metrics:
            self._last_metrics = metrics

    @property
    def last_dsl(self) -> Optional[str]:
        return self._last_dsl

    @property
    def last_metrics(self) -> Optional[Dict[str, Any]]:
        return self._last_metrics

    def history_text(self, max_chars: int = 1500) -> str:
        """返回对话历史摘要（限制 Token 用量）。"""
        lines = []
        for rec in self._buffer:
            role_label = "User" if rec.role == "user" else "Assistant"
            # 助手消息仅保留首 200 字符，避免 Token 膨胀
            content = rec.content[:200] if rec.role == "assistant" else rec.content
            lines.append(f"{role_label}: {content}")
        text = "\n".join(lines)
        return text[-max_chars:]  # 尾部最新优先

    def to_langchain_messages(self) -> list:
        """转换为 LangChain HumanMessage/AIMessage 列表。"""
        try:
            from langchain_core.messages import HumanMessage, AIMessage
        except ImportError:
            return []
        msgs = []
        for rec in self._buffer:
            if rec.role == "user":
                msgs.append(HumanMessage(content=rec.content))
            else:
                msgs.append(AIMessage(content=rec.content))
        return msgs


# ---------------------------------------------------------------------------
# 工具函数（纯 Python，0 LLM Token）
# ---------------------------------------------------------------------------

def _make_synthetic_dataset(n_tickers: int = 20, n_days: int = 252,
                             seed: int = 42) -> Dict[str, pd.DataFrame]:
    """生成合成回测数据集（复用 router.py 同款，无外部依赖）。"""
    rng     = np.random.default_rng(seed)
    dates   = pd.bdate_range("2021-01-04", periods=n_days)
    tickers = [f"T{i:03d}" for i in range(n_tickers)]

    close = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.012, (n_days, n_tickers)), axis=0),
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


def _partition(dataset: Dict[str, pd.DataFrame],
               oos_ratio: float) -> Tuple[Dict, Dict]:
    """IS/OOS 物理分区，返回 (is_data, oos_data)。"""
    from app.core.data_engine.data_partitioner import DataPartitioner
    dates = next(iter(dataset.values())).index
    dp = DataPartitioner(
        start     = str(dates[0].date()),
        end       = str(dates[-1].date()),
        oos_ratio = oos_ratio,
    )
    part = dp.partition(dataset)
    return part.train(), part.test()


def _run_backtest_core(
    dsl:     str,
    cfg,                         # SimulationConfig
    is_data: Dict,
    oos_data: Optional[Dict],
) -> Dict[str, Any]:
    """
    执行 RealisticBacktester IS+OOS 回测，返回结构化指标字典。
    纯 Python，0 LLM Token。
    """
    from app.core.backtest_engine.realistic_backtester import RealisticBacktester

    bt     = RealisticBacktester(config=cfg)
    result = bt.run(dsl, is_data, oos_dataset=oos_data)

    is_r   = result.is_report
    oos_r  = result.oos_report

    def _f(v):
        return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None

    # 过拟合评分（纯 Python 逻辑）
    is_sharpe  = _f(is_r.sharpe_ratio)  or 0.0
    oos_sharpe = _f(oos_r.sharpe_ratio) if oos_r else None

    if oos_sharpe is not None and abs(is_sharpe) > 1e-9:
        degradation = (is_sharpe - oos_sharpe) / abs(is_sharpe)
        overfit_score = float(np.clip(degradation, 0.0, 1.0))
    else:
        overfit_score = 0.0

    is_overfit = overfit_score > _OVERFIT_THRESHOLD

    return {
        "is_sharpe":        is_sharpe,
        "oos_sharpe":       oos_sharpe,
        "is_return":        _f(is_r.annualized_return),
        "is_turnover":      _f(is_r.ann_turnover),
        "is_ic":            _f(is_r.mean_ic),
        "overfitting_score": overfit_score,
        "is_overfit":        is_overfit,
        "summary":           result.summary(),
    }


# ---------------------------------------------------------------------------
# 4 Agent Tools（可独立调用，也被 LangChain @tool 包装）
# ---------------------------------------------------------------------------

class QuantTools:
    """
    量化工具函数集合。

    每个方法对应一个 Agent Tool，参数和返回值均为 JSON 字符串，
    便于被 LangChain @tool 包装或直接测试调用。
    """

    def __init__(
        self,
        n_tickers:  int   = _DEFAULT_N_TICKERS,
        n_days:     int   = _DEFAULT_N_DAYS,
        oos_ratio:  float = _DEFAULT_OOS_RATIO,
        n_trials:   int   = _DEFAULT_N_TRIALS,
        seed:       int   = 42,
        llm: Any          = None,   # Optional LangChain LLM
    ) -> None:
        self._n_tickers = n_tickers
        self._n_days    = n_days
        self._oos_ratio = oos_ratio
        self._n_trials  = n_trials
        self._seed      = seed
        self._llm       = llm

        # 预生成并分区（每次 chat 复用同一份，避免重复生成）
        ds = _make_synthetic_dataset(n_tickers, n_days, seed)
        self._is_data, self._oos_data = _partition(ds, oos_ratio)

    # ------------------------------------------------------------------
    # Tool 1: 自然语言 → DSL
    # ------------------------------------------------------------------

    def tool_generate_alpha_dsl(self, hypothesis: str) -> str:
        """
        将市场假设翻译为 Alpha DSL。
        - 有 LLM：调用 LLM，~200 tokens
        - 无 LLM：关键词映射（0 tokens）
        返回 JSON: {"dsl": str, "explanation": str}
        """
        if self._llm is not None:
            return self._llm_generate_dsl(hypothesis)
        return self._fallback_generate_dsl(hypothesis)

    def _fallback_generate_dsl(self, hypothesis: str) -> str:
        """无 LLM 时基于关键词映射生成 DSL。"""
        h_lower = hypothesis.lower()
        for kw, dsl in _FALLBACK_DSL_MAP.items():
            if kw in h_lower:
                return json.dumps({
                    "dsl": dsl,
                    "explanation": f"[Fallback] Mapped '{kw}' keyword to DSL.",
                })
        return json.dumps({
            "dsl": _FALLBACK_DSL_MAP["default"],
            "explanation": "[Fallback] No keyword match, using default momentum DSL.",
        })

    def _llm_generate_dsl(self, hypothesis: str) -> str:
        """使用 LLM 生成 DSL（最小化 Token 的 few-shot prompt）。"""
        system = (
            "You are a DSL compiler for alpha strategies. "
            "Translate the user's market hypothesis into ONE Alpha DSL expression. "
            "Supported operators: rank, ts_mean, ts_std, ts_delta, ts_decay_linear, log, returns. "
            "Output ONLY valid JSON: {\"dsl\": \"<expression>\", \"explanation\": \"<one_sentence>\"}. "
            "No markdown, no extra text."
        )
        few_shot = (
            "Example: hypothesis='volume spike momentum' → "
            '{"dsl": "rank(ts_delta(log(volume), 5))", "explanation": "Rank of 5-day log-volume change."}'
        )
        prompt = f"{system}\n{few_shot}\n\nHypothesis: {hypothesis}"

        try:
            response = self._llm.invoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            # 提取 JSON（防止 LLM 返回 markdown 包装）
            import re
            m = re.search(r'\{.*\}', text, re.DOTALL)
            if m:
                parsed = json.loads(m.group())
                return json.dumps(parsed)
        except Exception as exc:
            logger.warning("LLM DSL 生成失败，降级到关键词映射: %s", exc)

        return self._fallback_generate_dsl(hypothesis)

    # ------------------------------------------------------------------
    # Tool 2: Optuna 超参优化（纯 Python，0 Token）
    # ------------------------------------------------------------------

    def tool_run_optuna(self, dsl: str, n_trials: int = 0) -> str:
        """
        对给定 DSL 在 IS 数据集上运行 Optuna 超参优化。
        OOS 数据全程物理隔离。
        返回 JSON: {"best_config": {...}, "best_fitness": float, "n_trials": int}
        """
        from app.core.ml_engine.alpha_optimizer import AlphaOptimizer, SearchSpace

        n = n_trials if n_trials > 0 else self._n_trials
        optimizer = AlphaOptimizer(
            dsl          = dsl,
            is_dataset   = self._is_data,   # 仅传 IS
            search_space = SearchSpace(
                delay_range     = (0, 3),
                decay_range     = (0, 8),
                portfolio_modes = ("long_short",),
            ),
            n_trials = n,
            seed     = self._seed,
        )
        try:
            best_cfg, summary = optimizer.optimize()
        except Exception as exc:
            logger.warning("Optuna 优化失败: %s", exc)
            # 降级：使用默认配置
            from app.core.alpha_engine.signal_processor import SimulationConfig
            best_cfg = SimulationConfig()
            return json.dumps({
                "best_config":  {"delay": 1, "decay_window": 0,
                                 "portfolio_mode": "long_short"},
                "best_fitness": -999.0,
                "n_trials":     0,
                "error":        str(exc),
            })

        return json.dumps({
            "best_config": {
                "delay":            best_cfg.delay,
                "decay_window":     best_cfg.decay_window,
                "truncation_min_q": best_cfg.truncation_min_q,
                "truncation_max_q": best_cfg.truncation_max_q,
                "portfolio_mode":   best_cfg.portfolio_mode,
            },
            "best_fitness": summary.best_value,
            "n_trials":     summary.n_trials,
        })

    # ------------------------------------------------------------------
    # Tool 3: IS+OOS 完整回测（纯 Python，0 Token）
    # ------------------------------------------------------------------

    def tool_run_backtest(self, dsl: str, config_json: str = "{}") -> str:
        """
        用给定 DSL 和可选配置运行完整 IS+OOS 双段回测。
        返回 JSON: {is_sharpe, oos_sharpe, overfitting_score, is_overfit, summary}
        """
        from app.core.alpha_engine.signal_processor import SimulationConfig

        cfg_dict = json.loads(config_json) if config_json.strip() else {}
        cfg = SimulationConfig(
            delay            = cfg_dict.get("delay", 1),
            decay_window     = cfg_dict.get("decay_window", 0),
            truncation_min_q = cfg_dict.get("truncation_min_q", 0.05),
            truncation_max_q = cfg_dict.get("truncation_max_q", 0.95),
            portfolio_mode   = cfg_dict.get("portfolio_mode", "long_short"),
        )
        try:
            metrics = _run_backtest_core(dsl, cfg, self._is_data, self._oos_data)
        except Exception as exc:
            logger.warning("回测失败: %s", exc)
            metrics = {
                "is_sharpe": None, "oos_sharpe": None,
                "overfitting_score": 0.0, "is_overfit": False,
                "summary": f"回测失败: {exc}",
            }
        return json.dumps(metrics, default=str)

    # ------------------------------------------------------------------
    # Tool 4: 保存到 AlphaStore（纯 Python，0 Token）
    # ------------------------------------------------------------------

    def tool_save_alpha(self, name: str, dsl: str,
                        metrics_json: str = "{}") -> str:
        """
        将 Alpha 策略持久化到 SQLite AlphaStore。
        返回 JSON: {"id": int, "status": "saved"}
        """
        from app.db.alpha_store import AlphaStore, AlphaResult

        metrics = json.loads(metrics_json) if metrics_json.strip() else {}

        result = AlphaResult(
            dsl          = dsl,
            hypothesis   = name,
            sharpe       = float(metrics.get("is_sharpe")     or 0.0),
            ann_return   = float(metrics.get("is_return")     or 0.0),
            ann_turnover = float(metrics.get("is_turnover")   or 0.0),
            ic_ir        = float(metrics.get("is_ic")         or 0.0),
        )
        try:
            store   = AlphaStore()
            alpha_id = store.save(result)
            return json.dumps({"id": alpha_id, "status": "saved", "dsl": dsl})
        except Exception as exc:
            logger.warning("AlphaStore.save 失败: %s", exc)
            return json.dumps({"id": -1, "status": "error", "detail": str(exc)})


# ---------------------------------------------------------------------------
# Anti-Overfitting Critic（纯 Python 逻辑，0 LLM Token）
# ---------------------------------------------------------------------------

class OverfitCritic:
    """
    过拟合检测器（无 LLM）。

    规则：
      PASS: oos_sharpe > MIN_OOS_SHARPE AND overfitting_score < OVERFIT_THRESHOLD
      FAIL: 否则
    """

    @staticmethod
    def check(metrics: Dict[str, Any]) -> Tuple[bool, str]:
        """
        返回 (passed, reason)。
        passed=True 表示策略通过反过拟合检验。
        """
        oos_sharpe     = metrics.get("oos_sharpe")     or 0.0
        overfit_score  = metrics.get("overfitting_score") or 0.0
        is_overfit_flag = metrics.get("is_overfit")    or False

        if is_overfit_flag or overfit_score > _OVERFIT_THRESHOLD:
            return False, (
                f"过拟合！IS vs OOS Sharpe 退化 {overfit_score*100:.1f}%。"
                "建议：缩短时间窗口，增加 decay 平滑，或降低 DSL 复杂度。"
            )
        if oos_sharpe is not None and oos_sharpe < _MIN_OOS_SHARPE:
            return False, (
                f"OOS Sharpe={oos_sharpe:.3f} < {_MIN_OOS_SHARPE}，OOS 表现不足。"
                "建议：尝试更强的截面排名信号或更长回望窗口。"
            )
        return True, "通过反过拟合检验。"


# ---------------------------------------------------------------------------
# FallbackOrchestrator — 无 LLM 时的直接编排器
# ---------------------------------------------------------------------------

class FallbackOrchestrator:
    """
    无 LLM 依赖的直接工作流编排器（用于测试 / 无 API Key 场景）。

    实现与 LangChain AgentExecutor 相同的两种工作流，
    通过简单规则判断意图并串联 QuantTools。
    """

    def __init__(self, tools: QuantTools) -> None:
        self._tools  = tools
        self._critic = OverfitCritic()

    def run_workflow_a(self, hypothesis: str) -> Tuple[str, Dict]:
        """Workflow A：从假设生成 DSL → 优化 → 回测 → 验证 → 保存。"""
        dsl, final_metrics = None, {}

        for attempt in range(_MAX_CORRECTION_ROUNDS + 1):
            # Step 1: 生成 DSL
            dsl_result = json.loads(
                self._tools.tool_generate_alpha_dsl(hypothesis)
            )
            dsl = dsl_result["dsl"]

            # Step 2: Optuna 优化（IS only）
            opt_result  = json.loads(self._tools.tool_run_optuna(dsl))
            best_config = json.dumps(opt_result.get("best_config", {}))

            # Step 3: 完整回测
            final_metrics = json.loads(
                self._tools.tool_run_backtest(dsl, best_config)
            )

            # Step 4: Anti-Overfitting Check（纯 Python）
            passed, reason = self._critic.check(final_metrics)
            if passed:
                break

            logger.info("Attempt %d 过拟合: %s", attempt + 1, reason)
            if attempt < _MAX_CORRECTION_ROUNDS:
                # 自我修正：在假设中追加"简化"指令
                hypothesis = (
                    f"Simplify and reduce complexity of: {dsl}. "
                    f"Reason: {reason}"
                )

        # Step 5: 保存
        self._tools.tool_save_alpha(
            name         = f"auto_{int(time.time())}",
            dsl          = dsl or "",
            metrics_json = json.dumps(final_metrics),
        )
        return dsl or "", final_metrics

    def run_workflow_b(self, user_dsl: str) -> Tuple[str, Dict]:
        """Workflow B：优化已有 DSL → 验证 → 保存。"""
        dsl = user_dsl

        for attempt in range(_MAX_CORRECTION_ROUNDS + 1):
            # Step 1: Optuna 优化
            opt_result  = json.loads(self._tools.tool_run_optuna(dsl))
            best_config = json.dumps(opt_result.get("best_config", {}))

            # Step 2: 完整回测
            final_metrics = json.loads(
                self._tools.tool_run_backtest(dsl, best_config)
            )

            # Step 3: 过拟合检查
            passed, reason = self._critic.check(final_metrics)
            if passed:
                break

            logger.info("Workflow B attempt %d 过拟合: %s", attempt + 1, reason)
            if attempt < _MAX_CORRECTION_ROUNDS:
                # 简化 DSL（降级：回退到 rank(close)）
                dsl = "rank(ts_delta(log(close), 3))"

        self._tools.tool_save_alpha(
            name         = f"optimized_{int(time.time())}",
            dsl          = dsl,
            metrics_json = json.dumps(final_metrics),
        )
        return dsl, final_metrics


# ---------------------------------------------------------------------------
# LangChainOrchestrator — 有 LLM 时的完整 AgentExecutor
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a professional quantitative researcher assistant.
You orchestrate specialized tools to generate, optimize, and validate alpha strategies.

WORKFLOW A (Generate from scratch):
1. Call tool_generate_alpha_dsl with the user's market hypothesis.
2. Call tool_run_optuna with the generated DSL to find best hyperparameters (IS only).
3. Call tool_run_backtest with the DSL and best config.
4. Check the returned overfitting_score and is_overfit flag.
   IF is_overfit=true: call tool_generate_alpha_dsl again with instruction to
   REDUCE COMPLEXITY. Repeat at most 2 times.
5. IF performance passes (oos_sharpe > 0.2 AND overfitting_score < 0.5):
   call tool_save_alpha to persist the result.

WORKFLOW B (Optimize existing DSL — user provides DSL directly):
1. Call tool_run_optuna directly on the provided DSL.
2. Call tool_run_backtest to validate.
3. IF overfitting detected: simplify DSL and repeat. Max 2 rounds.
4. Call tool_save_alpha.

ANTI-OVERFITTING RULES (Python enforces numbers; you enforce intent):
- PASS: oos_sharpe > 0.2 AND overfitting_score < 0.5
- FAIL: oos_sharpe drops > 50% vs is_sharpe
- When failing: always suggest SIMPLER DSL (shorter windows, add decay)

TOKEN EFFICIENCY (strict):
- Never re-explain tool results; reference numbers directly.
- Final reply must be under 100 words.
- Do not describe computation steps already completed by tools.

MEMORY:
- If user says "the last alpha" or "make it less sensitive to X",
  use the DSL from the most recent successful workflow.
"""


def _build_langchain_agent(llm, tools_obj: QuantTools):
    """
    构造 LangChain AgentExecutor（需要 langchain / langchain_openai 已安装）。
    """
    try:
        from langchain.tools import tool as lc_tool
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.messages import SystemMessage
    except ImportError as exc:
        raise ImportError(
            "需要安装 langchain 和 langchain-openai: pip install langchain langchain-openai"
        ) from exc

    # --- 包装 4 个工具 ---
    @lc_tool
    def tool_generate_alpha_dsl(hypothesis: str) -> str:
        """Translate a natural language market hypothesis into an Alpha DSL expression."""
        return tools_obj.tool_generate_alpha_dsl(hypothesis)

    @lc_tool
    def tool_run_optuna(dsl: str, n_trials: int = 15) -> str:
        """Run Optuna hyperparameter optimization on IS dataset. Returns best config and fitness."""
        return tools_obj.tool_run_optuna(dsl, n_trials)

    @lc_tool
    def tool_run_backtest(dsl: str, config_json: str = "{}") -> str:
        """Run full IS+OOS backtest. Returns IS/OOS Sharpe, overfitting_score, is_overfit flag."""
        return tools_obj.tool_run_backtest(dsl, config_json)

    @lc_tool
    def tool_save_alpha(name: str, dsl: str, metrics_json: str = "{}") -> str:
        """Save the alpha strategy to the AlphaStore SQLite ledger."""
        return tools_obj.tool_save_alpha(name, dsl, metrics_json)

    lc_tools = [tool_generate_alpha_dsl, tool_run_optuna,
                 tool_run_backtest, tool_save_alpha]

    # --- Prompt（最小 Token 结构）---
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent    = create_tool_calling_agent(llm, lc_tools, prompt)
    executor = AgentExecutor(
        agent       = agent,
        tools       = lc_tools,
        verbose     = False,
        max_iterations = 10,
        handle_parsing_errors = True,
    )
    return executor


# ---------------------------------------------------------------------------
# QuantAgent — 主入口（统一封装 LangChain 和 Fallback 两条路径）
# ---------------------------------------------------------------------------

class QuantAgent:
    """
    对话式量化 Agent 主类。

    - 有 OPENAI_API_KEY：使用 LangChain AgentExecutor + GPT-4o-mini
    - 无 API Key：使用 FallbackOrchestrator（规则映射，0 LLM Token）
    - 两条路径暴露相同的 .chat() 接口，调用方无需感知差异

    Parameters
    ----------
    n_tickers  : 合成数据集股票数量
    n_days     : 合成数据集交易日
    oos_ratio  : IS/OOS 分割比例
    n_trials   : Optuna 试验次数（Agent 内部）
    seed       : 随机种子
    api_key    : OpenAI API Key（None 则从环境变量读取）
    model      : OpenAI 模型名称
    """

    def __init__(
        self,
        n_tickers:  int   = _DEFAULT_N_TICKERS,
        n_days:     int   = _DEFAULT_N_DAYS,
        oos_ratio:  float = _DEFAULT_OOS_RATIO,
        n_trials:   int   = _DEFAULT_N_TRIALS,
        seed:       int   = 42,
        api_key:    Optional[str] = None,
        model:      str   = "gpt-4o-mini",
    ) -> None:
        self._memory = ConversationMemory(max_turns=10)
        self._llm    = None
        self._lc_executor = None

        # 尝试初始化 LLM
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if key:
            try:
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(
                    model       = model,
                    api_key     = key,
                    temperature = 0.1,
                    max_tokens  = 512,   # 严格限制单次 Token 上限
                )
                logger.info("LLM 初始化成功: %s", model)
            except Exception as exc:
                logger.warning("LLM 初始化失败，降级到 Fallback: %s", exc)

        # 工具实例
        self._tools = QuantTools(
            n_tickers = n_tickers,
            n_days    = n_days,
            oos_ratio = oos_ratio,
            n_trials  = n_trials,
            seed      = seed,
            llm       = self._llm,
        )

        # 编排器
        if self._llm is not None:
            try:
                self._lc_executor = _build_langchain_agent(self._llm, self._tools)
            except Exception as exc:
                logger.warning("LangChain AgentExecutor 构建失败，降级: %s", exc)

        self._fallback = FallbackOrchestrator(self._tools)

    # ------------------------------------------------------------------
    # 主对话接口
    # ------------------------------------------------------------------

    def chat(self, message: str) -> Dict[str, Any]:
        """
        处理一条用户消息，返回结构化回复。

        Returns
        -------
        {
            "reply":   str,           # 自然语言回复（≤100 字）
            "dsl":     Optional[str], # 本轮生成/优化的 DSL
            "metrics": Optional[dict] # IS/OOS 绩效指标
        }
        """
        self._memory.add_user(message)

        # 注入上下文：若用户引用"上一个"，替换为实际 DSL
        enriched = self._enrich_message(message)

        # 路由：判断意图
        intent, dsl_hint = self._detect_intent(enriched)

        try:
            if self._lc_executor is not None:
                result = self._lc_chat(enriched)
            else:
                result = self._fallback_chat(intent, enriched, dsl_hint)
        except Exception as exc:
            logger.exception("QuantAgent.chat 失败")
            result = {
                "reply":   f"处理失败: {exc}",
                "dsl":     None,
                "metrics": None,
            }

        self._memory.add_assistant(
            result.get("reply", ""),
            dsl     = result.get("dsl"),
            metrics = result.get("metrics"),
        )
        return result

    # ------------------------------------------------------------------
    # 内部：LangChain 路径
    # ------------------------------------------------------------------

    def _lc_chat(self, message: str) -> Dict[str, Any]:
        history = self._memory.to_langchain_messages()[:-1]  # 不含最新 user msg
        resp = self._lc_executor.invoke({
            "input":        message,
            "chat_history": history,
        })
        reply   = resp.get("output", "")
        dsl     = self._memory.last_dsl
        metrics = self._memory.last_metrics
        return {"reply": reply[:500], "dsl": dsl, "metrics": metrics}

    # ------------------------------------------------------------------
    # 内部：Fallback 路径
    # ------------------------------------------------------------------

    def _fallback_chat(self, intent: str, message: str,
                       dsl_hint: Optional[str]) -> Dict[str, Any]:
        if intent == "workflow_b" and dsl_hint:
            dsl, metrics = self._fallback.run_workflow_b(dsl_hint)
        else:
            dsl, metrics = self._fallback.run_workflow_a(message)

        oos_s   = metrics.get("oos_sharpe")
        is_s    = metrics.get("is_sharpe")
        overfit = metrics.get("overfitting_score", 0.0)

        reply = (
            f"DSL: {dsl} | "
            f"IS Sharpe={is_s:.3f} OOS Sharpe={oos_s:.3f} "
            f"过拟合评分={overfit:.2f} "
            f"{'⚠ 过拟合' if metrics.get('is_overfit') else '✓ 通过'}"
        ) if is_s is not None else f"生成 DSL: {dsl}（回测数据不足）"

        return {"reply": reply, "dsl": dsl, "metrics": metrics}

    # ------------------------------------------------------------------
    # 内部：消息增强 & 意图检测
    # ------------------------------------------------------------------

    def _enrich_message(self, message: str) -> str:
        """若消息引用上一个 Alpha，注入实际 DSL。"""
        lower = message.lower()
        if self._memory.last_dsl and any(
            kw in lower for kw in ["last", "previous", "上一个", "刚才", "that one"]
        ):
            return f"{message} [Context: last DSL was '{self._memory.last_dsl}']"
        return message

    def _detect_intent(self, message: str) -> Tuple[str, Optional[str]]:
        """
        快速意图检测（纯字符串规则，0 LLM Token）。

        Returns
        -------
        intent   : "workflow_a" | "workflow_b"
        dsl_hint : 从消息中提取的 DSL（Workflow B 时）
        """
        import re
        # Workflow B：消息中直接包含 DSL（含括号的算子表达式）
        # Fix: capture optional leading minus so `-rank(returns)` is not silently
        # stripped to `rank(returns)`, which would invert the intended strategy.
        # Also use a balanced-paren scan to handle nested expressions like
        # rank(ts_delta(log(close),5)).
        dsl_match = re.search(
            r'(-?\s*(?:rank|ts_mean|ts_std|ts_delta|ts_decay_linear|log|abs|sign|zscore|scale)'
            r'\s*\()',
            message,
        )
        if dsl_match:
            dsl_hint = _extract_balanced(message, dsl_match.start())
            if dsl_hint:
                return "workflow_b", dsl_hint

        # Workflow B：引用上一个 + "optimize/improve"
        lower = message.lower()
        if self._memory.last_dsl and any(
            kw in lower for kw in ["optimize", "improve", "refine", "优化", "改进"]
        ):
            return "workflow_b", self._memory.last_dsl

        return "workflow_a", None

    # ------------------------------------------------------------------
    # 便捷属性
    # ------------------------------------------------------------------

    @property
    def memory(self) -> ConversationMemory:
        return self._memory

    @property
    def tools(self) -> QuantTools:
        return self._tools
