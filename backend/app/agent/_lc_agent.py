"""
_lc_agent.py — _build_langchain_agent.

把 QuantTools 接到 LangChain 的 `create_agent`（langchain 1.x），并用
`_SessionScopedAgent` 适配器接上本项目自己的 ChatStore 做按会话持久化记忆。

【缺陷 D-3，2026-09-20 迁移】原实现用 `AgentExecutor` +
`create_tool_calling_agent` + `RunnableWithMessageHistory`，这三个符号在
langchain 1.x 都已移出 `langchain.agents`（搬进了未安装的 `langchain-classic`）。
而 `requirements.txt` 写的是 `langchain>=0.2` **没有上界**、`requirements.lock`
锁的是 `1.4.0` —— 也就是说**按声明的依赖装出来的环境，这条链路必然构建失败**，
然后被 `_agent.py` 的 `except Exception` 接住、只打一条 warning 就静默降级到
FallbackOrchestrator。对外表现：`/api/chat` 照常返回，LLM 研究链路整条死掉。
（交易回路本来就不含 LLM，所以不影响下单；影响的是因子发现。）

迁移中**两处语义不是自动等价的**，已逐个实测并显式补回：

  · 旧 `AgentExecutor(handle_parsing_errors=True)` 把工具异常转成观测喂回模型，
    模型可以据此改口；`create_agent` 默认让异常**直接冒出 invoke**，整轮对话挂掉。
    → 用 `wrap_tool_call` 中间件补回（`_TOOL_ERROR_PREFIX`）。

  · 旧 `max_iterations=15` 数的是 agent 步，到顶**停下返回**；新的
    `recursion_limit` 数的是 LangGraph super-step，到顶**抛 GraphRecursionError**。
    实测换算见 `RECURSION_LIMIT` 的注释（装了中间件时是 `4L+4`，不是没装时的
    `2L+2` —— 照搬后者会让护栏比收口更早触发）。但那只是防护网：
    真正对应 `max_iterations` 的是 `ModelCallLimitMiddleware(run_limit=...)`，
    它按**单次调用**计模型调用数且 `exit_behavior="end"` 时优雅收尾。
    注意必须用 `run_limit` 而不是 `thread_limit` —— 后者跨轮累积，
    会让同一会话聊到后面被慢性截断。

**没有引入 langgraph 的 checkpointer**：本项目已有 ChatStore 作为会话历史的
唯一真相（`_chat_history.SQLAlchemyChatMessageHistory`）。再挂一个 checkpointer
等于同一个概念两份实现，迟早对不上账 —— 这一轮修的 A-2/A-3、C-1/B-3 都是这种形态。

Tool ordering (weaknesses 1-7 compliant):
  tool_generate_alpha_dsl  — seed DSL from hypothesis
  tool_interpret_factor    — DSL → factor_family + design diagnosis  [NEW Tool 7]
  tool_run_gp_optimization — PRIMARY optimizer; accepts factor_family + dataset_name
  tool_run_backtest        — IS+OOS validation
  tool_mutate_ast          — single AST mutation; accepts mutation_target for direct dispatch
  tool_run_optuna          — SECONDARY (parameter fine-tuning only)
  tool_save_alpha          — persist to AlphaStore

Key parameter links (weakness-7 fix):
  tool_interpret_factor(seed_dsl) → {factor_family, ...}
  tool_run_gp_optimization(seed_dsl, factor_family=<above>)
    → GP mutation weights biased toward financially appropriate operators

  tool_run_backtest → {overfitting_score, is_turnover, ...}
  tool_mutate_ast(dsl, reason, mutation_target=<from CriticResult>)
    → targeted AST correction instead of random mutation
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from app.agent._chat_history import _make_history_factory
from app.agent._prompts import _SYSTEM_PROMPT
from app.agent._tools import QuantTools

logger = logging.getLogger(__name__)

#: 单次对话允许的模型调用次数上限。等价于旧 `AgentExecutor(max_iterations=15)`。
MAX_MODEL_CALLS_PER_TURN = 15

#: LangGraph 的兜底护栏，**必须宽到让 ModelCallLimitMiddleware 先收口**。
#: 它一旦先触发，抛的就是 GraphRecursionError —— 正是迁移要避免的那种硬失败。
#:
#: 实测（`ModelCallLimitMiddleware` **在图里**时，L = MAX_MODEL_CALLS_PER_TURN）：
#:   · 被 run_limit 截停          需要 4L + 2   （L=15 → 62）
#:   · 模型自己在第 L 轮讲完       需要 4L + 4   （L=15 → 64）
#: 取两者上界 4L + 4。
#:
#: **不要用没装中间件时量出来的 2L+2。** 中间件给每轮加了一个节点，
#: 同一个 L 的开销直接翻倍；照搬那个数会让护栏比收口路径更早触发
#: （第一版就是这么写的，被 test_the_model_call_limit_ends_gracefully 当场抓住）。
RECURSION_LIMIT = 4 * MAX_MODEL_CALLS_PER_TURN + 4

#: 工具异常被转成观测时的前缀，模型据此知道这一步失败了。
_TOOL_ERROR_PREFIX = "TOOL_ERROR"


class LangChainIncompatibleError(ImportError):
    """
    已安装的 langchain 提供不了本模块需要的符号 —— **大版本不兼容**，不是"没装"。

    单独立一个类型，是为了让 `_agent.py` 能把「依赖不兼容」与「没配 API key」、
    「LLM 初始化失败」区分开并如实上报，而不是都归进一条 warning。
    按字符串匹配报错文案来区分是不可靠的（文案一改就失效）。
    """


def _build_tools(tools_obj: QuantTools) -> list:
    """
    把 QuantTools 的方法包成 LangChain 工具，**按给 LLM 看的顺序**返回。

    单独抽出来不是为了好看：这一层全是**接线**，而接线错了全部是静默的 ——
    少注册一个工具，LLM 永远不会调用它，对外只表现为「agent 好像不太会用 GP」；
    参数忘了透传（比如 `factor_family`），GP 退回随机权重，照样返回一条 DSL。
    抽成函数之后，接线可以用**真的** `@lc_tool` 直接验，
    不必为了测它而把整个 langchain 换成替身。

    （`tests/unit/agent/test_lc_agent_wiring.py` 原先正是整套 mock 掉 langchain 的，
    理由写在它的 docstring 里：缺陷 D-3 让真实 import 必然失败。D-3 修好之后
    那个理由不成立了，而「mock 掉构建器再宣称兼容」恰恰是 D-3 潜伏这么久的原因。）
    """
    from langchain.tools import tool as lc_tool

    # ── Tool 1: hypothesis → seed DSL ─────────────────────────────────

    @lc_tool
    def tool_generate_alpha_dsl(hypothesis: str) -> str:
        """Translate a natural language market hypothesis into a seed Alpha DSL expression.
        CRITICAL: if the hypothesis mentions a specific data field (vwap, volume, etc.),
        the generated DSL MUST include that field. This seed is then evolved by GP."""
        return tools_obj.tool_generate_alpha_dsl(hypothesis)

    # ── Tool 2: GP structural search (PRIMARY optimizer) ──────────────

    @lc_tool
    def tool_run_gp_optimization(
        seed_dsl:        str = "",
        n_generations:   int = 4,
        pop_size:        int = 12,
        n_optuna_trials: int = 8,
        factor_family:   str = "",
        dataset_name:    str = "",
        dataset_start:   str = "2021-01-01",
        dataset_end:     str = "2024-01-01",
    ) -> str:
        """
        PRIMARY OPTIMIZER. Run GP-driven alpha evolution.

        1. Initialises population with seed_dsl + random AST variants.
        2. Evolves for n_generations using AST mutation + subtree crossover + selection.
        3. Multi-objective fitness: sharpe_oos - 0.2*turnover - 0.3*overfit_penalty.
        4. Diversity filter: rejects alphas with signal correlation > 0.9.
        5. After evolution, Optuna fine-tunes ONLY the best structure's parameters.

        factor_family: pass the value from tool_interpret_factor to bias GP mutations
          toward financially appropriate operators for this factor type.
          (e.g. "momentum", "reversion", "volatility", "liquidity", "composite")
        dataset_name: optional real market dataset. Valid names come from the
                      dataset registry (e.g. "us_tech_large", "us_broad_large").
                      An unknown name is REJECTED, not silently downgraded.
          When omitted, uses the session-default dataset.

        Returns: best_dsl, metrics, generations_run, pool_top5, evolution_log.
        Use this as the MAIN optimizer — NOT tool_run_optuna standalone.
        """
        return tools_obj.tool_run_gp_optimization(
            seed_dsl        = seed_dsl,
            n_generations   = n_generations,
            pop_size        = pop_size,
            n_optuna_trials = n_optuna_trials,
            factor_family   = factor_family,
            dataset_name    = dataset_name,
            dataset_start   = dataset_start,
            dataset_end     = dataset_end,
        )

    # ── Tool 3: IS+OOS validation backtest ────────────────────────────

    @lc_tool
    def tool_run_backtest(
        dsl:          str  = "",
        config_json:  str  = "{}",
        use_test_set: bool = False,
    ) -> str:
        """Run full IS+OOS validation backtest.
        Returns IS/OOS Sharpe, overfitting_score, is_overfit, is_true_holdout.

        use_test_set=True  : OOS = true held-out Test set (GP never saw this data).
                             Use ONCE for the final selected alpha after GP completes.
        use_test_set=False : OOS = Validate set (used during GP fitness evaluation).
        """
        return tools_obj.tool_run_backtest(dsl, config_json, use_test_set)

    # ── Tool 4: Single AST structural mutation ────────────────────────

    @lc_tool
    def tool_mutate_ast(
        current_dsl:     str,
        overfit_reason:  str = "",
        mutation_target: str = "",
    ) -> str:
        """
        Apply ONE real AST-level structural mutation (NOT string templates).
        Uses GP engine operations: point_mutation, hoist_mutation, param_mutation.
        Adaptive weights based on overfit_reason (turnover / sharpe / overfitting).
        Call when GP result still shows overfitting after tool_run_gp_optimization.

        mutation_target: directly dispatch to a specific GP operator. Accepted values:
          replace_subtree, hoist, add_ts_smoothing, add_condition, wrap_rank, point
          (pass the recommended_mutation from tool_interpret_factor diagnosis).
        """
        return tools_obj.tool_mutate_ast(current_dsl, overfit_reason, mutation_target)

    # ── Tool 5: Optuna parameter fine-tuning (SECONDARY) ──────────────

    @lc_tool
    def tool_run_optuna(dsl: str, n_trials: int = 10) -> str:
        """SECONDARY optimizer — fine-tunes execution parameters (delay, decay, truncation)
        for a FIXED DSL structure. OOS is never seen during optimization.
        NOTE: GP already calls this internally. Only call manually if GP was skipped."""
        return tools_obj.tool_run_optuna(dsl, n_trials)

    # ── Tool 6: Persist to AlphaStore ─────────────────────────────────

    @lc_tool
    def tool_save_alpha(name: str, dsl: str, metrics_json: str = "{}") -> str:
        """Save the validated alpha strategy to the AlphaStore SQLite ledger."""
        return tools_obj.tool_save_alpha(name, dsl, metrics_json)

    # ── Tool 7: Financial interpretation + diagnosis ───────────────────

    @lc_tool
    def tool_interpret_factor(dsl: str, metrics_json: str = "{}") -> str:
        """
        Interpret a DSL alpha expression in financial terms and diagnose weaknesses.

        ALWAYS call this BEFORE tool_run_gp_optimization.
        The returned factor_family MUST be passed to tool_run_gp_optimization so that
        GP mutation weights are biased toward financially appropriate operators.

        Returns JSON with:
          factor_family   : use as factor_family param in tool_run_gp_optimization
          description     : what the factor measures in plain language
          design_issues   : list of detected design flaws (missing normalization, etc.)
          design_suggestions : DSL-level fix patches for each issue
          diagnosis       : (if metrics_json provided) metric-driven financial diagnosis
            recommended_mutation : pass as mutation_target to tool_mutate_ast
        """
        return tools_obj.tool_interpret_factor(dsl, metrics_json)

    lc_tools = [
        tool_generate_alpha_dsl,
        tool_interpret_factor,
        tool_run_gp_optimization,
        tool_run_backtest,
        tool_mutate_ast,
        tool_run_optuna,
        tool_save_alpha,
    ]
    return lc_tools


def _build_langchain_agent(
    llm:        Any,
    tools_obj:  QuantTools,
    chat_store: Optional[Any] = None,
) -> Any:
    """
    用 `create_agent` 组装 agent（7 个工具），并用 `_SessionScopedAgent`
    接上 ChatStore 做按会话持久化记忆。

    工具本身由 `_build_tools` 构造 —— 接线与组装分开，接线才测得动。
    """
    try:
        import langchain
        from langchain.agents import create_agent
        from langchain.agents.middleware import (
            ModelCallLimitMiddleware, wrap_tool_call)
        from langchain.tools import tool as lc_tool   # noqa: F401 —— 兼容性探针
    except ImportError as exc:
        # 文案必须指向**版本**，不能说"需要安装 langchain"。
        # 原文案正是这么写的，而 langchain 明明装着 —— 照着做只会再装一遍
        # 同样的版本，真正的原因（大版本 API 搬家）一次都不会被看见。
        installed = getattr(locals().get("langchain", None), "__version__", "未知")
        raise LangChainIncompatibleError(
            f"已安装的 langchain（{installed}）提供不了 create_agent / middleware："
            f"{exc}。这是**大版本不兼容**，不是没装 —— 重装同一版本无效。"
            f"本模块需要 langchain>=1.0（`create_agent` API）；"
            f"若环境被降到 0.x，请按 requirements.lock 对齐依赖。"
        ) from exc

    lc_tools = _build_tools(tools_obj)

    # ── 中间件：把两处**不自动等价**的旧语义显式补回来 ────────────────

    @wrap_tool_call
    def _tool_errors_become_observations(request, handler):
        """
        工具抛异常 → 转成 ToolMessage 观测，模型下一轮能看见并改口。

        等价于旧 `AgentExecutor(handle_parsing_errors=True)`。
        `create_agent` 默认让异常冒出 `invoke`，一个工具出错就把整轮对话打断 ——
        而这些工具里跑的是 GP 进化、回测、Optuna，失败是常态而非例外。
        """
        from langchain_core.messages import ToolMessage
        try:
            return handler(request)
        except Exception as exc:                      # noqa: BLE001 - 故意兜全部
            logger.warning("工具 %s 抛出 %s: %s",
                           request.tool_call.get("name"), type(exc).__name__, exc)
            return ToolMessage(
                content=f"{_TOOL_ERROR_PREFIX}: {type(exc).__name__}: {exc}",
                tool_call_id=request.tool_call["id"],
            )

    middleware = [
        ModelCallLimitMiddleware(
            run_limit     = MAX_MODEL_CALLS_PER_TURN,
            exit_behavior = "end",        # 到顶优雅收尾，不抛异常
        ),
        _tool_errors_become_observations,
    ]

    agent = create_agent(
        llm,
        lc_tools,
        system_prompt = _SYSTEM_PROMPT,
        middleware    = middleware,
    )
    logger.info("create_agent 构建成功（%d 个工具，单轮模型调用上限 %d）",
                len(lc_tools), MAX_MODEL_CALLS_PER_TURN)

    return _SessionScopedAgent(agent, chat_store)


class _SessionScopedAgent:
    """
    把 `create_agent` 的图包成**旧调用契约**，让上游一行都不用改。

    对外仍然是
        .invoke({"input": str}, config={"configurable": {"session_id": sid}})
            -> {"output": str}
    这正是 `RunnableWithMessageHistory(AgentExecutor)` 的形状，于是
    `_agent._lc_chat`、`/api/chat`、`/api/chat/stream` 的契约全部不变。

    它替 `RunnableWithMessageHistory` 做的三件事：
      ① 按 session_id 从 ChatStore 读历史 → 拼进 messages
      ② 调图
      ③ 把**新增的两条**（用户输入 + 最终回复）写回 ChatStore

    第 ③ 步是"重复追加"的风险点：图返回的 `messages` 里**含**传进去的历史，
    整段回写就会把旧消息又写一遍。所以这里只写新增的两条，与
    `RunnableWithMessageHistory` 当初的行为一致。
    """

    def __init__(self, agent: Any, chat_store: Optional[Any] = None) -> None:
        self._agent = agent
        self._store = chat_store
        self._history_factory = (
            _make_history_factory(chat_store) if chat_store is not None else None)

    # `_agent._lc_chat` 用这个属性判断"有没有 DB 历史"，保持同名同义。
    @property
    def get_session_history(self) -> Any:
        if self._history_factory is None:
            raise AttributeError("no chat_store configured")
        return self._history_factory

    def invoke(self, payload: dict, config: Optional[dict] = None) -> dict:
        from langchain_core.messages import AIMessage, HumanMessage

        message = payload.get("input", "")
        session_id = ((config or {}).get("configurable") or {}).get("session_id")

        history: list = []
        store_history = None
        if self._history_factory is not None and session_id:
            store_history = self._history_factory(session_id)
            history = list(store_history.messages)
        elif payload.get("chat_history"):
            # 无 DB 时上游自己带历史进来（`_agent._lc_chat` 的 else 分支）
            history = list(payload["chat_history"])

        human = HumanMessage(content=message)
        result = self._agent.invoke(
            {"messages": [*history, human]},
            config={"recursion_limit": RECURSION_LIMIT},
        )

        reply = ""
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content:
                reply = msg.content
                break

        # 只回写新增的两条 —— result["messages"] 里前面那段是传进去的历史
        if store_history is not None:
            store_history.add_message(human)
            store_history.add_message(AIMessage(content=reply))

        return {"output": reply}
