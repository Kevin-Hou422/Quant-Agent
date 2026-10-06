# Quant Agent — Autonomous Alpha Research & Paper-Trading Platform

An end-to-end quantitative research platform: a typed-AST alpha DSL, realistic backtesting with a
frozen out-of-sample holdout, Genetic-Programming factor discovery, a **portfolio-manager layer**
that turns factors into a dollar-denominated book, strategy-level validation gates, a paper-trading
engine with a daily scheduler, and a React/TypeScript workspace UI.

> **What this README is.** It describes what exists in the code **today**, including the parts that
> are deliberately *not* finished. Where a capability is planned but absent, it says so — see
> [§16 Status & Roadmap](#16-status--roadmap). The active plan is
> [backend/ULTIMATE_GOAL_ROADMAP.md](backend/ULTIMATE_GOAL_ROADMAP.md); the governing principles are
> [backend/RESEARCH_OPERATING_MODEL.md](backend/RESEARCH_OPERATING_MODEL.md).

> **⚠️ How to read the numbers this system produces.** Backtest Sharpe/return figures here are
> **not validated P&L**. The selection-bias half of the problem is fixed (frozen holdout, purged
> cross-validation, multiple-testing correction — see [§12](#12-statistical-integrity)), so
> *relative* conclusions ("is this selection process overfitting?", "does factor B add to strategy
> A?") are meaningful. The *absolute* level is not: the free price data has no delisted names, so
> historical survivorship bias remains, and forward paper-trading evidence has not been accumulated
> yet. Details in [§12](#12-statistical-integrity).

---

## Table of Contents

1. [What It Does Today](#1-what-it-does-today)
2. [Architecture](#2-architecture)
3. [Quick Start](#3-quick-start)
4. [Project Structure](#4-project-structure)
5. [Alpha DSL](#5-alpha-dsl)
6. [Research Engines](#6-research-engines)
7. [Data Engine & Datasets](#7-data-engine--datasets)
8. [From Factors to a Book: Portfolio Manager & Gates](#8-from-factors-to-a-book-portfolio-manager--gates)
9. [Trading Reality & Paper Execution](#9-trading-reality--paper-execution)
10. [API Reference](#10-api-reference)
11. [Frontend](#11-frontend)
12. [Statistical Integrity](#12-statistical-integrity)
13. [Database Schema](#13-database-schema)
14. [Configuration](#14-configuration)
15. [Testing & Engineering Discipline](#15-testing--engineering-discipline)
16. [Status & Roadmap](#16-status--roadmap)
17. [License, Data & Disclaimer](#17-license-data--disclaimer)

---

## 1. What It Does Today

- **Alpha DSL** — a typed-AST factor language (Lark grammar), ~50 operators (cross-sectional,
  time-series, group/sector, arithmetic), static validation (depth / window / look-ahead), and a
  vectorized NumPy/pandas executor.
- **Research-grade backtesting** — three-way IS / Validate / **frozen Test** split with embargo
  between every segment, `prev_w·price_chg` accounting, T+1 delay, square-root market impact, ADV
  liquidity caps, walk-forward folds, Deflated Sharpe, and CPCV-based overfitting probability.
- **Autonomous discovery** — `MarketObserver` derives hypothesis directions from price-volume
  observations (regime, cross-sectional dispersion, momentum/reversal breadth); `DiscoveryEngine`
  drives GP from those directions with **no user text and no LLM**, storing winners as CANDIDATE.
- **Genetic Programming** — `PopulationEvolver` evolves typed ASTs with structural mutations, a
  diversity-filtered `AlphaPool`, and a fitness measured by **purged K-fold cross-validation inside
  the in-sample block** (not by peeking at the holdout).
- **Portfolio-manager layer** — combines N factors into **one** dollar book: marginal-contribution
  factor admission, capacity limits (water-filling over ADV), capital allocation, gross/net and
  concentration risk gates, volatility targeting, no-trade bands, and fast/slow horizon profiling.
- **Strategy-level gating** — the strict gate is applied to the *traded strategy*, not to individual
  factors: per-segment OOS positivity, Deflated Sharpe with a **global** trial counter, Sharpe
  t ≥ 3.0 (Harvey-Liu-Zhu), and PBO via combinatorial purged cross-validation.
- **Risk attribution** — a Barra-lite factor risk model (`Σ = BΣ_fBᵀ + D`) decomposing portfolio
  variance into factor vs. specific risk, per-factor contributions, and net style exposures.
- **Lifecycle & approval** — a 7-state factor machine plus a separate **strategy-config** entity with
  its own proposed→approved→active state machine and an append-only decision ledger.
- **Paper trading** — `PaperBroker` reconciles day-by-day fills with the backtester to ~2.2e-16;
  `PositionStore` is idempotent and crash-recoverable; an APScheduler daily ingest + trading loop
  skips non-trading days via a DST-aware US market calendar.
- **AI chat agent** — a LangChain tool-calling agent over the research tools, with a **deterministic
  zero-LLM fallback** that runs the whole seed→GP→critic→save pipeline when no API key is set. The
  **LLM never touches the trading loop.**
- **Frontend** — React 19 / TypeScript workspace: Chat, Compiler, Ledger, Dataset, Live dashboard,
  approval queue, portfolio/strategy console, and a trading-reality panel.

---

## 2. Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        React 19 / TypeScript UI                            │
│  Chat · Compiler · Dataset · AlphaDashboard(+ApprovalQueue)                 │
│  PortfolioView (strategy console) · TradingRealityPanel                     │
└──────────────────────────────┬─────────────────────────────────────────────┘
                     HTTP / SSE (Vite proxy → :8000)
┌──────────────────────────────▼─────────────────────────────────────────────┐
│                             FastAPI (:8000)                                │
│  /api/chat[/stream] /api/workflow/* /api/gp/evolve /api/backtest/*         │
│  /api/alphas/* (validate|approve|reject|pending|decisions)                  │
│  /api/strategies/* (propose|pending|approve|reject)                         │
│  /api/portfolio/diagnostics /api/trading/status /api/datasets /api/regime   │
└─┬────────┬─────────┬──────────┬───────────┬──────────┬──────────┬──────────┘
  ▼        ▼         ▼          ▼           ▼          ▼          ▼
alpha_  backtest_  gp_engine  discovery  lifecycle  portfolio_  risk_
engine  engine     Evolver    Market     Validation manager     engine
DSL     Realistic  AlphaPool  Observer   LeakFilter Combiner    Exposures
AST     +costs     Mutations  Discovery  Promotion  Capacity    FactorRet
Parser  3-way      PurgedCV   Engine     Gate(A/B/C)Allocator   Σ=BΣfB'+D
Exec    CPCV/PBO                                    RiskGate    Attribution
  │        │          │           │          │       StrategyGate   │
  ▼        ▼          ▼           ▼          ▼          ▼            ▼
 data_engine (registry, PIT store, moomoo/yahoo providers,   trading_context
  health gate, market calendar, sectors, regime)             spread/borrow/
  │                                                          account (T1/T2/T3)
  ▼                                                                │
 execution/PaperBroker ── tasks/scheduler · daily_ingest · daily_trading_loop
  │                                                                │
  └──── db/ AlphaStore · StrategyStore · PositionStore · TrialLedger ─────────┘
         HoldoutLedger · DiagnosticsStore · RunManifest · ChatStore  (SQLite/WAL)
```

| Mode | Trigger | Pipeline |
|---|---|---|
| **Chat** | user message | `QuantAgent.chat()` → LangChain tools (or deterministic fallback) → DSL → backtest → reply |
| **Workflow A** | Generate | hypothesis → `GenerationWorkflow` → seeds → GP + Optuna → save (SSE) |
| **Workflow B** | Optimize | DSL → `OptimizationWorkflow` → GP + Optuna → save (SSE) |
| **Discovery** | nightly job | `MarketObserver` → `DiscoveryEngine` → GP → CANDIDATE (no LLM, no user text) |
| **Validation** | endpoint / gate | `LeakFilter` (factor) → `StrategyGate` (strategy: OOS/DSR/t/PBO) |
| **Paper** | scheduler | `DailyIngest` (health gate) → `DailyTradingLoop.run_portfolio` → `PaperBroker` → `AlphaMonitor` |

---

## 3. Quick Start

**One-command launcher (bash):**
```bash
./start.sh                 # backend (:8000) + frontend (:5173)
```

**Manual — backend:**
```bash
cd backend
python -m venv ../venv && source ../venv/Scripts/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --app-dir .        # http://localhost:8000  (docs: /docs)
```

**Manual — frontend:**
```bash
cd frontend
npm install
npm run dev                                       # http://localhost:5173 (proxies /api → :8000)
```

Optional `backend/.env`:
- `OPENAI_API_KEY=...` enables the LLM agent; without it the deterministic fallback runs automatically.
- `PRICE_SOURCE=moomoo` switches US price data to the moomoo OpenD gateway (requires OpenD running
  locally on port 11111). Default is `yahoo`.

The server **refuses to start** if bound to a non-loopback address unless `ALLOW_INSECURE_BIND=true`
— there is no authentication layer, so this is deliberate (see [§14](#14-configuration)).

---

## 4. Project Structure

```
Quant Agent/
├── start.sh · LICENSE (Apache-2.0) · docker/ · scripts/ · venv/
│
├── backend/
│   ├── ULTIMATE_GOAL_ROADMAP.md     # active plan (Phase S/9/TR/PM done; 10/12/13/14 + R open)
│   ├── RESEARCH_OPERATING_MODEL.md  # long-term principles (living)
│   ├── OPERATIONS.md                # validation-period operating rules + holdout discipline
│   ├── DEV_LESSONS.md               # post-mortems, each one enforced by a test
│   ├── MUTATION_LEDGER.md           # mutation-testing criteria + pointers to machine-readable state
│   ├── backend_retired_report/      # frozen archives (Phase 0–5, 6–8 plans, audits)
│   ├── tools/mutation/              # mutation runner (reproducible kill-rate measurement)
│   ├── tests/                       # 152 test files: unit/ integration/ meta/ golden/ performance/
│   └── app/
│       ├── main.py · config.py · dependencies.py
│       ├── api/                     # router.py (research + lifecycle + strategy + diagnostics),
│       │                            #   chat_router.py
│       ├── core/
│       │   ├── alpha_engine/        # parser, typed AST, validator, executor, fast_ops,
│       │   │                        #   generator, signal_processor, financial_interpreter
│       │   ├── backtest_engine/     # realistic_backtester, backtest_engine, transaction_cost,
│       │   │                        #   performance_analyzer, portfolio_constructor, risk_report,
│       │   │                        #   overfit_stats (PBO/CPCV), alpha_combiner, capacity
│       │   ├── data_engine/         # dataset_registry, pit_store, market_calendar, schema,
│       │   │                        #   health_report, regime_detector, sector_mapper,
│       │   │                        #   data_partitioner (3-way + PurgedKFold), providers/
│       │   ├── gp_engine/           # population_evolver, alpha_pool, mutations, fitness,
│       │   │                        #   evaluation_utils (purged-CV fitness), _rng
│       │   ├── discovery/           # market_observer, discovery_engine
│       │   ├── lifecycle/           # validation_gate, leak_filter, promotion_gate (A/B/C)
│       │   ├── portfolio_manager/   # manager, strategy_gate, strategy_builder, risk_gate, horizon
│       │   ├── risk_engine/         # factor_model (exposures, factor returns, Σ=BΣfB'+D, attribution)
│       │   ├── trading_context/     # context, spread (Corwin-Schultz), providers (quote/borrow/account)
│       │   ├── strategies/          # baselines (8 classic anomalies as a fallback book)
│       │   ├── ml_engine/           # proxy_model (XGBoost), alpha_evaluator, alpha_optimizer
│       │   ├── execution/           # paper_broker
│       │   ├── monitor/             # alpha_monitor (realized IC, decay)
│       │   └── workflows/           # alpha_workflows (Generation/Optimization)
│       ├── agent/                   # quant_agent, alpha_agent, _lc_agent, _tools, _fallback,
│       │                            #   _critic, _prompts, _memory, _chat_history
│       ├── db/                      # alpha_store, strategy_store, position_store, trial_ledger
│       │                            #   (+ HoldoutLedger), diagnostics_store, run_manifest,
│       │                            #   chat_store, alpha_lifecycle, _sqlite_utils (WAL)
│       └── tasks/                   # scheduler, daily_ingest, daily_trading_loop,
│                                    #   cost_calibration, backup, reasoning_log
│
└── frontend/src/
    ├── api/client.ts · store/workspaceStore.ts · hooks/useQuantWorkspace.ts
    └── components/
        ├── layout/ chat/ compiler/ dataset/ analysis/
        ├── dashboard/   # AlphaDashboard, ApprovalQueue
        └── portfolio/   # PortfolioView (strategy console), TradingRealityPanel
```

---

## 5. Alpha DSL

A typed expression language over a daily panel `(time × assets)`. Parsed by a Lark grammar into a
typed AST, statically validated, then executed vectorized.

- **Arithmetic / unary:** `add sub mul div pow signed_power neg abs log sqrt sign max2 min2`
- **Logical / conditional:** `logical_and logical_or logical_not if_else where trade_when`
- **Time-series:** `ts_mean ts_std ts_var ts_sum ts_delta ts_delay ts_max ts_min ts_rank ts_argmax
  ts_argmin ts_zscore ts_skew ts_kurt ts_entropy ts_decay_linear ts_corr ts_cov`
- **Cross-sectional:** `rank zscore scale normalize winsorize demean ind_neutralize`
- **Group / sector:** `group_rank group_zscore group_mean group_neutralize`
- **Inputs:** `data` — `open high low close volume vwap returns`, plus derived `sector` (GICS L1)

Example: `rank(ts_delta(log(close), 5))` — 5-day log-price momentum, cross-sectionally ranked.

The validator enforces max AST depth, max time-series window, and rejects look-ahead constructs.
Signals pass through `SignalProcessor` (truncation → decay → neutralization → **T+1 delay**) before
portfolio construction.

---

## 6. Research Engines

- **backtest_engine** — `RealisticBacktester` runs the segments with `prev_w·price_chg` accounting.
  `TransactionCostEngine` applies `slip_bps = spread/2 + impact_coef·vol·1e4·√participation` plus
  fixed and borrow costs; liquidity is enforced by `simulate_partial_fills` (the *same* primitive the
  paper broker uses, so the two engines cannot drift apart). `PerformanceAnalyzer` computes Sharpe,
  IC/IC-IR, turnover, drawdown, walk-forward folds and Deflated Sharpe; `overfit_stats` computes PBO
  by CSCV and by **CPCV** (combinatorial purged CV — the default, because CSCV's adjacent blocks let
  rolling operators leak across the boundary and *understate* overfitting).
- **gp_engine** — typed-AST evolution with point/hoist/param/subtree mutations, a diversity-filtered
  `AlphaPool`, a scale-stability structural penalty, and a bindable shared RNG for determinism.
  Fitness uses **purged K-fold CV inside the in-sample block** (`s_fitness_mode=purged_cv`), so the
  score no longer depends on what regime the last segment happened to be.
- **discovery** — `MarketObserver` scores six factor families from regime / dispersion / breadth /
  autocorrelation / volatility; `DiscoveryEngine` runs GP on the top families and saves winners as
  CANDIDATE. Registered as a nightly job under `ENABLE_DISCOVERY=true`. Deterministic, no LLM.
- **ml_engine** — `ProxyModel` (XGBoost overfitting proxy), `AlphaEvaluator`, `AlphaOptimizer`
  (Optuna over an IS-only objective).
- **agent** — `QuantAgent` picks the LangChain tool path when an LLM is configured, else the
  deterministic `FallbackOrchestrator`; `OverfitCritic` drives targeted correction rounds. Every
  response carries a `data_source` badge (`real:<name>` / `synthetic`) end-to-end, on both the POST
  and the SSE path.

---

## 7. Data Engine & Datasets

- **Providers** — `YahooFinanceProvider` (default), **`MoomooProvider`** (via a local OpenD gateway —
  the same source used for execution, which removes train/serve skew), `CcxtBinanceProvider`,
  `AkshareProvider`, `LocalParquetProvider`. All return the 7 standard daily fields.
- **Loading** — `load_registry_dataset()` → provider fetch → schema enforce → universe filter →
  sector attach → **health gate**. The health gate is fail-closed by default
  (`RESEARCH_HEALTH_FAIL_CLOSED=true`): low-quality data raises instead of silently flowing into
  research.
- **Point-in-time store** — `PITStore` appends `(field, date, as_of)` vintages, never rewriting
  history, so any replay can reconstruct the view as of a past date. `DailyIngest.ingest_incremental`
  backfills once and then appends only new bars.
- **Market calendar** — `market_calendar.py` (DST-aware session close, half-days, cross-checks data
  against the calendar and warns on disagreement). The scheduler skips non-trading days.
- **Datasets** (`dataset_registry.py`):

  | Name | Size | Region |
  |---|---|---|
  | `us_broad_large` | 95 tickers, all 11 GICS sectors | US |
  | `us_tech_large` / `us_financials` / `us_healthcare` / `us_energy` | 35 / 30 / 29 / 21 | US |
  | `crypto_major` / `crypto_alt` | 23 / 26 | Global |

  > `us_broad_large` is capped at ~95 names because a free moomoo account allows **100 historical-K
  > line symbols per month**. At the current $10k paper AUM the binding constraint is capital, not
  > breadth (only 10–30 names can be held), and forward paper accumulates its own data after the
  > initial backfill. Raising AUM later requires revisiting universe size, quota strategy, capacity
  > modelling and concentration limits together — documented in the roadmap.

---

## 8. From Factors to a Book: Portfolio Manager & Gates

A validated factor is not a strategy. The PM layer turns N factors into **one** dollar-denominated
book, and the strict gate is applied to that book rather than to the individual factors.

- **Combination & capacity** — `AlphaCombiner` (IC-IR weighted / equal / min-variance) produces one
  composite signal; `capacity.py` derives per-factor AUM ceilings from ADV participation and
  water-fills the allocation.
- **Marginal admission** (`marginal_factor_selection`) — a candidate factor joins only if it raises
  the *strategy's* out-of-sample Sharpe by `pm_marginal_min_improve`. This is the point of the
  design: a factor's value is its marginal contribution (∝ √(1−ρ²)), so per-factor high bars would
  reject exactly the low-correlation diversifiers that help most.
- **Risk gate** (`PortfolioRiskGate`) — gross/net exposure caps, single-name and sector concentration
  limits (reduce-only), volatility targeting against realised portfolio vol, and a drawdown circuit
  breaker.
- **Horizon** — factors are classified fast/slow by annualised turnover; a **no-trade band** (derived
  from estimated spreads, or configured) suppresses small drifts to cut turnover.
- **Strategy gate** (`StrategyGate`) — per-segment OOS positivity, **Deflated Sharpe using the global
  cross-session trial count**, Sharpe **t ≥ 3.0**, and **PBO via CPCV**. Fail-closed.
- **Graded promotion** (`promotion_gate.py`) — entry to PAPER is graded **A/B/C** (A = passed the
  strict gate, B = failed but Sharpe > 0, C = unfit). Experiment mode (default on) admits B/C to
  collect forward evidence **but labels the grade honestly**. Promotion to ACTIVE requires ≥60
  *forward* trading days with realised IC mean > 0 and t > 2.
- **Strategy configs as first-class entities** — `StrategyStore` persists a config
  (proposed→approved→active→retired/rejected) with an append-only decision ledger. What gets approved
  is a *portfolio configuration*, not a factor.
- **Baseline fallback** — when no in-house factor qualifies, `strategies/baselines.py` supplies eight
  documented classic anomalies (12-1 momentum, short-term reversal, low-vol, 52-week high, time-series
  trend, liquidity, idiosyncratic skew, 12M) so the book still trades and the run is labelled
  `used_baseline=True`.

### The unified gate chain

| Stage | Bar | Applied to |
|---|---|---|
| 1. Pool entry | low bar: leak / obvious-garbage filter only | factor |
| 2. Strategy validation | **strict**: all-fold OOS > 0 + DSR > 0.90 (global trials) + t ≥ 3.0 + PBO | **strategy** |
| 3. Factor admission | marginal OOS improvement after real costs | factor → strategy |
| 4. Enter PAPER | graded A/B/C, thresholds configurable | strategy |
| 5. → ACTIVE | strictest: ≥60 **forward** days, realised IC mean > 0, t > 2 | strategy |

---

## 9. Trading Reality & Paper Execution

Financial parameters that decide "can a $10k retail account actually make money here" are not
hidden defaults in the code. They are split into three disciplines:

- **T1 — explicit facts** (configured): account type, short permission, broker (`moomoo_us`,
  commission-free US equities).
- **T2 — derived estimates** (recomputed from data): per-name effective spread via Corwin-Schultz /
  Abdi-Ranaldi from free high/low, one-way cost, tradability filters, no-trade band.
- **T3 — knowable only at trade time** (must go through a provider): quotes, borrow availability,
  buying power. `get_trade_providers("live")` reads the moomoo paper account through the broker
  gateway and quotes through market snapshots; if either source is missing it **raises** —
  estimates are never allowed to masquerade as live data. Live borrow fees are deliberately not
  wired (the snapshot's unit is unverified; long-only does not need it).

`PaperBroker` fills at the close reusing the exact cost and liquidity primitives of the backtester
(reconciles to ~2.2e-16). `PositionStore` keeps idempotent positions/fills/PnL with
`state_before()` so an interrupted day can be replayed deterministically. Every `run_portfolio` call
persists a diagnostics record (`DiagnosticsStore`) exposing selection trace, strategy verdict, risk
report, horizon, T3 state, trading context, **risk attribution** and the **execution report** at
`GET /api/portfolio/diagnostics`.

### Execution on moomoo paper trading (Phase 12)

Off by default (`EXECUTION_MODE=off`). With `EXECUTION_MODE=moomoo_paper`, after the simulated book
has traded each day, the **last row** of the same risk-gated target weights is turned into share
orders on a moomoo **paper** account:

1. **Reconcile first** (this is also crash recovery): read broker orders and positions, difference
   cumulative fills into increments, and check *last trusted snapshot + our fills = broker
   positions*. Any mismatch (a split, a manual trade, an order that cannot be located) **stops
   trading** until a human accepts the broker state.
2. **Freshness** — the decision date must be the most recent *closed* session (real NYSE calendar).
3. **Account size** must agree with `PAPER_AUM` (one money figure, two descriptions of it).
4. Stale open orders from earlier decision dates are cancelled.
5. Targets → integer shares (always rounded toward zero) → **pre-trade gate** → **write intent,
   then place**. Each order carries a deterministic client id in moomoo's `remark` field, so a re-run
   never places a second order and a crash between "placed" and "recorded" is recovered by remark.

Limit orders (`ref close × (1 ± 50bps)`, DAY) are placed after the close and fill at the next open;
the gap to the simulated close fill is exactly what the fidelity report measures.

- **Pre-trade gate** (fail-closed on every rule): kill switch, daily-loss breaker (only risk-reducing
  orders pass), reference price vs broker price, no accidental shorts, order notional vs ADV
  (same `max_participation_pct` as the cost model), post-trade name and gross limits (same settings
  as the portfolio risk gate), and buying power at the limit price without counting same-batch sell
  proceeds.
- **Kill switch** — two-step (one-time token, actor and reason): persist the halt first, then cancel
  every open order and place flatten orders for every position. If the broker is unreachable the
  halt still takes effect and flattening resumes on the next cycle.
- **Real money is refused in code**, not by configuration: the moomoo SDK defaults `trd_env` to
  `REAL`, so every trading call passes `SIMULATE` explicitly and the gateway rejects anything else.
- **OpenD offline fails fast.** Measured: when OpenD is not listening, constructing an SDK context
  never returns (it retries every 8 s and leaves a non-daemon thread behind). Every connection is
  therefore preceded by a 2-second TCP probe; if the port is closed the cycle is skipped with a clear
  reason instead of hanging the scheduler.
- **Fidelity ladder** (`GET /api/execution/fidelity`, and monthly with the cost calibration) —
  internal sim (close fill) vs moomoo paper (next-open fill), split into overnight gap and at-open
  execution, fill ratios, and a bounded `impact_coef` suggestion that is never auto-applied.
  Permanent impact is reported as **unidentifiable** from paper fills rather than estimated.

> The moomoo simulate account is assumed to be **dedicated** to this system: broker positions are
> the book's truth, and positions outside the target are traded down to zero.

> The daily loop replays a **historical** window and then appends incrementally as new bars arrive.
> Real forward operation requires the OpenD gateway to stay online each trading day; no forward
> track record has been accumulated yet (see [§16](#16-status--roadmap)).

---

## 10. API Reference

Base URL `http://localhost:8000/api` (interactive docs at `/docs`).

**Agent & Chat** — `POST /agent/run` · `POST /chat` · `POST /chat/stream` ·
`POST|GET|PATCH|DELETE /chat/sessions[...]`

**Discovery & Workflows** — `POST /gp/evolve` · `POST /workflow/generate[/stream]` ·
`POST /workflow/optimize[/stream]`

**Backtest & Evaluate** — `POST /backtest/run` · `/backtest/realistic` · `/backtest/multi` ·
`/backtest/walk_forward` · `POST /alpha/simulate` · `/alpha/optimize` · `/alpha/save`

> Every one of these applies the three-way split and returns a `partition` block describing it.
> Touching the frozen Test segment requires an explicit `report_test=true` and is **metered**
> (see [§12](#12-statistical-integrity)).

**Factor lifecycle** — `GET /alphas/dashboard` · `/alphas/pending` · `/alphas/{id}/ic_history` ·
`/alphas/{id}/walk_forward` · `/alphas/{id}/decisions` · `PATCH /alphas/{id}/status` ·
`POST /alphas/{id}/validate` · `/alphas/{id}/approve` · `/alphas/{id}/reject` · `/alphas/{id}/retrain`

**Strategy configs** — `POST /strategies/propose` · `GET /strategies` · `/strategies/pending` ·
`/strategies/{id}` · `POST /strategies/{id}/approve[?activate]` · `/strategies/{id}/reject`

**Paper, data & diagnostics** — `GET /paper/{id}/pnl` · `/portfolio/diagnostics` ·
`/trading/status` · `/scheduler/status` · `/datasets` · `/datasets/{name}/health` · `/regime` ·
`/report/query`

**Execution (moomoo paper)** — `GET /execution/status` · `POST /execution/reconcile` ·
`POST /execution/reconcile/accept` · `POST /execution/kill_switch/arm?action=engage|reset` →
`POST /execution/kill_switch/confirm` · `GET /execution/fidelity?start&end`

---

## 11. Frontend

React 19 + TypeScript + Vite + Zustand + ECharts, styled as an OS-style workspace.

- **Chat** — streaming agent chat with step-by-step `ThoughtBlock` reveal and a data-source badge
  (green for real data, amber **"⚠ synthetic (metrics invalid)"**).
- **Compiler** — DSL editor, run config, console output.
- **Dataset** — catalog + health.
- **AlphaDashboard** — paper equity curve, IC history, walk-forward charts, plus **ApprovalQueue**
  (pending candidates with approve/reject and the decision lineage).
- **PortfolioView** — strategy console: composition and per-factor quota bars, gate verdict
  (Sharpe / DSR / t / PBO), risk report (name-clipped / sector-scaled / gross-scaled / vol-scaled),
  turnover and no-trade band, A/B/C grade badge, and propose/approve/reject actions.
- **TradingRealityPanel** — data source and OpenD connectivity, estimated spreads and one-way cost,
  tradability/shortability, **T3 provider mode (sim vs live shown prominently)**, and gate grading
  with the thresholds actually in force.
- **ExecutionPanel** — moomoo paper positions, open and recent orders (gate rejections included),
  fills with slippage against the simulated close, reconciliation status and discrepancies with an
  accept action, the two-step kill switch, the sim-vs-paper fidelity report and the audit log.

---

## 12. Statistical Integrity

This is the part of the system most likely to be quietly wrong, so it is built as machinery rather
than convention.

- **Three-way split, everywhere** — IS | embargo | Validate | embargo | **Test**. The last ~2 years
  are frozen by *calendar* (not by ratio), so the window does not drift when you change the request
  dates. Every research endpoint returns the split it used; when a panel is too short to honour the
  freeze, the response says so (`degraded: true`) instead of silently weakening.
- **The holdout is metered** — touching the Test segment is an explicit opt-in and every use is
  recorded in an append-only ledger. Exceeding the budget (default: once) returns
  `over_budget: true` alongside the numbers. It does not block — blocking would just push people to
  work around the ledger — but the count travels with the result.
- **Selection happens only on Validate** — GP fitness uses purged K-fold CV inside IS; the validation
  gate and the strategy gate both see only the selection panel.
- **Multiple-testing correction** — a persistent cross-session `TrialLedger` counts every GP
  individual and Optuna trial ever run; Deflated Sharpe deflates by that global count, not by 1.
- **PBO via CPCV** — combinatorial purged cross-validation with purging around each held-out block.
  The reported `pbo_method` always states which estimator produced the number.
- **Reproducibility** — `RunManifest` records dataset SHA-256 + git commit + seed + config + the
  split used; a bindable shared RNG removes global-random order dependence; golden-master tests pin
  backtest arithmetic to 1e-12.
- **Risk attribution** — factor vs specific variance decomposition with an exactly checkable identity
  (`factor_var + specific_var == total_var`).

**What is still not fixed, and why it matters:**

- **Survivorship bias remains.** Free price sources carry no delisted names. This inflates historical
  backtests in an unknown direction and magnitude. The decision on record is *not* to buy a
  delisting-inclusive source but to rely on forward paper trading, which is free of the bias by
  construction.
- **No forward evidence yet.** The replay/forward split exists (`alpha_ic_history.is_forward`) and
  the →ACTIVE gate consumes forward samples only, but the count is currently zero. The gate is
  therefore not enforced by default (`TR_ENFORCE_ACTIVE_GATE=false`).
- **Risk attribution covers 5 price-volume styles + sectors.** `size` and `value` need fundamentals
  (Phase 10), so every attribution payload carries `styles_missing: ["size","value"]` — a "66% factor
  risk" figure must not be read as "the major styles are covered".

---

## 13. Database Schema

SQLite via SQLAlchemy, hardened with WAL + `busy_timeout`. Live databases must not sit in a
cloud-sync folder (a lesson learned the hard way — see `DEV_LESSONS.md` §Q).

**`backend/alphas.db`** (`DATABASE_URL`)

| Table | Contents |
|---|---|
| `alpha_records` | factor ledger: `dsl, hypothesis, ann_return, sharpe, max_drawdown, ic_ir, ann_turnover, status, reasoning(JSON)` (append-only) |
| `alpha_ic_history` | per-day realised IC, with an **`is_forward`** flag that only ever upgrades (a replay cannot erase accumulated forward evidence) |
| `alpha_decisions` | append-only approve/reject lineage for factors |
| `strategy_configs` / `strategy_decisions` | strategy configs + their approval lineage |
| `paper_positions` / `paper_fills` / `paper_daily_pnl` | idempotent paper trading state |
| `trial_ledger` | global multiple-testing counter (cross-session) |
| `holdout_usages` | append-only frozen-Test usage ledger |
| `portfolio_diagnostics` | per-run diagnostics (verdicts, risk, horizon, T3, attribution, execution) |
| `exec_orders` / `exec_fills` | moomoo paper orders (write-ahead intent, unique client id) and fill increments (unique per cumulative quantity) |
| `exec_snapshots` / `exec_events` / `exec_state` | reconciliation snapshots, audit log (kill switch, accepts, recovery), kill-switch state |
| `run_manifests` | reproducibility ledger |
| `chat_sessions` / `chat_messages` | agent memory |

**`backend/scheduler_jobs.db`** (`SCHEDULER_DB_URL`) — APScheduler jobstore.
**`backend/pit_store/`** — point-in-time parquet vintages (gitignored).

---

## 14. Configuration

Environment variables (or `backend/.env`), read by [config.py](backend/app/config.py). Selected keys —
the file itself is the authority and documents *why* each default is what it is.

| Var | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./alphas.db` | main ledger |
| `OPENAI_API_KEY` | *(empty)* | enables the LLM agent; empty → deterministic fallback |
| `DEFAULT_DATASET` / `DEFAULT_START` / `DEFAULT_END` | `us_tech_large` / `2020-01-01` / `2024-01-01` | research window |
| `PRICE_SOURCE` | `yahoo` | `yahoo` \| `moomoo` (single authoritative source for research+execution) |
| `MOOMOO_HOST` / `MOOMOO_PORT` | `127.0.0.1` / `11111` | OpenD gateway |
| `EXECUTION_MODE` | `off` | `moomoo_paper` = place real orders on the moomoo **paper** account (real money is refused in code) |
| `MOOMOO_SECURITY_FIRM` / `MOOMOO_TRD_ACC_ID` | `FUTUINC` / `0` | broker entity; `0` = first US paper account |
| `EXEC_LIMIT_BAND_BPS` / `EXEC_FLATTEN_BAND_BPS` | `50` / `500` | limit band for rebalancing and for flattening |
| `EXEC_MAX_DAILY_LOSS` / `EXEC_MAX_PRICE_DEVIATION` / `EXEC_MAX_AUM_MISMATCH` | `0.05` / `0.15` / `0.5` | daily-loss breaker, stale-price guard, account-vs-`PAPER_AUM` tolerance |
| `ENABLE_SCHEDULER` / `ENABLE_PAPER_TRADING` / `ENABLE_DISCOVERY` | `false` | nothing starts trading or mining by itself |
| `AUTONOMY_MODE` | `manual` | `manual` = human approves every promotion |
| `PAPER_AUM` | `1_000_000` | the single source of AUM for PM capacity and broker costs |
| `S_THREE_WAY_ENABLED` | `true` | force the three-way split on every research path |
| `S_TEST_FREEZE_YEARS` / `S_HOLDOUT_BUDGET` | `2.0` / `1` | frozen holdout window and one-shot budget |
| `S_FITNESS_MODE` / `S_CV_FOLDS` / `S_EMBARGO_DAYS` | `purged_cv` / `5` / `20` | GP fitness discipline |
| `S_USE_CPCV` | `true` | PBO estimator (CSCV understates overfitting) |
| `FACTOR_GATE_MODE` | `leak` | factor-level bar: leak filter only; the strict gate is at strategy level |
| `TR_EXPERIMENT_MODE` / `TR_ENFORCE_ACTIVE_GATE` | `true` / `false` | admit graded B/C to paper; →ACTIVE gate records but does not block yet |
| `RISK_MAX_GROSS` / `RISK_MAX_NAME_WEIGHT` / `RISK_MAX_SECTOR_WEIGHT` | `1.0` / `0.10` / `0.30` | portfolio risk limits |
| `RISK_TARGET_VOL_ANN` / `RISK_MAX_DRAWDOWN` / `RISK_HALT_ON_DRAWDOWN` | `0.0` / `0.20` / `false` | vol targeting (0 = off), drawdown breaker |
| `RISK_ATTR_LOOKBACK` | `252` | risk-model estimation window (diagnostic only) |
| `RESEARCH_HEALTH_FAIL_CLOSED` / `RESEARCH_MIN_HEALTH` | `true` / `0.7` | reject low-quality data instead of degrading |
| `CALENDAR_ALLOW_HEURISTIC` | `false` | never guess trading days |
| `ALLOW_INSECURE_BIND` | `false` | refuse to bind a non-loopback address (there is no auth layer) |

> **No hard gate is enabled by default.** `pm_strategy_gate_block`, `risk_halt_on_drawdown` and
> `tr_enforce_active_gate` are all `false` during the evidence-gathering phase: verdicts are recorded
> but do not stop trading. This is a deliberate choice, pinned by a test so it cannot drift silently.

---

## 15. Testing & Engineering Discipline

```bash
cd backend  && pytest -q          # 4448 passed, 1 skipped  (173 test files)
cd frontend && npm run test       # 101 passed (9 files)
cd frontend && npm run build      # tsc type check
```

Passing tests are treated as a weak signal on their own. Three mechanisms back them up:

- **Mutation testing** (`tools/mutation/`, `MUTATION_LEDGER.md`) — kill rate is measured per module
  and recorded in a machine-readable manifest that a meta-test reconciles against the source tree.
  Every surviving mutant must either be killed by a new test or carry a **mechanically verifiable**
  equivalence proof; proofs that cannot be defended are listed as unproven rather than hand-waved.
  New code added to a module is verified point-by-point before merge.
- **Enforced lessons** (`DEV_LESSONS.md` + `tests/meta/test_lessons_enforced.py`) — each post-mortem
  becomes a test that fails on relapse: no assertions that tolerate 5xx, no silently swallowed
  exceptions in gates, fallbacks must lean conservative, every endpoint that loads a dataset must
  also freeze a holdout, fixtures must not fake ±1% high/low, and so on. A meta-rule checks that
  every lesson has a check.
- **Debt ledgers** — untested routes, synthetic-only endpoints, silent `except` counts and unmeasured
  mutation points are all tracked as ratchets that may shrink but never grow.

`tests/` is split into `unit/`, `integration/`, `meta/` (invariants and lesson enforcement),
`golden/` (1e-12 numerical baselines) and `performance/`.

---

## 16. Status & Roadmap

**Done**

| Area | Status |
|---|---|
| DSL, data engine, backtesting, GP, ML proxy, agent, lifecycle, monitoring, paper broker, reproducibility | ✅ Phases 0–8 |
| Autonomous discovery + validation gate + human approval | ✅ Phase 9 |
| Trading reality: single authoritative source (moomoo), derived costs, T3 providers, graded gates | ✅ Phase TR |
| Portfolio manager: combination, capacity, allocation, risk gate, horizon, strategy configs | ✅ Phase PM |
| Data-contract audit (fail-closed synthetic data, provenance badges) + full test audit | ✅ Phases A/B |
| Statistical foundations: three-way split, frozen metered holdout, purged-CV fitness, CPCV/PBO | ✅ Phase S |
| Forward incremental ingest, PIT append, DST-aware calendar, replay/forward split | ✅ Phase 11 (code; not yet exercised live) |
| Risk attribution + structured covariance | ✅ Phase R.2 (part) |
| moomoo paper execution: reconciliation/crash recovery, pre-trade gate, two-step kill switch, sim-vs-paper fidelity, live T3 providers, execution panel | ✅ Phase 12 + FE-12 (code; **not yet run against a live OpenD gateway**) |

**Not built yet** — the honest list:

- **Phase 12 — live verification.** Everything is tested against a fake trading context whose call
  signatures and returned columns are checked mechanically against the installed moomoo SDK, but no
  order has been placed on a real paper account yet. Two behaviours can only be confirmed there:
  whether the paper account serves history-order queries, and whether orders placed after the close
  appear in the next session's order list.
- **Phase 10 — alternative data.** Fundamentals/earnings with point-in-time visibility and sparse
  quarterly fields in the DSL. Blocks the `size`/`value` risk styles and any valuation factor.
- **Phase 13/14 — red-team agent, fully autonomous mode, validation-period operations.**
- **R.2 remainder** — residual-based style neutralisation (changes signals, so it needs its own
  validation) and alpha-vs-risk-premium separation; `beta_neutral` closure is deferred until shorting
  is enabled.
- **R.4** — Ledoit-Wolf shrinkage, HRP, turnover in the optimisation objective.
- **Frontend** — FE-8/10/11/13/R panels (cost calibration, alt-data, forward status, red-team,
  research-credibility charts). Backend data for the research-credibility panel is already available
  at `/api/portfolio/diagnostics`.
- **Operational milestone M5** — 60 trading days of forward paper evidence and a first validation
  report. Not started; this is what would turn "research-grade estimate" into "measured".

---

## 17. License, Data & Disclaimer

> Not legal advice. Consult a professional before commercial use.

### Code license
This framework/code is licensed under the **Apache License 2.0** (see [LICENSE](LICENSE)) —
permissive, with an explicit patent grant.

The **edge is separate from the code** and is never part of this repository: profitable
factors/alphas, tuned parameters, live position sizing, API keys and broker credentials stay private
(see `.gitignore`). Open-sourcing the framework costs nothing; open-sourcing a working strategy would
crowd it out of existence. If a real edge emerges later, an **open-core** split (framework open,
strategies proprietary) or a source-available licence is the natural next step — narrowing from
permissive is easy, the reverse is not.

### Data — bring your own
This repo ships **no market data**.
- **yfinance / Yahoo** data is for personal research; Yahoo's ToS restricts commercial use and
  redistribution — do not bundle or redistribute it.
- **moomoo OpenAPI** data is subject to your brokerage agreement and market-data entitlements.
- Paid/academic sources (Sharadar, CRSP, Compustat, …) have strict redistribution limits — never
  commit or ship their data with the code.

Point-in-time data you accumulate (`pit_store/`) and all databases (`*.db`) are gitignored and stay
local. Supply your own data and keys via `backend/.env`.

### Disclaimer
For **research and educational purposes only**. Nothing here is financial or investment advice.
Provided **"as is", without warranty of any kind**. Backtest and paper-trading results are not
indicative of future performance and, as [§12](#12-statistical-integrity) states plainly, are
currently research-grade estimates rather than validated P&L. You are solely responsible for any use,
including any real capital at risk.

### Dependencies
Third-party libraries retain their own licences (FastAPI, NumPy, pandas, LangChain, React, …
predominantly MIT / Apache-2.0 / BSD). Verify compatibility before redistribution.
