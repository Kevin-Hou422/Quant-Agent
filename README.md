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
> **not validated P&L**. Selection bias is *controlled, not eliminated*: a frozen, metered holdout,
> purged cross-validation and multiple-testing corrections (DSR, HAC t, PBO) are in place — see
> [§12](#12-statistical-integrity) — but each has a limited scope (PBO covers the selected factors'
> returns, not the whole GP / combination / human selection history; the global trial count is only
> as complete as the recorded runs). *Relative* conclusions ("is this selection process overfitting?",
> "does factor B add to strategy A?") are informative within those limits. The *absolute* level is not: the free price data has no delisted names, so
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
  observations (regime, cross-sectional dispersion, momentum/reversal breadth) and, when the dataset
  carries fundamentals, from the valuation spread and the earnings calendar; `DiscoveryEngine`
  drives GP from those directions with **no user text and no LLM**, storing winners as CANDIDATE.
- **Point-in-time fundamentals** — SEC EDGAR facts aligned to their **actual filing date** (visible
  from the next trading day), giving eight daily fields: market cap, earnings yield, book-to-price,
  ROE, gross margin, asset growth, standardized earnings surprise, and days since the last earnings
  release.
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
│   ├── tests/                       # 191 test files: unit/ integration/ meta/ golden/ performance/
│   │                                #   fixtures/ = trimmed real SEC EDGAR + price responses
│   └── app/
│       ├── main.py · config.py · dependencies.py
│       ├── api/                     # router.py (research + lifecycle + strategy + diagnostics),
│       │                            #   chat_router.py
│       ├── core/
│       │   ├── alpha_engine/        # parser, typed AST, validator, executor, fast_ops, fields,
│       │   │                        #   generator, signal_processor, financial_interpreter
│       │   ├── backtest_engine/     # realistic_backtester, backtest_engine, transaction_cost,
│       │   │                        #   performance_analyzer, portfolio_constructor, risk_report,
│       │   │                        #   overfit_stats (PBO/CPCV), alpha_combiner, capacity
│       │   ├── data_engine/         # dataset_registry, pit_store, market_calendar, schema,
│       │   │                        #   health_report, regime_detector, sector_mapper,
│       │   │                        #   data_partitioner (3-way + PurgedKFold), providers/,
│       │   │                        #   fundamentals/ (edgar, pit, panels, prices, store, service)
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
- **Cross-sectional:** `rank zscore scale normalize winsorize demean ind_neutralize fund_rank`
- **Group / sector:** `group_rank group_zscore group_mean group_neutralize`
- **Inputs:** `data` — `open high low close volume vwap returns`, plus derived `sector` (GICS L1)
- **Fundamental inputs** (US datasets, `ENABLE_FUNDAMENTALS=true`) — fields carry a class, because
  the same operator means different things on each:

  | Class | Fields | Behaviour |
  |---|---|---|
  | valuation | `market_cap` `earnings_yield` `book_to_price` | market cap moves daily; the numerators update quarterly |
  | quarterly | `roe` `gross_margin` `asset_growth` `sue` | a **step function**: constant between filings |
  | event | `days_since_earnings` | counter, reset the trading day after an earnings release |

Examples: `rank(ts_delta(log(close), 5))` — 5-day log-price momentum, cross-sectionally ranked.
`fund_rank(earnings_yield)` — cheapness; `fund_rank(trade_when(days_since_earnings < 60, sue))` —
post-earnings-announcement drift.

The validator enforces max AST depth, max time-series window, and rejects look-ahead constructs.
It also rejects short-window dispersion operators on step fields (`ts_std(roe, 20)` is zero almost
everywhere — as a denominator it yields `inf`, not a volatility). `fund_rank` is `rank` with a
coverage floor: a day on which fewer than half the names have a value produces no signal, instead of
ranking three stocks as if they were a cross-section. A dataset without fundamentals has no such
fields, so an expression that references one **raises** rather than evaluating to NaN.
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
- **discovery** — `MarketObserver` scores six price-volume factor families from regime / dispersion /
  breadth / autocorrelation / volatility, plus `value` / `profitability` / `earnings` when the
  dataset has those fields at ≥ 50% coverage (valuation spread vs. its own history; defensive
  regimes; share of names that reported in the last 20 sessions). `DiscoveryEngine` runs GP on the
  top families and saves winners as CANDIDATE. Seeds that reference fields the dataset lacks are
  dropped before GP. Registered as a nightly job under `ENABLE_DISCOVERY=true`. Deterministic, no LLM.
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
- **Fundamentals** (`data_engine/fundamentals/`, off by default) — facts come from SEC EDGAR
  `companyfacts`, not from a vendor's "latest financials" table, because the one thing that must not
  be wrong is *when a number became knowable*. Every fact carries its `filed` date and a restatement
  is a later-`filed` row for the same period, so "the latest version disclosed by day D" is
  reconstructed without assumptions. A fact filed on day F is visible from the **first trading day
  after F**; values forward-fill and expire after 200 days without a new filing. TTM figures are
  assembled only from directly reported numbers (prior fiscal year + current YTD − prior-year YTD).
  Market cap is **unadjusted** close × shares, with the share count carried across splits (the two
  price sources disagree on what "unadjusted" means — yfinance's `Close` is already split-adjusted —
  and each is handled explicitly). Facts are stored append-only under `<PIT_STORE_DIR>/fundamentals`;
  the daily panels are recomputed from (facts, prices) and are *not* written to the price PIT store.
  Unadjusted closes are historical facts, so they are cached append-only and only the missing tail
  is requested from the price source.
  Known gaps are reported, not papered over: multi-class issuers without a usable share count
  (`BRK-B`, `V`) have no valuation fields, and a company whose recent filings are missing upstream
  is named in the load report as stale.
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

### Going live on the moomoo paper account

```powershell
copy backend\.env.forward.example backend\.env      # then edit PAPER_AUM etc.
cd backend
..\venv\Scripts\python -m app.tasks.forward preflight   # every check must be OK
..\venv\Scripts\python -m app.tasks.forward run-now     # optional: first supervised run now
cd ..
powershell -ExecutionPolicy Bypass -File scripts\start_forward.ps1 [-RegisterLogonTask]
```

- **Preflight** (`python -m app.tasks.forward preflight`, or `POST /api/execution/preflight`) has
  two layers and reports both: `connected` (OpenD listening, simulate account resolved, account size
  vs `PAPER_AUM`, kill switch, reconcile — the first reconcile establishes the trusted baseline) and
  `ready` (additionally: execution mode, scheduler switches *and* the jobs the scheduler actually
  registers, live-data location and writability, trading calendar, a US dataset, an **approved and
  activated strategy**, the fundamentals that strategy depends on — if any of its factors reference
  a fundamental field, the switch must be on, an SEC identity configured and local filings fresh for
  at least half the universe — and a live fetch proving the data source already has the last closed
  session's bar). It never places an order. `run-now` refuses to run unless every blocking check
  passes, runs the scheduler's own `daily_trading_job`, and exits non-zero unless execution actually
  completed (blocked, submit errors, portfolio errors and rejected ingests are all failures).
- **Only an activated strategy trades at the broker.** Without an active strategy config the daily
  loop still runs the simulated book (PAPER factors or the baseline library) but the broker side only
  reconciles. An active config whose components are unavailable halts new risk instead of trading
  other factors; its frozen combination weights are used as approved, not refitted daily.
- **The active config must carry an executable spec** (external review 2026-10-18, F07). Proposals
  store a hashed spec — component expressions, frozen weights, method, AUM, no-trade band, risk
  limits and vol-estimation window, signal parameters, broker/short flags, dataset and price source,
  the universe (ticker hash), the execution account and order/quote rules, and the submission
  schedule (spec v2, review §5.2). The daily loop trades exactly that spec and stops rebalancing if
  any runtime parameter differs; configs approved before 2026-10-18 have no spec and must be
  re-proposed and re-approved (roadmap D13).
- **One strategy contract** (review N05-e). `strategy_contract.StrategyContract` is frozen at
  proposal, the gate evaluates that contract, and the proposal's risk snapshot, the strategy gate's
  backtest, the daily loop's simulated book and broker targets, and forward evidence are all
  generated from it through a single `generate_targets` (book → vol targeting → hard limits →
  drawdown flatten → no-trade band → post-band projection). Only the fill adapter differs
  (BacktestEngine vs PaperBroker + broker).
- **Risk-increasing orders need an independent quote** (F09, review N06): opening, adding to or
  flipping a position requires a moomoo snapshot quote with a price, a timestamp that is not in the
  future, agreement with the reference price, and an eligible session (exchange time): within 30
  minutes of the last closed session's close, that day's after-hours, or a later session's
  pre-market / regular hours — earlier intraday, same-day pre-market and non-trading-day timestamps
  are rejected (review N06-d). Reductions and flattening need no quote. Account operations from the
  scheduler, `run-now`, the API and recovery are serialised by an OS file lock keyed by the **broker
  account** in a machine-wide directory (`ACCOUNT_LOCK_DIR`, default `%PROGRAMDATA%\QuantAgent`), and
  the execution DB refuses to trade a different account than the one it was first bound to (review
  §5.3); the kill switch is re-read before every order in a batch (N01).
- **One clock for signal, fill, label and archive** (decision D14 (a) + C3, clock version `c2`,
  `backend/app/core/clock.py`). Information cutoff = T close (after the health gate); the signal and
  targets are frozen then, with no extra signal delay. Execution is a T+1 DAY limit order (reference
  = T close, band = `EXEC_LIMIT_BAND_BPS`); the backtest and the internal paper book fill it by the
  same rule — open price if the open is inside the limit, else the limit price if the day's range
  reaches it, else no fill — so a planned open execution is not assumed to fill at the open. The
  predictive label of signal T is open(T+1) → open(T+2); the archive deadline is open(T+1), the
  label's start, so no part of the scored return has happened when the signal is archived. Realized
  book / broker P&L is recorded separately from predictive IC. Validate-segment features are
  computed on the known history before the segment (252-day factors are scorable from the first
  Validate day; only Validate dates are scored; the book starts flat).
- **Forward evidence = signal archived before the outcome** (F06). A day's IC counts as forward only
  if the signal it uses was archived before its label starts (the open of T+1, see the clock above);
  backfilled, re-run or offline results are replay. Strategy promotion reads that strategy's own record (id + spec hash), and the
  t-stat is the smaller of the plain and the Newey-West one. Evidence rows carry the input identity
  (data digest, universe, actual price source, PIT vintage, code version), and a spec/runtime
  mismatch quarantines the day instead of crediting it (review N02). Days whose signal was not
  archived in time (first run, backfill after downtime, late archive, source change) are kept as
  `replay` evidence for research but never count as forward (review N03-b). Promotion to ACTIVE uses
  the validation design **registered at proposal time** — thresholds, t-stat rule, evidence rule,
  clock version, registration time, hashed — and counts only forward evidence dated after
  registration; later changes to the `TR_*` settings do not move an existing strategy's bar (review
  N05-d). Every stored result carries the signal/fill clock version (`c2` since 2026-10-22;
  older rows are `c1`). Factor monitoring and the IC history statistics read the current clock only;
  `c1` rows stay in the database, labelled, and are not inherited (review §5.1).
- **Fundamentals have a source / period quality gate** (review §5.4, F13). A strategy whose
  expressions use fundamental fields passes proposal and trades only if the fundamentals quality
  report for that very panel shows last-day coverage ≥ `FUND_MIN_COVERAGE` (0.8) for each field
  used, and at most `FUND_MAX_ESTIMATED_SHARES` (10%) of names valued with an estimated share count
  (diluted weighted average). SUE is paired with earnings releases by **fiscal quarter**, not by date
  order; the share count expires with its own fact; suspected small stock dividends (1.15–1.5× jumps
  in the adjustment factor) quarantine valuation fields until a newer share count arrives.
- **Formal forward days are recorded, and started by a human** (review §7.3). `forward_record.py`
  writes one record per strategy per decision day (signal / target times, expected and actual
  session, target weights and shares, fills, unfilled reasons, cash and positions, costs, net return,
  benchmark excess, drawdown, gross/net, capital utilisation, reconcile status and anomalies; values
  that cannot be computed are `None` with a reason). A day is *formal* only after a human start
  record (unique start time, spec hash, code version, data identity, environment) and only if it ran
  between the decision-day close and the next open with execution live; late reruns cannot overwrite
  a formal day. No formal forward has been started. The benchmark is the equal-weight universe
  with rules fixed in advance (`backend/app/core/benchmark.py`, decision D16): members frozen when
  the formal forward starts, daily rebalance, close-to-close window, and a member with a missing price
  or a quarantined corporate action is held at a 0 return *in the denominator* (never dropped and
  re-averaged). The plain difference to this fully-invested benchmark is labelled as such — it is
  not alpha for a low-exposure book — and an exposure-matched difference is recorded next to it;
  SPY as a second benchmark is deferred. Operating limits (daily-only breaker, log-only alerts) are
  spelled out in `backend/OPERATIONS.md` §7.
- **Unknown corporate actions are quarantined** (decision 18 of 2026-10-22). A one-day move of
  |log return| ≥ log 1.4 on any of the last few bars (an unadjusted split, a special dividend, …)
  flags the name; the flag persists until a person reconciles broker shares / cash and the data
  adjustment and resolves it (`backend/tools/ops/ca_flags.py`). While flagged, the paper book and the
  broker order gate may reduce but never add risk on that name; broker share counts that disagree
  with the ledger by a split-like ratio are annotated for reconciliation. The full corporate-action
  ledger is still deferred.
- **The daily job runs twice** (21:30 UTC and a 22:45 UTC retry for late data). Every run reconciles
  and continues a pending flatten even when there is no new bar or the ingest is rejected; reruns are
  idempotent (client-id dedupe at the broker, `last_pnl_date` resume in the simulated book).
- **Live data must not sit in a cloud-synced folder.** With forward trading enabled, the service
  refuses to start if the resolved database, scheduler store or PIT directory is inside
  OneDrive/Dropbox/Google Drive/iCloud. The default `./alphas.db` resolves into OneDrive when the
  repo lives there, so `.env.forward.example` uses absolute paths under `C:\QuantAgentData`.
- `scripts/start_forward.ps1` starts OpenD if needed, waits for it to listen (log in once in its
  window, or enable auto-login), runs the preflight, and only then starts the API with the scheduler.
  The machine must be on with OpenD logged in at 21:30 UTC on trading days.
- The template follows the roadmap's TR.2 decision: `PRICE_SOURCE=moomoo` (research and execution
  from one source) on `us_broad_large` (≤100 names, fits the free account's monthly kline quota).

> An external audit (2026-10-07, 16 findings) blocked go-live. 15 are fixed and F15 partially
> (forward evidence is filtered to dates after the strategy's creation, but the portfolio book is
> still shared rather than kept per strategy version). Each fix has a correct-behaviour regression
> test, including a multi-day end-to-end test that drives the real
> daily pipeline through a contract fake of `yfinance` (exclusive `end`, delayed publication) and a
> pessimistic broker fake (asynchronous cancels, delayed order visibility). A full trading-session
> loop against a live OpenD gateway has still **not** been run.

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
> (see [§12](#12-statistical-integrity)). The split comes from one place — `research_context.open_research`
> (review N09-c) — which the CLI, chat, workflows, discovery, the API, the validation gate and the
> proposal all use; its context (dataset / universe / experiment / frozen boundary) is written into
> every GP / Optuna run record, and a search handed frozen-segment dates refuses to start.

**Factor lifecycle** — `GET /alphas/dashboard` · `/alphas/pending` · `/alphas/{id}/ic_history` ·
`/alphas/{id}/walk_forward` · `/alphas/{id}/decisions` · `PATCH /alphas/{id}/status` ·
`POST /alphas/{id}/validate` · `/alphas/{id}/approve` · `/alphas/{id}/reject` · `/alphas/{id}/retrain`

**Strategy configs** — `POST /strategies/propose` · `GET /strategies` · `/strategies/pending` ·
`/strategies/{id}` · `POST /strategies/{id}/approve[?activate]` · `/strategies/{id}/reject`

**Paper, data & diagnostics** — `GET /paper/{id}/pnl` · `/portfolio/diagnostics` ·
`/trading/status` · `/scheduler/status` · `/datasets` · `/datasets/{name}/health` · `/datasets/{name}/fundamentals` · `/regime` ·
`/report/query`

**Execution (moomoo paper)** — `GET /execution/status` · `POST /execution/reconcile` ·
`POST /execution/reconcile/accept` · `POST /execution/kill_switch/arm?action=engage|reset` →
`POST /execution/kill_switch/confirm` · `GET /execution/fidelity?start&end` ·
`POST /execution/preflight`

---

## 11. Frontend

React 19 + TypeScript + Vite + Zustand + ECharts, styled as an OS-style workspace.

- **Chat** — streaming agent chat with step-by-step `ThoughtBlock` reveal and a data-source badge
  (green for real data, amber **"⚠ synthetic (metrics invalid)"**).
- **Compiler** — DSL editor, run config, console output.
- **Dataset** — catalog + health, and a **Fundamentals** panel: field classes, coverage, each
  ticker's last filing date (the day after which its numbers became visible), stale flags and the
  fields a ticker cannot provide. It reads local filings only — opening it sends no external request.
- **AlphaDashboard** — paper equity curve, IC history, walk-forward charts, plus **ApprovalQueue**
  (pending candidates with approve/reject and the decision lineage; a candidate that uses
  fundamental fields lists them with the rule for when those numbers became knowable).
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
- **A real dataset's Test start is registered by a person** (decision D12). Nothing pins it as a
  side effect of the first call any more: until a date is registered
  (`backend/tools/research/register_research_boundary.py`, write-once, with actor and reason) research
  runs on a *provisional* split and formal verdicts — the validation gate and strategy proposals —
  refuse. `backend/tools/research/research_boundary_report.py` lists existing boundaries, earliest
  exposure, exposure counts, universes and experiments from the live database over a read-only,
  lock-free connection and suggests dates. Exposure counts are per dataset and overlapping window,
  so renaming the experiment does not refresh them.
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
- **No forward evidence yet.** Collecting evidence and being verified are two different strategy
  states: `paper_running` trades the simulated account and accumulates forward evidence without any
  evidence requirement; `verified_active` can only be reached through the pre-registered validation
  gate on that strategy's own forward evidence — there is no setting that bypasses it
  (`TR_ENFORCE_ACTIVE_GATE` no longer does anything). No strategy is verified today.
- **Risk attribution covers 5 price-volume styles + sectors, plus `size` and `value` only when the
  dataset carries fundamentals.** Without them (the default), every attribution payload carries
  `styles_missing: ["size","value"]` with the reason — a "66% factor risk" figure must then not be
  read as "the major styles are covered".
- **Fundamental factors are untested as strategies.** The fields are point-in-time correct and
  cross-checked against the broker's own P/E and P/B on the live universe, but no fundamental factor
  has been through the validation gate. A preview on the selection segments only (95 large caps,
  2021-01 to 2024-09, frozen test segment untouched) showed **no usable signal** in any classic
  value / profitability / earnings / size factor. `sue` only becomes visible when the 10-Q is filed —
  days to weeks after the earnings press release for many companies.
- **IC figures stored before 2026-10-13 are wrong.** The backtest report paired signal rows with
  return rows by position; with a warm-up window the signal was matched to a day *inside its own
  formation window* (a 5-day reversal reported IC −0.42 on a pure random walk). P&L and Sharpe were
  never affected. The computation is fixed; `ic_ir` on already-saved factor records was not recomputed.

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
| `ENABLE_FUNDAMENTALS` | `false` | attach the eight fundamental fields to US datasets (SEC EDGAR, filing-date aligned) |
| `EDGAR_USER_AGENT` | *(empty)* | `"Your Name you@example.com"` — the SEC requires every request to declare an identity. No default: the system will not invent one; without it only already-stored facts are used |
| `FUNDAMENTALS_DIR` / `FUNDAMENTALS_MAX_AGE_H` | `<PIT_STORE_DIR>/fundamentals` / `20` | fact store; how long before a ticker is re-fetched |
| `EXECUTION_MODE` | `off` | `moomoo_paper` = place real orders on the moomoo **paper** account (real money is refused in code) |
| `MOOMOO_SECURITY_FIRM` / `MOOMOO_TRD_ACC_ID` | `FUTUINC` / `0` | broker entity; `0` = first US paper account |
| `EXEC_LIMIT_BAND_BPS` / `EXEC_FLATTEN_BAND_BPS` | `50` / `500` | limit band for rebalancing and for flattening |
| `EXEC_MAX_DAILY_LOSS` / `EXEC_MAX_PRICE_DEVIATION` | `0.05` / `0.15` | daily-loss breaker, stale-price guard |
| `PAPER_AUM` | `1000000` | capital **allocated to the system** inside the broker account (a slice). Share counts, risk limits, the daily-loss breaker and the equity curve all use slice equity = `PAPER_AUM` + account P&L since the baseline; the rest of the account stays idle and the slice never borrows. Must not exceed the account's assets |
| `ENABLE_SCHEDULER` / `ENABLE_PAPER_TRADING` / `ENABLE_DISCOVERY` | `false` | nothing starts trading or mining by itself |
| `AUTONOMY_MODE` | `manual` | `manual` = human approves every promotion |
| `PAPER_AUM` | `1_000_000` | the single source of AUM for PM capacity and broker costs |
| `S_THREE_WAY_ENABLED` | `true` | force the three-way split on every research path; `false` cannot unfreeze a dataset whose Test boundary is already pinned (review N09-c) |
| `S_TEST_FREEZE_YEARS` / `S_HOLDOUT_BUDGET` | `2.0` / `1` | frozen holdout window and one-shot budget |
| `S_FITNESS_MODE` / `S_CV_FOLDS` / `S_EMBARGO_DAYS` | `purged_cv` / `5` / `20` | GP fitness discipline |
| `S_USE_CPCV` | `true` | PBO estimator (CSCV understates overfitting) |
| `FACTOR_GATE_MODE` | `leak` | factor-level bar: leak filter only; the strict gate is at strategy level |
| `TR_EXPERIMENT_MODE` | `true` | admit graded B/C proposals to `paper_running` (grades are shown as they are; a failed or unevaluated required gate is never grade A) |
| `RISK_MAX_GROSS` / `RISK_MAX_NAME_WEIGHT` / `RISK_MAX_SECTOR_WEIGHT` | `1.0` / `0.10` / `0.30` | portfolio risk limits |
| `RISK_TARGET_VOL_ANN` / `RISK_MAX_DRAWDOWN` / `RISK_HALT_ON_DRAWDOWN` | `0.0` / `0.20` / `false` | vol targeting (0 = off), drawdown breaker |
| `RISK_ATTR_LOOKBACK` | `252` | risk-model estimation window (diagnostic only) |
| `RESEARCH_HEALTH_FAIL_CLOSED` / `RESEARCH_MIN_HEALTH` | `true` / `0.7` | reject low-quality data instead of degrading |
| `CALENDAR_ALLOW_HEURISTIC` | `false` | never guess trading days |
| `ALLOW_INSECURE_BIND` | `false` | refuse to bind a non-loopback address (there is no auth layer) |

> **Trading-time hard gates are off by default; verification is not.** `pm_strategy_gate_block` and
> `risk_halt_on_drawdown` are `false` during the evidence-gathering phase: the daily strategy gate's
> verdict is recorded but does not stop paper trading. This is a deliberate choice, pinned by a test.
> Being *verified* (`verified_active`) always requires the pre-registered validation gate.

---

## 15. Testing & Engineering Discipline

```bash
cd backend  && pytest -q          # 5280 tests (2 skipped, ~53 min; last full run 2026-10-21). One skip is the live-SEC
                                  # contract test (needs EDGAR_USER_AGENT); the live-OpenD test also skips when OpenD is down
cd frontend && npm run test       # 121 passed (11 files)
cd frontend && npx tsc -b         # type check (what CI runs; `--noEmit` on the root tsconfig checks nothing)
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
| Point-in-time fundamentals (SEC EDGAR), sparse DSL field classes, `fund_rank`, fundamental discovery directions | ✅ Phase 10 (run against live EDGAR + moomoo on the 95-name universe; no fundamental factor validated yet) |
| Risk attribution + structured covariance | ✅ Phase R.2 (part) |
| moomoo paper execution: reconciliation/crash recovery, pre-trade gate, two-step kill switch, sim-vs-paper fidelity, live T3 providers, execution panel | ✅ Phase 12 + FE-12 (code; **not yet run against a live OpenD gateway**) |
| External review 2026-10-18 (F01–F14): health-gate hard checks, re-fetch on re-adjustment, frozen Test boundaries + overlap-metered holdout, selection-only research paths, trial counting at the search layer, executable strategy spec, archived-signal forward evidence per strategy version, atomic order claims + account lock, independent quotes, weight drift, content-hashed cost cache, factor-return min-variance, SUE/8-K alignment, single-transaction approvals | ✅ (see the roadmap's fix log; the strategy approved before 2026-10-18 must be re-proposed — D13) |
| Re-review 2026-10-19 (N01–N09, §5, §7.3): mid-batch kill switch, input identity on evidence, open-time archive deadline, health gate on the consumed panel, freeze-then-evaluate proposals, quote sessions, trial-run lifecycle, pre-trade-value costs + post-trade renormalisation (checked against a share + cash ledger), one research context, one strategy contract, spec v2, broker-account lock, formal-forward daily records | ✅ code (per-item status in `backend/docs/RE_REVIEW_FIX_TRACKER.md`; mutation review owed — D5) |

**Not built yet** — the honest list:

- **Phase 12 — live verification.** Everything is tested against a fake trading context whose call
  signatures and returned columns are checked mechanically against the installed moomoo SDK, but no
  order has been placed on a real paper account yet. Two behaviours can only be confirmed there:
  whether the paper account serves history-order queries, and whether orders placed after the close
  appear in the next session's order list.
- **Phase 10 remainder — decided 2026-10-22, not open.** Fundamental factors are not promoted this
  round (D1): no path can move a factor that uses a fundamental field to validated / paper / active,
  the validation gate only gives such factors a selection-segment diagnostic, and pre-registered
  diagnostics are counted as trials without touching the frozen Test. Multi-class share counts stay
  NaN (D7). Press-release-level SUE is not added (D8); whether those numbers require a paid source has
  **not been verified**.
- **Open** (per-item status in `backend/docs/DECISION_IMPLEMENTATION_TRACKER.md`): the Test start dates
  for the live database — a read-only report with suggested dates is waiting for your confirmation
  (D12); re-proposal of the active strategy (D13); an alert channel (D15, alerts are log-only). Not
  built: an intraday breaker watchdog, a backup-restore drill, and the full corporate-action ledger
  (unknown corporate actions are already quarantined, see above).
- **Phase 13/14 — red-team agent, fully autonomous mode, validation-period operations.**
- **R.2 remainder** — residual-based style neutralisation (changes signals, so it needs its own
  validation) and alpha-vs-risk-premium separation; `beta_neutral` closure is deferred until shorting
  is enabled.
- **R.4** — Ledoit-Wolf shrinkage, HRP, turnover in the optimisation objective.
- **Frontend** — FE-8/11/13/R panels (cost calibration, forward status, red-team,
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
