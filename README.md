# Quant Agent — Autonomous Alpha Research & Paper-Trading Platform

An end-to-end quantitative alpha research platform modelled after WorldQuant Brain. It combines a
typed-AST alpha DSL, realistic IS/OOS/Test backtesting, Genetic-Programming structural evolution,
Optuna parameter search, an XGBoost overfitting proxy, a LangChain chat agent (with a fully
deterministic no-LLM fallback), a factor **lifecycle state machine**, a **paper-trading engine**
with a daily scheduler, and a React/TypeScript OS-style UI with real-time SSE streaming.

> **Scope of this README:** it documents everything that exists in the codebase **today**
> (Phases 0–8 complete). Planned-but-not-yet-built capabilities (real-time incremental data,
> alt-data, Alpaca broker execution, autonomous market-observation-driven discovery, red-team
> multi-agent) are **not** described here — see the active roadmap
> [backend/ULTIMATE_GOAL_ROADMAP.md](backend/ULTIMATE_GOAL_ROADMAP.md). The completed Phase 6–8
> plan is archived at
> [backend/backend_retired_report/PAPER_TRADING_ROADMAP.md](backend/backend_retired_report/PAPER_TRADING_ROADMAP.md).

---

## Table of Contents

1. [What It Does Today](#1-what-it-does-today)
2. [Architecture](#2-architecture)
3. [Quick Start](#3-quick-start)
4. [Project Structure](#4-project-structure)
5. [Alpha DSL](#5-alpha-dsl)
6. [Backend Engines](#6-backend-engines)
7. [Data Engine & Datasets](#7-data-engine--datasets)
8. [Lifecycle, Monitoring & Paper Trading](#8-lifecycle-monitoring--paper-trading)
9. [API Reference](#9-api-reference)
10. [Frontend](#10-frontend)
11. [Database Schema](#11-database-schema)
12. [Reproducibility & Anti-Overfitting](#12-reproducibility--anti-overfitting)
13. [Configuration](#13-configuration)
14. [CLI Modes](#14-cli-modes)
15. [Testing](#15-testing)
16. [Project Status & Roadmap](#16-project-status--roadmap)
17. [License, Data & Disclaimer](#17-license-data--disclaimer)

---

## 1. What It Does Today

- **Alpha DSL** — a typed-AST factor language (Lark grammar) with ~50 operators (cross-sectional,
  time-series, group/sector, arithmetic), static validation (depth / window / look-ahead), and a
  vectorized NumPy/pandas executor.
- **Realistic backtesting** — immutable IS/OOS/Test three-way split with embargo, `prev_w·price_chg`
  PnL accounting, T+1 signal delay, a square-root market-impact cost model, ADV liquidity caps, and
  a water-filling L1 portfolio projection. Walk-Forward multi-fold validation and Deflated Sharpe.
- **Genetic Programming discovery** — `PopulationEvolver` evolves a typed-AST population with
  structural mutations, a diversity-filtered `AlphaPool`, fitness on IS+OOS Sharpe/turnover/drawdown,
  and Optuna fine-tuning of the pool-best.
- **XGBoost overfitting proxy** — `ProxyModel` / `AlphaEvaluator` score overfitting risk.
- **AI chat agent** — a LangChain tool-calling agent (7 quant tools) over GPT-4o-mini, with a
  **deterministic `FallbackOrchestrator`** that runs the whole seed→GP→critic→save pipeline with
  **zero LLM** when no API key is set. SSE streaming of every step.
- **Factor lifecycle** — a 7-state machine (CANDIDATE→VALIDATED→PAPER→ACTIVE→DECAYING→RETIRED/
  SUPERSEDED) with enforced transitions and a REST PATCH gate.
- **Monitoring & paper trading** — `AlphaMonitor` (realized IC, decay detection), a `PaperBroker`
  that reconciles day-by-day fills with the backtester to ~2.2e-16, a `PositionStore` (idempotent,
  crash-recoverable), and an APScheduler daily ingest + trading loop.
- **Reproducibility** — `RunManifest` ledger (dataset SHA-256 + git commit + seed + config), a
  bindable deterministic RNG, and WAL-hardened SQLite.
- **Frontend** — React 19 / TypeScript OS-style workspace: Chat, Compiler, Ledger, Dataset, and a
  Live dashboard with PnL / IC / walk-forward charts.

---

## 2. Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                        React 19 / TypeScript UI                           │
│  GlobalSidebar · ChatView · CompilerView · DatasetView · AlphaDashboard   │
│  LeftLedgerPane · SessionHistory · RightPane (PnL/IC charts)              │
└──────────────────────────────┬────────────────────────────────────────────┘
                    HTTP / SSE (Vite proxy → :8000)
┌──────────────────────────────▼────────────────────────────────────────────┐
│                            FastAPI (:8000)                                │
│  /api/chat[/stream] /api/chat/sessions   /api/workflow/generate|optimize  │
│  /api/agent/run /api/gp/evolve /api/backtest/* /api/alpha/*               │
│  /api/alphas/dashboard|status|ic_history|retrain|walk_forward             │
│  /api/paper/{id}/pnl /api/scheduler/status /api/datasets /api/regime      │
└──┬──────────┬───────────┬────────────┬────────────┬───────────┬───────────┘
   ▼          ▼           ▼            ▼            ▼           ▼
 alpha_    backtest_    gp_engine    ml_engine   agent/     tasks/
 engine    engine       Population   ProxyModel  QuantAgent  scheduler
 (DSL AST) Realistic    Evolver +    +Alpha      +Fallback   DailyIngest
 Parser/   BT + cost    AlphaPool +  Evaluator   Orchestr.   DailyTradingLoop
 Executor  IS/OOS/Test  Mutations                LangChain   AlphaMonitor
   │          │            │                                    │
   ▼          ▼            ▼                                     ▼
 portfolio_engine   data_engine (providers, PIT-lite,   execution/PaperBroker
 SignalProcessor    health, regime, sectors)            db/PositionStore
 PortfolioConstr.          │                                    │
        └─────────────── db/ AlphaStore · ChatStore · RunManifest (SQLite/WAL) ┘
```

| Mode | Trigger | Pipeline |
|---|---|---|
| **Chat** | user message | `QuantAgent.chat()` → LangChain tools (or Fallback) → DSL → backtest → reply |
| **Workflow A** | Generate | hypothesis → `GenerationWorkflow` → seeds → GP + Optuna → save (SSE) |
| **Workflow B** | Optimize | DSL → `OptimizationWorkflow` → GP + Optuna → save (SSE) |
| **Backtest** | Run | DSL → `RealisticBacktester` (IS+OOS/Test) → `AlphaEvaluator` → metrics |
| **Paper** | scheduler | `DailyIngest` (health gate) → `DailyTradingLoop` → `PaperBroker` → `AlphaMonitor` |

---

## 3. Quick Start

**One-command launcher (bash):**
```bash
./start.sh                 # starts backend (:8000) + frontend (:5173)
```

**Manual — backend:**
```bash
cd backend
python -m venv ../venv && source ../venv/Scripts/activate   # Windows: ../venv/Scripts/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --app-dir .                   # http://localhost:8000  (docs: /docs)
```

**Manual — frontend:**
```bash
cd frontend
npm install
npm run dev                                                 # http://localhost:5173 (proxies /api → :8000)
```

Optional: create `backend/.env` with `OPENAI_API_KEY=...` to enable the LLM agent. Without it, the
deterministic fallback path is used automatically.

---

## 4. Project Structure

```
Quant Agent/
├── start.sh                         # backend + frontend launcher
├── README.md                        # this file
├── docker/                          # container assets
├── scripts/                         # helper scripts
├── venv/                            # Python virtualenv
│
├── backend/
│   ├── requirements.txt / requirements.lock
│   ├── alphas.db                    # main SQLite ledger (auto-created, WAL)
│   ├── scheduler_jobs.db            # APScheduler jobstore (auto-created)
│   ├── ULTIMATE_GOAL_ROADMAP.md     # active long-horizon plan (Phase 9–14 + R)
│   ├── OPERATIONS.md                # validation-period operating rules (Phase 8.3)
│   ├── RESEARCH_OPERATING_MODEL.md  # long-term principles (living)
│   ├── backend_retired_report/      # archived reports (frozen, not updated):
│   │   ├── PAPER_TRADING_ROADMAP.md #   Phase 6–8 plan (completed 2026-08-19)
│   │   ├── DEV_ROADMAP.md           #   Phase 0–5 plan
│   │   └── AUDIT_REPORT.md / STAGE1_COMPLETION_REPORT.md
│   ├── tests/                       # 16 pytest modules + conftest
│   └── app/
│       ├── main.py                  # FastAPI app + lifespan + CLI entry
│       ├── config.py                # pydantic-settings (env vars)
│       ├── dependencies.py          # DI singletons
│       ├── api/
│       │   ├── router.py            # core / backtest / workflow / lifecycle / paper endpoints
│       │   └── chat_router.py       # /api/chat + session CRUD
│       ├── core/
│       │   ├── alpha_engine/        # DSL: parser, ast, validator, executor, operators,
│       │   │                        #   fast_ops, generator, signal_processor,
│       │   │                        #   financial_interpreter/diagnostics
│       │   ├── backtest_engine/     # realistic_backtester, backtest_engine, transaction_cost,
│       │   │                        #   performance_analyzer, portfolio_constructor, risk_report,
│       │   │                        #   multi_dataset_backtester, alpha_combiner, visualizer
│       │   ├── portfolio_engine/    # signal_processor, portfolio_constructor, realistic_backtester
│       │   ├── gp_engine/           # population_evolver, alpha_pool, mutations, fitness,
│       │   │                        #   gp_engine, evaluation_utils, _rng (deterministic RNG)
│       │   ├── ml_engine/           # proxy_model (XGBoost), alpha_evaluator, alpha_optimizer
│       │   ├── optimization_engine/ # alpha_optimizer (Optuna), alpha_evaluator, data_partitioner
│       │   ├── data_engine/         # data_manager, providers, dataset_registry, feature_store,
│       │   │                        #   preprocessor, schema, panel_factory, health_report,
│       │   │                        #   regime_detector, sector_mapper, data_partitioner
│       │   ├── monitor/             # alpha_monitor (realized IC, decay)
│       │   ├── execution/           # paper_broker (day-by-day stateful fills)
│       │   ├── workflows/           # alpha_workflows (Generation/Optimization + seed diversity)
│       │   └── utils/               # fast_ops
│       ├── agent/                   # quant_agent, alpha_agent, _agent, _lc_agent, _tools,
│       │                            #   _fallback, _critic, _prompts, _memory, _chat_history
│       ├── db/                      # alpha_store, chat_store, position_store, run_manifest,
│       │                            #   alpha_lifecycle (state machine), _sqlite_utils (WAL)
│       └── tasks/                   # scheduler, daily_ingest, daily_trading_loop, reasoning_log
│
└── frontend/
    └── src/
        ├── App.tsx / main.tsx
        ├── api/client.ts            # typed fetch + SSE client
        ├── store/workspaceStore.ts  # Zustand store
        ├── hooks/useQuantWorkspace.ts
        ├── types/index.ts
        └── components/
            ├── layout/              # GlobalSidebar, WorkspaceLayout, LeftLedgerPane,
            │                        #   RightPane, SessionHistoryPanel
            ├── chat/                # ChatView, ChatMessage, ThoughtBlock
            ├── compiler/            # CompilerView, ConfigModal, ConsoleOutput
            ├── dataset/             # DatasetView
            ├── dashboard/           # AlphaDashboard (Live: PnL/IC/walk-forward)
            └── analysis/            # MetricsGrid, OverfitBadge, PnLChart, WalkForwardChart,
                                     #   RegimeBadge, AlphaPoolPanel
```

---

## 5. Alpha DSL

A typed expression language over a daily panel `(time × assets)`. Parsed by a Lark grammar into a
typed AST (`typed_nodes.py` / `ast.py`), statically validated, then executed vectorized
(`executor.py` / `dsl_executor.py` / `fast_ops.py`).

**Operator families** (registry in [ast.py](backend/app/core/alpha_engine/ast.py)):

- **Arithmetic / unary:** `add sub mul div pow signed_power neg abs log sqrt sign max2 min2`
- **Logical / conditional:** `logical_and logical_or logical_not if_else where trade_when`
- **Time-series:** `ts_mean ts_std ts_var ts_sum ts_delta ts_delay ts_max ts_min ts_rank
  ts_argmax ts_argmin ts_zscore ts_skew ts_kurt ts_entropy ts_decay_linear ts_corr ts_cov`
- **Cross-sectional:** `rank zscore scale normalize winsorize demean ind_neutralize`
- **Group / sector:** `group_rank group_zscore group_mean group_neutralize`
- **Inputs:** `data` (fields: `open high low close volume vwap returns`, + derived `sector`)

Example: `rank(ts_delta(log(close), 5))` — 5-day log-price momentum, cross-sectionally ranked.

**Validation** ([validator.py](backend/app/core/alpha_engine/validator.py)) enforces max AST depth,
max time-series window, and rejects look-ahead constructs. Signals pass through `SignalProcessor`
(T+1 delay, truncation, decay, neutralization) before portfolio construction.

---

## 6. Backend Engines

- **alpha_engine** — DSL parse/validate/execute; `generator.py` produces random valid ASTs;
  `financial_interpreter.py` / `financial_diagnostics.py` label a factor's family and diagnose design.
- **backtest_engine / portfolio_engine** — `RealisticBacktester` runs IS+OOS (or held-out Test) with
  `prev_w·price_chg` accounting; `TransactionCostEngine` applies the square-root impact law
  (`slip_bps = spread/2 + impact_coef·vol·1e4·√participation`) + fixed/borrow costs;
  `LiquidityConstraint` caps by 20-day ADV; `project_to_capped_l1` water-fills the L1 cap;
  `PerformanceAnalyzer` computes Sharpe, IC/IC-IR, turnover, drawdown, Walk-Forward, Deflated Sharpe.
- **gp_engine** — `PopulationEvolver` (typed-AST evolution), `AlphaPool` (diversity-filtered
  hall-of-fame), `mutations.py` (point/hoist/param/subtree), `fitness.py`, and `_rng.py` (a bindable
  shared RNG for deterministic runs).
- **ml_engine / optimization_engine** — `ProxyModel` (XGBoost overfitting proxy), `AlphaEvaluator`
  (overfitting scoring), `AlphaOptimizer` (Optuna parameter search over an IS-only objective).
- **workflows** — `GenerationWorkflow` (hypothesis→seeds→GP), `OptimizationWorkflow` (DSL→GP),
  and `_generate_diverse_seeds` (4-layer seed diversity: LLM → keyword templates → AST mutation →
  random).
- **agent** — `QuantAgent` picks the LangChain tool-calling path (7 tools: generate/interpret/
  gp_optimize/backtest/mutate/optuna/save) when an LLM is configured, else the deterministic
  `FallbackOrchestrator`; `OverfitCritic` drives targeted correction rounds.

---

## 7. Data Engine & Datasets

- **Providers** — `YahooFinanceProvider` (yfinance), `CcxtBinanceProvider` (crypto), plus
  `AkshareProvider`, `AlphaVantageProvider`, `LocalParquetProvider`. All fetch **daily OHLCV
  historical batch** over `[start, end]` — 7 standard fields
  `open high low close volume vwap returns`.
- **Pipeline** — `DataManager.get_panel()` does cache → provider fetch → schema enforce → master
  reindex → universe filter → preprocess (adjust/ffill/synthetic fields) → feature-store write →
  health check. `ParquetFeatureStore` caches panels; `DataHealthChecker` scores data quality (the
  daily ingest rejects data below a health threshold rather than silently downgrading).
- **Sectors & regime** — `sector_mapper.py` attaches GICS L1 codes (used by `group_*` operators);
  `regime_detector.py` classifies market regime (surfaced at `/api/regime`).
- **Active datasets** ([dataset_registry.py](backend/app/core/data_engine/dataset_registry.py)):
  - **US equities:** `us_broad_large` (~90 tickers, all 11 GICS sectors), `us_tech_large`,
    `us_financials`, `us_healthcare`, `us_energy`
  - **Crypto:** `crypto_major`, `crypto_alt`
  - **Shelved** (recorded, not loadable): China A-share + HK specs, popped from the active registry.

---

## 8. Lifecycle, Monitoring & Paper Trading

- **Lifecycle state machine** ([alpha_lifecycle.py](backend/app/db/alpha_lifecycle.py)) —
  CANDIDATE → VALIDATED → PAPER → ACTIVE → DECAYING → {RETIRED | SUPERSEDED}. `validate_transition`
  enforces legal edges; illegal → HTTP 409. (Note: current creation paths save new alphas directly
  as `active`; the graded gates are enforced for manual PATCH — closing this gap is Phase 9.)
- **Monitoring** — `AlphaMonitor` computes per-day realized IC, detects decay, and powers
  `/api/alphas/dashboard` and `/api/alphas/{id}/ic_history`.
- **Paper trading** — `PaperBroker.step()` fills at close, reusing the exact `TransactionCostEngine`
  + `LiquidityConstraint` from the backtester (reconciles to ~2.2e-16). `PositionStore` persists
  idempotent `paper_positions` / `paper_fills` / `paper_daily_pnl` with `state_before()` for
  deterministic crash-recovery re-runs.
- **Scheduler** ([scheduler.py](backend/app/tasks/scheduler.py)) — an APScheduler
  `BackgroundScheduler` with a SQLAlchemy jobstore. `daily_monitor_job` (decay check) always
  registers; `daily_trading_job` (`DailyIngest` health-gate → `DailyTradingLoop`) registers only when
  `ENABLE_PAPER_TRADING=true`. Enabled via `ENABLE_SCHEDULER=true` at server startup.

> The daily loop currently runs a **historical replay** over a re-pulled batch (incrementally
> resumed per-alpha), not a live tick/bar feed. True incremental/live ingestion is planned — see the
> roadmaps.

---

## 9. API Reference

Base URL `http://localhost:8000/api` (interactive docs at `/docs`).

**Agent & Chat**
| Method | Path | Purpose |
|---|---|---|
| POST | `/agent/run` | run the quant agent on a hypothesis/DSL |
| POST | `/chat` · `/chat/stream` | chat turn (JSON / SSE) |
| POST/GET/PATCH/DELETE | `/chat/sessions[...]` | session CRUD + history |

**Discovery & Workflows**
| Method | Path | Purpose |
|---|---|---|
| POST | `/gp/evolve` | run GP evolution |
| POST | `/workflow/generate` · `/workflow/generate/stream` | Workflow A (hypothesis → alpha) |
| POST | `/workflow/optimize` · `/workflow/optimize/stream` | Workflow B (DSL → optimized alpha) |

**Backtest & Evaluate**
| Method | Path | Purpose |
|---|---|---|
| POST | `/backtest/run` | IS+OOS backtest |
| POST | `/backtest/realistic` | realistic backtest (costs + liquidity) |
| POST | `/backtest/multi` | multi-dataset backtest |
| POST | `/backtest/walk_forward` | walk-forward multi-fold |
| POST | `/alpha/simulate` · `/alpha/optimize` · `/alpha/save` | eval / Optuna / persist |

**Lifecycle, Paper & Data**
| Method | Path | Purpose |
|---|---|---|
| GET | `/alphas/dashboard` | lifecycle dashboard (+ `allowed_next`) |
| GET | `/alphas/{id}/ic_history` | realized IC history |
| PATCH | `/alphas/{id}/status` | lifecycle transition (human gate) |
| POST | `/alphas/{id}/retrain` | retrain / re-optimize |
| GET | `/alphas/{id}/walk_forward` | walk-forward for a saved alpha |
| GET | `/paper/{id}/pnl` | paper equity/PnL curve |
| GET | `/scheduler/status` | scheduler + job state |
| GET | `/datasets` · `/datasets/{name}/health` | dataset catalog + health |
| GET | `/regime` | current market regime |
| GET | `/report/query` | saved alpha report query |

---

## 10. Frontend

React 19 + TypeScript + Vite + Zustand, styled as an OS-style workspace.

- **Layout** — `GlobalSidebar` (Chat / Compiler / Ledger / Data), `WorkspaceLayout`,
  `LeftLedgerPane` (alpha ledger + sessions), `RightPane` (charts), `SessionHistoryPanel`.
- **Views** — `ChatView` (streaming agent chat with `ThoughtBlock` step reveal), `CompilerView`
  (DSL editor + `ConfigModal` + `ConsoleOutput`), `DatasetView`, `AlphaDashboard` (Live: paper
  equity curve + IC + walk-forward via ECharts).
- **Analysis widgets** — `MetricsGrid`, `OverfitBadge`, `PnLChart`, `WalkForwardChart`,
  `RegimeBadge`, `AlphaPoolPanel`.
- **API layer** — `api/client.ts` (typed fetch + SSE); Vite proxies `/api` → `:8000`.

---

## 11. Database Schema

SQLite via SQLAlchemy, hardened with WAL + `busy_timeout` ([_sqlite_utils.py](backend/app/db/_sqlite_utils.py)).
Two physical files, both configurable via env:

- **`backend/alphas.db`** (`DATABASE_URL`, default `sqlite:///./alphas.db`)
  - `alpha_records` — factor ledger: `dsl, hypothesis, ann_return, sharpe, max_drawdown, ic_ir,
    ann_turnover, status, reasoning(JSON)` (append-only; managed by `AlphaStore`)
  - `alpha_ic_history` — per-day realized IC (`AlphaMonitor`)
  - `paper_positions` / `paper_fills` / `paper_daily_pnl` — paper trading (`PositionStore`, idempotent)
  - `run_manifest` — reproducibility ledger (`RunManifestStore`: dataset SHA-256 + git commit + seed + config)
  - `chat_sessions` / `chat_messages` — agent memory (`ChatStore`)
- **`backend/scheduler_jobs.db`** (`SCHEDULER_DB_URL`) — APScheduler jobstore.

Factor `status` transitions are constrained by the lifecycle state machine.

---

## 12. Reproducibility & Anti-Overfitting

- **Three-way split** — immutable IS/OOS/Test partitioning with **embargo**; the Test set is touched
  only for final held-out validation.
- **Walk-Forward + Deflated Sharpe** — multi-fold out-of-sample validation and Bailey/López de Prado
  DSR to discount multiple-testing selection bias.
- **Realistic costs** — square-root market-impact slippage, spread, fixed fees, short borrow, and ADV
  liquidity caps, so in-sample edge must survive frictions.
- **Overfitting proxy** — XGBoost `ProxyModel` + `AlphaEvaluator` scoring; `OverfitCritic` triggers
  targeted structural corrections during discovery.
- **Determinism** — a bindable shared RNG (`gp_engine/_rng.py`) removes global-random order
  dependence; `RunManifest` records dataset hash + git commit + seed + config so any run can be
  verified/replayed (`verify_dataset` re-hashes before replay).

---

## 13. Configuration

Environment variables (or `backend/.env`), read by [config.py](backend/app/config.py):

| Var | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./alphas.db` | main ledger |
| `OPENAI_API_KEY` | *(empty)* | enables the LLM agent; empty → deterministic fallback |
| `DEFAULT_DATASET` / `DEFAULT_START` / `DEFAULT_END` | `us_tech_large` / `2020-01-01` / `2024-01-01` | default research window |
| `DEFAULT_POP_SIZE` / `DEFAULT_N_GEN` | `20` / `5` | GP defaults |
| `INITIAL_CAPITAL` | `1_000_000` | backtest capital |
| `ENABLE_SCHEDULER` | `false` | start APScheduler on server startup |
| `SCHEDULER_DB_URL` / `SCHEDULER_TIMEZONE` | `sqlite:///scheduler_jobs.db` / `UTC` | scheduler jobstore |
| `ENABLE_PAPER_TRADING` | `false` | register daily ingest + trading loop jobs |
| `PAPER_DATASET` / `PAPER_START` / `PAPER_END` | `us_tech_large` / … | paper pipeline window |

---

## 14. CLI Modes

`python app/main.py --mode <mode>` (run from `backend/`):

| Mode | Example |
|---|---|
| `agent` | `--mode agent --hypothesis "Post-earnings drift"` |
| `gp` | `--mode gp --generations 5 --pop-size 20 --dataset crypto_major` |
| `backtest` | `--mode backtest --dsl "rank(ts_delta(log(close),5))"` |
| `realistic` | `--mode realistic --dsl "..." --walk-forward --wf-splits 5` |
| `report` | `--mode report --alpha-id 1` |

Common flags: `--dataset --start --end --optuna-trials --oos-ratio --embargo-days --delay
--decay-window --use-synthetic` (synthetic random-walk data is a deprecated offline-only fallback).

---

## 15. Testing

- **Backend** — 14 pytest modules in [backend/tests/](backend/tests/) (DSL, backtest, data engine,
  discovery, and per-phase suites `test_phase1..7`, reproducibility, supplementary fixes).
  Current baseline: **432 passed, 3 skipped**.
  ```bash
  cd backend && pytest -q
  ```
- **Frontend** — Vitest unit + integration tests under `frontend/src/__tests__/`, plus `tsc` type check.
  ```bash
  cd frontend && npm run test && npm run build
  ```

---

## 16. Project Status & Roadmap

**Complete:** Phases 0–8 (DSL, data, backtest, GP, ML proxy, agent, lifecycle, monitoring, paper
broker + daily loop, reproducibility, US+crypto datasets, **PIT point-in-time store +
cost-calibration loop + operations rules**). The Phase 6–8 plan is archived (completed) at
[backend/backend_retired_report/PAPER_TRADING_ROADMAP.md](backend/backend_retired_report/PAPER_TRADING_ROADMAP.md);
validation-period operating rules live in [backend/OPERATIONS.md](backend/OPERATIONS.md).

**Planned (not yet built)** — documented, not described in this README:

- **Phase 9–14** (long horizon) — autonomous market-observation-driven factor discovery, graded
  validation gate + human approve/reject, alt-data (fundamentals/earnings) with PIT + DSL sparse
  fields, forward incremental data, **Alpaca paper-broker execution** with deterministic trading
  workflow + risk gates, paper-vs-real fidelity modeling, and a conditional red-team multi-agent:
  [backend/ULTIMATE_GOAL_ROADMAP.md](backend/ULTIMATE_GOAL_ROADMAP.md)
- **Governing principles** — LLM never in the trading loop, state transitions never as LLM tools,
  alt-data PIT ordering, and when multi-agent is justified:
  [backend/RESEARCH_OPERATING_MODEL.md](backend/RESEARCH_OPERATING_MODEL.md)

---

## 17. License, Data & Disclaimer

> Not legal advice. Consult a professional before commercial use.

### Code license
This **framework/code** is licensed under the **Apache License 2.0** — permissive
(free to use, modify, redistribute) with an explicit patent grant. Add a top-level `LICENSE` file
with the Apache-2.0 text to make it binding. (MIT is a simpler permissive alternative; AGPL-3.0 if
you specifically want to prevent others from running a closed-source hosted service off it.)

The **edge is separate from the code** and is **never** part of this repository: actual profitable
factors/alphas, tuned parameters/configs, live position sizing, API keys, and broker credentials
stay private (see `.gitignore`). Open-sourcing the framework costs nothing; open-sourcing a working
strategy would crowd it out of existence. If a real edge or a commercial product emerges later,
move to an **open-core** model (framework open, strategies + "pro" service proprietary) or a
**source-available** license (e.g., BSL) — narrowing from permissive is easy, the reverse is not.

### Data — bring your own
This repo ships **no market data**. Data providers carry their own terms:
- **yfinance / Yahoo** data is for personal research; Yahoo's ToS restricts commercial use and
  redistribution — **do not** bundle or redistribute it, and do not build a commercial product on it.
- Paid/academic sources (Sharadar, CRSP, Compustat, …) have strict redistribution limits — **never**
  commit or ship their data with the code.

Point-in-time / proprietary data you accumulate (`pit_store/`) and all databases (`*.db`) are
gitignored and stay local. Users must supply their own data and API keys via `backend/.env`.

### Disclaimer
For **research and educational purposes only**. Nothing here is financial or investment advice.
Provided **"as is", without warranty of any kind**; backtest and paper-trading results are not
indicative of future performance and, per the roadmap, are currently research-grade estimates,
not validated P&L. You are solely responsible for any use, including any real capital at risk.

### Dependencies
Third-party libraries retain their own licenses (FastAPI, NumPy, pandas, LangChain, React, etc. —
predominantly MIT / Apache-2.0 / BSD). Verify compatibility before redistribution.
