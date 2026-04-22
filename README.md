# Quant Agent — Production-Grade Alpha Research Platform

An end-to-end autonomous quantitative alpha research platform modelled after WorldQuant Brain. It combines a high-performance typed-AST DSL engine, realistic IS/OOS backtesting, Genetic Programming (GP) structural evolution, Optuna parameter optimization, an AI-driven LangChain chat agent, and a React/TypeScript OS-style UI with real-time SSE streaming.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Complete File Structure](#2-complete-file-structure)
3. [Quick Start](#3-quick-start)
4. [Installation Details](#4-installation-details)
5. [Alpha DSL Language](#5-alpha-dsl-language)
6. [Backend Engines](#6-backend-engines)
   - [6.1 Alpha DSL Engine](#61-alpha-dsl-engine)
   - [6.2 Data Engine](#62-data-engine)
   - [6.3 Backtest Engine](#63-backtest-engine)
   - [6.4 GP Engine](#64-gp-engine)
   - [6.5 ML Engine (Optuna + Evaluator)](#65-ml-engine-optuna--evaluator)
   - [6.6 Workflow Pipelines](#66-workflow-pipelines)
   - [6.7 AI Agent (LangChain)](#67-ai-agent-langchain)
   - [6.8 Database Layer](#68-database-layer)
7. [API Reference](#7-api-reference)
8. [Frontend Architecture](#8-frontend-architecture)
   - [8.1 State Management](#81-state-management)
   - [8.2 Components](#82-components)
   - [8.3 SSE Streaming](#83-sse-streaming)
9. [Anti-Overfitting Design](#9-anti-overfitting-design)
10. [Key Metrics & Formulas](#10-key-metrics--formulas)
11. [Configuration](#11-configuration)
12. [CLI Modes](#12-cli-modes)
13. [Testing](#13-testing)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         React 19 / TypeScript UI                            │
│                                                                             │
│  ┌──────────────┐  ┌────────────────────┐  ┌──────────────────────────┐   │
│  │ GlobalSidebar│  │  SessionHistory /  │  │  ChatView (CHAT mode)    │   │
│  │  (icon tbar) │  │  LeftLedgerPane    │  │  CompilerView + Console  │   │
│  │              │  │  (alpha ledger /   │  │  (COMPILER mode)         │   │
│  │              │  │   chat sessions)   │  │  RightPane (PnL chart)   │   │
│  └──────────────┘  └────────────────────┘  └──────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────────────────┘
             HTTP / SSE (Vite proxy  →  :8000)
┌──────────────────────────────▼──────────────────────────────────────────────┐
│                           FastAPI  (:8000)                                  │
│                                                                             │
│  /api/chat          /api/workflow/generate[/stream]                        │
│  /api/chat/sessions /api/workflow/optimize[/stream]                        │
│  /alpha/simulate    /alpha/optimize   /alpha/save                          │
│  /backtest/run      /backtest/realistic                                    │
│  /report/query      /gp/evolve        /agent/run                          │
└──┬────────────────┬──────────────────┬────────────────┬─────────────────────┘
   │                │                  │                │
   ▼                ▼                  ▼                ▼
Alpha DSL       Backtest           GP Engine        LangChain Agent
Engine (AST)    Engine             (Population      (GPT-4o / Fallback)
Parser +        RealisticBT        Evolver +             │
Executor +      IS+OOS split       DEAP +           ConversationMemory
Validator +     RiskReport         AlphaPool +      QuantTools (5 tools)
fast_ops        Perf Analyzer      Mutations)       FallbackOrchestrator
   │                │                  │
   ▼                ▼                  ▼
Signal          DataPartitioner    ML Engine
Processor       (immutable         AlphaOptimizer
(truncation     IS/OOS split)      (Optuna) +
 decay                             AlphaEvaluator
 neutral.                          (overfitting
 delay)                            scoring)
                                       │
                              Workflow Pipelines
                           GenerationWorkflow (A)
                           OptimizationWorkflow (B)
                                       │
                               AlphaStore (SQLite)
                               ChatStore  (SQLite)
```

### Data Flow Summary

| Mode | Trigger | Pipeline |
|---|---|---|
| **Chat** | User message | `QuantAgent.chat()` → LangChain tools → DSL → Backtest → Reply |
| **Backtest** | Run button | DSL → `RealisticBacktester` (IS+OOS) → `AlphaEvaluator` → Metrics |
| **Workflow A** | Generate | Hypothesis → `GenerationWorkflow` → GP+Optuna → SSE stream |
| **Workflow B** | Optimize | DSL → `OptimizationWorkflow` → GP+Optuna → SSE stream |

---

## 2. Complete File Structure

```
Quant Agent/
├── start.sh                              # One-command launcher (bash)
├── README.md
│
├── backend/
│   ├── requirements.txt                  # Python dependencies
│   ├── alphas.db                         # SQLite alpha ledger (auto-created)
│   └── app/
│       ├── main.py                       # FastAPI app + CORS + CLI entry point
│       ├── config.py                     # pydantic-settings (env vars)
│       ├── dependencies.py               # DI singletons (AlphaStore, ChatStore)
│       │
│       ├── api/
│       │   ├── router.py                 # Core + Phase2 + Workflow endpoints
│       │   └── chat_router.py            # /api/chat  /api/chat/sessions/*
│       │
│       ├── core/
│       │   ├── alpha_engine/             # DSL parsing, validation, execution
│       │   │   ├── ast.py                # Node dataclass, operator registry
│       │   │   ├── parser.py             # Lark grammar → typed AST Node
│       │   │   ├── validator.py          # Static: depth / window / lookahead
│       │   │   ├── dsl_executor.py       # Memoized AST → pd.DataFrame signal
│       │   │   ├── signal_processor.py   # SimulationConfig + processing pipeline
│       │   │   ├── typed_nodes.py        # Node, Cache, Dataset types
│       │   │   ├── fast_ops.py           # Bottleneck/Numba accelerated ops
│       │   │   ├── operators.py          # Operator dispatch table
│       │   │   └── generator.py          # Random DSL generation templates
│       │   │
│       │   ├── backtest_engine/
│       │   │   ├── realistic_backtester.py   # IS+OOS dual-leg backtest
│       │   │   ├── backtest_engine.py        # Pure portfolio simulation
│       │   │   ├── risk_report.py            # Sharpe/IC/Turnover/Drawdown report
│       │   │   ├── performance_analyzer.py   # PnL, equity curve, drawdown series
│       │   │   ├── portfolio_constructor.py  # SignalWeightedPortfolio + Neutral.
│       │   │   ├── transaction_cost.py       # Slippage (sqrt-law) + commissions
│       │   │   └── visualizer.py             # ECharts serialization
│       │   │
│       │   ├── data_engine/
│       │   │   ├── data_partitioner.py       # Immutable IS/OOS physical split
│       │   │   ├── data_manager.py           # OHLCV panel construction + cache
│       │   │   ├── yahoo_provider.py         # yfinance integration
│       │   │   ├── alpha_vantage_provider.py # Alternative data source
│       │   │   ├── local_parquet_provider.py # Parquet cache
│       │   │   ├── preprocessor.py           # Cleaning + alignment
│       │   │   ├── schema.py                 # Data validation schemas
│       │   │   ├── feature_store.py          # Feature matrix storage
│       │   │   ├── panel_factory.py          # Multi-asset panel builder
│       │   │   ├── dataset_loader.py         # Unified loader interface
│       │   │   ├── health_report.py          # Data quality checks
│       │   │   └── base.py                   # Abstract provider base class
│       │   │
│       │   ├── gp_engine/
│       │   │   ├── gp_engine.py              # AlphaEvolver (DEAP-based)
│       │   │   ├── population_evolver.py     # PopulationEvolver (true GP core)
│       │   │   ├── mutations.py              # AST-level: point/hoist/param/crossover
│       │   │   ├── fitness.py                # compute_fitness() + mutation_weights()
│       │   │   └── alpha_pool.py             # AlphaPool diversity filter (corr<0.9)
│       │   │
│       │   ├── ml_engine/
│       │   │   ├── alpha_optimizer.py        # AlphaOptimizer (Optuna, IS-only)
│       │   │   ├── alpha_evaluator.py        # AlphaEvaluator (overfitting score)
│       │   │   └── proxy_model.py            # XGBoost/LightGBM proxy scorer
│       │   │
│       │   ├── workflows/
│       │   │   └── alpha_workflows.py        # GenerationWorkflow + OptimizationWorkflow
│       │   │
│       │   └── utils/
│       │       └── fast_ops.py               # ts_mean / ts_std / ts_rank aliases
│       │
│       ├── db/
│       │   ├── alpha_store.py                # AlphaRecord ORM + AlphaStore CRUD
│       │   └── chat_store.py                 # ChatMessage ORM + ChatStore CRUD
│       │
│       ├── agent/
│       │   ├── quant_agent.py                # Public entry point (re-exports)
│       │   ├── _agent.py                     # QuantAgent main class (LangChain)
│       │   ├── _chat_history.py              # SQLAlchemyChatMessageHistory
│       │   ├── _memory.py                    # ConversationMemory (LRU, 50 sessions)
│       │   ├── _tools.py                     # QuantTools (5 LangChain tools)
│       │   ├── _critic.py                    # OverfitCritic gate
│       │   ├── _prompts.py                   # System prompt + message templates
│       │   ├── _lc_agent.py                  # RunnableWithMessageHistory builder
│       │   ├── _fallback.py                  # FallbackOrchestrator (no-LLM mode)
│       │   ├── _constants.py                 # Thresholds, DSL keyword maps
│       │   ├── _helpers.py                   # _extract_balanced, _safe_json_loads
│       │   ├── _data_utils.py                # Synthetic data gen + backtest core
│       │   └── alpha_agent.py                # Legacy AlphaAgent (Workflow A/B seeds)
│       │
│       └── tasks/
│           └── reasoning_log.py              # ReasoningLog dataclass
│
├── tests/
│   ├── test_phase1_upgrade.py
│   ├── test_phase2.py
│   ├── test_phase3.py
│   ├── test_alpha_discovery.py
│   ├── test_backtest_engine.py
│   ├── test_data_engine_smoke.py
│   └── test_dsl_engine.py
│
└── frontend/
    ├── package.json
    ├── vite.config.ts                    # Vite + Tailwind + /api proxy → :8000
    ├── tsconfig.json
    ├── index.html
    └── src/
        ├── main.tsx                      # React 19 bootstrap
        ├── App.tsx                       # Root: <WorkspaceLayout />
        ├── index.css                     # Tailwind v4 import
        │
        ├── types/
        │   └── index.ts                  # All domain types (see §8)
        │
        ├── api/
        │   └── client.ts                 # Axios wrappers + SSE stream functions
        │
        ├── store/
        │   └── workspaceStore.ts         # Zustand store (WorkspaceState)
        │
        ├── hooks/
        │   └── useQuantWorkspace.ts      # sendChat / runBacktest / runOptimize
        │
        └── components/
            ├── ErrorBoundary.tsx
            ├── layout/
            │   ├── WorkspaceLayout.tsx   # 3-pane resizable root
            │   ├── GlobalSidebar.tsx     # 64px icon toolbar
            │   ├── LeftLedgerPane.tsx    # Alpha ledger (COMPILER mode)
            │   ├── SessionHistoryPanel.tsx # Chat sessions (CHAT mode)
            │   └── RightPane.tsx         # PnL chart + metrics grid
            │
            ├── chat/
            │   ├── ChatView.tsx          # Scrollable feed + input
            │   ├── ChatMessage.tsx       # Bubble + streaming cursor + tag colors
            │   └── ThoughtBlock.tsx      # Agent reasoning block
            │
            ├── compiler/
            │   ├── CompilerView.tsx      # Monaco editor + tab bar + toolbar
            │   ├── ConfigModal.tsx       # SimulationConfig sliders
            │   └── ConsoleOutput.tsx     # Bottom log console (tagged coloring)
            │
            └── analysis/
                ├── PnLChart.tsx          # ECharts IS/OOS with markArea + split line
                ├── MetricsGrid.tsx       # IS vs OOS metrics comparison table
                └── OverfitBadge.tsx      # Overfitting indicator (pulsing)
```

---

## 3. Quick Start

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| Node.js | 18+ |
| bash | Git Bash / WSL on Windows |

### One-command launch

```bash
bash start.sh
```

The script will:
1. Create `backend/.venv` if missing and install Python dependencies
2. Install `frontend/node_modules` if missing
3. Start the FastAPI backend on **:8000** in the background
4. Start the Vite frontend on **:5173** in the foreground

| Service | URL |
|---|---|
| Frontend | http://localhost:5173 |
| Backend API | http://127.0.0.1:8000 |
| Swagger docs | http://127.0.0.1:8000/docs |
| ReDoc | http://127.0.0.1:8000/redoc |

### Optional: set OpenAI API key

```bash
echo "OPENAI_API_KEY=sk-..." > backend/.env
```

Without it, the system uses `FallbackOrchestrator` (rule-based DSL generation). All backtest and GP optimization features remain fully functional.

---

## 4. Installation Details

### Backend (manual)

```bash
cd backend
python -m venv .venv

# Windows (Git Bash / PowerShell)
source .venv/Scripts/activate       # Git Bash
# .venv\Scripts\Activate.ps1        # PowerShell

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend (manual)

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
```

### Python Dependencies (key packages)

```
# Server
fastapi>=0.111
uvicorn[standard]
pydantic>=2
pydantic-settings

# Data & Math
numpy pandas bottleneck scipy
yfinance pyarrow fastparquet

# DSL
lark

# ML / Optimization
optuna xgboost lightgbm

# GP
deap

# Database
sqlalchemy

# LLM / Agent
langchain langchain-openai langchain-community langchain-core

# Visualization
plotly
```

### Node Dependencies (key packages)

```
react@19 react-dom
@vitejs/plugin-react
tailwindcss @tailwindcss/vite
zustand
axios
echarts echarts-for-react
@monaco-editor/react
lucide-react
react-resizable-panels
```

---

## 5. Alpha DSL Language

The DSL is a typed expression language for constructing cross-sectional alpha signals. Every expression evaluates to a 2-D `pd.DataFrame` of shape `(T × N)` — one row per trading day, one column per asset.

### Data Fields (leaf nodes)

| Field | Description |
|---|---|
| `close` | Adjusted closing price |
| `open` | Opening price |
| `high` | Daily high |
| `low` | Daily low |
| `volume` | Trading volume |
| `vwap` | Volume-weighted average price |
| `returns` | Daily log returns (auto-computed from close) |

### Time-Series Operators

| Operator | Signature | Description |
|---|---|---|
| `ts_mean` | `(x, window)` | Rolling mean over `window` trading days |
| `ts_std` | `(x, window)` | Rolling standard deviation |
| `ts_delta` | `(x, window)` | `x - x[t - window]` (momentum/change) |
| `ts_delay` | `(x, window)` | `x[t - window]` (look-back) |
| `ts_max` | `(x, window)` | Rolling maximum |
| `ts_min` | `(x, window)` | Rolling minimum |
| `ts_rank` | `(x, window)` | Percentile rank within window |
| `ts_decay_linear` | `(x, window)` | Linearly-weighted decay |
| `ts_corr` | `(x, y, window)` | Rolling Pearson correlation |
| `ts_cov` | `(x, y, window)` | Rolling covariance |

### Cross-Sectional Operators

| Operator | Signature | Description |
|---|---|---|
| `rank` | `(x)` | Cross-sectional percentile rank [0, 1] |
| `zscore` | `(x)` | Cross-sectional z-score normalisation |
| `demean` | `(x)` | Subtract cross-sectional mean |
| `winsorize` | `(x, k)` | Clip outliers beyond ±k σ |
| `group_rank` | `(x, g)` | Within-group rank |
| `group_zscore` | `(x, g)` | Within-group z-score |
| `scale` | `(x)` | Scale to unit L1 norm |

### Arithmetic / Unary Operators

| Operator | Description |
|---|---|
| `+`, `-`, `*`, `/` | Element-wise arithmetic |
| `log(x)` | Natural logarithm |
| `abs(x)` | Absolute value |
| `neg(x)` or `x * -1` | Negation |
| `sqrt(x)` | Square root |
| `sign(x)` | Signum (−1, 0, +1) |

### Example DSL Expressions

```
# Classic price momentum
rank(ts_delta(log(close), 5))

# Mean-reversion on volume-weighted price
zscore(ts_mean(close, 20)) * -1

# Short-term reversal combined with volume signal
(rank(ts_delta(returns, 3)) * -1 + rank(ts_mean(volume, 10))) / 2

# Decay-smoothed momentum with z-score normalisation
ts_decay_linear(zscore(ts_delta(close, 10)), 5)

# Correlation between price and volume (volume impact)
rank(ts_corr(close, volume, 10))

# Volatility-scaled momentum
rank(ts_delta(close, 20)) / rank(ts_std(returns, 20))
```

### Validation Rules

The `AlphaValidator` enforces:
- Window size ∈ [1, 252] (valid trading days)
- No look-ahead bias: `ts_delay` / `ts_delta` require window ≥ 1
- AST nesting depth ≤ 10
- All data references must be recognised field names

---

## 6. Backend Engines

### 6.1 Alpha DSL Engine

**Module:** `backend/app/core/alpha_engine/`

#### Architecture

```
DSL string
    │
    ▼
parser.py  (Lark CFG)
    │
    ▼
typed_nodes.py  (Node dataclass)
    │
    ├─► validator.py  (static analysis — no runtime execution)
    │       WindowValidator    (window ∈ [1, 252])
    │       LookAheadValidator (window ≥ 1 for ts_delta/ts_delay)
    │       DepthValidator     (nesting ≤ 10)
    │
    └─► dsl_executor.py  (memoised AST → pd.DataFrame)
            operators.py  (dispatch table)
            fast_ops.py   (Bottleneck-accelerated kernels)
```

#### Key Classes

**`Parser`** (`parser.py`)
- Lark-based grammar converts DSL strings to typed `Node` objects
- Bidirectional: `node = parser.parse(dsl)`, `dsl = repr(node)`
- Raises `ParseError` on invalid syntax

**`AlphaValidator`** (`validator.py`)
- Composite of `WindowValidator`, `LookAheadValidator`, `DepthValidator`
- Collects all errors before raising `ValidationError`
- Used before any execution to prevent expensive failures

**`Executor`** (`dsl_executor.py`)
- Memoisation: identical subtrees computed once per `run()` call
- `run_expr(dsl, dataset)` — shortcut accepting raw DSL string
- Returns `pd.DataFrame (T × N)` with assets as columns

**`SimulationConfig`** (`signal_processor.py`)

| Parameter | Default | Range | Description |
|---|---|---|---|
| `delay` | 1 | 0–20 | Days to delay signal before trading |
| `decay_window` | 0 | 0–60 | `ts_decay_linear` smoothing window (0 = off) |
| `truncation_min_q` | 0.05 | 0.0–0.5 | Clip signals below this quantile |
| `truncation_max_q` | 0.95 | 0.5–1.0 | Clip signals above this quantile |
| `portfolio_mode` | "long_short" | enum | "long_short" or "decile" |
| `top_pct` | 0.10 | 0–0.5 | Top fraction for decile mode |

Signal processing pipeline: `truncation → decay → neutralisation → delay`

---

### 6.2 Data Engine

**Module:** `backend/app/core/data_engine/`

#### `DataPartitioner` (anti-overfitting core)

Strict physical IS/OOS split — the single most important anti-overfitting mechanism:

```python
partitioner = DataPartitioner(
    start     = "2020-01-01",
    end       = "2024-12-31",
    oos_ratio = 0.30,          # last 30% = OOS
)
part = partitioner.partition(dataset)
is_data  = part.train()        # deep copy, read-only
oos_data = part.test()         # deep copy, read-only, locked until final eval
```

- `PartitionedDataset` uses `__slots__` + custom `__setattr__` to prevent mutation
- Both partitions are **deep copies** — modifying IS data cannot contaminate OOS
- `split_date`, `is_days`, `oos_days` properties for reporting

#### Data Providers

| Provider | Source | Use Case |
|---|---|---|
| `YahooProvider` | yfinance | Live market data |
| `AlphaVantageProvider` | Alpha Vantage API | Alternative pricing |
| `LocalParquetProvider` | Local `.parquet` | Cached / offline data |

All providers implement the abstract `BaseProvider` interface: `load(tickers, start, end)` → `Dict[str, pd.DataFrame]`

---

### 6.3 Backtest Engine

**Module:** `backend/app/core/backtest_engine/`

#### `RealisticBacktester`

The primary evaluation engine used throughout the system:

```python
from app.core.alpha_engine.signal_processor import SimulationConfig
from app.core.backtest_engine.realistic_backtester import RealisticBacktester

cfg    = SimulationConfig(delay=1, decay_window=5)
bt     = RealisticBacktester(config=cfg)
result = bt.run(dsl, is_data, oos_dataset=oos_data)

print(result.is_report.sharpe_ratio)   # IS Sharpe
print(result.oos_report.sharpe_ratio)  # OOS Sharpe
```

#### `RiskReport` Metrics

| Metric | Description |
|---|---|
| `sharpe_ratio` | Annualised Sharpe (252-day) |
| `annualized_return` | CAGR |
| `annualized_vol` | Annualised volatility |
| `max_drawdown` | Peak-to-trough decline |
| `mean_ic` | Mean Spearman IC (signal ↔ next-day returns) |
| `ic_ir` | IC Information Ratio (mean / std) |
| `ann_turnover` | Average daily L1 signal change (% of book) |
| `net_returns` | Daily net return series (pd.Series) |

#### Portfolio Modes

| Mode | Description |
|---|---|
| `long_short` | Equal-weight long top / short bottom decile |
| `decile` | Long only, top `top_pct` fraction |

#### Transaction Costs

- **Commissions**: basis-points per trade
- **Slippage**: square-root-law model (`k × σ × √(trade_size / ADV)`)
- **Liquidity cap**: maximum 10% of ADV per position

---

### 6.4 GP Engine

**Module:** `backend/app/core/gp_engine/`

The GP engine performs **true structural evolution** — all mutations operate on typed AST nodes, never on strings. This guarantees every candidate is syntactically valid by construction.

#### `PopulationEvolver` (main GP class)

```python
evolver = PopulationEvolver(
    is_data        = is_data,
    oos_data       = oos_data,
    pop_size       = 20,
    n_generations  = 7,
    elite_ratio    = 0.25,
    corr_threshold = 0.90,
    seed           = 42,
)
result = evolver.run(
    seed_dsls         = initial_dsl_list,
    n_optuna_trials   = 10,
    on_generation_end = callback,         # fired after each generation
)
```

#### Evolution Loop (per generation)

```
1. Evaluate population
   └─ RealisticBacktester IS+OOS for each individual
   └─ compute: sharpe_is, sharpe_oos, turnover, overfitting_score

2. Compute fitness
   └─ fitness = sharpe_oos
              - 0.2 × turnover
              - 0.3 × max(0, sharpe_is - sharpe_oos)

3. AlphaPool update (diversity filter)
   └─ Add each individual's signal to AlphaPool
   └─ Reject if signal correlation > 0.9 with existing pool

4. Log generation summary
   └─ generation, population_size, best_fitness, best_oos_sharpe, best_dsl

5. Adaptive mutation weights
   └─ mutation_weights_from_metrics(oos_sharpe, turnover, overfit_score)
   └─ Adjusts crossover / point / hoist / param ratios dynamically

6. Generate next population
   ├─ Elite preservation (top 25%)
   ├─ Tournament selection (k=3)
   └─ Apply GP operators:
      ├─ crossover  (40%): subtree_crossover(p1, p2) → c1, c2
      ├─ point      (30%): swap leaf node or operator
      ├─ hoist      (15%): extract and promote subtree
      └─ param      (15%): adjust window size ±1–3
```

#### `AlphaPool`

Cross-generation memory for diversity enforcement:
- Maximum 200 entries; culls by lowest fitness when full
- Adds entry only if signal correlation < 0.90 with all existing entries
- `population_diagnostics()` returns mean OOS Sharpe, turnover, overfit score
- `top_k(k)` retrieves the k best entries by fitness

#### Fitness Formula

```
fitness = sharpe_oos
        - 0.20 × turnover
        - 0.30 × max(0, sharpe_is - sharpe_oos)
```

OOS Sharpe is the primary objective. Turnover penalises excessive trading. The last term penalises IS−OOS degradation (overfitting).

---

### 6.5 ML Engine (Optuna + Evaluator)

**Module:** `backend/app/core/ml_engine/`

#### `AlphaOptimizer` (Optuna)

Parameter fine-tuning **after** GP selects the best structure. Optuna searches **IS data only** — OOS data is physically locked:

```python
optimizer = AlphaOptimizer(
    dsl          = best_dsl,
    is_dataset   = is_data,    # OOS never passed here
    search_space = SearchSpace(
        delay_range     = (0, 3),
        decay_range     = (0, 8),
        portfolio_modes = ("long_short",),
    ),
    n_trials = 10,
    seed     = 42,
)
best_config, summary = optimizer.optimize()
```

**Search space parameters:**

| Parameter | Range | Description |
|---|---|---|
| `delay` | 0–3 | Execution lag |
| `decay_window` | 0–8 | Signal smoothing |
| `truncation_min_q` | 0.01–0.10 | Lower percentile clip |
| `truncation_max_q` | 0.90–0.99 | Upper percentile clip |
| `portfolio_mode` | enum | long_short / decile |

**Optuna Objective:**
```
objective = Sharpe_IS + 0.5 × IC_IS - 0.1 × Turnover_IS
```
(OOS is never accessed; final IS+OOS evaluation is run separately after Optuna completes.)

#### `AlphaEvaluator`

High-level metrics aggregation:

```python
evaluator = AlphaEvaluator()
result = evaluator.evaluate(
    is_report=..., is_prices=..., is_signal=...,
    oos_report=..., oos_prices=..., oos_signal=...,
)

print(result.overfitting_score)  # 0.0 – 1.0
print(result.is_overfit)         # True if score > 0.5
print(result.ic_decay)           # {"t1": ..., "t5": ...}
```

**Overfitting score:**
```
overfitting_score = clip((Sharpe_IS - Sharpe_OOS) / |Sharpe_IS|, 0, 1)
```
Score 0 = no degradation; Score 1 = complete OOS failure.

---

### 6.6 Workflow Pipelines

**Module:** `backend/app/core/workflows/alpha_workflows.py`

Two production pipelines sharing the same GP core:

#### Workflow A — `GenerationWorkflow` (Hypothesis → Alpha)

```
Input: natural language hypothesis (e.g. "short-term momentum with volume filter")

Step 1  Generate ≥12 diverse seed DSLs
        ├─ Layer 1: LLM (AlphaAgent._generate_dsls) — hypothesis-specific
        ├─ Layer 2: Keyword templates (momentum/reversion/volume/volatility)
        ├─ Layer 3: AST mutations of valid seeds (point/hoist/param)
        └─ Layer 4: Random alphas (generate_random_alpha + _SEED_DSLS)

Step 2  IS/OOS partition (DataPartitioner, oos_ratio=0.30)

Step 3  PopulationEvolver
        └─ all seed DSLs → initial population
        └─ GP evolution: 7 generations, pop=max(20, n_seeds+4)
        └─ fitness = OOS Sharpe - 0.20×turnover - 0.30×overfit_penalty
        └─ AlphaPool diversity filter (corr < 0.90)

Step 4  Optuna fine-tune (10 trials, IS-only)

Step 5  Final IS+OOS backtest with best config

Step 6  Auto-save to AlphaStore

Output: WorkflowResult(best_dsl, metrics, evolution_log, pool_top5, best_config, explanation)
```

#### Workflow B — `OptimizationWorkflow` (DSL → Optimised Alpha)

```
Input: existing DSL string

Step 1  Parse + validate → canonical AST

Step 2  IS/OOS partition + quick_metrics (initial quality diagnosis)
        → is_sharpe, oos_sharpe, turnover, overfitting_score

Step 3  Expand population
        ├─ original canonical DSL
        ├─ n_mutations structural variants (point/hoist/param)
        └─ random fill to meet pop_size

Step 4  Targeted structural mutations (based on metric diagnosis)
        ├─ High turnover (>3.0) → ts_decay_linear wrappers (w=3,5,8)
        ├─ Low OOS Sharpe (<0.3) → rank(), scale(), signal combination
        └─ Overfit (>0.5) → ts_mean smoothing (w=3,5,10)

Step 5  PopulationEvolver
        └─ expanded population → GP evolution (7 gen)
        └─ same fitness / AlphaPool logic as Workflow A

Step 6  Optuna fine-tune (10 trials, IS-only)

Step 7  Auto-save to AlphaStore

Output: WorkflowResult with delta vs input (OOS Sharpe improvement)
```

#### SSE Streaming

Both workflows have dedicated streaming endpoints that push progress via Server-Sent Events as the GP evolves:

```
Event stream format (NDJSON over text/event-stream):

{"type": "text",  "text": "[Workflow B] Parsing and diagnosing initial quality..."}
{"type": "text",  "text": "[Diagnose] IS Sharpe=0.4521 | OOS Sharpe=0.2834 | ..."}
{"type": "text",  "text": "[GP] Launching evolution: pop=24 | gen=7 | optuna_trials=10"}
{"type": "text",  "text": "[GP] Gen 1/7 | pop=24 | fitness=0.1823 | oos_sharpe=0.3012\n     → rank(ts_delta(close, 5))"}
{"type": "text",  "text": "[GP] Gen 2/7 | pop=24 | fitness=0.2145 | oos_sharpe=0.3567\n     → zscore(ts_mean(returns, 10))"}
...
{"type": "text",  "text": "[Optuna] Fine-tuning complete | config: {\"delay\": 1, ...}"}
{"type": "text",  "text": "[Result] IS Sharpe=0.6123 | OOS Sharpe=0.4891 (↑0.2057 vs input)"}
{"type": "text",  "text": "✓ Anti-overfitting check passed"}
{"type": "ping"}                          # keep-alive every ~60s
{"type": "done",  "result": {...}}        # full WorkflowResponse payload
{"type": "error", "message": "..."}      # terminal error (if any)
```

---

### 6.7 AI Agent (LangChain)

**Module:** `backend/app/agent/`

#### `QuantAgent`

Main conversational interface. Wraps LangChain `RunnableWithMessageHistory`:

```python
agent = QuantAgent(
    n_tickers = 20,
    n_days    = 252,
    oos_ratio = 0.30,
    n_trials  = 10,
    chat_store = chat_store,
)
reply, dsl, metrics = agent.chat(message="Generate a momentum alpha", session_id="abc")
```

#### Tools (`QuantTools`)

Five LangChain tools available to the agent:

| Tool | Description |
|---|---|
| `generate_alpha_dsl` | Generate DSL from hypothesis keywords |
| `run_gp_optimization` | Run GP evolution (PopulationEvolver) |
| `run_optuna` | Optuna parameter fine-tuning |
| `run_backtest` | IS+OOS backtest for a DSL |
| `save_alpha` | Persist result to AlphaStore |

#### `FallbackOrchestrator`

Rule-based orchestrator for zero-LLM mode (no `OPENAI_API_KEY`):
- Keyword matching: "momentum", "reversion", "volume", "volatility" → template DSLs
- Dispatches `run_workflow_a` / `run_workflow_b` internally
- All GP + Optuna capabilities remain available

#### Memory & Sessions

- `ConversationMemory`: LRU dict of `{session_id → SQLAlchemyChatMessageHistory}`
- Maximum 50 concurrent sessions (evicts least-recently-used)
- `ChatStore`: SQLite-backed `ChatMessage` ORM (session_id, role, content, created_at)

#### `OverfitCritic`

Anti-overfitting gate applied after every backtest:
- Rejects alphas where `overfitting_score > threshold` (default 0.5)
- Returns corrective feedback for agent self-improvement loop

---

### 6.8 Database Layer

**Module:** `backend/app/db/`

#### `AlphaStore`

SQLite-backed alpha ledger:

```python
store = AlphaStore(db_url="sqlite:///./alphas.db")

# Save
alpha_id = store.save(AlphaResult(
    dsl          = "rank(ts_delta(log(close), 5))",
    hypothesis   = "5-day price momentum",
    sharpe       = 0.92,
    ann_return   = 0.18,
    ann_turnover = 1.42,
    ic_ir        = 0.31,
    status       = "active",
))

# Query
records = store.query(min_sharpe=0.5, status="active", limit=30)

# Export
store.export_csv("./alpha_ledger.csv")
```

**`AlphaRecord` ORM schema:**

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `dsl` | TEXT | Alpha expression |
| `hypothesis` | TEXT | Natural language description |
| `sharpe` | FLOAT | IS Sharpe ratio |
| `ann_return` | FLOAT | Annualised return |
| `ann_turnover` | FLOAT | Annualised turnover |
| `ic_ir` | FLOAT | IC Information Ratio |
| `max_drawdown` | FLOAT | Maximum drawdown |
| `status` | TEXT | "active" / "retired" |
| `reasoning` | TEXT | Agent explanation |
| `created_at` | DATETIME | Auto-set |

#### `ChatStore`

SQLite-backed conversation history:

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `session_id` | TEXT | UUID session identifier |
| `role` | TEXT | "user" / "assistant" |
| `content` | TEXT | Message text |
| `created_at` | DATETIME | Auto-set |

---

## 7. API Reference

All endpoints are served under the FastAPI app at `http://127.0.0.1:8000`. Interactive Swagger UI at `/docs`.

### Chat Endpoints

| Method | Path | Request | Response | Description |
|---|---|---|---|---|
| `POST` | `/api/chat` | `{message, session_id}` | `{session_id, reply, dsl?, metrics?}` | Chat with QuantAgent |
| `POST` | `/api/chat/sessions` | `{title}` | `{session_id, title, created_at}` | Create new session |
| `GET` | `/api/chat/sessions` | — | `{sessions[], count}` | List all sessions |
| `GET` | `/api/chat/sessions/{id}` | — | `{session_id, title, messages[]}` | Get session history |

### Workflow Endpoints (Primary)

| Method | Path | Request Body | Response | Description |
|---|---|---|---|---|
| `POST` | `/api/workflow/generate` | `WorkflowGenerateRequest` | `WorkflowResponse` | Hypothesis → GP-evolved alpha |
| `POST` | `/api/workflow/optimize` | `WorkflowOptimizeRequest` | `WorkflowResponse` | DSL → GP-optimised alpha |
| `POST` | `/api/workflow/generate/stream` | `WorkflowGenerateRequest` | SSE stream | Streaming Workflow A |
| `POST` | `/api/workflow/optimize/stream` | `WorkflowOptimizeRequest` | SSE stream | Streaming Workflow B |

**`WorkflowGenerateRequest`:**
```json
{
  "hypothesis":    "momentum",
  "n_tickers":     20,
  "n_days":        252,
  "n_generations": 7,
  "pop_size":      20,
  "n_optuna":      10,
  "n_seed_dsls":   12,
  "oos_ratio":     0.30,
  "seed":          42
}
```

**`WorkflowOptimizeRequest`:**
```json
{
  "dsl":           "rank(ts_delta(log(close), 5))",
  "n_tickers":     20,
  "n_days":        252,
  "n_generations": 7,
  "pop_size":      20,
  "n_optuna":      10,
  "n_mutations":   8,
  "oos_ratio":     0.30,
  "seed":          42
}
```

**`WorkflowResponse`:**
```json
{
  "workflow":        "optimization",
  "best_dsl":        "zscore(ts_mean(returns, 10))",
  "metrics": {
    "is_sharpe":   0.8432,
    "oos_sharpe":  0.6127,
    "is_return":   0.1823,
    "is_turnover": 1.2341,
    "is_ic":       0.0412,
    "overfitting_score": 0.2732,
    "is_overfit":  false
  },
  "evolution_log": [
    {
      "generation": 1, "population_size": 24,
      "best_fitness": 0.3241, "best_oos_sharpe": 0.4123,
      "best_dsl": "...", "mean_fitness": 0.1823
    }
  ],
  "pool_top5":       [...],
  "best_config":     {"delay": 1, "decay_window": 3, "portfolio_mode": "long_short"},
  "seed_dsls":       [...],
  "generations_run": 7,
  "explanation":     "...",
  "pnl_is":          [...],
  "pnl_oos":         [...],
  "split_date":      "2023-06-30",
  "overfitting_score": 0.2732,
  "is_overfit":      false
}
```

### Backtest / Simulate Endpoints

| Method | Path | Request Body | Response | Description |
|---|---|---|---|---|
| `POST` | `/alpha/simulate` | `SimulateRequest` | `EvalResponse` | Manual IS+OOS backtest |
| `POST` | `/alpha/optimize` | `OptimizeRequest` | `EvalResponse` | Optuna param tune |
| `POST` | `/backtest/run` | `{dsl, n_tickers, n_days}` | `{dsl, report}` | Basic backtest |
| `POST` | `/backtest/realistic` | `RealisticBacktestRequest` | full IS+OOS | Enhanced backtest |

**`SimulateRequest`:**
```json
{
  "dsl":       "rank(ts_delta(log(close), 5))",
  "config": {
    "delay": 1, "decay_window": 0,
    "truncation_min_q": 0.05, "truncation_max_q": 0.95,
    "portfolio_mode": "long_short", "top_pct": 0.10
  },
  "n_tickers": 20,
  "n_days":    252,
  "oos_ratio": 0.30
}
```

**`EvalResponse`:**
```json
{
  "dsl":               "...",
  "is_metrics":        {"sharpe_ratio": 0.84, "annualized_return": 0.18, ...},
  "oos_metrics":       {"sharpe_ratio": 0.61, ...},
  "overfitting_score": 0.27,
  "is_overfit":        false,
  "ic_decay":          {"t1": 0.032, "t5": 0.018},
  "best_config":       null,
  "n_trials_run":      null,
  "pnl_is":            [...],
  "pnl_oos":           [...],
  "split_date":        "2023-06-30"
}
```

### Alpha Ledger Endpoints

| Method | Path | Params | Response | Description |
|---|---|---|---|---|
| `POST` | `/alpha/save` | `SaveAlphaRequest` body | `{id, status}` | Save alpha result |
| `GET` | `/report/query` | `min_sharpe`, `status`, `limit`, `alpha_id` | `{total, records[]}` | Query ledger |

### Legacy / Low-level Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/agent/run` | AlphaAgent run (hypothesis-driven, returns initial DSLs) |
| `POST` | `/gp/evolve` | Raw DEAP GP evolution (Hall of Fame) |
| `GET` | `/health` | Health check |

---

## 8. Frontend Architecture

### 8.1 State Management

**`workspaceStore.ts`** — single Zustand store

```typescript
interface WorkspaceState {
  // View mode
  activeView:  'CHAT' | 'COMPILER'
  setActiveView: (v: ActiveView) => void

  // Monaco editor tabs
  editorTabs:    EditorTab[]       // {id, label, dsl, alphaId?, isModified}
  activeTabId:   string
  editorDsl:     string            // mirrors active tab's dsl

  // Session management
  sessionId:   string
  sessions:    ChatSession[]       // {id, title, createdAt}

  // Chat messages
  chatMessages: ChatMessage[]      // {id, role, content, dsl?, metrics?, type?, isStreaming?}

  // Streaming message actions
  startStreamingMessage:    (id: string) => void
  appendToStreamingMessage: (id: string, chunk: string) => void
  finalizeStreamingMessage: (id: string, extras?: Partial<ChatMessage>) => void

  // Alpha history (ledger)
  alphaHistory: AlphaRecord[]

  // Backtest result (drives RightPane chart)
  simulationResult: SimResult | null

  // Status bar
  status:      'idle' | 'optimizing' | 'backtesting' | 'ready' | 'error'

  // Console log lines
  consoleLogs: string[]

  // Simulation parameters
  simConfig:   SimulationConfig

  // Ledger panel
  ledgerOpen:  boolean
}
```

**`useQuantWorkspace` hook** — main action layer

```typescript
const { sendChat, runBacktest, runOptimize, loadHistory,
        loadSessions, newSession, switchSession, store } = useQuantWorkspace()
```

| Action | Trigger | Behaviour |
|---|---|---|
| `sendChat(text)` | Chat input enter | POST /api/chat → typing-effect reply |
| `runBacktest()` | Run button | POST /alpha/simulate → metrics + PnL chart |
| `runOptimize()` | Optimize button | SSE /workflow/optimize/stream → typed output → new tab |
| `loadHistory()` | After save/run | GET /report/query → updates alpha ledger |
| `newSession()` | + button | POST /api/chat/sessions → fresh chat |
| `switchSession(id)` | Session click | GET /api/chat/sessions/{id} → restore history |

### 8.2 Components

#### Layout

**`WorkspaceLayout`** — root 3-pane layout (react-resizable-panels)
- Left panel (64px fixed): `GlobalSidebar`
- Centre-left panel (resizable): `SessionHistoryPanel` (CHAT) or `LeftLedgerPane` (COMPILER)
- Main panel: `ChatView` (CHAT) or `CompilerView` + `ConsoleOutput` + `RightPane` (COMPILER)

**`GlobalSidebar`** — icon toolbar (64px)
- View toggle: Chat / Compiler
- Run / Optimize buttons
- Session management controls

**`LeftLedgerPane`** (COMPILER mode)
- Sorted alpha history table (by Sharpe)
- Per-record: DSL preview, Sharpe, IC-IR, turnover, date
- Click to open alpha in new Monaco editor tab

**`SessionHistoryPanel`** (CHAT mode)
- List of saved chat sessions
- Create new session, switch between sessions

**`RightPane`** (COMPILER mode)
- `PnLChart` — ECharts cumulative PnL with IS/OOS regions
- `MetricsGrid` — IS vs OOS metrics comparison table
- `OverfitBadge` — overfitting indicator

#### Chat

**`ChatMessage`** — renders a single message bubble:
- User messages: right-aligned, emerald bubble
- Assistant text messages: left-aligned, dark bubble
- Streaming GP output: dark monospace bubble with per-line tag coloring + blinking cursor
- DSL chip with "Edit" button (opens in compiler)
- Metrics chips (hidden while streaming)

Tag coloring in streaming bubbles:

| Tag | Color |
|---|---|
| `[GP]` | Violet |
| `[Optuna]` | Violet (lighter) |
| `[Diagnose]` | Sky |
| `[Result]` | Emerald |
| `[Workflow A/B]` | Slate |
| `[ERROR]` | Rose |
| `[WARN]`, `⚠` | Amber |
| `✓` | Emerald |

**`ThoughtBlock`** — expandable/collapsible agent reasoning block

#### Compiler

**`CompilerView`** — Monaco editor workspace:
- Tab bar with create/close tabs
- DSL-syntax Monaco editor
- Toolbar: Run Backtest, AI Optimize, Config
- Auto-opens GP-evolved best DSL in new tab after streaming

**`ConfigModal`** — `SimulationConfig` parameter sliders:
- Delay, Decay Window, Truncation (min/max), Portfolio Mode, Top Pct
- Changes reflected immediately in next backtest run

**`ConsoleOutput`** — bottom log console with tag-based coloring:

| Tag | Color | Typical content |
|---|---|---|
| `[GP]` | Violet | GP generation summaries |
| `[Optuna]` | Violet (lighter) | Parameter tuning results |
| `[Backtest]` / `[BKTEST]` | Sky | Backtest metric output |
| `[AST]` | Cyan | AST parsing telemetry |
| `[DATA]` | Teal | Data engine state / NaN warnings |
| `[TELEM]` | Indigo | System telemetry |
| `[PERF]` | Lime | Performance / resource overhead |
| `[NaN]` | Amber | NaN warnings |
| `[OK]` | Emerald | Success messages |
| `[WARN]` | Amber | Warnings |
| `[ERROR]` / `[Syntax Error]` | Rose | Errors |
| `[System]` | Slate | System messages |

#### Analysis

**`PnLChart`** — cumulative PnL visualisation:
- Blue `markArea` tint: IS period ("In-Sample")
- Red `markArea` tint: OOS period ("UNSEEN DATA")
- Dashed `markLine`: IS/OOS split date
- Area gradient fill under cumulative return curve

**`MetricsGrid`** — IS vs OOS comparison table:
- Sharpe Ratio, Annualised Return, Max Drawdown, IC, IC-IR, Turnover

**`OverfitBadge`** — overfitting indicator:

| Score | Badge | Animation |
|---|---|---|
| < 40% | Emerald "Healthy" | static |
| 40–60% | Amber "Overfit Risk" | pulse |
| > 60% | Rose "Overfit!" | pulse |

### 8.3 SSE Streaming

When the user triggers "AI Optimize" from the compiler:

```
1. store.setActiveView('CHAT')          → auto-switch to chat view
2. store.startStreamingMessage(id)      → create empty streaming message
3. streamWorkflowOptimize(dsl, onEvent) → fetch /workflow/optimize/stream
4. onEvent callback:
   ├─ type="text"  → typeIntoMessage(id, text, 10ms/char)  [character-by-character]
   ├─ type="ping"  → ignored
   ├─ type="error" → store.finalizeStreamingMessage(id)
   └─ type="done"  → finalizeStreamingMessage with DSL+metrics chip
                     setSimulationResult(...)   → RightPane chart updates
                     newEmptyTab() + setEditorDsl(best_dsl)
                     loadHistory()              → ledger refreshes
```

The `typeIntoMessage` function appends one character at a time with a 10ms delay, creating the character-by-character animation effect. A sequential text queue prevents simultaneous typing from multiple events:

```typescript
const flushQueue = async () => {
  if (typing) return
  typing = true
  while (textQueue.length > 0) {
    const chunk = textQueue.shift()!
    await typeIntoMessage(streamId, chunk + '\n', 10)
  }
  typing = false
}
```

Similarly, `sendChat` responses are typed out character by character (14ms/char) via `typeIntoMessage`.

---

## 9. Anti-Overfitting Design

The platform enforces strict anti-overfitting at every layer of the stack:

### Layer 1 — Data Isolation (`DataPartitioner`)

```
Full dataset: Jan 2020 – Dec 2024
      │
      ├─── IS (70%):  Jan 2020 – Jun 2023   ← GP trains here, Optuna searches here
      │
      └─── OOS (30%): Jul 2023 – Dec 2024   ← Seen ONLY during final evaluation
```

- `PartitionedDataset` uses `__slots__` + custom `__setattr__` — modifying one partition cannot leak into the other
- OOS data is **never** passed into `AlphaOptimizer`
- `RealisticBacktester` accepts `oos_dataset=None` by default — caller must explicitly unlock it

### Layer 2 — Fitness Function (GP Evolution)

The GP fitness function directly penalises IS/OOS degradation:

```
fitness = sharpe_oos - 0.20 × turnover - 0.30 × max(0, sharpe_is - sharpe_oos)
```

The third term forces the GP to prefer alphas that generalise — if IS Sharpe greatly exceeds OOS Sharpe, fitness is penalised proportionally.

### Layer 3 — Diversity Filter (`AlphaPool`)

Rejects any candidate whose signal has correlation > 0.9 with an already-accepted alpha. This prevents the population from converging to a single overfit idea.

### Layer 4 — Optuna Objective

```
optuna_objective = sharpe_IS + 0.5 × IC_IS - 0.1 × turnover_IS
```

OOS is never visible. After Optuna completes, the winning config is locked and a single IS+OOS validation is run to measure true generalisation.

### Layer 5 — `OverfitCritic` (Agent)

Applied to every alpha the agent generates. Alphas with `overfitting_score > 0.5` are rejected and the agent receives corrective feedback to try a different structural approach.

### Layer 6 — UI Badge

Real-time visual feedback on every backtest result:
- Users cannot accidentally ignore overfitting — the pulsing badge forces attention
- Full IS vs OOS metrics side-by-side in MetricsGrid

---

## 10. Key Metrics & Formulas

| Metric | Formula | Interpretation |
|---|---|---|
| **Sharpe Ratio** | `mean(r) / std(r) × √252` | Risk-adjusted return; target > 0.5 for IS, > 0.3 for OOS |
| **IC (Information Coefficient)** | `Spearman(signal_t, return_{t+1})` | Signal predictive power; good if |IC| > 0.02 |
| **IC-IR** | `mean(IC) / std(IC)` | IC consistency; target > 0.3 |
| **Annualised Turnover** | `mean(L1 signal change) × 252` | Trading frequency; penalised in fitness |
| **Max Drawdown** | `max(peak - trough) / peak` | Worst loss from peak; < 25% preferred |
| **Overfitting Score** | `clip((Sharpe_IS - Sharpe_OOS) / \|Sharpe_IS\|, 0, 1)` | 0=no overfit, 1=complete failure |
| **GP Fitness** | `oos_sharpe - 0.20×turnover - 0.30×max(0, is_sharpe - oos_sharpe)` | GP selection objective |
| **Optuna Objective** | `is_sharpe + 0.50×ic_ir - 0.10×turnover` | Optuna search objective (IS-only) |
| **IC Decay T+1** | `Spearman(signal_t, return_{t+1})` | Next-day predictive power |
| **IC Decay T+5** | `Spearman(signal_t, return_{t+5})` | 5-day predictive power |

---

## 11. Configuration

### Environment Variables

Create `backend/.env` (auto-created with defaults if absent):

```env
# LLM (optional — FallbackOrchestrator used if absent)
OPENAI_API_KEY=sk-...

# Database
DATABASE_URL=sqlite:///./alphas.db

# Debug
DEBUG=false

# Default dataset parameters
DEFAULT_N_TICKERS=20
DEFAULT_N_DAYS=120
```

### `SimulationConfig` Defaults

```python
SimulationConfig(
    delay            = 1,     # 1-day execution lag
    decay_window     = 0,     # no decay smoothing
    truncation_min_q = 0.05,  # winsorise below 5th percentile
    truncation_max_q = 0.95,  # winsorise above 95th percentile
    portfolio_mode   = "long_short",
    top_pct          = 0.10,  # top 10% for decile mode
)
```

### Workflow Defaults

| Parameter | Workflow A | Workflow B |
|---|---|---|
| `pop_size` | 20 | 20 |
| `n_generations` | 7 | 7 |
| `n_optuna` | 10 | 10 |
| `n_seed_dsls` | 12 | — |
| `n_mutations` | — | 8 |
| `oos_ratio` | 0.30 | 0.30 |

---

## 12. CLI Modes

The backend can also be run as a CLI tool:

```bash
cd backend
source .venv/bin/activate

# Run LangChain agent (1 hypothesis iteration)
python -m app.main --mode agent --hypothesis "short-term mean reversion"

# Run GP evolution (DEAP-based)
python -m app.main --mode gp --n-gen 10 --pop-size 30

# Run single backtest
python -m app.main --mode backtest --dsl "rank(ts_delta(log(close),5))"

# Query alpha ledger
python -m app.main --mode report --limit 20 --min-sharpe 0.5

# Run realistic IS+OOS backtest
python -m app.main --mode realistic --dsl "zscore(ts_mean(returns,10))"
```

---

## 13. Testing

```bash
cd backend
source .venv/bin/activate
pip install pytest pytest-asyncio httpx

# Run all tests
pytest tests/ -v

# Individual test files
pytest tests/test_dsl_engine.py -v           # DSL parser + executor
pytest tests/test_backtest_engine.py -v      # Backtest pipeline
pytest tests/test_data_engine_smoke.py -v    # DataPartitioner + providers
pytest tests/test_phase2.py -v               # Simulate + optimize endpoints
pytest tests/test_phase3.py -v               # Workflow A + B pipelines
pytest tests/test_alpha_discovery.py -v      # End-to-end alpha discovery
```

### Test Coverage Areas

| File | Covers |
|---|---|
| `test_dsl_engine.py` | Parser, Validator, Executor, all operators |
| `test_backtest_engine.py` | RealisticBacktester, RiskReport, portfolio modes |
| `test_data_engine_smoke.py` | DataPartitioner isolation, provider interfaces |
| `test_phase1_upgrade.py` | Agent run, backtest/run, report/query endpoints |
| `test_phase2.py` | /alpha/simulate, /alpha/optimize endpoints |
| `test_phase3.py` | /workflow/generate, /workflow/optimize endpoints |
| `test_alpha_discovery.py` | Full end-to-end alpha discovery loop |

---

## Licence

MIT — see `LICENSE` for details.
