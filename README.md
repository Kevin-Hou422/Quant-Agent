# Quant Agent — Production-Grade Alpha Research Platform

An end-to-end, autonomous quantitative alpha research platform modelled after WorldQuant Brain. It combines a high-performance DSL engine, realistic backtesting, ML-powered optimization (Optuna), an AI-driven LangChain agent, and a React/TypeScript OS-style UI.

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [File Structure](#file-structure)
3. [Quick Start](#quick-start)
4. [Installation Details](#installation-details)
5. [Backend Modules](#backend-modules)
6. [Frontend Modules](#frontend-modules)
7. [API Reference](#api-reference)
8. [Configuration](#configuration)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        React / TypeScript UI                         │
│  GlobalSidebar │ LeftLedgerPane │ CenterPane (Chat/Compiler) │ Right │
└───────────────────────────┬──────────────────────────────────────────┘
                            │ HTTP (Vite proxy → :8000)
┌───────────────────────────▼──────────────────────────────────────────┐
│                       FastAPI  (:8000)                               │
│  /alpha/simulate  /alpha/optimize  /api/chat  /api/report/query      │
└──┬────────────────┬───────────────┬────────────────┬─────────────────┘
   │                │               │                │
   ▼                ▼               ▼                ▼
Alpha DSL       Backtest       Optuna ML         LangChain
Engine (AST)    Engine         Optimizer         Agent (GPT-4o)
   │                │
   ▼                ▼
GP Evolution    Data Engine (yfinance / Parquet)
Engine          + DataPartitioner (IS / OOS split)
```

---

## File Structure

```
Quant Agent/
├── start.sh                        # One-command launcher (bash)
├── README.md
│
├── backend/
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py                 # FastAPI app entry-point + CLI runner
│   │   ├── config.py               # pydantic-settings global config
│   │   ├── dependencies.py         # DI: AlphaStore, DB session
│   │   │
│   │   ├── api/
│   │   │   ├── router.py           # API route aggregator
│   │   │   ├── backtest_router.py  # POST /api/backtest/run
│   │   │   ├── optimize_router.py  # POST /alpha/optimize
│   │   │   ├── simulate_router.py  # POST /alpha/simulate
│   │   │   ├── chat_router.py      # POST /api/chat (LangChain agent)
│   │   │   └── report_router.py    # GET /api/report/query
│   │   │
│   │   └── core/
│   │       ├── alpha_engine/       # DSL parser, AST nodes, validator, executor
│   │       │   ├── __init__.py
│   │       │   ├── ast_nodes.py    # Node classes (TimeSeries/CrossSectional/Scalar)
│   │       │   ├── parser.py       # DSL string → AST
│   │       │   ├── executor.py     # AST → pandas signal (vectorized)
│   │       │   ├── validator.py    # Static analysis, lookahead guard
│   │       │   └── fast_ops.py     # Bottleneck / Numba accelerated operators
│   │       │
│   │       ├── backtest_engine/    # Realistic backtester
│   │       │   ├── __init__.py
│   │       │   ├── data_engine.py  # yfinance loader, PIT data, Parquet cache
│   │       │   ├── data_partitioner.py  # IS / OOS strict split
│   │       │   ├── simulator.py    # Long-short / Decile portfolio + costs
│   │       │   ├── metrics.py      # Sharpe, IC, Turnover, Drawdown
│   │       │   └── risk_report.py  # Full report builder
│   │       │
│   │       ├── optimization_engine/  # Optuna ML optimizer
│   │       │   ├── __init__.py
│   │       │   ├── alpha_optimizer.py  # AlphaOptimizer (Optuna)
│   │       │   └── evaluator.py        # AlphaEvaluator (IS+OOS metrics)
│   │       │
│   │       ├── gp_engine/          # Genetic Programming
│   │       │   ├── __init__.py
│   │       │   └── gp_engine.py    # GP: crossover, mutation, fitness
│   │       │
│   │       ├── ml_engine/          # Alpha store, reasoning logs, ML scorer
│   │       │   ├── __init__.py
│   │       │   ├── alpha_store.py  # SQLite-backed alpha ledger
│   │       │   ├── ml_scorer.py    # XGBoost/LightGBM feature pruning
│   │       │   └── reasoning_log.py
│   │       │
│   │       ├── portfolio_engine/   # Portfolio construction helpers
│   │       │   └── __init__.py
│   │       │
│   │       └── utils/
│   │           ├── __init__.py
│   │           ├── fast_ops.py     # Aliases: ts_mean / ts_std / ts_rank
│   │           └── simulation_config.py  # SimulationConfig dataclass
│   │
│   └── tests/
│       ├── test_phase1.py
│       ├── test_phase2.py
│       ├── test_phase3.py
│       └── test_alpha_discovery.py
│
└── frontend/
    ├── package.json
    ├── vite.config.ts              # Vite + Tailwind plugin + /api proxy
    ├── tsconfig.json
    ├── index.html
    └── src/
        ├── main.tsx
        ├── App.tsx
        ├── index.css               # Tailwind v4 import
        │
        ├── types/
        │   └── index.ts            # ChatMessage, AlphaRecord, SimResult, SimulationConfig
        │
        ├── api/
        │   └── client.ts           # Axios wrappers for all backend endpoints
        │
        ├── store/
        │   └── workspaceStore.ts   # Zustand: activeView, chat, alphaHistory, results
        │
        ├── hooks/
        │   └── useQuantWorkspace.ts  # sendChat, runBacktest, runOptimize actions
        │
        └── components/
            ├── layout/
            │   ├── WorkspaceLayout.tsx   # 3-pane root layout
            │   ├── GlobalSidebar.tsx     # w-20 icon sidebar + Run button
            │   ├── LeftLedgerPane.tsx    # Alpha history + status badges
            │   └── RightPane.tsx         # PnL chart + metrics grid
            │
            ├── chat/
            │   ├── ChatView.tsx          # Scrollable chat interface
            │   ├── ChatMessage.tsx       # User / assistant / tool bubbles
            │   └── ThoughtBlock.tsx      # Agent reasoning display
            │
            ├── compiler/
            │   ├── CompilerView.tsx      # Monaco Editor + toolbar
            │   ├── ConsoleOutput.tsx     # Bottom log console
            │   └── ConfigModal.tsx       # SimulationConfig sliders modal
            │
            └── analysis/
                ├── PnLChart.tsx          # ECharts IS/OOS PnL with markArea
                ├── MetricsGrid.tsx       # IS vs OOS metrics table
                └── OverfitBadge.tsx      # Amber/red pulsing overfitting indicator
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- `bash` (Git Bash / WSL on Windows)

### One-command launch

```bash
bash start.sh
```

- **Frontend**: http://localhost:5173
- **Backend API**: http://127.0.0.1:8000
- **Swagger docs**: http://127.0.0.1:8000/docs

---

## Installation Details

### Backend (manual)

```bash
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend (manual)

```bash
cd frontend
npm install
npm run dev
```

### Key Python Dependencies

```
fastapi>=0.111
uvicorn[standard]
pydantic>=2
pydantic-settings
sqlalchemy
yfinance
pandas numpy bottleneck
optuna
langchain langchain-openai langchain-community
xgboost lightgbm
pyarrow fastparquet
```

Install all:
```bash
pip install -r backend/requirements.txt
```

### Key Node Dependencies

```
react react-dom
@vitejs/plugin-react
tailwindcss @tailwindcss/vite
zustand
axios
echarts echarts-for-react
@monaco-editor/react
lucide-react
```

---

## Backend Modules

### Alpha DSL Engine (`core/alpha_engine/`)
- Node-based AST with `TimeSeriesNode`, `CrossSectionalNode`, `ScalarNode`
- Vectorized operators via Bottleneck/Numba — zero Python loops
- Static validator blocks look-ahead functions and deep recursion
- Sub-expression memoization cache

### Data Engine (`core/backtest_engine/data_engine.py`)
- yfinance loader with Parquet cache
- Point-in-time aligned panel (NaN for delisted assets)
- Auto-adjustment factors, VWAP, returns pre-computation

### DataPartitioner (`core/backtest_engine/data_partitioner.py`)
- Strict IS / OOS physical split — OOS data is **never** visible to optimizer
- Configurable `oos_ratio` (default 0.3)
- Immutable `PartitionedDataset` (frozen slots)

### Realistic Backtester (`core/backtest_engine/simulator.py`)
- Long-Short and Decile portfolio modes
- Transaction costs (bps) + square-root-law slippage
- ADV liquidity cap (default 10%)
- Signal pipeline: truncation → decay → neutralization → delay

### Optuna Optimizer (`core/optimization_engine/alpha_optimizer.py`)
- Fitness = `Sharpe_IS + 0.5 × IC_IS - 0.1 × Turnover_IS`
- Searches: decay (0–10), truncation (0.01–0.10), neutralization (None/Sector)
- **Never** evaluates on OOS during search phase

### LangChain Agent (`core/ml_engine/` + `api/chat_router.py`)
- GPT-4o class LLM for DSL generation only (~200 tokens per call)
- Tools: `generate_alpha_dsl`, `run_optuna`, `run_backtest`, `save_alpha`
- Workflow A: Generate → Optimize → Validate → Anti-overfitting check
- Workflow B: Parse existing DSL → GP mutation → Optuna tune → Validate
- `FallbackOrchestrator` for zero-LLM mode (keyword heuristics)
- Session-based `ConversationMemory` (LRU eviction at 50 sessions)

---

## Frontend Modules

### State Management (Zustand)
```ts
interface WorkspaceState {
  activeView: 'CHAT' | 'COMPILER'
  editorDsl: string
  sessionId: string
  chatMessages: ChatMessage[]
  alphaHistory: AlphaRecord[]
  simulationResult: SimResult | null
  status: 'idle' | 'optimizing' | 'backtesting' | 'ready' | 'error'
  consoleLogs: string[]
  simConfig: SimulationConfig
}
```

### ECharts PnL Configuration
- IS region: blue `markArea` tint
- OOS region: red `markArea` + "UNSEEN DATA" label
- `markLine` dashed vertical at train/test split date
- Area gradient fill under cumulative PnL curve

### Anti-Overfitting UI
| Overfitting Score | Badge Color | Animation |
|---|---|---|
| < 40% | Emerald "Healthy" | static |
| 40–60% | Amber "Overfit Risk" | pulse |
| > 60% | Rose "Overfit!" | pulse |

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/alpha/simulate` | Manual backtest with `SimulationConfig` |
| `POST` | `/alpha/optimize` | Optuna auto-tune + IS/OOS evaluation |
| `POST` | `/api/chat` | Chat with LangChain agent |
| `GET`  | `/api/report/query` | Alpha history ledger |
| `POST` | `/api/backtest/run` | Basic backtest (no IS/OOS split) |
| `GET`  | `/docs` | Swagger UI |

---

## Configuration

Edit `backend/.env` (created automatically if missing):

```env
OPENAI_API_KEY=sk-...         # Required for LangChain agent
DATABASE_URL=sqlite:///./alphas.db
DEBUG=false
DEFAULT_N_TICKERS=20
DEFAULT_N_DAYS=120
```

Without `OPENAI_API_KEY`, the system uses `FallbackOrchestrator` (rule-based DSL generation, full backtest/optimize pipeline still functional).
