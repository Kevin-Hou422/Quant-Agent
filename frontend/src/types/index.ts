// ─── Shared domain types ──────────────────────────────────────────────────

export type ActiveView = 'CHAT' | 'COMPILER' | 'DATASET' | 'DASHBOARD'
export type Status = 'idle' | 'optimizing' | 'backtesting' | 'walkforward' | 'ready' | 'error'

export interface DatasetInfo {
  name:     string
  region:   string
  industry: string
  provider: string
  universe: string[]
  n_assets: number
  start:    string
}

export interface ChatSession {
  id:        string
  title:     string
  createdAt: string
}

export interface EditorTab {
  id:         string
  label:      string   // display name in tab bar
  dsl:        string   // current DSL content
  alphaId?:   number   // set when loaded from a saved alpha
  isModified: boolean  // unsaved edits since last save/load
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  dsl?: string | null
  metrics?: SimMetrics | null
  type?: 'message' | 'thought' | 'tool_output'
  timestamp: number
  isStreaming?: boolean
}

export interface AlphaRecord {
  id: number
  dsl: string
  hypothesis?: string
  sharpe?: number
  ic_ir?: number
  ann_turnover?: number
  ann_return?: number
  status?: string
  created_at?: string
}

export interface SimulationConfig {
  delay: number
  decay_window: number
  truncation_min_q: number
  truncation_max_q: number
  portfolio_mode: 'long_short' | 'decile'
  top_pct: number
  dataset:    string
  start_date: string
  end_date:   string
}

export interface SimMetrics {
  sharpe_ratio?: number | null
  annualized_return?: number | null
  annualized_vol?: number | null
  max_drawdown?: number | null
  mean_ic?: number | null
  ic_ir?: number | null
  ann_turnover?: number | null
  ic_decay_t1?: number | null
  ic_decay_t5?: number | null
  // Task 3.3: portfolio beta vs market benchmark (null when no benchmark provided)
  portfolio_beta?: number | null
  // Task 4.3: Deflated Sharpe — P(true Sharpe > 0) after multiple-testing correction
  deflated_sharpe?: number | null
}

/** Phase 5 (FE-5.1): one row of GET /api/alphas/dashboard */
export interface AlphaDashboardRow {
  alpha_id:        number
  dsl:             string
  status:          string
  sharpe:          number
  n_ic_days:       number
  latest_ic:       number | null
  latest_date:     string | null
  rolling_mean_ic: number | null
  rolling_ic_ir:   number | null
  consecutive_neg: number
  has_alert:       boolean
  allowed_next:    string[]
}

/** Phase 5 (FE-5.2): GET /api/alphas/{id}/ic_history */
export interface ICHistoryData {
  alpha_id: number
  points:   Array<{ date: string; realized_ic: number; realized_return: number }>
  rolling_mean_ic: number | null
  rolling_ic_ir:   number | null
}

/** Phase 5 (FE-5.3): GET /api/scheduler/status */
export interface SchedulerStatus {
  running: boolean
  jobs: Array<{ id: string; name: string; next_run: string | null; trigger: string }>
}

/** Task 4.1: market regime info from GET /api/regime */
export interface RegimeInfo {
  dataset:       string
  regime:        'bull' | 'bear' | 'sideways' | 'high_vol'
  as_of:         string
  counts:        Record<string, number>
  trend_window:  number
  vol_window:    number
  recent_labels: string[]
}

export interface SimResult {
  dsl: string
  is_metrics: SimMetrics & Record<string, unknown>
  oos_metrics?: (SimMetrics & Record<string, unknown>) | null
  overfitting_score: number
  is_overfit: boolean
  ic_decay: Record<string, number>
  best_config?: Record<string, unknown> | null
  n_trials_run?: number | null
  /** PnL daily returns from backtest (IS period) */
  pnl_is: number[]
  /** PnL daily returns from backtest (OOS period) */
  pnl_oos: number[]
  /** Last IS date (ISO string) */
  split_date?: string | null
}

/** Single entry from AlphaPool top-5 (Task 3.2) */
export interface PoolEntry {
  dsl:               string
  fitness:           number
  sharpe_is:         number
  sharpe_oos:        number
  turnover:          number
  overfitting_score: number
  generation:        number
}

/** Combined multi-Alpha signal metrics (Task 3.2 AlphaCombiner) */
export interface CombinedAlphaMetrics {
  n_alphas:           number
  weights:            Record<string, number>   // {dsl: ic_weighted_weight}
  combined_ic_ir:     number
  combined_mean_ic:   number
}

/** Response from /api/workflow/generate and /api/workflow/optimize */
export interface WorkflowResponse {
  workflow:          string
  best_dsl:          string
  metrics:           Record<string, unknown>   // {is_sharpe, oos_sharpe, is_return, is_turnover, is_ic, overfitting_score, is_overfit}
  evolution_log:     Array<{
    generation:       number
    population_size:  number
    best_fitness:     number
    best_oos_sharpe:  number
    best_dsl:         string
    mean_fitness:     number
  }>
  pool_top5:         PoolEntry[]
  best_config:       Record<string, unknown> | null
  seed_dsls:         string[]
  generations_run:   number
  explanation:       string
  pnl_is:            number[]
  pnl_oos:           number[]
  split_date:        string | null
  overfitting_score: number
  is_overfit:        boolean
  // IC Decay: {"t1": number, "t5": number} — populated by _add_workflow_pnl
  ic_decay?:         Record<string, number | null>
  // Task 3.2: multi-Alpha combined signal (present when pool has ≥ 2 entries)
  combined_metrics?: CombinedAlphaMetrics | null
}

export interface WalkForwardFoldReport {
  fold_idx:     number
  is_start:     string
  is_end:       string
  oos_start:    string
  oos_end:      string
  is_days:      number
  oos_days:     number
  is_sharpe:    number
  oos_sharpe:   number
  oos_maxdd:    number
  oos_turnover: number
  oos_ic_ir:    number
  overfitting:  number
}

export interface WalkForwardResult {
  dsl:              string
  n_folds:          number
  mean_oos_sharpe:  number
  std_oos_sharpe:   number
  min_oos_sharpe:   number
  pct_positive:     number
  mean_overfitting: number
  fold_reports:     WalkForwardFoldReport[]
}

export interface DatasetHealth {
  name:          string
  overall_score: number
  n_tickers:     number
  n_dates:       number
  n_gaps:        number
  n_spikes:      number
  n_zero_volume: number
  mean_nan_pct:  number
  notes:         string[]
}

export interface BacktestRunResponse {
  dsl: string
  report: {
    sharpe_ratio?: number
    annualized_return?: number
    max_drawdown?: number
    mean_ic?: number
    ic_ir?: number
    ann_turnover?: number
    net_returns?: Record<string, number>
  }
}
