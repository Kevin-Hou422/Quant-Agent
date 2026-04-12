// ─── Shared domain types ──────────────────────────────────────────────────

export type ActiveView = 'CHAT' | 'COMPILER'
export type Status = 'idle' | 'optimizing' | 'backtesting' | 'ready' | 'error'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  dsl?: string | null
  metrics?: SimMetrics | null
  type?: 'message' | 'thought' | 'tool_output'
  timestamp: number
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
}

export interface SimResult {
  dsl: string
  is_metrics: SimMetrics
  oos_metrics?: SimMetrics | null
  // Aliases used by analysis components
  is_report?: SimMetrics | null
  oos_report?: SimMetrics | null
  overfitting_score: number
  is_overfit: boolean
  ic_decay: Record<string, number>
  best_config?: Record<string, unknown> | null
  n_trials_run?: number | null
  // PnL series (optional)
  pnl_dates?: string[]
  pnl_is?: number[]
  pnl_oos?: number[]
  split_date?: string
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
