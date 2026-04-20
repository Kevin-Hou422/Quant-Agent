import axios from 'axios'
import type {
  SimulationConfig, SimResult, AlphaRecord, BacktestRunResponse, ChatSession,
} from '../types'

const http = axios.create({ baseURL: '/api', timeout: 120_000 })

// ── Chat ──────────────────────────────────────────────────────────────────
export const apiChat = (message: string, sessionId: string) =>
  http.post<{ session_id: string; reply: string; dsl?: string | null; metrics?: Record<string, unknown> | null }>(
    '/chat',
    { message, session_id: sessionId },
  )

// ── Alpha Simulate (Phase 2 endpoint) ────────────────────────────────────
export const apiSimulate = (
  dsl: string,
  config: SimulationConfig,
  nTickers = 20,
  nDays = 252,
  oos_ratio = 0.3,
) =>
  http.post<SimResult>('/alpha/simulate', {
    dsl, config, n_tickers: nTickers, n_days: nDays, oos_ratio,
  })

// ── Alpha Optimize (Phase 2 Optuna endpoint) ──────────────────────────────
export const apiOptimize = (dsl: string, nTrials = 20) =>
  http.post<SimResult>('/alpha/optimize', {
    dsl,
    search_space: { portfolio_modes: ['long_short'] },
    n_trials: nTrials,
    n_tickers: 20,
    n_days: 252,
    oos_ratio: 0.3,
  })

// ── Basic backtest (Phase 1 endpoint) ─────────────────────────────────────
export const apiBacktest = (dsl: string) =>
  http.post<BacktestRunResponse>('/backtest/run', { dsl, n_tickers: 20, n_days: 252 })

// ── Alpha Ledger (history) ────────────────────────────────────────────────
export const apiFetchAlphaHistory = (limit = 30) =>
  http.get<{ total: number; records: AlphaRecord[] }>(`/report/query?limit=${limit}`)

// ── Chat Session Management ───────────────────────────────────────────────
export const apiCreateSession = (title: string) =>
  http.post<{ session_id: string; title: string; created_at: string }>(
    '/chat/sessions',
    { title },
  )

export const apiListSessions = () =>
  http.get<{ sessions: Array<{ session_id: string; title: string; created_at: string }>; count: number }>(
    '/chat/sessions',
  )

export const apiGetSession = (sessionId: string) =>
  http.get<{
    session_id: string
    title:      string
    created_at: string
    messages:   Array<{ id: number; role: string; content: string; created_at: string }>
  }>(`/chat/sessions/${sessionId}`)
