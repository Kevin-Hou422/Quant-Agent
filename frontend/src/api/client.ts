import axios from 'axios'
import type {
  SimulationConfig, SimResult, AlphaRecord, BacktestRunResponse, ChatSession,
  WorkflowResponse,
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

// ── Workflow A: hypothesis → GP-evolved alpha ─────────────────────────────
export const apiWorkflowGenerate = (hypothesis: string, nDays = 252) =>
  http.post<WorkflowResponse>('/workflow/generate', {
    hypothesis,
    n_tickers:     20,
    n_days:        nDays,
    n_generations: 7,
    pop_size:      20,
    n_optuna:      10,
    n_seed_dsls:   12,
    oos_ratio:     0.3,
  })

// ── Workflow B: DSL → GP-evolved + Optuna-tuned alpha ─────────────────────
export const apiWorkflowOptimize = (dsl: string, nDays = 252) =>
  http.post<WorkflowResponse>('/workflow/optimize', {
    dsl,
    n_tickers:     20,
    n_days:        nDays,
    n_generations: 7,
    pop_size:      20,
    n_optuna:      10,
    n_mutations:   8,
    oos_ratio:     0.3,
  })

// ── Basic backtest (Phase 1 endpoint) ─────────────────────────────────────
export const apiBacktest = (dsl: string) =>
  http.post<BacktestRunResponse>('/backtest/run', { dsl, n_tickers: 20, n_days: 252 })

// ── Alpha Ledger (history) ────────────────────────────────────────────────
export const apiFetchAlphaHistory = (limit = 30) =>
  http.get<{ total: number; records: AlphaRecord[] }>(`/report/query?limit=${limit}`)

// ── Alpha Save (persist manual backtest result) ───────────────────────────
export const apiSaveAlpha = (body: {
  dsl: string
  hypothesis?: string
  sharpe?: number
  ic_ir?: number
  ann_turnover?: number
  ann_return?: number
}) =>
  http.post<{ id: number; status: string }>('/alpha/save', body)

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

// ── SSE Streaming Workflows ───────────────────────────────────────────────

type SSEEvent =
  | { type: 'text';  text: string }
  | { type: 'ping' }
  | { type: 'done';  result: Record<string, unknown> }
  | { type: 'error'; message: string }

export function streamChat(
  message:    string,
  sessionId:  string,
  onEvent:    (e: SSEEvent) => void,
  signal?:    AbortSignal,
): Promise<void> {
  return fetch('/api/chat/stream', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ message, session_id: sessionId }),
    signal,
  }).then(async (res) => {
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const reader  = res.body!.getReader()
    const decoder = new TextDecoder()
    let   buf     = ''
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      const lines = buf.split('\n')
      buf = lines.pop() ?? ''
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try { onEvent(JSON.parse(line.slice(6)) as SSEEvent) } catch { /* ignore */ }
        }
      }
    }
  })
}

export function streamWorkflowOptimize(
  dsl:     string,
  onEvent: (e: SSEEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  return fetch('/api/workflow/optimize/stream', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({
      dsl,
      n_tickers: 20, n_days: 252, n_generations: 7,
      pop_size: 20, n_optuna: 10, n_mutations: 8, oos_ratio: 0.3,
    }),
    signal,
  }).then(async (res) => {
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const reader  = res.body!.getReader()
    const decoder = new TextDecoder()
    let   buf     = ''
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      const lines = buf.split('\n')
      buf = lines.pop() ?? ''
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try { onEvent(JSON.parse(line.slice(6)) as SSEEvent) } catch { /* ignore */ }
        }
      }
    }
  })
}

export function streamWorkflowGenerate(
  hypothesis: string,
  onEvent:    (e: SSEEvent) => void,
  signal?:    AbortSignal,
): Promise<void> {
  return fetch('/api/workflow/generate/stream', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({
      hypothesis,
      n_tickers: 20, n_days: 252, n_generations: 7,
      pop_size: 20, n_optuna: 10, n_seed_dsls: 12, oos_ratio: 0.3,
    }),
    signal,
  }).then(async (res) => {
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const reader  = res.body!.getReader()
    const decoder = new TextDecoder()
    let   buf     = ''
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      const lines = buf.split('\n')
      buf = lines.pop() ?? ''
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try { onEvent(JSON.parse(line.slice(6)) as SSEEvent) } catch { /* ignore */ }
        }
      }
    }
  })
}
