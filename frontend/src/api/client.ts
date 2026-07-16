import axios from 'axios'
import type {
  SimulationConfig, SimResult, AlphaRecord, BacktestRunResponse,
  WorkflowResponse, DatasetInfo, WalkForwardResult, DatasetHealth,
  RegimeInfo,
} from '../types'

const http = axios.create({ baseURL: '/api', timeout: 120_000 })

// Translate 429 / 408 into user-readable messages so the UI can toast them
http.interceptors.response.use(
  (r) => r,
  (err) => {
    const status = err?.response?.status
    if (status === 429)
      return Promise.reject(new Error('GP 任务正在运行，请稍后重试'))
    if (status === 408)
      return Promise.reject(new Error('GP 任务超时，请缩小种群规模后重试'))
    return Promise.reject(err)
  },
)

// ── Datasets ──────────────────────────────────────────────────────────────
export const apiFetchDatasets = () =>
  http.get<{ datasets: DatasetInfo[]; total: number }>('/datasets')

export const apiFetchDatasetHealth = (name: string, start: string, end: string) =>
  http.get<DatasetHealth>(`/datasets/${encodeURIComponent(name)}/health`, {
    params: { start, end },
    timeout: 60_000,   // health check loads real data — may be slow on first call
  })

// ── Market Regime (Task 4.1 / FE-4.1) ─────────────────────────────────────
export const apiFetchRegime = (dataset: string, start: string, end: string) =>
  http.get<RegimeInfo>('/regime', {
    params: { dataset_name: dataset, start, end },
    timeout: 60_000,   // first call loads the dataset — may be slow
  })

// ── Walk-Forward Backtest ────────────────────────────────────────────────
export const apiWalkForwardBacktest = (
  dsl:          string,
  config:       SimulationConfig,
  nSplits      = 5,
  embargoDays  = 20,
) =>
  http.post<WalkForwardResult>('/backtest/walk_forward', {
    dsl,
    dataset_name:  config.dataset    || 'us_tech_large',
    dataset_start: config.start_date || '2020-01-01',
    dataset_end:   config.end_date   || '2024-01-01',
    n_splits:      nSplits,
    embargo_days:  embargoDays,
    portfolio_mode: config.portfolio_mode,
    delay:         config.delay,
  }, { timeout: 300_000 })

// ── Chat ──────────────────────────────────────────────────────────────────
export const apiChat = (message: string, sessionId: string) =>
  http.post<{ session_id: string; reply: string; dsl?: string | null; metrics?: Record<string, unknown> | null }>(
    '/chat',
    { message, session_id: sessionId },
  )

// ── Alpha Simulate (Phase 2 endpoint) ────────────────────────────────────
export const apiSimulate = (dsl: string, config: SimulationConfig, oos_ratio = 0.3) =>
  http.post<SimResult>('/alpha/simulate', {
    dsl,
    config,
    dataset_name:  config.dataset    || 'us_tech_large',
    dataset_start: config.start_date || '2020-01-01',
    dataset_end:   config.end_date   || '2024-01-01',
    oos_ratio,
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
export const apiWorkflowGenerate = (
  hypothesis: string,
  dataset = 'us_tech_large',
  startDate = '2020-01-01',
  endDate   = '2024-01-01',
) =>
  http.post<WorkflowResponse>('/workflow/generate', {
    hypothesis,
    dataset_name:  dataset,
    dataset_start: startDate,
    dataset_end:   endDate,
    n_generations: 7,
    pop_size:      20,
    n_optuna:      10,
    n_seed_dsls:   12,
    oos_ratio:     0.3,
  })

// ── Workflow B: DSL → GP-evolved + Optuna-tuned alpha ─────────────────────
export const apiWorkflowOptimize = (
  dsl: string,
  dataset = 'us_tech_large',
  startDate = '2020-01-01',
  endDate   = '2024-01-01',
) =>
  http.post<WorkflowResponse>('/workflow/optimize', {
    dsl,
    dataset_name:  dataset,
    dataset_start: startDate,
    dataset_end:   endDate,
    n_generations: 7,
    pop_size:      20,
    n_optuna:      10,
    n_mutations:   8,
    oos_ratio:     0.3,
  })

// ── Basic backtest (Phase 1 endpoint) ─────────────────────────────────────
export const apiBacktest = (dsl: string, dataset = 'us_tech_large') =>
  http.post<BacktestRunResponse>('/backtest/run', {
    dsl,
    dataset_name: dataset,
  })

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

export const apiRenameSession = (sessionId: string, title: string) =>
  http.patch<{ session_id: string; title: string; created_at: string }>(
    `/chat/sessions/${sessionId}`,
    { title },
  )

export const apiDeleteSession = (sessionId: string) =>
  http.delete(`/chat/sessions/${sessionId}`)

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
  dsl:       string,
  onEvent:   (e: SSEEvent) => void,
  dataset?:  string,
  startDate?: string,
  endDate?:   string,
  signal?:   AbortSignal,
): Promise<void> {
  return fetch('/api/workflow/optimize/stream', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({
      dsl,
      dataset_name:  dataset    || 'us_tech_large',
      dataset_start: startDate  || '2020-01-01',
      dataset_end:   endDate    || '2024-01-01',
      n_generations: 7, pop_size: 20, n_optuna: 10, n_mutations: 8, oos_ratio: 0.3,
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
  dataset?:   string,
  startDate?: string,
  endDate?:   string,
  signal?:    AbortSignal,
): Promise<void> {
  return fetch('/api/workflow/generate/stream', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({
      hypothesis,
      dataset_name:  dataset    || 'us_tech_large',
      dataset_start: startDate  || '2020-01-01',
      dataset_end:   endDate    || '2024-01-01',
      n_generations: 7, pop_size: 20, n_optuna: 10, n_seed_dsls: 12, oos_ratio: 0.3,
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
