import { useWorkspaceStore } from '../store/workspaceStore'
import {
  apiChat, apiSimulate, apiOptimize, apiFetchAlphaHistory, apiSaveAlpha,
  apiCreateSession, apiListSessions, apiGetSession,
} from '../api/client'
import type { ChatSession } from '../types'

// ── Progress log sequences ────────────────────────────────────────────────────

const BACKTEST_STEPS = [
  '[System] Parsing DSL expression...',
  '[System] Compiling AST to signal tree...',
  '[System] Computing cross-sectional signals (IS 2020-2022)...',
  '[System] Running IS portfolio simulation...',
  '[System] Computing OOS signals (2023-2024)...',
  '[System] Running OOS portfolio simulation...',
  '[System] Evaluating anti-overfitting metrics...',
]

const OPTIMIZE_STEPS = [
  '[GP] Initializing population (12 individuals)...',
  '[GP] Generation 1/4 — evaluating fitness scores...',
  '[GP] Generation 2/4 — AST mutation + subtree crossover...',
  '[GP] Generation 3/4 — tournament selection + diversity filter...',
  '[GP] Generation 4/4 — elite preservation...',
  '[Optuna] Fine-tuning winner structure parameters...',
  '[System] Running final IS+OOS validation...',
]

function startProgressStream(steps: string[], appendLog: (l: string) => void, ms = 750): () => void {
  let i = 0
  const timer = setInterval(() => {
    if (i < steps.length) appendLog(steps[i++])
    else clearInterval(timer)
  }, ms)
  return () => clearInterval(timer)
}

function classifyError(err: any): string {
  const status = err?.response?.status
  const detail = err?.response?.data?.detail ?? err?.message ?? 'Unknown error'
  if (status === 400) return `[Syntax Error] Invalid DSL formula — ${detail}`
  if (status === 422) return `[Syntax Error] Validation failed — ${detail}`
  if (status === 500) return `[ERROR] Server error — ${detail}`
  return `[ERROR] ${detail}`
}

// ── Hook ─────────────────────────────────────────────────────────────────────

export function useQuantWorkspace() {
  const store = useWorkspaceStore()

  // ── Session management ────────────────────────────────────────────────
  const loadSessions = async () => {
    try {
      const res = await apiListSessions()
      const sessions: ChatSession[] = res.data.sessions.map((s) => ({
        id:        s.session_id,
        title:     s.title,
        createdAt: s.created_at,
      }))
      store.setSessions(sessions)
    } catch { /* non-critical */ }
  }

  const newSession = async () => {
    try {
      const res = await apiCreateSession('New Research')
      const s: ChatSession = {
        id:        res.data.session_id,
        title:     res.data.title,
        createdAt: res.data.created_at,
      }
      store.setSessionId(s.id)
      store.setMessages([])
      store.addSession(s)
    } catch {
      const id = Math.random().toString(36).slice(2)
      store.setSessionId(id)
      store.setMessages([])
      store.addSession({ id, title: 'New Research', createdAt: new Date().toISOString() })
    }
  }

  const switchSession = async (sessionId: string) => {
    if (sessionId === store.sessionId) return
    store.setSessionId(sessionId)
    store.setMessages([])
    try {
      const res = await apiGetSession(sessionId)
      const msgs = res.data.messages.map((m) => ({
        id:        String(m.id),
        role:      m.role as 'user' | 'assistant',
        content:   m.content,
        timestamp: new Date(m.created_at).getTime(),
        type:      'message' as const,
      }))
      store.setMessages(msgs)
    } catch { /* start fresh */ }
  }

  // ── Chat ──────────────────────────────────────────────────────────────
  const sendChat = async (text: string) => {
    if (!text.trim()) return
    store.addMessage({ role: 'user', content: text })
    store.setStatus('optimizing')
    try {
      const res = await apiChat(text, store.sessionId)
      const { reply, dsl, metrics } = res.data
      store.addMessage({ role: 'assistant', content: reply, dsl, metrics: metrics as any, type: 'message' })
      if (dsl) store.setEditorDsl(dsl)
      store.setStatus('ready')
    } catch (err: any) {
      store.addMessage({
        role:    'assistant',
        content: err?.response?.status === 400
          ? 'DSL syntax error — please check your formula and try again.'
          : `Error: ${err?.message ?? 'Unknown error'}`,
        type: 'tool_output',
      })
      store.setStatus('error')
    }
  }

  // ── Run backtest ──────────────────────────────────────────────────────
  const runBacktest = async () => {
    const dsl = store.editorDsl.trim()
    if (!dsl) return
    store.setStatus('backtesting')
    store.clearLogs()
    store.appendLog(`[Backtest] DSL: ${dsl}`)

    const cancelStream = startProgressStream(BACKTEST_STEPS, store.appendLog)
    try {
      const res = await apiSimulate(dsl, store.simConfig)
      cancelStream()
      store.setSimulationResult(res.data)

      const is  = res.data.is_metrics
      const oos = res.data.oos_metrics
      store.appendLog(`[Backtest] IS  Sharpe : ${is?.sharpe_ratio?.toFixed(4) ?? 'N/A'}`)
      store.appendLog(`[Backtest] OOS Sharpe : ${oos?.sharpe_ratio?.toFixed(4) ?? 'N/A'}`)
      store.appendLog(`[Backtest] Overfitting: ${res.data.overfitting_score?.toFixed(3)}`)
      store.appendLog(`[Backtest] PnL pts    : IS=${res.data.pnl_is?.length ?? 0}  OOS=${res.data.pnl_oos?.length ?? 0}`)
      store.appendLog(res.data.is_overfit ? '[WARN] Overfitting detected!' : '[OK] Passed anti-overfitting check.')

      // Auto-save to Alpha Ledger
      const activeTab = store.editorTabs.find((t) => t.id === store.activeTabId)
      try {
        await apiSaveAlpha({
          dsl,
          hypothesis:   activeTab?.label ?? dsl.slice(0, 40),
          sharpe:       is?.sharpe_ratio      ?? 0,
          ic_ir:        is?.ic_ir             ?? 0,
          ann_turnover: is?.ann_turnover      ?? 0,
          ann_return:   is?.annualized_return ?? 0,
        })
        store.appendLog('[OK] Alpha saved to Ledger.')
      } catch { /* save failure is non-fatal */ }

      store.setStatus('ready')
      await loadHistory()
    } catch (err: any) {
      cancelStream()
      store.appendLog(classifyError(err))
      store.setStatus('error')
    }
  }

  // ── AI Optimize ───────────────────────────────────────────────────────
  const runOptimize = async () => {
    const dsl = store.editorDsl.trim()
    if (!dsl) return
    store.setStatus('optimizing')
    store.clearLogs()
    store.appendLog(`[GP] Seed DSL: ${dsl}`)

    const cancelStream = startProgressStream(OPTIMIZE_STEPS, store.appendLog, 900)
    try {
      const res = await apiOptimize(dsl, 20)
      cancelStream()
      store.setSimulationResult(res.data)

      if (res.data.best_config) {
        store.appendLog(`[Optuna] Best config: ${JSON.stringify(res.data.best_config)}`)
      }
      const is  = res.data.is_metrics
      const oos = res.data.oos_metrics
      store.appendLog(`[GP] IS  Sharpe : ${is?.sharpe_ratio?.toFixed(4) ?? 'N/A'}`)
      store.appendLog(`[GP] OOS Sharpe : ${oos?.sharpe_ratio?.toFixed(4) ?? 'N/A'}`)
      store.appendLog(res.data.is_overfit ? '[WARN] Overfitting in optimized result!' : '[OK] Optimization passed anti-overfitting check.')

      // Auto-save optimized alpha
      const activeTab = store.editorTabs.find((t) => t.id === store.activeTabId)
      try {
        await apiSaveAlpha({
          dsl,
          hypothesis:   `Optimized: ${activeTab?.label ?? dsl.slice(0, 30)}`,
          sharpe:       is?.sharpe_ratio      ?? 0,
          ic_ir:        is?.ic_ir             ?? 0,
          ann_turnover: is?.ann_turnover      ?? 0,
          ann_return:   is?.annualized_return ?? 0,
        })
        store.appendLog('[OK] Optimized alpha saved to Ledger.')
      } catch { /* non-fatal */ }

      store.setStatus('ready')
      await loadHistory()
    } catch (err: any) {
      cancelStream()
      store.appendLog(classifyError(err))
      store.setStatus('error')
    }
  }

  // ── Alpha history ─────────────────────────────────────────────────────
  const loadHistory = async () => {
    try {
      const res = await apiFetchAlphaHistory(30)
      store.setAlphaHistory(res.data.records)
    } catch { /* silently ignore */ }
  }

  return {
    sendChat, runBacktest, runOptimize, loadHistory,
    loadSessions, newSession, switchSession,
    store,
  }
}
