import { useWorkspaceStore } from '../store/workspaceStore'
import {
  apiChat, apiSimulate, apiWorkflowOptimize, apiFetchAlphaHistory, apiSaveAlpha,
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
  '[GP] Parsing DSL → AST node tree...',
  '[GP] Diagnosing initial quality (IS+OOS backtest)...',
  '[GP] Expanding population: original + mutations + targeted variants...',
  '[GP] Generation 1/7 — evaluating fitness (OOS Sharpe - overfitting penalty)...',
  '[GP] Generation 2/7 — AST mutation + subtree crossover...',
  '[GP] Generation 3/7 — tournament selection + diversity filter (corr < 0.9)...',
  '[GP] Generation 4/7 — adaptive mutation weights from pool diagnostics...',
  '[GP] Generation 5/7 — elite preservation + random exploration...',
  '[GP] Generation 6/7 — AlphaPool diversity-filtered accumulation...',
  '[GP] Generation 7/7 — selecting best structure from pool...',
  '[Optuna] Fine-tuning execution parameters (delay, decay, truncation)...',
  '[System] Running final IS+OOS validation with tuned config...',
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

  // ── AI Optimize (Workflow B: GP evolution + Optuna fine-tuning) ──────────
  const runOptimize = async () => {
    const dsl = store.editorDsl.trim()
    if (!dsl) return
    store.setStatus('optimizing')
    store.clearLogs()
    store.appendLog(`[GP] Input DSL: ${dsl}`)

    const cancelStream = startProgressStream(OPTIMIZE_STEPS, store.appendLog, 850)
    try {
      const res = await apiWorkflowOptimize(dsl)
      cancelStream()

      const wf = res.data

      // Log GP evolution trace
      for (const gen of (wf.evolution_log ?? [])) {
        store.appendLog(
          `[GP] Gen ${gen.generation}/${wf.generations_run} | ` +
          `pop=${gen.population_size} | ` +
          `best_fitness=${gen.best_fitness.toFixed(4)} | ` +
          `oos_sharpe=${gen.best_oos_sharpe.toFixed(4)} | ` +
          `dsl=${gen.best_dsl.slice(0, 60)}`
        )
      }

      // Log pool top-5
      if (wf.pool_top5?.length) {
        store.appendLog(`[GP] AlphaPool top-${wf.pool_top5.length}:`)
        wf.pool_top5.forEach((e: any, i: number) => {
          store.appendLog(`  #${i + 1} fitness=${(e.fitness ?? 0).toFixed(4)}  oos=${(e.sharpe_oos ?? 0).toFixed(4)}  ${(e.dsl ?? '').slice(0, 50)}`)
        })
      }

      // Best alpha
      store.appendLog(`[GP] Best DSL: ${wf.best_dsl}`)
      if (wf.best_config) {
        store.appendLog(`[Optuna] Tuned config: ${JSON.stringify(wf.best_config)}`)
      }

      const m = wf.metrics as Record<string, any>
      store.appendLog(`[GP] IS  Sharpe : ${m?.is_sharpe != null  ? Number(m.is_sharpe ).toFixed(4) : 'N/A'}`)
      store.appendLog(`[GP] OOS Sharpe : ${m?.oos_sharpe != null ? Number(m.oos_sharpe).toFixed(4) : 'N/A'}`)
      store.appendLog(wf.is_overfit ? '[WARN] Overfitting detected in GP result!' : '[OK] GP result passed anti-overfitting check.')
      store.appendLog(`[GP] Explanation: ${wf.explanation}`)

      // Adapt WorkflowResponse → SimResult format for the chart (RightPane)
      const simResult = {
        dsl:               wf.best_dsl,
        is_metrics:        {
          sharpe_ratio:      m?.is_sharpe      ?? null,
          annualized_return: m?.is_return      ?? null,
          ann_turnover:      m?.is_turnover    ?? null,
          ic_ir:             m?.is_ic          ?? null,
          mean_ic:           m?.is_ic          ?? null,
        },
        oos_metrics:       m?.oos_sharpe != null
          ? { sharpe_ratio: m.oos_sharpe }
          : null,
        overfitting_score: wf.overfitting_score,
        is_overfit:        wf.is_overfit,
        ic_decay:          {},
        best_config:       wf.best_config ?? null,
        n_trials_run:      null,
        pnl_is:            wf.pnl_is  ?? [],
        pnl_oos:           wf.pnl_oos ?? [],
        split_date:        wf.split_date ?? null,
      }
      store.setSimulationResult(simResult)

      // Update editor to show the GP-evolved best DSL in a new tab
      if (wf.best_dsl && wf.best_dsl !== dsl) {
        store.newEmptyTab()
        store.setEditorDsl(wf.best_dsl)
        store.appendLog(`[GP] Opened best DSL in new editor tab.`)
      }

      // Auto-save
      try {
        await apiSaveAlpha({
          dsl:          wf.best_dsl,
          hypothesis:   `GP-evolved: ${dsl.slice(0, 40)}`,
          sharpe:       Number(m?.is_sharpe)   || 0,
          ic_ir:        Number(m?.is_ic)       || 0,
          ann_turnover: Number(m?.is_turnover) || 0,
          ann_return:   Number(m?.is_return)   || 0,
        })
        store.appendLog('[OK] GP-evolved alpha saved to Ledger.')
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
