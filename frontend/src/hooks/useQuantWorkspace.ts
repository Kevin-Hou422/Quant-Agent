import { useWorkspaceStore } from '../store/workspaceStore'
import {
  apiSimulate, apiFetchAlphaHistory, apiSaveAlpha,
  apiCreateSession, apiListSessions, apiGetSession,
  apiRenameSession, apiDeleteSession,
  streamWorkflowOptimize, streamChat,
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

  // ── Typing-effect helper ──────────────────────────────────────────────
  const typeIntoMessage = async (id: string, text: string, msPerChar = 14) => {
    for (const char of text) {
      store.appendToStreamingMessage(id, char)
      await new Promise<void>((r) => setTimeout(r, msPerChar))
    }
  }

  // ── Session init (called once on app mount) ───────────────────────────
  /**
   * 1. Fetch all sessions from DB
   * 2. If the stored sessionId is in the list → restore its messages
   * 3. If not → use the most-recent session (or create a fresh one)
   */
  const initSessions = async () => {
    try {
      const res = await apiListSessions()
      const sessions: ChatSession[] = res.data.sessions.map((s) => ({
        id:        s.session_id,
        title:     s.title,
        createdAt: s.created_at,
      }))
      store.setSessions(sessions)

      const stored = store.sessionId
      const found  = sessions.find((s) => s.id === stored)

      if (found) {
        // Restore messages for the stored session
        try {
          const r = await apiGetSession(stored)
          const msgs = r.data.messages.map((m) => ({
            id:        String(m.id),
            role:      m.role as 'user' | 'assistant',
            content:   m.content,
            timestamp: new Date(m.created_at).getTime(),
            type:      'message' as const,
          }))
          store.setMessages(msgs)
        } catch { /* start fresh */ }
      } else if (sessions.length > 0) {
        // Stored id not in DB → switch to most-recent session
        const latest = sessions[0]
        store.setSessionId(latest.id)
        try {
          const r = await apiGetSession(latest.id)
          const msgs = r.data.messages.map((m) => ({
            id:        String(m.id),
            role:      m.role as 'user' | 'assistant',
            content:   m.content,
            timestamp: new Date(m.created_at).getTime(),
            type:      'message' as const,
          }))
          store.setMessages(msgs)
        } catch { /* start fresh */ }
      } else {
        // No sessions at all → create one
        await _createFreshSession()
      }
    } catch {
      // Network/server down — create an in-memory session so the UI works
      await _createFreshSession()
    }
  }

  const _createFreshSession = async () => {
    try {
      const res = await apiCreateSession('New Chat')
      const s: ChatSession = {
        id:        res.data.session_id,
        title:     res.data.title,
        createdAt: res.data.created_at,
      }
      store.setSessionId(s.id)
      store.setMessages([])
      store.addSession(s)
    } catch {
      // Offline fallback: keep the random id, messages stay empty
      store.setMessages([])
    }
  }

  // ── Load sessions list ────────────────────────────────────────────────
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

  // ── Create new session ────────────────────────────────────────────────
  const newSession = async () => {
    try {
      const res = await apiCreateSession('New Chat')
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
      store.addSession({ id, title: 'New Chat', createdAt: new Date().toISOString() })
    }
  }

  // ── Switch session ────────────────────────────────────────────────────
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

  // ── Rename session ────────────────────────────────────────────────────
  const renameSession = async (id: string, title: string) => {
    if (!title.trim()) return
    store.updateSessionTitle(id, title.trim())   // optimistic
    try {
      await apiRenameSession(id, title.trim())
    } catch {
      await loadSessions()  // rollback: reload from server
    }
  }

  // ── Delete session ────────────────────────────────────────────────────
  const deleteSession = async (id: string) => {
    store.removeSession(id)                       // optimistic

    // If deleting the active session → switch to next available or create new
    if (store.sessionId === id) {
      const remaining = store.sessions.filter((s) => s.id !== id)
      if (remaining.length > 0) {
        await switchSession(remaining[0].id)
      } else {
        await _createFreshSession()
      }
    }

    try {
      await apiDeleteSession(id)
    } catch {
      await loadSessions()  // rollback
    }
  }

  // ── Chat (streaming) ──────────────────────────────────────────────────
  const sendChat = async (text: string) => {
    if (!text.trim()) return

    // Detect first message BEFORE addMessage increments the count
    const isFirstMessage = store.chatMessages.length === 0

    store.addMessage({ role: 'user', content: text })
    store.setStatus('optimizing')

    const streamId = Math.random().toString(36).slice(2)
    store.startStreamingMessage(streamId)

    // Sequential text queue — same pattern as runOptimize
    const textQueue: string[] = []
    let   typing = false

    const flushQueue = async () => {
      if (typing) return
      typing = true
      while (textQueue.length > 0) {
        const chunk = textQueue.shift()!
        await typeIntoMessage(streamId, chunk + '\n', 10)
      }
      typing = false
    }

    const enqueueText = (line: string) => {
      textQueue.push(line)
      flushQueue()
    }

    try {
      let finalResult: any = null

      await streamChat(text, store.sessionId, (event) => {
        if (event.type === 'text') {
          enqueueText(event.text)
          if (
            event.text.startsWith('[GP]')       ||
            event.text.startsWith('[Optuna]')   ||
            event.text.startsWith('[Workflow')  ||
            event.text.startsWith('[Diagnose]')
          ) {
            store.appendLog(event.text.split('\n')[0])
          }
        } else if (event.type === 'done') {
          finalResult = event.result
        } else if (event.type === 'error') {
          enqueueText(`[ERROR] ${event.message}`)
          store.appendLog(`[ERROR] ${event.message}`)
        }
      })

      // Wait for remaining typing to finish
      await new Promise<void>((resolve) => {
        const check = setInterval(() => {
          if (!typing && textQueue.length === 0) { clearInterval(check); resolve() }
        }, 50)
      })

      if (finalResult) {
        const { dsl, metrics } = finalResult as any
        store.finalizeStreamingMessage(streamId, { dsl, metrics, type: 'message' })
        if (dsl) store.setEditorDsl(dsl)
        store.setStatus('ready')
      } else {
        store.finalizeStreamingMessage(streamId)
        store.setStatus('error')
      }

      // Auto-name session on first message, then refresh the sessions list
      const sessionId = store.sessionId
      if (isFirstMessage) {
        const title = text.length > 40 ? text.slice(0, 40) + '…' : text
        store.updateSessionTitle(sessionId, title)     // optimistic update
        try { await apiRenameSession(sessionId, title) } catch { /* non-fatal */ }
      }

      // Ensure the current session appears in the sidebar list
      await loadSessions()

    } catch (err: any) {
      store.finalizeStreamingMessage(streamId, {
        content: `Error: ${err?.message ?? 'Unknown error'}`,
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

  // ── AI Optimize (Workflow B: GP evolution + Optuna fine-tuning, SSE) ─
  const runOptimize = async () => {
    const dsl = store.editorDsl.trim()
    if (!dsl) return

    store.setActiveView('CHAT')
    store.setStatus('optimizing')
    store.clearLogs()
    store.appendLog(`[GP] Input DSL: ${dsl}`)

    const streamId = Math.random().toString(36).slice(2)
    store.startStreamingMessage(streamId)

    const textQueue: string[] = []
    let   typing = false

    const flushQueue = async () => {
      if (typing) return
      typing = true
      while (textQueue.length > 0) {
        const chunk = textQueue.shift()!
        await typeIntoMessage(streamId, chunk + '\n', 10)
      }
      typing = false
    }

    const enqueueText = (line: string) => {
      textQueue.push(line)
      flushQueue()
    }

    try {
      let finalWf: any = null

      await streamWorkflowOptimize(dsl, (event) => {
        if (event.type === 'text') {
          enqueueText(event.text)
          if (event.text.startsWith('[GP]') || event.text.startsWith('[Optuna]')) {
            store.appendLog(event.text.split('\n')[0])
          }
        } else if (event.type === 'done') {
          finalWf = event.result
        } else if (event.type === 'error') {
          enqueueText(`[ERROR] ${event.message}`)
          store.appendLog(`[ERROR] ${event.message}`)
        }
      })

      await new Promise<void>((resolve) => {
        const check = setInterval(() => {
          if (!typing && textQueue.length === 0) { clearInterval(check); resolve() }
        }, 50)
      })

      if (finalWf) {
        const wf = finalWf
        const m  = wf.metrics as Record<string, any>

        store.finalizeStreamingMessage(streamId, {
          dsl:     wf.best_dsl,
          metrics: {
            sharpe_ratio:      m?.is_sharpe      ?? null,
            annualized_return: m?.is_return      ?? null,
            ic_ir:             m?.is_ic          ?? null,
          } as any,
          type: 'message',
        })

        for (const gen of (wf.evolution_log ?? [])) {
          store.appendLog(
            `[GP] Gen ${gen.generation}/${wf.generations_run} | ` +
            `fitness=${(gen.best_fitness as number).toFixed(4)} | ` +
            `oos_sharpe=${(gen.best_oos_sharpe as number).toFixed(4)}`
          )
        }
        if (wf.best_config) {
          store.appendLog(`[Optuna] Tuned config: ${JSON.stringify(wf.best_config)}`)
        }
        store.appendLog(wf.is_overfit ? '[WARN] Overfitting detected!' : '[OK] GP result passed anti-overfitting check.')

        store.setSimulationResult({
          dsl:               wf.best_dsl,
          is_metrics:        {
            sharpe_ratio:      m?.is_sharpe      ?? null,
            annualized_return: m?.is_return      ?? null,
            ann_turnover:      m?.is_turnover    ?? null,
            ic_ir:             m?.is_ic          ?? null,
            mean_ic:           m?.is_ic          ?? null,
          },
          oos_metrics:       m?.oos_sharpe != null ? { sharpe_ratio: m.oos_sharpe } : null,
          overfitting_score: wf.overfitting_score ?? 0,
          is_overfit:        wf.is_overfit        ?? false,
          ic_decay:          {},
          best_config:       wf.best_config ?? null,
          n_trials_run:      null,
          pnl_is:            wf.pnl_is   ?? [],
          pnl_oos:           wf.pnl_oos  ?? [],
          split_date:        wf.split_date ?? null,
        })

        if (wf.best_dsl && wf.best_dsl !== dsl) {
          store.newEmptyTab()
          store.setEditorDsl(wf.best_dsl)
          store.appendLog('[GP] Opened best DSL in new editor tab.')
        }

        store.setStatus('ready')
        await loadHistory()
      } else {
        store.finalizeStreamingMessage(streamId)
        store.setStatus('error')
      }
    } catch (err: any) {
      store.finalizeStreamingMessage(streamId)
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
    initSessions, loadSessions, newSession, switchSession,
    renameSession, deleteSession,
    store,
  }
}
