import { useWorkspaceStore } from '../store/workspaceStore'
import { apiChat, apiSimulate, apiOptimize, apiFetchAlphaHistory } from '../api/client'

export function useQuantWorkspace() {
  const store = useWorkspaceStore()

  // ── Send a chat message ─────────────────────────────────────────────────
  const sendChat = async (text: string) => {
    if (!text.trim()) return
    store.addMessage({ role: 'user', content: text })
    store.setStatus('optimizing')
    try {
      const res = await apiChat(text, store.sessionId)
      const { reply, dsl, metrics } = res.data
      store.addMessage({
        role: 'assistant',
        content: reply,
        dsl,
        metrics: metrics as any,
        type: 'message',
      })
      // Auto-sync generated DSL into editor
      if (dsl) store.setEditorDsl(dsl)
      store.setStatus('ready')
    } catch (err: any) {
      store.addMessage({
        role: 'assistant',
        content: `Error: ${err.message}`,
        type: 'tool_output',
      })
      store.setStatus('error')
    }
  }

  // ── Run manual backtest (simulate) ──────────────────────────────────────
  const runBacktest = async () => {
    const dsl = store.editorDsl.trim()
    if (!dsl) return
    store.setStatus('backtesting')
    store.clearLogs()
    store.appendLog(`[Backtest] DSL: ${dsl}`)
    store.appendLog('[Backtest] Sending to /alpha/simulate...')
    try {
      const res = await apiSimulate(dsl, store.simConfig)
      console.log('[runBacktest] raw response:', JSON.stringify(res.data).slice(0, 400))
      console.log('[runBacktest] pnl_is length:', res.data.pnl_is?.length, 'pnl_oos length:', res.data.pnl_oos?.length)
      store.setSimulationResult(res.data)
      store.appendLog(`[Backtest] IS Sharpe: ${res.data.is_metrics?.sharpe_ratio?.toFixed(4) ?? 'N/A'}`)
      store.appendLog(`[Backtest] OOS Sharpe: ${res.data.oos_metrics?.sharpe_ratio?.toFixed(4) ?? 'N/A'}`)
      store.appendLog(`[Backtest] Overfitting Score: ${res.data.overfitting_score?.toFixed(3)}`)
      store.appendLog(`[Backtest] PnL series: IS=${res.data.pnl_is?.length ?? 0} pts, OOS=${res.data.pnl_oos?.length ?? 0} pts`)
      store.appendLog(res.data.is_overfit ? '[WARN] Overfitting detected!' : '[OK] Passed anti-overfitting check.')
      store.setStatus('ready')
      await loadHistory()
    } catch (err: any) {
      store.appendLog(`[ERROR] ${err.message}`)
      store.setStatus('error')
    }
  }

  // ── AI Optimize via Optuna ──────────────────────────────────────────────
  const runOptimize = async () => {
    const dsl = store.editorDsl.trim()
    if (!dsl) return
    store.setStatus('optimizing')
    store.clearLogs()
    store.appendLog(`[Optuna] Optimizing DSL: ${dsl}`)
    store.appendLog('[Optuna] Running 20 trials on IS dataset...')
    try {
      const res = await apiOptimize(dsl, 20)
      store.setSimulationResult(res.data)
      if (res.data.best_config) {
        store.appendLog(`[Optuna] Best config: ${JSON.stringify(res.data.best_config)}`)
      }
      store.appendLog(`[Optuna] IS Sharpe: ${res.data.is_metrics?.sharpe_ratio?.toFixed(4) ?? 'N/A'}`)
      store.appendLog(`[Optuna] OOS Sharpe: ${res.data.oos_metrics?.sharpe_ratio?.toFixed(4) ?? 'N/A'}`)
      store.setStatus('ready')
      await loadHistory()
    } catch (err: any) {
      store.appendLog(`[ERROR] ${err.message}`)
      store.setStatus('error')
    }
  }

  // ── Load alpha history ──────────────────────────────────────────────────
  const loadHistory = async () => {
    try {
      const res = await apiFetchAlphaHistory(30)
      store.setAlphaHistory(res.data.records)
    } catch { /* silently ignore */ }
  }

  return { sendChat, runBacktest, runOptimize, loadHistory, store }
}
