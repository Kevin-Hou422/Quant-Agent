import { useCallback, useEffect, useState } from 'react'
import { Inbox, Check, X, RefreshCw } from 'lucide-react'
import { apiFetchPending, apiApproveAlpha, apiRejectAlpha } from '../../api/client'
import type { PendingAlpha } from '../../types'

/**
 * FE-9 (Phase 9.4): 待批准队列 —— 默认模式的人机接口。
 * agent 自主走到 VALIDATED，用户在这里只做「批准（→PAPER）」或「拒绝（→RETIRED）」。
 */
export default function ApprovalQueue({ onChange }: { onChange?: () => void }) {
  const [items, setItems]   = useState<PendingAlpha[]>([])
  const [loading, setLoading] = useState(true)
  const [busy, setBusy]     = useState<number | null>(null)
  const [error, setError]   = useState<string | null>(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const { data } = await apiFetchPending()
      setItems(data)
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { refresh() }, [refresh])

  const decide = useCallback(
    async (id: number, action: 'approve' | 'reject') => {
      let reason = ''
      if (action === 'reject') {
        reason = window.prompt('拒绝原因（可选）：') ?? ''
      }
      setBusy(id)
      try {
        if (action === 'approve') await apiApproveAlpha(id, reason)
        else await apiRejectAlpha(id, reason)
        await refresh()
        onChange?.()
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e))
      } finally {
        setBusy(null)
      }
    },
    [refresh, onChange],
  )

  return (
    <div className="border border-slate-800 rounded-lg bg-slate-900/40">
      <div className="flex items-center gap-2 px-3 py-2 border-b border-slate-800">
        <Inbox size={14} className="text-sky-400" />
        <span className="text-xs font-semibold text-slate-200">待批准队列</span>
        <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded bg-sky-900/60 text-sky-400">
          {items.length}
        </span>
        <button
          onClick={refresh}
          className="ml-auto flex items-center gap-1 text-[11px] text-slate-400 hover:text-slate-200 transition-colors"
        >
          <RefreshCw size={12} /> 刷新
        </button>
      </div>

      {error && (
        <div className="px-3 py-2 text-[11px] text-rose-400">{error}</div>
      )}

      {loading ? (
        <div className="px-3 py-4 text-[11px] text-slate-500">加载中…</div>
      ) : items.length === 0 ? (
        <div className="px-3 py-4 text-[11px] text-slate-500">
          暂无待批准候选（agent 挖掘 + 验证门通过后会出现在这里）
        </div>
      ) : (
        <ul className="divide-y divide-slate-800">
          {items.map((it) => (
            <li key={it.alpha_id} className="px-3 py-2 flex items-center gap-3">
              <div className="min-w-0 flex-1">
                <div className="text-[11px] font-mono text-slate-300 truncate">{it.dsl}</div>
                <div className="text-[10px] text-slate-500 truncate">
                  #{it.alpha_id} · {it.hypothesis || '—'} · Sharpe {it.sharpe.toFixed(2)} · IC-IR {it.ic_ir.toFixed(2)}
                </div>
              </div>
              <button
                disabled={busy === it.alpha_id}
                onClick={() => decide(it.alpha_id, 'approve')}
                className="flex items-center gap-1 text-[11px] px-2 py-1 rounded bg-emerald-900/50 text-emerald-400 hover:bg-emerald-900/80 disabled:opacity-40 transition-colors"
              >
                <Check size={12} /> 批准
              </button>
              <button
                disabled={busy === it.alpha_id}
                onClick={() => decide(it.alpha_id, 'reject')}
                className="flex items-center gap-1 text-[11px] px-2 py-1 rounded bg-rose-900/50 text-rose-400 hover:bg-rose-900/80 disabled:opacity-40 transition-colors"
              >
                <X size={12} /> 拒绝
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
