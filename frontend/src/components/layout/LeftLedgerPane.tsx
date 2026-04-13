import { useWorkspaceStore } from '../../store/workspaceStore'
import { useQuantWorkspace } from '../../hooks/useQuantWorkspace'
import { RefreshCw, X } from 'lucide-react'

const statusColor: Record<string, string> = {
  active:      'bg-emerald-500',
  optimizing:  'bg-amber-400 animate-pulse',
  backtesting: 'bg-violet-500 animate-pulse',
  ready:       'bg-sky-400',
  error:       'bg-rose-500',
}

const StatusBadge = ({ s }: { s?: string }) => (
  <span className={`inline-block w-2 h-2 rounded-full shrink-0 ${statusColor[s ?? 'active'] ?? 'bg-slate-600'}`} />
)

export default function LeftLedgerPane() {
  const { alphaHistory, setEditorDsl, setActiveView, setLedgerOpen } = useWorkspaceStore()
  const { loadHistory } = useQuantWorkspace()

  return (
    <aside className="w-64 h-full flex flex-col bg-slate-900 border-r border-slate-800 shadow-2xl">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-3 border-b border-slate-800">
        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Alpha Ledger</span>
        <div className="flex items-center gap-2">
          <button
            onClick={loadHistory}
            className="text-slate-500 hover:text-slate-300 transition-colors"
            title="Refresh"
          >
            <RefreshCw size={13} />
          </button>
          <button
            onClick={() => setLedgerOpen(false)}
            className="text-slate-500 hover:text-slate-300 transition-colors"
            title="Close"
          >
            <X size={13} />
          </button>
        </div>
      </div>

      {/* List */}
      <div className="flex-1 overflow-y-auto">
        {alphaHistory.length === 0 ? (
          <div className="px-3 py-6 text-center text-xs text-slate-600">
            No alphas yet.<br />Run a backtest to generate results.
          </div>
        ) : (
          alphaHistory.map((rec) => (
            <button
              key={rec.id}
              onClick={() => {
                setEditorDsl(rec.dsl)
                setActiveView('COMPILER')
                setLedgerOpen(false)
              }}
              className="w-full text-left px-3 py-2.5 border-b border-slate-800 hover:bg-slate-800 transition-colors group"
            >
              <div className="flex items-center gap-2 mb-1">
                <StatusBadge s={rec.status} />
                <span className="text-xs font-mono text-slate-300 truncate flex-1">
                  #{rec.id} {rec.hypothesis ? rec.hypothesis.slice(0, 20) : rec.dsl.slice(0, 22)}
                </span>
              </div>
              <div className="flex gap-3 text-xs text-slate-500 ml-4">
                <span>
                  Sharpe <span className={`font-semibold ${(rec.sharpe ?? 0) > 0.5 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {rec.sharpe?.toFixed(2) ?? '—'}
                  </span>
                </span>
                {rec.ic_ir != null && (
                  <span>IC-IR <span className="text-sky-400">{rec.ic_ir.toFixed(2)}</span></span>
                )}
              </div>
              {rec.created_at && (
                <div className="text-xs text-slate-700 ml-4 mt-0.5">
                  {new Date(rec.created_at).toLocaleDateString()}
                </div>
              )}
            </button>
          ))
        )}
      </div>

      {/* Footer */}
      <div className="px-3 py-2 border-t border-slate-800 text-xs text-slate-600">
        {alphaHistory.length} alphas stored
      </div>
    </aside>
  )
}
