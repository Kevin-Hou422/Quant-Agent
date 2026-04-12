import { useWorkspaceStore } from '../../store/workspaceStore'

export default function ConsoleOutput() {
  const { consoleLogs, clearLogs, status } = useWorkspaceStore()

  const statusColor: Record<string, string> = {
    idle: 'text-slate-500',
    backtesting: 'text-amber-400',
    optimizing: 'text-violet-400',
    ready: 'text-emerald-400',
    error: 'text-rose-400',
  }

  return (
    <div className="h-36 flex flex-col border-t border-slate-800 bg-slate-950">
      <div className="flex items-center px-3 py-1.5 border-b border-slate-800">
        <span className={`text-xs font-mono font-semibold ${statusColor[status]}`}>
          ▶ Console [{status.toUpperCase()}]
        </span>
        <button
          onClick={clearLogs}
          className="ml-auto text-xs text-slate-600 hover:text-slate-400 transition-colors"
        >
          clear
        </button>
      </div>
      <div className="flex-1 overflow-y-auto px-3 py-2 font-mono text-xs text-slate-400 space-y-0.5">
        {consoleLogs.length === 0 ? (
          <span className="text-slate-700">Waiting for execution…</span>
        ) : (
          consoleLogs.map((l, i) => (
            <div key={i} className={`
              ${l.includes('[ERROR]') ? 'text-rose-400' :
                l.includes('[WARN]') ? 'text-amber-400' :
                l.includes('[OK]') ? 'text-emerald-400' : 'text-slate-400'}
            `}>{l}</div>
          ))
        )}
      </div>
    </div>
  )
}
