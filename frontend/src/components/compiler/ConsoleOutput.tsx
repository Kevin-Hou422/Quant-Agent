import { useWorkspaceStore } from '../../store/workspaceStore'
import { useQuantWorkspace } from '../../hooks/useQuantWorkspace'
import { Copy, RefreshCw, FileText } from 'lucide-react'

interface LogAction {
  label: string
  icon: React.ElementType
  onClick: () => void
}

function LogLine({ line, actions }: { line: string; actions?: LogAction[] }) {
  const textClass =
    line.includes('[Syntax Error]') ? 'text-rose-400 font-semibold' :
    line.includes('[ERROR]')        ? 'text-rose-400' :
    line.includes('[WARN]')         ? 'text-amber-400' :
    line.includes('[OK]')           ? 'text-emerald-400' :
    line.includes('[GP]')           ? 'text-violet-400' :
    line.includes('[Optuna]')       ? 'text-violet-300' :
    line.includes('[Backtest]')     ? 'text-sky-400' :
    line.includes('[System]')       ? 'text-slate-500' :
    'text-slate-400'

  return (
    <div className={`flex items-start gap-2 group ${textClass}`}>
      <span className="flex-1 leading-relaxed">{line}</span>
      {actions && (
        <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
          {actions.map((a) => (
            <button
              key={a.label}
              onClick={a.onClick}
              title={a.label}
              className="flex items-center gap-0.5 text-[10px] text-slate-500 hover:text-slate-200 bg-slate-800 hover:bg-slate-700 rounded px-1.5 py-0.5 transition-colors"
            >
              <a.icon size={9} />
              <span>{a.label}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

export default function ConsoleOutput() {
  const { consoleLogs, clearLogs, status, setActiveView } = useWorkspaceStore()
  const { runOptimize } = useQuantWorkspace()

  const statusColor: Record<string, string> = {
    idle:        'text-slate-500',
    backtesting: 'text-amber-400',
    optimizing:  'text-violet-400',
    ready:       'text-emerald-400',
    error:       'text-rose-400',
  }

  function getActions(line: string): LogAction[] | undefined {
    if (line.includes('[Backtest]') || line.includes('[OK]')) {
      return [{
        label: 'View',
        icon: FileText,
        onClick: () => setActiveView('COMPILER'),
      }]
    }
    if (line.includes('[Optuna]')) {
      return [{
        label: 'Rerun',
        icon: RefreshCw,
        onClick: () => runOptimize(),
      }]
    }
    if (line.includes('[ERROR]')) {
      return [{
        label: 'Copy',
        icon: Copy,
        onClick: () => navigator.clipboard?.writeText(line),
      }]
    }
    return undefined
  }

  return (
    <div className="h-full flex flex-col border-t border-slate-800 bg-slate-950">
      <div className="flex items-center px-3 py-1.5 border-b border-slate-800 shrink-0">
        <span className={`text-xs font-mono font-semibold ${statusColor[status] ?? 'text-slate-500'}`}>
          ▶ Console [{status.toUpperCase()}]
        </span>
        <button
          onClick={clearLogs}
          className="ml-auto text-xs text-slate-600 hover:text-slate-400 transition-colors"
        >
          clear
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-2 font-mono text-xs space-y-0.5">
        {consoleLogs.length === 0 ? (
          <span className="text-slate-700">Waiting for execution…</span>
        ) : (
          consoleLogs.map((l, i) => (
            <LogLine key={i} line={l} actions={getActions(l)} />
          ))
        )}
      </div>
    </div>
  )
}
