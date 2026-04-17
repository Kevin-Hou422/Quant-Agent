import { MessageSquare, Code2, BookOpen, Play, Zap } from 'lucide-react'
import { useWorkspaceStore } from '../../store/workspaceStore'
import { useQuantWorkspace } from '../../hooks/useQuantWorkspace'

const NavBtn = ({
  icon: Icon,
  label,
  active,
  onClick,
}: {
  icon: React.ElementType
  label: string
  active?: boolean
  onClick: () => void
}) => (
  <button
    onClick={onClick}
    title={label}
    className={`
      flex flex-col items-center justify-center gap-1 w-full py-3 text-xs transition-colors
      ${active
        ? 'text-emerald-400 bg-slate-800'
        : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800/60'}
    `}
  >
    <Icon size={20} strokeWidth={1.5} />
    <span className="text-[10px] leading-none">{label}</span>
  </button>
)

export default function GlobalSidebar() {
  const { activeView, setActiveView, status, ledgerOpen, toggleLedger } = useWorkspaceStore()
  const { runBacktest, runOptimize } = useQuantWorkspace()

  const isRunning = status === 'backtesting' || status === 'optimizing'

  return (
    <aside className="w-full h-full flex flex-col bg-slate-900 border-r border-slate-800 overflow-hidden">
      {/* Logo */}
      <div className="flex items-center justify-center h-14 border-b border-slate-800">
        <span className="text-lg font-bold text-emerald-400 tracking-tight">QA</span>
      </div>

      {/* Nav */}
      <nav className="flex flex-col flex-1 pt-2">
        <NavBtn
          icon={MessageSquare}
          label="Chat"
          active={activeView === 'CHAT'}
          onClick={() => setActiveView('CHAT')}
        />
        <NavBtn
          icon={Code2}
          label="Compiler"
          active={activeView === 'COMPILER'}
          onClick={() => setActiveView('COMPILER')}
        />
        <NavBtn
          icon={BookOpen}
          label="Ledger"
          active={ledgerOpen}
          onClick={toggleLedger}
        />
      </nav>

      {/* Action buttons */}
      <div className="flex flex-col gap-2 px-2 pb-4 border-t border-slate-800 pt-3">
        <button
          onClick={runOptimize}
          disabled={isRunning}
          title="AI Optimize"
          className="flex flex-col items-center justify-center gap-1 py-2 rounded-lg bg-violet-800/60 hover:bg-violet-700 disabled:opacity-40 text-violet-300 text-[10px] transition-colors"
        >
          <Zap size={16} strokeWidth={1.5} />
          AI Opt
        </button>
        <button
          onClick={runBacktest}
          disabled={isRunning}
          title="Run Backtest"
          className={`
            flex flex-col items-center justify-center gap-1 py-2 rounded-lg text-[10px] transition-colors
            ${isRunning
              ? 'bg-amber-800/60 text-amber-300 animate-pulse'
              : 'bg-emerald-700 hover:bg-emerald-600 text-white'}
            disabled:opacity-40
          `}
        >
          <Play size={16} strokeWidth={1.5} />
          {isRunning ? 'Running' : 'Run'}
        </button>
      </div>
    </aside>
  )
}
