import { MessageSquare, Code2, Play, Zap } from 'lucide-react'
import { useWorkspaceStore } from '../../store/workspaceStore'
import { useQuantWorkspace } from '../../hooks/useQuantWorkspace'

const NavBtn = ({
  icon: Icon,
  label,
  active,
  onClick,
}: { icon: React.FC<any>; label: string; active?: boolean; onClick: () => void }) => (
  <button
    title={label}
    onClick={onClick}
    className={`flex flex-col items-center gap-1 px-2 py-3 rounded-lg w-full text-xs transition-colors
      ${active
        ? 'bg-emerald-500/20 text-emerald-400'
        : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800'}`}
  >
    <Icon size={20} />
    <span className="leading-tight text-center">{label}</span>
  </button>
)

export default function GlobalSidebar() {
  const { activeView, setActiveView, status } = useWorkspaceStore()
  const { runBacktest, runOptimize } = useQuantWorkspace()

  const isRunning = status === 'backtesting' || status === 'optimizing'

  return (
    <aside className="w-20 flex flex-col items-center gap-2 bg-slate-900 border-r border-slate-800 py-4 px-2 shrink-0">
      {/* Logo */}
      <div className="mb-3 text-emerald-400">
        <Zap size={28} />
      </div>

      <div className="w-full h-px bg-slate-700 mb-2" />

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

      <div className="flex-1" />

      {/* Run Backtest */}
      <button
        title="Run Manual Backtest"
        onClick={() => { setActiveView('COMPILER'); runBacktest() }}
        disabled={isRunning}
        className="flex flex-col items-center gap-1 px-2 py-3 rounded-lg w-full text-xs
          bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed
          text-white transition-colors"
      >
        <Play size={20} />
        <span className="leading-tight text-center">{isRunning ? '...' : 'Run'}</span>
      </button>

      {/* AI Optimize */}
      <button
        title="AI Optimize via Optuna"
        onClick={() => { setActiveView('COMPILER'); runOptimize() }}
        disabled={isRunning}
        className="flex flex-col items-center gap-1 px-2 py-3 rounded-lg w-full text-xs
          bg-violet-700 hover:bg-violet-600 disabled:opacity-40 disabled:cursor-not-allowed
          text-white transition-colors"
      >
        <Zap size={20} />
        <span className="leading-tight text-center">Optuna</span>
      </button>
    </aside>
  )
}
