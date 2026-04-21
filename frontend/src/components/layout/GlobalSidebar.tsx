import { MessageSquare, Code2, BookOpen, Play, Zap } from 'lucide-react'
import { useWorkspaceStore } from '../../store/workspaceStore'
import { useQuantWorkspace } from '../../hooks/useQuantWorkspace'

function NavBtn({
  icon: Icon,
  label,
  active,
  onClick,
}: {
  icon:    React.ElementType
  label:   string
  active?: boolean
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      title={label}
      className={`
        flex flex-col items-center justify-center gap-1 w-full py-3 text-xs
        transition-colors duration-100
        ${active
          ? 'text-emerald-400 bg-slate-800 border-r-2 border-emerald-500'
          : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60 border-r-2 border-transparent'}
      `}
    >
      <Icon size={18} strokeWidth={1.5} />
      <span className="text-[9px] leading-none font-medium">{label}</span>
    </button>
  )
}

export default function GlobalSidebar() {
  const {
    activeView, setActiveView,
    ledgerOpen, toggleLedger, setLedgerOpen,
    status,
  } = useWorkspaceStore()
  const { runBacktest, runOptimize } = useQuantWorkspace()

  const isRunning = status === 'backtesting' || status === 'optimizing'
  const inChat     = activeView === 'CHAT'
  const inCompiler = activeView === 'COMPILER'

  // Chat button: toggle between CHAT ↔ COMPILER
  const handleChat = () => {
    if (inChat) {
      setActiveView('COMPILER')
    } else {
      setActiveView('CHAT')
      setLedgerOpen(false)   // hide ledger when switching to chat
    }
  }

  // Compiler button: always go to COMPILER
  const handleCompiler = () => {
    setActiveView('COMPILER')
  }

  // Ledger button: if in CHAT → switch to COMPILER + open ledger
  //                if in COMPILER → toggle ledger
  const handleLedger = () => {
    if (inChat) {
      setActiveView('COMPILER')
      setLedgerOpen(true)
    } else {
      toggleLedger()
    }
  }

  return (
    <aside className="w-full h-full flex flex-col bg-slate-900 border-r border-slate-700">
      {/* Logo */}
      <div className="flex items-center justify-center h-14 border-b border-slate-800 shrink-0">
        <span className="text-base font-bold text-emerald-400 tracking-tight select-none">QA</span>
      </div>

      {/* Nav buttons */}
      <nav className="flex flex-col flex-1 pt-1">
        <NavBtn
          icon={MessageSquare}
          label="Chat"
          active={inChat}
          onClick={handleChat}
        />
        <NavBtn
          icon={Code2}
          label="Compiler"
          active={inCompiler && !ledgerOpen}
          onClick={handleCompiler}
        />
        <NavBtn
          icon={BookOpen}
          label="Ledger"
          active={inCompiler && ledgerOpen}
          onClick={handleLedger}
        />
      </nav>

      {/* Action buttons — only shown in COMPILER mode */}
      {inCompiler && (
        <div className="flex flex-col gap-2 px-2 pb-4 pt-3 border-t border-slate-800 shrink-0">
          <button
            onClick={runOptimize}
            disabled={isRunning}
            title="AI Optimize"
            className="flex flex-col items-center justify-center gap-1 py-2 rounded-md bg-violet-800/60 hover:bg-violet-700 disabled:opacity-40 text-violet-300 text-[9px] font-medium transition-colors"
          >
            <Zap size={14} strokeWidth={1.5} />
            AI Opt
          </button>
          <button
            onClick={runBacktest}
            disabled={isRunning}
            title="Run Backtest"
            className={`
              flex flex-col items-center justify-center gap-1 py-2 rounded-md text-[9px] font-medium transition-colors
              ${isRunning
                ? 'bg-amber-800/60 text-amber-300 animate-pulse'
                : 'bg-emerald-700 hover:bg-emerald-600 text-white'}
              disabled:opacity-40
            `}
          >
            <Play size={14} strokeWidth={1.5} />
            {isRunning ? '…' : 'Run'}
          </button>
        </div>
      )}
    </aside>
  )
}
