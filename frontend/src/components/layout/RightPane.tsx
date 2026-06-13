import PnLChart from '../analysis/PnLChart'
import MetricsGrid from '../analysis/MetricsGrid'
import OverfitBadge from '../analysis/OverfitBadge'
import WalkForwardChart from '../analysis/WalkForwardChart'
import AlphaPoolPanel from '../analysis/AlphaPoolPanel'
import { useWorkspaceStore } from '../../store/workspaceStore'
import { BarChart2, GitBranch, Layers } from 'lucide-react'

export default function RightPane() {
  const {
    simulationResult, walkForwardResult, workflowResult,
    analysisTab, setAnalysisTab,
  } = useWorkspaceStore()

  const hasBacktest  = simulationResult !== null
  const hasWalkFwd   = walkForwardResult !== null
  const hasPool      = workflowResult !== null && (workflowResult.pool_top5?.length ?? 0) > 0

  return (
    <aside className="h-full flex flex-col bg-slate-900 border-l border-slate-800 overflow-hidden">
      {/* Header with tab toggle */}
      <div className="flex items-center gap-1 px-2 py-2 border-b border-slate-800 shrink-0">
        <button
          onClick={() => setAnalysisTab('backtest')}
          className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-[11px] font-medium transition-colors ${
            analysisTab === 'backtest'
              ? 'bg-slate-800 text-emerald-400'
              : 'text-slate-500 hover:text-slate-300'
          }`}
        >
          <BarChart2 size={12} />
          Backtest
          {hasBacktest && <OverfitBadge score={simulationResult!.overfitting_score} inline />}
        </button>

        <button
          onClick={() => setAnalysisTab('walkforward')}
          className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-[11px] font-medium transition-colors ${
            analysisTab === 'walkforward'
              ? 'bg-slate-800 text-sky-400'
              : 'text-slate-500 hover:text-slate-300'
          }`}
        >
          <GitBranch size={12} />
          Walk-Fwd
          {hasWalkFwd && (
            <span className={`ml-1 text-[9px] font-semibold rounded px-1 py-0.5 ${
              walkForwardResult!.mean_oos_sharpe > 0.3
                ? 'bg-emerald-900/60 text-emerald-400'
                : 'bg-amber-900/60 text-amber-400'
            }`}>
              {walkForwardResult!.n_folds}F
            </span>
          )}
        </button>

        {/* FE-3.1: Pool tab — shows AlphaPool top-5 + combined signal metrics */}
        <button
          onClick={() => setAnalysisTab('pool')}
          className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-[11px] font-medium transition-colors ${
            analysisTab === 'pool'
              ? 'bg-slate-800 text-violet-400'
              : 'text-slate-500 hover:text-slate-300'
          }`}
        >
          <Layers size={12} />
          Pool
          {hasPool && (
            <span className="ml-1 text-[9px] font-semibold rounded px-1 py-0.5 bg-violet-900/60 text-violet-400">
              {workflowResult!.pool_top5.length}
            </span>
          )}
        </button>
      </div>

      <div className="flex-1 overflow-y-auto flex flex-col gap-2 p-2">
        {analysisTab === 'backtest' ? (
          hasBacktest ? (
            <>
              <PnLChart />
              <MetricsGrid />
            </>
          ) : (
            <div className="flex flex-col items-center justify-center flex-1 text-slate-700 text-sm text-center gap-2">
              <BarChart2 size={40} className="opacity-30" />
              <p>Run a backtest or send a chat<br/>message to see analysis here.</p>
            </div>
          )
        ) : analysisTab === 'walkforward' ? (
          <WalkForwardChart />
        ) : (
          <AlphaPoolPanel />
        )}
      </div>
    </aside>
  )
}
