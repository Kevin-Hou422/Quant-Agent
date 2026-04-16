import PnLChart from '../analysis/PnLChart'
import MetricsGrid from '../analysis/MetricsGrid'
import OverfitBadge from '../analysis/OverfitBadge'
import { useWorkspaceStore } from '../../store/workspaceStore'
import { BarChart2 } from 'lucide-react'

export default function RightPane() {
  const { simulationResult } = useWorkspaceStore()

  return (
    <aside className="h-full flex flex-col bg-slate-900 border-l border-slate-800 overflow-hidden">
      <div className="flex items-center gap-2 px-3 py-3 border-b border-slate-800">
        <BarChart2 size={16} className="text-emerald-400" />
        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Analysis</span>
        {simulationResult && <OverfitBadge score={simulationResult.overfitting_score} />}
      </div>

      <div className="flex-1 overflow-y-auto flex flex-col gap-2 p-2">
        {simulationResult ? (
          <>
            <PnLChart />
            <MetricsGrid />
          </>
        ) : (
          <div className="flex flex-col items-center justify-center flex-1 text-slate-700 text-sm text-center gap-2">
            <BarChart2 size={40} className="opacity-30" />
            <p>Run a backtest or send a chat<br/>message to see analysis here.</p>
          </div>
        )}
      </div>
    </aside>
  )
}
