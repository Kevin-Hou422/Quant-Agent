import { useWorkspaceStore } from '../../store/workspaceStore'
import OverfitBadge from './OverfitBadge'

interface MetricRow {
  label: string
  isKey: string
  oosKey: string
  fmt?: (v: any) => string
}

const PCT = (v: number | null | undefined) =>
  v == null ? '—' : `${(v * 100).toFixed(2)}%`
const NUM = (v: number | null | undefined) =>
  v == null ? '—' : v.toFixed(3)

const ROWS: MetricRow[] = [
  { label: 'Annualized Return', isKey: 'annualized_return', oosKey: 'annualized_return', fmt: PCT },
  { label: 'Sharpe Ratio',      isKey: 'sharpe_ratio',      oosKey: 'sharpe_ratio',      fmt: NUM },
  { label: 'Max Drawdown',      isKey: 'max_drawdown',      oosKey: 'max_drawdown',      fmt: PCT },
  { label: 'Mean IC',           isKey: 'mean_ic',           oosKey: 'mean_ic',           fmt: NUM },
  { label: 'IC-IR',             isKey: 'ic_ir',             oosKey: 'ic_ir',             fmt: NUM },
  { label: 'Turnover',          isKey: 'ann_turnover',      oosKey: 'ann_turnover',      fmt: PCT },
]

export default function MetricsGrid() {
  const { simulationResult } = useWorkspaceStore()

  if (!simulationResult) {
    return (
      <div className="flex items-center justify-center h-full text-slate-600 text-sm">
        No results yet
      </div>
    )
  }

  const is = simulationResult.is_metrics ?? {}
  const oos = simulationResult.oos_metrics ?? {}
  const score = simulationResult.overfitting_score ?? 0

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Metrics</span>
        <OverfitBadge score={score} />
      </div>

      <table className="w-full text-xs">
        <thead>
          <tr className="text-slate-500">
            <th className="text-left pb-2 font-medium">Metric</th>
            <th className="text-right pb-2 font-medium text-blue-400">IS</th>
            <th className="text-right pb-2 font-medium text-rose-400">OOS</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800">
          {ROWS.map(row => {
            const isVal = (is as any)[row.isKey]
            const oosVal = (oos as any)[row.oosKey]
            const fmt = row.fmt ?? NUM

            // Highlight degradation
            let oosClass = 'text-slate-300'
            if (isVal != null && oosVal != null && typeof isVal === 'number' && typeof oosVal === 'number') {
              if (row.isKey === 'sharpe_ratio' && isVal > 0 && oosVal < isVal * 0.6) {
                oosClass = score > 0.6 ? 'text-rose-400' : 'text-amber-400'
              }
            }

            return (
              <tr key={row.label} className="hover:bg-slate-800/50 transition-colors">
                <td className="py-1.5 text-slate-400">{row.label}</td>
                <td className="py-1.5 text-right text-blue-300 font-mono">{fmt(isVal)}</td>
                <td className={`py-1.5 text-right font-mono ${oosClass}`}>{fmt(oosVal)}</td>
              </tr>
            )
          })}
        </tbody>
      </table>

      {simulationResult.ic_decay && (
        <div className="mt-1">
          <p className="text-xs text-slate-500 mb-1.5">IC Decay</p>
          <div className="flex gap-3">
            {Object.entries(simulationResult.ic_decay).map(([k, v]) => (
              <div key={k} className="flex-1 bg-slate-800 rounded-lg px-2 py-1.5 text-center">
                <div className="text-xs text-slate-500">{k}</div>
                <div className="text-sm font-mono text-emerald-400">{NUM(v as number)}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
