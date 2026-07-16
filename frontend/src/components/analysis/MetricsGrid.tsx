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

  // FE-3.2: portfolio beta — from is_metrics or oos_metrics (shown only when non-null)
  const portfolioBeta = (is as any).portfolio_beta ?? (oos as any).portfolio_beta ?? null

  // FE-4.2: Deflated Sharpe (OOS-first) + trial count used for the correction
  const dsr = (oos as any).deflated_sharpe ?? (is as any).deflated_sharpe ?? null
  const nTrials = simulationResult.n_trials_run ?? null

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

      {/* FE-3.2: Risk Exposure section — shown only when portfolio_beta is available */}
      {portfolioBeta != null && !isNaN(portfolioBeta) && (
        <div className="border-t border-slate-800 pt-2">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
            Risk Exposure
          </p>
          <div className="flex items-center justify-between text-xs py-1.5">
            <span className="text-slate-400">Market Beta (β)</span>
            <span className={`font-mono font-semibold ${
              Math.abs(portfolioBeta) < 0.1
                ? 'text-emerald-400'
                : Math.abs(portfolioBeta) < 0.3
                  ? 'text-amber-400'
                  : 'text-rose-400'
            }`}>
              {portfolioBeta.toFixed(3)}
            </span>
          </div>
          <p className="text-[10px] text-slate-600 mt-1">
            {Math.abs(portfolioBeta) < 0.1
              ? '✓ Near market-neutral'
              : Math.abs(portfolioBeta) < 0.3
                ? '⚠ Moderate market exposure'
                : '✗ High market exposure — consider beta_neutral()'}
          </p>
        </div>
      )}

      {/* FE-4.2: Deflated Sharpe Ratio — multiple-testing-corrected significance */}
      {dsr != null && !isNaN(dsr) && (
        <div className="border-t border-slate-800 pt-2">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
            Statistical Significance
          </p>
          <div className="flex items-center justify-between text-xs py-1.5">
            <span className="text-slate-400">
              Deflated Sharpe{nTrials != null ? ` (n=${nTrials})` : ''}
            </span>
            <span className={`font-mono font-semibold ${
              dsr > 0.95
                ? 'text-emerald-400'
                : dsr > 0.5
                  ? 'text-amber-400'
                  : 'text-rose-400'
            }`}>
              {(dsr * 100).toFixed(1)}%
            </span>
          </div>
          <p className="text-[10px] text-slate-600 mt-1">
            {dsr > 0.95
              ? '✓ Sharpe significant after multiple-testing correction'
              : dsr > 0.5
                ? '⚠ Weak evidence — may not survive more trials'
                : '✗ Likely selection bias — Sharpe not distinguishable from noise'}
          </p>
        </div>
      )}

      {/* IC Decay section */}
      {simulationResult.ic_decay && Object.keys(simulationResult.ic_decay).length > 0 && (
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
