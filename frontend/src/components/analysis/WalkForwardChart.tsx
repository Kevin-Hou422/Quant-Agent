import { useMemo, useRef, useEffect } from 'react'
import ReactECharts from 'echarts-for-react'
import type { ECharts } from 'echarts'
import { useWorkspaceStore } from '../../store/workspaceStore'
import ErrorBoundary from '../ErrorBoundary'
import type { WalkForwardFoldReport } from '../../types'
import { TrendingUp, AlertTriangle, CheckCircle2, Activity } from 'lucide-react'

// ── Summary metric chip ───────────────────────────────────────────────────────

function StatChip({
  label, value, good, warn,
}: { label: string; value: string; good: boolean; warn: boolean }) {
  const cls = good ? 'text-emerald-400' : warn ? 'text-amber-400' : 'text-rose-400'
  return (
    <div className="flex flex-col items-center bg-slate-800 rounded-lg px-3 py-2 min-w-[72px]">
      <span className={`text-sm font-mono font-semibold ${cls}`}>{value}</span>
      <span className="text-[9px] text-slate-500 mt-0.5 text-center leading-tight">{label}</span>
    </div>
  )
}

// ── Fold row in table ─────────────────────────────────────────────────────────

function FoldRow({ fold }: { fold: WalkForwardFoldReport }) {
  const isGood    = fold.oos_sharpe > 0.3
  const isWarn    = fold.oos_sharpe > 0 && fold.oos_sharpe <= 0.3
  const overfit   = fold.overfitting > 0.5
  return (
    <tr className="border-b border-slate-800 hover:bg-slate-800/40 transition-colors text-xs">
      <td className="py-1.5 px-2 text-slate-500 font-mono">F{fold.fold_idx + 1}</td>
      <td className="py-1.5 px-2 text-slate-500 font-mono text-[10px]">
        {fold.oos_start.slice(0, 10)} – {fold.oos_end.slice(0, 10)}
      </td>
      <td className="py-1.5 px-2 text-right font-mono text-sky-300">{fold.is_sharpe.toFixed(3)}</td>
      <td className={`py-1.5 px-2 text-right font-mono ${isGood ? 'text-emerald-400' : isWarn ? 'text-amber-400' : 'text-rose-400'}`}>
        {fold.oos_sharpe.toFixed(3)}
      </td>
      <td className={`py-1.5 px-2 text-right font-mono ${overfit ? 'text-rose-400' : 'text-slate-400'}`}>
        {(fold.overfitting * 100).toFixed(0)}%
      </td>
    </tr>
  )
}

// ── Inner chart ───────────────────────────────────────────────────────────────

function WalkForwardChartInner() {
  const { walkForwardResult } = useWorkspaceStore()
  const chartRef = useRef<{ getEchartsInstance(): ECharts } | null>(null)

  useEffect(() => {
    chartRef.current?.getEchartsInstance?.()?.resize()
  })

  const option = useMemo(() => {
    if (!walkForwardResult || walkForwardResult.fold_reports.length === 0) return null

    const folds    = walkForwardResult.fold_reports
    const labels   = folds.map((f) => `F${f.fold_idx + 1}`)
    const isSharpe = folds.map((f) => +f.is_sharpe.toFixed(4))
    const oosSharpe= folds.map((f) => +f.oos_sharpe.toFixed(4))
    const mean     = walkForwardResult.mean_oos_sharpe

    return {
      backgroundColor: 'transparent',
      textStyle: { color: '#94a3b8' },
      animation: false,
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#1e293b',
        borderColor: '#334155',
        textStyle: { color: '#e2e8f0', fontSize: 11 },
        formatter: (params: any[]) => {
          const fold = folds[params[0].dataIndex]
          return [
            `<b>Fold ${fold.fold_idx + 1}</b>`,
            `OOS: ${fold.oos_start.slice(0, 10)} → ${fold.oos_end.slice(0, 10)}`,
            `IS  Sharpe: <b style="color:#7dd3fc">${fold.is_sharpe.toFixed(3)}</b>`,
            `OOS Sharpe: <b style="color:${fold.oos_sharpe > 0.3 ? '#34d399' : fold.oos_sharpe > 0 ? '#fbbf24' : '#f87171'}">${fold.oos_sharpe.toFixed(3)}</b>`,
            `Overfitting: ${(fold.overfitting * 100).toFixed(0)}%`,
            `IC-IR: ${fold.oos_ic_ir.toFixed(3)}`,
          ].join('<br/>')
        },
      },
      legend: {
        show: true, top: 0, right: 4,
        data: [
          { name: 'IS Sharpe',  icon: 'roundRect', itemStyle: { color: '#38bdf8' } },
          { name: 'OOS Sharpe', icon: 'roundRect', itemStyle: { color: '#10b981' } },
        ],
        textStyle: { color: '#64748b', fontSize: 10 },
        itemWidth: 14, itemHeight: 4,
      },
      grid: { left: 44, right: 8, top: 32, bottom: 28 },
      xAxis: {
        type: 'category', data: labels, boundaryGap: true,
        axisLine: { lineStyle: { color: '#334155' } },
        axisTick: { show: false },
        axisLabel: { color: '#475569', fontSize: 10 },
      },
      yAxis: {
        type: 'value',
        axisLine: { show: false }, axisTick: { show: false },
        axisLabel: { color: '#64748b', fontSize: 10, formatter: (v: number) => v.toFixed(2) },
        splitLine: { lineStyle: { color: '#1e293b', type: 'dashed' } },
      },
      series: [
        {
          name: 'IS Sharpe', type: 'bar', barWidth: '28%',
          data: isSharpe,
          itemStyle: { color: '#38bdf8', borderRadius: [3, 3, 0, 0] },
        },
        {
          name: 'OOS Sharpe', type: 'bar', barWidth: '28%',
          data: oosSharpe.map((v) => ({
            value: v,
            itemStyle: {
              color: v > 0.3 ? '#10b981' : v > 0 ? '#fbbf24' : '#f87171',
              borderRadius: [3, 3, 0, 0],
            },
          })),
        },
        {
          name: 'Mean OOS', type: 'line', symbol: 'none',
          data: oosSharpe.map(() => +mean.toFixed(4)),
          lineStyle: { color: '#10b981', type: 'dashed', width: 1.5, opacity: 0.7 },
          silent: true,
        },
        // Zero baseline
        {
          name: 'Zero', type: 'line', symbol: 'none',
          data: oosSharpe.map(() => 0),
          lineStyle: { color: '#475569', type: 'solid', width: 1, opacity: 0.5 },
          silent: true,
        },
      ],
    }
  }, [walkForwardResult])

  if (!walkForwardResult) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-slate-700 text-sm text-center gap-2">
        <Activity size={36} className="opacity-30" />
        <p className="text-xs">Run Walk-Forward to validate alpha stability<br/>across multiple time periods</p>
      </div>
    )
  }

  const wf   = walkForwardResult
  const pctP = wf.pct_positive * 100
  const isStable   = wf.mean_oos_sharpe > 0.3 && pctP >= 60
  const isWarn     = wf.mean_oos_sharpe > 0 && (wf.mean_oos_sharpe <= 0.3 || pctP < 60)
  const isBad      = wf.mean_oos_sharpe <= 0

  const overallIcon = isStable
    ? <CheckCircle2 size={13} className="text-emerald-400" />
    : isWarn
    ? <AlertTriangle size={13} className="text-amber-400" />
    : <AlertTriangle size={13} className="text-rose-400" />

  return (
    <div className="flex flex-col gap-2">
      {/* Summary header */}
      <div className="flex items-center gap-2 mb-1">
        {overallIcon}
        <span className={`text-xs font-semibold ${isStable ? 'text-emerald-400' : isWarn ? 'text-amber-400' : 'text-rose-400'}`}>
          {isStable ? 'Stable' : isWarn ? 'Marginal' : 'Unstable'} — {wf.n_folds}-fold Walk-Forward
        </span>
      </div>

      {/* Stat chips */}
      <div className="flex gap-1.5 flex-wrap">
        <StatChip
          label="Mean OOS SR"
          value={wf.mean_oos_sharpe.toFixed(3)}
          good={wf.mean_oos_sharpe > 0.3}
          warn={wf.mean_oos_sharpe > 0}
        />
        <StatChip
          label="±Std"
          value={`±${wf.std_oos_sharpe.toFixed(3)}`}
          good={wf.std_oos_sharpe < 0.3}
          warn={wf.std_oos_sharpe < 0.6}
        />
        <StatChip
          label="Min OOS SR"
          value={wf.min_oos_sharpe.toFixed(3)}
          good={wf.min_oos_sharpe > 0}
          warn={wf.min_oos_sharpe > -0.2}
        />
        <StatChip
          label="% Positive"
          value={`${pctP.toFixed(0)}%`}
          good={pctP >= 60}
          warn={pctP >= 40}
        />
        <StatChip
          label="Overfit"
          value={`${(wf.mean_overfitting * 100).toFixed(0)}%`}
          good={wf.mean_overfitting < 0.3}
          warn={wf.mean_overfitting < 0.5}
        />
      </div>

      {/* Bar chart */}
      <div className="w-full" style={{ height: 160 }}>
        {option && (
          <ReactECharts
            ref={chartRef as any}
            option={option}
            style={{ height: '100%', width: '100%' }}
            opts={{ renderer: 'canvas' }}
            notMerge
          />
        )}
      </div>

      {/* Fold detail table */}
      <table className="w-full text-xs mt-1">
        <thead>
          <tr className="text-slate-600">
            <th className="text-left pb-1.5 px-2 font-medium">#</th>
            <th className="text-left pb-1.5 px-2 font-medium">OOS Period</th>
            <th className="text-right pb-1.5 px-2 font-medium text-sky-400/80">IS SR</th>
            <th className="text-right pb-1.5 px-2 font-medium text-emerald-400/80">OOS SR</th>
            <th className="text-right pb-1.5 px-2 font-medium text-slate-500">Overfit</th>
          </tr>
        </thead>
        <tbody>
          {wf.fold_reports.map((f) => <FoldRow key={f.fold_idx} fold={f} />)}
        </tbody>
      </table>
    </div>
  )
}

export default function WalkForwardChart() {
  return (
    <ErrorBoundary>
      <WalkForwardChartInner />
    </ErrorBoundary>
  )
}
