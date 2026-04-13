import ReactECharts from 'echarts-for-react'
import { useWorkspaceStore } from '../../store/workspaceStore'
import { useMemo } from 'react'
import ErrorBoundary from '../ErrorBoundary'

// Defensive cumulative product — handles empty arrays and NaN gracefully
function cumulativeReturns(rets: number[]): number[] {
  if (!rets || rets.length === 0) return []
  let cum = 1
  return rets.map((r) => {
    const safe = isFinite(r) ? r : 0
    cum *= 1 + safe
    return +(cum - 1).toFixed(4)
  })
}

function PnLChartInner() {
  const { simulationResult } = useWorkspaceStore()

  const option = useMemo(() => {
    // ── Empty state ─────────────────────────────────────────────────────────
    if (!simulationResult) {
      return {
        backgroundColor: 'transparent',
        graphic: [{
          type: 'text', left: 'center', top: 'middle',
          style: { text: 'Run a backtest to see results', fill: '#475569', fontSize: 13 },
        }],
        xAxis: { show: false }, yAxis: { show: false }, series: [],
      }
    }

    const isReturns: number[] = simulationResult.pnl_is ?? []
    const oosReturns: number[] = simulationResult.pnl_oos ?? []

    // ── No data guard ────────────────────────────────────────────────────────
    if (isReturns.length === 0 && oosReturns.length === 0) {
      return {
        backgroundColor: 'transparent',
        graphic: [{
          type: 'text', left: 'center', top: 'middle',
          style: { text: 'No PnL data returned by backend', fill: '#64748b', fontSize: 12 },
        }],
        xAxis: { show: false }, yAxis: { show: false }, series: [],
      }
    }

    const totalLen = isReturns.length + oosReturns.length

    // Synthesize date labels (descending days from today)
    const today = new Date()
    const dates: string[] = Array.from({ length: totalLen }, (_, i) => {
      const d = new Date(today)
      d.setDate(d.getDate() - (totalLen - i))
      return d.toISOString().slice(0, 10)
    })

    const isCum = cumulativeReturns(isReturns)
    // OOS curve starts where IS ends
    const isEndValue = isCum[isCum.length - 1] ?? 0
    const oosCumRaw = cumulativeReturns(oosReturns)
    const oosCum = oosCumRaw.map(v => +(v + isEndValue).toFixed(4))

    // Build padded series data: IS series fills OOS positions with null, vice versa
    const isData: (number | null)[] = [
      ...isCum,
      ...Array(oosReturns.length).fill(null),
    ]
    // OOS starts at the IS endpoint so lines connect
    const oosData: (number | null)[] = [
      ...Array(isReturns.length - 1).fill(null),
      isEndValue,   // connect point
      ...oosCum,
    ]

    const yPct = (v: number) => `${(v * 100).toFixed(2)}%`

    return {
      backgroundColor: 'transparent',
      textStyle: { color: '#94a3b8' },
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#1e293b',
        borderColor: '#334155',
        textStyle: { color: '#e2e8f0', fontSize: 11 },
        formatter: (params: any[]) => {
          const date = params[0]?.axisValue ?? ''
          const lines = params
            .filter(p => p.value != null)
            .map(p => `${p.marker}${p.seriesName}: <b>${yPct(p.value)}</b>`)
          return `${date}<br/>${lines.join('<br/>')}`
        },
      },
      legend: {
        top: 4, right: 8,
        textStyle: { color: '#64748b', fontSize: 10 },
        itemWidth: 12, itemHeight: 4,
      },
      grid: { left: 52, right: 12, top: 32, bottom: 40 },
      xAxis: {
        type: 'category',
        data: dates,
        axisLine: { lineStyle: { color: '#334155' } },
        axisLabel: { color: '#475569', fontSize: 9, rotate: 30,
          formatter: (v: string) => v.slice(5) },  // show MM-DD
        splitLine: { show: false },
      },
      yAxis: {
        type: 'value',
        axisLine: { show: false },
        axisLabel: {
          color: '#64748b', fontSize: 10,
          formatter: yPct,
        },
        splitLine: { lineStyle: { color: '#1e293b', type: 'dashed' } },
      },
      series: [
        {
          name: 'IN-SAMPLE',
          type: 'line',
          data: isData,
          symbol: 'none',
          connectNulls: false,
          lineStyle: { color: '#10b981', width: 2 },
          areaStyle: {
            color: {
              type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(16,185,129,0.18)' },
                { offset: 1, color: 'rgba(16,185,129,0)' },
              ],
            },
          },
          // Label on the IS region
          markArea: {
            silent: true,
            label: { show: true, color: '#10b981', fontSize: 9, position: 'insideTopLeft' },
            itemStyle: { color: 'rgba(16,185,129,0.04)' },
            data: [[{ name: 'IS', xAxis: dates[0] }, { xAxis: dates[isReturns.length - 1] ?? dates[0] }]],
          },
        },
        {
          name: 'OUT-OF-SAMPLE',
          type: 'line',
          data: oosData,
          symbol: 'none',
          connectNulls: false,
          lineStyle: { color: '#38bdf8', width: 2 },
          areaStyle: {
            color: {
              type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(56,189,248,0.15)' },
                { offset: 1, color: 'rgba(56,189,248,0)' },
              ],
            },
          },
          markArea: {
            silent: true,
            label: { show: true, color: '#38bdf8', fontSize: 9, position: 'insideTopRight' },
            itemStyle: { color: 'rgba(56,189,248,0.04)' },
            data: [[
              { name: 'OOS (UNSEEN)', xAxis: dates[isReturns.length] ?? dates[0] },
              { xAxis: dates[dates.length - 1] },
            ]],
          },
        },
      ],
    }
  }, [simulationResult])

  return (
    <ReactECharts
      option={option}
      style={{ height: '100%', width: '100%' }}
      theme="dark"
      opts={{ renderer: 'canvas' }}
      notMerge
    />
  )
}

export default function PnLChart() {
  return (
    <ErrorBoundary>
      <PnLChartInner />
    </ErrorBoundary>
  )
}
