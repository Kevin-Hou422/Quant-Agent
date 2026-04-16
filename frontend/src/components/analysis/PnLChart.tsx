import ReactECharts from 'echarts-for-react'
import { useWorkspaceStore } from '../../store/workspaceStore'
import { useMemo } from 'react'
import ErrorBoundary from '../ErrorBoundary'

/** Safely compute cumulative returns from a returns array */
function cumRet(rets: number[]): number[] {
  if (!rets?.length) return []
  let cum = 1
  return rets.map(r => {
    cum *= 1 + (isFinite(r) ? r : 0)
    return +(cum - 1).toFixed(5)
  })
}

function PnLChartInner() {
  const { simulationResult } = useWorkspaceStore()

  const option = useMemo(() => {
    /* ── Empty / loading state ─────────────────────────────────────── */
    const empty = (text: string) => ({
      backgroundColor: 'transparent',
      graphic: [{ type: 'text', left: 'center', top: 'middle',
        style: { text, fill: '#475569', fontSize: 12, fontFamily: 'Inter,sans-serif' } }],
      xAxis: { show: false }, yAxis: { show: false }, series: [],
    })

    if (!simulationResult) return empty('Run a backtest to see PnL')

    const rawIS  = simulationResult.pnl_is  ?? []
    const rawOOS = simulationResult.pnl_oos ?? []

    if (!rawIS.length && !rawOOS.length) return empty('No PnL data returned by backend')

    /* ── Build unified data array ──────────────────────────────────── */
    const isCum   = cumRet(rawIS)
    const isEnd   = isCum[isCum.length - 1] ?? 0
    const oosCum  = cumRet(rawOOS).map(v => +(v + isEnd).toFixed(5))

    // IS points, then OOS points — one continuous series
    const allValues: number[] = [...isCum, ...oosCum]
    const splitIdx = isCum.length   // index where OOS begins

    // Synthesize date labels
    const total = allValues.length
    const today = new Date()
    const xData = Array.from({ length: total }, (_, i) => {
      const d = new Date(today)
      d.setDate(d.getDate() - (total - i))
      return d.toISOString().slice(5, 10) // MM-DD
    })

    /* ── ECharts option ────────────────────────────────────────────── */
    return {
      backgroundColor: 'transparent',
      textStyle: { color: '#94a3b8', fontFamily: 'Inter,sans-serif' },

      // visualMap colours the line by x-index range
      visualMap: [{
        show: false,
        type: 'piecewise',
        dimension: 0,          // x-axis index
        seriesIndex: 0,
        pieces: [
          // IS segment — emerald green
          { min: 0,         max: splitIdx - 1, color: '#10b981' },
          // OOS segment — sky blue
          { min: splitIdx,  max: total,        color: '#38bdf8' },
        ],
      }],

      tooltip: {
        trigger: 'axis',
        backgroundColor: '#1e293b',
        borderColor: '#334155',
        textStyle: { color: '#e2e8f0', fontSize: 11 },
        formatter: (params: any[]) => {
          const p = params[0]
          if (!p) return ''
          const idx   = p.dataIndex
          const phase = idx < splitIdx ? '<span style="color:#10b981">▮ IN-SAMPLE</span>'
                                       : '<span style="color:#38bdf8">▮ OUT-OF-SAMPLE</span>'
          return `${p.axisValue} · ${phase}<br/><b>${(p.value * 100).toFixed(2)}%</b>`
        },
      },

      // Legend as static labels (top right)
      legend: {
        show: true, top: 2, right: 4,
        data: [
          { name: 'IS',  itemStyle: { color: '#10b981' } },
          { name: 'OOS', itemStyle: { color: '#38bdf8' } },
        ],
        textStyle: { color: '#64748b', fontSize: 10 },
        icon: 'roundRect',
        itemWidth: 12, itemHeight: 4,
      },

      grid: { left: 50, right: 10, top: 36, bottom: 38 },

      xAxis: {
        type: 'category',
        data: xData,
        boundaryGap: false,
        axisLine: { lineStyle: { color: '#334155' } },
        axisTick: { show: false },
        axisLabel: {
          color: '#475569', fontSize: 9, rotate: 30,
          interval: Math.floor(total / 5),
        },
        splitLine: { show: false },
      },

      yAxis: {
        type: 'value',
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: {
          color: '#64748b', fontSize: 10,
          formatter: (v: number) => `${(v * 100).toFixed(1)}%`,
        },
        splitLine: { lineStyle: { color: '#1e293b', type: 'dashed' } },
      },

      series: [{
        name: 'Cumulative PnL',
        type: 'line',
        data: allValues,
        smooth: false,
        symbol: 'none',
        lineStyle: { width: 2 },   // color controlled by visualMap
        areaStyle: {
          // gradient controlled by index — gradient follows the dominant color
          color: {
            type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: 'rgba(16,185,129,0.12)' },
              { offset: 0.5, color: 'rgba(56,189,248,0.06)' },
              { offset: 1, color: 'rgba(16,185,129,0)' },
            ],
          },
        },

        // IS / OOS region labels via markArea (no line, only fill)
        markArea: {
          silent: true,
          data: [
            [
              { name: 'IN-SAMPLE',
                xAxis: 0,
                label: { show: true, color: '#10b981', fontSize: 9, position: 'insideTopLeft' },
                itemStyle: { color: 'rgba(16,185,129,0.04)' } },
              { xAxis: splitIdx - 1 },
            ],
            [
              { name: 'UNSEEN DATA',
                xAxis: splitIdx,
                label: { show: true, color: '#38bdf8', fontSize: 9, position: 'insideTopRight' },
                itemStyle: { color: 'rgba(56,189,248,0.04)' } },
              { xAxis: total - 1 },
            ],
          ],
        },
      }],
    }
  }, [simulationResult])

  return (
    <div className="w-full" style={{ height: 220 }}>
      <ReactECharts
        option={option}
        style={{ height: '100%', width: '100%' }}
        opts={{ renderer: 'canvas' }}
        notMerge
      />
    </div>
  )
}

export default function PnLChart() {
  return (
    <ErrorBoundary>
      <PnLChartInner />
    </ErrorBoundary>
  )
}
