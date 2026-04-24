import ReactECharts from 'echarts-for-react'
import { useWorkspaceStore } from '../../store/workspaceStore'
import { useMemo, useRef, useEffect } from 'react'
import ErrorBoundary from '../ErrorBoundary'
import type { ECharts } from 'echarts'

/** Build cumulative return series from daily returns */
function toCumulative(dailyReturns: number[]): number[] {
  if (!dailyReturns?.length) return []
  let cum = 1
  return dailyReturns.map(r => {
    cum *= 1 + (isFinite(r) ? r : 0)
    return +(cum - 1).toFixed(6)
  })
}

function PnLChartInner() {
  const { simulationResult } = useWorkspaceStore()
  const chartRef = useRef<{ getEchartsInstance(): ECharts } | null>(null)

  const { option, hasData } = useMemo(() => {
    if (!simulationResult) {
      return {
        hasData: false,
        option: {
          backgroundColor: 'transparent',
          graphic: [{ type: 'text', left: 'center', top: 'middle',
            style: { text: 'Run a backtest to see PnL', fill: '#475569', fontSize: 13 } }],
          xAxis: { show: false }, yAxis: { show: false }, series: [],
        },
      }
    }

    const isRaw  = simulationResult.pnl_is  ?? []
    const oosRaw = simulationResult.pnl_oos ?? []

    if (!isRaw.length && !oosRaw.length) {
      return {
        hasData: false,
        option: {
          backgroundColor: 'transparent',
          graphic: [{ type: 'text', left: 'center', top: 'middle',
            style: { text: 'Backtest ran — no PnL series returned.', fill: '#64748b', fontSize: 11 } }],
          xAxis: { show: false }, yAxis: { show: false }, series: [],
        },
      }
    }

    // ── Build cumulative series ──────────────────────────────────────
    const isCum  = toCumulative(isRaw)
    const isEnd  = isCum.length ? isCum[isCum.length - 1] : 0
    const oosCum = toCumulative(oosRaw).map(v => +(v + isEnd).toFixed(6))

    const splitIdx = isCum.length         // first OOS index in the combined x-axis
    const total    = splitIdx + oosCum.length

    // MM-DD x-axis labels
    const today = new Date()
    const xData = Array.from({ length: total }, (_, i) => {
      const d = new Date(today)
      d.setDate(d.getDate() - (total - 1 - i))
      return d.toISOString().slice(5, 10)
    })

    // ── Two separate series so legend names match exactly ────────────
    //
    //  IS series : real values at [0, splitIdx-1], null elsewhere
    // OOS series : null until splitIdx-1 (join point = last IS value),
    //              then real OOS values  — connectNulls:false keeps them separate
    const isNulls  = new Array(oosCum.length).fill(null)
    const oosNulls = new Array(Math.max(0, splitIdx - 1)).fill(null)

    const isSeriesData: (number | null)[] = [...isCum, ...isNulls]
    const oosSeriesData: (number | null)[] = oosCum.length
      ? [...oosNulls, isEnd, ...oosCum]   // start OOS from the last IS value
      : []

    const legendItems = [
      { name: 'IN-SAMPLE',     icon: 'roundRect', itemStyle: { color: '#10b981' } },
      ...(oosCum.length
        ? [{ name: 'OUT-OF-SAMPLE', icon: 'roundRect', itemStyle: { color: '#0ea5e9' } }]
        : []),
    ]

    const series: object[] = [
      {
        name:         'IN-SAMPLE',
        type:         'line',
        data:         isSeriesData,
        connectNulls: false,
        symbol:       'none',
        lineStyle:    { width: 2, color: '#10b981' },
        areaStyle: {
          color: {
            type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0,   color: 'rgba(16,185,129,0.12)' },
              { offset: 1,   color: 'rgba(16,185,129,0.01)' },
            ],
          },
        },
      },
      ...(oosCum.length ? [{
        name:         'OUT-OF-SAMPLE',
        type:         'line',
        data:         oosSeriesData,
        connectNulls: false,
        symbol:       'none',
        lineStyle:    { width: 2, color: '#0ea5e9' },
        areaStyle: {
          color: {
            type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0,   color: 'rgba(14,165,233,0.10)' },
              { offset: 1,   color: 'rgba(14,165,233,0.01)' },
            ],
          },
        },
      }] : []),
    ]

    return {
      hasData: true,
      option: {
        backgroundColor: 'transparent',
        textStyle: { color: '#94a3b8' },
        animation: false,

        legend: {
          show: true, top: 0, right: 4,
          data: legendItems,
          textStyle: { color: '#64748b', fontSize: 10 },
          itemWidth: 14, itemHeight: 4,
        },

        tooltip: {
          trigger: 'axis',
          backgroundColor: '#1e293b',
          borderColor: '#334155',
          textStyle: { color: '#e2e8f0', fontSize: 11 },
          formatter: (params: { seriesName: string; dataIndex: number; axisValue: string; value: number | null }[]) => {
            const p = params.find(x => x.value != null)
            if (!p || p.value == null) return ''
            const isIS = p.seriesName === 'IN-SAMPLE'
            const badge = isIS
              ? '<span style="color:#10b981">● IS</span>'
              : '<span style="color:#0ea5e9">● OOS</span>'
            return `${p.axisValue} ${badge}<br/><b>${(p.value * 100).toFixed(2)}%</b>`
          },
        },

        grid: { left: 52, right: 8, top: 28, bottom: 36 },

        xAxis: {
          type: 'category',
          data: xData,
          boundaryGap: false,
          axisLine: { lineStyle: { color: '#334155' } },
          axisTick: { show: false },
          axisLabel: {
            color: '#475569', fontSize: 9,
            interval: Math.max(1, Math.floor(total / 6)),
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

        series,
      },
    }
  }, [simulationResult])

  // Re-fit chart when panel is resized
  useEffect(() => {
    const chart = chartRef.current?.getEchartsInstance?.()
    chart?.resize()
  })

  return (
    <div className="w-full h-full min-h-[180px]">
      {!hasData && !simulationResult && (
        <div className="flex items-center justify-center h-full">
          <div className="text-center text-slate-600 text-xs">
            <p className="text-2xl mb-2">📈</p>
            <p>Run a backtest to visualise PnL</p>
          </div>
        </div>
      )}
      <ReactECharts
        ref={chartRef as any}
        option={option}
        style={{ height: '100%', width: '100%', minHeight: 180 }}
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
