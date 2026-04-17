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

    // ── Debug: log raw payload ───────────────────────────────────────
    console.log('[PnLChart] simulationResult keys:', Object.keys(simulationResult))
    console.log('[PnLChart] pnl_is length:', simulationResult.pnl_is?.length ?? 'MISSING')
    console.log('[PnLChart] pnl_oos length:', simulationResult.pnl_oos?.length ?? 'MISSING')
    console.log('[PnLChart] overfitting_score:', simulationResult.overfitting_score)

    const isRaw  = simulationResult.pnl_is  ?? []
    const oosRaw = simulationResult.pnl_oos ?? []

    if (!isRaw.length && !oosRaw.length) {
      return {
        hasData: false,
        option: {
          backgroundColor: 'transparent',
          graphic: [{ type: 'text', left: 'center', top: 'middle',
            style: { text: 'Backtest ran — no PnL series returned.\nCheck backend net_returns field.', fill: '#64748b', fontSize: 11 } }],
          xAxis: { show: false }, yAxis: { show: false }, series: [],
        },
      }
    }

    // ── Build cumulative series ──────────────────────────────────────
    const isCum   = toCumulative(isRaw)
    const isEnd   = isCum[isCum.length - 1] ?? 0
    const oosCum  = toCumulative(oosRaw).map(v => +(v + isEnd).toFixed(6))

    // Single continuous array: IS then OOS
    const allValues = [...isCum, ...oosCum]
    const splitIdx  = isCum.length   // first OOS index

    // MM-DD labels
    const total = allValues.length
    const today = new Date()
    const xData = Array.from({ length: total }, (_, i) => {
      const d = new Date(today)
      d.setDate(d.getDate() - (total - 1 - i))
      return d.toISOString().slice(5, 10)
    })

    console.log('[PnLChart] chart data:', { total, splitIdx, firstVal: allValues[0], lastVal: allValues[total - 1] })

    return {
      hasData: true,
      option: {
        backgroundColor: 'transparent',
        textStyle: { color: '#94a3b8' },
        animation: false,

        // ── visualMap: colour line by x-index ───────────────────────
        visualMap: [{
          show: false,
          type: 'piecewise',
          dimension: 0,
          seriesIndex: 0,
          pieces: [
            { min: 0,         max: splitIdx - 1, color: '#10b981' },   // IS: emerald
            { min: splitIdx,  max: total,        color: '#0ea5e9' },   // OOS: sky
          ],
        }],

        tooltip: {
          trigger: 'axis',
          backgroundColor: '#1e293b',
          borderColor: '#334155',
          textStyle: { color: '#e2e8f0', fontSize: 11 },
          formatter: (params: { dataIndex: number; axisValue: string; value: number }[]) => {
            const p = params[0]
            if (!p) return ''
            const isIS = p.dataIndex < splitIdx
            const badge = isIS
              ? '<span style="color:#10b981">● IS</span>'
              : '<span style="color:#0ea5e9">● OOS</span>'
            return `${p.axisValue} ${badge}<br/><b>${(p.value * 100).toFixed(2)}%</b>`
          },
        },

        legend: {
          show: true, top: 0, right: 4,
          data: [
            { name: 'IN-SAMPLE',      icon: 'roundRect', itemStyle: { color: '#10b981' } },
            { name: 'OUT-OF-SAMPLE',  icon: 'roundRect', itemStyle: { color: '#0ea5e9' } },
          ],
          textStyle: { color: '#64748b', fontSize: 10 },
          itemWidth: 14, itemHeight: 4,
        },

        grid: { left: 52, right: 8, top: 32, bottom: 36 },

        xAxis: {
          type: 'category',
          data: xData,
          boundaryGap: false,
          axisLine: { lineStyle: { color: '#334155' } },
          axisTick: { show: false },
          axisLabel: {
            color: '#475569', fontSize: 9, rotate: 0,
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

        series: [{
          name: 'PnL',
          type: 'line',
          data: allValues,
          smooth: false,
          symbol: 'none',
          lineStyle: { width: 2 },  // colour controlled by visualMap
          areaStyle: {
            color: {
              type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [
                { offset: 0,   color: 'rgba(16,185,129,0.15)' },
                { offset: 0.6, color: 'rgba(14,165,233,0.06)' },
                { offset: 1,   color: 'rgba(16,185,129,0)' },
              ],
            },
          },
          // IS / OOS area shading
          markArea: {
            silent: true,
            data: [
              [
                { name: 'IS', xAxis: 0,
                  label: { color: '#10b981', fontSize: 9, position: 'insideTopLeft' },
                  itemStyle: { color: 'rgba(16,185,129,0.04)' } },
                { xAxis: Math.max(0, splitIdx - 1) },
              ],
              ...(oosCum.length ? [[
                { name: 'OOS', xAxis: splitIdx,
                  label: { color: '#0ea5e9', fontSize: 9, position: 'insideTopRight' },
                  itemStyle: { color: 'rgba(14,165,233,0.04)' } },
                { xAxis: total - 1 },
              ]] : []),
            ],
          },
        }],
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
