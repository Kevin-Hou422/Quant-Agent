import ReactECharts from 'echarts-for-react'
import { useWorkspaceStore } from '../../store/workspaceStore'
import { useMemo } from 'react'

export default function PnLChart() {
  const { simulationResult } = useWorkspaceStore()

  const option = useMemo(() => {
    if (!simulationResult) {
      return {
        backgroundColor: 'transparent',
        textStyle: { color: '#94a3b8' },
        graphic: [{
          type: 'text',
          left: 'center', top: 'middle',
          style: { text: 'Run a backtest to see results', fill: '#475569', fontSize: 14 },
        }],
        xAxis: { show: false },
        yAxis: { show: false },
        series: [],
      }
    }

    // Use pnl_is/pnl_oos if available, otherwise show empty series
    const isReturns: number[] = simulationResult.pnl_is ?? []
    const oosReturns: number[] = simulationResult.pnl_oos ?? []

    // Generate synthetic date labels
    const totalLen = isReturns.length + oosReturns.length
    const today = new Date()
    const dates: string[] = Array.from({ length: totalLen }, (_, i) => {
      const d = new Date(today)
      d.setDate(d.getDate() - (totalLen - i))
      return d.toISOString().slice(0, 10)
    })

    const splitDate = dates[isReturns.length - 1] ?? dates[0]

    // Cumulative product
    const cumProd = (rets: number[]) => {
      let cum = 1
      return rets.map(r => { cum *= (1 + r); return +(cum - 1).toFixed(4) })
    }

    const isCum = cumProd(isReturns)
    const oosCum = cumProd(oosReturns).map(v => v + (isCum[isCum.length - 1] ?? 0))

    const allCum = [...isCum, ...oosCum]

    return {
      backgroundColor: 'transparent',
      textStyle: { color: '#94a3b8' },
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#1e293b',
        borderColor: '#334155',
        textStyle: { color: '#e2e8f0', fontSize: 12 },
        formatter: (params: any[]) => {
          const p = params[0]
          return `${p.axisValue}<br/>${p.marker} PnL: <b>${(p.value * 100).toFixed(2)}%</b>`
        },
      },
      grid: { left: 48, right: 24, top: 32, bottom: 40 },
      xAxis: {
        type: 'category',
        data: dates,
        axisLine: { lineStyle: { color: '#334155' } },
        axisLabel: { color: '#64748b', fontSize: 10, rotate: 30 },
        splitLine: { show: false },
      },
      yAxis: {
        type: 'value',
        axisLine: { show: false },
        axisLabel: {
          color: '#64748b', fontSize: 10,
          formatter: (v: number) => `${(v * 100).toFixed(1)}%`,
        },
        splitLine: { lineStyle: { color: '#1e293b' } },
      },
      series: [
        {
          type: 'line',
          data: allCum,
          smooth: false,
          symbol: 'none',
          lineStyle: { color: '#10b981', width: 2 },
          areaStyle: {
            color: {
              type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(16,185,129,0.15)' },
                { offset: 1, color: 'rgba(16,185,129,0)' },
              ],
            },
          },
          markArea: {
            silent: true,
            data: [
              [
                { name: 'IN-SAMPLE', xAxis: dates[0], itemStyle: { color: 'rgba(59,130,246,0.06)' }, label: { color: '#3b82f6', fontSize: 10 } },
                { xAxis: splitDate },
              ],
              [
                { name: 'UNSEEN DATA', xAxis: splitDate, itemStyle: { color: 'rgba(244,63,94,0.06)' }, label: { color: '#f43f5e', fontSize: 10 } },
                { xAxis: dates[dates.length - 1] },
              ],
            ],
          },
          markLine: {
            silent: true,
            data: [{ xAxis: splitDate, name: 'OOS Split' }],
            lineStyle: { color: '#f43f5e', type: 'dashed', width: 1.5 },
            label: { show: true, position: 'insideEndTop', color: '#f43f5e', fontSize: 10, formatter: 'OOS →' },
            symbol: ['none', 'none'],
          },
        },
      ],
    }
  }, [simulationResult])

  return (
    <div className="w-full h-full">
      <ReactECharts
        option={option}
        style={{ height: '100%', width: '100%' }}
        theme="dark"
        opts={{ renderer: 'canvas' }}
      />
    </div>
  )
}
