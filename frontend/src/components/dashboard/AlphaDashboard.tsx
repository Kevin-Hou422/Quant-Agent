import { useCallback, useEffect, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import {
  Activity, AlertTriangle, RefreshCw, ArrowRight, Clock, CheckCircle2,
} from 'lucide-react'
import {
  apiFetchDashboard, apiFetchICHistory, apiPatchAlphaStatus, apiFetchSchedulerStatus,
} from '../../api/client'
import type { AlphaDashboardRow, ICHistoryData, SchedulerStatus } from '../../types'

/**
 * FE-5.1/5.2/5.3: Alpha lifecycle dashboard.
 *
 * - Card list of all non-terminal alphas with rolling IC-IR + decay alerts
 * - Click a card → IC history line chart (FE-5.2)
 * - Status transition buttons validated by the backend state machine
 * - Scheduler status bar (FE-5.3)
 */

const STATUS_STYLE: Record<string, string> = {
  candidate:  'bg-slate-800 text-slate-400',
  validated:  'bg-sky-900/60 text-sky-400',
  paper:      'bg-violet-900/60 text-violet-400',
  active:     'bg-emerald-900/60 text-emerald-400',
  decaying:   'bg-amber-900/60 text-amber-400',
}

const NUM = (v: number | null | undefined, d = 4) => (v == null ? '—' : v.toFixed(d))

export default function AlphaDashboard() {
  const [rows, setRows]         = useState<AlphaDashboardRow[]>([])
  const [nAlerts, setNAlerts]   = useState(0)
  const [selected, setSelected] = useState<number | null>(null)
  const [icData, setIcData]     = useState<ICHistoryData | null>(null)
  const [sched, setSched]       = useState<SchedulerStatus | null>(null)
  const [loading, setLoading]   = useState(true)
  const [error, setError]       = useState<string | null>(null)

  const refresh = useCallback(() => {
    setLoading(true)
    setError(null)
    Promise.all([apiFetchDashboard(), apiFetchSchedulerStatus()])
      .then(([dash, s]) => {
        setRows(dash.data.rows)
        setNAlerts(dash.data.n_alerts)
        setSched(s.data)
      })
      .catch((e) => setError(e?.message ?? 'Failed to load dashboard'))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => { refresh() }, [refresh])

  useEffect(() => {
    if (selected == null) { setIcData(null); return }
    let cancelled = false
    apiFetchICHistory(selected)
      .then((res) => { if (!cancelled) setIcData(res.data) })
      .catch(() => { if (!cancelled) setIcData(null) })
    return () => { cancelled = true }
  }, [selected])

  const transition = async (alphaId: number, status: string) => {
    try {
      await apiPatchAlphaStatus(alphaId, status)
      refresh()
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? e?.message ?? 'Transition failed')
    }
  }

  const icOption = icData && icData.points.length > 0 ? {
    grid: { left: 44, right: 16, top: 24, bottom: 28 },
    xAxis: {
      type: 'category',
      data: icData.points.map((p) => p.date),
      axisLabel: { color: '#64748b', fontSize: 9 },
      axisLine: { lineStyle: { color: '#1e293b' } },
    },
    yAxis: {
      type: 'value',
      axisLabel: { color: '#64748b', fontSize: 9 },
      splitLine: { lineStyle: { color: '#1e293b' } },
    },
    series: [{
      type: 'line',
      data: icData.points.map((p) => p.realized_ic),
      showSymbol: false,
      lineStyle: { color: '#34d399', width: 1.5 },
      markLine: {
        silent: true,
        symbol: 'none',
        data: [{ yAxis: 0 }],
        lineStyle: { color: '#f43f5e', type: 'dashed', width: 1 },
        label: { show: false },
      },
    }],
    tooltip: { trigger: 'axis', backgroundColor: '#0f172a', borderColor: '#334155',
               textStyle: { color: '#e2e8f0', fontSize: 11 } },
  } : null

  return (
    <div className="h-full flex flex-col bg-slate-950 overflow-hidden">
      {/* Top bar */}
      <div className="flex items-center gap-3 px-4 py-3 border-b border-slate-800 shrink-0">
        <Activity size={16} className="text-emerald-400" />
        <span className="text-sm font-semibold text-slate-200">Alpha Lifecycle Dashboard</span>
        {nAlerts > 0 && (
          <span className="flex items-center gap-1 text-[10px] font-semibold px-2 py-0.5 rounded bg-rose-900/60 text-rose-400">
            <AlertTriangle size={11} />
            {nAlerts} decay alert{nAlerts > 1 ? 's' : ''}
          </span>
        )}
        <button
          onClick={refresh}
          className="ml-auto flex items-center gap-1 text-[11px] text-slate-400 hover:text-slate-200 transition-colors"
        >
          <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      {/* FE-5.3: scheduler status bar */}
      <div className="flex items-center gap-2 px-4 py-1.5 border-b border-slate-800 bg-slate-900/60 text-[10px] shrink-0">
        <Clock size={11} className="text-slate-500" />
        {sched?.running ? (
          <>
            <span className="text-emerald-400 font-medium">Scheduler running</span>
            {sched.jobs.map((j) => (
              <span key={j.id} className="text-slate-500">
                {j.name} · next {j.next_run ? j.next_run.slice(0, 16) : '—'}
              </span>
            ))}
          </>
        ) : (
          <span className="text-slate-500">
            Scheduler stopped (set ENABLE_SCHEDULER=true to run daily monitoring)
          </span>
        )}
      </div>

      {error && (
        <div className="px-4 py-2 text-[11px] text-rose-400 bg-rose-950/40 border-b border-rose-900/40">
          {error}
        </div>
      )}

      <div className="flex-1 flex min-h-0">
        {/* Card list */}
        <div className="w-[420px] shrink-0 overflow-y-auto border-r border-slate-800 p-3 flex flex-col gap-2">
          {rows.length === 0 && !loading && (
            <div className="text-center text-slate-600 text-xs mt-8">
              No tracked alphas yet.<br />Run a backtest or GP optimize to populate the ledger.
            </div>
          )}
          {rows.map((r) => (
            <div
              key={r.alpha_id}
              onClick={() => setSelected(r.alpha_id === selected ? null : r.alpha_id)}
              className={`rounded-lg border p-3 cursor-pointer transition-colors ${
                r.alpha_id === selected
                  ? 'border-emerald-700 bg-slate-900'
                  : 'border-slate-800 bg-slate-900/50 hover:border-slate-700'
              }`}
            >
              <div className="flex items-center gap-2">
                <span className={`text-[9px] font-semibold px-1.5 py-0.5 rounded uppercase ${
                  STATUS_STYLE[r.status] ?? 'bg-slate-800 text-slate-400'
                }`}>
                  {r.status}
                </span>
                <span className="text-[10px] text-slate-500">#{r.alpha_id}</span>
                {r.has_alert && (
                  <span className="flex items-center gap-0.5 text-[9px] text-rose-400 font-semibold">
                    <AlertTriangle size={10} /> decay
                  </span>
                )}
                <span className="ml-auto text-[11px] font-mono text-slate-300">
                  SR {NUM(r.sharpe, 2)}
                </span>
              </div>
              <p className="mt-1.5 text-[11px] font-mono text-slate-400 truncate" title={r.dsl}>
                {r.dsl}
              </p>
              <div className="mt-2 flex gap-4 text-[10px] text-slate-500">
                <span>IC days: <b className="text-slate-300">{r.n_ic_days}</b></span>
                <span>roll IC: <b className={r.rolling_mean_ic != null && r.rolling_mean_ic < 0 ? 'text-rose-400' : 'text-slate-300'}>{NUM(r.rolling_mean_ic)}</b></span>
                <span>IC-IR: <b className="text-slate-300">{NUM(r.rolling_ic_ir, 2)}</b></span>
                {r.consecutive_neg > 0 && (
                  <span className="text-amber-500">neg×{r.consecutive_neg}</span>
                )}
              </div>
              {/* Status transition buttons */}
              {r.allowed_next.length > 0 && (
                <div className="mt-2 flex gap-1.5 flex-wrap">
                  {r.allowed_next.map((s) => (
                    <button
                      key={s}
                      onClick={(e) => { e.stopPropagation(); transition(r.alpha_id, s) }}
                      className="flex items-center gap-0.5 text-[9px] px-1.5 py-0.5 rounded border border-slate-700 text-slate-400 hover:text-slate-200 hover:border-slate-500 transition-colors"
                    >
                      <ArrowRight size={9} /> {s}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>

        {/* FE-5.2: IC history chart */}
        <div className="flex-1 min-w-0 p-4 overflow-y-auto">
          {selected == null ? (
            <div className="h-full flex flex-col items-center justify-center text-slate-700 text-sm gap-2">
              <Activity size={40} className="opacity-30" />
              <p>Select an alpha to view its realized IC history</p>
            </div>
          ) : icData == null || icData.points.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-slate-600 text-sm gap-2">
              <CheckCircle2 size={32} className="opacity-30" />
              <p>No IC history recorded for alpha #{selected} yet.</p>
              <p className="text-[11px] text-slate-700">
                IC records are written by AlphaMonitor.update() during daily monitoring.
              </p>
            </div>
          ) : (
            <div className="flex flex-col gap-3">
              <div className="flex items-center gap-4 text-xs text-slate-400">
                <span className="font-semibold text-slate-200">Alpha #{selected} — Realized IC</span>
                <span>rolling mean: <b className="font-mono text-slate-200">{NUM(icData.rolling_mean_ic)}</b></span>
                <span>rolling IC-IR: <b className="font-mono text-slate-200">{NUM(icData.rolling_ic_ir, 2)}</b></span>
                <span className="text-slate-600">{icData.points.length} days</span>
              </div>
              {icOption && (
                <ReactECharts option={icOption} style={{ height: 320 }} notMerge />
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
