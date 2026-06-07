import { useEffect, useState } from 'react'
import { Database, Globe2, Building2, ChevronRight, Check, ArrowLeft, RefreshCw, Calendar, Hash } from 'lucide-react'
import { useWorkspaceStore } from '../../store/workspaceStore'
import { apiFetchDatasets } from '../../api/client'
import type { DatasetInfo } from '../../types'

// ── Region / provider badges ─────────────────────────────────────────────────

const REGION_COLORS: Record<string, string> = {
  US:       'bg-blue-900/60 text-blue-300 border-blue-800',
  China:    'bg-red-900/60 text-red-300 border-red-800',
  HongKong: 'bg-violet-900/60 text-violet-300 border-violet-800',
  Global:   'bg-amber-900/60 text-amber-300 border-amber-800',
}

const PROVIDER_LABELS: Record<string, string> = {
  yfinance:      'Yahoo Finance',
  akshare:       'AkShare',
  ccxt_binance:  'Binance (CCXT)',
}

const REGION_LABELS: Record<string, string> = {
  US:       'US Equities',
  China:    'China A-shares',
  HongKong: 'Hong Kong',
  Global:   'Crypto',
}

// ── Ticker chip ───────────────────────────────────────────────────────────────

function TickerChip({ ticker }: { ticker: string }) {
  return (
    <span className="px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-[10px] font-mono text-slate-300">
      {ticker}
    </span>
  )
}

// ── Dataset card (left panel) ────────────────────────────────────────────────

function DatasetCard({
  ds,
  isActive,
  isSelected,
  onClick,
}: {
  ds:         DatasetInfo
  isActive:   boolean
  isSelected: boolean
  onClick:    () => void
}) {
  const regionCls = REGION_COLORS[ds.region] ?? 'bg-slate-800 text-slate-400 border-slate-700'

  return (
    <button
      onClick={onClick}
      className={`
        w-full text-left px-4 py-3.5 border-b border-slate-800 transition-all duration-100
        ${isSelected
          ? 'bg-slate-800 border-l-2 border-l-emerald-500'
          : 'hover:bg-slate-800/60 border-l-2 border-l-transparent'}
      `}
    >
      <div className="flex items-start justify-between gap-2 mb-2">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-slate-200 truncate">
              {ds.name}
            </span>
            {isActive && (
              <span className="flex items-center gap-1 text-[9px] font-semibold text-emerald-400 bg-emerald-900/40 border border-emerald-800/60 rounded px-1.5 py-0.5 shrink-0">
                <Check size={8} /> ACTIVE
              </span>
            )}
          </div>
          <p className="text-[11px] text-slate-500 mt-0.5">{ds.industry}</p>
        </div>
        <span className={`text-[9px] font-medium border rounded px-1.5 py-0.5 shrink-0 ${regionCls}`}>
          {REGION_LABELS[ds.region] ?? ds.region}
        </span>
      </div>

      <div className="flex items-center gap-3 text-[10px] text-slate-500">
        <span className="flex items-center gap-1">
          <Hash size={9} />
          {ds.n_assets} assets
        </span>
        <span className="flex items-center gap-1">
          <Calendar size={9} />
          from {ds.start}
        </span>
      </div>
    </button>
  )
}

// ── Detail panel (right side) ────────────────────────────────────────────────

function DatasetDetail({
  ds,
  isActive,
  startDate,
  endDate,
  onStartChange,
  onEndChange,
  onActivate,
}: {
  ds:            DatasetInfo
  isActive:      boolean
  startDate:     string
  endDate:       string
  onStartChange: (v: string) => void
  onEndChange:   (v: string) => void
  onActivate:    () => void
}) {
  const regionCls = REGION_COLORS[ds.region] ?? 'bg-slate-800 text-slate-400 border-slate-700'

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-6 py-5 border-b border-slate-800 shrink-0">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-base font-semibold text-slate-100">{ds.name}</h2>
            <p className="text-sm text-slate-500 mt-0.5">{ds.industry}</p>
          </div>
          <span className={`text-xs font-medium border rounded-md px-2.5 py-1 ${regionCls}`}>
            {REGION_LABELS[ds.region] ?? ds.region}
          </span>
        </div>

        {/* Metadata row */}
        <div className="flex flex-wrap gap-4 mt-4 text-xs text-slate-400">
          <div className="flex items-center gap-1.5">
            <Globe2 size={12} className="text-slate-600" />
            <span>{PROVIDER_LABELS[ds.provider] ?? ds.provider}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <Building2 size={12} className="text-slate-600" />
            <span>{ds.n_assets} tickers</span>
          </div>
          <div className="flex items-center gap-1.5">
            <Calendar size={12} className="text-slate-600" />
            <span>Data available from {ds.start}</span>
          </div>
        </div>
      </div>

      {/* Date range config */}
      <div className="px-6 py-4 border-b border-slate-800 bg-slate-900/50 shrink-0">
        <p className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-3">
          Backtest Date Range
        </p>
        <div className="flex items-center gap-3">
          <div className="flex-1">
            <label className="text-[10px] text-slate-600 block mb-1">Start</label>
            <input
              type="date"
              value={startDate}
              min={ds.start}
              max={endDate}
              onChange={(e) => onStartChange(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded px-2.5 py-1.5 text-xs text-slate-200
                         focus:outline-none focus:border-emerald-600 transition-colors"
            />
          </div>
          <ChevronRight size={14} className="text-slate-600 mt-4 shrink-0" />
          <div className="flex-1">
            <label className="text-[10px] text-slate-600 block mb-1">End</label>
            <input
              type="date"
              value={endDate}
              min={startDate}
              onChange={(e) => onEndChange(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded px-2.5 py-1.5 text-xs text-slate-200
                         focus:outline-none focus:border-emerald-600 transition-colors"
            />
          </div>
        </div>
      </div>

      {/* Activate button */}
      <div className="px-6 py-4 border-b border-slate-800 shrink-0">
        <button
          onClick={onActivate}
          className={`
            w-full flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-medium transition-colors
            ${isActive
              ? 'bg-emerald-900/40 border border-emerald-700 text-emerald-400 cursor-default'
              : 'bg-emerald-600 hover:bg-emerald-500 text-white'}
          `}
        >
          {isActive ? (
            <>
              <Check size={14} />
              Active dataset
            </>
          ) : (
            <>
              <Database size={14} />
              Use this dataset
            </>
          )}
        </button>
      </div>

      {/* Universe list */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        <p className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-3">
          Universe ({ds.universe.length} tickers)
        </p>
        <div className="flex flex-wrap gap-1.5">
          {ds.universe.map((t) => (
            <TickerChip key={t} ticker={t} />
          ))}
        </div>
      </div>
    </div>
  )
}

// ── Empty state ───────────────────────────────────────────────────────────────

function EmptyDetail() {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center gap-3 text-slate-700">
      <Database size={40} className="opacity-30" />
      <p className="text-sm">Select a dataset to view details</p>
    </div>
  )
}

// ── DatasetView ───────────────────────────────────────────────────────────────

export default function DatasetView() {
  const { datasets, setDatasets, simConfig, setSimConfig, prevView, setActiveView } = useWorkspaceStore()
  const [selected, setSelected]   = useState<DatasetInfo | null>(null)
  const [loading, setLoading]     = useState(false)
  const [localStart, setLocalStart] = useState(simConfig.start_date)
  const [localEnd,   setLocalEnd]   = useState(simConfig.end_date)

  // Keep local dates in sync when a different card is selected
  useEffect(() => {
    setLocalStart(simConfig.start_date)
    setLocalEnd(simConfig.end_date)
  }, [selected])

  // Load datasets on first mount if not cached
  useEffect(() => {
    if (datasets.length > 0) return
    setLoading(true)
    apiFetchDatasets()
      .then((r) => setDatasets(r.data.datasets))
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  const handleActivate = () => {
    if (!selected) return
    setSimConfig({
      dataset:    selected.name,
      start_date: localStart,
      end_date:   localEnd,
    })
    setActiveView(prevView)   // return to previous view
  }

  const handleRefresh = () => {
    setLoading(true)
    apiFetchDatasets()
      .then((r) => setDatasets(r.data.datasets))
      .catch(() => {})
      .finally(() => setLoading(false))
  }

  // Group datasets by region for the sidebar
  const grouped: Record<string, DatasetInfo[]> = {}
  for (const ds of datasets) {
    const key = REGION_LABELS[ds.region] ?? ds.region
    ;(grouped[key] ??= []).push(ds)
  }
  const regionOrder = ['US Equities', 'China A-shares', 'Hong Kong', 'Crypto']

  return (
    <div className="h-full flex flex-col bg-slate-950 overflow-hidden">
      {/* Top bar */}
      <div className="flex items-center gap-3 px-4 py-3 border-b border-slate-800 bg-slate-900 shrink-0">
        <button
          onClick={() => setActiveView(prevView)}
          className="text-slate-500 hover:text-slate-300 transition-colors"
          title="Back"
        >
          <ArrowLeft size={15} />
        </button>
        <Database size={15} className="text-emerald-400" />
        <span className="text-sm font-semibold text-slate-200">Dataset Registry</span>
        <span className="ml-1 text-xs text-slate-600">
          {datasets.length > 0 ? `${datasets.length} datasets` : ''}
        </span>

        {/* Active badge */}
        <div className="ml-auto flex items-center gap-2">
          <span className="text-[10px] text-slate-600">Active:</span>
          <span className="text-[11px] font-mono font-semibold text-emerald-400 bg-emerald-900/30 border border-emerald-800/50 rounded px-2 py-0.5">
            {simConfig.dataset}
          </span>
          <span className="text-[10px] text-slate-600">{simConfig.start_date} → {simConfig.end_date}</span>

          <button
            onClick={handleRefresh}
            disabled={loading}
            className="ml-2 text-slate-600 hover:text-slate-400 transition-colors disabled:opacity-40"
            title="Refresh"
          >
            <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />
          </button>
        </div>
      </div>

      {/* Body: left list + right detail */}
      <div className="flex flex-1 min-h-0">

        {/* Left: dataset list */}
        <div className="w-72 shrink-0 border-r border-slate-800 overflow-y-auto bg-slate-900/40">
          {loading && datasets.length === 0 ? (
            <div className="flex items-center justify-center h-32 text-slate-600 text-sm gap-2">
              <RefreshCw size={14} className="animate-spin" />
              Loading datasets…
            </div>
          ) : (
            regionOrder
              .filter((r) => grouped[r])
              .map((regionLabel) => (
                <div key={regionLabel}>
                  {/* Region header */}
                  <div className="px-4 py-2 border-b border-slate-800 bg-slate-900/70 sticky top-0">
                    <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">
                      {regionLabel}
                    </span>
                  </div>
                  {grouped[regionLabel].map((ds) => (
                    <DatasetCard
                      key={ds.name}
                      ds={ds}
                      isActive={ds.name === simConfig.dataset}
                      isSelected={selected?.name === ds.name}
                      onClick={() => setSelected(ds)}
                    />
                  ))}
                </div>
              ))
          )}
        </div>

        {/* Right: detail */}
        <div className="flex-1 min-w-0 overflow-hidden">
          {selected ? (
            <DatasetDetail
              ds={selected}
              isActive={selected.name === simConfig.dataset}
              startDate={localStart}
              endDate={localEnd}
              onStartChange={setLocalStart}
              onEndChange={setLocalEnd}
              onActivate={handleActivate}
            />
          ) : (
            <EmptyDetail />
          )}
        </div>
      </div>
    </div>
  )
}
