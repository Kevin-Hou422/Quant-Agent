import { useWorkspaceStore } from '../../store/workspaceStore'
import type { PoolEntry } from '../../types'
import { Layers, TrendingUp, BarChart2 } from 'lucide-react'

// ── Formatting helpers ────────────────────────────────────────────────────────

const NUM = (v: number | null | undefined, d = 3) =>
  v == null || isNaN(v as number) ? '—' : (v as number).toFixed(d)

const PCT = (v: number | null | undefined) =>
  v == null || isNaN(v as number) ? '—' : `${((v as number) * 100).toFixed(1)}%`

const sharpeColor = (v: number) => {
  if (v >= 0.8) return 'text-emerald-400'
  if (v >= 0.3) return 'text-sky-400'
  if (v > 0)    return 'text-slate-300'
  return 'text-rose-400'
}

// ── DSL preview (truncate long expressions) ───────────────────────────────────

function DslPreview({ dsl, maxLen = 42 }: { dsl: string; maxLen?: number }) {
  const short = dsl.length > maxLen ? dsl.slice(0, maxLen) + '…' : dsl
  return (
    <span
      className="font-mono text-[10px] text-slate-300"
      title={dsl}
    >
      {short}
    </span>
  )
}

// ── Weight bar (visualises IC-weighted distribution) ─────────────────────────

function WeightBar({ weights, pool }: { weights: Record<string, number>; pool: PoolEntry[] }) {
  // Map DSL → weight; show bars in rank order
  const entries = pool
    .map((e, i) => ({ rank: i + 1, dsl: e.dsl, weight: weights[e.dsl] ?? 0 }))
    .sort((a, b) => b.weight - a.weight)

  const maxW = Math.max(...entries.map((e) => e.weight), 0.01)

  return (
    <div className="space-y-1">
      {entries.map((e) => (
        <div key={e.dsl} className="flex items-center gap-2">
          <span className="text-[9px] text-slate-500 w-4 text-right shrink-0">#{e.rank}</span>
          <div className="flex-1 bg-slate-800 rounded-full h-1.5 overflow-hidden">
            <div
              className="h-full bg-emerald-500 rounded-full transition-all"
              style={{ width: `${(e.weight / maxW) * 100}%` }}
            />
          </div>
          <span className="text-[9px] text-slate-400 w-8 text-right shrink-0 font-mono">
            {(e.weight * 100).toFixed(0)}%
          </span>
        </div>
      ))}
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export default function AlphaPoolPanel() {
  const { workflowResult } = useWorkspaceStore()

  if (!workflowResult) {
    return (
      <div className="flex flex-col items-center justify-center flex-1 text-slate-700 text-sm text-center gap-2 py-8">
        <Layers size={36} className="opacity-30" />
        <p>Run GP Optimize to see<br />the Alpha Pool here.</p>
      </div>
    )
  }

  const pool    = workflowResult.pool_top5 ?? []
  const combined = workflowResult.combined_metrics

  if (pool.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center flex-1 text-slate-600 text-sm gap-2 py-8">
        <Layers size={28} className="opacity-30" />
        <p>Pool is empty.</p>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-3">

      {/* ── Combined signal metrics ───────────────────────────────────── */}
      {combined && (
        <div className="rounded-lg border border-slate-700 bg-slate-800/60 p-3">
          <div className="flex items-center gap-1.5 mb-2">
            <TrendingUp size={11} className="text-emerald-400" />
            <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
              Combined Signal ({combined.n_alphas} Alphas)
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2 mb-3">
            <div className="bg-slate-900/60 rounded p-2 text-center">
              <div className="text-[9px] text-slate-500 mb-0.5">IC-IR</div>
              <div className={`text-sm font-mono font-semibold ${
                (combined.combined_ic_ir ?? 0) > 0.3 ? 'text-emerald-400' : 'text-amber-400'
              }`}>
                {NUM(combined.combined_ic_ir)}
              </div>
            </div>
            <div className="bg-slate-900/60 rounded p-2 text-center">
              <div className="text-[9px] text-slate-500 mb-0.5">Mean IC</div>
              <div className={`text-sm font-mono font-semibold ${
                (combined.combined_mean_ic ?? 0) > 0 ? 'text-sky-400' : 'text-slate-400'
              }`}>
                {NUM(combined.combined_mean_ic)}
              </div>
            </div>
          </div>

          {/* Weight distribution */}
          {combined.weights && Object.keys(combined.weights).length > 0 && (
            <div>
              <p className="text-[9px] text-slate-500 mb-1.5">IC-Weighted distribution</p>
              <WeightBar weights={combined.weights} pool={pool} />
            </div>
          )}
        </div>
      )}

      {/* ── Pool top-5 table ─────────────────────────────────────────── */}
      <div>
        <div className="flex items-center gap-1.5 mb-2">
          <BarChart2 size={11} className="text-slate-500" />
          <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
            Pool Top-{pool.length}
          </span>
          <span className="text-[9px] text-slate-600 ml-auto">
            Gen {workflowResult.generations_run}
          </span>
        </div>

        <div className="space-y-1.5">
          {pool.map((entry, idx) => (
            <div
              key={entry.dsl}
              className="rounded border border-slate-800 bg-slate-900/40 p-2 hover:border-slate-700 transition-colors"
            >
              {/* Rank + DSL */}
              <div className="flex items-start gap-2 mb-1.5">
                <span className="text-[10px] font-bold text-slate-600 w-4 shrink-0 pt-0.5">
                  #{idx + 1}
                </span>
                <DslPreview dsl={entry.dsl} />
              </div>

              {/* Metrics row */}
              <div className="grid grid-cols-3 gap-1 ml-6 text-center">
                <div>
                  <div className="text-[8px] text-slate-600">OOS Sharpe</div>
                  <div className={`text-[10px] font-mono font-semibold ${sharpeColor(entry.sharpe_oos)}`}>
                    {NUM(entry.sharpe_oos)}
                  </div>
                </div>
                <div>
                  <div className="text-[8px] text-slate-600">Fitness</div>
                  <div className="text-[10px] font-mono text-slate-300">
                    {NUM(entry.fitness)}
                  </div>
                </div>
                <div>
                  <div className="text-[8px] text-slate-600">Turnover</div>
                  <div className="text-[10px] font-mono text-slate-400">
                    {PCT(entry.turnover / 252)}
                  </div>
                </div>
              </div>

              {/* Overfitting badge */}
              {entry.overfitting_score > 0.5 && (
                <div className="ml-6 mt-1">
                  <span className="text-[8px] bg-amber-900/60 text-amber-400 rounded px-1 py-0.5">
                    overfit {(entry.overfitting_score * 100).toFixed(0)}%
                  </span>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Evolution summary */}
      {workflowResult.explanation && (
        <details className="group">
          <summary className="text-[10px] text-slate-600 hover:text-slate-400 cursor-pointer list-none select-none">
            <span className="group-open:hidden">▸ Show explanation</span>
            <span className="hidden group-open:inline">▾ Hide explanation</span>
          </summary>
          <p className="mt-1 text-[10px] text-slate-500 leading-relaxed whitespace-pre-wrap">
            {workflowResult.explanation}
          </p>
        </details>
      )}
    </div>
  )
}
