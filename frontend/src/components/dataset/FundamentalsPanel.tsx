/**
 * FundamentalsPanel — FE-10 基本面字段浏览 + 披露时点标注
 *
 * 数据来自 GET /api/datasets/{name}/fundamentals，那个接口只读本地已存的事实
 * （不向 SEC / 行情源发请求），所以选中数据集时直接取，不需要用户点按钮。
 *
 * 面板上每个数字都绑回接口字段：覆盖率是 `fields[].coverage`，披露日是
 * `tickers[].last_filed`。接口没给的（coverage 为 null、fields 为 null）显示 "—"，
 * 不拿 0 或默认值顶上。
 */
import { useEffect, useState } from 'react'
import { FileText, Loader2, AlertTriangle } from 'lucide-react'
import { apiFetchDatasetFundamentals } from '../../api/client'
import type { DatasetFundamentals, FundamentalFieldClass } from '../../types'

const CLASS_STYLE: Record<FundamentalFieldClass, string> = {
  valuation: 'bg-sky-900/50 text-sky-300 border-sky-800',
  quarterly: 'bg-violet-900/50 text-violet-300 border-violet-800',
  event:     'bg-amber-900/50 text-amber-300 border-amber-800',
}

const CLASS_LABEL: Record<FundamentalFieldClass, string> = {
  valuation: 'daily',
  quarterly: 'quarterly step',
  event:     'event counter',
}

export function coverageText(coverage: number | null): string {
  return coverage === null ? '—' : `${(coverage * 100).toFixed(0)}%`
}

export default function FundamentalsPanel({ datasetName }: { datasetName: string }) {
  const [data, setData]       = useState<DatasetFundamentals | null>(null)
  const [loading, setLoading] = useState(false)
  const [failed, setFailed]   = useState(false)

  useEffect(() => {
    let cancelled = false
    setData(null)
    setFailed(false)
    setLoading(true)
    apiFetchDatasetFundamentals(datasetName)
      .then((r) => { if (!cancelled) setData(r.data) })
      .catch(() => { if (!cancelled) setFailed(true) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [datasetName])

  return (
    <div className="px-6 py-4 border-b border-slate-800 shrink-0" data-testid="fundamentals-panel">
      <div className="flex items-center gap-2 mb-2">
        <FileText size={12} className="text-slate-600" />
        <p className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Fundamentals</p>
        {data && data.supported && (
          <span
            data-testid="fundamentals-switch"
            className={`text-[10px] border rounded px-1.5 py-0.5 ${
              data.enabled
                ? 'bg-emerald-900/40 text-emerald-300 border-emerald-800'
                : 'bg-slate-800 text-slate-500 border-slate-700'
            }`}
          >
            {data.enabled ? 'attached to this dataset' : 'off (ENABLE_FUNDAMENTALS=false)'}
          </span>
        )}
      </div>

      {loading && (
        <div className="flex items-center gap-1.5 text-[10px] text-slate-500">
          <Loader2 size={10} className="animate-spin" /> Reading local filings…
        </div>
      )}
      {failed && (
        <p className="text-[10px] text-rose-400" data-testid="fundamentals-error">
          Could not read the fundamentals status.
        </p>
      )}

      {data && !data.supported && (
        <p className="text-[10px] text-slate-600" data-testid="fundamentals-unsupported">
          Fundamentals come from SEC filings and exist for US equities only.
        </p>
      )}

      {data && data.supported && (
        <>
          <p className="text-[10px] text-slate-500 mb-2" data-testid="fundamentals-summary">
            {data.n_with_facts} / {data.n_tickers} tickers have filings on disk
            {data.n_stale > 0 && <span className="text-amber-400"> · {data.n_stale} stale</span>}
            {!data.identity_configured && (
              <span className="text-slate-600"> · EDGAR_USER_AGENT not set (no updates)</span>
            )}
          </p>
          <p className="text-[10px] text-slate-600 mb-2" data-testid="fundamentals-rule">
            {data.visibility_rule}
          </p>

          <table className="w-full text-[10px] mb-3">
            <thead>
              <tr className="text-slate-600 text-left">
                <th className="font-medium pb-1">Field</th>
                <th className="font-medium pb-1">Updates</th>
                <th className="font-medium pb-1 text-right">Coverage</th>
              </tr>
            </thead>
            <tbody>
              {data.fields.map((f) => (
                <tr key={f.name} data-testid={`fund-field-${f.name}`} title={f.note}>
                  <td className="py-0.5 font-mono text-slate-300">{f.name}</td>
                  <td className="py-0.5">
                    <span className={`border rounded px-1 ${CLASS_STYLE[f.class]}`}>{CLASS_LABEL[f.class]}</span>
                  </td>
                  <td className="py-0.5 text-right font-mono text-slate-400">{coverageText(f.coverage)}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {data.n_with_facts > 0 && (
            <div className="max-h-40 overflow-y-auto">
              <table className="w-full text-[10px]">
                <thead>
                  <tr className="text-slate-600 text-left">
                    <th className="font-medium pb-1">Ticker</th>
                    <th className="font-medium pb-1">Last filing (visible after)</th>
                    <th className="font-medium pb-1">Period end</th>
                    <th className="font-medium pb-1">Unavailable</th>
                  </tr>
                </thead>
                <tbody>
                  {data.tickers.map((t) => {
                    const missing = t.fields
                      ? Object.entries(t.fields).filter(([, ok]) => !ok).map(([name]) => name)
                      : null
                    return (
                      <tr key={t.ticker} data-testid={`fund-ticker-${t.ticker}`}>
                        <td className="py-0.5 font-mono text-slate-300">{t.ticker}</td>
                        <td className="py-0.5 font-mono text-slate-400">
                          {t.last_filed ?? '—'}
                          {t.stale && (
                            <span className="ml-1 text-amber-400" title="No new filing for too long — fields are blank">
                              <AlertTriangle size={9} className="inline" /> stale
                            </span>
                          )}
                        </td>
                        <td className="py-0.5 font-mono text-slate-500">{t.last_period_end ?? '—'}</td>
                        <td className="py-0.5 text-slate-600">
                          {!t.has_facts ? 'no filings' : missing === null ? '—' : missing.join(', ') || 'none'}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  )
}
