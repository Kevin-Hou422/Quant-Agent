import { useCallback, useEffect, useState } from 'react'
import {
  Radio, Wifi, WifiOff, Coins, Wallet, ShieldCheck, RefreshCw,
  AlertTriangle, Gauge,
} from 'lucide-react'
import { apiFetchTradingStatus, apiFetchPortfolioDiagnostics } from '../../api/client'
import type { TradingStatus, PortfolioDiagnostic } from '../../types'

/**
 * FE-TR (Phase TR): 交易现实面板。
 * 把决定"这笔交易在 $10k 散户下到底什么成本/能不能做空/买入力多少/证据够不够"的
 * 运行时事实呈现出来 —— 此前只在日志里，用户看不见。
 * 四块：数据源(TR.2) · 交易现实(TR.1) · T3账户(TR.3) · 门分级(TR.4)
 */

function num(v: unknown, d = 2, dash = '—'): string {
  const n = typeof v === 'number' ? v : Number(v)
  return Number.isFinite(n) ? n.toFixed(d) : dash
}

function Card({ title, icon: Icon, tone, children }: {
  title: string; icon: React.ElementType; tone?: string; children: React.ReactNode
}) {
  return (
    <section className="border border-slate-800 rounded-lg bg-slate-900/40">
      <div className="flex items-center gap-2 px-3 py-2 border-b border-slate-800">
        <Icon size={13} className={tone ?? 'text-sky-400'} />
        <span className="text-xs font-semibold text-slate-200">{title}</span>
      </div>
      <div className="px-3 py-2">{children}</div>
    </section>
  )
}

function Row({ k, v, tone, hint }: { k: string; v: string; tone?: string; hint?: string }) {
  return (
    <div className="flex items-baseline gap-2 py-0.5" title={hint}>
      <span className="text-[10px] text-slate-500 w-32 shrink-0">{k}</span>
      <span className={`text-[11px] font-mono ${tone ?? 'text-slate-200'}`}>{v}</span>
    </div>
  )
}

export default function TradingRealityPanel() {
  const [status, setStatus] = useState<TradingStatus | null>(null)
  const [diags, setDiags]   = useState<PortfolioDiagnostic[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError]   = useState<string | null>(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const [s, d] = await Promise.all([
        apiFetchTradingStatus(), apiFetchPortfolioDiagnostics(20),
      ])
      setStatus(s.data); setDiags(d.data); setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { refresh() }, [refresh])

  const latest = diags[0]
  const tc = latest?.trading_context ?? null
  const t3 = latest?.t3 ?? null
  const grade = (latest?.strategy_verdict as { paper_grade?: string } | null)?.paper_grade

  return (
    <div className="h-full overflow-y-auto p-4">
      <div className="flex items-center gap-3 mb-3">
        <Radio size={16} className="text-sky-400" />
        <h2 className="text-sm font-bold text-slate-100">交易现实</h2>
        <span className="text-[11px] text-slate-500">决定"$10k 散户下能否赚钱"的运行时事实</span>
        <button onClick={refresh} className="ml-auto flex items-center gap-1 text-[11px] text-slate-400 hover:text-slate-200">
          <RefreshCw size={12} /> 刷新
        </button>
      </div>

      {error && <div className="mb-3 px-3 py-2 rounded bg-rose-950/40 text-[11px] text-rose-400">{error}</div>}
      {loading && <div className="text-[11px] text-slate-500">加载中…</div>}

      {!loading && (
        <div className="grid grid-cols-2 gap-3">
          {/* ── 1. 数据源状态 (TR.2) ── */}
          <Card title="数据源 (TR.2)" icon={status?.moomoo.opend_reachable ? Wifi : WifiOff}
                tone={status?.moomoo.opend_reachable ? 'text-emerald-400' : 'text-rose-400'}>
            {status ? (
              <>
                <Row k="价格源" v={status.price_source}
                     tone={status.same_source ? 'text-emerald-300' : 'text-amber-300'}
                     hint="moomoo = 研究/执行同源，消除 train/serve skew" />
                {!status.same_source && (
                  <div className="mt-1 mb-1 text-[10px] text-amber-400 flex items-center gap-1">
                    <AlertTriangle size={11} /> 研究与执行**不同源**，存在 train/serve skew
                  </div>
                )}
                <Row k="OpenD 网关" v={`${status.moomoo.host}:${status.moomoo.port} · ${status.moomoo.opend_reachable ? '已连通' : '未连通'}`}
                     tone={status.moomoo.opend_reachable ? 'text-emerald-300' : 'text-rose-300'} />
                <Row k="券商档" v={status.broker} />
                <Row k="账户 / 做空" v={`${status.account_type} · ${status.allow_short ? '允许做空' : 'long-only'}`}
                     tone={status.allow_short ? 'text-slate-200' : 'text-sky-300'} />
                <Row k="AUM / 数据集" v={`$${status.paper_aum.toLocaleString()} · ${status.paper_dataset}`} />
              </>
            ) : <span className="text-[11px] text-slate-500">—</span>}
          </Card>

          {/* ── 2. 交易现实 (TR.1) ── */}
          <Card title="成本与可交易性 (TR.1)" icon={Coins} tone="text-amber-400">
            {tc ? (
              <>
                <Row k="中位估计价差" v={`${num(tc.median_spread_bps)} bps`}
                     hint="Corwin-Schultz 从免费 H/L 估计（估计值，非实时盘口）" />
                <Row k="中位单边成本" v={`${num(tc.median_cost_oneway_bps)} bps`}
                     hint="半价差 + 佣金 + 规费" />
                <Row k="可交易 / 可做空" v={`${tc.n_tradable ?? '—'} / ${tc.n_shortable ?? 0} 只`} />
                <Row k="无交易带" v={num(tc.rebalance_band, 4)}
                     hint="权重漂移小于此值不调仓 —— 减换手省成本" />
                {tc.notes?.slice(0, 2).map((n, i) => (
                  <div key={i} className="text-[10px] text-slate-500 mt-1">· {n}</div>
                ))}
              </>
            ) : <span className="text-[11px] text-slate-500">暂无（等一次组合运行）</span>}
          </Card>

          {/* ── 3. T3 账户 (TR.3) ── */}
          <Card title="账户 / T3 数据源 (TR.3)" icon={Wallet} tone="text-violet-400">
            {t3 ? (
              <>
                <div className="mb-1">
                  <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                    t3.mode === 'live' ? 'bg-emerald-900/60 text-emerald-300' : 'bg-amber-900/60 text-amber-300'}`}>
                    {t3.mode === 'live' ? 'LIVE 实时' : 'SIM 估计'}
                  </span>
                  <span className="ml-2 text-[10px] text-slate-500">
                    {t3.mode === 'live' ? '来自券商实时 API' : '盘口/借券为推导估计，非实时'}
                  </span>
                </div>
                <Row k="买入力" v={`$${num(t3.buying_power)}`} />
                <Row k="持仓数" v={`${t3.n_positions} 只`} />
                <Row k="末净值 / AUM" v={`${num(latest?.equity, 4)} · $${num(latest?.aum, 0)}`} />
              </>
            ) : <span className="text-[11px] text-slate-500">暂无（等一次组合运行）</span>}
          </Card>

          {/* ── 4. 门分级 (TR.4) ── */}
          <Card title="门分级与阈值 (TR.4)" icon={ShieldCheck} tone="text-emerald-400">
            {status ? (
              <>
                <div className="mb-1 flex items-center gap-2">
                  <span className="text-[10px] text-slate-500">当前策略等级</span>
                  {grade ? (
                    <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                      grade === 'A' ? 'bg-emerald-900/60 text-emerald-300'
                      : grade === 'B' ? 'bg-amber-900/60 text-amber-300'
                      : 'bg-rose-900/60 text-rose-300'}`}>{grade} 级</span>
                  ) : <span className="text-[11px] text-slate-500">—</span>}
                </div>
                <Row k="实验模式" v={status.gates.experiment_mode ? '开（B/C 也放行收证据）' : '关（仅 A 级）'}
                     tone={status.gates.experiment_mode ? 'text-amber-300' : 'text-emerald-300'} />
                <Row k="→ACTIVE 门" v={status.gates.enforce_active_gate ? '强制拦截' : '仅记录（不拦）'}
                     tone={status.gates.enforce_active_gate ? 'text-emerald-300' : 'text-amber-300'}
                     hint="ic_history 尚未分离回放/前向（Phase 11）前，默认不拦" />
                <Row k="→ACTIVE 阈值" v={`≥${status.gates.min_forward_days} 前向日 · IC t>${status.gates.min_ic_tstat}`} />
                <Row k="因子入池门" v={status.gates.factor_gate_mode === 'leak' ? 'leak（低门槛滤泄漏）' : 'strict（因子级严门）'} />
              </>
            ) : <span className="text-[11px] text-slate-500">—</span>}
          </Card>
        </div>
      )}

      {/* 最近运行 */}
      {!loading && diags.length > 0 && (
        <section className="mt-4">
          <div className="flex items-center gap-2 mb-2">
            <Gauge size={13} className="text-slate-400" />
            <span className="text-xs font-semibold text-slate-300">最近运行</span>
            <span className="text-[10px] text-slate-500">{diags.length} 条</span>
          </div>
          <div className="overflow-x-auto border border-slate-800 rounded-lg">
            <table className="w-full text-[11px]">
              <thead className="bg-slate-900/60 text-slate-500">
                <tr>
                  {['时间', '因子', '交易日', '末净值', '换手', '带', '等级', '衰减'].map(h => (
                    <th key={h} className="text-left font-medium px-2 py-1.5 whitespace-nowrap">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {diags.map(d => {
                  const g = (d.strategy_verdict as { paper_grade?: string } | null)?.paper_grade
                  return (
                    <tr key={d.id} className="text-slate-300">
                      <td className="px-2 py-1 whitespace-nowrap text-slate-500">{d.run_at?.slice(0, 19).replace('T', ' ') ?? '—'}</td>
                      <td className="px-2 py-1">{d.n_factors ?? '—'}{d.used_baseline ? ' (基准)' : ''}</td>
                      <td className="px-2 py-1">{d.days_processed ?? '—'}</td>
                      <td className="px-2 py-1 font-mono">{num(d.equity, 4)}</td>
                      <td className="px-2 py-1 font-mono">{num(d.turnover_ann)}</td>
                      <td className="px-2 py-1 font-mono">{num(d.no_trade_band, 4)}</td>
                      <td className="px-2 py-1">{g ?? '—'}</td>
                      <td className="px-2 py-1 text-[10px] text-amber-400">{d.strategy_decay ? '告警' : ''}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  )
}
