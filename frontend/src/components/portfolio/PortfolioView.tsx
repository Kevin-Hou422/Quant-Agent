import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Layers, Check, X, RefreshCw, PlusCircle, ShieldCheck, ShieldAlert,
  Gauge, Percent, ChevronDown, ChevronRight,
} from 'lucide-react'
import {
  apiFetchStrategiesPending, apiFetchStrategies, apiProposeStrategy,
  apiApproveStrategy, apiRejectStrategy,
} from '../../api/client'
import type { StrategyConfigItem } from '../../types'
import TradingRealityPanel from './TradingRealityPanel'

/**
 * FE-PM (Phase PM.7): 组合策略配置控制台。
 * 审批对象 = **一份组合策略配置**（成分 + 每因子配额 + 策略门 verdict + 风控 + 换手），
 * 不是单因子。对齐后端 /strategies/{propose,pending,list,{id},approve,reject}。
 */

const STATUS_STYLE: Record<string, string> = {
  proposed: 'bg-amber-900/50 text-amber-400',
  approved: 'bg-sky-900/50 text-sky-400',
  active:   'bg-emerald-900/60 text-emerald-400',
  retired:  'bg-slate-800 text-slate-400',
  rejected: 'bg-rose-900/50 text-rose-400',
}

function num(v: unknown, d = 3): string {
  const n = typeof v === 'number' ? v : Number(v)
  return Number.isFinite(n) ? n.toFixed(d) : '—'
}

function Stat({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <div className="flex flex-col">
      <span className="text-[9px] uppercase tracking-wider text-slate-500">{label}</span>
      <span className={`text-[12px] font-mono ${tone ?? 'text-slate-200'}`}>{value}</span>
    </div>
  )
}

function StrategyCard({ cfg, onDecision }: {
  cfg: StrategyConfigItem
  onDecision: (id: number, action: 'approve' | 'approve_activate' | 'reject') => void
}) {
  const [open, setOpen] = useState(false)
  const v = cfg.verdict || {}
  const r = cfg.risk_report || {}
  const gatePassed = Boolean((v as { passed?: boolean }).passed)
  const weights = cfg.combo_weights || {}
  const totalW = Object.values(weights).reduce((a, b) => a + Math.abs(b), 0) || 1
  const isProposed = cfg.status === 'proposed'

  return (
    <li className="border border-slate-800 rounded-lg bg-slate-900/40">
      {/* header */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-slate-800">
        <button onClick={() => setOpen(o => !o)} className="text-slate-500 hover:text-slate-300">
          {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </button>
        <Layers size={13} className="text-violet-400" />
        <span className="text-xs font-semibold text-slate-200">策略 #{cfg.id}</span>
        {cfg.name && <span className="text-[10px] text-slate-500">{cfg.name}</span>}
        <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded ${STATUS_STYLE[cfg.status] ?? 'bg-slate-800 text-slate-400'}`}>
          {cfg.status}
        </span>
        <span className="ml-2 flex items-center gap-1 text-[10px]">
          {gatePassed
            ? <><ShieldCheck size={12} className="text-emerald-400" /><span className="text-emerald-400">策略门通过</span></>
            : <><ShieldAlert size={12} className="text-amber-400" /><span className="text-amber-400">策略门未过</span></>}
        </span>
        {/* TR.4 进 PAPER 分级：A=严门全过 B=未过但Sharpe>0 C=不合格 */}
        {typeof v['paper_grade'] === 'string' && (
          <span
            title="TR.4 进 PAPER 分级：A=严门全过 · B=未过但 Sharpe>0 · C=不合格（实验模式下仍放行以收前向证据）"
            className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
              v['paper_grade'] === 'A' ? 'bg-emerald-900/60 text-emerald-300'
              : v['paper_grade'] === 'B' ? 'bg-amber-900/60 text-amber-300'
              : 'bg-rose-900/60 text-rose-300'}`}
          >
            {String(v['paper_grade'])} 级
          </span>
        )}
        <span className="ml-auto text-[10px] text-slate-500">{cfg.factors.length} 因子 · ${cfg.aum.toLocaleString()}</span>
      </div>

      {/* key stats row */}
      <div className="grid grid-cols-4 gap-3 px-3 py-2 border-b border-slate-800/60">
        <Stat label="Sharpe" value={num(v['sharpe'])} />
        <Stat label="DSR" value={num(v['deflated_sharpe'])} tone={Number(v['deflated_sharpe']) > 0.9 ? 'text-emerald-300' : 'text-slate-200'} />
        <Stat label="夏普 t" value={num(v['t_stat'], 2)} tone={Number(v['t_stat']) >= 3 ? 'text-emerald-300' : 'text-slate-200'} />
        <Stat label="PBO" value={v['pbo'] == null ? 'NA' : num(v['pbo'], 2)} tone={Number(v['pbo']) > 0.5 ? 'text-rose-300' : 'text-slate-200'} />
      </div>

      {open && (
        <div className="px-3 py-2 space-y-3">
          {/* composition */}
          <div>
            <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">成分 + 配额（combo 权重）</div>
            <div className="space-y-1">
              {Object.entries(weights).map(([f, w]) => (
                <div key={f} className="flex items-center gap-2">
                  <span className="text-[11px] font-mono text-slate-400 w-24 truncate">因子 {f}</span>
                  <div className="flex-1 h-2 bg-slate-800 rounded overflow-hidden">
                    <div className="h-full bg-violet-600" style={{ width: `${(Math.abs(w) / totalW) * 100}%` }} />
                  </div>
                  <span className="text-[10px] font-mono text-slate-400 w-14 text-right">{num(w, 3)}</span>
                </div>
              ))}
              {Object.keys(weights).length === 0 && <span className="text-[11px] text-slate-500">（无配额数据）</span>}
            </div>
          </div>

          {/* risk + horizon */}
          <div className="grid grid-cols-3 gap-3">
            <Stat label="单票削(次)" value={String(r['n_name_clipped'] ?? '—')} />
            <Stat label="行业缩(次)" value={String(r['n_sector_scaled'] ?? '—')} />
            <Stat label="gross缩(日)" value={String(r['n_gross_scaled'] ?? '—')} />
            <Stat label="vol 缩放" value={num(r['vol_scalar'], 3)} />
            <Stat label="年化换手" value={num(cfg.turnover_ann, 2)} />
            <Stat label="无交易带" value={num(cfg.no_trade_band, 4)} />
          </div>

          {/* gate reasons */}
          {Array.isArray((v as { reasons?: string[] }).reasons) && (v as { reasons?: string[] }).reasons!.length > 0 && (
            <div className="text-[10px] text-amber-400/90">
              未过原因：{(v as { reasons?: string[] }).reasons!.join('；')}
            </div>
          )}

          {/* lineage */}
          {cfg.decisions && cfg.decisions.length > 0 && (
            <div className="text-[10px] text-slate-500">
              谱系：{cfg.decisions.map((d, i) => (
                <span key={i}>{d.decision}（{d.from_status}→{d.to_status}）{i < cfg.decisions!.length - 1 ? ' · ' : ''}</span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* actions for proposed */}
      {isProposed && (
        <div className="flex items-center gap-2 px-3 py-2 border-t border-slate-800">
          <button onClick={() => onDecision(cfg.id, 'approve')}
            className="flex items-center gap-1 text-[11px] px-2 py-1 rounded bg-sky-900/50 text-sky-400 hover:bg-sky-900/80 transition-colors">
            <Check size={12} /> 批准
          </button>
          <button onClick={() => onDecision(cfg.id, 'approve_activate')}
            className="flex items-center gap-1 text-[11px] px-2 py-1 rounded bg-emerald-800/60 text-emerald-300 hover:bg-emerald-700 transition-colors">
            <Gauge size={12} /> 批准并启用（active）
          </button>
          <button onClick={() => onDecision(cfg.id, 'reject')}
            className="flex items-center gap-1 text-[11px] px-2 py-1 rounded bg-rose-900/50 text-rose-400 hover:bg-rose-900/80 transition-colors">
            <X size={12} /> 拒绝
          </button>
        </div>
      )}
    </li>
  )
}

type PMTab = 'configs' | 'reality'

function TabBar({ tab, setTab }: { tab: PMTab; setTab: (t: PMTab) => void }) {
  const item = (t: PMTab, label: string) => (
    <button
      key={t}
      onClick={() => setTab(t)}
      className={`px-3 py-2 text-xs font-medium border-b-2 transition-colors ${
        tab === t
          ? 'text-violet-300 border-violet-500'
          : 'text-slate-400 border-transparent hover:text-slate-200'}`}
    >
      {label}
    </button>
  )
  return (
    <div className="flex items-center gap-1 px-4 border-b border-slate-800 bg-slate-950 shrink-0">
      {item('configs', '策略配置')}
      {item('reality', '交易现实')}
    </div>
  )
}

export default function PortfolioView() {
  const [tab, setTab]         = useState<PMTab>('configs')
  const [pending, setPending] = useState<StrategyConfigItem[]>([])
  const [others, setOthers]   = useState<StrategyConfigItem[]>([])
  const [loading, setLoading] = useState(true)
  const [busy, setBusy]       = useState(false)
  const [error, setError]     = useState<string | null>(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const [p, all] = await Promise.all([apiFetchStrategiesPending(), apiFetchStrategies()])
      setPending(p.data)
      setOthers(all.data.filter(c => c.status !== 'proposed'))
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { refresh() }, [refresh])

  const propose = useCallback(async () => {
    setBusy(true); setError(null)
    try {
      await apiProposeStrategy()
      await refresh()
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      setError(msg.includes('400') ? '无 PAPER/ACTIVE 因子 —— 先让 agent 挖出并批准因子进 PAPER' : msg)
    } finally {
      setBusy(false)
    }
  }, [refresh])

  const decide = useCallback(async (id: number, action: 'approve' | 'approve_activate' | 'reject') => {
    setBusy(true); setError(null)
    try {
      if (action === 'reject') {
        const reason = window.prompt('拒绝原因（可选）：') ?? ''
        await apiRejectStrategy(id, reason)
      } else {
        await apiApproveStrategy(id, action === 'approve_activate', '')
      }
      await refresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }, [refresh])

  const active = useMemo(() => others.filter(c => c.status === 'active'), [others])

  // FE-TR：交易现实作为并列标签页（不再加顶层导航）
  if (tab === 'reality') {
    return (
      <div className="h-full flex flex-col bg-slate-950">
        <TabBar tab={tab} setTab={setTab} />
        <div className="flex-1 min-h-0"><TradingRealityPanel /></div>
      </div>
    )
  }

  return (
    <div className="h-full overflow-y-auto bg-slate-950">
      <TabBar tab={tab} setTab={setTab} />
      <div className="p-4">
      {/* header */}
      <div className="flex items-center gap-3 mb-4">
        <Layers size={18} className="text-violet-400" />
        <h1 className="text-sm font-bold text-slate-100">组合策略配置</h1>
        <span className="text-[11px] text-slate-500">审批对象 = 一份组合策略（成分+配额+门+风控），非单因子</span>
        <button onClick={propose} disabled={busy}
          className="ml-auto flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg bg-violet-700 hover:bg-violet-600 disabled:opacity-40 text-white transition-colors">
          <PlusCircle size={13} /> {busy ? '构建中…' : '提出新策略配置'}
        </button>
        <button onClick={refresh} className="flex items-center gap-1 text-[11px] text-slate-400 hover:text-slate-200 transition-colors">
          <RefreshCw size={12} /> 刷新
        </button>
      </div>

      {error && <div className="mb-3 px-3 py-2 rounded bg-rose-950/40 text-[11px] text-rose-400">{error}</div>}

      {/* active banner */}
      {active.length > 0 && (
        <div className="mb-4 px-3 py-2 rounded-lg bg-emerald-950/30 border border-emerald-900/50 text-[11px] text-emerald-300 flex items-center gap-2">
          <Percent size={13} /> 当前**在交易**的配置：#{active.map(c => c.id).join(', ')} —— run_portfolio 只交易其成分。
        </div>
      )}

      {loading ? (
        <div className="text-[11px] text-slate-500 px-1">加载中…</div>
      ) : (
        <>
          {/* pending */}
          <section className="mb-6">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xs font-semibold text-slate-300">待审批</span>
              <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded bg-amber-900/50 text-amber-400">{pending.length}</span>
            </div>
            {pending.length === 0 ? (
              <div className="text-[11px] text-slate-500 px-1 py-3 border border-dashed border-slate-800 rounded-lg">
                暂无待审批策略。点「提出新策略配置」，系统会从当前 PAPER 因子构建一份（边际准入→合成→策略门→风控→换手）。
              </div>
            ) : (
              <ul className="space-y-2">
                {pending.map(c => <StrategyCard key={c.id} cfg={c} onDecision={decide} />)}
              </ul>
            )}
          </section>

          {/* history / active */}
          {others.length > 0 && (
            <section>
              <div className="text-xs font-semibold text-slate-300 mb-2">已批准 / 在交易 / 历史</div>
              <ul className="space-y-2">
                {others.map(c => <StrategyCard key={c.id} cfg={c} onDecision={decide} />)}
              </ul>
            </section>
          )}
        </>
      )}
      </div>
    </div>
  )
}
