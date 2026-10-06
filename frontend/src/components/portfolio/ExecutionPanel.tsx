import { useCallback, useEffect, useState } from 'react'
import {
  Power, RefreshCw, ShieldAlert, ShieldCheck, ListOrdered, Scale, Activity, AlertTriangle,
} from 'lucide-react'
import {
  apiFetchExecutionStatus, apiReconcileExecution, apiAcceptReconcile,
  apiArmKillSwitch, apiConfirmKillSwitch, apiFetchFidelity,
} from '../../api/client'
import type { ExecutionStatus, ExecFillRow, ExecOrderRow, FidelityReport, FidelityStats } from '../../types'

/**
 * FE-12 (Phase 12): 执行监控 —— moomoo **纸交易**账户的持仓 / 订单 / 成交、
 * 本地账本 vs 券商的对账差、下单前风控拦单、一键全平（两步确认）、
 * 以及"内部模拟（收盘成交）vs 纸交易（次日开盘成交）"保真度对比。
 *
 * 前端只读 + 人工动作；所有数字直接来自 /api/execution/*，不做任何推断或装饰。
 */

function num(v: unknown, d = 2, dash = '—'): string {
  const n = typeof v === 'number' ? v : Number(v)
  return Number.isFinite(n) ? n.toFixed(d) : dash
}

/** 成交相对决策日收盘的滑点（bps，正 = 比模拟账本更差）。 */
export function slippageBps(f: Pick<ExecFillRow, 'qty' | 'price' | 'ref_price'>): number {
  if (!(f.ref_price > 0) || f.qty === 0) return NaN
  const s = f.qty > 0 ? 1 : -1
  return s * (f.price - f.ref_price) / f.ref_price * 1e4
}

const SNAP_TONE: Record<string, string> = {
  baseline:    'bg-sky-900/50 text-sky-300',
  clean:       'bg-emerald-900/50 text-emerald-300',
  accepted:    'bg-emerald-900/50 text-emerald-300',
  discrepancy: 'bg-rose-900/60 text-rose-300',
  unresolved:  'bg-rose-900/60 text-rose-300',
  unstable:    'bg-amber-900/50 text-amber-300',
}

const ORDER_TONE: Record<string, string> = {
  FILLED: 'text-emerald-300', PARTIAL: 'text-amber-300', SUBMITTED: 'text-sky-300',
  PENDING_SUBMIT: 'text-amber-300', REJECTED_BY_GATE: 'text-rose-300', FAILED: 'text-rose-300',
  CANCELLED: 'text-slate-400', NOT_SUBMITTED: 'text-slate-400',
}

function Card({ title, icon: Icon, tone, right, children }: {
  title: string; icon: React.ElementType; tone?: string; right?: React.ReactNode; children: React.ReactNode
}) {
  return (
    <section className="border border-slate-800 rounded-lg bg-slate-900/40">
      <div className="flex items-center gap-2 px-3 py-2 border-b border-slate-800">
        <Icon size={13} className={tone ?? 'text-sky-400'} />
        <span className="text-xs font-semibold text-slate-200">{title}</span>
        <div className="ml-auto">{right}</div>
      </div>
      <div className="px-3 py-2">{children}</div>
    </section>
  )
}

function Th({ children }: { children: React.ReactNode }) {
  return <th className="text-left font-medium text-slate-500 px-1.5 py-1">{children}</th>
}
function Td({ children, tone }: { children: React.ReactNode; tone?: string }) {
  return <td className={`px-1.5 py-0.5 font-mono ${tone ?? 'text-slate-300'}`}>{children}</td>
}

// ---------------------------------------------------------------------------
// 两步确认：全平 / 解除熔断
// ---------------------------------------------------------------------------

const PHRASE = { engage: '全平', reset: '解除' } as const

function KillSwitchDialog({ action, onClose, onDone }: {
  action: 'engage' | 'reset'
  onClose: () => void
  onDone: (msg: string) => void
}) {
  const [token, setToken]   = useState<string | null>(null)
  const [left, setLeft]     = useState(0)
  const [actor, setActor]   = useState('')
  const [reason, setReason] = useState('')
  const [phrase, setPhrase] = useState('')
  const [busy, setBusy]     = useState(false)
  const [err, setErr]       = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    apiArmKillSwitch(action)
      .then(r => { if (alive) { setToken(r.data.token); setLeft(r.data.expires_in_s) } })
      .catch(e => { if (alive) setErr(e instanceof Error ? e.message : String(e)) })
    return () => { alive = false }
  }, [action])

  useEffect(() => {
    if (left <= 0) return
    const t = setTimeout(() => setLeft(s => s - 1), 1000)
    return () => clearTimeout(t)
  }, [left])

  const ready = Boolean(token) && left > 0 && actor.trim() !== '' && reason.trim() !== ''
    && phrase.trim() === PHRASE[action] && !busy

  const confirm = async () => {
    if (!ready || !token) return
    setBusy(true); setErr(null)
    try {
      const r = await apiConfirmKillSwitch(token, actor.trim(), reason.trim())
      const e = (r.data as { error?: string }).error
      onDone(e ? `熔断已生效，但：${e}` : (action === 'engage' ? '全平熔断已开启' : '全平熔断已解除'))
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50" role="dialog">
      <div className="w-[440px] rounded-lg border border-rose-900 bg-slate-950 p-4 space-y-3">
        <div className="flex items-center gap-2">
          <AlertTriangle size={15} className="text-rose-400" />
          <span className="text-sm font-bold text-rose-300">
            {action === 'engage' ? '一键全平（纸交易账户）' : '解除全平熔断'}
          </span>
        </div>
        <p className="text-[11px] text-slate-400 leading-relaxed">
          {action === 'engage'
            ? '将撤销账户内全部在途单，按券商现价 ×(1−全平带) 挂限价卖出全部持仓；之后每个交易周期只继续全平、不再调仓，直到人工解除。'
            : '解除后，下一个交易周期恢复按目标权重调仓。'}
        </p>
        <div className="text-[10px] text-slate-500" data-testid="token-state">
          {err ? <span className="text-rose-400">{err}</span>
            : token ? (left > 0 ? `确认令牌剩余 ${left}s` : '令牌已过期，请关闭后重试')
            : '正在申请确认令牌…'}
        </div>
        <input aria-label="操作人" value={actor} onChange={e => setActor(e.target.value)}
          placeholder="操作人" className="w-full text-xs px-2 py-1 rounded bg-slate-900 border border-slate-700 text-slate-200" />
        <input aria-label="原因" value={reason} onChange={e => setReason(e.target.value)}
          placeholder="原因（写进审计日志）" className="w-full text-xs px-2 py-1 rounded bg-slate-900 border border-slate-700 text-slate-200" />
        <input aria-label="确认短语" value={phrase} onChange={e => setPhrase(e.target.value)}
          placeholder={`输入「${PHRASE[action]}」确认`} className="w-full text-xs px-2 py-1 rounded bg-slate-900 border border-slate-700 text-slate-200" />
        <div className="flex justify-end gap-2">
          <button onClick={onClose} className="text-[11px] px-3 py-1 rounded text-slate-400 hover:text-slate-200">取消</button>
          <button onClick={confirm} disabled={!ready}
            className="text-[11px] px-3 py-1 rounded bg-rose-800 text-white disabled:opacity-30 hover:bg-rose-700">
            {busy ? '执行中…' : '确认执行'}
          </button>
        </div>
      </div>
    </div>
  )
}

function AcceptDialog({ onClose, onDone }: { onClose: () => void; onDone: (msg: string) => void }) {
  const [actor, setActor]   = useState('')
  const [reason, setReason] = useState('')
  const [busy, setBusy]     = useState(false)
  const [err, setErr]       = useState<string | null>(null)
  const ok = actor.trim() !== '' && reason.trim() !== '' && !busy
  const submit = async () => {
    if (!ok) return
    setBusy(true); setErr(null)
    try {
      await apiAcceptReconcile(actor.trim(), reason.trim())
      onDone('已按券商当前状态重建可信基准')
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }
  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50" role="dialog">
      <div className="w-[420px] rounded-lg border border-amber-900 bg-slate-950 p-4 space-y-3">
        <span className="text-sm font-bold text-amber-300">接受对账差异</span>
        <p className="text-[11px] text-slate-400">以券商当前持仓为准重建基准；找不到的在途单按"未到达券商/已撤"结案。请先在 moomoo App 核对。</p>
        {err && <div className="text-[11px] text-rose-400">{err}</div>}
        <input aria-label="操作人" value={actor} onChange={e => setActor(e.target.value)} placeholder="操作人"
          className="w-full text-xs px-2 py-1 rounded bg-slate-900 border border-slate-700 text-slate-200" />
        <input aria-label="原因" value={reason} onChange={e => setReason(e.target.value)} placeholder="原因（如：AAA 2:1 拆股）"
          className="w-full text-xs px-2 py-1 rounded bg-slate-900 border border-slate-700 text-slate-200" />
        <div className="flex justify-end gap-2">
          <button onClick={onClose} className="text-[11px] px-3 py-1 rounded text-slate-400">取消</button>
          <button onClick={submit} disabled={!ok}
            className="text-[11px] px-3 py-1 rounded bg-amber-800 text-white disabled:opacity-30">确认接受</button>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// 面板
// ---------------------------------------------------------------------------

function statLine(label: string, s: FidelityStats | undefined) {
  return (
    <div className="flex items-baseline gap-2 py-0.5">
      <span className="text-[10px] text-slate-500 w-40 shrink-0">{label}</span>
      <span className="text-[11px] font-mono text-slate-200">
        {s && s.n > 0 ? `中位 ${num(s.median)} · 均值 ${num(s.mean)} · P90 ${num(s.p90)} bps（n=${s.n}）` : '无样本'}
      </span>
    </div>
  )
}

function OrdersTable({ rows, testid }: { rows: ExecOrderRow[]; testid: string }) {
  if (rows.length === 0) return <div className="text-[11px] text-slate-500">（无）</div>
  return (
    <table className="w-full text-[10px]" data-testid={testid}>
      <thead><tr><Th>决策日</Th><Th>标的</Th><Th>方向</Th><Th>数量</Th><Th>限价</Th><Th>成交</Th><Th>状态</Th><Th>原因</Th></tr></thead>
      <tbody>
        {rows.map(o => (
          <tr key={o.client_id} title={o.client_id}>
            <Td>{o.decision_date}</Td>
            <Td>{o.ticker}{o.purpose === 'flatten' ? ' ⚑' : ''}</Td>
            <Td tone={o.side === 'BUY' ? 'text-emerald-300' : 'text-rose-300'}>{o.side}</Td>
            <Td>{o.qty}</Td>
            <Td>{num(o.limit_price)}</Td>
            <Td>{o.dealt_qty > 0 ? `${o.dealt_qty} @ ${num(o.dealt_avg_price)}` : '—'}</Td>
            <Td tone={ORDER_TONE[o.status]}>{o.status}</Td>
            <Td tone="text-rose-300">{o.reject_reason || o.last_err_msg || ''}</Td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function isoDaysAgo(n: number): string {
  const d = new Date(Date.now() - n * 86400_000)
  return d.toISOString().slice(0, 10)
}

export default function ExecutionPanel() {
  const [st, setSt]           = useState<ExecutionStatus | null>(null)
  const [error, setError]     = useState<string | null>(null)
  const [notice, setNotice]   = useState<string | null>(null)
  const [dialog, setDialog]   = useState<null | 'engage' | 'reset' | 'accept'>(null)
  const [busy, setBusy]       = useState(false)
  const [fid, setFid]         = useState<FidelityReport | null>(null)
  const [fidRange, setFidRange] = useState({ start: isoDaysAgo(30), end: isoDaysAgo(0) })

  const refresh = useCallback(async () => {
    try {
      const r = await apiFetchExecutionStatus()
      setSt(r.data); setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  useEffect(() => { refresh() }, [refresh])

  const reconcile = async () => {
    setBusy(true); setNotice(null)
    try {
      const r = await apiReconcileExecution()
      setNotice(`对账完成：${String((r.data as { status?: string }).status ?? '')}`)
      await refresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const loadFidelity = async () => {
    try {
      const r = await apiFetchFidelity(fidRange.start, fidRange.end)
      setFid(r.data); setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const done = async (msg: string) => { setDialog(null); setNotice(msg); await refresh() }

  const snap = st?.last_snapshot ?? null
  const ks = st?.kill_switch
  const total = snap?.total_assets ?? 0
  const positions = Object.entries(snap?.positions ?? {})
  const blocking = snap && ['discrepancy', 'unresolved', 'unstable'].includes(snap.status)

  return (
    <div className="h-full overflow-y-auto p-4 space-y-3">
      <div className="flex items-center gap-3">
        <Activity size={16} className="text-sky-400" />
        <h2 className="text-sm font-bold text-slate-100">执行监控</h2>
        <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded bg-sky-900/50 text-sky-300">SIMULATE</span>
        <span data-testid="exec-mode" className={`text-[10px] font-semibold px-1.5 py-0.5 rounded ${
          st?.mode === 'moomoo_paper' ? 'bg-emerald-900/60 text-emerald-300' : 'bg-slate-800 text-slate-400'}`}>
          {st?.mode === 'moomoo_paper' ? '自动下单：开' : `自动下单：关（${st?.mode ?? '…'}）`}
        </span>
        <button onClick={reconcile} disabled={busy}
          className="ml-auto flex items-center gap-1 text-[11px] px-2 py-1 rounded bg-slate-800 text-slate-200 hover:bg-slate-700 disabled:opacity-40">
          <Scale size={12} /> {busy ? '对账中…' : '立即对账'}
        </button>
        <button onClick={refresh} className="flex items-center gap-1 text-[11px] text-slate-400 hover:text-slate-200">
          <RefreshCw size={12} /> 刷新
        </button>
      </div>

      {error && <div className="px-3 py-2 rounded bg-rose-950/40 text-[11px] text-rose-400">{error}</div>}
      {notice && <div className="px-3 py-2 rounded bg-slate-900 text-[11px] text-slate-300">{notice}</div>}

      {/* 全平熔断 */}
      <Card title="一键全平（kill switch）" icon={Power} tone={ks?.engaged ? 'text-rose-400' : 'text-slate-400'}
        right={ks?.engaged
          ? <button onClick={() => setDialog('reset')} className="text-[11px] px-2 py-1 rounded bg-slate-800 text-slate-200">解除熔断…</button>
          : <button onClick={() => setDialog('engage')} className="text-[11px] px-2 py-1 rounded bg-rose-900/70 text-rose-200 hover:bg-rose-800">一键全平…</button>}>
        {ks?.engaged
          ? <div data-testid="ks-engaged" className="text-[11px] text-rose-300">
              熔断已开启 · {ks.actor} · {ks.at} · {ks.reason} —— 每个周期只继续全平，不调仓
            </div>
          : <div className="text-[11px] text-slate-500">未开启。全平需两步确认：申请一次性令牌 → 填写操作人与原因并输入确认短语。</div>}
      </Card>

      {/* 对账 */}
      <Card title="对账（本地账本 vs 券商）" icon={blocking ? ShieldAlert : ShieldCheck}
        tone={blocking ? 'text-rose-400' : 'text-emerald-400'}
        right={blocking ? <button onClick={() => setDialog('accept')} className="text-[11px] px-2 py-1 rounded bg-amber-900/70 text-amber-200">接受差异…</button> : null}>
        {!snap ? <div className="text-[11px] text-slate-500">尚无对账快照（执行层未运行过或券商未连通）。</div> : (
          <div className="space-y-2">
            <div className="flex items-center gap-3 text-[11px]">
              <span data-testid="snap-status" className={`px-1.5 py-0.5 rounded font-semibold ${SNAP_TONE[snap.status] ?? 'bg-slate-800 text-slate-300'}`}>{snap.status}</span>
              <span className="text-slate-400">市场日 {snap.market_date}</span>
              <span className="text-slate-400">总资产 ${num(snap.total_assets)}</span>
              <span className="text-slate-400">现金 ${num(snap.cash)}</span>
              <span className="text-slate-400">买入力 ${num(snap.power)}</span>
            </div>
            {blocking && <div className="text-[11px] text-rose-300">对账不干净 —— 交易周期<b>不会下单</b>，直到人工核对并接受。</div>}
            {snap.discrepancies.length > 0 && (
              <table className="w-full text-[10px]" data-testid="discrepancies">
                <thead><tr><Th>标的 / 订单</Th><Th>账本推算</Th><Th>券商</Th><Th>说明</Th></tr></thead>
                <tbody>{snap.discrepancies.map((d, i) => (
                  <tr key={i}>
                    <Td>{String(d.ticker ?? d.client_id ?? '')}</Td>
                    <Td>{d.expected !== undefined ? num(d.expected, 0) : '—'}</Td>
                    <Td>{d.broker !== undefined ? num(d.broker, 0) : '—'}</Td>
                    <Td tone="text-amber-300">{d.status ? `在途单找不到（${String(d.status)}）` : ''}</Td>
                  </tr>))}
                </tbody>
              </table>
            )}
            {snap.note && <div className="text-[10px] text-amber-400">{snap.note}</div>}
          </div>
        )}
      </Card>

      {/* 持仓 */}
      <Card title="券商持仓" icon={ListOrdered}>
        {positions.length === 0 ? <div className="text-[11px] text-slate-500">（空仓）</div> : (
          <table className="w-full text-[10px]" data-testid="positions">
            <thead><tr><Th>标的</Th><Th>股数</Th><Th>现价</Th><Th>市值</Th><Th>权重</Th></tr></thead>
            <tbody>{positions.map(([tk, p]) => (
              <tr key={tk}>
                <Td>{tk}</Td><Td>{p.qty}</Td><Td>{num(p.price)}</Td><Td>{num(p.market_val)}</Td>
                <Td>{total > 0 ? `${num(p.market_val / total * 100, 1)}%` : '—'}</Td>
              </tr>))}
            </tbody>
          </table>
        )}
      </Card>

      {/* 订单 */}
      <Card title="在途订单" icon={ListOrdered}><OrdersTable rows={st?.open_orders ?? []} testid="open-orders" /></Card>
      <Card title="近期订单（含下单前风控拦单）" icon={ListOrdered}><OrdersTable rows={st?.recent_orders ?? []} testid="recent-orders" /></Card>

      {/* 成交 */}
      <Card title="近期成交（vs 决策日收盘 = 模拟账本成交价）" icon={Activity}>
        {(st?.recent_fills ?? []).length === 0 ? <div className="text-[11px] text-slate-500">（无）</div> : (
          <table className="w-full text-[10px]" data-testid="fills">
            <thead><tr><Th>决策日</Th><Th>成交日</Th><Th>标的</Th><Th>数量</Th><Th>成交价</Th><Th>决策日收盘</Th><Th>滑点 bps</Th></tr></thead>
            <tbody>{(st?.recent_fills ?? []).map((f, i) => {
              const s = slippageBps(f)
              return (
                <tr key={`${f.client_id}-${i}`}>
                  <Td>{f.decision_date}</Td><Td>{f.fill_date}</Td><Td>{f.ticker}</Td>
                  <Td tone={f.qty > 0 ? 'text-emerald-300' : 'text-rose-300'}>{f.qty}</Td>
                  <Td>{num(f.price)}</Td><Td>{num(f.ref_price)}</Td>
                  <Td tone={s > 0 ? 'text-rose-300' : 'text-emerald-300'}>{num(s, 1)}</Td>
                </tr>)
            })}</tbody>
          </table>
        )}
      </Card>

      {/* 保真度 */}
      <Card title="保真度：内部模拟（收盘成交）vs moomoo 纸交易（次日开盘成交）" icon={Scale}
        right={
          <div className="flex items-center gap-1">
            <input aria-label="开始日期" type="date" value={fidRange.start}
              onChange={e => setFidRange(r => ({ ...r, start: e.target.value }))}
              className="text-[10px] px-1 rounded bg-slate-900 border border-slate-700 text-slate-300" />
            <input aria-label="结束日期" type="date" value={fidRange.end}
              onChange={e => setFidRange(r => ({ ...r, end: e.target.value }))}
              className="text-[10px] px-1 rounded bg-slate-900 border border-slate-700 text-slate-300" />
            <button onClick={loadFidelity} className="text-[11px] px-2 py-0.5 rounded bg-slate-800 text-slate-200">生成</button>
          </div>}>
        {!fid ? <div className="text-[11px] text-slate-500">选择区间后生成。</div> : (
          <div data-testid="fidelity">
            <div className="text-[11px] text-slate-400 mb-1">
              纸交易调仓单 {fid.n_live_orders} · 成交 {fid.n_live_fills} · 与模拟账本匹配 {fid.n_matched_sim}
              · 成交率中位 纸交易 {num(fid.live_fill_ratio_median)} / 模拟 {num(fid.sim_fill_ratio_median)}
            </div>
            {statLine('总滑点（vs 决策日收盘）', fid.total_slippage_bps)}
            {statLine('隔夜缺口（次日开盘 vs 收盘）', fid.overnight_gap_bps)}
            {statLine('开盘执行（成交价 vs 开盘）', fid.at_open_exec_bps)}
            <div className="text-[11px] text-slate-300 mt-1">
              impact_coef 建议：{num(fid.current_impact_coef, 4)} → {num(fid.recommended_impact_coef, 4)}
              （×{num(fid.recommended_scale, 3)}，仅建议，需人工确认）
            </div>
            <div className="text-[11px] text-amber-300">永久冲击：不可识别 —— {fid.permanent_impact_note}</div>
            {fid.warnings.map((w, i) => <div key={i} className="text-[10px] text-amber-400">⚠ {w}</div>)}
          </div>
        )}
      </Card>

      {/* 审计 */}
      <Card title="审计日志" icon={ShieldCheck}>
        {(st?.events ?? []).length === 0 ? <div className="text-[11px] text-slate-500">（无）</div> : (
          <ul className="text-[10px] space-y-0.5">
            {(st?.events ?? []).map((e, i) => (
              <li key={i} className="font-mono text-slate-400">{e.at} · {e.kind} · {e.actor} · {e.reason}</li>
            ))}
          </ul>
        )}
      </Card>

      {(dialog === 'engage' || dialog === 'reset') &&
        <KillSwitchDialog action={dialog} onClose={() => setDialog(null)} onDone={done} />}
      {dialog === 'accept' && <AcceptDialog onClose={() => setDialog(null)} onDone={done} />}
    </div>
  )
}
