/**
 * ExecutionPanel.test.tsx — FE-12 执行监控
 *
 *  · 面板上的数字逐个绑回 /api/execution/status 的真实字段（§N：防装饰）
 *  · 全平必须两步：点按钮只申请令牌；令牌 + 操作人 + 原因 + 确认短语齐了才执行
 *  · 对账不干净时显示阻断提示与"接受差异"入口
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'

const api = vi.hoisted(() => ({
  apiFetchExecutionStatus: vi.fn(),
  apiReconcileExecution:   vi.fn(),
  apiAcceptReconcile:      vi.fn(),
  apiArmKillSwitch:        vi.fn(),
  apiConfirmKillSwitch:    vi.fn(),
  apiFetchFidelity:        vi.fn(),
}))
vi.mock('../../../api/client', () => api)

import ExecutionPanel, { slippageBps } from '../../../components/portfolio/ExecutionPanel'

const STATUS = {
  mode: 'moomoo_paper',
  kill_switch: { engaged: false },
  last_snapshot: {
    taken_at: '2025-03-04T21:00:00', market_date: '2025-03-04', status: 'clean',
    total_assets: 100130, cash: 76930, power: 76930, note: '',
    positions: { AAA: { qty: 200, price: 51, market_val: 10200 } },
    discrepancies: [],
  },
  open_orders: [{
    client_id: 'qaR-1-20250304-AAA-S', decision_date: '2025-03-04', purpose: 'rebalance',
    ticker: 'AAA', side: 'SELL', qty: 4, limit_price: 50.75, ref_price: 51, status: 'SUBMITTED',
    broker_status: 'SUBMITTED', broker_order_id: '90004', dealt_qty: 0, dealt_avg_price: 0,
    reject_reason: '', last_err_msg: '',
  }],
  recent_orders: [{
    client_id: 'qaR-1-20250304-CCC-B', decision_date: '2025-03-04', purpose: 'rebalance',
    ticker: 'CCC', side: 'BUY', qty: 500, limit_price: 10.05, ref_price: 10,
    status: 'REJECTED_BY_GATE', broker_status: '', broker_order_id: null, dealt_qty: 0,
    dealt_avg_price: 0, reject_reason: 'no_adv', last_err_msg: '',
  }],
  recent_fills: [{
    client_id: 'qaR-1-20250303-AAA-B', ticker: 'AAA', side: 'BUY', qty: 200, price: 50.2,
    ref_price: 50, decision_date: '2025-03-03', fill_date: '2025-03-04',
  }],
  events: [{ at: '2025-03-04T21:00:00', kind: 'startup_recovery', actor: 'system', reason: 'clean' }],
}

beforeEach(() => {
  Object.values(api).forEach(f => f.mockReset())
  api.apiFetchExecutionStatus.mockResolvedValue({ data: STATUS })
  api.apiArmKillSwitch.mockResolvedValue({ data: { token: 'tok-123', action: 'engage', expires_in_s: 120 } })
  api.apiConfirmKillSwitch.mockResolvedValue({ data: { action: 'engage', state: { engaged: true } } })
})

describe('slippageBps', () => {
  it('positive means worse than the simulated close fill, for both sides', () => {
    expect(slippageBps({ qty: 200, price: 50.2, ref_price: 50 })).toBeCloseTo(40, 6)
    expect(slippageBps({ qty: -50, price: 19.7, ref_price: 20 })).toBeCloseTo(150, 6)
    expect(Number.isNaN(slippageBps({ qty: 10, price: 1, ref_price: 0 }))).toBe(true)
  })
})

describe('ExecutionPanel — data binding', () => {
  it('renders broker state from the status endpoint', async () => {
    render(<ExecutionPanel />)
    await waitFor(() => expect(screen.getByTestId('snap-status')).toHaveTextContent('clean'))
    expect(screen.getByTestId('exec-mode')).toHaveTextContent('自动下单：开')
    const pos = screen.getByTestId('positions')
    expect(pos).toHaveTextContent('AAA')
    expect(pos).toHaveTextContent('10200.00')
    expect(pos).toHaveTextContent('10.2%')                       // 10200 / 100130
    expect(screen.getByTestId('open-orders')).toHaveTextContent('50.75')
    const recent = screen.getByTestId('recent-orders')
    expect(recent).toHaveTextContent('REJECTED_BY_GATE')
    expect(recent).toHaveTextContent('no_adv')
    expect(screen.getByTestId('fills')).toHaveTextContent('40.0')  // (50.2−50)/50 bps
  })

  it('shows the blocking banner and accept entry when reconciliation is not clean', async () => {
    api.apiFetchExecutionStatus.mockResolvedValue({ data: {
      ...STATUS,
      last_snapshot: { ...STATUS.last_snapshot, status: 'discrepancy',
        discrepancies: [{ ticker: 'AAA', expected: 200, broker: 400 }] },
    } })
    render(<ExecutionPanel />)
    await waitFor(() => expect(screen.getByTestId('snap-status')).toHaveTextContent('discrepancy'))
    expect(screen.getByText(/不会下单/)).toBeInTheDocument()
    const d = screen.getByTestId('discrepancies')
    expect(d).toHaveTextContent('AAA')
    expect(d).toHaveTextContent('400')
    expect(screen.getByText('接受差异…')).toBeInTheDocument()
  })
})

describe('ExecutionPanel — kill switch is two-step', () => {
  it('opening the dialog only arms; confirm needs actor, reason and the phrase', async () => {
    render(<ExecutionPanel />)
    await waitFor(() => expect(screen.getByText('一键全平…')).toBeInTheDocument())
    fireEvent.click(screen.getByText('一键全平…'))
    await waitFor(() => expect(screen.getByTestId('token-state')).toHaveTextContent('剩余'))
    expect(api.apiArmKillSwitch).toHaveBeenCalledWith('engage')
    expect(api.apiConfirmKillSwitch).not.toHaveBeenCalled()

    const go = screen.getByText('确认执行')
    expect(go).toBeDisabled()
    fireEvent.change(screen.getByLabelText('操作人'), { target: { value: 'kevin' } })
    fireEvent.change(screen.getByLabelText('原因'), { target: { value: 'drill' } })
    expect(go).toBeDisabled()                                   // 还差确认短语
    fireEvent.change(screen.getByLabelText('确认短语'), { target: { value: '解除' } })
    expect(go).toBeDisabled()                                   // 短语不对
    fireEvent.change(screen.getByLabelText('确认短语'), { target: { value: '全平' } })
    expect(go).not.toBeDisabled()
    fireEvent.click(go)
    await waitFor(() => expect(api.apiConfirmKillSwitch).toHaveBeenCalledWith('tok-123', 'kevin', 'drill'))
    await waitFor(() => expect(screen.getByText('全平熔断已开启')).toBeInTheDocument())
  })

  it('a broker outage is surfaced, not hidden', async () => {
    api.apiConfirmKillSwitch.mockResolvedValue({ data: {
      action: 'engage', state: { engaged: true }, error: '券商不可用，熔断已生效、全平待续：OpenD down' } })
    render(<ExecutionPanel />)
    await waitFor(() => expect(screen.getByText('一键全平…')).toBeInTheDocument())
    fireEvent.click(screen.getByText('一键全平…'))
    await waitFor(() => expect(screen.getByTestId('token-state')).toHaveTextContent('剩余'))
    fireEvent.change(screen.getByLabelText('操作人'), { target: { value: 'kevin' } })
    fireEvent.change(screen.getByLabelText('原因'), { target: { value: 'panic' } })
    fireEvent.change(screen.getByLabelText('确认短语'), { target: { value: '全平' } })
    fireEvent.click(screen.getByText('确认执行'))
    await waitFor(() => expect(screen.getByText(/OpenD down/)).toBeInTheDocument())
  })

  it('engaged state is displayed with who and why', async () => {
    api.apiFetchExecutionStatus.mockResolvedValue({ data: {
      ...STATUS, kill_switch: { engaged: true, actor: 'kevin', reason: 'drill', at: '2025-03-05T10:00:00' } } })
    render(<ExecutionPanel />)
    await waitFor(() => expect(screen.getByTestId('ks-engaged')).toHaveTextContent('kevin'))
    expect(screen.getByTestId('ks-engaged')).toHaveTextContent('drill')
    expect(screen.getByText('解除熔断…')).toBeInTheDocument()
  })
})

describe('ExecutionPanel — fidelity', () => {
  it('renders the report fields and the unidentifiable permanent impact', async () => {
    api.apiFetchFidelity.mockResolvedValue({ data: {
      period_start: '2025-03-01', period_end: '2025-03-31', n_live_orders: 3, n_live_fills: 3,
      n_matched_sim: 3, n_ref_mismatch: 0, live_fill_ratio_median: 1, sim_fill_ratio_median: 1,
      n_live_unfilled: 0,
      total_slippage_bps: { n: 3, mean: 4, median: 4.5, p90: 9 },
      overnight_gap_bps: { n: 3, mean: 3, median: 3.5, p90: 7 },
      at_open_exec_bps: { n: 3, mean: 1, median: 1.2, p90: 2 },
      assumed_spread_bps: 2, current_impact_coef: 0.1, recommended_impact_coef: 0.1,
      recommended_scale: 1, permanent_impact_bps: null,
      permanent_impact_note: '纸交易成交不改变真实市场价格', notes: [], warnings: ['样本不足'], markdown: '',
    } })
    render(<ExecutionPanel />)
    await waitFor(() => expect(screen.getByText('生成')).toBeInTheDocument())
    fireEvent.click(screen.getByText('生成'))
    await waitFor(() => expect(screen.getByTestId('fidelity')).toHaveTextContent('中位 4.50'))
    const f = screen.getByTestId('fidelity')
    expect(f).toHaveTextContent('中位 3.50')
    expect(f).toHaveTextContent('不可识别')
    expect(f).toHaveTextContent('样本不足')
    expect(api.apiFetchFidelity).toHaveBeenCalledTimes(1)
  })
})
