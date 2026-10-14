/**
 * ApprovalQueue.test.tsx — 待批准队列（FE-9）+ 基本面字段标注（FE-10 后半）
 *
 *  · 批准的人要看得到：这个因子用了哪些基本面字段、这些数字从哪天起才算"已知"
 *  · 只用价格的因子不显示这一行；老版本后端（没有 sparse_fields 字段）也不能崩
 *  · 批准 / 拒绝真的带着 id 和理由调了接口
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react'

const api = vi.hoisted(() => ({
  apiFetchPending: vi.fn(),
  apiApproveAlpha: vi.fn(),
  apiRejectAlpha:  vi.fn(),
}))
vi.mock('../../../api/client', () => api)

import ApprovalQueue from '../../../components/dashboard/ApprovalQueue'

const RULE = '披露日之后的第一个交易日起可见；超过 200 天没有新披露则置空'
const FUND = {
  alpha_id: 11, dsl: 'fund_rank(earnings_yield) + fund_rank(sue)', hypothesis: 'value + earnings',
  sharpe: 0.42, ic_ir: 0.11,
  sparse_fields: [
    { name: 'earnings_yield', class: 'valuation', note: '盈利收益率 E/P' },
    { name: 'sue',            class: 'quarterly', note: '标准化盈利超预期' },
  ],
  visibility_rule: RULE,
}
const PRICE = {
  alpha_id: 12, dsl: 'rank(ts_delta(log(close), 252))', hypothesis: 'momentum', sharpe: 0.8, ic_ir: 0.05,
  sparse_fields: [], visibility_rule: null,
}

beforeEach(() => {
  vi.clearAllMocks()
  vi.spyOn(window, 'prompt').mockReturnValue('looks fine')
})

describe('ApprovalQueue — fundamental field annotations', () => {
  it('lists the fundamental fields a candidate uses and the visibility rule', async () => {
    api.apiFetchPending.mockResolvedValue({ data: [FUND, PRICE] })
    render(<ApprovalQueue />)
    const row = await screen.findByTestId('pending-sparse-11')
    expect(within(row).getByText('earnings_yield')).toHaveAttribute('title', '盈利收益率 E/P')
    expect(within(row).getByText('sue')).toHaveAttribute('title', '标准化盈利超预期')
    expect(row).toHaveTextContent(RULE)
    expect(row).toHaveAttribute('title', RULE)
  })

  it('shows no annotation for a price-only candidate', async () => {
    api.apiFetchPending.mockResolvedValue({ data: [FUND, PRICE] })
    render(<ApprovalQueue />)
    await screen.findByTestId('pending-sparse-11')
    expect(screen.queryByTestId('pending-sparse-12')).toBeNull()
    expect(screen.getByText('rank(ts_delta(log(close), 252))')).toBeInTheDocument()
  })

  it('renders a payload from a backend that predates the field (no sparse_fields key)', async () => {
    const { sparse_fields: _s, visibility_rule: _v, ...legacy } = PRICE
    api.apiFetchPending.mockResolvedValue({ data: [legacy] })
    render(<ApprovalQueue />)
    expect(await screen.findByText('rank(ts_delta(log(close), 252))')).toBeInTheDocument()
    expect(screen.queryByTestId('pending-sparse-12')).toBeNull()
  })

  it('approving and rejecting call the API with the id (and the typed reason for a reject)', async () => {
    api.apiFetchPending.mockResolvedValue({ data: [FUND, PRICE] })
    api.apiApproveAlpha.mockResolvedValue({ data: {} })
    api.apiRejectAlpha.mockResolvedValue({ data: {} })
    const onChange = vi.fn()
    render(<ApprovalQueue onChange={onChange} />)
    await screen.findByTestId('pending-sparse-11')
    fireEvent.click(screen.getAllByText('批准')[0])
    await waitFor(() => expect(api.apiApproveAlpha).toHaveBeenCalledWith(11, ''))
    fireEvent.click(screen.getAllByText('拒绝')[1])
    await waitFor(() => expect(api.apiRejectAlpha).toHaveBeenCalledWith(12, 'looks fine'))
    await waitFor(() => expect(onChange).toHaveBeenCalledTimes(2))
    expect(api.apiFetchPending).toHaveBeenCalledTimes(3)              // 首次 + 每次决定后刷新
  })

  it('shows the fields even when the rule is missing', async () => {
    api.apiFetchPending.mockResolvedValue({ data: [{ ...FUND, visibility_rule: null }] })
    render(<ApprovalQueue />)
    const row = await screen.findByTestId('pending-sparse-11')
    expect(within(row).getByText('sue')).toBeInTheDocument()
    expect(row).not.toHaveTextContent('第一个交易日')
  })
})
