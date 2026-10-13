/**
 * FundamentalsPanel.test.tsx — FE-10 基本面字段浏览 + 披露时点标注
 *
 *  · 面板上的每个数字绑回 /api/datasets/{name}/fundamentals 的真实字段（§N：防装饰）
 *  · 接口没给的值显示 "—"，不拿 0 顶上（coverage=null、fields=null）
 *  · 开关关着 / 非美股 / 接口失败，各有明确的说法
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, within } from '@testing-library/react'

const api = vi.hoisted(() => ({ apiFetchDatasetFundamentals: vi.fn() }))
vi.mock('../../../api/client', () => api)

import FundamentalsPanel, { coverageText } from '../../../components/dataset/FundamentalsPanel'

const FIELDS = [
  { name: 'market_cap',          class: 'valuation', note: '市值', coverage: 0.9684 },
  { name: 'earnings_yield',      class: 'valuation', note: 'E/P',  coverage: 0.5 },
  { name: 'roe',                 class: 'quarterly', note: 'ROE',  coverage: 0.25 },
  { name: 'days_since_earnings', class: 'event',     note: '财报日', coverage: 1 },
]

const BODY = {
  dataset: 'us_broad_large', supported: true, enabled: true, identity_configured: true,
  visibility_rule: '披露日之后的第一个交易日起可见；超过 200 天没有新披露则置空',
  fields: FIELDS, n_tickers: 4, n_with_facts: 3, n_stale: 1,
  tickers: [
    { ticker: 'AAPL', has_facts: true, last_filed: '2026-07-31', last_period_end: '2026-06-27', stale: false,
      fields: { market_cap: true, earnings_yield: true, roe: true }, synced_at: '2026-10-12T00:00:00+00:00' },
    { ticker: 'BKNG', has_facts: true, last_filed: '2026-08-04', last_period_end: '2026-06-30', stale: false,
      fields: { market_cap: true, earnings_yield: true, roe: false, book_to_price: false }, synced_at: null },
    { ticker: 'C',    has_facts: true, last_filed: '2026-02-20', last_period_end: '2025-12-31', stale: true,
      fields: null, synced_at: null },
    { ticker: 'SPY',  has_facts: false, last_filed: null, last_period_end: null, stale: false,
      fields: null, synced_at: null },
  ],
}

const reply = (over: object = {}) =>
  api.apiFetchDatasetFundamentals.mockResolvedValue({ data: { ...BODY, ...over } })

beforeEach(() => vi.clearAllMocks())

describe('coverageText', () => {
  it('formats a share as a whole percent and shows a dash when the API gave nothing', () => {
    expect(coverageText(0.9684)).toBe('97%')
    expect(coverageText(0)).toBe('0%')
    expect(coverageText(1)).toBe('100%')
    expect(coverageText(null)).toBe('—')
  })
})

describe('FundamentalsPanel', () => {
  it('asks the API for the selected dataset and re-asks when it changes', async () => {
    reply()
    const { rerender } = render(<FundamentalsPanel datasetName="us_broad_large" />)
    await screen.findByTestId('fundamentals-summary')
    expect(api.apiFetchDatasetFundamentals).toHaveBeenCalledWith('us_broad_large')
    rerender(<FundamentalsPanel datasetName="us_tech_large" />)
    await waitFor(() => expect(api.apiFetchDatasetFundamentals).toHaveBeenLastCalledWith('us_tech_large'))
    expect(api.apiFetchDatasetFundamentals).toHaveBeenCalledTimes(2)
  })

  it('shows every field with its class and the coverage the API returned', async () => {
    reply()
    render(<FundamentalsPanel datasetName="us_broad_large" />)
    const mc = await screen.findByTestId('fund-field-market_cap')
    expect(within(mc).getByText('97%')).toBeInTheDocument()
    expect(within(mc).getByText('daily')).toBeInTheDocument()
    const roe = screen.getByTestId('fund-field-roe')
    expect(within(roe).getByText('25%')).toBeInTheDocument()
    expect(within(roe).getByText('quarterly step')).toBeInTheDocument()
    expect(within(screen.getByTestId('fund-field-days_since_earnings')).getByText('event counter')).toBeInTheDocument()
    expect(roe).toHaveAttribute('title', 'ROE')
    expect(screen.getAllByTestId(/^fund-field-/)).toHaveLength(FIELDS.length)
  })

  it('shows the filing date each ticker became visible after, and the visibility rule', async () => {
    reply()
    render(<FundamentalsPanel datasetName="us_broad_large" />)
    const aapl = await screen.findByTestId('fund-ticker-AAPL')
    expect(within(aapl).getByText('2026-07-31')).toBeInTheDocument()
    expect(within(aapl).getByText('2026-06-27')).toBeInTheDocument()
    expect(screen.getByTestId('fundamentals-rule')).toHaveTextContent('披露日之后的第一个交易日起可见')
  })

  it('names the fields a ticker cannot provide, and says "none" when all are available', async () => {
    reply()
    render(<FundamentalsPanel datasetName="us_broad_large" />)
    const bkng = await screen.findByTestId('fund-ticker-BKNG')
    expect(within(bkng).getByText('roe, book_to_price')).toBeInTheDocument()
    expect(within(screen.getByTestId('fund-ticker-AAPL')).getByText('none')).toBeInTheDocument()
  })

  it('flags stale tickers and does not guess fields that were never computed', async () => {
    reply()
    render(<FundamentalsPanel datasetName="us_broad_large" />)
    const c = await screen.findByTestId('fund-ticker-C')
    expect(within(c).getByText(/stale/)).toBeInTheDocument()
    expect(within(c).getAllByText('—')).toHaveLength(1)                 // fields=null → 不知道，不写 none
    expect(within(screen.getByTestId('fund-ticker-AAPL')).queryByText(/stale/)).toBeNull()
    const spy = screen.getByTestId('fund-ticker-SPY')
    expect(within(spy).getByText('no filings')).toBeInTheDocument()
    expect(within(spy).getAllByText('—')).toHaveLength(2)               // 披露日、期末日都没有
  })

  it('summarises how many tickers have filings and how many are stale', async () => {
    reply()
    render(<FundamentalsPanel datasetName="us_broad_large" />)
    const summary = await screen.findByTestId('fundamentals-summary')
    expect(summary).toHaveTextContent('3 / 4 tickers have filings on disk')
    expect(summary).toHaveTextContent('1 stale')
    expect(summary).not.toHaveTextContent('EDGAR_USER_AGENT')
  })

  it('omits the stale note when nothing is stale and warns when no identity is configured', async () => {
    reply({ n_stale: 0, identity_configured: false })
    render(<FundamentalsPanel datasetName="us_broad_large" />)
    const summary = await screen.findByTestId('fundamentals-summary')
    expect(summary).not.toHaveTextContent('stale')
    expect(summary).toHaveTextContent('EDGAR_USER_AGENT not set')
  })

  it('says whether the fields are actually attached to the dataset', async () => {
    reply()
    const { unmount } = render(<FundamentalsPanel datasetName="us_broad_large" />)
    expect(await screen.findByTestId('fundamentals-switch')).toHaveTextContent('attached to this dataset')
    unmount()
    reply({ enabled: false })
    render(<FundamentalsPanel datasetName="us_broad_large" />)
    expect(await screen.findByTestId('fundamentals-switch')).toHaveTextContent('off (ENABLE_FUNDAMENTALS=false)')
  })

  it('explains that non-US datasets have no fundamentals instead of showing empty tables', async () => {
    reply({ supported: false, enabled: false, tickers: [], n_with_facts: 0,
            fields: FIELDS.map((f) => ({ ...f, coverage: null })) })
    render(<FundamentalsPanel datasetName="crypto_major" />)
    expect(await screen.findByTestId('fundamentals-unsupported')).toHaveTextContent('US equities only')
    expect(screen.queryByTestId('fundamentals-summary')).toBeNull()
    expect(screen.queryByTestId('fund-field-roe')).toBeNull()
    expect(screen.queryByTestId('fundamentals-switch')).toBeNull()
  })

  it('hides the ticker table when no ticker has filings but still lists the fields', async () => {
    reply({ n_with_facts: 0, n_stale: 0,
            tickers: BODY.tickers.map((t) => ({ ...t, has_facts: false, last_filed: null })),
            fields: FIELDS.map((f) => ({ ...f, coverage: 0 })) })
    render(<FundamentalsPanel datasetName="us_broad_large" />)
    expect(within(await screen.findByTestId('fund-field-roe')).getByText('0%')).toBeInTheDocument()
    expect(screen.queryByTestId('fund-ticker-AAPL')).toBeNull()
  })

  it('reports a failed request instead of rendering stale or empty data', async () => {
    api.apiFetchDatasetFundamentals.mockRejectedValue(new Error('boom'))
    render(<FundamentalsPanel datasetName="us_broad_large" />)
    expect(await screen.findByTestId('fundamentals-error')).toBeInTheDocument()
    expect(screen.queryByTestId('fundamentals-summary')).toBeNull()
  })

  it('ignores a slow response for a dataset the user has already left', async () => {
    let resolveFirst: (v: unknown) => void = () => {}
    api.apiFetchDatasetFundamentals
      .mockImplementationOnce(() => new Promise((res) => { resolveFirst = res }))
      .mockResolvedValueOnce({ data: { ...BODY, dataset: 'us_tech_large', n_with_facts: 1, n_tickers: 2 } })
    const { rerender } = render(<FundamentalsPanel datasetName="us_broad_large" />)
    rerender(<FundamentalsPanel datasetName="us_tech_large" />)
    expect(await screen.findByTestId('fundamentals-summary')).toHaveTextContent('1 / 2 tickers')
    resolveFirst({ data: BODY })                                        // 旧请求这时才回来
    await new Promise((r) => setTimeout(r, 20))
    expect(screen.getByTestId('fundamentals-summary')).toHaveTextContent('1 / 2 tickers')
  })
})
