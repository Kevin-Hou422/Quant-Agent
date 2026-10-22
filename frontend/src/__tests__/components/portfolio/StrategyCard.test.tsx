/**
 * StrategyCard.test.tsx — 2026-10-22 用户核验 C1 / C6
 *
 *  · C1：卡片按**最终**结论显示（verdict.passed 与 cfg.passed 都为真才是"全过"），并逐门显示原始结论；
 *        冻结规格过、流程滚动前推没过 → 显示"未全过"且两道门分别标出
 *  · C6：状态按后端的中文标签显示；approved 只能"开始纸交易（未验证）"或"进入联调"；
 *        只有 paper_running 才有"申请验证"
 */
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'

vi.mock('../../../api/client', () => ({}))

import { StrategyCard } from '../../../components/portfolio/PortfolioView'
import type { StrategyConfigItem } from '../../../types'

function cfg(over: Partial<StrategyConfigItem>): StrategyConfigItem {
  return {
    id: 7, name: '', status: 'proposed', version: 1, factors: ['1'], combo_weights: { '1': 1 },
    aum: 10000, method: 'equal_weight', passed: false, verdict: {}, risk_report: {},
    turnover_ann: 1, no_trade_band: 0.01, created_at: null, ...over,
  }
}

const SPLIT = {
  passed: false, paper_grade: 'C',
  gates: {
    frozen_spec: { evaluated: true, passed: true },
    procedure_oos: { evaluated: true, passed: false },
    fundamentals_quality: { evaluated: true, passed: true },
  },
}

describe('StrategyCard', () => {
  it('shows the final verdict and every gate separately', () => {
    render(<StrategyCard cfg={cfg({ verdict: SPLIT })} onDecision={() => {}} busy={false} />)
    expect(screen.getByText('必须的门未全过')).toBeInTheDocument()
    expect(screen.getByTestId('gate-frozen_spec')).toHaveTextContent('冻结规格：过')
    expect(screen.getByTestId('gate-procedure_oos')).toHaveTextContent('流程滚动前推：未过')
    expect(screen.queryByText('必须的门全过')).toBeNull()
  })

  it('an unevaluated gate is shown as such', () => {
    const v = { passed: false, gates: { frozen_spec: { evaluated: false, passed: false, error: '门崩了' } } }
    render(<StrategyCard cfg={cfg({ verdict: v })} onDecision={() => {}} busy={false} />)
    expect(screen.getByTestId('gate-frozen_spec')).toHaveTextContent('未评估')
  })

  it('an approved strategy can start paper trading or integration, not verification', () => {
    const on = vi.fn()
    render(<StrategyCard cfg={cfg({ status: 'approved', status_label: '已批准（未运行）' })}
      onDecision={on} busy={false} />)
    expect(screen.getByText('已批准（未运行）')).toBeInTheDocument()
    fireEvent.click(screen.getByText(/开始纸交易（收集证据，未验证）/))
    expect(on).toHaveBeenCalledWith(7, 'activate')
    fireEvent.click(screen.getByText(/进入工程联调/))
    expect(on).toHaveBeenCalledWith(7, 'integration')
    expect(screen.queryByText(/申请验证/)).toBeNull()
  })

  it('only a paper-running strategy can apply for verification', () => {
    const on = vi.fn()
    render(<StrategyCard cfg={cfg({ status: 'paper_running', status_label: '纸交易·收集证据（未验证）' })}
      onDecision={on} busy={false} />)
    fireEvent.click(screen.getByText(/申请验证/))
    expect(on).toHaveBeenCalledWith(7, 'verify')
  })
})
