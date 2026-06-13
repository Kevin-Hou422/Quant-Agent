/**
 * MetricsGrid.test.tsx — MetricsGrid 组件测试
 *
 * MetricsGrid 依赖 useWorkspaceStore，通过设置 store state 来测试渲染。
 */
import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, act } from '@testing-library/react'
import MetricsGrid from '../../../components/analysis/MetricsGrid'
import { useWorkspaceStore } from '../../../store/workspaceStore'
import type { SimResult } from '../../../types'

const MOCK_RESULT: SimResult = {
  dsl: 'rank(close)',
  is_metrics: {
    sharpe_ratio: 1.5,
    annualized_return: 0.20,
    max_drawdown: -0.08,
    mean_ic: 0.05,
    ic_ir: 0.6,
    ann_turnover: 0.4,
  },
  oos_metrics: {
    sharpe_ratio: 1.1,
    annualized_return: 0.12,
    max_drawdown: -0.12,
    mean_ic: 0.03,
    ic_ir: 0.4,
    ann_turnover: 0.4,
  },
  overfitting_score: 0.2,
  is_overfit: false,
  ic_decay: { t1: 0.04, t5: 0.02 },
  pnl_is: [0.01, -0.02, 0.03],
  pnl_oos: [0.008, -0.01],
}

describe('MetricsGrid - no result', () => {
  beforeEach(() => {
    act(() => useWorkspaceStore.getState().setSimulationResult(null))
  })

  it('shows no results placeholder when simulationResult is null', () => {
    render(<MetricsGrid />)
    expect(screen.getByText(/No results yet/i)).toBeInTheDocument()
  })
})

describe('MetricsGrid - with result', () => {
  beforeEach(() => {
    act(() => useWorkspaceStore.getState().setSimulationResult(MOCK_RESULT))
  })

  afterEach(() => {
    act(() => useWorkspaceStore.getState().setSimulationResult(null))
  })

  it('renders Sharpe Ratio label', () => {
    render(<MetricsGrid />)
    expect(screen.getByText('Sharpe Ratio')).toBeInTheDocument()
  })

  it('renders IS Sharpe value', () => {
    render(<MetricsGrid />)
    expect(screen.getByText('1.500')).toBeInTheDocument()
  })

  it('renders Annualized Return label', () => {
    render(<MetricsGrid />)
    expect(screen.getByText('Annualized Return')).toBeInTheDocument()
  })

  it('renders Max Drawdown label', () => {
    render(<MetricsGrid />)
    expect(screen.getByText('Max Drawdown')).toBeInTheDocument()
  })

  it('renders IC-IR label', () => {
    render(<MetricsGrid />)
    expect(screen.getByText('IC-IR')).toBeInTheDocument()
  })

  it('shows OverfitBadge with Healthy for low score', () => {
    render(<MetricsGrid />)
    expect(screen.getByText('Healthy')).toBeInTheDocument()
  })

  it('shows IC Decay section when ic_decay is present', () => {
    render(<MetricsGrid />)
    expect(screen.getByText('IC Decay')).toBeInTheDocument()
    expect(screen.getByText('t1')).toBeInTheDocument()
  })
})

describe('MetricsGrid - overfit scenario', () => {
  it('shows Overfit! badge when overfitting_score is high', () => {
    act(() => useWorkspaceStore.getState().setSimulationResult({
      ...MOCK_RESULT,
      overfitting_score: 0.8,
      is_overfit: true,
    }))
    render(<MetricsGrid />)
    expect(screen.getByText(/Overfit!/i)).toBeInTheDocument()
    act(() => useWorkspaceStore.getState().setSimulationResult(null))
  })
})

// Fix: add afterEach at describe scope - already handled inline above
