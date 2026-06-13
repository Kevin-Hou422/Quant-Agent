/**
 * OverfitBadge.test.tsx — OverfitBadge 组件测试
 */
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import OverfitBadge from '../../../components/analysis/OverfitBadge'

describe('OverfitBadge - full mode', () => {
  it('shows Healthy when score < 0.4', () => {
    render(<OverfitBadge score={0.2} />)
    expect(screen.getByText('Healthy')).toBeInTheDocument()
  })

  it('shows Overfit Risk when score between 0.4 and 0.6', () => {
    render(<OverfitBadge score={0.5} />)
    expect(screen.getByText(/Overfit Risk/i)).toBeInTheDocument()
  })

  it('shows Overfit! when score >= 0.6', () => {
    render(<OverfitBadge score={0.8} />)
    expect(screen.getByText(/Overfit!/i)).toBeInTheDocument()
  })

  it('shows percentage in Overfit Risk badge', () => {
    render(<OverfitBadge score={0.5} />)
    expect(screen.getByText(/50%/i)).toBeInTheDocument()
  })

  it('shows percentage in Overfit! badge', () => {
    render(<OverfitBadge score={0.75} />)
    expect(screen.getByText(/75%/i)).toBeInTheDocument()
  })

  it('boundary: score 0.0 shows Healthy', () => {
    render(<OverfitBadge score={0.0} />)
    expect(screen.getByText('Healthy')).toBeInTheDocument()
  })

  it('boundary: score 1.0 shows Overfit!', () => {
    render(<OverfitBadge score={1.0} />)
    expect(screen.getByText(/Overfit!/i)).toBeInTheDocument()
  })
})

describe('OverfitBadge - inline mode', () => {
  it('renders inline without text labels', () => {
    const { container } = render(<OverfitBadge score={0.3} inline />)
    expect(screen.queryByText('Healthy')).not.toBeInTheDocument()
    // Should render a span with rounded-full class
    expect(container.querySelector('.rounded-full')).toBeInTheDocument()
  })

  it('inline with high score renders with animate-pulse', () => {
    const { container } = render(<OverfitBadge score={0.7} inline />)
    const dot = container.querySelector('.animate-pulse')
    expect(dot).toBeInTheDocument()
  })

  it('inline with low score has no animate-pulse', () => {
    const { container } = render(<OverfitBadge score={0.2} inline />)
    const dot = container.querySelector('.animate-pulse')
    expect(dot).not.toBeInTheDocument()
  })
})
