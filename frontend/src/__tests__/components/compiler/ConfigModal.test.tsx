/**
 * ConfigModal.test.tsx — ConfigModal 组件测试
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, act } from '@testing-library/react'
import { useWorkspaceStore } from '../../../store/workspaceStore'

vi.mock('lucide-react', () => ({
  X:           () => <span data-testid="icon-x">×</span>,
  Database:    () => <span data-testid="icon-db" />,
  ExternalLink: () => <span data-testid="icon-external" />,
}))

import ConfigModal from '../../../components/compiler/ConfigModal'

const onClose = vi.fn()

describe('ConfigModal - rendering', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders the modal title', () => {
    render(<ConfigModal onClose={onClose} />)
    expect(screen.getByText('Simulation Config')).toBeInTheDocument()
  })

  it('renders Execution Delay slider', () => {
    render(<ConfigModal onClose={onClose} />)
    expect(screen.getByText('Execution Delay (days)')).toBeInTheDocument()
  })

  it('renders Decay Window slider', () => {
    render(<ConfigModal onClose={onClose} />)
    expect(screen.getByText('Decay Window')).toBeInTheDocument()
  })

  it('renders Portfolio Mode select', () => {
    render(<ConfigModal onClose={onClose} />)
    expect(screen.getByText('Portfolio Mode')).toBeInTheDocument()
  })

  it('renders the active dataset name', () => {
    render(<ConfigModal onClose={onClose} />)
    expect(screen.getByText('us_tech_large')).toBeInTheDocument()
  })

  it('renders close button', () => {
    render(<ConfigModal onClose={onClose} />)
    expect(screen.getByTestId('icon-x')).toBeInTheDocument()
  })
})

describe('ConfigModal - interactions', () => {
  it('calls onClose when close button clicked', () => {
    render(<ConfigModal onClose={onClose} />)
    const closeBtn = screen.getByRole('button', { name: /×/i })
    fireEvent.click(closeBtn)
    expect(onClose).toHaveBeenCalledOnce()
  })

  it('calls onClose and sets view to DATASET when Change clicked', () => {
    render(<ConfigModal onClose={onClose} />)
    const changeBtn = screen.getByText('Change')
    fireEvent.click(changeBtn)
    expect(onClose).toHaveBeenCalled()
    expect(useWorkspaceStore.getState().activeView).toBe('DATASET')
    act(() => useWorkspaceStore.getState().setActiveView('COMPILER'))
  })

  it('updates simConfig when delay slider changes', () => {
    render(<ConfigModal onClose={onClose} />)
    const sliders = screen.getAllByRole('slider')
    // First slider is Execution Delay
    const delaySlider = sliders[0]
    fireEvent.change(delaySlider, { target: { value: '3' } })
    expect(useWorkspaceStore.getState().simConfig.delay).toBe(3)
    act(() => useWorkspaceStore.getState().setSimConfig({ delay: 1 }))
  })

  it('updates portfolio_mode when select changes', () => {
    render(<ConfigModal onClose={onClose} />)
    const select = screen.getByRole('combobox')
    fireEvent.change(select, { target: { value: 'decile' } })
    expect(useWorkspaceStore.getState().simConfig.portfolio_mode).toBe('decile')
    act(() => useWorkspaceStore.getState().setSimConfig({ portfolio_mode: 'long_short' }))
  })
})
