/**
 * GlobalSidebar.test.tsx — GlobalSidebar 导航测试
 *
 * Mock useQuantWorkspace hook 避免真实 API 调用
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, act } from '@testing-library/react'
import { useWorkspaceStore } from '../../../store/workspaceStore'

// Mock useQuantWorkspace to avoid real API calls
vi.mock('../../../hooks/useQuantWorkspace', () => ({
  useQuantWorkspace: () => ({
    runBacktest: vi.fn(),
    runOptimize: vi.fn(),
  }),
}))

// Mock lucide-react icons to avoid SVG rendering issues
vi.mock('lucide-react', () => ({
  MessageSquare: () => <span data-testid="icon-chat" />,
  Code2:         () => <span data-testid="icon-compiler" />,
  BookOpen:      () => <span data-testid="icon-ledger" />,
  Database:      () => <span data-testid="icon-data" />,
  Play:          () => <span data-testid="icon-play" />,
  Zap:           () => <span data-testid="icon-zap" />,
  Activity:      () => <span data-testid="icon-live" />,
}))

import GlobalSidebar from '../../../components/layout/GlobalSidebar'

describe('GlobalSidebar - navigation buttons', () => {
  beforeEach(() => {
    act(() => useWorkspaceStore.getState().setActiveView('COMPILER'))
    act(() => useWorkspaceStore.getState().setLedgerOpen(false))
    act(() => useWorkspaceStore.getState().setStatus('idle'))
  })

  it('renders Chat, Compiler, Ledger, Data buttons', () => {
    render(<GlobalSidebar />)
    expect(screen.getByTitle('Chat')).toBeInTheDocument()
    expect(screen.getByTitle('Compiler')).toBeInTheDocument()
    expect(screen.getByTitle('Ledger')).toBeInTheDocument()
    expect(screen.getByTitle('Data')).toBeInTheDocument()
  })

  it('clicking Chat button switches activeView to CHAT', () => {
    render(<GlobalSidebar />)
    fireEvent.click(screen.getByTitle('Chat'))
    expect(useWorkspaceStore.getState().activeView).toBe('CHAT')
  })

  it('clicking Data button switches activeView to DATASET', () => {
    render(<GlobalSidebar />)
    fireEvent.click(screen.getByTitle('Data'))
    expect(useWorkspaceStore.getState().activeView).toBe('DATASET')
  })

  it('clicking Compiler button switches activeView to COMPILER', () => {
    act(() => useWorkspaceStore.getState().setActiveView('CHAT'))
    render(<GlobalSidebar />)
    fireEvent.click(screen.getByTitle('Compiler'))
    expect(useWorkspaceStore.getState().activeView).toBe('COMPILER')
  })

  it('shows logo QA text', () => {
    render(<GlobalSidebar />)
    expect(screen.getByText('QA')).toBeInTheDocument()
  })
})

describe('GlobalSidebar - active state highlighting', () => {
  it('Compiler button appears active in COMPILER mode', () => {
    act(() => useWorkspaceStore.getState().setActiveView('COMPILER'))
    render(<GlobalSidebar />)
    const compilerBtn = screen.getByTitle('Compiler')
    expect(compilerBtn.className).toContain('emerald')
  })

  it('Chat button appears active in CHAT mode', () => {
    act(() => useWorkspaceStore.getState().setActiveView('CHAT'))
    render(<GlobalSidebar />)
    const chatBtn = screen.getByTitle('Chat')
    expect(chatBtn.className).toContain('emerald')
    act(() => useWorkspaceStore.getState().setActiveView('COMPILER'))
  })
})

describe('GlobalSidebar - action buttons in COMPILER mode', () => {
  beforeEach(() => {
    act(() => useWorkspaceStore.getState().setActiveView('COMPILER'))
    act(() => useWorkspaceStore.getState().setStatus('idle'))
  })

  it('shows Run button in COMPILER mode', () => {
    render(<GlobalSidebar />)
    expect(screen.getByTitle('Run Backtest')).toBeInTheDocument()
  })

  it('shows AI Opt button in COMPILER mode', () => {
    render(<GlobalSidebar />)
    expect(screen.getByTitle('AI Optimize')).toBeInTheDocument()
  })

  it('Run button disabled when status is backtesting', () => {
    act(() => useWorkspaceStore.getState().setStatus('backtesting'))
    render(<GlobalSidebar />)
    const runBtn = screen.getByTitle('Run Backtest')
    expect(runBtn).toBeDisabled()
    act(() => useWorkspaceStore.getState().setStatus('idle'))
  })
})

describe('GlobalSidebar - dataset info', () => {
  it('shows dataset name in sidebar footer', () => {
    render(<GlobalSidebar />)
    // The default config has 'us_tech_large'
    expect(screen.getByText('us_tech_large')).toBeInTheDocument()
  })
})
