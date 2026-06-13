/**
 * workflow.test.tsx — 前端工作流集成测试
 *
 * 测试关键状态流：view 切换、消息追加、模拟结果设置。
 * Mock 所有 API 调用避免真实网络请求。
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, act, waitFor } from '@testing-library/react'
import { useWorkspaceStore } from '../../store/workspaceStore'

// Mock all external dependencies
vi.mock('@monaco-editor/react', () => ({
  default: ({ value }: { value: string }) => (
    <textarea data-testid="monaco-editor" defaultValue={value} />
  ),
}))

vi.mock('echarts-for-react', () => ({
  default: () => <div data-testid="echarts" />,
}))

vi.mock('lucide-react', () => {
  const icons = ['MessageSquare', 'Code2', 'BookOpen', 'Database', 'Play', 'Zap',
    'ChevronDown', 'ChevronRight', 'X', 'Plus', 'Send', 'Settings', 'RefreshCw',
    'CheckCircle', 'AlertCircle', 'Clock', 'TrendingUp', 'BarChart2', 'Layers',
    'ExternalLink', 'Download', 'Filter', 'Info']
  const mocked: Record<string, () => JSX.Element> = {}
  icons.forEach(name => { mocked[name] = () => <span data-testid={`icon-${name}`} /> })
  return mocked
})

vi.mock('../../hooks/useQuantWorkspace', () => ({
  useQuantWorkspace: () => ({
    runBacktest: vi.fn(),
    runOptimize: vi.fn(),
    fetchDatasets: vi.fn(),
    initSession: vi.fn(),
  }),
}))

vi.mock('../../api/client', () => ({
  apiChat:          vi.fn().mockResolvedValue({ data: { reply: 'Mock reply', session_id: 'sid', dsl: 'rank(close)', metrics: null } }),
  apiFetchDatasets: vi.fn().mockResolvedValue({ data: { datasets: [], total: 0 } }),
  apiSimulate:      vi.fn().mockResolvedValue({ data: {} }),
  apiRunBacktest:   vi.fn().mockResolvedValue({ data: { dsl: 'rank(close)', report: {} } }),
  apiWalkForwardBacktest: vi.fn().mockResolvedValue({ data: {} }),
  apiCreateSession: vi.fn().mockResolvedValue({ data: { session_id: 'new-sid', title: 'New' } }),
  apiListSessions:  vi.fn().mockResolvedValue({ data: { sessions: [], count: 0 } }),
  apiGetSession:    vi.fn().mockResolvedValue({ data: { session_id: 'sid', messages: [] } }),
  apiDeleteSession: vi.fn().mockResolvedValue({ data: {} }),
  apiRenameSession: vi.fn().mockResolvedValue({ data: {} }),
  apiRunWorkflow:   vi.fn().mockResolvedValue({ data: {} }),
}))

describe('Store state flows', () => {
  beforeEach(() => {
    act(() => {
      useWorkspaceStore.getState().clearChat()
      useWorkspaceStore.getState().setActiveView('COMPILER')
      useWorkspaceStore.getState().setStatus('idle')
      useWorkspaceStore.getState().setSimulationResult(null)
    })
  })

  it('addMessage + clearChat cycle works', () => {
    act(() => {
      useWorkspaceStore.getState().addMessage({
        role: 'user', content: 'test', type: 'message',
      })
    })
    expect(useWorkspaceStore.getState().chatMessages).toHaveLength(1)
    act(() => useWorkspaceStore.getState().clearChat())
    expect(useWorkspaceStore.getState().chatMessages).toHaveLength(0)
  })

  it('setSimulationResult stores result and can be cleared', () => {
    const mockResult = {
      dsl: 'rank(close)',
      is_metrics: { sharpe_ratio: 1.2 },
      overfitting_score: 0.1,
      is_overfit: false,
      ic_decay: {},
      pnl_is: [],
      pnl_oos: [],
    }
    act(() => useWorkspaceStore.getState().setSimulationResult(mockResult as any))
    expect(useWorkspaceStore.getState().simulationResult?.dsl).toBe('rank(close)')
    act(() => useWorkspaceStore.getState().setSimulationResult(null))
    expect(useWorkspaceStore.getState().simulationResult).toBeNull()
  })

  it('switching views preserves chatMessages', () => {
    act(() => {
      useWorkspaceStore.getState().addMessage({ role: 'user', content: 'message A', type: 'message' })
    })
    act(() => useWorkspaceStore.getState().setActiveView('DATASET'))
    act(() => useWorkspaceStore.getState().setActiveView('COMPILER'))
    act(() => useWorkspaceStore.getState().setActiveView('CHAT'))
    expect(useWorkspaceStore.getState().chatMessages).toHaveLength(1)
    expect(useWorkspaceStore.getState().chatMessages[0].content).toBe('message A')
  })

  it('setAnalysisTab changes active tab', () => {
    act(() => useWorkspaceStore.getState().setAnalysisTab('walkforward'))
    expect(useWorkspaceStore.getState().analysisTab).toBe('walkforward')
    act(() => useWorkspaceStore.getState().setAnalysisTab('backtest'))
  })

  it('session management: addSession prepends to sessions', () => {
    act(() => useWorkspaceStore.getState().setSessions([]))
    act(() => useWorkspaceStore.getState().addSession({
      id: 'sid-test', title: 'Test Session', createdAt: new Date().toISOString(),
    }))
    expect(useWorkspaceStore.getState().sessions).toHaveLength(1)
    expect(useWorkspaceStore.getState().sessions[0].id).toBe('sid-test')
  })

  it('removeSession removes from sessions list', () => {
    act(() => {
      useWorkspaceStore.getState().setSessions([
        { id: 'to-remove', title: 'Remove Me', createdAt: '' },
        { id: 'keep',      title: 'Keep',      createdAt: '' },
      ])
    })
    act(() => useWorkspaceStore.getState().removeSession('to-remove'))
    const ids = useWorkspaceStore.getState().sessions.map(s => s.id)
    expect(ids).not.toContain('to-remove')
    expect(ids).toContain('keep')
  })

  it('alphaHistory tracking', () => {
    const records = [{ id: 1, dsl: 'rank(close)', sharpe: 1.5 }]
    act(() => useWorkspaceStore.getState().setAlphaHistory(records as any))
    expect(useWorkspaceStore.getState().alphaHistory).toHaveLength(1)
  })

  it('walkForwardResult stores and clears', () => {
    const mockWF = {
      dsl: 'rank(close)', n_folds: 3, mean_oos_sharpe: 0.8,
      std_oos_sharpe: 0.1, min_oos_sharpe: 0.5,
      pct_positive: 1.0, mean_overfitting: 0.1, fold_reports: [],
    }
    act(() => useWorkspaceStore.getState().setWalkForwardResult(mockWF as any))
    expect(useWorkspaceStore.getState().walkForwardResult?.n_folds).toBe(3)
    act(() => useWorkspaceStore.getState().setWalkForwardResult(null))
    expect(useWorkspaceStore.getState().walkForwardResult).toBeNull()
  })
})
