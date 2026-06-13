/**
 * workspaceStore.test.ts — Zustand store 状态管理单元测试
 */
import { describe, it, expect, beforeEach } from 'vitest'
import { act } from '@testing-library/react'
import { useWorkspaceStore } from '../../../store/workspaceStore'

describe('WorkspaceStore - activeView', () => {
  it('default activeView is COMPILER', () => {
    expect(useWorkspaceStore.getState().activeView).toBe('COMPILER')
  })

  it('setActiveView updates activeView', () => {
    act(() => useWorkspaceStore.getState().setActiveView('CHAT'))
    expect(useWorkspaceStore.getState().activeView).toBe('CHAT')
    act(() => useWorkspaceStore.getState().setActiveView('COMPILER'))
  })

  it('setActiveView to DATASET records prevView', () => {
    act(() => useWorkspaceStore.getState().setActiveView('COMPILER'))
    act(() => useWorkspaceStore.getState().setActiveView('DATASET'))
    expect(useWorkspaceStore.getState().activeView).toBe('DATASET')
    expect(useWorkspaceStore.getState().prevView).toBe('COMPILER')
    act(() => useWorkspaceStore.getState().setActiveView('COMPILER'))
  })
})

describe('WorkspaceStore - chatMessages', () => {
  beforeEach(() => {
    act(() => useWorkspaceStore.getState().clearChat())
  })

  it('initial chatMessages is empty after clear', () => {
    expect(useWorkspaceStore.getState().chatMessages).toHaveLength(0)
  })

  it('addMessage appends a message with id and timestamp', () => {
    act(() => useWorkspaceStore.getState().addMessage({ role: 'user', content: 'Hello', type: 'message' }))
    const msgs = useWorkspaceStore.getState().chatMessages
    expect(msgs).toHaveLength(1)
    expect(msgs[0].content).toBe('Hello')
    expect(msgs[0].role).toBe('user')
    expect(msgs[0].id).toBeTruthy()
    expect(msgs[0].timestamp).toBeGreaterThan(0)
  })

  it('addMessage appends multiple messages in order', () => {
    act(() => {
      useWorkspaceStore.getState().addMessage({ role: 'user',      content: 'First',  type: 'message' })
      useWorkspaceStore.getState().addMessage({ role: 'assistant', content: 'Second', type: 'message' })
    })
    expect(useWorkspaceStore.getState().chatMessages).toHaveLength(2)
  })

  it('clearChat empties chatMessages', () => {
    act(() => useWorkspaceStore.getState().addMessage({ role: 'user', content: 'test', type: 'message' }))
    act(() => useWorkspaceStore.getState().clearChat())
    expect(useWorkspaceStore.getState().chatMessages).toHaveLength(0)
  })

  it('setMessages replaces chatMessages', () => {
    const msgs = [
      { id: '1', role: 'user' as const, content: 'a', timestamp: 1, type: 'message' as const },
      { id: '2', role: 'assistant' as const, content: 'b', timestamp: 2, type: 'message' as const },
    ]
    act(() => useWorkspaceStore.getState().setMessages(msgs))
    expect(useWorkspaceStore.getState().chatMessages).toHaveLength(2)
    expect(useWorkspaceStore.getState().chatMessages[0].content).toBe('a')
  })
})

describe('WorkspaceStore - streaming messages', () => {
  beforeEach(() => {
    act(() => useWorkspaceStore.getState().clearChat())
  })

  it('startStreamingMessage adds empty assistant message', () => {
    act(() => useWorkspaceStore.getState().startStreamingMessage('stream-1'))
    const msgs = useWorkspaceStore.getState().chatMessages
    expect(msgs).toHaveLength(1)
    expect(msgs[0].id).toBe('stream-1')
    expect(msgs[0].isStreaming).toBe(true)
    expect(msgs[0].content).toBe('')
  })

  it('appendToStreamingMessage concatenates content', () => {
    act(() => useWorkspaceStore.getState().startStreamingMessage('s1'))
    act(() => useWorkspaceStore.getState().appendToStreamingMessage('s1', 'Hello'))
    act(() => useWorkspaceStore.getState().appendToStreamingMessage('s1', ' World'))
    expect(useWorkspaceStore.getState().chatMessages[0].content).toBe('Hello World')
  })

  it('finalizeStreamingMessage sets isStreaming to false', () => {
    act(() => useWorkspaceStore.getState().startStreamingMessage('s2'))
    act(() => useWorkspaceStore.getState().finalizeStreamingMessage('s2'))
    expect(useWorkspaceStore.getState().chatMessages[0].isStreaming).toBe(false)
  })
})

describe('WorkspaceStore - editorTabs', () => {
  it('default editorTabs has one tab', () => {
    expect(useWorkspaceStore.getState().editorTabs.length).toBeGreaterThanOrEqual(1)
  })

  it('newEmptyTab adds a new tab', () => {
    const initial = useWorkspaceStore.getState().editorTabs.length
    act(() => useWorkspaceStore.getState().newEmptyTab())
    expect(useWorkspaceStore.getState().editorTabs.length).toBeGreaterThan(initial)
  })

  it('setEditorDsl updates editorDsl', () => {
    act(() => useWorkspaceStore.getState().setEditorDsl('rank(close)'))
    expect(useWorkspaceStore.getState().editorDsl).toBe('rank(close)')
  })
})

describe('WorkspaceStore - status', () => {
  it('setStatus updates status', () => {
    act(() => useWorkspaceStore.getState().setStatus('backtesting'))
    expect(useWorkspaceStore.getState().status).toBe('backtesting')
    act(() => useWorkspaceStore.getState().setStatus('idle'))
    expect(useWorkspaceStore.getState().status).toBe('idle')
  })
})

describe('WorkspaceStore - consoleLogs', () => {
  it('appendLog adds a line', () => {
    act(() => useWorkspaceStore.getState().clearLogs())
    act(() => useWorkspaceStore.getState().appendLog('test log line'))
    expect(useWorkspaceStore.getState().consoleLogs).toContain('test log line')
  })

  it('clearLogs empties consoleLogs', () => {
    act(() => useWorkspaceStore.getState().appendLog('line'))
    act(() => useWorkspaceStore.getState().clearLogs())
    expect(useWorkspaceStore.getState().consoleLogs).toHaveLength(0)
  })
})

describe('WorkspaceStore - simulationResult', () => {
  it('setSimulationResult stores result and can be cleared', () => {
    const mockResult = {
      dsl: 'rank(close)',
      is_metrics: { sharpe_ratio: 1.5 },
      overfitting_score: 0.1,
      is_overfit: false,
      ic_decay: {},
      pnl_is: [0.01, -0.02],
      pnl_oos: [],
    }
    act(() => useWorkspaceStore.getState().setSimulationResult(mockResult as any))
    expect(useWorkspaceStore.getState().simulationResult?.dsl).toBe('rank(close)')
    act(() => useWorkspaceStore.getState().setSimulationResult(null))
    expect(useWorkspaceStore.getState().simulationResult).toBeNull()
  })
})

describe('WorkspaceStore - ledger', () => {
  it('toggleLedger flips ledgerOpen', () => {
    act(() => useWorkspaceStore.getState().setLedgerOpen(false))
    act(() => useWorkspaceStore.getState().toggleLedger())
    expect(useWorkspaceStore.getState().ledgerOpen).toBe(true)
    act(() => useWorkspaceStore.getState().toggleLedger())
    expect(useWorkspaceStore.getState().ledgerOpen).toBe(false)
  })

  it('setLedgerOpen sets value directly', () => {
    act(() => useWorkspaceStore.getState().setLedgerOpen(true))
    expect(useWorkspaceStore.getState().ledgerOpen).toBe(true)
    act(() => useWorkspaceStore.getState().setLedgerOpen(false))
  })
})

describe('WorkspaceStore - sessions', () => {
  it('addSession prepends to sessions', () => {
    act(() => useWorkspaceStore.getState().setSessions([]))
    act(() => useWorkspaceStore.getState().addSession({ id: 'sid1', title: 'S1', createdAt: '' }))
    expect(useWorkspaceStore.getState().sessions[0].id).toBe('sid1')
  })

  it('removeSession filters out by id', () => {
    act(() => useWorkspaceStore.getState().setSessions([
      { id: 'a', title: 'A', createdAt: '' },
      { id: 'b', title: 'B', createdAt: '' },
    ]))
    act(() => useWorkspaceStore.getState().removeSession('a'))
    expect(useWorkspaceStore.getState().sessions.map(s => s.id)).not.toContain('a')
  })

  it('updateSessionTitle changes title', () => {
    act(() => useWorkspaceStore.getState().setSessions([{ id: 'x', title: 'Old', createdAt: '' }]))
    act(() => useWorkspaceStore.getState().updateSessionTitle('x', 'New'))
    const sess = useWorkspaceStore.getState().sessions.find(s => s.id === 'x')
    expect(sess?.title).toBe('New')
  })
})
