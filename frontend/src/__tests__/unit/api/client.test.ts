/**
 * client.test.ts — API client 导出函数验证测试
 */
import { describe, it, expect } from 'vitest'

describe('API Client - core function exports', () => {
  it('exports apiFetchDatasets', async () => {
    const { apiFetchDatasets } = await import('../../../api/client')
    expect(typeof apiFetchDatasets).toBe('function')
  })

  it('exports apiChat', async () => {
    const { apiChat } = await import('../../../api/client')
    expect(typeof apiChat).toBe('function')
  })

  it('exports apiSimulate', async () => {
    const { apiSimulate } = await import('../../../api/client')
    expect(typeof apiSimulate).toBe('function')
  })

  it('exports apiBacktest (single DSL backtest)', async () => {
    const { apiBacktest } = await import('../../../api/client')
    expect(typeof apiBacktest).toBe('function')
  })

  it('exports apiWalkForwardBacktest', async () => {
    const { apiWalkForwardBacktest } = await import('../../../api/client')
    expect(typeof apiWalkForwardBacktest).toBe('function')
  })

  it('exports apiFetchDatasetHealth', async () => {
    const { apiFetchDatasetHealth } = await import('../../../api/client')
    expect(typeof apiFetchDatasetHealth).toBe('function')
  })

  it('exports apiWorkflowGenerate', async () => {
    const { apiWorkflowGenerate } = await import('../../../api/client')
    expect(typeof apiWorkflowGenerate).toBe('function')
  })

  it('exports apiWorkflowOptimize', async () => {
    const { apiWorkflowOptimize } = await import('../../../api/client')
    expect(typeof apiWorkflowOptimize).toBe('function')
  })
})

describe('API Client - session management exports', () => {
  it('exports apiCreateSession', async () => {
    const { apiCreateSession } = await import('../../../api/client')
    expect(typeof apiCreateSession).toBe('function')
  })

  it('exports apiListSessions', async () => {
    const { apiListSessions } = await import('../../../api/client')
    expect(typeof apiListSessions).toBe('function')
  })

  it('exports apiDeleteSession', async () => {
    const { apiDeleteSession } = await import('../../../api/client')
    expect(typeof apiDeleteSession).toBe('function')
  })

  it('exports apiGetSession', async () => {
    const { apiGetSession } = await import('../../../api/client')
    expect(typeof apiGetSession).toBe('function')
  })
})

describe('API Client - streaming exports', () => {
  it('exports streamChat', async () => {
    const { streamChat } = await import('../../../api/client')
    expect(typeof streamChat).toBe('function')
  })

  it('exports streamWorkflowGenerate', async () => {
    const { streamWorkflowGenerate } = await import('../../../api/client')
    expect(typeof streamWorkflowGenerate).toBe('function')
  })
})

describe('API Client - function signatures', () => {
  it('apiChat accepts 2 parameters (message, sessionId)', async () => {
    const { apiChat } = await import('../../../api/client')
    expect(apiChat.length).toBe(2)
  })

  it('apiFetchDatasets takes no required arguments', async () => {
    const { apiFetchDatasets } = await import('../../../api/client')
    expect(apiFetchDatasets.length).toBe(0)
  })

  it('apiSimulate accepts dsl and config as first params', async () => {
    const { apiSimulate } = await import('../../../api/client')
    expect(apiSimulate.length).toBeGreaterThanOrEqual(2)
  })
})
