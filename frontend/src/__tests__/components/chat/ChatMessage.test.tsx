/**
 * ChatMessage.test.tsx — ChatMessage 组件测试
 */
import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'

vi.mock('@monaco-editor/react', () => ({
  default: () => <div data-testid="monaco-editor" />,
}))

import ChatMessage from '../../../components/chat/ChatMessage'

const BASE_USER_MSG = {
  id: '1',
  role: 'user' as const,
  content: 'Hello agent',
  timestamp: Date.now(),
  type: 'message' as const,
}

const BASE_ASSISTANT_MSG = {
  id: '2',
  role: 'assistant' as const,
  content: 'Here is your alpha: rank(close)',
  timestamp: Date.now(),
  type: 'message' as const,
}

describe('ChatMessage - user message', () => {
  it('renders user message content', () => {
    render(<ChatMessage message={BASE_USER_MSG} />)
    expect(screen.getByText('Hello agent')).toBeInTheDocument()
  })

  it('does not render DSL block for user message without dsl', () => {
    render(<ChatMessage message={BASE_USER_MSG} />)
    expect(screen.queryByTestId('monaco-editor')).not.toBeInTheDocument()
  })
})

describe('ChatMessage - assistant message', () => {
  it('renders assistant message content', () => {
    render(<ChatMessage message={BASE_ASSISTANT_MSG} />)
    expect(screen.getByText(/Here is your alpha/i)).toBeInTheDocument()
  })

  it('renders DSL when dsl field is provided', () => {
    const msgWithDsl = { ...BASE_ASSISTANT_MSG, dsl: 'rank(close)' }
    render(<ChatMessage message={msgWithDsl} />)
    const dslElement = screen.queryByTestId('monaco-editor') || screen.queryByText('rank(close)')
    expect(dslElement).toBeInTheDocument()
  })

  it('does not render DSL block when dsl is null', () => {
    const msgNoDsl = { ...BASE_ASSISTANT_MSG, dsl: null }
    render(<ChatMessage message={msgNoDsl} />)
    expect(screen.queryByTestId('monaco-editor')).not.toBeInTheDocument()
  })
})

describe('ChatMessage - streaming state', () => {
  it('renders streaming message with content', () => {
    const streamingMsg = {
      ...BASE_ASSISTANT_MSG,
      content: 'Loading...',
      isStreaming: true,
    }
    render(<ChatMessage message={streamingMsg} />)
    expect(screen.getByText('Loading...')).toBeInTheDocument()
  })
})
