import React, { useRef, useEffect } from 'react'
import { useWorkspaceStore } from '../../store/workspaceStore'
import { useQuantWorkspace } from '../../hooks/useQuantWorkspace'
import ChatMessage from './ChatMessage'
import { Send, Trash2, Loader2 } from 'lucide-react'

// Status-aware thinking label
const THINKING_LABELS: Record<string, string> = {
  optimizing:  'Running Optuna optimization…',
  backtesting: 'Backtesting IS + OOS…',
}

function ThinkingIndicator({ status }: { status: string }) {
  const label = THINKING_LABELS[status] ?? 'Agent thinking…'

  const dotColor: Record<string, string> = {
    optimizing:  'bg-violet-500',
    backtesting: 'bg-amber-400',
  }
  const color = dotColor[status] ?? 'bg-emerald-500'

  return (
    <div className="flex items-center gap-2.5 py-1">
      <span className="inline-flex gap-1 shrink-0">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className={`w-1.5 h-1.5 ${color} rounded-full animate-bounce`}
            style={{ animationDelay: `${i * 0.15}s` }}
          />
        ))}
      </span>
      <span className="text-xs text-slate-400 italic">{label}</span>
    </div>
  )
}

export default function ChatView() {
  const { chatMessages, clearChat, status } = useWorkspaceStore()
  const { sendChat } = useQuantWorkspace()
  const [input, setInput] = React.useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatMessages, status])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return
    sendChat(input.trim())
    setInput('')
  }

  const isLoading = status === 'optimizing' || status === 'backtesting'

  return (
    <div className="flex flex-col h-full bg-slate-950">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800 bg-slate-900 shrink-0">
        <div>
          <h2 className="text-sm font-semibold text-slate-200">AI Quant Agent</h2>
          <p className="text-xs text-slate-500">
            LangChain · GPT-4o · Optuna · Anti-Overfitting
            {status !== 'idle' && status !== 'ready' && status !== 'error' && (
              <span className="ml-2 text-amber-400 animate-pulse">● {status}</span>
            )}
          </p>
        </div>
        <button
          onClick={clearChat}
          className="text-slate-600 hover:text-slate-400 transition-colors"
          title="Clear conversation"
        >
          <Trash2 size={16} />
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {chatMessages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center text-slate-600 gap-3">
            <p className="text-3xl">🧠</p>
            <p className="text-sm font-medium text-slate-500">Quant Research Assistant</p>
            <div className="text-xs max-w-xs space-y-1 text-slate-600">
              <p>Try: <span className="text-slate-400">"Generate an alpha based on volume spikes"</span></p>
              <p>Or: <span className="text-slate-400">"Optimize ts_mean(close, 10)"</span></p>
            </div>
          </div>
        )}

        {chatMessages.map((msg) => (
          <ChatMessage key={msg.id} message={msg} />
        ))}

        {isLoading && <ThinkingIndicator status={status} />}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="px-4 pb-4 shrink-0">
        <div className={`
          flex gap-2 border rounded-xl px-4 py-2.5 transition-colors
          ${isLoading
            ? 'bg-slate-800/50 border-slate-700/50'
            : 'bg-slate-800 border-slate-700 focus-within:border-emerald-600'}
        `}>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={isLoading
              ? 'Processing…'
              : 'Describe a market hypothesis or paste a DSL to optimize…'}
            disabled={isLoading}
            className="flex-1 bg-transparent outline-none text-sm text-slate-200 placeholder-slate-600 disabled:cursor-not-allowed"
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="text-emerald-500 hover:text-emerald-400 disabled:opacity-30 transition-colors shrink-0"
          >
            {isLoading
              ? <Loader2 size={18} className="animate-spin text-amber-400" />
              : <Send size={18} />}
          </button>
        </div>
      </form>
    </div>
  )
}
