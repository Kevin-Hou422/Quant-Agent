import React, { useRef, useEffect } from 'react'
import { useWorkspaceStore } from '../../store/workspaceStore'
import { useQuantWorkspace } from '../../hooks/useQuantWorkspace'
import ChatMessage from './ChatMessage'
import { Send, Trash2 } from 'lucide-react'

export default function ChatView() {
  const { chatMessages, clearChat, status } = useWorkspaceStore()
  const { sendChat } = useQuantWorkspace()
  const [input, setInput] = React.useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatMessages])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || status === 'optimizing') return
    sendChat(input.trim())
    setInput('')
  }

  const isLoading = status === 'optimizing' || status === 'backtesting'

  return (
    <div className="flex flex-col h-full bg-slate-950">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800 bg-slate-900">
        <div>
          <h2 className="text-sm font-semibold text-slate-200">AI Quant Agent</h2>
          <p className="text-xs text-slate-500">LangChain · GPT-4o · Optuna · Anti-Overfitting</p>
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
            <p className="text-2xl">🧠</p>
            <p className="text-sm font-medium text-slate-500">Quant Research Assistant</p>
            <p className="text-xs max-w-xs">
              Try: <em>"Generate an alpha based on volume spikes"</em><br/>
              or: <em>"Optimize ts_mean(close, 10)"</em>
            </p>
          </div>
        )}
        {chatMessages.map((msg) => (
          <ChatMessage key={msg.id} message={msg} />
        ))}
        {isLoading && (
          <div className="flex items-center gap-2 text-slate-500 text-sm">
            <span className="inline-flex gap-1">
              {[0,1,2].map(i => (
                <span key={i} className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-bounce"
                  style={{ animationDelay: `${i * 0.15}s` }} />
              ))}
            </span>
            <span>Agent thinking…</span>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="px-4 pb-4">
        <div className="flex gap-2 bg-slate-800 border border-slate-700 rounded-xl px-4 py-2">
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Describe a market hypothesis or paste a DSL to optimize…"
            disabled={isLoading}
            className="flex-1 bg-transparent outline-none text-sm text-slate-200 placeholder-slate-600 disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="text-emerald-500 hover:text-emerald-400 disabled:opacity-30 transition-colors"
          >
            <Send size={18} />
          </button>
        </div>
      </form>
    </div>
  )
}
