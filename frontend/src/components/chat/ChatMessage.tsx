import { useWorkspaceStore } from '../../store/workspaceStore'
import ThoughtBlock from './ThoughtBlock'
import type { ChatMessage as Msg } from '../../types'
import { Code2, Bot, User } from 'lucide-react'

interface Props { message: Msg }

export default function ChatMessage({ message }: Props) {
  const { setEditorDsl, setActiveView } = useWorkspaceStore()
  const isUser = message.role === 'user'

  if (message.type === 'thought') return <ThoughtBlock content={message.content} />

  const handleUseDsl = () => {
    if (message.dsl) {
      setEditorDsl(message.dsl)
      setActiveView('COMPILER')
    }
  }

  return (
    <div className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}>
      {/* Avatar */}
      {!isUser && (
        <div className="w-7 h-7 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center shrink-0 mt-0.5">
          <Bot size={14} />
        </div>
      )}

      <div className={`max-w-[80%] flex flex-col gap-2 ${isUser ? 'items-end' : 'items-start'}`}>
        {/* Bubble */}
        <div className={`rounded-2xl px-4 py-2.5 text-sm leading-relaxed
          ${isUser
            ? 'bg-emerald-600 text-white rounded-tr-sm'
            : 'bg-slate-800 text-slate-200 rounded-tl-sm'}`}>
          {message.content}
        </div>

        {/* DSL chip */}
        {message.dsl && (
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs bg-slate-900 border border-slate-700 rounded-lg px-3 py-1 text-emerald-400">
              {message.dsl}
            </span>
            <button
              onClick={handleUseDsl}
              className="flex items-center gap-1 text-xs text-slate-500 hover:text-emerald-400 transition-colors"
              title="Open in Compiler"
            >
              <Code2 size={12} />
              Edit
            </button>
          </div>
        )}

        {/* Metrics chips */}
        {message.metrics && (
          <div className="flex flex-wrap gap-1.5">
            {Object.entries(message.metrics as Record<string, number>).slice(0, 4).map(([k, v]) =>
              v != null ? (
                <span key={k} className="text-xs bg-slate-800 border border-slate-700 rounded px-2 py-0.5 text-slate-400">
                  {k.replace(/_/g, ' ')}: <span className="text-sky-400 font-mono">{typeof v === 'number' ? v.toFixed(3) : v}</span>
                </span>
              ) : null
            )}
          </div>
        )}
      </div>

      {isUser && (
        <div className="w-7 h-7 rounded-full bg-slate-700 text-slate-400 flex items-center justify-center shrink-0 mt-0.5">
          <User size={14} />
        </div>
      )}
    </div>
  )
}
