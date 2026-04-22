import { useWorkspaceStore } from '../../store/workspaceStore'
import ThoughtBlock from './ThoughtBlock'
import type { ChatMessage as Msg } from '../../types'
import { Code2, Bot, User } from 'lucide-react'

interface Props { message: Msg }

// Colour a tagged prefix line like "[GP] ...", "[Optuna] ..." etc.
function tagClass(line: string): string {
  if (line.startsWith('[GP]'))       return 'text-violet-400'
  if (line.startsWith('[Optuna]'))   return 'text-violet-300'
  if (line.startsWith('[Diagnose]')) return 'text-sky-400'
  if (line.startsWith('[Result]'))   return 'text-emerald-400'
  if (line.startsWith('[Workflow'))  return 'text-slate-400'
  if (line.startsWith('[ERROR]'))    return 'text-rose-400'
  if (line.startsWith('[WARN]'))     return 'text-amber-400'
  if (line.startsWith('⚠'))         return 'text-amber-400'
  if (line.startsWith('✓'))         return 'text-emerald-400'
  return ''
}

// Render content that may contain tagged GP lines and line-breaks
function StreamContent({ text, isStreaming }: { text: string; isStreaming?: boolean }) {
  const lines = text.split('\n')
  return (
    <span className="whitespace-pre-wrap break-words font-mono text-xs leading-5">
      {lines.map((line, i) => (
        <span key={i} className={tagClass(line) || undefined}>
          {line}
          {i < lines.length - 1 && '\n'}
        </span>
      ))}
      {isStreaming && (
        <span className="inline-block w-[6px] h-[13px] bg-emerald-400 ml-0.5 align-[-2px] animate-pulse" />
      )}
    </span>
  )
}

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

  // Streaming assistant messages use a monospace log-style bubble
  const isGpStream = !isUser && message.isStreaming

  return (
    <div className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}>
      {/* Avatar */}
      {!isUser && (
        <div className={`w-7 h-7 rounded-full flex items-center justify-center shrink-0 mt-0.5
          ${isGpStream ? 'bg-violet-500/20 text-violet-400' : 'bg-emerald-500/20 text-emerald-400'}`}>
          <Bot size={14} />
        </div>
      )}

      <div className={`max-w-[85%] flex flex-col gap-2 ${isUser ? 'items-end' : 'items-start'}`}>
        {/* Bubble */}
        <div className={`rounded-2xl px-4 py-2.5 text-sm leading-relaxed
          ${isUser
            ? 'bg-emerald-600 text-white rounded-tr-sm'
            : message.isStreaming || message.content.includes('[GP]') || message.content.includes('[Workflow')
              ? 'bg-slate-900 border border-slate-700/60 text-slate-300 rounded-tl-sm'
              : 'bg-slate-800 text-slate-200 rounded-tl-sm'}`}>
          {!isUser && (message.isStreaming || message.content.includes('[GP]') || message.content.includes('[Workflow'))
            ? <StreamContent text={message.content} isStreaming={message.isStreaming} />
            : message.content
          }
        </div>

        {/* DSL chip — only shown after streaming finishes */}
        {message.dsl && !message.isStreaming && (
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
        {message.metrics && !message.isStreaming && (
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
