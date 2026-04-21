import { useEffect } from 'react'
import { useWorkspaceStore } from '../../store/workspaceStore'
import { useQuantWorkspace } from '../../hooks/useQuantWorkspace'
import { MessageSquare, Plus } from 'lucide-react'
import type { ChatSession } from '../../types'

function SessionItem({
  session,
  active,
  onClick,
}: {
  session: ChatSession
  active:  boolean
  onClick: () => void
}) {
  const date = session.createdAt
    ? new Date(session.createdAt).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
    : ''

  return (
    <button
      onClick={onClick}
      className={`
        w-full text-left px-3 py-2.5 border-b border-slate-800/60 transition-colors
        ${active
          ? 'bg-slate-700/60 border-l-2 border-l-emerald-500'
          : 'hover:bg-slate-800/50 border-l-2 border-l-transparent'}
      `}
    >
      <div className="flex items-start gap-2">
        <MessageSquare
          size={11}
          className={`mt-0.5 shrink-0 ${active ? 'text-emerald-400' : 'text-slate-600'}`}
        />
        <div className="min-w-0 flex-1">
          <p className={`text-[11px] truncate leading-tight ${active ? 'text-slate-200' : 'text-slate-400'}`}>
            {session.title}
          </p>
          {date && (
            <p className="text-[10px] text-slate-600 mt-0.5">{date}</p>
          )}
        </div>
      </div>
    </button>
  )
}

export default function SessionHistoryPanel() {
  const { sessionId, sessions } = useWorkspaceStore()
  const { newSession, switchSession, loadSessions } = useQuantWorkspace()

  useEffect(() => { loadSessions() }, [])

  return (
    <aside className="h-full flex flex-col bg-slate-900 border-r border-slate-700">
      {/* New session button */}
      <div className="px-2 py-2.5 border-b border-slate-800 shrink-0">
        <button
          onClick={newSession}
          className="
            w-full flex items-center justify-center gap-1.5 py-2 rounded-lg
            bg-emerald-600 hover:bg-emerald-500 active:bg-emerald-700
            text-white text-[11px] font-medium transition-colors
          "
        >
          <Plus size={12} strokeWidth={2.5} />
          New Chat
        </button>
      </div>

      {/* Session list */}
      <div className="flex-1 overflow-y-auto">
        {sessions.length === 0 ? (
          <p className="px-3 py-5 text-[11px] text-slate-600 text-center leading-relaxed">
            No sessions yet.<br />Click "New Chat" to start.
          </p>
        ) : (
          sessions.map((s) => (
            <SessionItem
              key={s.id}
              session={s}
              active={s.id === sessionId}
              onClick={() => switchSession(s.id)}
            />
          ))
        )}
      </div>
    </aside>
  )
}
