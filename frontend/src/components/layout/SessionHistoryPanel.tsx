import { useEffect, useRef, useState } from 'react'
import { useWorkspaceStore } from '../../store/workspaceStore'
import { useQuantWorkspace } from '../../hooks/useQuantWorkspace'
import { MessageSquare, Plus, Pencil, Trash2, Check, X } from 'lucide-react'
import type { ChatSession } from '../../types'

// ── Single session row ────────────────────────────────────────────────────────

function SessionItem({
  session,
  active,
  onSelect,
  onRename,
  onDelete,
}: {
  session:  ChatSession
  active:   boolean
  onSelect: () => void
  onRename: (title: string) => void
  onDelete: () => void
}) {
  const [editing, setEditing]   = useState(false)
  const [draft,   setDraft]     = useState(session.title)
  const inputRef                = useRef<HTMLInputElement>(null)

  const date = session.createdAt
    ? new Date(session.createdAt).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
    : ''

  // Focus input when entering edit mode
  useEffect(() => {
    if (editing) {
      setDraft(session.title)
      setTimeout(() => inputRef.current?.select(), 0)
    }
  }, [editing, session.title])

  const commitRename = () => {
    const trimmed = draft.trim()
    if (trimmed && trimmed !== session.title) onRename(trimmed)
    setEditing(false)
  }

  const cancelRename = () => {
    setDraft(session.title)
    setEditing(false)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter')  { e.preventDefault(); commitRename() }
    if (e.key === 'Escape') { e.preventDefault(); cancelRename() }
  }

  return (
    <div
      className={`
        group relative flex items-start px-3 py-2.5 border-b border-slate-800/60
        transition-colors cursor-pointer select-none
        ${active
          ? 'bg-slate-700/60 border-l-2 border-l-emerald-500'
          : 'hover:bg-slate-800/50 border-l-2 border-l-transparent'}
      `}
      onClick={() => { if (!editing) onSelect() }}
    >
      <MessageSquare
        size={11}
        className={`mt-0.5 mr-2 shrink-0 ${active ? 'text-emerald-400' : 'text-slate-600'}`}
      />

      <div className="min-w-0 flex-1 pr-1">
        {editing ? (
          <div className="flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
            <input
              ref={inputRef}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={handleKeyDown}
              className="flex-1 min-w-0 text-[11px] bg-slate-700 border border-emerald-500/60
                         rounded px-1.5 py-0.5 text-slate-200 outline-none"
            />
            <button onClick={commitRename} title="Save" className="text-emerald-400 hover:text-emerald-300 shrink-0">
              <Check size={11} />
            </button>
            <button onClick={cancelRename} title="Cancel" className="text-slate-500 hover:text-slate-300 shrink-0">
              <X size={11} />
            </button>
          </div>
        ) : (
          <>
            <p className={`text-[11px] truncate leading-tight ${active ? 'text-slate-200' : 'text-slate-400'}`}>
              {session.title}
            </p>
            {date && <p className="text-[10px] text-slate-600 mt-0.5">{date}</p>}
          </>
        )}
      </div>

      {/* Action buttons — shown on hover when not editing */}
      {!editing && (
        <div
          className="flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity shrink-0"
          onClick={(e) => e.stopPropagation()}
        >
          <button
            onClick={() => setEditing(true)}
            title="Rename"
            className="p-1 text-slate-600 hover:text-slate-300 rounded transition-colors"
          >
            <Pencil size={10} />
          </button>
          <button
            onClick={onDelete}
            title="Delete"
            className="p-1 text-slate-600 hover:text-rose-400 rounded transition-colors"
          >
            <Trash2 size={10} />
          </button>
        </div>
      )}
    </div>
  )
}

// ── Panel ─────────────────────────────────────────────────────────────────────

export default function SessionHistoryPanel() {
  const { sessionId, sessions } = useWorkspaceStore()
  const { loadSessions, newSession, switchSession, renameSession, deleteSession } =
    useQuantWorkspace()

  // Refresh the sidebar list whenever the panel becomes visible
  useEffect(() => { loadSessions() }, [])  // eslint-disable-line react-hooks/exhaustive-deps

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
              onSelect={() => switchSession(s.id)}
              onRename={(title) => renameSession(s.id, title)}
              onDelete={() => deleteSession(s.id)}
            />
          ))
        )}
      </div>
    </aside>
  )
}
