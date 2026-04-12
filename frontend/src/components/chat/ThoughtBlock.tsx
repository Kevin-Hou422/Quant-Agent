import { useState } from 'react'
import { ChevronDown, ChevronRight, Brain } from 'lucide-react'

interface Props { content: string }

export default function ThoughtBlock({ content }: Props) {
  const [open, setOpen] = useState(false)
  return (
    <div className="border border-amber-900/40 bg-amber-950/20 rounded-lg overflow-hidden text-xs">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-2 px-3 py-2 text-amber-400 hover:bg-amber-950/30 transition-colors"
      >
        <Brain size={13} />
        <span className="font-medium">Agent Thought</span>
        {open ? <ChevronDown size={13} className="ml-auto" /> : <ChevronRight size={13} className="ml-auto" />}
      </button>
      {open && (
        <div className="px-3 pb-2 text-amber-300/70 font-mono leading-relaxed whitespace-pre-wrap">
          {content}
        </div>
      )}
    </div>
  )
}
