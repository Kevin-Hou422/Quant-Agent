import React from 'react'
import GlobalSidebar from './GlobalSidebar'
import LeftLedgerPane from './LeftLedgerPane'
import RightPane from './RightPane'
import ChatView from '../chat/ChatView'
import CompilerView from '../compiler/CompilerView'
import { useWorkspaceStore } from '../../store/workspaceStore'
import { useQuantWorkspace } from '../../hooks/useQuantWorkspace'

export default function WorkspaceLayout() {
  const { activeView } = useWorkspaceStore()
  const { loadHistory } = useQuantWorkspace()

  React.useEffect(() => { loadHistory() }, [])

  return (
    <div className="flex h-screen w-screen bg-slate-950 overflow-hidden">
      <GlobalSidebar />
      <LeftLedgerPane />

      {/* Center pane */}
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden border-x border-slate-800">
        {activeView === 'CHAT' ? <ChatView /> : <CompilerView />}
      </main>

      <RightPane />
    </div>
  )
}
