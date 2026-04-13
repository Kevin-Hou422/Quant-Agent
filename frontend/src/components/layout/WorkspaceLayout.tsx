import { Group as PanelGroup, Panel, Separator as PanelResizeHandle } from 'react-resizable-panels'
import GlobalSidebar from './GlobalSidebar'
import LeftLedgerPane from './LeftLedgerPane'
import RightPane from './RightPane'
import ChatView from '../chat/ChatView'
import CompilerView from '../compiler/CompilerView'
import { useWorkspaceStore } from '../../store/workspaceStore'

function ResizeHandle({ vertical = false }: { vertical?: boolean }) {
  return (
    <PanelResizeHandle
      className={`
        group flex items-center justify-center
        ${vertical ? 'h-1.5 w-full cursor-row-resize' : 'w-1.5 h-full cursor-col-resize'}
        bg-slate-800 hover:bg-emerald-600/40 transition-colors duration-150
      `}
    />
  )
}

export default function WorkspaceLayout() {
  const { activeView, ledgerOpen } = useWorkspaceStore()

  return (
    <div className="flex h-screen w-screen bg-slate-950 overflow-hidden">
      {/* Fixed sidebar */}
      <GlobalSidebar />

      {/* Ledger slide-out drawer */}
      <div
        className={`
          absolute left-20 top-0 h-full z-30 transition-transform duration-200 ease-in-out
          ${ledgerOpen ? 'translate-x-0' : '-translate-x-full'}
        `}
      >
        <LeftLedgerPane />
      </div>

      {/* Main resizable area */}
      <div className="flex-1 min-w-0">
        <PanelGroup orientation="horizontal" className="h-full">
          <Panel defaultSize={65} minSize={40}>
            <div className="h-full overflow-hidden">
              {activeView === 'CHAT' ? <ChatView /> : <CompilerView />}
            </div>
          </Panel>

          <ResizeHandle />

          <Panel defaultSize={35} minSize={20} maxSize={55}>
            <RightPane />
          </Panel>
        </PanelGroup>
      </div>
    </div>
  )
}
