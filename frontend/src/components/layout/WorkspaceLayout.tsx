import { PanelGroup, Panel, PanelResizeHandle } from 'react-resizable-panels'
import GlobalSidebar from './GlobalSidebar'
import LeftLedgerPane from './LeftLedgerPane'
import RightPane from './RightPane'
import ChatView from '../chat/ChatView'
import CompilerView from '../compiler/CompilerView'
import ConsoleOutput from '../compiler/ConsoleOutput'
import { useWorkspaceStore } from '../../store/workspaceStore'

/** Vertical drag handle (between Editor and Console) */
function HResizeHandle() {
  return (
    <PanelResizeHandle className="
      h-1.5 w-full flex items-center justify-center cursor-row-resize
      bg-slate-800/80 hover:bg-emerald-600/30 transition-colors duration-150 group
    ">
      <div className="w-12 h-0.5 rounded-full bg-slate-700 group-hover:bg-emerald-500 transition-colors" />
    </PanelResizeHandle>
  )
}

/** Horizontal drag handle (between panels) */
function VResizeHandle() {
  return (
    <PanelResizeHandle className="
      w-1.5 h-full flex items-center justify-center cursor-col-resize
      bg-slate-800/80 hover:bg-emerald-600/30 transition-colors duration-150 group
    ">
      <div className="w-0.5 h-12 rounded-full bg-slate-700 group-hover:bg-emerald-500 transition-colors" />
    </PanelResizeHandle>
  )
}

export default function WorkspaceLayout() {
  const { activeView, ledgerOpen } = useWorkspaceStore()

  return (
    <div className="flex h-screen w-screen bg-slate-950 overflow-hidden font-sans">

      {/* ── Fixed Global Sidebar (not part of panel group) ───────────── */}
      <GlobalSidebar />

      {/* ── Alpha Ledger slide-out drawer (overlays, absolute) ──────── */}
      <div className={`
        absolute left-20 top-0 h-full z-40
        transition-transform duration-200 ease-in-out
        ${ledgerOpen ? 'translate-x-0' : '-translate-x-full'}
      `}>
        <LeftLedgerPane />
      </div>

      {/* ── Main 2-column resizable area ─────────────────────────────── */}
      <div className="flex-1 min-w-0 h-full">
        <PanelGroup direction="horizontal" className="h-full w-full">

          {/* Center pane */}
          <Panel defaultSize={65} minSize={35} className="flex flex-col min-h-0">
            {activeView === 'CHAT' ? (
              /* Chat mode: full height chat, no console */
              <div className="h-full overflow-hidden">
                <ChatView />
              </div>
            ) : (
              /* Compiler mode: Monaco (top) + Console (bottom), resizable */
              <PanelGroup direction="vertical" className="h-full">
                <Panel defaultSize={65} minSize={30} className="min-h-0 flex flex-col">
                  <CompilerView />
                </Panel>
                <HResizeHandle />
                <Panel defaultSize={35} minSize={12} maxSize={60} className="min-h-0 flex flex-col">
                  <ConsoleOutput />
                </Panel>
              </PanelGroup>
            )}
          </Panel>

          <VResizeHandle />

          {/* Right analysis pane */}
          <Panel defaultSize={35} minSize={20} maxSize={55} className="flex flex-col min-h-0">
            <RightPane />
          </Panel>

        </PanelGroup>
      </div>
    </div>
  )
}
