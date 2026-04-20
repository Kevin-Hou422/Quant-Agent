import { PanelGroup, Panel, PanelResizeHandle } from 'react-resizable-panels'
import GlobalSidebar from './GlobalSidebar'
import LeftLedgerPane from './LeftLedgerPane'
import RightPane from './RightPane'
import ChatView from '../chat/ChatView'
import CompilerView from '../compiler/CompilerView'
import ConsoleOutput from '../compiler/ConsoleOutput'
import { useWorkspaceStore } from '../../store/workspaceStore'

/** Vertical drag handle between horizontal panels */
function VHandle() {
  return (
    <PanelResizeHandle className="relative w-1.5 bg-slate-800 hover:bg-emerald-600/60 active:bg-emerald-500 transition-colors duration-150 cursor-col-resize group">
      <div className="absolute inset-y-0 left-1/2 -translate-x-1/2 w-px bg-slate-700 group-hover:bg-emerald-500 transition-colors" />
      {/* Grip dots */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 flex flex-col gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
        {[0,1,2].map(i => (
          <div key={i} className="w-0.5 h-0.5 rounded-full bg-emerald-400" />
        ))}
      </div>
    </PanelResizeHandle>
  )
}

/** Horizontal drag handle between vertical panels */
function HHandle() {
  return (
    <PanelResizeHandle className="relative h-1 bg-slate-800 hover:bg-emerald-600/50 transition-colors duration-150 cursor-row-resize group">
      <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 h-0.5 bg-slate-700 group-hover:bg-emerald-500 transition-colors" />
    </PanelResizeHandle>
  )
}

export default function WorkspaceLayout() {
  const { activeView, ledgerOpen } = useWorkspaceStore()

  return (
    <div className="h-screen w-screen bg-slate-950 overflow-hidden" style={{ fontFamily: "'Inter','system-ui',sans-serif" }}>
      <PanelGroup direction="horizontal" className="h-full w-full">

        {/* ── Panel 1: Sidebar (collapsible, never squeezed to 0) ────────── */}
        <Panel defaultSize={15} minSize={12} collapsible={true}>
          <div className="relative h-full">
            {/* Alpha Ledger slide-out (absolute, overlays workspace) */}
            <div className={`
              absolute left-full top-0 h-full z-50
              transition-transform duration-200 ease-in-out
              ${ledgerOpen ? 'translate-x-0' : '-translate-x-full pointer-events-none'}
            `}>
              <LeftLedgerPane />
            </div>
            <GlobalSidebar />
          </div>
        </Panel>

        <VHandle />

        {/* ── Panel 2: Center workspace ──────────────────────────────── */}
        <Panel defaultSize={60} minSize={30}>
          {activeView === 'CHAT' ? (
            /* Chat: full height */
            <div className="h-full overflow-hidden">
              <ChatView />
            </div>
          ) : (
            /* Compiler: Monaco (top) + Console (bottom), resizable */
            <PanelGroup direction="vertical" className="h-full">
              <Panel defaultSize={65} minSize={30}>
                <div className="h-full overflow-hidden">
                  <CompilerView />
                </div>
              </Panel>
              <HHandle />
              <Panel defaultSize={35} minSize={12} maxSize={60}>
                <div className="h-full overflow-hidden">
                  <ConsoleOutput />
                </div>
              </Panel>
            </PanelGroup>
          )}
        </Panel>

        <VHandle />

        {/* ── Panel 3: Analysis dashboard (always visible) ───────────── */}
        <Panel defaultSize={35} minSize={22} maxSize={55}>
          <div className="h-full overflow-hidden">
            <RightPane />
          </div>
        </Panel>

      </PanelGroup>
    </div>
  )
}
