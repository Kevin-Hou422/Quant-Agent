import { useEffect } from 'react'
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels'
import GlobalSidebar from './GlobalSidebar'
import LeftLedgerPane from './LeftLedgerPane'
import SessionHistoryPanel from './SessionHistoryPanel'
import RightPane from './RightPane'
import ChatView from '../chat/ChatView'
import CompilerView from '../compiler/CompilerView'
import ConsoleOutput from '../compiler/ConsoleOutput'
import DatasetView from '../dataset/DatasetView'
import { useWorkspaceStore } from '../../store/workspaceStore'
import { useQuantWorkspace } from '../../hooks/useQuantWorkspace'

/** Horizontal drag handle — Monaco ↔ Console (only resizable boundary) */
function HHandle() {
  return (
    <PanelResizeHandle className="
      relative h-1.5 bg-slate-800
      hover:bg-emerald-600/60 active:bg-emerald-500
      transition-colors duration-150 cursor-row-resize group
    ">
      <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 h-px bg-slate-700 group-hover:bg-emerald-500 transition-colors" />
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 flex flex-row gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
        {[0, 1, 2].map((i) => (
          <div key={i} className="w-0.5 h-0.5 rounded-full bg-emerald-400" />
        ))}
      </div>
    </PanelResizeHandle>
  )
}

export default function WorkspaceLayout() {
  const { activeView, ledgerOpen } = useWorkspaceStore()
  const { initSessions } = useQuantWorkspace()

  // Run once on app mount: restore or create the active session
  useEffect(() => { initSessions() }, [])  // eslint-disable-line react-hooks/exhaustive-deps

  const inDataset  = activeView === 'DATASET'
  const inChat     = activeView === 'CHAT'
  const inCompiler = activeView === 'COMPILER'

  return (
    <div
      className="h-screen w-screen flex bg-slate-950 overflow-hidden"
      style={{ fontFamily: "'Inter','system-ui',sans-serif" }}
    >
      {/* ── Col 1: Icon toolbar (always visible) ─────────────────────── */}
      <div style={{ width: 64, minWidth: 64, flexShrink: 0 }} className="h-full">
        <GlobalSidebar />
      </div>

      {/* ── Dataset view: full-width main area, no side panels ────────── */}
      {inDataset && (
        <div className="flex-1 min-w-0 h-full overflow-hidden">
          <DatasetView />
        </div>
      )}

      {/* ── Chat / Compiler layout ────────────────────────────────────── */}
      {!inDataset && (
        <>
          {/*
           * Col 2 (conditional):
           *  CHAT mode     → Session history list  (192px)
           *  COMPILER mode → Alpha Ledger panel    (240px), only when ledgerOpen
           */}
          {inChat && (
            <div style={{ width: 192, minWidth: 192, flexShrink: 0 }} className="h-full">
              <SessionHistoryPanel />
            </div>
          )}
          {inCompiler && ledgerOpen && (
            <div style={{ width: 240, minWidth: 240, flexShrink: 0 }} className="h-full">
              <LeftLedgerPane />
            </div>
          )}

          {/* Col 3: Main content */}
          <div className="flex-1 min-w-0 h-full overflow-hidden">
            {inChat ? (
              <ChatView />
            ) : (
              <PanelGroup direction="vertical" className="h-full">
                <Panel defaultSize={65} minSize={30}>
                  <div className="h-full overflow-hidden">
                    <CompilerView />
                  </div>
                </Panel>
                <HHandle />
                <Panel defaultSize={35} minSize={12} maxSize={65}>
                  <div className="h-full overflow-hidden">
                    <ConsoleOutput />
                  </div>
                </Panel>
              </PanelGroup>
            )}
          </div>

          {/* Col 4: Analysis panel (COMPILER only) */}
          {inCompiler && (
            <div style={{ width: 360, minWidth: 360, flexShrink: 0 }} className="h-full border-l border-slate-800 overflow-hidden">
              <RightPane />
            </div>
          )}
        </>
      )}
    </div>
  )
}
