import { create } from 'zustand'
import type {
  ActiveView, Status, ChatMessage, AlphaRecord,
  SimResult, SimulationConfig, ChatSession, EditorTab,
} from '../types'

const genId = () => Math.random().toString(36).slice(2)

const DEFAULT_DSL = 'rank(ts_delta(log(close), 5))'

const DEFAULT_CONFIG: SimulationConfig = {
  delay: 1,
  decay_window: 0,
  truncation_min_q: 0.05,
  truncation_max_q: 0.95,
  portfolio_mode: 'long_short',
  top_pct: 0.10,
}

function makeDefaultTab(): EditorTab {
  return { id: genId(), label: 'New Alpha', dsl: DEFAULT_DSL, isModified: false }
}

interface WorkspaceState {
  activeView: ActiveView
  setActiveView: (v: ActiveView) => void

  // ── Editor Tabs ─────────────────────────────────────────────────────────
  editorTabs:    EditorTab[]
  activeTabId:   string
  editorDsl:     string   // always mirrors active tab's dsl

  setEditorDsl:        (dsl: string) => void
  setActiveTab:        (id: string) => void
  closeTab:            (id: string) => void
  newEmptyTab:         () => void
  openAlphaInNewTab:   (alpha: AlphaRecord) => void

  // ── Session management ───────────────────────────────────────────────────
  sessionId:   string
  setSessionId: (id: string) => void

  sessions:    ChatSession[]
  setSessions: (sessions: ChatSession[]) => void
  addSession:  (s: ChatSession) => void

  // ── Chat messages ────────────────────────────────────────────────────────
  chatMessages: ChatMessage[]
  addMessage:   (msg: Omit<ChatMessage, 'id' | 'timestamp'>) => void
  setMessages:  (msgs: ChatMessage[]) => void
  clearChat:    () => void

  alphaHistory:    AlphaRecord[]
  setAlphaHistory: (records: AlphaRecord[]) => void

  simulationResult:    SimResult | null
  setSimulationResult: (r: SimResult | null) => void

  status:    Status
  setStatus: (s: Status) => void

  consoleLogs: string[]
  appendLog:   (line: string) => void
  clearLogs:   () => void

  simConfig:    SimulationConfig
  setSimConfig: (c: Partial<SimulationConfig>) => void

  // Ledger panel (second column in COMPILER mode)
  ledgerOpen:   boolean
  toggleLedger: () => void
  setLedgerOpen:(open: boolean) => void
}

const _defaultTab = makeDefaultTab()

export const useWorkspaceStore = create<WorkspaceState>((set) => ({
  // Start in COMPILER mode
  activeView: 'COMPILER',
  setActiveView: (v) => set({ activeView: v }),

  // ── Editor Tabs ──────────────────────────────────────────────────────────
  editorTabs:  [_defaultTab],
  activeTabId: _defaultTab.id,
  editorDsl:   _defaultTab.dsl,

  setEditorDsl: (dsl) => set((s) => ({
    editorDsl:   dsl,
    editorTabs:  s.editorTabs.map((t) =>
      t.id === s.activeTabId ? { ...t, dsl, isModified: true } : t,
    ),
  })),

  setActiveTab: (id) => set((s) => {
    const tab = s.editorTabs.find((t) => t.id === id)
    return tab ? { activeTabId: id, editorDsl: tab.dsl } : {}
  }),

  closeTab: (id) => set((s) => {
    const remaining = s.editorTabs.filter((t) => t.id !== id)
    if (remaining.length === 0) {
      const fresh = makeDefaultTab()
      return { editorTabs: [fresh], activeTabId: fresh.id, editorDsl: fresh.dsl }
    }
    if (s.activeTabId !== id) return { editorTabs: remaining }
    // Active tab closed — activate adjacent tab
    const idx     = s.editorTabs.findIndex((t) => t.id === id)
    const nextTab = remaining[Math.min(idx, remaining.length - 1)]
    return { editorTabs: remaining, activeTabId: nextTab.id, editorDsl: nextTab.dsl }
  }),

  newEmptyTab: () => set((s) => {
    const t = makeDefaultTab()
    return { editorTabs: [...s.editorTabs, t], activeTabId: t.id, editorDsl: t.dsl }
  }),

  openAlphaInNewTab: (alpha) => set((s) => {
    // Re-activate if already open
    const existing = s.editorTabs.find((t) => t.alphaId === alpha.id)
    if (existing) {
      return {
        activeTabId: existing.id,
        editorDsl:   existing.dsl,
        activeView:  'COMPILER',
        ledgerOpen:  false,
      }
    }
    const t: EditorTab = {
      id:         genId(),
      label:      alpha.hypothesis
                    ? alpha.hypothesis.slice(0, 24)
                    : `Alpha #${alpha.id}`,
      dsl:        alpha.dsl,
      alphaId:    alpha.id,
      isModified: false,
    }
    return {
      editorTabs:  [...s.editorTabs, t],
      activeTabId: t.id,
      editorDsl:   t.dsl,
      activeView:  'COMPILER',
      ledgerOpen:  false,
    }
  }),

  // ── Session management ───────────────────────────────────────────────────
  sessionId:    genId(),
  setSessionId: (id) => set({ sessionId: id }),

  sessions:    [],
  setSessions: (sessions) => set({ sessions }),
  addSession:  (s) => set((st) => ({
    sessions: [s, ...st.sessions.filter((x) => x.id !== s.id)],
  })),

  // ── Chat messages ────────────────────────────────────────────────────────
  chatMessages: [],
  addMessage: (msg) =>
    set((s) => ({
      chatMessages: [...s.chatMessages, { ...msg, id: genId(), timestamp: Date.now() }],
    })),
  setMessages: (msgs) => set({ chatMessages: msgs }),
  clearChat:   () => set({ chatMessages: [] }),

  alphaHistory:    [],
  setAlphaHistory: (records) => set({ alphaHistory: records }),

  simulationResult:    null,
  setSimulationResult: (r) => set({ simulationResult: r }),

  status:    'idle',
  setStatus: (s) => set({ status: s }),

  consoleLogs: [],
  appendLog: (line) =>
    set((s) => ({ consoleLogs: [...s.consoleLogs.slice(-200), line] })),
  clearLogs: () => set({ consoleLogs: [] }),

  simConfig:    DEFAULT_CONFIG,
  setSimConfig: (c) => set((s) => ({ simConfig: { ...s.simConfig, ...c } })),

  ledgerOpen:   false,
  toggleLedger: () => set((s) => ({ ledgerOpen: !s.ledgerOpen })),
  setLedgerOpen:(open) => set({ ledgerOpen: open }),
}))
