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

// ── localStorage helpers ──────────────────────────────────────────────────────

const LS_SESSION_KEY = 'qagent_session_id'

function readStoredSessionId(): string {
  try { return localStorage.getItem(LS_SESSION_KEY) || genId() } catch { return genId() }
}

function writeStoredSessionId(id: string): void {
  try { localStorage.setItem(LS_SESSION_KEY, id) } catch { /* ignore */ }
}

// ── Store interface ───────────────────────────────────────────────────────────

interface WorkspaceState {
  activeView: ActiveView
  setActiveView: (v: ActiveView) => void

  // ── Editor Tabs ─────────────────────────────────────────────────────────
  editorTabs:    EditorTab[]
  activeTabId:   string
  editorDsl:     string

  setEditorDsl:        (dsl: string) => void
  setActiveTab:        (id: string) => void
  closeTab:            (id: string) => void
  newEmptyTab:         () => void
  openAlphaInNewTab:   (alpha: AlphaRecord) => void

  // ── Session management ───────────────────────────────────────────────────
  sessionId:   string
  setSessionId: (id: string) => void

  sessions:             ChatSession[]
  setSessions:          (sessions: ChatSession[]) => void
  addSession:           (s: ChatSession) => void
  updateSessionTitle:   (id: string, title: string) => void
  removeSession:        (id: string) => void

  // ── Chat messages ────────────────────────────────────────────────────────
  chatMessages: ChatMessage[]
  addMessage:   (msg: Omit<ChatMessage, 'id' | 'timestamp'>) => void
  setMessages:  (msgs: ChatMessage[]) => void
  clearChat:    () => void

  // ── Streaming messages ───────────────────────────────────────────────────
  startStreamingMessage:    (id: string) => void
  appendToStreamingMessage: (id: string, chunk: string) => void
  finalizeStreamingMessage: (id: string, extras?: Partial<ChatMessage>) => void

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

  ledgerOpen:    boolean
  toggleLedger:  () => void
  setLedgerOpen: (open: boolean) => void
}

const _defaultTab = makeDefaultTab()

export const useWorkspaceStore = create<WorkspaceState>((set) => ({
  activeView: 'COMPILER',
  setActiveView: (v) => set({ activeView: v }),

  // ── Editor Tabs ──────────────────────────────────────────────────────────
  editorTabs:  [_defaultTab],
  activeTabId: _defaultTab.id,
  editorDsl:   _defaultTab.dsl,

  setEditorDsl: (dsl) => set((s) => ({
    editorDsl:  dsl,
    editorTabs: s.editorTabs.map((t) =>
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
    const idx     = s.editorTabs.findIndex((t) => t.id === id)
    const nextTab = remaining[Math.min(idx, remaining.length - 1)]
    return { editorTabs: remaining, activeTabId: nextTab.id, editorDsl: nextTab.dsl }
  }),

  newEmptyTab: () => set((s) => {
    const t = makeDefaultTab()
    return { editorTabs: [...s.editorTabs, t], activeTabId: t.id, editorDsl: t.dsl }
  }),

  openAlphaInNewTab: (alpha) => set((s) => {
    const existing = s.editorTabs.find((t) => t.alphaId === alpha.id)
    if (existing) {
      return { activeTabId: existing.id, editorDsl: existing.dsl, activeView: 'COMPILER', ledgerOpen: false }
    }
    const t: EditorTab = {
      id:         genId(),
      label:      alpha.hypothesis ? alpha.hypothesis.slice(0, 24) : `Alpha #${alpha.id}`,
      dsl:        alpha.dsl,
      alphaId:    alpha.id,
      isModified: false,
    }
    return { editorTabs: [...s.editorTabs, t], activeTabId: t.id, editorDsl: t.dsl, activeView: 'COMPILER', ledgerOpen: false }
  }),

  // ── Session management ───────────────────────────────────────────────────
  // Restore sessionId from localStorage on startup so page-reload restores context
  sessionId:    readStoredSessionId(),
  setSessionId: (id) => {
    writeStoredSessionId(id)
    set({ sessionId: id })
  },

  sessions:    [],
  setSessions: (sessions) => set({ sessions }),

  addSession: (s) => set((st) => ({
    sessions: [s, ...st.sessions.filter((x) => x.id !== s.id)],
  })),

  updateSessionTitle: (id, title) => set((st) => ({
    sessions: st.sessions.map((s) => s.id === id ? { ...s, title } : s),
  })),

  removeSession: (id) => set((st) => ({
    sessions: st.sessions.filter((s) => s.id !== id),
  })),

  // ── Chat messages ────────────────────────────────────────────────────────
  chatMessages: [],
  addMessage: (msg) =>
    set((s) => ({
      chatMessages: [...s.chatMessages, { ...msg, id: genId(), timestamp: Date.now() }],
    })),
  setMessages: (msgs) => set({ chatMessages: msgs }),
  clearChat:   () => set({ chatMessages: [] }),

  // ── Streaming messages ───────────────────────────────────────────────────
  startStreamingMessage: (id) =>
    set((s) => ({
      chatMessages: [
        ...s.chatMessages,
        { id, role: 'assistant', content: '', timestamp: Date.now(), type: 'message', isStreaming: true },
      ],
    })),

  appendToStreamingMessage: (id, chunk) =>
    set((s) => ({
      chatMessages: s.chatMessages.map((m) =>
        m.id === id ? { ...m, content: m.content + chunk } : m,
      ),
    })),

  finalizeStreamingMessage: (id, extras = {}) =>
    set((s) => ({
      chatMessages: s.chatMessages.map((m) =>
        m.id === id ? { ...m, isStreaming: false, ...extras } : m,
      ),
    })),

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

  ledgerOpen:    false,
  toggleLedger:  () => set((s) => ({ ledgerOpen: !s.ledgerOpen })),
  setLedgerOpen: (open) => set({ ledgerOpen: open }),
}))
