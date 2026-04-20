import { create } from 'zustand'
import type { ActiveView, Status, ChatMessage, AlphaRecord, SimResult, SimulationConfig, ChatSession } from '../types'

const genId = () => Math.random().toString(36).slice(2)

const DEFAULT_CONFIG: SimulationConfig = {
  delay: 1,
  decay_window: 0,
  truncation_min_q: 0.05,
  truncation_max_q: 0.95,
  portfolio_mode: 'long_short',
  top_pct: 0.10,
}

interface WorkspaceState {
  activeView: ActiveView
  setActiveView: (v: ActiveView) => void

  editorDsl: string
  setEditorDsl: (dsl: string) => void

  // ── Session management ──────────────────────────────────────────────────
  sessionId: string
  setSessionId: (id: string) => void

  sessions: ChatSession[]
  setSessions: (sessions: ChatSession[]) => void
  addSession: (s: ChatSession) => void

  // ── Chat messages ───────────────────────────────────────────────────────
  chatMessages: ChatMessage[]
  addMessage: (msg: Omit<ChatMessage, 'id' | 'timestamp'>) => void
  setMessages: (msgs: ChatMessage[]) => void
  clearChat: () => void

  alphaHistory: AlphaRecord[]
  setAlphaHistory: (records: AlphaRecord[]) => void

  simulationResult: SimResult | null
  setSimulationResult: (r: SimResult | null) => void

  status: Status
  setStatus: (s: Status) => void

  consoleLogs: string[]
  appendLog: (line: string) => void
  clearLogs: () => void

  simConfig: SimulationConfig
  setSimConfig: (c: Partial<SimulationConfig>) => void

  // Ledger slide-out state
  ledgerOpen: boolean
  toggleLedger: () => void
  setLedgerOpen: (open: boolean) => void
}

export const useWorkspaceStore = create<WorkspaceState>((set) => ({
  activeView: 'CHAT',
  setActiveView: (v) => set({ activeView: v }),

  editorDsl: 'rank(ts_delta(log(close), 5))',
  setEditorDsl: (dsl) => set({ editorDsl: dsl }),

  // ── Session management ────────────────────────────────────────────────
  sessionId: genId(),
  setSessionId: (id) => set({ sessionId: id }),

  sessions: [],
  setSessions: (sessions) => set({ sessions }),
  addSession: (s) => set((st) => ({
    sessions: [s, ...st.sessions.filter((x) => x.id !== s.id)],
  })),

  // ── Chat messages ─────────────────────────────────────────────────────
  chatMessages: [],
  addMessage: (msg) =>
    set((s) => ({
      chatMessages: [...s.chatMessages, { ...msg, id: genId(), timestamp: Date.now() }],
    })),
  setMessages: (msgs) => set({ chatMessages: msgs }),
  clearChat: () => set({ chatMessages: [] }),

  alphaHistory: [],
  setAlphaHistory: (records) => set({ alphaHistory: records }),

  simulationResult: null,
  setSimulationResult: (r) => set({ simulationResult: r }),

  status: 'idle',
  setStatus: (s) => set({ status: s }),

  consoleLogs: [],
  appendLog: (line) =>
    set((s) => ({ consoleLogs: [...s.consoleLogs.slice(-200), line] })),
  clearLogs: () => set({ consoleLogs: [] }),

  simConfig: DEFAULT_CONFIG,
  setSimConfig: (c) => set((s) => ({ simConfig: { ...s.simConfig, ...c } })),

  ledgerOpen: false,
  toggleLedger: () => set((s) => ({ ledgerOpen: !s.ledgerOpen })),
  setLedgerOpen: (open) => set({ ledgerOpen: open }),
}))
