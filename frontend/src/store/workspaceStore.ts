import { create } from 'zustand'
import type { ActiveView, Status, ChatMessage, AlphaRecord, SimResult, SimulationConfig } from '../types'

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
  // View
  activeView: ActiveView
  setActiveView: (v: ActiveView) => void

  // Editor
  editorDsl: string
  setEditorDsl: (dsl: string) => void

  // Chat
  sessionId: string
  chatMessages: ChatMessage[]
  addMessage: (msg: Omit<ChatMessage, 'id' | 'timestamp'>) => void
  clearChat: () => void

  // Alpha history
  alphaHistory: AlphaRecord[]
  setAlphaHistory: (records: AlphaRecord[]) => void

  // Simulation result
  simulationResult: SimResult | null
  setSimulationResult: (r: SimResult | null) => void

  // Status
  status: Status
  setStatus: (s: Status) => void

  // Console log
  consoleLogs: string[]
  appendLog: (line: string) => void
  clearLogs: () => void

  // Simulation config
  simConfig: SimulationConfig
  setSimConfig: (c: Partial<SimulationConfig>) => void
}

export const useWorkspaceStore = create<WorkspaceState>((set) => ({
  activeView: 'CHAT',
  setActiveView: (v) => set({ activeView: v }),

  editorDsl: 'rank(ts_delta(log(close), 5))',
  setEditorDsl: (dsl) => set({ editorDsl: dsl }),

  sessionId: genId(),
  chatMessages: [],
  addMessage: (msg) =>
    set((s) => ({
      chatMessages: [
        ...s.chatMessages,
        { ...msg, id: genId(), timestamp: Date.now() },
      ],
    })),
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
  setSimConfig: (c) =>
    set((s) => ({ simConfig: { ...s.simConfig, ...c } })),
}))
