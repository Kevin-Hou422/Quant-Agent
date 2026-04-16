import { useState } from 'react'
import MonacoEditor from '@monaco-editor/react'
import { useWorkspaceStore } from '../../store/workspaceStore'
import { useQuantWorkspace } from '../../hooks/useQuantWorkspace'
import { Play, Zap, Settings } from 'lucide-react'
import ConfigModal from './ConfigModal'

// ── DSL operator catalogue with documentation ────────────────────────────────

const TS_OPS = [
  { name: 'ts_mean',         doc: 'ts_mean(x, window) — Rolling mean over past `window` days.' },
  { name: 'ts_std',          doc: 'ts_std(x, window) — Rolling standard deviation.' },
  { name: 'ts_delta',        doc: 'ts_delta(x, window) — x[t] - x[t-window].' },
  { name: 'ts_delay',        doc: 'ts_delay(x, d) — Lag x by d days.' },
  { name: 'ts_rank',         doc: 'ts_rank(x, window) — Time-series percentile rank.' },
  { name: 'ts_max',          doc: 'ts_max(x, window) — Rolling maximum.' },
  { name: 'ts_min',          doc: 'ts_min(x, window) — Rolling minimum.' },
  { name: 'ts_decay_linear', doc: 'ts_decay_linear(x, window) — Linearly-weighted decay.' },
]

const CS_OPS = [
  { name: 'rank',         doc: 'rank(x) — Cross-sectional percentile rank [0,1].' },
  { name: 'zscore',       doc: 'zscore(x) — Cross-sectional z-score.' },
  { name: 'cs_rank',      doc: 'cs_rank(x) — Alias for rank().' },
  { name: 'cs_zscore',    doc: 'cs_zscore(x) — Alias for zscore().' },
  { name: 'ind_neutralize', doc: 'ind_neutralize(x) — Industry-neutral residual.' },
  { name: 'scale',        doc: 'scale(x) — Scale x to unit L1 norm.' },
]

const SCALAR_OPS = [
  { name: 'log',           doc: 'log(x) — Natural logarithm.' },
  { name: 'abs',           doc: 'abs(x) — Absolute value.' },
  { name: 'sign',          doc: 'sign(x) — Sign of x: -1, 0, 1.' },
  { name: 'sqrt',          doc: 'sqrt(x) — Square root.' },
  { name: 'signed_power',  doc: 'signed_power(x, p) — sign(x)*|x|^p.' },
]

const PRICE_FIELDS = [
  { name: 'close',   doc: 'Daily close price.' },
  { name: 'open',    doc: 'Daily open price.' },
  { name: 'high',    doc: 'Daily high price.' },
  { name: 'low',     doc: 'Daily low price.' },
  { name: 'volume',  doc: 'Daily trading volume.' },
  { name: 'vwap',    doc: 'Volume-weighted average price.' },
  { name: 'returns', doc: 'Daily pct_change of close.' },
]

const ALL_OPS = [...TS_OPS, ...CS_OPS, ...SCALAR_OPS]

// ── Monaco setup (called once before mount) ─────────────────────────────────

function setupMonaco(monaco: any) {
  // Avoid re-registering on hot-reload
  const lang = 'quantdsl'
  const existingLangs: string[] = monaco.languages.getLanguages().map((l: any) => l.id)
  if (!existingLangs.includes(lang)) {
    monaco.languages.register({ id: lang, extensions: ['.dsl'], aliases: ['QuantDSL'] })
  }

  // ── Monarch tokenizer (syntax highlighting) ──────────────────────────────
  monaco.languages.setMonarchTokensProvider(lang, {
    tokenizer: {
      root: [
        // TS operators
        [/\b(ts_mean|ts_std|ts_delta|ts_delay|ts_rank|ts_max|ts_min|ts_decay_linear)\b/, 'keyword.ts'],
        // CS operators
        [/\b(rank|zscore|cs_rank|cs_zscore|ind_neutralize|scale|signed_power)\b/, 'keyword.cs'],
        // Scalar operators
        [/\b(log|abs|sign|sqrt)\b/, 'keyword.scalar'],
        // Price fields
        [/\b(close|open|high|low|volume|vwap|returns)\b/, 'variable.price'],
        // Numbers
        [/\b\d+(\.\d+)?\b/, 'number'],
        // Operators
        [/[+\-*/]/, 'operator'],
        // Brackets
        [/[(),]/, 'delimiter'],
      ],
    },
  })

  // ── Custom token colors ─────────────────────────────────────────────────
  monaco.editor.defineTheme('quantdark', {
    base: 'vs-dark',
    inherit: true,
    rules: [
      { token: 'keyword.ts',     foreground: '10b981', fontStyle: 'bold' },  // emerald
      { token: 'keyword.cs',     foreground: '34d399', fontStyle: 'bold' },  // emerald-light
      { token: 'keyword.scalar', foreground: 'a7f3d0' },                     // emerald-pale
      { token: 'variable.price', foreground: '38bdf8' },                     // sky blue
      { token: 'number',         foreground: 'fbbf24' },                     // amber
      { token: 'operator',       foreground: 'e2e8f0' },
      { token: 'delimiter',      foreground: '94a3b8' },
    ],
    colors: {
      'editor.background': '#020617',  // slate-950
      'editor.foreground': '#e2e8f0',
      'editorLineNumber.foreground': '#334155',
      'editor.selectionBackground': '#1e3a5f',
      'editorCursor.foreground': '#10b981',
    },
  })

  // ── Auto-completion provider ────────────────────────────────────────────
  monaco.languages.registerCompletionItemProvider(lang, {
    triggerCharacters: ['(', '_', ...Array.from('abcdefghijklmnopqrstuvwxyz')],
    provideCompletionItems: (_model: any, position: any) => ({
      suggestions: [
        ...ALL_OPS.map((op) => ({
          label: op.name,
          kind: monaco.languages.CompletionItemKind.Function,
          insertText: op.name + '(',
          insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
          documentation: { value: `**${op.name}**\n\n${op.doc}` },
          range: {
            startLineNumber: position.lineNumber,
            endLineNumber: position.lineNumber,
            startColumn: position.column,
            endColumn: position.column,
          },
        })),
        ...PRICE_FIELDS.map((f) => ({
          label: f.name,
          kind: monaco.languages.CompletionItemKind.Variable,
          insertText: f.name,
          documentation: { value: `**${f.name}**\n\n${f.doc}` },
          range: {
            startLineNumber: position.lineNumber,
            endLineNumber: position.lineNumber,
            startColumn: position.column,
            endColumn: position.column,
          },
        })),
      ],
    }),
  })
}

// ── Component ────────────────────────────────────────────────────────────────

export default function CompilerView() {
  const { editorDsl, setEditorDsl, status } = useWorkspaceStore()
  const { runBacktest, runOptimize } = useQuantWorkspace()
  const [showConfig, setShowConfig] = useState(false)
  const [monacoReady, setMonacoReady] = useState(false)

  const isRunning = status === 'backtesting' || status === 'optimizing'

  return (
    <div className="flex flex-col h-full bg-slate-950">
      {/* Toolbar */}
      <div className="flex items-center gap-2 px-4 py-2.5 border-b border-slate-800 bg-slate-900 shrink-0">
        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex-1">
          DSL Compiler
          {monacoReady && (
            <span className="ml-2 text-emerald-500/60 font-normal normal-case">● QuantDSL</span>
          )}
        </span>

        <button
          onClick={() => setShowConfig(true)}
          className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-300 px-2 py-1 rounded transition-colors"
        >
          <Settings size={13} /> Config
        </button>

        <button
          onClick={runOptimize}
          disabled={isRunning}
          className="flex items-center gap-1.5 text-xs bg-violet-700 hover:bg-violet-600 disabled:opacity-40 text-white px-3 py-1.5 rounded-lg transition-colors"
        >
          <Zap size={13} /> AI Optimize
        </button>

        <button
          onClick={runBacktest}
          disabled={isRunning}
          className="flex items-center gap-1.5 text-xs bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white px-3 py-1.5 rounded-lg transition-colors"
        >
          <Play size={13} /> {isRunning ? 'Running…' : 'Run Backtest'}
        </button>
      </div>

      {/* Monaco Editor */}
      <div className="flex-1 min-h-0">
        <MonacoEditor
          height="100%"
          language="quantdsl"
          theme="quantdark"
          value={editorDsl}
          onChange={(v) => setEditorDsl(v ?? '')}
          beforeMount={setupMonaco}
          onMount={() => setMonacoReady(true)}
          options={{
            fontSize: 15,
            fontFamily: "'Fira Code', 'Cascadia Code', 'Consolas', monospace",
            fontLigatures: true,
            minimap: { enabled: false },
            lineNumbers: 'on',
            wordWrap: 'on',
            padding: { top: 16 },
            scrollBeyondLastLine: false,
            suggestOnTriggerCharacters: true,
            quickSuggestions: { other: true, comments: false, strings: false },
            parameterHints: { enabled: true },
            hover: { enabled: true },
            bracketPairColorization: { enabled: true },
          }}
        />
      </div>

      {showConfig && <ConfigModal onClose={() => setShowConfig(false)} />}
    </div>
  )
}
