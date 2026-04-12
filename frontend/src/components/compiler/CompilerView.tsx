import { useState } from 'react'
import MonacoEditor from '@monaco-editor/react'
import { useWorkspaceStore } from '../../store/workspaceStore'
import { useQuantWorkspace } from '../../hooks/useQuantWorkspace'
import ConsoleOutput from './ConsoleOutput'
import { Play, Zap, Settings } from 'lucide-react'
import ConfigModal from './ConfigModal'

// DSL autocomplete suggestions
const DSL_OPS = [
  'rank','ts_mean','ts_std','ts_delta','ts_rank','ts_decay_linear',
  'log','abs','sign','returns','vwap','close','volume','open','high','low',
  'ind_neutralize','cs_rank','cs_zscore','cs_scale',
]

function beforeMount(monaco: any) {
  monaco.languages.registerCompletionItemProvider('plaintext', {
    provideCompletionItems: (_: any, position: any) => ({
      suggestions: DSL_OPS.map(op => ({
        label: op,
        kind: monaco.languages.CompletionItemKind.Function,
        insertText: op,
        range: {
          startLineNumber: position.lineNumber,
          endLineNumber: position.lineNumber,
          startColumn: position.column,
          endColumn: position.column,
        },
      })),
    }),
  })
}

export default function CompilerView() {
  const { editorDsl, setEditorDsl, status } = useWorkspaceStore()
  const { runBacktest, runOptimize } = useQuantWorkspace()
  const [showConfig, setShowConfig] = useState(false)

  const isRunning = status === 'backtesting' || status === 'optimizing'

  return (
    <div className="flex flex-col h-full bg-slate-950">
      {/* Toolbar */}
      <div className="flex items-center gap-2 px-4 py-2.5 border-b border-slate-800 bg-slate-900">
        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex-1">
          DSL Compiler
        </span>

        <button
          onClick={() => setShowConfig(true)}
          className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-300 px-2 py-1 rounded transition-colors"
          title="Simulation Config"
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
          defaultLanguage="plaintext"
          theme="vs-dark"
          value={editorDsl}
          onChange={(v) => setEditorDsl(v ?? '')}
          beforeMount={beforeMount}
          options={{
            fontSize: 15,
            fontFamily: "'Fira Code', 'Cascadia Code', 'Consolas', monospace",
            fontLigatures: true,
            minimap: { enabled: false },
            lineNumbers: 'off',
            wordWrap: 'on',
            padding: { top: 20 },
            scrollBeyondLastLine: false,
            suggestOnTriggerCharacters: true,
            quickSuggestions: { other: true, comments: false, strings: false },
          }}
        />
      </div>

      {/* Console */}
      <ConsoleOutput />

      {showConfig && <ConfigModal onClose={() => setShowConfig(false)} />}
    </div>
  )
}
