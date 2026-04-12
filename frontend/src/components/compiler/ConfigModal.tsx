import { useWorkspaceStore } from '../../store/workspaceStore'
import { X } from 'lucide-react'

interface Props { onClose: () => void }

const Slider = ({
  label, field, min, max, step, fmt,
}: {
  label: string
  field: keyof import('../../types').SimulationConfig
  min: number; max: number; step: number
  fmt?: (v: number) => string
}) => {
  const { simConfig, setSimConfig } = useWorkspaceStore()
  const val = simConfig[field] as number
  return (
    <div className="flex items-center gap-3 text-sm">
      <span className="text-slate-400 w-40 shrink-0">{label}</span>
      <input
        type="range" min={min} max={max} step={step}
        value={val}
        onChange={e => setSimConfig({ [field]: Number(e.target.value) } as any)}
        className="flex-1 accent-emerald-500"
      />
      <span className="text-emerald-400 font-mono w-12 text-right text-xs">
        {fmt ? fmt(val) : val}
      </span>
    </div>
  )
}

export default function ConfigModal({ onClose }: Props) {
  const { simConfig, setSimConfig } = useWorkspaceStore()

  return (
    <div className="absolute inset-0 bg-black/70 flex items-center justify-center z-50">
      <div className="bg-slate-900 border border-slate-700 rounded-xl p-6 w-[480px] shadow-2xl">
        <div className="flex items-center justify-between mb-5">
          <h3 className="text-sm font-semibold text-slate-200">Simulation Config</h3>
          <button onClick={onClose} className="text-slate-500 hover:text-slate-300"><X size={16} /></button>
        </div>

        <div className="space-y-4">
          <Slider label="Execution Delay (days)" field="delay" min={0} max={10} step={1} />
          <Slider label="Decay Window" field="decay_window" min={0} max={30} step={1} />
          <Slider label="Truncation Min Q" field="truncation_min_q" min={0.01} max={0.20} step={0.01}
            fmt={v => `${(v * 100).toFixed(0)}%`} />
          <Slider label="Truncation Max Q" field="truncation_max_q" min={0.80} max={0.99} step={0.01}
            fmt={v => `${(v * 100).toFixed(0)}%`} />

          <div className="flex items-center gap-3 text-sm">
            <span className="text-slate-400 w-40 shrink-0">Portfolio Mode</span>
            <select
              value={simConfig.portfolio_mode}
              onChange={e => setSimConfig({ portfolio_mode: e.target.value as any })}
              className="flex-1 bg-slate-800 border border-slate-700 rounded px-2 py-1 text-slate-200 text-xs"
            >
              <option value="long_short">Long-Short</option>
              <option value="decile">Decile</option>
            </select>
          </div>
        </div>

        <button
          onClick={onClose}
          className="mt-6 w-full bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg py-2 text-sm transition-colors"
        >
          Apply
        </button>
      </div>
    </div>
  )
}
