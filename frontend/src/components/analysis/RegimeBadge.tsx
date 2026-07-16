import { useEffect, useState } from 'react'
import { TrendingUp, TrendingDown, MoveRight, Zap } from 'lucide-react'
import { apiFetchRegime } from '../../api/client'
import { useWorkspaceStore } from '../../store/workspaceStore'
import type { RegimeInfo } from '../../types'

/**
 * FE-4.1: Market regime badge (Task 4.1).
 *
 * Fetches GET /api/regime for the active dataset and renders a compact
 * BULL / BEAR / SIDEWAYS / HIGH VOL indicator. Silently renders nothing
 * on fetch failure (regime is auxiliary info, never blocks the UI).
 */

const STYLE: Record<RegimeInfo['regime'], { cls: string; label: string; Icon: typeof TrendingUp }> = {
  bull:     { cls: 'bg-emerald-900/60 text-emerald-400', label: 'BULL',     Icon: TrendingUp },
  bear:     { cls: 'bg-rose-900/60 text-rose-400',       label: 'BEAR',     Icon: TrendingDown },
  sideways: { cls: 'bg-slate-800 text-slate-400',        label: 'SIDEWAYS', Icon: MoveRight },
  high_vol: { cls: 'bg-amber-900/60 text-amber-400',     label: 'HIGH VOL', Icon: Zap },
}

export default function RegimeBadge() {
  const { simConfig } = useWorkspaceStore()
  const [info, setInfo] = useState<RegimeInfo | null>(null)

  useEffect(() => {
    let cancelled = false
    setInfo(null)
    apiFetchRegime(simConfig.dataset, simConfig.start_date, simConfig.end_date)
      .then((res) => { if (!cancelled) setInfo(res.data) })
      .catch(() => { /* auxiliary info — fail silently */ })
    return () => { cancelled = true }
  }, [simConfig.dataset, simConfig.start_date, simConfig.end_date])

  if (!info) return null

  const { cls, label, Icon } = STYLE[info.regime] ?? STYLE.sideways

  return (
    <span
      className={`ml-auto flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-semibold ${cls}`}
      title={`Market regime for ${info.dataset} as of ${info.as_of} (trend ${info.trend_window}d / vol ${info.vol_window}d)`}
    >
      <Icon size={11} />
      {label}
    </span>
  )
}
