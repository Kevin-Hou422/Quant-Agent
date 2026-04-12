interface Props { score: number }

export default function OverfitBadge({ score }: Props) {
  if (score < 0.4) {
    return (
      <span className="inline-flex items-center gap-1 text-xs bg-emerald-900/40 text-emerald-400 border border-emerald-800 rounded-full px-2 py-0.5">
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
        Healthy
      </span>
    )
  }

  if (score < 0.6) {
    return (
      <span className="inline-flex items-center gap-1 text-xs bg-amber-900/40 text-amber-400 border border-amber-700 rounded-full px-2 py-0.5 animate-pulse">
        <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
        Overfit Risk {(score * 100).toFixed(0)}%
      </span>
    )
  }

  return (
    <span className="inline-flex items-center gap-1 text-xs bg-rose-900/40 text-rose-400 border border-rose-700 rounded-full px-2 py-0.5 animate-pulse">
      <span className="w-1.5 h-1.5 rounded-full bg-rose-400" />
      Overfit! {(score * 100).toFixed(0)}%
    </span>
  )
}
