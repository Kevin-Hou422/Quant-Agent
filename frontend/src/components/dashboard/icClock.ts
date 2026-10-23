/**
 * IC 历史按时钟版本拆开（2026-10-22 用户决定 D14："旧历史保留旧版本，先隔离重算"）。
 * c1 的 IC 是前收盘 → 今收盘的标签，c2 是开盘(T+1) → 开盘(T+2)：两者不是同一个量，不连成一条线。
 * 没有时钟版本的行是版本化之前写的，那时只有 c1。
 */
export interface IcPoint {
  date: string
  realized_ic: number
  realized_return: number
  clock_version?: string
}

export interface IcByClock {
  dates: string[]
  current: (number | null)[]
  legacy: (number | null)[]
  nCurrent: number
  nLegacy: number
}

export function splitIcByClock(points: IcPoint[], currentClock: string): IcByClock {
  const isCurrent = (p: IcPoint) => (p.clock_version ?? 'c1') === currentClock
  return {
    dates: points.map((p) => p.date),
    current: points.map((p) => (isCurrent(p) ? p.realized_ic : null)),
    legacy: points.map((p) => (isCurrent(p) ? null : p.realized_ic)),
    nCurrent: points.filter(isCurrent).length,
    nLegacy: points.filter((p) => !isCurrent(p)).length,
  }
}
