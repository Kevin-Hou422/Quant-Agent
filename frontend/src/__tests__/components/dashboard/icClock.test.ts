/**
 * 2026-10-22 用户决定 D14："旧历史保留旧版本，先隔离重算。"
 * c1（旧时钟）的 IC 是另一种标签（前收盘 → 今收盘），不能和 c2 连成一条线、也不能算进当前统计。
 */
import { describe, expect, it } from 'vitest'
import { splitIcByClock } from '../../../components/dashboard/icClock'

const pts = [
  { date: '2026-10-01', realized_ic: 0.5, realized_return: 0, clock_version: 'c1' },
  { date: '2026-10-02', realized_ic: 0.4, realized_return: 0, clock_version: 'c1' },
  { date: '2026-10-23', realized_ic: 0.1, realized_return: 0, clock_version: 'c2' },
]

describe('splitIcByClock', () => {
  it('puts each clock in its own series on a shared date axis', () => {
    const s = splitIcByClock(pts, 'c2')
    expect(s.dates).toEqual(['2026-10-01', '2026-10-02', '2026-10-23'])
    expect(s.current).toEqual([null, null, 0.1])
    expect(s.legacy).toEqual([0.5, 0.4, null])
    expect(s.nCurrent).toBe(1)
    expect(s.nLegacy).toBe(2)
  })

  it('treats a point without a clock version as c1 (rows before versioning)', () => {
    const s = splitIcByClock([{ date: 'd', realized_ic: 0.3, realized_return: 0 }], 'c2')
    expect(s.current).toEqual([null])
    expect(s.legacy).toEqual([0.3])
  })
})
