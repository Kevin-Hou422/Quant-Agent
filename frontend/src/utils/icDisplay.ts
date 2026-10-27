import type { IcStatus } from '../types'

const STATUS_LABEL: Record<IcStatus, string> = {
  evaluated: '',
  not_computed: '未算 IC',
  never_evaluated: '未评估',
  legacy_unknown: '旧值，口径未知',
  values_without_provenance: '无来源',
}

const fmt = (v: number | null | undefined) => (v == null || !Number.isFinite(v) ? '—' : v.toFixed(3))

/**
 * IC 均值与 IC-IR 分开显示（审计决定 2）。旧行（legacy_unknown）的 ic_ir 列多半是 IC 均值：
 * 原值照样给出，但标明口径未知，不当作 IC-IR。
 */
export function icSummary(r: { ic_mean?: number | null; ic_ir?: number | null; ic_status?: IcStatus }): string {
  const status = r.ic_status ?? 'legacy_unknown'
  if (status === 'legacy_unknown') return `IC 列 ${fmt(r.ic_ir)}（${STATUS_LABEL.legacy_unknown}）`
  const note = STATUS_LABEL[status] ? `（${STATUS_LABEL[status]}）` : ''
  return `IC 均值 ${fmt(r.ic_mean)} · IC-IR ${fmt(r.ic_ir)}${note}`
}
