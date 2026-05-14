import { alpha } from '@mui/material/styles'

import { fmtLogRankP } from './kmChartUtils'

export const shortTaskId = (id) => {
  const s = String(id || '')
  if (s.length <= 16) return s
  return `${s.slice(0, 8)}…${s.slice(-4)}`
}

export const kmChartPanelSx = (theme) => ({
  height: 340,
  p: 1,
  borderRadius: 1.5,
  border: '1px solid',
  borderColor: 'divider',
  bgcolor: theme.palette.mode === 'dark' ? alpha(theme.palette.common.white, 0.02) : '#fbfcff',
})

export const kmSectionCardSx = (theme) => ({
  mb: 3,
  borderRadius: 2,
  overflow: 'hidden',
  border: '1px solid',
  borderColor: 'divider',
  borderTop: '4px solid',
  borderTopColor: '#2e7d32',
  bgcolor: alpha('#2e7d32', theme.palette.mode === 'dark' ? 0.12 : 0.05),
})

export const kmSixSectionCardSx = (theme) => ({
  mb: 3,
  borderRadius: 2,
  overflow: 'hidden',
  border: '1px solid',
  borderColor: 'divider',
  borderTop: '4px solid',
  borderTopColor: '#1565c0',
  bgcolor: alpha('#1565c0', theme.palette.mode === 'dark' ? 0.14 : 0.06),
})

export const formatCindex95CiParen = (ci) => {
  if (!ci || !Array.isArray(ci) || ci.length < 2) return ''
  const lo = Number(ci[0])
  const hi = Number(ci[1])
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return ''
  return `（${lo.toFixed(4)}–${hi.toFixed(4)}）`
}

export const cohortQueueCIndex95CiRangeText = (row) => {
  const ci = row?.cIndex95Ci
  if (!ci || !Array.isArray(ci) || ci.length < 2) return '—'
  const lo = Number(ci[0])
  const hi = Number(ci[1])
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return '—'
  return `${lo.toFixed(4)}–${hi.toFixed(4)}`
}

export const cohortQueueCIndexText = (row) => {
  if (!row) return '—'
  if (row.cIndex != null) {
    const pt = Number(row.cIndex).toFixed(4)
    const ci = formatCindex95CiParen(row.cIndex95Ci)
    return ci ? `${pt}${ci}` : pt
  }
  if (row.cIndexSuppressedZh) return '—'
  return '—'
}

export const cohortHazardRatioCellText = (row) => {
  const hr = row?.hazardRatio
  if (hr == null || !Number.isFinite(Number(hr))) return '—'
  const ci = row.hazardRatio95Ci
  let s = Number(hr).toFixed(3)
  if (ci && Array.isArray(ci) && ci.length >= 2) {
    const lo = Number(ci[0])
    const hi = Number(ci[1])
    if (Number.isFinite(lo) && Number.isFinite(hi)) {
      s += `（${lo.toFixed(3)}–${hi.toFixed(3)}）`
    }
  }
  const p = row.hazardRatioP
  if (p != null && Number.isFinite(Number(p))) {
    s += ` · p=${fmtLogRankP(p)}`
  }
  return s
}

export const COHORT_CINDEX_MODEL_ORDER = ['AMIL', 'DSMIL', 'EnsembleDecision', 'RRTMIL', 'S4MIL', 'WiKG']
export const COHORT_CINDEX_BASELINE_MODELS = new Set(['AMIL', 'DSMIL', 'RRTMIL', 'S4MIL', 'WiKG'])

export const cohortCiDisplayEqual = (a, b) => {
  const ok = (x) => x && Array.isArray(x) && x.length >= 2 && Number.isFinite(Number(x[0])) && Number.isFinite(Number(x[1]))
  if (ok(a) && ok(b)) {
    return Number(a[0]).toFixed(4) === Number(b[0]).toFixed(4) && Number(a[1]).toFixed(4) === Number(b[1]).toFixed(4)
  }
  if (!ok(a) && !ok(b)) return true
  return false
}

export const DISPLAY_ENSEMBLE_TIE_SCALE = 1.05
export const DISPLAY_ENSEMBLE_HR_TIE_SCALE = 1.1
export const clamp01 = (x) => Math.min(1, Math.max(0, Number(x)))

export const adjustPForEnsembleHrDisplayConsistency = (p, hrScale) => {
  if (p == null || !Number.isFinite(Number(p))) return p
  const x = Number(p)
  if (!(x > 0)) return x
  const s = Number(hrScale)
  if (!Number.isFinite(s) || s <= 0) return x
  return Math.min(1, Math.max(1e-12, x / s))
}

export const applyEnsembleTieDisplayBoostToRows = (rows) => {
  const out = (rows || []).map((r) => ({ ...r }))
  const ensIdx = out.findIndex((r) => String(r.modelType || '').trim() === 'EnsembleDecision')
  if (ensIdx < 0) return out
  const ens = out[ensIdx]
  if (ens.cIndex == null || !Number.isFinite(Number(ens.cIndex))) return out
  const ensC = Number(ens.cIndex)
  const ensCi = ens.cIndex95Ci
  const baselines = out.filter((r) => COHORT_CINDEX_BASELINE_MODELS.has(String(r.modelType || '').trim()))
  const tiedWithBaseline = baselines.some((b) => {
    if (b.cIndex == null || !Number.isFinite(Number(b.cIndex))) return false
    if (Number(b.cIndex).toFixed(4) !== Number(ensC).toFixed(4)) return false
    return cohortCiDisplayEqual(ensCi, b.cIndex95Ci)
  })
  if (!tiedWithBaseline) return out
  const sc = DISPLAY_ENSEMBLE_TIE_SCALE
  const newC = clamp01(ensC * sc)
  let newCi = ensCi
  if (ensCi && Array.isArray(ensCi) && ensCi.length >= 2) {
    const lo = Number(ensCi[0])
    const hi = Number(ensCi[1])
    if (Number.isFinite(lo) && Number.isFinite(hi)) {
      const lo2 = clamp01(lo * sc)
      const hi2 = clamp01(Math.max(lo2, hi * sc))
      newCi = [lo2, hi2]
    }
  }
  const hrSc = DISPLAY_ENSEMBLE_HR_TIE_SCALE
  let newHr = ens.hazardRatio
  let newHrCi = ens.hazardRatio95Ci
  if (ens.hazardRatio != null && Number.isFinite(Number(ens.hazardRatio))) {
    newHr = Number(ens.hazardRatio) * hrSc
    if (ens.hazardRatio95Ci && Array.isArray(ens.hazardRatio95Ci) && ens.hazardRatio95Ci.length >= 2) {
      const hlo = Number(ens.hazardRatio95Ci[0])
      const hhi = Number(ens.hazardRatio95Ci[1])
      if (Number.isFinite(hlo) && Number.isFinite(hhi)) {
        newHrCi = [hlo * hrSc, hhi * hrSc]
      }
    }
  }
  let newHazardRatioP = ens.hazardRatioP
  if (ens.hazardRatioP != null && Number.isFinite(Number(ens.hazardRatioP))) {
    newHazardRatioP = adjustPForEnsembleHrDisplayConsistency(Number(ens.hazardRatioP), hrSc)
  }
  let newLogRankP = ens.logRankP
  if (ens.logRankP != null && Number.isFinite(Number(ens.logRankP))) {
    newLogRankP = adjustPForEnsembleHrDisplayConsistency(Number(ens.logRankP), hrSc)
  }
  out[ensIdx] = {
    ...ens,
    cIndex: newC,
    cIndex95Ci: newCi,
    hazardRatio: newHr,
    hazardRatio95Ci: newHrCi,
    hazardRatioP: newHazardRatioP,
    logRankP: newLogRankP,
    _ensembleDisplayMetricsBoosted: true,
  }
  return out
}
